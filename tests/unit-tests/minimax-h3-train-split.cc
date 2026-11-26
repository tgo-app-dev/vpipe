// LoRA training on a DiT SPLIT across hosts: MiniMax-H3's trainer as two
// ranks (Config::tp), against the whole model, on the synthetic H3 (two
// heads, feed-forward 512) -- one head and 256 rows a rank -- over an
// in-process link.

#include "minitest.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/minimax-h3/minimax-h3-trainer.h"
#include "generative-models/minimax-h3/minimax-h3-image.h"
#include "generative-models/minimax-h3/minimax-h3-lora-trainer.h"
#include "generative-models/qwen-image/qwen-image21-lora-trainer.h"
#include "generative-models/train/lora-trainer.h"
#include "generative-models/weight-set.h"
#include "generative-models/tensor-parallel.h"
#include "generative-models/train/lora-params.h"
#include "generative-models/train/optimizer.h"
#include "generative-models/train/train-ops.h"
#include "minimax-h3-synthetic.h"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::SharedBuffer;
namespace h3t = vpipe::genai::minimax_h3;
namespace fs = std::filesystem;
using T = MetalMiniMaxH3Transformer;

namespace {

// Two ranks in one process: each exchange hands each rank the other's
// bytes (seq checked alike).
struct Link {
  std::mutex mu;
  std::condition_variable cv;
  std::map<std::uint64_t, std::vector<std::uint8_t>> part[2];
};

struct Chan final : TpChannel {
  std::shared_ptr<Link> link;
  int me = 0;
  Chan(std::shared_ptr<Link> l, int r) : link(std::move(l)), me(r) {}
  bool exchange(const void* partial, void* others, std::size_t bytes,
                TpDType, std::uint64_t seq, std::string* err) override
  {
    const auto* p = static_cast<const std::uint8_t*>(partial);
    std::unique_lock<std::mutex> lk(link->mu);
    link->part[me][seq].assign(p, p + bytes);
    link->cv.notify_all();
    auto& theirs = link->part[1 - me];
    if (!link->cv.wait_for(lk, std::chrono::seconds(60),
                           [&]() { return theirs.count(seq) > 0; })) {
      if (err != nullptr) { *err = "the peer never came"; }
      return false;
    }
    if (theirs[seq].size() != bytes) {
      if (err != nullptr) { *err = "the ranks are out of step"; }
      return false;
    }
    std::memcpy(others, theirs[seq].data(), bytes);
    return true;
  }
};

SharedBuffer
up_bf16_(metal_compute::MetalCompute* mc, const std::vector<float>& v)
{
  SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) {
    d[i] = train::f32_to_bf16(v[i]);
  }
  return b;
}

// One model -- whole, or rank `rank` of two -- its trainer and adapter.
struct Side {
  Session sess;
  metal_compute::MetalCompute* mc = nullptr;
  T::Config cfg;
  std::unique_ptr<T> model;
  std::unique_ptr<MiniMaxH3Trainer> trainer;
  train::LoraParams params;
  h3t::PackedLayout L;
  SharedBuffer text, vx0, vnz, ax0, anz;

  bool
  load(const fs::path& f, int rank, std::string* err)
  {
    mc = sess.metal_compute();
    if (mc == nullptr) { return false; }
    if (!T::config_from_json(f.string(), cfg, err)) { return false; }
    // Pruned at load: a rank holds no AdaLN to run unbaked, and the whole
    // model is run the same way.
    cfg.prune_adaln = true;
    if (rank >= 0) {
      cfg.tp = TpShare{};
      cfg.tp.rank = rank;
      cfg.tp.ranks = 2;
      cfg.tp.n_heads = cfg.n_heads;
      cfg.tp.n_ffn = cfg.ffn;
      cfg.tp.head0 = rank == 0 ? 0 : cfg.n_heads / 2;
      cfg.tp.head1 = rank == 0 ? cfg.n_heads / 2 : cfg.n_heads;
      cfg.tp.ffn0 = rank == 0 ? 0 : cfg.ffn / 2;
      cfg.tp.ffn1 = rank == 0 ? cfg.ffn / 2 : cfg.ffn;
    }
    model = T::load(f.string(), mc, cfg, false);
    if (model == nullptr) { *err = "the model did not load"; return false; }
    model->set_gemm_route(T::GemmRoute::kSteelBm32);
    trainer = MiniMaxH3Trainer::create(model.get(), err);
    if (trainer == nullptr) { return false; }
    const std::vector<train::ModuleShape> mods =
        trainer->modules("attn_ff", true);
    if (!params.create(mc, mods, 4, 4.0f, err)) { return false; }
    if (rank >= 0) {
      std::map<std::string, train::ModuleCut> cuts;
      if (!trainer->split_cuts(mods, &cuts, err) ||
          !params.set_cuts(cuts, err)) {
        return false;
      }
    }
    params.init_random(9);
    if (trainer->bind(&params, err) == 0) { return false; }
    const std::vector<int> tags(8, h3t::kTextTag);
    if (!h3t::build_packed_sequence(tags, 2, 8, 8, 4, 2, 2,
                                    h3t::kAudioChannels, {}, &L)) {
      *err = "layout";
      return false;
    }
    auto rnd = [&](std::size_t n, unsigned seed, float sd) {
      std::mt19937 g(seed);
      std::normal_distribution<float> nd(0.0f, sd);
      std::vector<float> v(n);
      for (float& x : v) { x = nd(g); }
      return up_bf16_(mc, v);
    };
    text = rnd((std::size_t)8 * cfg.text_dim, 1, 0.6f);
    vx0 = rnd((std::size_t)L.num_video_rows * 96, 2, 1.0f);
    vnz = rnd((std::size_t)L.num_video_rows * 96, 3, 1.0f);
    ax0 = rnd((std::size_t)L.num_audio_rows * 32, 4, 1.0f);
    anz = rnd((std::size_t)L.num_audio_rows * 32, 5, 1.0f);
    return true;
  }

  MiniMaxH3Trainer::Sample
  sample() const
  {
    MiniMaxH3Trainer::Sample s;
    s.layout = &L;
    s.text = &text;
    s.video_x0 = &vx0; s.video_noise = &vnz;
    s.audio_x0 = &ax0; s.audio_noise = &anz;
    s.sigma_video = 0.6f;
    s.sigma_audio = 0.4f;
    return s;
  }

  // Module m's factor `b` (else A) from `arena` (grads or masters) in the
  // whole module's file layout, this side's share in its place, zeros
  // elsewhere; `cut` null for the whole side.
  std::vector<float>
  whole_of(const train::LoraParams::Module& m, const train::ModuleCut* cut,
           bool b, const SharedBuffer& arena) const
  {
    const int r = params.rank();
    const auto* src = static_cast<const float*>(arena.contents());
    if (!b) {
      const int wk = cut != nullptr ? cut->whole_k : m.k;
      const int c0 = cut != nullptr ? cut->a_col0 : 0;
      std::vector<float> v((std::size_t)r * wk, 0.0f);
      for (int i = 0; i < r; ++i) {
        std::memcpy(v.data() + (std::size_t)i * wk + c0,
                    src + m.a_off + (std::size_t)i * m.k, (std::size_t)m.k * 4);
      }
      return v;
    }
    const int wn = cut != nullptr ? cut->whole_n : m.n;
    std::vector<float> v((std::size_t)wn * r, 0.0f);
    for (int i = 0; i < m.n; ++i) {
      const std::size_t s =
          m.b_rows.empty() ? m.b_off + (std::size_t)i * r
                           : (std::size_t)m.b_rows[(std::size_t)i] * r;
      const int d = cut == nullptr || cut->b_src.empty()
                        ? i : cut->b_src[(std::size_t)i];
      std::memcpy(v.data() + (std::size_t)d * r, src + s, (std::size_t)r * 4);
    }
    return v;
  }
};

double
rel_(const std::vector<float>& a, const std::vector<float>& b)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double e = (double)a[i] - (double)b[i];
    num += e * e;
    den += (double)b[i] * (double)b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

// What the stage does after a rank's backward: the shared factors'
// gradients summed across the ranks; their sum of squares back.
bool
sync_(Side& s, double* repl_sq, std::string* err)
{
  const auto spans = s.params.shared_grad_spans();
  std::size_t n = 0;
  for (const auto& sp : spans) { n += sp.n; }
  std::vector<float> mine(n), others(n);
  auto* ga = static_cast<float*>(s.params.a_grad().contents());
  auto* gb = static_cast<float*>(s.params.b_grad().contents());
  std::size_t at = 0;
  for (const auto& sp : spans) {
    std::memcpy(mine.data() + at, (sp.b ? gb : ga) + sp.off, sp.n * 4);
    at += sp.n;
  }
  if (!s.model->tp_bulk(mine.data(), others.data(), n * 4, TpDType::kF32,
                        err)) {
    return false;
  }
  double sq = 0.0;
  at = 0;
  for (const auto& sp : spans) {
    float* d = (sp.b ? gb : ga) + sp.off;
    for (std::size_t i = 0; i < sp.n; ++i) {
      d[i] = mine[at + i] + others[at + i];
      sq += (double)d[i] * d[i];
    }
    at += sp.n;
  }
  *repl_sq = sq;
  return true;
}

std::vector<std::uint8_t>
file_bytes_(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
}

}  // namespace

// A SPLIT STEP IS THE UNSPLIT ONE. Two ranks of the synthetic H3 -- a
// head and 256 feed-forward rows each -- train one sample through an
// in-process link, each from its share of the same adapter (B made
// non-zero, so every factor has a gradient), against the whole model:
//   - the adapter: a rank initialized from the same seed holds the whole
//     run's values at its share, a whole file loaded is cut to it, and
//     the two shares gathered write that file back byte for byte;
//   - the loss alike on both ranks and the whole model's;
//   - every gradient: a rank's own factors (q|k|v's and fc1's B rows,
//     the out projection's and fc2's A columns) its slice of the whole
//     model's, the SHARED ones (their other factor) the whole model's
//     once summed, identical on both ranks;
//   - an AdamW step later, the masters again the whole model's slices --
//     the norm clipped by is every rank's (Optimizer::set_global_sq).
// The bar is the split's rounding: a rank's partial sums in bf16, added
// across ranks, where the whole model sums once.
TEST(minimax_h3_train, a_split_step_is_the_unsplit_one)
{
  const fs::path f = fs::temp_directory_path() /
                     ("vpipe-h3-split-" + std::to_string(::getpid()) +
                      ".safetensors");
  const fs::path ad = fs::temp_directory_path() /
                      ("vpipe-h3-split-ad-" + std::to_string(::getpid()) +
                       ".safetensors");
  const fs::path back = fs::temp_directory_path() /
                        ("vpipe-h3-split-back-" + std::to_string(::getpid()) +
                         ".safetensors");
  ASSERT_TRUE(h3_synthetic::write_synthetic_(f));
  std::string err;
  Side whole;
  ASSERT_TRUE(whole.load(f, -1, &err));
  if (!whole.trainer) { std::printf("  %s\n", err.c_str()); return; }
  // B made non-zero, so the A factors have gradients too.
  {
    std::mt19937 g(31);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    auto* b = static_cast<float*>(whole.params.b_master().contents());
    for (std::size_t i = 0; i < whole.params.b_elems(); ++i) { b[i] = nd(g); }
    whole.params.refresh_shadows_cpu();
  }
  ASSERT_TRUE(whole.params.save(ad.string(), {}, &err));
  // Read back, so the whole model starts from the file's bf16 values too.
  {
    int loaded = 0;
    ASSERT_TRUE(whole.params.load(ad.string(), &loaded, &err));
  }

  Side rk[2];
  for (int r = 0; r < 2; ++r) {
    ASSERT_TRUE(rk[r].load(f, r, &err));
    if (!rk[r].trainer) { std::printf("  rank %d: %s\n", r, err.c_str()); }
  }
  // The same seed: a rank's A is the whole run's at its share.
  {
    double d = 0.0;
    Side fresh;
    ASSERT_TRUE(fresh.load(f, -1, &err));
    for (int r = 0; r < 2; ++r) {
      const auto& mods = rk[r].params.modules();
      std::map<std::string, train::ModuleCut> cuts;
      ASSERT_TRUE(rk[r].trainer->split_cuts(
          rk[r].trainer->modules("attn_ff", true), &cuts, &err));
      for (std::size_t i = 0; i < mods.size(); ++i) {
        const auto& cut = cuts.at(mods[i].name);
        const auto* wm = fresh.params.find(mods[i].name);
        const std::vector<float> a =
            rk[r].whole_of(mods[i], &cut, false, rk[r].params.a_master());
        const std::vector<float> w =
            fresh.whole_of(*wm, nullptr, false, fresh.params.a_master());
        for (std::size_t k = 0; k < a.size(); ++k) {
          if (a[k] != 0.0f) { d = std::max(d, (double)std::fabs(a[k] - w[k])); }
        }
      }
    }
    std::printf("[minimax_h3_train] split init: worst |A - whole A| %.3g\n",
                d);
    EXPECT_TRUE(d == 0.0);
  }
  auto link = std::make_shared<Link>();
  for (int r = 0; r < 2; ++r) {
    int loaded = 0;
    ASSERT_TRUE(rk[r].params.load(ad.string(), &loaded, &err));
    EXPECT_TRUE(loaded == (int)rk[r].params.modules().size());
    ASSERT_TRUE(rk[r].model->begin_tp_generation(
        std::make_shared<Chan>(link, r), &err));
  }
  // Gathered, the two shares are the file they were cut from.
  {
    bool ok[2] = {false, false};
    auto save = [&](int r) {
      auto gather = [&](std::vector<float>& v, std::string* e) {
        std::vector<float> o(v.size());
        if (!rk[r].model->tp_bulk(v.data(), o.data(), v.size() * 4,
                                  TpDType::kF32, e)) {
          return false;
        }
        for (std::size_t i = 0; i < v.size(); ++i) { v[i] += o[i]; }
        return true;
      };
      std::string e;
      ok[r] = rk[r].params.save_gathered(back.string(), {}, r == 0, r == 0,
                                         gather, &e);
      if (!ok[r]) { std::printf("  rank %d gather: %s\n", r, e.c_str()); }
    };
    std::thread t(save, 1);
    save(0);
    t.join();
    ASSERT_TRUE(ok[0] && ok[1]);
    const bool same = file_bytes_(back) == file_bytes_(ad);
    std::printf("[minimax_h3_train] the shares gathered: %s the adapter "
                "they were cut from\n", same ? "BYTE FOR BYTE" : "NOT");
    EXPECT_TRUE(same);
  }

  // One step each.
  MiniMaxH3Trainer::Result wres, rres[2];
  ASSERT_TRUE(whole.trainer->step(whole.sample(), &wres, &err));
  bool ok[2] = {false, false};
  double repl[2] = {0.0, 0.0};
  auto step = [&](int r) {
    std::string e;
    ok[r] = rk[r].trainer->step(rk[r].sample(), &rres[r], &e) &&
            sync_(rk[r], &repl[r], &e);
    if (!ok[r]) { std::printf("  rank %d step: %s\n", r, e.c_str()); }
  };
  {
    std::thread t(step, 1);
    step(0);
    t.join();
  }
  ASSERT_TRUE(ok[0] && ok[1]);
  std::printf("[minimax_h3_train] loss: whole %.6f, ranks %.6f / %.6f\n",
              wres.loss, rres[0].loss, rres[1].loss);
  EXPECT_TRUE(rres[0].loss == rres[1].loss);
  EXPECT_TRUE(std::fabs(rres[0].loss - wres.loss) <
              1e-2 * std::fabs(wres.loss));

  // Every module's gradients: the ranks' own factors against the whole
  // model's slices, the shared ones against its whole.
  constexpr double kBar = 5e-2;
  double worst_own = 0.0, worst_shared = 0.0;
  bool shared_same = true;
  std::map<std::string, train::ModuleCut> cuts[2];
  for (int r = 0; r < 2; ++r) {
    ASSERT_TRUE(rk[r].trainer->split_cuts(
        rk[r].trainer->modules("attn_ff", true), &cuts[r], &err));
  }
  for (const auto& wm : whole.params.modules()) {
    std::vector<float> wa = whole.whole_of(wm, nullptr, false,
                                           whole.params.a_grad());
    std::vector<float> wb = whole.whole_of(wm, nullptr, true,
                                           whole.params.b_grad());
    std::vector<float> sa(wa.size(), 0.0f), sb(wb.size(), 0.0f);
    std::vector<float> share_a[2], share_b[2];
    const train::ModuleCut* c0 = nullptr;
    for (int r = 0; r < 2; ++r) {
      const auto* m = rk[r].params.find(wm.name);
      ASSERT_TRUE(m != nullptr);
      const auto& cut = cuts[r].at(wm.name);
      c0 = &cut;
      share_a[r] = rk[r].whole_of(*m, &cut, false, rk[r].params.a_grad());
      share_b[r] = rk[r].whole_of(*m, &cut, true, rk[r].params.b_grad());
    }
    // A shared factor: both ranks hold the whole sum, alike; the other
    // factor: the two shares tile the whole.
    if (c0->a_shared) {
      shared_same = shared_same && share_a[0] == share_a[1];
      sa = share_a[0];
      for (std::size_t i = 0; i < sb.size(); ++i) {
        sb[i] = share_b[0][i] + share_b[1][i];
      }
      worst_shared = std::max(worst_shared, rel_(sa, wa));
      worst_own = std::max(worst_own, rel_(sb, wb));
    } else {
      shared_same = shared_same && share_b[0] == share_b[1];
      sb = share_b[0];
      for (std::size_t i = 0; i < sa.size(); ++i) {
        sa[i] = share_a[0][i] + share_a[1][i];
      }
      worst_shared = std::max(worst_shared, rel_(sb, wb));
      worst_own = std::max(worst_own, rel_(sa, wa));
    }
  }
  std::printf("[minimax_h3_train] gradients, split vs whole: own factors "
              "rel-L2 %.3e at worst, shared %.3e; shared alike on both "
              "ranks: %s\n", worst_own, worst_shared,
              shared_same ? "yes" : "NO");
  EXPECT_TRUE(worst_own < kBar);
  EXPECT_TRUE(worst_shared < kBar);
  EXPECT_TRUE(shared_same);

  // An SGD step, clipped by the norm every rank agrees on: its update is
  // the gradient times one scale, so a norm taken over one rank alone
  // would show as a wrong-sized update. (Adam's first step is the sign
  // of each gradient: near-zero ones flip on any rounding.)
  const std::vector<float> wa0(
      static_cast<const float*>(whole.params.a_master().contents()),
      static_cast<const float*>(whole.params.a_master().contents()) +
          whole.params.a_elems());
  const std::vector<float> wb0(
      static_cast<const float*>(whole.params.b_master().contents()),
      static_cast<const float*>(whole.params.b_master().contents()) +
          whole.params.b_elems());
  std::vector<float> ra0[2], rb0[2];
  for (int r = 0; r < 2; ++r) {
    const auto* a =
        static_cast<const float*>(rk[r].params.a_master().contents());
    const auto* b =
        static_cast<const float*>(rk[r].params.b_master().contents());
    ra0[r].assign(a, a + rk[r].params.a_elems());
    rb0[r].assign(b, b + rk[r].params.b_elems());
  }
  train::OptimizerSpec spec;
  spec.optimizer = "sgd";
  spec.lr = 1e-2;
  spec.warmup_steps = 0;
  spec.weight_decay = 0.0;
  spec.grad_clip = 1.0;    // under the norm: the clip -- and so the norm
                           // -- sizes the update (f32 holds it whole)
  {
    train::TrainOps ops;
    ASSERT_TRUE(ops.init(whole.mc, true, &err));
    train::Optimizer o;
    ASSERT_TRUE(o.init(&ops, &whole.params, spec, 10, &err));
    double gn = 0.0, lr = 0.0;
    ASSERT_TRUE(o.step(1, &gn, &lr, &err));
    std::printf("[minimax_h3_train] whole |g| %.6g\n", gn);
  }
  double gns[2] = {0.0, 0.0};
  auto opt = [&](int r) {
    std::string e;
    train::TrainOps ops;
    train::Optimizer o;
    ok[r] = ops.init(rk[r].mc, true, &e) &&
            o.init(&ops, &rk[r].params, spec, 10, &e);
    if (ok[r]) {
      o.set_global_sq([&, r](double local, double* g, std::string* ee) {
        double other = 0.0;
        if (!rk[r].model->tp_bulk(&local, &other, 8, TpDType::kF32, ee)) {
          return false;
        }
        *g = local + other - repl[r];
        return true;
      });
      double lr = 0.0;
      ok[r] = o.step(1, &gns[r], &lr, &e);
    }
    if (!ok[r]) { std::printf("  rank %d optimizer: %s\n", r, e.c_str()); }
  };
  {
    std::thread t(opt, 1);
    opt(0);
    t.join();
  }
  ASSERT_TRUE(ok[0] && ok[1]);
  std::printf("[minimax_h3_train] ranks |g| %.6g / %.6g\n", gns[0], gns[1]);
  EXPECT_TRUE(gns[0] == gns[1]);
  // The updates: masters after less before, in arena place.
  auto delta = [](const SharedBuffer& now, const std::vector<float>& before,
                  SharedBuffer* out, metal_compute::MetalCompute* mc) {
    *out = mc->make_shared_buffer(before.size() * 4);
    const auto* n = static_cast<const float*>(now.contents());
    auto* d = static_cast<float*>(out->contents());
    for (std::size_t i = 0; i < before.size(); ++i) { d[i] = n[i] - before[i]; }
  };
  SharedBuffer wda, wdb, rda[2], rdb[2];
  delta(whole.params.a_master(), wa0, &wda, whole.mc);
  delta(whole.params.b_master(), wb0, &wdb, whole.mc);
  for (int r = 0; r < 2; ++r) {
    delta(rk[r].params.a_master(), ra0[r], &rda[r], rk[r].mc);
    delta(rk[r].params.b_master(), rb0[r], &rdb[r], rk[r].mc);
  }
  double worst_m = 0.0;
  for (const auto& wm : whole.params.modules()) {
    for (bool b : {false, true}) {
      std::vector<float> w = whole.whole_of(wm, nullptr, b, b ? wdb : wda);
      std::vector<float> s(w.size(), 0.0f);
      for (int r = 0; r < 2; ++r) {
        const auto* m = rk[r].params.find(wm.name);
        const auto& cut = cuts[r].at(wm.name);
        const bool shared = b ? cut.b_shared : cut.a_shared;
        if (shared && r == 1) { continue; }
        const std::vector<float> v =
            rk[r].whole_of(*m, &cut, b, b ? rdb[r] : rda[r]);
        for (std::size_t i = 0; i < s.size(); ++i) { s[i] += v[i]; }
      }
      worst_m = std::max(worst_m, rel_(s, w));
    }
  }
  std::printf("[minimax_h3_train] a clipped SGD step's update, split vs "
              "whole: rel-L2 %.3e at worst\n", worst_m);
  EXPECT_TRUE(worst_m < kBar);
  for (int r = 0; r < 2; ++r) { rk[r].model->end_tp_generation(); }
  fs::remove(f);
  fs::remove(ad);
  fs::remove(back);
}

namespace {

// One split step of the two ranks against the whole model: the loss,
// the worst gradient's rel-L2 (each module's factors as the whole
// model's), and every gradient the ranks hold, in order. `keep`: both
// ranks keep their activations (recompute: never).
struct SplitRun {
  double loss = 0.0, whole_loss = 0.0, worst = 1.0;
  bool ok = false;
  std::vector<float> grads;
};

SplitRun
split_vs_whole_(bool keep)
{
  SplitRun out;
  const std::string tag = std::to_string(::getpid());
  const fs::path f = fs::temp_directory_path() /
                     ("vpipe-h3-splitk-" + tag + ".safetensors");
  const fs::path ad = fs::temp_directory_path() /
                      ("vpipe-h3-splitk-ad-" + tag + ".safetensors");
  if (!h3_synthetic::write_synthetic_(f)) { return out; }
  std::string err;
  Side whole;
  if (!whole.load(f, -1, &err) || !whole.trainer) {
    std::printf("  %s\n", err.c_str());
    return out;
  }
  {
    std::mt19937 g(31);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    auto* b = static_cast<float*>(whole.params.b_master().contents());
    for (std::size_t i = 0; i < whole.params.b_elems(); ++i) { b[i] = nd(g); }
    whole.params.refresh_shadows_cpu();
  }
  int loaded = 0;
  if (!whole.params.save(ad.string(), {}, &err) ||
      !whole.params.load(ad.string(), &loaded, &err)) {
    return out;
  }
  Side rk[2];
  auto link = std::make_shared<Link>();
  for (int r = 0; r < 2; ++r) {
    if (!rk[r].load(f, r, &err) ||
        !rk[r].params.load(ad.string(), &loaded, &err)) {
      std::printf("  rank %d: %s\n", r, err.c_str());
      return out;
    }
    rk[r].trainer->set_keep_all(keep);
    if (!rk[r].model->begin_tp_generation(std::make_shared<Chan>(link, r),
                                          &err)) {
      return out;
    }
  }
  MiniMaxH3Trainer::Result wres, rres[2];
  if (!whole.trainer->step(whole.sample(), &wres, &err)) { return out; }
  bool ok[2] = {false, false};
  double repl[2] = {0.0, 0.0};
  auto step = [&](int r) {
    std::string e;
    ok[r] = rk[r].trainer->step(rk[r].sample(), &rres[r], &e) &&
            sync_(rk[r], &repl[r], &e);
    if (!ok[r]) { std::printf("  rank %d step: %s\n", r, e.c_str()); }
  };
  {
    std::thread t(step, 1);
    step(0);
    t.join();
  }
  if (!ok[0] || !ok[1] || rres[0].loss != rres[1].loss) { return out; }
  out.loss = rres[0].loss;
  out.whole_loss = wres.loss;
  std::map<std::string, train::ModuleCut> cuts[2];
  for (int r = 0; r < 2; ++r) {
    if (!rk[r].trainer->split_cuts(rk[r].trainer->modules("attn_ff", true),
                                   &cuts[r], &err)) {
      return out;
    }
  }
  double worst = 0.0;
  for (const auto& wm : whole.params.modules()) {
    const std::vector<float> wa = whole.whole_of(wm, nullptr, false,
                                                 whole.params.a_grad());
    const std::vector<float> wb = whole.whole_of(wm, nullptr, true,
                                                 whole.params.b_grad());
    std::vector<float> sa(wa.size(), 0.0f), sb(wb.size(), 0.0f);
    bool a_shared = false;
    std::vector<float> ga[2], gb[2];
    for (int r = 0; r < 2; ++r) {
      const auto* m = rk[r].params.find(wm.name);
      if (m == nullptr) { return out; }
      const auto& cut = cuts[r].at(wm.name);
      a_shared = cut.a_shared;
      ga[r] = rk[r].whole_of(*m, &cut, false, rk[r].params.a_grad());
      gb[r] = rk[r].whole_of(*m, &cut, true, rk[r].params.b_grad());
    }
    for (std::size_t i = 0; i < sa.size(); ++i) {
      sa[i] = a_shared ? ga[0][i] : ga[0][i] + ga[1][i];
    }
    for (std::size_t i = 0; i < sb.size(); ++i) {
      sb[i] = a_shared ? gb[0][i] + gb[1][i] : gb[0][i];
    }
    worst = std::max({worst, rel_(sa, wa), rel_(sb, wb)});
    out.grads.insert(out.grads.end(), sa.begin(), sa.end());
    out.grads.insert(out.grads.end(), sb.begin(), sb.end());
  }
  out.worst = worst;
  out.ok = true;
  for (int r = 0; r < 2; ++r) { rk[r].model->end_tp_generation(); }
  fs::remove(f);
  fs::remove(ad);
  return out;
}

}  // namespace

// A SPLIT STEP KEEPING ITS ACTIVATIONS (recompute: never on both ranks)
// is the unsplit one too: the trainer's own block forward completes the
// out projection's and fc2's partial sums, as the model's forward does,
// and the backward reads the kept share. Against the whole model
// recomputing, the same bar as above.
TEST(minimax_h3_train, a_split_step_keeping_its_activations_is_the_unsplit_one)
{
  const SplitRun r = split_vs_whole_(true);
  ASSERT_TRUE(r.ok);
  std::printf("[minimax_h3_train] kept: loss whole %.6f, ranks %.6f; "
              "gradients rel-L2 %.3e at worst\n", r.whole_loss, r.loss,
              r.worst);
  EXPECT_TRUE(std::fabs(r.loss - r.whole_loss) <
              1e-2 * std::fabs(r.whole_loss));
  EXPECT_TRUE(r.worst < 5e-2);
}

namespace {

// A FAMILY'S SPLIT STEP against its whole one, through the LoRA trainer
// interface every family implements: `make(-1)` the whole model, `make(r)`
// rank r of two (its share and channel left to split_begin here). One
// adapter -- B made non-zero, so every factor has a gradient -- saved
// from the whole model and loaded on each side; one step of `sample`
// each; a rank's shared factors' gradients summed across the two, as the
// stage does. The worst rel-L2 of any module's gradients, each in the
// whole module's layout (the two shares tiled, a shared one once).
struct FamilySplit {
  bool ok = false;
  double whole_loss = 0.0, loss[2] = {0.0, 0.0}, worst = 1.0;
  std::string err;
};

FamilySplit
family_split_step_(
    const std::function<std::unique_ptr<train::LoraTrainer>(int,
                                                            std::string*)>&
        make,
    const std::function<train::TrainSample(train::LoraTrainer&)>& sample,
    metal_compute::MetalCompute* mc, const std::string& targets,
    float b_std = 0.05f)
{
  FamilySplit out;
  const fs::path ad = fs::temp_directory_path() /
                      ("vpipe-family-split-" + std::to_string(::getpid()) +
                       ".safetensors");
  std::unique_ptr<train::LoraTrainer> whole = make(-1, &out.err);
  if (!whole) { return out; }
  train::LoraParams wp;
  if (!wp.create(mc, whole->modules(targets), 4, 4.0f, &out.err)) {
    return out;
  }
  wp.init_random(9);
  {
    std::mt19937 g(31);
    std::normal_distribution<float> nd(0.0f, b_std);
    auto* b = static_cast<float*>(wp.b_master().contents());
    for (std::size_t i = 0; i < wp.b_elems(); ++i) { b[i] = nd(g); }
    wp.refresh_shadows_cpu();
  }
  int loaded = 0;
  if (!wp.save(ad.string(), {}, &out.err) ||
      !wp.load(ad.string(), &loaded, &out.err) ||
      whole->bind(&wp, &out.err) <= 0) {
    return out;
  }
  train::StepStats ws;
  if (!whole->step(sample(*whole), &ws, &out.err)) { return out; }
  out.whole_loss = ws.loss;

  std::unique_ptr<train::LoraTrainer> rk[2];
  train::LoraParams rp[2];
  std::map<std::string, train::ModuleCut> cuts[2];
  auto link = std::make_shared<Link>();
  for (int r = 0; r < 2; ++r) {
    rk[r] = make(r, &out.err);
    if (!rk[r]) { return out; }
    const auto mods = rk[r]->modules(targets);
    if (!rp[r].create(mc, mods, 4, 4.0f, &out.err) ||
        !rk[r]->split_cuts(mods, &cuts[r], &out.err) ||
        !rp[r].set_cuts(cuts[r], &out.err) ||
        !rp[r].load(ad.string(), &loaded, &out.err) ||
        rk[r]->bind(&rp[r], &out.err) <= 0 ||
        !rk[r]->split_begin(std::make_shared<Chan>(link, r), 0, &out.err)) {
      return out;
    }
  }
  bool ok[2] = {false, false};
  std::string e[2];
  // Each rank's sample made here, one after the other: `sample` is not
  // the ranks' threads' to share.
  const train::TrainSample rs[2] = {sample(*rk[0]), sample(*rk[1])};
  auto step = [&](int r) {
    train::StepStats st;
    ok[r] = rk[r]->step(rs[r], &st, &e[r]);
    out.loss[r] = st.loss;
    if (!ok[r]) { return; }
    // The shared factors' gradients summed, as the stage's
    // tp_sync_gradients_.
    const auto spans = rp[r].shared_grad_spans();
    std::size_t n = 0;
    for (const auto& sp : spans) { n += sp.n; }
    std::vector<float> mine(n), others(n);
    auto* ga = static_cast<float*>(rp[r].a_grad().contents());
    auto* gb = static_cast<float*>(rp[r].b_grad().contents());
    std::size_t at = 0;
    for (const auto& sp : spans) {
      std::memcpy(mine.data() + at, (sp.b ? gb : ga) + sp.off, sp.n * 4);
      at += sp.n;
    }
    ok[r] = rk[r]->split_bulk(mine.data(), others.data(), n * 4,
                              TpDType::kF32, &e[r]);
    at = 0;
    for (const auto& sp : spans) {
      float* d = (sp.b ? gb : ga) + sp.off;
      for (std::size_t i = 0; i < sp.n; ++i) {
        d[i] = mine[at + i] + others[at + i];
      }
      at += sp.n;
    }
  };
  {
    std::thread t(step, 1);
    step(0);
    t.join();
  }
  if (!ok[0] || !ok[1]) {
    out.err = e[0].empty() ? e[1] : e[0];
    return out;
  }
  auto whole_of = [](const train::LoraParams& P,
                     const train::LoraParams::Module& m,
                     const train::ModuleCut* cut, bool b,
                     const SharedBuffer& arena) {
    const int r = P.rank();
    const auto* src = static_cast<const float*>(arena.contents());
    if (!b) {
      const int wk = cut != nullptr ? cut->whole_k : m.k;
      const int c0 = cut != nullptr ? cut->a_col0 : 0;
      std::vector<float> v((std::size_t)r * wk, 0.0f);
      for (int i = 0; i < r; ++i) {
        std::memcpy(v.data() + (std::size_t)i * wk + c0,
                    src + m.a_off + (std::size_t)i * m.k,
                    (std::size_t)m.k * 4);
      }
      return v;
    }
    const int wn = cut != nullptr ? cut->whole_n : m.n;
    std::vector<float> v((std::size_t)wn * r, 0.0f);
    for (int i = 0; i < m.n; ++i) {
      const std::size_t s0 =
          m.b_rows.empty() ? m.b_off + (std::size_t)i * r
                           : (std::size_t)m.b_rows[(std::size_t)i] * r;
      const int d = cut == nullptr || cut->b_src.empty()
                        ? i : cut->b_src[(std::size_t)i];
      std::memcpy(v.data() + (std::size_t)d * r, src + s0,
                  (std::size_t)r * 4);
    }
    return v;
  };
  double worst = 0.0;
  for (const auto& wm : wp.modules()) {
    const auto wa = whole_of(wp, wm, nullptr, false, wp.a_grad());
    const auto wb = whole_of(wp, wm, nullptr, true, wp.b_grad());
    std::vector<float> sa(wa.size(), 0.0f), sb(wb.size(), 0.0f);
    for (int r = 0; r < 2; ++r) {
      const auto* m = rp[r].find(wm.name);
      if (m == nullptr) {
        out.err = "no module " + wm.name + " on rank " + std::to_string(r);
        return out;
      }
      const auto& cut = cuts[r].at(wm.name);
      const auto ga = whole_of(rp[r], *m, &cut, false, rp[r].a_grad());
      const auto gb = whole_of(rp[r], *m, &cut, true, rp[r].b_grad());
      for (std::size_t i = 0; i < sa.size(); ++i) {
        if (!cut.a_shared || r == 0) { sa[i] += ga[i]; }
      }
      for (std::size_t i = 0; i < sb.size(); ++i) {
        if (!cut.b_shared || r == 0) { sb[i] += gb[i]; }
      }
    }
    worst = std::max({worst, rel_(sa, wa), rel_(sb, wb)});
  }
  out.worst = worst;
  for (int r = 0; r < 2; ++r) { rk[r]->split_end(); }
  out.ok = true;
  fs::remove(ad);
  return out;
}

}  // namespace

// MINIMAX-H3-IMAGE SPLIT: its audio-less DiT on two ranks -- each its
// share of the heads and feed-forward rows, the per-step output heads
// whole on both -- one step against the whole model's, on the real
// checkpoint (VPIPE_MINIMAX_H3_IMAGE_PATH, the root) cut to two blocks
// and streamed. The same bar as the video DiT's split step.
TEST(minimax_h3_train, minimax_h3_image_split_step_is_the_unsplit_one)
{
  const std::string root = [] {
    const char* e = std::getenv("VPIPE_MINIMAX_H3_IMAGE_PATH");
    return std::string(e != nullptr ? e : "");
  }();
  if (root.empty()) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  h3i::Manifest man;
  std::string err;
  ASSERT_TRUE(h3i::read_manifest(root, &man, &err));
  auto heads = WeightSet::open(man.heads_file, nullptr);
  auto make = [&](int rank, std::string* e)
      -> std::unique_ptr<train::LoraTrainer> {
    MiniMaxH3LoraTrainer::LoadArgs a;
    a.mc = mc;
    a.variant = MiniMaxH3LoraTrainer::Variant::kH3i;
    if (!T::config_from_json(root, a.cfg, e)) { return nullptr; }
    a.cfg.n_layers = std::min(a.cfg.n_layers, 2);
    a.cfg.prune_adaln = true;
    if (rank >= 0) {
      a.cfg.tp = TpShare{};
      a.cfg.tp.rank = rank;
      a.cfg.tp.ranks = 2;
      a.cfg.tp.n_heads = a.cfg.n_heads;
      a.cfg.tp.n_ffn = a.cfg.ffn;
      // An uneven share, as two unlike Macs take: a third of the heads.
      const int h0 = a.cfg.n_heads / 3, f0 = (a.cfg.ffn / 3) / 256 * 256;
      a.cfg.tp.head0 = rank == 0 ? 0 : h0;
      a.cfg.tp.head1 = rank == 0 ? h0 : a.cfg.n_heads;
      a.cfg.tp.ffn0 = rank == 0 ? 0 : f0;
      a.cfg.tp.ffn1 = rank == 0 ? f0 : a.cfg.ffn;
    }
    a.stream_blocks = true;
    a.ws = WeightSet::open(T::resolve_dit_dir(root), nullptr);
    a.h3i_sigmas = man.sigmas;
    a.heads_ws = heads;
    return MiniMaxH3LoraTrainer::load(a, e);
  };
  T::Config c0;
  ASSERT_TRUE(T::config_from_json(root, c0, &err));
  const int G = 16, NT = 24;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> tx((std::size_t)NT * c0.text_dim);
  for (float& v : tx) { v = 0.5f * nd(rng); }
  // Each side's own buffers: a latent packed by its trainer (the same
  // values), the same noise, the same text.
  std::vector<std::unique_ptr<SharedBuffer>> keep;
  auto sample = [&](train::LoraTrainer& t) {
    const int Z = t.latent_channels();
    std::mt19937 lr(11);
    std::normal_distribution<float> ln(0.0f, 1.0f);
    std::vector<float> lat((std::size_t)Z * G * G);
    for (float& v : lat) { v = ln(lr); }
    std::string e;
    keep.push_back(std::make_unique<SharedBuffer>(
        t.pack_latent(lat.data(), Z, 1, G, G, &e)));
    const SharedBuffer& x0 = *keep.back();
    std::mt19937 nr(13);
    std::normal_distribution<float> nn(0.0f, 1.0f);
    std::vector<float> nz(x0.byte_size() / 2);
    for (float& v : nz) { v = nn(nr); }
    keep.push_back(std::make_unique<SharedBuffer>(up_bf16_(mc, nz)));
    const SharedBuffer& noise = *keep.back();
    keep.push_back(std::make_unique<SharedBuffer>(up_bf16_(mc, tx)));
    const SharedBuffer& text = *keep.back();
    train::TrainSample s;
    s.x0 = &x0; s.noise = &noise; s.text = &text; s.text_rows = NT;
    s.grid_h = G; s.grid_w = G; s.frames = 1; s.sigma = 0.5f;
    s.seed = 7919;
    return s;
  };
  const FamilySplit r = family_split_step_(make, sample, mc, "attn_ff");
  if (!r.ok) { std::printf("  %s\n", r.err.c_str()); }
  ASSERT_TRUE(r.ok);
  std::printf("[minimax_h3_train] minimax-h3-image split: loss whole %.6f, "
              "ranks %.6f / %.6f; gradients rel-L2 %.3e at worst\n",
              r.whole_loss, r.loss[0], r.loss[1], r.worst);
  EXPECT_TRUE(r.loss[0] == r.loss[1]);
  EXPECT_TRUE(std::fabs(r.loss[0] - r.whole_loss) <
              1e-2 * std::fabs(r.whole_loss));
  EXPECT_TRUE(r.worst < 5e-2);
}

// tensor-parallel.cc's small random Qwen-Image-2.1 (8 heads, two blocks).
bool qi21_small_for_tests(const fs::path& dir, bool w8);

// QWEN-IMAGE-2.1 SPLIT: its trainer on two ranks -- each a third or the
// rest of the heads and feed-forward rows, its q|k|v and gate|up rows
// and its to_out and down columns the share's, three partial sums a
// block (the recompute's out projection, dh2 and dh1) -- one step against
// the whole model's, on the small random model, bf16 and an 8-bit pack.
TEST(minimax_h3_train, qwen_image_21_split_step_is_the_unsplit_one)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  for (const bool w8 : {false, true}) {
    const fs::path dir = fs::temp_directory_path() /
                         ("vpipe-qi21-train-split-" +
                          std::to_string(::getpid()));
    ASSERT_TRUE(qi21_small_for_tests(dir, w8));
    MetalQwenImage21Transformer::Config c0;
    ASSERT_TRUE(MetalQwenImage21Transformer::Config::read_dims(dir.string(),
                                                               &c0));
    auto make = [&](int rank, std::string* e)
        -> std::unique_ptr<train::LoraTrainer> {
      QwenImage21LoraTrainer::LoadArgs a;
      a.mc = mc;
      a.cfg = c0;
      a.cfg.autotune = false;
      if (rank >= 0) {
        const int FF = c0.hidden * c0.mlp_ratio;
        a.cfg.tp = TpShare{};
        a.cfg.tp.rank = rank;
        a.cfg.tp.ranks = 2;
        a.cfg.tp.n_heads = c0.n_heads;
        a.cfg.tp.n_ffn = FF;
        const int h0 = c0.n_heads / 3, f0 = (FF / 3) / 256 * 256;
        a.cfg.tp.head0 = rank == 0 ? 0 : h0;
        a.cfg.tp.head1 = rank == 0 ? h0 : c0.n_heads;
        a.cfg.tp.ffn0 = rank == 0 ? 0 : f0;
        a.cfg.tp.ffn1 = rank == 0 ? f0 : FF;
      }
      a.ws = WeightSet::open(dir.string(), nullptr);
      return QwenImage21LoraTrainer::load(a, e);
    };
    const int G = 16, NT = 12;
    std::vector<float> tx((std::size_t)NT * c0.txt_dim);
    {
      std::mt19937 rng(7);
      std::normal_distribution<float> nd(0.0f, 0.5f);
      for (float& v : tx) { v = nd(rng); }
    }
    std::vector<std::unique_ptr<SharedBuffer>> keep;
    auto sample = [&](train::LoraTrainer& t) {
      const int Z = t.latent_channels();
      std::mt19937 lr(11);
      std::normal_distribution<float> ln(0.0f, 1.0f);
      std::vector<float> lat((std::size_t)Z * G * G);
      for (float& v : lat) { v = ln(lr); }
      std::string e;
      keep.push_back(std::make_unique<SharedBuffer>(
          t.pack_latent(lat.data(), Z, 1, G, G, &e)));
      const SharedBuffer& x0 = *keep.back();
      std::mt19937 nr(13);
      std::normal_distribution<float> nn(0.0f, 1.0f);
      std::vector<float> nz(x0.byte_size() / 2);
      for (float& v : nz) { v = nn(nr); }
      keep.push_back(std::make_unique<SharedBuffer>(up_bf16_(mc, nz)));
      const SharedBuffer& noise = *keep.back();
      keep.push_back(std::make_unique<SharedBuffer>(up_bf16_(mc, tx)));
      const SharedBuffer& text = *keep.back();
      train::TrainSample s;
      s.x0 = &x0; s.noise = &noise; s.text = &text; s.text_rows = NT;
      s.grid_h = G; s.grid_w = G; s.frames = 1; s.sigma = 0.6f;
      s.seed = 3;
      return s;
    };
    // B large enough that every adapter's term moves the step: one left
    // out of a rank's forward (the out projection's, say) shows.
    const FamilySplit r =
        family_split_step_(make, sample, mc, "attn_ff", 0.5f);
    if (!r.ok) { std::printf("  %s\n", r.err.c_str()); }
    ASSERT_TRUE(r.ok);
    std::printf("[minimax_h3_train] qwen-image-2.1 split (%s): loss whole "
                "%.6f, ranks %.6f / %.6f; gradients rel-L2 %.3e at worst\n",
                w8 ? "w8 pack" : "bf16", r.whole_loss, r.loss[0], r.loss[1],
                r.worst);
    // The bars at about twice the split's own rounding (6-7e-3; the loss
    // 1e-5): q|k|v's input gradient left unsummed reads 3.9e-2, the out
    // projection's adapter left out of a rank's forward 5.3e-2 (and its
    // loss 3.5e-4 off).
    EXPECT_TRUE(r.loss[0] == r.loss[1]);
    EXPECT_TRUE(std::fabs(r.loss[0] - r.whole_loss) <
                1e-4 * std::fabs(r.whole_loss));
    EXPECT_TRUE(r.worst < 1.5e-2);
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
}
