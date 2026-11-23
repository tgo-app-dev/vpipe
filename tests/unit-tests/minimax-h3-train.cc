// minimax-h3-train.cc -- LoRA training on MiniMax-H3, against the
// reference's own autograd, and the AdaLN pruned at load against the
// released form.
//
// The golden (tools/dump_minimax_h3_train_golden.py, over the FL2VA
// transformer) is one training step of the REAL DiT cut to two main
// blocks -- the real token refiner, embeddings, final layer and heads --
// on a T2VA sequence [text | audio | video], a LoRA on all six projections
// of every block (main and refiner, the diffusers decomposition), H3's
// two-modality flow loss, and every factor's gradient as torch's autograd
// computes it. Its fp32 run is the truth, its bf16 run the bar.
//
// Engagement first: every module must end the step with a NONZERO
// gradient -- a projection the backward forgot trains nothing and says
// nothing.
//
//   VPIPE_MINIMAX_H3_TEST_MODEL_PATH=<MiniMax-H3 root or FL2VA>
//   VPIPE_MINIMAX_H3_TRAIN_GOLDEN=<dir> vpipe_test \
//       --filter 'minimax_h3_train*'

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-image.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/minimax-h3/minimax-h3-lora-trainer.h"
#include "generative-models/minimax-h3/minimax-h3-trainer.h"
#include "generative-models/train/lora-params.h"
#include "generative-models/train/lora-trainer.h"
#include "generative-models/train/optimizer.h"
#include "generative-models/train/train-ops.h"
#include "generative-models/weight-set.h"
#include "minimax-h3-synthetic.h"
#include "stages/model-memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::SharedBuffer;
namespace h3t = vpipe::genai::minimax_h3;

namespace {

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return v != nullptr ? std::string(v) : std::string();
}

struct Golden {
  std::filesystem::path dir;
  FlexData man;
  int geti(const char* k, int d) const
  {
    auto o = man.as_object();
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  }
  double getr(const char* k, double d) const
  {
    auto o = man.as_object();
    return o.contains(k) ? o.at(k).as_real(d) : d;
  }
  // grad_bf16_rel[name]
  double grad_bar(const std::string& nm) const
  {
    auto o = man.as_object();
    if (!o.contains("grad_bf16_rel")) { return 0.0; }
    const FlexData g = o.at("grad_bf16_rel");
    auto go = g.as_object();
    return go.contains(nm) ? go.at(nm).as_real(0.0) : 0.0;
  }
  template <typename T>
  std::vector<T>
  raw(const std::string& file) const
  {
    std::ifstream in(dir / file, std::ios::binary);
    std::vector<T> v;
    if (!in) { return v; }
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    v.resize((std::size_t)n / sizeof(T));
    in.read(reinterpret_cast<char*>(v.data()), n);
    return v;
  }
  std::vector<float> f32(const std::string& nm) const
  {
    return raw<float>(nm + ".f32");
  }
};

bool
load_golden_(Golden* g)
{
  const std::string d = env_("VPIPE_MINIMAX_H3_TRAIN_GOLDEN");
  if (d.empty()) { return false; }
  g->dir = d;
  std::ifstream mf(g->dir / "train.json");
  if (!mf) { return false; }
  try {
    g->man = FlexData::from_json(mf);
  } catch (...) {
    return false;
  }
  return g->man.is_object();
}

SharedBuffer
up_bf16_(metal_compute::MetalCompute* mc, const std::vector<float>& v)
{
  SharedBuffer b =
      mc->make_shared_buffer(std::max<std::size_t>(v.size(), 1) * 2);
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) {
    d[i] = train::f32_to_bf16(v[i]);
  }
  return b;
}

std::vector<float>
down_(const SharedBuffer& b, std::size_t off_elems, std::size_t n)
{
  std::vector<float> v(n);
  const auto* d =
      static_cast<const std::uint16_t*>(b.contents()) + off_elems;
  for (std::size_t i = 0; i < n; ++i) { v[i] = train::bf16_to_f32(d[i]); }
  return v;
}

double
rel_l2_(const float* got, const float* want, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double e = (double)got[i] - (double)want[i];
    num += e * e;
    den += (double)want[i] * (double)want[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

// A safetensors file's tensor as f32 (F32 or BF16).
bool
read_tensor_(const std::string& path, const std::string& nm,
             std::vector<float>* out)
{
  std::shared_ptr<WeightSet> ws = WeightSet::open(path, nullptr);
  if (!ws) { return false; }
  const auto* info = ws->src().info(nm);
  if (info == nullptr) { return false; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  std::vector<std::uint8_t> raw(info->nbytes);
  if (!ws->src().read_into(nm, raw.data(), raw.size())) { return false; }
  out->resize(n);
  if (info->dtype == "F32") {
    std::memcpy(out->data(), raw.data(), n * 4);
  } else if (info->dtype == "BF16") {
    const auto* s = reinterpret_cast<const std::uint16_t*>(raw.data());
    for (std::size_t i = 0; i < n; ++i) {
      (*out)[i] = train::bf16_to_f32(s[i]);
    }
  } else {
    return false;
  }
  return true;
}

struct Rig {
  Session sess;
  metal_compute::MetalCompute* mc = nullptr;
  std::string root;
  MetalMiniMaxH3Transformer::Config cfg;
  std::unique_ptr<MetalMiniMaxH3Transformer> model;
  h3t::PackedLayout L;
  SharedBuffer text, vx0, vnz, ax0, anz;
  train::LoraParams params;
  std::unique_ptr<MiniMaxH3Trainer> trainer;
  float sv = 0.0f, sa = 0.0f;
  int bound = 0;

  // `pin`: steel's 32-row GEMM tile for every projection (the bit-exact
  // comparisons: two loads may tune differently); else the engine's.
  bool
  load_model(const Golden& g, bool stream, bool prune, bool pin = true)
  {
    mc = sess.metal_compute();
    root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
    if (mc == nullptr || root.empty()) { return false; }
    std::string cerr;
    if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
      std::printf("  config: %s\n", cerr.c_str());
      return false;
    }
    cfg.n_layers = g.geti("layers", 2);
    cfg.prune_adaln = prune;
    model = MetalMiniMaxH3Transformer::load(root, mc, cfg, stream);
    if (model == nullptr) { return false; }
    if (pin) {
      model->set_gemm_route(
          MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
    }
    return true;
  }

  bool
  build(const Golden& g, bool load_adapter, bool stream = false,
        bool prune = false, bool pin = true)
  {
    if (!load_model(g, stream, prune, pin)) { return false; }
    const int nt = g.geti("n_text", 8);
    const std::vector<int> tags((std::size_t)nt, h3t::kTextTag);
    if (!h3t::build_packed_sequence(tags, g.geti("frames", 2),
                                    g.geti("lh", 12), g.geti("lw", 20),
                                    g.geti("audio_latents", 8), cfg.patch_h,
                                    cfg.patch_w, h3t::kAudioChannels, {},
                                    &L)) {
      return false;
    }
    text = up_bf16_(mc, g.f32("text"));
    vx0 = up_bf16_(mc, g.f32("video_x0"));
    vnz = up_bf16_(mc, g.f32("video_noise"));
    ax0 = up_bf16_(mc, g.f32("audio_x0"));
    anz = up_bf16_(mc, g.f32("audio_noise"));
    sv = (float)g.getr("sigma_video", 0.7);
    sa = (float)g.getr("sigma_audio", 0.5);
    std::string err;
    trainer = MiniMaxH3Trainer::create(model.get(), &err);
    if (trainer == nullptr) {
      std::printf("  trainer: %s\n", err.c_str());
      return false;
    }
    const int r = g.geti("rank", 4);
    if (!params.create(mc, trainer->modules("attn_ff", true), r,
                       (float)g.geti("alpha", r), &err)) {
      return false;
    }
    params.init_random(7);
    // Bound FIRST: the binder sets the row maps the file is read through.
    bound = trainer->bind(&params, &err);
    if (bound == 0) {
      std::printf("  bind: %s\n", err.c_str());
      return false;
    }
    if (load_adapter) {
      int loaded = 0;
      if (!params.load((g.dir / "adapter.safetensors").string(), &loaded,
                       &err) ||
          loaded != (int)params.modules().size()) {
        std::printf("  adapter: %s (%d)\n", err.c_str(), loaded);
        return false;
      }
    }
    return true;
  }

  // Another rig's layout and inputs, aliased.
  void
  share_inputs(const Rig& o)
  {
    auto al = [](const SharedBuffer& b) {
      return b.subview(0, b.byte_size());
    };
    L = o.L;
    text = al(o.text); vx0 = al(o.vx0); vnz = al(o.vnz);
    ax0 = al(o.ax0); anz = al(o.anz);
    sv = o.sv; sa = o.sa;
  }

  MiniMaxH3Trainer::Sample
  sample() const
  {
    MiniMaxH3Trainer::Sample s;
    s.layout = &L;
    s.text = &text;
    s.video_x0 = &vx0;
    s.video_noise = &vnz;
    s.audio_x0 = &ax0;
    s.audio_noise = &anz;
    s.sigma_video = sv;
    s.sigma_audio = sa;
    return s;
  }

  // Module m's gradient B in the FILE's row order (the arena's is the
  // forward's: fused q|k|v, fc1 gate first).
  std::vector<float>
  grad_b(const train::LoraParams::Module& m) const
  {
    const int r = params.rank();
    const auto* gb = static_cast<const float*>(params.b_grad().contents());
    std::vector<float> v((std::size_t)m.n * r);
    for (int i = 0; i < m.n; ++i) {
      const std::size_t src =
          m.b_rows.empty() ? m.b_off + (std::size_t)i * r
                           : (std::size_t)m.b_rows[(std::size_t)i] * r;
      std::memcpy(v.data() + (std::size_t)i * r, gb + src,
                  (std::size_t)r * 4);
    }
    return v;
  }

  std::vector<float>
  grad_a(const train::LoraParams::Module& m) const
  {
    const int r = params.rank();
    const auto* ga = static_cast<const float*>(params.a_grad().contents());
    return std::vector<float>(ga + m.a_off,
                              ga + m.a_off + (std::size_t)r * m.k);
  }
};

}  // namespace

// The layout the trainer is handed is generate-video's own: the golden's
// restatement of build_packed_sequence agrees with it, position by
// position.
TEST(minimax_h3_train, the_golden_layout_is_build_packed_sequence)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  const int nt = g.geti("n_text", 8);
  const std::vector<int> tags((std::size_t)nt, h3t::kTextTag);
  h3t::PackedLayout L;
  ASSERT_TRUE(h3t::build_packed_sequence(tags, g.geti("frames", 2),
                                         g.geti("lh", 12), g.geti("lw", 20),
                                         g.geti("audio_latents", 8), 2, 2,
                                         h3t::kAudioChannels, {}, &L));
  const std::vector<double> pos = g.raw<double>("pos.f64");
  const std::vector<int> gt = g.raw<int>("tags.i32");
  ASSERT_TRUE(pos.size() == L.position_ids.size());
  ASSERT_TRUE(gt.size() == L.token_tags.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < pos.size(); ++i) {
    worst = std::max(worst, std::fabs(pos[i] - L.position_ids[i]));
  }
  int tag_off = 0;
  for (std::size_t i = 0; i < gt.size(); ++i) {
    tag_off += gt[i] != L.token_tags[i];
  }
  std::printf("[minimax_h3_train] layout: %d rows, positions max |d| %.3g, "
              "%d tags off\n", L.seq_len, worst, tag_off);
  EXPECT_TRUE(worst < 1e-9);
  EXPECT_TRUE(tag_off == 0);
  EXPECT_TRUE(L.audio_start == g.geti("audio_start", -1));
  EXPECT_TRUE(L.video_start == g.geti("video_start", -1));
}

// ONE STEP: the loss, the velocities, and every factor's gradient.
TEST(minimax_h3_train, one_step_matches_autograd)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  // The engine's own GEMM routes (the matrix cores on an M5).
  Rig rig;
  ASSERT_TRUE(rig.build(g, /*load_adapter=*/true, false, false,
                        /*pin=*/false));
  if (!rig.trainer) { return; }
  const int nb = g.geti("layers", 2) + rig.cfg.n_refiner;
  EXPECT_TRUE(rig.bound == nb * 6);

  MiniMaxH3Trainer::Result res;
  std::string err;
  ASSERT_TRUE(rig.trainer->step(rig.sample(), &res, &err, true));
  if (!err.empty()) { std::printf("[minimax_h3_train] %s\n", err.c_str()); }

  const double want = g.getr("loss_f32", 0.0);
  const double ref16 = g.getr("loss_bf16", 0.0);
  const double lbar =
      std::max(2.0 * std::fabs(ref16 - want) / want, 1e-2);
  std::printf("[minimax_h3_train] loss %.6f (video %.6f, audio %.6f), "
              "reference %.6f, its bf16 %.6f; %.1f + %.1f ms\n", res.loss,
              res.loss_video, res.loss_audio, want, ref16, res.forward_ms,
              res.backward_ms);
  EXPECT_TRUE(std::fabs(res.loss - want) / want < lbar);

  const int VPE = rig.cfg.video_patch_elems(), AC = rig.cfg.audio_channels;
  const std::vector<float> pv = g.f32("pred_video_f32");
  const std::vector<float> pa = g.f32("pred_audio_f32");
  ASSERT_TRUE(pv.size() == (std::size_t)rig.L.num_video_rows * VPE);
  ASSERT_TRUE(pa.size() == (std::size_t)rig.L.num_audio_rows * AC);
  const std::vector<float> gv = down_(res.pred_video, 0, pv.size());
  const std::vector<float> ga = down_(res.pred_audio, 0, pa.size());
  const double ev = rel_l2_(gv.data(), pv.data(), pv.size());
  const double ea = rel_l2_(ga.data(), pa.data(), pa.size());
  const double bv = g.getr("pred_video_bf16_rel", 0.0);
  const double ba = g.getr("pred_audio_bf16_rel", 0.0);
  std::printf("[minimax_h3_train] velocity rel-L2 video %.5f (ref bf16 "
              "%.5f), audio %.5f (%.5f)\n", ev, bv, ea, ba);
  EXPECT_TRUE(ev < std::max(2.0 * bv, 0.01));
  EXPECT_TRUE(ea < std::max(2.0 * ba, 0.01));

  // Every module's two gradients, against the reference's.
  const std::string gf = (g.dir / "grads_f32.safetensors").string();
  int zero = 0, bad = 0;
  double worst = 0.0, worst_bar = 0.0;
  for (const auto& m : rig.params.modules()) {
    std::vector<float> wa, wb;
    ASSERT_TRUE(read_tensor_(gf, m.name + ".A", &wa));
    ASSERT_TRUE(read_tensor_(gf, m.name + ".B", &wb));
    const std::vector<float> a = rig.grad_a(m);
    const std::vector<float> b = rig.grad_b(m);
    ASSERT_TRUE(wa.size() == a.size() && wb.size() == b.size());
    double na = 0.0, nbn = 0.0;
    for (float x : a) { na += (double)x * x; }
    for (float x : b) { nbn += (double)x * x; }
    if (na == 0.0 || nbn == 0.0) { ++zero; }
    const double e_a = rel_l2_(a.data(), wa.data(), a.size());
    const double e_b = rel_l2_(b.data(), wb.data(), b.size());
    const double bar_a = std::max(2.5 * g.grad_bar(m.name + ".A"), 0.05);
    const double bar_b = std::max(2.5 * g.grad_bar(m.name + ".B"), 0.05);
    worst = std::max({worst, e_a, e_b});
    worst_bar = std::max({worst_bar, e_a / bar_a, e_b / bar_b});
    std::printf("[minimax_h3_train] %-46s dA %.4f (bf16 %.4f)  dB %.4f "
                "(bf16 %.4f)\n", m.name.c_str(), e_a,
                g.grad_bar(m.name + ".A"), e_b, g.grad_bar(m.name + ".B"));
    if (e_a > bar_a || e_b > bar_b) { ++bad; }
  }
  std::printf("[minimax_h3_train] worst gradient rel-L2 %.4f, %.2f of its "
              "bar\n", worst, worst_bar);
  EXPECT_TRUE(zero == 0);
  EXPECT_TRUE(bad == 0);
}

namespace {

std::vector<std::uint16_t>
forward_words_(MetalMiniMaxH3Transformer& m, const Rig& rig)
{
  const int VPE = rig.cfg.video_patch_elems(), AC = rig.cfg.audio_channels;
  // x_t as the trainer forms it.
  auto xt = [&](const SharedBuffer& a, const SharedBuffer& e, float s,
                std::size_t n) {
    SharedBuffer b = rig.mc->make_shared_buffer(n * 2);
    const auto* pa = static_cast<const std::uint16_t*>(a.contents());
    const auto* pe = static_cast<const std::uint16_t*>(e.contents());
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      d[i] = train::f32_to_bf16((1.0f - s) * train::bf16_to_f32(pa[i]) +
                                s * train::bf16_to_f32(pe[i]));
    }
    return b;
  };
  const SharedBuffer v = xt(rig.vx0, rig.vnz, rig.sv,
                            (std::size_t)rig.L.num_video_rows * VPE);
  const SharedBuffer a = xt(rig.ax0, rig.anz, rig.sa,
                            (std::size_t)rig.L.num_audio_rows * AC);
  std::vector<float> uniq;
  std::vector<int> idx;
  h3t::build_row_timesteps(rig.L, 1.0f - rig.sv, 1.0f - rig.sa, 1.0f, &uniq,
                           &idx);
  MetalMiniMaxH3Transformer::Step st;
  st.video = &v;
  st.audio = &a;
  st.text = &rig.text;
  st.layout = &rig.L;
  st.timesteps = &uniq;
  st.row_timestep_index = &idx;
  st.dense_attention = true;
  std::string err;
  const auto out = m.forward(st, &err);
  std::vector<std::uint16_t> w;
  if (out.empty()) {
    std::printf("  forward: %s\n", err.c_str());
    return w;
  }
  const auto* pv = static_cast<const std::uint16_t*>(out.video.contents());
  const auto* pa = static_cast<const std::uint16_t*>(out.audio.contents());
  w.assign(pv, pv + (std::size_t)rig.L.num_video_rows * VPE);
  w.insert(w.end(), pa, pa + (std::size_t)rig.L.num_audio_rows * AC);
  return w;
}

std::size_t
diff_(const std::vector<std::uint16_t>& a,
      const std::vector<std::uint16_t>& b)
{
  if (a.size() != b.size() || a.empty()) { return (std::size_t)-1; }
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) { n += a[i] != b[i]; }
  return n;
}

}  // namespace

// ONE FORWARD. A fresh adapter (B = 0) is exactly the base; and the
// adapter the trainer saves, bound again from the file by the model's own
// binder -- the diffusers decomposition, q|k|v fused and banded for this
// base's grouping, fc1's halves swapped -- runs exactly the forward the
// trainer ran.
TEST(minimax_h3_train, a_saved_adapter_reloads_bit_exact)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  Rig rig;
  ASSERT_TRUE(rig.build(g, /*load_adapter=*/false));
  if (!rig.trainer) { return; }
  std::vector<std::uint16_t> base;
  {
    Rig plain;
    ASSERT_TRUE(plain.load_model(g, false, false));
    plain.share_inputs(rig);
    base = forward_words_(*plain.model, plain);
  }
  const std::vector<std::uint16_t> zero_b = forward_words_(*rig.model, rig);
  const std::size_t d0 = diff_(zero_b, base);
  std::printf("[minimax_h3_train] B = 0 against the base: %zu of %zu "
              "differ\n", d0, base.size());
  EXPECT_TRUE(d0 == 0);

  // The golden's factors (B nonzero), saved, bound again from the file.
  int loaded = 0;
  std::string err;
  ASSERT_TRUE(rig.params.load((g.dir / "adapter.safetensors").string(),
                              &loaded, &err));
  const std::vector<std::uint16_t> live = forward_words_(*rig.model, rig);
  const std::size_t moved = diff_(live, base);
  const std::filesystem::path f =
      std::filesystem::temp_directory_path() / "vpipe_h3_train_reload.st";
  ASSERT_TRUE(rig.params.save(f.string(), {{"vpipe.family", "minimax-h3"}},
                              &err));
  std::vector<std::uint16_t> again;
  {
    Rig file;
    ASSERT_TRUE(file.load_model(g, false, false));
    MetalMiniMaxH3Transformer::LoraSpec sp;
    sp.path = f.string();
    file.model = MetalMiniMaxH3Transformer::load(file.root, file.mc,
                                                 file.cfg, false, {sp});
    ASSERT_TRUE(file.model != nullptr);
    if (!file.model) { return; }
    file.model->set_gemm_route(
        MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
    std::printf("[minimax_h3_train] the file binds %d modules (%d banded)\n",
                file.model->lora_modules(0), file.model->lora_banded(0));
    file.share_inputs(rig);
    again = forward_words_(*file.model, file);
  }
  std::filesystem::remove(f);
  const std::size_t d1 = diff_(again, live);
  std::printf("[minimax_h3_train] the adapter moves %zu of %zu; bound from "
              "its file: %zu differ\n", moved, live.size(), d1);
  EXPECT_TRUE(moved > live.size() / 10);
  EXPECT_TRUE(d1 == 0);
}

// STREAMED == RESIDENT: the backward reads the blocks 1 -> 0 through the
// model's own slot pair, the next one read under the current one's work.
TEST(minimax_h3_train, streamed_equals_resident)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  std::vector<float> ga[2], gb[2];
  double loss[2] = {0.0, 0.0};
  for (int arm = 0; arm < 2; ++arm) {
    Rig rig;
    ASSERT_TRUE(rig.build(g, /*load_adapter=*/true, /*stream=*/arm == 1));
    if (!rig.trainer) { return; }
    MiniMaxH3Trainer::Result res;
    std::string err;
    ASSERT_TRUE(rig.trainer->step(rig.sample(), &res, &err));
    if (!err.empty()) { std::printf("  %s\n", err.c_str()); }
    loss[arm] = res.loss;
    const std::size_t na = rig.params.a_elems(), nb = rig.params.b_elems();
    const auto* a = static_cast<const float*>(rig.params.a_grad().contents());
    const auto* b = static_cast<const float*>(rig.params.b_grad().contents());
    ga[arm].assign(a, a + na);
    gb[arm].assign(b, b + nb);
    std::printf("[minimax_h3_train] %s: loss %.6f, %.1f + %.1f ms\n",
                arm == 0 ? "resident" : "streamed", res.loss, res.forward_ms,
                res.backward_ms);
  }
  std::size_t da = 0, db = 0;
  for (std::size_t i = 0; i < ga[0].size(); ++i) { da += ga[0][i] != ga[1][i]; }
  for (std::size_t i = 0; i < gb[0].size(); ++i) { db += gb[0][i] != gb[1][i]; }
  std::printf("[minimax_h3_train] streamed vs resident: loss %s, %zu of %zu "
              "dA and %zu of %zu dB differ\n",
              loss[0] == loss[1] ? "identical" : "DIFFERS", da, ga[0].size(),
              db, gb[0].size());
  EXPECT_TRUE(loss[0] == loss[1]);
  EXPECT_TRUE(da == 0 && db == 0);
}

// PRUNED AT LOAD (Config::prune_adaln): the released AdaLN folded onto the
// timestep embedding's 8-dim curve, W' = W U and b' = W m + b, each
// forward reading exact coordinates. Against the released form, kept:
// the velocities, and a training step's loss and gradients.
TEST(minimax_h3_train, pruned_at_load_runs_as_the_released_model)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  Rig rel, pru;
  ASSERT_TRUE(rel.build(g, /*load_adapter=*/true, false, /*prune=*/false));
  ASSERT_TRUE(pru.build(g, /*load_adapter=*/true, false, /*prune=*/true));
  if (!rel.trainer || !pru.trainer) { return; }
  ASSERT_TRUE(pru.model->adaln_pruned_at_load());
  EXPECT_TRUE(!rel.model->adaln_pruned_at_load());
  std::printf("[minimax_h3_train] pruned at load: the curve's residual "
              "%.3g\n", pru.model->adaln_prune_residual());
  EXPECT_TRUE(pru.model->adaln_prune_residual() < 1e-4);

  // The modulation tables and the velocities through each.
  MiniMaxH3Trainer::Result r0, r1;
  std::string err;
  ASSERT_TRUE(rel.trainer->step(rel.sample(), &r0, &err, true));
  ASSERT_TRUE(pru.trainer->step(pru.sample(), &r1, &err, true));
  const int VPE = rel.cfg.video_patch_elems();
  const std::size_t nv = (std::size_t)rel.L.num_video_rows * VPE;
  const std::vector<float> v0 = down_(r0.pred_video, 0, nv);
  const std::vector<float> v1 = down_(r1.pred_video, 0, nv);
  const double ev = rel_l2_(v1.data(), v0.data(), nv);
  const std::vector<float> pv = g.f32("pred_video_f32");
  const double e_rel = rel_l2_(v0.data(), pv.data(), nv);
  const double e_pru = rel_l2_(v1.data(), pv.data(), nv);
  std::printf("[minimax_h3_train] velocity: pruned vs released %.5f; from "
              "the fp32 reference released %.5f, pruned %.5f; loss %.6f / "
              "%.6f\n", ev, e_rel, e_pru, r0.loss, r1.loss);
  EXPECT_TRUE(ev < 0.01);
  EXPECT_TRUE(e_pru < std::max(1.5 * e_rel, 0.01));
  // The gradients.
  const auto* a0 = static_cast<const float*>(rel.params.a_grad().contents());
  const auto* a1 = static_cast<const float*>(pru.params.a_grad().contents());
  const auto* b0 = static_cast<const float*>(rel.params.b_grad().contents());
  const auto* b1 = static_cast<const float*>(pru.params.b_grad().contents());
  const double ea = rel_l2_(a1, a0, rel.params.a_elems());
  const double eb = rel_l2_(b1, b0, rel.params.b_elems());
  std::printf("[minimax_h3_train] gradients pruned vs released: dA %.4f, "
              "dB %.4f\n", ea, eb);
  EXPECT_TRUE(ea < 0.05 && eb < 0.05);
}

// ---- a synthetic H3, both q|k|v groupings ---------------------------------

namespace {

namespace fs = std::filesystem;

using h3_synthetic::write_synthetic_;

// One synthetic rig: the model at a grouping, its trainer and adapter, a
// sample over [8 text | 2 x 4 audio | 2 frames of 4 x 4] rows.
struct Syn {
  Session sess;
  metal_compute::MetalCompute* mc = nullptr;
  fs::path file;
  MetalMiniMaxH3Transformer::Config cfg;
  std::unique_ptr<MetalMiniMaxH3Transformer> model;
  h3t::PackedLayout L;
  SharedBuffer text, vx0, vnz, ax0, anz;
  train::LoraParams params;
  std::unique_ptr<MiniMaxH3Trainer> trainer;
  Rig io;   // its rows, as forward_words_ reads them

  bool
  load(const fs::path& f, bool per_head,
       const std::vector<MetalMiniMaxH3Transformer::LoraSpec>& specs = {})
  {
    mc = sess.metal_compute();
    if (mc == nullptr) { return false; }
    file = f;
    std::string err;
    if (!MetalMiniMaxH3Transformer::config_from_json(f.string(), cfg,
                                                     &err)) {
      std::printf("  config: %s\n", err.c_str());
      return false;
    }
    cfg.qkv_per_head = per_head;
    model = MetalMiniMaxH3Transformer::load(f.string(), mc, cfg, false,
                                            specs);
    if (model == nullptr) { return false; }
    model->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
    const std::vector<int> tags(8, h3t::kTextTag);
    if (!h3t::build_packed_sequence(tags, 2, 8, 8, 4, 2, 2,
                                    h3t::kAudioChannels, {}, &L)) {
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
    io.mc = mc;
    io.cfg = cfg;
    io.L = L;
    auto al = [](const SharedBuffer& b) {
      return b.subview(0, b.byte_size());
    };
    io.text = al(text); io.vx0 = al(vx0); io.vnz = al(vnz);
    io.ax0 = al(ax0); io.anz = al(anz);
    io.sv = 0.6f; io.sa = 0.4f;
    return true;
  }

  bool
  train(std::string* err)
  {
    trainer = MiniMaxH3Trainer::create(model.get(), err);
    if (trainer == nullptr) { return false; }
    if (!params.create(mc, trainer->modules("attn_ff", true), 4, 4.0f,
                       err)) {
      return false;
    }
    params.init_random(9);
    return trainer->bind(&params, err) > 0;
  }

  MiniMaxH3Trainer::Sample
  sample() const
  {
    MiniMaxH3Trainer::Sample s;
    s.layout = &L;
    s.text = &text;
    s.video_x0 = &vx0; s.video_noise = &vnz;
    s.audio_x0 = &ax0; s.audio_noise = &anz;
    s.sigma_video = io.sv;
    s.sigma_audio = io.sa;
    return s;
  }
};

}  // namespace

// On a synthetic H3 in BOTH q|k|v groupings -- MiniMaxAI's per head and
// Comfy-Org's flat -- every optimizer lowers the loss over a few steps,
// and the adapter it leaves (B no longer zero), saved in the diffusers
// decomposition and bound again by the model's binder, runs exactly the
// forward the trainer ran: the fused q|k|v placed for each grouping by
// the binder's own qkv_fused_row, fc1's halves swapped back.
TEST(minimax_h3_train, both_groupings_learn_and_reload_bit_exact)
{
  const fs::path f = fs::temp_directory_path() /
                     ("vpipe-h3-train-" + std::to_string(::getpid()) +
                      ".safetensors");
  ASSERT_TRUE(write_synthetic_(f));
  for (bool per_head : {true, false}) {
    for (const char* opt : {"adamw", "prodigy", "lion", "sgd"}) {
      Syn s;
      ASSERT_TRUE(s.load(f, per_head));
      std::string err;
      ASSERT_TRUE(s.train(&err));
      if (!s.trainer) { std::printf("  %s\n", err.c_str()); break; }
      train::TrainOps ops;
      ASSERT_TRUE(ops.init(s.mc, true, &err));
      train::OptimizerSpec spec;
      spec.optimizer = opt;
      spec.lr = std::string(opt) == "adamw" ? 1e-2
                : std::string(opt) == "lion" ? 1e-3
                : std::string(opt) == "sgd" ? 2.0 : 1.0;
      spec.warmup_steps = 0;
      spec.weight_decay = 0.0;
      const int steps = 20;
      train::Optimizer o;
      ASSERT_TRUE(o.init(&ops, &s.params, spec, steps, &err));
      double first = 0.0, last = 0.0;
      for (int t = 0; t < steps; ++t) {
        MiniMaxH3Trainer::Result res;
        ASSERT_TRUE(s.trainer->step(s.sample(), &res, &err));
        double gn = 0.0, lr = 0.0;
        ASSERT_TRUE(o.step(1, &gn, &lr, &err));
        if (t == 0) { first = res.loss; }
        last = res.loss;
      }
      std::printf("[minimax_h3_train] %s %-7s loss %.4f -> %.4f\n",
                  per_head ? "per-head" : "flat    ", opt, first, last);
      EXPECT_TRUE(std::isfinite(last));
      EXPECT_TRUE(last < 0.9 * first);
      if (std::string(opt) != "adamw") { continue; }
      // The adapter, saved and bound again from its file.
      const std::vector<std::uint16_t> live = forward_words_(*s.model, s.io);
      const fs::path ad = fs::temp_directory_path() /
                          ("vpipe-h3-train-ad-" + std::to_string(::getpid()) +
                           ".safetensors");
      ASSERT_TRUE(s.params.save(ad.string(), {}, &err));
      Syn b;
      MetalMiniMaxH3Transformer::LoraSpec sp;
      sp.path = ad.string();
      ASSERT_TRUE(b.load(f, per_head, {sp}));
      const std::vector<std::uint16_t> again = forward_words_(*b.model, b.io);
      Syn base;
      ASSERT_TRUE(base.load(f, per_head));
      const std::vector<std::uint16_t> plain =
          forward_words_(*base.model, base.io);
      fs::remove(ad);
      const std::size_t d = diff_(again, live);
      const std::size_t moved = diff_(live, plain);
      std::printf("[minimax_h3_train] %s: the trained adapter moves %zu of "
                  "%zu; bound from its file (%d modules, %d banded): %zu "
                  "differ\n", per_head ? "per-head" : "flat",
                  moved, live.size(), b.model->lora_modules(0),
                  b.model->lora_banded(0), d);
      EXPECT_TRUE(moved > live.size() / 10);
      EXPECT_TRUE(d == 0);
    }
  }
  fs::remove(f);
}

// ---- the family behind train-lora, on the real checkpoint ------------------

namespace {

template <typename Fn>
std::size_t
device_peak_of_(metal_compute::MetalCompute* mc, Fn&& fn)
{
  const std::size_t base = mc->memory_budget_settled().allocated;
  std::atomic<std::size_t> peak{base};
  std::atomic<bool> done{false};
  std::thread sampler([&] {
    while (!done.load()) {
      const std::size_t a = mc->memory_budget().allocated;
      if (a > peak.load()) { peak.store(a); }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });
  fn();
  done.store(true);
  sampler.join();
  const std::size_t top =
      std::max(peak.load(), mc->memory_budget().allocated);
  return top > base ? top - base : 0;
}

const train::LoraFamily*
h3_family_()
{
  for (const train::LoraFamily* f : train::lora_families()) {
    if (std::string(f->name()) == "minimax-h3") { return f; }
  }
  return nullptr;
}

}  // namespace

// THE PLAN BOOKS WHAT A STEP HOLDS. The stage sizes its stream / keep
// decisions and the residency reserve from plan_arena_bytes -- the
// trainer's arena and the tap forward's own scratch -- before anything
// loads; a step's device allocation, sampled from a thread, must stay
// under it at every geometry: stills at 512^2 and 1024^2, and a two-
// latent clip (5 frames) at 640x352. Each in a fresh load, so the arena
// is built for that geometry alone.
TEST(minimax_h3_train, the_plan_books_what_a_step_holds)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root.empty()) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  struct Geo { int t, h, w; } geos[] = {{1, 32, 32}, {1, 64, 64},
                                        {2, 22, 40}};
  const int kText = 300, kRank = 32;
  for (int keep = 0; keep < 2; ++keep)
  for (const Geo& g : geos) {
    if (keep && g.h != 32) { continue; }   // one geometry kept: it is big
    MiniMaxH3LoraTrainer::LoadArgs a;
    a.mc = mc;
    std::string err;
    ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(root, a.cfg,
                                                            &err));
    a.cfg.n_layers = std::min(a.cfg.n_layers, 2);
    a.ws = WeightSet::open(MetalMiniMaxH3Transformer::resolve_dit_dir(root),
                           nullptr);
    auto t = MiniMaxH3LoraTrainer::load(a, &err);
    ASSERT_TRUE(t != nullptr);
    if (t == nullptr) { std::printf("  %s\n", err.c_str()); return; }
    train::LoraParams params;
    ASSERT_TRUE(params.create(mc, t->modules("attn_ff"), kRank,
                              (float)kRank, &err));
    params.init_random(3);
    ASSERT_TRUE(t->bind(&params, &err) > 0);
    {
      FlexData o = FlexData::make_object();
      o.as_object().insert_or_assign(train::opt::kKeepAll,
                                     FlexData::make_bool(keep != 0));
      t->set_options(o);
    }
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> lat((std::size_t)a.cfg.video_channels * g.t * g.h *
                           g.w);
    for (float& v : lat) { v = nd(rng); }
    SharedBuffer x0 = t->pack_latent(lat.data(), a.cfg.video_channels, g.t,
                                     g.h, g.w, &err);
    ASSERT_TRUE(!x0.empty());
    std::vector<float> nz(x0.byte_size() / 2), tx((std::size_t)kText *
                                                  a.cfg.text_dim);
    for (float& v : nz) { v = nd(rng); }
    for (float& v : tx) { v = 0.5f * nd(rng); }
    SharedBuffer noise = up_bf16_(mc, nz), text = up_bf16_(mc, tx);
    train::TrainSample s;
    s.x0 = &x0;
    s.noise = &noise;
    s.text = &text;
    s.text_rows = kText;
    s.grid_h = g.h;
    s.grid_w = g.w;
    s.frames = g.t;
    s.sigma = 0.6f;
    s.seed = 17;
    train::StepStats st;
    bool ok = false;
    const std::size_t peak = device_peak_of_(mc, [&] {
      ok = t->step(s, &st, &err);
    });
    ASSERT_TRUE(ok);
    if (!ok) { std::printf("  %s\n", err.c_str()); return; }
    const std::size_t booked =
        t->arena_bytes(g.t, g.h, g.w, kText, kRank, keep != 0);
    const int seq = MiniMaxH3LoraTrainer::sequence_rows(a.cfg, g.t, g.h,
                                                        g.w, kText);
    std::printf("[minimax_h3_train] plan%s: %d x %dx%d latents (%d rows), a "
                "step's device peak %.1f MB, booked %.1f MB (%.2fx); loss "
                "%.4f (video %.4f, audio %.4f: no soundtrack)\n",
                keep ? " (every block kept)" : "", g.t, g.w, g.h,
                seq, peak / 1048576.0, booked / 1048576.0,
                (double)booked / std::max<std::size_t>(peak, 1), st.loss,
                st.loss_video, st.loss_audio);
    EXPECT_TRUE(peak <= booked);
    EXPECT_TRUE(st.loss_video >= 0.0 &&
                std::fabs(st.loss - st.loss_video) < 1e-6 * (1 + st.loss));
  }
}

// THE AdaLN, KEPT OR PRUNED BY THE BOX. The family keeps the released
// AdaLN resident where the whole DiT fits beside a step (the M5 Ultra's
// case) and prunes it at load where it does not; the plan says what each
// holds, and a load of the decision holds what the plan said.
// VPIPE_H3_TRAIN_ADALN=keep|prune pins it.
TEST(minimax_h3_train, the_family_keeps_the_adaln_where_it_fits)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root.empty()) { return; }
  const train::LoraFamily* fam = h3_family_();
  ASSERT_TRUE(fam != nullptr);
  std::string err;
  ASSERT_TRUE(fam->claims(root, "", train::RunShape{}, &err));
  ASSERT_TRUE(::getenv("VPIPE_H3_TRAIN_ADALN") == nullptr);
  train::RunShape r;
  r.max_grid_h = r.max_grid_w = 64;
  r.max_cells = 64 * 64;
  r.max_text = 300;
  r.rank = 32;
  r.headroom = (std::size_t)8 << 30;
  train::LoadPlan big, small;
  r.ram = (std::size_t)1 << 42;   // 4 TB: everything fits
  ASSERT_TRUE(fam->plan(root, "", r, &big, &err));
  r.ram = (std::size_t)1 << 30;   // 1 GB: nothing does
  ASSERT_TRUE(fam->plan(root, "", r, &small, &err));
  const std::string dit = MetalMiniMaxH3Transformer::resolve_dit_dir(root);
  const std::size_t full = model_memory::dir_weights_bytes(dit);
  const std::size_t adaln = MetalMiniMaxH3Transformer::adaln_retired_bytes(dit);
  auto pruned_of = [](const train::LoadPlan& p) {
    FlexData o = p.opts;
    return o.as_object().at("prune_adaln").as_bool(false);
  };
  std::printf("[minimax_h3_train] family: %zu MB of DiT, %zu MB of AdaLN; "
              "roomy box: %s, held %zu MB; small box: %s, held %zu MB, "
              "floor %zu MB; adapter state %zu MB, arena %zu MB\n",
              full >> 20, adaln >> 20, big.note.c_str(),
              big.held_bytes >> 20, small.note.c_str(),
              small.held_bytes >> 20, small.floor_bytes >> 20,
              big.state_bytes >> 20, big.arena_recompute >> 20);
  EXPECT_TRUE(adaln > 0);
  EXPECT_TRUE(!pruned_of(big) && big.held_bytes == full);
  EXPECT_TRUE(pruned_of(small) && small.held_bytes < full - adaln / 2);
  EXPECT_TRUE(small.floor_bytes < big.floor_bytes);
  // A load of each decision holds what its plan said -- where the
  // checkpoint is small enough to load whole here (a cut of the release).
  MetalMiniMaxH3Transformer::Config cfg;
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(root, cfg, &err));
  if (cfg.n_layers > 4) {
    std::printf("[minimax_h3_train] family: %d blocks -- the loads are "
                "checked on a cut of at most 4\n", cfg.n_layers);
    return;
  }
  Session sess;
  for (const train::LoadPlan* p : {&big, &small}) {
    train::LoraFamily::LoadArgs a;
    a.mc = sess.metal_compute();
    a.ws = WeightSet::open(p->dit_dir, nullptr);
    a.root = root;
    a.plan = p;
    a.run = &r;
    auto t = fam->load(a, &err);
    ASSERT_TRUE(t != nullptr);
    if (t == nullptr) { std::printf("  %s\n", err.c_str()); return; }
    auto* lt = static_cast<MiniMaxH3LoraTrainer*>(t.get());
    const std::size_t held = lt->resident_bytes();
    std::printf("[minimax_h3_train] family: %s -- planned %.1f MB, held "
                "%.1f MB, pruned at load: %s\n", p->note.c_str(),
                p->held_bytes / 1048576.0, held / 1048576.0,
                lt->model()->adaln_pruned_at_load() ? "yes" : "no");
    EXPECT_TRUE(lt->model()->adaln_pruned_at_load() == pruned_of(*p));
    const double ratio = (double)held / (double)p->held_bytes;
    EXPECT_TRUE(ratio > 0.97 && ratio < 1.03);
  }
}

// KEYFRAME TASKS on a synthetic H3. A sample that pins its first frame
// (i2va), its last (l2va) or both (fl2va) packs exactly the layout
// generate-video builds for that request -- every position, tag and row
// count, the presentation's vision rows tagged as the conditioner tags
// them -- and the anchors reach the forward: the same sample with its
// keyframes trains to another loss than without, and a step trains on it.
TEST(minimax_h3_train, keyframe_tasks_pack_generate_videos_layout)
{
  const fs::path f = fs::temp_directory_path() /
                     ("vpipe-h3-kf-" + std::to_string(::getpid()) +
                      ".safetensors");
  ASSERT_TRUE(write_synthetic_(f));
  Session sess;
  auto* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  MiniMaxH3LoraTrainer::LoadArgs a;
  a.mc = mc;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(f.string(), a.cfg,
                                                          &err));
  a.ws = WeightSet::open(f.string(), nullptr);
  auto t = MiniMaxH3LoraTrainer::load(a, &err);
  ASSERT_TRUE(t != nullptr);
  if (t == nullptr) { std::printf("  %s\n", err.c_str()); return; }
  t->model()->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
  train::LoraParams params;
  ASSERT_TRUE(params.create(mc, t->modules("attn_ff"), 4, 4.0f, &err));
  params.init_random(5);
  ASSERT_TRUE(t->bind(&params, &err) > 0);
  // A 22-frame window (7 latent frames) of 8 x 8 latents; 12 text rows,
  // the first 4 a vision block (tag 0), as a presentation tags them.
  const int T = 7, G = 8, NT = 12, Z = a.cfg.video_channels;
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> lat((std::size_t)Z * T * G * G), still((std::size_t)Z *
                                                           G * G);
  for (float& v : lat) { v = nd(rng); }
  for (float& v : still) { v = nd(rng); }
  SharedBuffer x0 = t->pack_latent(lat.data(), Z, T, G, G, &err);
  SharedBuffer k1 = t->pack_latent(still.data(), Z, 1, G, G, &err);
  ASSERT_TRUE(!x0.empty() && !k1.empty());
  SharedBuffer cond = mc->make_shared_buffer(2 * k1.byte_size());
  std::memcpy(cond.contents(), k1.contents(), k1.byte_size());
  std::memcpy(static_cast<std::uint8_t*>(cond.contents()) + k1.byte_size(),
              x0.contents(), k1.byte_size());
  std::vector<float> nz(x0.byte_size() / 2), tx((std::size_t)NT *
                                                a.cfg.text_dim);
  for (float& v : nz) { v = nd(rng); }
  for (float& v : tx) { v = 0.6f * nd(rng); }
  SharedBuffer noise = up_bf16_(mc, nz), text = up_bf16_(mc, tx);
  FlexData sb = FlexData::make_object();
  {
    FlexData tags = FlexData::make_array();
    for (int i = 0; i < NT; ++i) {
      tags.as_array().push_back(FlexData::make_int(i < 4 ? 0 : h3t::kTextTag));
    }
    sb.as_object().insert_or_assign("token_tags", std::move(tags));
  }
  std::vector<int> tags(NT, h3t::kTextTag);
  for (int i = 0; i < 4; ++i) { tags[(std::size_t)i] = 0; }
  const int alat = h3t::audio_latent_num_frames(22);
  for (int anchors : {1, 2, 3}) {
    train::TrainSample s;
    s.x0 = &x0; s.noise = &noise; s.text = &text; s.text_rows = NT;
    s.sideband = &sb; s.grid_h = G; s.grid_w = G; s.frames = T;
    s.sigma = 0.6f; s.seed = 11; s.anchors = anchors; s.cond_x0 = &cond;
    h3t::PackedLayout mine, gv;
    ASSERT_TRUE(t->layout_of(s, &mine, &err));
    std::vector<h3t::Anchor> ends;
    if (anchors & 1) { ends.push_back(h3t::Anchor::kFirst); }
    if (anchors & 2) { ends.push_back(h3t::Anchor::kLast); }
    ASSERT_TRUE(h3t::build_packed_sequence(tags, T, G, G, alat,
                                           a.cfg.patch_h, a.cfg.patch_w,
                                           h3t::kAudioChannels, ends, &gv));
    const bool same = mine.seq_len == gv.seq_len &&
                      mine.position_ids == gv.position_ids &&
                      mine.token_tags == gv.token_tags &&
                      mine.num_condition_video_rows ==
                          gv.num_condition_video_rows &&
                      mine.video_indices == gv.video_indices &&
                      mine.audio_indices == gv.audio_indices;
    std::printf("[minimax_h3_train] %s: %d rows, %d conditioning, the "
                "layout generate-video builds: %s\n",
                anchors == 1 ? "i2va" : anchors == 2 ? "l2va" : "fl2va",
                mine.seq_len, mine.num_condition_video_rows,
                same ? "yes" : "NO");
    EXPECT_TRUE(same);
    EXPECT_TRUE(mine.num_condition_video_rows ==
                __builtin_popcount((unsigned)anchors) * (G / 2) * (G / 2));
  }
  // The anchors are read: one sample's loss with and without them.
  train::TrainSample s;
  s.x0 = &x0; s.noise = &noise; s.text = &text; s.text_rows = NT;
  s.grid_h = G; s.grid_w = G; s.frames = T; s.sigma = 0.6f; s.seed = 11;
  double l0 = 0.0, l3 = 0.0;
  ASSERT_TRUE(t->eval_loss(s, &l0, &err));
  s.anchors = 3; s.cond_x0 = &cond; s.sideband = &sb;
  ASSERT_TRUE(t->eval_loss(s, &l3, &err));
  train::StepStats st;
  ASSERT_TRUE(t->step(s, &st, &err));
  std::printf("[minimax_h3_train] keyframes read: loss %.6f without, %.6f "
              "with both anchors; a step trains on it (%.6f)\n", l0, l3,
              st.loss);
  EXPECT_TRUE(l0 != l3);
  EXPECT_TRUE(std::fabs(st.loss - l3) < 1e-4 * (1 + l3));
  // A presentation without an anchor is refused.
  train::TrainSample bad = s;
  bad.anchors = 0;
  bad.cond_x0 = nullptr;
  train::StepStats bst;
  EXPECT_TRUE(!t->step(bad, &bst, &err));
  fs::remove(f);
}

// P-H3-IMAGE: FL2VA under PrunaAI's adapter. On a synthetic H3 with a
// synthetic "PrunaAI adapter" (a few AdamW steps of its own) and a fake
// P-H3-Image root (its manifest; the DiT named apart, as dit_dir does):
// the family refuses the root without the adapter, saying so; with it,
// it trains a NEW adapter in the slot above the frozen one -- one frame,
// the grid's shift 1 -- and the file it saves, bound by the model's own
// binder beside the base adapter, runs exactly the trainer's forward
// (the one-forward contract over a frozen slot). The metadata names the
// base adapter inference must stack beneath it.
TEST(minimax_h3_train, p_h3_image_trains_over_its_frozen_adapter)
{
  const fs::path tmp = fs::temp_directory_path() /
                       ("vpipe-h3-ph3-" + std::to_string(::getpid()));
  fs::remove_all(tmp);
  fs::create_directories(tmp / "root");
  const fs::path f = tmp / "dit.safetensors";
  ASSERT_TRUE(write_synthetic_(f));
  std::string err;
  // The base adapter.
  const fs::path base = tmp / "pruna.safetensors";
  {
    Syn s;
    ASSERT_TRUE(s.load(f, true));
    ASSERT_TRUE(s.train(&err));
    train::TrainOps ops;
    ASSERT_TRUE(ops.init(s.mc, true, &err));
    train::OptimizerSpec spec;
    spec.lr = 1e-2;
    spec.warmup_steps = 0;
    train::Optimizer o;
    ASSERT_TRUE(o.init(&ops, &s.params, spec, 6, &err));
    for (int i = 0; i < 6; ++i) {
      MiniMaxH3Trainer::Result res;
      ASSERT_TRUE(s.trainer->step(s.sample(), &res, &err));
      double gn = 0.0, lr = 0.0;
      ASSERT_TRUE(o.step(1, &gn, &lr, &err));
    }
    ASSERT_TRUE(s.params.save(base.string(), {}, &err));
  }
  std::ofstream(tmp / "root" / "modular_model_index.json")
      << "{\"_class_name\": \"PH3ImageModularPipeline\"}";
  const train::LoraFamily* fam = h3_family_();
  ASSERT_TRUE(fam != nullptr);
  train::RunShape r;
  r.dit_dir = f.string();
  r.rank = 4;
  r.max_grid_h = r.max_grid_w = 8;
  r.max_cells = 64;
  r.max_text = 8;
  r.ram = (std::size_t)1 << 40;
  const std::string root = (tmp / "root").string();
  err.clear();
  EXPECT_TRUE(!fam->claims(root, "", r, &err) &&
              err.find("base_lora") != std::string::npos);
  std::printf("[minimax_h3_train] p-h3-image without its adapter: %s\n",
              err.c_str());
  r.base_loras = {base.string()};
  err.clear();
  ASSERT_TRUE(fam->claims(root, "", r, &err));
  train::LoadPlan plan;
  ASSERT_TRUE(fam->plan(root, "", r, &plan, &err));
  Session sess;
  train::LoraFamily::LoadArgs la;
  la.mc = sess.metal_compute();
  la.ws = WeightSet::open(plan.dit_dir, nullptr);
  la.root = root;
  la.plan = &plan;
  la.run = &r;
  auto t = fam->load(la, &err);
  ASSERT_TRUE(t != nullptr);
  if (t == nullptr) { std::printf("  %s\n", err.c_str()); return; }
  auto* lt = static_cast<MiniMaxH3LoraTrainer*>(t.get());
  lt->model()->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
  EXPECT_TRUE(lt->variant() == MiniMaxH3LoraTrainer::Variant::kPh3);
  EXPECT_TRUE(lt->schedule_mu(8, 8) == 0.0);
  EXPECT_TRUE(lt->parent_model_type() == "p-h3-image");
  EXPECT_TRUE(lt->metadata()["vpipe.base_adapter"] == base.string());
  const int base_mods = lt->model()->lora_modules(0);
  std::printf("[minimax_h3_train] p-h3-image: %s; the base adapter bound "
              "%d modules in slot 0\n", plan.note.c_str(), base_mods);
  EXPECT_TRUE(base_mods > 0);
  train::LoraParams params;
  ASSERT_TRUE(params.create(la.mc, lt->modules("attn_ff"), 4, 4.0f, &err));
  params.init_random(13);
  ASSERT_TRUE(lt->bind(&params, &err) > 0);
  const int Z = lt->latent_channels();
  std::mt19937 rng(17);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> lat((std::size_t)Z * 8 * 8), nz, tx((std::size_t)8 *
                                                        256);
  for (float& v : lat) { v = nd(rng); }
  for (float& v : tx) { v = 0.6f * nd(rng); }
  SharedBuffer x0 = lt->pack_latent(lat.data(), Z, 1, 8, 8, &err);
  nz.resize(x0.byte_size() / 2);
  for (float& v : nz) { v = nd(rng); }
  SharedBuffer noise = up_bf16_(la.mc, nz), text = up_bf16_(la.mc, tx);
  train::TrainOps ops;
  ASSERT_TRUE(ops.init(la.mc, true, &err));
  train::OptimizerSpec spec;
  spec.lr = 1e-2;
  spec.warmup_steps = 0;
  train::Optimizer o;
  ASSERT_TRUE(o.init(&ops, &params, spec, 12, &err));
  double first = 0.0, last = 0.0;
  for (int i = 0; i < 12; ++i) {
    train::TrainSample s;
    s.x0 = &x0; s.noise = &noise; s.text = &text; s.text_rows = 8;
    s.grid_h = 8; s.grid_w = 8; s.frames = 1; s.sigma = 0.55f;
    s.seed = 3;
    train::StepStats st;
    ASSERT_TRUE(lt->step(s, &st, &err));
    double gn = 0.0, lr = 0.0;
    ASSERT_TRUE(o.step(1, &gn, &lr, &err));
    if (i == 0) { first = st.loss; }
    last = st.loss;
  }
  std::printf("[minimax_h3_train] p-h3-image over its adapter: loss %.4f "
              "-> %.4f\n", first, last);
  EXPECT_TRUE(last < 0.9 * first);
  // The contract over the frozen slot -- on the q|k|v grouping the
  // family's load read off the file (random weights read the other way
  // are another model).
  const bool ph = lt->model()->config().qkv_per_head;
  Syn ref;
  ASSERT_TRUE(ref.load(f, ph));
  const std::vector<std::uint16_t> live = forward_words_(*lt->model(), ref.io);
  const fs::path trained = tmp / "trained.safetensors";
  ASSERT_TRUE(params.save(trained.string(), {}, &err));
  MetalMiniMaxH3Transformer::LoraSpec b0, b1;
  b0.path = base.string();
  b1.path = trained.string();
  Syn both, only;
  ASSERT_TRUE(both.load(f, ph, {b0, b1}));
  ASSERT_TRUE(only.load(f, ph, {b0}));
  const std::vector<std::uint16_t> again = forward_words_(*both.model, both.io);
  const std::vector<std::uint16_t> base_only =
      forward_words_(*only.model, only.io);
  const std::size_t d = diff_(again, live), moved = diff_(live, base_only);
  std::printf("[minimax_h3_train] p-h3-image: the trained adapter moves %zu "
              "of %zu over the base; bound beside it from its file: %zu "
              "differ\n", moved, live.size(), d);
  EXPECT_TRUE(moved > live.size() / 10);
  EXPECT_TRUE(d == 0);
  fs::remove_all(tmp);
}

// MINIMAX-H3-IMAGE: audio-less, one output head a step, its release grid.
// On the real checkpoint (VPIPE_MINIMAX_H3_IMAGE_PATH, the root) cut to
// two blocks: every sample draws one release step by its seed -- the
// step's sigma as the release runs it, through that step's own head --
// so twenty seeds over one sample land on exactly the grid's six distinct
// losses; a few steps lower the loss, and a preview runs the release's
// grid (a head a step, the velocity form) to a finite picture latent.
TEST(minimax_h3_train, minimax_h3_image_trains_through_its_step_heads)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_IMAGE_PATH");
  if (root.empty()) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  h3i::Manifest man;
  std::string err;
  ASSERT_TRUE(h3i::read_manifest(root, &man, &err));
  MiniMaxH3LoraTrainer::LoadArgs a;
  a.mc = mc;
  a.variant = MiniMaxH3LoraTrainer::Variant::kH3i;
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(root, a.cfg, &err));
  a.cfg.n_layers = std::min(a.cfg.n_layers, 2);
  EXPECT_TRUE(!a.cfg.audio);
  a.ws = WeightSet::open(MetalMiniMaxH3Transformer::resolve_dit_dir(root),
                         nullptr);
  a.h3i_sigmas = man.sigmas;
  a.heads_ws = WeightSet::open(man.heads_file, nullptr);
  auto t = MiniMaxH3LoraTrainer::load(a, &err);
  ASSERT_TRUE(t != nullptr);
  if (t == nullptr) { std::printf("  %s\n", err.c_str()); return; }
  EXPECT_TRUE(t->parent_model_type() == "minimax-h3-image");
  EXPECT_TRUE(t->audio_latents(1) == 0);
  train::LoraParams params;
  ASSERT_TRUE(params.create(mc, t->modules("attn_ff"), 8, 8.0f, &err));
  params.init_random(5);
  ASSERT_TRUE(t->bind(&params, &err) > 0);
  const int Z = a.cfg.video_channels, G = 32, NT = 40;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> lat((std::size_t)Z * G * G), tx((std::size_t)NT *
                                                    a.cfg.text_dim);
  for (float& v : lat) { v = nd(rng); }
  for (float& v : tx) { v = 0.5f * nd(rng); }
  SharedBuffer x0 = t->pack_latent(lat.data(), Z, 1, G, G, &err);
  std::vector<float> nz(x0.byte_size() / 2);
  for (float& v : nz) { v = nd(rng); }
  SharedBuffer noise = up_bf16_(mc, nz), text = up_bf16_(mc, tx);
  auto sample = [&](std::uint64_t seed) {
    train::TrainSample s;
    s.x0 = &x0; s.noise = &noise; s.text = &text; s.text_rows = NT;
    s.grid_h = G; s.grid_w = G; s.frames = 1; s.sigma = 0.5f;
    s.seed = seed;
    return s;
  };
  std::vector<double> losses;
  for (std::uint64_t sd = 1; sd <= 20; ++sd) {
    double l = 0.0;
    ASSERT_TRUE(t->eval_loss(sample(sd * 7919), &l, &err));
    bool seen = false;
    for (double x : losses) { seen = seen || x == l; }
    if (!seen) { losses.push_back(l); }
  }
  std::printf("[minimax_h3_train] minimax-h3-image: 20 seeds, %zu distinct "
              "losses (the grid has %d steps)\n", losses.size(), man.steps());
  EXPECT_TRUE((int)losses.size() == man.steps());
  train::TrainOps ops;
  ASSERT_TRUE(ops.init(mc, true, &err));
  train::OptimizerSpec spec;
  spec.lr = 2e-3;
  spec.warmup_steps = 0;
  train::Optimizer o;
  ASSERT_TRUE(o.init(&ops, &params, spec, 8, &err));
  double first = 0.0, last = 0.0;
  for (int i = 0; i < 8; ++i) {
    // One release step throughout (the same seed), so the loss compares.
    train::StepStats st;
    ASSERT_TRUE(t->step(sample(7919), &st, &err));
    double gn = 0.0, lr = 0.0;
    ASSERT_TRUE(o.step(1, &gn, &lr, &err));
    if (i == 0) { first = st.loss; }
    last = st.loss;
  }
  std::printf("[minimax_h3_train] minimax-h3-image: loss %.4f -> %.4f\n",
              first, last);
  EXPECT_TRUE(last < first);
  train::PreviewRequest pr;
  pr.text = &text;
  pr.text_rows = NT;
  pr.grid_h = G;
  pr.grid_w = G;
  pr.seed = 42;
  std::vector<float> out;
  ASSERT_TRUE(t->preview(pr, &out, &err));
  bool finite = out.size() == (std::size_t)Z * G * G;
  for (float v : out) { finite = finite && std::isfinite(v); }
  EXPECT_TRUE(finite);
}

// REF2VA: a caption video-ref-encoder encoded with a sample's references
// carries the plan on its sideband (each reference's kind and latent
// geometry, the presentation's tags) and its rows beside it. On the
// synthetic H3 as the Ref2VA partition: the layout is the one
// generate-video builds for that plan (a picture, a clip with its
// soundtrack, a soundtrack alone), the rows are read (the loss moves
// with them), a step trains on them, and a caption whose rows did not
// arrive -- or one sent to the FL2VA partition -- is refused by name.
TEST(minimax_h3_train, references_pack_generate_videos_ref2va_layout)
{
  const fs::path f = fs::temp_directory_path() /
                     ("vpipe-h3-r2v-" + std::to_string(::getpid()) +
                      ".safetensors");
  ASSERT_TRUE(write_synthetic_(f));
  Session sess;
  auto* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  std::string err;
  auto load = [&](const char* part) {
    MiniMaxH3LoraTrainer::LoadArgs a;
    a.mc = mc;
    std::string e;
    MetalMiniMaxH3Transformer::config_from_json(f.string(), a.cfg, &e);
    a.ws = WeightSet::open(f.string(), nullptr);
    a.partition = part;
    auto t = MiniMaxH3LoraTrainer::load(a, &e);
    if (t != nullptr) {
      t->model()->set_gemm_route(
          MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
    }
    return t;
  };
  auto t = load("ref2va");
  ASSERT_TRUE(t != nullptr);
  if (t == nullptr) { return; }
  const auto& c = t->model()->config();
  train::LoraParams params;
  ASSERT_TRUE(params.create(mc, t->modules("attn_ff"), 4, 4.0f, &err));
  params.init_random(5);
  ASSERT_TRUE(t->bind(&params, &err) > 0);
  // The target: a 22-frame window of 8 x 8 latents. The references: a
  // picture at 6 x 10, a 2-latent-frame clip at 8 x 8 with 12 audio
  // latents, a soundtrack of 9 alone. 14 text rows, 3 of them the
  // picture's vision block.
  const int T = 7, G = 8, NT = 14, Z = c.video_channels, VPE =
      c.video_patch_elems(), AC = c.audio_channels;
  std::vector<h3t::Reference> refs(3);
  refs[0].kind = h3t::Reference::Kind::kImage;
  refs[0].num_latent_frames = 1;
  refs[0].latent_height = 6;
  refs[0].latent_width = 10;
  refs[1].kind = h3t::Reference::Kind::kVideo;
  refs[1].num_latent_frames = 2;
  refs[1].latent_height = 8;
  refs[1].latent_width = 8;
  refs[1].num_audio_latents = 12;
  refs[2].kind = h3t::Reference::Kind::kAudio;
  refs[2].num_audio_latents = 9;
  const int nrv = 1 * 3 * 5 + 2 * 4 * 4;          // patch 2 x 2
  const int nra = h3t::kAudioChannels * (12 + 9);
  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> lat((std::size_t)Z * T * G * G);
  for (float& v : lat) { v = nd(rng); }
  SharedBuffer x0 = t->pack_latent(lat.data(), Z, T, G, G, &err);
  std::vector<float> nz(x0.byte_size() / 2), tx((std::size_t)NT * c.text_dim),
      rv((std::size_t)nrv * VPE), ra((std::size_t)nra * AC);
  for (float& v : nz) { v = nd(rng); }
  for (float& v : tx) { v = 0.6f * nd(rng); }
  for (float& v : rv) { v = nd(rng); }
  for (float& v : ra) { v = nd(rng); }
  SharedBuffer noise = up_bf16_(mc, nz), text = up_bf16_(mc, tx),
               rvb = up_bf16_(mc, rv), rab = up_bf16_(mc, ra);
  std::vector<int> tags(NT, h3t::kTextTag);
  for (int i = 2; i < 5; ++i) { tags[(std::size_t)i] = 0; }
  FlexData sb = FlexData::make_object();
  {
    auto o = sb.as_object();
    FlexData tg = FlexData::make_array();
    for (int v : tags) { tg.as_array().push_back(FlexData::make_int(v)); }
    o.insert_or_assign("token_tags", std::move(tg));
    FlexData rl = FlexData::make_array();
    const char* kinds[] = {"image", "video", "audio"};
    for (int i = 0; i < 3; ++i) {
      FlexData r = FlexData::make_object();
      auto ro = r.as_object();
      ro.insert_or_assign("kind", FlexData::make_string(kinds[i]));
      ro.insert_or_assign("latent_frames",
                          FlexData::make_int(refs[(std::size_t)i]
                                                 .num_latent_frames));
      ro.insert_or_assign("latent_height",
                          FlexData::make_int(refs[(std::size_t)i]
                                                 .latent_height));
      ro.insert_or_assign("latent_width",
                          FlexData::make_int(refs[(std::size_t)i]
                                                 .latent_width));
      ro.insert_or_assign("audio_latents",
                          FlexData::make_int(refs[(std::size_t)i]
                                                 .num_audio_latents));
      rl.as_array().push_back(std::move(r));
    }
    o.insert_or_assign("references", std::move(rl));
  }
  train::TrainSample s;
  s.x0 = &x0; s.noise = &noise; s.text = &text; s.text_rows = NT;
  s.sideband = &sb; s.grid_h = G; s.grid_w = G; s.frames = T;
  s.sigma = 0.6f; s.seed = 21; s.cond_x0 = &rvb; s.cond_audio = &rab;
  h3t::PackedLayout mine, gv;
  ASSERT_TRUE(t->layout_of(s, &mine, &err));
  ASSERT_TRUE(h3t::build_ref2va_packed_sequence(
      tags, refs, T, G, G, h3t::audio_latent_num_frames(22), c.patch_h,
      c.patch_w, h3t::kAudioChannels, &gv));
  const bool same = mine.seq_len == gv.seq_len &&
                    mine.position_ids == gv.position_ids &&
                    mine.token_tags == gv.token_tags &&
                    mine.num_condition_video_rows ==
                        gv.num_condition_video_rows &&
                    mine.num_condition_audio_rows ==
                        gv.num_condition_audio_rows &&
                    mine.video_indices == gv.video_indices &&
                    mine.audio_indices == gv.audio_indices;
  std::printf("[minimax_h3_train] ref2va: %d rows, %d + %d reference rows, "
              "the layout generate-video builds: %s\n", mine.seq_len,
              mine.num_condition_video_rows, mine.num_condition_audio_rows,
              same ? "yes" : "NO");
  EXPECT_TRUE(same);
  EXPECT_TRUE(mine.num_condition_video_rows == nrv &&
              mine.num_condition_audio_rows == nra);
  // The rows are read: the loss with and without them (the plan alike).
  double l0 = 0.0, l1 = 0.0;
  ASSERT_TRUE(t->eval_loss(s, &l0, &err));
  std::vector<float> rv2 = rv;
  for (float& v : rv2) { v = -v; }
  SharedBuffer rvb2 = up_bf16_(mc, rv2);
  train::TrainSample s2 = s;
  s2.cond_x0 = &rvb2;
  ASSERT_TRUE(t->eval_loss(s2, &l1, &err));
  train::StepStats st;
  ASSERT_TRUE(t->step(s, &st, &err));
  std::printf("[minimax_h3_train] references read: loss %.6f, %.6f with "
              "their video rows negated; a step trains on them (%.6f)\n",
              l0, l1, st.loss);
  EXPECT_TRUE(l0 != l1);
  EXPECT_TRUE(std::fabs(st.loss - l0) < 1e-4 * (1 + l0));
  // No rows beside the plan: refused, naming the ports.
  train::TrainSample bad = s;
  bad.cond_audio = nullptr;
  err.clear();
  EXPECT_TRUE(!t->step(bad, &st, &err) &&
              err.find("ref_audio") != std::string::npos);
  std::printf("[minimax_h3_train] rows missing: %s\n", err.c_str());
  // FL2VA does not take references.
  auto fl = load("fl2va");
  ASSERT_TRUE(fl != nullptr);
  h3t::PackedLayout L;
  err.clear();
  EXPECT_TRUE(fl != nullptr && !fl->layout_of(s, &L, &err) &&
              err.find("Ref2VA") != std::string::npos);
  std::printf("[minimax_h3_train] on fl2va: %s\n", err.c_str());
  fs::remove(f);
}

// RECOMPUTE: NEVER. The forward runs each main block through the
// trainer's own block forward and keeps its activations; the backward
// reads them and recomputes nothing. Against the recompute arm on the
// golden (both on steel's pinned tile): the same loss, velocities and
// gradients to the bit, or -- where the model's own block and the
// trainer's round apart -- within the reference's own bf16 distance; and
// both within the autograd golden's bars. A streamed model keeping its
// activations is the resident one to the bit.
TEST(minimax_h3_train, keep_all_matches_recompute)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  Rig rec, keep, keep_s;
  ASSERT_TRUE(rec.build(g, true));
  ASSERT_TRUE(keep.build(g, true));
  ASSERT_TRUE(keep_s.build(g, true, /*stream=*/true));
  if (!rec.trainer || !keep.trainer || !keep_s.trainer) { return; }
  keep.trainer->set_keep_all(true);
  keep_s.trainer->set_keep_all(true);
  MiniMaxH3Trainer::Result r0, r1, r2;
  std::string err;
  ASSERT_TRUE(rec.trainer->step(rec.sample(), &r0, &err, true));
  ASSERT_TRUE(keep.trainer->step(keep.sample(), &r1, &err, true));
  if (!err.empty()) { std::printf("  %s\n", err.c_str()); }
  ASSERT_TRUE(keep_s.trainer->step(keep_s.sample(), &r2, &err, true));
  auto words = [](const SharedBuffer& b) {
    const auto* p = static_cast<const std::uint16_t*>(b.contents());
    return std::vector<std::uint16_t>(p, p + b.byte_size() / 2);
  };
  const std::size_t dv = diff_(words(r0.pred_video), words(r1.pred_video));
  const std::size_t ds = diff_(words(r1.pred_video), words(r2.pred_video));
  double worst = 0.0, worst_bar = 0.0, gap = 0.0;
  std::size_t bits = 0, bits_s = 0, n = 0;
  const std::string gf = (g.dir / "grads_f32.safetensors").string();
  for (std::size_t i = 0; i < keep.params.modules().size(); ++i) {
    const auto& m = keep.params.modules()[i];
    const auto& m0 = rec.params.modules()[i];
    const auto& m2 = keep_s.params.modules()[i];
    std::vector<float> wa, wb;
    ASSERT_TRUE(read_tensor_(gf, m.name + ".A", &wa));
    ASSERT_TRUE(read_tensor_(gf, m.name + ".B", &wb));
    const auto a = keep.grad_a(m), b = keep.grad_b(m);
    const auto a0 = rec.grad_a(m0), b0 = rec.grad_b(m0);
    const auto a2 = keep_s.grad_a(m2), b2 = keep_s.grad_b(m2);
    for (std::size_t k = 0; k < a.size(); ++k) {
      bits += a[k] != a0[k];
      bits_s += a[k] != a2[k];
    }
    for (std::size_t k = 0; k < b.size(); ++k) {
      bits += b[k] != b0[k];
      bits_s += b[k] != b2[k];
    }
    n += a.size() + b.size();
    gap = std::max({gap, rel_l2_(a.data(), a0.data(), a.size()),
                    rel_l2_(b.data(), b0.data(), b.size())});
    const double ea = rel_l2_(a.data(), wa.data(), a.size());
    const double eb = rel_l2_(b.data(), wb.data(), b.size());
    worst = std::max({worst, ea, eb});
    worst_bar = std::max(
        {worst_bar, ea / std::max(2.5 * g.grad_bar(m.name + ".A"), 0.05),
         eb / std::max(2.5 * g.grad_bar(m.name + ".B"), 0.05)});
  }
  std::printf("[minimax_h3_train] keep_all: loss %.6f against recompute's "
              "%.6f; %zu of %zu velocity words and %zu of %zu gradient "
              "values differ (worst module gap %.5f); %.1f + %.1f ms "
              "against %.1f + %.1f; worst gradient from the reference "
              "%.4f, %.2f of its bar; streamed keep: %zu velocity words, "
              "%zu gradients differ\n", r1.loss, r0.loss, dv,
              r1.pred_video.byte_size() / 2, bits, n, gap, r1.forward_ms,
              r1.backward_ms, r0.forward_ms, r0.backward_ms, worst,
              worst_bar, ds, bits_s);
  EXPECT_TRUE(worst_bar <= 1.0);
  EXPECT_TRUE(gap < std::max(g.getr("pred_video_bf16_rel", 0.0), 0.01));
  EXPECT_TRUE(ds == 0 && bits_s == 0);
}

namespace {

// A conditioning sideband naming `refs` (each a picture at lh x lw) and
// the text rows' tags.
FlexData
edit_sideband_(const std::vector<int>& tags,
               const std::vector<std::pair<int, int>>& refs, bool draws,
               const char* kind = "image")
{
  FlexData sb = FlexData::make_object();
  auto o = sb.as_object();
  FlexData tg = FlexData::make_array();
  for (int v : tags) { tg.as_array().push_back(FlexData::make_int(v)); }
  o.insert_or_assign("token_tags", std::move(tg));
  FlexData rl = FlexData::make_array();
  for (const auto& r : refs) {
    FlexData e = FlexData::make_object();
    auto eo = e.as_object();
    eo.insert_or_assign("kind", FlexData::make_string(kind));
    eo.insert_or_assign("latent_frames", FlexData::make_int(1));
    eo.insert_or_assign("latent_height", FlexData::make_int(r.first));
    eo.insert_or_assign("latent_width", FlexData::make_int(r.second));
    eo.insert_or_assign("audio_latents", FlexData::make_int(0));
    if (draws) {
      eo.insert_or_assign("posterior_draws",
                          FlexData::make_int(24 * r.first * r.second));
    }
    rl.as_array().push_back(std::move(e));
  }
  o.insert_or_assign("references", std::move(rl));
  return sb;
}

}  // namespace

// EDIT PAIRS on P-H3-Image (synthetic H3 as its FL2VA, a stand-in grid):
// a caption encoded with the picture to edit packs generate-video's edit
// layout -- the reference builder, the picture ahead of the generated
// one, its two audio latents -- with the picture's rows read as given (the
// encoder noised them) at the condition level; a caption without one
// packs the text-to-image layout; a clip as an edit's reference is
// refused by name.
TEST(minimax_h3_train, p_h3_image_trains_edit_pairs)
{
  const fs::path f = fs::temp_directory_path() /
                     ("vpipe-h3-ph3e-" + std::to_string(::getpid()) +
                      ".safetensors");
  ASSERT_TRUE(write_synthetic_(f));
  Session sess;
  auto* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  MiniMaxH3LoraTrainer::LoadArgs a;
  a.mc = mc;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(f.string(), a.cfg,
                                                          &err));
  a.ws = WeightSet::open(f.string(), nullptr);
  a.variant = MiniMaxH3LoraTrainer::Variant::kPh3;
  a.partition = "fl2va";
  a.ph3_sigmas = {1.0f, 0.75f, 0.4f, 0.0f};
  auto t = MiniMaxH3LoraTrainer::load(a, &err);
  ASSERT_TRUE(t != nullptr);
  if (t == nullptr) { std::printf("  %s\n", err.c_str()); return; }
  t->model()->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
  train::LoraParams params;
  ASSERT_TRUE(params.create(mc, t->modules("attn_ff"), 4, 4.0f, &err));
  params.init_random(5);
  ASSERT_TRUE(t->bind(&params, &err) > 0);
  const auto& c = t->model()->config();
  const int G = 8, NT = 10, Z = c.video_channels, VPE = c.video_patch_elems();
  std::mt19937 rng(4);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> lat((std::size_t)Z * G * G), tx((std::size_t)NT *
                                                    c.text_dim);
  for (float& v : lat) { v = nd(rng); }
  for (float& v : tx) { v = 0.6f * nd(rng); }
  SharedBuffer x0 = t->pack_latent(lat.data(), Z, 1, G, G, &err);
  std::vector<float> nz(x0.byte_size() / 2), rv((std::size_t)(6 / 2) *
                                                (10 / 2) * VPE);
  for (float& v : nz) { v = nd(rng); }
  for (float& v : rv) { v = nd(rng); }
  SharedBuffer noise = up_bf16_(mc, nz), text = up_bf16_(mc, tx),
               rows = up_bf16_(mc, rv), none = up_bf16_(mc, {0.0f});
  std::vector<int> tags(NT, h3t::kTextTag);
  for (int i = 1; i < 4; ++i) { tags[(std::size_t)i] = 0; }
  const FlexData sb = edit_sideband_(tags, {{6, 10}}, false);
  train::TrainSample s;
  s.x0 = &x0; s.noise = &noise; s.text = &text; s.text_rows = NT;
  s.sideband = &sb; s.grid_h = G; s.grid_w = G; s.frames = 1;
  s.sigma = 0.5f; s.seed = 9; s.cond_x0 = &rows; s.cond_audio = &none;
  h3t::PackedLayout mine, gv;
  ASSERT_TRUE(t->layout_of(s, &mine, &err));
  std::vector<h3t::Reference> refs(1);
  refs[0].kind = h3t::Reference::Kind::kImage;
  refs[0].num_latent_frames = 1;
  refs[0].latent_height = 6;
  refs[0].latent_width = 10;
  ASSERT_TRUE(h3t::build_ref2va_packed_sequence(
      tags, refs, 1, G, G, h3t::audio_latent_num_frames(1), c.patch_h,
      c.patch_w, h3t::kAudioChannels, &gv));
  const bool same = mine.seq_len == gv.seq_len &&
                    mine.position_ids == gv.position_ids &&
                    mine.token_tags == gv.token_tags &&
                    mine.video_indices == gv.video_indices &&
                    mine.audio_indices == gv.audio_indices;
  double le = 0.0, lneg = 0.0, lt2i = 0.0;
  ASSERT_TRUE(t->eval_loss(s, &le, &err));
  std::vector<float> rn = rv;
  for (float& v : rn) { v = -v; }
  SharedBuffer rows2 = up_bf16_(mc, rn);
  train::TrainSample s2 = s;
  s2.cond_x0 = &rows2;
  ASSERT_TRUE(t->eval_loss(s2, &lneg, &err));
  // Text-to-image: no sideband references, the T2VA layout.
  train::TrainSample s3 = s;
  s3.sideband = nullptr;
  s3.cond_x0 = nullptr;
  ASSERT_TRUE(t->layout_of(s3, &gv, &err));
  h3t::PackedLayout t2i;
  ASSERT_TRUE(h3t::build_packed_sequence(
      std::vector<int>(NT, h3t::kTextTag), 1, G, G,
      h3t::audio_latent_num_frames(1), c.patch_h, c.patch_w,
      h3t::kAudioChannels, {}, &t2i));
  const bool same_t2i = gv.seq_len == t2i.seq_len &&
                        gv.position_ids == t2i.position_ids &&
                        gv.video_indices == t2i.video_indices;
  ASSERT_TRUE(t->eval_loss(s3, &lt2i, &err));
  train::StepStats st;
  ASSERT_TRUE(t->step(s, &st, &err));
  std::printf("[minimax_h3_train] p-h3-image edit: %d rows (%d of the "
              "picture), generate-video's edit layout: %s; t2i layout: %s; "
              "loss %.5f edit, %.5f its rows negated, %.5f t2i; a step "
              "trains (%.5f)\n", mine.seq_len, mine.num_condition_video_rows,
              same ? "yes" : "NO", same_t2i ? "yes" : "NO", le, lneg, lt2i,
              st.loss);
  EXPECT_TRUE(same && same_t2i);
  EXPECT_TRUE(mine.num_condition_video_rows == 15);
  EXPECT_TRUE(le != lneg && le != lt2i);
  EXPECT_TRUE(std::fabs(st.loss - le) < 1e-4 * (1 + le));
  // A clip is not an edit's picture.
  const FlexData bad = edit_sideband_(tags, {{6, 10}}, false, "video");
  train::TrainSample sb2 = s;
  sb2.sideband = &bad;
  err.clear();
  EXPECT_TRUE(!t->layout_of(sb2, &mine, &err) &&
              err.find("pictures") != std::string::npos);
  std::printf("[minimax_h3_train] a clip as the edit's reference: %s\n",
              err.c_str());
  fs::remove(f);
}

// EDIT PAIRS on MiniMax-H3-Image (the real checkpoint, two blocks): the
// pictures' rows arrive CLEAN with their posterior draws named, and the
// trainer noises them to the release's 0.999 with the sample's own noise
// -- so two seeds at one release step read different rows, and the
// layout is generate-video's [text | picture | the picture].
TEST(minimax_h3_train, minimax_h3_image_trains_edit_pairs)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_IMAGE_PATH");
  if (root.empty()) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  h3i::Manifest man;
  std::string err;
  ASSERT_TRUE(h3i::read_manifest(root, &man, &err));
  MiniMaxH3LoraTrainer::LoadArgs a;
  a.mc = mc;
  a.variant = MiniMaxH3LoraTrainer::Variant::kH3i;
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(root, a.cfg, &err));
  a.cfg.n_layers = std::min(a.cfg.n_layers, 2);
  a.ws = WeightSet::open(MetalMiniMaxH3Transformer::resolve_dit_dir(root),
                         nullptr);
  a.h3i_sigmas = man.sigmas;
  a.heads_ws = WeightSet::open(man.heads_file, nullptr);
  auto t = MiniMaxH3LoraTrainer::load(a, &err);
  ASSERT_TRUE(t != nullptr);
  if (t == nullptr) { std::printf("  %s\n", err.c_str()); return; }
  train::LoraParams params;
  ASSERT_TRUE(params.create(mc, t->modules("attn_ff"), 8, 8.0f, &err));
  params.init_random(5);
  ASSERT_TRUE(t->bind(&params, &err) > 0);
  const auto& c = t->model()->config();
  const int G = 32, NT = 40, Z = c.video_channels, VPE = c.video_patch_elems();
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> lat((std::size_t)Z * G * G), tx((std::size_t)NT *
                                                    c.text_dim);
  for (float& v : lat) { v = nd(rng); }
  for (float& v : tx) { v = 0.5f * nd(rng); }
  SharedBuffer x0 = t->pack_latent(lat.data(), Z, 1, G, G, &err);
  std::vector<float> nz(x0.byte_size() / 2), rv((std::size_t)(G / 2) *
                                                (G / 2) * VPE);
  for (float& v : nz) { v = nd(rng); }
  for (float& v : rv) { v = nd(rng); }
  SharedBuffer noise = up_bf16_(mc, nz), text = up_bf16_(mc, tx),
               rows = up_bf16_(mc, rv), none = up_bf16_(mc, {0.0f});
  std::vector<int> tags(NT, h3t::kTextTag);
  for (int i = 3; i < 20; ++i) { tags[(std::size_t)i] = 0; }
  const FlexData sb = edit_sideband_(tags, {{G, G}}, true);
  auto sample = [&](std::uint64_t seed, const FlexData* side) {
    train::TrainSample s;
    s.x0 = &x0; s.noise = &noise; s.text = &text; s.text_rows = NT;
    s.sideband = side; s.grid_h = G; s.grid_w = G; s.frames = 1;
    s.sigma = 0.5f; s.seed = seed;
    if (side != nullptr) { s.cond_x0 = &rows; s.cond_audio = &none; }
    return s;
  };
  h3t::PackedLayout mine, gv;
  ASSERT_TRUE(t->layout_of(sample(1, &sb), &mine, &err));
  std::vector<h3t::Reference> refs(1);
  refs[0].kind = h3t::Reference::Kind::kImage;
  refs[0].num_latent_frames = 1;
  refs[0].latent_height = G;
  refs[0].latent_width = G;
  ASSERT_TRUE(h3t::build_ref2va_packed_sequence(tags, refs, 1, G, G, 0,
                                                c.patch_h, c.patch_w,
                                                h3t::kAudioChannels, &gv));
  const bool same = mine.seq_len == gv.seq_len &&
                    mine.position_ids == gv.position_ids &&
                    mine.video_indices == gv.video_indices;
  // Seeds that draw the same release step: the noise on the picture's
  // rows differs, so the loss does.
  double l1 = 0.0, l2 = 0.0, l0 = 0.0;
  std::uint64_t s1 = 0, s2 = 0;
  {
    // Find two seeds sharing a step by their t2i loss (one per step).
    std::map<double, std::vector<std::uint64_t>> by;
    for (std::uint64_t sd = 1; sd <= 24; ++sd) {
      double l = 0.0;
      ASSERT_TRUE(t->eval_loss(sample(sd * 7919, nullptr), &l, &err));
      by[l].push_back(sd * 7919);
    }
    for (const auto& e : by) {
      if (e.second.size() >= 2) {
        s1 = e.second[0];
        s2 = e.second[1];
        l0 = e.first;
        break;
      }
    }
  }
  ASSERT_TRUE(s1 != 0);
  ASSERT_TRUE(t->eval_loss(sample(s1, &sb), &l1, &err));
  ASSERT_TRUE(t->eval_loss(sample(s2, &sb), &l2, &err));
  std::printf("[minimax_h3_train] minimax-h3-image edit: %d rows (%d of the "
              "picture), generate-video's layout: %s; one release step: "
              "t2i %.4f, edit %.4f / %.4f at two seeds (the picture's noise "
              "the sample's)\n", mine.seq_len, mine.num_condition_video_rows,
              same ? "yes" : "NO", l0, l1, l2);
  EXPECT_TRUE(same);
  EXPECT_TRUE(l1 != l0 && l1 != l2);
  train::StepStats st;
  ASSERT_TRUE(t->step(sample(s1, &sb), &st, &err));
}
