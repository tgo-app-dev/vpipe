// qwen-image-21-train.cc -- LoRA training on Qwen-Image-2.1, against the
// reference's own autograd.
//
// The golden (tools/dump_qwen_image21_golden.py --what train) is one
// training step of a small random model of the real architecture: two
// blocks of two 128-wide heads, a LoRA on all seven projections of each,
// the flow-matching loss vpipe trains with, and every factor's gradient as
// torch computes it through diffusers' own module code. Weights, factors
// and inputs are bf16-representable, so what separates the two is vpipe's
// bf16 activations alone.
//
// Engagement comes first: every bound module must end the step with a
// NONZERO gradient. A projection the backward forgot trains nothing and
// says nothing -- the training form of a tier declared and never
// dispatched -- and a numeric bar over the modules that did work cannot
// see it.
//
//   VPIPE_QWEN_IMAGE21_TRAIN_GOLDEN=<dir> vpipe_test \
//       --filter 'qwen_image_21_train*'

#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/qwen-image/metal-qwen-image21-transformer.h"
#include "generative-models/qwen-image/qwen-image21-layout.h"
#include "generative-models/qwen-image/qwen-image21-trainer.h"
#include "generative-models/train/lora-params.h"
#include "generative-models/train/optimizer.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::SharedBuffer;

namespace {

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
  std::vector<int>
  arr(const char* k) const
  {
    std::vector<int> v;
    auto o = man.as_object();
    if (!o.contains(k)) { return v; }
    const FlexData f = o.at(k);
    if (!f.is_array()) { return v; }
    auto a = f.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      v.push_back((int)a.at(i).as_int(0));
    }
    return v;
  }
  std::vector<float>
  f32(const std::string& nm) const
  {
    std::ifstream in(dir / (nm + ".f32"), std::ios::binary);
    std::vector<float> v;
    if (!in) { return v; }
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    v.resize((std::size_t)n / 4);
    in.read(reinterpret_cast<char*>(v.data()), n);
    return v;
  }
};

bool
load_golden_(Golden* g)
{
  const char* d = std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_GOLDEN");
  if (d == nullptr || *d == '\0') { return false; }
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

double
rel_l2_(const float* got, const std::vector<float>& want)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double e = (double)got[i] - (double)want[i];
    num += e * e;
    den += (double)want[i] * (double)want[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

// Everything a test needs from one golden: the model, its layout, the
// sample, and a trainer bound to the golden's own adapter.
struct Rig {
  Session sess;
  metal_compute::MetalCompute* mc = nullptr;
  std::unique_ptr<MetalQwenImage21Transformer> model;
  qi21::Layout lay;
  SharedBuffer x0, noise, txt;
  train::LoraParams params;
  std::unique_ptr<QwenImage21Trainer> trainer;
  int bound = 0;
  float sigma = 0.0f;

  bool
  build(const Golden& g, bool load_adapter, bool stream = false)
  {
    mc = sess.metal_compute();
    if (mc == nullptr) { return false; }
    MetalQwenImage21Transformer::Config cfg;
    const std::string ckpt = (g.dir / "train").string();
    if (!MetalQwenImage21Transformer::Config::read_dims(ckpt, &cfg)) {
      return false;
    }
    model = MetalQwenImage21Transformer::load(ckpt, mc, cfg, stream);
    if (model == nullptr) { return false; }
    const std::vector<int> slot_i = g.arr("slot");
    const std::vector<int> shp = g.arr("img_shapes");
    if (slot_i.empty() || shp.size() != 3) { return false; }
    std::vector<std::uint8_t> slot(slot_i.begin(), slot_i.end());
    std::string lerr;
    if (!qi21::build_layout(slot, {}, {qi21::ImgBlock{shp[0], shp[1], shp[2]}},
                            &lay, &lerr)) {
      return false;
    }
    x0 = up_bf16_(mc, g.f32("train_x0"));
    noise = up_bf16_(mc, g.f32("train_noise"));
    txt = up_bf16_(mc, g.f32("train_txt"));
    sigma = (float)g.getr("sigma", 0.6);
    std::string err;
    trainer = QwenImage21Trainer::create(model.get(), &err);
    if (trainer == nullptr) {
      std::printf("[qwen_image_21_train] trainer: %s\n", err.c_str());
      return false;
    }
    if (!params.create(mc, trainer->modules("attn_ff"), g.geti("rank", 4),
                       (float)g.getr("alpha", 8.0), &err)) {
      return false;
    }
    if (load_adapter) {
      int loaded = 0;
      if (!params.load((g.dir / "train_lora.safetensors").string(), &loaded,
                       &err) ||
          loaded != (int)params.modules().size()) {
        std::printf("[qwen_image_21_train] adapter: %s (%d)\n", err.c_str(),
                    loaded);
        return false;
      }
    } else {
      params.init_random(7);
    }
    bound = trainer->bind(&params, &err);
    if (bound == 0) {
      std::printf("[qwen_image_21_train] bind: %s\n", err.c_str());
    }
    return bound > 0;
  }

  QwenImage21Trainer::Sample
  sample() const
  {
    QwenImage21Trainer::Sample s;
    s.x0 = &x0;
    s.noise = &noise;
    s.txt = &txt;
    s.layout = &lay;
    s.sigma = sigma;
    return s;
  }
};

}  // namespace

// ONE STEP: the loss, the prediction, and every factor's gradient.
TEST(qwen_image_21_train, one_step_matches_autograd)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  Rig rig;
  ASSERT_TRUE(rig.build(g, /*load_adapter=*/true));
  if (!rig.trainer) { return; }
  EXPECT_TRUE(rig.bound == 2 * 7);

  QwenImage21Trainer::Result res;
  std::string err;
  ASSERT_TRUE(rig.trainer->step(rig.sample(), &res, &err, true));
  if (!err.empty()) { std::printf("[qwen_image_21_train] %s\n", err.c_str()); }

  const double want_loss = g.getr("loss", 0.0);
  std::printf("[qwen_image_21_train] loss %.6f, reference %.6f\n", res.loss,
              want_loss);
  EXPECT_TRUE(std::fabs(res.loss - want_loss) / want_loss < 1e-2);

  const std::vector<float> want_pred = g.f32("train_pred");
  if (!res.pred.empty() && !want_pred.empty()) {
    std::vector<float> got(want_pred.size());
    const auto* d = static_cast<const std::uint16_t*>(res.pred.contents());
    for (std::size_t i = 0; i < got.size(); ++i) {
      got[i] = train::bf16_to_f32(d[i]);
    }
    const double e = rel_l2_(got.data(), want_pred);
    std::printf("[qwen_image_21_train] prediction rel-L2 %.4f\n", e);
    EXPECT_TRUE(e < 0.05);
  }

  // Every module's two gradients, against the reference's.
  const auto* ga = static_cast<const float*>(rig.params.a_grad().contents());
  const auto* gb = static_cast<const float*>(rig.params.b_grad().contents());
  const char* shorts[] = {"q", "k", "v", "o", "gate", "proj", "out"};
  int zero = 0, bad = 0;
  double worst = 0.0;
  for (const auto& m : rig.params.modules()) {
    int L = -1;
    std::sscanf(m.name.c_str(), "transformer_blocks.%d.", &L);
    int j = 0;
    const std::string leaf = m.name.substr(m.name.find('.', 19) + 1);
    const char* leaves[] = {"attn.to_q", "attn.to_k", "attn.to_v",
                            "attn.to_out.0", "img_mlp.gate_layer",
                            "img_mlp.proj", "img_mlp.out"};
    for (int i = 0; i < 7; ++i) { if (leaf == leaves[i]) { j = i; } }
    const std::string sfx = std::to_string(L) + "_" + shorts[j];
    const std::vector<float> wa = g.f32("train_dA_" + sfx);
    const std::vector<float> wb = g.f32("train_dB_" + sfx);
    ASSERT_TRUE(wa.size() == (std::size_t)rig.params.rank() * m.k);
    ASSERT_TRUE(wb.size() == (std::size_t)m.n * rig.params.rank());
    const float* a = ga + m.a_off;
    const float* b = gb + m.b_off;
    double na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < wa.size(); ++i) { na += (double)a[i] * a[i]; }
    for (std::size_t i = 0; i < wb.size(); ++i) { nb += (double)b[i] * b[i]; }
    if (na == 0.0 || nb == 0.0) { ++zero; }
    const double ea = rel_l2_(a, wa);
    const double eb = rel_l2_(b, wb);
    worst = std::max({worst, ea, eb});
    std::printf("[qwen_image_21_train] %-34s dA %.4f  dB %.4f\n",
                m.name.c_str(), ea, eb);
    if (ea > 0.05 || eb > 0.05) { ++bad; }
  }
  std::printf("[qwen_image_21_train] worst gradient rel-L2 %.4f\n", worst);
  EXPECT_TRUE(zero == 0);
  EXPECT_TRUE(bad == 0);
}

namespace {

std::vector<std::uint16_t>
forward_words_(MetalQwenImage21Transformer& m, const Rig& rig)
{
  SharedBuffer xt = rig.mc->make_shared_buffer(rig.x0.byte_size());
  std::memcpy(xt.contents(), rig.x0.contents(), rig.x0.byte_size());
  MetalQwenImage21Transformer::Request req;
  req.latents = &xt;
  req.txt = &rig.txt;
  req.layout = &rig.lay;
  req.timestep = rig.sigma;
  std::string err;
  SharedBuffer out = m.forward(req, &err);
  std::vector<std::uint16_t> v;
  if (out.empty()) { return v; }
  const std::size_t n =
      (std::size_t)rig.lay.target_len * m.config().out_channels;
  const auto* d = static_cast<const std::uint16_t*>(out.contents());
  v.assign(d, d + n);
  return v;
}

std::size_t
diff_(const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b)
{
  if (a.size() != b.size() || a.empty()) { return (std::size_t)-1; }
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) { n += a[i] != b[i]; }
  return n;
}

}  // namespace

// ONE FORWARD. A fresh adapter (B = 0) leaves the model exactly the base;
// and an adapter the trainer saves, bound again from the file by the
// runtime-LoRA loader, runs exactly the forward the trainer ran.
TEST(qwen_image_21_train, a_saved_adapter_reloads_bit_exact)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  Rig rig;
  ASSERT_TRUE(rig.build(g, /*load_adapter=*/false));
  if (!rig.trainer) { return; }

  // B = 0: the base model, bit for bit.
  MetalQwenImage21Transformer::Config cfg = rig.model->config();
  const std::string ckpt = (g.dir / "train").string();
  auto base = MetalQwenImage21Transformer::load(ckpt, rig.mc, cfg, false);
  ASSERT_TRUE(base != nullptr);
  if (base == nullptr) { return; }
  const auto w_base = forward_words_(*base, rig);
  const auto w_init = forward_words_(*rig.model, rig);
  EXPECT_TRUE(diff_(w_base, w_init) == 0);

  // A trained-looking adapter, saved and bound from the file.
  int loaded = 0;
  std::string err;
  ASSERT_TRUE(rig.params.load((g.dir / "train_lora.safetensors").string(),
                              &loaded, &err));
  const auto w_live = forward_words_(*rig.model, rig);
  EXPECT_TRUE(diff_(w_base, w_live) > 0);
  const std::string path =
      (std::filesystem::temp_directory_path() / "vpipe-qi21-train.safetensors")
          .string();
  ASSERT_TRUE(rig.params.save(path, {{"vpipe.trigger_word", "ohwx"}}, &err));
  auto file = MetalQwenImage21Transformer::load(
      ckpt, rig.mc, cfg, false,
      {MetalQwenImage21Transformer::LoraSpec{path, 1.0f}});
  ASSERT_TRUE(file != nullptr);
  if (file == nullptr) { return; }
  EXPECT_TRUE(file->lora_modules(0) == rig.bound);
  const auto w_file = forward_words_(*file, rig);
  const std::size_t d = diff_(w_live, w_file);
  std::printf("[qwen_image_21_train] live vs reloaded: %zu words differ\n",
              d);
  EXPECT_TRUE(d == 0);
  std::filesystem::remove(path);
}

// A STREAMED step is the same step: the backward acquires blocks in
// reverse through the slots, and the gradients must not care.
TEST(qwen_image_21_train, streamed_step_matches_resident)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  std::vector<float> grads[2];
  double loss[2] = {0.0, 0.0};
  for (int arm = 0; arm < 2; ++arm) {
    Rig rig;
    ASSERT_TRUE(rig.build(g, /*load_adapter=*/true, /*stream=*/arm == 1));
    if (!rig.trainer) { return; }
    EXPECT_TRUE(rig.model->streaming() == (arm == 1));
    QwenImage21Trainer::Result res;
    std::string err;
    ASSERT_TRUE(rig.trainer->step(rig.sample(), &res, &err));
    loss[arm] = res.loss;
    const auto* a = static_cast<const float*>(rig.params.a_grad().contents());
    const auto* b = static_cast<const float*>(rig.params.b_grad().contents());
    grads[arm].assign(a, a + rig.params.a_elems());
    grads[arm].insert(grads[arm].end(), b, b + rig.params.b_elems());
  }
  EXPECT_TRUE(loss[0] == loss[1]);
  std::size_t diff = 0;
  for (std::size_t i = 0; i < grads[0].size(); ++i) {
    diff += grads[0][i] != grads[1][i];
  }
  std::printf("[qwen_image_21_train] streamed vs resident: %zu of %zu "
              "gradient values differ\n", diff, grads[0].size());
  EXPECT_TRUE(diff == 0);
}

// KEEP ALL: reading the forward's own intermediates instead of
// recomputing them is the same backward -- bit for bit on a GPU-only
// forward, where the recompute runs the same kernels on the same inputs.
TEST(qwen_image_21_train, keep_all_matches_recompute)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  std::vector<float> grads[2];
  double loss[2] = {0.0, 0.0};
  for (int arm = 0; arm < 2; ++arm) {
    Rig rig;
    ASSERT_TRUE(rig.build(g, /*load_adapter=*/true));
    if (!rig.trainer) { return; }
    rig.trainer->set_keep_all(arm == 1);
    QwenImage21Trainer::Result res;
    std::string err;
    ASSERT_TRUE(rig.trainer->step(rig.sample(), &res, &err));
    loss[arm] = res.loss;
    const auto* a = static_cast<const float*>(rig.params.a_grad().contents());
    const auto* b = static_cast<const float*>(rig.params.b_grad().contents());
    grads[arm].assign(a, a + rig.params.a_elems());
    grads[arm].insert(grads[arm].end(), b, b + rig.params.b_elems());
  }
  EXPECT_TRUE(loss[0] == loss[1]);
  std::size_t diff = 0;
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < grads[0].size(); ++i) {
    diff += grads[0][i] != grads[1][i];
    const double e = (double)grads[0][i] - grads[1][i];
    num += e * e;
    den += (double)grads[0][i] * grads[0][i];
  }
  std::printf("[qwen_image_21_train] keep-all vs recompute: %zu of %zu "
              "gradient values differ, rel-L2 %.2e\n", diff,
              grads[0].size(), std::sqrt(num / std::max(den, 1e-30)));
  EXPECT_TRUE(diff == 0);
}

// It LEARNS: a fresh adapter trained on one sample drives its loss down,
// under every optimizer.
// VALIDATION'S LOSS is the training step's, from the inference forward
// alone: the same prediction (the tap only copies), the same target, the
// same normalisation -- only the sum runs in double on the host rather
// than in f32 partials on the GPU.
TEST(qwen_image_21_train, a_forward_only_loss_matches_the_step)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  Rig rig;
  ASSERT_TRUE(rig.build(g, /*load_adapter=*/true));
  if (!rig.trainer) { return; }
  std::string err;
  QwenImage21Trainer::Result res;
  ASSERT_TRUE(rig.trainer->step(rig.sample(), &res, &err));
  double l = 0.0;
  ASSERT_TRUE(rig.trainer->loss(rig.sample(), &l, &err));
  const double rel = std::abs(l - res.loss) / std::max(res.loss, 1e-12);
  std::printf("[qwen_image_21_train] forward-only loss %.6f, step %.6f "
              "(rel %.1e)\n", l, res.loss, rel);
  EXPECT_TRUE(rel < 1e-5);
}

TEST(qwen_image_21_train, training_reduces_the_loss)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  for (const char* opt : {"adamw", "prodigy", "lion", "sgd"}) {
    Rig rig;
    ASSERT_TRUE(rig.build(g, /*load_adapter=*/false));
    if (!rig.trainer) { return; }
    train::TrainOps ops;
    std::string err;
    ASSERT_TRUE(ops.init(rig.mc, true, &err));
    train::OptimizerSpec spec;
    spec.optimizer = opt;
    spec.lr = std::string(opt) == "adamw" ? 1e-2
              : std::string(opt) == "lion" ? 1e-3
              : std::string(opt) == "sgd" ? 2.0 : 1.0;
    spec.warmup_steps = 0;
    spec.weight_decay = 0.0;
    const int steps = 30;
    train::Optimizer o;
    ASSERT_TRUE(o.init(&ops, &rig.params, spec, steps, &err));
    double first = 0.0, last = 0.0;
    for (int t = 0; t < steps; ++t) {
      QwenImage21Trainer::Result res;
      ASSERT_TRUE(rig.trainer->step(rig.sample(), &res, &err));
      double gn = 0.0, lr = 0.0;
      ASSERT_TRUE(o.step(1, &gn, &lr, &err));
      if (t == 0) { first = res.loss; }
      last = res.loss;
    }
    std::printf("[qwen_image_21_train] %-7s loss %.4f -> %.4f (d %.3g)\n",
                opt, first, last, o.prodigy_d());
    EXPECT_TRUE(std::isfinite(last));
    EXPECT_TRUE(last < 0.9 * first);
  }
}

// OPT-IN BENCH on the real checkpoint: a training step's forward and
// backward, GPU-only against the ANE tiers, at the two sizes training
// uses. VPIPE_QWEN_IMAGE21_TRAIN_BENCH=<the model's transformer dir>;
// VPIPE_QWEN_IMAGE21_TRAIN_BENCH_PX picks one size (default 512 and 1024).
TEST(qwen_image_21_train, bench_real_step)
{
  const char* dir = std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_BENCH");
  if (dir == nullptr || *dir == '\0') { return; }
  std::vector<int> sizes = {512, 1024};
  if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_BENCH_PX")) {
    sizes = {std::atoi(e)};
  }
  const bool i8 = std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_BENCH_I8") != nullptr;
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  // 0: GPU, recompute; 1: GPU, keep all; 2: ANE tiers, recompute;
  // 3: ANE q/k/v tier, keep all; 4: keep all + the ANE backward;
  // 5: recompute + every ANE tier.
  for (int arm = 0; arm < 6; ++arm) {
    if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_BENCH_ARM")) {
      if (std::atoi(e) != arm) { continue; }
    }
    MetalQwenImage21Transformer::Config cfg;
    ASSERT_TRUE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));
    cfg.ane_ffn = arm >= 2;
    cfg.ane_qkv = arm >= 2;
    cfg.i8_gemm = i8;
    auto model = MetalQwenImage21Transformer::load(dir, mc, cfg, false);
    ASSERT_TRUE(model != nullptr);
    if (model == nullptr) { return; }
    std::string err;
    auto tr = QwenImage21Trainer::create(model.get(), &err);
    ASSERT_TRUE(tr != nullptr);
    train::LoraParams params;
    ASSERT_TRUE(params.create(mc, tr->modules("attn_ff"), 16, 16.0f, &err));
    params.init_random(1);
    ASSERT_TRUE(tr->bind(&params, &err) > 0);
    tr->set_keep_all(arm == 1 || arm == 3 || arm == 4);
    tr->set_ane_backward(arm >= 4);
    for (int px : sizes) {
      const int grid = px / 16;
      const int txt_rows = 128;
      std::vector<std::uint8_t> slot(txt_rows, 0);
      slot.insert(slot.end(), (std::size_t)grid * grid / 4, 1);
      qi21::Layout lay;
      ASSERT_TRUE(qi21::build_layout(
          slot, {}, {qi21::ImgBlock{1, grid, grid}}, &lay, &err));
      std::mt19937 g(3);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      auto rnd = [&](std::size_t n) {
        std::vector<float> v(n);
        for (float& x : v) { x = nd(g); }
        return up_bf16_(mc, v);
      };
      SharedBuffer x0 = rnd((std::size_t)lay.target_len * 64);
      SharedBuffer ns = rnd((std::size_t)lay.target_len * 64);
      SharedBuffer tx = rnd((std::size_t)txt_rows * cfg.txt_dim);
      QwenImage21Trainer::Sample smp;
      smp.x0 = &x0; smp.noise = &ns; smp.txt = &tx; smp.layout = &lay;
      smp.sigma = 0.5f;
      double fsum = 0.0, bsum = 0.0;
      const int reps = 3;
      for (int r = 0; r <= reps; ++r) {
        QwenImage21Trainer::Result res;
        ASSERT_TRUE(tr->step(smp, &res, &err));
        if (r == 0) { continue; }     // warm: modules, residency, caches
        fsum += res.forward_ms;
        bsum += res.backward_ms;
      }
      std::printf("[qwen_image_21_train] bench %s%s%s %dpx (joint %d): "
                  "forward %.0f ms, backward %.0f ms, step %.0f ms\n",
                  arm >= 4 ? "ane+bwd" : arm >= 2 ? "ane" : "gpu",
                  i8 ? "+i8" : "",
                  (arm == 1 || arm == 3 || arm == 4) ? " keep" : "", px,
                  lay.joint_len, fsum / reps, bsum / reps,
                  (fsum + bsum) / reps);
    }
  }
}

// The ANE's backward, on the real checkpoint (it needs whole 256-row
// chunks, which the small golden never has): the factor gradients with
// the three backward GEMMs split onto the ANE, against the GPU's.
// VPIPE_QWEN_IMAGE21_TRAIN_BENCH=<transformer dir>.
TEST(qwen_image_21_train, ane_backward_matches_gpu)
{
  const char* dir = std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_BENCH");
  if (dir == nullptr || *dir == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  ASSERT_TRUE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));
  auto model = MetalQwenImage21Transformer::load(dir, mc, cfg, false);
  ASSERT_TRUE(model != nullptr);
  if (model == nullptr) { return; }
  std::string err;
  const int px = 512, grid = px / 16, txt_rows = 128;
  std::vector<std::uint8_t> slot(txt_rows, 0);
  slot.insert(slot.end(), (std::size_t)grid * grid / 4, 1);
  qi21::Layout lay;
  ASSERT_TRUE(qi21::build_layout(slot, {}, {qi21::ImgBlock{1, grid, grid}},
                                 &lay, &err));
  std::mt19937 g(5);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto rnd = [&](std::size_t n) {
    std::vector<float> v(n);
    for (float& x : v) { x = nd(g); }
    return up_bf16_(mc, v);
  };
  SharedBuffer x0 = rnd((std::size_t)lay.target_len * 64);
  SharedBuffer ns = rnd((std::size_t)lay.target_len * 64);
  SharedBuffer tx = rnd((std::size_t)txt_rows * cfg.txt_dim);
  std::vector<float> grads[2];
  double loss[2] = {0.0, 0.0};
  for (int arm = 0; arm < 2; ++arm) {
    auto tr = QwenImage21Trainer::create(model.get(), &err);
    ASSERT_TRUE(tr != nullptr);
    train::LoraParams params;
    ASSERT_TRUE(params.create(mc, tr->modules("attn_ff"), 16, 16.0f, &err));
    params.init_random(11);
    // A nonzero B, so the A factors see a gradient too.
    {
      std::mt19937 gb(12);
      std::normal_distribution<float> nb(0.0f, 0.02f);
      auto* bm = static_cast<float*>(params.b_master().contents());
      for (std::size_t i = 0; i < params.b_elems(); ++i) { bm[i] = nb(gb); }
      params.refresh_shadows_cpu();
    }
    ASSERT_TRUE(tr->bind(&params, &err) > 0);
    tr->set_keep_all(true);
    tr->set_ane_backward(arm == 1);
    QwenImage21Trainer::Sample smp;
    smp.x0 = &x0; smp.noise = &ns; smp.txt = &tx; smp.layout = &lay;
    smp.sigma = 0.5f;
    // A few steps, so an auto share has left its probes behind; the
    // gradients accumulate, so compare the last step's alone.
    for (int r = 0; r < 3; ++r) {
      QwenImage21Trainer::Result res;
      std::memset(params.a_grad().contents(), 0, params.a_elems() * 4);
      std::memset(params.b_grad().contents(), 0, params.b_elems() * 4);
      ASSERT_TRUE(tr->step(smp, &res, &err));
      loss[arm] = res.loss;
    }
    if (arm == 1) {
      std::printf("[qwen_image_21_train] ane backward armed: %d, loss "
                  "scale %.0f\n", tr->ane_backward_armed() ? 1 : 0,
                  (double)tr->loss_scale());
    }
    const auto* a = static_cast<const float*>(params.a_grad().contents());
    const auto* b = static_cast<const float*>(params.b_grad().contents());
    grads[arm].assign(a, a + params.a_elems());
    grads[arm].insert(grads[arm].end(), b, b + params.b_elems());
  }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < grads[0].size(); ++i) {
    const double e = (double)grads[1][i] - grads[0][i];
    num += e * e;
    den += (double)grads[0][i] * grads[0][i];
  }
  const double rel = std::sqrt(num / std::max(den, 1e-30));
  std::printf("[qwen_image_21_train] ane vs gpu backward: loss %.6f / "
              "%.6f, gradient rel-L2 %.4f\n", loss[0], loss[1], rel);
  EXPECT_TRUE(loss[0] == loss[1]);
  EXPECT_TRUE(rel < 0.05);
}

// I8 GEMM, forward AND backward. The backward's dX GEMMs run through the
// forward's own route over a transposed weight, so with i8_gemm on they
// take the int8 pipe too. The floors are the context's: M >= 1024 and
// K >= 1024 -- 512^2 is 1024 image rows plus the text, and the model's
// narrowest reduction is its hidden width (3072), so every block GEMM of
// the step clears them. Matrix-core GPUs only; elsewhere the context
// turns itself off and the test says so and returns.
//
// WHAT TO HOLD IT TO. Not a norm against bf16: on this checkpoint the
// early blocks' factor gradients swing by half for ANY small change in
// the forward -- bf16 itself, with the clean latent moved by 1%, moves
// the whole gradient by rel-L2 ~0.5 (blocks 0-8 by 0.5-0.7, the last by
// 0.02). So i8 is held to that CONTROL, to the well-conditioned last
// block, and to what matters: over a short run it must LEARN like bf16.
TEST(qwen_image_21_train, i8_gemm_trains_like_bf16)
{
  const char* dir = std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_BENCH");
  if (dir == nullptr || *dir == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[qwen_image_21_train] no matrix cores: i8 not tested\n");
    return;
  }
  MetalQwenImage21Transformer::Config cfg;
  ASSERT_TRUE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));
  const int px = 512, grid = px / 16, txt_rows = 128;
  std::vector<std::uint8_t> slot(txt_rows, 0);
  slot.insert(slot.end(), (std::size_t)grid * grid / 4, 1);
  qi21::Layout lay;
  std::string err;
  ASSERT_TRUE(qi21::build_layout(slot, {}, {qi21::ImgBlock{1, grid, grid}},
                                 &lay, &err));
  std::mt19937 g(5);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto rnd_host = [&](std::size_t n) {
    std::vector<float> v(n);
    for (float& x : v) { x = nd(g); }
    return v;
  };
  const std::vector<float> x0h = rnd_host((std::size_t)lay.target_len * 64);
  SharedBuffer x0 = up_bf16_(mc, x0h);
  SharedBuffer ns = up_bf16_(mc, rnd_host((std::size_t)lay.target_len * 64));
  SharedBuffer tx =
      up_bf16_(mc, rnd_host((std::size_t)txt_rows * cfg.txt_dim));
  SharedBuffer x0p;
  {
    std::vector<float> v = x0h;
    std::mt19937 gp(99);
    for (float& x : v) { x *= 1.0f + 0.01f * nd(gp); }
    x0p = up_bf16_(mc, v);
  }
  auto sample = [&](const SharedBuffer* x) {
    QwenImage21Trainer::Sample smp;
    smp.x0 = x; smp.noise = &ns; smp.txt = &tx; smp.layout = &lay;
    smp.sigma = 0.5f;
    return smp;
  };

  // 0: bf16; 1: bf16 with the latent moved by 1% (the control); 2: i8.
  // Recompute, not keep-all: they must fit the 24 GB boxes this runs on
  // beside a 13 GB model.
  constexpr int kArms = 3;
  std::vector<float> grads[kArms];
  double loss[kArms] = {};
  std::vector<double> curve[2];     // the short run: bf16, i8
  struct Mod {
    std::string name;
    std::size_t a_off = 0, na = 0, b_off = 0, nb = 0;   // into `grads`
  };
  std::vector<Mod> mods;
  bool engaged = false;
  const int rank = 16, run_steps = 10;
  std::unique_ptr<MetalQwenImage21Transformer> model;
  for (int arm = 0; arm < kArms; ++arm) {
    const bool i8 = arm == 2;
    if (arm != 1) {
      // One model at a time: the context is chosen at load.
      model.reset();
      cfg.i8_gemm = i8;
      model = MetalQwenImage21Transformer::load(dir, mc, cfg, false);
      ASSERT_TRUE(model != nullptr);
      if (model == nullptr) { return; }
      if (i8) { engaged = model->uses_i8_gemm(); }
    }
    auto tr = QwenImage21Trainer::create(model.get(), &err);
    ASSERT_TRUE(tr != nullptr);
    if (tr == nullptr) { return; }
    // The gradient: a nonzero B, so the A factors see one too.
    {
      train::LoraParams params;
      ASSERT_TRUE(params.create(mc, tr->modules("attn_ff"), rank, 16.0f,
                                &err));
      params.init_random(11);
      std::mt19937 gb(12);
      std::normal_distribution<float> nb(0.0f, 0.02f);
      auto* bm = static_cast<float*>(params.b_master().contents());
      for (std::size_t i = 0; i < params.b_elems(); ++i) { bm[i] = nb(gb); }
      params.refresh_shadows_cpu();
      ASSERT_TRUE(tr->bind(&params, &err) > 0);
      // Two steps: the first tunes the context's splits.
      for (int r = 0; r < 2; ++r) {
        QwenImage21Trainer::Result res;
        std::memset(params.a_grad().contents(), 0, params.a_elems() * 4);
        std::memset(params.b_grad().contents(), 0, params.b_elems() * 4);
        ASSERT_TRUE(tr->step(sample(arm == 1 ? &x0p : &x0), &res, &err));
        loss[arm] = res.loss;
      }
      const auto* a = static_cast<const float*>(params.a_grad().contents());
      const auto* b = static_cast<const float*>(params.b_grad().contents());
      grads[arm].assign(a, a + params.a_elems());
      grads[arm].insert(grads[arm].end(), b, b + params.b_elems());
      if (arm == 0) {
        for (const auto& m : params.modules()) {
          mods.push_back(Mod{m.name, m.a_off, (std::size_t)rank * m.k,
                             params.a_elems() + m.b_off,
                             (std::size_t)m.n * rank});
        }
      }
    }
    tr.reset();
    if (arm == 1) { continue; }
    // The short run, from the standard init (B = 0), on one sample.
    auto tr2 = QwenImage21Trainer::create(model.get(), &err);
    ASSERT_TRUE(tr2 != nullptr);
    if (tr2 == nullptr) { return; }
    train::LoraParams params;
    ASSERT_TRUE(params.create(mc, tr2->modules("attn_ff"), rank, 16.0f,
                              &err));
    params.init_random(11);
    ASSERT_TRUE(tr2->bind(&params, &err) > 0);
    train::TrainOps ops;
    ASSERT_TRUE(ops.init(mc, true, &err));
    train::OptimizerSpec spec;
    spec.optimizer = "adamw";
    spec.lr = 2e-3;
    spec.warmup_steps = 0;
    spec.weight_decay = 0.0;
    train::Optimizer o;
    ASSERT_TRUE(o.init(&ops, &params, spec, run_steps, &err));
    for (int t = 0; t < run_steps; ++t) {
      QwenImage21Trainer::Result res;
      ASSERT_TRUE(tr2->step(sample(&x0), &res, &err));
      double gn = 0.0, lr = 0.0;
      ASSERT_TRUE(o.step(1, &gn, &lr, &err));
      curve[i8 ? 1 : 0].push_back(res.loss);
    }
  }
  ASSERT_TRUE(engaged);
  if (curve[0].size() != (std::size_t)run_steps ||
      curve[1].size() != (std::size_t)run_steps) {
    return;
  }

  // The gradient against the control, whole and per block.
  auto rel_blocks = [&](int arm, int L) {
    double bn = 0.0, bd = 0.0;
    const std::string pre =
        L < 0 ? "" : "transformer_blocks." + std::to_string(L) + ".";
    for (const auto& m : mods) {
      if (m.name.compare(0, pre.size(), pre) != 0) { continue; }
      for (auto [off, n] : {std::pair{m.a_off, m.na},
                            std::pair{m.b_off, m.nb}}) {
        for (std::size_t i = off; i < off + n; ++i) {
          const double e = (double)grads[arm][i] - grads[0][i];
          bn += e * e;
          bd += (double)grads[0][i] * grads[0][i];
        }
      }
    }
    return std::sqrt(bn / std::max(bd, 1e-30));
  };
  const int last = cfg.n_layers - 1;
  const double ctrl = rel_blocks(1, -1), ctrl_last = rel_blocks(1, last);
  const double rel = rel_blocks(2, -1), rel_last = rel_blocks(2, last);
  std::printf("[qwen_image_21_train] gradient vs bf16: i8 rel-L2 %.4f "
              "(last block %.4f), a 1%% input change %.4f (%.4f); loss "
              "%.6f / %.6f\n", rel, rel_last, ctrl, ctrl_last, loss[2],
              loss[0]);
  std::string c0, c1;
  for (int t = 0; t < run_steps; ++t) {
    c0 += " " + std::to_string(curve[0][(std::size_t)t]).substr(0, 6);
    c1 += " " + std::to_string(curve[1][(std::size_t)t]).substr(0, 6);
  }
  std::printf("[qwen_image_21_train] adamw run, bf16:%s\n", c0.c_str());
  std::printf("[qwen_image_21_train] adamw run, i8:  %s\n", c1.c_str());

  // It engaged (it moved the gradient), it moved it no more than a small
  // input change does, the last block -- where nothing amplifies -- is
  // close, and the run learned at least 80% of what bf16's did.
  EXPECT_TRUE(rel > 0.0);
  EXPECT_TRUE(rel < 2.5 * ctrl);
  EXPECT_TRUE(rel_last < 0.1);
  const double gain0 = curve[0].front() - curve[0].back();
  const double gain1 = curve[1].front() - curve[1].back();
  EXPECT_TRUE(gain0 > 0.0);
  EXPECT_TRUE(gain1 > 0.8 * gain0);
}

// RESUME. A run stopped, saved and continued from its saved state is the
// run that was never stopped -- bit for bit, masters and moments both,
// since nothing in a step depends on anything but the state and the
// sample.
TEST(qwen_image_21_train, a_resumed_run_matches_one_never_stopped)
{
  Golden g;
  if (!load_golden_(&g)) { return; }
  const std::string state =
      (std::filesystem::temp_directory_path() / "vpipe-qi21-resume.bin")
          .string();
  train::OptimizerSpec spec;
  spec.optimizer = "adamw";
  spec.lr = 1e-2;
  spec.warmup_steps = 0;
  const int total = 6;
  auto masters = [](const train::LoraParams& p) {
    std::vector<float> v(p.a_elems() + p.b_elems());
    std::memcpy(v.data(), p.a_master().contents(), p.a_elems() * 4);
    std::memcpy(v.data() + p.a_elems(), p.b_master().contents(),
                p.b_elems() * 4);
    return v;
  };
  // steps [from, to) on a fresh rig, loading `load` first when given and
  // saving to `save` after when given.
  auto run = [&](int from, int to, const std::string* load,
                 const std::string* save, std::vector<float>* out) {
    Rig rig;
    if (!rig.build(g, /*load_adapter=*/false)) { return false; }
    train::TrainOps ops;
    std::string err;
    if (!ops.init(rig.mc, true, &err)) { return false; }
    train::Optimizer o;
    if (!o.init(&ops, &rig.params, spec, total, &err)) { return false; }
    if (load != nullptr && !o.load_state(*load, &err)) {
      std::printf("[qwen_image_21_train] load: %s\n", err.c_str());
      return false;
    }
    if (o.steps_taken() != from) { return false; }
    for (int t = from; t < to; ++t) {
      QwenImage21Trainer::Result res;
      if (!rig.trainer->step(rig.sample(), &res, &err)) { return false; }
      double gn = 0.0, lr = 0.0;
      if (!o.step(1, &gn, &lr, &err)) { return false; }
    }
    if (save != nullptr && !o.save_state(*save, &err)) { return false; }
    *out = masters(rig.params);
    return true;
  };
  std::vector<float> straight, first_half, resumed;
  ASSERT_TRUE(run(0, total, nullptr, nullptr, &straight));
  ASSERT_TRUE(run(0, total / 2, nullptr, &state, &first_half));
  ASSERT_TRUE(run(total / 2, total, &state, nullptr, &resumed));
  ASSERT_TRUE(straight.size() == resumed.size() && !straight.empty());
  std::size_t diff = 0;
  for (std::size_t i = 0; i < straight.size(); ++i) {
    diff += straight[i] != resumed[i];
  }
  std::printf("[qwen_image_21_train] resumed vs straight: %zu of %zu "
              "master values differ\n", diff, straight.size());
  EXPECT_TRUE(diff == 0);
  // And the halfway point really is halfway: it moved from the start and
  // is not yet the end.
  std::size_t moved = 0;
  for (std::size_t i = 0; i < straight.size(); ++i) {
    moved += first_half[i] != straight[i];
  }
  EXPECT_TRUE(moved > 0);
  std::filesystem::remove(state);
}
