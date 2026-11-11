// MiniMax-H3-Image (MiniMaxAI/MiniMax-H3-Image): the rules, the layout,
// the sampling grid, and -- against tools/dump_minimax_h3_image_golden.py
// -- the audio-less DiT with its per-step heads and the image VAE (H3's
// encoder, a 2D ViT decoder).
//
// Env: VPIPE_MINIMAX_H3_IMAGE_PATH (the repo root: transformer/, vae/,
// pdd_fused_heads.safetensors, model_index.json, pdd_config.json) and
// VPIPE_MINIMAX_H3_IMAGE_GOLDEN (the dumper's output: `dit`, `vae`). The
// rules tests need neither.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/minimax-h3/minimax-h3-image.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/minimax-h3/minimax-h3-scheduler.h"
#include "generative-models/weight-set.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/vae-decode-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

namespace fs = std::filesystem;

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return v != nullptr ? std::string(v) : std::string();
}

template <typename T>
std::vector<T>
read_raw_(const std::string& p)
{
  std::ifstream f(p, std::ios::binary);
  std::vector<T> v;
  if (!f) { return v; }
  f.seekg(0, std::ios::end);
  const std::size_t n = (std::size_t)f.tellg() / sizeof(T);
  f.seekg(0);
  v.resize(n);
  f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * sizeof(T)));
  return v;
}

FlexData
read_json_(const std::string& p)
{
  std::ifstream f(p);
  if (!f) { return FlexData(); }
  try {
    return FlexData::from_json(f);
  } catch (...) {
    return FlexData();
  }
}

float
bf_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t
rne_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

SharedBuffer
bf16_buf_(MetalCompute* mc, const std::vector<float>& v)
{
  SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
  if (b.empty()) { return b; }
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { d[i] = rne_(v[i]); }
  return b;
}

double
rel_bf16_(const SharedBuffer& got, const std::vector<float>& ref,
          std::size_t n)
{
  const auto* g = static_cast<const std::uint16_t*>(got.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n && i < ref.size(); ++i) {
    const double d = (double)bf_(g[i]) - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : 1.0;
}

double
rel_f32_(const std::vector<float>& a, const std::vector<float>& b)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : 1.0;
}

}  // namespace

// THE REFERENCE'S ARITHMETIC: the canvas a request is generated on, an
// edit picture's size for each consumer, and the grid as the release
// steps it. Expected values computed by the reference's own functions
// (diffusers' minimax_h3_image get_image_canvas_size / bound_short_edge,
// and the set-timesteps step's float32 round trip).
TEST(minimax_h3_image, sizing_and_grid_follow_the_reference)
{
  struct C { int h, w; std::int64_t area; int eh, ew; };
  // 1024^2 at 4 MP doubles; 4736x3552 is past it and only rounds; a
  // 1000x1500 request scales 1.672x to 1664x2496.
  for (const C c : {C{1024, 1024, h3i::kDefaultMinArea, 2048, 2048},
                    C{3552, 4736, h3i::kDefaultMinArea, 3552, 4736},
                    C{1500, 1000, h3i::kDefaultMinArea, 2496, 1664},
                    C{1000, 1000, 1024, 992, 992},
                    C{720, 1280, h3i::kDefaultMinArea, 1536, 2720}}) {
    int h = 0, w = 0;
    h3i::canvas_size(c.h, c.w, c.area, h3i::kCanvasMultiple, &h, &w);
    std::printf("[minimax_h3_image] canvas %dx%d at %lld -> %dx%d\n", c.w,
                c.h, (long long)c.area, w, h);
    EXPECT_TRUE(h == c.eh && w == c.ew);
  }
  struct E { int w, h, lo, hi, ew, eh; };
  for (const E e : {E{1024, 1536, 1024, 2048, 1024, 1536},   // kept
                    E{3300, 2200, 1024, 2048, 3072, 2048},   // down
                    E{800, 600, 1024, 2048, 1376, 1024},     // up
                    E{1500, 1200, 1024, 1024, 1280, 1024}}) {
    int w = 0, h = 0;
    h3i::bound_short_edge(e.w, e.h, e.lo, e.hi, h3i::kCanvasMultiple, &w,
                          &h);
    std::printf("[minimax_h3_image] picture %dx%d in [%d, %d] -> %dx%d\n",
                e.w, e.h, e.lo, e.hi, w, h);
    EXPECT_TRUE(w == e.ew && h == e.eh);
  }
  EXPECT_TRUE(h3i::picture_label(3) == "Picture 3: ");
  // The grid: the released sigmas each through f32(1 - sigma) and back.
  // Against the reference's own float32 arithmetic where it is dumped.
  const h3i::Manifest man;
  const std::vector<float> g = h3i::grid_as_run(man.sigmas);
  ASSERT_TRUE(g.size() == 7 && g.front() == 1.0f && g.back() == 0.0f);
  EXPECT_TRUE(g[3] == 0.75f);
  const std::string gd = env_("VPIPE_MINIMAX_H3_IMAGE_GOLDEN");
  if (!gd.empty() && fs::exists(fs::path(gd) / "grid_sigmas.f32")) {
    const auto ref = read_raw_<float>(gd + "/grid_sigmas.f32");
    const auto rt = read_raw_<float>(gd + "/grid_timesteps.f32");
    ASSERT_TRUE(ref.size() == g.size() && rt.size() + 1 == g.size());
    MiniMaxH3Scheduler sv(1.0);
    std::vector<double> gd64(g.begin(), g.end());
    ASSERT_TRUE(sv.set_sigmas_mapped(gd64, 1.0));
    for (std::size_t i = 0; i < g.size(); ++i) {
      EXPECT_TRUE(g[i] == ref[i] && sv.sigmas()[i] == ref[i]);
      if (i < rt.size()) { EXPECT_TRUE(sv.timesteps()[i] == rt[i]); }
    }
  }
}

// THE VELOCITY FORM: x + (sigma - sigma_next) v, every operation in f32
// on the grid's own sigmas -- not the blend form step() takes, which is
// the same update in exact arithmetic and differs in the last bit.
TEST(minimax_h3_image, the_velocity_step_is_the_released_update)
{
#pragma clang fp contract(off)
  const h3i::Manifest man;
  const std::vector<float> g = h3i::grid_as_run(man.sigmas);
  MiniMaxH3Scheduler s(1.0);
  ASSERT_TRUE(s.set_sigmas_mapped(std::vector<double>(g.begin(), g.end()),
                                  1.0));
  std::vector<float> x(64), v(64);
  for (int i = 0; i < 64; ++i) {
    x[(std::size_t)i] = std::sin(0.37f * (float)i);
    v[(std::size_t)i] = std::cos(0.11f * (float)i) * 1.7f;
  }
  int differ = 0;
  for (int step = 0; step < 6; ++step) {
    std::vector<float> a = x, b = x;
    ASSERT_TRUE(s.step_velocity(v.data(), step, a.data(), a.size()));
    ASSERT_TRUE(s.step(v.data(), step, b.data(), b.size()));
    const float d = g[(std::size_t)step] - g[(std::size_t)step + 1];
    for (std::size_t i = 0; i < x.size(); ++i) {
      const float dv = d * v[i];
      EXPECT_TRUE(a[i] == x[i] + dv);
      differ += a[i] != b[i] ? 1 : 0;
    }
  }
  std::printf("[minimax_h3_image] velocity form vs blend form: %d of %d "
              "values differ in the last bits\n", differ, 6 * 64);
}

// THE MANIFEST, off the release's own files: the grid, the heads' file,
// and where the text encoder lives (MiniMax-H3's, not this repository's).
TEST(minimax_h3_image, the_manifest_is_read_off_the_release)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_IMAGE_PATH");
  if (root.empty()) { return; }
  ASSERT_TRUE(h3i::is_root(root));
  h3i::Manifest m;
  std::string err;
  ASSERT_TRUE(h3i::read_manifest(root, &m, &err));
  std::printf("[minimax_h3_image] %d steps, heads %s, text encoder %s/%s\n",
              m.steps(), m.heads_file.c_str(), m.text_encoder_repo.c_str(),
              m.text_encoder_subfolder.c_str());
  EXPECT_TRUE(m.steps() == 6 && !m.heads_file.empty() &&
              m.text_encoder_repo == "MiniMaxAI/MiniMax-H3" &&
              m.text_encoder_subfolder == "text_encoder");
  MetalMiniMaxH3Transformer::Config c;
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(root, c, &err));
  EXPECT_TRUE(!c.audio && c.hidden == 5376 && c.n_layers == 50);
}

// THE LAYOUT: [text | a reference picture | the picture], no audio rows,
// the reference on the integer slot after the text and the picture on the
// next -- H3's reference layout with no soundtrack. Every position and tag
// equal to the reference's own builder (the dumper's restatement of
// build_image_packed_sequence).
TEST(minimax_h3_image, the_layout_is_the_reference_s)
{
  const std::string gd = env_("VPIPE_MINIMAX_H3_IMAGE_GOLDEN");
  if (gd.empty() || !fs::exists(fs::path(gd) / "dit.json")) { return; }
  FlexData j = read_json_(gd + "/dit.json");
  ASSERT_TRUE(j.is_object());
  auto o = j.as_object();
  std::vector<int> tags;
  {
    FlexData t = o.at("text_tags");
    auto a = t.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      tags.push_back((int)a.at(i).as_int(0));
    }
  }
  minimax_h3::Reference r;
  r.kind = minimax_h3::Reference::Kind::kImage;
  r.num_latent_frames = 1;
  r.latent_height = (int)o.at("ref_lh").as_int(0);
  r.latent_width  = (int)o.at("ref_lw").as_int(0);
  minimax_h3::PackedLayout L;
  ASSERT_TRUE(minimax_h3::build_ref2va_packed_sequence(
      tags, {r}, 1, (int)o.at("lh").as_int(0), (int)o.at("lw").as_int(0), 0,
      2, 2, minimax_h3::kAudioChannels, &L));
  const auto pos = read_raw_<double>(gd + "/dit_pos.f64");
  const auto rtag = read_raw_<std::int32_t>(gd + "/dit_tags.i32");
  ASSERT_TRUE((int)rtag.size() == L.seq_len &&
              pos.size() == (std::size_t)3 * L.seq_len);
  int bad_pos = 0, bad_tag = 0;
  for (int i = 0; i < L.seq_len; ++i) {
    for (int a = 0; a < 3; ++a) {
      const std::size_t k = (std::size_t)i * 3 + (std::size_t)a;
      if (L.position_ids[k] != pos[k]) { ++bad_pos; }
    }
    if (L.token_tags[(std::size_t)i] != rtag[(std::size_t)i]) { ++bad_tag; }
  }
  std::printf("[minimax_h3_image] layout: %d rows (%d text, %d reference), "
              "%d positions and %d tags off\n", L.seq_len, L.num_text_rows,
              L.num_condition_video_rows, bad_pos, bad_tag);
  EXPECT_TRUE(bad_pos == 0 && bad_tag == 0 && L.audio_indices.empty() &&
              L.num_condition_video_rows == (int)o.at("n_ref_rows").as_int(0));
}

// THE DiT: two real blocks with the real refiner, embeddings and final
// norm, through the per-step head of step 2, over the layout above with
// the reference rows held at t = 0.999 -- against the fp32 restatement at
// most 1.25x the release's own bf16 distance. And the checkpoint's own
// head (no step head) is a different answer: the heads are what the
// steps run.
TEST(minimax_h3_image, dit_matches_reference)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_IMAGE_PATH");
  const std::string gd = env_("VPIPE_MINIMAX_H3_IMAGE_GOLDEN");
  if (root.empty() || gd.empty() || !fs::exists(fs::path(gd) / "dit.json")) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  FlexData j = read_json_(gd + "/dit.json");
  auto o = j.as_object();
  MetalMiniMaxH3Transformer::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(root, cfg, &err));
  cfg.n_layers = (int)o.at("layers").as_int(2);
  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg, false, {});
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  h3i::Manifest man;
  ASSERT_TRUE(h3i::read_manifest(root, &man, &err));
  ASSERT_TRUE(m->set_step_heads(WeightSet::open(man.heads_file, nullptr),
                                &err));
  EXPECT_TRUE(m->step_heads() == 6);
  std::vector<int> tags;
  {
    FlexData t = o.at("text_tags");
    auto a = t.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      tags.push_back((int)a.at(i).as_int(0));
    }
  }
  minimax_h3::Reference r;
  r.kind = minimax_h3::Reference::Kind::kImage;
  r.num_latent_frames = 1;
  r.latent_height = (int)o.at("ref_lh").as_int(0);
  r.latent_width  = (int)o.at("ref_lw").as_int(0);
  minimax_h3::PackedLayout L;
  ASSERT_TRUE(minimax_h3::build_ref2va_packed_sequence(
      tags, {r}, 1, (int)o.at("lh").as_int(0), (int)o.at("lw").as_int(0), 0,
      cfg.patch_h, cfg.patch_w, minimax_h3::kAudioChannels, &L));
  const float t_gen = (float)o.at("t_gen").as_real(0.0);
  std::vector<float> uniq;
  std::vector<int> row_idx;
  minimax_h3::build_row_timesteps(L, t_gen, t_gen, h3i::kReferenceNoiseAug,
                                  &uniq, &row_idx);
  const auto text = read_raw_<float>(gd + "/dit_text.f32");
  const auto video = read_raw_<float>(gd + "/dit_video.f32");
  const auto ref = read_raw_<float>(gd + "/dit_out.f32");
  const auto bar = read_raw_<float>(gd + "/dit_bf16.f32");
  SharedBuffer tb = bf16_buf_(mc, text), vb = bf16_buf_(mc, video);
  SharedBuffer ab = mc->make_shared_buffer(16);
  MetalMiniMaxH3Transformer::Step st;
  st.video = &vb;  st.audio = &ab;  st.text = &tb;
  st.layout = &L;  st.timesteps = &uniq;
  st.row_timestep_index = &row_idx;
  st.out_head = (int)o.at("step").as_int(2);
  auto v = m->forward(st, &err);
  if (v.empty()) { std::printf("[minimax_h3_image] %s\n", err.c_str()); }
  ASSERT_TRUE(!v.empty());
  if (v.empty()) { return; }
  const double ours = rel_bf16_(v.video, ref, ref.size());
  const double rb = rel_f32_(bar, ref);
  // The heads sit 1-2% from the checkpoint's own (and from each other):
  // the wrong one is a small error, but a measurable one.
  auto other = [&](int head) {
    st.out_head = head;
    auto w = m->forward(st, &err);
    return w.empty() ? 0.0 : rel_bf16_(w.video, ref, ref.size());
  };
  const double own_d = other(-1), first_d = other(0);
  std::printf("[minimax_h3_image] DiT, %d blocks, step %d's head: %.5f from "
              "fp32 (the reference's bf16 %.5f); the checkpoint's own head "
              "%.5f, step 0's %.5f\n", cfg.n_layers,
              (int)o.at("step").as_int(2), ours, rb, own_d, first_d);
  EXPECT_TRUE(ours < 1.25 * rb + 1e-4);
  EXPECT_TRUE(own_d > 1.3 * ours && first_d > 1.3 * ours);
  // A head it does not have is refused.
  st.out_head = 99;
  EXPECT_TRUE(m->forward(st, &err).empty());
}

// THE IMAGE VAE: H3's encoder over one frame, tiled at 256 px with the
// release's blend, against the fp32 restatement (the release runs it in
// fp32; ours keeps H3's bf16 activations: bar 2.5e-2); and the 2D decoder
// -- (h, w) rotary, one pixel frame a token -- on one tile and a 2 x 2
// tiling, at most 1.25x the release's own bf16 distance.
TEST(minimax_h3_image, vae_matches_reference)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_IMAGE_PATH");
  const std::string gd = env_("VPIPE_MINIMAX_H3_IMAGE_GOLDEN");
  if (root.empty() || gd.empty() || !fs::exists(fs::path(gd) / "vae.json")) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string vdir = MetalMiniMaxH3VideoVae::resolve_vae_dir(root);
  MetalMiniMaxH3VideoVae::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(vdir, cfg, &err));
  std::printf("[minimax_h3_image] VAE %s: %d blocks x %d, %d heads, rope %d "
              "axes over %d channels\n", vdir.c_str(), cfg.n_layers, cfg.dim,
              cfg.n_heads, cfg.rope_axes, cfg.rope_dim());
  EXPECT_TRUE(cfg.rope_axes == 2 && cfg.dim == 1024 && cfg.n_layers == 24 &&
              cfg.out_patch_t() == 1 && cfg.t1_direct &&
              cfg.patch_elems() == 768 && cfg.rope_dim() == 32);
  auto m = MetalMiniMaxH3VideoVae::load(vdir, mc, cfg);
  ASSERT_TRUE(m != nullptr && m->can_decode_image());
  if (m == nullptr) { return; }
  struct Case { const char* name; int h, w; };
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float sd[3] = {0.229f, 0.224f, 0.225f};
  for (const Case c : {Case{"a", 352, 480}, Case{"b", 256, 256}}) {
    const std::string p = gd + "/vae_" + c.name;
    const auto rgb = read_raw_<std::uint8_t>(p + "_rgb.u8");
    const auto mom = read_raw_<float>(p + "_moments.f32");
    const auto ref = read_raw_<float>(p + "_out.f32");
    const auto bar = read_raw_<float>(p + "_bf16.f32");
    const int lh = c.h / 16, lw = c.w / 16;
    ASSERT_TRUE(rgb.size() == (std::size_t)3 * c.h * c.w &&
                mom.size() == (std::size_t)48 * lh * lw);
    // Encode: ImageNet-normalized bf16 [3, 1, H, W].
    std::vector<float> x(rgb.size());
    for (int ch = 0; ch < 3; ++ch) {
      for (std::size_t i = 0; i < (std::size_t)c.h * c.w; ++i) {
        const std::size_t k = (std::size_t)ch * c.h * c.w + i;
        x[k] = ((float)rgb[k] / 255.0f - mean[ch]) / sd[ch];
      }
    }
    int lf = 0;
    SharedBuffer xb = bf16_buf_(mc, x);
    SharedBuffer got = m->encode_video(xb, 1, c.h, c.w, &lf, &err);
    ASSERT_TRUE(!got.empty() && lf == 1);
    if (got.empty()) { std::printf("  %s\n", err.c_str()); return; }
    const std::size_t nz = (std::size_t)24 * lh * lw;
    std::vector<float> mref(mom.begin(), mom.begin() + (long)nz);
    const double enc = rel_bf16_(got, mref, nz);
    // Decode the reference's own mean.
    SharedBuffer zb = bf16_buf_(mc, mref);
    SharedBuffer out = m->decode_image(zb, lh, lw, &err);
    ASSERT_TRUE(!out.empty());
    if (out.empty()) { std::printf("  %s\n", err.c_str()); return; }
    const double ours = rel_bf16_(out, ref, ref.size());
    const double rb = rel_f32_(bar, ref);
    std::printf("[minimax_h3_image] VAE %s %dx%d: encode %.5f from fp32; "
                "decode %.5f (the reference's bf16 %.5f)\n", c.name, c.w,
                c.h, enc, ours, rb);
    // The release encodes in fp32; this encoder keeps bf16 activations
    // (H3's, shared): 1.5% from fp32 here, the conditioning's own
    // posterior noise far above it.
    EXPECT_TRUE(enc < 2.5e-2);
    EXPECT_TRUE(ours < 1.25 * rb + 1e-4);
  }
}

namespace {

// One F32 latent [24, 1, lh, lw] with a sideband, then the end.
class SourceLatent : public TypedStage<SourceLatent> {
public:
  static constexpr const char* kTypeName = "ut-source-latent-h3i";
  SourceLatent(const SessionContextIntf* s, std::string id,
               std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  int lh = 0, lw = 0;
  FlexData sideband;
  bool done = false;
  Job
  process(RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::F32;
      tb->shape = {24, 1, lh, lw};
      const std::size_t n = (std::size_t)24 * lh * lw;
      tb->resize_contiguous(n);
      std::uint32_t x = 1234u;
      for (std::size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        tb->as_f32()[i] = ((float)(x >> 8) / 16777216.0f - 0.5f) * 2.0f;
      }
      tb->sideband = sideband;
      co_await ctx.write(0, std::move(tb));
    }
    ctx.signal_done();
    co_return;
  }
};

class SinkPicture : public TypedStage<SinkPicture> {
public:
  static constexpr const char* kTypeName = "ut-sink-picture-h3i";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> captured;
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

}  // namespace

// vae-decode returns a MiniMax-H3-Image picture at the size its latent's
// sideband asks (sideband::kResizeTo) WHETHER OR NOT it keeps its VAE: a
// tight box lets the VAE go after each beat, and the 2D decoder was
// asked about after that -- the picture came back at the canvas,
// unresized (an edit with its encoder in the graph: 2048^2 for 1024^2).
TEST(minimax_h3_image, a_picture_is_resized_back_with_its_vae_let_go)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_IMAGE_PATH");
  if (root.empty()) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  for (const char* unload : {"never", "always"}) {
    auto pl = std::make_unique<Pipeline>("h3i-resize", &sess);
    auto su = std::make_unique<SourceLatent>(&sess, "src",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
    su->lh = 16;   // a 256 x 256 canvas
    su->lw = 16;
    FlexData sb = FlexData::make_object();
    FlexData sz = FlexData::make_array();
    sz.as_array().push_back(FlexData::make_int(128));
    sz.as_array().push_back(FlexData::make_int(96));
    sb.as_object().insert_or_assign(sideband::kResizeTo, std::move(sz));
    sb.as_object().insert_or_assign(sideband::kFrames,
                                    FlexData::make_int(1));
    su->sideband = std::move(sb);
    auto* src = pl->insert_stage(std::move(su));
    FlexData vc = FlexData::make_object();
    vc.as_object().insert("hf_dir", FlexData::make_string(root));
    vc.as_object().insert("unload_when_idle", FlexData::make_string(unload));
    auto* dec = pl->insert_stage(std::make_unique<VaeDecodeStage>(
        &sess, "dec", std::vector<InEdge>{{src, 0}}, std::move(vc)));
    auto* sink = static_cast<SinkPicture*>(pl->insert_stage(
        std::make_unique<SinkPicture>(&sess, "sink",
                                      std::vector<InEdge>{{dec, 0}},
                                      FlexData::make_object())));
    PipelineRuntime rt(pl.get(), &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    ASSERT_TRUE(sink->captured.size() == 1);
    if (sink->captured.size() != 1) { return; }
    const auto* tb =
        dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
    ASSERT_TRUE(tb != nullptr && tb->shape.size() == 3);
    if (tb == nullptr || tb->shape.size() != 3) { return; }
    std::printf("[minimax_h3_image] unload %s: decoded 256x256, emitted "
                "%lldx%lld\n", unload, (long long)tb->shape[2],
                (long long)tb->shape[1]);
    EXPECT_TRUE(tb->shape[0] == 3 && tb->shape[1] == 96 &&
                tb->shape[2] == 128);
  }
}
