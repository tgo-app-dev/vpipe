// P-H3-Image (PrunaAI/P-H3-Image): the rules, the layouts at one latent
// frame, and -- against tools/dump_p_h3_image_golden.py -- the connector,
// the text and edit conditioning, and the single-frame image decoder.
//
// Env: VPIPE_P_H3_IMAGE_PATH (the repo root: text_encoder/, text_connector/,
// tokenizer/, vae/, modular_model_index.json; transformer/ not needed) and
// VPIPE_P_H3_IMAGE_GOLDEN (the dumper's output: `text`, `edit`, `vae`).
// The rules and layout tests need neither.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"
#include "generative-models/minimax-h3/p-h3-image.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/weight-set.h"
#include "stages/model-memory.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

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

// rel-L2 of a bf16 buffer's rows [row0, rows) against an f32 reference.
double
rel_rows_(const SharedBuffer& got, const std::vector<float>& ref, int width,
          int row0, int rows)
{
  const auto* g = static_cast<const std::uint16_t*>(got.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = (std::size_t)row0 * width;
       i < (std::size_t)rows * width && i < ref.size(); ++i) {
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

// The dumper's prompts (tools/dump_p_h3_image_golden.py PROMPTS).
const char* const kTextPrompts[2] = {
    "A German Expressionist oil painting of a Berlin street at night in the "
    "manner of Ernst Ludwig Kirchner, with gaunt figures hurrying past a "
    "glowing gas lamp.",
    "A macro photograph of a dew-covered spider web at sunrise, the droplets "
    "catching warm orange light, shallow depth of field, the background a "
    "soft blur of green meadow. A small brass sign in the corner reads "
    "\"MORNING\".",
};

}  // namespace

// THE REFERENCE'S ARITHMETIC, Python's round() included: half to even,
// which a C round() (half away from zero) gets wrong at every .5 -- 48 /
// 32 and 80 / 32 both land on 64 here. Expected values computed by the
// reference's own functions (diffusers' p_h3_image before_encoder.py).
TEST(p_h3_image, sizing_follows_the_reference)
{
  EXPECT_TRUE(ph3::py_round(2.5) == 2 && ph3::py_round(3.5) == 4 &&
              ph3::py_round(-2.5) == -2);
  struct Case { int w, h, ew, eh; };
  for (const Case c : {Case{1920, 1080, 1344, 736}, Case{4000, 3000, 1152, 864},
                       Case{800, 600, 1152, 864}, Case{1024, 1024, 1024, 1024},
                       Case{1000, 1000, 992, 992}, Case{3000, 500, 2464, 416},
                       Case{48, 40, 1088, 928}}) {
    int w = 0, h = 0;
    ph3::size_at_area(c.w, c.h, 1000000, 32, &w, &h);
    EXPECT_TRUE(w == c.ew && h == c.eh);
  }
  struct Rng { int w, h, ew, eh, fw, fh; };
  for (const Rng c : {Rng{1920, 1080, 1920, 1088, 2720, 1536},
                      Rng{4000, 3000, 3360, 2496, 3360, 2496},
                      Rng{800, 600, 1184, 896, 2368, 1760},
                      Rng{5000, 4000, 3232, 2592, 3232, 2592},
                      Rng{1500, 1000, 1504, 992, 2496, 1664}}) {
    int w = 0, h = 0, fw = 0, fh = 0;
    ph3::size_in_area_range(c.w, c.h, 1048576, 8388608, 32, &w, &h);
    ph3::size_in_area_range(c.w, c.h, 4194304, 8388608, 32, &fw, &fh);
    EXPECT_TRUE(w == c.ew && h == c.eh && fw == c.fw && fh == c.fh);
  }
  struct Rc { int h, w, eh, ew; };
  for (const Rc c : {Rc{2912, 2912, 2912, 2912}, Rc{1080, 1920, 1088, 1920},
                     Rc{1000, 1000, 992, 992}, Rc{16, 16, 32, 32},
                     Rc{48, 80, 64, 64}, Rc{1040, 1040, 1024, 1024}}) {
    int h = 0, w = 0;
    ph3::round_canvas(c.h, c.w, 32, &h, &w);
    EXPECT_TRUE(h == c.eh && w == c.ew);
  }
  // The edit prompt: prefixed once, never twice; a brief kept as written.
  EXPECT_TRUE(ph3::edit_prompt("make it red") ==
              "Edit <Picture 1>. make it red");
  EXPECT_TRUE(ph3::edit_prompt("  Edit <Picture 1>. Keep the cat.") ==
              "  Edit <Picture 1>. Keep the cat.");
  EXPECT_TRUE(ph3::edit_prompt("subject_definitions:\n...") ==
              "subject_definitions:\n...");
}

// The manifest: the released grid and sizes read, a grid that is not a
// 1 -> 0 decreasing one refused, another pipeline's manifest refused.
TEST(p_h3_image, the_manifest_is_read_and_checked)
{
  const fs::path d =
      fs::temp_directory_path() /
      ("p-h3-manifest-" + std::to_string(::getpid()));
  fs::create_directories(d);
  auto write = [&](const std::string& body) {
    std::ofstream(d / ph3::kManifestFile) << body;
  };
  write("{\"_class_name\": \"PH3ImageModularPipeline\", \"sample_sigmas\": "
        "[1.0, 0.5, 0.0], \"canvas_min_pixels\": 1000}");
  ph3::Manifest m;
  std::string err;
  EXPECT_TRUE(ph3::is_root(d.string()));
  EXPECT_TRUE(ph3::read_manifest(d.string(), &m, &err));
  EXPECT_TRUE(m.sample_sigmas.size() == 3 && m.canvas_min_pixels == 1000 &&
              m.vision_pixels == 1000000);
  write("{\"_class_name\": \"PH3ImageModularPipeline\", \"sample_sigmas\": "
        "[1.0, 0.5, 0.6, 0.0]}");
  EXPECT_TRUE(!ph3::read_manifest(d.string(), &m, &err));
  write("{\"_class_name\": \"MiniMaxH3ModularPipeline\"}");
  EXPECT_TRUE(!ph3::is_root(d.string()));
  EXPECT_TRUE(!ph3::read_manifest(d.string(), &m, &err));
  std::error_code ec;
  fs::remove_all(d, ec);
  const std::string root = env_("VPIPE_P_H3_IMAGE_PATH");
  if (root.empty()) { return; }
  ph3::Manifest real;
  ASSERT_TRUE(ph3::read_manifest(root, &real, &err));
  EXPECT_TRUE(real.sample_sigmas.size() == 9 &&
              real.sample_sigmas.back() == 0.0f &&
              std::fabs(real.sample_sigmas[7] - 0.528718f) < 1e-7f);
}

// THE LAYOUTS, as the reference builds them: text-to-image is H3's t2va
// at ONE latent frame and two audio latents (text | audio | image), the
// audio rows on t = n_text + i pinned to the width grid's two ends, the
// image on t = n_text; an edit is ref2va with image references, each on
// one integer slot after the text, the target audio and image on the
// slot after the last.
TEST(p_h3_image, layouts_are_h3s_at_one_latent_frame)
{
  namespace h3 = minimax_h3;
  const int n = 7, lh = 8, lw = 12;   // 4 x 6 = 24 image rows
  std::vector<int> tags((std::size_t)n, h3::kTextTag);
  h3::PackedLayout L;
  ASSERT_TRUE(h3::build_packed_sequence(tags, 1, lh, lw, ph3::kAudioLatents,
                                        2, 2, h3::kAudioChannels, {}, &L));
  const int audio = h3::kAudioChannels * ph3::kAudioLatents;
  EXPECT_TRUE(L.num_video_rows == 24 && L.num_condition_rows == 0);
  EXPECT_TRUE(L.num_audio_rows == audio);
  EXPECT_TRUE((int)L.position_ids.size() == 3 * (n + audio + 24));
  const auto wg = h3::spatial_position_grid(lw, 2, std::sqrt((double)lh * lw));
  auto pos = [&](int r, int a) {
    return L.position_ids[(std::size_t)r * 3 + a];
  };
  bool ok = true;
  for (int i = 0; i < n; ++i) { ok = ok && pos(i, 0) == (double)i; }
  for (int c = 0; c < h3::kAudioChannels; ++c) {
    for (int i = 0; i < ph3::kAudioLatents; ++i) {
      const int r = n + c * ph3::kAudioLatents + i;
      ok = ok && pos(r, 0) == (double)(n + i) && pos(r, 1) == 0.0 &&
           pos(r, 2) == (c == 0 ? wg.front() : wg.back());
    }
  }
  for (int r = n + audio; r < n + audio + 24; ++r) {
    ok = ok && pos(r, 0) == (double)n;
  }
  EXPECT_TRUE(ok);
  // The edit: two pictures, one at the canvas and one smaller.
  std::vector<h3::Reference> refs(2);
  refs[0].num_latent_frames = 1; refs[0].latent_height = lh;
  refs[0].latent_width = lw;
  refs[1].num_latent_frames = 1; refs[1].latent_height = 4;
  refs[1].latent_width = 4;
  h3::PackedLayout E;
  ASSERT_TRUE(h3::build_ref2va_packed_sequence(
      tags, refs, 1, lh, lw, ph3::kAudioLatents, 2, 2, h3::kAudioChannels,
      &E));
  EXPECT_TRUE(E.num_condition_rows == 24 + 4 && E.num_video_rows == 24);
  bool eok = true;
  for (int r = n; r < n + 24; ++r) {
    eok = eok && E.position_ids[(std::size_t)r * 3] == (double)n;
  }
  for (int r = n + 24; r < n + 28; ++r) {
    eok = eok && E.position_ids[(std::size_t)r * 3] == (double)(n + 1);
  }
  const int t0 = n + 28 + audio;
  for (int r = t0; r < t0 + 24; ++r) {
    eok = eok && E.position_ids[(std::size_t)r * 3] == (double)(n + 2);
  }
  EXPECT_TRUE(eok);
}

// THE TEXT CONDITIONING. (1) The connector alone, fed the reference's own
// bf16 taps (the last RAW, normed here as the encoder's final norm does):
// the reference's connector output on those taps, to rounding. (2) The
// whole encoder on the prompt, verbatim: against the fp32 reference, at
// most 1.25x the reference's own bf16 distance; the sink row exact.
TEST(p_h3_image, text_conditioning_matches_reference)
{
  const std::string root = env_("VPIPE_P_H3_IMAGE_PATH");
  const std::string gd = env_("VPIPE_P_H3_IMAGE_GOLDEN");
  if (root.empty() || gd.empty() ||
      !fs::exists(fs::path(gd) / "text.json")) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MiniMaxH3TextEncoder::Config cfg;
  std::string err;
  ASSERT_TRUE(MiniMaxH3TextEncoder::config_from_json(root, cfg, &err));
  EXPECT_TRUE(cfg.taps == std::vector<int>({13, 19, 25, 31, 36}) &&
              cfg.tap == 36 && cfg.text_dim == 5120 &&
              !cfg.connector.empty());
  auto enc = MiniMaxH3TextEncoder::load(root, mc, &sess, cfg);
  ASSERT_TRUE(enc != nullptr && enc->has_connector());
  if (enc == nullptr) { return; }
  const int D = cfg.text_dim, H = cfg.lm.hidden;
  for (int k = 0; k < 2; ++k) {
    const std::string p = gd + "/text" + std::to_string(k);
    const auto ids = read_raw_<std::int32_t>(p + "_ids.i32");
    const auto ref = read_raw_<float>(p + "_cond.f32");
    const auto bar = read_raw_<float>(p + "_cond_bf16.f32");
    const auto taps = read_raw_<float>(p + "_taps_bf16.f32");
    const auto from_taps = read_raw_<float>(p + "_cond_from_taps.f32");
    const int n = (int)ids.size();
    ASSERT_TRUE(n > 0 && (int)(ref.size() / D) == n &&
                taps.size() == (std::size_t)5 * n * H);
    if (n <= 0 || taps.size() != (std::size_t)5 * n * H) { return; }
    // (1) the connector on the reference's taps
    SharedBuffer tb = mc->make_shared_buffer(taps.size() * 2);
    auto* t = static_cast<std::uint16_t*>(tb.contents());
    for (std::size_t i = 0; i < taps.size(); ++i) { t[i] = rne_(taps[i]); }
    std::string cerr;
    SharedBuffer c = enc->connect(tb, n, &cerr);
    ASSERT_TRUE(!c.empty());
    const double conn = rel_rows_(c, from_taps, D, 0, n);
    // (2) the whole encoder
    // Tokenized VERBATIM -- no template, no special tokens -- to the
    // reference's ids.
    std::vector<std::int32_t> v(ids.begin(), ids.end());
    EXPECT_TRUE(enc->tokenize(kTextPrompts[k]) == v);
    SharedBuffer h = enc->encode(kTextPrompts[k], nullptr, &err);
    ASSERT_TRUE(!h.empty());
    if (h.empty()) { return; }
    const double ours = rel_rows_(h, ref, D, 1, n);
    std::vector<float> refr(ref.begin() + D, ref.end());
    std::vector<float> barr(bar.begin() + D, bar.end());
    const double rb = rel_f32_(barr, refr);
    bool sink = true;
    const auto* hb = static_cast<const std::uint16_t*>(h.contents());
    for (int o = 0; o < D; ++o) { sink = sink && bf_(hb[o]) == bar[o]; }
    std::printf("[p_h3_image] text%d: %d tokens; the connector on the "
                "reference's taps %.2e; conditioning %.5f from fp32 (the "
                "reference's bf16 %.5f); sink row %s\n", k, n, conn, ours, rb,
                sink ? "exact" : "DIFFERS");
    EXPECT_TRUE(conn < 2e-3);
    EXPECT_TRUE(ours < 1.25 * rb + 1e-4);
    EXPECT_TRUE(sink);
  }
}

// THE EDIT PRESENTATION: "<Picture 1>: " + the vision block + "Edit
// <Picture 1>. " + the instruction, verbatim -- the ids and per-row tags
// exactly the reference's -- through the 4B's tower (deepstack included)
// on the reference's own ~1 MP copy, against the fp32 reference at most
// 1.25x its bf16's distance.
TEST(p_h3_image, edit_conditioning_matches_reference)
{
  const std::string root = env_("VPIPE_P_H3_IMAGE_PATH");
  const std::string gd = env_("VPIPE_P_H3_IMAGE_GOLDEN");
  if (root.empty() || gd.empty() ||
      !fs::exists(fs::path(gd) / "edit.json")) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MiniMaxH3TextEncoder::Config cfg;
  std::string err;
  ASSERT_TRUE(MiniMaxH3TextEncoder::config_from_json(root, cfg, &err));
  cfg.lm.max_seq = 8192;
  auto tower = MiniMaxH3TextEncoder::load_vision(root, mc, cfg, &err);
  ASSERT_TRUE(tower != nullptr);
  if (tower == nullptr) { return; }
  std::ifstream jf(gd + "/edit.json");
  std::string js((std::istreambuf_iterator<char>(jf)),
                 std::istreambuf_iterator<char>());
  auto num = [&](const char* k) {
    const std::size_t p = js.find(std::string("\"") + k + "\":");
    return p == std::string::npos ? 0 : std::atoi(js.c_str() + p +
                                                  std::strlen(k) + 3);
  };
  const int vw = num("vision_w"), vh = num("vision_h");
  const auto rgb = read_raw_<std::uint8_t>(gd + "/edit_vision_rgb.u8");
  ASSERT_TRUE(vw > 0 && vh > 0 && rgb.size() == (std::size_t)3 * vw * vh);
  if (rgb.size() != (std::size_t)3 * vw * vh) { return; }
  MetalQwenVisionEncoder::Result res = tower->encode(rgb.data(), vh, vw);
  ASSERT_TRUE(!res.embeddings.empty());
  tower.reset();
  auto enc = MiniMaxH3TextEncoder::load(root, mc, &sess, cfg);
  ASSERT_TRUE(enc != nullptr);
  if (enc == nullptr) { return; }
  MiniMaxH3TextEncoder::Reference r;
  r.kind = MiniMaxH3TextEncoder::Reference::Kind::kImage;
  r.vision = &res;
  const std::string prompt = ph3::edit_prompt(
      "Make the background a snowy mountain village at dusk.");
  MiniMaxH3TextEncoder::Presentation P;
  ASSERT_TRUE(enc->build_presentation({r}, prompt, &P, &err));
  const auto ids = read_raw_<std::int32_t>(gd + "/edit_ids.i32");
  const auto tags = read_raw_<std::int32_t>(gd + "/edit_tags.i32");
  const bool same_ids =
      P.ids == std::vector<std::int32_t>(ids.begin(), ids.end());
  const bool same_tags = P.tags == std::vector<int>(tags.begin(), tags.end());
  std::vector<int> got_tags;
  int n = 0;
  SharedBuffer h = enc->encode_references({r}, prompt, &got_tags, &n, &err);
  ASSERT_TRUE(!h.empty());
  if (h.empty()) { return; }
  const int D = cfg.text_dim;
  const auto ref = read_raw_<float>(gd + "/edit_cond.f32");
  const auto bar = read_raw_<float>(gd + "/edit_cond_bf16.f32");
  const double ours = rel_rows_(h, ref, D, 1, n);
  std::vector<float> refr(ref.begin() + D, ref.end());
  std::vector<float> barr(bar.begin() + D, bar.end());
  const double rb = rel_f32_(barr, refr);
  std::printf("[p_h3_image] edit: %d tokens, ids %s, tags %s; conditioning "
              "%.5f from fp32 (the reference's bf16 %.5f)\n", n,
              same_ids ? "equal" : "DIFFER", same_tags ? "equal" : "DIFFER",
              ours, rb);
  EXPECT_TRUE(same_ids && same_tags && got_tags == P.tags);
  EXPECT_TRUE(ours < 1.25 * rb + 1e-4);
}

// THE IMAGE DECODER: one latent frame into one picture, the token's
// pinned rotary time and the 256-pixel tiling with its linear blend, on
// one tile and on a 2 x 3 tiling -- against the fp32 restatement at most
// 1.25x the bf16 restatement's distance. And the video decode REFUSED:
// this decoder expands a token into one pixel frame.
TEST(p_h3_image, image_decoder_matches_reference)
{
  const std::string root = env_("VPIPE_P_H3_IMAGE_PATH");
  const std::string gd = env_("VPIPE_P_H3_IMAGE_GOLDEN");
  if (root.empty() || gd.empty() || !fs::exists(fs::path(gd) / "vae.json")) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMiniMaxH3VideoVae::Config cfg;
  std::string err;
  const std::string vdir = MetalMiniMaxH3VideoVae::resolve_vae_dir(root);
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(vdir, cfg, &err));
  EXPECT_TRUE(cfg.t1_direct && cfg.out_patch_t() == 1 &&
              cfg.patch_t == 4 && !cfg.decodes_video() &&
              cfg.patch_elems() == 768 &&
              std::fabs(cfg.t1_time - ph3::kFramePosition) < 1e-12);
  auto m = MetalMiniMaxH3VideoVae::load(vdir, mc, cfg);
  ASSERT_TRUE(m != nullptr && m->can_decode_image());
  if (m == nullptr) { return; }
  struct Case { const char* name; int lh, lw; };
  for (const Case c : {Case{"a", 16, 16}, Case{"b", 32, 40}}) {
    const std::string p = gd + "/vae_" + c.name;
    const auto z = read_raw_<float>(p + "_z.f32");
    const auto ref = read_raw_<float>(p + "_out.f32");
    const auto bar = read_raw_<float>(p + "_bf16.f32");
    ASSERT_TRUE(z.size() == (std::size_t)24 * c.lh * c.lw);
    SharedBuffer zb = mc->make_shared_buffer(z.size() * 2);
    auto* zp = static_cast<std::uint16_t*>(zb.contents());
    for (std::size_t i = 0; i < z.size(); ++i) { zp[i] = rne_(z[i]); }
    SharedBuffer out = m->decode_image(zb, c.lh, c.lw, &err);
    if (out.empty()) { std::printf("[p_h3_image] %s\n", err.c_str()); }
    ASSERT_TRUE(!out.empty());
    if (out.empty()) { return; }
    const int px = c.lh * 16 * c.lw * 16;
    const double ours = rel_rows_(out, ref, px, 0, 3);
    const double rb = rel_f32_(bar, ref);
    std::printf("[p_h3_image] image decoder %s: %dx%d latent -> %dx%d, %.5f "
                "from fp32 (the reference's bf16 %.5f)\n", c.name, c.lh, c.lw,
                c.lw * 16, c.lh * 16, ours, rb);
    EXPECT_TRUE(ours < 1.25 * rb + 1e-4);
  }
  SharedBuffer zb = mc->make_shared_buffer((std::size_t)24 * 7 * 16 * 16 * 2);
  std::memset(zb.contents(), 0, zb.byte_size());
  EXPECT_TRUE(m->decode_video(zb, 7, 16, 16, nullptr, &err).empty());
}

// THE PLAN AGAINST WHAT RUNS. The VAE's two halves load at their first
// use and each holds what held_bytes books for it, to the byte -- the
// encoder (byte for byte H3's) at an edit picture's encode, the
// single-frame decoder at a picture's decode; the image decode's sampled
// device peak inside decode_working_bytes, which both generate-video's
// arena and vae-decode's revision book; and the connector held as its
// claim counts it (f32 as stored, cached in its own set).
TEST(p_h3_image, the_plan_books_what_the_codec_and_connector_hold)
{
  const std::string root = env_("VPIPE_P_H3_IMAGE_PATH");
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMiniMaxH3VideoVae::Config cfg;
  std::string err;
  const std::string vdir = MetalMiniMaxH3VideoVae::resolve_vae_dir(root);
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(vdir, cfg, &err));
  {
    auto ws = WeightSet::open(vdir, nullptr);
    ASSERT_TRUE(ws != nullptr);
    if (!ws) { return; }
    auto m = MetalMiniMaxH3VideoVae::load(ws, mc, cfg);
    ASSERT_TRUE(m != nullptr);
    if (!m) { return; }
    const std::size_t at_load = ws->stats().bytes;
    const int H = 512, W = 640;
    const std::size_t n = (std::size_t)cfg.in_channels * H * W;
    SharedBuffer xb = mc->make_shared_buffer(n * 2);
    std::memset(xb.contents(), 0, n * 2);
    SharedBuffer mom = m->encode(xb, 1, H, W, &err);
    ASSERT_TRUE(!mom.empty());
    const std::size_t after_enc = ws->stats().bytes;
    const int lh = H / cfg.patch, lw = W / cfg.patch;
    SharedBuffer zb = mc->make_shared_buffer((std::size_t)cfg.z_channels *
                                             lh * lw * 2);
    std::memcpy(zb.contents(), mom.contents(), zb.byte_size());
    // A small warm-up first: it loads the decoder half (and its load's
    // transients) with a scratch for 8 x 8 tokens. The picture's decode
    // after it, sampled, then builds the scratch its 256-pixel tiles need
    // -- its whole working set, the weights already resident.
    {
      SharedBuffer z8 = mc->make_shared_buffer((std::size_t)cfg.z_channels *
                                               8 * 8 * 2);
      std::memset(z8.contents(), 0, z8.byte_size());
      ASSERT_TRUE(!m->decode_image(z8, 8, 8, &err).empty());
    }
    const std::size_t after_dec = ws->stats().bytes;
    const std::size_t base = mc->memory_budget_settled().allocated;
    std::atomic<std::size_t> peak{base};
    std::atomic<bool> stop{false};
    std::thread sampler([&] {
      while (!stop.load()) {
        const std::size_t a = mc->memory_budget().allocated;
        if (a > peak.load()) { peak.store(a); }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    });
    SharedBuffer pic = m->decode_image(zb, lh, lw, &err);
    stop.store(true);
    sampler.join();
    ASSERT_TRUE(!pic.empty());
    using Half = MetalMiniMaxH3VideoVae::Half;
    const std::size_t enc = MetalMiniMaxH3VideoVae::held_bytes(vdir,
                                                               Half::kEncoder);
    const std::size_t dec = MetalMiniMaxH3VideoVae::held_bytes(vdir,
                                                               Half::kDecoder);
    const std::size_t top =
        std::max(peak.load(), mc->memory_budget().allocated);
    const std::size_t used = top > base ? top - base : 0;
    const std::size_t booked =
        MetalMiniMaxH3VideoVae::decode_working_bytes(cfg, 1, lh, lw);
    std::printf("[p_h3_image] VAE held: at load %zu, encoder %zu (booked "
                "%zu), decoder %zu (booked %zu); a %dx%d decode peaked %.1f "
                "MB against %.1f MB booked\n", at_load,
                after_enc - at_load, enc, after_dec - after_enc, dec, W, H,
                used / 1048576.0, booked / 1048576.0);
    EXPECT_TRUE(at_load == 0);
    EXPECT_TRUE(after_enc - at_load == enc);
    EXPECT_TRUE(after_dec - after_enc == dec);
    EXPECT_TRUE(used <= booked + ((std::size_t)2 << 20));
  }
  {
    const std::string cd = ph3::connector_dir(root);
    ASSERT_TRUE(!cd.empty());
    auto ws = WeightSet::open(cd, nullptr);
    ASSERT_TRUE(ws != nullptr);
    if (!ws) { return; }
    for (const char* nm : {"input_mean", "input_std", "proj.weight",
                           "output_mean", "output_std", "sink_embedding"}) {
      ASSERT_TRUE(!ws->tensor(nm, mc, WeightSet::Residency::Copied).empty());
    }
    const std::size_t held = ws->stats().bytes;
    const std::size_t claimed = model_memory::dir_weights_bytes(cd);
    // The claim is the file: the tensors and its header.
    std::printf("[p_h3_image] connector held %zu, claimed %zu (the file, "
                "its header included)\n", held, claimed);
    EXPECT_TRUE(held <= claimed && claimed - held < ((std::size_t)64 << 10));
  }
}
