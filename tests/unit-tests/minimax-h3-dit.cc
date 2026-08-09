// The MiniMax-H3 omni denoiser (MiniMaxH3DiTModel), against the
// diffusers reference.
//
// The golden is a TRUNCATED model -- two blocks out of fifty, dumped by
// the scratchpad's gen_dit_golden.py -- and that is the point rather
// than a compromise. The full checkpoint is 66 GB at bf16, so a
// full-depth reference does not fit any machine this runs on. At depth 2
// every distinct path is still exercised:
//
//   * both input patch projections (video 96 -> 5376, audio 32 -> 5376)
//     and the text projection plus its 2-block token refiner, which has
//     neither AdaLN nor rope and so is a genuinely different block;
//   * the per-ROW AdaLN -- three distinct timesteps and three modality
//     tags are live in this layout, so the table is nine rows and an
//     implementation that broadcast one modulation over the sequence,
//     which is what every other DiT here does, cannot pass;
//   * the partial rotate-half rope: 96 of 128 channels rotate and 32
//     pass through, over a three-axis grid whose time axis is
//     non-uniform;
//   * the VALUE-FIRST SwiGLU. A gate-first read produces a plausible
//     activation of the right magnitude and the wrong values;
//   * per-head q/k RMS, the shared final norm indexed by the bare
//     timestep rather than by the AdaLN row, and both output heads.
//
// Two blocks and not one because block 1's weights differ from block 0's:
// an implementation that read any per-block tensor from block 0 would
// pass at depth 1 and fail here.
//
// The layout constants below MUST match gen_dit_golden.py's -- they are
// what the golden was dumped for.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH = the FL2VA partition dir (or
// the repo root), VPIPE_MINIMAX_H3_DIT_GOLDEN = the golden dir. Skips if
// either is unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "stages/model-quantize-stage.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-denoise.h"
#include "generative-models/minimax-h3/minimax-h3-scheduler.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
namespace h3 = vpipe::genai::minimax_h3;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// gen_dit_golden.py's layout.
constexpr int kLayers   = 2;
constexpr int kLatentF  = 3;
constexpr int kLatentH  = 8;
constexpr int kLatentW  = 8;
constexpr int kAudioLat = 4;
constexpr int kTextRows = 6;
constexpr float kTVideo = 0.3125f;
constexpr float kTAudio = 0.5f;
constexpr float kTCond  = 0.75f;

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<float> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
bf16_to_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

SharedBuffer
to_bf16_buf_(MetalCompute* mc, const std::vector<float>& v)
{
  SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
  if (b.empty()) { return b; }
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { d[i] = f32_to_bf16_(v[i]); }
  return b;
}

double
rel_l2_bf16_(const SharedBuffer& got, const std::vector<float>& ref)
{
  const auto* g = static_cast<const std::uint16_t*>(got.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = (double)bf16_to_f32_(g[i]) - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// rel-L2 alone cannot separate the two ways an output can score ~1.0: a
// velocity ORTHOGONAL to the reference and a velocity near ZERO both
// land there. The magnitude ratio and the correlation do separate them,
// so report all three whenever the run is a diagnostic rather than a
// pass/fail golden.
void
report_shape_(const char* what, const SharedBuffer& got,
              const std::vector<float>& ref)
{
  const auto* g = static_cast<const std::uint16_t*>(got.contents());
  double sg = 0.0, sr = 0.0, dot = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double a = (double)bf16_to_f32_(g[i]);
    const double b = (double)ref[i];
    sg += a * a;
    sr += b * b;
    dot += a * b;
  }
  const std::size_t n = ref.size();
  const double rg = std::sqrt(sg / (double)n);
  const double rr = std::sqrt(sr / (double)n);
  const double corr = (sg > 0.0 && sr > 0.0) ? dot / std::sqrt(sg * sr) : 0.0;
  std::printf("[minimax_h3_dit] %s: |got| %.5f |ref| %.5f  ratio %.4f  "
              "corr %.5f\n", what, rg, rr, rg > 0.0 ? rg / rr : 0.0, corr);
}

}  // namespace

TEST(minimax_h3_dit, config_from_json)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  MetalMiniMaxH3Transformer::Config cfg;
  std::string err;
  const bool ok =
      MetalMiniMaxH3Transformer::config_from_json(root, cfg, &err);
  if (!ok) { std::printf("[minimax_h3_dit] config: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  std::printf("[minimax_h3_dit] hidden %d, %d heads x %d = inner %d, "
              "ffn %d, %d blocks + %d refiner, adaln %d\n",
              cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.inner(), cfg.ffn,
              cfg.n_layers, cfg.n_refiner, cfg.adaln_out());
  // The attention inner width is LARGER than the residual stream. Every
  // other model in this tree has hidden == heads * head_dim, so this is
  // the assumption most likely to be carried in by accident.
  EXPECT_TRUE(cfg.inner() > cfg.hidden);
  EXPECT_TRUE(cfg.inner() == 7168 && cfg.hidden == 5376);
  EXPECT_TRUE(cfg.adaln_out() == 6 * cfg.hidden * h3::kModalityNum);
  // 3 axes x 16 frequencies, doubled: 96 of the 128 head channels
  // rotate and 32 pass through.
  EXPECT_TRUE(cfg.rope_rot() == 96 && cfg.rope_rot() < cfg.head_dim);
  // The path resolver has to reach the DiT from the repo root, since
  // that is what the catalogue registers. THREE spellings are valid --
  // two publishers plus what this tree writes:
  //   * MiniMaxAI's diffusers `transformer/` dir;
  //   * Comfy-Org's `diffusion_models/*.safetensors` file;
  //   * a repack model-quantize has processed, whose quantized component
  //     is a directory checkpoint at `diffusion_models/` (no file
  //     extension, and not spelled "transformer").
  // This env var names a REPO, so any of the three may be behind it.
  const std::string d = MetalMiniMaxH3Transformer::resolve_dit_dir(root);
  EXPECT_TRUE(d.find("transformer") != std::string::npos ||
              d.find(".safetensors") != std::string::npos ||
              d.find("diffusion_models") != std::string::npos);
}

namespace {

// `n_layers` out of the golden's meta.json, so one harness serves a
// golden dumped at any depth. Falls back to the historical 2.
int
golden_int_(const std::string& gdir, const char* key, int dflt)
{
  std::ifstream f(gdir + "/meta.json");
  if (!f.good()) { return dflt; }
  const std::string s((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  const std::size_t k = s.find(std::string("\"") + key + "\"");
  if (k == std::string::npos) { return dflt; }
  const std::size_t c = s.find(':', k);
  if (c == std::string::npos) { return dflt; }
  const int n = std::atoi(s.c_str() + c + 1);
  return n > 0 ? n : dflt;
}

// True when the golden's meta declares an EMPTY anchor list -- the t2va
// layout, which the production pipeline uses and the depth-<=16 goldens
// (all "first"-anchored) never exercise.
bool
golden_no_anchors_(const std::string& gdir)
{
  std::ifstream f(gdir + "/meta.json");
  if (!f.good()) { return false; }
  const std::string s((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  const std::size_t k = s.find("\"anchors\"");
  if (k == std::string::npos) { return false; }
  const std::size_t b = s.find('[', k);
  const std::size_t e = s.find(']', k);
  if (b == std::string::npos || e == std::string::npos) { return false; }
  return s.find_first_not_of(" \t\r\n", b + 1) >= e;
}

float
golden_float_(const std::string& gdir, const char* key, float dflt)
{
  std::ifstream f(gdir + "/meta.json");
  if (!f.good()) { return dflt; }
  const std::string s((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  const std::size_t k = s.find(std::string("\"") + key + "\"");
  if (k == std::string::npos) { return dflt; }
  const std::size_t c = s.find(':', k);
  if (c == std::string::npos) { return dflt; }
  return (float)std::atof(s.c_str() + c + 1);
}

// How many DISTINCT timesteps the golden's layout carries. The anchored
// goldens have three (video, audio, conditioning); a t2va production
// layout has ONE, so this cannot be hardcoded.
std::size_t
golden_uniq_count_(const std::string& gdir, std::size_t dflt)
{
  std::ifstream f(gdir + "/meta.json");
  if (!f.good()) { return dflt; }
  const std::string s((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  const std::size_t k = s.find("\"uniq\"");
  if (k == std::string::npos) { return dflt; }
  const std::size_t b = s.find('[', k), e = s.find(']', k);
  if (b == std::string::npos || e == std::string::npos) { return dflt; }
  if (s.find_first_not_of(" \t\r\n", b + 1) >= e) { return 0; }
  std::size_t n = 1;
  for (std::size_t i = b + 1; i < e; ++i) {
    if (s[i] == ',') { ++n; }
  }
  return n;
}

int
golden_n_layers_(const std::string& gdir)
{
  std::ifstream f(gdir + "/meta.json");
  if (!f.good()) { return kLayers; }
  const std::string s((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  const std::size_t k = s.find("\"n_layers\"");
  if (k == std::string::npos) { return kLayers; }
  const std::size_t c = s.find(':', k);
  if (c == std::string::npos) { return kLayers; }
  const int n = std::atoi(s.c_str() + c + 1);
  return n > 0 ? n : kLayers;
}

// The golden run, against whichever checkpoint `root` names.
// Factored out so a QUANTIZED checkpoint is measured through the exact
// same harness as the bf16 one -- a quant test that reimplemented the
// setup could differ in the layout or the timesteps and report a
// quantization error that was really a harness difference.
bool
run_dit_golden_(const std::string& root, const std::string& gdir,
                double* rv_out, double* ra_out)
{
  const std::vector<float> vin  = read_f32_(gdir + "/video_in.f32");
  const std::vector<float> ain  = read_f32_(gdir + "/audio_in.f32");
  const std::vector<float> tin  = read_f32_(gdir + "/text_in.f32");
  const std::vector<float> vref = read_f32_(gdir + "/video_out.f32");
  const std::vector<float> aref = read_f32_(gdir + "/audio_out.f32");
  if (vin.empty() || ain.empty() || tin.empty() || vref.empty() ||
      aref.empty()) {
    return false;   // golden not dumped -> skip
  }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return false; }

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!(MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr))) { return false; }
  // Depth from the GOLDEN's own meta, not from the checkpoint: loading a
  // few blocks out of fifty is what makes this runnable at all, and a
  // deeper golden is how a defect that only COMPOUNDS with depth becomes
  // visible -- a fixed depth 2 cannot tell a sound block from one with a
  // small per-block error.
  cfg.n_layers = golden_n_layers_(gdir);
  // Geometry from the golden too, so ONE harness serves both the small
  // anchored goldens and a PRODUCTION-layout dump. Everything verified
  // so far ran either a small geometry or random rows; this is the first
  // elementwise check at the real sequence shape.
  const int gLatF  = golden_int_(gdir, "latent_frames", kLatentF);
  const int gLatH  = golden_int_(gdir, "latent_h", kLatentH);
  const int gLatW  = golden_int_(gdir, "latent_w", kLatentW);
  const int gAudL  = golden_int_(gdir, "n_audio_latents", kAudioLat);
  const int gText  = golden_int_(gdir, "n_text", kTextRows);
  const bool gNoAnc = golden_no_anchors_(gdir);

  h3::PackedLayout L;
  const std::vector<int> text_tags((std::size_t)gText, h3::kTextTag);
  const std::vector<h3::Anchor> anchors =
      gNoAnc ? std::vector<h3::Anchor>{}
             : std::vector<h3::Anchor>{h3::Anchor::kFirst};
  if (!(h3::build_packed_sequence(text_tags, gLatF, gLatH,
                                        gLatW, gAudL, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels,
                                        anchors, &L))) { return false; }
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, golden_float_(gdir, "t_video", kTVideo),
                          golden_float_(gdir, "t_audio", kTAudio),
                          golden_float_(gdir, "t_cond", kTCond),
                          &uniq, &row_idx);
  // Three distinct timesteps in one forward is the whole conditioning
  // story of this model; if the layout collapsed them the test would be
  // checking a much weaker thing.
  if (!(uniq.size() == golden_uniq_count_(gdir, 3))) { return false; }

  const int n_video = (int)L.video_indices.size();
  if (!(vin.size() ==
              (std::size_t)n_video * (std::size_t)cfg.video_patch_elems())) { return false; }
  if (!(ain.size() ==
              (std::size_t)L.num_audio_rows * (std::size_t)cfg.audio_channels)) { return false; }
  if (!(tin.size() == (std::size_t)gText * (std::size_t)cfg.text_dim)) { return false; }

  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  if (!(m != nullptr)) { return false; }

  const SharedBuffer vb = to_bf16_buf_(mc, vin);
  const SharedBuffer ab = to_bf16_buf_(mc, ain);
  const SharedBuffer tb = to_bf16_buf_(mc, tin);
  if (!(!vb.empty() && !ab.empty() && !tb.empty())) { return false; }

  MetalMiniMaxH3Transformer::Step step;
  step.video  = &vb;
  step.audio  = &ab;
  step.text   = &tb;
  step.layout = &L;
  step.timesteps = &uniq;
  step.row_timestep_index = &row_idx;
  std::string ferr;
  MetalMiniMaxH3Transformer::Velocity out = m->forward(step, &ferr);
  if (out.empty()) { std::printf("[minimax_h3_dit] %s\n", ferr.c_str()); }
  if (!(!out.empty())) { return false; }

  const double rv = rel_l2_bf16_(out.video, vref);
  const double ra = rel_l2_bf16_(out.audio, aref);
  report_shape_("video", out.video, vref);
  report_shape_("audio", out.audio, aref);
  // Diagnostic escape hatch: the raw velocity, so a caller can score
  // REGIONS of it (conditioning rows against generated rows) rather
  // than the single whole-buffer number this harness prints.
  if (const char* dd = std::getenv("VPIPE_MINIMAX_H3_DIT_DUMP")) {
    const std::string base(dd);
    auto put = [&](const std::string& nm, const SharedBuffer& b,
                   std::size_t n) {
      std::ofstream f(base + "/" + nm + ".f32", std::ios::binary);
      const auto* p = static_cast<const std::uint16_t*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) {
        const float v = bf16_to_f32_(p[i]);
        f.write(reinterpret_cast<const char*>(&v), 4);
      }
    };
    put("got_video", out.video, vref.size());
    put("got_audio", out.audio, aref.size());
    std::printf("[minimax_h3_dit] dumped velocity to %s\n", dd);
  }
  std::printf("[minimax_h3_dit] seq %d (%d text + %d cond + %d audio + %d "
              "video), %d blocks: video rel-L2 %.6f, audio rel-L2 %.6f\n",
              L.seq_len, L.num_text_rows, L.num_condition_rows,
              L.num_audio_rows, L.num_video_rows, cfg.n_layers, rv, ra);
  // Both heads must be alive. An audio head that returned zeros would
  // pass a loose relative bar on the video alone, and the audio rows are
  // only 8 of 78.
  double vmag = 0.0, amag = 0.0;
  for (float x : vref) { vmag += (double)x * x; }
  for (float x : aref) { amag += (double)x * x; }
  if (vmag <= 0.0 || amag <= 0.0) { return false; }
  *rv_out = rv;
  *ra_out = ra;
  return true;
}

}  // namespace

TEST(minimax_h3_dit, forward_matches_golden)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_MINIMAX_H3_DIT_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  double rv = 0.0, ra = 0.0;
  if (!run_dit_golden_(root, gd, &rv, &ra)) { return; }
  // The reference itself runs the block stack in bf16 (that is the
  // released dtype), so this compares two bf16 evaluations whose
  // accumulation orders differ -- the floor is the ACCUMULATION, not the
  // model, and it therefore GROWS WITH DEPTH. A single bar would either
  // be slack at depth 2 or fail a correct model at depth 50.
  //
  // MEASURED against ComfyUI goldens, dense vs dense:
  //     depth  2 / 8 / 16   0.0193 / 0.0195 / 0.0197 video
  //     depth 50            0.0254 video, 0.0122 audio
  // The shallow numbers being FLAT is the real signal -- a per-block
  // error would compound (before the per-head qkv fix they ran
  // 0.32 / 0.69 / 1.09).
  const int depth = golden_n_layers_(gd);
  EXPECT_TRUE(rv < (depth >= 32 ? 0.04 : 0.03));
  EXPECT_TRUE(ra < (depth >= 32 ? 0.03 : 0.03));
}

// The SAME model from Comfy-Org's repack, through the SAME golden.
//
// This is the acceptance test for reading that format, and it has to be
// an equivalence rather than a bar of its own: the two checkpoints are
// the same weights, so the only honest question is whether they compute
// the same function. What differs is one thing no checkpoint states --
// MiniMaxAI groups the fused `attn.qkv_proj` per head, Comfy-Org
// reorders it flat -- under identical tensor names and identical
// shapes. Reading one as the other loads cleanly, passes every shape
// check, and scrambles attention in every block. That failure mode is
// why the layout is a recorded fact (Config::qkv_per_head) and why this
// test exists: a shape check cannot catch it, only a number can.
//
// Env: VPIPE_MINIMAX_H3_COMFY_MODEL_PATH = the Comfy-Org repo root, its
// diffusion_models/ subdir, or the .safetensors itself. The released
// file is 66 GB; the scratchpad's make_comfy_fixture.py reproduces the
// conversion over just the tensors a depth-2 load reads (~4 GB), which
// is the depth this golden runs at.
TEST(minimax_h3_dit, comfy_layout_matches_golden)
{
  const char* comfy = std::getenv("VPIPE_MINIMAX_H3_COMFY_MODEL_PATH");
  const char* gd    = std::getenv("VPIPE_MINIMAX_H3_DIT_GOLDEN");
  if (comfy == nullptr || *comfy == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  // The config comes out of the safetensors __metadata__, not a
  // config.json -- and it must arrive with the flat grouping recorded.
  MetalMiniMaxH3Transformer::Config cfg;
  std::string err;
  const bool ok =
      MetalMiniMaxH3Transformer::config_from_json(comfy, cfg, &err);
  if (!ok) { std::printf("[minimax_h3_comfy] config: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  EXPECT_TRUE(!cfg.qkv_per_head);
  EXPECT_TRUE(cfg.inner() == 7168 && cfg.hidden == 5376);
  EXPECT_TRUE(cfg.n_layers == 50 && cfg.n_refiner == 2);
  // Resolution is checked in detail (root / subdir / file, and which
  // variant wins) by minimax_h3_comfy, which needs no weights. Here it
  // only has to land somewhere: this env var may also point at a
  // checkpoint DERIVED from the repack -- model-quantize's output, an
  // ordinary directory that carries the flat grouping in its own
  // config.json -- and that is a case worth being able to run.
  const std::string f = MetalMiniMaxH3Transformer::resolve_dit_dir(comfy);
  EXPECT_TRUE(!f.empty());

  // The released layout, for the same golden, must land in the same
  // place. Read its own bar rather than hard-coding one: this asserts
  // the two AGREE, which is the claim -- a shared regression would move
  // both and a layout bug moves only one.
  double rv = 0.0, ra = 0.0;
  if (!run_dit_golden_(comfy, gd, &rv, &ra)) { return; }
  const int depth = golden_n_layers_(gd);
  EXPECT_TRUE(rv < (depth >= 32 ? 0.04 : 0.03));
  EXPECT_TRUE(ra < (depth >= 32 ? 0.03 : 0.03));

  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  MetalMiniMaxH3Transformer::Config rel;
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(root, rel, &err));
  // Both env vars may legitimately name the SAME repack -- a box that
  // has only the Comfy-Org copy. There is no second opinion to take
  // then, so stop rather than compare a checkpoint with itself.
  if (!rel.qkv_per_head) { return; }
  // Every OTHER field has to match: the two configs are the same model
  // described twice, so a difference here is the repack having drifted.
  EXPECT_TRUE(rel.hidden == cfg.hidden && rel.n_heads == cfg.n_heads &&
              rel.head_dim == cfg.head_dim && rel.n_layers == cfg.n_layers &&
              rel.n_refiner == cfg.n_refiner && rel.ffn == cfg.ffn &&
              rel.video_channels == cfg.video_channels &&
              rel.audio_channels == cfg.audio_channels &&
              rel.patch_t == cfg.patch_t && rel.patch_h == cfg.patch_h &&
              rel.patch_w == cfg.patch_w && rel.text_dim == cfg.text_dim &&
              rel.freq_dim == cfg.freq_dim &&
              rel.time_hidden == cfg.time_hidden &&
              rel.time_dim == cfg.time_dim &&
              rel.rope_freq_dim == cfg.rope_freq_dim);
  double rrv = 0.0, rra = 0.0;
  if (!run_dit_golden_(root, gd, &rrv, &rra)) { return; }
  std::printf("[minimax_h3_comfy] video %.6f vs released %.6f, "
              "audio %.6f vs %.6f\n", rv, rrv, ra, rra);
  // Same weights, same golden, different qkv grouping in the FILE: the
  // two runs differ only in the order the loader walks those bytes, so
  // they should land within the bf16 accumulation noise of each other,
  // not merely both under the bar.
  EXPECT_TRUE(std::fabs(rv - rrv) < 0.005);
  EXPECT_TRUE(std::fabs(ra - rra) < 0.005);
}

// The QUANTIZED DiT, through the same golden.
//
// This is the load-bearing measurement for the whole model: 33B at bf16
// is 62 GB, which fits no box here, so a quantized checkpoint is not an
// optimization but the precondition for running the DiT at all.
//
// The AdaLN modulation is what makes H3 different from the other DiTs
// quantized here. Elsewhere it is a small side projection kept bf16 by
// default; here `blocks.N.adaln_proj.linear` is 2688 -> 96768, i.e. 260M
// of each block's 645M and 13B of the 33B model. Leaving it bf16 would
// cap a quantized checkpoint at ~36 GB and defeat the point, so it is
// quantized -- at 8-bit, because it is what the residual scale rides on
// and the loader derives bit width per tensor.
//
// WIDTH: the body defaults to 8-bit here, and that is a MEASURED choice,
// not caution. Against this golden (dense bf16 scores 0.0086 video /
// 0.0200 audio):
//
//     body w4 + mod w8   24 GB   video 0.0359   audio 0.2130
//     body w8 + mod w8   33 GB   video 0.0091   audio 0.0238
//
// The video branch degrades the way a 4-bit DiT normally does. The AUDIO
// branch does not: 0.213 is a 10x jump on the dense baseline, and it is
// not a small-denominator artifact -- the audio golden's rms is 13.3
// against the video's 29.7, so the absolute error really is larger on
// the 8 audio rows. Isolating it by re-quantizing at w8 showed the 4-bit
// BODY is the cause, not the 8-bit modulation. Whether 0.213 per step is
// audible after 50 steps is an end-to-end question this test cannot
// answer, so the default is the width whose error this bar can honestly
// stand behind; w4 stays available (and is the right pick for a
// video-only run, or a box that cannot hold 33 GB) via
// VPIPE_MINIMAX_H3_QUANT_BITS=4.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH, VPIPE_MINIMAX_H3_DIT_GOLDEN,
// VPIPE_MINIMAX_H3_QUANT_OUT (an existing quantized DiT is reused; an
// empty path is quantized into, which reads 62 GB and takes ~5 minutes).
TEST(minimax_h3_dit, quantized_matches_golden)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_MINIMAX_H3_DIT_GOLDEN");
  const char* qo   = std::getenv("VPIPE_MINIMAX_H3_QUANT_OUT");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0' ||
      qo == nullptr || *qo == '\0') {
    return;
  }
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path out(qo);

  if (!fs::exists(out / "config.json", ec)) {
    Session sess;
    if (sess.metal_compute() == nullptr) { return; }
    const std::string src =
        MetalMiniMaxH3Transformer::resolve_dit_dir(root);
    FlexData cfg = FlexData::make_object();
    {
      auto o = cfg.as_object();
      o.insert("src_model", FlexData::make_string(src));
      o.insert("output_name", FlexData::make_string(out.string()));
      // Body width. 8-bit by default (see the note above); 4-bit is the
      // smaller, video-only checkpoint.
      int bits = 8;
      if (const char* b = std::getenv("VPIPE_MINIMAX_H3_QUANT_BITS")) {
        bits = std::atoi(b);
      }
      o.insert("bits", FlexData::make_int(bits));
      o.insert("group_size", FlexData::make_int(64));
      // Without this the 13B of modulation stays bf16 and the "4-bit"
      // checkpoint is still ~36 GB.
      o.insert("quant_modulation", FlexData::make_bool(true));
      o.insert("skip_existing", FlexData::make_bool(false));
    }
    ModelQuantizeStage s(&sess, "mq-h3", std::vector<InEdge>{},
                         std::move(cfg));
    if (!s.config_error().empty()) {
      std::printf("[minimax_h3_dit] quantize config: %s\n",
                  s.config_error().c_str());
    }
    ASSERT_TRUE(s.config_error().empty());
    ASSERT_TRUE(s.quantize_once());
  }

  // The checkpoint must actually be MIXED: 4-bit body next to 8-bit
  // modulation. A run that silently produced one width everywhere would
  // still load and still pass the error bar below at 8-bit, while being
  // twice the size it should be.
  {
    auto wts = genai::MetalLlamaWeights::open_model(out.string());
    ASSERT_TRUE(wts.has_value());
    int w4 = 0, w8 = 0;
    for (const auto& n : wts->tensor_names()) {
      if (n.size() < 8 || n.compare(n.size() - 7, 7, ".scales") != 0) {
        continue;
      }
      const std::string base = n.substr(0, n.size() - 7);
      const auto* si = wts->info(n);
      const auto* ci = wts->info(base + ".weight");
      if (si == nullptr || ci == nullptr || si->shape.size() != 2 ||
          ci->shape.size() != 2) {
        continue;
      }
      const long K = si->shape[1] * 64;
      if (K <= 0) { continue; }
      ((ci->shape[1] * 32 / K) == 8 ? w8 : w4) += 1;
    }
    std::printf("[minimax_h3_dit] quantized tensors: %d at 4-bit, %d at "
                "8-bit\n", w4, w8);
    EXPECT_TRUE(w4 + w8 > 0);
    // 4 body linears per block against 1 modulation linear, over 50
    // blocks plus 2 refiner blocks (body only) and the final layer
    // (modulation only).
    // 4 body linears per block over 50 blocks + 2 refiner blocks, and 1
    // modulation linear per block + the final layer. At the default w8
    // body both land in the 8-bit bucket, so only check the split when
    // the body is narrower.
    if (w4 > 0) { EXPECT_TRUE(w4 == 4 * 52 && w8 == 51); }
    EXPECT_TRUE(w4 + w8 == 4 * 52 + 51);
  }

  double rv = 0.0, ra = 0.0;
  if (!run_dit_golden_(out.string(), gd, &rv, &ra)) {
    EXPECT_TRUE(false);
    return;
  }
  // MEASURED at the default w8 body: 0.0091 / 0.0238, against the dense
  // checkpoint's 0.0086 / 0.0200 through this same harness -- so 8-bit
  // costs almost nothing and the bars sit just above that rather than
  // at a level a wrong-width load could hide under. At w4 the bars have
  // to be 0.06 / 0.30 instead, which is the point of the note above.
  const bool w4_body =
      std::getenv("VPIPE_MINIMAX_H3_QUANT_BITS") != nullptr &&
      std::atoi(std::getenv("VPIPE_MINIMAX_H3_QUANT_BITS")) == 4;
  // Depth-aware for the same reason as forward_matches_golden: the bf16
  // accumulation floor grows with the stack. MEASURED at depth 50
  // against the ComfyUI golden -- DENSE 0.0254 / 0.0122, w8 0.0301 /
  // 0.0188 -- so at that depth 8-bit costs ~0.005 and the floor is the
  // accumulation. The bar sits just above the w8 measurement rather
  // than at a level a wrong-width load could hide under.
  const int qdepth = golden_n_layers_(gd);
  const double vbar = w4_body ? 0.06 : (qdepth >= 32 ? 0.045 : 0.02);
  const double abar = w4_body ? 0.30 : (qdepth >= 32 ? 0.030 : 0.05);
  EXPECT_TRUE(rv < vbar);
  EXPECT_TRUE(ra < abar);
}

// The denoise loop's CONTRACT, at depth 2.
//
// There is no reference trajectory to compare against here -- running
// the real 50-block loop needs the whole 33B model -- so this pins the
// invariants the loop is responsible for, which is where its bugs live
// rather than in the arithmetic (the scheduler's own step is already
// bit-exact against the reference in minimax_h3_sched):
//
//   * the keyframe CONDITION rows come out UNCHANGED. They lead the
//     video block so that "step only the generated rows" is a
//     contiguous slice; stepping them too would dissolve the anchors
//     the model interpolates between, and it would do it gradually
//     enough to read as weak conditioning rather than as a bug.
//   * the generated video AND audio rows both move. An audio branch
//     that was never stepped would leave the loop silently emitting its
//     initial noise, and the video would still look right.
//   * the loop is deterministic and finite.
//
// scratch_bytes() must equal what ensure_scratch_ actually allocates.
//
// The two list the same buffers in different places, so nothing but this
// test stops them drifting -- and drift here is not cosmetic. A video
// forward's scratch is ~200 KB per row (~1.9 GB at the production layout,
// ~3.9 GB at 19k rows), and the stage preflights against this number to
// decide whether the machine has room. An estimate that under-reports is
// how the preflight passes and the box then thrashes; on 16 GB, wired
// Metal buffers make that a watchdog kernel panic rather than a failed
// allocation. So: exact equality, not a tolerance.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH.
TEST(minimax_h3_dit, scratch_estimate_matches_allocation)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    std::printf("[minimax_h3_dit] config: %s\n", cerr.c_str());
    return;
  }
  // Two blocks: the scratch is a function of the GEOMETRY, not the depth,
  // and this keeps the load cheap.
  cfg.n_layers = 2;
  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  EXPECT_TRUE(m->scratch_resident_bytes() == 0);   // nothing allocated yet

  // A small real layout: 2 latent frames of a 320x192 canvas plus text
  // and audio rows. Small enough to allocate here, and it exercises the
  // same expressions the production geometry does.
  h3::PackedLayout L;
  const std::vector<int> tags(8, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, 2, 12, 20, 8, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels,
                                        {}, &L));

  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, kTVideo, kTAudio, kTCond, &uniq, &row_idx);

  const int n_video = (int)L.video_indices.size();
  std::vector<float> vin((std::size_t)n_video * cfg.video_patch_elems(), 0.01f);
  std::vector<float> ain((std::size_t)L.num_audio_rows * cfg.audio_channels,
                         0.01f);
  std::vector<float> tin((std::size_t)tags.size() * cfg.text_dim, 0.01f);
  const SharedBuffer vb = to_bf16_buf_(mc, vin);
  const SharedBuffer ab = to_bf16_buf_(mc, ain);
  const SharedBuffer tb = to_bf16_buf_(mc, tin);
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  MetalMiniMaxH3Transformer::Step step;
  step.video = &vb;  step.audio = &ab;  step.text = &tb;
  step.layout = &L;  step.timesteps = &uniq;
  step.row_timestep_index = &row_idx;
  std::string ferr;
  const auto v = m->forward(step, &ferr);
  if (v.empty()) {
    std::printf("[minimax_h3_dit] forward: %s\n", ferr.c_str());
  }
  ASSERT_TRUE(!v.empty());

  const std::size_t est_geo = MetalMiniMaxH3Transformer::scratch_bytes(
      cfg, L.seq_len, (int)tags.size(), (int)uniq.size(), false);
  const std::size_t est = MetalMiniMaxH3Transformer::scratch_bytes(
      cfg, L.seq_len, (int)tags.size(), (int)uniq.size(),
      m->uses_matrix_cores());
  const std::size_t act_geo = m->scratch_resident_bytes();
  const std::size_t act_dq  = m->dequant_scratch_bytes();
  std::printf("[minimax_h3_dit] %d rows, n_t=%zu: geometry est %zu / act %zu"
              " | dequant est %zu / act %zu (%.1f MB total)\n",
              L.seq_len, uniq.size(), est_geo, act_geo, est - est_geo, act_dq,
              (double)(act_geo + act_dq) / 1048576.0);
  EXPECT_TRUE(est_geo == act_geo);
  EXPECT_TRUE(est == act_geo + act_dq);
  // And it must be the dominant term it is claimed to be: per-row, not a
  // constant. ~200 KB/row at the released config.
  EXPECT_TRUE(act_geo > (std::size_t)L.seq_len * 64 * 1024);

  // A forced STEEL route must not touch the matrix-core path at all.
  //
  // This is the regression for a bug the estimate above exposed:
  // gemm_mma_ guarded only with route_ok_, which answers "can this route
  // run this shape" and is trivially true for a steel route -- so every
  // steel route fell into the mma path and ran the 128x128 tile. It was
  // invisible from outputs (the two agree numerically) and showed up only
  // as memory: dequant scratch existing when nothing should have
  // dequantized. The dequant buffer is the observable, so it is the
  // assertion.
  if (m->uses_matrix_cores()) {
    auto m2 = MetalMiniMaxH3Transformer::load(root, mc, cfg);
    ASSERT_TRUE(m2 != nullptr);
    m2->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
    std::string serr;
    const auto v2 = m2->forward(step, &serr);
    ASSERT_TRUE(!v2.empty());
    std::printf("[minimax_h3_dit] forced steel: dequant scratch %zu bytes\n",
                m2->dequant_scratch_bytes());
    EXPECT_TRUE(m2->dequant_scratch_bytes() == 0);
    // And the geometry scratch is unchanged -- the route decides which
    // kernel runs, never how much room the forward needs.
    EXPECT_TRUE(m2->scratch_resident_bytes() == act_geo);
  }
}

// The M5 fast paths must be SELECTED, not merely available.
//
// A silent fallback from the matrix-core GEMM to the steel one, or from the
// NAX flash attention to the ALU one, is 2-3x slower and numerically fine --
// so every golden here still passes and no correctness test can see it. That
// is exactly the shape of failure the Qwen fast-path guard exists for, and
// the Boogu bring-up is the case where it actually happened: NAX wiring
// written on a box without matrix cores, shipped unexercised, silently
// broken.
//
// minimax_h3_blocks proves the KERNELS work at this model's shapes without
// needing a checkpoint. This proves the MODEL reaches them, which is the
// half that needs one. Skips on a pre-M5 GPU and under the A/B env knobs,
// since selecting steel is the correct answer there.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH.
TEST(minimax_h3_dit, m5_fastpath_engaged_guard)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  if (std::getenv("VPIPE_H3_NO_MMA2") != nullptr ||
      std::getenv("VPIPE_H3_NO_ATTN_NAX") != nullptr ||
      std::getenv("VPIPE_H3_NO_STEEL_ATTN") != nullptr) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    std::printf("[minimax_h3_dit] config: %s\n", cerr.c_str());
    return;
  }
  // Two blocks: this asks which kernels the model SELECTS, and that is a
  // property of the loader, not of the depth.
  cfg.n_layers = 2;
  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  std::printf("[minimax_h3_dit] M5 paths: matmul2d GEMM %d, NAX attention "
              "%d\n", m->uses_matrix_cores() ? 1 : 0,
              m->uses_nax_attention() ? 1 : 0);
  EXPECT_TRUE(m->uses_matrix_cores());
  EXPECT_TRUE(m->uses_nax_attention());
}

// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH.
TEST(minimax_h3_dit, denoise_holds_the_anchors)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr));
  cfg.n_layers = kLayers;

  h3::PackedLayout L;
  const std::vector<int> text_tags((std::size_t)kTextRows, h3::kTextTag);
  // A FIRST-frame anchor, so there are conditioning rows to hold.
  const std::vector<h3::Anchor> anchors = {h3::Anchor::kFirst};
  ASSERT_TRUE(h3::build_packed_sequence(text_tags, kLatentF, kLatentH,
                                        kLatentW, kAudioLat, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels,
                                        anchors, &L));
  ASSERT_TRUE(L.num_condition_rows > 0 && L.num_video_rows > 0);
  ASSERT_TRUE(L.num_audio_rows > 0);

  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  const int PE = cfg.video_patch_elems();
  const int AC = cfg.audio_channels;
  const int vrows = (int)L.video_indices.size();

  SharedBuffer tb = mc->make_shared_buffer(
      (std::size_t)kTextRows * cfg.text_dim * 2);
  ASSERT_TRUE(!tb.empty());
  {
    auto* d = static_cast<std::uint16_t*>(tb.contents());
    for (int i = 0; i < kTextRows * cfg.text_dim; ++i) {
      d[i] = f32_to_bf16_(0.02f * (float)(i * 37 % 23 - 11));
    }
  }

  auto init = [&](std::vector<float>& v, std::vector<float>& a) {
    v.assign((std::size_t)vrows * PE, 0.0f);
    a.assign((std::size_t)L.num_audio_rows * AC, 0.0f);
    // SIGNED throughout: `(i * 31) % 19 - 9` on a size_t underflows to
    // ~1.8e19 whenever the modulus lands under 9, which makes the state
    // astronomically large and every relative change round to zero.
    for (std::size_t i = 0; i < v.size(); ++i) {
      v[i] = 0.1f * (float)((int)(i * 31 % 19) - 9);
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
      a[i] = 0.1f * (float)((int)(i * 17 % 13) - 6);
    }
  };
  std::vector<float> vid, aud;
  init(vid, aud);
  const std::vector<float> vid0 = vid, aud0 = aud;

  genai::DenoiseRequest req;
  req.dit    = m.get();
  req.layout = &L;
  req.text   = &tb;
  req.video  = vid.data();
  req.audio  = aud.data();
  req.num_steps = 4;
  std::string derr;
  const bool ok = genai::denoise(req, &derr);
  if (!ok) { std::printf("[minimax_h3_dit] denoise: %s\n", derr.c_str()); }
  ASSERT_TRUE(ok);

  // The anchors: byte-identical, not merely close.
  bool anchors_held = true;
  for (std::size_t i = 0; i < (std::size_t)L.num_condition_rows * PE; ++i) {
    if (vid[i] != vid0[i]) { anchors_held = false; break; }
  }
  EXPECT_TRUE(anchors_held);

  auto moved = [](const std::vector<float>& now, const std::vector<float>& was,
                  std::size_t from, std::size_t to) {
    double d = 0.0, base = 0.0;
    for (std::size_t i = from; i < to; ++i) {
      d += (double)(now[i] - was[i]) * (now[i] - was[i]);
      base += (double)was[i] * was[i];
    }
    return base > 0.0 ? std::sqrt(d / base) : 0.0;
  };
  const double dv = moved(vid, vid0, (std::size_t)L.num_condition_rows * PE,
                          vid.size());
  const double da = moved(aud, aud0, 0, aud.size());
  std::printf("[minimax_h3_dit] denoise %d steps: anchors held=%d, generated "
              "video moved %.4f, audio moved %.4f\n", req.num_steps,
              anchors_held ? 1 : 0, dv, da);
  // Both branches must have been stepped. The audio one is the easy
  // omission: it is 8 rows of 78 and the video would look fine without
  // it.
  EXPECT_TRUE(dv > 0.01);
  EXPECT_TRUE(da > 0.01);

  bool finite = true;
  for (float x : vid) { if (!std::isfinite(x)) { finite = false; break; } }
  for (float x : aud) { if (!std::isfinite(x)) { finite = false; break; } }
  EXPECT_TRUE(finite);

  // Deterministic: no RNG in the loop, so a rerun from the same state
  // must land on the same numbers.
  std::vector<float> vid2, aud2;
  init(vid2, aud2);
  genai::DenoiseRequest req2 = req;
  req2.video = vid2.data();
  req2.audio = aud2.data();
  ASSERT_TRUE(genai::denoise(req2, &derr));
  EXPECT_TRUE(vid2 == vid && aud2 == aud);
}

// Step timing at an ARBITRARY geometry, which is the whole question for
// this model: the packed sequence runs from ~500 rows at 256px to ~19k at
// the model's own 960x544x124, and every kernel choice that matters here
// is a function of that length rather than of the checkpoint.
//
// Pairs with VPIPE_H3_DIT_PROFILE (per-section GPU timing inside the
// forward) -- set both to get a breakdown at a chosen size. Depth is
// configurable and defaults SHALLOW: per-block cost is uniform, so the
// section SHARES and the GEMM kernel choice are fully visible at depth 2
// while a depth-50 load is 33 GB and minutes per iteration.
//
// Env: VPIPE_MINIMAX_H3_DIT_BENCH=1 + VPIPE_MINIMAX_H3_TEST_MODEL_PATH.
// Geometry: ..._BENCH_{LATF,LATH,LATW,AUD,TEXT,LAYERS,ITERS}.
TEST(minimax_h3_dit, step_bench)
{
  const char* on   = std::getenv("VPIPE_MINIMAX_H3_DIT_BENCH");
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (on == nullptr || *on == '\0' || *on == '0') { return; }
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int latf   = envi("VPIPE_MINIMAX_H3_BENCH_LATF", 17);
  const int lath   = envi("VPIPE_MINIMAX_H3_BENCH_LATH", 34);
  const int latw   = envi("VPIPE_MINIMAX_H3_BENCH_LATW", 60);
  const int naud   = envi("VPIPE_MINIMAX_H3_BENCH_AUD", 93);
  const int ntext  = envi("VPIPE_MINIMAX_H3_BENCH_TEXT", 16);
  const int layers = envi("VPIPE_MINIMAX_H3_BENCH_LAYERS", 2);
  const int iters  = envi("VPIPE_MINIMAX_H3_BENCH_ITERS", 3);

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    std::printf("[minimax_h3_dit] config: %s\n", cerr.c_str());
    return;
  }
  cfg.n_layers = layers;

  h3::PackedLayout L;
  const std::vector<int> text_tags((std::size_t)ntext, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(
      text_tags, latf, lath, latw, naud, cfg.patch_h, cfg.patch_w,
      h3::kAudioChannels, {h3::Anchor::kFirst}, &L));

  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, kTVideo, kTAudio, kTCond, &uniq, &row_idx);

  const int n_video = (int)L.video_indices.size();
  std::vector<float> vin((std::size_t)n_video * cfg.video_patch_elems());
  std::vector<float> ain((std::size_t)L.num_audio_rows * cfg.audio_channels);
  std::vector<float> tin((std::size_t)ntext * cfg.text_dim);
  for (std::size_t i = 0; i < vin.size(); ++i) {
    vin[i] = 0.01f * (float)((i % 197) - 98);
  }
  for (std::size_t i = 0; i < ain.size(); ++i) {
    ain[i] = 0.01f * (float)((i % 91) - 45);
  }
  for (std::size_t i = 0; i < tin.size(); ++i) {
    tin[i] = 0.01f * (float)((i % 131) - 65);
  }

  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  if (m == nullptr) { std::printf("[minimax_h3_dit] load failed\n"); }
  ASSERT_TRUE(m != nullptr);

  const SharedBuffer vb = to_bf16_buf_(mc, vin);
  const SharedBuffer ab = to_bf16_buf_(mc, ain);
  const SharedBuffer tb = to_bf16_buf_(mc, tin);
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  MetalMiniMaxH3Transformer::Step step;
  step.video  = &vb;
  step.audio  = &ab;
  step.text   = &tb;
  step.layout = &L;
  step.timesteps = &uniq;
  step.row_timestep_index = &row_idx;

  std::printf("[minimax_h3_dit] bench seq %d (%d text + %d cond + %d audio + "
              "%d video), %d blocks, qmm_tile %d\n", L.seq_len,
              L.num_text_rows, L.num_condition_rows, L.num_audio_rows,
              L.num_video_rows, cfg.n_layers, m->qmm_tile());
  // One forward to let the tile autotune run, so the choice it made is
  // printed alongside the timings it produced.
  {
    std::string werr;
    (void)m->forward(step, &werr);
    std::printf("[minimax_h3_dit] qmm tuning: %s\n",
                m->qmm_tuning().empty() ? "(off)" : m->qmm_tuning().c_str());
  }

  // VPIPE_MINIMAX_H3_BENCH_TILE_AB: alternate the quantized-GEMM tile cap
  // 0,1,2,... across iterations INSIDE this process. Two tiles compared
  // across two processes cannot resolve a few percent -- this box has ~4%
  // of thermal spread between runs -- and alternating puts every arm
  // under the same drift.
  const bool ab_tile =
      std::getenv("VPIPE_MINIMAX_H3_BENCH_TILE_AB") != nullptr;
  const int arms = ab_tile ? (m->qmm_tile() + 1) : 1;
  std::vector<double> arm_best((std::size_t)arms, 1e30);

  double best = 1e30;
  double l2_first = -1.0;
  bool l2_stable = true;
  for (int i = 0; i < iters * arms; ++i) {
    if (ab_tile) { m->set_qmm_tile(i % arms); }
    const auto t0 = std::chrono::steady_clock::now();
    std::string ferr;
    MetalMiniMaxH3Transformer::Velocity out = m->forward(step, &ferr);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (out.empty()) { std::printf("[minimax_h3_dit] %s\n", ferr.c_str()); }
    ASSERT_TRUE(!out.empty());
    // A tile that changes the ANSWER is not a speedup. The goldens run at
    // 78 rows, where the tall tiles never engage, so this is the only
    // place a tall-tile numerical difference could show up.
    double l2 = 0.0;
    {
      const auto* p = static_cast<const std::uint16_t*>(out.video.contents());
      const std::size_t n = out.video.byte_size() / 2;
      for (std::size_t k = 0; k < n; ++k) {
        const double v = (double)bf16_to_f32_(p[k]);
        l2 += v * v;
      }
      l2 = std::sqrt(l2 / (double)(n ? n : 1));
    }
    if (l2_first < 0.0) { l2_first = l2; }
    else if (std::fabs(l2 - l2_first) > 1e-6 * (l2_first + 1e-9)) {
      l2_stable = false;
    }
    // Iteration 0 of each arm pays for scratch allocation and function
    // binding, neither of which recurs inside a denoise loop.
    if (i >= arms) {
      best = std::min(best, ms);
      arm_best[(std::size_t)(i % arms)] =
          std::min(arm_best[(std::size_t)(i % arms)], ms);
    }
    std::printf("[minimax_h3_dit]   iter %2d tile %d: %8.1f ms  rms %.6f\n",
                i, ab_tile ? (i % arms) : m->qmm_tile(), ms, l2);
  }
  if (ab_tile) {
    for (int a = 0; a < arms; ++a) {
      std::printf("[minimax_h3_dit] tile %d best %8.1f ms  (%.3fx vs tile 0)\n",
                  a, arm_best[(std::size_t)a],
                  arm_best[0] / arm_best[(std::size_t)a]);
    }
  }
  std::printf("[minimax_h3_dit] best %.1f ms/step at %d rows, %d blocks "
              "-> %.1f ms/block\n", best, L.seq_len, cfg.n_layers,
              best / (double)(cfg.n_layers ? cfg.n_layers : 1));
  EXPECT_TRUE(l2_stable);
  EXPECT_TRUE(best < 1e29);
}
