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
#include "generative-models/shared/riffle-rows.h"
#include "generative-models/shared/streamed-refill.h"
#include "generative-models/weight-set.h"
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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <memory>
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
      cfg, L.seq_len, (int)tags.size(), (int)uniq.size(), false,
      m->ff_scratch_narrow());
  const std::size_t est = MetalMiniMaxH3Transformer::scratch_bytes(
      cfg, L.seq_len, (int)tags.size(), (int)uniq.size(),
      m->uses_matrix_cores(), m->ff_scratch_narrow());
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
// The ref2va path through the REAL DiT: interleaved reference blocks.
//
// `t2va` / `fl2va` give every modality one contiguous range (video two),
// which is what the transformer's scatter and its two output heads were
// written against. `ref2va` breaks that -- a video reference's soundtrack
// rows sit immediately before its own video rows, so the two modalities
// interleave and neither is one range any more.
//
// What this pins is that the run-driven addressing puts every row where
// the layout says. A wrong offset does not fail: it feeds the reference
// block's latents to the target rows and back, and the model generates
// confidently from the wrong conditioning.
TEST(minimax_h3_dit, ref2va_denoise_holds_the_reference_blocks)
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

  // An image, then a video carrying its own soundtrack: the smallest
  // request in which audio is NOT one contiguous block.
  using Ref = h3::Reference;
  const std::vector<Ref> refs = {
      Ref{Ref::Kind::kImage, 1, kLatentH, kLatentW, 0},
      Ref{Ref::Kind::kVideo, 3, kLatentH, kLatentW, 12},
  };
  h3::PackedLayout L;
  const std::vector<int> text_tags((std::size_t)kTextRows, h3::kTextTag);
  ASSERT_TRUE(h3::build_ref2va_packed_sequence(
      text_tags, refs, kLatentF, kLatentH, kLatentW, kAudioLat, cfg.patch_h,
      cfg.patch_w, h3::kAudioChannels, &L));
  ASSERT_TRUE(L.num_condition_video_rows > 0);
  ASSERT_TRUE(L.num_condition_audio_rows > 0);
  ASSERT_TRUE(L.audio_runs.size() >= 2);      // reference + target
  ASSERT_TRUE(L.video_runs.size() >= 3);      // image + clip + target

  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  const int PE = cfg.video_patch_elems();
  const int AC = cfg.audio_channels;
  const int vrows = (int)L.video_indices.size();
  const int arows = (int)L.audio_indices.size();

  SharedBuffer tb = mc->make_shared_buffer(
      (std::size_t)kTextRows * cfg.text_dim * 2);
  ASSERT_TRUE(!tb.empty());
  {
    auto* d = static_cast<std::uint16_t*>(tb.contents());
    for (int i = 0; i < kTextRows * cfg.text_dim; ++i) {
      d[i] = f32_to_bf16_(0.02f * (float)(i * 37 % 23 - 11));
    }
  }

  std::vector<float> vid((std::size_t)vrows * PE, 0.0f);
  std::vector<float> aud((std::size_t)arows * AC, 0.0f);
  for (std::size_t i = 0; i < vid.size(); ++i) {
    vid[i] = 0.1f * (float)((int)(i * 31 % 19) - 9);
  }
  for (std::size_t i = 0; i < aud.size(); ++i) {
    aud[i] = 0.1f * (float)((int)(i * 17 % 13) - 6);
  }
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
  if (!ok) { std::printf("[minimax_h3_dit] ref2va denoise: %s\n",
                         derr.c_str()); }
  ASSERT_TRUE(ok);

  // Both modalities' reference rows survive byte-identical. Audio is the
  // new one: it has never had conditioning rows before, so a loop that
  // stepped the whole buffer would overwrite the soundtrack it is meant
  // to be conditioning ON.
  bool vheld = true, aheld = true;
  for (std::size_t i = 0; i < (std::size_t)L.num_condition_video_rows * PE;
       ++i) {
    if (vid[i] != vid0[i]) { vheld = false; break; }
  }
  for (std::size_t i = 0; i < (std::size_t)L.num_condition_audio_rows * AC;
       ++i) {
    if (aud[i] != aud0[i]) { aheld = false; break; }
  }
  EXPECT_TRUE(vheld);
  EXPECT_TRUE(aheld);

  auto moved = [](const std::vector<float>& now, const std::vector<float>& was,
                  std::size_t from, std::size_t to) {
    double d = 0.0, base = 0.0;
    for (std::size_t i = from; i < to; ++i) {
      d += (double)(now[i] - was[i]) * (now[i] - was[i]);
      base += (double)was[i] * was[i];
    }
    return base > 0.0 ? std::sqrt(d / base) : 0.0;
  };
  const double dv =
      moved(vid, vid0, (std::size_t)L.num_condition_video_rows * PE,
            vid.size());
  const double da =
      moved(aud, aud0, (std::size_t)L.num_condition_audio_rows * AC,
            aud.size());
  std::printf("[minimax_h3_dit] ref2va denoise %d steps: %d video runs, %d "
              "audio runs; references held v=%d a=%d; generated video moved "
              "%.4f, audio moved %.4f\n", req.num_steps,
              (int)L.video_runs.size(), (int)L.audio_runs.size(),
              vheld ? 1 : 0, aheld ? 1 : 0, dv, da);
  EXPECT_TRUE(dv > 0.01);
  EXPECT_TRUE(da > 0.01);
}

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
  // At least two: iteration 0 of each arm pays for scratch allocation and
  // function binding and is discarded, so ITERS=1 would time nothing and
  // fail the `best` assertion rather than report a number.
  const int iters  = std::max(2, envi("VPIPE_MINIMAX_H3_BENCH_ITERS", 3));

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
  // VPIPE_MINIMAX_H3_BENCH_FUSED_AB: the same interleave over the
  // attention layout masks -- arm 0 transposes all four activations,
  // arm 1 only stores the output row-major, arm 2 also reads q/k/v out
  // of the fused projection in place. Every arm is the SAME arithmetic
  // on the same bytes, so the rms check below is a correctness
  // assertion here and not just a sanity one.
  const bool ab_fused =
      std::getenv("VPIPE_MINIMAX_H3_BENCH_FUSED_AB") != nullptr;
  const int kFusedArm[] = {0, MetalMiniMaxH3Transformer::kFusedAttnOut,
                           MetalMiniMaxH3Transformer::kFusedAttnOut |
                               MetalMiniMaxH3Transformer::kFusedAttnQkv};
  const int n_fuse = (int)std::size(kFusedArm);
  const int arms = ab_tile ? (m->qmm_tile() + 1) : (ab_fused ? n_fuse : 1);

  // THE WEAVE, on buffers the size of one block's fc1 -- timed on
  // scratch of its own, because riffling a block that is already
  // interleaved would corrupt it. Pure CPU and independent of the row
  // count, which is the whole asymmetry: it is paid per block-load
  // while the fused saving is paid per row.
  {
    const std::size_t N  = (std::size_t)(2 * cfg.ffn);
    const int bits = m->quant_bits() > 0 ? m->quant_bits() : 8;
    const std::size_t rc = (std::size_t)cfg.hidden * (std::size_t)bits / 8;
    const std::size_t rg =
        (std::size_t)cfg.hidden / (std::size_t)m->quant_group() * 2;
    SharedBuffer wc = mc->make_shared_buffer(N * rc);
    SharedBuffer wsc = mc->make_shared_buffer(N * rg);
    SharedBuffer wqb = mc->make_shared_buffer(N * rg);
    if (!wc.empty() && !wsc.empty() && !wqb.empty()) {
      double bestw = 1e30;
      for (int r = 0; r < 5; ++r) {
        const auto w0 = std::chrono::steady_clock::now();
        vpipe::genai::riffle_rows(wc, N);
        vpipe::genai::riffle_rows(wsc, N);
        vpipe::genai::riffle_rows(wqb, N);
        bestw = std::min(bestw, std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - w0).count());
      }
      std::printf("[minimax_h3_dit] weave: %.2f ms per block "
                  "(w%d, %zu MB fc1), row-count independent\n",
                  bestw, bits, (N * rc + 2 * N * rg) >> 20);
    }
  }
  std::vector<double> arm_best((std::size_t)arms, 1e30);

  double best = 1e30;
  double l2_first = -1.0;
  bool l2_stable = true;
  for (int i = 0; i < iters * arms; ++i) {
    if (ab_tile) { m->set_qmm_tile(i % arms); }
    if (ab_fused) {
      m->set_fused_attention(kFusedArm[(std::size_t)(i % arms)]);
      // What the model ACCEPTED, not what the arm asked for: a build
      // without the in-place rope silently runs arm 1 twice, and this
      // is where that shows.
      if (m->fused_attention() != kFusedArm[(std::size_t)(i % arms)]) {
        std::printf("[minimax_h3_dit]   (arm %d refused: mask %d)\n",
                    i % arms, m->fused_attention());
      }
    }
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
    std::printf("[minimax_h3_dit]   iter %2d %s %d: %8.1f ms  rms %.6f\n",
                i, ab_fused ? "fusemask" : "tile",
                ab_fused ? kFusedArm[(std::size_t)(i % arms)]
                         : (ab_tile ? (i % arms) : m->qmm_tile()), ms, l2);
  }
  if (ab_tile || ab_fused) {
    const char* what = ab_fused ? "fusemask" : "tile";
    for (int a = 0; a < arms; ++a) {
      std::printf("[minimax_h3_dit] %s %d best %8.1f ms  (%.3fx vs %s 0)\n",
                  what, ab_fused ? kFusedArm[(std::size_t)a] : a,
                  arm_best[(std::size_t)a],
                  arm_best[0] / arm_best[(std::size_t)a], what);
    }
  }
  std::printf("[minimax_h3_dit] best %.1f ms/step at %d rows, %d blocks "
              "-> %.1f ms/block\n", best, L.seq_len, cfg.n_layers,
              best / (double)(cfg.n_layers ? cfg.n_layers : 1));
  EXPECT_TRUE(l2_stable);
  EXPECT_TRUE(best < 1e29);
}

// The GEMM tuner prunes its candidate list on a warm pass, so that it
// gets cheaper on every machine without being told anything about any of
// them. This is the RULE half, and the rule is what can go wrong
// silently: a pruned winner is not an error, it is just a slower model.
//
// Three invariants, and the first one is not academic -- the rule was
// written with both minima seeded from warm[0], which keeps ONE arm out
// of an already-ascending list instead of the two it promises.
//
// No GPU and no checkpoint: this is arithmetic.
TEST(minimax_h3_dit, warm_pass_prune_keeps_the_contenders)
{
  using M = MetalMiniMaxH3Transformer;
  auto has = [](const std::vector<int>& v, int i) {
    return std::find(v.begin(), v.end(), i) != v.end();
  };

  // Ascending, evenly spread -- the case the seeded-from-warm[0] version
  // got wrong.
  {
    const std::vector<int> live = M::tune_prune_survivors({1.0, 2.0, 3.0}, 0.6);
    EXPECT_TRUE(live.size() == 2);
    EXPECT_TRUE(has(live, 0) && has(live, 1));
  }
  // One leader, the rest far behind: the runner-up survives anyway, so
  // the vote always has something to compare against.
  {
    const std::vector<int> live =
        M::tune_prune_survivors({1.0, 9.0, 9.1, 9.2}, 0.6);
    EXPECT_TRUE(live.size() == 2);
    EXPECT_TRUE(has(live, 0) && has(live, 1));
  }
  // A close cluster is never reached into -- this is the M5 split sweep,
  // where the warm pass under-rated the eventual winner by 15%.
  {
    const std::vector<int> live =
        M::tune_prune_survivors({1.9, 1.25, 1.10, 1.00, 1.18, 1.11}, 0.6);
    EXPECT_TRUE(live.size() == 5);
    EXPECT_TRUE(!has(live, 0));           // 0.53 of the leader: hopeless
  }
  // The M5 route sweep: three steel arms at ~3x, three matmul2d arms
  // close together. Exactly the steel arms go.
  {
    const std::vector<int> live =
        M::tune_prune_survivors({3.42, 3.16, 3.21, 1.15, 1.07, 1.00}, 0.6);
    EXPECT_TRUE(live.size() == 3);
    EXPECT_TRUE(has(live, 3) && has(live, 4) && has(live, 5));
  }
  // All within the threshold: nothing is dropped. A machine with no
  // matrix cores sees this, and pays exactly what it paid before.
  {
    const std::vector<int> live =
        M::tune_prune_survivors({1.00, 1.19, 1.12}, 0.6);
    EXPECT_TRUE(live.size() == 3);
  }
  // The leader is in the survivors no matter what the field looks like.
  {
    const std::vector<double> warm{4.0, 1.0, 4.1, 40.0, 4.2};
    const std::vector<int> live = M::tune_prune_survivors(warm, 0.6);
    EXPECT_TRUE(has(live, 1));
    EXPECT_TRUE(!has(live, 3));
  }
  // A warm pass that failed to time is not evidence, and neither is a
  // field of two. Both prune nothing.
  {
    EXPECT_TRUE(M::tune_prune_survivors({1.0, 0.0, 5.0}, 0.6).size() == 3);
    EXPECT_TRUE(M::tune_prune_survivors({1.0, -1.0, 5.0}, 0.6).size() == 3);
    EXPECT_TRUE(M::tune_prune_survivors({1.0, 9.0}, 0.6).size() == 2);
    EXPECT_TRUE(M::tune_prune_survivors({}, 0.6).empty());
  }
}

// A tuned GEMM answer is filed under the row count it was MEASURED at,
// not the row count the caller asked for -- so two geometries above the
// tuner's ceiling share one measurement instead of running the identical
// sweep twice.
//
// Two claims, and the second is the one that can rot silently. The key
// ALGEBRA is arithmetic. But the store and the lookup compute that key in
// different functions, and if they ever disagreed nothing would fail --
// every projection would just fall back to the untuned rule and only a
// timing would show it. So this drives real forwards and counts the
// measurements.
//
// VPIPE_H3_TUNE_ROWS drops the ceiling so the sequences can be small: the
// property under test is about the ceiling, not about its default value,
// and at the default this test would cost two 4096-row sweeps. It is read
// at LOAD and kept per model, which is what makes setting it here work
// whatever else has run in this process first.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH.
TEST(minimax_h3_dit, tuned_routes_are_shared_above_the_tune_ceiling)
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
  cfg.n_layers = 1;

  ::setenv("VPIPE_H3_TUNE_ROWS", "128", 1);
  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  ::unsetenv("VPIPE_H3_TUNE_ROWS");
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  const int ceil = m->tune_row_ceiling();
  ASSERT_TRUE(ceil == 128);
  if (ceil != 128) { return; }
  // The algebra: shared at and above the ceiling, exact below it.
  EXPECT_TRUE(m->tune_row_key(ceil) == ceil);
  EXPECT_TRUE(m->tune_row_key(ceil + 1) == ceil);
  EXPECT_TRUE(m->tune_row_key(9382) == m->tune_row_key(16990));
  EXPECT_TRUE(m->tune_row_key(ceil - 1) == ceil - 1);
  EXPECT_TRUE(m->tune_row_key(ceil - 1) != m->tune_row_key(ceil));

  auto ramp = [](std::size_t n, float k) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = std::sin((float)i * k) * 0.5f;
    }
    return v;
  };
  // One forward at a layout of `latf` video frames. The two calls must
  // land at DIFFERENT sequence lengths, both above the ceiling, or this
  // proves nothing -- checked below rather than assumed.
  auto run_at = [&](int latf, int* seq) {
    h3::PackedLayout L;
    const std::vector<int> tags(8, h3::kTextTag);
    if (!h3::build_packed_sequence(tags, latf, 12, 20, 8, cfg.patch_h,
                                   cfg.patch_w, h3::kAudioChannels, {}, &L)) {
      return false;
    }
    *seq = L.seq_len;
    std::vector<float> uniq;
    std::vector<int>   row_idx;
    h3::build_row_timesteps(L, kTVideo, kTAudio, kTCond, &uniq, &row_idx);
    const int n_video = (int)L.video_indices.size();
    const SharedBuffer vb = to_bf16_buf_(
        mc, ramp((std::size_t)n_video * cfg.video_patch_elems(), 0.017f));
    const SharedBuffer ab = to_bf16_buf_(
        mc, ramp((std::size_t)L.num_audio_rows * cfg.audio_channels, 0.031f));
    const SharedBuffer tb = to_bf16_buf_(
        mc, ramp((std::size_t)tags.size() * cfg.text_dim, 0.005f));
    if (vb.empty() || ab.empty() || tb.empty()) { return false; }
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    const auto out = m->forward(step, &ferr);
    if (out.empty()) {
      std::printf("[minimax_h3_dit] forward: %s\n", ferr.c_str());
      return false;
    }
    return true;
  };

  int seq_a = 0, seq_b = 0;
  ASSERT_TRUE(run_at(2, &seq_a));
  // Nothing was tuned at all: no matrix cores and only one steel tile
  // built, so there was never a choice to make. Say so -- the counts
  // below would all be 0 and would pass while proving nothing.
  if (m->qmm_tune_count() == 0) {
    std::printf("[minimax_h3_dit] no GEMM route to tune on this build; "
                "sharing not exercised\n");
    return;
  }
  EXPECT_TRUE(m->qmm_tune_count() == 1);
  // Same geometry again: a cache hit whichever way the key is computed,
  // so this is the weak half and it comes first.
  ASSERT_TRUE(run_at(2, &seq_b));
  EXPECT_TRUE(seq_b == seq_a);
  EXPECT_TRUE(m->qmm_tune_count() == 1);
  // A DIFFERENT sequence length, still above the ceiling. This is the
  // claim: one measurement covers both.
  ASSERT_TRUE(run_at(3, &seq_b));
  if (seq_b == seq_a || seq_a <= ceil || seq_b <= ceil) {
    std::printf("[minimax_h3_dit] layouts %d/%d do not straddle the "
                "ceiling %d; sharing not exercised\n", seq_a, seq_b, ceil);
    return;
  }
  EXPECT_TRUE(m->qmm_tune_count() == 1);
}

// IS THE FORWARD DETERMINISTIC AT A VIDEO GEOMETRY?
//
// A 1376x768x243 generation returns whole-latent noise in both video and
// audio, while 175 and 209 frames are clean, and the author's own reading
// is that it may not fail EVERY time. Every kernel on the block path has
// been probed in isolation at and past that height and none of them
// moves -- elementwise, norms, rope, all five GEMM arms, the band loop,
// the matmul2d destination cliff, attention against float64, arena
// subviews past 4 GB, and the LoRA pair. So what is left is not a kernel:
// it is composition, and pressure.
//
// Neither reproduces in a kernel probe. Composition needs the real
// aliasing (ff over qh|kh|vh|oh, proj over ob) driven by the real block
// loop; pressure needs the real streamed checkpoint competing with ~10 GB
// of activation scratch on a box that has 24. This runs the actual
// forward TWICE on the same input and diffs it bit-exact, which catches
// both without needing to know which:
//
//   * a race, or a shared scratch read after the next writer touched it,
//     is a different answer on the second run
//   * a purged or parked page returning zeros is a different answer
//   * an uninitialised read is a different answer
//   * a deterministic index bug is NOT caught here -- it would produce
//     the same wrong bytes twice -- which is exactly why the kernel
//     probes came first
//
// It needs no golden and no reference implementation, and one forward is
// a fraction of a generation, so it is affordable on the box that
// actually reproduces the corruption where a full run is not.
//
// GEOMETRY FROM THE ENV, because the box that reproduces this is not the
// box this was written on. The defaults are the failing clip: 1376x768 is
// a 86x48 latent, 243 frames is 72 latent frames, and 243/24 s at 40
// latents/second is 405 audio latents.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH, VPIPE_H3_DETERMINISM=1,
// and optionally VPIPE_H3_DET_{LATF,LATH,LATW,AUD,LAYERS,RUNS}.
TEST(minimax_h3_dit, forward_is_deterministic_at_video_heights)
{
  if (std::getenv("VPIPE_H3_DETERMINISM") == nullptr) { return; }
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int latf   = envi("VPIPE_H3_DET_LATF", 72);    // 243 frames
  const int lath   = envi("VPIPE_H3_DET_LATH", 48);    // 768 / 16
  const int latw   = envi("VPIPE_H3_DET_LATW", 86);    // 1376 / 16
  const int naud   = envi("VPIPE_H3_DET_AUD", 405);    // 243/24 s x 40
  const int ntext  = envi("VPIPE_H3_DET_TEXT", 16);
  const int runs   = envi("VPIPE_H3_DET_RUNS", 2);

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    std::printf("[minimax_h3_dit] config: %s\n", cerr.c_str());
    return;
  }
  // FOUR BLOCKS by default, not the whole stack.
  //
  // The full stack was the original default, on the reasoning that only
  // it reproduces a real generation's memory pressure. That reasoning
  // was right and the default was still wrong: reproducing the pressure
  // means loading the entire checkpoint beside ~10 GB of scratch, and on
  // the 24 GB box that actually shows the bug it took the machine down.
  // A test that can only demonstrate the failure by causing it is not a
  // test, it is the failure.
  //
  // Four blocks still exercise composition, the arena aliasing and the
  // slot alternation, which is what a kernel probe cannot reach. The
  // pressure half belongs to instrumenting a real run, not to
  // manufacturing it here.
  const int meta_layers = cfg.n_layers;
  cfg.n_layers = envi("VPIPE_H3_DET_LAYERS", 4);

  h3::PackedLayout L;
  const std::vector<int> text_tags((std::size_t)ntext, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(
      text_tags, latf, lath, latw, naud, cfg.patch_h, cfg.patch_w,
      h3::kAudioChannels, {h3::Anchor::kFirst}, &L));
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, kTVideo, kTAudio, kTCond, &uniq, &row_idx);
  std::printf("[minimax_h3_dit] determinism: seq %d (%d text + %d cond + "
              "%d audio + %d video), %d blocks, %d runs\n", L.seq_len,
              L.num_text_rows, L.num_condition_rows, L.num_audio_rows,
              L.num_video_rows, cfg.n_layers, runs);
  {
    const std::size_t sc = MetalMiniMaxH3Transformer::scratch_bytes(
        cfg, L.seq_len, L.num_text_rows, (int)uniq.size(), true);
    const auto b = mc->memory_budget();
    std::printf("[minimax_h3_dit] scratch estimate %.2f GB; headroom %.2f "
                "GB, available physical %.2f GB\n",
                (double)sc / 1073741824.0,
                (double)b.headroom / 1073741824.0,
                (double)b.available_physical / 1073741824.0);
    // REFUSE A GEOMETRY THIS BOX CANNOT HOLD, rather than finding out by
    // running it. The scratch is mlock-WIRED, so the kernel cannot
    // reclaim it under pressure: an over-commit here does not fail the
    // process, it takes the machine down. Asked before the model loads,
    // because by then the weights are resident too.
    //
    // Two gigabytes of margin covers the trunk, the block slots and the
    // dequant scratch, none of which are in `sc`. VPIPE_H3_DET_FORCE
    // overrides for a box with room to spare -- and is deliberately not
    // something the geometry knobs imply.
    // The WEIGHTS TOO, which the first version of this check left out --
    // and they are the larger term whenever n_layers is more than a
    // handful. A block of this model is ~0.6 GB at 8 bits, the trunk and
    // the embedders are another couple, and scratch_bytes() knows about
    // none of it. Leaving them out is what let a full-stack run at the
    // video geometry look affordable when it was three times the box.
    std::size_t weights = 0;
    {
      namespace fs = std::filesystem;
      std::error_code wec;
      for (const auto& de : fs::recursive_directory_iterator(
               fs::path(root), fs::directory_options::skip_permission_denied,
               wec)) {
        if (wec) { break; }
        if (de.is_regular_file(wec) &&
            de.path().extension() == ".safetensors") {
          weights += (std::size_t)de.file_size(wec);
        }
      }
      // Blocks scale with the stack this run asks for; the trunk, the
      // embedders and the two heads do not, so they get a flat two
      // gigabytes on top. Both terms err HIGH on purpose -- the
      // checkpoint directory may hold more than the one partition being
      // read, and an under-estimate here is what takes a box down.
      const double frac = meta_layers > 0
                              ? (double)cfg.n_layers / (double)meta_layers
                              : 1.0;
      weights = (std::size_t)((double)weights * std::min(1.0, frac)) +
                ((std::size_t)2 << 30);
    }
    // Idle physical, not `available_physical`: the latter counts
    // file-backed and purgeable pages the OS *could* reclaim, which on a
    // box already deep in swap reads as room that does not exist.
    // free_physical is free + purgeable + speculative and deliberately
    // EXCLUDES the file cache -- which is what makes it the right number
    // here. available_physical counts clean file pages as room, and on a
    // box already deep in swap that reads as space which does not exist.
    const std::size_t idle = b.free_physical != 0 ? b.free_physical
                                                  : b.available_physical;
    const std::size_t margin = (std::size_t)3 << 30;
    const std::size_t need = sc + weights;
    std::printf("[minimax_h3_dit] need ~%.2f GB (scratch %.2f + weights "
                "~%.2f); idle physical %.2f GB\n",
                (double)need / 1073741824.0, (double)sc / 1073741824.0,
                (double)weights / 1073741824.0,
                (double)idle / 1073741824.0);
    if (std::getenv("VPIPE_H3_DET_FORCE") == nullptr &&
        idle != 0 && need + margin > idle) {
      std::printf("[minimax_h3_dit] SKIPPED: ~%.2f GB needed plus %.0f GB "
                  "of margin does not fit in %.2f GB idle. Lower "
                  "VPIPE_H3_DET_LAYERS or VPIPE_H3_DET_LATF; do NOT reach "
                  "for VPIPE_H3_DET_FORCE on a box that is already "
                  "swapping.\n", (double)need / 1073741824.0,
                  (double)margin / 1073741824.0,
                  (double)idle / 1073741824.0);
      return;
    }
  }

  const int n_video = (int)L.video_indices.size();
  auto ramp = [](std::size_t n, float k) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = std::sin((float)i * k) * 0.5f;
    }
    return v;
  };
  const SharedBuffer vb = to_bf16_buf_(
      mc, ramp((std::size_t)n_video * cfg.video_patch_elems(), 0.017f));
  const SharedBuffer ab = to_bf16_buf_(
      mc, ramp((std::size_t)L.audio_indices.size() * cfg.audio_channels,
               0.031f));
  const SharedBuffer tb = to_bf16_buf_(
      mc, ramp((std::size_t)ntext * cfg.text_dim, 0.005f));
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  if (m == nullptr) { std::printf("[minimax_h3_dit] load failed\n"); }
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  MetalMiniMaxH3Transformer::Step step;
  step.video  = &vb;
  step.audio  = &ab;
  step.text   = &tb;
  step.layout = &L;
  step.timesteps = &uniq;
  step.row_timestep_index = &row_idx;

  // Raw bf16 both times: the claim is that the bytes are identical, so
  // they are compared as bytes. A float compare would let a NaN pass as
  // equal to itself, and NaN is one of the things being looked for.
  std::vector<std::uint16_t> ref_v, ref_a;
  bool diverged = false;
  for (int r = 0; r < runs; ++r) {
    std::string ferr;
    const auto t0 = std::chrono::steady_clock::now();
    MetalMiniMaxH3Transformer::Velocity out = m->forward(step, &ferr);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (out.empty()) { std::printf("[minimax_h3_dit] %s\n", ferr.c_str()); }
    ASSERT_TRUE(!out.empty());
    if (out.empty()) { return; }
    auto copy = [](const SharedBuffer& b) {
      const auto* p = static_cast<const std::uint16_t*>(b.contents());
      return std::vector<std::uint16_t>(p, p + b.byte_size() / 2);
    };
    std::vector<std::uint16_t> v = copy(out.video), a = copy(out.audio);
    // A finite, non-degenerate velocity is its own check: an all-zero or
    // all-NaN output would otherwise be perfectly reproducible.
    std::size_t bad_v = 0;
    double l2 = 0.0;
    for (std::uint16_t h : v) {
      const float f = bf16_to_f32_(h);
      if (!std::isfinite(f)) { ++bad_v; }
      else { l2 += (double)f * (double)f; }
    }
    l2 = std::sqrt(l2 / (double)(v.empty() ? 1 : v.size()));
    const auto b = mc->memory_budget();
    std::printf("[minimax_h3_dit]   run %d: %8.0f ms  rms %.6f  non-finite "
                "%zu  scratch %.2f GB  avail %.2f GB\n", r, ms, l2, bad_v,
                (double)m->scratch_resident_bytes() / 1073741824.0,
                (double)b.available_physical / 1073741824.0);
    EXPECT_TRUE(bad_v == 0);
    if (r == 0) { ref_v = std::move(v); ref_a = std::move(a); continue; }
    std::size_t dv = 0, da = 0;
    for (std::size_t i = 0; i < v.size() && i < ref_v.size(); ++i) {
      if (v[i] != ref_v[i]) { ++dv; }
    }
    for (std::size_t i = 0; i < a.size() && i < ref_a.size(); ++i) {
      if (a[i] != ref_a[i]) { ++da; }
    }
    if (dv != 0 || da != 0) {
      diverged = true;
      std::printf("[minimax_h3_dit]   run %d DIFFERS from run 0: video %zu "
                  "of %zu, audio %zu of %zu\n", r, dv, v.size(), da,
                  a.size());
    }
    EXPECT_TRUE(dv == 0 && da == 0);
  }
  std::printf("[minimax_h3_dit] determinism: %s over %d runs\n",
              diverged ? "NOT REPRODUCIBLE" : "identical", runs);
}

// Attention must compute the same thing whichever of its four
// activations it addresses in place.
//
// The bar here is BIT-EXACT, not close: the three masks are the same
// arithmetic in the same order on the same values, and all that differs
// is which addresses those values live at. Nothing rounds differently,
// so anything but equality is a layout error -- a wrong stride, a head
// grouping read as the other one, rope applied to the wrong channel
// pair, the in-place hazard not actually settled by the barrier.
//
// And a layout error here is SILENT. Every wrong version still produces
// a full velocity of plausible magnitude; the rms barely moves, because
// scrambling which head attends to what does not change how big the
// numbers are. Only equality separates them.
//
// One load, three masks, because the mask is settable at runtime --
// which also means this is the test that the setter takes effect at
// all, and it says so rather than passing vacuously if a mask is
// refused.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH.
TEST(minimax_h3_dit, fused_attention_matches_the_transposes)
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
  // Four blocks, and a layout with text, condition, audio and video rows
  // -- the refiner blocks run unmodulated over the text rows alone and
  // take a different rope path (rot 0), so a mask that is only right for
  // the main blocks has to be able to show it.
  cfg.n_layers = 4;

  h3::PackedLayout L;
  const std::vector<int> tags(8, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, 2, 12, 20, 8, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels,
                                        {}, &L));
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, kTVideo, kTAudio, kTCond, &uniq, &row_idx);
  const int n_video = (int)L.video_indices.size();
  auto ramp = [](std::size_t n, float k) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = std::sin((float)i * k) * 0.5f;
    }
    return v;
  };
  const SharedBuffer vb = to_bf16_buf_(
      mc, ramp((std::size_t)n_video * cfg.video_patch_elems(), 0.017f));
  const SharedBuffer ab = to_bf16_buf_(
      mc, ramp((std::size_t)L.num_audio_rows * cfg.audio_channels, 0.031f));
  const SharedBuffer tb = to_bf16_buf_(
      mc, ramp((std::size_t)tags.size() * cfg.text_dim, 0.005f));
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  MetalMiniMaxH3Transformer::Step step;
  step.video = &vb;  step.audio = &ab;  step.text = &tb;
  step.layout = &L;  step.timesteps = &uniq;
  step.row_timestep_index = &row_idx;

  // Raw bf16, not floats: the claim is that the bytes are identical, and
  // comparing them as bytes is the way to say so.
  auto run = [&](int mask, std::vector<std::uint16_t>* v,
                 std::vector<std::uint16_t>* a) {
    m->set_fused_attention(mask);
    if (m->fused_attention() != mask) { return false; }
    std::string ferr;
    const auto out = m->forward(step, &ferr);
    if (out.empty()) {
      std::printf("[minimax_h3_dit] forward: %s\n", ferr.c_str());
      return false;
    }
    auto copy = [](const SharedBuffer& b) {
      const auto* p = static_cast<const std::uint16_t*>(b.contents());
      return std::vector<std::uint16_t>(p, p + b.byte_size() / 2);
    };
    *v = copy(out.video);
    *a = copy(out.audio);
    return !v->empty();
  };

  const int kOut = MetalMiniMaxH3Transformer::kFusedAttnOut;
  const int kQkv = MetalMiniMaxH3Transformer::kFusedAttnQkv;
  std::vector<std::uint16_t> bv, ba;
  const bool base = run(0, &bv, &ba);
  ASSERT_TRUE(base);
  if (!base) { return; }

  for (int mask : {kOut, kOut | kQkv}) {
    std::vector<std::uint16_t> v, a;
    if (!run(mask, &v, &a)) {
      // Refused, not wrong: a build without the steel attention or the
      // in-place rope cannot offer this mask, and comparing the default
      // path with itself would pass while proving nothing.
      std::printf("[minimax_h3_dit] fuse mask %d unavailable; not "
                  "compared\n", mask);
      continue;
    }
    ASSERT_TRUE(v.size() == bv.size() && a.size() == ba.size());
    std::size_t bad = 0;
    for (std::size_t i = 0; i < v.size() && i < bv.size(); ++i) {
      if (v[i] != bv[i]) { ++bad; }
    }
    for (std::size_t i = 0; i < a.size() && i < ba.size(); ++i) {
      if (a[i] != ba[i]) { ++bad; }
    }
    if (bad != 0) {
      std::printf("[minimax_h3_dit] fuse mask %d: %zu of %zu elements "
                  "differ from the transposing path\n", mask, bad,
                  v.size() + a.size());
    }
    EXPECT_TRUE(bad == 0);
  }
  // The default the model ships with, restored for anything that reuses
  // this process.
  m->set_fused_attention(kOut);
}

// The fused-SwiGLU FF must be the same function as the GEMM-then-split
// pair it replaces, on the REAL checkpoint and through a whole stack of
// blocks.
//
// This is where the fusion is actually checked, and it has to be here
// rather than at the end of a generation: the sampler is chaotic, so a
// last-ulp difference in one forward and a genuinely wrong FF both come
// out as two unrelated videos. Only the velocity out of one forward
// separates them.
//
// The bar is the one rounding the two paths genuinely differ by -- the
// split path stores fc1's gate and up as bf16 and reads them back, the
// fused one keeps the pair in the accumulator -- carried through the
// block stack. A gate/up mix-up, the interleave written the other way
// round, or a stale weight scores O(1) here, three orders clear.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (a quantized g64 DiT; nothing
// else can fuse, and the test says so rather than passing vacuously).
TEST(minimax_h3_dit, fused_ff_matches_split)
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
  // Eight blocks: enough that a per-block difference accumulates the way
  // it would over fifty, cheap enough to load twice.
  cfg.n_layers = 8;

  h3::PackedLayout L;
  const std::vector<int> tags(8, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, 2, 12, 20, 8, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels,
                                        {}, &L));
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, kTVideo, kTAudio, kTCond, &uniq, &row_idx);
  const int n_video = (int)L.video_indices.size();
  // Deterministic non-constant inputs: a constant one would hide a
  // column permutation, which is exactly the mistake the interleave can
  // make.
  auto ramp = [](std::size_t n, float k) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = std::sin((float)i * k) * 0.5f;
    }
    return v;
  };
  const std::vector<float> vin =
      ramp((std::size_t)n_video * cfg.video_patch_elems(), 0.017f);
  const std::vector<float> ain =
      ramp((std::size_t)L.num_audio_rows * cfg.audio_channels, 0.031f);
  const std::vector<float> tin =
      ramp((std::size_t)tags.size() * cfg.text_dim, 0.005f);
  const SharedBuffer vb = to_bf16_buf_(mc, vin);
  const SharedBuffer ab = to_bf16_buf_(mc, ain);
  const SharedBuffer tb = to_bf16_buf_(mc, tin);
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  // One arm. The env is read at LOAD, so each arm is its own load --
  // which is also what makes `narrow` below a real check that the arms
  // differ rather than two runs of the same path.
  auto arm = [&](bool fused, std::vector<float>* vel_v,
                 std::vector<float>* vel_a, bool* narrow) {
    if (fused) { ::unsetenv("VPIPE_H3_NO_FUSED_FF"); }
    else       { ::setenv("VPIPE_H3_NO_FUSED_FF", "1", 1); }
    auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
    if (m == nullptr) { return false; }
    *narrow = m->ff_scratch_narrow();
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    const auto v = m->forward(step, &ferr);
    if (v.empty()) {
      std::printf("[minimax_h3_dit] forward: %s\n", ferr.c_str());
      return false;
    }
    auto copy = [](const SharedBuffer& b, std::size_t n) {
      std::vector<float> out(n);
      const auto* p = static_cast<const std::uint16_t*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) { out[i] = bf16_to_f32_(p[i]); }
      return out;
    };
    *vel_v = copy(v.video, (std::size_t)n_video * cfg.video_patch_elems());
    *vel_a = copy(v.audio,
                  (std::size_t)L.audio_indices.size() * cfg.audio_channels);
    return true;
  };

  std::vector<float> fv, fa, sv, sa;
  bool f_narrow = false, s_narrow = true;
  const bool ok_f = arm(true, &fv, &fa, &f_narrow);
  const bool ok_s = arm(false, &sv, &sa, &s_narrow);
  ::unsetenv("VPIPE_H3_NO_FUSED_FF");
  ASSERT_TRUE(ok_f && ok_s);
  // A quantized g64 checkpoint fuses; anything else cannot, and then
  // this test has compared one path with itself. Say so instead.
  if (!f_narrow) {
    std::printf("[minimax_h3_dit] fused FF unavailable for this "
                "checkpoint; nothing compared\n");
    return;
  }
  EXPECT_TRUE(!s_narrow);

  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };
  const double rv = rel(fv, sv), ra = rel(fa, sa);
  std::printf("[minimax_h3_dit] fused vs split over %d blocks: video "
              "rel-L2 %.3e, audio %.3e\n", cfg.n_layers, rv, ra);
  EXPECT_TRUE(rv < 2e-2);
  EXPECT_TRUE(ra < 2e-2);
}

// The streamed block stack reads into TWO reusable destinations and
// refills them in place, instead of allocating a block's worth of buffers
// per block and filling them out of the shard mapping. This is where that
// is checked to be the same function.
//
// BIT-EXACT, not a tolerance. Slots change where the bytes land and how
// they get there -- a pread off the shard rather than a memcpy out of the
// mapping -- and nothing about which bytes they are or what is computed
// over them. Any difference at all is a bug, and the interesting bugs
// here all produce SMALL ones: a stale tensor left over from the previous
// block in a slot that was not fully refilled, the f16 scales converted
// twice (or not at all) because a refill ran over an already-converted
// buffer, or the prefetch writing the slot the GPU is reading. Every one
// of those yields a plausible video with slightly wrong numbers, which a
// tolerance would wave through.
//
// Both arms STREAM. Preloading does not touch this path at all, so an arm
// that preloaded would compare the slot path against nothing.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH.
TEST(minimax_h3_dit, streamed_slots_match_per_block_loads)
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
  // Four blocks: the slot pair alternates every block, so four covers
  // both slots twice and the wrap between them three times, which is
  // where an off-by-one in the alternation would show.
  cfg.n_layers = 4;

  h3::PackedLayout L;
  const std::vector<int> tags(8, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, 2, 12, 20, 8, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels,
                                        {}, &L));
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, kTVideo, kTAudio, kTCond, &uniq, &row_idx);
  const int n_video = (int)L.video_indices.size();
  auto ramp = [](std::size_t n, float k) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = std::sin((float)i * k) * 0.5f;
    }
    return v;
  };
  const SharedBuffer vb = to_bf16_buf_(
      mc, ramp((std::size_t)n_video * cfg.video_patch_elems(), 0.017f));
  const SharedBuffer ab = to_bf16_buf_(
      mc, ramp((std::size_t)L.num_audio_rows * cfg.audio_channels, 0.031f));
  const SharedBuffer tb = to_bf16_buf_(
      mc, ramp((std::size_t)tags.size() * cfg.text_dim, 0.005f));
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  // Raw bf16 words, not floats: this is an exactness check, so nothing
  // should pass through a conversion that could round two different
  // values to one.
  // mode 0 = the slot PAIR, 1 = slots OFF (per-block allocations),
  // 2 = SINGLE slot -- which is what a box takes when the budget refuses
  // the second slot, and a path neither of the original two arms reaches.
  int pf_started = 0, pf_hit = 0;
  bool saw_pair = false;
  auto arm = [&](int mode, std::vector<std::uint16_t>* out_v,
                 std::vector<std::uint16_t>* out_a) {
    ::setenv("VPIPE_H3_STREAM", "1", 1);
    if (mode == 1) { ::setenv("VPIPE_H3_NO_SLOTS", "1", 1); }
    else           { ::unsetenv("VPIPE_H3_NO_SLOTS"); }
    if (mode == 2) { ::setenv("VPIPE_H3_SINGLE_SLOT", "1", 1); }
    else           { ::unsetenv("VPIPE_H3_SINGLE_SLOT"); }
    auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
    if (m == nullptr) { return false; }
    // The GEMM route is chosen by MEASUREMENT, so two loads of the same
    // model can pick different winners for the same shape and the
    // forward is not reproducible run to run. MEASURED on the bf16
    // checkpoint: the slot arm compared against ITSELF came out at
    // rel-L2 3.2e-03 over half the words, which is the same spread as
    // any other pair -- so without pinning, this test measures the
    // tuner's noise floor and can say nothing about the weights.
    m->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    const auto v = m->forward(step, &ferr);
    if (v.empty()) {
      std::printf("[minimax_h3_dit] forward: %s\n", ferr.c_str());
      return false;
    }
    auto raw = [](const SharedBuffer& b, std::size_t n) {
      const auto* p = static_cast<const std::uint16_t*>(b.contents());
      return std::vector<std::uint16_t>(p, p + n);
    };
    *out_v = raw(v.video, (std::size_t)n_video * cfg.video_patch_elems());
    *out_a = raw(v.audio,
                 (std::size_t)L.audio_indices.size() * cfg.audio_channels);
    pf_started = m->prefetch_started();
    pf_hit     = m->prefetch_hits();
    saw_pair   = m->slot_pair();
    std::printf("[minimax_h3_dit] slots mode %d: pair=%d prefetch %d/%d "
                "hit\n", mode, saw_pair ? 1 : 0, pf_hit, pf_started);
    return true;
  };

  std::vector<std::uint16_t> sv, sa, pv, pa, gv, ga;
  const bool ok_s = arm(0, &sv, &sa);
  const int pair_started = pf_started, pair_hit = pf_hit;
  const bool was_pair = saw_pair;
  const bool ok_p = arm(1, &pv, &pa);
  const bool ok_g = arm(2, &gv, &ga);
  const int single_started = pf_started, single_hit = pf_hit;
  const bool single_pair = saw_pair;
  ::unsetenv("VPIPE_H3_NO_SLOTS");
  ::unsetenv("VPIPE_H3_SINGLE_SLOT");
  ::unsetenv("VPIPE_H3_STREAM");
  ASSERT_TRUE(ok_s && ok_p && ok_g);

  // THE SINGLE-SLOT ARM IS A THIRD PATH, and it has to be the same
  // function as the other two. A box tight enough to refuse the second
  // slot takes it without saying so, which is why it needs forcing to be
  // testable at all.
  {
    std::size_t dv = 0, da = 0;
    for (std::size_t i = 0; i < gv.size() && i < sv.size(); ++i) {
      if (gv[i] != sv[i]) { ++dv; }
    }
    for (std::size_t i = 0; i < ga.size() && i < sa.size(); ++i) {
      if (ga[i] != sa[i]) { ++da; }
    }
    if (dv != 0 || da != 0) {
      std::printf("[minimax_h3_dit] SINGLE-SLOT DIFFERS from the pair: "
                  "video %zu of %zu, audio %zu of %zu\n", dv, gv.size(), da,
                  ga.size());
    }
    EXPECT_TRUE(dv == 0 && da == 0);
  }
  // And the prefetch has to be ACCOUNTED FOR. A read issued and never
  // consumed is not merely wasted: on the fallback destination the block
  // it filled stays live for the rest of the forward, which is a block's
  // worth of memory added on exactly the runs that were already too tight
  // to afford a second slot. `started == hit` is the property; a run that
  // issued nothing trivially has it.
  std::printf("[minimax_h3_dit] pair(pair=%d) %d/%d, single(pair=%d) %d/%d\n",
              was_pair ? 1 : 0, pair_hit, pair_started,
              single_pair ? 1 : 0, single_hit, single_started);
  EXPECT_TRUE(single_pair == false);          // the force took effect
  EXPECT_TRUE(pair_hit == pair_started);
  EXPECT_TRUE(single_hit == single_started);
  ASSERT_TRUE(!sv.empty() && !sa.empty());
  EXPECT_TRUE(sv.size() == pv.size());
  EXPECT_TRUE(sa.size() == pa.size());

  std::size_t diff = 0;
  double num = 0.0, den = 0.0, worst = 0.0;
  auto cmp = [&](const std::vector<std::uint16_t>& a,
                 const std::vector<std::uint16_t>& b) {
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
      if (a[i] != b[i]) { ++diff; }
      const double x = bf16_to_f32_(a[i]), y = bf16_to_f32_(b[i]);
      num += (x - y) * (x - y);
      den += y * y;
      worst = std::max(worst, std::fabs(x - y));
    }
  };
  cmp(sv, pv);
  cmp(sa, pa);
  const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  std::printf("[minimax_h3_dit] slots vs per-block loads over %d streamed "
              "blocks: %zu of %zu words differ, rel-L2 %.3e, max |d| "
              "%.3e\n", cfg.n_layers, diff, sv.size() + sa.size(), rel,
              worst);
  EXPECT_TRUE(diff == 0);
}

// A baked schedule has to produce the SAME velocity as the projections
// it replaces.
//
// This is a memory optimization, so the bar is exact agreement, not a
// tolerance: baking changes WHERE the modulation is computed, not how.
// Both arms run the same GEMM against the same weights over the same
// timesteps -- the baked one just does it once for every step up front,
// at M = all rows instead of M = this step's rows. The only way that
// moves a bit is if the GEMM's route depends on M, which it does (the
// tuner keys on it), so the route is pinned in both arms and what is
// left must be bit-identical.
//
// What it is really guarding is the INDEXING. A step's rows sit at an
// offset in the table, blocks bind their slice rather than copying it,
// and the modulation kernels read it through an adaln index built per
// step -- so an off-by-one in row0, a stale n_t, or a table laid out
// per-distinct-value instead of per-step all produce a plausible video
// made from another step's noise level.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH.
// The AdaLN bake's READ, against the read it replaces.
//
// The bake is the largest single read in a run -- 13.86 GB at the
// released 8-bit config, 39.3% of the checkpoint -- and it now takes it
// through one reusable destination per tensor (pread) instead of
// allocating a fresh set per block and copying out of the shard's mmap.
// Same bytes by a different route, so the bytes are what to check, on
// every dtype an AdaLN projection carries.
//
// It also pins the REUSE, which is what makes the fast read available:
// a destination sized for block 0 has to serve block 1 and block 2. If
// the projections were not shape-identical the refill would refuse and
// the bake would silently fall back to the slow path -- correct, and
// none of the point.
//
// No model is loaded and nothing runs on the GPU: this is about the
// weight set, and the equivalent forward-level test
// (baked_adaln_matches_the_projections) cannot run on a 16 GB box
// against a 35 GB checkpoint, because its UNBAKED arm is the thing the
// bake exists to avoid.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH.
TEST(minimax_h3_dit, adaln_refill_matches_the_mapped_read)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::shared_ptr<genai::WeightSet> ws = genai::WeightSet::open(root, nullptr);
  if (!ws) {
    std::printf("[minimax_h3_dit] no weight set at %s\n", root);
    return;
  }
  // bf16 is what the model reads a projection as, so it is what the
  // refill is asked for: the u32 codes land raw and the f16 scales and
  // biases are converted in place, which is what to_bf16_() does on the
  // route this replaces.
  const char* kSuffix[] = {".weight", ".scales", ".biases", ".bias"};
  std::vector<metal_compute::SharedBuffer> slot(4);
  int checked = 0, reused = 0;
  std::size_t bytes = 0, differ = 0;

  for (int b = 0; b < 3; ++b) {
    for (int k = 0; k < 4; ++k) {
      const std::string nm =
          "blocks." + std::to_string(b) + ".adaln_proj.linear" + kSuffix[k];
      const auto* ti = ws->src().info(nm);
      if (ti == nullptr) { continue; }

      // The reference: the read this replaces. F16 is converted on the
      // host exactly as to_bf16_ does -- f16 -> f32 -> round-to-nearest-
      // even bf16 -- because that is the value the forward consumed
      // before and has to consume now.
      metal_compute::SharedBuffer raw =
          ws->stream_tensor(nm, mc, genai::WeightSet::Residency::Copied);
      if (raw.empty()) { continue; }
      std::vector<std::uint16_t> want;
      const std::size_t nbytes = ti->nbytes;
      if (ti->dtype == "F16") {
        const std::size_t n = nbytes / 2;
        want.resize(n);
        const auto* src = static_cast<const _Float16*>(raw.contents());
        for (std::size_t i = 0; i < n; ++i) {
          float f = (float)src[i];
          std::uint32_t u;
          std::memcpy(&u, &f, 4);
          want[i] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
        }
      } else {
        want.resize(nbytes / 2);
        std::memcpy(want.data(), raw.contents(), nbytes);
      }

      // Block 0 allocates; 1 and 2 must fit the SAME buffer.
      if (slot[(std::size_t)k].empty()) {
        slot[(std::size_t)k] = mc->make_shared_buffer(nbytes);
      } else {
        ++reused;
      }
      ASSERT_TRUE(!slot[(std::size_t)k].empty());
      const genai::Refill r = genai::refill_streamed_tensor(
          *ws, nm, slot[(std::size_t)k], genai::RefillDst::kBf16);
      EXPECT_TRUE(r == genai::Refill::kFilled);
      if (r != genai::Refill::kFilled) { continue; }
      ++checked;
      bytes += nbytes;
      if (std::memcmp(slot[(std::size_t)k].contents(), want.data(),
                      nbytes) != 0) {
        ++differ;
        std::printf("[minimax_h3_dit] MISMATCH %s (%s)\n", nm.c_str(),
                    ti->dtype.c_str());
      }
    }
  }
  std::printf("[minimax_h3_dit] adaln refill: %d tensors (%.1f MB), %d into "
              "a reused destination, %zu differ\n", checked,
              (double)bytes / 1e6, reused, differ);
  EXPECT_TRUE(checked >= 8);
  // The reuse is the mechanism, not an incidental: without it there is
  // no destination to pread into.
  EXPECT_TRUE(reused >= 4);
  EXPECT_TRUE(differ == 0);
}

TEST(minimax_h3_dit, baked_adaln_matches_the_projections)
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
  cfg.n_layers = 4;

  h3::PackedLayout L;
  const std::vector<int> tags(8, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, 5, 20, 12, 8, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels,
                                        {}, &L));
  // Three steps at DIFFERENT noise levels, so a table read at the wrong
  // offset lands on a different timestep rather than on a copy of the
  // right one.
  const float kV[] = {0.9f, 0.5f, 0.15f};
  const float kA[] = {0.8f, 0.45f, 0.1f};
  const int kSteps = 3;

  const int n_video = (int)L.video_indices.size();
  auto ramp = [](std::size_t n, float k) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) { v[i] = std::sin((float)i * k) * 0.5f; }
    return v;
  };
  const metal_compute::SharedBuffer vb = to_bf16_buf_(
      mc, ramp((std::size_t)n_video * cfg.video_patch_elems(), 0.017f));
  const metal_compute::SharedBuffer ab = to_bf16_buf_(
      mc, ramp((std::size_t)L.num_audio_rows * cfg.audio_channels, 0.031f));
  const metal_compute::SharedBuffer tb = to_bf16_buf_(
      mc, ramp((std::size_t)tags.size() * cfg.text_dim, 0.005f));
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  std::vector<std::vector<float>> sched;
  std::vector<std::vector<float>> uniqs(kSteps);
  std::vector<std::vector<int>>   ridx(kSteps);
  for (int i = 0; i < kSteps; ++i) {
    h3::build_row_timesteps(L, kV[i], kA[i], 1.0f, &uniqs[(std::size_t)i],
                            &ridx[(std::size_t)i]);
    sched.push_back(uniqs[(std::size_t)i]);
  }

  // The GEMM route is chosen by measurement and keys on M, and the two
  // arms run the AdaLN projection at DIFFERENT M (all rows at once vs
  // one step's). Pinned, so the comparison is about the table.
  const auto kPin = MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32;
  auto run = [&](bool bake, std::vector<std::vector<float>>* out) {
    auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg, false);
    if (m == nullptr) { return false; }
    m->set_gemm_route(kPin);
    if (bake) {
      std::string berr;
      if (!m->bake_adaln(sched, &berr)) {
        std::printf("[minimax_h3_dit] bake: %s\n", berr.c_str());
        return false;
      }
      if (!m->adaln_baked()) { return false; }
      std::printf("[minimax_h3_dit] baked %zu MB of AdaLN tables\n",
                  (std::size_t)(m->adaln_table_bytes() >> 20));
    }
    out->assign((std::size_t)kSteps, {});
    for (int i = 0; i < kSteps; ++i) {
      MetalMiniMaxH3Transformer::Step st;
      st.video = &vb;  st.audio = &ab;  st.text = &tb;
      st.layout = &L;
      st.timesteps          = &uniqs[(std::size_t)i];
      st.row_timestep_index = &ridx[(std::size_t)i];
      st.schedule_index     = bake ? i : -1;
      std::string ferr;
      const auto v = m->forward(st, &ferr);
      if (v.empty()) {
        std::printf("[minimax_h3_dit] forward: %s\n", ferr.c_str());
        return false;
      }
      const std::size_t n =
          (std::size_t)n_video * cfg.video_patch_elems();
      auto& dst = (*out)[(std::size_t)i];
      dst.resize(n);
      const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
      for (std::size_t k = 0; k < n; ++k) { dst[k] = bf16_to_f32_(p[k]); }
    }
    return true;
  };

  std::vector<std::vector<float>> plain, baked;
  ASSERT_TRUE(run(false, &plain));
  ASSERT_TRUE(run(true, &baked));
  for (int i = 0; i < kSteps; ++i) {
    const auto& a = plain[(std::size_t)i];
    const auto& b = baked[(std::size_t)i];
    ASSERT_TRUE(a.size() == b.size() && !a.empty());
    std::size_t diff = 0;
    double num = 0.0, den = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
      if (a[k] != b[k]) { ++diff; }
      const double d = (double)a[k] - (double)b[k];
      num += d * d;
      den += (double)a[k] * (double)a[k];
    }
    std::printf("[minimax_h3_dit] step %d (t_v=%.2f): %zu/%zu elements "
                "differ, rel-L2 %.3e\n", i, (double)kV[i], diff, a.size(),
                den > 0.0 ? std::sqrt(num / den) : 0.0);
    EXPECT_TRUE(diff == 0);
  }
}
