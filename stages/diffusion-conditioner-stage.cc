#include "stages/diffusion-conditioner-stage.h"
#include <chrono>

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-config-source.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/context-manager.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#endif

#include <memory>
#include <string>
#include <vector>

namespace vpipe {

namespace {

// The model iport (a model-select source) overrides hf_dir. Inserted after the
// prompt + negative inputs, so it is iport2 (ref_image follows at iport3).
// (Referenced only from the Apple-gated code; maybe_unused for non-Apple.)
[[maybe_unused]] constexpr unsigned kModelPort = 2;

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "model dir (text_encoder/, transformer/, tokenizer/); the "
          "transformer's _class_name selects the family + encoder. OPTIONAL: a "
          "model-select source on the model iport overrides it",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "krea2,flux2,qwen-image-edit,mage-flow,mage-flow-edit,"
       "boogu-image,boogu-image-edit,"
       "wan-t2v,wan-i2v,minimax-h3-fl2va",
   .model_channel = "diffusion-model"},
  {.key = "grounded_negative", .type = ConfigType::Bool, .required = false,
   .doc = "image-aware families only: always emit a negative conditioning on "
          "oport1 -- a GROUNDED encode of the (possibly empty) negative prompt "
          "-- so the DiT can run classifier-free guidance (CFG>1). Matches the "
          "Krea-2 edit deletion recipe (empty grounded negative). Default false "
          "(emit a negative only when a non-empty negative prompt is wired)"},
  {.key = "unload_when_idle", .type = ConfigType::String, .required = false,
   .doc = "drop the text encoder (and vision tower) after each conditioning is "
          "emitted and reload it on the next prompt. The encoder is idle for the "
          "whole denoise, so on a memory-bounded box this is what lets a large "
          "DiT and a large encoder share one machine (a 4-bit Boogu-Image is a "
          "~5.6 GB DiT beside a ~4.7 GB Qwen3-VL mllm). \"auto\" (default) "
          "decides from physical RAM, whether a peer decided to stream, and "
          "the pipeline's weight bytes. Force it with \"destroy\" (free the "
          "bytes; costs a reload per prompt), \"park\" (hand them to the "
          "kernel as purgeable -- reclaimed only if the box needs the RAM, "
          "reused without a reload if not) or \"keep\" (hold them pinned). "
          "Legacy: \"always\" = destroy, \"never\" = keep",
   .def_str = "auto"},
};
const PortSpec kIports[] = {
  {.name = "prompt", .doc = "prompt text (FlexData string or {text: ...})",
   .type = &typeid(FlexDataPayload), .tags = "text", .clock_group = 0},
  {.name = "negative", .doc = "OPTIONAL negative prompt (FlexData) for the DiT's "
                              "classifier-free guidance; its conditioning is "
                              "emitted on oport1",
   .type = &typeid(FlexDataPayload), .tags = "text", .clock_group = 0},
  {.name = "model", .doc = "OPTIONAL shared model reference from a model-select "
                           "source; overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "ref_image", .doc = "OPTIONAL raw reference image (planar U8 RGB "
                               "TensorBeat [3,H,W], load-image format). Image-"
                               "aware families (Qwen-Image-Edit) run it through "
                               "the Qwen2.5-VL vision tower; others ignore it.",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "ref_image2", .doc = "OPTIONAL SECOND reference image (same format). "
                                "Qwen-Image-Edit-2511 is multi-reference and "
                                "Mage-Flow-Edit's template has a per-reference "
                                "body, so on those families the VLM must see "
                                "BOTH pictures (the DiT's ref_latent1 only "
                                "carries the second one's spatial detail). Each "
                                "reference gets its own vision block, mROPE band "
                                "and deepstack run. Krea-2 is single-reference "
                                "by design and ignores it.",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "model_config",
   .doc = "OPTIONAL model-specific parameters from the resident family's own "
          "config source (krea2-model-config, mage-flow-model-config, "
          "boogu-image-model-config, qwen-image-edit-model-config). Today "
          "that means how a REFERENCE IMAGE is prepared for the grounded "
          "encode: each family's reference pipeline bounds it differently "
          "and the numbers are not interchangeable. Unwired, the family's "
          "own numbers apply",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
[[maybe_unused]] constexpr unsigned kModelCfgPort = 5;
const PortSpec kOports[] = {
  {.name = "conditioning",
   .doc = "conditioning tensor for the generate-image DiT (family-shaped + typed: "
          "krea2 f16 [n,12,2560]; flux2 f16 [n,3*enc_hidden]; qwen-image-edit "
          "bf16 [n_real,3584] image-aware; mage-flow bf16 [n_real,2560] "
          "image-aware)",
   .type = &typeid(TensorBeatPayload), .tags = "conditioning", .clock_group = 0},
  {.name = "neg_conditioning",
   .doc = "conditioning for the negative prompt (same shape/type); emitted only "
          "when a negative prompt is wired",
   .type = &typeid(TensorBeatPayload), .tags = "conditioning", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "diffusion-conditioner",
  .doc       = "Prompt (+ optional reference image) -> conditioning embeddings "
               "for a diffusion DiT. Owns the tokenizer + text encoder + (for "
               "image-aware models) the Qwen2.5-VL vision tower. The encoder "
               "half of the generate-image split; pair it with a generate-image "
               "stage on the same hf_dir. On the Mage-Flow families every "
               "prompt (and, for an edit, the source image) is first screened "
               "by the model's own content-policy classifier -- mandatory, no "
               "config key; a refused prompt yields a blank image instead of "
               "a generation.",
  .display_name = "Diffusion Conditioner",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

#ifdef VPIPE_BUILD_APPLE_SILICON

using metal_compute::SharedBuffer;

inline std::uint16_t f32_to_bf16_(float f)
{
  std::uint32_t u; std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

// Downscale planar U8 RGB [3,H,W] so the longest side <= `cap`, preserving the
// aspect ratio with an area-average box filter (matching the ComfyUI grounded-
// encode node's "area" grounding resize). Writes into `out` + sets *oh/*ow when
// it resizes; leaves `out` empty (no-op, caller keeps the original) when the
// image already fits. The Qwen3-VL tower smart-resizes internally, but its cap
// (~1M px area) is looser than the node's 768 grounding_px default, so bound the
// longest side here to stay in the LoRA's 384-768px training distribution.
// PIL-faithful separable LANCZOS-3 resample of a planar U8 RGB image. The
// reference pipelines that preprocess a VLM conditioning image with
// `resample="lanczos"` (Boogu-Image's BooguImageProcessor) are matched only by
// this filter -- a box/area average differs enough that, once the 36-layer LM
// amplifies it, the image-row conditioning lands ~3x further from the reference
// (measured: rel-L2 0.56 box vs 0.18 lanczos on the same 1024->384 downscale).
//
// Mirrors Pillow's ImagingResample: filterscale = max(1, in/out), support = 3 *
// filterscale, weights lanczos((k + 0.5 - center) / filterscale) normalised to
// sum 1, and -- as Pillow does -- the horizontal pass is CLAMPED BACK TO U8
// before the vertical pass, which is observable in the result.
void resize_lanczos_(const std::uint8_t* rgb, int H, int W, int nh, int nw,
                     std::vector<std::uint8_t>& out)
{
  auto lanczos = [](double x) {
    if (x < 0.0) { x = -x; }
    if (x < 1e-9) { return 1.0; }
    if (x >= 3.0) { return 0.0; }
    const double px = M_PI * x;
    return (std::sin(px) / px) * (std::sin(px / 3.0) / (px / 3.0));
  };
  // Per-output-pixel weight tables for one axis.
  struct Axis {
    std::vector<int> lo, hi;
    std::vector<std::vector<double>> w;
  };
  auto build = [&](int in, int outn) {
    Axis ax;
    const double scale = (double)in / (double)outn;
    const double fs = scale < 1.0 ? 1.0 : scale;
    const double support = 3.0 * fs;
    ax.lo.resize((std::size_t)outn);
    ax.hi.resize((std::size_t)outn);
    ax.w.resize((std::size_t)outn);
    for (int i = 0; i < outn; ++i) {
      const double center = ((double)i + 0.5) * scale;
      int lo = (int)std::floor(center - support);
      int hi = (int)std::ceil(center + support);
      if (lo < 0) { lo = 0; }
      if (hi > in) { hi = in; }
      if (hi <= lo) { lo = in > 0 ? std::min(in - 1, std::max(0, lo)) : 0;
                      hi = lo + 1; }
      std::vector<double> w((std::size_t)(hi - lo));
      double sum = 0.0;
      for (int k = lo; k < hi; ++k) {
        const double t = ((double)k + 0.5 - center) / fs;
        const double v = lanczos(t);
        w[(std::size_t)(k - lo)] = v;
        sum += v;
      }
      if (sum != 0.0) { for (double& v : w) { v /= sum; } }
      ax.lo[(std::size_t)i] = lo;
      ax.hi[(std::size_t)i] = hi;
      ax.w[(std::size_t)i] = std::move(w);
    }
    return ax;
  };
  const Axis axw = build(W, nw);
  const Axis axh = build(H, nh);
  auto clamp8 = [](double v) {
    const double r = std::round(v);
    return (std::uint8_t)(r < 0.0 ? 0.0 : (r > 255.0 ? 255.0 : r));
  };
  out.assign((std::size_t)3 * nh * nw, 0);
  std::vector<std::uint8_t> mid((std::size_t)H * nw);   // one plane, H x nw
  for (int c = 0; c < 3; ++c) {
    const std::uint8_t* src = rgb + (std::size_t)c * H * W;
    for (int y = 0; y < H; ++y) {                       // horizontal pass
      for (int x = 0; x < nw; ++x) {
        const auto& w = axw.w[(std::size_t)x];
        double acc = 0.0;
        for (int k = axw.lo[(std::size_t)x]; k < axw.hi[(std::size_t)x]; ++k) {
          acc += w[(std::size_t)(k - axw.lo[(std::size_t)x])] *
                 (double)src[(std::size_t)y * W + k];
        }
        mid[(std::size_t)y * nw + x] = clamp8(acc);      // U8 between passes
      }
    }
    std::uint8_t* dst = out.data() + (std::size_t)c * nh * nw;
    for (int y = 0; y < nh; ++y) {                       // vertical pass
      const auto& w = axh.w[(std::size_t)y];
      for (int x = 0; x < nw; ++x) {
        double acc = 0.0;
        for (int k = axh.lo[(std::size_t)y]; k < axh.hi[(std::size_t)y]; ++k) {
          acc += w[(std::size_t)(k - axh.lo[(std::size_t)y])] *
                 (double)mid[(std::size_t)k * nw + x];
        }
        dst[(std::size_t)y * nw + x] = clamp8(acc);
      }
    }
  }
}

// `max_pixels` (0 = unbounded) additionally bounds the AREA, and `align`
// (0 = none) floors both dims to a multiple of it -- Boogu's
// BooguImageProcessor.get_new_height_width takes the min of the pixel and
// side-length ratios, clamps to <= 1 (never upscales) and floor-aligns to
// vae_scale_factor 16. `lanczos` picks the filter (see resize_lanczos_);
// false keeps the historical box average.
void cap_longest_side_(const std::uint8_t* rgb, int H, int W, int cap,
                       std::vector<std::uint8_t>& out, int* oh, int* ow,
                       std::size_t max_pixels = 0, int align = 0,
                       bool lanczos = false)
{
  *oh = H; *ow = W;
  if (H <= 0 || W <= 0) { return; }
  const int longest = std::max(H, W);
  double s = 1.0;
  if (cap > 0 && longest > 0) { s = std::min(s, (double)cap / (double)longest); }
  if (max_pixels > 0) {
    const double cur = (double)H * (double)W;
    if (cur > 0.0) {
      s = std::min(s, std::sqrt((double)max_pixels / cur));
    }
  }
  if (s >= 1.0) { return; }                    // never upscale
  int nh, nw;
  if (align > 0) {
    nh = std::max(align, (int)(H * s) / align * align);
    nw = std::max(align, (int)(W * s) / align * align);
  } else {
    nh = std::max(1, (int)std::lround(H * s));
    nw = std::max(1, (int)std::lround(W * s));
  }
  if (nh == H && nw == W) { return; }
  if (lanczos) {
    resize_lanczos_(rgb, H, W, nh, nw, out);
    *oh = nh; *ow = nw;
    return;
  }
  out.assign((std::size_t)3 * nh * nw, 0);
  for (int c = 0; c < 3; ++c) {
    const std::uint8_t* src = rgb + (std::size_t)c * H * W;
    std::uint8_t* dst = out.data() + (std::size_t)c * nh * nw;
    for (int y = 0; y < nh; ++y) {
      const int y0 = (int)((double)y * H / nh);
      const int y1 = std::max(y0 + 1, (int)((double)(y + 1) * H / nh));
      for (int x = 0; x < nw; ++x) {
        const int x0 = (int)((double)x * W / nw);
        const int x1 = std::max(x0 + 1, (int)((double)(x + 1) * W / nw));
        std::uint32_t acc = 0, cnt = 0;
        for (int yy = y0; yy < y1 && yy < H; ++yy) {
          for (int xx = x0; xx < x1 && xx < W; ++xx) {
            acc += src[(std::size_t)yy * W + xx]; ++cnt;
          }
        }
        dst[(std::size_t)y * nw + x] = (std::uint8_t)(cnt ? acc / cnt : 0u);
      }
    }
  }
  *oh = nh; *ow = nw;
}

// ---- Prompt templates (shared with the generate-image stage) ----
constexpr const char* kPrefix =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n";
constexpr const char* kSuffix = "<|im_end|>\n<|im_start|>assistant\n";
constexpr int kDropPrefix = 34;
const int kSelectLayers[12] = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35};
const int kFluxTaps[3] = {9, 18, 27};

constexpr const char* kQiePrefix =
    "<|im_start|>system\nDescribe the key features of the input image (color, "
    "shape, size, texture, objects, background), then explain how the user's "
    "text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the "
    "original input where appropriate.<|im_end|>\n<|im_start|>user\n";
constexpr const char* kQieSuffix = "<|im_end|>\n<|im_start|>assistant\n";
constexpr int kQieDropPrefix = 64;

// Mage-Flow reuses BOTH templates verbatim -- its models/utils.py
// PROMPT_TEMPLATE["mage-flow"] is byte-identical to kPrefix + "{}" + kSuffix
// (start_idx 34 == kDropPrefix) and PROMPT_TEMPLATE["mage-flow-edit"] to
// kQiePrefix + "{}" + kQieSuffix (start_idx 64 == kQieDropPrefix); both
// descend from the same Qwen-Image conventions. Only the multi-reference body
// is its own: `Image {j}: <|vision_start|><|image_pad|><|vision_end|>` per
// reference (no separator), then the instruction -- pipeline.py
// _edit_prompt_body. Qwen-Image-Edit says "Picture {j}: " instead, and Boogu
// uses bare unlabelled blocks; ref_blocks_() below renders all three.
// Boogu-Image's two system prompts, verbatim from BooguImagePipeline
// (SYSTEM_PROMPT_4_T2I_UNIFIED / SYSTEM_PROMPT_4_TI2I_UNIFIED -- the latter is
// byte-identical to the Qwen-Image-Edit one above). The pipeline picks between
// them by whether an input image is present. It renders through the stock
// Qwen3-VL chat template WITHOUT a generation prompt (no trailing assistant
// turn), and -- unlike every other family here -- DROPS NOTHING: the whole
// templated sequence, system prompt included, is the conditioning the DiT's
// context_refiner sees.
constexpr const char* kBooguT2I =
    "<|im_start|>system\nYou are a helpful assistant that generates "
    "high-quality images based on user instructions. The instructions are as "
    "follows.<|im_end|>\n<|im_start|>user\n";
constexpr const char* kBooguTi2i =
    "<|im_start|>system\nDescribe the key features of the input image (color, "
    "shape, size, texture, objects, background), then explain how the user's "
    "text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the "
    "original input where appropriate.<|im_end|>\n<|im_start|>user\n";
constexpr const char* kBooguSuffix = "<|im_end|>\n";

// ---- multi-reference helpers -------------------------------------------
// One vision block per reference, with the family's own label convention:
// Qwen-Image-Edit says "Picture N: ", Mage-Flow "Image N: ", Boogu uses bare
// back-to-back blocks (verified against the reference's rendered template).
// `label` nullptr/empty => unlabelled.
std::string
ref_blocks_(int n_ref, const char* label)
{
  static const char* kBlock = "<|vision_start|><|image_pad|><|vision_end|>";
  std::string out;
  for (int i = 0; i < n_ref; ++i) {
    if (label != nullptr && *label != '\0') {
      out += fmt("{} {}: ", label, i + 1)();
    }
    out += kBlock;
  }
  return out;
}

// Expand each <|image_pad|> placeholder -- one per reference, in prompt order --
// to THAT reference's own vision-token count, and report the resulting image
// token RUNS as {row0, rows}. With several references the runs are DISJOINT
// (the labels sit between the blocks), which is why the callers cannot assume
// one contiguous image span.
std::vector<std::pair<int, int>>
expand_pads_(std::vector<std::int32_t>& ids, std::int32_t pad_id,
             const int* tok, int n_ref)
{
  std::vector<std::pair<int, int>> runs;
  std::vector<std::int32_t> out;
  out.reserve(ids.size() + 64);
  int k = 0;
  for (const std::int32_t id : ids) {
    if (id == pad_id && k < n_ref) {
      const int row0 = (int)out.size();
      const int cnt = tok[k] > 0 ? tok[k] : 0;
      for (int j = 0; j < cnt; ++j) { out.push_back(pad_id); }
      runs.push_back({row0, cnt});
      ++k;
    } else {
      out.push_back(id);
    }
  }
  ids.swap(out);
  return runs;
}

// 3-axis mROPE position_ids [3*n] for a text+vision sequence: text advances
// sequentially, each image run gets its OWN 2-D band (t=base, h=base+row,
// w=base+col) and the following text resumes at base + max(mh, mw). Exactly the
// stock Qwen3-VL / Qwen2.5-VL rule, applied per reference.
std::vector<std::int32_t>
mrope_positions_(int n, const std::vector<std::pair<int, int>>& runs,
                 const int* mh, const int* mw)
{
  std::vector<std::int32_t> pos((std::size_t)3 * n, 0);
  int cur = 0;
  std::size_t nr = 0;
  for (int i = 0; i < n;) {
    if (nr < runs.size() && i == runs[nr].first && runs[nr].second > 0) {
      const int base = cur;
      const int w = mw[nr] > 0 ? mw[nr] : 1;
      for (int j = 0; j < runs[nr].second && i < n; ++j, ++i) {
        pos[(std::size_t)i] = base;
        pos[(std::size_t)n + i] = base + j / w;
        pos[(std::size_t)2 * n + i] = base + j % w;
      }
      cur = base + std::max(mh[nr] > 0 ? mh[nr] : 1, w);
      ++nr;
    } else {
      pos[(std::size_t)i] = cur;
      pos[(std::size_t)n + i] = cur;
      pos[(std::size_t)2 * n + i] = cur;
      ++cur; ++i;
    }
  }
  return pos;
}

// Special-token-aware encode: split at the markers, BPE each text run, splice
// the markers' ids (matches the HF fast tokenizer). Includes the Qwen2.5-VL
// vision markers so the QIE image-aware template isolates them.
std::vector<std::int32_t>
encode_with_specials_(const genai::Tokenizer& tok, const std::string& text)
{
  static const char* kMarkers[] = {"<|im_start|>", "<|im_end|>",
                                   "<|vision_start|>", "<|vision_end|>",
                                   "<|image_pad|>", "<think>", "</think>"};
  static const int kN = (int)(sizeof(kMarkers) / sizeof(kMarkers[0]));
  std::vector<std::int32_t> out;
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t best = std::string::npos;
    int which = -1;
    for (int mi = 0; mi < kN; ++mi) {
      const std::size_t f = text.find(kMarkers[mi], pos);
      if (f != std::string::npos && (best == std::string::npos || f < best)) {
        best = f; which = mi;
      }
    }
    if (which < 0) {
      const auto seg = tok.encode(text.substr(pos));
      out.insert(out.end(), seg.begin(), seg.end());
      break;
    }
    if (best > pos) {
      const auto seg = tok.encode(text.substr(pos, best - pos));
      out.insert(out.end(), seg.begin(), seg.end());
    }
    const std::int32_t sid = tok.special_token_id(kMarkers[which]);
    if (sid >= 0) { out.push_back(sid); }
    pos = best + std::strlen(kMarkers[which]);
  }
  return out;
}

// ---- Encoder configs (per family) ----
genai::MetalQwenModel::Config encoder_config_krea2_()
{
  genai::MetalQwenModel::Config c;
  c.n_layers = 36; c.hidden = 2560; c.n_heads = 32; c.n_kv_heads = 8;
  c.head_dim = 128; c.ffn_inner = 9728; c.vocab = 151936; c.rope_theta = 5.0e6f;
  c.rms_eps = 1e-6f; c.rotary_dim = 128; c.full_attn_interval = 1;
  c.tie_embeddings = true; c.use_bf16 = true; c.dense = true;
  c.zero_centered_norm = false; c.attn_output_gate = false;
  c.backbone_only = true; c.weight_prefix = "language_model."; c.model_seg = "";
  c.max_seq = 1024; c.page_tokens = 256;
  return c;
}
genai::MetalQwenModel::Config encoder_config_flux2_(const std::string& enc_dir)
{
  genai::MetalQwenModel::Config c = encoder_config_krea2_();
  c.rope_theta = 1.0e6f; c.weight_prefix = "model.";
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(enc_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      auto geti = [&](const char* k, int cur) {
        return o.contains(k) ? (int)o.at(k).as_int(cur) : cur; };
      auto getf = [&](const char* k, float cur) {
        return o.contains(k) ? (float)o.at(k).as_real(cur) : cur; };
      c.n_layers = geti("num_hidden_layers", c.n_layers);
      c.hidden = geti("hidden_size", c.hidden);
      c.n_heads = geti("num_attention_heads", c.n_heads);
      c.n_kv_heads = geti("num_key_value_heads", c.n_kv_heads);
      c.head_dim = geti("head_dim",
                        c.n_heads > 0 ? c.hidden / c.n_heads : c.head_dim);
      c.rotary_dim = c.head_dim;
      c.ffn_inner = geti("intermediate_size", c.ffn_inner);
      c.vocab = geti("vocab_size", c.vocab);
      c.rope_theta = getf("rope_theta", c.rope_theta);
      c.rms_eps = getf("rms_norm_eps", c.rms_eps);
      if (o.contains("tie_word_embeddings")) {
        c.tie_embeddings = o.at("tie_word_embeddings").as_bool(c.tie_embeddings);
      }
    }
  }
  return c;
}
// Mage-Flow's text encoder IS the same Qwen3-VL 4B krea2 drives; the only
// difference is the checkpoint layout -- Mage-Flow wraps everything in
// `model.` (398 LM tensors under "model.language_model.", 315 tower tensors
// under "model.visual."), krea2 omits that wrapper.
genai::MetalQwenModel::Config encoder_config_mage_()
{
  genai::MetalQwenModel::Config c = encoder_config_krea2_();
  c.weight_prefix = "model.language_model.";
  // NOT backbone-only, unlike every other diffusion text encoder here: the
  // MANDATORY content screen (mage-screen.h) GENERATES a JSON verdict on
  // these same weights, which needs the token-embedding muxer + the (tied)
  // lm_head. The conditioning path takes its embeddings from the same muxer,
  // so this binds one embed table rather than the stage keeping a second copy
  // beside the model's (~780 MB at vocab 151936 x 2560 bf16).
  c.backbone_only = false;
  // The classifier's system prompt is the policy itself -- a few thousand
  // tokens, far past the ~100-token conditioning prompts the other families
  // size for. This is only a page-pool CAP (pages are allocated lazily and
  // returned on release), not a resident allocation.
  c.max_seq = 8192;
  return c;
}
// Boogu-Image's mllm is a stock Qwen3VLForConditionalGeneration -- the 10B
// ships an 8B Qwen3-VL (36L, hidden 4096, 32q/8kv, rope theta 5e6, UNTIED
// embeddings) wrapped as `model.language_model.` / `model.visual.` like
// Mage-Flow. Sized from mllm/config.json's text_config so one path serves any
// Boogu size.
genai::MetalQwenModel::Config encoder_config_boogu_(const std::string& enc_dir)
{
  genai::MetalQwenModel::Config c = encoder_config_krea2_();
  c.n_layers = 36; c.hidden = 4096; c.n_heads = 32; c.n_kv_heads = 8;
  c.head_dim = 128; c.rotary_dim = 128; c.ffn_inner = 12288;
  c.rope_theta = 5.0e6f; c.tie_embeddings = false;
  c.weight_prefix = "model.language_model.";
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(enc_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto root = fd.as_object();
      if (root.contains("text_config")) {
        FlexData tc = root.at("text_config");
        if (tc.is_object()) {
          auto o = tc.as_object();
          auto geti = [&](const char* k, int cur) {
            return o.contains(k) ? (int)o.at(k).as_int(cur) : cur; };
          auto getf = [&](const char* k, float cur) {
            return o.contains(k) ? (float)o.at(k).as_real(cur) : cur; };
          c.n_layers = geti("num_hidden_layers", c.n_layers);
          c.hidden = geti("hidden_size", c.hidden);
          c.n_heads = geti("num_attention_heads", c.n_heads);
          c.n_kv_heads = geti("num_key_value_heads", c.n_kv_heads);
          c.head_dim = geti("head_dim",
                            c.n_heads > 0 ? c.hidden / c.n_heads : c.head_dim);
          c.rotary_dim = c.head_dim;
          c.ffn_inner = geti("intermediate_size", c.ffn_inner);
          c.vocab = geti("vocab_size", c.vocab);
          c.rope_theta = getf("rope_theta", c.rope_theta);
          c.rms_eps = getf("rms_norm_eps", c.rms_eps);
        }
      }
      if (root.contains("tie_word_embeddings")) {
        c.tie_embeddings =
            root.at("tie_word_embeddings").as_bool(c.tie_embeddings);
      }
    }
  }
  return c;
}
genai::MetalQwenModel::Config encoder_config_qie_()
{
  genai::MetalQwenModel::Config c;
  c.n_layers = 28; c.hidden = 3584; c.n_heads = 28; c.n_kv_heads = 4;
  c.head_dim = 128; c.ffn_inner = 18944; c.vocab = 152064; c.rope_theta = 1.0e6f;
  c.rms_eps = 1e-6f; c.rotary_dim = 128; c.full_attn_interval = 1;
  c.tie_embeddings = false; c.use_bf16 = true; c.dense = true;
  c.zero_centered_norm = false; c.attn_output_gate = false;
  c.qk_norm = false; c.attention_bias = true; c.backbone_only = true;
  c.weight_prefix = ""; c.model_seg = "model."; c.max_seq = 1024;
  c.page_tokens = 256;
  return c;
}

// The transformer family from <root>/transformer/config.json `_class_name`.
std::string family_(const std::string& transformer_dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(transformer_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto obj = fd.as_object();
      if (obj.contains("_class_name")) {
        const std::string cls(obj.at("_class_name").as_string(""));
        if (cls == "Flux2Transformer2DModel") { return "flux2"; }
        if (cls == "QwenImageTransformer2DModel") { return "qwen-image-edit"; }
        // Mage-Flow (microsoft/Mage-Flow*). Named EXPLICITLY, never left to
        // the "krea2" default: its text encoder is the same Qwen3-VL 4B krea2
        // drives, so an unrecognized repo would LOAD and silently produce
        // krea2's 12-tap conditioning instead of Mage-Flow's single
        // last-hidden tap (and with the wrong weight prefix).
        if (cls == "MageFlow") { return "mage-flow"; }
        // Boogu-Image. The t2i and edit repos ship the SAME transformer config
        // (only the weights differ), so there is one family string; the edit
        // path turns on when a reference image is wired, exactly as Mage-Flow
        // switches templates.
        if (cls == "BooguImageTransformer2DModel") { return "boogu-image"; }
        // Wan video. Its tower is a umT5-XXL ENCODER rather than a
        // decoder-only LM, so this must be named explicitly -- falling
        // through to the "krea2" default would try to load a Qwen3-VL out
        // of a T5 checkpoint and fail late and confusingly.
        if (cls == "WanTransformer3DModel") { return "wan"; }
        // MiniMax-H3. Its tower IS a Qwen3-VL, but tapped at layer 50 of
        // 64 rather than run to the last hidden state, so it cannot go
        // through the shared Qwen encoder path -- that would condition
        // the DiT on a tensor 14 layers further along than the one it
        // was trained against.
        if (cls == "MiniMaxH3DiTModel") { return "minimax-h3"; }
      }
    }
  }
  return "krea2";
}

// Extract prompt text from a FlexData beat (string or {text: ...}).
std::string flex_text_(const FlexData& fd)
{
  if (fd.is_string()) { return std::string(fd.as_string("")); }
  if (fd.is_object()) {
    auto o = fd.as_object();
    if (o.contains("text")) { return std::string(o.at("text").as_string("")); }
  }
  return "";
}

#endif  // VPIPE_BUILD_APPLE_SILICON

}  // namespace

DiffusionConditionerStage::DiffusionConditionerStage(
    const SessionContextIntf* s, std::string id, std::vector<InEdge> iports,
    FlexData config)
  : TypedStage<DiffusionConditionerStage>(s, std::move(id), std::move(iports),
                                          std::move(config))
{
  // hf_dir is OPTIONAL: a model-select source on the model iport can supply it
  // instead, so "no model at all" is reported at initialize()/process() time
  // (when iport connectivity is known), not at construction.
  _hf_dir    = attr_str("hf_dir");
  _grounded_negative = attr_bool("grounded_negative");
#ifdef VPIPE_BUILD_APPLE_SILICON
  {
    bool bad = false;
    _unload_cfg = model_memory::parse_unload_policy(
        attr_str("unload_when_idle"), &bad);
    if (bad) {
      // Deferred-validated config: warn and take the default rather than throw.
      session()->warn(fmt(
          "DiffusionConditionerStage('{}'): unload_when_idle '{}' is not "
          "auto|always|never; using auto", this->id(),
          attr_str("unload_when_idle")));
    }
  }
#endif
  allocate_oports(spec().oports.size());
}

DiffusionConditionerStage::~DiffusionConditionerStage() = default;

const StageSpec&
DiffusionConditionerStage::spec() const noexcept
{
  return kSpec;
}

void
DiffusionConditionerStage::apply_constant(unsigned iport, const FlexData& beat)
{
  // Pre-launch twin of the runtime latch in process(): the same
  // beat and the same parse, early enough that declare_resources()
  // sees the model. Bookkeeping only -- nothing loads here; the
  // pipeline is not assembled yet (see Stage::apply_constant).
  if (iport != kModelPort) { return; }
  apply_model_select_beat(beat, _hf_dir);
}

#ifndef VPIPE_BUILD_APPLE_SILICON
Job DiffusionConditionerStage::initialize(RuntimeContext&) { co_return; }
Job DiffusionConditionerStage::process(RuntimeContext&) { co_return; }
#else

bool
DiffusionConditionerStage::load_encoder_(metal_compute::MetalCompute* mc)
{
  if (_family == "wan") {
    genai::MetalUmt5Encoder::Config ucfg;
    std::string uerr;
    if (!genai::MetalUmt5Encoder::config_from_json(_enc_dir, ucfg, &uerr)) {
      session()->error(fmt("DiffusionConditionerStage('{}'): {}",
                           this->id(), uerr));
      return false;
    }
    _enc_hidden = ucfg.d_model;
    _enc_ws = genai::open_weight_set(_enc_dir, session());
    if (!_enc_ws) {
      session()->error(fmt("DiffusionConditionerStage('{}'): cannot open text "
                           "encoder checkpoint: {}", this->id(), _enc_dir));
      return false;
    }
    _umt5 = genai::MetalUmt5Encoder::load(_enc_ws, mc, ucfg);
    if (!_umt5) {
      session()->error(fmt("DiffusionConditionerStage('{}'): umT5 encoder load "
                           "failed: {}", this->id(), _enc_dir));
      return false;
    }
    // No separate embedding table to cache: umT5 gathers from its own
    // `shared.weight`, which the encoder already holds.
    return true;
  }
  if (_family == "minimax-h3") {
    genai::MiniMaxH3TextEncoder::Config h3cfg;
    std::string h3err;
    if (!genai::MiniMaxH3TextEncoder::config_from_json(_enc_dir, h3cfg,
                                                       &h3err)) {
      session()->error(fmt("DiffusionConditionerStage('{}'): {}", this->id(),
                           h3err));
      return false;
    }
    _enc_hidden = h3cfg.text_dim;
    // The Qwen3-VL-32B backbone is ~48 GB bf16 (~26 GB at w8) for the 50
    // tapped layers, so on a small box it cannot be resident -- and it
    // only ever PREFILLS, which is exactly what layer streaming serves.
    // Same rule the DiTs use, with the encoder as the streamed component
    // and everything else the graph declared as what stays resident.
    {
      // Size the COMPONENT, not what the config happened to name: a
      // Comfy-Org repo root holds the 66 GB DiT beside the encoder, and
      // summing the tree would decide to stream on the DiT's bytes.
      const std::string ed =
          genai::MiniMaxH3TextEncoder::resolve_encoder_dir(_enc_dir);
      const auto plan = model_memory::plan_streaming(
          session(), ed, std::string(), model_memory::kStreamHeadroom);
      h3cfg.lm.stream_layers = plan.stream;
      h3cfg.lm.pin_frac      = plan.pin_frac;
      if (const char* e = std::getenv("VPIPE_H3_ENC_STREAM")) {
        h3cfg.lm.stream_layers = (std::atoi(e) != 0);
        if (!h3cfg.lm.stream_layers) { h3cfg.lm.pin_frac = 0.0; }
      }
      if (const char* e = std::getenv("VPIPE_H3_ENC_PIN_FRAC")) {
        h3cfg.lm.pin_frac = std::atof(e);
      }
      session()->log_debug(fmt(
          "DiffusionConditionerStage('{}'): MiniMax-H3 encoder footprint "
          "{} GB (others {} GB) + {} GB headroom -> {}", this->id(),
          plan.footprint >> 30, plan.others >> 30,
          model_memory::kStreamHeadroom >> 30,
          h3cfg.lm.stream_layers ? "STREAM layers" : "PRELOAD"));
    }
    _h3_enc = genai::MiniMaxH3TextEncoder::load(_enc_dir, mc,
                                                const_cast<SessionContextIntf*>(
                                                    session()), h3cfg);
    if (!_h3_enc) {
      session()->error(fmt("DiffusionConditionerStage('{}'): MiniMax-H3 text "
                           "encoder load failed: {}", this->id(), _enc_dir));
      return false;
    }
    if (_h3_enc->streaming()) {
      session()->info(fmt(
          "DiffusionConditionerStage('{}'): MiniMax-H3 encoder streams its "
          "layers ({} of {} pinned) -- ~one layer resident instead of the "
          "stack, at one command buffer per layer per prompt",
          this->id(), _h3_enc->pinned_layers(), h3cfg.tap));
      // Deliberately NOT revising the manager declaration down: unlike a
      // DiT, an LM reads its tensors UNCACHED into its own buffers, so the
      // weight set's stats report a fraction of what the model holds and
      // would revise to a number that is wrong in the unsafe direction.
    }
    // The encoder owns its tokenizer and embedding table; nothing else
    // to bind here.
    return true;
  }
  genai::MetalQwenModel::Config ecfg =
      _family == "flux2" ? encoder_config_flux2_(_enc_dir)
      : _family == "qwen-image-edit" ? encoder_config_qie_()
      : _family == "mage-flow" ? encoder_config_mage_()
      : _family == "boogu-image" ? encoder_config_boogu_(_enc_dir)
      : encoder_config_krea2_();
  _enc_hidden = ecfg.hidden;
  // The encoder may be affine-quantized (model-quantize target=text_encoder).
  // The loader auto-detects quantized-vs-dense weights but needs the bit-width
  // to pick the w4g64 vs w8g64 kernel, so read it from the encoder's
  // config.json quantization block (absent => dense bf16, quant_bits unused).
  {
    std::ifstream in(std::filesystem::path(_enc_dir) / "config.json");
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto o = fd.as_object();
        if (o.contains("quantization")) {
          FlexData q = o.at("quantization");
          if (q.is_object()) {
            auto qo = q.as_object();
            const int b = qo.contains("bits") ? (int)qo.at("bits").as_int(0) : 0;
            if (b == 4 || b == 8) { ecfg.quant_bits = b; }
          }
        }
      }
    }
  }
  // LAYER STREAMING, on the same rule the DiTs use.
  //
  // declare_resources() puts this encoder on the books with a FLOOR --
  // what it holds if it streams -- and that floor is what the resource
  // phase judges the graph against. Until this decision existed the
  // floor was a promise nothing kept: the encoder loaded resident
  // whatever the box, so a graph could be admitted on the strength of a
  // reduction that never happened. This is the other half of that
  // claim.
  //
  // BACKBONE-ONLY ONLY, which is not a limitation so much as the
  // mechanism's definition: a streamed layer is built inside the
  // PREFILL and freed after it, so decode would re-read the stack per
  // token. Every conditioning encoder here taps hidden states and never
  // decodes -- except mage-flow, whose content screen generates a
  // verdict through the lm_head. MetalQwenModel::load refuses the
  // combination anyway; asking only when it can be served keeps a
  // warning off a path that is behaving correctly.
  if (ecfg.backbone_only) {
    const auto plan = model_memory::plan_streaming(
        session(), _enc_dir, std::string(), model_memory::kStreamHeadroom);
    ecfg.stream_layers = plan.stream;
    ecfg.pin_frac      = plan.pin_frac;
    if (const char* e = std::getenv("VPIPE_ENC_STREAM")) {
      ecfg.stream_layers = (std::atoi(e) != 0);
      if (!ecfg.stream_layers) { ecfg.pin_frac = 0.0; }
    }
    session()->log_debug(fmt(
        "DiffusionConditionerStage('{}'): text encoder footprint {} MB "
        "(others {} MB) + {} MB headroom -> {}", this->id(),
        plan.footprint >> 20, plan.others >> 20,
        model_memory::kStreamHeadroom >> 20,
        ecfg.stream_layers ? "STREAM layers" : "PRELOAD"));
  }
  // One set for this checkpoint, shared with the vision tower and the
  // embedding table below and with anything else naming the same dir.
  _enc_ws = genai::open_weight_set(_enc_dir, session());
  if (!_enc_ws) {
    session()->error(fmt("DiffusionConditionerStage('{}'): cannot open text "
                         "encoder checkpoint: {}", this->id(), _enc_dir));
    return false;
  }
  _encoder = genai::MetalQwenModel::load(_enc_ws, mc, ecfg);
  if (!_encoder) {
    session()->error(fmt("DiffusionConditionerStage('{}'): text encoder load "
                         "failed: {}", this->id(), _enc_dir));
    return false;
  }
  if (_encoder->streaming_layers()) {
    session()->info(fmt(
        "DiffusionConditionerStage('{}'): text encoder streams its layers -- "
        "~one layer resident instead of the stack, rebuilt inside each "
        "prefill", this->id()));
    // Deliberately NOT revising the manager declaration down. An LM reads
    // its tensors UNCACHED into its own buffers, so the weight set's
    // stats report a fraction of what the model holds and a revision
    // would move the declared figure in the unsafe direction. The
    // streamable claim declare_resources() made already says what this
    // reduces to; there is nothing to correct.
  }
  if (_family == "mage-flow") {
    // Mage-Flow takes its embeddings from the model's own muxer (loaded
    // because the content screen has to generate), so there is no second
    // table to load here. Probe it once: an encoder that cannot gather a
    // token embedding can neither condition NOR screen, and a mage-flow
    // encoder that cannot screen must not run at all.
    if (_encoder->embed_text_buf(std::vector<std::int32_t>{0}).empty()) {
      session()->error(fmt(
          "DiffusionConditionerStage('{}'): Mage-Flow encoder has no usable "
          "token-embedding table -- it could neither condition nor run the "
          "mandatory content screen; inert", this->id()));
      return false;
    }
    return true;
  }
  const std::string emb_name =
      (_family == "flux2" || _family == "qwen-image-edit")
          ? "model.embed_tokens.weight"
      : (_family == "boogu-image")
          ? "model.language_model.embed_tokens.weight"
          : "language_model.embed_tokens.weight";
  _embed = _enc_ws->tensor(emb_name, mc,
                           genai::WeightSet::Residency::Copied);
  return !_embed.empty();
}

void
DiffusionConditionerStage::reset_run_state()
{
  // Per-launch reset: the stage survives a stop/relaunch, and the
  // select sources upstream re-emit on every launch. Without this the
  // re-emitted beat is never latched and this stage keeps the previous
  // run's selection.
  _model_latched    = false;
  _cfg_latched      = false;
  _model_cfg        = FlexData{};
  _negative_latched = false;
  // Re-decided next launch: peers may differ.
  _unload_resolved  = false;
  // Same for the cached reference images: `_ref_rgb[i]` non-empty makes
  // the iport3/iport4 read conditional, so a relaunch would never
  // consume the new reference and would re-encode the previous run's
  // picture instead.
  for (int i = 0; i < kMaxRefs; ++i) {
    _ref_rgb[i].clear();
    _ref_rgb_h[i] = 0;
    _ref_rgb_w[i] = 0;
  }
  _n_ref = 0;

}

// The encoder and DiT directories this stage's model resolves to.
//
// Shared by both planning passes so they cannot disagree about which
// checkpoint is being talked about: the declare pass puts them on the
// books and the decide pass refines one of them, keyed by path.
void
DiffusionConditionerStage::resolve_component_dirs_(std::string* enc_out,
                                                   std::string* dit_out) const
{
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  // Boogu names its text encoder `mllm/`; every other family uses
  // text_encoder/. Detected from the filesystem rather than by parsing
  // the family, so this stays a cheap pre-init query -- and declaring a
  // directory that turns out to be empty is harmless (0 bytes).
  const std::string mllm = (fs::path(root) / "mllm").string();
  std::error_code ec;
  std::string enc = fs::exists(mllm, ec)
                        ? mllm
                        : (fs::path(root) / "text_encoder").string();
  std::string dit = (fs::path(root) / "transformer").string();
  // A Comfy-Org repack spells neither of those. Resolving both here is not
  // cosmetic: this claim is what the resource-planning phase sizes every
  // peer against, and a component that resolves to nothing is declared as
  // 0 bytes -- the silent under-count the phase exists to prevent.
  if (!fs::exists(enc, ec)) {
    const std::string e =
        genai::MiniMaxH3TextEncoder::resolve_encoder_dir(root);
    if (!e.empty() && e != root) { enc = e; }
  }
  if (!fs::exists(dit, ec)) {
    const std::string d =
        genai::MetalMiniMaxH3Transformer::resolve_dit_dir(root);
    if (!d.empty() && d != root) { dit = d; }
  }
  if (enc_out != nullptr) { *enc_out = std::move(enc); }
  if (dit_out != nullptr) { *dit_out = std::move(dit); }
}

std::vector<ResourceClaim>
DiffusionConditionerStage::declare_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  std::string enc, dit;
  resolve_component_dirs_(&enc, &dit);
  // Both, unconditionally. Whether the encoder is phase-limited is a
  // DECISION and belongs in decide_resources() below -- taken here it
  // would read a half-built picture, and a graph would size itself
  // differently depending on where the flattener put this stage.
  //
  // The encoder's FLOOR, though, is not a decision: it is a fact about
  // the checkpoint -- what it holds if it streams its layers -- and it
  // is the term that decides the peak of a generation graph, because the
  // conditioning phase carries the encoder at whatever size it is
  // counted at. A 48 GB encoder with no floor makes every such graph
  // report a requirement it does not have.
  std::vector<ResourceClaim> out;
  const std::size_t enc_floor =
      enc.empty() ? 0
                  : genai::MiniMaxH3TextEncoder::streaming_floor_bytes(enc);
  if (!enc.empty()) {
    out.push_back(model_memory::weight_claim_streamable(enc, enc_floor));
  }
  if (!dit.empty()) {
    for (auto& c : model_memory::weight_claims({dit})) {
      out.push_back(std::move(c));
    }
  }
  return out;
}

StageMemory
DiffusionConditionerStage::declare_memory() const
{
  StageMemory m;
  if (_hf_dir.empty()) { return m; }
  std::string enc, dit;
  resolve_component_dirs_(&enc, &dit);
  if (enc.empty()) { return m; }
  m.hold(enc, model_memory::dir_weights_bytes(enc),
         genai::MiniMaxH3TextEncoder::streaming_floor_bytes(enc),
         // `destroy` only, for the reason decide_resources gives.
         _unload_cfg == model_memory::UnloadPolicy::kDestroy,
         _unload_cfg == model_memory::UnloadPolicy::kAuto);
  // The conditioning it emits is deliberately NOT declared here.
  //
  // Its shape is built per beat from the prompt -- rows are tokens, and
  // an image reference adds its own -- so there is no honest plan-time
  // figure, and it is ~10 MB at any plausible caption length against a
  // peak measured in gigabytes. Declaring a guess would put a number in
  // the plan that nothing checks; leaving it out leaves the plan short
  // by an amount smaller than its own rounding. The revise path is
  // where this belongs once declare_memory has one.
  return m;
}

std::vector<ResourceClaim>
DiffusionConditionerStage::decide_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  std::string enc, dit;
  resolve_component_dirs_(&enc, &dit);
  if (enc.empty()) { return {}; }

  // The ENCODER is claimed for the CONDITION phase when this stage will
  // certainly let go of it, and left persistent when it might not.
  //
  // Why it has to be said in the PLAN rather than at unload time: a DiT
  // takes its block-streaming decision, which is irreversible, on the
  // first conditioning beat. That is after this stage has produced its
  // output and before it has dropped the encoder, so an announcement at
  // unload arrives one decision too late. MEASURED on a 64 GB box with
  // the bf16 MiniMax-H3: the DiT streamed against a footprint of 57 GB,
  // of which 20 GB was an encoder that no longer existed by the first
  // denoise step, and the verdict turned on 1 GB.
  //
  // ONLY `destroy` counts, and `auto` only when it will resolve to
  // destroy. `park` releases NOTHING here: park_weights() walks a weight
  // set's CACHED entries, and every text encoder in this stage reads
  // uncached into the model's own members -- so it parks 0 bytes and the
  // encoder stays entirely resident. A peer that subtracted it would be
  // short by the encoder's whole size and would preload into a box that
  // cannot hold it.
  //
  // bounded() is unphased on purpose, and by this point it is also
  // COMPLETE: every stage has declared, and no refinement from this pass
  // has been applied yet, so every stage deciding gets the same answer.
  const bool releases =
      _unload_cfg == model_memory::UnloadPolicy::kDestroy ||
      (_unload_cfg == model_memory::UnloadPolicy::kAuto &&
       model_memory::bounded(session(), {enc, dit},
                             model_memory::kHeadroom));
  if (!releases) { return {}; }
  return model_memory::weight_claims_in_phase(
      {enc}, model_memory::kPhaseCondition);
}

Job
DiffusionConditionerStage::initialize(RuntimeContext& ctx)
{
  // If a previous run left the weights UNLOADED (the idle-unload
  // policy drops them between beats), let this launch load them again:
  // ensure_loaded_'s once-only guard is per-Stage, not per-launch, so
  // without this the stage stays inert for the whole run. When the
  // weights are still held we deliberately leave the guard set --
  // reloading on top of a resident copy is exactly what doubles peak
  // memory.
  if (!_encoder) {
    _load_attempted = false;
    _unloaded       = false;
  }

  // Defer the encoder load when a model-select source feeds the model iport
  // (its beat only arrives after the init barrier, in process()). Otherwise
  // load now from the config hf_dir, as before.
  const bool model_from_iport =
      ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort);
  if (!model_from_iport) { ensure_loaded_(); }
  co_return;
}

// Re-seed the grounded-encode parameters for the family that is now
// resident and overlay whatever the config beat set. Both directions of
// one question, run on every change to either, because a config beat and
// a model reference arrive on different ports and either can be first.
void
DiffusionConditionerStage::apply_model_config_()
{
  // The FAMILY's numbers first, ALWAYS -- not merged onto whatever was
  // there. A previous family's bounds surviving a checkpoint change is
  // the failure this ordering exists to prevent, and it would be
  // invisible: the encode succeeds either way, just at a resolution the
  // model was not trained against.
  _ground = genai::GroundedEncodeParams::for_family(_family);
  const std::string want = model_config::family_of(_model_cfg);
  if (!want.empty() && want != _family) {
    session()->warn(fmt(
        "DiffusionConditionerStage('{}'): the model_config beat is for the "
        "'{}' family but the resident checkpoint is '{}'; IGNORING it and "
        "using the family's own numbers. Wire the config stage that matches "
        "the checkpoint", this->id(), want, _family));
    return;
  }
  std::string perr;
  _ground.merge_flex(_model_cfg, &perr);
  if (!perr.empty() && !_model_cfg.is_null()) {
    session()->warn(fmt("DiffusionConditionerStage('{}'): model_config: {}",
                        this->id(), perr));
  }
}

void
DiffusionConditionerStage::ensure_loaded_()
{
  if (_load_attempted) { return; }   // idempotent: load at most once
  _load_attempted = true;
  if (_hf_dir.empty()) {
    session()->error(fmt("DiffusionConditionerStage('{}'): no model -- set "
        "config.hf_dir or wire a model-select source to the model iport; "
        "inert", this->id()));
    return;
  }
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) {
    session()->error(fmt(
        "DiffusionConditionerStage('{}'): no metal-compute backend; inert",
        this->id()));
    return;
  }
  const std::string root = resolve_model_dir(session(), _hf_dir);
  _family = family_((std::filesystem::path(root) / "transformer").string());
  // Seeded here and again below: the H3 probe can still change the
  // family, and `_ground` has to describe whichever one wins.
  apply_model_config_();
  // `transformer/` is the DIFFUSERS spelling, and MiniMax-H3 also ships as
  // a Comfy-Org repack whose DiT is one .safetensors under
  // `diffusion_models/` (or, after model-quantize, a directory checkpoint
  // there). On such a root the read above finds nothing and falls through
  // to the "krea2" default, which then looks for a Qwen3-VL 4B that is not
  // there. Ask H3's own reader, which resolves every layout and refuses a
  // checkpoint that is not H3.
  if (_family != "minimax-h3") {
    genai::MetalMiniMaxH3Transformer::Config probe;
    if (genai::MetalMiniMaxH3Transformer::config_from_json(root, probe,
                                                           nullptr)) {
      _family = "minimax-h3";
      apply_model_config_();
    }
  }
  // Boogu names its text encoder `mllm/` (it is a full multimodal LLM, not a
  // text_encoder in the diffusers sense); every other family uses
  // text_encoder/.
  _enc_dir = (std::filesystem::path(root) /
              (_family == "boogu-image" ? "mllm" : "text_encoder")).string();
  if (_family == "minimax-h3") {
    // `text_encoders/` on a repack, `text_encoder/` on a diffusers tree,
    // and a directory checkpoint in either once quantized -- one resolver
    // for all of them.
    const std::string enc =
        genai::MiniMaxH3TextEncoder::resolve_encoder_dir(root);
    if (!enc.empty() && enc != root) { _enc_dir = enc; }
  }
  // NOTE: the encoder's phase claim is made in declare_resources(),
  // which is where it belongs and where the planning phase can see it.
  // It used to be repeated here as well, because declare_resources()
  // returned nothing for a graph whose `hf_dir` arrives on the model
  // iport from `model-select` -- which is how the shipped pipelines are
  // written. Stage::apply_constant now delivers that choice before
  // planning, so the declaration is real and this second site is gone.
  // Do not restore it: two places stating one intent means only one of
  // them is exercised by any given graph shape.

  namespace fs = std::filesystem;
  std::string tok_path = (fs::path(root) / "tokenizer" / "tokenizer.json").string();
  if (!fs::exists(tok_path)) {
    tok_path = (fs::path(root) / "processor" / "tokenizer.json").string();
  }
  if (!fs::exists(tok_path)) {
    // Mage-Flow ships no separate tokenizer/ or processor/ dir -- the
    // Qwen3-VL tokenizer + processor configs live beside the weights.
    tok_path = (fs::path(_enc_dir) / "tokenizer.json").string();
  }
  _tokenizer = genai::Tokenizer::from_huggingface_json(tok_path, session());
  if (!_tokenizer) {
    session()->error(fmt(
        "DiffusionConditionerStage('{}'): tokenizer load failed: {}; inert",
        this->id(), tok_path));
    return;
  }
  if (!load_encoder_(mc)) {
    session()->error(fmt(
        "DiffusionConditionerStage('{}'): encoder/embeds load failed; inert",
        this->id()));
    return;
  }
  session()->info(fmt(
      "DiffusionConditionerStage('{}'): family {} encoder ({}), hidden {}",
      this->id(), _family,
      _family == "flux2" ? "Qwen3 dense"
      : _family == "qwen-image-edit" ? "Qwen2.5-VL"
      : _family == "boogu-image" ? "Qwen3-VL (mllm)"
      : _family == "wan" ? "umT5-XXL"
      : _family == "minimax-h3" ? "Qwen3-VL-32B (layer-50 tap)" : "Qwen3-VL",
      _enc_hidden));

  // The component dirs the idle-unload decision sizes against. The
  // DECISION itself is deferred to the first process() -- see
  // resolve_unload_policy_().
  _root_dir  = root;
  // `transformer/` is the diffusers spelling. A Comfy-Org repack keeps its
  // DiT under `diffusion_models/`, so naming only the former left the DiT
  // contributing ZERO to the footprint this decision sizes against -- and
  // a decision that cannot see the largest peer in the graph always says
  // there is room.
  std::string dit = (std::filesystem::path(root) / "transformer").string();
  {
    std::error_code ec;
    if (!std::filesystem::exists(dit, ec)) {
      const std::string d =
          genai::MetalMiniMaxH3Transformer::resolve_dit_dir(root);
      if (!d.empty() && d != root) { dit = d; }
    }
  }
  _peer_dirs = {_enc_dir, dit};
}

// Decided at the FIRST process(), not at load, because process() runs
// strictly after the init barrier -- by which point every stage in the
// graph has finished loading. Two things are only true then:
//
//   * every peer's weights exist, so nothing is missed. Declarations
//     (Stage::declare_resources) already cover peers this stage cannot see
//     from its own config, but a declaration is an ESTIMATE from the
//     files on disk.
//   * what each model is really holding is authoritative, and is often
//     far less. A streaming DiT holds only its pinned prefix, so sizing
//     against its full on-disk bytes would drop this stage's encoder
//     after every prompt to make room for weights that were never
//     resident.
//
// Unlike block streaming this decision is cheap to defer: it is a
// per-beat behaviour flag, not a constructor argument, so getting it
// right on the first beat costs nothing.
void
DiffusionConditionerStage::resolve_unload_policy_()
{
  if (_unload_resolved) { return; }
  _unload_resolved = true;
  switch (_unload_cfg) {
    case model_memory::UnloadPolicy::kDestroy:
    case model_memory::UnloadPolicy::kPark:
    case model_memory::UnloadPolicy::kKeep:
      _idle_action = _unload_cfg;
      break;
    default: {
      // Two independent reasons to let go, and they want DIFFERENT
      // answers.
      //
      // A peer that decided to stream is the harder of the two. It is
      // already paying per-step disk reads for want of RAM, and the
      // only thing that gives it room deterministically is DESTROYING
      // this encoder: parked bytes are still held bytes as far as
      // weight_footprint() and the manager's accounting are concerned,
      // and they only come back to the box if the kernel gets round to
      // reclaiming them. A DiT that has to decide, this step, whether to
      // keep a block cannot wait on that.
      //
      // Absent a streaming peer, the box is merely tight-ish, and PARK
      // is the better trade: the second prompt reuses the encoder for
      // free if the pages survived, and the kernel takes them if
      // something genuinely needs the RAM. Keeping it pinned is the one
      // answer that is never right here -- unreclaimable read-only
      // weights just push the pressure onto the compressor, which pays
      // to put them back.
      const bool streaming = model_memory::peer_streams(session());
      const bool tight     = model_memory::bounded(session(), _peer_dirs,
                                                   model_memory::kHeadroom);
      _idle_action = streaming ? model_memory::UnloadPolicy::kDestroy
                   : tight     ? model_memory::UnloadPolicy::kDestroy
                               : model_memory::UnloadPolicy::kPark;
      break;
    }
  }
  session()->log_debug(fmt(
      "DiffusionConditionerStage('{}'): encoder + DiT footprint {} MB + {} MB "
      "headroom vs {} MB RAM, a peer streams={}, unload_when_idle={} -> {}",
      this->id(),
      model_memory::weight_footprint(session(), _peer_dirs) >> 20,
      model_memory::kHeadroom >> 20, model_memory::phys_ram() >> 20,
      model_memory::peer_streams(session()) ? "yes" : "no",
      model_memory::unload_policy_name(_unload_cfg),
      model_memory::unload_policy_name(_idle_action)));
  if (_idle_action == model_memory::UnloadPolicy::kDestroy) {
    session()->info(fmt(
        "DiffusionConditionerStage('{}'): memory-bounded -- the encoder is "
        "dropped after each conditioning and reloaded on the next prompt",
        this->id()));
  } else if (_idle_action == model_memory::UnloadPolicy::kPark) {
    session()->info(fmt(
        "DiffusionConditionerStage('{}'): the encoder is parked after each "
        "conditioning -- reclaimable if the box needs the RAM, and reused "
        "without a reload if it does not", this->id()));
  }
}

// Park rather than destroy: the weights go purgeable and the next
// prompt reactivates them, re-reading from disk only if the kernel
// actually took the pages.
//
// Parking walks the weight SET's cached tensors, so what this reports
// is a property of the loader, not of the box: a model that reads its
// weights uncached (every LM does -- see WeightSet::read) holds them in
// its own members, which the registry cannot see, and parks 0. That is
// logged rather than hidden, because a park that frees nothing looks
// exactly like a park that worked until someone measures the box.
void
DiffusionConditionerStage::park_encoder_()
{
  if (_enc_dir.empty()) { return; }
  auto* mgr = session()->services()->generative_model_manager();
  if (mgr == nullptr) { return; }
  // THE BORROW HAS TO END FIRST, and that is why this drops the models
  // rather than parking around them. The manager owns the checkpoint and
  // refuses to park one anything is still borrowing -- it cannot know a
  // live model is idle, and a park under one hands its reader pages the
  // kernel may discard. This stage parking its own encoder while holding
  // it was exactly that case: the embedding table is a cached tensor,
  // held here across the park and read on the next prompt.
  //
  // So `park` and `destroy` differ in what the MANAGER does with the
  // bytes, not in what this stage does with its models. Both let go;
  // park keeps them (purgeable, reactivated on the next read), destroy
  // hands them back.
  unload_encoder_();
  const std::size_t got = mgr->park_weights(_enc_dir);
  session()->log_debug(fmt(
      "DiffusionConditionerStage('{}'): parked {} MB of the encoder{}",
      this->id(), got >> 20,
      got == 0 ? " -- nothing parkable; its loader reads uncached, so the "
                 "bytes live in the model, not the weight set"
               : ""));
}

void
DiffusionConditionerStage::release_encoder_when_idle_()
{
  switch (_idle_action) {
    case model_memory::UnloadPolicy::kDestroy: unload_encoder_(); break;
    case model_memory::UnloadPolicy::kPark:    park_encoder_();   break;
    default: break;                            // kKeep: hold it pinned
  }
}

void
DiffusionConditionerStage::unload_encoder_()
{
  if (!_encoder && !_umt5 && !_h3_enc) { return; }
  // Everything weight-sized: the LM (or the wan family's umT5 tower),
  // either vision tower, and the embedding table. The tokenizer stays
  // (kilobytes, and it is pure CPU state).
  // SETTLED when the policy is `auto`. The manager owns the checkpoint,
  // so dropping these models only ends this stage's borrow; what the
  // call does is ask for it to be settled now -- parked if nobody else
  // is borrowing it, left alone if someone is. The order against the
  // resets below no longer matters (it did when the manager's reference
  // was weak and there was nothing left to pool afterwards).
  //
  // The encoder is the checkpoint a relaunch most wants back -- it is
  // the largest thing this graph loads and is dropped after every
  // conditioning by design. `destroy` is the caller asking for the bytes
  // now, so it keeps dropping them.
  _encoder.reset();
  _umt5.reset();
  _h3_enc.reset();
  _vision.reset();
  _vision3.reset();
  _embed = metal_compute::SharedBuffer{};
  _ds_feats.clear();
  // AND THE BORROW ITSELF. Keeping `_enc_ws` across an unload used to be
  // harmless bookkeeping; with the manager owning the checkpoint it is
  // the difference between released and not, because a set this stage
  // still holds is one the manager will not park or drop. It is
  // reopened by load_encoder_() on the next prompt, which is where it
  // came from.
  _enc_ws.reset();
  _unloaded = true;
  // SETTLE IT NOW rather than at whatever unrelated moment the manager
  // is next doing work. `destroy` is the caller asking for the bytes
  // back, which parking does not do; anything else keeps the checkpoint
  // for the next prompt, purgeable.
  if (!_enc_dir.empty()) {
    if (auto* mgr = session()->services()->generative_model_manager()) {
      if (_unload_cfg == model_memory::UnloadPolicy::kDestroy) {
        mgr->drop_weights(_enc_dir);
      } else {
        mgr->pool_weights(_enc_dir);
      }
    }
  }

  // SAY SO to the manager, which is what makes this a memory decision
  // rather than a private one.
  //
  // A declaration PERSISTS for the run at max(held, estimate) -- on
  // purpose, so a peer never sizes against weights that are merely
  // between uses. Destroying is the case that rule cannot see: the bytes
  // are gone and will not come back until the next prompt, but the claim
  // still reads as this stage's full encoder, and every peer sizing
  // after this point adds it.
  //
  // That matters most for the one decision that cannot be revisited. A
  // DiT takes its block-streaming decision at construction, on the first
  // conditioning beat -- i.e. strictly AFTER this -- and it counts the
  // encoder as a coexisting peer. So an encoder this stage has already
  // released can push a DiT into streaming for RAM that is free, and
  // nothing later can undo it. Revising to 0 is how the encoder stops
  // being the DiT's constraint. Reloading needs no matching call: it
  // opens the weight set again, and the manager counts what it holds.
  if (!_enc_dir.empty() && session() != nullptr &&
      session()->services() != nullptr) {
    auto* mgr = session()->services()->generative_model_manager();
    if (mgr != nullptr) {
      mgr->revise_declaration(_enc_dir, 0);
      // And redeem the promise the phase claim made. A claim whose
      // release never arrives is warned about at the end of the run --
      // see GenerativeModelManager::clear_declarations -- because a DiT
      // has already spent an irreversible decision on it.
      mgr->note_phase_released(_enc_dir);
    }
  }
  session()->log_debug(fmt(
      "DiffusionConditionerStage('{}'): encoder unloaded (idle), declaration "
      "revised to 0", this->id()));
}

bool
DiffusionConditionerStage::reload_encoder_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _enc_dir.empty()) { return false; }
  if (!load_encoder_(mc)) {
    session()->error(fmt(
        "DiffusionConditionerStage('{}'): encoder reload failed: {}",
        this->id(), _enc_dir));
    return false;
  }
  _unloaded = false;
  session()->log_debug(fmt(
      "DiffusionConditionerStage('{}'): encoder reloaded for a new prompt",
      this->id()));
  return true;
}

// True for the families whose conditioning is a SINGLE post-final-norm
// last-hidden tap [n_real, hidden] (bf16), as opposed to krea2's 12-tap stack
// or flux2's 3-tap concat.
static bool
single_tap_(const std::string& family)
{
  return family == "qwen-image-edit" || family == "mage-flow" ||
         family == "boogu-image" || family == "wan" ||
         family == "minimax-h3";
}

SharedBuffer
DiffusionConditionerStage::vision_tokens_(metal_compute::MetalCompute* mc,
                                          int& n_img) const
{
  n_img = 0;
  _img_n = 0;
  for (int i = 0; i < kMaxRefs; ++i) {
    _img_mh[i] = 0; _img_mw[i] = 0; _img_tok[i] = 0;
  }
  if (_n_ref <= 0 || _ref_rgb[0].empty()) { return {}; }

  // Qwen-Image-Edit: Qwen2.5-VL tower -> bf16 [n_img_total, 3584]. QIE-2511 is
  // a MULTI-reference edit model, so every wired picture is encoded and the
  // rows are CONCATENATED in prompt order; encode_ then splices them over the
  // pad rows of the per-reference "Picture N: " blocks in that same order.
  // (Qwen2.5-VL has no deepstack, so multi-reference needs nothing else here.)
  if (_family == "qwen-image-edit") {
    if (!_vision) {
      genai::MetalQwen25Vision::Config vcfg;
      _vision = genai::MetalQwen25Vision::load(_enc_ws, mc, vcfg);
      if (!_vision) {
        session()->warn(fmt("DiffusionConditionerStage('{}'): vision tower load "
                            "failed; text-only conditioning", this->id()));
        return {};
      }
    }
    std::vector<SharedBuffer> per;
    int total = 0;
    const int mm = _vision->config().merge;
    for (int i = 0; i < _n_ref; ++i) {
      int vgh = 0, vgw = 0;
      // Qwen2.5-VL takes the pixel budget directly and smart-resizes from
      // it, so this family has no separate long-edge cap. A budget of 0
      // would mean "no image at all" rather than "unbounded", so the
      // family's own number stands in -- 0 here is a config mistake, not
      // a mode.
      const std::size_t budget = _ground.pixel_budget > 0
                                     ? _ground.pixel_budget
                                     : (std::size_t)384 * 384;
      SharedBuffer vt = _vision->encode_rgb(_ref_rgb[i].data(), _ref_rgb_h[i],
                                            _ref_rgb_w[i], budget, vgh, vgw);
      if (vt.empty()) { return {}; }
      const int tok = (vgh / mm) * (vgw / mm);
      _img_mh[i] = vgh / mm; _img_mw[i] = vgw / mm; _img_tok[i] = tok;
      total += tok;
      per.push_back(std::move(vt));
      session()->info(fmt(
          "DiffusionConditionerStage('{}'): reference {} -> {} vision tokens "
          "(grid {}x{})", this->id(), i, tok, vgh, vgw));
    }
    _img_n = (int)per.size();
    n_img = total;
    if (per.size() == 1) { return std::move(per[0]); }
    SharedBuffer all =
        mc->make_shared_buffer((std::size_t)total * _enc_hidden * 2);
    if (all.empty()) { return {}; }
    std::size_t off = 0;
    for (int i = 0; i < _img_n; ++i) {
      const std::size_t nb = (std::size_t)_img_tok[i] * _enc_hidden * 2;
      std::memcpy(static_cast<std::uint8_t*>(all.contents()) + off,
                  per[(std::size_t)i].contents(), nb);
      off += nb;
    }
    return all;
  }

  // Krea-2 edit (identity-edit LoRA): Qwen3-VL tower -> f16 [n_img, 2560]. The
  // instruction is encoded WITH the source image (training-matched grounded
  // encode); the raw source RGB comes through the ref_image iport. The 315
  // visual.* tower tensors ship inside text_encoder/, so it loads from _enc_dir.
  // Mage-Flow rides the SAME Qwen3-VL tower + deepstack path as krea2; only
  // the checkpoint prefix ("model.visual." vs "visual."), the conditioning
  // long-edge cap (384 vs 768) and the processor's min_pixels differ.
  if (_family == "krea2" || _family == "mage-flow" ||
      _family == "boogu-image") {
    // Boogu's mllm shares Mage-Flow's checkpoint wrapper ("model.visual."),
    // its bf16 pipeline dtype and its preprocessor bounds (shortest_edge
    // 65536), and its pipeline caps the VLM conditioning image at 384x384
    // pixels -- so it takes the same branch.
    const bool mage = (_family == "mage-flow" || _family == "boogu-image");
    if (!_vision3) {
      genai::ModelLoader loader(session());
      const auto mcfg = loader.load_config(_enc_dir);
      if (!mcfg.has_value()) {
        session()->warn(fmt("DiffusionConditionerStage('{}'): Qwen3-VL config "
                            "parse failed; text-only conditioning", this->id()));
        return {};
      }
      auto vcfg = genai::MetalQwenVisionEncoder::config_from(*mcfg);
      vcfg.weight_prefix = mage ? "model.visual." : "visual.";
      // Mage-Flow's pipeline casts its whole text encoder -- the Qwen3-VL
      // tower included -- to bf16, so the conditioning it was tuned against
      // carries bf16 tower numerics. Match that here. (f16 is the more
      // ACCURATE tower, ~3x closer to an fp32 oracle; this is fidelity to
      // the reference, which is what the goldens measure and what the DiT
      // was trained alongside.) Krea-2 stays f16 -- its own verified state.
      vcfg.use_bf16 = mage;
      // The processor bounds, from the model layer rather than from
      // literals here. `min_pixels` is what makes a small or very wide
      // reference get UPSCALED before patching: past ~2.25:1 aspect a
      // 384-capped image falls under Mage-Flow's 65536 and the Qwen
      // default of 3136 would silently skip that upscale. 0 means the
      // family did not set one, so the tower's own default stands.
      if (_ground.min_pixels > 0) { vcfg.min_pixels = _ground.min_pixels; }
      if (_ground.max_pixels > 0) { vcfg.max_pixels = _ground.max_pixels; }
      // Qwen3-VL normalizes images to [-1,1] (mean/std 0.5), NOT the OpenAI-CLIP
      // defaults the loader assumes when the repo ships no preprocessor_config.
      // With CLIP normalization the vision tokens -- hence the grounded image
      // conditioning -- are wrong, and the edit mis-targets (removes the
      // foreground subject instead of the background). Qwen3-VL's official
      // preprocessor_config uses 0.5/0.5.
      for (int i = 0; i < 3; ++i) {
        vcfg.image_mean[i] = 0.5f;
        vcfg.image_std[i] = 0.5f;
      }
      _vision3 = genai::MetalQwenVisionEncoder::load(_enc_dir, mc, vcfg);
      if (!_vision3) {
        session()->warn(fmt("DiffusionConditionerStage('{}'): Qwen3-VL vision "
                            "tower load failed; text-only conditioning",
                            this->id()));
        return {};
      }
      _vision3->set_session(session());
    }
    // Krea-2 is single-reference by design (the ComfyUI-Krea2Edit node takes
    // one source image); the others encode every wired picture.
    const int use_refs = (_family == "krea2") ? 1 : _n_ref;
    if (_family == "krea2" && _n_ref > 1) {
      session()->warn(fmt(
          "DiffusionConditionerStage('{}'): Krea-2 grounded encode is "
          "single-reference; ignoring {} extra reference image(s)",
          this->id(), _n_ref - 1));
    }
    std::vector<SharedBuffer> per_emb;
    std::vector<std::vector<SharedBuffer>> per_ds;
    int total = 0;
    for (int ri = 0; ri < use_refs; ++ri) {
    std::vector<std::uint8_t> capped;
    int rh = _ref_rgb_h[ri], rw = _ref_rgb_w[ri];
    // Mage-Flow caps the VL conditioning image's long edge at 384 (its
    // training preprocessing -- pipeline.py `vl_cond_long_edge`); the VAE
    // reference path keeps the full target resolution. krea2's grounding
    // node uses 768.
    // The GEOMETRY comes from the model layer (_ground); the FILTER and
    // the alignment stay per-family code. That split is deliberate: a
    // bound is a number someone might reasonably tune for a fine-tune,
    // while "LANCZOS, floor-aligned to 16" is Boogu's algorithm, and
    // matching the filter matters as much as the geometry -- a box
    // average leaves the image-row conditioning ~3x further from the
    // reference after LM amplification.
    if (_family == "boogu-image") {
      cap_longest_side_(_ref_rgb[ri].data(), _ref_rgb_h[ri], _ref_rgb_w[ri],
                        _ground.long_edge, capped, &rh, &rw,
                        _ground.pixel_budget, 16, /*lanczos=*/true);
    } else {
      cap_longest_side_(_ref_rgb[ri].data(), _ref_rgb_h[ri], _ref_rgb_w[ri],
                        _ground.long_edge, capped, &rh, &rw,
                        _ground.pixel_budget);
    }
    const std::uint8_t* rgb = capped.empty() ? _ref_rgb[ri].data()
                                             : capped.data();
    auto r = _vision3->encode(rgb, rh, rw);
    if (r.embeddings.empty() || r.n_tokens <= 0) { return {}; }
    // Per-reference merged grid (mh, mw) for the 2-D mROPE band. The encoder
    // returns the PATCH grid; the LM tokens are the S x S-merged set in merger
    // (row-major mh x mw) order.
    const int S = _vision3->config().spatial_merge > 0
                      ? _vision3->config().spatial_merge : 2;
    _img_mh[ri] = r.grid_h / S;
    _img_mw[ri] = r.grid_w / S;
    _img_tok[ri] = r.n_tokens;
    total += r.n_tokens;
    // Deepstack features (f16 [n_img, EH]) -> bf16 (the encoder residual dtype),
    // for injection into the text encoder at layers 0.. (see encode_).
    std::vector<SharedBuffer> ds_this;
    for (auto& df : r.deepstack) {
      if (df.empty()) { continue; }
      const std::size_t ne = (std::size_t)r.n_tokens * _enc_hidden;
      SharedBuffer b = mc->make_shared_buffer(ne * 2);
      if (b.empty()) { _ds_feats.clear(); break; }
      // The tower's element type is its own business (bf16 for Mage-Flow,
      // f16 for Krea-2); the encoder residual is bf16 either way. When they
      // already agree this is a straight copy.
      const auto* s = static_cast<const std::uint16_t*>(df.contents());
      auto* d = static_cast<std::uint16_t*>(b.contents());
      if (_vision3->is_bf16()) {
        std::memcpy(d, s, ne * 2);
      } else {
        for (std::size_t i = 0; i < ne; ++i) {
          _Float16 h; std::memcpy(&h, &s[i], 2);
          d[i] = f32_to_bf16_((float)h);
        }
      }
      ds_this.push_back(std::move(b));
    }
    session()->info(fmt(
        "DiffusionConditionerStage('{}'): reference {} -> {} vision tokens "
        "(grid {}x{}), {} deepstack", this->id(), ri, r.n_tokens,
        r.grid_h, r.grid_w, ds_this.size()));
    per_emb.push_back(std::move(r.embeddings));
    per_ds.push_back(std::move(ds_this));
    }   // for ri
    _img_n = (int)per_emb.size();
    if (_img_n == 0) { return {}; }
    n_img = total;
    // Concatenate the references' tower rows (and each deepstack level) in
    // prompt order; encode_ splices them over the pad rows in the same order,
    // and the deepstack Segs address the per-reference runs.
    const bool bf = _vision3->is_bf16();
    _ds_feats.clear();
    const std::size_t nlev = per_ds.empty() ? 0 : per_ds[0].size();
    for (std::size_t l = 0; l < nlev; ++l) {
      SharedBuffer b =
          mc->make_shared_buffer((std::size_t)total * _enc_hidden * 2);
      if (b.empty()) { _ds_feats.clear(); break; }
      std::size_t off = 0;
      bool ok = true;
      for (int i = 0; i < _img_n; ++i) {
        if (l >= per_ds[(std::size_t)i].size()) { ok = false; break; }
        const std::size_t nb = (std::size_t)_img_tok[i] * _enc_hidden * 2;
        std::memcpy(static_cast<std::uint8_t*>(b.contents()) + off,
                    per_ds[(std::size_t)i][l].contents(), nb);
        off += nb;
      }
      if (!ok) { _ds_feats.clear(); break; }
      _ds_feats.push_back(std::move(b));
    }
    if (_img_n == 1) { return std::move(per_emb[0]); }
    const std::size_t esz = bf ? 2u : 2u;   // both f16 and bf16 are 2 bytes
    SharedBuffer all =
        mc->make_shared_buffer((std::size_t)total * _enc_hidden * esz);
    if (all.empty()) { return {}; }
    std::size_t off = 0;
    for (int i = 0; i < _img_n; ++i) {
      const std::size_t nb = (std::size_t)_img_tok[i] * _enc_hidden * esz;
      std::memcpy(static_cast<std::uint8_t*>(all.contents()) + off,
                  per_emb[(std::size_t)i].contents(), nb);
      off += nb;
    }
    return all;
  }

  return {};   // flux2 etc.: text-only
}

SharedBuffer
DiffusionConditionerStage::encode_(const std::string& text, const char* which,
                                   int& n_real_out,
                                   const SharedBuffer& vtok, int n_img) const
{
  auto* mc = session()->services()->metal_compute();
  const int EH = _enc_hidden;
  (void)mc;

  if (_family == "wan") {
    // Wan tokenizes to a FIXED 512-token window and pads with the T5 pad
    // token, and the encoder zeroes every row past the real tokens -- so
    // what the DiT cross-attends to beyond the prompt is zero, not the
    // encoder's opinion of padding. That zero tail is part of the
    // conditioning contract (see MetalUmt5Encoder::encode), which is why
    // the full 512 rows are emitted rather than only the real ones.
    if (!_umt5) { return {}; }
    // The reference cleans the prompt before tokenizing (diffusers
    // WanPipeline.prompt_clean, and transformers' own T5 preprocessing
    // does the same): every whitespace run collapses to one space and
    // the ends are stripped. Not cosmetic -- a trailing space becomes a
    // real extra token and a tab or newline becomes UNKNOWN, since the
    // metaspace substitution only knows about ' '.
    auto ids = _tokenizer->encode(genai::Tokenizer::whitespace_clean(text));
    const std::int32_t eos = _tokenizer->special_token_id("</s>");
    if (eos >= 0) { ids.push_back(eos); }
    const int kWanMaxSeq = 512;
    if ((int)ids.size() > kWanMaxSeq) {
      session()->warn(fmt(
          "DiffusionConditionerStage('{}'): {} prompt is {} tokens; Wan's "
          "text window is {} -- truncating", this->id(), which, ids.size(),
          kWanMaxSeq));
      ids.resize((std::size_t)kWanMaxSeq);
    }
    const int n_real = (int)ids.size();
    std::int32_t pad = _tokenizer->special_token_id("<pad>");
    if (pad < 0) { pad = 0; }
    ids.resize((std::size_t)kWanMaxSeq, pad);
    n_real_out = kWanMaxSeq;
    std::string eerr;
    SharedBuffer h = _umt5->encode(ids, n_real, &eerr);
    if (h.empty()) {
      session()->warn(fmt("DiffusionConditionerStage('{}'): umT5 {} encode: {}",
                          this->id(), which, eerr));
    }
    return h;
  }

  if (_family == "minimax-h3") {
    // VERBATIM: no chat template, no BOS/EOS, no padding to a window.
    // Every other family here wraps the prompt in something, so the
    // wrong default is a live hazard rather than a hypothetical one --
    // and a template's tokens would land in the conditioning as real
    // rows the DiT attends to.
    if (!_h3_enc) { return {}; }
    int n = 0;
    std::string eerr;
    SharedBuffer h = _h3_enc->encode(text, &n, &eerr);
    if (h.empty()) {
      session()->warn(fmt("DiffusionConditionerStage('{}'): MiniMax-H3 {} "
                          "encode: {}", this->id(), which, eerr));
      return h;
    }
    n_real_out = n;
    return h;
  }

  if (_family == "flux2") {
    const int JD = 3 * EH;
    // Match diffusers Flux2Klein encode_prompt: apply_chat_template with
    // enable_thinking=False appends the Qwen3 empty thinking block after the
    // assistant turn. Omitting its 4 tokens (the causal encoder's most-
    // contextualized) drifts the conditioning -> degraded edits.
    const std::string tmpl = std::string("<|im_start|>user\n") + text +
                             "<|im_end|>\n<|im_start|>assistant\n"
                             "<think>\n\n</think>\n\n";
    auto ids = encode_with_specials_(*_tokenizer, tmpl);
    if (ids.empty()) { return {}; }
    const int real_n = (int)ids.size();   // real tokens, before padding
    // diffusers Flux2Klein tokenizes with padding="max_length", max_length=512
    // (pad id <|endoftext|>) and feeds ALL 512 Qwen3 hidden states to the DiT
    // (the DiT does NOT mask the conditioning). Under right-padding + causal
    // attention the 19 real tokens are unaffected; the padding positions carry
    // their own position-dependent hidden states that the DiT still attends to.
    // Emitting only the real tokens (the old behavior) fed the DiT a 19-token
    // conditioning where the reference feeds 512 -> a per-step attention bias
    // that compounds over the trajectory (color cast + composition drift).
    {
      const int kFluxMaxSeq = 512;
      std::int32_t pad = _tokenizer->special_token_id("<|endoftext|>");
      if (pad < 0) { pad = 151643; }
      if ((int)ids.size() > kFluxMaxSeq) { ids.resize(kFluxMaxSeq); }
      else { ids.resize(kFluxMaxSeq, pad); }
    }
    const int n = (int)ids.size();
    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
    if (x.empty()) { return {}; }
    const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    const std::size_t vocab = _embed.byte_size() / ((std::size_t)EH * 2);
    for (int i = 0; i < n; ++i) {
      const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
      if (id >= vocab) { return {}; }
      std::memcpy(xb + (std::size_t)i * EH * 2, tbl + (std::size_t)id * EH * 2,
                  (std::size_t)EH * 2);
    }
    std::vector<int> taps_l; for (int k : kFluxTaps) { taps_l.push_back(k - 1); }
    genai::ContextManager* cm = _encoder->context_manager();
    const genai::ContextId cid = cm->acquire_root();
    // key_valid_len = real_n: the padding tokens attend ONLY to the real
    // prefix (HF attention_mask semantics), matching diffusers' 512-token
    // conditioning exactly (vs plain causal, where padding attends to padding).
    const auto _enc_t0 = std::chrono::steady_clock::now();
    // LLM-lane perf event (perf-visualizer): the DiT text-conditioning encoder
    // prefill; value = the padded conditioning length.
    SharedBuffer taps;
    {
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmDitText,
                         kPerfLlmDitTextBegin, (std::uint64_t)n);
      taps = _encoder->forward_embeddings_taps(cid, x, n, taps_l, real_n);
    }
    if (std::getenv("VPIPE_COND_PROFILE")) {
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - _enc_t0).count();
      session()->log_normal(fmt("DiffusionConditionerStage: encoder prefill "
                                "n={} real={} -> {:.1f} ms", n, real_n, ms));
    }
    cm->release(cid);
    if (taps.empty()) { return {}; }
    // The FLUX.2 DiT consumes f16 context ([n, 3*EH]); convert bf16 taps to
    // _Float16 here (byte-identical to the old inline encode path).
    SharedBuffer ctxb = mc->make_shared_buffer((std::size_t)n * JD * 2);
    const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
    auto* cp = static_cast<_Float16*>(ctxb.contents());
    for (int p = 0; p < n; ++p) {
      for (int j = 0; j < 3; ++j) {
        const std::size_t src = ((std::size_t)j * n + p) * EH;
        const std::size_t dst = (std::size_t)p * JD + (std::size_t)j * EH;
        for (int h = 0; h < EH; ++h) {
          cp[dst + h] = (_Float16)bf16_to_f32_(tp[src + h]);
        }
      }
    }
    n_real_out = n;
    session()->log_debug(fmt("DiffusionConditionerStage('{}'): [{}] flux2 -> "
                             "[{}, {}] f16", this->id(), which, n, JD));
    return ctxb;
  }

  if (_family == "qwen-image-edit") {
    const int TD = EH;   // 3584
    const int NL = 28;
    const bool img_aware = (n_img > 0) && !vtok.empty();
    const std::int32_t pad_id =
        img_aware ? _tokenizer->special_token_id("<|image_pad|>") : -1;
    // QIE-2511 is MULTI-reference: one "Picture N: " labelled block per wired
    // picture, each expanded to its own vision-token count.
    const int nref = (img_aware && pad_id >= 0) ? _img_n : 0;
    std::string tmpl =
        std::string(kQiePrefix) + ref_blocks_(nref, "Picture") +
        text + std::string(kQieSuffix);
    std::vector<std::int32_t> ids = encode_with_specials_(*_tokenizer, tmpl);
    std::vector<std::pair<int, int>> runs;
    if (nref > 0) {
      runs = expand_pads_(ids, pad_id, _img_tok, nref);
    }
    if ((int)ids.size() <= kQieDropPrefix) { return {}; }
    const int n = (int)ids.size();
    const int n_real = n - kQieDropPrefix;
    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
    if (x.empty()) { return {}; }
    {
      const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
      auto* xb = static_cast<std::uint8_t*>(x.contents());
      const std::size_t vocab = _embed.byte_size() / ((std::size_t)EH * 2);
      for (int i = 0; i < n; ++i) {
        const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
        if (id >= vocab) { return {}; }
        std::memcpy(xb + (std::size_t)i * EH * 2, tbl + (std::size_t)id * EH * 2,
                    (std::size_t)EH * 2);
      }
    }
    if (img_aware && pad_id >= 0) {
      const auto* vt = static_cast<const std::uint16_t*>(vtok.contents());
      auto* xh = static_cast<std::uint16_t*>(x.contents());
      int j = 0;
      for (int i = 0; i < n && j < n_img; ++i) {
        if (ids[(std::size_t)i] == pad_id) {
          std::memcpy(xh + (std::size_t)i * EH, vt + (std::size_t)j * EH,
                      (std::size_t)EH * 2);
          ++j;
        }
      }
    }
    genai::ContextManager* cm = _encoder->context_manager();
    const genai::ContextId cid = cm->acquire_root();
    SharedBuffer taps;
    {   // LLM-lane perf event: DiT text-conditioning encoder prefill.
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmDitText,
                         kPerfLlmDitTextBegin, (std::uint64_t)n);
      taps =
          _encoder->forward_embeddings_taps(cid, x, n, std::vector<int>{NL - 1});
    }
    cm->release(cid);
    if (taps.empty()) { return {}; }
    // final-RMSNorm weight, applied on the host.
    std::vector<float> fnorm(EH, 1.0f);
    {
      if (_enc_ws) {
        // Through the set: this used to re-open (and re-parse the headers
        // of) the whole checkpoint on EVERY conditioning call to read one
        // vector. Now it is a lookup on a checkpoint already open.
        SharedBuffer nw =
            _enc_ws->tensor("model.norm.weight", mc,
                            genai::WeightSet::Residency::Copied);
        if (!nw.empty()) {
          const auto* p = static_cast<const std::uint16_t*>(nw.contents());
          for (int h = 0; h < EH; ++h) { fnorm[h] = bf16_to_f32_(p[h]); }
        }
      }
    }
    SharedBuffer txt = mc->make_shared_buffer((std::size_t)n_real * TD * 2);
    const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
    auto* op = static_cast<std::uint16_t*>(txt.contents());
    for (int p = 0; p < n_real; ++p) {
      const auto* row = tp + (std::size_t)(p + kQieDropPrefix) * EH;
      double ss = 0.0;
      for (int h = 0; h < EH; ++h) { const double v = bf16_to_f32_(row[h]);
                                     ss += v * v; }
      const double inv = 1.0 / std::sqrt(ss / (double)EH + 1e-6);
      for (int h = 0; h < TD; ++h) {
        op[(std::size_t)p * TD + h] =
            f32_to_bf16_((float)(bf16_to_f32_(row[h]) * inv * fnorm[h]));
      }
    }
    n_real_out = n_real;
    session()->log_debug(fmt("DiffusionConditionerStage('{}'): [{}] qie -> "
                             "[{}, {}]{}", this->id(), which, n_real, TD,
                             img_aware ? ", image-aware" : ""));
    return txt;
  }

  if (_family == "boogu-image") {
    // Boogu conditioning: the mllm's LAST hidden state over the WHOLE templated
    // sequence (no prefix drop -- the DiT's context_refiner is trained on the
    // system prompt too), image-grounded when a reference is wired. The system
    // prompt itself switches on that: t2i vs ti2i, exactly as the pipeline
    // picks it by "are there input images".
    const int NL = _encoder->config().n_layers;
    const bool img_aware = (n_img > 0) && !vtok.empty();
    const std::int32_t pad_id =
        img_aware ? _tokenizer->special_token_id("<|image_pad|>") : -1;
    const bool grounded = img_aware && pad_id >= 0;
    // Boogu renders BARE back-to-back vision blocks (no "Picture N:" label) --
    // verified against the reference's own rendered template.
    const int nref = grounded ? _img_n : 0;
    const std::string tmpl =
        (grounded ? std::string(kBooguTi2i) + ref_blocks_(nref, nullptr)
                  : std::string(kBooguT2I)) +
        text + std::string(kBooguSuffix);
    std::vector<std::int32_t> ids = encode_with_specials_(*_tokenizer, tmpl);
    std::vector<std::pair<int, int>> runs;
    if (nref > 0) { runs = expand_pads_(ids, pad_id, _img_tok, nref); }
    if (ids.empty()) { return {}; }
    const int n = (int)ids.size();
    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
    if (x.empty()) { return {}; }
    {
      const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
      auto* xb = static_cast<std::uint8_t*>(x.contents());
      const std::size_t vocab = _embed.byte_size() / ((std::size_t)EH * 2);
      for (int i = 0; i < n; ++i) {
        const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
        if (id >= vocab) { return {}; }
        std::memcpy(xb + (std::size_t)i * EH * 2, tbl + (std::size_t)id * EH * 2,
                    (std::size_t)EH * 2);
      }
    }
    // Splice the tower rows over the image_pad embeddings (the tower runs bf16
    // here, matching the encoder residual -- ask it rather than assume).
    int first_pad = -1;
    if (grounded) {
      const bool vt_bf16 = _vision3 && _vision3->is_bf16();
      const auto* vt = static_cast<const std::uint16_t*>(vtok.contents());
      auto* xh = static_cast<std::uint16_t*>(x.contents());
      int j = 0;
      for (int i = 0; i < n && j < n_img; ++i) {
        if (ids[(std::size_t)i] == pad_id) {
          if (first_pad < 0) { first_pad = i; }
          const std::uint16_t* src = vt + (std::size_t)j * EH;
          if (vt_bf16) {
            std::memcpy(xh + (std::size_t)i * EH, src, (std::size_t)EH * 2);
          } else {
            for (int h = 0; h < EH; ++h) {
              _Float16 hf; std::memcpy(&hf, &src[h], 2);
              xh[(std::size_t)i * EH + h] = f32_to_bf16_((float)hf);
            }
          }
          ++j;
        }
      }
    }
    genai::MetalQwenModel::DeepstackInject ds;
    const bool use_ds = grounded && first_pad >= 0 && !_ds_feats.empty();
    if (use_ds) {
      for (int i = 0; i < (int)_ds_feats.size(); ++i) {
        ds.feats.push_back(&_ds_feats[(std::size_t)i]);
        ds.layers.push_back(i);
      }
      ds.row0 = first_pad;
      ds.rows = n_img;
      if (runs.size() > 1) {
        // Boogu's blocks are adjacent, so one span would still cover them --
        // but be explicit rather than rely on that, since the features are
        // concatenated per reference.
        int feat = 0;
        for (const auto& rn : runs) {
          ds.segs.push_back({rn.first, rn.second, feat});
          feat += rn.second;
        }
      }
    }
    // Boogu calls the STOCK Qwen3-VL forward, so the image rows carry the
    // normal 2-D mROPE grid (t=base, h=base+row, w=base+col; text resumes at
    // base + max(mh,mw)) -- NOT Mage-Flow's flat arange override. Each
    // reference gets its OWN band.
    const bool use_mrope = grounded && !runs.empty() && _img_mw[0] > 0;
    std::vector<std::int32_t> pos;
    if (use_mrope) { pos = mrope_positions_(n, runs, _img_mh, _img_mw); }
    genai::ContextManager* cm = _encoder->context_manager();
    const genai::ContextId cid = cm->acquire_root();
    SharedBuffer taps;
    {   // LLM-lane perf event: DiT text-conditioning encoder prefill.
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmDitText,
                         kPerfLlmDitTextBegin, (std::uint64_t)n);
      taps = use_mrope
                 ? _encoder->forward_embeddings_taps_mrope(
                       cid, x, n, pos, std::vector<int>{NL - 1},
                       /*key_valid_len=*/0, use_ds ? &ds : nullptr)
                 : _encoder->forward_embeddings_taps(
                       cid, x, n, std::vector<int>{NL - 1},
                       /*key_valid_len=*/0, use_ds ? &ds : nullptr);
    }
    cm->release(cid);
    if (taps.empty()) { return {}; }
    // last_hidden_state is POST the model's final RMSNorm; the tap is the last
    // layer's pre-norm output, so apply it on the host.
    std::vector<float> fnorm((std::size_t)EH, 1.0f);
    {
      if (_enc_ws) {
        // Through the set: this used to re-open (and re-parse the headers
        // of) the whole checkpoint on EVERY conditioning call to read one
        // vector. Now it is a lookup on a checkpoint already open.
        SharedBuffer nw =
            _enc_ws->tensor("model.language_model.norm.weight", mc,
                            genai::WeightSet::Residency::Copied);
        if (!nw.empty()) {
          const auto* p = static_cast<const std::uint16_t*>(nw.contents());
          for (int h = 0; h < EH; ++h) { fnorm[(std::size_t)h] = bf16_to_f32_(p[h]); }
        }
      }
    }
    const float reps = _encoder->config().rms_eps;
    SharedBuffer txt = mc->make_shared_buffer((std::size_t)n * EH * 2);
    if (txt.empty()) { return {}; }
    const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
    auto* op = static_cast<std::uint16_t*>(txt.contents());
    for (int p = 0; p < n; ++p) {
      const auto* row = tp + (std::size_t)p * EH;
      double ss = 0.0;
      for (int h = 0; h < EH; ++h) {
        const double v = bf16_to_f32_(row[h]); ss += v * v;
      }
      const double inv = 1.0 / std::sqrt(ss / (double)EH + (double)reps);
      for (int h = 0; h < EH; ++h) {
        op[(std::size_t)p * EH + h] =
            f32_to_bf16_((float)(bf16_to_f32_(row[h]) * inv * fnorm[(std::size_t)h]));
      }
    }
    n_real_out = n;   // Boogu keeps EVERY token
    session()->log_debug(fmt("DiffusionConditionerStage('{}'): [{}] boogu-image "
                             "-> [{}, {}] bf16{}", this->id(), which, n, EH,
                             grounded ? ", image-grounded (edit)" : ""));
    return txt;
  }

  if (_family == "mage-flow") {
    const int NL = 36;   // Qwen3-VL 4B layers
    const bool img_aware = (n_img > 0) && !vtok.empty();
    const std::int32_t pad_id =
        img_aware ? _tokenizer->special_token_id("<|image_pad|>") : -1;
    const bool grounded = img_aware && pad_id >= 0;
    // Edit template (drop 64) when a reference rides along, t2i template
    // (drop 34) otherwise -- the reference picks the template from the CALL
    // (generate_edits vs generate_images), which is the same distinction.
    const int drop = grounded ? kQieDropPrefix : kDropPrefix;
    const std::string tmpl =
        (grounded ? std::string(kQiePrefix) + ref_blocks_(_img_n, "Image")
                  : std::string(kPrefix)) +
        text + (grounded ? std::string(kQieSuffix) : std::string(kSuffix));
    std::vector<std::int32_t> ids = encode_with_specials_(*_tokenizer, tmpl);
    std::vector<std::pair<int, int>> runs;
    if (grounded) { runs = expand_pads_(ids, pad_id, _img_tok, _img_n); }
    if ((int)ids.size() <= drop) { return {}; }
    const int n = (int)ids.size();
    const int n_real = n - drop;
    // Gather through the model's embed muxer (the same table, the same rows:
    // a dense gather kernel where the other families memcpy). Mage-Flow is
    // the one family that binds the muxer -- see encoder_config_mage_().
    for (const std::int32_t id : ids) {
      if (id < 0 || id >= _encoder->config().vocab) { return {}; }
    }
    SharedBuffer x = _encoder->embed_text_buf(ids);
    if (x.empty() || x.byte_size() < (std::size_t)n * EH * 2) { return {}; }
    // Splice the tower rows over the image_pad embeddings (f16 tower -> bf16
    // encoder input, as in the krea2 path below).
    int first_pad = -1;
    if (grounded) {
      // Tower rows in the tower's OWN element type -> the encoder's bf16
      // residual. Reading a bf16 tower's buffer as f16 gives values of
      // roughly the right magnitude and entirely the wrong content, so this
      // asks the tower rather than assuming.
      const bool vt_bf16 = _vision3 && _vision3->is_bf16();
      const auto* vt = static_cast<const std::uint16_t*>(vtok.contents());
      auto* xh = static_cast<std::uint16_t*>(x.contents());
      int j = 0;
      for (int i = 0; i < n && j < n_img; ++i) {
        if (ids[(std::size_t)i] == pad_id) {
          if (first_pad < 0) { first_pad = i; }
          const std::uint16_t* src = vt + (std::size_t)j * EH;
          if (vt_bf16) {
            std::memcpy(xh + (std::size_t)i * EH, src, (std::size_t)EH * 2);
          } else {
            for (int h = 0; h < EH; ++h) {
              _Float16 hf; std::memcpy(&hf, &src[h], 2);
              xh[(std::size_t)i * EH + h] = f32_to_bf16_((float)hf);
            }
          }
          ++j;
        }
      }
    }
    genai::MetalQwenModel::DeepstackInject ds;
    const bool use_ds = grounded && first_pad >= 0 && !_ds_feats.empty();
    if (use_ds) {
      for (int i = 0; i < (int)_ds_feats.size(); ++i) {
        ds.feats.push_back(&_ds_feats[(std::size_t)i]);
        ds.layers.push_back(i);
      }
      ds.row0 = first_pad;
      ds.rows = n_img;
      if (runs.size() > 1) {
        // The "Image N: " labels sit BETWEEN the blocks, so the image-token runs
        // are genuinely disjoint here -- one span would inject into the label
        // tokens. Segs address each run against the concatenated features.
        int feat = 0;
        for (const auto& rn : runs) {
          ds.segs.push_back({rn.first, rn.second, feat});
          feat += rn.second;
        }
      }
    }
    // NOTE: SEQUENTIAL positions, NOT the 2-D mROPE grid krea2 uses. Mage-Flow
    // overrides position_ids with a per-sequence `torch.arange(length)`
    // (text_encoder.py TextEncoder.forward), which the patched Qwen3-VL text
    // model expands to three IDENTICAL axes -- so the image rows carry plain
    // sequential positions even though the tower is multimodal.
    genai::ContextManager* cm = _encoder->context_manager();
    const genai::ContextId cid = cm->acquire_root();
    SharedBuffer taps;
    {   // LLM-lane perf event: DiT text-conditioning encoder prefill.
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmDitText,
                         kPerfLlmDitTextBegin, (std::uint64_t)n);
      taps = _encoder->forward_embeddings_taps(
          cid, x, n, std::vector<int>{NL - 1}, /*key_valid_len=*/0,
          use_ds ? &ds : nullptr);
    }
    cm->release(cid);
    if (taps.empty()) { return {}; }
    // Final RMSNorm on the host (the tap is the last layer's pre-norm output).
    std::vector<float> fnorm(EH, 1.0f);
    {
      if (_enc_ws) {
        // Through the set: this used to re-open (and re-parse the headers
        // of) the whole checkpoint on EVERY conditioning call to read one
        // vector. Now it is a lookup on a checkpoint already open.
        SharedBuffer nw =
            _enc_ws->tensor("model.language_model.norm.weight", mc,
                            genai::WeightSet::Residency::Copied);
        if (!nw.empty()) {
          const auto* p = static_cast<const std::uint16_t*>(nw.contents());
          for (int h = 0; h < EH; ++h) { fnorm[h] = bf16_to_f32_(p[h]); }
        }
      }
    }
    SharedBuffer txt = mc->make_shared_buffer((std::size_t)n_real * EH * 2);
    if (txt.empty()) { return {}; }
    const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
    auto* op = static_cast<std::uint16_t*>(txt.contents());
    for (int p = 0; p < n_real; ++p) {
      const auto* row = tp + (std::size_t)(p + drop) * EH;
      double ss = 0.0;
      for (int h = 0; h < EH; ++h) {
        const double v = bf16_to_f32_(row[h]); ss += v * v;
      }
      const double inv = 1.0 / std::sqrt(ss / (double)EH + 1e-6);
      for (int h = 0; h < EH; ++h) {
        op[(std::size_t)p * EH + h] =
            f32_to_bf16_((float)(bf16_to_f32_(row[h]) * inv * fnorm[h]));
      }
    }
    n_real_out = n_real;
    session()->log_debug(fmt("DiffusionConditionerStage('{}'): [{}] mage-flow "
                             "-> [{}, {}] bf16{}", this->id(), which, n_real,
                             EH, grounded ? ", image-grounded (edit)" : ""));
    return txt;
  }

  // krea2: 12-tap -> [n_real, 12, EH] (the DiT fuses via forward_text). With a
  // reference image (edit / identity-edit LoRA) the instruction is encoded
  // GROUNDED on the source: insert <|vision_start|><|image_pad|><|vision_end|>
  // in the user turn and overwrite the pad positions with the Qwen3-VL vision
  // tokens (training-matched). kPrefix -- hence kDropPrefix -- is unchanged: the
  // vision block sits after the (identical) system prefix, so it survives the
  // prefix drop and joins the retained conditioning.
  const bool img_aware = (n_img > 0) && !vtok.empty();
  const std::int32_t pad_id =
      img_aware ? _tokenizer->special_token_id("<|image_pad|>") : -1;
  const std::string tmpl =
      std::string(kPrefix) +
      (img_aware && pad_id >= 0
           ? "<|vision_start|><|image_pad|><|vision_end|>" : "") +
      text + std::string(kSuffix);
  std::vector<std::int32_t> ids = encode_with_specials_(*_tokenizer, tmpl);
  if (img_aware && pad_id >= 0) {   // expand the single pad id to n_img copies
    std::vector<std::int32_t> ex; ex.reserve(ids.size() + (std::size_t)n_img);
    for (const std::int32_t id : ids) {
      if (id == pad_id) { for (int j = 0; j < n_img; ++j) ex.push_back(pad_id); }
      else { ex.push_back(id); }
    }
    ids.swap(ex);
  }
  if ((int)ids.size() <= kDropPrefix) { return {}; }
  const int n = (int)ids.size();
  const int n_real = n - kDropPrefix;
  const int NL = 12;
  SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
  if (x.empty()) { return {}; }
  {
    const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    const std::size_t vocab = _embed.byte_size() / ((std::size_t)EH * 2);
    for (int i = 0; i < n; ++i) {
      const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
      if (id >= vocab) { return {}; }
      std::memcpy(xb + (std::size_t)i * EH * 2, tbl + (std::size_t)id * EH * 2,
                  (std::size_t)EH * 2);
    }
  }
  // Splice the vision-tower rows over the image_pad embeddings. The Qwen3-VL
  // tower emits f16; the encoder input is bf16, so convert per element (the QIE
  // tower already emits bf16 and raw-copies -- this is the krea2 difference).
  int first_pad = -1;   // row0 of the contiguous image block (for deepstack)
  if (img_aware && pad_id >= 0) {
    const auto* vt = static_cast<const _Float16*>(vtok.contents());
    auto* xh = static_cast<std::uint16_t*>(x.contents());
    int j = 0;
    for (int i = 0; i < n && j < n_img; ++i) {
      if (ids[(std::size_t)i] == pad_id) {
        if (first_pad < 0) { first_pad = i; }
        for (int h = 0; h < EH; ++h) {
          xh[(std::size_t)i * EH + h] =
              f32_to_bf16_((float)vt[(std::size_t)j * EH + h]);
        }
        ++j;
      }
    }
  }
  std::vector<int> taps_l; for (int k : kSelectLayers) { taps_l.push_back(k - 1); }
  // Qwen3-VL deepstack: after LM layers 0..N-1, ADD the vision deepstack
  // features to the image rows (contiguous from first_pad). Only when grounded
  // and the tower produced deepstack features.
  genai::MetalQwenModel::DeepstackInject ds;
  const bool use_ds = img_aware && first_pad >= 0 && !_ds_feats.empty();
  if (use_ds) {
    for (int i = 0; i < (int)_ds_feats.size(); ++i) {
      ds.feats.push_back(&_ds_feats[(std::size_t)i]);
      ds.layers.push_back(i);
    }
    ds.row0 = first_pad;
    ds.rows = n_img;
  }
  // Qwen3-VL image tokens require 3-axis mROPE: text tokens sequential, the
  // n_img image tokens (contiguous from first_pad, merger row-major order) get
  // 2-D grid positions (t=base, h=base+row, w=base+col); the next text token
  // resumes at base + max(mh, mw). Without this the image rows carry sequential
  // positions and the grounded conditioning collapses (image tokens diverge
  // from the reference). Sequential path kept for the text-only case.
  const bool use_mrope = img_aware && first_pad >= 0
                         && _img_mh[0] > 0 && _img_mw[0] > 0;
  std::vector<std::int32_t> pos;
  if (use_mrope) {
    pos.assign((std::size_t)3 * n, 0);
    int cur = 0;
    for (int i = 0; i < n;) {
      if (i == first_pad) {
        const int base = cur;
        for (int j = 0; j < n_img && i < n; ++j, ++i) {
          pos[(std::size_t)i] = base;                          // T
          pos[(std::size_t)n + i] = base + j / _img_mw[0];     // H
          pos[(std::size_t)2 * n + i] = base + j % _img_mw[0]; // W
        }
        cur = base + std::max(_img_mh[0], _img_mw[0]);
      } else {
        pos[(std::size_t)i] = cur;
        pos[(std::size_t)n + i] = cur;
        pos[(std::size_t)2 * n + i] = cur;
        ++cur; ++i;
      }
    }
  }
  genai::ContextManager* cm = _encoder->context_manager();
  const genai::ContextId cid = cm->acquire_root();
  SharedBuffer taps;
  {   // LLM-lane perf event: DiT text-conditioning encoder prefill.
    PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmDitText,
                       kPerfLlmDitTextBegin, (std::uint64_t)n);
    taps = use_mrope
               ? _encoder->forward_embeddings_taps_mrope(
                     cid, x, n, pos, taps_l, /*key_valid_len=*/0,
                     use_ds ? &ds : nullptr)
               : _encoder->forward_embeddings_taps(
                     cid, x, n, taps_l, /*key_valid_len=*/0,
                     use_ds ? &ds : nullptr);
  }
  cm->release(cid);
  if (taps.empty()) { return {}; }
  // The Krea-2 DiT's forward_text consumes f16 ([n_real, 12, EH]); convert the
  // bf16 taps to _Float16 here (byte-identical to the old inline encode path).
  SharedBuffer ehs = mc->make_shared_buffer((std::size_t)n_real * NL * EH * 2);
  const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
  auto* ep = static_cast<_Float16*>(ehs.contents());
  for (int p = 0; p < n_real; ++p) {
    for (int j = 0; j < NL; ++j) {
      const std::size_t src = ((std::size_t)j * n + (kDropPrefix + p)) * EH;
      const std::size_t dst = ((std::size_t)p * NL + j) * EH;
      for (int h = 0; h < EH; ++h) {
        ep[dst + h] = (_Float16)bf16_to_f32_(tp[src + h]);
      }
    }
  }
  n_real_out = n_real;
  session()->log_debug(fmt("DiffusionConditionerStage('{}'): [{}] krea2 -> "
                           "[{}, 12, {}] f16{}", this->id(), which, n_real, EH,
                           img_aware ? ", image-grounded" : ""));
  return ehs;
}

// bf16 -> f32 for the opt-in conditioning trace below.
inline float
bf16_to_f32_dbg_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Wrap a 2-byte/elt metal buffer [rows, ...] as a TensorBeatPayload of `shape`.
// `dt` records the element type the paired DiT consumes -- F16 for krea2/flux2
// (forward_text/forward_dit read _Float16), Bf16 for qwen-image-edit.
static std::unique_ptr<TensorBeatPayload>
to_beat_(const SharedBuffer& buf, std::vector<std::int64_t> shape,
         TensorBeat::DType dt)
{
  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = dt;
  out->shape = std::move(shape);
  std::size_t n = 1; for (auto d : out->shape) { n *= (std::size_t)d; }
  out->resize_contiguous(n);
  std::memcpy(out->as_u8(), buf.contents(), n * 2);
  return out;
}

// The conditioning beat a BLOCKED prompt gets: a single zero row carrying
// `content_blocked` on its sideband. The DiT never looks at the numbers -- it
// checks the flag, skips the denoise entirely and passes the flag on to
// vae-decode, which emits the blank refusal image. Emitting a beat at all (a
// refusal is still an answer) is what keeps a blocked prompt from stalling a
// pipeline that is blocked on the DiT's iport0.
static std::unique_ptr<TensorBeatPayload>
blocked_beat_(int enc_hidden)
{
  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::Bf16;
  out->shape = {1, enc_hidden};
  out->resize_contiguous((std::size_t)enc_hidden);
  std::memset(out->as_u8(), 0, (std::size_t)enc_hidden * 2);
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign("content_blocked", FlexData::make_bool(true));
  out->sideband = std::move(sb);
  return out;
}

genai::MageScreenVerdict
DiffusionConditionerStage::screen_(const std::string& prompt,
                                   const SharedBuffer& vtok, int n_img) const
{
  genai::MageScreenRequest req;
  req.prompt = prompt;
  if (n_img > 0 && !vtok.empty()) {
    req.vision = &vtok;
    req.n_img  = n_img;
    // The content screen judges the source picture(s) as one block; with
    // several references the grids differ, so pass the FIRST (the screen only
    // needs a consistent 2-D band, and Mage-Flow's own screen path is
    // single-image).
    req.img_mh = _img_mh[0];
    req.img_mw = _img_mw[0];
    for (const auto& f : _ds_feats) { req.deepstack.push_back(&f); }
  }
  return genai::mage_screen(*_encoder, *_tokenizer, req, session());
}

Job
DiffusionConditionerStage::process(RuntimeContext& ctx)
{
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) { co_return; }

  // The model config. Latched, but RE-READ whenever another beat is
  // waiting: a config source with no trigger emits once for the whole
  // run, while a trigger-driven one emits per request. Blocking on the
  // first beat and polling after serves both -- the first prompt waits
  // for the parameters it was wired to use, later ones pick up a change
  // without waiting for one that may never come. Unlike the DiT's, every
  // parameter here is read per encode, so a late change simply applies.
  if (ctx.num_iports() > kModelCfgPort && ctx.iport_connected(kModelCfgPort) &&
      (!_cfg_latched || ctx.backlog(kModelCfgPort) > 0)) {
    auto gb = co_await ctx.read(kModelCfgPort);
    _cfg_latched = true;
    if (const auto* gfd =
            gb ? dynamic_cast<const FlexDataPayload*>(gb.get()) : nullptr) {
      _model_cfg = gfd->data;
      // Only once the family is known; otherwise ensure_loaded_ applies
      // it at the moment it becomes known.
      if (_load_attempted) { apply_model_config_(); }
    }
  }
  // Latch the shared model (iport2) once -- a model-select source overrides the
  // hf_dir config -- then lazily load the encoder before we need it.
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      if (apply_model_select_beat(mfd->data, _hf_dir)) {
        ensure_loaded_();
      }
    }
  }
  // The wan family's tower is _umt5, not _encoder (a umT5 ENCODER rather
  // than a decoder-only LM), so both members have to be consulted --
  // testing only _encoder made a loaded wan conditioner look inert.
  const bool have_enc =
      _encoder != nullptr || _umt5 != nullptr || _h3_enc != nullptr;
  // Post-barrier AND post-load: only here are both `_peer_dirs` and the
  // peers' resident bytes real.
  //
  // This used to run at the top of process(), which read as "after the
  // init barrier, so the footprint is real" and was not: `_peer_dirs` is
  // filled by ensure_loaded_(), which for a model-select graph runs LATER
  // IN THIS SAME CALL. So the first resolution sized against an EMPTY dir
  // list, measured a footprint of ~0, concluded the box was roomy, and
  // latched that answer for the life of the stage. On a 16 GB box running
  // MiniMax-H3 that kept ~8 GB of pinned Qwen3-VL-32B resident through a
  // 33B denoise the encoder takes no part in, and the DiT's forward was
  // refused for want of working set.
  resolve_unload_policy_();
  // NOTE: an unloaded encoder is NOT reloaded here. The reload waits until
  // a prompt has actually been read -- see below. `_unloaded` is what
  // distinguishes "dropped on purpose, reloadable" from genuinely inert,
  // so the gate consults it rather than treating an absent encoder as
  // proof of nothing to do.
  if (!have_enc && !_unloaded) {
    // Inert: consume the prompt rather than returning immediately. A
    // process() that neither blocks nor signals done is re-invoked at
    // once, and the stage then spins a core for the life of the pipeline
    // doing nothing but building and tearing down its own coroutine
    // frame -- which is what this looked like when the gate was wrong.
    auto drop = co_await ctx.read(0);
    if (!drop) { ctx.signal_done(); }
    co_return;
  }

  // Latch the negative prompt (iport1) + reference image (iport3) once.
  if (!_negative_latched && (int)ctx.num_iports() > 1 &&
      ctx.iport_connected(1)) {
    auto nb = co_await ctx.read(1);
    if (const auto* fp = nb ? dynamic_cast<const FlexDataPayload*>(nb.get())
                            : nullptr) {
      _negative_prompt = flex_text_(fp->data);
    }
    _negative_latched = true;
  }
  // Reference images latch independently on iport3 / iport4, then compact to a
  // contiguous [0, _n_ref) run so "reference i" always means the i-th picture
  // the VLM sees even if only the second port was wired.
  for (int i = 0; i < kMaxRefs; ++i) {
    const unsigned port = 3u + (unsigned)i;
    if (!_ref_rgb[i].empty()) { continue; }
    if ((int)ctx.num_iports() <= (int)port || !ctx.iport_connected(port)) {
      continue;
    }
    auto rb = co_await ctx.read(port);
    const auto* tb = rb ? dynamic_cast<const TensorBeatPayload*>(rb.get())
                        : nullptr;
    if (tb != nullptr && tb->dtype == TensorBeat::DType::U8 &&
        tb->shape.size() == 3 && tb->shape[0] == 3) {
      const auto bytes = tb->materialize_contiguous();
      _ref_rgb[i].assign(bytes.begin(), bytes.end());
      _ref_rgb_h[i] = (int)tb->shape[1];
      _ref_rgb_w[i] = (int)tb->shape[2];
    }
  }
  {
    int w = 0;
    for (int i = 0; i < kMaxRefs; ++i) {
      if (_ref_rgb[i].empty()) { continue; }
      if (w != i) {
        _ref_rgb[w] = std::move(_ref_rgb[i]);
        _ref_rgb_h[w] = _ref_rgb_h[i]; _ref_rgb_w[w] = _ref_rgb_w[i];
        _ref_rgb[i].clear();
      }
      ++w;
    }
    _n_ref = w;
  }

  // Read the prompt (iport0). A null beat means the upstream source is
  // exhausted: signal done so the runtime tears the stage down instead of
  // re-invoking process() in a tight EOF-read loop.
  auto pb = co_await ctx.read(0);
  if (!pb) { ctx.signal_done(); co_return; }
  const auto* fp = dynamic_cast<const FlexDataPayload*>(pb.get());
  if (fp == nullptr) { co_return; }
  const std::string prompt = flex_text_(fp->data);
  if (prompt.empty()) { co_return; }

  // ONLY NOW is the encoder worth having again.
  //
  // A previous prompt may have dropped it to leave the DiT room. The
  // reload used to sit above the read, which meant the LAST process() of
  // a run -- the one that exists only to observe EOS -- brought the whole
  // encoder back to discover there was nothing to do. On MiniMax-H3 that
  // put a streaming Qwen3-VL-32B (~1.5 GB pinned, and the disk traffic to
  // rebuild it) back alongside a 33B denoise that had just been given
  // room by dropping it. Reading first costs nothing: the beat is already
  // in hand, and a null one returns above without touching the encoder.
  if (_unloaded) { reload_encoder_(); }
  if (_encoder == nullptr && _umt5 == nullptr && _h3_enc == nullptr) {
    session()->warn(fmt(
        "DiffusionConditionerStage('{}'): the encoder could not be reloaded; "
        "dropping this prompt", this->id()));
    co_return;
  }

  // Vision tokens (QIE + ref image) -- shared by the positive + negative encode.
  int n_img = 0;
  SharedBuffer vtok = vision_tokens_(mc, n_img);

  // Shape helper for the family conditioning tensor.
  auto shape_for = [&](int rows) -> std::vector<std::int64_t> {
    if (_family == "krea2") { return {rows, 12, _enc_hidden}; }
    if (_family == "flux2") { return {rows, 3 * _enc_hidden}; }
    return {rows, _enc_hidden};   // qwen-image-edit / mage-flow
  };
  // Element type the paired DiT consumes: krea2/flux2 -> f16, the single-tap
  // families (qwen-image-edit / mage-flow) -> bf16.
  const TensorBeat::DType cdt = single_tap_(_family) ? TensorBeat::DType::Bf16
                                                     : TensorBeat::DType::F16;

  // ---- MANDATORY content screen (Mage-Flow) ----------------------------
  // Runs on the encoder this stage already owns, on every prompt, with no
  // config key to turn it off -- microsoft/Mage-Flow puts the classifier on
  // the text encoder precisely so it cannot be skipped, and a graph that
  // could omit it would be a bypass. FAIL-CLOSED: mage_screen() returns a
  // BLOCKING verdict for every failure mode, so a classifier that will not
  // run stops generation instead of waving it through.
  if (_family == "mage-flow") {
    const genai::MageScreenVerdict verdict = screen_(prompt, vtok, n_img);
    if (verdict.violates) {
      ++_blocked;
      // Say THAT it was blocked, not WHICH category tripped: the reference
      // surfaces nothing at all (its refusal banner is deliberately empty)
      // because naming the category turns the gate into an oracle to probe
      // against. The full verdict stays at debug level for diagnosis.
      session()->warn(fmt(
          "DiffusionConditionerStage('{}'): prompt blocked by the Mage-Flow "
          "content policy; emitting a refusal", this->id()));
      std::string cats;
      for (const auto& c : verdict.categories) {
        if (!cats.empty()) { cats += ","; }
        cats += c;
      }
      session()->log_debug(fmt(
          "DiffusionConditionerStage('{}'): content screen verdict: [{}] {}",
          this->id(), cats.empty() ? "-" : cats.c_str(), verdict.reason));
      co_await ctx.write(0, blocked_beat_(_enc_hidden));
      ++_emitted;
      release_encoder_when_idle_();
      co_return;
    }
  }

  int n_real = 0;
  SharedBuffer cond = encode_(prompt, "prompt", n_real, vtok, n_img);
  if (cond.empty()) {
    session()->warn(fmt("DiffusionConditionerStage('{}'): prompt encode failed",
                        this->id()));
    co_return;
  }
  // Emit the negative conditioning (oport1) BEFORE the positive (oport0): the
  // generate-image stage blocks on iport0, so enqueuing the negative first
  // guarantees its paired beat is already in iport1's FIFO when the positive
  // arrives (a race-free backlog poll on the consumer side).
  // Emit the negative conditioning (oport1) for the DiT's CFG. Normally only
  // when a non-empty negative prompt is wired; the Krea-2 edit deletion recipe
  // instead uses a GROUNDED encode of an EMPTY instruction as the negative
  // (README: "negative (leave empty) -- only used at CFG > 1"), enabled by
  // `grounded_negative` on an image-aware run. encode_("") still splices the
  // vision tokens, so the empty negative stays image-grounded.
  const bool emit_neg =
      !_negative_prompt.empty()
      || (_grounded_negative && n_img > 0 && !vtok.empty());
  if (emit_neg) {
    int n_neg = 0;
    SharedBuffer nc = encode_(_negative_prompt, "negative", n_neg, vtok, n_img);
    if (!nc.empty()) {
      co_await ctx.write(1, to_beat_(nc, shape_for(n_neg), cdt));
    }
  }
  // Drop the encoder BEFORE the conditioning is published: the DiT stage starts
  // its denoise as soon as the beat lands, so releasing here is what gives it
  // the working set. It reloads when the next prompt arrives.
  release_encoder_when_idle_();
  {
    auto beat = to_beat_(cond, shape_for(n_real), cdt);
    // Opt-in trace of the conditioning ITSELF. Two prompts that produce
    // the same downstream generation are ambiguous between "the encoder
    // emitted near-identical embeddings" and "the DiT ignored different
    // ones", and only the tensor at this boundary separates them. The
    // per-row spread is the load-bearing number: a mean-collapsed
    // encoder still has a plausible global RMS.
    if (std::getenv("VPIPE_COND_PROFILE") != nullptr) {
      const std::size_t n = beat->element_count();
      const int rows = n_real > 0 ? n_real : 1;
      const int cols = rows > 0 ? (int)(n / (std::size_t)rows) : 0;
      double s2 = 0.0, s1 = 0.0;
      std::vector<double> rrms((std::size_t)rows, 0.0);
      const auto* p = static_cast<const std::uint16_t*>(
          (const void*)beat->as_u8());
      for (int r = 0; r < rows; ++r) {
        double rs = 0.0;
        for (int c = 0; c < cols; ++c) {
          const std::size_t k = (std::size_t)r * cols + c;
          const float v = (cdt == TensorBeat::DType::Bf16)
                              ? bf16_to_f32_dbg_(p[k])
                              : (float)((const _Float16*)p)[k];
          s2 += (double)v * v;
          s1 += (double)v;
          rs += (double)v * v;
        }
        rrms[(std::size_t)r] = cols > 0 ? std::sqrt(rs / cols) : 0.0;
      }
      const double rms = n > 0 ? std::sqrt(s2 / (double)n) : 0.0;
      double rmin = 1e30, rmax = -1e30, rsum = 0.0;
      for (double v : rrms) {
        rmin = std::min(rmin, v); rmax = std::max(rmax, v); rsum += v;
      }
      const double rmean = rows > 0 ? rsum / rows : 0.0;
      double rvar = 0.0;
      for (double v : rrms) { rvar += (v - rmean) * (v - rmean); }
      rvar = rows > 0 ? std::sqrt(rvar / rows) : 0.0;
      session()->info(fmt(
          "cond-profile rows {} x {}  rms {:.5f}  mean {:+.6f}  "
          "row-rms [{:.4f} .. {:.4f}] sd {:.5f}", rows, cols, rms,
          n > 0 ? s1 / (double)n : 0.0, rmin, rmax, rvar));
    }
    co_await ctx.write(0, std::move(beat));
  }
  ++_emitted;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(DiffusionConditionerStage)
VPIPE_REGISTER_SPEC(DiffusionConditionerStage, kSpec)

}  // namespace vpipe
