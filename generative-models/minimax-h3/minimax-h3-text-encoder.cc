#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/context-manager.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <cstring>
#include <filesystem>
#include <fstream>

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// The Comfy-Org single-file encoder's `__metadata__` key. Unlike the DiT
// and the two VAEs, what it holds is NOT a config -- only
// `{"num_hidden_layers": 50, "output": "unnormalized_hidden_after_layer_50"}`,
// i.e. the tap and nothing else. Every architecture field has to come
// from the TENSOR SHAPES instead (see comfy_config_).
constexpr const char* kComfyKey = "minimax_h3_te";

// bf16 is the conditioner's residual element, so an f16 tower's rows
// have to be narrowed on the way in. Truncating (rather than rounding)
// matches every other splice in the tree.
inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)(u >> 16);
}

// The released Qwen3-VL-32B vision tower, and the two PIXEL BUDGETS
// MiniMax-H3's processors carry.
//
// The budgets are the part worth stating: the tower's own defaults are
// Qwen's generic ones (3136 / 1003520 for images), and H3 ships bounds
// an order of magnitude larger, with a SECOND pair for video. Left at
// the defaults a reference image is downscaled to a fraction of the
// resolution the model was conditioned on -- which produces a perfectly
// ordinary-looking embedding from the wrong pixels.
//
// From processor/preprocessor_config.json and
// processor/video_preprocessor_config.json, alongside the vision_config
// a diffusers checkout also carries (and a Comfy-Org repack does not).
void
h3_vision_defaults_(MetalQwenVisionEncoder::Config& v)
{
  v.depth          = 27;
  v.hidden         = 1152;
  v.n_heads        = 16;         // head_dim 72 -- the bd128 zero-pad path
  v.intermediate   = 4304;
  v.patch_size     = 16;
  v.spatial_merge  = 2;
  v.temporal_patch = 2;
  v.out_hidden     = 5120;       // == the DiT's text_dim
  v.num_pos_embed  = 2304;
  v.deepstack_indexes = {8, 16, 24};
  v.min_pixels       = 65536;
  v.max_pixels       = 16777216;
  v.video_min_pixels = 4096;
  v.video_max_pixels = 25165824;
  // The conditioner runs bf16 end to end, so the tower has to as well:
  // its output is spliced into a bf16 residual stream, and an f16 tower
  // would carry a different rounding into it than the reference's.
  v.use_bf16 = true;
}

// Size a Config from the Comfy-Org encoder's TENSOR SHAPES.
//
// Every other Comfy-Org component here embeds a real config; this one
// embeds only the tap, so the architecture has to be measured. That is
// sound for the fields below -- each is a matrix dimension, so reading
// it wrong would not load -- but NOT for the two constants at the
// bottom, which no shape encodes. They keep the released Qwen3-VL-32B
// values, and a checkpoint that changed either would load and produce
// subtly wrong conditioning rather than fail, so they are named here
// instead of being silently inherited from a default.
//
// The file is also PRUNED to the tap: 50 layers, not 64. So the layer
// count read here is both `total_layers` and `tap` -- there is nothing
// past it to tap at.
bool
comfy_config_(const std::string& file, MiniMaxH3TextEncoder::Config& out,
              std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  auto wts = MetalLlamaWeights::open(file);
  if (!wts) { return fail("cannot open " + file); }
  auto dim = [&](const char* nm, int axis) -> int {
    const auto* i = wts->info(nm);
    if (i == nullptr || (int)i->shape.size() <= axis) { return 0; }
    return (int)i->shape[axis];
  };
  // The weight prefix is "model." here, not the diffusers
  // "model.language_model." -- Comfy-Org's conversion drops the
  // language_model segment. MetalQwenModel re-derives this from the
  // tensor names on its own; it is set explicitly so the probes below
  // and the embedding bind in load() agree with it.
  MetalQwenModel::Config& c = out.lm;
  c.weight_prefix = "model.";
  c.model_seg     = "";

  int layers = 0;
  while (dim(("model.layers." + std::to_string(layers) +
              ".input_layernorm.weight").c_str(), 0) > 0) {
    ++layers;
  }
  if (layers <= 0) { return fail(file + ": no model.layers.N.* tensors"); }
  out.total_layers = layers;
  out.tap          = layers;
  h3_vision_defaults_(out.vision);

  c.vocab     = dim("model.embed_tokens.weight", 0);
  c.hidden    = dim("model.embed_tokens.weight", 1);
  c.head_dim  = dim("model.layers.0.self_attn.q_norm.weight", 0);
  const int q  = dim("model.layers.0.self_attn.q_proj.weight", 0);
  const int kv = dim("model.layers.0.self_attn.k_proj.weight", 0);
  c.ffn_inner = dim("model.layers.0.mlp.gate_proj.weight", 0);
  if (c.hidden <= 0 || c.head_dim <= 0 || q <= 0 || kv <= 0 ||
      c.ffn_inner <= 0 || c.vocab <= 0) {
    return fail(file + ": cannot measure the encoder geometry from its "
                       "tensor shapes");
  }
  c.n_heads    = q / c.head_dim;
  c.n_kv_heads = kv / c.head_dim;
  c.rotary_dim = c.head_dim;
  out.text_dim = c.hidden;
  if (c.n_heads <= 0 || c.n_kv_heads <= 0 ||
      c.n_heads * c.head_dim != q || c.n_kv_heads * c.head_dim != kv) {
    return fail(file + ": q/k projection widths are not a whole number of "
                       "head_dim heads");
  }
  // NOT derivable from any shape. These are the released Qwen3-VL-32B
  // values, which is what this checkpoint is a repack of.
  c.rope_theta = 5.0e6f;
  c.rms_eps    = 1e-6f;
  return true;
}

}  // namespace

std::string
MiniMaxH3TextEncoder::resolve_encoder_dir(const std::string& path)
{
  namespace fs = std::filesystem;
  fs::path p(path);
  {
    const std::string f = comfy::resolve_component(path, "text_encoders",
                                                   kComfyKey, {"qwen3vl"});
    if (!f.empty()) { return f; }
  }
  if (!fs::is_directory(p)) { return path; }
  // A QUANTIZED repack: model-quantize keeps the repack's role subdirs and
  // writes the quantized component as a directory checkpoint inside its own,
  // so `text_encoders/` holds config.json + shards rather than one
  // .safetensors. A source repo has no config.json there, so this and the
  // repack probe above never both match.
  if (fs::exists(p / "text_encoders" / "config.json")) {
    return (p / "text_encoders").string();
  }
  if (fs::exists(p / "config.json") &&
      (fs::exists(p / "model.safetensors.index.json") ||
       fs::exists(p / "model.safetensors"))) {
    return p.string();                                  // already text_encoder
  }
  if (fs::exists(p / "text_encoder" / "config.json")) {
    return (p / "text_encoder").string();               // a partition root
  }
  if (fs::exists(p / "FL2VA" / "text_encoder" / "config.json")) {
    return (p / "FL2VA" / "text_encoder").string();
  }
  return path;
}

bool
MiniMaxH3TextEncoder::config_from_json(const std::string& enc_dir, Config& out,
                                       std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const fs::path dir(resolve_encoder_dir(enc_dir));
  // The Comfy-Org single file: no config.json, and its `__metadata__`
  // carries only the tap -- so the geometry is measured off the tensor
  // shapes instead. See comfy_config_.
  if (comfy::is_component(dir.string(), kComfyKey)) {
    MiniMaxH3TextEncoder::Config c;
    if (!comfy_config_(dir.string(), c, err)) { return false; }
    // The invariant fields, shared with the diffusers path below: a
    // stock Qwen3-VL text stack, backbone only, read for an
    // intermediate hidden state rather than for tokens.
    MetalQwenModel::Config& lm = c.lm;
    lm.full_attn_interval = 1;
    lm.dense = true;
    lm.use_bf16 = true;
    lm.qk_norm = true;
    lm.attention_bias = false;
    lm.attn_output_gate = false;
    lm.zero_centered_norm = false;
    lm.tie_embeddings = false;
    lm.backbone_only = true;
    lm.max_seq = 4096;
    lm.page_tokens = 256;
    lm.n_layers = c.tap;
    out = std::move(c);
    return true;
  }
  std::ifstream f(dir / "config.json");
  if (!f) { return fail("cannot open " + (dir / "config.json").string()); }
  FlexData cfg;
  try {
    cfg = FlexData::from_json(f);
  } catch (...) {
    return fail("cannot parse " + (dir / "config.json").string());
  }
  if (!cfg.is_object()) { return fail("encoder config is not a JSON object"); }
  auto root = cfg.as_object();

  MetalQwenModel::Config& c = out.lm;
  // A stock Qwen3-VL text stack: dense SwiGLU, GQA with per-head q/k
  // RMS, no attention bias and no output gate.
  c.full_attn_interval = 1;
  c.dense = true;
  c.use_bf16 = true;
  c.qk_norm = true;
  c.attention_bias = false;
  c.attn_output_gate = false;
  c.zero_centered_norm = false;
  c.tie_embeddings = false;
  // Nothing past the tap runs, so there is no head to load. This is also
  // what lets the layer count below be the TAP rather than the
  // checkpoint's depth.
  c.backbone_only = true;
  c.weight_prefix = "model.language_model.";
  c.model_seg = "";
  c.max_seq = 4096;
  c.page_tokens = 256;
  c.rms_eps = 1e-6f;

  if (!root.contains("text_config")) {
    return fail("no text_config in the encoder config: this does not look "
                "like a Qwen3-VL checkpoint");
  }
  FlexData tcd = root.at("text_config");
  if (!tcd.is_object()) { return fail("text_config is not an object"); }
  auto o = tcd.as_object();
  auto geti = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  auto getf = [&](const char* k, float d) {
    return o.contains(k) ? (float)o.at(k).as_real(d) : d;
  };
  h3_vision_defaults_(out.vision);
  out.total_layers = geti("num_hidden_layers", 64);
  c.hidden      = geti("hidden_size", 5120);
  c.n_heads     = geti("num_attention_heads", 64);
  c.n_kv_heads  = geti("num_key_value_heads", 8);
  c.head_dim    = geti("head_dim", c.n_heads > 0 ? c.hidden / c.n_heads : 128);
  c.rotary_dim  = c.head_dim;
  c.ffn_inner   = geti("intermediate_size", 25600);
  c.vocab       = geti("vocab_size", 151936);
  c.rope_theta  = getf("rope_theta", 5.0e6f);
  c.rms_eps     = getf("rms_norm_eps", c.rms_eps);
  out.text_dim  = c.hidden;
  if (root.contains("tie_word_embeddings")) {
    c.tie_embeddings = root.at("tie_word_embeddings").as_bool(false);
  }
  // A quantized encoder (model-quantize target=text_encoder). The loader
  // auto-detects quantized-vs-dense weights, but it needs the bit width
  // to pick the w4g64 vs w8g64 kernel -- left at the default a 8-bit
  // checkpoint decodes with the 4-bit kernel and returns garbage rather
  // than failing, so read it rather than assuming.
  if (root.contains("quantization")) {
    FlexData q = root.at("quantization");
    if (q.is_object()) {
      auto qo = q.as_object();
      const int b = qo.contains("bits") ? (int)qo.at("bits").as_int(0) : 0;
      if (b == 4 || b == 8) { c.quant_bits = b; out.quantized = true; }
    }
  }
  if (out.tap <= 0 || out.tap > out.total_layers) {
    return fail("tap layer " + std::to_string(out.tap) + " is outside the "
                "checkpoint's " + std::to_string(out.total_layers) +
                " layers");
  }
  // The vision tower, when the checkpoint describes one. Its defaults
  // are already the released geometry (see h3_vision_defaults_), so a
  // config without a vision_config is not an error -- Comfy-Org's
  // repack has none and still carries all 351 tower tensors.
  if (root.contains("vision_config")) {
    FlexData v = root.at("vision_config");
    if (v.is_object()) {
      auto vo = v.as_object();
      auto vget = [&](const char* k, int d) {
        return vo.contains(k) ? (int)vo.at(k).as_int(d) : d;
      };
      MetalQwenVisionEncoder::Config& vc = out.vision;
      vc.depth          = vget("depth", vc.depth);
      vc.hidden         = vget("hidden_size", vc.hidden);
      vc.n_heads        = vget("num_heads", vc.n_heads);
      vc.patch_size     = vget("patch_size", vc.patch_size);
      vc.spatial_merge  = vget("spatial_merge_size", vc.spatial_merge);
      vc.temporal_patch = vget("temporal_patch_size", vc.temporal_patch);
      vc.out_hidden     = vget("out_hidden_size", vc.out_hidden);
      vc.num_pos_embed  = vget("num_position_embeddings", vc.num_pos_embed);
      vc.intermediate   = vget("intermediate_size", vc.intermediate);
      if (vo.contains("deepstack_visual_indexes")) {
        FlexData di = vo.at("deepstack_visual_indexes");
        if (di.is_array()) {
          std::vector<int> ix;
          for (auto e : di.as_array()) { ix.push_back((int)e.as_int(-1)); }
          if (!ix.empty()) { vc.deepstack_indexes = std::move(ix); }
        }
      }
    }
  }

  // Load exactly as deep as the tap. Layers past it never run.
  c.n_layers = out.tap;
  return true;
}

MiniMaxH3TextEncoder::~MiniMaxH3TextEncoder() = default;

std::unique_ptr<MetalQwenVisionEncoder>
MiniMaxH3TextEncoder::load_vision(const std::string& enc_dir, MetalCompute* mc,
                                  const Config& cfg, std::string* err)
{
  auto fail = [&](std::string m) -> std::unique_ptr<MetalQwenVisionEncoder> {
    if (err != nullptr) { *err = std::move(m); }
    return nullptr;
  };
  if (mc == nullptr) { return fail("no metal compute"); }
  const std::string dir = resolve_encoder_dir(enc_dir);
  auto ws = WeightSet::open(dir, mc->session());
  if (!ws) { return fail("cannot open " + dir); }

  // Two spellings of the same tower, as everywhere else in this file:
  // the diffusers checkpoint nests it under `model.visual.`, Comfy-Org's
  // conversion drops the `model.` segment. Probed rather than assumed
  // because a wrong prefix loads NOTHING and the tower would come back
  // as a plain load failure with no hint which layout was expected.
  MetalQwenVisionEncoder::Config vc = cfg.vision;
  const char* found = nullptr;
  for (const char* p : {"visual.", "model.visual."}) {
    if (ws->src().info(std::string(p) + "patch_embed.proj.weight") != nullptr) {
      found = p;
      break;
    }
  }
  if (found == nullptr) {
    return fail("no vision tower in " + dir +
                " (looked for visual. and model.visual.)");
  }
  vc.weight_prefix = found;

  auto tower = MetalQwenVisionEncoder::load(std::move(ws), mc, vc);
  if (!tower) { return fail("the vision tower failed to build"); }
  return tower;
}

std::unique_ptr<MiniMaxH3TextEncoder>
MiniMaxH3TextEncoder::load(const std::string& enc_dir, MetalCompute* mc,
                           SessionContextIntf* session, const Config& cfg)
{
  if (mc == nullptr) { return nullptr; }
  const std::string dir = resolve_encoder_dir(enc_dir);
  auto m = std::unique_ptr<MiniMaxH3TextEncoder>(new MiniMaxH3TextEncoder());
  m->_mc = mc;
  m->_cfg = cfg;
  if (m->_cfg.tap <= 0 || m->_cfg.tap > m->_cfg.total_layers) {
    return nullptr;
  }
  m->_cfg.lm.n_layers = m->_cfg.tap;

  m->_ws = open_weight_set(dir, session);
  if (!m->_ws) { return nullptr; }
  m->_lm = MetalQwenModel::load(m->_ws, mc, m->_cfg.lm);
  if (!m->_lm) { return nullptr; }

  // The embedding table is not part of a backbone-only load, so bind it
  // here. It comes from the same set, so a peer naming this checkpoint
  // shares the bytes rather than mapping a second copy. Two spellings:
  // the diffusers checkpoint nests the text stack under
  // `model.language_model.`, Comfy-Org's conversion drops that segment.
  for (const char* nm : {"model.language_model.embed_tokens.weight",
                         "model.embed_tokens.weight"}) {
    m->_embed = m->_ws->tensor(nm, mc, WeightSet::Residency::Copied);
    if (!m->_embed.empty()) { break; }
  }
  if (m->_embed.empty()) { return nullptr; }

  // The tokenizer. A diffusers text_encoder/ ships its own; the
  // Comfy-Org repack ships NONE -- it is weights only -- so the search
  // also covers the repo root and a tokenizer/ beside it, which is
  // where a user pairing the repack with MiniMaxAI's tokenizer would
  // put one. Missing, this fails LOUDLY rather than returning a nullptr
  // the caller has to guess the meaning of: encode(prompt) genuinely
  // cannot work, and "no tokenizer" is a fixable setup problem where a
  // bare load failure reads as an unsupported checkpoint.
  namespace fs = std::filesystem;
  const fs::path d(dir);
  const fs::path base = fs::is_directory(d) ? d : d.parent_path();
  const fs::path root = base.parent_path();
  for (const fs::path& p : {base / "tokenizer.json",
                            base / "tokenizer" / "tokenizer.json",
                            root / "tokenizer.json",
                            root / "tokenizer" / "tokenizer.json"}) {
    if (fs::exists(p)) {
      m->_tok = Tokenizer::from_huggingface_json(p.string(), session);
      if (m->_tok) { break; }
    }
  }
  if (!m->_tok) {
    if (session != nullptr) {
      session->warn(fmt(
          "MiniMaxH3TextEncoder: no tokenizer.json for '{}' -- looked in "
          "'{}' and '{}' (and their tokenizer/ subdirs). The Comfy-Org "
          "repack ships weights only; copy MiniMaxAI/MiniMax-H3's "
          "FL2VA/text_encoder/tokenizer.json next to it",
          dir, base.string(), root.string()));
    }
    return nullptr;
  }

  if (session != nullptr) {
    session->log_debug(
        fmt("MiniMaxH3TextEncoder: Qwen3-VL hidden {} x {} heads ({} kv), "
            "tapped at layer {} of {} -- {} layers loaded",
            m->_cfg.lm.hidden, m->_cfg.lm.n_heads, m->_cfg.lm.n_kv_heads,
            m->_cfg.tap, m->_cfg.total_layers, m->_cfg.lm.n_layers));
  }
  return m;
}

std::vector<std::int32_t>
MiniMaxH3TextEncoder::tokenize(std::string_view prompt) const
{
  if (!_tok) { return {}; }
  // Tokenizer::encode adds no BOS/EOS of its own, which IS the
  // reference's `add_special_tokens=False`.
  return _tok->encode(prompt);
}

SharedBuffer
MiniMaxH3TextEncoder::encode_ids(const std::vector<std::int32_t>& ids,
                                 std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  if (!_lm || _embed.empty()) { return fail("encoder is not loaded"); }
  const int n = (int)ids.size();
  if (n <= 0) { return fail("empty prompt"); }
  if (n > _cfg.lm.max_seq) {
    return fail("prompt is " + std::to_string(n) + " tokens, past the " +
                std::to_string(_cfg.lm.max_seq) + "-token pool");
  }
  const int H = _cfg.lm.hidden;
  const std::size_t vocab = _embed.byte_size() / ((std::size_t)H * 2);

  SharedBuffer x = _mc->make_shared_buffer((std::size_t)n * H * 2);
  if (x.empty()) { return fail("embedding allocation failed"); }
  {
    const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    for (int i = 0; i < n; ++i) {
      const std::int32_t id = ids[(std::size_t)i];
      if (id < 0 || (std::size_t)id >= vocab) {
        return fail("token id " + std::to_string(id) + " is outside the "
                    "vocabulary");
      }
      std::memcpy(xb + (std::size_t)i * H * 2,
                  tbl + (std::size_t)id * H * 2, (std::size_t)H * 2);
    }
  }

  // The tap is 0-INDEXED over layers; HF's output_hidden_states[k] is
  // the residual after layer k-1. So conditioning on hidden_states[50]
  // means tapping after 0-indexed layer 49.
  const std::vector<int> taps{_cfg.tap - 1};
  ContextManager* cm = _lm->context_manager();
  if (cm == nullptr) { return fail("encoder has no context manager"); }
  const ContextId cid = cm->acquire_root();
  SharedBuffer h = _lm->forward_embeddings_taps(cid, x, n, taps);
  cm->release(cid);
  if (h.empty()) { return fail("encoder forward failed"); }
  return h;
}

std::vector<float>
MiniMaxH3TextEncoder::video_block_seconds(int num_sampled_frames,
                                          float sample_fps, int temporal_patch)
{
  std::vector<float> out;
  if (num_sampled_frames <= 0 || !(sample_fps > 0.0f) || temporal_patch <= 0) {
    return out;
  }
  // The reference pads the timestamp list by REPEATING the last one so
  // it divides, then labels each group with the mean of its members --
  // so a trailing odd frame is labelled with its own time, not with a
  // time no frame is at.
  const int padded = ((num_sampled_frames + temporal_patch - 1) /
                      temporal_patch) * temporal_patch;
  auto stamp = [&](int j) {
    return (float)(std::min(j, num_sampled_frames - 1)) / sample_fps;
  };
  for (int i = 0; i < padded; i += temporal_patch) {
    out.push_back((stamp(i) + stamp(i + temporal_patch - 1)) * 0.5f);
  }
  return out;
}

bool
MiniMaxH3TextEncoder::build_presentation(const std::vector<Reference>& refs,
                                         std::string_view prompt,
                                         Presentation* out,
                                         std::string* err) const
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (out == nullptr) { return fail("null output"); }
  if (!_tok) { return fail("encoder has no tokenizer"); }
  if (refs.empty()) {
    return fail("ref2va needs at least one reference; use encode() for a "
                "text-only prompt");
  }

  const std::int32_t vs = _tok->special_token_id("<|vision_start|>");
  const std::int32_t ve = _tok->special_token_id("<|vision_end|>");
  const std::int32_t ip = _tok->special_token_id("<|image_pad|>");
  const std::int32_t vp = _tok->special_token_id("<|video_pad|>");
  if (vs < 0 || ve < 0 || ip < 0 || vp < 0) {
    return fail("this tokenizer has no Qwen3-VL vision markers");
  }

  Presentation P;
  std::vector<std::int32_t>& ids = P.ids;
  std::vector<int>&          tags = P.tags;
  auto emit_text = [&](const std::string& str) {
    const std::vector<std::int32_t> t = _tok->encode(str);
    ids.insert(ids.end(), t.begin(), t.end());
    tags.insert(tags.end(), t.size(), minimax_h3::kTextTag);
  };
  auto emit_vision = [&](std::int32_t pad, int rows,
                         const MetalQwenVisionEncoder::Result* vis, int cell,
                         int mh, int mw) {
    ids.push_back(vs);
    tags.push_back(minimax_h3::kVideoTag);
    Presentation::Run r;
    r.start = (int)ids.size();
    r.rows  = rows;
    r.mh = mh; r.mw = mw; r.cell = cell; r.vision = vis;
    P.runs.push_back(r);
    ids.insert(ids.end(), (std::size_t)rows, pad);
    tags.insert(tags.end(), (std::size_t)rows, minimax_h3::kVideoTag);
    ids.push_back(ve);
    tags.push_back(minimax_h3::kVideoTag);
  };

  int n_image = 0, n_video = 0, n_audio = 0;
  for (const Reference& r : refs) {
    // A video that carries sound is labelled "<Audio j>: " BEFORE its
    // "<Video k>: ", mirroring the order its rows are packed in.
    if (r.has_audio) {
      emit_text("<Audio " + std::to_string(++n_audio) + ">: ");
    }
    if (r.kind == Reference::Kind::kAudio) { continue; }
    if (r.vision == nullptr || r.vision->n_tokens <= 0) {
      return fail("a visual reference has no encoded vision");
    }
    const int S  = _cfg.vision.spatial_merge > 0 ? _cfg.vision.spatial_merge : 1;
    const int mh = r.vision->grid_h / S;
    const int mw = r.vision->grid_w / S;
    const int gt = r.vision->grid_t > 0 ? r.vision->grid_t : 1;
    const int per_cell = r.vision->n_tokens / gt;
    if (per_cell <= 0 || per_cell * gt != r.vision->n_tokens) {
      return fail("the tower's token count is not a whole number of cells");
    }
    if (r.kind == Reference::Kind::kImage) {
      if (gt != 1) { return fail("an image reference has more than one cell"); }
      emit_text("<Picture " + std::to_string(++n_image) + ">: ");
      emit_vision(ip, per_cell, r.vision, 0, mh, mw);
    } else {
      if ((int)r.block_seconds.size() != gt) {
        return fail("a video reference has " + std::to_string(gt) +
                    " blocks but " + std::to_string(r.block_seconds.size()) +
                    " timestamps");
      }
      emit_text("<Video " + std::to_string(++n_video) + ">: ");
      for (int cell = 0; cell < gt; ++cell) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "<%.1f seconds>",
                      (double)r.block_seconds[(std::size_t)cell]);
        emit_text(buf);
        emit_vision(vp, per_cell, r.vision, cell, mh, mw);
      }
    }
  }
  emit_text(std::string(prompt));

  const int n = (int)ids.size();
  if (n <= 0) { return fail("the presentation tokenized to nothing"); }

  // The rotary layout. Qwen3-VL derives this itself from the token-type
  // ids; reproduced here because the tap is taken off the backbone
  // directly. Per MODALITY RUN: a text run advances one position per
  // token, a vision run puts every row at the run's base on the t axis
  // and at base + (row, col) on h/w, and the text after it resumes at
  // base + max(mh, mw) -- NOT one past the last vision position.
  //
  // VERIFIED against transformers' own get_rope_index on a presentation
  // of this exact shape (h3ref/probe_presentation_rope.py), including
  // the case that decides it: a video's cells are separated by their
  // timestamp labels, so each cell is its OWN run laid out from its own
  // base rather than one run spanning the clip.
  P.mrope.assign((std::size_t)3 * n, 0);
  {
    int cur = 0;
    std::size_t nr = 0;
    for (int i = 0; i < n;) {
      if (nr < P.runs.size() && i == P.runs[nr].start && P.runs[nr].rows > 0) {
        const Presentation::Run& r = P.runs[nr];
        const int w = r.mw > 0 ? r.mw : 1;
        for (int j = 0; j < r.rows && i < n; ++j, ++i) {
          P.mrope[(std::size_t)i]             = cur;
          P.mrope[(std::size_t)n + i]         = cur + j / w;
          P.mrope[(std::size_t)2 * n + i]     = cur + j % w;
        }
        cur += std::max(r.mh > 0 ? r.mh : 1, w);
        ++nr;
      } else {
        P.mrope[(std::size_t)i]             = cur;
        P.mrope[(std::size_t)n + i]         = cur;
        P.mrope[(std::size_t)2 * n + i]     = cur;
        ++cur;
        ++i;
      }
    }
  }

  *out = std::move(P);
  return true;
}

SharedBuffer
MiniMaxH3TextEncoder::encode_references(const std::vector<Reference>& refs,
                                        std::string_view prompt,
                                        std::vector<int>* token_tags,
                                        int* n_tokens, std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  if (!_lm || _embed.empty()) { return fail("encoder is not loaded"); }

  Presentation P;
  if (!build_presentation(refs, prompt, &P, err)) { return {}; }
  const std::vector<std::int32_t>& ids = P.ids;
  const std::vector<Presentation::Run>& runs = P.runs;
  const int n = P.size();
  if (n > _cfg.lm.max_seq) {
    return fail("the presentation is " + std::to_string(n) + " tokens, past "
                "the " + std::to_string(_cfg.lm.max_seq) + "-token pool");
  }

  // ---- 2. the embedding stream ---------------------------------------
  const int H = _cfg.lm.hidden;
  const std::size_t vocab = _embed.byte_size() / ((std::size_t)H * 2);
  SharedBuffer x = _mc->make_shared_buffer((std::size_t)n * H * 2);
  if (x.empty()) { return fail("embedding allocation failed"); }
  {
    const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    for (int i = 0; i < n; ++i) {
      const std::int32_t id = ids[(std::size_t)i];
      if (id < 0 || (std::size_t)id >= vocab) {
        return fail("token id " + std::to_string(id) + " is outside the "
                    "vocabulary");
      }
      std::memcpy(xb + (std::size_t)i * H * 2,
                  tbl + (std::size_t)id * H * 2, (std::size_t)H * 2);
    }
  }
  // The tower's rows replace the pad embeddings. A video reference's
  // tokens are one contiguous [grid_t * per_cell, out_hidden] block, so
  // a cell reads from its own offset inside it.
  {
    auto* xh = static_cast<std::uint16_t*>(x.contents());
    for (const Presentation::Run& r : runs) {
      if (r.vision->out_hidden != H) {
        return fail("the vision tower emits " +
                    std::to_string(r.vision->out_hidden) + " channels but the "
                    "conditioner is " + std::to_string(H) + " wide");
      }
      const bool vt_bf16 = r.vision->is_bf16;
      const auto* vt = static_cast<const std::uint16_t*>(
          r.vision->embeddings.contents());
      const std::size_t base = (std::size_t)r.cell * r.rows * H;
      for (int j = 0; j < r.rows; ++j) {
        const std::uint16_t* src = vt + base + (std::size_t)j * H;
        std::uint16_t* dst = xh + (std::size_t)(r.start + j) * H;
        if (vt_bf16) {
          std::memcpy(dst, src, (std::size_t)H * 2);
        } else {
          for (int h = 0; h < H; ++h) {
            _Float16 hf;
            std::memcpy(&hf, &src[h], 2);
            dst[h] = f32_to_bf16_((float)hf);
          }
        }
      }
    }
  }

  const std::vector<std::int32_t>& pos = P.mrope;

  // ---- 4. deepstack ----------------------------------------------------
  //
  // The tower's per-layer features are added back at the vision rows
  // after the first few LM layers. The runs are disjoint, so this needs
  // the SEGMENTED form -- and the features of a video reference are one
  // buffer across its cells, so a cell's feature offset is its own.
  MetalQwenModel::DeepstackInject ds;
  std::vector<SharedBuffer> ds_feats;
  const std::size_t n_ds =
      runs.empty() ? 0 : runs[0].vision->deepstack.size();
  bool ds_ok = n_ds > 0;
  for (const Presentation::Run& r : runs) {
    if (r.vision->deepstack.size() != n_ds) { ds_ok = false; }
  }
  if (ds_ok) {
    // One CONCATENATED feature buffer per injected layer, in run order,
    // which is the layout DeepstackInject::Seg indexes with `feat_row`.
    int total = 0;
    for (const Presentation::Run& r : runs) { total += r.rows; }
    for (std::size_t d = 0; d < n_ds && ds_ok; ++d) {
      SharedBuffer f = _mc->make_shared_buffer((std::size_t)total * H * 2);
      if (f.empty()) { ds_ok = false; break; }
      auto* fb = static_cast<std::uint16_t*>(f.contents());
      int row = 0;
      for (const Presentation::Run& r : runs) {
        const auto* src = static_cast<const std::uint16_t*>(
            r.vision->deepstack[d].contents());
        const std::size_t base = (std::size_t)r.cell * r.rows * H;
        for (int j = 0; j < r.rows; ++j, ++row) {
          const std::uint16_t* s = src + base + (std::size_t)j * H;
          std::uint16_t* dst = fb + (std::size_t)row * H;
          if (r.vision->is_bf16) {
            std::memcpy(dst, s, (std::size_t)H * 2);
          } else {
            for (int h = 0; h < H; ++h) {
              _Float16 hf;
              std::memcpy(&hf, &s[h], 2);
              dst[h] = f32_to_bf16_((float)hf);
            }
          }
        }
      }
      ds_feats.push_back(std::move(f));
    }
  }
  if (ds_ok && ds_feats.size() == n_ds) {
    int feat = 0;
    for (const Presentation::Run& r : runs) {
      ds.segs.push_back({r.start, r.rows, feat});
      feat += r.rows;
    }
    ds.row0 = runs.front().start;
    ds.rows = feat;
    for (std::size_t d = 0; d < n_ds; ++d) {
      ds.feats.push_back(&ds_feats[d]);
      ds.layers.push_back((int)d);
    }
  }

  // ---- 5. the tap ------------------------------------------------------
  const std::vector<int> taps{_cfg.tap - 1};
  ContextManager* cm = _lm->context_manager();
  if (cm == nullptr) { return fail("encoder has no context manager"); }
  const ContextId cid = cm->acquire_root();
  SharedBuffer h = _lm->forward_embeddings_taps_mrope(
      cid, x, n, pos, taps, /*key_valid_len=*/0,
      ds.feats.empty() ? nullptr : &ds);
  cm->release(cid);
  if (h.empty()) { return fail("encoder forward failed"); }

  if (token_tags != nullptr) { *token_tags = std::move(P.tags); }
  if (n_tokens != nullptr) { *n_tokens = n; }
  return h;
}

SharedBuffer
MiniMaxH3TextEncoder::encode(std::string_view prompt, int* n_tokens,
                             std::string* err)
{
  const std::vector<std::int32_t> ids = tokenize(prompt);
  if (ids.empty()) {
    if (err != nullptr) { *err = "prompt tokenized to nothing"; }
    return {};
  }
  SharedBuffer h = encode_ids(ids, err);
  if (!h.empty() && n_tokens != nullptr) { *n_tokens = (int)ids.size(); }
  return h;
}

}  // namespace genai
}  // namespace vpipe
