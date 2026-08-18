#include "stages/model-quantize-stage.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/temp-root.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/boogu/metal-boogu-calibration.h"
#include "generative-models/flux2/metal-flux2-calibration.h"
#include "generative-models/krea2/metal-krea2-calibration.h"
#include "generative-models/qwen-image/metal-qwen-image-calibration.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/quantize/arch-detect.h"
#include "generative-models/quantize/calibration.h"
#include "generative-models/quantize/model-quantizer.h"
#include "generative-models/tokenizer.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "interfaces/ui-delegate-intf.h"
#include "stages/model-detect.h"
#include "stages/model-registry.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {
// Defined below; forward-declared so the constructor can validate `target`.
std::string t2i_target_subdir_(const std::string& target);
}

ModelQuantizeStage::ModelQuantizeStage(
    const SessionContextIntf* s,
    std::string               id,
    std::vector<InEdge>       iports,
    FlexData                  config)
  : TypedStage<ModelQuantizeStage>(s, std::move(id), std::move(iports),
                                   std::move(config))
{
  // Validation is deferred to launch (see Stage::fail_config). Attribute
  // defaults live in kSpec.attrs; attr_* resolves the configured value else
  // the default.
  _src_model     = attr_str("src_model");
  _output_name   = attr_str("output_name");
  _arch          = attr_str("arch");
  _target        = attr_str("target");
  _bits          = static_cast<int>(attr_uint("bits"));
  _group_size    = static_cast<int>(attr_uint("group_size"));
  _skip_existing = attr_bool("skip_existing");
  _awq           = attr_bool("awq");
  _awq_clip      = attr_bool("awq_clip");
  _calib_dir     = attr_str("calib_dir");
  _mixed         = attr_bool("mixed");
  _high_bits     = static_cast<int>(attr_uint("high_bits"));
  _mixed_frac    = static_cast<float>(attr_real("mixed_frac"));
  _layer_prefix  = attr_str("layer_prefix");
  _quant_exclude = attr_str("quant_exclude");
  _quant_modulation = attr_bool("quant_modulation");
  _quant_vision     = attr_bool("quant_vision");
  _klein_kv      = attr_bool("klein_kv");
  _n_layers      = static_cast<int>(attr_uint("n_layers"));

  if (_src_model.empty()) {
    fail_config(fmt("ModelQuantizeStage('{}'): config.src_model is required",
                    this->id()));
  }
  if (_output_name.empty()) {
    fail_config(fmt("ModelQuantizeStage('{}'): config.output_name is required",
                    this->id()));
  }
  if (_bits != 4 && _bits != 8) {
    fail_config(fmt("ModelQuantizeStage('{}'): bits must be 4 or 8 (got {})",
                    this->id(), _bits));
  }
  if (_group_size != 32 && _group_size != 64) {
    fail_config(fmt(
        "ModelQuantizeStage('{}'): group_size must be 32 or 64 (got {})",
        this->id(), _group_size));
  }
  if (_awq_clip && !_awq) {
    fail_config(fmt(
        "ModelQuantizeStage('{}'): awq_clip requires awq=true", this->id()));
  }
  // `target` is validated at run time: its valid values depend on the model
  // (Krea-2 components dit|text_encoder|vae vs a general-LLM submodule scope
  // all|text|vision|audio|<prefix>), which is only known once src is resolved.
  // The mixed-affine decode kernels are w4g64 + w8g64 only.
  if (_mixed && (_bits != 4 || _high_bits != 8 || _group_size != 64)) {
    fail_config(fmt(
        "ModelQuantizeStage('{}'): mixed requires bits=4, high_bits=8, "
        "group_size=64", this->id()));
  }
  // NOTE: arch / n_layers / layer_prefix and the awq-calibratable check are
  // resolved at run time (quantize_once) -- they auto-detect from the source
  // config.json, which is only available after src_model is resolved.

  allocate_oports(spec().oports.size());
}

namespace {

// On-device auto-calibration corpus shape: an effective AWQ pass at 4-bit
// needs ~128 sequences of ~512 tokens. The text comes from the curated
// built-in corpus (genai::build_builtin_calibration_corpus).
constexpr int kCalibSeqs   = 128;
constexpr int kCalibSeqLen = 512;

// The text-to-image DiTs (Krea-2's Qwen-Image MMDiT and FLUX.2's FLUX-topology
// transformer) are diffusion transformers, not LMs: the config.json carries a
// `_class_name` and no LM stack (no model_type / layers.N / embed_tokens), so
// the LM arch-detect + AWQ/mixed/embedding machinery doesn't apply. They
// quantize as a plain group-affine linear pass over a per-family DiT leaf set.
//
// Map a transformer config `_class_name` to the vpipe family tag, or "" when it
// is not a recognised text-to-image DiT. Krea-2 and FLUX.2 ship the SAME
// diffusers pipeline shape (transformer/ + text_encoder/ + vae/ + tokenizer/ +
// scheduler/), so they share the self-contained, chainable pipeline-quantize
// path below; only the DiT class-name and the DiT quant leaf set differ.
std::string
dit_class_family_(const std::string& class_name)
{
  if (class_name == "Krea2Transformer2DModel") { return "krea2"; }
  if (class_name == "Flux2Transformer2DModel") { return "flux2"; }
  if (class_name == "QwenImageTransformer2DModel") { return "qwen-image-edit"; }
  // Mage-Flow's NR-MMDiT is the Qwen-Image dual-stream MMDiT under a different
  // config (see metal-mage-flow-transformer.h), so it shares that leaf set.
  if (class_name == "MageFlow") { return "mage-flow"; }
  if (class_name == "BooguImageTransformer2DModel") { return "boogu-image"; }
  // Wan video. Its two A14B experts are separate checkpoints, so each is
  // quantized by pointing src_model at that expert's own directory.
  if (class_name == "WanTransformer3DModel") { return "wan"; }
  // MiniMax-H3: one 33B single-stream stack emitting video AND audio. Its
  // partitions (FL2VA / Ref2VA) ship byte-identical transformer configs, so
  // the class name is all there is to go on -- and all there needs to be,
  // since the quant leaf set does not depend on the partition.
  if (class_name == "MiniMaxH3DiTModel") { return "minimax-h3"; }
  return {};
}

// Resolve the DiT weight dir + family from what the user pointed `src_model`
// at: either the transformer/ dir itself (its config.json IS the DiT config),
// or the stock pipeline ROOT (a diffusers layout with no top-level config.json
// -- the DiT lives under transformer/, siblings text_encoder/, tokenizer/,
// vae/). Sets *family to the vpipe tag and returns the DiT dir, or "" if it's
// not a text-to-image DiT. The rest of the DiT path relies on parent_path()
// being the pipeline root (for AWQ calibration, which loads the encoder +
// tokenizer), so returning the transformer/ subdir keeps that invariant for the
// root case too.
std::string
resolve_t2i_dit_dir_(const std::string& src_dir, std::string* family,
                     const std::string& partition = {})
{
  namespace fs = std::filesystem;
  auto family_of = [](const fs::path& cfg) -> std::string {
    std::ifstream in(cfg);
    if (!in) { return {}; }
    FlexData c = FlexData::from_json(in);
    if (!c.is_object()) { return {}; }
    auto o = c.as_object();
    if (!o.contains("_class_name")) { return {}; }
    return dit_class_family_(std::string(o.at("_class_name").as_string("")));
  };
  // A Comfy-Org single-file DiT, named directly or through the repo root
  // / diffusion_models subdir. It has no config.json -- the transformer
  // config is in the safetensors `__metadata__` -- so the quantizer is
  // handed the FILE and lifts the config out of it (model-quantizer.cc),
  // writing a directory checkpoint that records the qkv order the source
  // was in.
  {
    // `partition` is what the models DB said this reference is. It has
    // to rank first: MiniMax-H3's two partitions live in one repo and
    // differ only in the DiT filename, so without it a quantize of the
    // Ref2VA record reads FL2VA's 66 GB and writes it out under the
    // Ref2VA name. The historical `fl2va` preference stays behind it.
    const std::vector<std::string> prefer =
        partition.empty() ? std::vector<std::string>{"fl2va"}
                          : std::vector<std::string>{partition, "fl2va"};
    const std::string f = genai::comfy::resolve_component(
        src_dir, "diffusion_models", "config", prefer);
    if (!f.empty()) {
      FlexData md;
      if (genai::comfy::metadata_json(f, "config", md, nullptr) &&
          md.is_object() && md.as_object().contains("transformer")) {
        const FlexData t = md.as_object().at("transformer");
        auto to = t.as_object();
        const FlexData im_fd =
            to.contains("image_model") ? to.at("image_model") : FlexData();
        if (std::string(im_fd.as_string("")) == "minimax_h3") {
          if (family != nullptr) { *family = "minimax-h3"; }
          return f;
        }
      }
    }
  }
  // A repack this stage has ALREADY quantized: the role subdirs survive and
  // the quantized component is a directory checkpoint inside its own, so the
  // DiT is at `diffusion_models/config.json` rather than being a file with
  // an embedded config. Probed after the repack file above -- a source repo
  // has no config.json there, so the two never both match -- and before the
  // diffusers spellings below, which would otherwise not match at all and
  // send a chain's second pass down the LM path to fail with "no submodule
  // matching target 'text_encoder'".
  {
    const fs::path q = fs::path(src_dir) / "diffusion_models" / "config.json";
    const std::string fam_q = family_of(q);
    if (!fam_q.empty()) {
      if (family != nullptr) { *family = fam_q; }
      return q.parent_path().string();
    }
  }
  std::string fam = family_of(fs::path(src_dir) / "config.json");
  if (!fam.empty()) { if (family) { *family = fam; } return src_dir; }
  const fs::path sub = fs::path(src_dir) / "transformer";
  fam = family_of(sub / "config.json");
  if (!fam.empty()) { if (family) { *family = fam; } return sub.string(); }
  // Fallback: the diffusers pipeline root carries a model_index.json whose
  // `transformer` entry names the DiT class ([library, class_name]). Use it when
  // transformer/config.json is absent or has lost its _class_name (e.g. a
  // re-exported checkpoint) -- it is the canonical pipeline descriptor.
  {
    std::ifstream in(fs::path(src_dir) / "model_index.json");
    if (in) {
      FlexData mi = FlexData::from_json(in);
      if (mi.is_object()) {
        auto o = mi.as_object();
        if (o.contains("transformer")) {
          FlexData t = o.at("transformer");
          if (t.is_array()) {
            auto arr = t.as_array();
            if (arr.size() >= 2) {
              fam = dit_class_family_(std::string(arr[1].as_string("")));
              if (!fam.empty() && fs::is_directory(sub)) {
                if (family) { *family = fam; }
                return sub.string();
              }
            }
          }
        }
      }
    }
  }
  return {};
}

// Canonicalise the `target` config value to a pipeline component sub-dir name
// (transformer | text_encoder | vae), or "" if unrecognised. Empty input
// defaults to the DiT (the transformer). Family-agnostic (both share layout).
std::string
t2i_target_subdir_(const std::string& target)
{
  if (target.empty() || target == "dit" || target == "transformer") {
    return "transformer";
  }
  if (target == "text_encoder" || target == "text-encoder" ||
      target == "encoder" || target == "text") {
    return "text_encoder";
  }
  if (target == "vae") { return "vae"; }
  return {};
}

// The fast tokenizer.json that goes WITH a text encoder, wherever its
// publisher put it: beside the weights (a diffusers text_encoder/), in
// the pipeline's tokenizer/ (Krea-2, FLUX.2, and the Comfy-Org repack,
// whose borrowed companion lands there), or in processor/
// (Qwen-Image-Edit, whose tokenizer/ holds only the slow vocab.json +
// merges). "" when there is none.
//
// Shared by the two callers that need it -- AWQ auto-calibration, which
// reads it, and the output copy, which carries it -- so they cannot
// drift into disagreeing about where an encoder's tokenizer lives.
// `enc_dir` is the encoder DIRECTORY or, for a Comfy-Org repack, the
// single .safetensors FILE; only the directory form has a tokenizer
// beside it.
std::string
find_tokenizer_json_(const std::string& enc_dir, const std::string& root)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  std::vector<fs::path> cands;
  if (fs::is_directory(fs::path(enc_dir), ec) && !ec) {
    cands.push_back(fs::path(enc_dir) / "tokenizer.json");
    cands.push_back(fs::path(enc_dir) / "tokenizer" / "tokenizer.json");
  }
  // The first four are the encoder's OWN search
  // (MiniMaxH3TextEncoder::load), deliberately: a tokenizer that the
  // runtime would load and the quantizer would not find is the worst of
  // the three outcomes, because it works until the copy leaves the repo.
  // A bare tokenizer.json at the repo root is what dropping MiniMaxAI's
  // beside a repack looks like when nobody made a tokenizer/ for it.
  cands.push_back(fs::path(root) / "tokenizer.json");
  cands.push_back(fs::path(root) / "tokenizer" / "tokenizer.json");
  cands.push_back(fs::path(root) / "processor" / "tokenizer.json");
  for (const fs::path& p : cands) {
    std::error_code lec;
    if (fs::exists(p, lec) && !lec) { return p.string(); }
  }
  return {};
}

// Regular files under `p` (recursive) -- the denominator for the copy bar.
std::size_t
count_files_(const std::filesystem::path& p)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  if (fs::is_regular_file(p, ec)) { return 1; }
  if (!fs::is_directory(p, ec)) { return 0; }
  std::size_t n = 0;
  for (fs::recursive_directory_iterator it(p, ec), end; it != end;
       it.increment(ec)) {
    if (it->is_regular_file(ec)) { ++n; }
  }
  return n;
}

bool
link_or_copy_tree_(const std::filesystem::path& src,
                   const std::filesystem::path& dst, std::error_code& ec,
                   const std::function<bool()>& stop,
                   const std::function<void()>& on_file = {})
{
  namespace fs = std::filesystem;
  if (stop()) { return false; }
  if (fs::is_directory(src)) {
    fs::create_directories(dst, ec);
    for (const auto& e : fs::directory_iterator(src, ec)) {
      if (!link_or_copy_tree_(e.path(), dst / e.path().filename(), ec, stop,
                              on_file)) {
        return false;
      }
    }
    return true;
  }
  if (!fs::is_regular_file(src)) { return true; }
  fs::remove(dst, ec);
  std::error_code le;
  fs::create_hard_link(src, dst, le);
  if (le) {   // cross-device / unsupported -> real copy
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
  }
  if (on_file) { on_file(); }
  return true;
}

// Resolve a general-LLM `target` to a submodule quantization SCOPE. Sets
// *scope (a tensor-name substring; "" = the whole model) and *all_in_scope
// (quantize every 2D linear in scope -- for towers whose linear leaves are
// non-standard). Named aliases (text/language, vision, audio) resolve against
// the model's ACTUAL tensor names; an unrecognised value is used as a literal
// prefix. Returns false + sets *err when a named submodule is absent.
bool
resolve_llm_target_(const std::string& src_dir, const std::string& target,
                    std::string* scope, bool* all_in_scope, std::string* err)
{
  *scope = "";
  *all_in_scope = false;
  std::string t;
  for (char c : target) { t.push_back((char)std::tolower((unsigned char)c)); }
  if (t.empty() || t == "all" || t == "model" || t == "full" ||
      t == "everything") {
    return true;                       // whole model, standard leaf set
  }

  std::vector<std::string> cands;
  bool broad = false, backbone = false;
  if (t == "text" || t == "language" || t == "language_model" ||
      t == "backbone" || t == "llm" || t == "decoder") {
    cands = {"language_model.", "model.language_model."};
    backbone = true;                   // standard leaves -> leaf-gated
  } else if (t == "vision" || t == "visual" || t == "image") {
    cands = {"visual.", "vision_tower.", "vision_model.", "vision."};
    broad = true;
  } else if (t == "audio" || t == "speech") {
    cands = {"audio_tower.", "audio_model.", "audio."};
    broad = true;
  } else {
    cands = {target};                  // explicit tensor-name prefix
    broad = true;
  }

  std::vector<std::string> names;
  auto wts = genai::MetalLlamaWeights::open_model(src_dir);
  if (wts.has_value()) { names = wts->tensor_names(); }
  auto present = [&](const std::string& pre) {
    for (const auto& n : names) {
      if (n.find(pre) != std::string::npos) { return true; }
    }
    return false;
  };
  for (const auto& c : cands) {
    if (present(c)) { *scope = c; *all_in_scope = broad; return true; }
  }
  if (backbone) {
    return true;   // a plain LLM has no separate backbone submodule => all
  }
  *err = fmt("no submodule matching target '{}' found in the model",
             target)();
  return false;
}

// The DiT's quantizable Linear leaves (the segment before ".weight"), per
// family. Krea-2 (Qwen-Image MMDiT): the 28 blocks' attn (to_q/k/v/gate +
// to_out.0 -> leaf "0") and SwiGLU (gate/up/down), the text-fusion blocks,
// txt_in / time_embed (linear_1/2), img_in, time_mod_proj and
// final_layer.linear. `projector` (K=12) stays dense (not group-divisible).
// "0" uniquely matches attn.to_out.0 in the DiT.
//
// FLUX.2 (FLUX topology, Flux2Transformer2DModel): the double-stream blocks'
// attn (to_q/k/v + to_out.0 + to_add_out, plus add_{q,k,v}_proj when present)
// and gated MLP (ff.linear_in/out, ff_context.linear_in/out); the single-stream
// blocks' fused attn+MLP (to_qkv_mlp_proj in, to_out out); and the embedders
// (x_embedder, context_embedder, final proj_out). The small AdaLayerNorm
// modulation linears (norm*.linear) are LEFT bf16 (matched by nothing here) --
// they are cheap and quality-sensitive. NOTE: verify against the actual
// safetensors tensor names on first run + keep in lock-step with the flux2 DiT
// loader's expected quant leaf set.
std::vector<std::string>
dit_quant_linears_(const std::string& family)
{
  if (family == "flux2") {
    // The big per-block compute Linears only. The embedders (x_embedder K=128
    // -> only 2 groups/row at g64; context_embedder; final proj_out) are left
    // bf16: they are precision-sensitive input/output projections that all image
    // info flows through, and 4-bit them dominates the DiT quant error.
    return {"to_q", "to_k", "to_v", "to_out", "to_add_out",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_qkv_mlp_proj",
            "linear_in", "linear_out"};
  }
  if (family == "qwen-image-edit" || family == "mage-flow") {
    // Dual-stream QwenImageTransformer2DModel -- and Mage-Flow's NR-MMDiT,
    // whose checkpoint tensor NAMES are identical (that is what let the metal
    // port reuse the transformer wholesale), so one leaf set covers both.
    // Mage-Flow simply has no single_transformer_blocks tail.
    // The quantizer matches the LAST
    // dot-component before ".weight", so use those: to_out is "attn.to_out.0"
    // -> "0" (60, unique); both FeedForward up-projs "*_mlp.net.0.proj" ->
    // "proj" (120); both down-projs "*_mlp.net.2" -> "2" (120). The adaLN
    // modulation ("*_mod.1" -> "1"), img_in/txt_in and the norm_out/proj_out
    // head stay bf16 (precision-sensitive; the residual reaches ~1e7).
    return {"to_q", "to_k", "to_v", "0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            "proj", "2"};
  }
  if (family == "wan") {
    // Wan's single-stream block: self-attention, cross-attention into the
    // text, and an UNGATED feed-forward. Leaves are matched on the last
    // dot-component, so attn{1,2}.to_out.0 -> "0", ffn.net.0.proj ->
    // "proj", ffn.net.2 -> "2". That is 6 Linears per block x 40 blocks x
    // 2 experts, and it is all of the weight: everything left bf16 here
    // is small AND precision-sensitive --
    //   patch_embedding   every pixel of the latent enters through it, and
    //                     at K = 144 it is barely two groups per row;
    //   proj_out          every velocity value leaves through its 64 rows;
    //   condition_embedder  the timestep and text projections, which feed
    //                     the modulation of all 40 blocks;
    //   scale_shift_table the modulation itself.
    // None of their leaves ("patch_embedding", "proj_out", "linear_1",
    // "linear_2", "time_proj") is in this set, so no exclude list is
    // needed -- unlike Boogu, this family has no leaf collisions.
    return {"to_q", "to_k", "to_v", "0", "proj", "2"};
  }
  if (family == "minimax-h3") {
    // One single-stream block kind, repeated 50 times plus 2 token-refiner
    // blocks that share the same leaves: a FUSED qkv, the attention output,
    // and a gated feed-forward whose fc1 carries value|gate concatenated.
    //
    // What is deliberately NOT here is the whole rest of the model, and all
    // of it is both small and precision-sensitive:
    //   video_patch_proj / audio_patch_proj / condition_proj
    //                     every latent voxel, audio frame and text row
    //                     enters through these (~28M params together);
    //   time_embedder.proj_in / proj_out
    //                     feeds the modulation of all 50 blocks -- and vpipe
    //                     runs this MLP on the host in f32 on purpose;
    //   final_layer.video_out / audio_out
    //                     every velocity value leaves through them.
    // None of their leaves collides with this set, so no exclude list is
    // needed. `adaln_proj.linear` is handled separately: it is 40% of the
    // model and only quantized under quant_modulation.
    return {"qkv_proj", "out_proj", "fc1", "fc2"};
  }
  if (family == "boogu-image") {
    // Boogu's NextDiT. Every block kind contributes: the refiners + single
    // stream use attn.to_{q,k,v}/to_out.0 and feed_forward.linear_{1,2,3}; the
    // dual-stream blocks add the processor-owned joint projections
    // (img/instruct_to_{q,k,v}, img_out, instruct_out), their shared to_out.0,
    // the image self-attention (same to_* leaves) and the two per-stream FFs.
    // Leaves are matched on the LAST dot-component, so "0" is to_out.0.
    // The adaLN modulation linears (norm*.linear -> "linear") stay bf16 like
    // every other family here -- they are what the residual scale rides on.
    // NOTE the collision the exclude list below handles: feed_forward.linear_1
    // /2/3 share their leaves with norm_out.linear_1/2 and the timestep
    // embedder's linear_1/2, which must NOT be quantized.
    return {"to_q", "to_k", "to_v", "0",
            "img_to_q", "img_to_k", "img_to_v",
            "instruct_to_q", "instruct_to_k", "instruct_to_v",
            "img_out", "instruct_out",
            "linear_1", "linear_2", "linear_3"};
  }
  return {"to_q", "to_k", "to_v", "to_gate", "0", "gate", "up", "down",
          "linear_1", "linear_2", "img_in", "time_mod_proj", "linear"};
}

// The DiT's hidden width from transformer/config.json, 0 when unreadable. Used
// only to sanity-check the group size against it (see the quantize path): the
// key differs per family -- Boogu/Lumina say hidden_size, Qwen-Image/FLUX say
// (joint_)attention_dim or num_attention_heads * attention_head_dim.
int
dit_hidden_size_(const std::string& src_dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(src_dir) / "config.json");
  if (!in) { return 0; }
  FlexData cfg = FlexData::from_json(in);
  if (!cfg.is_object()) { return 0; }
  auto obj = cfg.as_object();
  for (const char* k : {"hidden_size", "joint_attention_dim",
                        "attention_dim", "inner_dim"}) {
    if (obj.contains(k)) {
      const int v = (int)obj.at(k).as_int(0);
      if (v > 0) { return v; }
    }
  }
  if (obj.contains("num_attention_heads") &&
      obj.contains("attention_head_dim")) {
    const int h = (int)obj.at("num_attention_heads").as_int(0);
    const int d = (int)obj.at("attention_head_dim").as_int(0);
    if (h > 0 && d > 0) { return h * d; }
  }
  return 0;
}

// The DiT's main-block count (config.json num_layers; default 28) -- the
// per-layer mixed-precision ranking runs over transformer_blocks.{0..N-1}.
int
dit_num_layers_(const std::string& src_dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(src_dir) / "config.json");
  if (in) {
    FlexData cfg = FlexData::from_json(in);
    if (cfg.is_object()) {
      auto obj = cfg.as_object();
      if (obj.contains("num_layers")) {
        const int n = (int)obj.at("num_layers").as_int(0);
        if (n > 0) { return n; }
      }
      // Mage-Flow's transformer/config.json is FLUX-shaped: the block count is
      // `depth` (12), with no num_layers. Without this the mixed / AWQ
      // per-layer ranking would silently run over the default 28 blocks.
      if (obj.contains("depth")) {
        const int n = (int)obj.at("depth").as_int(0);
        if (n > 0) { return n; }
      }
    }
  }
  return 28;
}

constexpr ConfigKey kAttrs[] = {
  {.key = "src_model", .type = ConfigType::String, .required = true,
   .doc = "source model: a models-DB key (registered by model-fetch) or a "
          "bf16/f16 safetensors directory path",
   .suggest_db = kModelRegistryDb},
  {.key = "output_name", .type = ConfigType::String, .required = true,
   .doc = "result name -> <cwd>/models/<output_name> (registered in the "
          "models DB under this key), or an explicit path (\"/..\", \"./..\") "
          "used verbatim and not registered"},
  {.key = "bits", .type = ConfigType::Uint,
   .doc = "backbone affine bit-width (4 | 8)", .def_uint = 8},
  {.key = "group_size", .type = ConfigType::Uint,
   .doc = "affine group size (32 | 64)", .def_uint = 64},
  {.key = "arch", .type = ConfigType::String,
   .doc = "model family tag; empty => auto-detect from config.json model_type",
   .def_str = ""},
  {.key = "target", .type = ConfigType::String,
   .doc = "which part of the model to quantize. Text-to-image (Krea-2 / "
          "FLUX.2 / Boogu-Image): a component -- dit (default) | text_encoder "
          "(Boogu: its mllm/) | vae -- the "
          "output is a SELF-CONTAINED pipeline (all copied, the target "
          "quantized) usable as an hf_dir; chain passes. General / multi-modal "
          "LLM: a submodule "
          "SCOPE -- all (default, whole model) | text (language backbone only) "
          "| vision | audio | an explicit tensor-name prefix -- quantizes just "
          "that submodule, leaving the rest bf16 (e.g. quantize the LM "
          "backbone, keep the vision tower full precision)",
   .def_str = ""},
  {.key = "skip_existing", .type = ConfigType::Bool,
   .doc = "skip if the output config.json already exists", .def_bool = true},
  {.key = "awq", .type = ConfigType::Bool,
   .doc = "AWQ activation-aware smoothing (per-layer fp-equivalent scale "
          "search). For standard-layernorm stacks -- full attention AND "
          "Qwen3.5 gated-DeltaNet (Llama / Qwen2 / Qwen3 / Qwen3.5 / MOSS); "
          "blocked for Gemma FFN norms and MoE (use mixed or plain). "
          "Calibration: supply calib_dir, else it auto-calibrates on-device "
          "for the Qwen3 family (dense or Qwen3.5 hybrid)",
   .def_bool = false},
  {.key = "awq_clip", .type = ConfigType::Bool,
   .doc = "paired AWQ per-group weight clip search (needs awq); OFF by "
          "default -- can regress end drift on small calibration sets",
   .def_bool = false},
  {.key = "klein_kv", .type = ConfigType::Bool, .required = false,
   .doc = "the flux2 source is FLUX.2-klein-9b-kv rather than plain klein-9B. "
          "Only affects on-device CALIBRATION, whose sweep conditions on a "
          "reference image: that checkpoint isolates reference tokens, so "
          "calibrating it under the plain joint attention would clip against "
          "a distribution it never sees. Weight quantization itself is "
          "identical (same tensor names/shapes). Default false"},
  {.key = "calib_dir", .type = ConfigType::String,
   .doc = "dir with calib_{qkv,o,gateup,down}.f32 activation stats; empty "
          "(default) => on-device auto-calibration for a known arch",
   .def_str = ""},
  {.key = "mixed", .type = ConfigType::Bool,
   .doc = "unsloth-style PER-LAYER mixed precision: promote the most-"
          "sensitive mixed_frac of layers to high_bits. Works for any "
          "standard-named family (skips non-matching layers). Requires "
          "bits=4, high_bits=8, group_size=64", .def_bool = false},
  {.key = "high_bits", .type = ConfigType::Uint,
   .doc = "promoted bit-width for mixed precision (8)", .def_uint = 8},
  {.key = "mixed_frac", .type = ConfigType::Real,
   .doc = "fraction of LAYERS promoted to high_bits", .def_real = 0.25},
  {.key = "layer_prefix", .type = ConfigType::String,
   .doc = "arch layer-root prefix for awq/mixed per-layer tensors; empty => "
          "auto-detect from config.json", .def_str = ""},
  {.key = "n_layers", .type = ConfigType::Uint,
   .doc = "layer count for awq/mixed; 0 => auto-detect from config.json",
   .def_uint = 0},
  {.key = "quant_exclude", .type = ConfigType::String,
   .doc = "comma-separated tensor-name substrings to LEAVE dense. Needed "
          "whenever `target` names a wholesale scope, because that rule "
          "takes every 2D floating-point tensor whose leaf is not a norm "
          "or an embedding -- which is right for linears and wrong for "
          "anything else shaped like one. Two kinds get caught: modulation "
          "TABLES (a DiT's f32 scale/shift rows are 2D and are not "
          "matrices), and tiny gating projections, where a per-group "
          "affine over a handful of rows is all error and no saving. "
          "Neither shows up as a load failure -- the checkpoint quantizes, "
          "loads and generates something wrong",
   .def_str = ""},
  {.key = "quant_vision", .type = ConfigType::Bool,
   .doc = "text-encoder passes: also quantize the checkpoint's VISION "
          "tower (visual.*) instead of passing it through bf16. Off by "
          "default because a tower is precision-sensitive and, for the "
          "encoders quantized here, small next to the language stack. "
          "Worth turning on for a multimodal encoder used TEXT-ONLY -- "
          "MiniMax-H3 taps a Qwen3-VL backbone and never runs its tower, "
          "so the ~1.5 GB it costs at bf16 is pure carry",
   .def_bool = false},
  {.key = "quant_modulation", .type = ConfigType::Bool,
   .doc = "Qwen-Image-Edit, Boogu-Image, Wan and MiniMax-H3 DiTs: also "
          "quantize the AdaLN modulation projections (QIE *_mod.1; Boogu "
          "norm*.linear; H3 adaln_proj.linear). They are the largest weights "
          "left bf16 -- QIE ~13 GB -> ~3.4 GB, Boogu 2.1 GB -> ~1.1 GB, and "
          "on H3 the modulation is 13B of the 33B, so leaving it bf16 holds a "
          "4-bit checkpoint at ~36 GB -- which is what lets the whole DiT fit "
          "a 16 GB box, but precision-sensitive, so they are kept bf16 by "
          "default and forced to 8-bit (not the body's bit-width) when you "
          "opt in here",
   .def_bool = false},
};
// Trigger iport (optional, any beat) + summary oport -- see model-fetch
// for the shared "preparation recipe" rationale.
const PortSpec kIports[] = {
  {.name = "trigger",
   .doc  = "optional pacing trigger (any beat type); when wired, the work "
           "waits for one beat before running -- lets these preparation "
           "stages cascade into a recipe",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "summary",
   .doc  = "FlexData summary of the completed work; its `text` field "
           "renders a report via save-text, and the beat also triggers "
           "the next stage in a recipe",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "model-quantize",
  .doc       = "Source: one-shot offline quantization of a safetensors model "
               "to the MLX-affine group-quant format, across the families "
               "vpipe supports (Llama / Qwen2 / Qwen3 / Qwen3.5 / Gemma-4 / "
               "MOSS). arch / layer_prefix / n_layers auto-detect from the "
               "source. Linears are quantized; embeddings/heads/norms/aux "
               "modules pass through. For a Krea-2 (text-to-image) model, "
               "`target` selects the component (dit | text_encoder | vae) and "
               "the output is a SELF-CONTAINED pipeline (all components copied, "
               "the target quantized) usable directly as a generate-image "
               "hf_dir -- chain passes to quantize more than one component. "
               "Optional trigger in / summary out.",
  .display_name = "Model Quantize",
  .category  = StageCategory::Preparation,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
ModelQuantizeStage::spec() const noexcept
{
  return kSpec;
}

// Register the quantized result in the models DB under `key` so downstream
// stages (e.g. text-to-speech) can reference it by key. Non-fatal: a registry
// failure just logs (the model is on disk regardless).
void
ModelQuantizeStage::register_output_(const std::string& key,
                                     const std::string& dir,
                                     const std::string& arch, int bits)
{
  LmdbEnv* env = session() ? session()->services()->lmdb_env() : nullptr;
  if (env == nullptr) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): no lmdb_env(); not registering '{}'",
        this->id(), key));
    return;
  }
  try {
    FlexData rec = FlexData::make_object();
    auto ro = rec.as_object();
    ro.insert_or_assign("local_path", FlexData::make_string(dir));
    ro.insert_or_assign("source", FlexData::make_string(_src_model));
    ro.insert_or_assign("quantized", FlexData::make_bool(true));
    ro.insert_or_assign("bits", FlexData::make_uint((std::uint64_t)bits));
    // Describe the OUTPUT by probing it (model-detect.h), so a quantized
    // model carries the same runtime type + I/O modalities as a fetched
    // one. This is what gets a quantized DiT its "<family>-dit" tag --
    // the arch-detect below is an LM probe and says nothing about a DiT.
    const DetectedModel d = detect_model_dir(dir);
    record_detected_fields(ro, d);
    // Fall back to the source model's arch tag when the output itself
    // isn't recognizable. "unknown" is arch-detect's miss value, not a
    // family -- writing it would be a false compatibility hint.
    if (d.model_type.empty() && !arch.empty() && arch != "unknown") {
      ro.insert_or_assign("model_type", FlexData::make_string(arch));
    }
    LmdbDb  db(*env, kModelRegistryDb);
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadWrite);
    const std::string bytes = rec.to_binary();
    db.put(txn, key, bytes);
    txn.commit();
    session()->info(fmt(
        "ModelQuantizeStage('{}'): registered '{}' -> '{}' in the "
        "model registry", this->id(), key, dir));
  } catch (const std::exception& e) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): registry write for '{}' failed: {}",
        this->id(), key, e.what()));
  }
}

bool
ModelQuantizeStage::quantize_once(const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;

  // Resolve the source: a models-DB key wins over a same-named path.
  const std::string src_dir =
      resolve_model_dir(session(), _src_model);

  // Resolve the output. A bare name (or "org/name") -> <cwd>/models/<name>,
  // registered under that key; an absolute/relative path ("/..", "./..") is
  // used verbatim and NOT registered.
  const bool explicit_path =
      _output_name[0] == '/' ||
      _output_name.rfind("./", 0) == 0 || _output_name.rfind("../", 0) == 0;
  const std::string out_dir = explicit_path
      ? _output_name
      : (fs::current_path() / "models" / _output_name).string();

  if (_skip_existing && fs::exists(fs::path(out_dir) / "config.json", ec)) {
    session()->info(fmt(
        "ModelQuantizeStage('{}'): output '{}' already exists; skipping",
        this->id(), out_dir));
    return true;
  }

  metal_compute::MetalCompute* mc = session()->services()->metal_compute();
  if (mc == nullptr) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): no metal-compute backend", this->id()));
    return false;
  }

  // ---- A REGISTERED out-of-tree family, before the built-in detection.
  // Mirrors how VideoModelRegistry sits ahead of the built-in wan /
  // minimax-h3 dispatch: this adds a path, it reroutes none. A family
  // that does not claim `src_dir` costs one cheap probe.
  if (const genai::QuantizableFamily* fam =
          genai::QuantizeFamilyRegistry::get().claim_for(src_dir)) {
    return quantize_registered_pipeline_(src_dir, *fam, out_dir, stop);
  }

  // ---- Text-to-image (Krea-2 / FLUX.2): a multi-component pipeline. ----
  std::string t2i_family;
  // Which model the REFERENCE meant, from the record rather than from
  // the directory -- two records can share one local_path.
  const std::string h3_part =
      genai::MetalMiniMaxH3Transformer::partition_of_model_type(
          resolve_model(session(), _src_model).model_type);
  const std::string dit_dir =
      resolve_t2i_dit_dir_(src_dir, &t2i_family, h3_part);
  if (!dit_dir.empty()) {
    // A Comfy-Org REPACK root is tested FIRST, and the order is
    // load-bearing rather than stylistic. A repack's components live in
    // role subdirs (diffusion_models/, text_encoders/, vae/), and once
    // this pass has quantized one of them that subdir holds a DIRECTORY
    // checkpoint -- so `resolve` returns a directory and the is_root test
    // below would read the output of a first pass as a diffusers root,
    // then fail looking for a `transformer/` that a repack never has.
    // Testing the repack shape first is what makes a chain work.
    if (is_comfy_root_(src_dir)) {
      return quantize_comfy_pipeline_(src_dir, t2i_family, dit_dir, out_dir,
                                      stop);
    }
    // src is the pipeline ROOT -- a directory holding transformer/ and
    // its siblings. A resolution to a FILE is not that: a bare component
    // named directly has nothing to copy around it and takes the
    // single-component path below.
    const bool is_root =
        (dit_dir != src_dir) && fs::is_directory(fs::path(dit_dir), ec);
    if (is_root) {
      // Self-contained pass: copy every component, quantize the `target` one
      // (default the DiT), register as a full pipeline usable as an hf_dir.
      // Chainable across passes (DiT, then text_encoder, ...).
      return quantize_t2i_pipeline_(src_dir, t2i_family, out_dir, stop);
    }
    // A single component named DIRECTLY (a path to one .safetensors, not
    // to the repo around it). Sibling selection is not possible from here
    // -- there is no repo to select out of -- and it is not a silent
    // fallback either: naming a file and asking for a different component
    // is a contradiction worth reporting, since the fix is to point
    // src_model at the model root and take the self-contained path above.
    if (!_target.empty() && t2i_target_subdir_(_target) != "transformer" &&
        !fs::is_directory(fs::path(dit_dir), ec)) {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): src '{}' names a single component; "
          "target='{}' selects a SIBLING, which needs the repo root -- "
          "point src_model at the model root", this->id(), src_dir, _target));
      return false;
    }
    // src is a bare transformer dir -> transformer-only output for the
    // generate-image `dit_dir` override (registered "<family>-dit"). Only the
    // DiT is present, so `target` other than dit doesn't apply.
    if (!_target.empty() && t2i_target_subdir_(_target) != "transformer") {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): src '{}' is a bare DiT dir; target='{}' "
          "needs the full pipeline root (with {}/) -- point src_model at the "
          "model root", this->id(), src_dir, _target,
          t2i_target_subdir_(_target)));
      return false;
    }
    if (!quantize_dit_component_(dit_dir, out_dir, t2i_family,
                                 fs::path(dit_dir).parent_path().string(),
                                 stop)) {
      return false;
    }
    session()->log_normal(fmt(
        "ModelQuantizeStage('{}'): quantized {} DiT '{}' -> '{}' ({}-bit "
        "g{}{}{})", this->id(), t2i_family, src_dir, out_dir, _bits,
        _group_size, _mixed ? " mixed" : "", _awq ? " awq-clip" : ""));
    // "<family>-dit" (transformer-only) -> the generate-image dit_dir slot.
    if (!explicit_path) {
      register_output_(_output_name, out_dir, t2i_family + "-dit", _bits);
    }
    return true;
  }

  // A diffusers PIPELINE ROOT that did not route to the DiT path above is not
  // an LLM, and must not be diagnosed as one. Falling through runs the LM
  // arch-detect on a directory that has no LM in it and reports
  // `arch='unknown'` plus a `target` resolution failure ("no submodule
  // matching target 'dit'") -- which reads as "this family is unsupported"
  // when the actual cause is almost always that the component is not on disk
  // yet, an interrupted or still-running download. Say which it is, here.
  {
    const bool has_index = fs::exists(fs::path(src_dir) / "model_index.json");
    const fs::path tdir = fs::path(src_dir) / "transformer";
    if (has_index || fs::is_directory(tdir, ec)) {
      std::string why;
      if (!fs::is_directory(tdir, ec)) {
        why = "its 'transformer/' component is not present";
      } else if (!fs::exists(tdir / "config.json")) {
        why = "'transformer/config.json' is missing";
      } else {
        std::string cls;
        std::ifstream in(tdir / "config.json");
        if (in) {
          FlexData c = FlexData::from_json(in);
          if (c.is_object()) {
            auto o = c.as_object();
            if (o.contains("_class_name")) {
              cls = std::string(o.at("_class_name").as_string(""));
            }
          }
        }
        why = cls.empty()
                  ? std::string("its transformer config carries no _class_name")
                  : fmt("its transformer _class_name '{}' is not a supported "
                        "DiT", cls)();
      }
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): '{}' looks like a text-to-image pipeline "
          "but {}. If the model is still downloading, wait for it to finish; "
          "otherwise this build does not support that DiT. (Not falling back "
          "to the language-model quantizer -- there is no LM at this path.)",
          this->id(), src_dir, why));
      return false;
    }
  }

  // Auto-detect arch / n_layers / layer_prefix + capabilities from the source
  // (config.json + a probe of the safetensors layer layout); explicit config
  // wins over the detection.
  const genai::QuantArchInfo meta = genai::detect_quant_arch(session(), src_dir);
  const std::string eff_arch = _arch.empty() ? meta.arch : _arch;
  const std::string eff_layer_prefix =
      _layer_prefix.empty() ? meta.layer_prefix : _layer_prefix;
  const int eff_n_layers = _n_layers > 0 ? _n_layers : meta.n_layers;
  session()->info(fmt(
      "ModelQuantizeStage('{}'): arch='{}' layer_prefix='{}' n_layers={} "
      "(attn {}) awq_ok={} calib_ok={}", this->id(), eff_arch,
      eff_layer_prefix, eff_n_layers, meta.n_attn_layers, meta.awq_ok,
      meta.calib_ok));
  session()->log_verbose(fmt(
      "ModelQuantizeStage('{}'): calib_dir='{}' awq={} mixed={}",
      this->id(), _calib_dir, _awq, _mixed));

  // Submodule scope (general / multi-modal LLM `target`): restrict the pass to
  // one part of the model (e.g. quantize the language backbone, keep the
  // vision tower bf16). Resolved against the model's tensor names.
  std::string quant_scope;
  bool quant_all_in_scope = false;
  {
    std::string serr;
    if (!resolve_llm_target_(src_dir, _target, &quant_scope,
                             &quant_all_in_scope, &serr)) {
      session()->warn(fmt("ModelQuantizeStage('{}'): {}", this->id(), serr));
      return false;
    }
    if (!quant_scope.empty() && quant_all_in_scope && (_awq || _mixed)) {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): awq/mixed operate on the language "
          "backbone; target '{}' (a vision/audio/explicit submodule) supports "
          "plain quantization only", this->id(), _target));
      return false;
    }
    if (!quant_scope.empty()) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): target '{}' -> quantizing only the "
          "submodule scope '{}'{}", this->id(), _target, quant_scope,
          quant_all_in_scope ? " (all linears in scope)" : ""));
    }
  }

  // AWQ / mixed need the per-layer geometry; validate it resolved (here, not
  // in the ctor -- it depends on the resolved source).
  if (_awq || _mixed) {
    if (eff_n_layers <= 0 || eff_layer_prefix.empty()) {
      // (A diffusers pipeline root never reaches here: it is diagnosed and
      // rejected right after resolve_t2i_dit_dir_ fails, above.)
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): awq/mixed need n_layers + layer_prefix; "
          "auto-detect from '{}' failed -- set them explicitly (got "
          "n_layers={}, layer_prefix='{}')",
          this->id(), src_dir, eff_n_layers, eff_layer_prefix));
      return false;
    }
  }
  // AWQ's fp-equivalent fold needs every layer to be a foldable block rooted
  // at input_layernorm (full attention OR Qwen3.5 gated-DeltaNet) with the
  // standard layernorm pair and a dense MLP. Block it for Gemma-style FFN
  // norms and MoE MLPs, where it would mis-fold or hard-fail -- mixed
  // precision and plain quant still work for those.
  if (_awq && meta.detected && !meta.awq_ok) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): AWQ is not supported for arch '{}' (not a "
        "uniform foldable stack -- Gemma FFN norms or a non-dense/MoE MLP). "
        "Use mixed=true or plain quantization instead.",
        this->id(), eff_arch));
    return false;
  }

  // On-device auto-calibration: when AWQ is on and no calib_dir was supplied,
  // produce the activation stats here (8-bit base + tapped forward over a
  // built-in corpus) instead of needing the offline HF script.
  // On-device AWQ auto-calibration (8-bit base + tapped forward over the
  // built-in text corpus) when AWQ is on and no calib_dir was supplied.
  std::string eff_calib = _calib_dir;
  fs::path auto_calib;
  if (stop()) { return false; }
  if (_awq && _calib_dir.empty()) {
    eff_calib = auto_calibrate_backbone_(src_dir, meta, eff_n_layers,
                                         src_dir + "/tokenizer.json", stop);
    if (eff_calib.empty()) { return false; }   // logged inside
    auto_calib = eff_calib;
  }

  genai::QuantizeOptions opt;
  opt.bits  = _bits;
  opt.group = _group_size;
  // Standard LMs (Qwen/Llama/Gemma) quantize the embed table + lm_head (MLX
  // convention) so they reload through the affine inference path. Every MOSS
  // variant (moss-tts / moss-tts-local / moss-tts-realtime) keeps bf16
  // embeddings + heads (its wrapper gathers them host-side, and the realtime
  // tied text head reads language_model.embed_tokens), so leave it off.
  const bool moss = eff_arch.rfind("moss-tts", 0) == 0;
  opt.quant_embeddings = !eff_arch.empty() && !moss;
  // Qwen3.5/3.6 store zero-centered RMSNorm weights and apply (1 + weight)
  // (Gemma-style); vpipe's RMSNorm kernel multiplies by `weight` directly (as
  // MLX-converted checkpoints, which pre-fold the +1), so fold +1 into the
  // affected norms when quantizing a raw-HF qwen3.5 checkpoint. Other families
  // (Qwen3/Llama standard norms; Gemma handled in the runtime) leave it off.
  opt.norm_offset = (eff_arch == "qwen3.5");
  // Gemma-4 (e4b / E2B PLE family) also quantizes the per-layer-input tensors
  // and the per-layer embed table (MLX convention): the affine gemma inference
  // path binds embed_tokens_per_layer / per_layer_input_gate /
  // per_layer_projection as quantized triples (per_layer_model_projection +
  // the norms stay bf16). Extend the default linear set with these gemma leaves
  // so a raw-HF gemma reloads through the affine path. (embed_tokens + tied
  // lm_head are already covered by quant_embeddings above.)
  if (eff_arch == "gemma4") {
    opt.quant_linears = genai::ModelQuantizer::default_quant_linears();
    opt.quant_linears.push_back("embed_tokens_per_layer");
    opt.quant_linears.push_back("per_layer_input_gate");
    opt.quant_linears.push_back("per_layer_projection");
  }
  if (_awq) {
    opt.smoothquant  = true;       // the SQ pass runs the AWQ scale search
    opt.awq_clip     = _awq_clip;
    opt.calib_dir    = eff_calib;
    opt.layer_prefix = eff_layer_prefix;
    opt.n_layers     = eff_n_layers;
  }
  if (_mixed) {
    opt.mixed      = true;
    opt.high_bits  = _high_bits;
    opt.mixed_frac = _mixed_frac;
    opt.layer_prefix = eff_layer_prefix;   // mixed needs it too (+ n_layers)
    opt.n_layers   = eff_n_layers;
  }
  opt.quant_scope        = quant_scope;         // "" => the whole model
  opt.quant_all_in_scope = quant_all_in_scope;
  // Comma-separated, whitespace trimmed. Appended to whatever a family
  // branch below sets rather than replacing it: the built-in exclusions
  // are architectural facts, not defaults to override.
  {
    std::string cur;
    auto flush = [&]() {
      std::size_t a = cur.find_first_not_of(" \t");
      std::size_t b = cur.find_last_not_of(" \t");
      if (a != std::string::npos) {
        opt.quant_exclude.push_back(cur.substr(a, b - a + 1));
      }
      cur.clear();
    };
    for (char c : _quant_exclude) {
      if (c == ',') { flush(); } else { cur.push_back(c); }
    }
    flush();
  }

  if (stop()) {
    if (!auto_calib.empty()) { fs::remove_all(auto_calib, ec); }
    return false;
  }
  genai::ModelQuantizer mq(mc);
  std::string err;
  const bool ran = mq.run(src_dir, out_dir, opt, &err, stop);
  if (!auto_calib.empty()) { fs::remove_all(auto_calib, ec); }   // temp calib
  if (!ran) {
    // On a stop request the run aborted mid-pass: config.json is written only
    // at the very end, so the output is incomplete (partial shards are inert).
    // Leave it in place but flag it for removal -- info, not the scary warn.
    if (stop()) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): quantize stopped; output '{}' is "
          "incomplete -- remove it before reuse", this->id(), out_dir));
      return false;
    }
    // warn (non-fatal): error() throws by design; an offline quantize
    // failure should surface and return false, not abort the process.
    session()->warn(fmt("ModelQuantizeStage('{}'): {}", this->id(), err));
    return false;
  }

  session()->log_normal(fmt(
      "ModelQuantizeStage('{}'): quantized '{}' -> '{}' ({}-bit g{})",
      this->id(), src_dir, out_dir, _bits, _group_size));

  // Register the result so it is referenceable by key (skip explicit paths).
  if (!explicit_path) {
    register_output_(_output_name, out_dir, eff_arch, _bits);
  }
  return true;
}

bool
ModelQuantizeStage::quantize_dit_component_(
    const std::string& dit_dir, const std::string& out_dir,
    const std::string& family, const std::string& calib_root,
    const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  metal_compute::MetalCompute* mc = session()->services()->metal_compute();
  if (mc == nullptr) { return false; }

  const bool is_flux2 = (family == "flux2");
  const bool is_mage  = (family == "mage-flow");
  const bool is_boogu = (family == "boogu-image");
  // Mage-Flow shares Qwen-Image's block topology and tensor names, so the
  // modulation handling below applies to it too.
  const bool is_qie   = (family == "qwen-image-edit") || is_mage;
  const bool awq = _awq;

  // Plain group-affine over the DiT leaf set (no LM arch-detect / embedding
  // quant; the DiT loader folds the zero-centered norms' +1 itself, so
  // norm_offset stays off). dit_family lets the quantizer pick the flux2
  // AWQ-calib mapping + smoothing fold + two-prefix mixed ranking.
  genai::QuantizeOptions opt;
  opt.bits  = _bits;
  opt.group = _group_size;
  // The group size has to DIVIDE the projections' input width or the quantizer
  // leaves them bf16 -- a "4-bit" checkpoint the size of the source. Boogu's
  // hidden_size is 3360, which is 52.5 groups of 64, so its attention and FF
  // gate/up projections are all group-32 work. Pick the largest supported group
  // that divides the DiT's hidden width and say so, rather than shipping a
  // checkpoint that only quantized the layers that happened to fit.
  {
    const int hid = dit_hidden_size_(dit_dir);
    if (hid > 0 && (hid % opt.group) != 0) {
      int g = 0;
      for (int cand : {64, 32}) {
        if (cand <= opt.group && (hid % cand) == 0) { g = cand; break; }
      }
      if (g > 0) {
        session()->warn(fmt(
            "ModelQuantizeStage('{}'): hidden_size {} is not a multiple of "
            "group_size {} -- the projections reading it would stay bf16; "
            "using group_size {} instead", this->id(), hid, opt.group, g));
        opt.group = g;
      } else {
        session()->warn(fmt(
            "ModelQuantizeStage('{}'): hidden_size {} is not a multiple of any "
            "supported group size (32 | 64); the projections reading it will "
            "stay bf16", this->id(), hid));
      }
    }
  }
  opt.quant_linears    = dit_quant_linears_(family);
  const bool is_wan = (family == "wan");
  const bool is_h3  = (family == "minimax-h3");
  if (_quant_modulation && is_h3) {
    // H3's modulation is not a small side projection like the others': its
    // per-block `adaln_proj.linear` is 2688 -> 96768 (6 modulation vectors x
    // 3 modalities x hidden), which is 260M of each block's 645M -- 13B of
    // the 33B model. Leaving it bf16 caps a 4-bit checkpoint at ~36 GB,
    // which is most of the reason to quantize at all, so this opt-in matters
    // more here than anywhere else.
    //
    // Still w8: the modulation is what the residual scale rides on, and the
    // per-tensor bit detection in the loader means body @ 4 + modulation @ 8
    // loads with no config to keep in sync. The leaf is unique to the two
    // adaln_proj sites (per-block and final_layer).
    opt.quant_linears.push_back("linear");
    opt.high_bit_leaves.push_back("linear");
    opt.high_bits = 8;
    session()->info(fmt(
        "ModelQuantizeStage('{}'): quantizing the AdaLN modulation "
        "(adaln_proj.linear, 13B of 33B) at 8-bit; body stays {}-bit",
        this->id(), _bits));
  }
  if (_quant_modulation && (is_qie || is_boogu || is_wan)) {
    // The adaLN modulation projections are the largest weights in these DiTs
    // and are kept bf16 by default because they are what the residual scale
    // rides on. This is the opt-in that quantizes them anyway, for a box that
    // cannot otherwise hold the DiT.
    //   Qwen-Image-Edit / Mage-Flow: `*_mod.1`             -> leaf "1"
    //   Boogu-Image: `norm1.linear` (refiners + single) and
    //     `img_norm{1,2,3}.linear` / `instruct_norm{1,2}.linear` (dual-stream)
    //                                                       -> leaf "linear"
    // On Boogu that leaf is EXACTLY the 76 modulation linears (32 single + 4
    // modulated refiners + 8x5 dual-stream); norm_out./time_caption_embed's
    // linear_1/linear_2 are a different leaf and stay excluded by name.
    //   Wan: the 6-way modulation comes out of
    //     condition_embedder.time_proj      -> leaf "time_proj"
    //   (scale_shift_table is 3-D and never quantized at all, and the
    //   per-block modulation is that table plus this projection's output).
    const char* leaf = is_boogu ? "linear" : is_wan ? "time_proj" : "1";
    opt.quant_linears.push_back(leaf);
    // The modulation carries large-magnitude scale/gate values (on QIE they
    // drive the ~1e7 residual), so 4-bit wrecks it (block-0 rel-L2 ~0.75) while
    // 8-bit is fine. Force w8 for the modulation regardless of the body's
    // bit-width; the DiT loaders derive bits PER TENSOR from the code/scale
    // shapes, so body @ bits, modulation @ 8 loads correctly.
    opt.high_bit_leaves.push_back(leaf);
    opt.high_bits = 8;
    session()->info(fmt(
        "ModelQuantizeStage('{}'): quantizing the AdaLN modulation ({}) at "
        "8-bit (precision-sensitive; body stays {}-bit)", this->id(),
        is_boogu ? "norm*.linear" : is_wan ? "condition_embedder.time_proj"
                                            : "*_mod.1", _bits));
  }
  if (is_boogu) {
    // Keep the precision-sensitive heads out despite their shared leaf names
    // (see dit_quant_linears_): the final LuminaLayerNormContinuous projection
    // -- every velocity value flows through its 64 rows -- and the timestep
    // embedder, which feeds every modulation vector in the model. The patch
    // embedders are excluded on the same grounds as FLUX.2's (K = 64 is one
    // group at g64, and all image information enters through them).
    opt.quant_exclude = {"norm_out.", "time_caption_embed.",
                         "x_embedder", "ref_image_patch_embedder"};
  }
  opt.quant_embeddings = false;
  opt.norm_offset      = false;
  opt.dit_family       = family;   // "krea2" | "flux2" | "boogu-image" | ...
  const int dit_layers = dit_num_layers_(dit_dir);
  fs::path tmp_cal;
  if (_mixed) {
    // Per-block mixed precision (bits=4, high_bits=8, group=64 -- enforced in
    // the ctor). Krea-2 ranks the single transformer_blocks prefix; FLUX.2 ranks
    // both block prefixes (flagged via dit_family in the quantizer).
    opt.mixed      = true;
    opt.high_bits  = _high_bits;
    opt.mixed_frac = _mixed_frac;
    opt.n_layers   = dit_layers;
    if (is_boogu) {
      // Rank Boogu's single-stream tail: 32 of its 46 blocks and the bulk of
      // the weights. (The 8 dual-stream blocks and the 6 refiners keep the
      // base width -- the dual-stream blocks carry the joint attention that
      // first mixes the streams, so they are the wrong place to save bits.)
      opt.layer_prefix = "single_stream_layers.";
    } else if (!is_flux2) {
      opt.layer_prefix = "transformer_blocks.";
    }
  }
  if (awq) {
    // DiT AWQ = activation-aware weight CLIPPING. On-device auto-calibrate over
    // prompts x sigmas (reading the encoder + tokenizer from calib_root) when
    // no calib_dir is supplied. Krea-2 and FLUX.2 use family-specific collectors
    // (different encoder / template / tap groups); the quantizer reads the
    // right calib layout via opt.dit_family.
    if (is_mage && _calib_dir.empty()) {
      // The on-device collectors drive a family's own encoder + template; no
      // Mage-Flow collector exists yet, and the Qwen-Image one would build a
      // QIE DiT config against Mage weights. Refuse rather than silently
      // calibrate the wrong model -- an explicit calib_dir still works.
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): Mage-Flow DiT AWQ has no on-device "
          "collector yet; supply calib_dir, or drop awq (plain / mixed "
          "quantization is supported)", this->id()));
      return false;
    }
    if (!_calib_dir.empty()) {
      opt.calib_dir = _calib_dir;
    } else {
      tmp_cal = vpipe::temp_root() /
                ((is_flux2 ? "vpipe-flux2-ditcal-"
                  : is_boogu ? "vpipe-boogu-ditcal-"
                  : is_qie ? "vpipe-qie-ditcal-"
                  : "vpipe-krea2-ditcal-") + this->id());
      fs::remove_all(tmp_cal, ec);
      session()->info(fmt(
          "ModelQuantizeStage('{}'): on-device {} DiT AWQ calibration "
          "(prompts x sigmas)", this->id(), family));
      std::string ce;
      const bool ok = is_flux2
          ? genai::collect_flux2_calibration(
                mc, calib_root, genai::default_dit_calibration_prompts(), 8, 256,
                256, 0, tmp_cal.string(), &ce, stop, _klein_kv)
          : is_boogu
          // 4 steps, not 8: the Boogu collector walks the DMD student's own
          // ascending schedule, which IS 4 steps.
          ? genai::collect_boogu_calibration(
                mc, calib_root, genai::default_dit_calibration_prompts(), 4, 256,
                256, 0, tmp_cal.string(), &ce, stop)
          : is_qie
          ? genai::collect_qwen_image_calibration(
                mc, calib_root, genai::default_dit_calibration_prompts(), 8, 256,
                256, 0, tmp_cal.string(), &ce, stop)
          : genai::collect_dit_calibration(
                mc, calib_root, genai::default_dit_calibration_prompts(), 8, 256,
                256, 0, tmp_cal.string(), &ce, stop);
      if (!ok) {
        fs::remove_all(tmp_cal, ec);
        if (stop()) {
          session()->info(fmt(
              "ModelQuantizeStage('{}'): stopped during DiT calibration",
              this->id()));
        } else {
          session()->warn(fmt("ModelQuantizeStage('{}'): DiT calib: {}",
                              this->id(), ce));
        }
        return false;
      }
      opt.calib_dir = tmp_cal.string();
    }
    opt.dit_awq    = true;
    opt.dit_family = family;   // "krea2" | "flux2"
    opt.n_layers   = dit_layers;
  }
  genai::ModelQuantizer mq(mc);
  std::string err;
  const bool ran = mq.run(dit_dir, out_dir, opt, &err, stop);
  if (!tmp_cal.empty()) { fs::remove_all(tmp_cal, ec); }
  if (!ran) {
    if (stop()) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): quantize stopped; output '{}' is "
          "incomplete", this->id(), out_dir));
    } else {
      session()->warn(fmt("ModelQuantizeStage('{}'): {}", this->id(), err));
    }
    return false;
  }
  return true;
}

bool
ModelQuantizeStage::quantize_text_encoder_(
    const std::string& enc_dir, const std::string& out_dir,
    const std::string& root, const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  metal_compute::MetalCompute* mc = session()->services()->metal_compute();
  if (mc == nullptr) { return false; }

  // The text encoder is a dense Qwen3-VL backbone under the language_model.
  // prefix (q/k/v/o_proj, gate/up/down_proj) -- the default linear set matches
  // it prefix-agnostically. CRITICAL: keep embed_tokens bf16 (quant_embeddings
  // off) -- the generate-image stage host-gathers it as a plain table.
  genai::QuantArchInfo meta = genai::detect_quant_arch(session(), enc_dir);
  // `detect_quant_arch` gets the layer prefix and depth by PROBING the
  // weights, which works on a Comfy-Org single file -- but the family
  // tag comes from config.json's model_type, which a repack has none of,
  // so it lands on "unknown". The repo around the file does know: its
  // diffusion_models/ entry names the architecture. Read that rather
  // than shipping an encoder whose record says "unknown", which is what
  // a picker filters on.
  if (meta.arch.empty() || meta.arch == "unknown") {
    for (const auto& c : genai::comfy::scan_repo(root)) {
      if (c.role != "diffusion_models") { continue; }
      FlexData md;
      if (!genai::comfy::metadata_json(c.file, "config", md, nullptr) ||
          !md.is_object() || !md.as_object().contains("transformer")) {
        continue;
      }
      const FlexData t = md.as_object().at("transformer");
      auto to = t.as_object();
      const FlexData im =
          to.contains("image_model") ? to.at("image_model") : FlexData();
      if (std::string(im.as_string("")) == "minimax_h3") {
        meta.arch = "minimax-h3";
        break;
      }
    }
  }
  genai::QuantizeOptions opt;
  opt.bits  = _bits;
  opt.group = _group_size;
  opt.quant_embeddings = false;
  opt.norm_offset      = false;   // qwen3_vl uses standard RMSNorm
  // Record the ROLE in the output, so a bare quantized encoder is still
  // recognizable after it leaves the pipeline it came from. Its own
  // config only says "qwen3_vl", which is the architecture, not the job.
  if (meta.arch == "minimax-h3") {
    opt.component_tag = "minimax-h3-text-encoder";
  }
  if (_quant_vision) {
    // The tower's linears are named nothing like the backbone's
    // (attn.qkv / attn.proj / mlp.linear_fc{1,2} / merger.linear_fc*),
    // so they need the wholesale rule while the backbone keeps the leaf
    // set. Both run in ONE pass -- two passes would each write a full
    // set of shards, and the second would not see the first's.
    opt.quant_extra_scopes = {"visual."};
  }
  session()->info(fmt(
      "ModelQuantizeStage('{}'): text encoder '{}' (arch '{}', {} layers, "
      "prefix '{}')", this->id(), enc_dir, meta.arch, meta.n_layers,
      meta.layer_prefix));
  if (_mixed) {
    if (meta.n_layers <= 0 || meta.layer_prefix.empty()) {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): text-encoder mixed precision needs "
          "n_layers + layer_prefix; auto-detect from '{}' failed", this->id(),
          enc_dir));
      return false;
    }
    opt.mixed        = true;
    opt.high_bits    = _high_bits;
    opt.mixed_frac   = _mixed_frac;
    opt.layer_prefix = meta.layer_prefix;
    opt.n_layers     = meta.n_layers;
  }
  fs::path auto_calib;
  if (_awq) {
    if (meta.n_layers <= 0 || meta.layer_prefix.empty()) {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): text-encoder AWQ needs n_layers + "
          "layer_prefix; auto-detect from '{}' failed", this->id(), enc_dir));
      return false;
    }
    std::string calib = _calib_dir;
    if (calib.empty()) {
      // On-device auto-calibration. The text encoder is a dense Qwen3
      // backbone, so a plain text corpus exercises exactly the encoded
      // path; see find_tokenizer_json_ for where the tokenizer it needs
      // can be.
      const std::string tok = find_tokenizer_json_(enc_dir, root);
      calib = auto_calibrate_backbone_(enc_dir, meta, meta.n_layers, tok, stop);
      if (calib.empty()) { return false; }   // logged inside
      auto_calib = calib;
    }
    opt.smoothquant  = true;
    opt.awq_clip     = _awq_clip;
    opt.calib_dir    = calib;
    opt.layer_prefix = meta.layer_prefix;
    opt.n_layers     = meta.n_layers;
  }
  genai::ModelQuantizer mq(mc);
  std::string err;
  const bool ran = mq.run(enc_dir, out_dir, opt, &err, stop);
  if (!auto_calib.empty()) { fs::remove_all(auto_calib, ec); }
  if (ran) {
    // Carry the tokenizer into the output. The quantizer copies sidecars
    // only from a DIRECTORY source, and a Comfy-Org repack is a single
    // file whose tokenizer is not beside it but at the repo root -- so
    // without this, quantizing that encoder writes a checkpoint that
    // registers fine and then fails to load for want of a tokenizer.json,
    // which is a setup problem wearing the mask of a bad checkpoint.
    const fs::path have = fs::path(out_dir) / "tokenizer.json";
    if (!fs::exists(have, ec)) {
      const std::string tok = find_tokenizer_json_(enc_dir, root);
      if (tok.empty()) {
        session()->warn(fmt(
            "ModelQuantizeStage('{}'): no tokenizer.json found for '{}' "
            "(looked beside it, and in '{}'s tokenizer/ and processor/); "
            "'{}' will not encode a prompt until one is copied in",
            this->id(), enc_dir, root, out_dir));
      } else {
        std::error_code cec;
        fs::copy_file(tok, have, fs::copy_options::overwrite_existing, cec);
        // Cheap and expected beside it by anything reading the output as
        // a stock HF tokenizer; absent upstream is not an error.
        const fs::path cfg_src =
            fs::path(tok).parent_path() / "tokenizer_config.json";
        std::error_code sec;
        if (fs::exists(cfg_src, sec) && !sec) {
          std::error_code c2;
          fs::copy_file(cfg_src,
                        fs::path(out_dir) / "tokenizer_config.json",
                        fs::copy_options::overwrite_existing, c2);
        }
        if (cec) {
          session()->warn(fmt(
              "ModelQuantizeStage('{}'): could not copy '{}' into '{}': {}",
              this->id(), tok, out_dir, cec.message()));
        } else {
          session()->log_debug(fmt(
              "ModelQuantizeStage('{}'): carried '{}' into '{}'",
              this->id(), tok, out_dir));
        }
      }
    }
  }
  if (!ran) {
    if (stop()) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): text-encoder quantize stopped; '{}' is "
          "incomplete", this->id(), out_dir));
    } else {
      session()->warn(fmt("ModelQuantizeStage('{}'): text-encoder: {}",
                          this->id(), err));
    }
    return false;
  }
  return true;
}

std::string
ModelQuantizeStage::auto_calibrate_backbone_(
    const std::string& src_dir, const genai::QuantArchInfo& meta,
    int n_layers, const std::string& tok_path,
    const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  metal_compute::MetalCompute* mc = session()->services()->metal_compute();
  if (mc == nullptr) { return {}; }
  if (!meta.calib_ok) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): on-device auto-calibration is not available "
        "for arch '{}' (only dense full-attention Qwen3-family backbones run "
        "through MetalQwenModel); supply calib_dir", this->id(), meta.arch));
    return {};
  }
  genai::MetalQwenModel::Config bcfg = meta.backbone;
  bcfg.n_layers = n_layers;   // honor an explicit n_layers override
  auto tok = genai::Tokenizer::from_huggingface_json(tok_path, session());
  if (!tok) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): auto-calibration needs a tokenizer at '{}'",
        this->id(), tok_path));
    return {};
  }
  // Built-in corpus: ~128 sequences of ~512 tokens from the curated mlx-lm
  // calibration set, chat-template-wrapped so control tokens are exercised.
  std::vector<std::vector<std::int32_t>> corpus =
      genai::build_builtin_calibration_corpus(
          *tok, kCalibSeqs, kCalibSeqLen, /*apply_chat_template=*/true);
  if (corpus.empty()) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): built-in calibration corpus produced no "
        "sequences", this->id()));
    return {};
  }
  const fs::path tmp_q8 =
      vpipe::temp_root() / ("vpipe-mqcal8-" + this->id());
  const fs::path tmp_cal =
      vpipe::temp_root() / ("vpipe-mqcal-" + this->id());
  fs::remove_all(tmp_q8, ec);
  fs::remove_all(tmp_cal, ec);
  auto on_fail = [&](const std::string& what, const std::string& detail) {
    if (stop()) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): quantize stopped during calibration",
          this->id()));
    } else if (!detail.empty()) {
      session()->warn(fmt("ModelQuantizeStage('{}'): {}{}", this->id(), what,
                          detail));
    }
  };
  if (bcfg.is_moe()) {
    // MoE: memory-safe layer-by-layer streaming forward over the bf16 SOURCE
    // (never resides the whole expert stack; no 8-bit base needed).
    session()->info(fmt(
        "ModelQuantizeStage('{}'): auto-calibrating MoE (streaming per-layer "
        "forward over {} corpus seqs)", this->id(), corpus.size()));
    std::string ce;
    if (!genai::collect_backbone_calibration_streaming(
            mc, src_dir, bcfg, corpus, tmp_cal.string(), &ce,
            (std::uint64_t)8 << 30, stop)) {
      on_fail("", ce);
      fs::remove_all(tmp_cal, ec);
      return {};
    }
    return tmp_cal.string();
  }
  session()->info(fmt(
      "ModelQuantizeStage('{}'): auto-calibrating (8-bit base + on-device taps "
      "over {} corpus seqs)", this->id(), corpus.size()));
  {
    genai::QuantizeOptions q8; q8.bits = 8; q8.group = 64;
    genai::ModelQuantizer mq8(mc);
    std::string e8;
    if (!mq8.run(src_dir, tmp_q8.string(), q8, &e8, stop)) {
      on_fail("calib 8-bit base: ", e8);
      fs::remove_all(tmp_q8, ec);
      return {};
    }
  }
  std::string ce;
  if (!genai::collect_backbone_calibration(
          mc, tmp_q8.string(), bcfg, corpus, tmp_cal.string(), &ce, stop)) {
    on_fail("", ce);
    fs::remove_all(tmp_q8, ec);
    fs::remove_all(tmp_cal, ec);
    return {};
  }
  fs::remove_all(tmp_q8, ec);   // the 8-bit base is consumed
  return tmp_cal.string();
}

bool
ModelQuantizeStage::quantize_t2i_pipeline_(
    const std::string& root, const std::string& family,
    const std::string& out_dir, const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;

  const std::string tgt_sub = t2i_target_subdir_(_target);
  if (!_target.empty() && tgt_sub.empty()) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): target '{}' is not a {} component "
        "(want dit | text_encoder | vae)", this->id(), _target, family));
    return false;
  }
  std::string tgt = tgt_sub.empty() ? std::string("transformer") : tgt_sub;
  // Boogu names its text encoder `mllm/` (it is a full Qwen3-VL, not a
  // diffusers text_encoder). Accept the family-neutral `text_encoder` target
  // and resolve it to whichever dir the checkpoint actually ships.
  if (tgt == "text_encoder" && !fs::is_directory(fs::path(root) / tgt, ec) &&
      fs::is_directory(fs::path(root) / "mllm", ec)) {
    tgt = "mllm";
  }
  const fs::path tgt_src = fs::path(root) / tgt;
  if (!fs::is_directory(tgt_src, ec)) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): {} root '{}' has no '{}' component to "
        "quantize", this->id(), family, root, tgt));
    return false;
  }
  if (tgt == "vae") {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): VAE quantization is not supported yet (the "
        "diffusion VAE is a conv net, not a Linear stack); no pass performed",
        this->id()));
    return false;
  }
  if (_skip_existing &&
      fs::exists(fs::path(out_dir) / tgt / "config.json", ec)) {
    session()->info(fmt(
        "ModelQuantizeStage('{}'): output '{}' already has a quantized {}; "
        "skipping", this->id(), out_dir, tgt));
    return true;
  }

  // Wan 2.2's A14B is TWO experts -- transformer/ (high noise) and
  // transformer_2/ (low noise) -- and both are the same DiT class. They are
  // one MODEL, so quantizing them in one pass is the only thing that makes
  // sense: quantizing half of a pair would produce a pipeline that silently
  // changes precision partway down the sigma schedule.
  std::string expert2;
  if (family == "wan" && tgt == "transformer" &&
      fs::exists(fs::path(root) / "transformer_2" / "config.json", ec)) {
    expert2 = "transformer_2";
  }

  session()->info(fmt(
      "ModelQuantizeStage('{}'): {} pipeline '{}' -> '{}' (quantizing {}{}, "
      "copying the other components)", this->id(), family, root, out_dir, tgt,
      expert2.empty() ? std::string() : " + " + expert2));

  // 1. Assemble the self-contained output: hard-link/copy every component
  //    except the target (quantized next), so the result carries all of
  //    tokenizer/scheduler/model_index.json + the other (un- or already-
  //    quantized) sub-models -- usable directly as a generate-image hf_dir.
  fs::create_directories(out_dir, ec);
  {
    // Progress over FILES, not components: on a cross-device destination this
    // is a real byte copy of the sibling sub-models (a ~9 GB text encoder, a
    // multi-GB DiT), which otherwise looks like a hang before the quantizer's
    // own bar appears.
    std::vector<fs::path> comps;
    std::size_t total = 0;
    for (const auto& e : fs::directory_iterator(root, ec)) {
      const std::string leaf = e.path().filename().string();
      if (leaf == tgt) { continue; }
      // Wan A14B's second expert is a DiT too, and it is quantized in this
      // same pass (below) -- so skip it here rather than copying 53 GB of
      // fp32 only to overwrite it.
      if (!expert2.empty() && leaf == "transformer_2") { continue; }
      comps.push_back(e.path());
      total += count_files_(e.path());
    }
    UiProgress bar = session()->open_progress("copy components");
    std::size_t done = 0;
    for (const auto& c : comps) {
      if (!link_or_copy_tree_(c, fs::path(out_dir) / c.filename(), ec, stop,
                              [&] {
                                ++done;
                                bar.update(done, total);
                              })) {
        session()->info(fmt(
            "ModelQuantizeStage('{}'): stopped while assembling '{}'",
            this->id(), out_dir));
        return false;
      }
    }
    bar.finish();
  }
  session()->log_debug(fmt(
      "ModelQuantizeStage('{}'): copied {} components (all but {}) into '{}'",
      this->id(), family, tgt, out_dir));

  // 2. Quantize the target component into out_dir/<tgt>.
  const std::string tgt_out = (fs::path(out_dir) / tgt).string();
  bool ok = false;
  if (tgt == "transformer") {
    ok = quantize_dit_component_(tgt_src.string(), tgt_out, family, root, stop);
    if (ok && !expert2.empty()) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): quantizing the second expert ({})",
          this->id(), expert2));
      ok = quantize_dit_component_((fs::path(root) / expert2).string(),
                                   (fs::path(out_dir) / expert2).string(),
                                   family, root, stop);
    }
  } else if (tgt == "text_encoder" || tgt == "mllm") {
    ok = quantize_text_encoder_(tgt_src.string(), tgt_out, root, stop);
  }
  if (!ok) { return false; }

  session()->log_normal(fmt(
      "ModelQuantizeStage('{}'): quantized {} {} '{}' -> self-contained "
      "'{}' ({}-bit g{}{}{})", this->id(), family, tgt, root, out_dir, _bits,
      _group_size, _mixed ? " mixed" : "", _awq ? " awq" : ""));

  // 3. Register as a full pipeline (usable directly as an hf_dir).
  const bool explicit_path =
      _output_name[0] == '/' ||
      _output_name.rfind("./", 0) == 0 || _output_name.rfind("../", 0) == 0;
  if (!explicit_path) {
    register_output_(_output_name, out_dir, family, _bits);
  }
  return true;
}

std::string
ModelQuantizeStage::comfy_target_subdir_(const std::string& target)
{
  const std::string sub = t2i_target_subdir_(target);
  if (sub == "transformer") { return "diffusion_models"; }
  if (sub == "text_encoder" || sub == "mllm") { return "text_encoders"; }
  if (sub == "vae") { return "vae"; }
  return {};
}

bool
ModelQuantizeStage::is_comfy_root_(const std::string& dir)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  // `diffusion_models/` is the whole test, and it is deliberately the ONLY
  // one. A repack names its DiT role directory that; a diffusers pipeline
  // root never does -- it uses transformer/ (or unet/).
  //
  // The obvious-looking alternative, "scan_repo finds any component", is
  // WRONG and was caught by model_quantize_stage.mage_flow_dit_quantizes:
  // scan_repo's role list includes `vae/`, which every diffusers root also
  // has, and a diffusers VAE's safetensors carries a `__metadata__` like
  // any other -- so every Krea-2 / FLUX.2 / QIE / Mage / Wan checkpoint
  // matched and got routed down the repack path, which then failed
  // looking for a diffusion_models/ that a diffusers tree never has.
  //
  // One check also covers the already-quantized case for free: a pass
  // writes its output as a directory INSIDE the role dir, so
  // `diffusion_models/` exists whether it holds the repack file or the
  // quantized checkpoint. That is what lets a chain re-enter here.
  return fs::is_directory(fs::path(dir) / "diffusion_models", ec) && !ec;
}

// The self-contained pass for a Comfy-Org repack.
//
// Structurally this is quantize_t2i_pipeline_ with one difference, and it is
// worth naming because it is the only thing that makes the layout awkward: a
// repack component is a FILE (`diffusion_models/minimax_h3_fl2va_bf16.
// safetensors`) while a quantized component is a DIRECTORY checkpoint
// (config.json + shards). So the output cannot be byte-for-byte the same
// shape as the input -- the quantized role subdir becomes a directory.
//
// Everything downstream is taught to accept either, which is why that is a
// tolerable asymmetry rather than a new format: MetalMiniMaxH3Transformer::
// resolve_dit_dir and MiniMaxH3TextEncoder::resolve_encoder_dir both probe
// the repack file first and the quantized directory second, so one path
// serves an original repack, a half-quantized chain output, and a fully
// quantized one.
bool
ModelQuantizeStage::quantize_registered_pipeline_(
    const std::string& root, const genai::QuantizableFamily& fam,
    const std::string& out_dir, const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  const std::string family(fam.tag());

  // Pick the component `target` names. Case-insensitive, and an empty
  // target takes the family's FIRST -- which is why the interface says
  // to order the list with the likely one first.
  std::string want;
  for (char c : _target) {
    want.push_back((char)std::tolower((unsigned char)c));
  }
  const std::vector<genai::QuantizableComponent> comps = fam.components();
  const genai::QuantizableComponent* comp = nullptr;
  for (const auto& c : comps) {
    std::string t;
    for (char ch : c.target) {
      t.push_back((char)std::tolower((unsigned char)ch));
    }
    if (want.empty() || t == want) { comp = &c; break; }
  }
  if (comp == nullptr) {
    std::string have;
    for (const auto& c : comps) {
      if (!have.empty()) { have += " | "; }
      have += c.target;
    }
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): '{}' is not a {} component (want {})",
        this->id(), _target, family, have.empty() ? "<none>" : have));
    return false;
  }
  if (comp->role.empty()) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): {} component '{}' names no role subdir",
        this->id(), family, comp->target));
    return false;
  }

  // Resolve what to read. Either the published repack FILE, or -- when a
  // previous pass already quantized this component -- the directory
  // checkpoint it left in the same place. Accepting both is what lets a
  // chain quantize the components in any order.
  std::string tgt_src = genai::comfy::resolve_component(
      root, comp->role, comp->meta_key, comp->prefer);
  if (tgt_src.empty() &&
      fs::exists(fs::path(root) / comp->role / "config.json", ec)) {
    tgt_src = (fs::path(root) / comp->role).string();
  }
  if (tgt_src.empty()) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): '{}' has no readable {} component in "
        "'{}/' (looked for `{}` in the safetensors metadata, preferring {})",
        this->id(), root, family, comp->role, comp->meta_key,
        comp->prefer.empty() ? "any" : comp->prefer.front()));
    return false;
  }
  if (_skip_existing &&
      fs::exists(fs::path(out_dir) / comp->role / "config.json", ec)) {
    session()->info(fmt(
        "ModelQuantizeStage('{}'): output '{}' already has a quantized {}; "
        "skipping", this->id(), out_dir, comp->role));
    return true;
  }

  session()->info(fmt(
      "ModelQuantizeStage('{}'): {} repack '{}' -> '{}' (quantizing {}, "
      "copying the other components)", this->id(), family, root, out_dir,
      comp->role));

  // 1. Assemble the self-contained output. Hard-linked where the
  //    destination is on the same device, so the components this pass
  //    does not touch cost no bytes -- which is what makes chaining a
  //    second component affordable.
  fs::create_directories(out_dir, ec);
  {
    std::vector<fs::path> others;
    std::size_t total = 0;
    for (const auto& e : fs::directory_iterator(root, ec)) {
      if (e.path().filename().string() == comp->role) { continue; }
      others.push_back(e.path());
      total += count_files_(e.path());
    }
    UiProgress bar = session()->open_progress("copy components");
    std::size_t done = 0;
    for (const auto& c : others) {
      if (!link_or_copy_tree_(c, fs::path(out_dir) / c.filename(), ec, stop,
                              [&] {
                                ++done;
                                bar.update(done, total);
                              })) {
        session()->info(fmt(
            "ModelQuantizeStage('{}'): stopped while assembling '{}'",
            this->id(), out_dir));
        return false;
      }
    }
    bar.finish();
  }

  // 2. Quantize the component into out_dir/<role>, with the family's own
  //    scope. The stage's `quant_exclude` is APPENDED to the family's
  //    rather than replacing it: the family's entries are architectural
  //    facts about its own weights, not defaults for a caller to drop.
  metal_compute::MetalCompute* mc = session()->services()->metal_compute();
  if (mc == nullptr) { return false; }
  genai::QuantizeOptions opt;
  opt.bits              = _bits;
  opt.group             = _group_size;
  opt.quant_embeddings  = false;
  opt.quant_scope       = comp->scope;
  opt.quant_all_in_scope = comp->all_in_scope;
  opt.component_tag     = comp->component_tag;
  {
    auto add_csv = [&opt](const std::string& csv) {
      std::string cur;
      auto flush = [&]() {
        const std::size_t a = cur.find_first_not_of(" \t");
        const std::size_t b = cur.find_last_not_of(" \t");
        if (a != std::string::npos) {
          opt.quant_exclude.push_back(cur.substr(a, b - a + 1));
        }
        cur.clear();
      };
      for (char c : csv) {
        if (c == ',') { flush(); } else { cur.push_back(c); }
      }
      flush();
    };
    add_csv(comp->exclude);
    add_csv(_quant_exclude);
  }

  const std::string tgt_out = (fs::path(out_dir) / comp->role).string();
  genai::ModelQuantizer mq(mc);
  std::string err;
  if (!mq.run(tgt_src, tgt_out, opt, &err, stop)) {
    if (stop()) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): {} quantize stopped; '{}' is incomplete "
          "-- remove it before reuse", this->id(), family, tgt_out));
    } else {
      session()->warn(fmt("ModelQuantizeStage('{}'): {} {}: {}", this->id(),
                          family, comp->role, err));
    }
    return false;
  }

  session()->log_normal(fmt(
      "ModelQuantizeStage('{}'): quantized {} {} '{}' -> self-contained "
      "'{}' ({}-bit g{}{})", this->id(), family, comp->role, root, out_dir,
      _bits, _group_size,
      comp->scope.empty() ? "" : (", scope '" + comp->scope + "'")));

  // 3. Register as a full model root, exactly as the built-in paths do.
  const bool explicit_path =
      _output_name[0] == '/' ||
      _output_name.rfind("./", 0) == 0 || _output_name.rfind("../", 0) == 0;
  if (!explicit_path) {
    register_output_(_output_name, out_dir, family, _bits);
  }
  return true;
}

bool
ModelQuantizeStage::quantize_comfy_pipeline_(
    const std::string& root, const std::string& family,
    const std::string& dit_path, const std::string& out_dir,
    const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;

  const std::string tgt = comfy_target_subdir_(_target);
  if (tgt.empty()) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): target '{}' is not a {} component "
        "(want dit | text_encoder)", this->id(), _target, family));
    return false;
  }
  if (tgt == "vae") {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): VAE quantization is not supported (the "
        "diffusion VAE is a conv net, not a Linear stack); no pass performed",
        this->id()));
    return false;
  }
  if (!fs::is_directory(fs::path(root) / tgt, ec)) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): {} repack '{}' has no '{}/' component to "
        "quantize", this->id(), family, root, tgt));
    return false;
  }
  // Resolve the component to read. Both arms accept the repack FILE or an
  // already-quantized directory, so a chain can quantize in either order.
  std::string tgt_src;
  if (tgt == "diffusion_models") {
    tgt_src = dit_path;
  } else {
    tgt_src = genai::comfy::resolve_component(root, "text_encoders",
                                              "minimax_h3_te", {"qwen3vl"});
    if (tgt_src.empty() &&
        fs::exists(fs::path(root) / tgt / "config.json", ec)) {
      tgt_src = (fs::path(root) / tgt).string();   // already quantized
    }
  }
  if (tgt_src.empty()) {
    // The catalogue pins the bf16 encoder, so an empty resolution means
    // either a partial checkout or a repo holding only the int8_convrot /
    // nvfp4_awq packings, which resolve_component skips because nothing
    // here reads them. Name both: the fix differs.
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): '{}' has no READABLE {} -- either the "
        "checkout is partial (the catalogue pins the bf16 one) or it holds "
        "only packings this build cannot read", this->id(), root, tgt));
    return false;
  }
  if (_skip_existing &&
      fs::exists(fs::path(out_dir) / tgt / "config.json", ec)) {
    session()->info(fmt(
        "ModelQuantizeStage('{}'): output '{}' already has a quantized {}; "
        "skipping", this->id(), out_dir, tgt));
    return true;
  }

  session()->info(fmt(
      "ModelQuantizeStage('{}'): {} repack '{}' -> '{}' (quantizing {}, "
      "copying the other components)", this->id(), family, root, out_dir,
      tgt));

  // 1. Assemble the self-contained output: every top-level entry except the
  //    target role, hard-linked where the destination is on the same device
  //    so the pass-through costs no extra bytes.
  fs::create_directories(out_dir, ec);
  {
    std::vector<fs::path> comps;
    std::size_t total = 0;
    for (const auto& e : fs::directory_iterator(root, ec)) {
      if (e.path().filename().string() == tgt) { continue; }
      comps.push_back(e.path());
      total += count_files_(e.path());
    }
    UiProgress bar = session()->open_progress("copy components");
    std::size_t done = 0;
    for (const auto& c : comps) {
      if (!link_or_copy_tree_(c, fs::path(out_dir) / c.filename(), ec, stop,
                              [&] {
                                ++done;
                                bar.update(done, total);
                              })) {
        session()->info(fmt(
            "ModelQuantizeStage('{}'): stopped while assembling '{}'",
            this->id(), out_dir));
        return false;
      }
    }
    bar.finish();
  }
  session()->log_debug(fmt(
      "ModelQuantizeStage('{}'): copied {} components (all but {}) into '{}'",
      this->id(), family, tgt, out_dir));

  // 2. Quantize the target into out_dir/<role>.
  const std::string tgt_out = (fs::path(out_dir) / tgt).string();
  bool ok = false;
  if (tgt == "diffusion_models") {
    // calib_root is the REPO, which is where an on-device DiT calibration
    // would read the encoder and tokenizer from.
    ok = quantize_dit_component_(tgt_src, tgt_out, family, root, stop);
  } else {
    ok = quantize_text_encoder_(tgt_src, tgt_out, root, stop);
  }
  if (!ok) { return false; }

  session()->log_normal(fmt(
      "ModelQuantizeStage('{}'): quantized {} {} '{}' -> self-contained "
      "'{}' ({}-bit g{}{}{})", this->id(), family, tgt, root, out_dir, _bits,
      _group_size, _mixed ? " mixed" : "", _awq ? " awq" : ""));

  // 3. Register as a full model root, exactly as the diffusers path does.
  const bool explicit_path =
      _output_name[0] == '/' ||
      _output_name.rfind("./", 0) == 0 || _output_name.rfind("../", 0) == 0;
  if (!explicit_path) {
    register_output_(_output_name, out_dir, family, _bits);
  }
  return true;
}

Job
ModelQuantizeStage::process(RuntimeContext& ctx)
{
  if (ctx.stop_requested()) {
    ctx.signal_done();
    co_return;
  }
  // Optional trigger (see model-fetch): gate the work on one beat when the
  // iport is wired so this stage can cascade in a recipe.
  if (ctx.iport_connected(0)) {
    auto trig = co_await ctx.read(0);
    if (!trig) {
      ctx.signal_done();
      co_return;
    }
  }
  // Source availability is checked HERE, AFTER the trigger -- in a recipe
  // the upstream model-fetch may not have downloaded it at config time.
  // Missing source => log + halt WITHOUT emitting a summary, so the
  // cascade stops here instead of quantizing a nonexistent model.
  if (!model_dir_available(session(), _src_model)) {
    session()->error(fmt(
        "ModelQuantizeStage('{}'): source model '{}' is not available "
        "(not downloaded yet?); skipping quantization",
        this->id(), _src_model));
    ctx.signal_done();
    co_return;
  }
  session()->info(fmt(
      "ModelQuantizeStage('{}'): quantizing '{}' -> '{}' ({}-bit g{}{}{})",
      this->id(), _src_model, _output_name, _bits, _group_size,
      _awq ? (_awq_clip ? " awq+clip" : " awq") : "",
      _mixed ? " mixed" : ""));
  const bool ok = quantize_once([&ctx] { return ctx.stop_requested(); });
  // Emit the summary only on success, so a failed quantization halts the
  // cascade (mirrors model-benchmark / -eval, which emit only on success).
  if (ok && ctx.has_consumers(0)) {
    FlexData summary = FlexData::make_object();
    auto so = summary.as_object();
    so.insert_or_assign("stage", FlexData::make_string("model-quantize"));
    so.insert_or_assign("source", FlexData::make_string(_src_model));
    so.insert_or_assign("output", FlexData::make_string(_output_name));
    so.insert_or_assign("bits", FlexData::make_int(_bits));
    so.insert_or_assign("group_size", FlexData::make_int(_group_size));
    so.insert_or_assign("quantized", FlexData::make_bool(ok));
    so.insert_or_assign("text", FlexData::make_string(
        fmt("[model-quantize] {} -> {} ({}-bit, group {}) [{}]",
            _src_model, _output_name, _bits, _group_size,
            ok ? "ok" : "failed")()));
    co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(summary)));
  }
  ctx.signal_done();
  co_return;
}

VPIPE_REGISTER_STAGE(ModelQuantizeStage)
VPIPE_REGISTER_SPEC(ModelQuantizeStage, kSpec)

}  // namespace vpipe
