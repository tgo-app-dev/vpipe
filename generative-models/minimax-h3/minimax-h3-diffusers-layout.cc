#include "generative-models/minimax-h3/minimax-h3-diffusers-layout.h"

#include "common/flex-data.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/fp8-layout.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <utility>

namespace vpipe {
namespace genai {
namespace minimax_h3 {

namespace {

// The layer-prefix renames, longest first where one prefixes another.
// The suffix after the prefix -- `weight`, `bias`, `weight_scale`, a
// comfy_quant record -- is carried as it is.
constexpr std::pair<std::string_view, std::string_view> kTop[] = {
    {"audio_proj_in.", "audio_patch_proj."},
    {"audio_proj_out.", "final_layer.audio_out."},
    {"proj_in.", "video_patch_proj."},
    {"proj_out.", "final_layer.video_out."},
    {"time_embedder.linear_1.", "time_embedder.proj_in."},
    {"time_embedder.linear_2.", "time_embedder.proj_out."},
    {"context_embedder.", "condition_proj."},
    {"norm_out.linear.", "final_layer.adaln_proj.linear."},
    {"norm_out.norm.", "final_layer.norm."},
    {"token_refiner.final_norm.", "token_refiner.final_norm."},
};

// Inside a block (main or refiner).
constexpr std::pair<std::string_view, std::string_view> kBlock[] = {
    {"attn.to_q.", "attn.qkv_proj.q."},
    {"attn.to_k.", "attn.qkv_proj.k."},
    {"attn.to_v.", "attn.qkv_proj.v."},
    {"attn.to_out.0.", "attn.out_proj."},
    {"attn.norm_q.", "attn.q_norm."},
    {"attn.norm_k.", "attn.k_norm."},
    {"ff.net.0.proj.", "mlp.fc1.ug."},
    {"ff.net.2.", "mlp.fc2."},
    {"adaln_proj.linear.", "adaln_proj.linear."},
    {"norm1.", "norm1."},
    {"norm2.", "norm2."},
};

bool
starts_(std::string_view s, std::string_view p)
{
  return s.size() >= p.size() && s.substr(0, p.size()) == p;
}

// `rest` after a "<prefix><digits>." block head: the new block head plus
// the translated remainder, or "" when the remainder is not a block
// tensor this knows.
std::string
block_(std::string_view name, std::string_view from, std::string_view to)
{
  if (!starts_(name, from)) { return {}; }
  std::size_t i = from.size();
  const std::size_t d0 = i;
  while (i < name.size() && std::isdigit((unsigned char)name[i])) { ++i; }
  if (i == d0 || i >= name.size() || name[i] != '.') { return {}; }
  const std::string_view idx = name.substr(d0, i - d0);
  const std::string_view rest = name.substr(i + 1);
  for (const auto& [a, b] : kBlock) {
    if (starts_(rest, a)) {
      return std::string(to) + std::string(idx) + "." + std::string(b) +
             std::string(rest.substr(a.size()));
    }
  }
  return {};
}

// The model's name for `name`, "" when it has none. Encoding
// bookkeeping at the top level (a legacy `scaled_fp8` marker) keeps its
// name; a per-layer record rides with its layer like any suffix.
std::string
native_(const std::string& name)
{
  if (name == "scaled_fp8") { return name; }
  std::string s = block_(name, "transformer_blocks.", "blocks.");
  if (!s.empty()) { return s; }
  s = block_(name, "token_refiner.refiner_blocks.", "token_refiner.blocks.");
  if (!s.empty()) { return s; }
  for (const auto& [a, b] : kTop) {
    if (starts_(name, a)) {
      return std::string(b) + name.substr(a.size());
    }
  }
  return {};
}

}  // namespace

bool
is_diffusers_layout(const MetalLlamaWeights& w)
{
  // Three tensors no other checkpoint in this tree spells together: the
  // per-block AdaLN projection, the refiner's split attention, and the
  // text projection. A LoRA for this layout names the first two with a
  // `.lora_A` in the middle, so it is not mistaken for the model.
  auto has = [&](const char* a, const char* b) {
    return w.has(a) || w.has(b);
  };
  return has("transformer_blocks.0.adaln_proj.linear.weight",
             "transformer_blocks.0.adaln_proj.linear._weight_qdata") &&
         has("token_refiner.refiner_blocks.0.attn.to_q.weight",
             "token_refiner.refiner_blocks.0.attn.to_q._weight_qdata") &&
         w.has("context_embedder.weight");
}

std::string
untranslatable_tensor(const MetalLlamaWeights& w)
{
  if (!is_diffusers_layout(w)) { return {}; }
  for (const std::string& n : w.tensor_names()) {
    if (native_(n).empty()) { return n; }
  }
  return {};
}

namespace {

// A regular file named `.safetensors`, in any case -- as config_from_json
// takes a single file, so a file it opens is never refused here for the
// spelling of its extension.
bool
safetensors_file_(const std::filesystem::path& p)
{
  std::error_code ec;
  if (!std::filesystem::is_regular_file(p, ec) || ec) { return false; }
  std::string ext = p.extension().string();
  for (char& c : ext) { c = (char)std::tolower((unsigned char)c); }
  return ext == ".safetensors";
}

// The ComfyUI prefix every tensor of `w` carries over an H3 DiT's names,
// "" for none. Longest first: the second is the first's tail.
std::string
comfy_prefix_(const MetalLlamaWeights& w)
{
  const std::vector<std::string> names = w.tensor_names();
  if (names.empty()) { return {}; }
  for (const char* p : {"model.diffusion_model.", "diffusion_model."}) {
    const std::string pre(p);
    bool all = true;
    for (const std::string& n : names) {
      if (!n.starts_with(pre)) { all = false; break; }
    }
    if (all && w.has(pre + "blocks.0.attn.qkv_proj.weight") &&
        w.has(pre + "audio_patch_proj.weight") &&
        w.has(pre + "condition_proj.weight")) {
      return pre;
    }
  }
  return {};
}

}  // namespace

std::size_t
translate_comfy_prefix(MetalLlamaWeights& w)
{
  const std::string pre = comfy_prefix_(w);
  if (pre.empty()) { return 0; }
  const std::size_t n = w.tensor_names().size();
  const bool ok = w.rename_tensors(
      [&](const std::string& name, MetalLlamaWeights::TensorInfo&) {
        return name.substr(pre.size());
      });
  return ok ? n : 0;
}

bool
is_comfy_saved_dit_file(const std::string& path)
{
  if (!safetensors_file_(path)) { return false; }
  // Opened RAW, with no translation, so the prefix is still there.
  const auto w = MetalLlamaWeights::open(path);
  return w.has_value() && !comfy_prefix_(*w).empty();
}

bool
is_diffusers_dit_file(const std::string& path)
{
  if (!safetensors_file_(path)) { return false; }
  // Opened, so the translation has run: its assembled pieces are the
  // tell, and a file it could not place whole does not have them.
  const auto w = MetalLlamaWeights::open_model(path);
  return w.has_value() && w->has("blocks.0.attn.qkv_proj.q.weight") &&
         w->has("blocks.0.mlp.fc1.ug.weight");
}

std::size_t
translate_diffusers(MetalLlamaWeights& w)
{
  if (!is_diffusers_layout(w) || !untranslatable_tensor(w).empty()) {
    return 0;
  }
  const std::size_t n = w.tensor_names().size();
  const bool ok = w.rename_tensors(
      [](const std::string& name, MetalLlamaWeights::TensorInfo&) {
        return native_(name);
      });
  return ok ? n : 0;
}

std::vector<Part>
assembled_parts(const MetalLlamaWeights& w, const std::string& nm)
{
  if (w.has(nm + ".weight")) { return {}; }
  auto rows_of = [&](const std::string& layer) -> std::int64_t {
    const auto* ti = w.info(layer + ".weight");
    return ti != nullptr && ti->shape.size() == 2 ? ti->shape[0] : -1;
  };
  // qkv: to_q, to_k, to_v stacked, in that order.
  {
    std::vector<Part> p;
    for (const char* s : {".q", ".k", ".v"}) {
      const std::int64_t r = rows_of(nm + s);
      if (r <= 0) { p.clear(); break; }
      p.push_back({nm + s, 0, r});
    }
    if (!p.empty()) { return p; }
  }
  // fc1: up | gate on disk, gate | up in the model.
  {
    const std::int64_t r = rows_of(nm + ".ug");
    if (r > 0 && r % 2 == 0) {
      return {{nm + ".ug", r / 2, r / 2}, {nm + ".ug", 0, r / 2}};
    }
  }
  return {};
}

// ---- the VAEs ---------------------------------------------------------

namespace {

// Inside a decoder block.
constexpr std::pair<std::string_view, std::string_view> kVaeBlock[] = {
    {"attn.to_q.", "attn.to_qkv.q."},
    {"attn.to_k.", "attn.to_qkv.k."},
    {"attn.to_v.", "attn.to_qkv.v."},
    {"attn.to_out.0.", "attn.to_out."},
    {"ff.net.0.proj.", "ff.w1.ug."},
    {"ff.net.2.", "ff.w2."},
    {"norm1.", "norm1."},
    {"norm2.", "norm2."},
};

// Names that are the same in both, whole or as a layer prefix.
constexpr std::string_view kVaeSame[] = {
    "decoder.proj_out.", "decoder.norm_out.", "encoder.conv_in.",
    "encoder.conv_out.", "encoder.norm_out.", "quant_conv.",
    "post_quant_conv.",
};

// "<digits>." at the start of `s`: the index and what follows it, or
// false.
bool
index_(std::string_view s, std::string_view* idx, std::string_view* rest)
{
  std::size_t i = 0;
  while (i < s.size() && std::isdigit((unsigned char)s[i])) { ++i; }
  if (i == 0 || i >= s.size() || s[i] != '.') { return false; }
  *idx = s.substr(0, i);
  *rest = s.substr(i + 1);
  return true;
}

// The video VAE's native name for `name`, "" when it has none.
std::string
native_vae_(const std::string& name)
{
  const std::string_view n(name);
  if (n == "decoder.register_tokens") { return name; }
  if (starts_(n, "decoder.proj_in.")) {
    return "decoder.x_embedder." + name.substr(16);
  }
  for (std::string_view s : kVaeSame) {
    if (starts_(n, s)) { return name; }
  }
  std::string_view idx, rest;
  constexpr std::string_view kBlk = "decoder.transformer_blocks.";
  if (starts_(n, kBlk) && index_(n.substr(kBlk.size()), &idx, &rest)) {
    const std::string head = std::string(kBlk) + std::string(idx) + ".";
    if (rest == "scale1" || rest == "scale2") {
      return head + std::string(rest);
    }
    for (const auto& [a, b] : kVaeBlock) {
      if (starts_(rest, a)) {
        return head + std::string(b) + std::string(rest.substr(a.size()));
      }
    }
    return {};
  }
  constexpr std::string_view kDown = "encoder.down_blocks.";
  if (starts_(n, kDown) && index_(n.substr(kDown.size()), &idx, &rest)) {
    const std::string lvl = "encoder.down." + std::string(idx) + ".";
    constexpr std::string_view kRes = "resnets.";
    std::string_view j, tail;
    if (starts_(rest, kRes) && index_(rest.substr(kRes.size()), &j, &tail)) {
      const std::string blk = lvl + "block." + std::string(j) + ".";
      if (starts_(tail, "conv_shortcut.")) {
        return blk + "nin_shortcut." + std::string(tail.substr(14));
      }
      for (std::string_view p : {"norm1.", "norm2.", "conv1.", "conv2."}) {
        if (starts_(tail, p)) { return blk + std::string(tail); }
      }
      return {};
    }
    constexpr std::string_view kDs = "downsamplers.0.conv.";
    if (starts_(rest, kDs)) {
      return lvl + "downsample.conv." + std::string(rest.substr(kDs.size()));
    }
  }
  return {};
}

}  // namespace

bool
is_diffusers_vae_dir(const std::string& dir, const char* cls)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::is_directory(fs::path(dir), ec) || ec) { return false; }
  std::ifstream in(fs::path(dir) / "config.json");
  if (!in) { return false; }
  FlexData fd;
  try {
    fd = FlexData::from_json(in);
  } catch (...) {
    return false;
  }
  if (!fd.is_object() || !fd.as_object().contains("_class_name")) {
    return false;
  }
  const FlexData c = fd.as_object().at("_class_name");   // a VIEW: bind it
  return c.is_string() && c.as_string("") == cls;
}

bool
is_diffusers_pipeline_root(const std::string& dir)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path p(dir);
  if (!fs::is_directory(p, ec) || ec) { return false; }
  if (fs::exists(p / "transformer_ref" / "config.json", ec)) { return true; }
  std::ifstream in(p / "model_index.json");
  if (!in) { return false; }
  FlexData fd;
  try {
    fd = FlexData::from_json(in);
  } catch (...) {
    return false;
  }
  if (!fd.is_object() || !fd.as_object().contains("_class_name")) {
    return false;
  }
  const FlexData c = fd.as_object().at("_class_name");   // a VIEW: bind it
  return c.is_string() && c.as_string("") == kDiffusersPipelineClass;
}

std::string
diffusers_pipeline_dit(const std::string& root, const std::string& partition)
{
  namespace fs = std::filesystem;
  std::string want = partition;
  for (char& c : want) { c = (char)std::tolower((unsigned char)c); }
  const char* sub = want == "ref2va" ? "transformer_ref"
                  : (want.empty() || want == "fl2va") ? "transformer"
                                                       : nullptr;
  if (sub == nullptr) { return {}; }
  const fs::path d = fs::path(root) / sub;
  std::error_code ec;
  return fs::exists(d / "config.json", ec) ? d.string() : std::string();
}

std::string
diffusers_pipeline_partition(const std::string& dit_dir)
{
  namespace fs = std::filesystem;
  fs::path d = fs::path(dit_dir).lexically_normal();
  if (d.filename().empty()) { d = d.parent_path(); }   // a trailing '/'
  if (!is_diffusers_pipeline_root(d.parent_path().string())) { return {}; }
  if (d.filename() == "transformer_ref") { return "ref2va"; }
  if (d.filename() == "transformer") { return "fl2va"; }
  return {};
}

bool
is_diffusers_vae_layout(const MetalLlamaWeights& w)
{
  // The decoder's renamed input projection, its split attention and the
  // encoder's diffusers block names, together. The native file has none
  // of them, and the DiT in either naming has no `decoder.` at all.
  return w.has("decoder.proj_in.weight") &&
         w.has("decoder.register_tokens") &&
         w.has("decoder.transformer_blocks.0.attn.to_q.weight") &&
         w.has("encoder.down_blocks.0.resnets.0.conv1.weight");
}

std::size_t
translate_diffusers_vae(MetalLlamaWeights& w)
{
  if (!is_diffusers_vae_layout(w)) { return 0; }
  const std::vector<std::string> names = w.tensor_names();
  for (const std::string& n : names) {
    if (native_vae_(n).empty()) { return 0; }   // keeps its names
  }
  const bool ok = w.rename_tensors(
      [](const std::string& name, MetalLlamaWeights::TensorInfo&) {
        return native_vae_(name);
      });
  return ok ? names.size() : 0;
}

std::vector<Part>
vae_assembled_parts(const MetalLlamaWeights& w, const std::string& nm,
                    int head_dim)
{
  if (w.has(nm + ".weight") || head_dim <= 0) { return {}; }
  auto rows_of = [&](const std::string& layer) -> std::int64_t {
    const auto* ti = w.info(layer + ".weight");
    return ti != nullptr && ti->shape.size() == 2 ? ti->shape[0] : -1;
  };
  // to_qkv: one head of q, then of k, then of v, head after head.
  {
    const std::int64_t r = rows_of(nm + ".q");
    if (r > 0 && r % head_dim == 0 && rows_of(nm + ".k") == r &&
        rows_of(nm + ".v") == r) {
      std::vector<Part> p;
      for (std::int64_t h = 0; h < r / head_dim; ++h) {
        for (const char* s : {".q", ".k", ".v"}) {
          p.push_back({nm + s, h * head_dim, head_dim});
        }
      }
      return p;
    }
  }
  // w1: up | gate on disk, gate | up in the model -- as the DiT's fc1.
  {
    const std::int64_t r = rows_of(nm + ".ug");
    if (r > 0 && r % 2 == 0) {
      return {{nm + ".ug", r / 2, r / 2}, {nm + ".ug", 0, r / 2}};
    }
  }
  return {};
}

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
