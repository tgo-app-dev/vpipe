#include "generative-models/minimax-h3/minimax-h3-diffusers-layout.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/fp8-layout.h"

#include <cctype>
#include <filesystem>
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

bool
is_diffusers_dit_file(const std::string& path)
{
  std::error_code ec;
  const std::filesystem::path p(path);
  if (!std::filesystem::is_regular_file(p, ec) || ec ||
      p.extension() != ".safetensors") {
    return false;
  }
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

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
