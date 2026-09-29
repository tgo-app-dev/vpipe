#include "generative-models/krea2/krea2-native-checkpoint.h"

#include "common/flex-data.h"
#include "generative-models/llama3/metal-llama-weights.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <utility>

namespace vpipe::genai::krea2 {

namespace {

bool
starts_(std::string_view s, std::string_view p)
{
  return s.size() >= p.size() && s.substr(0, p.size()) == p;
}

// "<digits>.<rest>" -> the index and the rest; false otherwise.
bool
split_index_(std::string_view s, std::string* idx, std::string_view* rest)
{
  std::size_t i = 0;
  while (i < s.size() && s[i] >= '0' && s[i] <= '9') { ++i; }
  if (i == 0 || i >= s.size() || s[i] != '.') { return false; }
  *idx  = std::string(s.substr(0, i));
  *rest = s.substr(i + 1);
  return true;
}

// The names inside one block, shared by the main stack and the two
// text-fusion stacks. The modulation table is the main stack's alone.
bool
map_block_(std::string_view leaf, std::string* out)
{
  static const std::pair<std::string_view, std::string_view> kMap[] = {
    {"attn.wq.weight",          "attn.to_q.weight"},
    {"attn.wk.weight",          "attn.to_k.weight"},
    {"attn.wv.weight",          "attn.to_v.weight"},
    {"attn.wo.weight",          "attn.to_out.0.weight"},
    {"attn.gate.weight",        "attn.to_gate.weight"},
    {"attn.qknorm.qnorm.scale", "attn.norm_q.weight"},
    {"attn.qknorm.knorm.scale", "attn.norm_k.weight"},
    {"mlp.gate.weight",         "ff.gate.weight"},
    {"mlp.up.weight",           "ff.up.weight"},
    {"mlp.down.weight",         "ff.down.weight"},
    {"prenorm.scale",           "norm1.weight"},
    {"postnorm.scale",          "norm2.weight"},
  };
  for (const auto& [from, to] : kMap) {
    if (leaf == from) { *out = std::string(to); return true; }
  }
  return false;
}

// Model-level tensors, one to one.
bool
map_top_(std::string_view n, std::string* out)
{
  static const std::pair<std::string_view, std::string_view> kMap[] = {
    {"first.weight",               "img_in.weight"},
    {"first.bias",                 "img_in.bias"},
    {"tmlp.0.weight",              "time_embed.linear_1.weight"},
    {"tmlp.0.bias",                "time_embed.linear_1.bias"},
    {"tmlp.2.weight",              "time_embed.linear_2.weight"},
    {"tmlp.2.bias",                "time_embed.linear_2.bias"},
    {"tproj.1.weight",             "time_mod_proj.weight"},
    {"tproj.1.bias",               "time_mod_proj.bias"},
    {"txtmlp.0.scale",             "txt_in.norm.weight"},
    {"txtmlp.1.weight",            "txt_in.linear_1.weight"},
    {"txtmlp.1.bias",              "txt_in.linear_1.bias"},
    {"txtmlp.3.weight",            "txt_in.linear_2.weight"},
    {"txtmlp.3.bias",              "txt_in.linear_2.bias"},
    {"txtfusion.projector.weight", "text_fusion.projector.weight"},
    {"last.linear.weight",         "final_layer.linear.weight"},
    {"last.linear.bias",           "final_layer.linear.bias"},
    {"last.norm.scale",            "final_layer.norm.weight"},
    {"last.modulation.lin",        "final_layer.scale_shift_table"},
  };
  for (const auto& [from, to] : kMap) {
    if (n == from) { *out = std::string(to); return true; }
  }
  return false;
}

// Leading dimension of a tensor, 0 when absent or not at least 1-D.
std::int64_t
dim0_(const MetalLlamaWeights& w, const std::string& name)
{
  const auto* ti = w.info(name);
  return (ti != nullptr && !ti->shape.empty()) ? ti->shape[0] : 0;
}

std::int64_t
dim1_(const MetalLlamaWeights& w, const std::string& name)
{
  const auto* ti = w.info(name);
  return (ti != nullptr && ti->shape.size() >= 2) ? ti->shape[1] : 0;
}

// How many consecutive `<stack>N.` blocks carry `probe`.
int
count_blocks_(const MetalLlamaWeights& w, const std::string& stack,
              const std::string& probe)
{
  int n = 0;
  while (w.has(stack + std::to_string(n) + "." + probe)) { ++n; }
  return n;
}

}  // namespace

std::optional<std::string>
native_prefix(const MetalLlamaWeights& w)
{
  for (const char* p : {"", "model.diffusion_model.", "diffusion_model."}) {
    const std::string pre = p;
    if (w.has(pre + "txtfusion.projector.weight") &&
        w.has(pre + "blocks.0.mod.lin")) {
      return pre;
    }
  }
  return std::nullopt;
}

bool
is_native_dit_file(const std::string& path)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::is_regular_file(path, ec) || ec ||
      fs::path(path).extension() != ".safetensors") {
    return false;
  }
  const auto w = MetalLlamaWeights::open(path);
  return w.has_value() && native_prefix(*w).has_value();
}

bool
to_diffusers(std::string_view                 native,
             const std::vector<std::int64_t>& shape,
             std::string*                     name,
             std::vector<std::int64_t>*       out_shape)
{
  std::string      idx;
  std::string_view rest;
  std::string      leaf;
  *out_shape = shape;
  // Scaled FP8's bookkeeping (shared/fp8-layout.h) travels WITH its
  // layer: `<layer>.scale_weight` is renamed exactly as `<layer>.weight`
  // is, so the scale is still found beside the weight it multiplies. The
  // legacy `scaled_fp8` marker sits at the model root and keeps its name.
  if (native == "scaled_fp8") {
    *name = "scaled_fp8";
    return true;
  }
  for (const char* suf : {".weight_scale", ".scale_weight", ".input_scale",
                          ".scale_input", ".comfy_quant"}) {
    const std::string_view sv(suf);
    if (native.size() <= sv.size() ||
        native.substr(native.size() - sv.size()) != sv) {
      continue;
    }
    const std::string base =
        std::string(native.substr(0, native.size() - sv.size())) +
        ".weight";
    std::string mapped;
    std::vector<std::int64_t> unused;
    if (!to_diffusers(base, {1, 1}, &mapped, &unused) ||
        mapped.size() <= 7 ||
        mapped.compare(mapped.size() - 7, 7, ".weight") != 0) {
      return false;
    }
    *name = mapped.substr(0, mapped.size() - 7) + std::string(sv);
    return true;
  }
  if (starts_(native, "blocks.") &&
      split_index_(native.substr(7), &idx, &rest)) {
    const std::string pre = "transformer_blocks." + idx + ".";
    if (rest == "mod.lin") {
      // [6*H] -> [6, H]: the six modulation vectors, row-major, which is
      // how the reference chunk(6)s the flat parameter.
      if (shape.size() != 1 || shape[0] % 6 != 0) { return false; }
      *name      = pre + "scale_shift_table";
      *out_shape = {6, shape[0] / 6};
      return true;
    }
    if (!map_block_(rest, &leaf)) { return false; }
    *name = pre + leaf;
    return true;
  }
  for (const char* stack : {"layerwise_blocks.", "refiner_blocks."}) {
    const std::string from = std::string("txtfusion.") + stack;
    if (starts_(native, from) &&
        split_index_(native.substr(from.size()), &idx, &rest)) {
      if (!map_block_(rest, &leaf)) { return false; }
      *name = std::string("text_fusion.") + stack + idx + "." + leaf;
      return true;
    }
  }
  return map_top_(native, name);
}

bool
translate_native(MetalLlamaWeights& w)
{
  const std::optional<std::string> pre = native_prefix(w);
  if (!pre.has_value() || !untranslatable_tensor(w).empty()) {
    return false;
  }
  const std::string p = *pre;
  return w.rename_tensors(
      [&p](const std::string& n, MetalLlamaWeights::TensorInfo& ti) {
        if (n.compare(0, p.size(), p) != 0) { return std::string(); }
        std::string out;
        std::vector<std::int64_t> shape;
        if (!to_diffusers(std::string_view(n).substr(p.size()), ti.shape,
                          &out, &shape)) {
          return std::string();   // unreachable: checked above
        }
        ti.shape = std::move(shape);
        return out;
      });
}

std::string
untranslatable_tensor(const MetalLlamaWeights& w)
{
  const std::optional<std::string> pre = native_prefix(w);
  if (!pre.has_value()) { return {}; }
  std::vector<std::string> names = w.tensor_names();
  std::sort(names.begin(), names.end());
  for (const std::string& n : names) {
    if (n.compare(0, pre->size(), *pre) != 0) { continue; }
    const auto* ti = w.info(n);
    std::string out;
    std::vector<std::int64_t> shape;
    if (ti == nullptr ||
        !to_diffusers(std::string_view(n).substr(pre->size()), ti->shape,
                      &out, &shape)) {
      return n;
    }
  }
  return {};
}

bool
diffusers_config(const MetalLlamaWeights& w, FlexData& out, std::string* err,
                 const std::string& source_layout)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "krea2 checkpoint config: " + m; }
    return false;
  };
  const std::string blk = "transformer_blocks.0.";
  const std::string tb  = "text_fusion.layerwise_blocks.0.";
  const std::int64_t hidden = dim0_(w, "img_in.weight");
  const std::int64_t in_ch  = dim1_(w, "img_in.weight");
  const std::int64_t hd     = dim0_(w, blk + "attn.norm_q.weight");
  const std::int64_t q_rows = dim0_(w, blk + "attn.to_q.weight");
  const std::int64_t k_rows = dim0_(w, blk + "attn.to_k.weight");
  const std::int64_t ffn    = dim0_(w, blk + "ff.gate.weight");
  const std::int64_t tdim   = dim1_(w, "time_embed.linear_1.weight");
  const std::int64_t mod_r  = dim0_(w, blk + "scale_shift_table");
  const std::int64_t mod_c  = dim1_(w, blk + "scale_shift_table");
  const std::int64_t t_hid  = dim0_(w, tb + "norm1.weight");
  const std::int64_t t_hd   = dim0_(w, tb + "attn.norm_q.weight");
  const std::int64_t t_q    = dim0_(w, tb + "attn.to_q.weight");
  const std::int64_t t_k    = dim0_(w, tb + "attn.to_k.weight");
  const std::int64_t t_ffn  = dim0_(w, tb + "ff.gate.weight");
  const std::int64_t t_lay  = dim1_(w, "text_fusion.projector.weight");
  if (hidden <= 0 || in_ch <= 0 || hd <= 0 || q_rows <= 0 || k_rows <= 0 ||
      ffn <= 0 || tdim <= 0 || t_hid <= 0 || t_hd <= 0 || t_q <= 0 ||
      t_k <= 0 || t_ffn <= 0 || t_lay <= 0) {
    return fail("a tensor the config is read from is missing");
  }
  if (q_rows % hd != 0 || k_rows % hd != 0 || q_rows != hidden ||
      t_q % t_hd != 0 || t_k % t_hd != 0 || mod_r != 6 || mod_c != hidden ||
      hd % 16 != 0) {
    return fail("tensor shapes do not describe a Krea-2 DiT");
  }
  const int n_layers =
      count_blocks_(w, "transformer_blocks.", "scale_shift_table");
  const int n_lw = count_blocks_(w, "text_fusion.layerwise_blocks.",
                                 "norm1.weight");
  const int n_rf = count_blocks_(w, "text_fusion.refiner_blocks.",
                                 "norm1.weight");
  if (n_layers <= 0 || n_lw <= 0 || n_rf <= 0) {
    return fail("no transformer or text-fusion blocks");
  }

  out = FlexData::make_object();
  auto o = out.as_object();
  auto put = [&](const char* k, std::int64_t v) {
    o.insert_or_assign(k, FlexData::make_int(v));
  };
  o.insert_or_assign("_class_name",
                     FlexData::make_string("Krea2Transformer2DModel"));
  put("attention_head_dim", hd);
  // The reference model's rope split, which the diffusers config states
  // as a list: 32/48/48 at head_dim 128.
  {
    FlexData axes = FlexData::make_array();
    auto a = axes.as_array();
    const std::int64_t s = hd / 16;
    a.push_back(FlexData::make_int(hd - 12 * s));
    a.push_back(FlexData::make_int(6 * s));
    a.push_back(FlexData::make_int(6 * s));
    o.insert_or_assign("axes_dims_rope", std::move(axes));
  }
  put("in_channels", in_ch);
  put("intermediate_size", ffn);
  o.insert_or_assign("norm_eps", FlexData::make_real(1e-5));
  put("num_attention_heads", q_rows / hd);
  put("num_key_value_heads", k_rows / hd);
  put("num_layers", n_layers);
  put("num_layerwise_text_blocks", n_lw);
  put("num_refiner_text_blocks", n_rf);
  put("num_text_layers", t_lay);
  o.insert_or_assign("rope_theta", FlexData::make_real(1000.0));
  put("text_hidden_dim", t_hid);
  put("text_intermediate_size", t_ffn);
  put("text_num_attention_heads", t_q / t_hd);
  put("text_num_key_value_heads", t_k / t_hd);
  put("timestep_embed_dim", tdim);
  // Where these weights came from. Nothing reads it; it is the only
  // record, once the output is a directory, that its names were
  // translated rather than published this way.
  if (!source_layout.empty()) {
    o.insert_or_assign("_vpipe_source_layout",
                       FlexData::make_string(source_layout));
  }
  return true;
}

}  // namespace vpipe::genai::krea2
