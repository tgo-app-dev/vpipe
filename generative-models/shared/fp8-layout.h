#ifndef VPIPE_GENERATIVE_MODELS_SHARED_FP8_LAYOUT_H
#define VPIPE_GENERATIVE_MODELS_SHARED_FP8_LAYOUT_H

// fp8-layout.h -- how a checkpoint stores its FP8 weights, PLAIN or SCALED.
//
// Plain FP8 is a dtype: an F8_E4M3 / F8_E5M2 tensor is its own value
// (shared/fp8.h). SCALED FP8 is the same codes with a multiplier beside
// them, value = code * scale, and the multiplier is only found by name.
// Three spellings are in circulation, all from ComfyUI's tooling:
//
//   legacy     a `scaled_fp8` MARKER tensor at the model prefix (its dtype
//              names the format; an f32 marker means E4M3), and per layer
//              `<layer>.scale_weight` beside `<layer>.weight`.
//   comfy_quant  per layer a `<layer>.comfy_quant` u8 tensor holding JSON
//              {"format": "float8_e4m3fn" | "float8_e5m2" | ...} and the
//              scale as `<layer>.weight_scale`. The weight may be stored as
//              plain U8 -- the JSON is then the ONLY thing that says it is
//              FP8.
//   header     the same per-layer formats in the safetensors header, as
//              `__metadata__["_quantization_metadata"]` = {"layers":
//              {"<layer>": {"format": ...}}}.
//
// A fourth, torchao's serialized Float8Tensor (unsloth's pre-quantized
// checkpoints), spells a layer `<layer>._weight_qdata` + `<layer>.
// _weight_scale`; translate_torchao() renames it into the comfy_quant
// spelling at open, so everything below sees one form.
//
// The scale is one value for the tensor (what every ComfyUI writer
// produces), or one per output row (torchao's per-row granularity).
// Anything else is not FP8-with-a-scale and is REFUSED by name: block
// scales, MXFP8, NVFP4. So is a scale beside a weight that is not FP8
// (or int8 below), and a second-stage scale (`weight_scale_2`,
// `pre_quant_scale`): each means an encoding whose codes decode to
// garbage without it.
//
// INT8, for a model that asks for it (`int8` on scan / resolve): a
// comfy_quant record {"format": "int8_tensorwise"} over I8 codes and
// their `weight_scale`, one value or one per row -- ComfyUI's int8 --
// value = code * scale; with `"convrot": true` the codes quantize the
// weight ROTATED, W * H^T over groups of `convrot_groupsize` input
// columns (256 by default; H the regular Hadamard of fp8.h), and the
// value is (code * scale) * H, group by group (Weight::rot_group; H is
// symmetric, so H = H^T undoes the rotation).
// Opt-in because the rotation is not elementwise: a reader that widens a
// weight code by code, and every LoRA reader, would load a rotated one as
// noise, so only a model whose widening carries rot_group takes it; to
// any other int8 is refused as before, by name.
//
// W4A8, under the same opt-in: a record {"format": "asym_w4a8_int8",
// "group_size": 16, "convrot": ...} over I8 nibble codes [N, K / 2] and
// `<layer>.weight_s_rel` (E4M3, [N, K / group_size]),
// `<layer>.weight_s_channel` (the row scale, [N]) and an optional
// `<layer>.weight_codebook` ([16]); fp8.h says how they make an int8
// code. A `weight_correction` (the asymmetric form) and the 6-bit
// `w6a8_int8` are refused by name.
//
// `input_scale` / `scale_input` are ACTIVATION scales for an FP8 matmul.
// A model that dequantizes its weights and multiplies in bf16 has no use
// for them, so they are bookkeeping here, exactly like the markers.
//
// Everything is resolved from names, dtypes and small JSON tensors, so a
// reader can ask about one tensor without holding any state (resolve), or
// survey a whole checkpoint once (scan).

#include "generative-models/shared/fp8.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// TOOLKIT: compiled into each plugin (libvpipe_toolkit.a), not exported
// by libvpipe -- see the vpipe_toolkit target and plugin/plugin-abi.h.

namespace vpipe::genai {
class MetalLlamaWeights;
}

namespace vpipe::genai::fp8 {

// How one FP8 (or int8) tensor is stored.
struct Weight {
  Format      format  = Format::kNone;
  std::string scale;              // the scale tensor's name; "" = unscaled
  bool        per_row = false;    // one scale per output row (else one)
  int         rot_group = 0;      // int8 ConvRot: its group; 0 = unrotated
  // W4A8: the relative scales' and the codebook's names ("" = none: code
  // - 8), and the columns a relative scale covers. `scale` is s_channel.
  std::string s_rel, codebook;
  int         qgroup = 0;
};

// A whole checkpoint's FP8 tensors, and the tensors that belong to the
// ENCODING rather than the model (scales, markers, comfy_quant records,
// activation scales) -- which a model must never be handed.
struct Layout {
  std::unordered_map<std::string, Weight> weights;
  std::unordered_set<std::string>         aux;

  bool any() const noexcept { return !weights.empty(); }
  bool scaled() const noexcept;
  const Weight* find(const std::string& name) const noexcept;
  bool is_aux(const std::string& name) const noexcept {
    return aux.count(name) != 0;
  }
};

// Whether `name` is spelled like encoding bookkeeping (a scale, marker,
// comfy_quant record or activation scale). A spelling test only; scan()
// is what decides whether one actually belongs to an FP8 weight.
bool is_aux_name(const std::string& name) noexcept;

// Survey `w`. False + *err for an encoding this build cannot read -- the
// message names the tensor and the format. `int8`: ComfyUI's int8 weights
// read too (see above); else refused by name.
bool scan(const MetalLlamaWeights& w, Layout* out, std::string* err,
          bool int8 = false);

// One tensor. False with *out untouched (format kNone) when `name` is not
// FP8; false + *err when it is FP8 in a form this build cannot read --
// or int8 where `int8` is false.
bool resolve(const MetalLlamaWeights& w, const std::string& name,
             Weight* out, std::string* err, bool int8 = false);

// The scale of `x` as f32: one value, or `rows` of them. Empty for an
// unscaled weight. False + *err when the scale tensor cannot be read.
bool read_scale(const MetalLlamaWeights& w, const Weight& x, std::size_t rows,
                std::vector<float>* out, std::string* err);

// torchao's `<layer>._weight_qdata` / `<layer>._weight_scale` renamed to
// `<layer>.weight` / `<layer>.weight_scale`, in place. Returns how many
// weights were renamed (0 for a checkpoint that does not use the
// spelling, or one where a new name would collide -- then nothing is
// renamed). Whether the result is FP8 this build reads is resolve()'s
// question, as for any other file: an int8 qdata keeps its dtype and is
// refused there by name.
std::size_t translate_torchao(MetalLlamaWeights& w);

// `rows` x `cols` codes to bf16 / f32, with `scale` applied (empty: none;
// one value; or one per row). Unscaled is EXACT. Scaled rounds once --
// the f32 product to bf16, nearest-even -- which is what the GPU
// expansion and ComfyUI compute, so a runtime expansion and a dense
// conversion of the same file agree to the bit. `rot_group` (int8 only,
// ConvRot) un-rotates each row's CODES first, exactly (fp8.h's
// convrot_unrotate), and then scales them: still one rounding.
void decode_bf16(const std::uint8_t* codes, std::uint16_t* dst,
                 std::size_t rows, std::size_t cols, Format f,
                 const std::vector<float>& scale, int rot_group = 0);
// W4A8 (kW4A8) to bf16, [rows, cols] with `cols` the LOGICAL width:
// each code to its int8 code (fp8.h), un-rotated exactly, times the row
// scale and convrot_norm, rounded once -- kI8's decode of the same
// codes, and the GPU widening (w4a8_convrot_dequant) to the bit.
// `packed` [rows, cols / 2], `s_rel` E4M3 [rows, cols / qgroup],
// `codebook` 16 floats or null.
void decode_w4a8_bf16(const std::uint8_t* packed, const std::uint8_t* s_rel,
                      const float* codebook, const std::vector<float>& scale,
                      std::size_t rows, std::size_t cols, int qgroup,
                      int rot_group, std::uint16_t* dst);
// One code's int8 value: round(codebook[c] * s_rel), clamped to +-127.
std::int32_t w4a8_code(std::uint8_t nibble, float s_rel,
                       const float* codebook);

void decode_f32(const std::uint8_t* codes, float* dst, std::size_t rows,
                std::size_t cols, Format f, const std::vector<float>& scale,
                int rot_group = 0);

}  // namespace vpipe::genai::fp8

#endif
