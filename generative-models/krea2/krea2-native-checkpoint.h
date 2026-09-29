#ifndef VPIPE_GENERATIVE_MODELS_KREA2_NATIVE_CHECKPOINT_H
#define VPIPE_GENERATIVE_MODELS_KREA2_NATIVE_CHECKPOINT_H

// Krea-2 checkpoints in Krea's OWN layout, as community fine-tunes ship.
//
// The published Krea-2 repos are diffusers pipelines (transformer/ with a
// config.json and `transformer_blocks.N.attn.to_q` names). Fine-tunes made
// in ComfyUI or on Krea's reference code are published instead as ONE
// .safetensors in the reference model's naming:
//
//   blocks.N.attn.{wq,wk,wv,wo,gate}      transformer_blocks.N.attn.
//                                           {to_q,to_k,to_v,to_out.0,to_gate}
//   blocks.N.attn.qknorm.{q,k}norm.scale  ....attn.norm_{q,k}.weight
//   blocks.N.mlp.{gate,up,down}           ....ff.{gate,up,down}
//   blocks.N.{pre,post}norm.scale         ....norm{1,2}.weight
//   blocks.N.mod.lin          [6*H]       ....scale_shift_table [6, H]
//   txtfusion.*               (same block names)   text_fusion.*
//   first / last.linear / last.norm.scale / last.modulation.lin
//                             img_in / final_layer.{linear, norm.weight,
//                                                   scale_shift_table}
//   tmlp.{0,2} / tproj.1      time_embed.linear_{1,2} / time_mod_proj
//   txtmlp.{0.scale,1,3}      txt_in.{norm.weight, linear_1, linear_2}
//
// with no config.json and no `__metadata__` -- the architecture is only in
// the tensor shapes. The names are the whole difference: every norm is
// zero-centered in BOTH layouts (the model applies 1 + scale), and the one
// reshape is the per-block modulation table, whose flat [6*H] row-major is
// exactly the [6, H] the diffusers model adds its timestep projection to.
// MEASURED on lustifyNSFWCheckpoint_v10Krea2_fp8 against Krea-2-Turbo:
// every mapped tensor, the tables included, agrees to within the FP8
// rounding of Turbo's own values.
//
// The file may carry its DiT keys under a prefix -- a full ComfyUI
// checkpoint says `model.diffusion_model.blocks.0...` -- so detection
// reports the prefix, and a key outside it (a bundled text encoder or
// VAE) is simply not part of the DiT.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

class FlexData;

namespace genai {

class MetalLlamaWeights;

namespace krea2 {

// The prefix under which `w` carries a native Krea-2 DiT ("",
// "model.diffusion_model." or "diffusion_model."), or nullopt when it is
// not one. Keyed on the two tensors no other family has (the same probe
// ComfyUI's model detection uses): `txtfusion.projector.weight` and the
// first block's `mod.lin`.
std::optional<std::string> native_prefix(const MetalLlamaWeights& w);

// Whether `path` is a single .safetensors file holding a native Krea-2
// DiT. Reads the header only.
bool is_native_dit_file(const std::string& path);

// One native tensor (prefix already stripped) in diffusers terms: its
// name, and its shape, which differs only for the modulation tables.
// Scaled FP8's per-layer records (`.scale_weight`, `.weight_scale`,
// `.comfy_quant`, the activation scales) are renamed with their layer,
// and the root `scaled_fp8` marker keeps its name.
// False when the name is not part of the Krea-2 DiT as this layout
// defines it -- a caller converting a checkpoint must treat that as an
// error rather than drop the tensor, since a missing weight loads as
// zeros or not at all.
bool to_diffusers(std::string_view                 native,
                  const std::vector<std::int64_t>& shape,
                  std::string*                     name,
                  std::vector<std::int64_t>*       out_shape);

// Rename `w` IN PLACE from the native layout to the diffusers one (the
// name table only -- see MetalLlamaWeights::rename_tensors), dropping
// anything outside the DiT's prefix. False, with `w` untouched, when `w`
// is not a native Krea-2 DiT or holds a DiT tensor to_diffusers() cannot
// place; untranslatable_tensor() names that tensor. Called by
// MetalLlamaWeights::open_model for every single-file checkpoint, so a
// caller of that never needs to.
bool translate_native(MetalLlamaWeights& w);

// The first tensor in `w`'s DiT namespace that has no diffusers name,
// "" when there is none or `w` is not a native Krea-2 DiT. For the error
// message when translate_native() refuses.
std::string untranslatable_tensor(const MetalLlamaWeights& w);

// The diffusers transformer config.json for a Krea-2 DiT read under the
// DIFFUSERS names (a translated native file, or any diffusers-named
// checkpoint that shipped without one), off its tensor shapes:
// `_class_name` Krea2Transformer2DModel and every field the published
// config carries. False + *err when a shape the config needs is missing
// or inconsistent. `source_layout`, when given, is recorded as
// `_vpipe_source_layout` -- the only trace, once the output is a
// directory, that its names were translated.
bool diffusers_config(const MetalLlamaWeights& w, FlexData& out,
                      std::string* err,
                      const std::string& source_layout = {});

}  // namespace krea2
}  // namespace genai
}  // namespace vpipe

#endif
