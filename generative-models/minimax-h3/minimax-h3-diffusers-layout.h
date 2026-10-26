#ifndef VPIPE_GENERATIVE_MODELS_MINIMAX_H3_DIFFUSERS_LAYOUT_H
#define VPIPE_GENERATIVE_MODELS_MINIMAX_H3_DIFFUSERS_LAYOUT_H

// minimax-h3-diffusers-layout.h -- MiniMax-H3 in the DIFFUSERS naming,
// read under the model's own.
//
// MiniMaxAI publishes the DiT twice: in its own naming (`blocks.N.attn.
// qkv_proj`, what this tree's loader reads, and what Comfy-Org repacks)
// and as a diffusers `MiniMaxH3Transformer3DModel` (`transformer_blocks.N.
// attn.to_q`), which is also what community derivatives start from --
// lightx2v's FP8 turbo DiT is one, as a single file with no config.
//
// The two are the SAME numbers, VERIFIED tensor for tensor against the
// Comfy-Org Ref2VA file, under three changes of spelling:
//
//   renames   transformer_blocks.N -> blocks.N, refiner_blocks -> blocks,
//             norm_q/k -> q_norm/k_norm, to_out.0 -> out_proj,
//             ff.net.2 -> mlp.fc2, proj_in/out -> video_patch_proj /
//             final_layer.video_out (audio likewise), time_embedder.
//             linear_1/2 -> proj_in/out, context_embedder -> condition_
//             proj, norm_out.{linear,norm} -> final_layer.{adaln_proj.
//             linear,norm}
//   qkv       SPLIT: to_q, to_k, to_v are the rows of qkv_proj in that
//             order -- the FLAT grouping, not per head.
//   fc1       ff.net.0.proj is mlp.fc1 with its two halves SWAPPED:
//             diffusers stores up | gate, this model gate | up.
//
// The renames are applied at open (translate_diffusers), so every reader
// of the checkpoint -- sizing, streaming floors, the quantizer -- sees
// the model's names. The other two cannot be a rename, since a name is
// one tensor: the split projection becomes `<qkv_proj>.q/.k/.v` and the
// swapped one `<fc1>.ug`, and the loader assembles each from its parts
// (assembled_parts). `rope.inv_freq` has no diffusers tensor; the loader
// computes it from the config.

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
class MetalLlamaWeights;
namespace minimax_h3 {

// Whether `w` is an H3 DiT in the diffusers naming.
bool is_diffusers_layout(const MetalLlamaWeights& w);

// Rename in place. Returns the number of tensors renamed; 0 (nothing
// touched) for a checkpoint that is not this layout, or one with a
// tensor this cannot place -- which then keeps its names and fails to
// load as what it is.
std::size_t translate_diffusers(MetalLlamaWeights& w);

// The diffusers name that stops translate_diffusers, "" when none does.
std::string untranslatable_tensor(const MetalLlamaWeights& w);

// A single .safetensors that opens as an H3 DiT in this layout -- what
// lightx2v publishes, with no config and no metadata beside it.
bool is_diffusers_dit_file(const std::string& path);

// ---- a DiT ComfyUI SAVED ------------------------------------------------
//
// What ComfyUI writes when it saves a model -- a community fine-tune, a
// merge, an int8 or FP8 requantization of one -- is its own model state:
// H3's names under `model.diffusion_model.` (or `diffusion_model.`), no
// config, no metadata, and the fused qkv in ComfyUI's FLAT order (the
// Comfy-Org repack's, which ComfyUI's H3 splits by; VERIFIED on a
// community int8 file against the released per-head rows). The prefix
// is dropped at open (translate_comfy_prefix, in every open's
// translations); is_comfy_saved_dit_file says a file is one, which is
// what decides its grouping, since nothing after the rename can.

// Drop the prefix in place, when EVERY tensor carries it and the names
// under it are an H3 DiT's. Returns the number renamed; 0 otherwise.
std::size_t translate_comfy_prefix(MetalLlamaWeights& w);

// Whether `path` is a .safetensors H3 DiT under ComfyUI's prefix.
bool is_comfy_saved_dit_file(const std::string& path);

// One piece of a weight the checkpoint stores in pieces: rows
// [row0, row0 + rows) of the layer `layer` (a name WITHOUT ".weight").
struct Part {
  std::string  layer;
  std::int64_t row0 = 0;
  std::int64_t rows = 0;
};

// How the linear `nm` (e.g. "blocks.3.attn.qkv_proj") is assembled, in
// output-row order. Empty when the checkpoint holds `<nm>.weight` as one
// tensor, or does not hold the pieces either.
std::vector<Part> assembled_parts(const MetalLlamaWeights& w,
                                  const std::string& nm);

// ---- the VAEs ---------------------------------------------------------
//
// The diffusers pipeline at the repository's ROOT (`vae/`, `audio_vae/`,
// beside the `FL2VA/` and `Ref2VA/` partitions) carries both VAEs too,
// and VERIFIED byte for byte against `FL2VA/video_vae/source` and
// `FL2VA/audio_vae` they are the same numbers:
//
//   video   every tensor renamed (decoder.proj_in -> decoder.x_embedder,
//           attn.to_out.0 -> attn.to_out, ff.net.2 -> ff.w2,
//           encoder.down_blocks.N.resnets.M -> encoder.down.N.block.M,
//           conv_shortcut -> nin_shortcut, downsamplers.0 -> downsample)
//           except two that are stored in pieces: to_q / to_k / to_v
//           are attn.to_qkv split PER HEAD -- the fused tensor is
//           [h0(q,k,v) | h1(q,k,v) | ...], each piece one head of 64
//           rows -- and ff.net.0.proj is ff.w1 with its halves swapped
//           (up | gate on disk, gate | up in the model). Weights and
//           biases alike. The native file's decoder.mask_token has no
//           counterpart, and nothing here reads it.
//   audio   the same tensors under the same names; only the config and
//           the file name differ.
//
// The video renames are applied at open (translate_diffusers_vae), and
// the two split tensors become `<to_qkv>.q/.k/.v` and `<w1>.ug`, which
// the VAE's loader assembles from vae_assembled_parts.

// The diffusers class names, as `_class_name` in each config.json.
inline constexpr const char* kDiffusersVideoVaeClass =
    "AutoencoderKLMiniMaxH3";
inline constexpr const char* kDiffusersAudioVaeClass =
    "AutoencoderKLMiniMaxH3Audio";
inline constexpr const char* kDiffusersDitClass =
    "MiniMaxH3Transformer3DModel";

// ---- the diffusers PIPELINE at the repository root ----------------------
//
// The root's model_index.json (`MiniMaxH3ModularPipeline`) is one pipeline
// over BOTH task partitions: `transformer/` is FL2VA's DiT and
// `transformer_ref/` Ref2VA's -- VERIFIED by their bytes against the
// released partitions -- beside one text encoder, tokenizer, processor
// and pair of VAEs (byte-identical to each partition's). So on such a
// root `transformer/` is a PARTITION's DiT, not the model's, and the
// task decides which directory answers.
inline constexpr const char* kDiffusersPipelineClass =
    "MiniMaxH3ModularPipeline";

// Whether `dir` is that root: its model_index.json names the pipeline,
// or it carries `transformer_ref/` (a partial checkout without the
// manifest).
bool is_diffusers_pipeline_root(const std::string& dir);

// The DiT directory a pipeline root keeps for `partition` ("fl2va" ->
// transformer/, "ref2va" -> transformer_ref/, empty -> the historical
// fl2va), or "" when that directory is not present.
std::string diffusers_pipeline_dit(const std::string& root,
                                   const std::string& partition);

// The partition a DiT directory holds BY ITS PLACE in a pipeline root
// (transformer/ -> "fl2va", transformer_ref/ -> "ref2va"), "" when it is
// not one of those.
std::string diffusers_pipeline_partition(const std::string& dit_dir);

// Whether `dir` is a directory whose config.json names `cls`. A file, a
// missing config or another class answer false.
bool is_diffusers_vae_dir(const std::string& dir, const char* cls);

// Whether `w` is the H3 video VAE in the diffusers naming.
bool is_diffusers_vae_layout(const MetalLlamaWeights& w);

// Rename in place, as translate_diffusers does for the DiT: the whole
// checkpoint or nothing. Returns the number renamed.
std::size_t translate_diffusers_vae(MetalLlamaWeights& w);

// How the video VAE's linear `nm` (e.g. "decoder.transformer_blocks.3.
// attn.to_qkv") is assembled, in output-row order, with `head_dim`-row
// heads. Empty when `<nm>.weight` is one tensor or the pieces are not
// there either.
std::vector<Part> vae_assembled_parts(const MetalLlamaWeights& w,
                                      const std::string& nm, int head_dim);

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
