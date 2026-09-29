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

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
