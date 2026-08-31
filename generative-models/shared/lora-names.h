#ifndef VPIPE_GENERATIVE_MODELS_SHARED_LORA_NAMES_H
#define VPIPE_GENERATIVE_MODELS_SHARED_LORA_NAMES_H

#include <string>

namespace vpipe {
namespace genai {
namespace lora {

// A module has ONE name in the model and several in the wild.
//
// Diffusers-convention adapters name the model's own modules, so
// matching them is a lookup. The ai-toolkit / ComfyUI convention is a
// different thing: it renames the CONTAINERS and the PROJECTIONS
// (`blocks` for `transformer_blocks`, `mlp` for `ff`, `wq` for `to_q`),
// so such a file names nothing the model has and a plain lookup finds
// zero modules -- which is indistinguishable, at runtime, from an
// adapter that had no effect.
//
// This header is the ONE map between the two. It lives here because two
// unrelated callers need it and a second copy would be a second thing
// to keep right: `fuse_lora` (which writes the delta into the weights)
// and the runtime binder (which keeps it as a side GEMM) must agree
// about which base weight a module means, or the same adapter would
// fuse one set of projections and bind another.

// Map an ai-toolkit / ComfyUI adapter module name to the diffusers
// base-weight name.
//
// The submodule renames are KREA-2's topology (`to_gate` exists in no
// other family here), and that is not a problem for a generic caller:
// a remapped name a base does not carry simply misses, which is the
// same answer as not remapping at all. What it must never do is map a
// name onto a DIFFERENT real module, and it cannot -- every rewrite is
// gated on the `diffusion_model.` prefix, so a diffusers-spelled name
// is returned as "" rather than rewritten.
//
// Returns "" when `m` is not in that convention.
std::string remap_ai_toolkit_module(std::string m);

}  // namespace lora
}  // namespace genai
}  // namespace vpipe

#endif
