#ifndef GENERATIVE_MODELS_SHARED_COMFY_CHECKPOINT_H
#define GENERATIVE_MODELS_SHARED_COMFY_CHECKPOINT_H

#include <string>
#include <vector>

namespace vpipe {

class FlexData;

namespace genai {
namespace comfy {

// Reading the Comfy-Org repackaging of a diffusers model.
//
// Comfy-Org republishes upstream checkpoints (Comfy-Org/MiniMax-H3,
// Comfy-Org/Wan-Animate-2, ...) as ONE safetensors file per component
// under a fixed set of repo subdirectories:
//
//   diffusion_models/<model>_<precision>.safetensors
//   text_encoders/<enc>_<precision>.safetensors
//   vae/<vae>_<precision>.safetensors
//   clip_vision/, loras/
//
// and marks the repo with the `comfyui` HuggingFace tag. Two things
// change against the upstream diffusers layout, and both have to be
// handled or the file is unloadable:
//
//   * there is no config.json. Each file carries its component's config
//     INSIDE the safetensors `__metadata__` block, as a JSON *string*
//     under a per-model key. That makes the file self-describing, which
//     is why detection here reads the file rather than trusting a
//     directory name.
//   * tensors may be REPACKED. Comfy-Org's MiniMax-H3 conversion
//     reorders the DiT's fused `attn.qkv_proj` from the released
//     per-head grouping to a flat [all q | all k | all v] -- same names,
//     same shapes, different bytes. Nothing in the file distinguishes
//     the two, so the layout has to be decided by WHERE the checkpoint
//     came from. That is what these helpers are for: a component that
//     resolves through here is the Comfy-Org repack, and the model's
//     config records it.
//
// The repacking is per COMPONENT, not per repo -- MiniMax-H3's video VAE
// keeps the released per-head qkv while its DiT does not -- so never
// infer one component's layout from another's.
//
// Precisions this build reads are bf16 / fp16 / fp32. The int8_convrot,
// fp8_scaled and nvfp4_awq variants use Comfy-Org's own packing and are
// deliberately not supported (nor fetched); `unsupported_quant()` names
// them so a resolve skips them rather than loading garbage.

// The whole `__metadata__` object of a single-file safetensors
// checkpoint. Values are strings (safetensors metadata is string ->
// string); use metadata_json() to get one parsed. False when `file` is
// not a readable safetensors file or carries no `__metadata__`.
bool read_metadata(const std::string& file, FlexData& out,
                   std::string* err = nullptr);

// One `__metadata__` entry, parsed as JSON. False when the file has no
// metadata, no such key, or the value is not JSON.
bool metadata_json(const std::string& file, const std::string& key,
                   FlexData& out, std::string* err = nullptr);

// True when `path` is a regular .safetensors file whose `__metadata__`
// carries `key` -- the test a `config_from_json` uses to decide it is
// looking at a Comfy-Org component rather than a diffusers directory.
bool is_component(const std::string& path, const std::string& key);

// True when a filename names a packing this build cannot read
// (int8_convrot / fp8_scaled / nvfp4 / gguf). Matched case-insensitively
// on the whole filename.
bool unsupported_quant(const std::string& file);

// Resolve a Comfy-Org single-file component. `path` may be
//   * the .safetensors file itself,
//   * the component subdirectory (`<repo>/diffusion_models`), or
//   * the repository root (`<repo>`),
// and `subdir` is the repo subdirectory to look in ("diffusion_models",
// "vae", "text_encoders"). A candidate must be a .safetensors carrying
// `meta_key` in its `__metadata__`, must not be an unsupported_quant(),
// and -- when `prefer` is non-empty -- a filename containing one of
// those substrings wins, earliest entry first. Ties break on the
// shortest name, so `minimax_h3_fl2va_bf16` beats
// `minimax_h3_fl2va_pruned_bf16` without either being named here.
//
// Returns the file path, or "" when nothing in `path` matches -- which
// is the signal to fall back to the diffusers layout.
std::string resolve_component(const std::string& path,
                              const std::string& subdir,
                              const std::string& meta_key,
                              const std::vector<std::string>& prefer = {});

// One readable single-file component found in a Comfy-Org repo.
struct Component {
  std::string role;      // the repo subdir: "diffusion_models", "vae", ...
  std::string file;      // absolute path to the .safetensors
  std::string meta_key;  // its `__metadata__` key, which names the model
                         // ("config" for a DiT, "minimax_h3_video_vae", ...)
};

// Every readable component under a Comfy-Org repo root, in a stable
// order (by role, then filename). Empty when `root` is not a directory
// or holds none of the known subdirectories -- which is also the test
// for "is this a Comfy-Org repo at all".
//
// This is the WHOLE-REPO view the per-component resolvers above cannot
// give: those answer "where is the DiT?", this answers "what is in
// here?". Detection needs it to describe a directory, and a
// component-selecting caller (model-quantize's `target`) needs it to
// map a role onto a file without knowing the naming of every model.
//
// Unreadable packings (unsupported_quant) are skipped, so the result is
// what this build can actually open -- not an inventory of the repo.
std::vector<Component> scan_repo(const std::string& root);

}  // namespace comfy
}  // namespace genai
}  // namespace vpipe

#endif
