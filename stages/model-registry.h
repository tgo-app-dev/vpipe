#ifndef STAGES_MODEL_REGISTRY_H
#define STAGES_MODEL_REGISTRY_H

#include <string>
#include <string_view>

namespace vpipe {

class SessionContextIntf;
class FlexData;

// The one LMDB sub-db the model registry lives in: model-fetch writes it,
// every model-consuming stage reads it. Fixed system-wide rather than a
// per-stage config key, so a fetcher and a consumer can never disagree
// about where a model reference resolves. The leading underscores keep it
// out of the way of the user-named sub-dbs (cameras, videos, ...).
inline constexpr std::string_view kModelRegistryDb = "__vpipe_model_registry";

// Resolve a model reference to a directory. If `ref` is a key in the
// model registry (kModelRegistryDb, the sub-db model-fetch writes, keyed
// by the huggingface.co path), return that record's local_path -- a DB
// entry WINS over a same-named filesystem path. Any missing env / key
// miss / DB error returns `ref` unchanged, so a plain filesystem path
// keeps working. On a hit, an info line is logged through `session`. A
// null `session` returns `ref`.
//
// Shared by the LM stages (audio-transcribe, text-chat, visual-qa,
// realtime-vqa) so a configured hf_dir may name a registered model.
std::string
resolve_model_dir(const SessionContextIntf* session,
                  const std::string&        ref);

// True when `ref` resolves (via resolve_model_dir) to a path that exists
// on disk -- i.e. the model is actually present, not just a registry key
// or a not-yet-downloaded path. Cheap: one LMDB read + a stat. The
// preparation stages (model-quantize / -benchmark / -eval) call this
// AFTER their trigger fires, not at config time, because a cascaded
// model-fetch may not have downloaded the model when the pipeline is
// built. Empty `ref` -> false.
bool
model_dir_available(const SessionContextIntf* session,
                    const std::string&        ref);

// Apply a `model-select` beat to a stage's hf_dir. The beat is the
// FlexData a `model-select` source emits so the diffusion-conditioner /
// text-to-image / vae-encode / vae-decode stages can share ONE model
// choice through their `model` iport (which OVERRIDES the hf_dir config
// key). The beat is either a plain string (the model dir/registry key) or
// an object with "hf_dir" (alias "model"). `hf_dir` is overwritten when
// the beat supplies a non-empty reference. Returns true iff a usable
// model reference was found.
bool
apply_model_select_beat(const FlexData& beat,
                        std::string&    hf_dir);

}

#endif
