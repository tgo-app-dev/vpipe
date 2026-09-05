#ifndef STAGES_MODEL_REGISTRY_H
#define STAGES_MODEL_REGISTRY_H

#include <string>
#include <string_view>
#include <vector>

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
// A model reference resolved through the registry.
//
// The directory alone is not always enough to say WHICH model was meant.
// Two records can share one `local_path` on purpose -- MiniMax-H3's two
// task partitions are published from one repo and differ only in which
// DiT file they pin, so `model-fetch --model_key` registers them
// separately over one download. A consumer handed only the path then
// re-probes the tree and cannot tell them apart; what disambiguates is
// the record, so this carries it.
//
// `model_type` is the record's own field (the catalogue's, when the
// fetch was catalogued), and is EMPTY for a plain filesystem path or an
// unrecognised repo -- an empty answer means "nothing said", never a
// guess.
struct ResolvedModel {
  std::string dir;          // local_path, or `ref` when not a DB key
  std::string key;          // the registry key hit, else empty
  std::string model_type;   // the record's model_type, else empty
  // Repo-relative paths the record PINNED, in catalogue order; empty for
  // a plain path or a whole-repo fetch. This is what says which of two
  // records sharing a directory is meant -- the case a directory scan
  // cannot answer, and answers WRONGLY rather than loudly.
  std::vector<std::string> files;
  bool from_registry = false;
};

// Resolve `ref` to a directory AND whatever the registry says about it.
// Same lookup and same fallbacks as resolve_model_dir; a miss returns
// `{ref, "", "", false}`.
ResolvedModel
resolve_model(const SessionContextIntf* session, const std::string& ref);

std::string
resolve_model_dir(const SessionContextIntf* session,
                  const std::string&        ref);

// The directory the record's pinned files actually live in.
//
// `dir` is where the repo was downloaded, and a record may pin a SUBTREE
// of it rather than the whole thing. VDN-H3's two stages are published
// from one repo as `stage-b-step-2000/...` and `stage-dmd-step-250/...`,
// so both keys resolve to the same root and only the pinned paths say
// which subtree was meant -- a consumer handed the root finds neither
// stage's config and reports the repo as malformed.
//
// Returns `dir` joined with the common DIRECTORY prefix of the pinned
// paths, and `dir` unchanged when they share none: a whole-repo fetch, a
// plain filesystem path, and every record whose files sit at the top all
// answer the same as before.
//
// OPT-IN, not folded into resolve_model_dir(), because "the directory
// the files are in" is not always "the directory the consumer wants" --
// a record pinning only `transformer/*.safetensors` of a multi-component
// repo pins a subtree whose consumer still wants the root next to it.
std::string
resolved_subtree_dir(const ResolvedModel& rm);

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

// Resolve a reference to ONE `.safetensors` -- an adapter, which is a
// file where every other model reference here is a directory.
//
// Three shapes, in order: a direct path to a file; a registry key, whose
// record NAMES its file (the only thing that disambiguates two records
// over one directory -- both MiniMax-H3 Turbo checkpoints are published
// from one repo and land side by side); and a bare directory, scanned
// for a single .safetensors. The scan is last because it is the only one
// that can be ambiguous, and it REFUSES rather than picking when it is:
// a directory-iteration order is not a choice a user made.
//
// Empty + *err on any failure. `what` names the caller in the message.
std::string
resolve_adapter_file(const SessionContextIntf* session,
                     const std::string& ref, std::string* err);

// Apply a `model-select` beat to a stage's hf_dir. The beat is the
// FlexData a `model-select` source emits so the diffusion-conditioner /
// generate-image / vae-encode / vae-decode stages can share ONE model
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
