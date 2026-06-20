#ifndef STAGES_MODEL_REGISTRY_H
#define STAGES_MODEL_REGISTRY_H

#include <string>

namespace vpipe {

class SessionContextIntf;

// Resolve a model reference to a directory. If `ref` is a key in the
// `models_db` LMDB sub-db (the registry model-fetch writes, keyed by the
// huggingface.co path), return that record's local_path -- a DB entry
// WINS over a same-named filesystem path. Any missing env / key miss /
// DB error returns `ref` unchanged, so a plain filesystem path keeps
// working. On a hit, an info line is logged through `session`. A null
// `session` or empty `models_db` returns `ref`.
//
// Shared by the LM stages (audio-transcribe, text-chat, visual-qa,
// realtime-vqa) so a configured hf_dir may name a registered model.
std::string
resolve_model_dir(const SessionContextIntf* session,
                  const std::string&        models_db,
                  const std::string&        ref);

}

#endif
