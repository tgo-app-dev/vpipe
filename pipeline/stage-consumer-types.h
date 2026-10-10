#ifndef PIPELINE_STAGE_CONSUMER_TYPES_H
#define PIPELINE_STAGE_CONSUMER_TYPES_H

// Model types a REGISTERED FAMILY adds to one stage's own model picker.
//
// register_channel_types() (stage-config.h) puts a plugin family on the
// shared-model CHANNEL, so a channel SOURCE (model-select) offers it. A
// channel CONSUMER keeps the static suggest_db_type it was compiled
// with, so generate-image's own `hf_dir` picker never offered a plugin
// image family it runs -- and vae-decode's carried one plugin's tag
// written in by hand. The channel cannot fix that, because it does not
// know which stage runs a family; the registry that accepted the family
// does. Host-internal: the registries and Stage::config_params() are the
// only callers, so nothing here is plugin ABI.

#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

// Idempotent per (stage, type); first-appearance order is kept.
void register_consumer_types(std::string_view stage_type,
                             std::string_view csv_model_types);

// What registered families added for `stage_type`, in order, each
// followed by its aliases (below).
std::vector<std::string> consumer_types(std::string_view stage_type);

// A family TAG also stands for these catalogue model types. A family is
// registered under one tag, while its checkpoints are catalogued under
// one type per instantiation (a t2i and an edit repo of one
// architecture), and a picker filters on the type. The detect profile
// that names those types is what registers them here; every list above,
// and the shared-model channel's, expands a tag by its aliases.
void register_type_aliases(std::string_view tag,
                           std::string_view csv_model_types);

}  // namespace vpipe

#endif
