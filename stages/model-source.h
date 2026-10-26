#ifndef STAGES_MODEL_SOURCE_H
#define STAGES_MODEL_SOURCE_H

#include "common/flex-data.h"

#include <cstddef>
#include <string>
#include <vector>

#include "common/vpipe-api.h"

// Where a model's files are FETCHED from: the part of it a plugin and a
// host see. Installed with the SDK since feature "model-sources/1".
//
// The catalogue is written in HuggingFace terms -- `hf_path` is both the
// download coordinate and the registry key -- because that is where these
// models are published. It is not always where they can be REACHED:
// huggingface.co is not routable from mainland China, and much of the
// catalogue is mirrored on modelscope.cn instead. `model-fetch` reads from
// either (its `source`, or $VPIPE_MODEL_SOURCE).
//
// WHAT A SOURCE DOES NOT CHANGE IS IDENTITY. A model fetched from a
// mirror registers under the SAME key and lands in the SAME directory as
// one fetched from HuggingFace, so nothing downstream -- model-select, a
// stage's `model_path`, a registry record, a `.vpipeline` recipe -- can
// tell which was used, and a user who switches sources does not acquire a
// second copy of a model they already have.
//
// What is here: the sources' names and labels (a host offering its user a
// choice), whether a source carries a repo and under what name, and the
// rows a plugin adds for its own families' repos. How a source is asked
// for a listing and a file -- the ModelSource interface -- is the host's
// own (stages/model-source-host.h, not installed), so it can grow without
// touching the plugin ABI.
//
// OPTIONAL TO A PLUGIN. Outside libvpipe these functions are WEAK
// imports: on a host built before them (no model-sources/1) they resolve
// to null and model_sources_available() says false, so a plugin that adds
// mirror rows when it can still loads there. One that cannot run without
// them lists model-sources/1 in VpipePluginInfo::required_features, and
// such a host refuses it by name -- which only works BECAUSE the imports
// are weak: a strong one is a missing symbol, and dyld refuses the plugin
// at dlopen, before its required_features are ever read.

#if defined(VPIPE_BUILDING_LIBRARY)
#define VPIPE_MODEL_SOURCES_IMPORT
#else
#define VPIPE_MODEL_SOURCES_IMPORT __attribute__((weak_import))
#endif

VPIPE_API_BEGIN

namespace vpipe {

// Every source name ("huggingface", "modelscope"), in registration order:
// what `model-fetch`'s `source` attribute takes.
VPIPE_MODEL_SOURCES_IMPORT std::vector<std::string> model_source_names();

// How source `name` is spelled to a human ("ModelScope"); "" when the
// name is not one.
VPIPE_MODEL_SOURCES_IMPORT std::string
model_source_label(const std::string& name);

// ---- mirror paths ------------------------------------------------------

// Whether a source carries a repo, and under what name. A later host may
// add a value; a caller treats one it does not know as Same.
enum class MirrorStatus {
  Same,      // mirrored under the same owner/repo as HuggingFace
  Renamed,   // mirrored, but under a different path (see `out`)
  Absent,    // the source is KNOWN not to carry it
};

// Resolve `hf_path` to its path on `source_name`, writing the answer to
// `out`.
//
// The default is Same, and that is deliberate: ModelScope mirrors most of
// this catalogue under the identical owner/repo, so requiring every entry
// to declare a mapping would be a table that says "unchanged" a hundred
// times and goes stale the moment a model is added. What the table
// records is only the DIFFERENCES -- the renames, and the repos a source
// is known to lack -- so a repo that is simply not mirrored fails at the
// listing step with that source's own 404, which is an honest answer.
//
// Absent exists so the failure can be better than a 404 where we already
// know the answer: three of the Krea-2 LoRAs are individual-author
// uploads that have no ModelScope counterpart, and telling a user that
// outright beats letting them read an HTTP code. A host offering a choice
// of source greys the choice for such a repo.
VPIPE_MODEL_SOURCES_IMPORT MirrorStatus
mirror_repo(const std::string& source_name, const std::string& hf_path,
            std::string& out);

// One mirror-table row. `path` empty marks the repo ABSENT on `source`.
struct MirrorEntry {
  std::string source;    // "modelscope"
  std::string hf_path;   // "MiniMaxAI/MiniMax-H3"
  std::string path;      // "MiniMax/MiniMax-H3"; "" -> absent there
  // A later field is a key here (lower_snake, a named constant; absent =
  // today's behaviour), so the row's layout never changes. None yet.
  FlexData extra;
};

// Contribute mirror rows from a PLUGIN, for a family whose catalogue
// entries do not live in this tree.
//
// FIRST-WINS per (source, hf_path), built-ins included, and the same
// lifetime rule as register_catalog_entries: each call publishes a new
// snapshot and the previous ones are retained, so a pointer or reference
// handed out earlier never dangles. Rows registered while a plugin LOADS
// -- a static initialiser or vpipe_plugin_register alike -- are held back
// until the host admits the plugin and dropped if it refuses it, as its
// stage types are, so a refused plugin cannot claim a repo ahead of one
// that loads. Returns how many rows were taken (held, during a load).
VPIPE_MODEL_SOURCES_IMPORT std::size_t
register_mirror_repos(std::vector<MirrorEntry> entries);

// Whether this host has the functions above: always inside libvpipe; in a
// plugin, whether the weak imports resolved. Ask before calling them.
inline bool
model_sources_available() noexcept
{
#if defined(VPIPE_BUILDING_LIBRARY)
  return true;
#else
  return &register_mirror_repos != nullptr;
#endif
}

}  // namespace vpipe

VPIPE_API_END

#endif
