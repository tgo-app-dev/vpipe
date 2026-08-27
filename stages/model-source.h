#ifndef STAGES_MODEL_SOURCE_H
#define STAGES_MODEL_SOURCE_H

#include "stages/model-catalog.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vpipe {

class FlexData;

// Where a model's files are FETCHED from.
//
// The catalogue is written in HuggingFace terms -- `hf_path` is both the
// download coordinate and the registry key -- because that is where these
// models are published. It is not always where they can be REACHED:
// huggingface.co is not routable from mainland China, and much of the
// catalogue is mirrored on modelscope.cn instead. A source is the small
// amount that actually differs between the two: how a repo's file list is
// asked for, how that answer is parsed, and how one file's bytes are
// addressed.
//
// WHAT A SOURCE DOES NOT CHANGE IS IDENTITY. A model fetched from a
// mirror registers under the SAME key and lands in the SAME directory as
// one fetched from HuggingFace, so nothing downstream -- model-select, a
// stage's `model_path`, a registry record, a `.vpipeline` recipe -- can
// tell which was used, and a user who switches sources does not acquire a
// second copy of a model they already have. The registry record notes
// which source the bytes came from; that is provenance, not identity.
//
// The interface is small on purpose. Everything hard about a large
// download -- the stall window, resume, the integrity check, the progress
// report -- is source-independent and stays in resumable-fetch.h.
class ModelSource
{
public:
  virtual ~ModelSource() = default;

  // Registry name; what `model-fetch`'s `source` attribute takes.
  virtual const char* name() const = 0;
  // How the source is spelled in a message to a human.
  virtual const char* label() const = 0;

  // The branch a repo is read from when the caller names none.
  //
  // HuggingFace says "main"; ModelScope says "master". Getting this wrong
  // is not a clean failure on ModelScope: an unknown revision comes back
  // HTTP 200 with `Code: 200` and an EMPTY file list, so "main" there
  // reads as a repo that exists and holds nothing. MEASURED against
  // Lightricks/LTX-2.5: Revision=master -> 24 entries, Revision=main and
  // Revision=bogus-xyz -> 0 apiece.
  virtual const char* default_revision() const = 0;

  // Env vars consulted in order for a bearer token, most specific first.
  virtual std::vector<std::string> token_env() const = 0;

  // Where a human reads about the repo; the registry record's source_url.
  virtual std::string web_url(const std::string& repo) const = 0;

  // The file-listing request for `repo` at `rev`, and the parse of what
  // comes back. Split so the transport (and its token, timeout and TLS
  // policy) stays with the caller.
  virtual std::string tree_url(const std::string& repo,
                               const std::string& rev) const = 0;
  virtual std::vector<HfFile> parse_tree(const FlexData& body) const = 0;

  // One file's bytes.
  virtual std::string file_url(const std::string& repo,
                               const std::string& rev,
                               const std::string& path) const = 0;

  // Whether a file can be rebuilt from a content-addressed store
  // (HuggingFace Xet). A mirror that has none streams every file, which
  // is what the empty `XetSource` in a FetchRequest already means -- this
  // is here so the caller does not have to know which sources those are.
  virtual bool supports_xet() const { return false; }

  // What an empty file list most likely means, in this source's terms.
  // Worth a per-source sentence: on HuggingFace an empty listing is a
  // private or misspelled repo, on ModelScope it is far more often a
  // revision that does not exist (see default_revision above).
  virtual std::string empty_tree_hint(const std::string& repo,
                                      const std::string& rev) const = 0;
};

// Look a source up by name ("huggingface", "modelscope"); nullptr when
// the name is not one.
const ModelSource* model_source(const std::string& name);

// The source a fetch uses when its config names none: $VPIPE_MODEL_SOURCE
// when that names a known source, else HuggingFace. This is the knob a
// user behind the Great Firewall sets ONCE, rather than per stage.
//
// An unknown $VPIPE_MODEL_SOURCE falls back to HuggingFace; the fetch
// stage warns about it, because silently ignoring the one setting that
// makes downloads work at all is the wrong failure.
const ModelSource& default_model_source();

// Every registered source name, in registration order.
std::vector<std::string> model_source_names();

// Extract the downloadable files from a ModelScope
// "/api/v1/models/<repo>/repo/files" answer -- the counterpart to
// hf_tree_files, and the parse behind ModelScope's parse_tree(). Pure;
// exposed so it can be tested against a captured page without a network.
// Directories (Type "tree") and malformed entries are skipped.
std::vector<HfFile> modelscope_tree_files(const FlexData& body);

// ---- mirror paths ------------------------------------------------------

// Whether a source carries a repo, and under what name.
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
// times and goes stale the moment a model is added. What the table below
// records is only the DIFFERENCES -- the renames, and the repos a source
// is known to lack -- so a repo that is simply not mirrored fails at the
// listing step with that source's own 404, which is an honest answer.
//
// Absent exists so the failure can be better than a 404 where we already
// know the answer: three of the Krea-2 LoRAs are individual-author
// uploads that have no ModelScope counterpart, and telling a user that
// outright beats letting them read an HTTP code.
MirrorStatus mirror_repo(const std::string& source_name,
                         const std::string& hf_path, std::string& out);

// One mirror-table row. `path` empty marks the repo ABSENT on `source`.
struct MirrorEntry {
  std::string source;    // "modelscope"
  std::string hf_path;   // "MiniMaxAI/MiniMax-H3"
  std::string path;      // "MiniMax/MiniMax-H3"; "" -> absent there
};

// Contribute mirror rows from a PLUGIN, for a family whose catalogue
// entries do not live in this tree.
//
// FIRST-WINS per (source, hf_path), and same lifetime rule as
// register_catalog_entries: each call publishes a new snapshot and the
// previous ones are retained, so a pointer or reference handed out
// earlier never dangles. Returns how many rows were taken.
std::size_t register_mirror_repos(std::vector<MirrorEntry> entries);

// ---- user-typed references --------------------------------------------

// Normalise a user-typed model reference to a bare "owner/repo", for
// EITHER source's URL spelling:
//   "https://huggingface.co/owner/repo"          -> "owner/repo"
//   "https://modelscope.cn/models/owner/repo"    -> "owner/repo"
//   "modelscope.cn/models/owner/repo/files"      -> "owner/repo"
//   "owner/repo"                                 -> "owner/repo"
// Returns "" when the input carries no owner/repo pair.
//
// ModelScope's web URL puts a "models/" segment ahead of the pair where
// HuggingFace has none, so the two cannot share one blind "first two
// segments" rule -- pasting a ModelScope URL into the HuggingFace-shaped
// parser yields "models/<owner>", a repo that does not exist.
std::string normalize_model_ref(const std::string& input);

}

#endif
