#ifndef STAGES_MODEL_SOURCE_HOST_H
#define STAGES_MODEL_SOURCE_HOST_H

// The host's side of model sources: how each one is ASKED for a listing
// and a file (model-fetch reads these), and the parsers behind them. NOT
// installed -- a plugin sees stages/model-source.h, the names, labels and
// mirror table -- so this interface can grow without touching the plugin
// ABI.

#include "stages/model-catalog.h"
#include "stages/model-source.h"

#include <string>
#include <vector>

namespace vpipe {

class FlexData;

// A source is the small amount that actually differs between HuggingFace
// and a mirror: how a repo's file list is asked for, how that answer is
// parsed, and how one file's bytes are addressed (identity does not
// change; see model-source.h). The registry record notes which source the
// bytes came from; that is provenance, not identity.
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

// Extract the downloadable files from a ModelScope
// "/api/v1/models/<repo>/repo/files" answer -- the counterpart to
// hf_tree_files, and the parse behind ModelScope's parse_tree(). Pure;
// exposed so it can be tested against a captured page without a network.
// Directories (Type "tree") and malformed entries are skipped.
std::vector<HfFile> modelscope_tree_files(const FlexData& body);

// The plugin manager's load quarantine for mirror rows (as the stage
// registry's): between begin and end, register_mirror_repos holds what it
// takes; end_mirror_quarantine(true) publishes it, (false) drops it.
void begin_mirror_quarantine();
void end_mirror_quarantine(bool admit);

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

}  // namespace vpipe

#endif
