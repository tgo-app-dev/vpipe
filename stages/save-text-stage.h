#ifndef VPIPE_STAGES_SAVE_TEXT_STAGE_H
#define VPIPE_STAGES_SAVE_TEXT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace vpipe {

// Sink stage: saves text extracted from incoming FlexData beats to a text file.
//
// On each beat it pulls a string field (config `key`, default "text"; or the
// whole payload when it is itself a plain string) and appends it to the file
// under the newline policy. Empty / missing extractions are skipped (no blank
// lines). The file is flushed after every write so it is readable live.
//
//   iport0  FlexDataPayload. The `key` field (default "text") is written; a
//           plain-string payload is written whole. A missing / non-string
//           field => the beat is skipped.
//
//   no oports (sink).
//
// Config (FlexData object):
//   path     (string, required)       -- output text file path.
//   key      (string, default "text") -- FlexData field to write. A plain
//                                        string payload is always written
//                                        whole regardless of `key`.
//   newline  (string, default "after")-- entry separator policy:
//                                        "after"  each entry followed by "\n"
//                                                 (one entry per line),
//                                        "before" each entry preceded by "\n"
//                                                 (except the first write),
//                                        "none"   entries concatenated as-is.
//   append   (bool,   default true)   -- append to an existing file; false
//                                        truncates it at initialize().
class SaveTextStage final : public TypedStage<SaveTextStage> {
public:
  static constexpr const char* kTypeName = "save-text";

  SaveTextStage(const SessionContextIntf* session,
                 std::string               id,
                 std::vector<InEdge>       iports,
                 FlexData                  config);

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& path() const noexcept { return _path; }
  const std::string& key()  const noexcept { return _key; }
  std::uint64_t entries_written() const noexcept { return _entries_written; }

private:
  enum class Newline { After, Before, None };

  std::string _path;
  std::string _key;
  Newline     _newline = Newline::After;
  bool        _append  = true;

  std::ofstream _out;
  bool          _wrote_any     = false;
  std::uint64_t _entries_written = 0;
};

}

#endif
