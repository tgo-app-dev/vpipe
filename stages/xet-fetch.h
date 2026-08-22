#ifndef STAGES_XET_FETCH_H
#define STAGES_XET_FETCH_H

#include "interfaces/session-context-intf.h"
#include "interfaces/ui-delegate-intf.h"
#include "stages/resumable-fetch.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace vpipe {

// Rebuilding a file from HuggingFace's Xet content-addressed store,
// instead of streaming it from the plain resolve URL.
//
// Why bother: a plain download is ONE connection, and a connection can
// be the throughput ceiling well before the link is. A Xet file is
// described by a reconstruction manifest, a list of byte ranges over
// content-addressed objects ("xorbs"), which can be fetched MANY AT A
// TIME. The store also holds those ranges compressed and deduplicated,
// so there are fewer bytes to move.
//
// MEASURED, three interleaved pairs on a 5.2 GB bf16 shard
// (MiniMaxAI/MiniMax-H3), whole-file wall clock through fetch_file:
//
//   plain single stream   89.5, 88.3, 82.7 MB/s   median 88.3
//   store, 8 streams      98.7, 92.8, 94.4 MB/s   median 94.4  = 1.07x
//
// So on a link where ONE stream already reaches ~88 MB/s there is very
// little headroom to win back, and what shows up is mostly the smaller
// download: the store serves that shard in 0.873x its size, because
// bf16 weights compress once their byte planes are separated. A 4-bit
// checkpoint does not compress at all and comes to 0.959x, dedup only.
//
// The win is bigger where a single stream is the constraint rather than
// the link. Fetching the same ranges in isolation, where one stream was
// getting ~52 MB/s, 16 of them reached 104 MB/s -- twice as much. Which
// of those two worlds a given machine is in is not something this code
// can tell, so the parallelism is a setting (`xet_streams`), not a
// promise.

// Append the whole file to `part`, in order, resuming whatever is
// already there.
//
// Appending in order is what makes it resumable at all: `part` is
// always a prefix of the finished file, so a killed fetch picks up at
// the first reconstruction term the file does not yet reach -- and the
// plain path can continue the same part if Xet is unavailable next
// time. `total` is the finished size, so the caller's progress and the
// resume arithmetic agree with what the manifest says.
//
// False + `err` when the file cannot be reconstructed -- no manifest,
// an expired grant that would not renew, a range that would not come
// down. The caller falls back to the plain download; anything already
// appended stays, because it is still a valid prefix.
bool xet_fetch(const SessionContextIntf* s, const XetSource& src,
               const std::string& hf_token, const FetchOpts& o,
               const std::filesystem::path& part, std::uint64_t total,
               std::string& err, UiProgress* progress,
               const std::function<bool()>* cancel = nullptr);

// ---- exposed for tests -------------------------------------------------

// Decode one CAS chunk: an 8-byte header -- version, a 24-bit
// compressed length, the compression scheme, a 24-bit uncompressed
// length -- followed by the payload. Schemes: 0 stores the bytes as
// they are, 1 wraps them in an LZ4 frame, 2 wraps an LZ4 frame around
// the same bytes with their four byte-planes separated first (BG4),
// which is what makes bf16 weights compress at all.
//
// Returns the number of INPUT bytes consumed, 0 on a malformed chunk.
// `out` gets the decoded bytes appended.
std::size_t xet_decode_chunk(const unsigned char* p, std::size_t n,
                             std::string& out, std::string& err);

// Undo the BG4 split in place over `n` bytes: the four planes are
// stored back to back, longest first, and interleave back to the
// original stream.
void xet_bg4_regroup(char* buf, std::size_t n, std::string& scratch);

}

#endif
