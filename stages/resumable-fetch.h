#ifndef STAGES_RESUMABLE_FETCH_H
#define STAGES_RESUMABLE_FETCH_H

#include "interfaces/session-context-intf.h"
#include "interfaces/ui-delegate-intf.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace vpipe {

// HTTP downloads that survive a bad link.
//
// Written for the model fetch, where one file can be twenty gigabytes.
// At that size the two things a plain GET does wrong both bite: a total
// deadline cannot tell a slow link from a dead one and cuts off healthy
// transfers, and starting over is an expensive answer to a socket that
// dropped after 19 of the 20 GB. So a transfer here is bounded by how
// long it has made NO progress, retries continue from the bytes already
// on disk, and the result is checked against the checksum the source
// published before it counts as downloaded.

// "3.4 GB" and friends. Powers of 1024, one decimal.
std::string human_bytes(std::uint64_t n);

// Files at least this large are worth a live progress report.
inline constexpr std::uint64_t kBigFileBytes = std::uint64_t{256} << 20;

// One-shot global libcurl init (idempotent). Safe from any thread.
void ensure_curl_global_init();

// GET into a string, under a hard deadline -- which is right for the
// API pages this serves, kilobytes apiece, where one still open after
// the timeout is stuck. `err` set + false on transport / HTTP error;
// `*status` carries the HTTP code, so a caller can spot a 401/403 and
// go find a token.
bool http_get_text(const std::string& url, const std::string& token,
                   bool verify_tls, long timeout_s, std::string& out,
                   long& status, std::string& err);

// GET a byte range into memory, under the same stall window a file
// transfer gets. This is the unit of work on the Xet path, where a file
// arrives as many ranges of content-addressed objects rather than as
// one stream. `off`/`last` are inclusive, as HTTP counts them.
//
// `on_bytes` (optional) is called as the range arrives with the count
// SO FAR in this attempt -- back to a small number if the transfer is
// retried, since the range starts over. Without it a range is invisible
// until it is whole, which is what leaves a parallel fetch's progress
// report standing still while eight of them are in flight.
bool http_get_range(const std::string& url, const std::string& token,
                    bool verify_tls, long stall_s, std::uint64_t off,
                    std::uint64_t last, std::string& out, long& status,
                    std::string& err,
                    const std::function<bool()>* cancel = nullptr,
                    const std::function<void(std::uint64_t)>* on_bytes
                        = nullptr);

// ---- integrity ---------------------------------------------------------

// What a repo publishes about one file, for the check after download.
//
// HuggingFace stores anything big in LFS and names the object by the
// SHA-256 of its CONTENT; everything small enough to live in git itself
// is named by the git blob id, a SHA-1 over "blob <size>\0" followed by
// the content. Every file carries one or the other -- and no MD5: the
// API publishes none, for any file, so there is no MD5 to check
// against.
struct FileDigest {
  std::uint64_t size = 0;   // 0 -> not published
  std::string   sha256;     // LFS object id
  std::string   git_oid;    // git blob id
};

// The outcome of checking one downloaded file.
enum class FileCheck {
  Ok,             // matched what the repo published
  NotPublished,   // nothing to compare against
  Mismatch,       // wrong size or wrong digest -- the bytes are bad
  Unreadable,     // could not be hashed at all
};

// Hash `path` the way the repo named it. `git_blob` picks the git
// flavour: SHA-1 primed with the "blob <size>\0" header git hashes
// ahead of the content, which is what makes a small file's id
// reproducible here without a checkout. Streams in a fixed buffer, so a
// 20 GB shard costs a megabyte rather than a copy of itself.
bool file_digest_hex(const std::filesystem::path& path, bool git_blob,
                     std::string& hex, std::string& err);

// Compare `path` against whatever the repo published for it, saying in
// `detail` which check ran. A file the repo publishes nothing for is
// NotPublished, NOT a pass: a fetch should be able to report that it
// checked nothing rather than claim a check that never happened.
FileCheck check_file_digest(const std::filesystem::path& path,
                            const FileDigest& want, std::string& detail);

// ---- download ----------------------------------------------------------

// Where a Xet-backed file can be reconstructed from. `hash` is the
// repo's `xetHash`; empty means the repo publishes none and the file
// can only be taken as one stream. See xet-fetch.h.
struct XetSource {
  std::string repo;        // "owner/name"
  std::string revision;    // "main", a branch, or a commit sha
  std::string hash;        // xetHash
};

// One file to fetch, and everything known about where it comes from.
struct FetchRequest {
  std::string url;      // the plain byte-stream URL, and the fallback
  std::string token;    // bearer token; "" for a public repo
  FileDigest  want;     // what the repo published about the bytes
  XetSource   xet;      // optional; empty hash -> plain download only
};

// How a fetch is bounded end to end.
struct FetchOpts {
  bool     verify_tls = true;
  // Abandon a transfer after this long below one kilobyte a second, and
  // retry it from where it stopped. This is the timeout that matters
  // for big files: it fires on a connection that has died, never on one
  // that is merely slow. 0 -> none.
  long     stall_s = 0;
  // Extra attempts after the first, each resuming from the partial file
  // (waits 2/5/15/30/60s between).
  unsigned retries = 0;
  // Check the bytes against the repo's id before accepting them.
  bool     verify = true;
  // How many ranges the Xet path pulls at once, when the repo publishes
  // a hash to reconstruct from. 0 turns Xet off and takes every file as
  // one stream. One connection is the throughput ceiling here, not the
  // link, which is the whole reason the setting exists.
  unsigned xet_streams = 8;
  // How much fetched-but-not-yet-written CAS data may be held. Bounds
  // the memory a fast link can run ahead by; ranges are up to ~64 MB
  // apiece, so this is a handful of them.
  std::size_t xet_window_bytes = std::size_t{256} << 20;
};

// GET `url` into `dest` (parent dirs created), resuming an interrupted
// attempt and checking the bytes against what the repo published.
//
// When the request carries a Xet source and `o.xet_streams` is not 0,
// the bytes come from the content store -- many ranges at once -- and
// the plain URL is the fallback for whatever that cannot deliver.
//
// A source that publishes no checksum is downloaded but not resumed:
// what makes resume safe is the check at the end, and without one a
// stale part splices into a file of the right length that is wrong in
// the middle.
//
// The bytes land in "<dest>.part" and are renamed into place only once
// they check out, so `dest` existing always means a COMPLETE file --
// which is what lets a caller trust a size match when it skips files
// already present, and what lets a fetch that was killed be RE-RUN
// rather than restarted. It is also what makes resume safe: a part left
// behind by a different revision fails the check and is taken again
// whole, instead of being spliced onto the new bytes.
//
// `*status` carries the HTTP code; `err` set + false on failure. `s`
// (optional) gets a line per retry -- silence while a multi-GB shard
// restarts itself reads as a hang -- and a line naming the checksum for
// files over kBigFileBytes, which are the ones where "it was checked"
// is worth the room. `progress` (optional)
// is driven with the byte counts, the resumed prefix included. When
// `cancel` is non-null it is polled mid-transfer and between attempts.
//
// `checked` (optional) receives the verdict on the bytes, so a caller
// can tally how much of what it fetched was actually checked. Written
// only when `o.verify` is on and the file was accepted.
bool fetch_file(const SessionContextIntf* s, const FetchRequest& req,
                const FetchOpts& o, const std::filesystem::path& dest,
                long& status, std::string& err, UiProgress* progress,
                const std::function<bool()>* cancel = nullptr,
                FileCheck* checked = nullptr);

}

#endif
