#include "stages/xet-fetch.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include <compression.h>

using std::string;
using std::vector;
namespace fs = std::filesystem;

namespace vpipe {

namespace {

// ---- LZ4 ---------------------------------------------------------------

// One LZ4 frame, as the CAS writes them: magic, FLG, BD, header
// checksum, then length-prefixed blocks until a zero length. Apple's
// libcompression decodes the BLOCKS (COMPRESSION_LZ4_RAW is exactly the
// raw block format); the frame around them is ours to walk, which is
// why there is no third-party LZ4 here.
bool
lz4_frame_decode_(const unsigned char* p, std::size_t n, std::size_t want,
                  string& out, string& err)
{
  if (n < 7 || p[0] != 0x04 || p[1] != 0x22 || p[2] != 0x4d
      || p[3] != 0x18) {
    err = "not an LZ4 frame";
    return false;
  }
  const unsigned flg = p[4];
  if ((flg & 0xc0) != 0x40) {
    err = "unsupported LZ4 frame version";
    return false;
  }
  std::size_t i = 7;
  if (flg & 0x08) { i += 8; }        // content size
  if (flg & 0x01) { i += 4; }        // dictionary id
  const std::size_t base = out.size();
  out.resize(base + want);
  std::size_t done = 0;
  while (i + 4 <= n) {
    std::uint32_t bs = 0;
    std::memcpy(&bs, p + i, 4);
    i += 4;
    if (bs == 0) { break; }                       // end mark
    const bool     stored = (bs & 0x80000000u) != 0;
    const std::size_t len = bs & 0x7fffffffu;
    if (i + len > n) {
      err = "LZ4 block runs past the chunk";
      return false;
    }
    if (stored) {
      if (done + len > want) { err = "LZ4 block overruns"; return false; }
      std::memcpy(&out[base + done], p + i, len);
      done += len;
    } else {
      const std::size_t got = compression_decode_buffer(
          reinterpret_cast<std::uint8_t*>(&out[base + done]), want - done,
          p + i, len, nullptr, COMPRESSION_LZ4_RAW);
      if (got == 0) { err = "LZ4 block would not decode"; return false; }
      done += got;
    }
    i += len;
    if (flg & 0x10) { i += 4; }      // per-block checksum
  }
  if (done != want) {
    out.resize(base + done);
    err = fmt("LZ4 frame gave {} bytes, the chunk header says {}",
              done, want)();
    return false;
  }
  return true;
}

}

// ---- byte grouping -----------------------------------------------------

void
xet_bg4_regroup(char* buf, std::size_t n, string& scratch)
{
  // The writer split the stream into four planes -- bytes 0,4,8..., then
  // 1,5,9..., and so on -- and stored them back to back, so that the
  // high and low halves of adjacent bf16 values sit next to their own
  // kind and LZ4 has something to find. Undo it by walking the planes
  // in order and scattering each back across the stream.
  if (n < 2) { return; }
  scratch.assign(buf, n);
  const std::size_t q = n / 4;
  const std::size_t r = n % 4;
  std::size_t at = 0;
  for (std::size_t plane = 0; plane < 4; ++plane) {
    const std::size_t len = q + (plane < r ? 1 : 0);
    for (std::size_t k = 0; k < len; ++k) {
      buf[k * 4 + plane] = scratch[at + k];
    }
    at += len;
  }
}

std::size_t
xet_decode_chunk(const unsigned char* p, std::size_t n, string& out,
                 string& err)
{
  if (n < 8) {
    err = "truncated chunk header";
    return 0;
  }
  const std::size_t clen = static_cast<std::size_t>(p[1])
                         | (static_cast<std::size_t>(p[2]) << 8)
                         | (static_cast<std::size_t>(p[3]) << 16);
  const unsigned    scheme = p[4];
  const std::size_t ulen = static_cast<std::size_t>(p[5])
                         | (static_cast<std::size_t>(p[6]) << 8)
                         | (static_cast<std::size_t>(p[7]) << 16);
  if (8 + clen > n) {
    err = "chunk payload runs past the buffer";
    return 0;
  }
  const unsigned char* body = p + 8;
  switch (scheme) {
    case 0:                                   // stored as-is
      if (clen != ulen) {
        err = "uncompressed chunk with mismatched lengths";
        return 0;
      }
      out.append(reinterpret_cast<const char*>(body), clen);
      return 8 + clen;
    case 1:                                   // LZ4 frame
      if (!lz4_frame_decode_(body, clen, ulen, out, err)) { return 0; }
      return 8 + clen;
    case 2: {                                 // LZ4 frame over BG4
      const std::size_t at = out.size();
      if (!lz4_frame_decode_(body, clen, ulen, out, err)) { return 0; }
      string scratch;
      xet_bg4_regroup(&out[at], ulen, scratch);
      return 8 + clen;
    }
    default:
      err = fmt("unknown chunk compression scheme {}", scheme)();
      return 0;
  }
}

namespace {

// ---- the reconstruction manifest ---------------------------------------

// One fetch: a byte range of one content-addressed object, holding a
// known run of chunks.
struct Unit {
  string        url;
  std::uint64_t off    = 0;   // inclusive byte range, as signed for
  std::uint64_t last   = 0;   // ... and as the signature demands
  std::uint32_t chunk0 = 0;   // chunk index the range starts at
  std::uint32_t uses   = 0;   // terms still to be written from it
};

// One slice of the output file: chunks [first, last) of a unit.
struct Term {
  std::size_t   unit  = 0;
  std::uint32_t first = 0;
  std::uint32_t last  = 0;
  std::uint64_t bytes = 0;    // what it contributes to the file
};

struct Plan {
  vector<Unit>  units;
  vector<Term>  terms;
  std::uint64_t skip = 0;     // offset_into_first_range
};

std::uint64_t
u64_(const FlexData& o, const char* key)
{
  auto obj = o.as_object();
  return obj.contains(key) ? obj.at(key).as_uint(0) : 0;
}

// GET the CAS grant for a repo: where the store is, and a bearer token
// for it. Both expire, which is why the whole plan is rebuilt rather
// than patched when a range comes back refused.
bool
xet_auth_(const XetSource& src, const string& hf_token, bool verify_tls,
          long timeout_s, string& cas_url, string& cas_token, string& err)
{
  const string url = "https://huggingface.co/api/models/" + src.repo
                   + "/xet-read-token/"
                   + (src.revision.empty() ? "main" : src.revision);
  string body;
  long   status = 0;
  if (!http_get_text(url, hf_token, verify_tls, timeout_s, body, status,
                     err)) {
    err = fmt("xet grant for '{}': {}", src.repo, err)();
    return false;
  }
  FlexData d;
  try {
    d = FlexData::from_json(body);
  } catch (const std::exception& e) {
    err = fmt("xet grant is not JSON: {}", e.what())();
    return false;
  }
  if (!d.is_object()) {
    err = "xet grant is not an object";
    return false;
  }
  auto o = d.as_object();
  cas_url = o.contains("casUrl") ? string(o.at("casUrl").as_string(""))
                                 : string();
  cas_token = o.contains("accessToken")
      ? string(o.at("accessToken").as_string("")) : string();
  if (cas_url.empty() || cas_token.empty()) {
    err = "xet grant carries no casUrl/accessToken";
    return false;
  }
  return true;
}

// GET and flatten the reconstruction manifest into something that can
// be walked start to finish: the ordered slices of the file, and the
// unique ranges they are cut from.
bool
xet_plan_(const XetSource& src, const string& cas_url,
          const string& cas_token, bool verify_tls, long timeout_s,
          Plan& plan, string& err)
{
  string body;
  long   status = 0;
  const string url = cas_url + "/v1/reconstructions/" + src.hash;
  if (!http_get_text(url, cas_token, verify_tls, timeout_s, body, status,
                     err)) {
    err = fmt("reconstruction for {}: {}", src.hash.substr(0, 12), err)();
    return false;
  }
  FlexData d;
  try {
    d = FlexData::from_json(body);
  } catch (const std::exception& e) {
    err = fmt("reconstruction is not JSON: {}", e.what())();
    return false;
  }
  if (!d.is_object()) {
    err = "reconstruction is not an object";
    return false;
  }
  auto root = d.as_object();
  plan.skip = u64_(d, "offset_into_first_range");
  if (!root.contains("terms") || !root.contains("fetch_info")) {
    err = "reconstruction has no terms/fetch_info";
    return false;
  }
  FlexData terms = root.at("terms");
  FlexData fetch = root.at("fetch_info");
  if (!terms.is_array() || !fetch.is_object()) {
    err = "reconstruction terms/fetch_info are the wrong shape";
    return false;
  }
  auto tarr = terms.as_array();
  auto fobj = fetch.as_object();
  // url+range -> index, so a range serving several slices is fetched
  // once. That is where the dedup saving actually lands.
  std::map<string, std::size_t> seen;
  for (std::size_t i = 0; i < tarr.size(); ++i) {
    FlexData t = tarr.at(i);
    if (!t.is_object()) { continue; }
    auto to = t.as_object();
    const string hash = to.contains("hash")
        ? string(to.at("hash").as_string("")) : string();
    FlexData tr = to.contains("range") ? to.at("range") : FlexData();
    if (hash.empty() || !tr.is_object()) {
      err = "reconstruction term is missing its hash/range";
      return false;
    }
    Term term;
    term.first = static_cast<std::uint32_t>(u64_(tr, "start"));
    term.last  = static_cast<std::uint32_t>(u64_(tr, "end"));
    term.bytes = u64_(t, "unpacked_length");
    if (!fobj.contains(hash)) {
      err = fmt("no fetch info for term {}", i)();
      return false;
    }
    FlexData entries = fobj.at(hash);
    if (!entries.is_array()) {
      err = "fetch info entry is not an array";
      return false;
    }
    auto earr = entries.as_array();
    bool placed = false;
    for (std::size_t k = 0; k < earr.size() && !placed; ++k) {
      FlexData e = earr.at(k);
      if (!e.is_object()) { continue; }
      auto eo = e.as_object();
      FlexData er = eo.contains("range") ? eo.at("range") : FlexData();
      FlexData ur = eo.contains("url_range") ? eo.at("url_range")
                                             : FlexData();
      if (!er.is_object() || !ur.is_object()) { continue; }
      const std::uint64_t cs = u64_(er, "start");
      const std::uint64_t ce = u64_(er, "end");
      // The entry has to COVER the slice; a partial overlap would need
      // stitching across two ranges, which the server never asks for.
      if (cs > term.first || ce < term.last) { continue; }
      Unit u;
      u.url    = eo.contains("url") ? string(eo.at("url").as_string(""))
                                    : string();
      u.off    = u64_(ur, "start");
      u.last   = u64_(ur, "end");
      u.chunk0 = static_cast<std::uint32_t>(cs);
      if (u.url.empty()) { continue; }
      const string key = hash + ":" + std::to_string(u.off) + "-"
                       + std::to_string(u.last);
      auto it = seen.find(key);
      if (it == seen.end()) {
        it = seen.emplace(key, plan.units.size()).first;
        plan.units.push_back(std::move(u));
      }
      term.unit = it->second;
      plan.units[term.unit].uses += 1;
      placed = true;
    }
    if (!placed) {
      err = fmt("no fetch range covers term {}", i)();
      return false;
    }
    plan.terms.push_back(term);
  }
  if (plan.terms.empty()) {
    err = "reconstruction lists no terms";
    return false;
  }
  return true;
}

// How often the report is pushed while ranges are in flight.
//
// The producer-side contract is "do not throttle" (UiProgressRegistry):
// one libcurl callback fires hundreds of times a second, the renderers
// coalesce, and update() is cheap. Eight of them are a different
// proposition -- they fired ~186 times a second on the measured run --
// and every push is a mutex plus two formatted byte counts for a number
// no renderer reads more than ten times a second. Half a second is what
// this report is worth: a bar that never stands still longer than that
// reads as live, and it is the web UI's own poll rate.
constexpr std::int64_t kReportEveryNs = std::int64_t{500} * 1000 * 1000;

std::int64_t
steady_ns_()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// ---- the parallel walk -------------------------------------------------

// Shared state for one pass over a plan: workers pull units in order,
// the writer appends slices in order, and the resident set is bounded so
// a 30 GB file does not want 30 GB of memory on the way in.
struct Walk {
  std::mutex              m;
  std::condition_variable cv;
  std::map<std::size_t, std::shared_ptr<const string>> resident;
  std::size_t             next_unit  = 0;
  std::size_t             held_bytes = 0;
  // The range the writer is currently blocked on. The memory budget
  // must never hold THIS one back: everything resident is there because
  // a later slice still wants it, so waiting for room that only the
  // writer can free, while the writer waits for this range, is a
  // deadlock. Fetching one range over the budget is the way out.
  std::size_t             want       = 0;
  unsigned                live       = 0;   // workers still going
  bool                    failed     = false;
  bool                    done       = false;
  bool                    drained    = false;
  string                  err;
};

}

bool
xet_fetch(const SessionContextIntf* s, const XetSource& src,
          const string& hf_token, const FetchOpts& o, const fs::path& part,
          std::uint64_t total, string& err, UiProgress* progress,
          const std::function<bool()>* cancel)
{
  if (src.hash.empty() || src.repo.empty() || o.xet_streams == 0) {
    err = "no xet hash for this file";
    return false;
  }
  // Two passes at most: a grant and the signed ranges inside a manifest
  // both expire, and the only fix is to ask for new ones. Whatever was
  // appended stays -- it is a prefix of the file either way.
  for (unsigned attempt = 0; attempt < 2; ++attempt) {
    string cas_url, cas_token;
    if (!xet_auth_(src, hf_token, o.verify_tls, 60, cas_url, cas_token,
                   err)) {
      return false;
    }
    Plan plan;
    if (!xet_plan_(src, cas_url, cas_token, o.verify_tls, 120, plan,
                   err)) {
      return false;
    }
    // Where the part already reaches. Slices are appended whole, so the
    // resume point is a slice boundary -- anything past the last whole
    // one is dropped rather than guessed at.
    std::error_code ec;
    std::uint64_t   have = 0;
    {
      const auto sz = fs::file_size(part, ec);
      if (!ec) { have = static_cast<std::uint64_t>(sz); }
    }
    std::size_t   start = 0;
    std::uint64_t at    = 0;
    while (start < plan.terms.size()
           && at + plan.terms[start].bytes <= have) {
      at += plan.terms[start].bytes;
      ++start;
    }
    if (at != have) {
      // A partial slice from a killed run. Cut back to the boundary.
      fs::resize_file(part, at, ec);
      if (ec) {
        err = fmt("cannot trim '{}': {}", part.string(), ec.message())();
        return false;
      }
      have = at;
    }
    if (start >= plan.terms.size()) { return true; }   // already whole
    if (s && attempt == 0) {
      s->info(fmt("    {}: {} ranges over {} objects, {} at a time",
                  part.stem().string(), plan.terms.size(),
                  plan.units.size(), o.xet_streams));
    }
    for (std::size_t i = 0; i < start; ++i) {
      Unit& u = plan.units[plan.terms[i].unit];
      if (u.uses > 0) { u.uses -= 1; }
    }

    // ---- what the report is driven from ------------------------------
    //
    // NOT the bytes the writer has appended. The writer advances only
    // when the ONE range it is waiting for lands, while the other N-1
    // are still coming down, so the file-bytes count stands still for as
    // long as that takes and then jumps. MEASURED on a 1.0 GB shard at 8
    // streams: a 1 Hz renderer saw six consecutive seconds of nothing
    // and then a 307 MB step, and the longest gap between appends was
    // 7.5 s. The bytes were arriving the whole time; nothing was
    // counting them.
    //
    // So each range is credited as it ARRIVES, for what it is worth in
    // FILE bytes -- the slices cut from it, which the manifest states up
    // front. The sum over ranges is the rest of the file exactly, so the
    // report still lands on the same number, and it can only run AHEAD
    // of the writer (a slice is appended from a range that is already
    // whole, hence already fully credited), never behind it.
    const std::uint64_t have0 = have;
    vector<std::uint64_t> worth(plan.units.size(), 0);
    for (std::size_t i = start; i < plan.terms.size(); ++i) {
      std::uint64_t b = plan.terms[i].bytes;
      // The manifest can start the first slice partway in; the writer
      // drops that prefix, so it is not worth anything here either.
      if (i == 0 && plan.skip > 0 && b > plan.skip) { b -= plan.skip; }
      worth[plan.terms[i].unit] += b;
    }
    std::atomic<std::uint64_t> credit{0};
    // When the next push is due. Shared, so the rate is the REPORT's,
    // not one worker's -- eight independent half-second timers would be
    // eight times the pushes for no more information.
    std::atomic<std::int64_t> due{0};

    std::ofstream ofs(part, std::ios::binary | std::ios::app);
    if (!ofs) {
      err = fmt("cannot open '{}' for writing", part.string())();
      return false;
    }
    Walk w;
    // From the FIRST unit, not from the one the resume point happens to
    // land on: ranges are numbered in first-use order, and a slice after
    // the resume point can still be cut from a range first used before
    // it. Starting anywhere else leaves that range unfetched and the
    // writer waiting for it forever. Ranges nothing needs any more are
    // skipped without being fetched, so this costs a loop, not bytes.
    w.next_unit = 0;
    const std::size_t budget = o.xet_window_bytes;

    // Every exit path runs this, so the writer is never left waiting on
    // a range that nobody is going to fetch.
    auto retire = [&] {
      std::lock_guard<std::mutex> lk(w.m);
      if (w.live > 0 && --w.live == 0) { w.drained = true; }
      w.cv.notify_all();
    };
    auto worker = [&] {
      for (;;) {
        std::size_t u = 0;
        {
          std::unique_lock<std::mutex> lk(w.m);
          w.cv.wait(lk, [&] {
            return w.failed || w.done || w.next_unit >= plan.units.size()
                || w.held_bytes < budget || w.next_unit <= w.want;
          });
          if (w.failed || w.done || w.next_unit >= plan.units.size()) {
            break;
          }
          u = w.next_unit++;
          if (plan.units[u].uses == 0) { continue; }   // nothing wants it
        }
        const Unit& un = plan.units[u];
        auto body = std::make_shared<string>();
        long   st = 0;
        string e2;
        bool   ok = false;
        // This range's share of the report, scaled by how much of it has
        // arrived. `given` is what this worker has already added, so a
        // retry -- which starts the range over at zero -- hands its own
        // credit back instead of counting the bytes twice.
        //
        // worth[u] is at most the whole file and a range is at most
        // ~64 MB, so the product cannot overflow short of an exabyte.
        const std::uint64_t wire =
            un.last >= un.off ? un.last - un.off + 1 : 0;
        std::uint64_t given = 0;
        std::function<void(std::uint64_t)> arrived =
            [&](std::uint64_t got) {
          const std::uint64_t c = (wire == 0 || got >= wire)
              ? worth[u]
              : (worth[u] * got) / wire;
          if (c > given)      { credit.fetch_add(c - given); }
          else if (c < given) { credit.fetch_sub(given - c); }
          given = c;
          if (!progress) { return; }
          // Whoever gets there first pushes for everyone, and the value
          // is the total across all eight, so it does not matter which
          // worker that is.
          const std::int64_t now = steady_ns_();
          std::int64_t       at  = due.load(std::memory_order_relaxed);
          if (now < at
              || !due.compare_exchange_strong(at, now + kReportEveryNs,
                                              std::memory_order_relaxed)) {
            return;
          }
          const std::uint64_t d = have0 + credit.load();
          progress->update(d, total,
                           human_bytes(d) + " / " + human_bytes(total));
        };
        for (unsigned a = 0; a <= o.retries && !ok; ++a) {
          if (cancel && (*cancel)()) { break; }
          ok = http_get_range(un.url, string(), o.verify_tls, o.stall_s,
                              un.off, un.last, *body, st, e2, cancel,
                              &arrived);
          if (!ok && (st == 401 || st == 403)) { break; }   // expired
        }
        // Settle exactly on success: a server that sent the range in
        // fewer bytes than asked for would otherwise leave the report a
        // little short of where the file actually is.
        if (ok && given < worth[u]) { credit.fetch_add(worth[u] - given); }
        std::lock_guard<std::mutex> lk(w.m);
        if (!ok) {
          w.failed = true;
          w.err = fmt("range {}-{}: {}", un.off, un.last, e2)();
          w.cv.notify_all();
          break;
        }
        w.held_bytes += body->size();
        w.resident.emplace(u, std::move(body));
        w.cv.notify_all();
      }
      retire();
    };

    vector<std::thread> pool;
    const unsigned      n = std::max(1u, o.xet_streams);
    pool.reserve(n);
    w.live = n;
    for (unsigned i = 0; i < n; ++i) { pool.emplace_back(worker); }

    bool   good  = true;
    string local_err;
    string slice;
    for (std::size_t i = start; i < plan.terms.size() && good; ++i) {
      const Term& t = plan.terms[i];
      std::shared_ptr<const string> body;
      {
        std::unique_lock<std::mutex> lk(w.m);
        w.want = t.unit;
        w.cv.notify_all();
        w.cv.wait(lk, [&] {
          return w.failed || w.drained || w.resident.count(t.unit) != 0
              || (cancel && (*cancel)());
        });
        if (w.failed) { good = false; local_err = w.err; break; }
        if (cancel && (*cancel)()) {
          good = false;
          local_err = "canceled";
          break;
        }
        auto it = w.resident.find(t.unit);
        if (it == w.resident.end()) {
          // Everyone has finished and what this slice needs is not
          // here. Say so rather than wait for a range that is never
          // coming.
          good = false;
          local_err = fmt("range {} was never fetched", t.unit)();
          break;
        }
        body = it->second;
      }
      // Walk the range's chunks from where it starts, skipping to the
      // first the slice wants and stopping after the last.
      slice.clear();
      const auto* p = reinterpret_cast<const unsigned char*>(body->data());
      std::size_t off = 0;
      const Unit& un  = plan.units[t.unit];
      for (std::uint32_t c = un.chunk0; c < t.last; ++c) {
        if (off >= body->size()) {
          good = false;
          local_err = fmt("range ran out of chunks at {}", c)();
          break;
        }
        if (c < t.first) {
          // Not wanted, but its length is only in its header.
          const std::size_t clen = static_cast<std::size_t>(p[off + 1])
                                 | (static_cast<std::size_t>(p[off + 2])
                                    << 8)
                                 | (static_cast<std::size_t>(p[off + 3])
                                    << 16);
          off += 8 + clen;
          continue;
        }
        string e3;
        const std::size_t used =
            xet_decode_chunk(p + off, body->size() - off, slice, e3);
        if (used == 0) {
          good = false;
          local_err = fmt("chunk {}: {}", c, e3)();
          break;
        }
        off += used;
      }
      if (!good) { break; }
      const char*  out_p = slice.data();
      std::size_t  out_n = slice.size();
      if (i == start && start == 0 && plan.skip > 0) {
        // The manifest can start the first slice partway in.
        if (plan.skip >= out_n) {
          good = false;
          local_err = "leading offset runs past the first slice";
          break;
        }
        out_p += plan.skip;
        out_n -= plan.skip;
      }
      ofs.write(out_p, static_cast<std::streamsize>(out_n));
      if (!ofs) {
        good = false;
        local_err = fmt("cannot write '{}'", part.string())();
        break;
      }
      // `have` is the file's own accounting -- the resume point, and
      // what the next pass trims back to. It is deliberately NOT the
      // report: see the credit table above.
      have += out_n;
      std::lock_guard<std::mutex> lk(w.m);
      Unit& mu = plan.units[t.unit];
      if (mu.uses > 0) { mu.uses -= 1; }
      if (mu.uses == 0) {
        auto it = w.resident.find(t.unit);
        if (it != w.resident.end()) {
          w.held_bytes -= it->second->size();
          w.resident.erase(it);
        }
      }
      w.cv.notify_all();
    }
    {
      std::lock_guard<std::mutex> lk(w.m);
      w.done = true;
      w.cv.notify_all();
    }
    for (std::thread& t : pool) { t.join(); }
    ofs.close();
    if (good) {
      // Land exactly on the finished size. The in-flight report is
      // pushed on a clock, so the last one can be half a second short of
      // the end -- and nothing reports this file again.
      if (progress) {
        progress->update(have, total,
                         human_bytes(have) + " / " + human_bytes(total));
      }
      return true;
    }
    if (cancel && (*cancel)()) {
      err = "canceled";
      return false;
    }
    err = local_err;
    // A refused range means the grant aged out mid-download; the second
    // pass takes a fresh one and picks up from what is already there.
    if (attempt == 0 && err.find("HTTP 40") != string::npos) {
      if (s) {
        s->info(fmt("    {}: the store grant expired; renewing",
                    part.stem().string()));
      }
      continue;
    }
    return false;
  }
  return false;
}

}
