#include "apps/web-ui/file-api.h"
#include "apps/web-ui/api-common.h"

#include "interfaces/session-context-intf.h"
#include "common/vpipe-format.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe::webui {

namespace {



// MIME type for a file path, keyed off its (lower-cased) extension.
// Covers the image / audio / video / text families the file-browser
// preview understands; anything else is served as a generic byte
// stream so the browser offers to download it rather than mis-render.
string
mime_for_path_(string_view path)
{
  const auto dot = path.rfind('.');
  string ext;
  if (dot != string_view::npos) {
    ext = string(path.substr(dot + 1));
    for (char& c : ext) { c = static_cast<char>(std::tolower(c)); }
  }
  static const std::unordered_map<string, string> kMime = {
    { "png", "image/png" },   { "jpg", "image/jpeg" },
    { "jpeg", "image/jpeg" }, { "webp", "image/webp" },
    { "gif", "image/gif" },   { "bmp", "image/bmp" },
    { "svg", "image/svg+xml" }, { "tiff", "image/tiff" },
    { "tif", "image/tiff" },  { "heic", "image/heic" },
    { "ico", "image/x-icon" },
    { "wav", "audio/wav" },   { "mp3", "audio/mpeg" },
    { "flac", "audio/flac" }, { "aac", "audio/aac" },
    { "m4a", "audio/mp4" },   { "ogg", "audio/ogg" },
    { "opus", "audio/ogg" },  { "aiff", "audio/aiff" },
    { "aif", "audio/aiff" },
    { "mp4", "video/mp4" },   { "m4v", "video/mp4" },
    { "mov", "video/quicktime" }, { "webm", "video/webm" },
    { "mkv", "video/x-matroska" }, { "avi", "video/x-msvideo" },
    { "ts", "video/mp2t" },   { "flv", "video/x-flv" },
    { "mpg", "video/mpeg" },  { "mpeg", "video/mpeg" },
    { "txt", "text/plain; charset=utf-8" },
    { "md", "text/markdown; charset=utf-8" },
    { "log", "text/plain; charset=utf-8" },
    { "csv", "text/csv; charset=utf-8" },
    { "srt", "text/plain; charset=utf-8" },
    { "vtt", "text/vtt; charset=utf-8" },
    { "json", "application/json; charset=utf-8" },
    { "xml", "application/xml; charset=utf-8" },
    { "yaml", "text/plain; charset=utf-8" },
    { "yml", "text/plain; charset=utf-8" },
    { "html", "text/html; charset=utf-8" },
    { "htm", "text/html; charset=utf-8" },
  };
  const auto it = kMime.find(ext);
  return it != kMime.end() ? it->second : "application/octet-stream";
}

// Parse a single-range HTTP "Range: bytes=START-END" header (END
// optional; a suffix form "bytes=-N" means the last N bytes). Only the
// first range of a list is honoured. Returns false when unparseable or
// unsatisfiable; on success *start/*end are inclusive absolute offsets
// clamped to [0, total-1].
bool
parse_range_(const string& hdr, uint64_t total, uint64_t* start,
             uint64_t* end)
{
  if (total == 0) { return false; }
  const string pfx = "bytes=";
  if (hdr.compare(0, pfx.size(), pfx) != 0) { return false; }
  string spec = hdr.substr(pfx.size());
  const auto comma = spec.find(',');
  if (comma != string::npos) { spec = spec.substr(0, comma); }
  const auto dash = spec.find('-');
  if (dash == string::npos) { return false; }
  const string a = trim(spec.substr(0, dash));
  const string b = trim(spec.substr(dash + 1));
  uint64_t s, e;
  if (a.empty()) {
    if (b.empty()) { return false; }
    uint64_t n = strtoull(b.c_str(), nullptr, 10);
    if (n == 0) { return false; }
    if (n > total) { n = total; }
    s = total - n;
    e = total - 1;
  } else {
    s = strtoull(a.c_str(), nullptr, 10);
    e = b.empty() ? (total - 1) : strtoull(b.c_str(), nullptr, 10);
  }
  if (s > e || s >= total) { return false; }
  if (e >= total) { e = total - 1; }
  *start = s;
  *end   = e;
  return true;
}

// Read `len` bytes starting at byte offset `off` from `p` into `out`.
// `out` is resized to the number of bytes actually read (short at EOF).
// Returns false only on an open/seek failure.
bool
read_file_slice_(const std::filesystem::path& p, uint64_t off,
                 uint64_t len, string* out)
{
  std::ifstream f(p, std::ios::binary);
  if (!f) { return false; }
  f.seekg(static_cast<std::streamoff>(off));
  if (!f) { return false; }
  out->resize(static_cast<size_t>(len));
  f.read(out->data(), static_cast<std::streamsize>(len));
  const std::streamsize got = f.gcount();
  out->resize(got > 0 ? static_cast<size_t>(got) : 0);
  return true;
}

}  // namespace

HttpResponse
FileApi::h_cwd_pipelines_(const HttpRequest&)
{
  // Returns the server-process cwd plus the .vpipeline filenames at
  // its top level. The Load-Pipeline dialog uses this to populate a
  // <datalist>, giving the browser native autocomplete-as-you-type
  // over the available files. Non-recursive on purpose -- it's a
  // quick picker, not a file browser. A user pointing to a path in
  // another directory just types it directly.
  namespace fs = std::filesystem;
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  std::error_code ec;
  fs::path cwd = fs::current_path(ec);
  oo.insert("cwd", fstr(ec ? string() : cwd.string()));
  FlexData arr = FlexData::make_array();
  auto av = arr.as_array();
  if (!ec) {
    vector<string> names;
    for (auto it = fs::directory_iterator(cwd, ec);
         !ec && it != fs::directory_iterator();
         it.increment(ec)) {
      std::error_code stat_ec;
      if (!it->is_regular_file(stat_ec) || stat_ec) { continue; }
      const auto name = it->path().filename().string();
      if (name.size() > 10
          && name.compare(name.size() - 10, 10, ".vpipeline") == 0) {
        names.push_back(name);
      }
    }
    std::sort(names.begin(), names.end());
    for (const auto& n : names) {
      av.push_back(FlexData::make_string(n));
    }
  }
  oo.insert("files", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
FileApi::h_list_(const HttpRequest& req)
{
  // Directory listing for the web-ui file open/save dialog and the file
  // browser. Operates in the session's filesystem namespace: when the
  // sandbox is active the client sees a chroot-like virtual tree whose
  // root is "/" (real host paths never leave the server); otherwise the
  // virtual path IS the host path. Read-only, non-recursive: one
  // directory per call, optionally windowed (offset/limit) + dirs-only
  // (see the query-param block below) for large directories.
  namespace fs = std::filesystem;
  const bool sb = _ctx.sctx && _ctx.sctx->fs_sandboxed();
  auto strip = [](string s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == '/' || s[i] == '\\')) { ++i; }
    return s.substr(i);
  };

  // The client encodes the path with encodeURIComponent (so '/', spaces
  // and unicode survive the query string) -- decode it before use.
  string   want = url_decode(query_param(req.query, "path"));
  string   vpath;      // clean virtual path echoed to the client
  fs::path real;       // real directory actually enumerated
  std::error_code ec;
  if (sb) {
    if (want.empty()) { want = "/"; }
    // Normalise inside the virtual space; lexically_normal collapses "."
    // and clamps ".." at the root so the browser cannot point above it.
    fs::path v =
        fs::path("/" + strip(want)).lexically_normal();
    vpath = v.generic_string();
    if (vpath.empty()) { vpath = "/"; }
    const string rel = strip(vpath);
    if (rel.empty()) {
      real = _ctx.sctx->fs_sandbox_root();
    } else {
      string cerr;
      real = _ctx.sctx->confine_path(vpath, /*for_write=*/false, &cerr);
      if (real.empty()) {
        return HttpResponse::error(
            400, "path '" + vpath + "' rejected: "
                     + (cerr.empty() ? "outside sandbox" : cerr));
      }
    }
  } else {
    fs::path v = want.empty() ? fs::current_path(ec) : fs::path(want);
    if (v.is_relative()) { v = fs::current_path(ec) / v; }
    fs::path canon = fs::weakly_canonical(v, ec);
    real  = (ec || canon.empty()) ? v.lexically_normal() : canon;
    vpath = real.generic_string();
  }

  std::error_code de;
  if (!fs::is_directory(real, de) || de) {
    return HttpResponse::error(400, "not a directory: " + vpath);
  }

  // Optional pagination + dirs-only filter (large-directory support).
  // The file browser requests fixed-size windows (offset/limit) so a
  // directory with thousands of entries is delivered as small responses
  // instead of one large one; `dirs_only` returns only sub-directories
  // (the tree pane, which never needs the files). Absent/zero limit =>
  // the whole sorted listing, preserving the open/save dialog's one-shot
  // behaviour. file_size() -- the one enumeration cost that always needs
  // a stat -- is charged only for the entries actually returned, so a
  // 50k-file directory is not fully stat'd to serve a 200-row window.
  const string dirs_only_s = query_param(req.query, "dirs_only");
  const bool   dirs_only =
      (dirs_only_s == "1" || dirs_only_s == "true");
  const string off_s = query_param(req.query, "offset");
  const string lim_s = query_param(req.query, "limit");
  const uint64_t off =
      off_s.empty() ? 0 : strtoull(off_s.c_str(), nullptr, 10);
  const uint64_t lim =
      lim_s.empty() ? 0 : strtoull(lim_s.c_str(), nullptr, 10);
  // Extension filter (the open/save dialog's category filters). A
  // comma-separated list of dot-led, lower-cased suffixes; a FILE is kept
  // only if its name ends with one. Empty => all files. Directories are
  // never filtered. Applied before sort/paging so `total` reflects the
  // filtered count and the client's windows stay consistent.
  std::vector<string> ext_filter;
  {
    const string raw = url_decode(query_param(req.query, "exts"));
    size_t p = 0;
    while (p <= raw.size()) {
      const size_t c = raw.find(',', p);
      string e = raw.substr(p, c == string::npos ? string::npos : c - p);
      for (char& ch : e) { ch = static_cast<char>(std::tolower(ch)); }
      if (!e.empty()) { ext_filter.push_back(std::move(e)); }
      if (c == string::npos) { break; }
      p = c + 1;
    }
  }
  auto ext_ok = [&ext_filter](const string& name) {
    if (ext_filter.empty()) { return true; }
    string lname = name;
    for (char& ch : lname) { ch = static_cast<char>(std::tolower(ch)); }
    for (const auto& e : ext_filter) {
      if (lname.size() >= e.size()
          && lname.compare(lname.size() - e.size(), e.size(), e) == 0) {
        return true;
      }
    }
    return false;
  };

  // Whitelisted pass-through prefixes ("mounts") the browser can jump
  // to, and whether `real` sits AT one -- if so, "up" returns to the
  // sandbox home rather than a re-rooted parent confine would reject.
  std::vector<fs::path> wl = _ctx.sctx ? _ctx.sctx->fs_whitelist()
                                   : std::vector<fs::path>{};
  bool at_wl_root = false;
  if (!wl.empty()) {
    std::error_code ce;
    fs::path rc = fs::weakly_canonical(real, ce);
    if (ce || rc.empty()) { rc = real.lexically_normal(); }
    for (const auto& w : wl) {
      if (rc == w) { at_wl_root = true; break; }
    }
  }

  // Parent (virtual). Empty string == "already at the root, no parent".
  string parent;
  if (strip(vpath).empty()) {
    parent = "";                       // sandbox root
  } else if (sb && at_wl_root) {
    parent = "/";                      // a granted mount -> up == home
  } else {
    fs::path pp = fs::path(vpath).parent_path();
    if (pp.empty() || pp == fs::path(vpath)) {
      parent = "";                     // native filesystem root
    } else {
      parent = pp.generic_string();
      if (sb && parent.empty()) { parent = "/"; }
    }
  }

  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("sandboxed", FlexData::make_bool(sb));
  oo.insert("path", fstr(vpath));
  oo.insert("parent", fstr(parent));
  // Granted mounts (empty unless --white-list-path was given). Surfaced
  // ONLY at the sandbox ROOT and only on the FIRST window (offset 0): they
  // are shortcuts to jump OUT of the sandbox home, so repeating them inside
  // every directory listing -- inside a mount, or on a later page -- is
  // just noise. Included for dirs_only too: the file browser's TREE builds
  // its root from a dirs_only listing and needs the mounts as sibling roots.
  FlexData mounts = FlexData::make_array();
  if (sb && strip(vpath).empty() && off == 0) {
    auto ma = mounts.as_array();
    for (const auto& w : wl) {
      FlexData mo = FlexData::make_object();
      auto x = mo.as_object();
      const std::string base = w.filename().string();
      x.insert("name", fstr(base.empty() ? w.generic_string() : base));
      x.insert("path", fstr(w.generic_string()));
      ma.push_back(std::move(mo));
    }
  }
  oo.insert("mounts", std::move(mounts));

  // Enumerate names + is-dir only (both come from the readdir d_type on
  // APFS, so no per-entry stat here); file_size is deferred to the slice.
  struct Ent { string name; bool dir; };
  vector<Ent> ents;
  for (auto it = fs::directory_iterator(
           real, fs::directory_options::skip_permission_denied, ec);
       !ec && it != fs::directory_iterator(); it.increment(ec)) {
    std::error_code se;
    const bool isdir = it->is_directory(se);
    if (se) { continue; }
    if (dirs_only && !isdir) { continue; }
    string nm = it->path().filename().string();
    if (nm.empty()) { continue; }
    if (!isdir) {
      std::error_code re;
      if (!it->is_regular_file(re) || re) { continue; }  // skip non-files
      if (!ext_ok(nm)) { continue; }                     // extension filter
    }
    ents.push_back(Ent{ std::move(nm), isdir });
  }
  auto ci_less = [](const string& a, const string& b) {
    return std::lexicographical_compare(
        a.begin(), a.end(), b.begin(), b.end(),
        [](unsigned char x, unsigned char y) {
          return std::tolower(x) < std::tolower(y);
        });
  };
  std::sort(ents.begin(), ents.end(), [&](const Ent& a, const Ent& b) {
    if (a.dir != b.dir) { return a.dir; }   // directories first
    return ci_less(a.name, b.name);
  });

  // Window [begin, end) of the sorted listing. lim == 0 => the whole list.
  const uint64_t total = static_cast<uint64_t>(ents.size());
  const uint64_t begin = std::min<uint64_t>(off, total);
  const uint64_t end   =
      (lim == 0) ? total : std::min<uint64_t>(begin + lim, total);

  FlexData arr = FlexData::make_array();
  auto av = arr.as_array();
  av.reserve(static_cast<size_t>(end - begin));
  for (uint64_t i = begin; i < end; ++i) {
    const Ent& e = ents[static_cast<size_t>(i)];
    FlexData eo = FlexData::make_object();
    auto x = eo.as_object();
    x.insert("name", fstr(e.name));
    x.insert("dir",  FlexData::make_bool(e.dir));
    if (!e.dir) {
      std::error_code ze;
      const auto sz = fs::file_size(real / e.name, ze);
      x.insert("size",
               FlexData::make_uint(ze ? 0 : static_cast<uint64_t>(sz)));
    }
    av.push_back(std::move(eo));
  }
  oo.insert("entries", std::move(arr));
  // total = full entry count (for the client's scrollbar / paging);
  // offset = the first index this window covers.
  oo.insert("total", FlexData::make_uint(total));
  oo.insert("offset", FlexData::make_uint(begin));
  return HttpResponse::json(200, o.to_json());
}

std::filesystem::path
FileApi::fs_resolve_(const string& want_in, bool for_write,
                        string* vpath_out, string* err_out) const
{
  // Map a client virtual path to a real host path using the SAME
  // namespace rules as h_fs_list_ (sandbox chroot-like "/"-root, or
  // native passthrough). Returns {} + sets *err_out on rejection.
  namespace fs = std::filesystem;
  const bool sb = _ctx.sctx && _ctx.sctx->fs_sandboxed();
  auto strip = [](string s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == '/' || s[i] == '\\')) { ++i; }
    return s.substr(i);
  };
  string   want = want_in;
  string   vpath;
  fs::path real;
  std::error_code ec;
  if (sb) {
    if (want.empty()) { want = "/"; }
    fs::path v = fs::path("/" + strip(want)).lexically_normal();
    vpath = v.generic_string();
    if (vpath.empty()) { vpath = "/"; }
    const string rel = strip(vpath);
    if (rel.empty()) {
      real = _ctx.sctx->fs_sandbox_root();
    } else {
      string cerr;
      real = _ctx.sctx->confine_path(vpath, for_write, &cerr);
      if (real.empty()) {
        if (err_out) {
          *err_out = "path '" + vpath + "' rejected: "
                     + (cerr.empty() ? "outside sandbox" : cerr);
        }
        return {};
      }
    }
  } else {
    fs::path v = want.empty() ? fs::current_path(ec) : fs::path(want);
    if (v.is_relative()) { v = fs::current_path(ec) / v; }
    fs::path canon = fs::weakly_canonical(v, ec);
    real  = (ec || canon.empty()) ? v.lexically_normal() : canon;
    vpath = real.generic_string();
  }
  if (vpath_out) { *vpath_out = vpath; }
  return real;
}

HttpResponse
FileApi::h_file_(const HttpRequest& req)
{
  // Serve one file's raw bytes for the file-browser preview (image /
  // audio / video / text). Honours a Range request (206 partial) and
  // bounds every response so a single request never buffers a whole
  // large media file into memory; the browser's media element (or the
  // text preview's Range fetch) pulls the rest in further chunks.
  namespace fs = std::filesystem;
  string   vpath, err;
  const string want = url_decode(query_param(req.query, "path"));
  fs::path real = fs_resolve_(want, /*for_write=*/false, &vpath, &err);
  if (real.empty()) {
    return HttpResponse::error(400, err.empty() ? "bad path" : err);
  }
  std::error_code ec;
  if (!fs::is_regular_file(real, ec) || ec) {
    return HttpResponse::error(404, "not a file: " + vpath);
  }
  const uint64_t total = static_cast<uint64_t>(fs::file_size(real, ec));
  if (ec) { return HttpResponse::error(500, "cannot stat: " + vpath); }

  constexpr uint64_t kChunkCap = 8ull * 1024 * 1024;    // per-206 ceiling
  constexpr uint64_t kFullCap  = 64ull * 1024 * 1024;   // full-GET ceiling

  uint64_t start = 0;
  uint64_t end   = total ? total - 1 : 0;
  bool     partial = false;
  const auto rit = req.headers.find("range");
  if (rit != req.headers.end()
      && parse_range_(rit->second, total, &start, &end)) {
    partial = true;
  } else if (total > kFullCap) {
    // No usable Range but the file is large: hand back a first bounded
    // slice as 206 so a media element continues via its own ranges.
    partial = true;
    start = 0;
    end   = kChunkCap - 1;
  }
  if (end >= total) { end = total ? total - 1 : 0; }
  if (partial && end >= start && (end - start + 1) > kChunkCap) {
    end = start + kChunkCap - 1;
  }
  const uint64_t len = (total == 0) ? 0 : (end - start + 1);

  string body;
  if (len && !read_file_slice_(real, start, len, &body)) {
    return HttpResponse::error(500, "read failed: " + vpath);
  }

  HttpResponse r;
  r.content_type = mime_for_path_(vpath);
  r.body = std::move(body);
  r.extra_headers.emplace_back("Accept-Ranges", "bytes");
  r.extra_headers.emplace_back("Cache-Control", "no-cache");
  if (partial) {
    r.status = 206;
    r.extra_headers.emplace_back(
        "Content-Range",
        "bytes " + std::to_string(start) + "-" + std::to_string(end)
            + "/" + std::to_string(total));
  }
  return r;
}

HttpResponse
FileApi::h_mkdir_(const HttpRequest& req)
{
  // Create a new directory `name` inside the (existing) virtual
  // directory `path`. Sandbox-confined for write.
  namespace fs = std::filesystem;
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "expected object {path, name}");
  }
  auto bo = body->as_object();
  const string dir  = string(bo.contains("path")
                                 ? bo.at("path").as_string("") : "");
  const string name = trim(string(bo.contains("name")
                                       ? bo.at("name").as_string("") : ""));
  if (name.empty() || name == "." || name == ".."
      || name.find('/') != string::npos
      || name.find('\\') != string::npos) {
    return HttpResponse::error(400, "invalid folder name");
  }
  string   pv, perr;
  fs::path parent = fs_resolve_(dir, /*for_write=*/false, &pv, &perr);
  if (parent.empty()) {
    return HttpResponse::error(400, perr.empty() ? "bad path" : perr);
  }
  std::error_code ec;
  if (!fs::is_directory(parent, ec) || ec) {
    return HttpResponse::error(400, "not a directory: " + pv);
  }
  const string childv = (pv == "/" ? "/" + name : pv + "/" + name);
  string   cv, cerr;
  fs::path real = fs_resolve_(childv, /*for_write=*/true, &cv, &cerr);
  if (real.empty()) {
    return HttpResponse::error(400, cerr.empty() ? "rejected" : cerr);
  }
  if (fs::exists(real, ec)) {
    return HttpResponse::error(409, "already exists: " + cv);
  }
  if (!fs::create_directory(real, ec) || ec) {
    return HttpResponse::error(
        500, "mkdir failed: " + (ec ? ec.message() : string("unknown")));
  }
  FlexData o = FlexData::make_object();
  o.as_object().insert("path", fstr(cv));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
FileApi::h_rename_(const HttpRequest& req)
{
  // Rename the item at virtual `path` to base name `to`, in place (same
  // parent directory). Works for files and directories.
  namespace fs = std::filesystem;
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "expected object {path, to}");
  }
  auto bo = body->as_object();
  const string srcv_in = string(bo.contains("path")
                                    ? bo.at("path").as_string("") : "");
  const string to = trim(string(bo.contains("to")
                                      ? bo.at("to").as_string("") : ""));
  if (srcv_in.empty()) {
    return HttpResponse::error(400, "missing 'path'");
  }
  if (to.empty() || to == "." || to == ".."
      || to.find('/') != string::npos || to.find('\\') != string::npos) {
    return HttpResponse::error(400, "invalid name");
  }
  string   sv, serr;
  fs::path src = fs_resolve_(srcv_in, /*for_write=*/false, &sv, &serr);
  if (src.empty()) {
    return HttpResponse::error(400, serr.empty() ? "bad path" : serr);
  }
  std::error_code ec;
  if (!fs::exists(src, ec) || ec) {
    return HttpResponse::error(404, "no such item: " + sv);
  }
  string parentv = fs::path(sv).parent_path().generic_string();
  if (parentv.empty()) { parentv = "/"; }
  const string dstv = (parentv == "/" ? "/" + to : parentv + "/" + to);
  string   dv, derr;
  fs::path dst = fs_resolve_(dstv, /*for_write=*/true, &dv, &derr);
  if (dst.empty()) {
    return HttpResponse::error(400, derr.empty() ? "rejected" : derr);
  }
  if (fs::exists(dst, ec)) {
    return HttpResponse::error(409, "already exists: " + dv);
  }
  fs::rename(src, dst, ec);
  if (ec) {
    return HttpResponse::error(500, "rename failed: " + ec.message());
  }
  FlexData o = FlexData::make_object();
  o.as_object().insert("path", fstr(dv));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
FileApi::h_write_(const HttpRequest& req)
{
  // Write a UTF-8 text file at virtual `path` (its parent must exist),
  // truncating any existing file. Sandbox-confined for write. Used by the
  // composer's Save-to-file (a JSON view arrangement).
  namespace fs = std::filesystem;
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "expected object {path, text}");
  }
  auto bo = body->as_object();
  const string pathv = string(bo.contains("path")
                                  ? bo.at("path").as_string("") : "");
  if (pathv.empty()) { return HttpResponse::error(400, "missing 'path'"); }
  const string text = string(bo.contains("text")
                                 ? bo.at("text").as_string("") : "");
  string   cv, cerr;
  fs::path real = fs_resolve_(pathv, /*for_write=*/true, &cv, &cerr);
  if (real.empty()) {
    return HttpResponse::error(400, cerr.empty() ? "rejected" : cerr);
  }
  std::error_code ec;
  if (fs::is_directory(real, ec)) {
    return HttpResponse::error(400, "is a directory: " + cv);
  }
  ofstream f(real, ios::binary | ios::trunc);
  if (!f) {
    return HttpResponse::error(500, "cannot open '" + cv + "' for write");
  }
  f.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!f) {
    return HttpResponse::error(500, "write to '" + cv + "' failed");
  }
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("ok", FlexData::make_bool(true));
  oo.insert("path", fstr(cv));
  return HttpResponse::json(200, o.to_json());
}

void
FileApi::register_routes(HttpServer& s)
{
  s.route("GET", "/api/cwd-pipelines",
          [this](const HttpRequest& r) { return h_cwd_pipelines_(r); });
  s.route("GET", "/api/fs/list",
          [this](const HttpRequest& r) { return h_list_(r); });
  s.route("GET", "/api/fs/file",
          [this](const HttpRequest& r) { return h_file_(r); });
  s.route("POST", "/api/fs/mkdir",
          [this](const HttpRequest& r) { return h_mkdir_(r); });
  s.route("POST", "/api/fs/rename",
          [this](const HttpRequest& r) { return h_rename_(r); });
  s.route("POST", "/api/fs/write",
          [this](const HttpRequest& r) { return h_write_(r); });
}

}
