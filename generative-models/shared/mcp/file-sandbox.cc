#include "generative-models/shared/mcp/file-sandbox.h"

#include "common/flex-data.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace fs = std::filesystem;

FileSandbox::FileSandbox(fs::path root, FileSandboxOptions opts)
  : _opts(opts)
{
  // Canonicalize the root once so containment compares resolved paths
  // (resolves symlinks and the /var -> /private/var alias). Fall back to
  // a lexically-normal form when the dir doesn't exist yet.
  std::error_code ec;
  fs::path c = fs::weakly_canonical(root, ec);
  _root = (ec || c.empty()) ? root.lexically_normal() : c;
}

fs::path
FileSandbox::resolve_(string rel, string* err) const
{
  // Treat every path as relative to the root: strip leading separators
  // so an absolute-looking "/etc/passwd" lands at "<root>/etc/passwd".
  while (!rel.empty() && (rel.front() == '/' || rel.front() == '\\')) {
    rel.erase(0, 1);
  }
  if (rel.empty()) {
    if (err) { *err = "empty path"; }
    return {};
  }
  std::error_code ec;
  const fs::path combined = _root / rel;
  // weakly_canonical resolves symlinks in the existing prefix AND
  // normalizes `..` in the whole path, so it catches both traversal and
  // symlink escapes in one shot. Fall back to a lexical normal form when
  // the target doesn't exist yet and canonicalization can't proceed.
  fs::path resolved = fs::weakly_canonical(combined, ec);
  if (ec || resolved.empty()) {
    resolved = combined.lexically_normal();
  }
  const fs::path relp = resolved.lexically_relative(_root);
  const string s = relp.generic_string();
  if (relp.empty() || s == ".." || s.rfind("../", 0) == 0) {
    if (err) { *err = "path escapes the sandbox"; }
    return {};
  }
  return resolved;
}

bool
FileSandbox::read_file(const string& rel,
                       string*       out,
                       bool*         truncated,
                       string*       err) const
{
  if (truncated) { *truncated = false; }
  const fs::path p = resolve_(rel, err);
  if (p.empty()) {
    return false;
  }
  std::error_code ec;
  const auto st = fs::status(p, ec);
  if (ec || !fs::exists(st)) {
    if (err) { *err = "no such file"; }
    return false;
  }
  if (fs::is_directory(st)) {
    if (err) { *err = "path is a directory"; }
    return false;
  }
  ifstream f(p, ios::binary);
  if (!f) {
    if (err) { *err = "cannot open file"; }
    return false;
  }
  // Read up to the cap + 1 byte so we can flag truncation.
  string data;
  data.resize(_opts.max_read_bytes + 1);
  f.read(data.data(), static_cast<std::streamsize>(data.size()));
  const size_t got = static_cast<size_t>(f.gcount());
  data.resize(got);
  if (got > _opts.max_read_bytes) {
    data.resize(_opts.max_read_bytes);
    if (truncated) { *truncated = true; }
  }
  // Refuse binary content: a NUL byte would make the JSON result invalid
  // and is a sign the model asked for the wrong thing.
  if (data.find('\0') != string::npos) {
    if (err) { *err = "binary file (contains NUL); not a text file"; }
    return false;
  }
  if (out) { *out = std::move(data); }
  return true;
}

bool
FileSandbox::write_file(const string&  rel,
                        const string&  content,
                        bool           append,
                        std::uint64_t* bytes_written,
                        string*        err) const
{
  if (content.size() > _opts.max_write_bytes) {
    if (err) { *err = "content exceeds the write size limit"; }
    return false;
  }
  const fs::path p = resolve_(rel, err);
  if (p.empty()) {
    return false;
  }
  std::error_code ec;
  if (fs::is_directory(p, ec)) {
    if (err) { *err = "path is a directory"; }
    return false;
  }
  // Parent dirs are inside the sandbox (resolve_ confined the whole
  // path), so creating them can't escape.
  const fs::path parent = p.parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent, ec);
  }
  ofstream f(p, append ? (ios::binary | ios::app)
                       : (ios::binary | ios::trunc));
  if (!f) {
    if (err) { *err = "cannot open file for writing"; }
    return false;
  }
  f.write(content.data(), static_cast<std::streamsize>(content.size()));
  f.flush();
  if (!f) {
    if (err) { *err = "write failed"; }
    return false;
  }
  if (bytes_written) {
    *bytes_written = static_cast<std::uint64_t>(content.size());
  }
  return true;
}

bool
FileSandbox::list_dir(const string&       rel,
                      vector<Entry>*      out,
                      bool*               truncated,
                      string*             err) const
{
  if (truncated) { *truncated = false; }
  const fs::path p = resolve_(rel, err);
  if (p.empty()) {
    return false;
  }
  std::error_code ec;
  if (!fs::is_directory(p, ec)) {
    if (err) { *err = "not a directory"; }
    return false;
  }
  fs::directory_iterator it(p, fs::directory_options::skip_permission_denied,
                            ec);
  if (ec) {
    if (err) { *err = "cannot list directory"; }
    return false;
  }
  for (const auto& de : it) {
    if (out && static_cast<int>(out->size()) >= _opts.max_list_entries) {
      if (truncated) { *truncated = true; }
      break;
    }
    Entry e;
    e.name   = de.path().filename().string();
    std::error_code ec2;
    e.is_dir = de.is_directory(ec2);
    if (!e.is_dir) {
      std::error_code ec3;
      const auto sz = de.file_size(ec3);
      e.size = ec3 ? 0 : static_cast<std::uint64_t>(sz);
    }
    if (out) { out->push_back(std::move(e)); }
  }
  return true;
}

namespace {

// Pull a string field from a parsed JSON args object; `def` on absence.
string
arg_str_(const FlexData& args, const char* key, const string& def)
{
  if (!args.is_object()) {
    return def;
  }
  auto o = args.as_object();
  if (!o.contains(key)) {
    return def;
  }
  FlexData v = o.at(key);
  return v.is_string() ? string(v.get_string()) : def;
}

bool
arg_bool_(const FlexData& args, const char* key, bool def)
{
  if (!args.is_object()) {
    return def;
  }
  auto o = args.as_object();
  return o.contains(key) ? o.at(key).as_bool(def) : def;
}

FlexData
error_obj_(const string& msg)
{
  FlexData fd = FlexData::make_object();
  fd.as_object().insert("error", FlexData::make_string(msg));
  return fd;
}

}  // namespace

void
add_file_tools(McpToolRegistry& reg, std::shared_ptr<FileSandbox> sandbox)
{
  if (!sandbox) {
    return;
  }

  reg.add(McpTool{
      .name = "read_file",
      .description =
          "Read a UTF-8 text file from your private workspace. `path` is "
          "relative to the workspace root; it cannot escape it.",
      .parameters_json =
          "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":"
          "\"string\",\"description\":\"workspace-relative file path\"}},"
          "\"required\":[\"path\"]}",
      .handler = [sandbox](const string& args_json) -> string {
        string path;
        try {
          path = arg_str_(FlexData::from_json(args_json), "path", "");
        } catch (const std::exception&) {}
        if (path.empty()) {
          return error_obj_("missing 'path' argument").to_json(false);
        }
        string data, err;
        bool truncated = false;
        if (!sandbox->read_file(path, &data, &truncated, &err)) {
          return error_obj_(err).to_json(false);
        }
        FlexData fd = FlexData::make_object();
        auto o = fd.as_object();
        o.insert("path", FlexData::make_string(path));
        o.insert("content", FlexData::make_string(data));
        o.insert("bytes",
                 FlexData::make_int(static_cast<int64_t>(data.size())));
        if (truncated) {
          o.insert("truncated", FlexData::make_bool(true));
        }
        return fd.to_json(false);
      },
  });

  reg.add(McpTool{
      .name = "write_file",
      .description =
          "Write (or append to) a text file in your private workspace, "
          "creating parent folders as needed. `path` is relative to the "
          "workspace root; it cannot escape it.",
      .parameters_json =
          "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":"
          "\"string\",\"description\":\"workspace-relative file path\"},"
          "\"content\":{\"type\":\"string\",\"description\":\"text to "
          "write\"},\"append\":{\"type\":\"boolean\",\"description\":"
          "\"append instead of overwrite (default false)\"}},"
          "\"required\":[\"path\",\"content\"]}",
      .handler = [sandbox](const string& args_json) -> string {
        FlexData args;
        try {
          args = FlexData::from_json(args_json);
        } catch (const std::exception&) {
          return error_obj_("invalid arguments JSON").to_json(false);
        }
        const string path = arg_str_(args, "path", "");
        if (path.empty()) {
          return error_obj_("missing 'path' argument").to_json(false);
        }
        if (!args.is_object() || !args.as_object().contains("content")) {
          return error_obj_("missing 'content' argument").to_json(false);
        }
        const string content = arg_str_(args, "content", "");
        const bool append = arg_bool_(args, "append", false);
        std::uint64_t written = 0;
        string err;
        if (!sandbox->write_file(path, content, append, &written, &err)) {
          return error_obj_(err).to_json(false);
        }
        FlexData fd = FlexData::make_object();
        auto o = fd.as_object();
        o.insert("ok", FlexData::make_bool(true));
        o.insert("path", FlexData::make_string(path));
        o.insert("bytes_written",
                 FlexData::make_int(static_cast<int64_t>(written)));
        return fd.to_json(false);
      },
  });

  reg.add(McpTool{
      .name = "list_files",
      .description =
          "List the entries of a directory in your private workspace. "
          "`path` is relative to the workspace root (default the root "
          "itself); it cannot escape it.",
      .parameters_json =
          "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":"
          "\"string\",\"description\":\"workspace-relative directory "
          "(default '.')\"}},\"required\":[]}",
      .handler = [sandbox](const string& args_json) -> string {
        string path = ".";
        try {
          path = arg_str_(FlexData::from_json(args_json), "path", ".");
        } catch (const std::exception&) {}
        if (path.empty()) {
          path = ".";
        }
        vector<FileSandbox::Entry> entries;
        bool truncated = false;
        string err;
        if (!sandbox->list_dir(path, &entries, &truncated, &err)) {
          return error_obj_(err).to_json(false);
        }
        FlexData fd = FlexData::make_object();
        auto o = fd.as_object();
        o.insert("path", FlexData::make_string(path));
        FlexData arr = FlexData::make_array();
        auto av = arr.as_array();
        for (const auto& e : entries) {
          FlexData ent = FlexData::make_object();
          auto eo = ent.as_object();
          eo.insert("name", FlexData::make_string(e.name));
          eo.insert("type",
                    FlexData::make_string(e.is_dir ? "dir" : "file"));
          if (!e.is_dir) {
            eo.insert("size",
                      FlexData::make_int(static_cast<int64_t>(e.size)));
          }
          av.push_back(std::move(ent));
        }
        o.insert("entries", std::move(arr));
        if (truncated) {
          o.insert("truncated", FlexData::make_bool(true));
        }
        return fd.to_json(false);
      },
  });
}

}
