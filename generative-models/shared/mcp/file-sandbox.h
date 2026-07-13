#ifndef VPIPE_GENERATIVE_MODELS_FILE_SANDBOX_H
#define VPIPE_GENERATIVE_MODELS_FILE_SANDBOX_H

#include "generative-models/shared/mcp/mcp-tools.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Size / count guards for one file-sandbox surface.
struct FileSandboxOptions {
  std::size_t max_read_bytes  = 256 * 1024;    // read_file cap
  std::size_t max_write_bytes = 1024 * 1024;   // write_file cap
  int         max_list_entries = 1000;         // list_files cap
};

// A directory-confined file I/O surface. Every operation takes a path
// the model supplied; the path is resolved RELATIVE to `root` (a leading
// "/" is stripped, so absolute-looking paths stay inside) and rejected
// if it would escape `root` via `..` traversal or a symlink. The root is
// canonicalized once at construction so the containment check compares
// resolved paths.
//
// This is in-process confinement (no seatbelt) -- appropriate because
// the tool code itself does the path checks; contrast the python tool,
// which runs an untrusted interpreter and needs the kernel sandbox.
class FileSandbox {
public:
  struct Entry {
    std::string   name;
    bool          is_dir = false;
    std::uint64_t size   = 0;
  };

  explicit FileSandbox(std::filesystem::path root,
                       FileSandboxOptions    opts = {});

  const std::filesystem::path& root() const noexcept { return _root; }
  const FileSandboxOptions&    options() const noexcept { return _opts; }

  // Read a text file. Returns false + *err on any failure (escape, not
  // found, directory, binary content). On success *out holds the bytes
  // (capped at max_read_bytes; *truncated set when the file was larger).
  bool read_file(const std::string& rel,
                 std::string*        out,
                 bool*               truncated,
                 std::string*        err) const;

  // Write (or append to) a text file, creating parent directories WITHIN
  // the sandbox as needed. Returns false + *err on failure.
  bool write_file(const std::string& rel,
                  const std::string& content,
                  bool               append,
                  std::uint64_t*     bytes_written,
                  std::string*       err) const;

  // List a directory's immediate entries (capped at max_list_entries;
  // *truncated set when there were more). Returns false + *err on failure.
  bool list_dir(const std::string&  rel,
                std::vector<Entry>*  out,
                bool*                truncated,
                std::string*         err) const;

private:
  // Resolve `rel` to an absolute path inside the sandbox, or return an
  // empty path + *err when it escapes / is invalid.
  std::filesystem::path resolve_(std::string rel, std::string* err) const;

  std::filesystem::path _root;
  FileSandboxOptions    _opts;
};

// Register the read_file / write_file / list_files tools, all backed by
// `sandbox` (captured, so it must outlive the registry's handlers).
void add_file_tools(McpToolRegistry&              reg,
                    std::shared_ptr<FileSandbox>  sandbox);

}

#endif
