// file-api.h -- /api/fs/* and /api/cwd-pipelines : the file browser's
// view of the session's (possibly sandboxed) filesystem namespace.
//
// Every path here is a CLIENT VIRTUAL path, mapped to a real host path
// by fs_resolve_ under the session's sandbox rules. No handler touches
// a caller-supplied path directly.

#ifndef WEBUI_FILE_API_H
#define WEBUI_FILE_API_H

#include "apps/web-ui/api-context.h"
#include "apps/web-ui/http-server.h"

#include <filesystem>
#include <string>

namespace vpipe::webui {

class FileApi {
public:
  explicit FileApi(ApiContext& ctx) : _ctx(ctx) {}

  void register_routes(HttpServer& s);

private:
  // {cwd, files: [...]} -- top-level .vpipeline files in the server's
  // cwd, used to populate the Load-Pipeline dialog's autocomplete.
  HttpResponse h_cwd_pipelines_(const HttpRequest&);
  // GET /api/fs/list?path= : one directory's entries for the file
  // open/save dialog, in the session's (possibly sandboxed) namespace.
  HttpResponse h_list_(const HttpRequest&);
  // GET /api/fs/file?path= : raw bytes of one file for the file-browser
  // preview (image / audio / video / text). Range-aware (206 partial)
  // and per-response size-bounded. Sandbox-confined (read-only).
  HttpResponse h_file_(const HttpRequest&);
  // POST /api/fs/mkdir {path, name} : create a directory `name` under
  // the existing virtual directory `path`. Sandbox-confined for write.
  HttpResponse h_mkdir_(const HttpRequest&);
  // POST /api/fs/rename {path, to} : rename the item at virtual `path`
  // to base name `to`, in place. Sandbox-confined for write.
  HttpResponse h_rename_(const HttpRequest&);
  // POST /api/fs/write {path, text} : write a UTF-8 text file at
  // virtual `path` (parent must exist), truncating any existing file.
  // Sandbox-confined for write. Used by the composer's Save-to-file.
  HttpResponse h_write_(const HttpRequest&);

  // Map a client virtual path to a real host path (sandbox chroot /
  // native passthrough). Returns {} and sets *err on rejection; *vpath
  // receives the cleaned virtual path. `for_write` is forwarded to
  // confine_path.
  std::filesystem::path fs_resolve_(const std::string& want,
                                    bool for_write, std::string* vpath,
                                    std::string* err) const;

  ApiContext& _ctx;
};

}

#endif
