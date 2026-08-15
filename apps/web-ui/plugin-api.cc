#include "apps/web-ui/plugin-api.h"

#include "apps/web-ui/api-common.h"
#include "common/flex-data.h"
#include "common/plugins-root.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/stage-registry.h"
#include "plugin/plugin-manager.h"

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

namespace vpipe::webui {

namespace fs = std::filesystem;

namespace {

// Does this file look like a loadable plugin?
//
// ON APPLE, ACCEPT BOTH .so AND .dylib. A plugin is built as a CMake
// MODULE library, and MODULE targets get `.so` on macOS -- only SHARED
// targets get `.dylib`. Every plugin in this tree is therefore a .so on
// the platform whose native suffix is .dylib: filtering to .dylib here
// listed nothing at all, which is how this was found. dlopen does not
// care about the suffix, so neither does this.
bool
is_plugin_file_(const fs::path& p)
{
  const std::string ext = p.extension().string();
#if defined(_WIN32)
  return ext == ".dll";
#elif defined(__APPLE__)
  return ext == ".so" || ext == ".dylib";
#else
  return ext == ".so";
#endif
}

}  // namespace

HttpResponse
PluginApi::h_list_(const HttpRequest&)
{
  auto& pm = PluginManager::get();
  const auto recs = pm.records();

  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  const fs::path root = plugins_root();
  oo.insert("root", fstr(root.string()));

  std::error_code ec;
  const bool have_root = fs::is_directory(root, ec) && !ec;
  oo.insert("root_exists", FlexData::make_bool(have_root));

  // How many stage types each plugin contributed, from the registry's
  // own provenance -- not from anything the plugin claimed. A plugin
  // that registered nothing shows 0 rather than looking equivalent to
  // one that registered ten.
  auto stage_count = [](const std::string& name) {
    std::size_t n = 0;
    for (auto& [id, type] : StageRegistry::get().all()) {
      (void)id;
      if (StageRegistry::get().origin(type) == name) { ++n; }
    }
    return n;
  };

  // ---- what is loaded ------------------------------------------------
  FlexData loaded = FlexData::make_array();
  auto la = loaded.as_array();
  for (const auto& r : recs) {
    FlexData e = FlexData::make_object();
    auto eo = e.as_object();
    eo.insert("name",        fstr(r.name));
    eo.insert("path",        fstr(r.path));
    eo.insert("version",     fstr(r.version));
    eo.insert("vendor",      fstr(r.vendor));
    eo.insert("license",     fstr(r.license));
    eo.insert("description", fstr(r.description));
    eo.insert("loaded",      FlexData::make_bool(true));
    eo.insert("enabled",     FlexData::make_bool(r.enabled));
    eo.insert("stage_count",
              FlexData::make_uint((std::uint64_t)stage_count(r.name)));
    la.push_back(std::move(e));
  }
  oo.insert("loaded", std::move(loaded));

  // ---- what is merely PRESENT ----------------------------------------
  //
  // A listing, not an instruction: files here are candidates the user
  // may choose to load. Nothing is loaded by virtue of sitting in the
  // directory -- see common/plugins-root.h.
  FlexData avail = FlexData::make_array();
  auto aa = avail.as_array();
  if (have_root) {
    std::vector<fs::path> files;
    for (const auto& de : fs::directory_iterator(root, ec)) {
      if (ec) { break; }
      if (!de.is_regular_file() && !de.is_symlink()) { continue; }
      if (!is_plugin_file_(de.path())) { continue; }
      files.push_back(de.path());
    }
    std::sort(files.begin(), files.end());
    for (const auto& f : files) {
      FlexData e = FlexData::make_object();
      auto eo = e.as_object();
      eo.insert("file", fstr(f.filename().string()));
      eo.insert("path", fstr(f.string()));
      eo.insert("loaded",
                FlexData::make_bool(pm.is_loaded_path(f.string())));
      aa.push_back(std::move(e));
    }
  }
  oo.insert("available", std::move(avail));

  // Stated by the API rather than assumed by the client: there is no
  // unload, and `enabled` is not one. See plugin-api.h.
  oo.insert("can_unload", FlexData::make_bool(false));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
PluginApi::h_load_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::json(400, R"({"error":"expected a JSON object"})");
  }
  const auto b = body->as_object();
  const std::string path =
      trim(std::string(b.contains("path") ? b.at("path").as_string("") : ""));
  if (path.empty()) {
    return HttpResponse::json(400, R"({"error":"path is required"})");
  }

  // Confined to the plugins root. The endpoint is reachable by anyone who
  // can reach the UI, and loading a dylib is arbitrary code execution in
  // this process -- so the path a request may name is the one the
  // convention already sanctions, not anywhere on disk. A deployment that
  // wants a plugin from elsewhere passes --plugin at startup, which is a
  // decision taken by whoever runs the process.
  std::error_code ec;
  const fs::path root = fs::weakly_canonical(plugins_root(), ec);
  const fs::path want = fs::weakly_canonical(fs::path(path), ec);
  if (ec) {
    return HttpResponse::json(400, R"({"error":"unreadable path"})");
  }
  const std::string rs = root.string();
  const std::string ws = want.string();
  const bool inside =
      ws.size() > rs.size() && ws.compare(0, rs.size(), rs) == 0 &&
      (ws[rs.size()] == fs::path::preferred_separator);
  if (!inside) {
    return HttpResponse::json(
        403,
        R"({"error":"only plugins under the plugins root may be loaded here; )"
        R"(use --plugin at startup for anything else"})");
  }
  if (!is_plugin_file_(want)) {
    return HttpResponse::json(400, R"({"error":"not a plugin dylib"})");
  }

  std::lock_guard<std::mutex> lk(_ctx.mu);
  const bool ok = PluginManager::get().load(_ctx.sctx, ws);
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("ok", FlexData::make_bool(ok));
  oo.insert("path", fstr(ws));
  if (!ok) {
    // PluginManager already logged the reason (ABI mismatch, missing
    // entry point, throwing register); point the operator at it rather
    // than inventing a second, vaguer explanation here.
    oo.insert("error",
              fstr("load failed -- see the log for the reason (ABI "
                   "mismatch, missing entry points, or a throwing "
                   "vpipe_plugin_register)"));
  }
  return HttpResponse::json(ok ? 200 : 400, o.to_json());
}

HttpResponse
PluginApi::h_enabled_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::json(400, R"({"error":"expected a JSON object"})");
  }
  const auto b = body->as_object();
  const std::string name =
      trim(std::string(b.contains("name") ? b.at("name").as_string("") : ""));
  if (name.empty()) {
    return HttpResponse::json(400, R"({"error":"name is required"})");
  }
  const bool on = b.contains("on") ? b.at("on").as_bool(true) : true;

  std::lock_guard<std::mutex> lk(_ctx.mu);
  auto& pm = PluginManager::get();
  bool known = false;
  for (const auto& r : pm.records()) {
    if (r.name == name) { known = true; break; }
  }
  if (!known) {
    return HttpResponse::json(404, R"({"error":"no such loaded plugin"})");
  }
  pm.set_enabled(name, on);
  if (_ctx.sctx != nullptr) {
    _ctx.sctx->info(fmt(
        "plugin '{}' {} -- its stages are {} the composer. The plugin "
        "stays loaded either way; vpipe does not unload plugins.",
        name, on ? "enabled" : "disabled",
        on ? "offered in" : "withheld from"));
  }
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("name", fstr(name));
  oo.insert("enabled", FlexData::make_bool(on));
  return HttpResponse::json(200, o.to_json());
}

void
PluginApi::register_routes(HttpServer& s)
{
  s.route("GET", "/api/plugins",
          [this](const HttpRequest& r) { return h_list_(r); });
  s.route("POST", "/api/plugins/load",
          [this](const HttpRequest& r) { return h_load_(r); });
  s.route("POST", "/api/plugins/enabled",
          [this](const HttpRequest& r) { return h_enabled_(r); });
}

}  // namespace vpipe::webui
