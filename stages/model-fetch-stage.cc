#include "stages/model-fetch-stage.h"
#include "stages/model-catalog.h"
#include "stages/qwen-asr-tokenizer.h"

#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <curl/curl.h>
#include <spawn.h>
#include <sys/wait.h>

using namespace std;
namespace fs = std::filesystem;

namespace vpipe {

namespace {

// One-shot global curl init (idempotent; rest-client uses the same
// pattern). Safe to call from any thread.
void
ensure_curl_global_init()
{
  static std::once_flag once;
  std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

void
trim_(string& s)
{
  auto ns = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), ns));
  s.erase(std::find_if(s.rbegin(), s.rend(), ns).base(), s.end());
}

// Expand a leading "~/" to $HOME (getline doesn't go through a shell).
string
expand_user_(const string& p)
{
  if (p.size() >= 2 && p[0] == '~' && p[1] == '/') {
    if (const char* home = std::getenv("HOME")) {
      return string(home) + p.substr(1);
    }
  }
  return p;
}

string
human_bytes_(uint64_t n)
{
  const char* u[] = { "B", "KB", "MB", "GB", "TB" };
  double v = static_cast<double>(n);
  int i = 0;
  while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
  return fmt("{:.1f} {}", v, u[i])();
}

// ---- interactive selection --------------------------------------------

// Outcome of one interactive level pick.
enum class SelectResult {
  Ok,       // a concrete option was chosen (returned in `out`)
  Auto,     // a single option was auto-selected (in `out`); no prompt shown
  Back,     // the user asked to step back to the previous level
  Aborted,  // input closed / canceled
};

// Prompt the user to pick one of `options` by index; the chosen string is
// returned via `out`. Auto-selects (-> Auto) when there is exactly one
// option. When `allow_back` is set the user may enter 'b' to step back a
// level (-> Back). Returns Aborted on Eof / Canceled.
SelectResult
select_from_(const SessionContextIntf* s, const string& title,
             const vector<string>& options,
             const function<bool()>& cancel, bool allow_back, string& out)
{
  if (options.empty()) {
    return SelectResult::Aborted;
  }
  if (options.size() == 1) {
    out = options[0];
    s->info(fmt("{}: {}", title, out));
    return SelectResult::Auto;
  }
  const VpipeFormat prompt = allow_back
      ? fmt("Select [0-{}] (b=back): ", options.size() - 1)
      : fmt("Select [0-{}]: ", options.size() - 1);
  for (;;) {
    s->info(fmt("{}", title));
    for (size_t i = 0; i < options.size(); ++i) {
      s->info(fmt("  [{}] {}", i, options[i]));
    }
    string line;
    if (s->getline(prompt, line, cancel) != UiInputStatus::Ok) {
      return SelectResult::Aborted;
    }
    trim_(line);
    if (allow_back && (line == "b" || line == "B")) {
      return SelectResult::Back;
    }
    try {
      size_t idx = static_cast<size_t>(std::stoul(line));
      if (idx < options.size()) {
        out = options[idx];
        return SelectResult::Ok;
      }
    } catch (...) {
    }
    s->info(fmt("Invalid selection '{}'; try again.", line));
  }
}

// ---- libcurl GET ------------------------------------------------------

size_t
write_to_string_(char* p, size_t s, size_t n, void* u)
{
  static_cast<string*>(u)->append(p, s * n);
  return s * n;
}

size_t
write_to_ofstream_(char* p, size_t s, size_t n, void* u)
{
  auto* o = static_cast<std::ofstream*>(u);
  o->write(p, static_cast<std::streamsize>(s * n));
  return o->good() ? s * n : 0;
}

// ---- download progress bar --------------------------------------------

// Files at least this large get a live progress bar.
constexpr curl_off_t kBigFileBytes = curl_off_t{256} * 1024 * 1024;

// Render a fixed-width bar, e.g. "[######------------------] 27%  ...".
// Padded to a stable width so a shorter redraw fully overwrites a longer
// one in a raw terminal (the web-ui handler clears the line itself).
string
render_bar_(int pct, curl_off_t now, curl_off_t total)
{
  if (pct < 0) { pct = 0; }
  if (pct > 100) { pct = 100; }
  constexpr int W = 24;
  const int filled = pct * W / 100;
  string bar = "[";
  for (int i = 0; i < W; ++i) { bar += (i < filled ? '#' : '-'); }
  bar += "] " + std::to_string(pct) + "%  "
       + human_bytes_(static_cast<uint64_t>(now   < 0 ? 0 : now)) + " / "
       + human_bytes_(static_cast<uint64_t>(total < 0 ? 0 : total));
  while (bar.size() < 56) { bar += ' '; }
  return bar;
}

// Per-download progress state handed to the libcurl xferinfo callback.
struct ProgressCtx {
  vpipe::UiTextStream* stream   = nullptr;
  int                  last_pct = -1;     // throttle: redraw on % change
  bool                 drawn    = false;  // emitted at least one frame
  curl_off_t           total    = 0;
  // Poll predicate: when set and it returns true, the xferinfo callback
  // aborts the transfer mid-flight (-> CURLE_ABORTED_BY_CALLBACK).
  const std::function<bool()>* cancel = nullptr;
};

// libcurl CURLOPT_XFERINFOFUNCTION. Redraws the bar (carriage-return +
// overwrite) only when the integer percentage changes, so an N-hundred-
// MB file produces ~100 updates regardless of callback frequency.
int
progress_cb_(void* p, curl_off_t dltotal, curl_off_t dlnow,
             curl_off_t /*ultotal*/, curl_off_t /*ulnow*/)
{
  auto* c = static_cast<ProgressCtx*>(p);
  if (c && c->cancel && (*c->cancel)()) { return 1; }   // abort transfer
  if (!c || !c->stream || dltotal <= 0) {
    return 0;
  }
  c->total = dltotal;
  const int pct = static_cast<int>((dlnow * 100) / dltotal);
  if (pct == c->last_pct) {
    return 0;
  }
  c->last_pct = pct;
  string line = c->drawn ? "\r" : "";   // \r overwrites the prior frame
  line += render_bar_(pct, dlnow, dltotal);
  c->stream->write(line);
  c->drawn = true;
  return 0;
}

// Shared easy-handle perform. `wcb`/`wdata` sink the body. Fills
// `*http_status` with the response code. When `progress` is non-null the
// transfer runs with the xferinfo callback (it draws the bar when a stream
// is set and/or polls the cancel predicate). Returns the CURLcode.
CURLcode
curl_perform_(const string& url, const string& token, bool verify_tls,
              long timeout_s, size_t (*wcb)(char*, size_t, size_t, void*),
              void* wdata, long* http_status, ProgressCtx* progress)
{
  CURL* c = curl_easy_init();
  if (!c) {
    return CURLE_FAILED_INIT;
  }
  struct curl_slist* hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, "User-Agent: vpipe-model-fetch/1");
  string auth;
  if (!token.empty()) {
    auth = "Authorization: Bearer " + token;
    hdrs = curl_slist_append(hdrs, auth.c_str());
  }
  curl_easy_setopt(c, CURLOPT_URL, url.c_str());
  curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdrs);
  curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
  if (progress) {
    // Attach the xferinfo callback whenever a ProgressCtx is present -- it
    // carries the bar stream and/or the cancel predicate (the callback
    // no-ops the bar when no stream is set).
    curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, &progress_cb_);
    curl_easy_setopt(c, CURLOPT_XFERINFODATA, progress);
  } else {
    curl_easy_setopt(c, CURLOPT_NOPROGRESS, 1L);
  }
  curl_easy_setopt(c, CURLOPT_SSL_VERIFYPEER, verify_tls ? 1L : 0L);
  curl_easy_setopt(c, CURLOPT_SSL_VERIFYHOST, verify_tls ? 2L : 0L);
  curl_easy_setopt(c, CURLOPT_TIMEOUT, timeout_s);
  curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 30L);
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, wcb);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, wdata);
  CURLcode rc = curl_easy_perform(c);
  if (http_status) {
    *http_status = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, http_status);
  }
  curl_slist_free_all(hdrs);
  curl_easy_cleanup(c);
  return rc;
}

// GET into a string. `err` set + false on transport / HTTP error;
// `*status` carries the HTTP code (so the caller can detect 401/403).
bool
http_get_(const string& url, const string& token, bool verify_tls,
          long timeout_s, string& out, long& status, string& err)
{
  out.clear();
  CURLcode rc = curl_perform_(url, token, verify_tls, timeout_s,
                              &write_to_string_, &out, &status, nullptr);
  if (rc != CURLE_OK) {
    err = curl_easy_strerror(rc);
    return false;
  }
  if (status < 200 || status >= 300) {
    err = fmt("HTTP {}", status)();
    return false;
  }
  return true;
}

// GET streaming to `dest` (parent dirs created). `*status` carries the
// HTTP code; `err` set + false on failure. When `progress` is non-null,
// a live progress bar is rendered to it for the duration of the
// transfer. When `cancel` is non-null it is polled mid-transfer; if it
// returns true the transfer aborts (curl error -> the partial dest is
// removed).
bool
http_download_(const string& url, const string& token, bool verify_tls,
               long timeout_s, const fs::path& dest, long& status,
               string& err, vpipe::UiTextStream* progress,
               const std::function<bool()>* cancel = nullptr)
{
  std::error_code ec;
  fs::create_directories(dest.parent_path(), ec);
  std::ofstream ofs(dest, std::ios::binary | std::ios::trunc);
  if (!ofs) {
    err = fmt("cannot open '{}' for writing", dest.string())();
    return false;
  }
  ProgressCtx pctx;
  pctx.stream = progress;
  pctx.cancel = cancel;
  CURLcode rc = curl_perform_(url, token, verify_tls, timeout_s,
                              &write_to_ofstream_, &ofs, &status,
                              (progress || cancel) ? &pctx : nullptr);
  ofs.close();
  if (rc != CURLE_OK) {
    err = curl_easy_strerror(rc);
    fs::remove(dest, ec);
    return false;
  }
  if (status < 200 || status >= 300) {
    err = fmt("HTTP {}", status)();
    fs::remove(dest, ec);
    return false;
  }
  // Snap the bar to a clean 100% frame (some servers omit the final
  // callback at dlnow == dltotal).
  if (progress && pctx.drawn) {
    progress->write("\r" + render_bar_(100, pctx.total, pctx.total));
  }
  return true;
}

// ---- archive extraction -----------------------------------------------

// Unpack `archive` into `dest_dir` with the system tar (no shell, so paths
// with spaces are safe). bsdtar (/usr/bin/tar on macOS) and GNU tar both
// accept `-x -f <archive> -C <dir>` and autodetect compression. Returns
// true on a clean exit(0); `err` carries a message otherwise.
bool
extract_tar_(const fs::path& archive, const fs::path& dest_dir, string& err)
{
  std::error_code ec;
  fs::create_directories(dest_dir, ec);
  const string a = archive.string();
  const string d = dest_dir.string();
  // posix_spawn wants a mutable argv; the strings outlive the call.
  const char* argv[] = { "tar", "-x", "-f", a.c_str(), "-C", d.c_str(),
                         nullptr };
  // A minimal environment is enough for tar and sidesteps the macOS dylib
  // `environ` symbol restriction.
  const char* envp[] = { "PATH=/usr/bin:/bin:/usr/local/bin", nullptr };
  pid_t pid = 0;
  int rc = posix_spawn(&pid, "/usr/bin/tar", nullptr, nullptr,
                       const_cast<char* const*>(argv),
                       const_cast<char* const*>(envp));
  if (rc != 0) {
    // Fall back to a PATH search (non-macOS layouts).
    rc = posix_spawnp(&pid, "tar", nullptr, nullptr,
                      const_cast<char* const*>(argv),
                      const_cast<char* const*>(envp));
  }
  if (rc != 0) {
    err = fmt("posix_spawn(tar) failed: {}", std::strerror(rc))();
    return false;
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    err = "waitpid(tar) failed";
    return false;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    err = fmt("tar exited with status {}",
              WIFEXITED(status) ? WEXITSTATUS(status) : -1)();
    return false;
  }
  return true;
}

// Find the single top-level `*.mlpackage` directory directly under `dir`
// (the vpipe-supplement archive layout). Empty path when none is present.
fs::path
find_mlpackage_(const fs::path& dir)
{
  std::error_code ec;
  if (!fs::exists(dir, ec)) {
    return {};
  }
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (ec) { break; }
    const fs::path& p = e.path();
    if (p.extension() == ".mlpackage") {
      return p;
    }
  }
  return {};
}

}  // namespace

ModelFetchStage::ModelFetchStage(const SessionContextIntf* s,
                                 string                    id,
                                 vector<InEdge>            iports,
                                 FlexData                  config)
  : TypedStage<ModelFetchStage>(s, std::move(id), std::move(iports),
                                std::move(config))
{
  ensure_curl_global_init();

  _base_path           = attr_str("base_path");
  _db_name             = attr_str("db_name");
  _model_path          = attr_str("model_path");
  _hf_token            = attr_str("hf_token");
  _overwrite_existing  = attr_bool("overwrite_existing");
  _prepare_tokenizer   = attr_bool("prepare_tokenizer");
  _skip_existing_files = attr_bool("skip_existing_files");
  _verify_tls          = attr_bool("verify_tls");
  _timeout_seconds     = static_cast<unsigned>(attr_uint("timeout_seconds"));

  if (_db_name.empty()) {
    _db_name = "models";
  }

  allocate_oports(spec().oports.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "base_path", .type = ConfigType::String,
   .doc = "download root; empty -> ./models. Files land under "
          "<base>/<owner>/<repo>"},
  {.key = "db_name", .type = ConfigType::String,
   .doc = "LMDB sub-db the model is registered in", .def_str = "models"},
  {.key = "model_path", .type = ConfigType::String,
   .doc = "non-interactive: a direct 'owner/repo' (or full URL); empty "
          "-> prompt / browse the catalogue"},
  {.key = "hf_token", .type = ConfigType::String,
   .doc = "HuggingFace token for gated/private repos; empty -> $HF_TOKEN "
          "(prompted via getpasswd if a download is gated)"},
  {.key = "overwrite_existing", .type = ConfigType::Bool,
   .doc = "re-download + re-register a model already in the registry",
   .def_bool = true},
  {.key = "prepare_tokenizer", .type = ConfigType::Bool,
   .doc = "synthesize tokenizer.json natively for Qwen3-ASR (no "
          "transformers needed)", .def_bool = true},
  {.key = "skip_existing_files", .type = ConfigType::Bool,
   .doc = "skip files already on disk whose size matches the repo",
   .def_bool = true},
  {.key = "verify_tls", .type = ConfigType::Bool,
   .doc = "enforce TLS certificate validation", .def_bool = true},
  {.key = "timeout_seconds", .type = ConfigType::Uint,
   .doc = "per-file network timeout (large shards need headroom)",
   .def_uint = 1800},
};
// One trigger iport (optional, any beat type) + one summary oport shared
// by all four "preparation" stages so they can be cascaded into a recipe
// (each stage's summary triggers the next) and/or dumped to a save-text
// report. See the ports doc below.
const PortSpec kIports[] = {
  {.name = "trigger",
   .doc  = "optional pacing trigger (any beat type); when wired, the work "
           "waits for one beat before running -- lets these preparation "
           "stages cascade into a recipe",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "summary",
   .doc  = "FlexData summary of the completed work; its `text` field "
           "renders a report via save-text, and the beat also triggers "
           "the next stage in a recipe",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "model-fetch",
  .doc       = "Interactive one-shot: identify a model (browse the "
               "internal HuggingFace catalogue or type a path), download "
               "it under a base path, synthesize the Qwen3-ASR "
               "tokenizer.json natively, and register it in LMDB sub-db "
               "'models' keyed by the huggingface.co path. Optional "
               "trigger in / summary out.",
  .display_name = "Model Fetch",
  .category  = StageCategory::Preparation,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
ModelFetchStage::spec() const noexcept
{
  return kSpec;
}

Job
ModelFetchStage::process(RuntimeContext& ctx)
{
  // Stable object (not a temporary) so its address can be threaded into the
  // libcurl xferinfo callback to abort an in-flight transfer mid-download.
  const std::function<bool()> cancel = [&ctx] {
    return ctx.stop_requested();
  };
  const SessionContextIntf* s = session();

  // Optional trigger: when the iport is wired, wait for one beat before
  // starting so this stage can cascade in a preparation recipe. Any beat
  // type works (payload ignored); upstream EOS -> nothing to do.
  if (ctx.iport_connected(0)) {
    auto trig = co_await ctx.read(0);
    if (!trig) {
      ctx.signal_done();
      co_return;
    }
  }

  // -------- 1. Identify the model --------------------------------------
  // Precedence: configured model_path > direct entry > catalogue browse.
  string hf_path;
  const ModelCatalogEntry* entry = nullptr;

  if (!_model_path.empty()) {
    hf_path = normalize_hf_path(_model_path);
    if (hf_path.empty()) {
      s->error(fmt("ModelFetchStage('{}'): invalid model_path '{}'",
                   this->id(), _model_path));
    }
  } else {
    string direct;
    if (s->getline(
            fmt("HuggingFace path (owner/repo or URL), or Enter to "
                "browse the catalogue: "),
            direct, cancel) != UiInputStatus::Ok) {
      s->error(fmt("ModelFetchStage('{}'): input closed", this->id()));
    }
    trim_(direct);
    if (!direct.empty()) {
      hf_path = normalize_hf_path(direct);
      if (hf_path.empty()) {
        s->error(fmt("ModelFetchStage('{}'): could not parse '{}' as "
                     "owner/repo", this->id(), direct));
      }
    } else {
      // Catalogue drill-down: family -> version -> param -> variant. At
      // any prompt the user may enter 'b' to step back to the previous
      // level. `dir` remembers whether we reached the current level going
      // forward (+1) or backward (-1) so an auto-selected (single-option)
      // level stays transparent to back-navigation instead of trapping it.
      string family, version, param, variant;
      bool aborted = false;
      for (int level = 0, dir = 1; level < 4; ) {
        string title;
        vector<string> opts;
        string* out = nullptr;
        switch (level) {
          case 0:
            title = "Model family"; opts = catalog_families();
            out = &family; break;
          case 1:
            title = "Version"; opts = catalog_versions(family);
            out = &version; break;
          case 2:
            title = "Parameter class";
            opts = catalog_param_classes(family, version);
            out = &param; break;
          default:
            title = "Variant";
            opts = catalog_variants(family, version, param);
            out = &variant; break;
        }
        switch (select_from_(s, title, opts, cancel, level > 0, *out)) {
          case SelectResult::Ok:
            dir = 1; ++level; break;
          case SelectResult::Auto:
            // A forced single-option level: keep moving the way we came,
            // bouncing forward if there is nothing before level 0.
            if (dir < 0 && level == 0) { dir = 1; }
            level += dir; break;
          case SelectResult::Back:
            dir = -1; --level; break;
          case SelectResult::Aborted:
            aborted = true; break;
        }
        if (aborted) { break; }
      }
      if (aborted) {
        s->error(fmt("ModelFetchStage('{}'): selection aborted",
                     this->id()));
      }
      entry = catalog_find(family, version, param, variant);
      if (!entry) {
        s->error(fmt("ModelFetchStage('{}'): no catalogue entry for "
                     "selection", this->id()));
      }
      hf_path = entry->hf_path;
    }
  }
  if (!entry) {
    entry = catalog_by_path(hf_path);   // enrich a typed path if known
  }

  // Registration key: a catalogue `name` (so several archives sharing one
  // hf_path repo register under distinct keys) else the hf_path itself.
  const string reg_key =
      (entry && !entry->name.empty()) ? entry->name : hf_path;

  // -------- 2. Resolve the download location ---------------------------
  string base_in = _base_path;
  if (base_in.empty()) {
    const fs::path def = fs::current_path() / "models";
    string line;
    if (s->getline(fmt("Base download path [default {}]: ", def.string()),
                   line, cancel) == UiInputStatus::Ok) {
      trim_(line);
      base_in = line;
    }
    if (base_in.empty()) {
      base_in = def.string();
    }
  }
  const fs::path base     = fs::path(expand_user_(base_in));
  const fs::path local_dir = base / hf_path;   // <base>/<owner>/<repo>

  // -------- 3. Registry pre-check --------------------------------------
  LmdbEnv* env = s->lmdb_env();
  if (!env) {
    s->error(fmt("ModelFetchStage('{}'): session lmdb_env() unavailable",
                 this->id()));
  }
  {
    LmdbDb  db(*env, _db_name);
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadOnly);
    auto existing = db.get(txn, reg_key);
    const bool present = existing.has_value();
    txn.abort();
    if (present && !_overwrite_existing) {
      s->info(fmt(
          "ModelFetchStage('{}'): '{}' already registered; set "
          "overwrite_existing=true to refresh. Done.",
          this->id(), reg_key));
      if (ctx.has_consumers(0)) {
        FlexData sum = FlexData::make_object();
        auto so = sum.as_object();
        so.insert_or_assign("stage", FlexData::make_string("model-fetch"));
        so.insert_or_assign("hf_path", FlexData::make_string(hf_path));
        so.insert_or_assign("already_present", FlexData::make_bool(true));
        so.insert_or_assign("text", FlexData::make_string(
            fmt("[model-fetch] {} already present (skipped)", hf_path)()));
        co_await ctx.write(0,
            make_payload<FlexDataPayload>(std::move(sum)));
      }
      ctx.signal_done();
      co_return;
    }
  }

  s->info(fmt("ModelFetchStage('{}'): fetching '{}' -> '{}'",
              this->id(), hf_path, local_dir.string()));

  // -------- 3b. Dataset fetch (eval datasets) -------------------------
  // A catalogue entry carrying explicit dataset_files is fetched VERBATIM from
  // the given URLs (the HF datasets-server /rows pages) into local_dir and
  // registered -- no model-repo tree walk. Keeps dataset text out of the binary
  // (the model-eval stage reads these rows-*.json pages on demand).
  if (entry != nullptr && !entry->dataset_files.empty()) {
    std::error_code ec;
    fs::create_directories(local_dir, ec);
    uint64_t total = 0;
    FlexData files_arr = FlexData::make_array();
    for (size_t i = 0; i < entry->dataset_files.size(); ++i) {
      if (ctx.stop_requested()) {
        s->error(fmt("ModelFetchStage('{}'): canceled", this->id()));
      }
      const string& url  = entry->dataset_files[i].first;
      const string& dest = entry->dataset_files[i].second;
      const fs::path out = local_dir / dest;
      s->info(fmt("  [{}/{}] {} ...",
                  i + 1, entry->dataset_files.size(), dest));
      long st = 0;
      string derr;
      // Datasets-server is public -- no auth token needed.
      if (!http_download_(url, string(), _verify_tls, _timeout_seconds, out,
                          st, derr, nullptr, &cancel)) {
        s->error(fmt("ModelFetchStage('{}'): dataset fetch '{}' failed: {}",
                     this->id(), dest, derr));
      }
      files_arr.as_array().push_back(FlexData::make_string(dest));
      total += static_cast<uint64_t>(fs::file_size(out, ec));
    }
    FlexData rec = FlexData::make_object();
    auto ro = rec.as_object();
    ro.insert_or_assign("local_path",
                        FlexData::make_string(local_dir.string()));
    ro.insert_or_assign("source_url",
                        FlexData::make_string(
                            "https://huggingface.co/datasets"));
    ro.insert_or_assign("dataset", FlexData::make_bool(true));
    ro.insert_or_assign("model_type",
                        FlexData::make_string(entry->model_type));
    ro.insert_or_assign("total_bytes", FlexData::make_uint(total));
    ro.insert_or_assign("files", std::move(files_arr));
    {
      LmdbDb  db(*env, _db_name);
      LmdbTxn txn(*env, LmdbTxn::Mode::ReadWrite);
      const string bytes = rec.to_binary();
      db.put(txn, reg_key, bytes);
      txn.commit();
    }
    s->info(fmt(
        "ModelFetchStage('{}'): dataset '{}' ({}) registered in sub-db '{}'",
        this->id(), reg_key, human_bytes_(total), _db_name));
    ro.insert_or_assign("stage", FlexData::make_string("model-fetch"));
    ro.insert_or_assign("text", FlexData::make_string(
        fmt("[model-fetch] dataset {}\n  -> {}\n  {} file(s), {} bytes",
            reg_key, local_dir.string(),
            entry->dataset_files.size(), total)()));
    if (ctx.has_consumers(0)) {
      co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(rec)));
    }
    ctx.signal_done();
    co_return;
  }

  // -------- 4. Resolve auth token -------------------------------------
  string token = _hf_token;
  if (token.empty()) {
    if (const char* e = std::getenv("HF_TOKEN")) { token = e; }
  }
  if (token.empty()) {
    if (const char* e = std::getenv("HUGGING_FACE_HUB_TOKEN")) {
      token = e;
    }
  }

  // -------- 5. List repo files (HF tree API) --------------------------
  const string tree_url =
      "https://huggingface.co/api/models/" + hf_path
      + "/tree/main?recursive=true";
  string body, err;
  long   status = 0;
  bool   ok = http_get_(tree_url, token, _verify_tls,
                        _timeout_seconds, body, status, err);
  // Gated/private repo without a usable token: prompt once (masked) and
  // retry. This is where getpasswd earns its keep.
  if (!ok && (status == 401 || status == 403)) {
    string t;
    if (s->getpasswd(
            fmt("'{}' is gated. HuggingFace token (blank to cancel): ",
                hf_path),
            t, cancel) == UiInputStatus::Ok) {
      trim_(t);
      if (!t.empty()) {
        token = t;
        ok = http_get_(tree_url, token, _verify_tls, _timeout_seconds,
                       body, status, err);
      }
    }
  }
  if (!ok) {
    s->error(fmt("ModelFetchStage('{}'): listing '{}' failed: {}",
                 this->id(), hf_path, err));
  }

  FlexData tree;
  try {
    tree = FlexData::from_json(body);
  } catch (const std::exception& e) {
    s->error(fmt("ModelFetchStage('{}'): bad tree JSON for '{}': {}",
                 this->id(), hf_path, e.what()));
  }
  vector<HfFile> files = hf_tree_files(tree);
  if (files.empty()) {
    s->error(fmt("ModelFetchStage('{}'): '{}' lists no files (private or "
                 "non-existent?)", this->id(), hf_path));
  }

  // A catalogue entry may pin a SUBSET of repo files (e.g. one GGUF quant
  // plus its mmproj / imatrix companions out of a multi-quant repo) --
  // fetch just those, not the whole repo. Preserves the pinned order.
  if (entry && !entry->files.empty()) {
    vector<HfFile> picked;
    for (const string& want : entry->files) {
      bool found = false;
      for (const HfFile& f : files) {
        if (f.path == want) {
          picked.push_back(f);
          found = true;
          break;
        }
      }
      if (!found) {
        s->error(fmt("ModelFetchStage('{}'): pinned file '{}' not found "
                     "in repo '{}'", this->id(), want, hf_path));
      }
    }
    files = std::move(picked);
  }

  // -------- 6. Download ------------------------------------------------
  uint64_t total_bytes = 0;
  uint64_t downloaded  = 0;
  uint64_t skipped     = 0;
  FlexData files_arr = FlexData::make_array();
  for (size_t i = 0; i < files.size(); ++i) {
    if (ctx.stop_requested()) {
      s->error(fmt("ModelFetchStage('{}'): canceled after {}/{} files",
                   this->id(), i, files.size()));
    }
    const HfFile& f = files[i];
    const fs::path dest = local_dir / f.path;
    files_arr.as_array().push_back(FlexData::make_string(f.path));
    total_bytes += f.size;

    std::error_code ec;
    if (_skip_existing_files && f.size > 0 && fs::exists(dest)
        && fs::file_size(dest, ec) == f.size && !ec) {
      s->info(fmt("  [{}/{}] {} ({}) -- present, skip",
                  i + 1, files.size(), f.path, human_bytes_(f.size)));
      ++skipped;
      continue;
    }
    s->info(fmt("  [{}/{}] {} ({}) ...",
                i + 1, files.size(), f.path,
                f.size ? human_bytes_(f.size) : string("size unknown")));
    const string file_url = "https://huggingface.co/" + hf_path
                          + "/resolve/main/" + f.path;
    // A live progress bar for big shards. It rides a verbatim text
    // stream (stdout in a terminal; an in-place console line in the
    // web-ui, which honours the \r the bar redraws with).
    std::unique_ptr<UiTextStream> bar;
    if (static_cast<curl_off_t>(f.size) >= kBigFileBytes) {
      bar = s->open_text_stream();
    }
    long dl_status = 0;
    string dl_err;
    if (!http_download_(file_url, token, _verify_tls, _timeout_seconds,
                        dest, dl_status, dl_err, bar.get(), &cancel)) {
      if (bar) { bar->end(); }
      s->error(fmt("ModelFetchStage('{}'): download of '{}' failed: {}",
                   this->id(), f.path, dl_err));
    }
    if (bar) { bar->end(); }
    ++downloaded;
  }
  s->info(fmt("ModelFetchStage('{}'): {} file(s) ({} downloaded, {} "
              "already present), {} total",
              this->id(), files.size(), downloaded, skipped,
              human_bytes_(total_bytes)));

  // -------- 7. Unpack archives (.tar -> *.mlpackage) ------------------
  // Catalogue archive entries (the vpipe-supplement CoreML packages) ship a
  // single *.mlpackage per .tar. Unpack each into <repo>/<name>/ and point
  // the registered local_path at the contained .mlpackage so a stage's
  // model_path / coreml_vision_path resolves straight to a loadable package.
  // Default (non-archive) entries register the repo dir, as before.
  string local_path_str = local_dir.string();
  if (entry && entry->extract_archive) {
    const string sub = entry->name.empty() ? string("extracted")
                                           : entry->name;
    const fs::path extract_dir = local_dir / sub;
    for (const HfFile& f : files) {
      if (f.path.size() < 4
          || f.path.compare(f.path.size() - 4, 4, ".tar") != 0) {
        continue;
      }
      const fs::path archive = local_dir / f.path;
      fs::path pkg = find_mlpackage_(extract_dir);
      // Re-unpack when nothing is there yet, or anything was (re)downloaded
      // this run (so a refreshed archive overwrites a stale extraction).
      if (pkg.empty() || downloaded > 0) {
        s->info(fmt("ModelFetchStage('{}'): unpacking '{}' -> '{}' ...",
                    this->id(), f.path, extract_dir.string()));
        string xerr;
        if (!extract_tar_(archive, extract_dir, xerr)) {
          s->error(fmt(
              "ModelFetchStage('{}'): unpack of '{}' failed: {}",
              this->id(), f.path, xerr));
        }
        pkg = find_mlpackage_(extract_dir);
      }
      if (!pkg.empty()) {
        local_path_str = pkg.string();
        s->info(fmt("ModelFetchStage('{}'): unpacked package '{}'",
                    this->id(), local_path_str));
      } else {
        local_path_str = extract_dir.string();
        s->warn(fmt(
            "ModelFetchStage('{}'): no *.mlpackage found under '{}' "
            "after unpacking '{}'; registering the dir itself",
            this->id(), extract_dir.string(), f.path));
      }
    }
  }

  // -------- 8. Prepare tokenizer.json (Qwen3-ASR, native) -------------
  // The Qwen3-ASR repos ship the tokenizer as vocab.json + merges.txt +
  // tokenizer_config.json but no consolidated tokenizer.json (which our
  // runtime needs). Synthesize it natively -- no transformers / Python.
  bool tokenizer_ready = true;
  const bool want_tok = _prepare_tokenizer && entry
                        && entry->needs_tokenizer_json;
  if (want_tok) {
    const fs::path tj = local_dir / "tokenizer.json";
    std::error_code ec;
    if (fs::exists(tj) && fs::file_size(tj, ec) > 0 && !ec) {
      s->info(fmt("ModelFetchStage('{}'): tokenizer.json already present",
                  this->id()));
    } else {
      s->info(fmt("ModelFetchStage('{}'): preparing tokenizer.json "
                  "(native byte-level BPE, no transformers) ...",
                  this->id()));
      string perr;
      tokenizer_ready =
          prepare_qwen_asr_tokenizer_json(local_dir.string(), perr);
      if (tokenizer_ready) {
        s->info(fmt("ModelFetchStage('{}'): tokenizer.json prepared",
                    this->id()));
      } else {
        // Non-fatal: the model is downloaded + still registers.
        s->warn(fmt(
            "ModelFetchStage('{}'): tokenizer.json prep failed: {} "
            "(repo must ship vocab.json + merges.txt + "
            "tokenizer_config.json)", this->id(), perr));
      }
    }
  }

  // -------- 9. Register in LMDB ---------------------------------------
  FlexData rec = FlexData::make_object();
  auto ro = rec.as_object();
  ro.insert_or_assign("hf_path", FlexData::make_string(hf_path));
  ro.insert_or_assign("source_url",
                      FlexData::make_string("https://huggingface.co/"
                                            + hf_path));
  ro.insert_or_assign("local_path",
                      FlexData::make_string(local_path_str));
  ro.insert_or_assign("file_count",
                      FlexData::make_uint(files.size()));
  ro.insert_or_assign("total_bytes", FlexData::make_uint(total_bytes));
  ro.insert_or_assign("files", std::move(files_arr));
  if (entry) {
    if (!entry->name.empty()) {
      ro.insert_or_assign("name", FlexData::make_string(entry->name));
    }
    ro.insert_or_assign("family", FlexData::make_string(entry->family));
    ro.insert_or_assign("version", FlexData::make_string(entry->version));
    ro.insert_or_assign("param_class",
                        FlexData::make_string(entry->param_class));
    ro.insert_or_assign("variant", FlexData::make_string(entry->variant));
    ro.insert_or_assign("model_type",
                        FlexData::make_string(entry->model_type));
    ro.insert_or_assign("needs_tokenizer_json",
                        FlexData::make_bool(entry->needs_tokenizer_json));
    ro.insert_or_assign("tokenizer_ready",
                        FlexData::make_bool(tokenizer_ready));
  }

  {
    LmdbDb      db(*env, _db_name);
    LmdbTxn     txn(*env, LmdbTxn::Mode::ReadWrite);
    const string bytes = rec.to_binary();
    db.put(txn, reg_key, bytes);
    txn.commit();
  }

  s->info(fmt("ModelFetchStage('{}'): registered '{}' in sub-db '{}'",
              this->id(), reg_key, _db_name));
  ro.insert_or_assign("stage", FlexData::make_string("model-fetch"));
  ro.insert_or_assign("text", FlexData::make_string(
      fmt("[model-fetch] {}\n  -> {}\n  {} file(s), {} bytes",
          hf_path, local_path_str, files.size(), total_bytes)()));
  if (ctx.has_consumers(0)) {
    co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(rec)));
  }
  ctx.signal_done();
  co_return;
}

VPIPE_REGISTER_STAGE(ModelFetchStage)
VPIPE_REGISTER_SPEC(ModelFetchStage, kSpec)

}
