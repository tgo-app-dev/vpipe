#ifndef WEBUI_WEB_UI_DELEGATE_H
#define WEBUI_WEB_UI_DELEGATE_H

#include "interfaces/ui-delegate-intf.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace vpipe::webui {

class WebUiTextStream;

// UiDelegateIntf that diverts a Session's user-facing I/O to the
// browser's User I/O page:
//   * error / warn / info  -> an in-memory console ring the browser
//     polls (GET /api/io/console?since=N).
//   * getline()            -> publishes a pending input request that
//     the browser reads (GET /api/io/pending) and answers (POST
//     /api/io/input). The calling worker blocks until the answer
//     arrives or the supplied cancel predicate fires.
//
// Threading: the UiDelegateIntf methods run on pool worker threads;
// the API accessors (console_since / pending_input / submit_input)
// run on the HTTP server thread. All state is guarded by one mutex
// and a condition variable. Only one getline() is serviced at a time;
// a second concurrent getline waits for the first to clear.
class WebUiDelegate final : public vpipe::UiDelegateIntf {
public:
  ~WebUiDelegate() override;

  // ---- UiDelegateIntf (worker threads) -----------------------------
  void error(const vpipe::VpipeFormat&) override;
  void warn(const vpipe::VpipeFormat&) override;
  void info(const vpipe::VpipeFormat&) override;
  vpipe::UiInputStatus
  getline(const vpipe::VpipeFormat& prompt, std::string& out,
          const std::function<bool()>& should_cancel) override;

  // Password variant: publishes the pending request with a `masked`
  // flag so the browser renders a masked input field. Identical
  // rendezvous semantics to getline otherwise.
  vpipe::UiInputStatus
  getpasswd(const vpipe::VpipeFormat& prompt, std::string& out,
            const std::function<bool()>& should_cancel) override;

  std::unique_ptr<vpipe::UiTextStream> open_text_stream() override;

  // ---- API surface (HTTP thread) -----------------------------------
  // Everything the user sees is a console line, including live text
  // streams (level "stream", `open` true while growing) and answered
  // getline echoes (level "input"). Keeping them all in one ordered
  // log means a client that reconnects / re-mounts the view rebuilds
  // the full transcript from seq 0 -- streams and input no longer
  // vanish after navigating away.
  struct ConsoleLine {
    std::uint64_t seq = 0;
    std::string   level;   // "error"|"warn"|"info"|"input"|"stream"
    std::string   text;
    bool          open = false;   // a "stream" line still growing
    bool          dirty = false;  // changed since last drain -> resend
  };

  // Lines that are new (seq > `since`) or have changed since the last
  // call (growing/closing streams) -- the latter are re-sent so the
  // client updates them in place (keyed by seq). Returned lines have
  // their dirty flag cleared. `*latest_seq` is the highest seq held
  // (or, if the result was capped by `limit`, the highest seq in the
  // returned batch so the next poll picks up after it).
  //
  // `limit` bounds the response size in entries. 0 means unlimited.
  // The iteration is in seq order, so when capped the next poll with
  // `since = *latest_seq` continues exactly where this one stopped;
  // dirty entries we couldn't fit stay dirty and are re-delivered on
  // the next pass. Not const: it clears drained dirty flags.
  std::vector<ConsoleLine>
  console_since(std::uint64_t since, std::uint64_t* latest_seq,
                std::size_t limit = 0);

  // If an input request is waiting, fill (*id, *prompt, *masked) and
  // return true. `masked` is set when the request came from getpasswd()
  // (the browser should mask the field). Any out-param may be null.
  bool pending_input(std::uint64_t* id, std::string* prompt,
                     bool* masked = nullptr) const;

  // Answer the pending getline identified by `id`. Returns false when
  // no request is pending or the id does not match (stale answer).
  // Echoes the answer into the transcript as an "input" line.
  bool submit_input(std::uint64_t id, std::string text);

  // Drop all console history (authoritative clear for the User I/O
  // "Clear" button) so nothing is re-delivered on the next poll. A
  // pending getline request is left intact -- clearing the transcript
  // must not cancel an outstanding input prompt.
  void clear_console();

  // Console ring cap. Lines beyond this count are dropped from the
  // front (oldest first) the way a terminal scrollback buffer does,
  // bounding memory and JSON-poll size for long-running sessions.
  // Default is kDefaultMaxConsole; the Settings page lets the user
  // raise/lower it. Trimming on `set` is immediate.
  static constexpr std::size_t kDefaultMaxConsole = 8192;
  static constexpr std::size_t kMinMaxConsole     = 16;
  static constexpr std::size_t kMaxMaxConsole     = 1'000'000;
  std::size_t max_console() const;
  void        set_max_console(std::size_t n);

private:
  friend class WebUiTextStream;

  void push_(const char* level, const vpipe::VpipeFormat&);

  // Shared rendezvous body for getline()/getpasswd(): echoes the
  // prompt, publishes a single pending request (masked or not), blocks
  // until the browser answers or `should_cancel` fires.
  vpipe::UiInputStatus
  request_input_(const vpipe::VpipeFormat& prompt, std::string& out,
                 const std::function<bool()>& should_cancel,
                 bool masked);
  // Incremental ring bound: at most one entry per call. Bulk drains
  // after a cap reduction are handled by trim_in_background_() so a
  // single shrink request doesn't block every other HTTP handler
  // (and every worker pushing log lines) for the duration of the
  // destructors. Caller holds _mu.
  void trim_();

  // Chunked background drain spawned by set_max_console. Loops:
  // acquire _mu, move out up to kTrimChunk entries, release _mu,
  // destruct the moved-out lines under no lock, repeat -- so the
  // expensive string deallocations don't extend the held-mutex
  // window. Exits when _console.size() <= _max_console.
  void trim_in_background_();

  // Called by WebUiTextStream (worker thread). A stream is just a
  // console line; its id IS that line's seq.
  std::uint64_t stream_open_();
  void          stream_write_(std::uint64_t id, std::string_view chunk);
  void          stream_end_(std::uint64_t id);

  mutable std::mutex      _mu;
  std::condition_variable _cv;

  std::deque<ConsoleLine> _console;
  std::uint64_t           _next_seq = 1;
  std::size_t             _max_console = kDefaultMaxConsole;
  // Background trim worker. `_trim_running` is guarded by _mu and
  // ensures at most one drainer is alive at a time; the std::thread
  // member is joined in the destructor (and replaced in
  // set_max_console after the previous one exits).
  std::thread             _trim_thread;
  bool                    _trim_running = false;

  // Single in-flight input request. _req_id != 0 means a getline is
  // blocked waiting; _have_resp signals the browser answered.
  std::uint64_t _req_id      = 0;
  std::string   _req_prompt;
  bool          _req_masked  = false;   // request came from getpasswd()
  bool          _have_resp   = false;
  std::string   _resp_text;
  std::uint64_t _next_req_id = 1;
};

}

#endif
