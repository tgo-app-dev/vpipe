#include "apps/web-ui/web-ui-delegate.h"
#include "common/media-line.h"
#include "common/vpipe-format.h"

#include <chrono>
#include <memory>
#include <string_view>
#include <utility>

using namespace std;

namespace vpipe::webui {

// Handle returned by WebUiDelegate::open_text_stream. Forwards writes
// to the delegate keyed by its stream id; end() (or the dtor) closes
// the stream so the next /api/io/console drain finalizes it.
class WebUiTextStream final : public vpipe::UiTextStream {
public:
  WebUiTextStream(WebUiDelegate* d, uint64_t id) : _d(d), _id(id) {}
  ~WebUiTextStream() override { end(); }

  void write(string_view chunk) override
  {
    if (_d && !_ended) { _d->stream_write_(_id, chunk); }
  }
  void end() override
  {
    if (_d && !_ended) {
      _ended = true;
      _d->stream_end_(_id);
    }
  }

private:
  WebUiDelegate* _d;
  uint64_t       _id;
  bool           _ended = false;
};

void
WebUiDelegate::trim_()
{
  // Drop at most one stale entry per call so push_'s held-mutex
  // window stays O(1). After a cap reduction the background trim
  // worker handles the bulk drain; under steady-state push load
  // size grows by 1 and this pops 1, so the ring stays at the cap.
  if (_console.size() > _max_console) {
    _console.pop_front();
  }
}

namespace {
// Bulk-trim chunk size. We hold _mu only long enough to move out
// kTrimChunk entries (a deque pop_front + a std::move of the
// ConsoleLine fields is ~ns each), then we release the lock and
// destruct the moved-out strings -- the actually expensive bit --
// without blocking pushes or polls.
constexpr std::size_t kTrimChunk = 64;

// Append `chunk` to a stream line's text, honouring the in-place control
// characters a TUI progress bar uses so the STORED text always equals
// what the console box should show (the browser just renders the text):
//   '\b'  erase the previous character (never crossing a line start)
//   '\r'  carriage return: drop back to just after the last newline
// All other bytes (including '\n') append verbatim. Chunks with no
// control characters take the fast straight-append path.
void
append_stream_chunk_(std::string& text, std::string_view chunk)
{
  if (chunk.find('\b') == std::string_view::npos
      && chunk.find('\r') == std::string_view::npos) {
    text.append(chunk);
    return;
  }
  for (char ch : chunk) {
    if (ch == '\b') {
      if (!text.empty() && text.back() != '\n') { text.pop_back(); }
    } else if (ch == '\r') {
      const auto nl = text.find_last_of('\n');
      text.erase(nl == std::string::npos ? 0 : nl + 1);
    } else {
      text.push_back(ch);
    }
  }
}
}

void
WebUiDelegate::trim_in_background_()
{
  std::vector<ConsoleLine> evicted;
  evicted.reserve(kTrimChunk);
  while (true) {
    // Destruct the previous chunk's strings outside the lock.
    evicted.clear();
    {
      std::lock_guard<std::mutex> lk(_mu);
      if (_console.size() <= _max_console) {
        _trim_running = false;
        return;
      }
      const std::size_t to_pop = std::min(
          kTrimChunk, _console.size() - _max_console);
      for (std::size_t i = 0; i < to_pop; ++i) {
        evicted.push_back(std::move(_console.front()));
        _console.pop_front();
      }
    }
    // evicted's elements destruct on the next iteration's clear()
    // (or when the function returns).
  }
}

std::size_t
WebUiDelegate::max_console() const
{
  std::lock_guard<std::mutex> lk(_mu);
  return _max_console;
}

void
WebUiDelegate::set_max_console(std::size_t n)
{
  if (n < kMinMaxConsole) { n = kMinMaxConsole; }
  if (n > kMaxMaxConsole) { n = kMaxMaxConsole; }
  bool spawn = false;
  {
    std::lock_guard<std::mutex> lk(_mu);
    _max_console = n;
    // If a drain is already running, it re-reads _max_console at the
    // top of each chunk -- the new cap takes effect on its next
    // iteration without us spawning another worker.
    if (_console.size() > _max_console && !_trim_running) {
      _trim_running = true;
      spawn = true;
    }
  }
  if (spawn) {
    // Join the previous trim_thread (already exited; this is a fast
    // no-op when the std::thread is in its detached-finished state)
    // before replacing it.
    if (_trim_thread.joinable()) { _trim_thread.join(); }
    _trim_thread = std::thread([this] { trim_in_background_(); });
  }
}

WebUiDelegate::~WebUiDelegate()
{
  if (_trim_thread.joinable()) { _trim_thread.join(); }
}

void
WebUiDelegate::push_(const char* level, const vpipe::VpipeFormat& f)
{
  string text;
  try {
    text = f();
  } catch (...) {
    text = "<formatter threw>";
  }
  lock_guard<mutex> lk(_mu);
  _console.push_back(ConsoleLine{ _next_seq++, level, std::move(text) });
  trim_();
  _cv.notify_all();
}

void
WebUiDelegate::error(const vpipe::VpipeFormat& f)
{
  push_("error", f);
}

void
WebUiDelegate::warn(const vpipe::VpipeFormat& f)
{
  push_("warn", f);
}

void
WebUiDelegate::info(const vpipe::VpipeFormat& f)
{
  push_("info", f);
}

vpipe::UiInputStatus
WebUiDelegate::getline(const vpipe::VpipeFormat&    prompt,
                       string&                      out,
                       const function<bool()>&      should_cancel)
{
  return request_input_(prompt, out, should_cancel,
                        /*masked=*/false, /*media=*/false);
}

vpipe::UiInputStatus
WebUiDelegate::getpasswd(const vpipe::VpipeFormat&    prompt,
                         string&                      out,
                         const function<bool()>&      should_cancel)
{
  return request_input_(prompt, out, should_cancel,
                        /*masked=*/true, /*media=*/false);
}

vpipe::UiInputStatus
WebUiDelegate::getmedialine(const vpipe::VpipeFormat&    prompt,
                            string&                      out,
                            const function<bool()>&      should_cancel)
{
  return request_input_(prompt, out, should_cancel,
                        /*masked=*/false, /*media=*/true);
}

vpipe::UiInputStatus
WebUiDelegate::request_input_(const vpipe::VpipeFormat&    prompt,
                              string&                      out,
                              const function<bool()>&      should_cancel,
                              bool                         masked,
                              bool                         media)
{
  string p;
  try {
    p = prompt();
  } catch (...) {
    p.clear();
  }

  // Echo the prompt into the console transcript so it has context even
  // for someone scrolling back; the live input box also shows it.
  if (!p.empty()) {
    lock_guard<mutex> lk(_mu);
    _console.push_back(ConsoleLine{ _next_seq++, "info", p });
    trim_();
  }

  unique_lock<mutex> lk(_mu);

  // One prompt at a time: wait for any in-flight request to clear.
  while (_req_id != 0) {
    if (should_cancel && should_cancel()) {
      return vpipe::UiInputStatus::Canceled;
    }
    _cv.wait_for(lk, chrono::milliseconds(50));
  }

  const uint64_t my_id = _next_req_id++;
  _req_id     = my_id;
  _req_prompt = p;
  _req_masked = masked;
  _req_media  = media;
  _have_resp  = false;
  _resp_text.clear();
  _cv.notify_all();

  vpipe::UiInputStatus status = vpipe::UiInputStatus::Canceled;
  for (;;) {
    if (_have_resp && _req_id == my_id) {
      out    = std::move(_resp_text);
      status = vpipe::UiInputStatus::Ok;
      break;
    }
    if (should_cancel && should_cancel()) {
      status = vpipe::UiInputStatus::Canceled;
      break;
    }
    _cv.wait_for(lk, chrono::milliseconds(50));
  }

  // Release the slot if it is still ours.
  if (_req_id == my_id) {
    _req_id     = 0;
    _have_resp  = false;
    _resp_text.clear();
    _req_prompt.clear();
    _req_masked = false;
    _req_media  = false;
    _cv.notify_all();
  }
  return status;
}

vector<WebUiDelegate::ConsoleLine>
WebUiDelegate::console_since(uint64_t since, uint64_t* latest_seq,
                             size_t limit)
{
  lock_guard<mutex> lk(_mu);
  vector<ConsoleLine> out;
  if (limit > 0) {
    out.reserve(std::min(limit, _console.size()));
  }
  bool capped = false;
  for (auto& l : _console) {
    // New lines (seq > since) plus changed ones (a growing/closing
    // stream). Clear dirty as we drain it.
    if (l.seq > since || l.dirty) {
      if (limit > 0 && out.size() >= limit) {
        capped = true;
        break;
      }
      out.push_back(l);
      l.dirty = false;
    }
  }
  if (latest_seq) {
    // When capped, advance `since` only as far as we actually drained
    // so the next poll resumes exactly where this one stopped.
    // Otherwise expose the true tip so the client can ignore
    // anything between the last returned dirty entry and the tip.
    *latest_seq = (capped && !out.empty())
                      ? out.back().seq
                      : (_next_seq - 1);
  }
  return out;
}

bool
WebUiDelegate::pending_input(uint64_t* id, string* prompt,
                             bool* masked, bool* media) const
{
  lock_guard<mutex> lk(_mu);
  if (_req_id == 0) {
    return false;
  }
  if (id) {
    *id = _req_id;
  }
  if (prompt) {
    *prompt = _req_prompt;
  }
  if (masked) {
    *masked = _req_masked;
  }
  if (media) {
    *media = _req_media;
  }
  return true;
}

bool
WebUiDelegate::submit_input(uint64_t id, string text)
{
  lock_guard<mutex> lk(_mu);
  if (_req_id == 0 || _req_id != id) {
    return false;
  }
  // Echo the answer into the transcript (so it persists across a view
  // re-mount and shows in the log), then hand it to the waiting
  // getline. Media-line attachment markers are compressed to glyphs in
  // the echo so a multi-megabyte base64 payload never enters the
  // console ring (and never re-polls to the browser).
  _console.push_back(ConsoleLine{ _next_seq++, "input",
                                  "> " + media_line::to_display(text) });
  trim_();
  _resp_text = std::move(text);
  _have_resp = true;
  _cv.notify_all();
  return true;
}

void
WebUiDelegate::clear_console()
{
  lock_guard<mutex> lk(_mu);
  _console.clear();
}

unique_ptr<vpipe::UiTextStream>
WebUiDelegate::open_text_stream()
{
  return make_unique<WebUiTextStream>(this, stream_open_());
}

// A text stream is just a console line; its id IS that line's seq.
uint64_t
WebUiDelegate::stream_open_()
{
  lock_guard<mutex> lk(_mu);
  uint64_t seq = _next_seq++;
  _console.push_back(ConsoleLine{ seq, "stream", string{}, /*open=*/true });
  trim_();
  _cv.notify_all();
  return seq;
}

void
WebUiDelegate::stream_write_(uint64_t id, string_view chunk)
{
  if (chunk.empty()) { return; }
  lock_guard<mutex> lk(_mu);
  for (auto& l : _console) {
    if (l.seq == id && l.open) {
      append_stream_chunk_(l.text, chunk);
      l.dirty = true;     // re-deliver so the client grows it in place
      break;
    }
  }
  _cv.notify_all();
}

void
WebUiDelegate::stream_end_(uint64_t id)
{
  lock_guard<mutex> lk(_mu);
  for (auto& l : _console) {
    if (l.seq == id) {
      l.open = false;
      l.dirty = true;     // final flush of the closed stream
      break;
    }
  }
  _cv.notify_all();
}

}
