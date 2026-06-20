#ifndef UI_DELEGATE_INTF_H
#define UI_DELEGATE_INTF_H

#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace vpipe {

struct VpipeFormat;

// Outcome of a UiDelegateIntf::getline() request.
enum class UiInputStatus {
  Ok,        // `out` holds one line of user input
  Eof,       // input stream closed -- no more input will arrive
  Canceled,  // the supplied cancel predicate fired before input arrived
};

// A live, append-only stream of text to the user, obtained from
// UiDelegateIntf::open_text_stream(). Unlike info() (one framed line
// per call), chunks written here are emitted VERBATIM and concatenate
// into a single logical message as they arrive -- the channel for
// token-by-token model output the user watches materialise. end()
// finalizes the message (terminates the line on stdio, closes the
// console entry in the web UI); the destructor calls end() if the
// caller didn't. A stream has a single producer and is not safe to
// write from multiple threads at once.
class UiTextStream {
public:
  virtual ~UiTextStream() = default;
  virtual void write(std::string_view chunk) = 0;
  virtual void end() = 0;
};

// No-op stream returned by contexts/delegates that have no live user
// to stream to (adapter SessionContextIntfs, capturing test delegates).
// Lets callers stream unconditionally without null checks.
class NullUiTextStream final : public UiTextStream {
public:
  void write(std::string_view) override {}
  void end() override {}
};

// User-facing I/O channel, distinct from LogDelegateIntf (the
// diagnostic-logging sink for debug/verbose/normal/always). This is
// where messages the operator is meant to read go -- error / warn /
// info -- and where interactive line input comes from -- getline.
//
// A Session routes SessionContextIntf::error/warn/info here and
// SessionContextIntf::getline here. The default implementation
// (StdioUiDelegate) writes error/warn to stderr, info to stdout, and
// reads getline from stdin; the web-ui app installs a delegate that
// diverts the same channels to a browser console.
//
// Implementations MUST be safe to call concurrently from multiple
// threads and MUST NOT throw out of error()/warn()/info() (a UI
// failure must not crash the caller). Note that the *Session* wrapper
// throws after error() returns -- that throw lives in Session, not in
// the delegate.
class UiDelegateIntf {
public:
  virtual ~UiDelegateIntf() = default;

  virtual void error(const VpipeFormat&) = 0;
  virtual void warn(const VpipeFormat&) = 0;
  virtual void info(const VpipeFormat&) = 0;

  // Blocking request for one line of user input. `prompt` is presented
  // to the user before the read (it may be empty). `should_cancel`, if
  // non-null, is polled periodically while waiting; once it returns
  // true the call abandons the read and returns Canceled so a pipeline
  // stop is observed promptly. On Ok, `out` holds the line with any
  // trailing newline stripped.
  virtual UiInputStatus
  getline(const VpipeFormat&           prompt,
          std::string&                 out,
          const std::function<bool()>& should_cancel) = 0;

  // Like getline(), but the typed input is a SECRET (a password):
  // implementations MASK it on screen so it is never shown in the clear
  // -- the stdio delegate disables terminal echo for the read, the
  // web-ui delegate flags the request so the browser renders a masked
  // input field. Prompt, cancellation, the Ok/Eof/Canceled outcomes,
  // and the newline-stripped `out` are otherwise identical to
  // getline(). The default forwards to getline() so a delegate that
  // cannot mask still functions (the input is simply not masked);
  // delegates capable of masking override this.
  virtual UiInputStatus
  getpasswd(const VpipeFormat&           prompt,
            std::string&                 out,
            const std::function<bool()>& should_cancel)
  {
    return getline(prompt, out, should_cancel);
  }

  // Open a live text-output stream (see UiTextStream). Used for
  // token-by-token model output that should appear as it is produced,
  // rather than as one info() line at the end. Never returns null.
  virtual std::unique_ptr<UiTextStream> open_text_stream() = 0;
};

}

#endif
