#ifndef UI_DELEGATE_INTF_H
#define UI_DELEGATE_INTF_H

#include "interfaces/ui-view-intf.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

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

// ---- user interrupts ------------------------------------------------
//
// "Stop what you are doing, but stay alive" -- Ctrl-C on the console,
// the Interrupt button in the web UI. A stage with long-running,
// abandonable work (a text-chat generation loop) registers a handler
// here; the front end calls every registered handler when the user
// asks to interrupt. This is NOT a pipeline stop: the stage keeps its
// state (a chat context keeps its K/V cache) and simply cuts the
// current unit of work short.
//
// The return value says whether the handler ACTUALLY interrupted
// something in flight. A front end uses it to tell a consumed
// interrupt from one that had nothing to act on -- the stdio front end
// escalates an unconsumed Ctrl-C to a process stop, so a pipeline with
// nothing interruptible still dies on one keystroke.
//
// Contract: handlers run on the front end's own thread (the CLI's wait
// loop, the web-ui HTTP thread) while the registry lock is held, so a
// handler MUST be quick, MUST NOT block, and MUST NOT add or remove
// handlers. Setting an atomic flag that the stage's own loop polls is
// the intended shape.
using UiInterruptHandler = std::function<bool()>;

// The handler table behind UiDelegateIntf::register_interrupt_handler.
// Held by shared_ptr so a UiInterruptToken outliving its delegate
// unregisters into thin air instead of dangling. Safe to call from
// multiple threads.
class UiInterruptRegistry {
public:
  // Returns the id to pass to remove(). Never 0.
  std::uint64_t add(std::string label, UiInterruptHandler fn);
  void          remove(std::uint64_t id) noexcept;

  // Invoke every registered handler; returns how many reported that
  // they interrupted something (i.e. returned true).
  int dispatch();

  std::size_t size() const;

private:
  struct Entry {
    std::uint64_t      id;
    std::string        label;
    UiInterruptHandler fn;
  };
  mutable std::mutex _mu;
  std::vector<Entry> _entries;
  std::uint64_t      _next_id = 1;
};

// RAII registration handle: destroying it unregisters the handler.
// Move-only; a default-constructed token is inert (holds nothing, and
// reset() on it is a no-op), which is what contexts with no UI hand
// back. Keep the token alive for exactly as long as the handler's
// captured state is valid -- a stage stores it as a member so the
// handler is gone before the stage is.
class UiInterruptToken {
public:
  UiInterruptToken() = default;
  UiInterruptToken(std::weak_ptr<UiInterruptRegistry> reg,
                   std::uint64_t                      id) noexcept
      : _reg(std::move(reg)), _id(id)
  {
  }
  ~UiInterruptToken() { reset(); }

  UiInterruptToken(const UiInterruptToken&)            = delete;
  UiInterruptToken& operator=(const UiInterruptToken&) = delete;

  UiInterruptToken(UiInterruptToken&& o) noexcept
      : _reg(std::move(o._reg)), _id(o._id)
  {
    o._id = 0;
  }
  UiInterruptToken& operator=(UiInterruptToken&& o) noexcept
  {
    if (this != &o) {
      reset();
      _reg  = std::move(o._reg);
      _id   = o._id;
      o._id = 0;
    }
    return *this;
  }

  // Unregister now (idempotent). A no-op once the owning registry --
  // and with it the delegate -- is gone.
  void reset() noexcept
  {
    if (_id != 0) {
      if (auto r = _reg.lock()) { r->remove(_id); }
      _id = 0;
    }
    _reg.reset();
  }

  explicit operator bool() const noexcept { return _id != 0; }

private:
  std::weak_ptr<UiInterruptRegistry> _reg;
  std::uint64_t                      _id = 0;
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

  // Like getline(), but the caller ACCEPTS media-line markers (see
  // common/media-line.h): the returned line may carry inline
  // image/audio attachments as special text sequences --
  //   <|__vpipe_fs_im_start__|>path<|__vpipe_fs_im_end__|> (a local
  // filesystem path; im=image, au=audio) or
  //   <|__vpipe_base64_im_start__|>length,data<|__vpipe_base64_im_end__|>
  // (an inline base64 payload). Delegates that can collect attachments
  // advertise the capability to their front end: the web-ui delegate
  // flags the pending request so the browser shows attach-image /
  // attach-audio buttons and drag-and-drop, emitting base64 markers;
  // the stdio delegate reads a plain line on which a local user types
  // fs-path markers by hand. The default forwards to getline() so a
  // delegate that knows nothing about media still functions (markers
  // typed literally still parse downstream). Prompt, cancellation, the
  // Ok/Eof/Canceled outcomes, and the newline-stripped `out` are
  // identical to getline().
  virtual UiInputStatus
  getmedialine(const VpipeFormat&           prompt,
               std::string&                 out,
               const std::function<bool()>& should_cancel)
  {
    return getline(prompt, out, should_cancel);
  }

  // Open a live text-output stream (see UiTextStream). Used for
  // token-by-token model output that should appear as it is produced,
  // rather than as one info() line at the end. Never returns null.
  virtual std::unique_ptr<UiTextStream> open_text_stream() = 0;

  // ---- stage-provided GUI views ------------------------------------
  //
  // A GRAPHICAL front end presents the panels stages register in the
  // UiViewRegistry (ui/ui-view-registry.h): it serves each view's
  // embedded module bytes and forwards the FlexData protocol between
  // the view and its backend. This accessor is how the rest of the
  // system asks "is such a front end attached, and what may a view
  // backend ask of it" -- the web-ui delegate returns its host, and
  // text-only delegates (StdioUiDelegate, the capturing test
  // delegates) return null, so a stage with a view simply has none
  // presented. See interfaces/ui-view-intf.h.
  virtual UiViewHostIntf* ui_view_host() { return nullptr; }

  // ---- stage interrupt handlers ------------------------------------
  //
  // Concrete (not virtual) and shared by every delegate: the table is
  // the same wherever the front end is, only the gesture that fires it
  // differs -- Ctrl-C for StdioUiDelegate, the User I/O view's
  // Interrupt button for WebUiDelegate. Stages reach this through
  // SessionContextIntf::register_interrupt_handler rather than
  // touching the delegate directly.
  //
  // `label` names the registrant for diagnostics (e.g. the stage id);
  // it has no effect on dispatch. Keep the returned token alive for as
  // long as `fn` is safe to call -- destroying it unregisters.
  UiInterruptToken
  register_interrupt_handler(std::string label, UiInterruptHandler fn)
  {
    const std::uint64_t id =
        _interrupts->add(std::move(label), std::move(fn));
    return UiInterruptToken(_interrupts, id);
  }

  // Fire every registered handler. Returns how many reported that they
  // actually interrupted work in flight (0 = nothing to interrupt).
  // Called by the front end, never by stages.
  int dispatch_interrupt() { return _interrupts->dispatch(); }

  // How many handlers are currently registered. Diagnostics/tests.
  std::size_t interrupt_handler_count() const
  {
    return _interrupts->size();
  }

private:
  // shared_ptr, not a plain member: tokens hold a weak_ptr to it, so a
  // token that outlives its delegate unregisters harmlessly.
  std::shared_ptr<UiInterruptRegistry> _interrupts =
      std::make_shared<UiInterruptRegistry>();
};

}

#endif
