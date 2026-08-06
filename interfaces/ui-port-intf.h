// ui-port-intf.h -- the session's user-facing side: interactive input,
// live text output, interrupt registration and localization.
//
// Every method is defaulted to an inert answer (Eof / a null stream /
// an inert token), so a context that never services a UI inherits the
// whole role without implementing any of it. That is what makes this
// safe to separate: the "no UI here" case is the default, not an
// override.

#ifndef UI_PORT_INTF_H
#define UI_PORT_INTF_H

#include "interfaces/ui-delegate-intf.h"
#include "common/i18n.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace vpipe {

struct VpipeFormat;

class UiPortIntf {
public:
  virtual ~UiPortIntf() = default;

  // The address the vpipe web-ui's HTTP server bound to, when this
  // session is being served by the web-ui; empty otherwise. Stages that
  // host their own HTTP endpoint (e.g. hls-broadcast) read this so, by
  // default, they bind to the same interface the UI is reachable on --
  // matching whatever the operator chose for the UI (en0, an explicit
  // --bind, 0.0.0.0, ...). The default returns empty: CLI sessions and
  // adapter contexts have no web-ui and callers fall back to en0.
  virtual std::string web_ui_bind_address() const { return {}; }

  // Current UI/message locale as an IETF tag (e.g. "en-us", "zh-cn").
  // Drives tr() and any user-facing string the session formats. The
  // default is English; the concrete Session returns the configured or
  // last-set tag. (Not the model/ASR language hint -- that is separate.)
  virtual std::string language() const
  {
    return std::string(default_language());
  }

  // Translate an application message `key` for the current language,
  // falling back to en-us and then to the key itself. Convenience over
  // the free vpipe::localize(); the key catalogue lives in common/i18n.
  virtual std::string tr(std::string_view key) const
  {
    return localize(language(), key);
  }

  // Blocking request for one line of interactive user input, routed to
  // the session's UI delegate. `prompt` is shown before the read;
  // `should_cancel` (if set) is polled while waiting so a pipeline
  // stop is observed promptly. See UiDelegateIntf::getline.
  virtual UiInputStatus
  getline(const VpipeFormat& /*prompt*/, std::string& /*out*/,
          const std::function<bool()>& /*should_cancel*/) const
  {
    return UiInputStatus::Eof;
  }

  // Blocking request for one line of SECRET interactive input (a
  // password), routed to the session's UI delegate, which masks it on
  // screen. See UiDelegateIntf::getpasswd.
  virtual UiInputStatus
  getpasswd(const VpipeFormat& /*prompt*/, std::string& /*out*/,
            const std::function<bool()>& /*should_cancel*/) const
  {
    return UiInputStatus::Eof;
  }

  // Blocking request for one line of interactive user input that MAY
  // carry inline media-line attachment markers (images/audio; see
  // common/media-line.h and UiDelegateIntf::getmedialine), routed to
  // the session's UI delegate. Calling this instead of getline tells
  // the delegate the caller will parse markers, so media-capable
  // front ends (web-ui) offer attach/drop controls.
  virtual UiInputStatus
  getmedialine(const VpipeFormat& /*prompt*/, std::string& /*out*/,
               const std::function<bool()>& /*should_cancel*/) const
  {
    return UiInputStatus::Eof;
  }

  // Open a live text-output stream routed to the session's UI delegate
  // (see UiDelegateIntf::open_text_stream / UiTextStream). The default
  // returns a no-op stream. Never returns null.
  virtual std::unique_ptr<UiTextStream>
  open_text_stream() const
  {
    return std::make_unique<NullUiTextStream>();
  }

  // Open a progress report routed to the session's UI delegate (see
  // UiDelegateIntf::open_progress / UiProgress). `desc` is the short
  // label a renderer shows. The default returns an INERT handle whose
  // update() is a no-op, so a stage reports progress unconditionally
  // and never checks whether anyone is watching.
  virtual UiProgress
  open_progress(std::string /*desc*/) const
  {
    return UiProgress();
  }

  // Register a handler for "the user asked to interrupt ongoing work"
  // (Ctrl-C on the console, the web UI's Interrupt button), routed to
  // the session's UI delegate. A stage with abandonable long-running
  // work registers here and cuts that work short when called; it is
  // NOT a pipeline stop, so the stage keeps its state. See
  // UiDelegateIntf::register_interrupt_handler for the handler
  // contract (quick, non-blocking, returns true if it actually
  // interrupted something).
  //
  // Store the returned token as a member: destroying it unregisters,
  // so the handler cannot outlive the state it captured. The default
  // returns an inert token.
  virtual UiInterruptToken
  register_interrupt_handler(std::string /*label*/,
                             UiInterruptHandler /*fn*/) const
  {
    return {};
  }
};

}

#endif
