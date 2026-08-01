#ifndef UI_VIEW_INTF_H
#define UI_VIEW_INTF_H

#include "common/flex-data.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

class Stage;

// ------------------------------------------------------------------
// Stage-provided GUI views
// ------------------------------------------------------------------
//
// A stage may ship its own front-end panel: an ES module (JavaScript +
// CSS + message catalogue) that is REGISTERED IN THE BACKEND and
// packaged with libvpipe (see ui/ui-view-registry.h), rather than
// living in the web-ui application. The view and its backend talk to
// each other over a FlexData message protocol they alone define; the
// front-end application (web-ui) only FORWARDS those messages and
// serves the module bytes -- it knows nothing about their content.
//
// The pieces:
//   UiViewChannel     -- backend -> view. Send a FlexData message, or a
//                        bulk binary payload carrying a FlexData header.
//   UiViewBackendIntf -- the stage-side handler for ONE mounted view.
//                        One instance per open panel.
//   UiViewHostIntf    -- what a backend may ask of the hosting
//                        application: which stages of a type exist and
//                        which are live. Implemented by the app (web-ui
//                        SessionApi); reached from a Session's UI
//                        delegate via UiDelegateIntf::ui_view_host().
//
// Threading: a channel's send*() methods are safe to call from any
// thread and never block on the client (they enqueue). A backend's
// on_*() methods are all invoked on the same connection thread, so a
// backend needs no internal locking against itself -- but it MAY be
// called concurrently with its own producer threads, which it owns.

// Backend -> front end. Handed to the backend on every callback; valid
// only for the duration of that call unless the backend captures the
// owning session (see UiViewBackendIntf::on_open).
class UiViewChannel {
public:
  virtual ~UiViewChannel() = default;

  // Send one structured message. Delivered to the view's onMessage.
  virtual void send(const FlexData& msg) = 0;

  // Send one BULK BINARY payload with a structured header -- the path
  // for media (encoded video fragments, images, PCM) that must not pay
  // JSON encoding. Delivered to the view's onBinary(header, bytes).
  virtual void send_binary(const FlexData&      header,
                           const std::uint8_t*  data,
                           std::size_t          size) = 0;

  // False once the client has gone away; a producer loop should stop.
  virtual bool alive() const = 0;
};

// The stage-side backend for one mounted view. Created per open panel
// by UiViewSpec::make_backend and destroyed when the panel closes.
class UiViewBackendIntf {
public:
  virtual ~UiViewBackendIntf() = default;

  // The view mounted and its channel is open. `ch` outlives the
  // backend, so a backend that pushes asynchronously may retain it.
  virtual void on_open(UiViewChannel& /*ch*/) {}

  // One message from the view. `msg` is whatever the view sent.
  virtual void on_message(UiViewChannel& ch, const FlexData& msg) = 0;

  // The panel closed / the client disconnected. Called exactly once,
  // before destruction; the channel is no longer usable.
  virtual void on_close() {}
};

// What a view backend may ask of the hosting application. The app owns
// the set of loaded pipelines, so stage discovery lives here rather
// than in libvpipe. Implementations must be safe to call from any
// thread.
class UiViewHostIntf {
public:
  virtual ~UiViewHostIntf() = default;

  // One stage of the requested type, in some loaded pipeline.
  struct StageRef {
    std::string pipeline;    // owning pipeline id
    std::string stage;       // stage id within it
    std::string state;       // pipeline state ("stopped"/"running"/...)
    bool        live = false;  // the pipeline is launched (or paused)
    // The stage's declared configuration object, so a backend can read
    // its own keys (a display title, a mode flag) for a stage that is
    // not running and has no live object to interrogate.
    FlexData    config;
  };

  // Every stage of `type_name` across the loaded pipelines, running or
  // not. Order is unspecified.
  virtual std::vector<StageRef>
  find_stages(std::string_view type_name) const = 0;

  // The live Stage object for (pipeline, stage), or nullptr when the
  // pipeline is not launched or the ids don't resolve. The pointer is
  // only valid while the pipeline stays launched -- a backend that
  // needs to outlive that must take a shared handle from the stage
  // (as the preview view does with its PreviewChannel).
  virtual Stage*
  live_stage(std::string_view pipeline, std::string_view stage) const = 0;
};

}

#endif
