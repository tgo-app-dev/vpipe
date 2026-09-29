#ifndef COMMON_MASK_EDITOR_CHANNEL_H
#define COMMON_MASK_EDITOR_CHANNEL_H

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vpipe {

// Transport from the create-mask stage to its GUI mask editor: a Frame
// carrying the background to paint over, the mask to start from, the
// canvas geometry, and the editor settings the stage was configured
// with. Latest-wins behind a version, exactly like CompareImageChannel
// -- an editor that falls behind wants the current background, never
// the intermediate ones.
//
// The OTHER direction -- the mask the user painted -- is not here. It
// is the stage's `commit` COMMAND (pipeline/stage-command.h), the same
// one a program sends: the view backend opens it with the PNG the
// browser's canvas produced, and the reply arrives once the beat is
// out. A queue of commands, not a latch, so two commits are two beats
// however close together they land.
//
// Thread-safe. The stage's coroutine publishes frames; the view
// backend's thread waits for them. Held by shared_ptr so a mounted
// panel outlives the stage's teardown; close() then ends it promptly.
class MaskEditorChannel {
public:
  using Bytes = std::shared_ptr<const std::vector<std::uint8_t>>;

  // How the mask is interpreted. Fixed by the stage's `mask_mode`
  // config and pushed to the editor so the brush behaves accordingly.
  enum class Mode : std::uint8_t {
    Binary = 0,   // two-state: a sample is 0 or 255
    Alpha  = 1,   // soft: 0..255 coverage, brush hardness shapes the falloff
    Class  = 2,   // multi-class: a sample is a class index 0..classes-1
  };

  // The stage's configuration, as far as the editor needs to know it.
  // Plain fields rather than a FlexData blob: the view backend is the
  // only translator, and it translates once, to JSON.
  struct Editor {
    Mode mode    = Mode::Binary;
    int  classes = 2;              // Class mode only; index 0 = background
    // 0x00RRGGBB per class in Class mode; a single entry (the overlay
    // colour) in Binary / Alpha mode.
    std::vector<std::uint32_t> colors;
    float overlay_opacity = 0.5f;
    // False when the stage was configured to run without the editor.
    // The panel still shows the mask; it just cannot commit one.
    bool  interactive = true;
  };

  // Stage -> editor. A null `background` means no reference image (the
  // editor paints over black); a null `mask` means "start empty".
  struct Frame {
    Bytes         background;          // PNG, RGB24
    Bytes         mask;                // PNG, GRAY8
    int           bg_width  = 0;       // background's own size
    int           bg_height = 0;
    int           width     = 0;       // mask canvas size
    int           height    = 0;
    Editor        editor;
    std::uint64_t version = 0;         // 0 only before the first publish
    bool          closed  = false;
  };

  MaskEditorChannel() = default;
  MaskEditorChannel(const MaskEditorChannel&)            = delete;
  MaskEditorChannel& operator=(const MaskEditorChannel&) = delete;

  // ---- stage side ----------------------------------------------------

  // Latch a new frame and wake every waiter. Each call bumps the
  // version, so republishing an identical frame still notifies.
  void publish(Frame f);

  // End the channel: every waiter wakes. Idempotent.
  void close();

  // ---- editor side ---------------------------------------------------

  Frame snapshot() const;

  // Block up to `timeout_ms` for the frame version to differ from
  // `since`, then return the current frame (whose version equals
  // `since` on timeout). Returns immediately when closed.
  Frame wait_change(std::uint64_t since, int timeout_ms) const;

  bool closed() const noexcept;

private:
  mutable std::mutex              _mu;
  mutable std::condition_variable _cv;
  Frame                           _frame;
  bool                            _closed = false;
};

// Abstract interface a stage implements to expose its editor channel
// WITHOUT its GUI view depending on the concrete (ffmpeg-heavy) stage
// type -- the same split PreviewSource and CompareImageSource make. The
// view backend dynamic_casts a live Stage to this.
class MaskEditorSource {
public:
  virtual ~MaskEditorSource() = default;
  virtual std::shared_ptr<MaskEditorChannel> mask_channel() const = 0;
};

}

#endif
