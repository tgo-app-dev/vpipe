#ifndef COMMON_COMPARE_IMAGE_CHANNEL_H
#define COMMON_COMPARE_IMAGE_CHANNEL_H

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace vpipe {

// Backend -> GUI transport for the compare-image stage.
//
// Unlike PreviewChannel this is not a stream: there is no per-subscriber
// queue and no history, because a comparison has no notion of a dropped
// frame -- only "the latest A and B". The stage LATCHES the current pair
// and bumps a version; readers wait for the version to move and take
// whatever is current. A reader that falls behind simply skips the
// intermediate pairs, which is the correct behaviour here and is why
// this needs none of PreviewChannel's backlog machinery.
//
// Both images are published already padded to a COMMON size (see
// CompareImageStage), so a consumer never has to reconcile two
// resolutions: the pair is directly overlayable, which is what makes the
// split/wipe modes in the view exact.
//
// Thread-safe: the stage's coroutine publishes; any number of reader
// threads (a view backend's watch loop) wait. Held by shared_ptr so a
// reader outlives the stage's teardown; close() then ends it promptly.
class CompareImageChannel {
public:
  using Png = std::shared_ptr<const std::vector<std::uint8_t>>;

  // One latched comparison pair. A null `a` / `b` means that input has
  // produced nothing yet (the view shows black for it). `version` is 0
  // only before the first publish.
  struct Snapshot {
    Png           a;
    Png           b;
    int           width   = 0;   // common padded size, 0 when empty
    int           height  = 0;
    std::uint64_t version = 0;
    bool          closed  = false;
  };

  CompareImageChannel() = default;
  CompareImageChannel(const CompareImageChannel&)            = delete;
  CompareImageChannel& operator=(const CompareImageChannel&) = delete;

  // One side as the producer holds it: packed RGB24 at its own size.
  struct Picture {
    std::shared_ptr<const std::vector<std::uint8_t>> rgb;
    int width  = 0;
    int height = 0;
  };
  // Turns a side into the PNG the view shows at the common size, padded
  // to it by the producer's policy; null when it cannot. Runs on a
  // READER's thread, so it must hold nothing of the producer's but what
  // it captured -- and it must stay callable for as long as the channel
  // lives.
  using Render =
      std::function<Png(const Picture& side, int width, int height)>;

  // ---- producer side ------------------------------------------------

  // Latch a new pair (either image may be null) at the common size and
  // wake every waiter. Each call bumps the version, so republishing an
  // identical pair still notifies.
  void publish(Png a, Png b, int width, int height);

  // The same for a pair still to be encoded: a side becomes a PNG only
  // when a reader asks for the pair (snapshot / wait_change), once, on
  // that reader's thread. A graph nobody watches -- every run without
  // the web UI -- encodes nothing, and a producer never waits on a PNG.
  // A side whose picture (the same `rgb`) and common size are the last
  // publish's keeps the PNG it had.
  void publish_pictures(Picture a, Picture b, int width, int height,
                        std::shared_ptr<const Render> render);

  // End the channel: waiters wake with closed=true. Idempotent.
  void close();

  // ---- reader side --------------------------------------------------

  Snapshot snapshot() const;

  // Block up to `timeout_ms` for the version to differ from `since`,
  // then return the current snapshot (whose version equals `since` on
  // timeout). Returns immediately when the channel is closed.
  Snapshot wait_change(std::uint64_t since, int timeout_ms) const;

  bool closed() const noexcept;

private:
  // A side published as a picture, and whether its PNG is still owed.
  struct Pending {
    Picture pic;
    bool    owed = false;
  };

  // Encode whatever the current pair still owes, then return it.
  Snapshot render_owed_() const;

  mutable std::mutex              _mu;
  mutable std::condition_variable _cv;
  // Serialises the renders: two readers asking at once encode a side
  // once. Never held with _mu.
  mutable std::mutex              _render_mu;
  mutable Snapshot                _cur;
  mutable Pending                 _pa, _pb;
  std::shared_ptr<const Render>   _render;
};

// Abstract interface a stage implements to expose its comparison pair
// WITHOUT its GUI view depending on the concrete (ffmpeg-heavy) stage
// type -- the same split PreviewSource makes for the preview stage. The
// view backend dynamic_casts a live Stage to this.
class CompareImageSource {
public:
  virtual ~CompareImageSource() = default;
  virtual std::shared_ptr<CompareImageChannel> compare_channel() const = 0;
};

}

#endif
