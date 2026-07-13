#ifndef COMMON_PREVIEW_CHANNEL_H
#define COMMON_PREVIEW_CHANNEL_H

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vpipe {

// Low-latency backend -> web-ui preview transport. A PreviewStage encodes
// its input as fragmented MP4 (H.264) + passes PCM audio through, and
// pushes both into a PreviewChannel; the web-ui's HTTP server subscribes
// and relays the messages to the browser over a WebSocket, where Media
// Source Extensions (video) + WebAudio (audio) render them. MSE is not
// gated to secure contexts, so this plays over plain-HTTP LAN origins --
// unlike WebCodecs, it needs no HTTPS.
//
// Wire: each message is one WebSocket binary frame = a 1-byte type tag
// followed by the payload (the WebSocket frame carries the length):
//
//   type 1 (config):   UTF-8 JSON, e.g.
//       {"video":{"codec":"avc1.640028","width":1280,"height":720},
//        "audio":{"sampleRate":48000,"channels":1}}
//     Sent first to every subscriber and again if it changes. "video" /
//     "audio" present only when that substream exists.
//   type 2 (init):     the fMP4 initialization segment (ftyp + moov).
//     Retained and replayed to each new subscriber before any fragment;
//     re-sent to all when the stream re-initializes (a resolution change),
//     at which point the browser rebuilds its MediaSource.
//   type 3 (fragment): one fMP4 media fragment (moof + mdat), each
//     starting on a keyframe so it is appendable right after the init.
//   type 4 (audio):    planar float32, `channels` planes of `frames`
//     samples (frames = payload_bytes / (channels*4)); channels +
//     sampleRate come from the config message.
//
// Thread-safe: one producer (the stage's cadence coroutine) calls the
// set_*/push_*/close methods; any number of subscriber threads (the HTTP
// server's per-connection threads) wait on their own queue. Held by
// shared_ptr so a subscriber outlives the stage's teardown; close() then
// ends it promptly.
class PreviewChannel {
public:
  static constexpr std::uint8_t kMsgConfig   = 1;
  static constexpr std::uint8_t kMsgInit     = 2;
  static constexpr std::uint8_t kMsgFragment = 3;
  static constexpr std::uint8_t kMsgAudio    = 4;

  // Opaque per-connection subscription; created by subscribe(), passed
  // back to wait_frame() / unsubscribe(). Defined in the .cc.
  struct Subscriber;

  PreviewChannel() = default;
  ~PreviewChannel();
  PreviewChannel(const PreviewChannel&)            = delete;
  PreviewChannel& operator=(const PreviewChannel&) = delete;

  // ---- producer side ------------------------------------------------

  // Latch + broadcast the config. `json` is the config payload; the flags
  // + dims feed stream discovery. Only a changed json is re-sent; the
  // discovery fields always update.
  void set_config(std::string json, bool has_video, bool has_audio,
                  int width, int height);

  // Latch + broadcast the fMP4 init segment. Replayed to every future
  // subscriber; re-broadcast to current ones (they rebuild MediaSource).
  void set_init(const std::uint8_t* data, std::size_t n);

  // Publish one fMP4 media fragment (moof + mdat).
  void push_fragment(const std::uint8_t* data, std::size_t n);

  // Publish one PCM chunk (planes[c] = `frames` float samples for ch c).
  void push_audio(const float* const* planes, int channels, int frames);

  // End the stream: every subscriber's wait_frame() returns null once its
  // queue drains. Idempotent.
  void close();

  // ---- discovery (any thread) ---------------------------------------

  bool has_video() const noexcept;
  bool has_audio() const noexcept;
  int  width()     const noexcept;
  int  height()    const noexcept;
  bool closed()    const noexcept;

  // ---- subscriber side ----------------------------------------------

  // Register a connection. Seeded with the current config + init messages.
  // Never null.
  std::shared_ptr<Subscriber> subscribe();
  void unsubscribe(const std::shared_ptr<Subscriber>& sub);

  // Block up to `timeout_ms` for the next message for `sub`. Returns the
  // blob (shared, zero-copy) or null on timeout / closed-and-drained --
  // distinguish via closed().
  std::shared_ptr<const std::vector<std::uint8_t>>
  wait_frame(const std::shared_ptr<Subscriber>& sub, int timeout_ms);

private:
  using Blob = std::shared_ptr<const std::vector<std::uint8_t>>;

  // Serialize one message: a 1-byte type tag + payload.
  static Blob msg_(std::uint8_t type, const std::uint8_t* payload,
                   std::size_t n);

  // Enqueue a ready blob to every subscriber. Caller holds _mu.
  void broadcast_(const Blob& blob);

  mutable std::mutex                       _mu;
  std::condition_variable                  _cv;
  std::vector<std::shared_ptr<Subscriber>> _subs;
  std::string                              _config_json;
  Blob                                     _config_blob;   // retained
  Blob                                     _init_blob;     // retained
  bool                                     _has_video = false;
  bool                                     _has_audio = false;
  int                                      _width     = 0;
  int                                      _height    = 0;
  bool                                     _closed    = false;
};

// Abstract interface a stage implements to expose its preview channel to
// the web-ui WITHOUT the web-ui depending on the concrete (ffmpeg-heavy)
// stage type. The web-ui's streaming route dynamic_casts a live Stage to
// this and relays the channel.
class PreviewSource {
public:
  virtual ~PreviewSource() = default;
  virtual std::shared_ptr<PreviewChannel> preview_channel() const = 0;
};

}

#endif
