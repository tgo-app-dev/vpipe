#ifndef VPIPE_GENERATIVE_MODELS_VIDEO_TURN_H
#define VPIPE_GENERATIVE_MODELS_VIDEO_TURN_H

#include "common/flex-data.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {
class SessionContextIntf;
}

namespace vpipe::genai {

class LoadedLanguageModel;

// A VISION-LANGUAGE MODEL watching a clip and answering about it, for a
// stage outside libvpipe -- a plugin -- that sees neither the model's
// vision tower nor its chat template:
//
//   auto turn = VideoTurn::make(lm, session());
//   for (each frame) turn->add_frame(rgb, h, w, seconds);
//   std::string words = turn->reply("", "What happens here?", ...);
//
// Each frame goes through the family's own tower as it is added (the
// Qwen3-VL tower of Qwen3.5, Gemma-4's, Pixtral's: whichever the model
// carries). The reply lays the frames into the family's own chat
// template as ONE video in a user turn -- each frame marked with its
// time, as the family's processor marks a video's frames -- after
// `before` and followed by `after`, prefills once, and decodes the answer
// with thinking off: MTP-drafted where the model carries a head (token
// for token the same reply), else the pipelined decode. The frames go
// with the reply, so one turn serves a clip's scenes one after another.
//
// WHY THIS AND NOT THE TOWERS. The towers, the video placeholders, the
// mROPE grids and the templates are each family's own, and change as fast
// as the models do; a stage needs this much of them. It is realtime-vqa's
// describe step without the camera's clock, the questions' fan-out and
// the database -- for a host reading a FILE, whose times are the clip's.
//
// One caller at a time (the stage that owns it); it holds the model for
// its own lifetime. Feature VPIPE_FEATURE_VIDEO_TURN (plugin/plugin-abi.h).
class VideoTurn {
  VPIPE_ABI_OPAQUE;   // host-owned: see vpipe/export.h
public:
  // The model's turn; null -- and why, in the session's log -- when the
  // model has no vision tower or no chat template to put a video in.
  static std::unique_ptr<VideoTurn>
  make(std::shared_ptr<LoadedLanguageModel> lm,
       const SessionContextIntf*            session);
  ~VideoTurn();

  VideoTurn(const VideoTurn&)            = delete;
  VideoTurn& operator=(const VideoTurn&) = delete;

  // One frame -- planar 8-bit RGB [3, h, w], `seconds` into the clip --
  // encoded now. Frames are added in time order. False (and logged) when
  // the tower refused it; the turn holds what it had.
  bool add_frame(const std::uint8_t* rgb, int h, int w, double seconds);

  // The frames added since the last reply, and their vision tokens: what
  // the next prefill will hold (a caller closes a scene before it grows
  // past what it can afford).
  int frames() const noexcept;
  int tokens() const noexcept;

  // The answer: the frames as a video between `before` and `after` (with
  // no frames, a turn of words alone), decoded to the model's stop or
  // `max_tokens`. `sampler` is a token-sampler spec -- what a
  // `sampler-select` stage emits (spec-keys.h token_sampler_spec); null
  // or empty: greedy. `piece`, when set, hears each decoded piece as it
  // comes; false from it stops the decode there, and what came so far is
  // the answer. "" when it failed (logged). The frames are dropped either
  // way.
  std::string reply(std::string_view before, std::string_view after,
                    const FlexData* sampler, int max_tokens,
                    const std::function<bool(std::string_view)>& piece = {});

  // The frames dropped, with no reply.
  void clear();

  // The last reply in numbers: {"prompt_tokens", "tokens", "frames",
  // "prefill_s", "decode_s"}; an empty object before the first.
  FlexData last_stats() const;

private:
  struct Impl;
  explicit VideoTurn(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> _impl;
};

}

VPIPE_API_END

#endif
