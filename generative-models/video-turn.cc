#include "generative-models/video-turn.h"

#include "common/vpipe-format.h"
#include "generative-models/chat-template.h"
#include "generative-models/gemma4/gemma4-unified-embedder.h"
#include "generative-models/gemma4/metal-gemma4-vision.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/mistral3/metal-pixtral-vision.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/sampler.h"
#include "generative-models/token-muxer.h"
#include "generative-models/tokenizer.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vpipe::genai {

namespace {

using clock = std::chrono::steady_clock;

double
seconds_since(clock::time_point t0)
{
  return std::chrono::duration<double>(clock::now() - t0).count();
}

// Pixtral sizes a picture by its longest edge, ~1200 tokens at camera
// sizes; a frame of a video gets what the other towers give one
// (realtime-vqa's cap, for the same reason: a scene of them must fit).
constexpr int kPixtralFrameTokens = 256;

// One frame through the tower: its rows (native f16 on the GPU, or host
// f32 for the encoder-less Gemma-4 embedder), its POST-merger grid, and
// its time.
struct Frame {
  metal_compute::SharedBuffer embeddings;
  std::vector<float>          embeddings_host;
  int                         n_tokens = 0;
  int                         mh = 0;
  int                         mw = 0;
  double                      seconds = 0;
};

// What a turn holds. Kept out of the class -- every member of which is
// exported -- so the API is the class and its insides stay the
// library's own.
struct TurnState {
  std::shared_ptr<LoadedLanguageModel> lm;
  const SessionContextIntf*            session = nullptr;
  std::unique_ptr<ChatTemplate>        tpl;
  // Borrowed from the model: whichever tower it carries.
  MetalQwenVisionEncoder*              qwen = nullptr;
  MetalGemma4VisionEncoder*            gemma = nullptr;
  MetalPixtralVisionEncoder*           pixtral = nullptr;
  Gemma4UnifiedEmbedder*               unified = nullptr;
  std::vector<Frame>                   frames;
  int                                  tokens = 0;
  FlexData                             stats = FlexData::make_object();
};

void
warn(const TurnState& m, const VpipeFormat& f)
{
  if (m.session) { m.session->warn(f); }
}

bool
encode(TurnState& m, const std::uint8_t* rgb, int h, int w, Frame* out)
{
  if (m.qwen) {
    // The metal tower reports PRE-merger patches.
    const int s = std::max(1, m.qwen->config().spatial_merge);
    auto r = m.qwen->encode(rgb, h, w);
    out->embeddings = std::move(r.embeddings);
    out->n_tokens   = r.n_tokens;
    out->mh         = r.grid_h / s;
    out->mw         = r.grid_w / s;
  } else if (m.gemma) {
    // The still budget: the dense-video one (~64 a frame) drew a frame
    // too coarsely (realtime-vqa found vague, made-up scenes).
    auto r = m.gemma->encode(rgb, h, w, -1);
    out->embeddings = std::move(r.embeddings);
    out->n_tokens   = r.n_tokens;
    out->mh         = r.grid_h;
    out->mw         = r.grid_w;
  } else if (m.pixtral) {
    auto r = m.pixtral->encode(rgb, h, w, kPixtralFrameTokens);
    out->embeddings = std::move(r.embeddings);
    out->n_tokens   = r.n_tokens;
    out->mh         = r.grid_h;
    out->mw         = r.grid_w;
  } else if (m.unified) {
    if (auto r = m.unified->encode_image(rgb, h, w)) {
      out->embeddings_host = std::move(r->rows);
      out->n_tokens        = r->n_tokens;
      out->mh              = r->grid_h;
      out->mw              = r->grid_w;
    }
  }
  return out->n_tokens > 0 &&
         (!out->embeddings.empty() || !out->embeddings_host.empty());
}

// One branch decoded, as text-chat decodes a turn: the drafter the model
// carries -- a DFlash block drafter attached, else its MTP head --
// drafting ahead (token for token the reply the plain decode gives),
// else the pipelined decode, else one token at a time.
std::string
decode(TurnState& m, LoadedLanguageModel::Context& ctx,
       const SamplerParams& sp, int max_tokens,
       const std::function<bool(std::string_view)>& piece, int* produced)
{
  LoadedLanguageModel& lm = *m.lm;
  const ChatTemplate& tpl = *m.tpl;
  const auto& tok = lm.tokenizer();
  auto sd = tok.make_stream_decoder();
  std::string out;
  int n = 0;
  bool go = true;
  const auto take = [&](std::int32_t id) {
    std::string s = tok.step(sd, id);
    ++n;
    if (!s.empty()) {
      out += s;
      if (piece && !piece(s)) { go = false; }
    }
  };
  const auto is_stop = [&tpl](std::int32_t id) {
    return tpl.is_stop_token(id);
  };
  std::int32_t cur = ctx.last_predicted_id();
  if (cur >= 0 && lm.spec_decode_available()) {
    lm.spec_generate(ctx, cur, max_tokens, sp, is_stop,
                     [&](std::span<const std::int32_t> ids) {
                       for (std::int32_t id : ids) {
                         take(id);
                         if (!go) { break; }
                       }
                       return go;
                     });
  } else if (lm.pdecode_begin(ctx, cur, {}, sp, max_tokens)) {
    // Drain before refill, as realtime-vqa learned: a depth-1 ring
    // (Qwen3.5) overflowed when the commit came first.
    const bool runahead = lm.pdecode_supports_runahead();
    bool committed = cur >= 0 && !is_stop(cur) ? lm.pdecode_commit(ctx)
                                                : false;
    if (runahead && committed && max_tokens > 1) {
      lm.pdecode_commit(ctx);
    }
    while (go && n < max_tokens && cur >= 0 && !is_stop(cur)) {
      take(cur);
      if (n >= max_tokens || !committed) { break; }
      cur = lm.pdecode_next(ctx);
      if (cur < 0) { break; }
      const bool more = n + 1 < max_tokens && !is_stop(cur);
      committed = more ? lm.pdecode_commit(ctx) : false;
    }
    lm.pdecode_end(ctx);
  } else {
    Sampler sampler(sp);
    if (cur >= 0) { sampler.prime(std::span<const std::int32_t>(&cur, 1)); }
    while (go && cur >= 0 && n < max_tokens && !is_stop(cur)) {
      take(cur);
      if (n >= max_tokens) { break; }
      const std::int32_t am = lm.next_token(ctx, cur);
      if (am < 0) { break; }
      const auto& logits = lm.last_logits_host();
      cur = sampler.is_argmax() || logits.empty()
                ? am
                : sampler.sample(std::span<const float>(logits.data(),
                                                        logits.size()));
    }
  }
  *produced = n;
  return tpl.sanitize_output(std::move(out));
}

}

struct VideoTurn::Impl : TurnState {};

std::unique_ptr<VideoTurn>
VideoTurn::make(std::shared_ptr<LoadedLanguageModel> lm,
                const SessionContextIntf*            session)
{
  if (!lm || !lm->valid()) {
    if (session) { session->warn(fmt("VideoTurn: no model")); }
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->session = session;
  impl->qwen    = lm->metal_vision_encoder();
  impl->gemma   = lm->metal_gemma4_vision_encoder();
  impl->pixtral = lm->metal_pixtral_vision_encoder();
  impl->unified = lm->gemma4_unified_embedder();
  if (impl->unified && !impl->unified->has_vision()) {
    impl->unified = nullptr;
  }
  if (!impl->qwen && !impl->gemma && !impl->pixtral && !impl->unified) {
    if (session) {
      session->warn(fmt("VideoTurn: '{}' has no vision tower on the "
                        "metal backend", lm->config().architecture));
    }
    return nullptr;
  }
  // Thinking off: a description, not a deliberation, and the budget is
  // the answer's.
  impl->tpl = make_chat_template(lm->config(), lm->tokenizer(),
                                 /*disable_thinking=*/true);
  if (!impl->tpl || (impl->tpl->video_pad_token_id() < 0 &&
                     impl->tpl->image_pad_token_id() < 0)) {
    if (session) {
      session->warn(fmt("VideoTurn: no chat template puts a video in "
                        "'{}'", lm->config().architecture));
    }
    return nullptr;
  }
  impl->lm = std::move(lm);
  return std::unique_ptr<VideoTurn>(new VideoTurn(std::move(impl)));
}

VideoTurn::VideoTurn(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}

VideoTurn::~VideoTurn() = default;

bool
VideoTurn::add_frame(const std::uint8_t* rgb, int h, int w, double seconds)
{
  if (rgb == nullptr || h <= 0 || w <= 0) {
    warn(*_impl, fmt("VideoTurn: a frame of {}x{} refused", w, h));
    return false;
  }
  const auto t0 = clock::now();
  Frame f;
  if (!encode(*_impl, rgb, h, w, &f)) {
    warn(*_impl, fmt("VideoTurn: the tower returned nothing for a "
                     "{}x{} frame at {:.2f} s", w, h, seconds));
    return false;
  }
  f.seconds = seconds;
  if (_impl->session) {
    _impl->session->log_normal(fmt(
        "VideoTurn: frame {} at {:.2f} s, {}x{} -> {} tokens in {:.3f} s",
        _impl->frames.size(), seconds, w, h, f.n_tokens,
        seconds_since(t0)));
  }
  _impl->tokens += f.n_tokens;
  _impl->frames.push_back(std::move(f));
  return true;
}

int
VideoTurn::frames() const noexcept
{
  return static_cast<int>(_impl->frames.size());
}

int
VideoTurn::tokens() const noexcept
{
  return _impl->tokens;
}

void
VideoTurn::clear()
{
  _impl->frames.clear();
  _impl->tokens = 0;
}

FlexData
VideoTurn::last_stats() const
{
  return _impl->stats;
}

std::string
VideoTurn::reply(std::string_view before, std::string_view after,
                 const FlexData* sampler, int max_tokens,
                 const std::function<bool(std::string_view)>& piece)
{
  TurnState& m = *_impl;
  std::vector<Frame> frames = std::move(m.frames);
  m.frames.clear();
  m.tokens = 0;
  const ChatTemplate& tpl = *m.tpl;
  max_tokens = std::max(1, max_tokens);

  // The prompt: the frames as one video, each marked with its time --
  // the template's own video turn -- between the words.
  std::vector<std::int32_t> ids;
  std::vector<float> times;
  std::vector<int> counts;
  std::vector<std::pair<int, int>> grids;
  for (const auto& f : frames) {
    times.push_back(static_cast<float>(f.seconds));
    counts.push_back(f.n_tokens);
    grids.emplace_back(f.mh, f.mw);
  }
  if (frames.empty()) {
    std::string words(before);
    if (!words.empty() && !after.empty()) { words += "\n\n"; }
    words += after;
    tpl.render_user_turn(words, /*is_first_turn=*/true, &ids);
  } else {
    bool split = tpl.render_video_prefix(times, counts,
                                         /*is_first_turn=*/true, before,
                                         &ids);
    if (split && !tpl.render_vlm_completion(after, &ids)) {
      split = false;
    }
    if (!split) {
      ids.clear();
      tpl.render_user_turn_video(after, times, counts,
                                 /*is_first_turn=*/true, before, &ids);
    }
  }
  // The pads take the frames' rows in order.
  std::int32_t pad = tpl.video_pad_token_id();
  if (pad < 0) { pad = tpl.image_pad_token_id(); }
  std::vector<TokenRef> refs;
  refs.reserve(ids.size());
  std::size_t fi = 0;
  int row = 0;
  for (std::int32_t id : ids) {
    TokenRef r;
    if (id == pad && fi < frames.size()) {
      r.kind = TokenRef::Kind::ImageTokens;
      if (!frames[fi].embeddings_host.empty()) {
        r.embeddings_host = &frames[fi].embeddings_host;
      } else {
        r.embeddings_buf = &frames[fi].embeddings;
      }
      r.image_token_offset = row++;
      if (row >= frames[fi].n_tokens) {
        ++fi;
        row = 0;
      }
    } else {
      r.kind = TokenRef::Kind::Text;
      r.text_id = id;
    }
    refs.push_back(r);
  }
  if (fi < frames.size()) {
    warn(m, fmt("VideoTurn: the template placed {} of {} frames",
                fi, frames.size()));
    return {};
  }

  auto wired = m.lm->wired_scope();
  auto ctx = m.lm->make_context();
  if (!ctx.valid()) {
    warn(m, fmt("VideoTurn: no context for the turn"));
    return {};
  }
  const auto t0 = clock::now();
  const std::int32_t first =
      frames.empty()
          ? m.lm->prefill(ctx, std::span<const std::int32_t>(ids))
          : m.lm->prefill_multimodal_metal(
                ctx, std::span<const TokenRef>(refs),
                std::span<const std::pair<int, int>>(grids));
  const double prefill_s = seconds_since(t0);
  if (first < 0) {
    warn(m, fmt("VideoTurn: the prefill of {} tokens ({} frames) failed",
                refs.size(), frames.size()));
    return {};
  }
  SamplerParams sp;
  if (sampler != nullptr && sampler->is_object()) {
    sp = parse_sampler_config(*sampler);
  }
  // The prefill decided the first token: greedy is its argmax; a
  // sampler draws it from the prefill's logits instead.
  if (!Sampler(sp).is_argmax()) {
    const auto& logits = m.lm->last_logits_host();
    if (!logits.empty()) {
      Sampler s(sp);
      ctx.set_last_predicted_id(s.sample(
          std::span<const float>(logits.data(), logits.size())));
    }
  }
  const auto t1 = clock::now();
  int produced = 0;
  std::string text = decode(m, ctx, sp, max_tokens, piece, &produced);
  const double decode_s = seconds_since(t1);
  if (m.session) {
    m.session->info(fmt(
        "VideoTurn: {} frames, {} prompt tokens in {:.2f} s "
        "({:.0f} tok/s); {} tokens decoded in {:.2f} s", frames.size(),
        refs.size(), prefill_s,
        prefill_s > 0 ? static_cast<double>(refs.size()) / prefill_s : 0.0,
        produced, decode_s));
  }
  FlexData st = FlexData::make_object();
  auto o = st.as_object();
  o.insert("prompt_tokens",
           FlexData::make_uint(static_cast<std::uint64_t>(refs.size())));
  o.insert("tokens",
           FlexData::make_uint(static_cast<std::uint64_t>(produced)));
  o.insert("frames",
           FlexData::make_uint(static_cast<std::uint64_t>(frames.size())));
  o.insert("prefill_s", FlexData::make_real(prefill_s));
  o.insert("decode_s", FlexData::make_real(decode_s));
  m.stats = std::move(st);
  return text;
}

}
