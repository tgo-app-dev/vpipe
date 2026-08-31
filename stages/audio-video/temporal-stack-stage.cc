#include "stages/audio-video/temporal-stack-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace vpipe {

namespace {

constexpr ConfigKey kAttrs[] = {
  {.key = "mode", .type = ConfigType::String, .required = false,
   .doc = "how to stack: auto (default) | video | audio | generic. VIDEO "
          "adds a new LEADING axis ([3,H,W] x T -> [T,3,H,W]); AUDIO "
          "concatenates on the LAST axis ([N] x k -> [sum N], planar "
          "[2,N] x k -> [2,sum N]); GENERIC is video's rule without the "
          "modality checks. AUTO reads it off the first beat -- rank 3 u8 "
          "is a frame, rank <= 2 f32 or anything carrying a `sample_rate` "
          "is audio -- and LATCHES it, so a producer that changes shape "
          "mid-group is an error rather than a half-stacked tensor",
   .def_str = "auto"},
  {.key = "group_size", .type = ConfigType::Int, .required = false,
   .doc = "beats per emitted group. 0 (the default) accumulates until the "
          "source ends and emits ONE beat -- which is what a single "
          "reference wants, and which also makes this stage single-shot: "
          "after that beat it is finished. A graph that serves several "
          "requests must set this, or every request after the first sees "
          "a producer that has already ended",
   .def_int = 0},
  {.key = "overlap", .type = ConfigType::Int, .required = false,
   .doc = "beats of each emitted group kept so the NEXT group opens with "
          "them -- context that lets a consumer see what came just before. "
          "BEATS, not seconds, the same unit group_size counts: at 24 fps "
          "an overlap of 24 is one second. Must be < group_size, and the "
          "group then advances by (group_size - overlap) beats per emit. "
          "VIDEO and generic groups only -- for audio use audio-to-pcm's "
          "chunk_overlap_s, which works in samples. 0 (default) = groups "
          "share nothing",
   .def_int = 0},
  {.key = "max_mb", .type = ConfigType::Int, .required = false,
   .doc = "ceiling on the accumulator, in MB (default 256). Reaching it "
          "emits what is held, WARNS, and ends the stream -- the group is "
          "no longer the one that was asked for, and continuing would emit "
          "a run of quietly short ones. The ceiling is spatial in "
          "practice: 256 MB is 171 frames of 960x544 u8 RGB but only 43 of "
          "1080p and 10 of 4K, while f32 audio would need 35 minutes to "
          "reach it. Resample the frames before stacking rather than "
          "raising this",
   .def_int = 256},
  {.key = "sideband", .type = ConfigType::Any, .required = false,
   .doc = "a JSON object merged into every emitted group's sideband, after "
          "this stage's own rate fields. For saying something about the "
          "group that no producer of its beats could know -- MiniMax-H3's "
          "reference ports, for instance, read `attach: true` on an audio "
          "beat to make it the soundtrack of the preceding reference "
          "rather than a reference of its own, and audio-to-pcm has no "
          "notion of either. Keys given here WIN over the computed ones, "
          "so it can also pin a rate a source got wrong",
   .def_str = ""},
  {.key = "fps", .type = ConfigType::Real, .required = false,
   .doc = "video only: the frame rate to publish on the stacked beat, when "
          "the frames do not carry one. Frame beats state `fps_num`/"
          "`fps_den` only when the source knew them, so this stage resolves "
          "a rate from that pair, else from the span of `timestamp_us`, "
          "else from this. 0 (default) means do not invent one -- a "
          "consumer that needs a rate should refuse rather than be handed a "
          "plausible wrong one",
   .def_real = 0.0},
};

// No tags: `generic` mode exists to stack whatever a graph produces, and
// a declared tag set is a connection CONSTRAINT here, not a hint -- it
// would refuse exactly the untagged producer generic is for. The
// TensorBeatPayload type is still enforced.
const PortSpec kIports[] = {
  {.name = "beats",
   .doc = "TensorBeatPayload, one time slice per beat: planar u8 RGB "
          "[3,H,W] frames, f32 PCM [N] or [channels,N] chunks, or any "
          "tensor of one shape for generic mode",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "stacked",
   .doc = "ONE TensorBeatPayload per group: [beats, ...] for video and "
          "generic, [..., sum] for audio. The sideband is rebuilt for the "
          "group -- `timestamp_us` of its first beat, `sample_rate` and "
          "`duration_us` for audio, a resolved `fps` for video",
   .type = &typeid(TensorBeatPayload), .clock_group = 1},
};

const StageSpec kSpec = {
  .type_name = "temporal-stack",
  .doc       = "Stacks many time-slice beats into one tensor: frames into "
               "a clip, PCM chunks into a waveform. The bridge between "
               "this tree's streaming convention and a consumer that "
               "conditions on a whole reference. Crosses beat-rate -> "
               "group-rate clocks.",
  .display_name = "Temporal Stack",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

std::string
shape_str_(const std::vector<std::int64_t>& sh)
{
  std::string s = "[";
  for (std::size_t i = 0; i < sh.size(); ++i) {
    if (i > 0) { s += ", "; }
    s += std::to_string(sh[i]);
  }
  return s + "]";
}

bool
sb_num_(const FlexData& sb, const char* key, double* out)
{
  if (!sb.is_object()) { return false; }
  auto o = sb.as_object();
  if (!o.contains(key)) { return false; }
  FlexData v = o.at(key);
  if (v.is_int())  { *out = (double)v.as_int(0);  return true; }
  if (v.is_uint()) { *out = (double)v.as_uint(0); return true; }
  if (v.is_real()) { *out = v.as_real(0.0);       return true; }
  return false;
}

}  // namespace

TemporalStackStage::TemporalStackStage(const SessionContextIntf* s,
                                       std::string               id,
                                       std::vector<InEdge>       iports,
                                       FlexData                  config)
  : TypedStage<TemporalStackStage>(s, std::move(id), std::move(iports),
                                   std::move(config))
{
  const std::string m = std::string(attr_str("mode"));
  if      (m == "auto"    || m.empty()) { _mode_cfg = Mode::kAuto; }
  else if (m == "video")                { _mode_cfg = Mode::kVideo; }
  else if (m == "audio")                { _mode_cfg = Mode::kAudio; }
  else if (m == "generic")              { _mode_cfg = Mode::kGeneric; }
  else {
    fail_config(fmt(
        "TemporalStackStage('{}'): mode '{}' is not auto|video|audio|generic",
        this->id(), m));
  }
  _overlap = (int)attr_int("overlap");
  if (_overlap < 0) {
    session()->warn(fmt(
        "TemporalStackStage('{}'): overlap {} is negative; using 0",
        this->id(), _overlap));
    _overlap = 0;
  }
  _group_size = (int)attr_int("group_size");
  if (_group_size < 0) {
    fail_config(fmt(
        "TemporalStackStage('{}'): group_size {} is negative; 0 means "
        "accumulate to end-of-stream", this->id(), _group_size));
  }
  if (_overlap > 0 && _group_size > 0 && _overlap >= _group_size) {
    // The group would reopen already full, emit again with no new beat,
    // and never advance. Refused rather than clamped: the graph asked
    // for a hop of zero or less, and quietly running a different one is
    // how a wrong number becomes a runtime mystery.
    fail_config(fmt(
        "TemporalStackStage('{}'): overlap {} must be < group_size {} -- "
        "the group advances by (group_size - overlap) beats, so this asks "
        "it to advance by {}", this->id(), _overlap, _group_size,
        _group_size - _overlap));
  }
  if (_overlap > 0 && _group_size == 0) {
    session()->warn(fmt(
        "TemporalStackStage('{}'): overlap {} has no effect with "
        "group_size 0 -- that accumulates to end-of-stream, so there is "
        "no next group to carry beats into", this->id(), _overlap));
  }
  const int mb = (int)attr_int("max_mb");
  if (mb <= 0) {
    fail_config(fmt(
        "TemporalStackStage('{}'): max_mb {} must be positive", this->id(),
        mb));
  }
  _max_bytes = (std::size_t)std::max(1, mb) << 20;
  {
    auto c = this->config().as_object();
    if (c.contains("sideband")) {
      FlexData v = c.at("sideband");
      if (v.is_object()) {
        _sideband_cfg = v;
      } else if (!v.is_null() && !(v.is_string() && v.as_string("").empty())) {
        fail_config(fmt(
            "TemporalStackStage('{}'): sideband must be a JSON object",
            this->id()));
      }
    }
  }
  _fps_cfg = attr_real("fps");
  if (_fps_cfg < 0.0) {
    fail_config(fmt("TemporalStackStage('{}'): fps {} is negative",
                    this->id(), _fps_cfg));
  }
  _mode = _mode_cfg;
  allocate_oports(kSpec.oports.size());
}

TemporalStackStage::~TemporalStackStage() = default;

const StageSpec&
TemporalStackStage::spec() const noexcept
{
  return kSpec;
}

Job
TemporalStackStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  // Per-RUN state. Stages survive a stop/relaunch while the runtime does
  // not, so a half-built group left by the previous run would be
  // prepended to the first group of this one.
  reset_group_();
  _mode    = _mode_cfg;
  _emitted = 0;
  _capped  = false;
  co_return;
}

TemporalStackStage::Mode
TemporalStackStage::sense(const TensorBeat& tb)
{
  const int rank = (int)tb.shape.size();
  double v = 0.0;
  // A stated sample rate settles it before any shape guess: it is the
  // producer saying what this is, and audio-to-pcm always states it.
  if (sb_num_(tb.sideband, "sample_rate", &v) && v > 0.0) {
    return Mode::kAudio;
  }
  if (rank == 3 && tb.dtype == TensorBeat::DType::U8) { return Mode::kVideo; }
  if (rank <= 2 && tb.dtype == TensorBeat::DType::F32) {
    return Mode::kAudio;
  }
  return Mode::kGeneric;
}

void
TemporalStackStage::reset_group_()
{
  _tail_meta.clear();
  _buf.clear();
  _elem_shape.clear();
  _count        = 0;
  _tail         = 0;
  _first_sideband = FlexData::make_null();
  _has_first_ts = false;
  _first_ts_us  = 0;
  _last_ts_us   = 0;
  _sample_rate  = 0;
  _fps_num      = 0.0;
  _fps_den      = 0.0;
}

int
TemporalStackStage::retain_beats_() const noexcept
{
  if (_overlap <= 0 || _count <= 0) { return 0; }
  // Audio beats are variable-length on the time axis, so N beats is not
  // a duration; audio-to-pcm's chunk_overlap_s is the control that is.
  if (_mode == Mode::kAudio) { return 0; }
  // Never the whole group: reopening it full would emit again with no
  // new beat. The ctor already refuses overlap >= group_size, so this
  // only binds a SHORT group -- one cut off by EOS or the byte ceiling.
  return _overlap < _count ? _overlap : _count - 1;
}

void
TemporalStackStage::retain_tail_(int keep)
{
  const std::size_t per = (std::size_t)_count > 0
                              ? _buf.size() / (std::size_t)_count
                              : 0;
  if (keep <= 0 || per == 0) { reset_group_(); return; }
  const std::size_t drop = (std::size_t)(_count - keep) * per;
  _buf.erase(_buf.begin(), _buf.begin() + (std::ptrdiff_t)drop);
  _count = keep;
  _tail  = 0;   // audio never gets here; see retain_beats_

  // The rate metadata has to describe the beats that REMAIN. Anything
  // else dates the next clip to a frame it no longer starts on, and the
  // timestamp-derived fps is a span over beats that already went out.
  while ((int)_tail_meta.size() > keep) { _tail_meta.pop_front(); }
  _has_first_ts = false;
  _first_ts_us  = 0;
  _last_ts_us   = 0;
  if (!_tail_meta.empty()) {
    _first_sideband = _tail_meta.front().sb;
    for (const auto& t : _tail_meta) {
      if (!t.has_ts) { continue; }
      if (!_has_first_ts) { _first_ts_us = t.ts; }
      _has_first_ts = true;
      _last_ts_us   = t.ts;
    }
  }
  // _elem_shape, _dtype, _mode and the declared fps_num/fps_den are
  // properties of the SOURCE, not of a group, and stay latched.
}

bool
TemporalStackStage::append_(const TensorBeat& tb, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  // byte_size()/bytes_(), never `data` directly: a Shared (Metal-buffer)
  // beat leaves `data` EMPTY and puts its bytes in the external handle,
  // and video-to-rgb's fast path produces exactly that. Reading `data`
  // reads a beat with pixels in it as an empty one.
  if (tb.shape.empty() || tb.byte_size() == 0) {
    return fail("an empty tensor cannot join a group");
  }
  if (_count == 0) {
    if (_mode == Mode::kAuto) { _mode = sense(tb); }
    _elem_shape = tb.shape;
    _dtype      = tb.dtype;
  } else {
    if (tb.dtype != _dtype) {
      return fail("dtype changed mid-group");
    }
    if (_mode == Mode::kAudio) {
      // Only the time axis may differ between chunks.
      if (tb.shape.size() != _elem_shape.size()) {
        return fail("rank changed mid-group");
      }
      for (std::size_t i = 0; i + 1 < tb.shape.size(); ++i) {
        if (tb.shape[i] != _elem_shape[i]) {
          return fail("channel layout changed mid-group");
        }
      }
    } else if (tb.shape != _elem_shape) {
      return fail("shape changed mid-group");
    }
  }
  const std::size_t n = tb.byte_size();
  const std::size_t prev = _buf.size();
  _buf.resize(prev + n);
  std::memcpy(_buf.data() + prev, tb.bytes_(), n);
  if (_mode == Mode::kAudio && !tb.shape.empty()) {
    _tail += tb.shape.back();
  }
  ++_count;

  // Sideband: the first beat's is the group's, and the timestamps of
  // every beat feed the rate resolution.
  if (_count == 1) { _first_sideband = tb.sideband; }
  double v = 0.0;
  if (sb_num_(tb.sideband, "timestamp_us", &v) && v >= 0.0) {
    if (!_has_first_ts) { _first_ts_us = (std::uint64_t)v; }
    _has_first_ts = true;
    _last_ts_us   = (std::uint64_t)v;
  }
  if (sb_num_(tb.sideband, "sample_rate", &v) && v > 0.0) {
    _sample_rate = (int)v;
  }
  if (sb_num_(tb.sideband, "fps_num", &v) && v > 0.0) { _fps_num = v; }
  if (sb_num_(tb.sideband, "fps_den", &v) && v > 0.0) { _fps_den = v; }

  // The last `_overlap` beats' rate metadata, for the retention after
  // the next emit. Kept only when an overlap is configured, and bounded
  // by it, so the default path allocates nothing.
  if (_overlap > 0) {
    TailMeta tm;
    tm.sb = tb.sideband;
    double ts = 0.0;
    if (sb_num_(tb.sideband, "timestamp_us", &ts) && ts >= 0.0) {
      tm.has_ts = true;
      tm.ts     = (std::uint64_t)ts;
    }
    _tail_meta.push_back(std::move(tm));
    while ((int)_tail_meta.size() > _overlap) { _tail_meta.pop_front(); }
  }
  return true;
}

TensorBeat
TemporalStackStage::build_() const
{
  TensorBeat tb;
  if (_count == 0) { return tb; }
  tb.dtype = _dtype;
  if (_mode == Mode::kAudio) {
    tb.shape = _elem_shape;
    tb.shape.back() = _tail;
  } else {
    tb.shape.reserve(_elem_shape.size() + 1);
    tb.shape.push_back((std::int64_t)_count);
    for (std::int64_t d : _elem_shape) { tb.shape.push_back(d); }
  }
  tb.strides.clear();
  tb.data.resize(_buf.size());
  std::memcpy(tb.data.data(), _buf.data(), _buf.size());

  // PLANAR audio needs the halves interleaved back apart: the group
  // arrived as k chunks of [C, n], so the buffer is C-major WITHIN each
  // chunk and chunk-major overall, while [C, sum n] wants one whole
  // channel then the next. Mono (rank 1) is already right.
  if (_mode == Mode::kAudio && _elem_shape.size() == 2 &&
      _elem_shape[0] > 1) {
    const std::size_t esz = TensorBeat::byte_size_of(_dtype);
    const std::size_t ch  = (std::size_t)_elem_shape[0];
    const std::size_t per = (std::size_t)_elem_shape[1];   // per chunk
    std::vector<std::uint8_t> out(_buf.size());
    for (std::size_t c = 0; c < ch; ++c) {
      for (int k = 0; k < _count; ++k) {
        const std::uint8_t* src =
            _buf.data() + ((std::size_t)k * ch + c) * per * esz;
        std::uint8_t* dst =
            out.data() + (c * (std::size_t)_tail + (std::size_t)k * per) * esz;
        std::memcpy(dst, src, per * esz);
      }
    }
    std::memcpy(tb.data.data(), out.data(), out.size());
  }

  // The group's sideband, rebuilt. Carried from the first beat so that
  // anything this stage does not understand -- provenance, a camera name
  // -- survives, then the rate fields overwritten with the GROUP's.
  FlexData sb = _first_sideband.is_object() ? _first_sideband
                                            : FlexData::make_object();
  auto o = sb.as_object();
  if (_has_first_ts) {
    o.insert_or_assign("timestamp_us", FlexData::make_uint(_first_ts_us));
  }
  if (_mode == Mode::kAudio) {
    if (_sample_rate > 0) {
      o.insert_or_assign("sample_rate", FlexData::make_int(_sample_rate));
      o.insert_or_assign(
          "duration_us",
          FlexData::make_uint((std::uint64_t)_tail * 1'000'000ULL /
                              (std::uint64_t)_sample_rate));
    }
  } else if (_mode == Mode::kVideo) {
    double fps = 0.0;
    if (_fps_num > 0.0 && _fps_den > 0.0) {
      fps = _fps_num / _fps_den;
    } else if (_has_first_ts && _count > 1 && _last_ts_us > _first_ts_us) {
      fps = (double)(_count - 1) /
            ((double)(_last_ts_us - _first_ts_us) / 1e6);
    } else if (_fps_cfg > 0.0) {
      fps = _fps_cfg;
    }
    if (fps > 0.0) { o.insert_or_assign("fps", FlexData::make_real(fps)); }
  }
  o.insert_or_assign("stacked", FlexData::make_int(_count));
  // Config LAST, so it overrides what was computed. A user pinning a
  // rate is correcting this stage, not asking to be corrected by it.
  if (_sideband_cfg.is_object()) {
    FlexData extra = _sideband_cfg;
    auto e = extra.as_object();
    for (const auto& kv : e) { o.insert_or_assign(kv.first, kv.second); }
  }
  tb.sideband = std::move(sb);
  return tb;
}

Job
TemporalStackStage::emit_(RuntimeContext& ctx)
{
  if (_count == 0) { co_return; }
  const int n = _count;
  TensorBeat tb = build_();
  if (_mode == Mode::kVideo && !_fps_reported) {
    _fps_reported = true;
    double fps = 0.0;
    const char* how = "none";
    if (sb_num_(tb.sideband, "fps", &fps)) {
      how = (_fps_num > 0.0 && _fps_den > 0.0) ? "declared by the source"
            : (_count > 1 && _last_ts_us > _first_ts_us)
                ? "derived from timestamps"
                : "from config";
    }
    // Said once, and said out loud: a clip carries its rate nowhere else,
    // and a consumer that resamples onto a fixed rate has no way to tell
    // a measured 23.976 from a guessed 24.
    if (fps > 0.0) {
      session()->info(fmt("TemporalStackStage('{}'): {:.4g} fps, {}",
                          this->id(), fps, how));
    } else {
      session()->warn(fmt(
          "TemporalStackStage('{}'): the frames state no rate and none "
          "could be derived, so the stacked clip carries no `fps`. Set the "
          "`fps` config if the consumer needs one -- it is not invented "
          "here, because a plausible wrong rate conditions a model at the "
          "wrong speed with nothing to complain about", this->id()));
    }
  }
  if (_overlap > 0 && _mode == Mode::kAudio && !_overlap_audio_warned) {
    _overlap_audio_warned = true;
    session()->warn(fmt(
        "TemporalStackStage('{}'): overlap {} is ignored for an AUDIO "
        "group -- its beats are variable-length on the time axis, so N "
        "beats is not a duration. Set audio-to-pcm's chunk_overlap_s "
        "instead; it works in samples and owns the chunking",
        this->id(), _overlap));
  }
  const int keep = retain_beats_();
  if (keep > 0) { retain_tail_(keep); } else { reset_group_(); }
  ++_emitted;
  session()->log_debug(fmt(
      "TemporalStackStage('{}'): group #{} of {} beat(s) -> {}{}", this->id(),
      _emitted, n, shape_str_(tb.shape),
      keep > 0 ? fmt(" (keeping {} for the next)", keep)() : std::string()));
  auto payload = make_payload<TensorBeatPayload>(std::move(tb));
  co_await ctx.write(0, std::move(payload));
  co_return;
}

Job
TemporalStackStage::process(RuntimeContext& ctx)
{
  auto p = co_await ctx.read(0);
  if (!p) {
    // End of stream: a partial group is still a group. An EMPTY one is
    // not -- emitting a zero-length tensor would make "the source had
    // nothing" indistinguishable from "the source had silence".
    if (_count > 0) { co_await emit_(ctx); }
    ctx.signal_done();
    co_return;
  }
  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(p.get());
  if (tbp == nullptr) {
    session()->warn(fmt(
        "TemporalStackStage('{}'): expected a tensor, got {}; dropping beat",
        this->id(), p->describe()));
    co_return;
  }
  if (_capped) { co_return; }

  // The ceiling is checked BEFORE the append, so the emitted group is
  // one that fits rather than one that already overshot.
  if (!_buf.empty() && _buf.size() + tbp->data.size() > _max_bytes) {
    session()->warn(fmt(
        "TemporalStackStage('{}'): the accumulator reached its {} MB "
        "ceiling at {} beat(s); emitting those and ending the stream. The "
        "group is short of what was asked for -- resample the frames "
        "smaller before stacking, or raise max_mb", this->id(),
        _max_bytes >> 20, _count));
    _capped = true;
    co_await emit_(ctx);
    ctx.signal_done();
    co_return;
  }

  std::string err;
  if (!append_(*tbp, &err)) {
    session()->warn(fmt(
        "TemporalStackStage('{}'): {}; dropping beat (a group is one "
        "geometry, and a stack across two is not the media)", this->id(),
        err));
    co_return;
  }
  if (_group_size > 0 && _count >= _group_size) {
    co_await emit_(ctx);
  }
  co_return;
}

VPIPE_REGISTER_STAGE(TemporalStackStage)
VPIPE_REGISTER_SPEC(TemporalStackStage, kSpec)

}  // namespace vpipe
