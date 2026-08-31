#include "stages/audio-video/temporal-slice-stage.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include "apple-silicon/tensor-beat.h"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

constexpr ConfigKey kAttrs[] = {
  // All three optional, like every part of a Python slice. With none of
  // them set this is `[:]` -- a pass-through, which is the harmless
  // thing for an unconfigured stage to be.
  {.key = "start", .type = ConfigType::Int, .required = false,
   .doc = "first beat of the slice, Python-style. NEGATIVE counts back "
          "from the end -- -1 is the last beat, -2 the one before it -- "
          "which the stage can only resolve once the source hits EOS, so "
          "it holds that many beats until then. Absent: start at the "
          "first beat",
   .def_int = 0},
  {.key = "step", .type = ConfigType::Int, .required = false,
   .doc = "keep every step-th beat from `start`. Must be POSITIVE: a "
          "stream is read once and in order, so a negative step would "
          "mean emitting beats this stage has already let go. Absent: 1",
   .def_int = 1},
  {.key = "sequence", .type = ConfigType::String, .required = false,
   .doc = "what the slice indexes. \"beats\" (default) is the beat "
          "STREAM -- keep some beats, drop the rest. \"frames\" is the "
          "LEADING AXIS OF ONE BEAT: a stacked clip [T, 3, H, W] in, the "
          "selected frames out, one beat in and one beat out. The frames "
          "mode holds nothing (T is known from the beat, so a negative "
          "index resolves at once) and does NOT change the beat rate, so "
          "unlike the stream mode it can sit inside a feedback loop",
   .def_str = "beats"},
  {.key = "squeeze", .type = ConfigType::Bool, .required = false,
   .doc = "frames mode only: when the slice selects exactly ONE frame, "
          "drop the time axis and emit [3, H, W] -- a STILL -- instead of "
          "[1, 3, H, W], a one-frame CLIP. The two are different requests "
          "to a reference-conditioned model and are sized by different "
          "rules, so there is no safe default and this is a switch rather "
          "than an inference. Asking to squeeze several frames is warned "
          "about once and the axis is kept",
   .def_bool = false},
  {.key = "end", .type = ConfigType::Int, .required = false,
   .doc = "one PAST the last beat of the slice, Python-style, so the "
          "slice is [start, end). Negative counts back from the end, on "
          "the same terms as `start`. Absent: run until the source hits "
          "EOS",
   .def_int = 0},
};

// Untyped ports: this stage never looks inside a beat, it only counts
// them, so constraining the payload would refuse producers it handles
// perfectly well. What it forwards is the beat it was handed, which is
// also how the sideband survives -- there is nothing to copy.
const PortSpec kIports[] = {
  {.name = "beats",
   .doc = "any beat stream; one beat is one position in the sequence",
   .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "beats",
   .doc = "the selected beats, FORWARDED UNCHANGED -- same payload, same "
          "shape, same sideband. Not stacked: a slice of N beats is N "
          "beats, not one tensor with a new axis (that is temporal-stack)",
   .clock_group = 1},
};

const StageSpec kSpec = {
  .type_name = "temporal-slice",
  .doc       = "Python's seq[start:end:step] over the beat stream: keeps "
               "the beats the slice names and drops the rest, forwarding "
               "each one untouched. `start: -1` into save-image writes "
               "the LAST frame of a clip. Negative indices are resolved "
               "at EOS, so they cost a held window of that many beats.",
  .display_name = "Temporal Slice",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

TemporalSliceStage::TemporalSliceStage(const SessionContextIntf* s,
                                       std::string               id,
                                       std::vector<InEdge>       iports,
                                       FlexData                  config)
  : TypedStage<TemporalSliceStage>(s, std::move(id), std::move(iports),
                                   std::move(config))
{
  allocate_oports(std::size(kOports));

  // PRESENCE, not value: 0 is a legal start and a legal end, so a
  // default cannot stand in for "not written". attr_int would hand back
  // the same 0 either way.
  const auto& c = this->config();
  if (c.is_object()) {
    auto o = c.as_object();
    _has_start = o.contains("start");
    _has_end   = o.contains("end");
  }
  _start = attr_int("start");
  _step  = attr_int("step");
  _end   = attr_int("end");
  _squeeze = attr_bool("squeeze");
  {
    const std::string seq = attr_str("sequence");
    if (seq.empty() || seq == "beats") {
      _within_beat = false;
    } else if (seq == "frames") {
      _within_beat = true;
    } else {
      fail_config(fmt("sequence '{}' is not beats|frames", seq));
    }
  }
  if (_squeeze && !_within_beat) {
    // Not fatal, but it means nothing: the stream mode forwards beats
    // untouched, so there is no axis to drop.
    session()->warn(fmt(
        "TemporalSliceStage('{}'): squeeze has no effect with "
        "sequence 'beats' -- that mode forwards each beat untouched, "
        "shape included", this->id()));
  }

  // Deferred validation: the constructor never throws, and the runtime
  // skips a stage whose config failed.
  if (_step <= 0) {
    fail_config(fmt("step must be positive (got {}); a stream is read "
                    "once and in order, so there is no beat to step back "
                    "to", _step));
  }
}

TemporalSliceStage::~TemporalSliceStage() = default;

const StageSpec&
TemporalSliceStage::spec() const noexcept
{
  return kSpec;
}

std::int64_t
TemporalSliceStage::hold_() const noexcept
{
  std::int64_t w = 0;
  if (_has_start && _start < 0) { w = std::max<std::int64_t>(w, -_start); }
  if (_has_end   && _end   < 0) { w = std::max<std::int64_t>(w, -_end); }
  return w;
}

bool
TemporalSliceStage::selected_(std::int64_t j, std::int64_t n) const noexcept
{
  // Python's slice resolution, both ends clamped into [0, n].
  std::int64_t s = 0;
  if (_has_start) {
    s = (_start < 0) ? std::max<std::int64_t>(n + _start, 0)
                     : std::min<std::int64_t>(_start, n);
  }
  std::int64_t e = n;
  if (_has_end) {
    e = (_end < 0) ? std::max<std::int64_t>(n + _end, 0)
                   : std::min<std::int64_t>(_end, n);
  }
  if (j < s || j >= e) { return false; }
  return ((j - s) % _step) == 0;
}

bool
TemporalSliceStage::selected_streaming_(std::int64_t j) const noexcept
{
  // Only ever asked about a beat that has fallen out of the hold
  // window, i.e. one with at least W beats behind it, so j < n - W.
  // That single fact settles both ends without knowing n:
  //
  //   start < 0 => s = n + start >= n - W > j, so j is NEVER selected.
  //   end   < 0 => e = n + end   >= n - W > j, so `end` cannot exclude
  //                it, and only `start` and `step` decide.
  //
  // Which leaves the non-negative ends, and those never needed n.
  if (_has_start && _start < 0) { return false; }
  const std::int64_t s = _has_start ? _start : 0;
  if (j < s) { return false; }
  if (_has_end && _end >= 0 && j >= _end) { return false; }
  return ((j - s) % _step) == 0;
}

Job
TemporalSliceStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  _index     = 0;
  _emitted   = 0;
  _peak_hold = 0;
  _hold.clear();

  const std::int64_t w = hold_();
  if (w > 0) {
    // Said out loud because it is the stage's whole memory cost and it
    // is set by a number the author wrote, not by the stream.
    session()->info(fmt(
        "TemporalSliceStage('{}'): a negative index holds the last {} "
        "beat(s) until EOS, where the slice is resolved", this->id(), w));
  }
  co_return;
}

Job
TemporalSliceStage::process_within_(RuntimeContext& ctx)
{
  auto p = co_await ctx.read(0);
  if (!p) { ctx.signal_done(); co_return; }

  const auto* tb = dynamic_cast<const TensorBeatPayload*>(p.get());
  if (tb == nullptr) {
    session()->warn(fmt(
        "TemporalSliceStage('{}'): sequence 'frames' slices the leading "
        "axis of a TENSOR, got {}; dropping beat", this->id(),
        p->describe()));
    co_return;
  }
  // Rank 2 is the floor: there has to be a leading axis to index AND
  // something left after it. A rank-1 tensor sliced this way would be
  // audio samples, which is temporal-stack's other modality and not a
  // clip at all.
  if (tb->shape.size() < 2 || tb->shape[0] <= 0 || tb->byte_size() == 0) {
    session()->warn(fmt(
        "TemporalSliceStage('{}'): sequence 'frames' needs a rank >= 2 "
        "tensor with a non-empty leading axis, got {}; dropping beat",
        this->id(), tb->describe()));
    co_return;
  }

  // T is KNOWN here, which is the whole difference from the stream
  // mode: a negative index resolves now instead of at EOS, so nothing
  // is held and `start: -1` costs nothing.
  const std::int64_t n = tb->shape[0];
  std::vector<std::int64_t> pick;
  for (std::int64_t j = 0; j < n; ++j) {
    if (selected_(j, n)) { pick.push_back(j); }
  }
  if (pick.empty()) {
    session()->log_debug(fmt(
        "TemporalSliceStage('{}'): the slice selected no frame of {}; "
        "dropping beat", this->id(), n));
    co_return;
  }

  // One element's geometry and byte size: everything after the leading
  // axis, which the rank check above guarantees is non-empty.
  std::vector<std::int64_t> elem(tb->shape.begin() + 1, tb->shape.end());
  std::size_t per_elems = 1;
  for (std::int64_t d : elem) { per_elems *= (std::size_t)(d > 0 ? d : 0); }
  const std::size_t esz = TensorBeat::byte_size_of(tb->dtype);
  const std::size_t per = per_elems * esz;
  if (per == 0 || tb->byte_size() < (std::size_t)n * per) {
    session()->warn(fmt(
        "TemporalSliceStage('{}'): {} does not hold {} whole elements; "
        "dropping beat", this->id(), tb->describe(), n));
    co_return;
  }

  // SQUEEZE: only when the slice named exactly one frame. Asking for it
  // over several is the caller describing a shape they cannot have
  // meant, so it is said out loud rather than fabricated -- once,
  // because a clip stream would otherwise say it every beat.
  bool squeeze = _squeeze && pick.size() == 1;
  if (_squeeze && pick.size() != 1 && !_squeeze_warned) {
    _squeeze_warned = true;
    session()->warn(fmt(
        "TemporalSliceStage('{}'): squeeze asked for but the slice "
        "selected {} frames, not 1; keeping the time axis. A multi-frame "
        "selection is a CLIP and has no still to squeeze to",
        this->id(), pick.size()));
  }

  TensorBeat out;
  out.dtype = tb->dtype;
  if (squeeze) {
    out.shape = elem;
  } else {
    out.shape.push_back((std::int64_t)pick.size());
    for (std::int64_t d : elem) { out.shape.push_back(d); }
  }
  out.data.resize(pick.size() * per);
  for (std::size_t k = 0; k < pick.size(); ++k) {
    std::memcpy(out.data.data() + k * per,
                tb->bytes_() + (std::size_t)pick[k] * per, per);
  }

  // The sideband is the input's, with the rate fields made true of what
  // came out. A SQUEEZED beat is a still: it carries neither `frames`
  // nor `fps`, because a picture that claims a frame count is a clip to
  // anything that reads one.
  FlexData sb = tb->sideband.is_object() ? tb->sideband
                                         : FlexData::make_object();
  auto o = sb.as_object();
  if (squeeze) {
    o.erase("frames");
    o.erase("fps");
  } else {
    o.insert_or_assign("frames",
                       FlexData::make_int((std::int64_t)pick.size()));
  }
  out.sideband = std::move(sb);

  ++_emitted;
  // Bound before the suspend; see audio-to-pcm's emit_chunk_ for what
  // building the payload inside the co_await argument costs.
  auto payload = make_payload<TensorBeatPayload>(std::move(out));
  co_await ctx.write(0, std::move(payload));
  co_return;
}

Job
TemporalSliceStage::process(RuntimeContext& ctx)
{
  if (_within_beat) {
    co_await process_within_(ctx);
    co_return;
  }
  auto p = co_await ctx.read(0);
  if (!p) {
    // EOS: the count is finally known, so everything still held can be
    // decided exactly. Emitted in ARRIVAL order -- a slice is a
    // subsequence, not a reordering.
    const std::int64_t n = _index;
    for (auto& [j, beat] : _hold) {
      if (!selected_(j, n)) { continue; }
      ++_emitted;
      co_await ctx.write(0, std::move(beat));
    }
    _hold.clear();
    ctx.signal_done();
    co_return;
  }

  const std::int64_t idx = _index++;
  const std::int64_t w   = hold_();

  // NO WINDOW: decide the beat on arrival and keep nothing. The whole
  // non-negative case, and the one that has to work on a source that
  // never reaches EOS.
  if (w == 0) {
    if (!selected_streaming_(idx)) { co_return; }
    ++_emitted;
    co_await ctx.write(0, std::move(p));
    co_return;
  }

  // Room is made BEFORE the new beat is taken, so the window holds W and
  // never W+1. A beat leaving it has W beats behind it, which is exactly
  // what makes it decidable without the total (see selected_streaming_).
  while ((std::int64_t)_hold.size() >= w) {
    auto [j, beat] = std::move(_hold.front());
    _hold.pop_front();
    if (!selected_streaming_(j)) { continue; }
    ++_emitted;
    co_await ctx.write(0, std::move(beat));
  }
  _hold.emplace_back(idx, std::move(p));
  _peak_hold = std::max(_peak_hold, (std::int64_t)_hold.size());
}

VPIPE_REGISTER_STAGE(TemporalSliceStage)
VPIPE_REGISTER_SPEC(TemporalSliceStage, kSpec)

}  // namespace vpipe
