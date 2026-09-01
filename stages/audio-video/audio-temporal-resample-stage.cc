#include "stages/audio-video/audio-temporal-resample-stage.h"

#include "common/beat-payload-intf.h"
#include "common/ffmpeg-libraries.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
}

namespace vpipe {

namespace {

bool
sb_num_(const FlexData& sb, const char* key, double* out)
{
  if (!sb.is_object()) { return false; }
  const auto o = sb.as_object();
  if (!o.contains(key)) { return false; }
  const FlexData v = o.at(key);
  if (v.is_int() || v.is_uint() || v.is_real()) {
    *out = v.as_real(0.0);
    return true;
  }
  return false;
}

// `atempo` takes one factor per instance and old builds cap it at 2.
// A chain of factors multiplies, so any ratio is reachable and the
// spelling works on every FFmpeg this dlopens.
std::string
atempo_chain_(double tempo)
{
  std::string out;
  auto add = [&](double f) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "atempo=%.9g", f);
    if (!out.empty()) { out += ","; }
    out += buf;
  };
  double t = tempo;
  while (t > 2.0)  { add(2.0); t /= 2.0; }
  while (t < 0.5)  { add(0.5); t *= 2.0; }
  if (std::fabs(t - 1.0) > 1e-9) { add(t); }
  return out;
}

constexpr ConfigKey kAttrs[] = {
  {.key = "output_sample_rate", .type = ConfigType::Int, .required = false,
   .doc = "the sample rate to resample TO -- the temporal RESOLUTION, "
          "which changes no duration and no pitch. 0 (default) keeps the "
          "input's. This is the knob a model forces: MiniMax-H3's audio "
          "VAE reads 32000 and nothing else, Qwen3-ASR 16000",
   .def_int = 0},
  {.key = "speed", .type = ConfigType::Real, .required = false,
   .doc = "the DURATION factor: the output lasts 1/speed as long as the "
          "input. 2.0 is twice as fast, 0.5 half. What happens to the "
          "PITCH is `pitch`'s business, not this key's. Default 1.0",
   .def_real = 1.0},
  {.key = "pitch", .type = ConfigType::String, .required = false,
   .doc = "\"maintain\" (default) holds the pitch while the duration "
          "changes -- a WSOLA time-stretch (ffmpeg `atempo`), which is "
          "what leaves a voice recognisable. \"follow\" lets the pitch "
          "track the speed, chipmunk or drawl: the tape behaviour, and "
          "free, because it is a resample and nothing else. \"raise\" / "
          "\"lower\" shift by `pitch_semitones` INDEPENDENTLY of speed, "
          "so the two knobs compose",
   .def_str = "maintain"},
  {.key = "pitch_semitones", .type = ConfigType::Real, .required = false,
   .doc = "how far `raise` / `lower` shifts, in equal-tempered "
          "semitones (12 = an octave). Must be positive for those two -- "
          "the DIRECTION is in `pitch`, so a signed value here would "
          "give two ways to spell the same request and one way to "
          "cancel it by accident. Ignored by maintain / follow",
   .def_real = 0.0},
  {.key = "stacked", .type = ConfigType::Bool, .required = false,
   .doc = "false (default): one PCM chunk per beat, and the oport is on "
          "a clock of its own because a speed change does not emit one "
          "beat per beat. true: the WHOLE waveform in one beat -- what "
          "`temporal-stack` builds -- so the stage is 1:1 and its oport "
          "shares the iport's clock, crossing no domain. STATED rather "
          "than sensed: the clock analysis runs at launch, before any "
          "beat exists",
   .def_bool = false},
};

const PortSpec kIports[] = {
  {.name = "pcm",
   .doc = "f32 PCM TensorBeat, [N] mono or PLANAR [C, N], with "
          "`sample_rate` on the sideband",
   .type = &typeid(TensorBeatPayload),
   .tags = "pcm-samples, audio-pcm", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "pcm",
   .doc = "the same layout at `output_sample_rate`, with the rate and "
          "duration rewritten on the sideband. Streaming, the chunk "
          "lengths are the filter's own. The DECLARED group is the "
          "streaming answer; oport_clock_group() shares the iport's when "
          "`stacked`",
   .type = &typeid(TensorBeatPayload),
   .tags = "pcm-samples, audio-pcm", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "audio-temporal-resample",
  .doc       = "Resamples PCM to another sample rate and/or speed "
               "through an FFmpeg filter chain, with pitch held, let to "
               "follow the speed, or shifted up or down by a stated "
               "number of semitones.",
  .display_name = "Audio Temporal Resample",
  .category  = StageCategory::Audio,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

AudioTemporalResampleStage::AudioTemporalResampleStage(
    const SessionContextIntf* s, std::string id, std::vector<InEdge> ip,
    FlexData cfg)
  : TypedStage<AudioTemporalResampleStage>(s, std::move(id), std::move(ip),
                                           std::move(cfg))
{
  allocate_oports(spec().oports.size());

  _out_rate  = (int)attr_int("output_sample_rate");
  _speed     = attr_real("speed");
  _semitones = attr_real("pitch_semitones");
  _stacked   = attr_bool("stacked");

  if (_out_rate < 0 || (_out_rate > 0 &&
                        (_out_rate < 1000 || _out_rate > 384000))) {
    fail_config(fmt("AudioTemporalResampleStage('{}'): "
                    "output_sample_rate {} is outside [1000, 384000] "
                    "(0 keeps the input's)", this->id(), _out_rate));
    return;
  }
  if (!(_speed > 0.0) || _speed > 100.0) {
    fail_config(fmt("AudioTemporalResampleStage('{}'): speed {} is "
                    "outside (0, 100]", this->id(), _speed));
    return;
  }
  const std::string p = attr_str("pitch");
  if (p == "maintain")     { _pitch = Pitch::kMaintain; }
  else if (p == "follow")  { _pitch = Pitch::kFollow; }
  else if (p == "raise")   { _pitch = Pitch::kRaise; }
  else if (p == "lower")   { _pitch = Pitch::kLower; }
  else {
    fail_config(fmt("AudioTemporalResampleStage('{}'): pitch '{}' is not "
                    "one of maintain|follow|raise|lower", this->id(), p));
    return;
  }
  const bool shifts = _pitch == Pitch::kRaise || _pitch == Pitch::kLower;
  if (shifts && !(_semitones > 0.0)) {
    fail_config(fmt("AudioTemporalResampleStage('{}'): pitch '{}' needs a "
                    "positive pitch_semitones -- the direction is in "
                    "`pitch`, so 0 asks to shift by nothing", this->id(),
                    p));
    return;
  }
  if (_semitones < 0.0 || _semitones > 48.0) {
    fail_config(fmt("AudioTemporalResampleStage('{}'): pitch_semitones {} "
                    "is outside [0, 48]", this->id(), _semitones));
    return;
  }
}

AudioTemporalResampleStage::~AudioTemporalResampleStage()
{
  const FFmpegLibraries* libs = session()->services()->ffmpeg_libraries();
  if (libs != nullptr && libs->avutil().api.frame_free != nullptr) {
    if (_in_frame  != nullptr) { libs->avutil().api.frame_free(&_in_frame); }
    if (_out_frame != nullptr) { libs->avutil().api.frame_free(&_out_frame); }
  }
  _graph.close();
}

const StageSpec&
AudioTemporalResampleStage::spec() const noexcept
{
  return kSpec;
}

unsigned
AudioTemporalResampleStage::oport_clock_group(unsigned p) const noexcept
{
  (void)p;
  return _stacked ? 0u : 1u;
}

const char*
AudioTemporalResampleStage::pitch_name(Pitch p) noexcept
{
  switch (p) {
    case Pitch::kMaintain: return "maintain";
    case Pitch::kFollow:   return "follow";
    case Pitch::kRaise:    return "raise";
    case Pitch::kLower:    return "lower";
  }
  return "maintain";
}

double
AudioTemporalResampleStage::pitch_factor(Pitch p, double semitones,
                                         double speed)
{
  switch (p) {
    case Pitch::kMaintain: return 1.0;
    // `follow` IS "the pitch factor equals the speed": the tempo below
    // then works out to 1 and no time-stretch is inserted at all, which
    // is the same thing as saying a plain resample was asked for.
    case Pitch::kFollow:   return speed;
    case Pitch::kRaise:    return std::pow(2.0, semitones / 12.0);
    case Pitch::kLower:    return std::pow(2.0, -semitones / 12.0);
  }
  return 1.0;
}

std::string
AudioTemporalResampleStage::chain_for(int in_rate, int out_rate,
                                      double speed, double k)
{
  if (in_rate <= 0 || out_rate <= 0 || !(speed > 0.0) || !(k > 0.0)) {
    return {};
  }
  std::string chain;
  auto add = [&](const std::string& f) {
    if (f.empty()) { return; }
    if (!chain.empty()) { chain += ","; }
    chain += f;
  };
  // asetrate RELABELS the rate: the same samples now play k times as
  // fast, so pitch x k and duration / k together. aresample brings the
  // rate back, keeping both. That is the pitch shift.
  if (std::fabs(k - 1.0) > 1e-9) {
    add("asetrate=" + std::to_string((long long)std::llround(
                          (double)in_rate * k)));
    add("aresample=" + std::to_string(in_rate));
  }
  // ...and the tempo puts the DURATION where it was asked to be. The
  // asetrate above already divided it by k, so what is left to apply is
  // speed / k -- which is 1 exactly when the pitch was asked to follow
  // the speed, and `speed` itself when the pitch was to be held.
  add(atempo_chain_(speed / k));
  if (out_rate != in_rate || chain.empty()) {
    add("aresample=" + std::to_string(out_rate));
  }
  return chain;
}

Job
AudioTemporalResampleStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  const FFmpegLibraries* libs = session()->services()->ffmpeg_libraries();
  if (libs == nullptr || !libs->avfilter().valid()) {
    session()->warn(fmt(
        "AudioTemporalResampleStage('{}'): libavfilter is not loaded, so "
        "this stage can only forward beats unchanged. Install a full "
        "FFmpeg or drop the stage from the graph", this->id()));
    co_return;
  }
  const bool shifts = _pitch == Pitch::kRaise || _pitch == Pitch::kLower;
  session()->info(fmt(
      "AudioTemporalResampleStage('{}'): {}, speed {:.4f}x, pitch {}{}"
      ", {} input", this->id(),
      _out_rate > 0 ? fmt("{} Hz", _out_rate)()
                    : std::string("the source's own rate"),
      _speed, pitch_name(_pitch),
      shifts ? fmt(" {:.3f} semitones", _semitones)() : std::string(),
      _stacked ? "one stacked waveform per beat" : "one chunk per beat"));
  co_return;
}

void
AudioTemporalResampleStage::reset_run_state()
{
  // atempo carries a WSOLA window and aresample a filter delay; a
  // relaunch is a new waveform and must not start inside the old one's
  // tail.
  _graph.close();
  _grate = _gch = 0;
  _pts = 0;
  _out_samples = 0;
  _emitted = 0;
  _have_seed = false;
  _sb_seed = FlexData{};
  _have_ts = false;
  _first_ts_us = 0;
  _shape_warned = false;
}

bool
AudioTemporalResampleStage::ensure_graph_(int rate, int channels)
{
  if (_graph.is_open() && _grate == rate && _gch == channels) {
    return true;
  }
  const FFmpegLibraries* libs = session()->services()->ffmpeg_libraries();
  if (libs == nullptr || !libs->avfilter().valid()) { return false; }
  if (_in_frame == nullptr)  { _in_frame  = libs->avutil().api.frame_alloc(); }
  if (_out_frame == nullptr) { _out_frame = libs->avutil().api.frame_alloc(); }
  if (_in_frame == nullptr || _out_frame == nullptr) { return false; }

  _graph.close();
  _pts = 0;
  AvFilterGraph::AudioIn in;
  in.sample_rate = rate;
  in.channels    = channels;
  in.sample_fmt  = AV_SAMPLE_FMT_FLTP;
  const int out_rate = _out_rate > 0 ? _out_rate : rate;
  const double k = pitch_factor(_pitch, _semitones, _speed);
  const std::string chain = chain_for(rate, out_rate, _speed, k);
  std::string err;
  if (!_graph.open(libs, in, chain, &err)) {
    session()->warn(fmt("AudioTemporalResampleStage('{}'): {}", this->id(),
                        err));
    return false;
  }
  _grate = rate; _gch = channels;
  session()->log_normal(fmt(
      "AudioTemporalResampleStage('{}'): {} Hz {}ch -> {} Hz via '{}'",
      this->id(), rate, channels, out_rate, chain));
  return true;
}

bool
AudioTemporalResampleStage::push_block_(const float* pcm, int channels,
                                        int samples, std::int64_t pts)
{
  _in_frame->format      = AV_SAMPLE_FMT_FLTP;
  _in_frame->sample_rate = _grate;
  _in_frame->nb_samples  = samples;
  _in_frame->pts         = pts;
  _in_frame->ch_layout   = channels == 2
                               ? AVChannelLayout AV_CHANNEL_LAYOUT_STEREO
                               : AVChannelLayout AV_CHANNEL_LAYOUT_MONO;
  for (int c = 0; c < channels; ++c) {
    // Planar and unreferenced, so these point straight at the beat: the
    // source's KEEP_REF copies them in.
    _in_frame->data[c] = reinterpret_cast<std::uint8_t*>(
        const_cast<float*>(pcm + (std::size_t)c * samples));
    _in_frame->linesize[c] = (int)((std::size_t)samples * sizeof(float));
  }
  // Manually populated frames leave extended_data null; every planar
  // consumer reads THAT rather than data[].
  _in_frame->extended_data = _in_frame->data;
  std::string err;
  if (!_graph.push(_in_frame, &err)) {
    session()->warn(fmt("AudioTemporalResampleStage('{}'): {}", this->id(),
                        err));
    return false;
  }
  return true;
}

bool
AudioTemporalResampleStage::drain_(std::vector<std::vector<float>>* blocks,
                                   int* channels)
{
  while (true) {
    std::string err;
    const auto r = _graph.pull(_out_frame, &err);
    if (r == AvFilterGraph::Pull::kAgain ||
        r == AvFilterGraph::Pull::kEof) {
      return true;
    }
    if (r == AvFilterGraph::Pull::kError) {
      session()->warn(fmt("AudioTemporalResampleStage('{}'): {}",
                          this->id(), err));
      return false;
    }
    const int ch = _out_frame->ch_layout.nb_channels;
    const int n  = _out_frame->nb_samples;
    if (ch <= 0 || n <= 0) { continue; }
    *channels = ch;
    std::vector<float> b((std::size_t)ch * n);
    for (int c = 0; c < ch; ++c) {
      const float* src =
          reinterpret_cast<const float*>(_out_frame->extended_data[c]);
      std::copy(src, src + n, b.data() + (std::size_t)c * n);
    }
    blocks->push_back(std::move(b));
  }
}

FlexData
AudioTemporalResampleStage::out_sideband_(int samples, int rate)
{
  FlexData sb = _have_seed ? _sb_seed : FlexData::make_object();
  if (!sb.is_object()) { sb = FlexData::make_object(); }
  auto o = sb.as_object();
  o.insert_or_assign("sample_rate", FlexData::make_int(rate));
  o.insert_or_assign(
      "duration_us",
      FlexData::make_uint((std::uint64_t)samples * 1000000ULL /
                          (std::uint64_t)std::max(1, rate)));
  if (_have_ts) {
    // Counted from the OUTPUT stream's own start, so the chunks are
    // contiguous whatever the filter's block sizes turned out to be.
    o.insert_or_assign(
        "timestamp_us",
        FlexData::make_uint(_first_ts_us +
                            _out_samples * 1000000ULL /
                                (std::uint64_t)std::max(1, rate)));
  }
  if (_stacked) {
    o.insert_or_assign("stacked", FlexData::make_int(1));
  } else {
    o.erase("stacked");
  }
  _out_samples += (std::uint64_t)samples;
  return sb;
}

Job
AudioTemporalResampleStage::process(RuntimeContext& ctx)
{
  auto p = co_await ctx.read(0);
  if (!p) {
    // EOS: atempo and aresample are both holding a tail, and dropping
    // it truncates the waveform by the filter's latency.
    if (!_stacked && _graph.is_open()) {
      std::string err;
      std::vector<std::vector<float>> tail;
      int ch = _gch;
      if (_graph.push(nullptr, &err) && drain_(&tail, &ch)) {
        const int rate = _out_rate > 0 ? _out_rate : _grate;
        for (auto& b : tail) {
          const int n = (int)(b.size() / (std::size_t)std::max(1, ch));
          auto t = std::make_unique<TensorBeatPayload>();
          t->dtype = TensorBeat::DType::F32;
          t->shape = ch == 1 ? std::vector<std::int64_t>{n}
                             : std::vector<std::int64_t>{ch, n};
          t->resize_contiguous(b.size());
          std::copy(b.begin(), b.end(), t->as_f32());
          t->sideband = out_sideband_(n, rate);
          ++_emitted;
          co_await ctx.write(0, std::move(t));
        }
      }
      _graph.close();
    }
    ctx.signal_done();
    co_return;
  }
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(p.get());
  if (tb == nullptr) {
    session()->warn(fmt(
        "AudioTemporalResampleStage('{}'): expected a TensorBeat, got {}; "
        "skipping", this->id(), p->describe()));
    co_return;
  }
  const FFmpegLibraries* libs = session()->services()->ffmpeg_libraries();
  if (libs == nullptr || !libs->avfilter().valid()) {
    co_await ctx.write(0, std::move(p));      // already warned at launch
    co_return;
  }
  if (tb->dtype != TensorBeat::DType::F32) {
    session()->warn(fmt(
        "AudioTemporalResampleStage('{}'): PCM must be f32, got {}; "
        "skipping", this->id(), TensorBeat::name_of(tb->dtype)));
    co_return;
  }
  const int rank = (int)tb->shape.size();
  if (rank < 1 || rank > 2) {
    session()->warn(fmt(
        "AudioTemporalResampleStage('{}'): PCM must be [N] or [C, N], got "
        "rank {}; skipping", this->id(), rank));
    co_return;
  }
  const int ch = rank == 2 ? (int)tb->shape[0] : 1;
  const int n  = rank == 2 ? (int)tb->shape[1] : (int)tb->shape[0];
  if (ch < 1 || ch > 2 || n <= 0) {
    session()->warn(fmt(
        "AudioTemporalResampleStage('{}'): mono or stereo only and not "
        "empty, got {} channels of {}; skipping", this->id(), ch, n));
    co_return;
  }
  double rv = 0.0;
  if ((!sb_num_(tb->sideband, "sample_rate", &rv) &&
       !sb_num_(tb->sideband, "sr", &rv)) || !(rv > 0.0)) {
    session()->warn(fmt(
        "AudioTemporalResampleStage('{}'): the beat carries no positive "
        "`sample_rate` in its sideband, and a rate cannot be guessed -- "
        "a waveform read at the wrong one is the wrong sound; skipping",
        this->id()));
    co_return;
  }
  const int in_rate = (int)rv;

  // `stacked` decides the clock domain at launch, so it cannot follow
  // the beat. A contradiction is named once rather than reinterpreted.
  const bool looks_stacked = [&] {
    double v = 0.0;
    return sb_num_(tb->sideband, "stacked", &v) && v > 0.0;
  }();
  if (looks_stacked != _stacked && !_shape_warned) {
    _shape_warned = true;
    session()->warn(fmt(
        "AudioTemporalResampleStage('{}'): `stacked` is {} but the beat "
        "{} say it is stacked. The setting decides the CLOCK DOMAIN at "
        "launch, so it cannot follow the beat; the waveform is filtered "
        "either way, but check the key", this->id(),
        _stacked ? "true" : "false", looks_stacked ? "does" : "does not"));
  }

  if (!_have_seed || _stacked) {
    _sb_seed   = tb->sideband;
    _have_seed = true;
    double ts = 0.0;
    _have_ts = sb_num_(tb->sideband, "timestamp_us", &ts) && ts >= 0.0;
    _first_ts_us = _have_ts ? (std::uint64_t)ts : 0;
    if (_stacked) { _out_samples = 0; }
  }

  // A stacked waveform gets a FRESH graph and is flushed at the end of
  // the beat -- the flush is what makes atempo's tail come out, and a
  // flushed graph is finished.
  if (_stacked) { _graph.close(); _grate = 0; }
  if (!ensure_graph_(in_rate, ch)) { co_return; }

  std::vector<std::vector<float>> out;
  int out_ch = ch;
  if (!push_block_(tb->as_f32(), ch, n, _pts)) { co_return; }
  _pts += n;
  if (!drain_(&out, &out_ch)) { co_return; }

  const int rate = _out_rate > 0 ? _out_rate : in_rate;
  if (_stacked) {
    std::string err;
    if (!_graph.push(nullptr, &err)) {
      session()->warn(fmt("AudioTemporalResampleStage('{}'): {}",
                          this->id(), err));
      co_return;
    }
    if (!drain_(&out, &out_ch)) { co_return; }
    _graph.close();
    _grate = 0;

    std::size_t total = 0;
    for (const auto& b : out) { total += b.size(); }
    const int m = (int)(total / (std::size_t)std::max(1, out_ch));
    auto t = std::make_unique<TensorBeatPayload>();
    t->dtype = TensorBeat::DType::F32;
    t->shape = out_ch == 1 ? std::vector<std::int64_t>{m}
                           : std::vector<std::int64_t>{out_ch, m};
    t->resize_contiguous(total);
    // The filter hands back planar BLOCKS; one waveform per channel is
    // the concatenation of each block's own plane, so the copy is
    // per-channel and not a memcpy of the lot.
    float* dst = t->as_f32();
    std::size_t at = 0;
    for (const auto& b : out) {
      const std::size_t bn = b.size() / (std::size_t)std::max(1, out_ch);
      for (int c = 0; c < out_ch; ++c) {
        std::copy(b.begin() + (std::size_t)c * bn,
                  b.begin() + (std::size_t)(c + 1) * bn,
                  dst + (std::size_t)c * m + at);
      }
      at += bn;
    }
    t->sideband = out_sideband_(m, rate);
    ++_emitted;
    session()->log_debug(fmt(
        "AudioTemporalResampleStage('{}'): {} -> {} samples/ch "
        "({} -> {} Hz)", this->id(), n, m, in_rate, rate));
    co_await ctx.write(0, std::move(t));
    co_return;
  }
  for (auto& b : out) {
    const int m = (int)(b.size() / (std::size_t)std::max(1, out_ch));
    auto t = std::make_unique<TensorBeatPayload>();
    t->dtype = TensorBeat::DType::F32;
    t->shape = out_ch == 1 ? std::vector<std::int64_t>{m}
                           : std::vector<std::int64_t>{out_ch, m};
    t->resize_contiguous(b.size());
    std::copy(b.begin(), b.end(), t->as_f32());
    t->sideband = out_sideband_(m, rate);
    ++_emitted;
    co_await ctx.write(0, std::move(t));
  }
}

VPIPE_REGISTER_STAGE(AudioTemporalResampleStage)
VPIPE_REGISTER_SPEC(AudioTemporalResampleStage, kSpec)

}  // namespace vpipe
