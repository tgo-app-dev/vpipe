#include "stages/beat-track-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/av-filter-graph.h"
#include "common/beat-payload-intf.h"
#include "common/ffmpeg-libraries.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/beat-tracker.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/frame.h>
#include <libavutil/samplefmt.h>
}

namespace vpipe {

namespace {

constexpr SpecExtra kModeChoices[] = {
  {spec_key::kChoices, "stream,whole"},
};
constexpr SpecExtra kUnitsChoices[] = {
  {spec_key::kChoices, "time,samples,frames"},
};

const ConfigKey kAttrs[] = {
  {.key = "mode", .type = ConfigType::String, .required = false,
   .doc = "\"stream\" (default): beats come out as they are decided, "
          "never revised, from 8 s past the first onset on and then one to "
          "three beats behind the sound -- librosa's whole-track "
          "statistics taken over what has been heard. \"whole\": "
          "librosa.beat.beat_track over the whole stream, at its end",
   .def_str = "stream", .extra = kModeChoices},
  {.key = "group", .type = ConfigType::Int, .required = false,
   .doc = "beats per output object (>= 1). The last object of a stream "
          "holds what is left",
   .def_int = 1},
  {.key = "units", .type = ConfigType::String, .required = false,
   .doc = "how a beat is given: \"time\" (seconds from the stream's first "
          "sample, default), \"samples\" (that sample at the input's rate) "
          "or \"frames\" (onset-envelope frames, hop_length samples at the "
          "analysis rate -- librosa's default)",
   .def_str = "time", .extra = kUnitsChoices},
  {.key = "channels", .type = ConfigType::String, .required = false,
   .doc = "\"mix\" (default): one tracker on the channels' mean, as "
          "librosa.load hears a file; \"each\": a tracker per channel, "
          "each object naming its `channel`; or a channel index",
   .def_str = "mix"},
  {.key = "sample_rate", .type = ConfigType::Int, .required = false,
   .doc = "the rate the tracker hears: 22050 (default), librosa.load's. "
          "Another input rate is converted with libavfilter's aresample. "
          "0 tracks at the input's own rate, as beat_track(y, sr=that "
          "rate) does",
   .def_int = 22050},
  {.key = "hop_length", .type = ConfigType::Int, .required = false,
   .doc = "the onset envelope's hop in samples at the analysis rate "
          "(librosa's 512: 23 ms at 22050 Hz); at most 1000 envelope "
          "frames a second",
   .def_int = 512},
  {.key = "start_bpm", .type = ConfigType::Real, .required = false,
   .doc = "the centre of the tempo estimate's log-normal prior, bpm "
          "(librosa's 120)",
   .def_real = 120.0},
  {.key = "tightness", .type = ConfigType::Real, .required = false,
   .doc = "how closely the beats keep to the tempo (librosa's 100); "
          "lower lets them follow the onsets more",
   .def_real = 100.0},
  {.key = "trim", .type = ConfigType::Bool, .required = false,
   .doc = "drop weak leading and trailing beats (librosa's default)",
   .def_bool = true},
  {.key = "bpm", .type = ConfigType::Real, .required = false,
   .doc = "> 0: track at this tempo instead of estimating one (librosa's "
          "bpm=)",
   .def_real = 0.0},
  {.key = "max_latency", .type = ConfigType::Real, .required = false,
   .doc = "stream: the longest a beat waits to be handed out, seconds "
          "(default 4). Most beats are certain 2-4 s after they are heard, "
          "but where two phases of the beat stay plausible certainty can "
          "take 10-20 s; past this the best guess so far is handed out, "
          "and a later decision that disagrees continues from it -- "
          "nothing handed out is revised. 0 waits for certainty. Measured "
          "against librosa: 98.1% of beats within a frame at 4 s, 96.9% at "
          "2 s, 95.5% at 1 s, 98.8-100% waiting",
   .def_real = 4.0},
};

const PortSpec kIports[] = {
  {.name = "pcm",
   .doc = "f32 PCM, [N] mono or PLANAR [C, N], `sample_rate` on the "
          "sideband. Any beat length; the beats until the end of stream "
          "are one stream",
   .type = &typeid(TensorBeatPayload),
   .tags = "pcm-samples, audio-pcm", .clock_group = 0},
};

const PortSpec kOports[] = {
  {.name = "beats",
   .doc = "FlexData per `group` beats: {beats: [..] in `units`, units, "
          "index (beats[0]'s ordinal), tempo (bpm), sample_rate, "
          "hop_length, analysis_rate, channel (\"each\" / an index), "
          "timestamp_us (the stream's first sample's UTC, when given)}. "
          "Comes out as beats are decided",
   .type = &typeid(FlexDataPayload), .clock_group = 1},
};

const StageSpec kSpec = {
  .type_name = "beat-track",
  .doc       = "Tracks the beats of PCM audio with librosa's beat tracker "
               "(beat_track: the mel onset envelope, the tempo from its "
               "autocorrelation, Ellis's dynamic program) and streams their "
               "positions -- in seconds, samples or envelope frames -- as "
               "FlexData, `group` beats an object. Mono, a mix of the "
               "channels or every channel apart; converted to 22050 Hz as "
               "librosa.load does.",
  .display_name = "Beat Track",
  .category  = StageCategory::Audio,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

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

const char*
units_name_(BeatTrackStage::Units u)
{
  switch (u) {
    case BeatTrackStage::Units::kSamples: return "samples";
    case BeatTrackStage::Units::kFrames:  return "frames";
    case BeatTrackStage::Units::kTime:    break;
  }
  return "time";
}

}  // namespace

// One tracker and what feeds it: the input channel it hears (-1, the
// mix), a resampler when the input is not at the analysis rate, and the
// beats waiting to fill a group.
struct BeatTrackStage::Channel {
  int index = -1;
  std::unique_ptr<beat::Tracker> tracker;
  const FFmpegLibraries* libs = nullptr;
  AvFilterGraph graph;
  AVFrame* fin = nullptr;
  AVFrame* fout = nullptr;
  std::int64_t pts = 0;
  std::vector<beat::Beat> pending;
  std::uint64_t index_out = 0;     // beats handed on

  ~Channel()
  {
    if (libs != nullptr && libs->avutil().api.frame_free != nullptr) {
      if (fin != nullptr) { libs->avutil().api.frame_free(&fin); }
      if (fout != nullptr) { libs->avutil().api.frame_free(&fout); }
    }
    graph.close();
  }
};

BeatTrackStage::BeatTrackStage(const SessionContextIntf* s, std::string id,
                               std::vector<InEdge> ip, FlexData cfg)
  : TypedStage<BeatTrackStage>(s, std::move(id), std::move(ip),
                               std::move(cfg))
{
  allocate_oports(spec().oports.size());
  const std::string mode = attr_str("mode");
  if (mode == "whole") {
    _whole = true;
  } else if (mode != "stream") {
    fail_config(fmt("BeatTrackStage('{}'): mode '{}' is neither \"stream\" "
                    "nor \"whole\"", this->id(), mode));
  }
  _group = (int)attr_int("group");
  if (_group < 1) {
    fail_config(fmt("BeatTrackStage('{}'): group {} is under 1",
                    this->id(), _group));
  }
  const std::string u = attr_str("units");
  if (u == "time") {
    _units = Units::kTime;
  } else if (u == "samples") {
    _units = Units::kSamples;
  } else if (u == "frames") {
    _units = Units::kFrames;
  } else {
    fail_config(fmt("BeatTrackStage('{}'): units '{}' is none of \"time\", "
                    "\"samples\", \"frames\"", this->id(), u));
  }
  _channels = attr_str("channels");
  if (_channels == "each") {
    _each = true;
  } else if (_channels != "mix") {
    char* end = nullptr;
    const long v = std::strtol(_channels.c_str(), &end, 10);
    if (_channels.empty() || end == nullptr || *end != '\0' || v < 0 ||
        v > 1023) {
      fail_config(fmt("BeatTrackStage('{}'): channels '{}' is none of "
                      "\"mix\", \"each\" or a channel index", this->id(),
                      _channels));
    } else {
      _pick = (int)v;
    }
  }
  _sr = (int)attr_int("sample_rate");
  _hop = (int)attr_int("hop_length");
  _start_bpm = attr_real("start_bpm");
  _tightness = attr_real("tightness");
  _trim = attr_bool("trim");
  _bpm = attr_real("bpm");
  _max_latency = attr_real("max_latency");
  if (_max_latency < 0.0) {
    fail_config(fmt("BeatTrackStage('{}'): max_latency {} is negative (0 "
                    "waits for certainty)", this->id(), _max_latency));
  }
  if (_sr < 0 || (_sr > 0 && _sr < 4000) || _sr > 768000) {
    fail_config(fmt("BeatTrackStage('{}'): sample_rate {} is outside "
                    "[4000, 768000] (0: the input's)", this->id(), _sr));
  }
  if (_hop < 16 || _hop > 8192) {
    fail_config(fmt("BeatTrackStage('{}'): hop_length {} is outside "
                    "[16, 8192]", this->id(), _hop));
  } else if (_sr > 0 && _sr > 1000 * _hop) {
    fail_config(fmt("BeatTrackStage('{}'): sample_rate {} over hop_length "
                    "{} is over 1000 envelope frames a second", this->id(),
                    _sr, _hop));
  }
  if (!(_start_bpm > 0.0) || !(_tightness > 0.0) || _bpm < 0.0) {
    fail_config(fmt("BeatTrackStage('{}'): start_bpm and tightness must be "
                    "positive and bpm not negative", this->id()));
  }
}

BeatTrackStage::~BeatTrackStage() = default;

const StageSpec&
BeatTrackStage::spec() const noexcept
{
  return kSpec;
}

void
BeatTrackStage::reset_run_state()
{
  // A relaunch is a new stream.
  _ch.clear();
  _open = _failed = false;
  _in_rate = _in_channels = _rate = 0;
  _have_ts = false;
  _first_ts_us = 0;
  _beats_out = _groups_out = 0;
  _out.clear();
}

bool
BeatTrackStage::open_(const TensorBeatPayload& tb, int channels, int rate)
{
  _in_rate = rate;
  _in_channels = channels;
  _rate = _sr > 0 ? _sr : rate;
  double ts = 0.0;
  _have_ts = sb_num_(tb.sideband, "timestamp_us", &ts) && ts >= 0.0;
  _first_ts_us = _have_ts ? (std::uint64_t)ts : 0;
  if (_pick >= channels) {
    session()->error(fmt("BeatTrackStage('{}'): channel {} asked of a "
                         "{}-channel stream; stopping", this->id(), _pick,
                         channels));
    return false;
  }
  beat::Params p;
  p.sample_rate = _rate;
  p.hop_length = _hop;
  p.start_bpm = _start_bpm;
  p.tightness = (float)_tightness;
  p.trim = _trim;
  p.bpm = _bpm;
  p.max_latency = _max_latency;
  const int n = _each ? channels : 1;
  const FFmpegLibraries* libs = nullptr;
  if (_rate != rate) {
    libs = session()->services()->ffmpeg_libraries();
    if (libs == nullptr || !libs->avfilter().valid()) {
      session()->error(fmt("BeatTrackStage('{}'): the input is {} Hz and "
                           "the tracker hears {} Hz, a conversion that needs "
                           "FFmpeg's libavfilter, which is not loaded. Feed "
                           "{} Hz, or set sample_rate 0 to track at {} Hz; "
                           "stopping", this->id(), rate, _rate, _rate,
                           rate));
      return false;
    }
  }
  for (int i = 0; i < n; ++i) {
    auto c = std::make_unique<Channel>();
    c->index = _each ? i : _pick;
    c->tracker = std::make_unique<beat::Tracker>(
        p, _whole ? beat::Mode::kWhole : beat::Mode::kStream);
    if (*c->tracker->invalid() != '\0') {
      session()->error(fmt("BeatTrackStage('{}'): {}; stopping", this->id(),
                           c->tracker->invalid()));
      return false;
    }
    if (libs != nullptr) {
      c->libs = libs;
      c->fin = libs->avutil().api.frame_alloc();
      c->fout = libs->avutil().api.frame_alloc();
      AvFilterGraph::AudioIn in;
      in.sample_rate = rate;
      in.channels = 1;
      in.sample_fmt = AV_SAMPLE_FMT_FLTP;
      std::string err;
      const std::string chain = "aresample=" + std::to_string(_rate);
      if (c->fin == nullptr || c->fout == nullptr ||
          !c->graph.open(libs, in, chain, &err)) {
        session()->error(fmt("BeatTrackStage('{}'): cannot convert {} Hz to "
                             "{} Hz: {}; stopping", this->id(), rate, _rate,
                             err));
        return false;
      }
    }
    _ch.push_back(std::move(c));
  }
  session()->info(fmt(
      "BeatTrackStage('{}'): {} tracker{} ({}) on {} Hz {}-channel PCM{}, "
      "hop {} ({:.1f} ms), {} mode, {} beat{} an object in {}", this->id(),
      n, n == 1 ? "" : "s",
      _each ? "a channel each"
            : _pick >= 0 ? fmt("channel {}", _pick)() : "the channels' mean",
      rate, channels,
      _rate != rate ? fmt(", heard at {} Hz", _rate)() : std::string(),
      _hop, 1000.0 * _hop / _rate, _whole ? "whole" : "stream", _group,
      _group == 1 ? "" : "s", units_name_(_units)));
  return true;
}

// The mono samples one tracker hears, at the input's rate, through the
// resampler when there is one.
bool
BeatTrackStage::resample_(Channel& c, const float* x, std::size_t n,
                          bool eos)
{
  std::string err;
  if (!eos) {
    c.fin->format = AV_SAMPLE_FMT_FLTP;
    c.fin->sample_rate = _in_rate;
    c.fin->nb_samples = (int)n;
    c.fin->pts = c.pts;
    c.fin->ch_layout = AVChannelLayout AV_CHANNEL_LAYOUT_MONO;
    c.fin->data[0] = reinterpret_cast<std::uint8_t*>(const_cast<float*>(x));
    c.fin->linesize[0] = (int)(n * sizeof(float));
    // A frame filled by hand leaves extended_data null; planar consumers
    // read that.
    c.fin->extended_data = c.fin->data;
    c.pts += (std::int64_t)n;
    if (!c.graph.push(c.fin, &err)) {
      session()->error(fmt("BeatTrackStage('{}'): {}", this->id(), err));
      return false;
    }
  } else if (!c.graph.push(nullptr, &err)) {
    session()->error(fmt("BeatTrackStage('{}'): {}", this->id(), err));
    return false;
  }
  while (true) {
    const auto r = c.graph.pull(c.fout, &err);
    if (r == AvFilterGraph::Pull::kAgain || r == AvFilterGraph::Pull::kEof) {
      return true;
    }
    if (r == AvFilterGraph::Pull::kError) {
      session()->error(fmt("BeatTrackStage('{}'): {}", this->id(), err));
      return false;
    }
    const int k = c.fout->nb_samples;
    if (k > 0) {
      c.tracker->push(reinterpret_cast<const float*>(
                          c.fout->extended_data[0]), (std::size_t)k);
    }
  }
}

bool
BeatTrackStage::feed_(Channel& c, const float* x, std::size_t n)
{
  if (!c.graph.is_open()) {
    c.tracker->push(x, n);
    return true;
  }
  return resample_(c, x, n, false);
}

// The tracker's beats into this channel's groups, whole groups out.
void
BeatTrackStage::collect_(Channel& c)
{
  for (const beat::Beat& b : c.tracker->take()) { c.pending.push_back(b); }
  while (c.pending.size() >= (std::size_t)_group) {
    _out.emplace_back(group_(c, (std::size_t)_group), (std::size_t)_group);
  }
}

FlexData
BeatTrackStage::group_(Channel& c, std::size_t n) const
{
  FlexData root = FlexData::make_object();
  auto o = root.as_object();
  FlexData list = FlexData::make_array();
  auto a = list.as_array();
  const double hop = (double)_hop;
  for (std::size_t i = 0; i < n; ++i) {
    const std::int64_t f = c.pending[i].frame;
    switch (_units) {
      case Units::kFrames:
        a.push_back(FlexData::make_int(f));
        break;
      case Units::kSamples:
        // At the input's rate: exact where the tracker hears that rate.
        a.push_back(FlexData::make_int(
            _rate == _in_rate
                ? f * _hop
                : (std::int64_t)std::llround((double)f * hop *
                                             (double)_in_rate /
                                             (double)_rate)));
        break;
      case Units::kTime:
        // librosa.frames_to_time: frame * hop samples, over the rate.
        a.push_back(FlexData::make_real((double)(f * _hop) /
                                        (double)_rate));
        break;
    }
  }
  o.insert_or_assign("beats", std::move(list));
  o.insert_or_assign("units", FlexData::make_string(units_name_(_units)));
  o.insert_or_assign("index", FlexData::make_uint(c.index_out));
  o.insert_or_assign("tempo", FlexData::make_real(
      _whole ? c.tracker->tempo() : c.pending[n - 1].bpm));
  o.insert_or_assign("sample_rate", FlexData::make_int(_in_rate));
  o.insert_or_assign("hop_length", FlexData::make_int(_hop));
  o.insert_or_assign("analysis_rate", FlexData::make_int(_rate));
  if (c.index >= 0) {
    o.insert_or_assign("channel", FlexData::make_int(c.index));
  }
  if (_have_ts) {
    o.insert_or_assign("timestamp_us", FlexData::make_uint(_first_ts_us));
  }
  c.index_out += (std::uint64_t)n;
  c.pending.erase(c.pending.begin(), c.pending.begin() + (long)n);
  return root;
}

Job
BeatTrackStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  if (!in) {
    if (_open && !_failed) {
      for (auto& c : _ch) {
        if (c->graph.is_open() && !resample_(*c, nullptr, 0, true)) {
          _failed = true;
        }
        c->tracker->finish();
        collect_(*c);
        if (!c->pending.empty()) {
          const std::size_t n = c->pending.size();
          _out.emplace_back(group_(*c, n), n);
        }
      }
      for (auto& g : _out) {
        _beats_out += g.second;
        ++_groups_out;
        co_await ctx.write(0, make_payload<FlexDataPayload>(
                                  std::move(g.first)));
      }
      _out.clear();
      std::string tempi;
      for (const auto& c : _ch) {
        tempi += fmt("{}{:.2f}", tempi.empty() ? "" : ", ",
                     c->tracker->tempo())();
      }
      session()->info(fmt("BeatTrackStage('{}'): {} beats in {} object{}; "
                          "tempo {} bpm", this->id(), _beats_out,
                          _groups_out, _groups_out == 1 ? "" : "s", tempi));
    }
    ctx.signal_done();
    co_return;
  }
  if (_failed) { co_return; }
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (tb == nullptr) {
    session()->warn(fmt("BeatTrackStage('{}'): expected a TensorBeat, got "
                        "{}; skipping", this->id(), in->describe()));
    co_return;
  }
  if (tb->dtype != TensorBeat::DType::F32) {
    session()->warn(fmt("BeatTrackStage('{}'): PCM must be f32, got {}; "
                        "skipping a beat", this->id(),
                        TensorBeat::name_of(tb->dtype)));
    co_return;
  }
  const int rank = (int)tb->shape.size();
  const int ch = rank == 2 ? (int)tb->shape[0] : 1;
  const std::int64_t len = rank == 2 ? tb->shape[1]
                         : rank == 1 ? tb->shape[0] : 0;
  if ((rank != 1 && rank != 2) || ch < 1 || len < 0) {
    session()->warn(fmt("BeatTrackStage('{}'): PCM must be [N] or [C, N]; "
                        "skipping a beat", this->id()));
    co_return;
  }
  double rv = 0.0;
  if ((!sb_num_(tb->sideband, "sample_rate", &rv) &&
       !sb_num_(tb->sideband, "sr", &rv)) || !(rv > 0.0)) {
    session()->error(fmt("BeatTrackStage('{}'): the beat carries no "
                         "`sample_rate`, and a beat placed at the wrong rate "
                         "is at the wrong time; stopping", this->id()));
    _failed = true;
    co_return;
  }
  const int rate = (int)std::lround(rv);
  if (!_open) {
    _open = true;
    if (!open_(*tb, ch, rate)) {
      _failed = true;
      co_return;
    }
  } else if (rate != _in_rate || ch != _in_channels) {
    session()->error(fmt("BeatTrackStage('{}'): the stream changed from {} "
                         "Hz {}-channel to {} Hz {}-channel; one stream is "
                         "one rate and layout, stopping", this->id(),
                         _in_rate, _in_channels, rate, ch));
    _failed = true;
    co_return;
  }
  if (len == 0) { co_return; }
  const auto host = tb->materialize_contiguous();
  const float* x = reinterpret_cast<const float*>(host.data());
  std::vector<float> mono;
  for (auto& c : _ch) {
    const float* src = nullptr;
    if (c->index >= 0) {
      src = x + (std::size_t)c->index * (std::size_t)len;
    } else if (ch == 1) {
      src = x;
    } else {
      // librosa.to_mono: np.mean over the channels -- the float32 sum,
      // divided by the count in float64 and rounded back.
      mono.resize((std::size_t)len);
      for (std::int64_t i = 0; i < len; ++i) {
        float s = 0.0f;
        for (int k = 0; k < ch; ++k) {
          s += x[(std::size_t)k * (std::size_t)len + (std::size_t)i];
        }
        mono[(std::size_t)i] = (float)((double)s / (double)ch);
      }
      src = mono.data();
    }
    if (!feed_(*c, src, (std::size_t)len)) {
      _failed = true;
      co_return;
    }
    collect_(*c);
  }
  for (auto& g : _out) {
    _beats_out += g.second;
    ++_groups_out;
    co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(g.first)));
  }
  _out.clear();
}

VPIPE_REGISTER_STAGE(BeatTrackStage)
VPIPE_REGISTER_SPEC(BeatTrackStage, kSpec)

}  // namespace vpipe
