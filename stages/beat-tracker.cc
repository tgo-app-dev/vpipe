#include "stages/beat-tracker.h"

#include "3rd-party/pocketfft.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <utility>

// librosa's sums are numpy's and numba's, neither of which contracts a
// multiply and an add into one rounding; a fused one here would move the
// scores a rounding apart and, at a near tie, a beat.
#pragma clang fp contract(off)

namespace vpipe::beat {

namespace {

constexpr double kPi = 3.141592653589793;   // np.pi
constexpr double kInf = std::numeric_limits<double>::infinity();
// The longest a stream holds a decision open, in envelope frames: a
// beat waiting for a strong frame after it, or paths that have not met
// (a long silence). 2^16 frames, 25 minutes at librosa's 43 a second.
constexpr std::int64_t kHoldFrames = std::int64_t(1) << 16;
// The most envelope frames a second: the tempogram's window and the
// dynamic program's reach grow with it, the program's work as its
// square.
constexpr double kMaxFrameRate = 1000.0;

// An absolutely indexed run of frames that can forget its front.
template <class T>
class Roll {
public:
  std::int64_t begin() const noexcept { return _b; }
  std::int64_t end() const noexcept
  {
    return _b + (std::int64_t)_v.size();
  }
  T&       operator[](std::int64_t i) { return _v[(std::size_t)(i - _b)]; }
  const T& operator[](std::int64_t i) const
  {
    return _v[(std::size_t)(i - _b)];
  }
  void push(T x) { _v.push_back(x); }
  // A frame before begin() read as zero: for a run whose dropped front
  // is known to be zero (a stream's leading silence).
  T at(std::int64_t i) const
  {
    return i < _b ? T{} : _v[(std::size_t)(i - _b)];
  }
  // An empty run starts at frame b.
  void
  start_at(std::int64_t b)
  {
    if (_v.empty()) { _b = b; }
  }
  // Frames before `i` are no longer needed; dropped in large steps.
  void
  forget_before(std::int64_t i)
  {
    const std::int64_t k = std::min(i, end()) - _b;
    if (k < 4096) { return; }
    _v.erase(_v.begin(), _v.begin() + k);
    _b += k;
  }
  const std::vector<T>& all() const noexcept { return _v; }

private:
  std::int64_t   _b = 0;
  std::vector<T> _v;
};

// np.add.reduce over a contiguous float32 run: numpy's pairwise sum
// (8 accumulators to 128 elements, halving above), in float32. `a(i)`
// is element i of the run.
template <class Get>
float
pairwise_f32_(const Get& a, std::size_t i0, std::size_t n)
{
  if (n < 8) {
    float r = 0.0f;
    for (std::size_t i = 0; i < n; ++i) { r += a(i0 + i); }
    return r;
  }
  if (n <= 128) {
    float r[8];
    for (std::size_t j = 0; j < 8; ++j) { r[j] = a(i0 + j); }
    std::size_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (std::size_t j = 0; j < 8; ++j) { r[j] += a(i0 + i + j); }
    }
    float s = ((r[0] + r[1]) + (r[2] + r[3])) +
              ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) { s += a(i0 + i); }
    return s;
  }
  std::size_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_f32_(a, i0, n2) + pairwise_f32_(a, i0 + n2, n - n2);
}

// np.std(x, ddof=1) of float32: the sums in float32, the divisions by
// the count (a numpy intp) in float64 and rounded back. `a(i)` is
// element i of n.
template <class Get>
float
std_f32_(const Get& a, std::size_t n)
{
  if (n < 2) { return 0.0f; }
  const float mean = (float)((double)pairwise_f32_(a, 0, n) / (double)n);
  auto sq = [&](std::size_t i) {
    const float d = a(i) - mean;
    return d * d;
  };
  const float var =
      (float)((double)pairwise_f32_(sq, 0, n) / (double)(n - 1));
  return std::sqrt(var);
}

// scipy.signal.get_window('hann', M, fftbins=True): general_cosine over
// linspace(-pi, pi, M + 1), the last point dropped.
std::vector<double>
hann_periodic_(int M)
{
  if (M <= 1) { return std::vector<double>((std::size_t)std::max(M, 0), 1.0); }
  const double step = (kPi - (-kPi)) / (double)M;
  std::vector<double> w((std::size_t)M);
  for (int k = 0; k < M; ++k) {
    const double fac = (double)k * step + (-kPi);
    w[(std::size_t)k] = 0.5 + 0.5 * std::cos(fac);
  }
  return w;
}

// librosa.core.convert's Slaney mel scale.
double
hz_to_mel_(double f)
{
  const double f_min = 0.0;
  const double f_sp = 200.0 / 3;
  double mels = (f - f_min) / f_sp;
  const double min_log_hz = 1000.0;
  const double min_log_mel = (min_log_hz - f_min) / f_sp;
  const double logstep = std::log(6.4) / 27.0;
  if (f >= min_log_hz) {
    mels = min_log_mel + std::log(f / min_log_hz) / logstep;
  }
  return mels;
}

double
mel_to_hz_(double m)
{
  const double f_min = 0.0;
  const double f_sp = 200.0 / 3;
  double freq = f_min + f_sp * m;
  const double min_log_hz = 1000.0;
  const double min_log_mel = (min_log_hz - f_min) / f_sp;
  const double logstep = std::log(6.4) / 27.0;
  if (m >= min_log_mel) {
    freq = min_log_hz * std::exp(logstep * (m - min_log_mel));
  }
  return freq;
}

// The tempogram window: librosa.time_to_frames(ac_size, sr, hop).
int
tempogram_frames_(const Params& p)
{
  const auto samples = (std::int64_t)(p.ac_size * p.sample_rate);
  return (int)(samples / p.hop_length);
}

}  // namespace

std::vector<float>
mel_filterbank(const Params& p)
{
  const int nb = 1 + p.n_fft / 2;
  const int M = p.n_mels;
  const double sr = p.sample_rate;
  // np.fft.rfftfreq(n_fft, d=1/sr)
  const double d = 1.0 / sr;
  const double val = 1.0 / ((double)p.n_fft * d);
  std::vector<double> fftf((std::size_t)nb);
  for (int k = 0; k < nb; ++k) { fftf[(std::size_t)k] = (double)k * val; }
  // mel_frequencies(n_mels + 2, fmin=0, fmax=sr / 2): a linspace on the
  // mel scale, its last point the end exactly.
  const double min_mel = hz_to_mel_(0.0);
  const double max_mel = hz_to_mel_(0.5 * sr);
  const int nm = M + 2;
  const double step = (max_mel - min_mel) / (double)(nm - 1);
  std::vector<double> mel_f((std::size_t)nm);
  for (int i = 0; i < nm; ++i) {
    const double m = i == nm - 1 ? max_mel : (double)i * step + min_mel;
    mel_f[(std::size_t)i] = mel_to_hz_(m);
  }
  std::vector<float> w((std::size_t)M * nb, 0.0f);
  for (int i = 0; i < M; ++i) {
    const double fd0 = mel_f[(std::size_t)i + 1] - mel_f[(std::size_t)i];
    const double fd1 = mel_f[(std::size_t)i + 2] - mel_f[(std::size_t)i + 1];
    const double enorm =
        2.0 / (mel_f[(std::size_t)i + 2] - mel_f[(std::size_t)i]);
    for (int k = 0; k < nb; ++k) {
      const double lower = -(mel_f[(std::size_t)i] - fftf[(std::size_t)k]) /
                           fd0;
      const double upper =
          (mel_f[(std::size_t)i + 2] - fftf[(std::size_t)k]) / fd1;
      // Stored float32, then scaled by the Slaney norm in float64 and
      // stored again, as `weights *= enorm` does.
      const float v = (float)std::max(0.0, std::min(lower, upper));
      w[(std::size_t)i * nb + k] = (float)((double)v * enorm);
    }
  }
  return w;
}

struct Tracker::Impl {
  Impl(const Params& prm, Mode md);

  void push_samples(const float* x, std::size_t n);
  void push_env(const float* e, std::size_t n);
  void finish();

  void spectrum_(std::int64_t f);
  float env_frame_(std::int64_t t, float floor) const;
  void add_tempogram_(std::int64_t c, std::int64_t n_final);
  void estimate_tempo_();
  float onset_std_(bool final) const;
  const std::vector<double>& local_window_(double f);
  double local_score_(std::int64_t i, double f, float denom,
                      std::int64_t n) const;
  void dp_step_(std::int64_t i, double thresh);
  void lead_dp_(std::int64_t n, double f);
  void note_localmax_(std::int64_t i);
  bool allowed_(std::int64_t f) const;
  void commit_();
  void commit_to_(std::int64_t m);
  void add_beat_(std::int64_t frame);
  double trim_square_(std::int64_t m) const;
  double stream_threshold_() const;
  std::vector<std::int64_t> best_path_(std::int64_t stop) const;
  double final_threshold_(std::int64_t n_frames) const;
  void emit_();
  void decide_end_();
  std::int64_t zero_end_() const;
  std::int64_t held_now_() const;
  void forget_rows_();
  void forget_();
  void advance_(bool final);

  Params p;
  Mode   mode;
  const char* bad = nullptr;

  int nfft = 0, hop = 0, nmels = 0, nbin = 0;
  std::int64_t shift = 0;      // lag + n_fft / (2 hop)
  double frame_rate = 0.0;
  int W = 0, Wh = 0, npad = 0; // tempogram window, half, FFT length
  std::vector<double> stft_win, ac_win, bpms, logprior;
  std::vector<int> mel_lo;
  std::vector<std::vector<float>> mel_w;
  std::unique_ptr<pocketfft::detail::pocketfft_r<double>> fft, acfft;
  std::vector<double> fbuf, acbuf;
  std::vector<float> pbuf, dbuf;

  // Input.
  bool took_samples = false, took_env = false, done = false;
  std::vector<float> y;
  std::int64_t y0 = 0, ylen = 0, nspec = 0;
  std::vector<float> db;       // unclamped dB rows, frames [db0, nspec)
  std::int64_t db0 = 0;
  float ceiling = -std::numeric_limits<float>::infinity();

  // The onset envelope.
  Roll<float> env;
  std::int64_t nenv = 0;
  bool any = false;
  std::int64_t onset0 = -1;        // the first frame with an onset
  double wmean = 0.0, wm2 = 0.0;   // Welford, for the running deviation
  float last_std = 0.0f;

  // The tempogram.
  std::vector<double> tgsum;
  std::int64_t ntg = 0;
  double bpm = 0.0;
  int bin = 0;

  // The dynamic program.
  Roll<double> ls, cs, tb;         // local, cumulative score; frame bpm
  Roll<std::int64_t> bl;           // backlinks
  Roll<char> ok;                   // descends from the commit
  // The child of the commit a frame descends through (before the first
  // commit, the start of its chain): -1 where it does not descend.
  Roll<std::int64_t> br;
  std::int64_t ndp = 0;
  std::int64_t dp0 = -1;           // the first decision's last frame
  std::int64_t held_most = 0;      // the most frames held so far
  bool lead_dp = true;             // VPIPE_BEAT_TRACK_LEAD_DP=0: off
  std::int64_t lag_frames = 0;     // max_latency in frames, 0 = none
  bool first = true;
  double lsmax = 0.0;
  std::vector<double> lmv;         // cumulative scores at local maxima
  std::map<double, std::vector<double>> lwin;

  // Commits and the hand-out.
  std::int64_t commit = -1;
  std::vector<Beat> beats;         // the path's beats, committed
  std::vector<double> beat_ls;     // their local scores
  double sq_stable = 0.0;          // the trim's squares settled so far
  std::size_t nout = 0;            // beats[0, nout) decided on
  bool lead = false;               // the leading trim is decided
  std::int64_t mx_from = -1, mx_to = 0;
  double mx = 0.0;                 // the strongest frame in them
  std::int64_t last_out = -1;      // the last beat handed out
  std::int64_t floor_out = 0;      // none before it: the leading trim
  std::vector<Beat> ready;
  std::int64_t handed = 0;
};

Tracker::Impl::Impl(const Params& prm, Mode md) : p(prm), mode(md)
{
  nfft = p.n_fft;
  hop = p.hop_length;
  nmels = p.n_mels;
  if (!(p.sample_rate > 0.0) || hop < 1 || nfft < 2 || nmels < 1) {
    bad = "sample_rate, hop_length, n_fft and n_mels must be positive";
    return;
  }
  if (!(p.start_bpm > 0.0) || !(p.std_bpm > 0.0) || !(p.tightness > 0.0f) ||
      p.bpm < 0.0) {
    bad = "start_bpm, std_bpm and tightness must be positive and bpm "
          "not negative";
    return;
  }
  nbin = 1 + nfft / 2;
  shift = 1 + nfft / (2 * hop);
  frame_rate = p.sample_rate / (double)hop;
  if (frame_rate > kMaxFrameRate) {
    bad = "sample_rate / hop_length is over 1000 envelope frames a second";
    return;
  }
  if (const char* e = std::getenv("VPIPE_BEAT_TRACK_LEAD_DP")) {
    lead_dp = std::string(e) != "0";
  }
  if (p.max_latency > 0.0) {
    lag_frames = std::max<std::int64_t>(
        1, (std::int64_t)std::ceil(p.max_latency * frame_rate));
  }
  W = tempogram_frames_(p);
  if (W < 2) {
    bad = "the tempogram window (ac_size) is under two frames";
    return;
  }
  Wh = W / 2;
  stft_win = hann_periodic_(nfft);
  ac_win = hann_periodic_(W);
  // scipy.fft.next_fast_len(2 W - 1, real=True): 2^a 3^b 5^c.
  for (npad = 2 * W - 1;; ++npad) {
    int m = npad;
    for (int q : {2, 3, 5}) {
      while (m % q == 0) { m /= q; }
    }
    if (m == 1) { break; }
  }
  fft = std::make_unique<pocketfft::detail::pocketfft_r<double>>(
      (std::size_t)nfft);
  acfft = std::make_unique<pocketfft::detail::pocketfft_r<double>>(
      (std::size_t)npad);
  fbuf.resize((std::size_t)nfft);
  acbuf.resize((std::size_t)npad);
  pbuf.resize((std::size_t)nbin);
  dbuf.resize((std::size_t)nmels);
  // The filterbank as rows of their nonzero span.
  const std::vector<float> w = mel_filterbank(p);
  mel_lo.resize((std::size_t)nmels);
  mel_w.resize((std::size_t)nmels);
  for (int m = 0; m < nmels; ++m) {
    const float* r = w.data() + (std::size_t)m * nbin;
    int lo = 0, hi = nbin;
    while (lo < nbin && r[lo] == 0.0f) { ++lo; }
    while (hi > lo && r[hi - 1] == 0.0f) { --hi; }
    mel_lo[(std::size_t)m] = lo;
    mel_w[(std::size_t)m].assign(r + lo, r + hi);
  }
  // librosa.tempo_frequencies and the log-normal prior about start_bpm.
  bpms.resize((std::size_t)W);
  logprior.resize((std::size_t)W);
  bpms[0] = kInf;
  for (int k = 1; k < W; ++k) {
    bpms[(std::size_t)k] = 60.0 * p.sample_rate / ((double)hop * (double)k);
  }
  int max_idx = 0;
  for (int k = 0; k < W; ++k) {
    if (bpms[(std::size_t)k] < p.max_tempo) { max_idx = k; break; }
  }
  for (int k = 0; k < W; ++k) {
    const double z = (std::log2(bpms[(std::size_t)k]) -
                      std::log2(p.start_bpm)) / p.std_bpm;
    logprior[(std::size_t)k] = k < max_idx ? -kInf : -0.5 * (z * z);
  }
  tgsum.assign((std::size_t)W, 0.0);
}

void
Tracker::Impl::spectrum_(std::int64_t f)
{
  // The frame centred on sample f * hop, zeros past either end.
  const std::int64_t s0 = f * hop - nfft / 2;
  for (int j = 0; j < nfft; ++j) {
    const std::int64_t i = s0 + j;
    const float v = i >= 0 && i < ylen ? y[(std::size_t)(i - y0)] : 0.0f;
    fbuf[(std::size_t)j] = stft_win[(std::size_t)j] * (double)v;
  }
  fft->exec(fbuf.data(), 1.0, true);
  // FFTPACK order: r0, (r1, i1), ..., r[n/2] for an even n. The
  // spectrum is stored complex64, its magnitude a float32 hypot, the
  // power its square.
  auto power = [](double re, double im) {
    const float m = std::hypot((float)re, (float)im);
    return m * m;
  };
  pbuf[0] = power(fbuf[0], 0.0);
  for (int k = 1; k < nbin; ++k) {
    const bool nyq = (nfft % 2 == 0) && k == nfft / 2;
    pbuf[(std::size_t)k] =
        nyq ? power(fbuf[(std::size_t)nfft - 1], 0.0)
            : power(fbuf[(std::size_t)(2 * k - 1)],
                    fbuf[(std::size_t)(2 * k)]);
  }
  float top = -std::numeric_limits<float>::infinity();
  for (int m = 0; m < nmels; ++m) {
    const auto& w = mel_w[(std::size_t)m];
    const float* pk = pbuf.data() + mel_lo[(std::size_t)m];
    double acc = 0.0;
    for (std::size_t k = 0; k < w.size(); ++k) {
      acc += (double)w[k] * (double)pk[k];
    }
    // power_to_db: 10 log10(max(amin, S)), ref 1; the top_db ceiling is
    // applied when the envelope is taken.
    const float d = 10.0f * std::log10(std::max(1e-10f, (float)acc));
    dbuf[(std::size_t)m] = d;
    top = std::max(top, d);
  }
  db.insert(db.end(), dbuf.begin(), dbuf.end());
  ceiling = std::max(ceiling, top);
}

float
Tracker::Impl::env_frame_(std::int64_t t, float floor) const
{
  // Rows before db0 were dropped only where every frame reading them
  // was zero (forget_rows_), and a zero frame stays zero under any
  // higher floor: max(a, F) - max(b, F) only shrinks toward 0 as F
  // rises.
  if (t < shift || t - shift < db0) { return 0.0f; }
  const float* a = db.data() + (std::size_t)(t - shift + 1 - db0) * nmels;
  const float* b = a - nmels;
  std::vector<float> d((std::size_t)nmels);
  for (int m = 0; m < nmels; ++m) {
    const float x = std::max(a[m], floor);
    const float z = std::max(b[m], floor);
    d[(std::size_t)m] = std::max(0.0f, x - z);
  }
  // np.median: the middle value, or the mean of the middle two.
  const std::size_t h = (std::size_t)nmels / 2;
  std::nth_element(d.begin(), d.begin() + (std::ptrdiff_t)h, d.end());
  const float hi = d[h];
  if (nmels % 2 == 1) { return hi; }
  const float lo = *std::max_element(d.begin(), d.begin() + (std::ptrdiff_t)h);
  return (lo + hi) * 0.5f;
}

// One centred tempogram frame, added to the sum: the envelope around c
// (np.pad's linear ramps to zero past its ends -- the right one only
// once the end is known, n_final >= 0), Hann-windowed, autocorrelated
// through an FFT as librosa.autocorrelate does, normalised by its peak.
// np.pad's end values are an int64 array, so its ramps are linspaces
// in float64, stored float32.
void
Tracker::Impl::add_tempogram_(std::int64_t c, std::int64_t n_final)
{
  const double l_step = nenv > 0 ? (double)env.at(0) / (double)Wh : 0.0;
  const double r_step =
      n_final > 0 ? (double)env.at(n_final - 1) / (double)Wh : 0.0;
  // A window of zeros (a leading silence) adds zeros: its peak is 0,
  // read as 1. Only the count moves.
  if (l_step == 0.0 && c - Wh + W <= zero_end_()) {
    ++ntg;
    return;
  }
  std::fill(acbuf.begin(), acbuf.end(), 0.0);
  for (int j = 0; j < W; ++j) {
    const std::int64_t t = c - Wh + j;
    float v;
    if (t < 0) {
      v = (float)((double)(Wh + t) * l_step);          // up from 0
    } else if (n_final >= 0 && t >= n_final) {
      v = (float)((double)(Wh - 1 - (t - n_final)) * r_step);  // to 0
    } else {
      v = env.at(t);
    }
    acbuf[(std::size_t)j] = (double)v * ac_win[(std::size_t)j];
  }
  acfft->exec(acbuf.data(), 1.0, true);
  // |X|^2 back into FFTPACK order, imaginary parts zero.
  acbuf[0] = acbuf[0] * acbuf[0];
  for (int k = 1; k < (npad + 1) / 2; ++k) {
    const double re = acbuf[(std::size_t)(2 * k - 1)];
    const double im = acbuf[(std::size_t)(2 * k)];
    acbuf[(std::size_t)(2 * k - 1)] = re * re + im * im;
    acbuf[(std::size_t)(2 * k)] = 0.0;
  }
  if (npad % 2 == 0) {
    const double re = acbuf[(std::size_t)npad - 1];
    acbuf[(std::size_t)npad - 1] = re * re;
  }
  acfft->exec(acbuf.data(), 1.0 / (double)npad, false);
  double peak = 0.0;
  for (int k = 0; k < W; ++k) {
    peak = std::max(peak, std::fabs(acbuf[(std::size_t)k]));
  }
  if (peak < DBL_MIN) { peak = 1.0; }
  for (int k = 0; k < W; ++k) {
    tgsum[(std::size_t)k] += acbuf[(std::size_t)k] / peak;
  }
  ++ntg;
}

void
Tracker::Impl::estimate_tempo_()
{
  if (p.bpm > 0.0) {
    bpm = p.bpm;
    return;
  }
  if (ntg == 0) { return; }
  double best = -kInf;
  int arg = 0;
  for (int k = 0; k < W; ++k) {
    const double mean = tgsum[(std::size_t)k] / (double)ntg;
    const double s = std::log1p(1e6 * mean) + logprior[(std::size_t)k];
    if (s > best) {
      best = s;
      arg = k;
    }
  }
  bin = arg;
  bpm = bpms[(std::size_t)arg];
}

float
Tracker::Impl::onset_std_(bool final) const
{
  // numpy's float32 deviation when the end is decided with the whole
  // envelope in hand -- librosa's -- and a running float64 one on the
  // way, which is the same figure to a few float32 roundings.
  // The frames dropped before the first onset are zeros, and numpy's
  // sums are rebuilt over them in place.
  if (final && env.begin() <= zero_end_()) {
    const std::int64_t z = env.begin();
    const std::vector<float>& v = env.all();
    auto at = [&](std::size_t i) {
      return (std::int64_t)i < z ? 0.0f : v[i - (std::size_t)z];
    };
    return std_f32_(at, (std::size_t)nenv);
  }
  if (nenv < 2) { return 0.0f; }
  return (float)std::sqrt(wm2 / (double)(nenv - 1));
}

// np.exp(-0.5 * (np.arange(-f, f + 1) * 32.0 / f) ** 2), in float64.
const std::vector<double>&
Tracker::Impl::local_window_(double f)
{
  auto it = lwin.find(f);
  if (it != lwin.end()) { return it->second; }
  const int K = 2 * (int)f + 1;
  std::vector<double> w((std::size_t)K);
  for (int k = 0; k < K; ++k) {
    const double t = (-f + (double)k) * 32.0 / f;
    w[(std::size_t)k] = std::exp(-0.5 * (t * t));
  }
  return lwin.emplace(f, std::move(w)).first->second;
}

// librosa's __beat_local_score at frame i, the envelope divided by
// `denom` in float32 and widened: its loop bounds as written, which
// leave the envelope's frame 0 out (it is always zero).
double
Tracker::Impl::local_score_(std::int64_t i, double f, float denom,
                            std::int64_t n) const
{
  const auto it = lwin.find(f);
  const std::vector<double>& w = it->second;
  const std::int64_t K = (std::int64_t)w.size();
  const std::int64_t h = K / 2;
  const std::int64_t k0 = std::max<std::int64_t>(0, i + h - n + 1);
  const std::int64_t k1 = std::min<std::int64_t>(i + h, K);
  double s = 0.0;
  for (std::int64_t k = k0; k < k1; ++k) {
    const float on = env.at(i + h - k) / denom;
    s += w[(std::size_t)k] * (double)on;
  }
  return s;
}

bool
Tracker::Impl::allowed_(std::int64_t f) const
{
  return commit < 0 || f == commit || (f > commit && ok[f] != 0);
}

// One step of librosa's __beat_track_dp: the best predecessor between
// two periods and half a period back (the nearest wins a tie), the
// score's tempo penalty in float64. A committed tracker only looks at
// frames descending from its commit.
void
Tracker::Impl::dp_step_(std::int64_t i, double thresh)
{
  const double f = tb[i] > 0.0 ? std::nearbyint(frame_rate * 60.0 / tb[i])
                               : 1.0;
  // A period under 2 frames would make the frame its own predecessor;
  // no tempo the tempogram can name gets there (320 bpm is 8 frames).
  const std::int64_t lo =
      std::min(i - 1, i - (std::int64_t)std::nearbyint(f / 2.0));
  const std::int64_t hi = i - 2 * (std::int64_t)f;
  const double lf = std::log(f);
  const double tight = (double)p.tightness;
  double best = -kInf;
  std::int64_t at = -1;
  for (std::int64_t loc = lo; loc >= hi; --loc) {
    if (loc < 0) { break; }
    if (!allowed_(loc)) { continue; }
    const double g = std::log((double)(i - loc)) - lf;
    const double s = cs[loc] - tight * (g * g);
    if (s > best) {
      best = s;
      at = loc;
    }
  }
  const double li = ls[i];
  cs.push(at >= 0 ? li + best : li);
  if (first && li < thresh) {
    bl.push(-1);
  } else {
    bl.push(at);
    first = false;
  }
  const std::int64_t b = bl[i];
  ok.push(commit < 0 ||
                  (b >= 0 && (b == commit || (b > commit && ok[b] != 0)))
              ? 1 : 0);
  br.push(b < 0 ? (commit < 0 ? i : -1)
                : b == commit ? i : b > commit ? br[b] : -1);
  ndp = i + 1;
  note_localmax_(i);
}

// dp_step_ over a stream's leading silence, frames [0, n), at the first
// decision, without holding them: every local score there is 0 (the
// frames' windows end before the first onset), so no chain starts
// (backlink -1) and each cumulative score is the best of the last two
// periods' less the tempo penalty. That recursion, run on zeros, turns
// periodic -- once 2f scores in a row equal those a period before, the
// next is computed from equal values, so it equals its own -- and is
// then carried a period at a time. The last 2f + 2 frames, all a later
// step reads, are held; the local maxima go to lmv as dp_step_'s do.
void
Tracker::Impl::lead_dp_(std::int64_t n, double f)
{
  const auto fi = (std::int64_t)f;
  const std::int64_t span = 2 * fi + 2;
  const std::int64_t b0 = n - span;
  std::vector<double> ring((std::size_t)span);
  auto at = [&](std::int64_t i) -> double& {
    return ring[(std::size_t)(i % span)];
  };
  auto same = [](double a, double b) {
    return std::memcmp(&a, &b, sizeof a) == 0;
  };
  const std::int64_t half = (std::int64_t)std::nearbyint(f / 2.0);
  const double lf = std::log(f);
  const double tight = (double)p.tightness;
  std::int64_t run = 0;   // the latest frames equal to a period before
  for (std::int64_t i = 0; i < n; ++i) {
    double c;
    if (run >= 2 * fi) {
      c = at(i - fi);
    } else {
      const std::int64_t lo = std::min(i - 1, i - half);
      const std::int64_t hi = i - 2 * fi;
      double best = -kInf;
      std::int64_t loc_at = -1;
      for (std::int64_t loc = lo; loc >= hi; --loc) {
        if (loc < 0) { break; }
        const double g = std::log((double)(i - loc)) - lf;
        const double sc = at(loc) - tight * (g * g);
        if (sc > best) {
          best = sc;
          loc_at = loc;
        }
      }
      c = loc_at >= 0 ? 0.0 + best : 0.0;
    }
    if (i >= fi && same(c, at(i - fi))) {
      ++run;
    } else {
      run = 0;
    }
    at(i) = c;
    const std::int64_t m = i - 1;
    if (m >= 1 && at(m) > at(m - 1) && at(m) >= c) { lmv.push_back(at(m)); }
  }
  for (Roll<double>* r : {&ls, &cs, &tb}) { r->start_at(b0); }
  bl.start_at(b0);
  ok.start_at(b0);
  br.start_at(b0);
  for (std::int64_t i = b0; i < n; ++i) {
    ls.push(0.0);
    tb.push(bpm);
    cs.push(at(i));
    bl.push(-1);
    ok.push(1);
    br.push(i);
  }
  ndp = n;
}

// util.localmax over the cumulative score, as frames arrive: frame i - 1
// once frame i is in (the last frame is decided at the end).
void
Tracker::Impl::note_localmax_(std::int64_t i)
{
  const std::int64_t m = i - 1;
  if (m < 1) { return; }
  if (cs[m] > cs[m - 1] && cs[m] >= cs[i]) { lmv.push_back(cs[m]); }
}

// The most recent frame every extensible path passes through. A future
// frame takes its predecessor at most two periods back, so the paths
// that matter end in the last two periods; their backtracks meet, and
// where they meet is final.
void
Tracker::Impl::commit_()
{
  if (ndp < 2) { return; }
  const std::int64_t last = ndp - 1;
  const double f = std::nearbyint(frame_rate * 60.0 / tb[last]);
  const std::int64_t lo = std::max<std::int64_t>(
      {commit + 1, last + 1 - 2 * (std::int64_t)f, ls.begin()});
  // The paths meet past the commit only if they all leave it through
  // one child (or, before the first, start one chain): asked first, it
  // spares the walks back where they do not (a silence, where they run
  // side by side to the commit every frame).
  std::int64_t one = -2;
  for (std::int64_t fr = last; fr >= lo; --fr) {
    if (!allowed_(fr)) { continue; }
    if (one == -2) {
      one = br[fr];
    } else if (br[fr] != one) {
      one = -1;
      break;
    }
  }
  if (one < 0) {
    // Paths that have not met for kHoldFrames (a long silence): the
    // best path's frame half that far back is committed, so what is
    // held stays bounded. librosa, with the whole track, would take
    // the path the sound after the silence chooses.
    if (last - std::max(commit, dp0) <= kHoldFrames) { return; }
    std::int64_t m = -1;
    for (std::int64_t fr : best_path_(commit)) {
      if (fr > last - kHoldFrames / 2) { break; }
      m = fr;
    }
    if (m > commit) { commit_to_(m); }
    return;
  }
  std::int64_t m = -2;
  for (std::int64_t fr = last; fr >= lo; --fr) {
    if (!allowed_(fr)) { continue; }
    if (m == -2) {
      m = fr;
      continue;
    }
    std::int64_t a = m, b = fr;
    const std::int64_t held = ls.begin();
    while (a != b && a >= held && b >= held) {
      if (a > b) {
        a = bl[a];
      } else {
        b = bl[b];
      }
    }
    if (a != b || a < held) {          // separate trees: not yet
      m = -2;
      break;
    }
    m = a;
  }
  if (m > commit) { commit_to_(m); }
}

// Commit frame m, which descends from the commit: the beats of the path
// back to it, and from now on only frames descending from m.
void
Tracker::Impl::commit_to_(std::int64_t m)
{
  const std::int64_t last = ndp - 1;
  std::vector<std::int64_t> seg;
  for (std::int64_t n = m; n > commit; n = bl[n]) {
    seg.push_back(n);
    if (bl[n] < 0) { break; }
  }
  for (auto it = seg.rbegin(); it != seg.rend(); ++it) { add_beat_(*it); }
  commit = m;
  for (std::int64_t fr = m + 1; fr <= last; ++fr) {
    const std::int64_t b = bl[fr];
    const bool d = b >= 0 && (b == m || (b > m && ok[b] != 0));
    ok[fr] = d ? 1 : 0;
    br[fr] = !d ? -1 : b == m ? fr : br[b];
  }
  ok[m] = 1;
  br[m] = -1;
}

// librosa's trim threshold is half the RMS of the beats' local scores
// convolved with np.hanning(5) = [0, .5, 1, .5, 0], over the full
// convolution's indices [2, n_frames + 2) -- two past a 'same'
// convolution. Index m of it is .5 b[m-1] + b[m-2] + .5 b[m-3].
void
Tracker::Impl::add_beat_(std::int64_t frame)
{
  beats.push_back({frame, tb[frame]});
  beat_ls.push_back(ls[frame]);
  // Index m = nb now has every term it will have.
  const std::int64_t nb = (std::int64_t)beat_ls.size();
  if (nb >= 2) { sq_stable += trim_square_(nb); }
}

double
Tracker::Impl::trim_square_(std::int64_t m) const
{
  const std::int64_t nb = (std::int64_t)beat_ls.size();
  auto b = [&](std::int64_t j) {
    return j >= 0 && j < nb ? beat_ls[(std::size_t)j] : 0.0;
  };
  const double s = 0.5 * b(m - 1) + b(m - 2) + 0.5 * b(m - 3);
  return s * s;
}

// On the way: over the beats committed so far, the settled indices
// [2, nb] summed as they settled, the last three as they stand.
double
Tracker::Impl::stream_threshold_() const
{
  if (!p.trim) { return 0.0; }
  const std::int64_t nb = (std::int64_t)beat_ls.size();
  if (nb == 0) { return 0.0; }
  double sum = sq_stable;
  for (std::int64_t m = std::max<std::int64_t>(2, nb + 1); m < nb + 4; ++m) {
    sum += trim_square_(m);
  }
  return 0.5 * std::sqrt(sum / (double)(nb + 2));
}

// At the end, exactly librosa's: every index in order, bounded by the
// frame count. 0 without `trim`.
double
Tracker::Impl::final_threshold_(std::int64_t n_frames) const
{
  if (!p.trim) { return 0.0; }
  const std::int64_t nb = (std::int64_t)beat_ls.size();
  if (nb == 0) { return 0.0; }
  const std::int64_t end = std::min<std::int64_t>(nb + 4, n_frames + 2);
  double sum = 0.0;
  std::int64_t cnt = 0;
  for (std::int64_t m = 2; m < end; ++m) {
    sum += trim_square_(m);
    ++cnt;
  }
  return cnt > 0 ? 0.5 * std::sqrt(sum / (double)cnt) : 0.0;
}

// Hand out the committed beats `trim` will not take back: the first
// time, from the first frame stronger than the threshold; then each
// beat once a frame at or after it is. With max_latency, a beat that
// old is handed out whatever is undecided: a held one, or the best path
// so far's -- and a later decision that disagrees continues after it,
// so nothing handed out is revised and the tracker is not bound to a
// guess. Beats held past kHoldFrames for want of a strong frame after
// them (a long silence) are let go -- the trailing trim's answer if the
// stream ended there -- so a silent stream holds nothing without bound.
void
Tracker::Impl::emit_()
{
  const std::int64_t heard = nenv - 1;
  // A beat is due max_latency after it was heard -- but the dynamic
  // program cannot place one nearer the sound than a period.
  const std::int64_t due =
      lag_frames > 0 ? std::min(ndp - 1, heard - lag_frames) : -1;
  if (!lead) {
    // The leading trim: the beats before the first frame stronger than
    // the threshold over the path's beats -- the committed ones, or once
    // a beat is due and none is, the best path's so far.
    double th = 0.0;
    if (nout < beats.size()) {
      th = stream_threshold_();
    } else if (due >= 0) {
      const auto path = best_path_(-1);
      if (path.empty() || path.front() > due) { return; }
      std::vector<double> sc;
      for (std::int64_t f : path) { sc.push_back(ls[f]); }
      std::swap(beat_ls, sc);
      th = stream_threshold_();
      std::swap(beat_ls, sc);
    } else {
      return;
    }
    if (!(lsmax > th)) { return; }
    std::int64_t f0 = ls.begin();
    while (f0 < ndp && !(ls[f0] > th)) { ++f0; }
    floor_out = f0;
    while (nout < beats.size() && beats[nout].frame < f0) { ++nout; }
    lead = true;
  }
  const double th = stream_threshold_();
  // No two beats nearer than the dynamic program allows: half a period.
  const std::int64_t gap = (std::int64_t)std::nearbyint(
      std::nearbyint(frame_rate * 60.0 / bpm) / 2.0);
  auto hand = [&](const Beat& b) {
    if (b.frame < floor_out) { return; }
    if (last_out >= 0 && b.frame < last_out + gap) { return; }
    ready.push_back(b);
    ++handed;
    last_out = b.frame;
  };
  while (nout < beats.size() && heard - beats[nout].frame > kHoldFrames) {
    ++nout;
  }
  while (due >= 0 && nout < beats.size() && beats[nout].frame <= due) {
    hand(beats[nout++]);
  }
  if (nout < beats.size()) {
    // The strongest frame from the oldest beat held on, kept as it grows.
    const std::int64_t b0 = beats[nout].frame;
    if (mx_from != b0) {
      mx_from = b0;
      mx_to = b0;
      mx = -kInf;
    }
    for (; mx_to < ndp; ++mx_to) { mx = std::max(mx, ls[mx_to]); }
    if (mx > th) {
      std::int64_t strong = ndp - 1;
      while (strong >= b0 && !(ls[strong] > th)) { --strong; }
      while (nout < beats.size() && beats[nout].frame <= strong) {
        hand(beats[nout++]);
      }
    }
  }
  if (due < 0 || due <= last_out) { return; }
  // The best path so far: its beats that are due and not yet said.
  for (std::int64_t f : best_path_(last_out)) {
    if (f <= due) { hand({f, tb[f]}); }
  }
  // What the committed path has at or before the last beat handed out
  // is said.
  while (nout < beats.size() && beats[nout].frame < last_out + gap) {
    ++nout;
  }
}

// The best path so far, ascending, its frames after `stop`: back from
// the frame of the last period with the most cumulative score.
std::vector<std::int64_t>
Tracker::Impl::best_path_(std::int64_t stop) const
{
  std::vector<std::int64_t> path;
  if (ndp < 1) { return path; }
  const std::int64_t last = ndp - 1;
  const auto f = (std::int64_t)std::nearbyint(frame_rate * 60.0 / bpm);
  std::int64_t best = -1;
  for (std::int64_t fr = last; fr > last - f && fr >= ls.begin(); --fr) {
    if (allowed_(fr) && (best < 0 || cs[fr] > cs[best])) { best = fr; }
  }
  for (std::int64_t n = best; n > stop && n >= ls.begin(); n = bl[n]) {
    path.push_back(n);
    if (bl[n] < 0) { break; }
  }
  std::reverse(path.begin(), path.end());
  return path;
}

// The end: librosa's last beat, among the frames the commits allow; the
// path back from it to the commit; the trim over every beat of the path.
void
Tracker::Impl::decide_end_()
{
  const std::int64_t N = ndp;
  if (N == 0) { return; }
  if (N >= 2 && cs[N - 1] > cs[N - 2]) { lmv.push_back(cs[N - 1]); }
  bool have_thr = !lmv.empty();
  double thr = 0.0;
  if (have_thr) {
    std::vector<double> v = lmv;
    const std::size_t h = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + (std::ptrdiff_t)h, v.end());
    double med = v[h];
    if (v.size() % 2 == 0) {
      const double lo =
          *std::max_element(v.begin(), v.begin() + (std::ptrdiff_t)h);
      med = (lo + med) / 2.0;
    }
    thr = 0.5 * med;
  }
  auto is_lmax = [&](std::int64_t n) {
    // The frame before the first held one is gone: not one to end on.
    if (n <= 0 || n <= cs.begin()) { return false; }
    if (n == N - 1) { return cs[n] > cs[n - 1]; }
    return cs[n] > cs[n - 1] && cs[n] >= cs[n + 1];
  };
  std::int64_t tail = -1;
  const std::int64_t floor = std::max<std::int64_t>(commit, ls.begin());
  if (have_thr) {
    for (std::int64_t n = N - 1; n >= floor && n >= 0; --n) {
      if (is_lmax(n) && cs[n] >= thr && allowed_(n)) {
        tail = n;
        break;
      }
    }
  }
  if (tail < 0) {
    // librosa's default is the last frame; a committed tracker's, the
    // last it may still end on.
    for (std::int64_t n = N - 1; n >= floor && n >= 0; --n) {
      if (allowed_(n)) {
        tail = n;
        break;
      }
    }
  }
  std::vector<std::int64_t> path;
  for (std::int64_t n = tail; n > commit && n >= 0; n = bl[n]) {
    path.push_back(n);
  }
  for (auto it = path.rbegin(); it != path.rend(); ++it) { add_beat_(*it); }
  const double th = final_threshold_(N);
  std::int64_t f0 = ls.begin();   // the frames before score 0
  if (!lead) {
    while (f0 < N && ls[f0] <= th) { ++f0; }
    lead = true;
  }
  std::int64_t fl = N - 1;
  while (fl >= ls.begin() && ls[fl] <= th) { --fl; }
  const std::int64_t gap = (std::int64_t)std::nearbyint(
      std::nearbyint(frame_rate * 60.0 / bpm) / 2.0);
  while (nout < beats.size()) {
    const Beat b = beats[nout++];
    if (b.frame >= f0 && b.frame >= floor_out && b.frame <= fl &&
        (last_out < 0 || b.frame >= last_out + gap)) {
      ready.push_back(b);
      ++handed;
      last_out = b.frame;
    }
  }
}

// The envelope frames known to be zero, [0, zero_end_()): those before
// the first onset (all of them, before one).
std::int64_t
Tracker::Impl::zero_end_() const
{
  if (!took_samples) { return 0; }
  return onset0 < 0 ? nenv : onset0;
}

// What a stream holds from its first frame on, warm-up and leading
// silence included. The dB rows: the next envelope frame reads two --
// and until the first decision, from the first onset's on, which a
// stream that ends there reads again under the whole track's ceiling
// (advance_). The envelope: the zeros before the first onset, read
// back as zeros (Roll::at).
void
Tracker::Impl::forget_rows_()
{
  if (mode == Mode::kWhole) { return; }
  std::int64_t row0 = nenv - shift;
  if (ndp == 0 && onset0 >= 0) { row0 = std::min(row0, onset0 - shift); }
  row0 = std::max<std::int64_t>(row0, 0);
  if (row0 - db0 >= 4096) {
    db.erase(db.begin(), db.begin() + (std::ptrdiff_t)((row0 - db0) * nmels));
    db0 = row0;
  }
  env.forget_before(zero_end_());
}

void
Tracker::Impl::forget_()
{
  if (mode == Mode::kWhole) { return; }
  // Nothing of the scores goes before the leading trim is decided, nor
  // before the first commit: until then a path's backtrack may reach
  // the stream's start.
  if (!lead || commit < 0) { return; }
  std::int64_t keep = ndp - 2 * W - 2;
  if (commit >= 0) { keep = std::min(keep, commit); }
  if (nout < beats.size()) { keep = std::min(keep, beats[nout].frame); }
  if (lag_frames > 0 && last_out >= 0) { keep = std::min(keep, last_out); }
  keep = std::max<std::int64_t>(keep, 0);
  ls.forget_before(keep);
  cs.forget_before(keep);
  tb.forget_before(keep);
  bl.forget_before(keep);
  ok.forget_before(keep);
  br.forget_before(keep);
  // The envelope: the next tempogram frame's window and the next local
  // scores' reach.
  std::int64_t ekeep = std::min(ntg - Wh, ndp - W - 2);
  ekeep = std::max<std::int64_t>(ekeep, 0);
  if (ekeep - env.begin() >= 4096) {
    env.forget_before(ekeep);
  }
}

// One step: everything a new spectrogram frame makes decidable (or, at
// the end, everything). Steps go a frame at a time, so the decisions
// depend on the frames alone and not on how the samples were cut.
void
Tracker::Impl::advance_(bool final)
{
  if (mode == Mode::kWhole && !final) { return; }
  // A stream that ends before its first decision is decided as a whole
  // track: its envelope again, under the whole track's ceiling.
  // The rows before db0 were dropped in a leading silence: the frames
  // reading them are zeros (as Welford's statistics over them are), so
  // the envelope starts after them.
  if (final && ndp == 0 && took_samples && nenv > 0 &&
      (db0 == 0 || (onset0 >= 0 && db0 + shift <= onset0))) {
    const std::int64_t z = db0 > 0 ? db0 + shift : 0;
    env = Roll<float>();
    env.start_at(z);
    nenv = z;
    any = false;
    onset0 = -1;
    wmean = wm2 = 0.0;
  }
  // The envelope, below the loudest bin so far less top_db: from the
  // tempogram's first window on (before it, nothing can be decided).
  if (took_samples && (final || nspec >= W)) {
    const float floor = ceiling - p.top_db;
    while (nenv < nspec) {
      const float e = env_frame_(nenv, floor);
      env.push(e);
      if (e > 0.0f && onset0 < 0) { onset0 = nenv; }
      any = any || e > 0.0f;
      ++nenv;
      const double d = (double)e - wmean;
      wmean += d / (double)nenv;
      wm2 += d * ((double)e - wmean);
    }
  }
  // Nothing is decided before the tempogram's window (ac_size) has been
  // heard past the first onset: librosa's statistics need a sample.
  if (!final && (onset0 < 0 || nenv - onset0 < W)) { return; }
  // The tempogram's frames whose windows are in; at the end all of them.
  if (p.bpm <= 0.0) {
    const std::int64_t reach = W - Wh - 1;
    while (ntg < nenv && (final || ntg + reach <= nenv - 1)) {
      add_tempogram_(ntg, final ? nenv : -1);
    }
  }
  estimate_tempo_();
  if (!any || !(bpm > 0.0)) {
    if (final) { bpm = 0.0; }
    return;   // librosa: no onset, no beats
  }
  const double f = std::nearbyint(frame_rate * 60.0 / bpm);
  const std::vector<double>& w = local_window_(f);
  const std::int64_t h = (std::int64_t)w.size() / 2;
  const float sd = onset_std_(final);
  last_std = sd;
  const float denom = sd + FLT_MIN;   // + util.tiny(float32)
  // The frames whose local score is in reach: all at the end, else
  // those a period or more behind the envelope. Before the end the
  // envelope's length bounds nothing (librosa's loop clamps at it).
  const std::int64_t upto = final ? nenv - 1 : nenv - 1 - h;
  const std::int64_t n_clamp =
      final ? nenv : std::numeric_limits<std::int64_t>::max() / 4;
  // A leading silence at the first decision: the frames whose window
  // ends before the first onset score zero, and lead_dp_ runs them
  // without holding them.
  std::int64_t from = ndp;
  if (lead_dp && !final && ndp == 0 && f >= 2.0 &&
      onset0 - h >= 2 * (std::int64_t)f + 2 + 4096) {
    from = std::min(onset0 - h, upto + 1);
  }
  std::vector<double> fresh;
  for (std::int64_t i = from; i <= upto; ++i) {
    fresh.push_back(local_score_(i, f, denom, n_clamp));
    lsmax = std::max(lsmax, fresh.back());
  }
  // No chain may start below 1% of the strongest frame: librosa's over
  // the whole track, here over every frame scored so far -- the first
  // decision's, over the whole warm-up.
  const double thresh = 0.01 * lsmax;
  if (from > ndp) {
    // A frame in reach of the onset scores above 0, so no chain starts
    // in the silence; were that not so, it is scored in full.
    if (thresh > 0.0) {
      lead_dp_(from, f);
    } else {
      for (std::int64_t i = ndp; i < from; ++i) {
        ls.push(local_score_(i, f, denom, n_clamp));
        tb.push(bpm);
      }
    }
  }
  for (std::int64_t i = from; i <= upto; ++i) {
    ls.push(fresh[(std::size_t)(i - from)]);
    tb.push(bpm);
  }
  for (std::int64_t i = ndp; i <= upto; ++i) { dp_step_(i, thresh); }
  if (dp0 < 0) { dp0 = upto; }
  held_most = std::max(held_most, held_now_());
  if (final) {
    decide_end_();
    return;
  }
  commit_();
  emit_();
  forget_();
}

void
Tracker::Impl::push_samples(const float* x, std::size_t n)
{
  if (bad != nullptr || done || took_env) { return; }
  took_samples = true;
  y.insert(y.end(), x, x + n);
  ylen += (std::int64_t)n;
  while (nspec * hop + nfft / 2 <= ylen) {
    spectrum_(nspec);
    ++nspec;
    advance_(false);
    held_most = std::max(held_most, held_now_());
    forget_rows_();
  }
  const std::int64_t keep = std::max<std::int64_t>(0, nspec * hop - nfft / 2);
  if (keep - y0 >= (1 << 16)) {
    y.erase(y.begin(), y.begin() + (std::ptrdiff_t)(keep - y0));
    y0 = keep;
  }
}

void
Tracker::Impl::push_env(const float* e, std::size_t n)
{
  if (bad != nullptr || done || took_samples || mode != Mode::kWhole) {
    return;
  }
  took_env = true;
  for (std::size_t i = 0; i < n; ++i) {
    env.push(e[i]);
    any = any || e[i] > 0.0f;
    ++nenv;
  }
  nspec = nenv;
}

void
Tracker::Impl::finish()
{
  if (bad != nullptr || done) { return; }
  done = true;
  if (took_samples) {
    // librosa's frame count: 1 + len // hop, zeros past the end.
    const std::int64_t frames = ylen > 0 ? 1 + ylen / hop : 0;
    while (nspec < frames) {
      spectrum_(nspec);
      ++nspec;
    }
  }
  advance_(true);
}

// ---- Tracker ----

Tracker::Tracker(const Params& p, Mode mode)
  : _m(std::make_unique<Impl>(p, mode))
{
}

Tracker::~Tracker() = default;

const char*
Tracker::invalid() const noexcept
{
  return _m->bad != nullptr ? _m->bad : "";
}

void
Tracker::push(const float* x, std::size_t n)
{
  _m->push_samples(x, n);
}

void
Tracker::push_envelope(const float* env, std::size_t n)
{
  _m->push_env(env, n);
}

void
Tracker::finish()
{
  _m->finish();
}

bool
Tracker::finished() const noexcept
{
  return _m->done;
}

std::vector<Beat>
Tracker::take()
{
  std::vector<Beat> out;
  out.swap(_m->ready);
  return out;
}

double
Tracker::tempo() const noexcept
{
  return _m->bpm;
}

std::int64_t
Tracker::frames() const noexcept
{
  return _m->nspec;
}

std::int64_t
Tracker::beats_out() const noexcept
{
  return _m->handed;
}

const std::vector<float>&
Tracker::onset_envelope() const noexcept
{
  return _m->env.all();
}

const std::vector<double>&
Tracker::local_score() const noexcept
{
  return _m->ls.all();
}

const std::vector<double>&
Tracker::cumulative_score() const noexcept
{
  return _m->cs.all();
}

int
Tracker::tempo_bin() const noexcept
{
  return _m->bin;
}

float
Tracker::onset_std() const noexcept
{
  return _m->last_std;
}

std::int64_t
Tracker::committed() const noexcept
{
  return (std::int64_t)_m->beats.size();
}

std::int64_t
Tracker::commit_frame() const noexcept
{
  return _m->commit;
}

std::int64_t
Tracker::Impl::held_now_() const
{
  const std::int64_t rows = (std::int64_t)db.size() / std::max(1, nmels);
  return std::max({rows, (std::int64_t)env.all().size(),
                   (std::int64_t)ls.all().size()});
}

std::int64_t
Tracker::held_frames() const noexcept
{
  return std::max(_m->held_most, _m->held_now_());
}

Result
beat_track(const float* y, std::size_t n, const Params& p)
{
  Tracker t(p, Mode::kWhole);
  t.push(y, n);
  t.finish();
  Result r;
  r.tempo = t.tempo();
  for (const Beat& b : t.take()) { r.beats.push_back(b.frame); }
  return r;
}

}  // namespace vpipe::beat
