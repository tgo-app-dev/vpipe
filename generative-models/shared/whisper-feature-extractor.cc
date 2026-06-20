#include "generative-models/shared/whisper-feature-extractor.h"

// pocketfft is a header-only FFT library (BSD-3-clause). Vendored into
// the vpipe tree so the feature extractor builds without MLX (the
// previous O(N^2) DFT inner loop was 87% of the encoder's wall time on
// a 6 s clip; pocketfft drops that to a few ms).
#include "3rd-party/pocketfft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

namespace vpipe::genai {

namespace {

// ---- Mel scale helpers (Slaney / HTK hybrid, matching librosa) ----

// Convert a frequency in Hz to a mel value using the Slaney scale
// (HF / librosa default). Linear below 1000 Hz, logarithmic above.
float hz_to_mel_(float hz)
{
  constexpr float f_min   = 0.0f;
  constexpr float f_sp    = 200.0f / 3.0f;        // linear slope
  constexpr float min_log_hz   = 1000.0f;
  constexpr float min_log_mel  = (min_log_hz - f_min) / f_sp;
  const float     logstep      = std::log(6.4f) / 27.0f;
  if (hz >= min_log_hz) {
    return min_log_mel + std::log(hz / min_log_hz) / logstep;
  }
  return (hz - f_min) / f_sp;
}

float mel_to_hz_(float mel)
{
  constexpr float f_min   = 0.0f;
  constexpr float f_sp    = 200.0f / 3.0f;
  constexpr float min_log_hz   = 1000.0f;
  constexpr float min_log_mel  = (min_log_hz - f_min) / f_sp;
  const float     logstep      = std::log(6.4f) / 27.0f;
  if (mel >= min_log_mel) {
    return min_log_hz * std::exp((mel - min_log_mel) * logstep);
  }
  return f_min + f_sp * mel;
}

// Build a [n_mels, n_freqs] mel filterbank. Slaney-normalised
// triangular filters; matches librosa.filters.mel(htk=False,
// norm="slaney") which is what HF WhisperFeatureExtractor uses.
//
// n_freqs == n_fft/2 + 1; bins span 0..sr/2 Hz uniformly. The output
// is stored row-major: out[m * n_freqs + k].
std::vector<float>
build_mel_filters_(int   n_fft,
                   int   n_mels,
                   int   sample_rate,
                   float f_min,
                   float f_max)
{
  const int n_freqs = n_fft / 2 + 1;
  std::vector<float> filt(static_cast<std::size_t>(n_mels) * n_freqs,
                          0.0f);

  // Frequency value at each DFT bin.
  std::vector<float> fft_freqs(n_freqs);
  for (int k = 0; k < n_freqs; ++k) {
    fft_freqs[k] = static_cast<float>(k) * sample_rate
                 / static_cast<float>(n_fft);
  }

  // mel-spaced center frequencies, including the two outer "edges"
  // (n_mels + 2 points total).
  const float mel_min = hz_to_mel_(f_min);
  const float mel_max = hz_to_mel_(f_max);
  std::vector<float> mels(n_mels + 2);
  for (int i = 0; i < n_mels + 2; ++i) {
    mels[i] = mel_min + (mel_max - mel_min)
                       * static_cast<float>(i)
                       / static_cast<float>(n_mels + 1);
  }
  std::vector<float> centers(n_mels + 2);
  for (int i = 0; i < n_mels + 2; ++i) {
    centers[i] = mel_to_hz_(mels[i]);
  }

  for (int m = 0; m < n_mels; ++m) {
    const float left   = centers[m];
    const float center = centers[m + 1];
    const float right  = centers[m + 2];
    const float left_w  = 1.0f / (center - left);
    const float right_w = 1.0f / (right - center);
    for (int k = 0; k < n_freqs; ++k) {
      float f  = fft_freqs[k];
      float lo = (f - left) * left_w;
      float hi = (right - f) * right_w;
      float w  = std::max(0.0f, std::min(lo, hi));
      filt[static_cast<std::size_t>(m) * n_freqs + k] = w;
    }
    // Slaney normalisation: each filter integrates to ~ 2 / (right
    // - left) so all filters carry equal energy across their span.
    const float norm = 2.0f / (right - left);
    for (int k = 0; k < n_freqs; ++k) {
      filt[static_cast<std::size_t>(m) * n_freqs + k] *= norm;
    }
  }
  return filt;
}

std::vector<float>
build_hann_window_(int n_fft)
{
  std::vector<float> w(n_fft);
  for (int i = 0; i < n_fft; ++i) {
    w[i] = 0.5f - 0.5f * std::cos(2.0f
                                  * static_cast<float>(std::numbers::pi)
                                  * static_cast<float>(i)
                                  / static_cast<float>(n_fft));
  }
  return w;
}

// Batched real-to-complex FFT of a [n_frames, n_fft] f32 matrix into
// the power spectrum [n_frames, n_fft/2 + 1] using pocketfft.
//
// Pocketfft's r2c only writes the non-redundant half-spectrum, so the
// result tensor has the same number of complex samples as a numpy
// rfft. We square-magnitude it inline to produce the power spectrum.
void
batched_rfft_power_(const float* frames,   // [n_frames, n_fft] f32
                    std::size_t  n_frames,
                    int          n_fft,
                    float*       out)      // [n_frames, n_freqs] f32
{
  const std::size_t n_freqs = static_cast<std::size_t>(n_fft / 2 + 1);
  std::vector<std::complex<float>> spec(n_frames * n_freqs);
  pocketfft::shape_t  shape_in{n_frames, static_cast<std::size_t>(n_fft)};
  pocketfft::stride_t stride_in{
      static_cast<ptrdiff_t>(n_fft * sizeof(float)),
      static_cast<ptrdiff_t>(sizeof(float))};
  pocketfft::stride_t stride_out{
      static_cast<ptrdiff_t>(n_freqs * sizeof(std::complex<float>)),
      static_cast<ptrdiff_t>(sizeof(std::complex<float>))};
  pocketfft::r2c<float>(shape_in, stride_in, stride_out,
                        /*axis=*/1, /*forward=*/true,
                        frames, spec.data(),
                        /*fct=*/1.0f);
  // |c|^2.
  for (std::size_t i = 0; i < spec.size(); ++i) {
    const float re = spec[i].real();
    const float im = spec[i].imag();
    out[i] = re * re + im * im;
  }
}

}  // namespace

// ---- WhisperFeatureExtractor ----------------------------------------

WhisperFeatureExtractor::WhisperFeatureExtractor(const Params& p)
  : _p(p)
  , _window(build_hann_window_(p.n_fft))
  , _mel_filters(build_mel_filters_(p.n_fft, p.n_mel_bins,
                                    p.sample_rate, p.f_min, p.f_max))
{}

std::size_t
WhisperFeatureExtractor::n_frames_for(std::size_t n_samples)
    const noexcept
{
  std::size_t eff = (_p.n_samples_target > 0)
      ? static_cast<std::size_t>(_p.n_samples_target)
      : n_samples;
  if (eff < static_cast<std::size_t>(_p.n_fft)) {
    return 0;
  }
  // librosa STFT with center=True pads n_fft/2 samples on each side
  // BEFORE framing; the number of frames is then
  // 1 + n_samples / hop. For Qwen3-ASR (and WhisperFeatureExtractor)
  // this gives exactly n_samples_target / hop frames at hop=160,
  // n_samples=480000 -> 3000 frames.
  return eff / static_cast<std::size_t>(_p.hop_length);
}

std::size_t
WhisperFeatureExtractor::extract(const float*        pcm,
                                 std::size_t         n_samples,
                                 std::vector<float>* out) const
{
  if (!out) {
    return 0;
  }
  out->clear();
  if (!pcm || n_samples == 0) {
    return 0;
  }

  const int n_fft   = _p.n_fft;
  const int hop     = _p.hop_length;
  const int n_mels  = _p.n_mel_bins;
  const int n_freqs = n_fft / 2 + 1;
  const int half_w  = n_fft / 2;

  // 1. Pad / truncate input to n_samples_target when requested.
  std::vector<float> samples;
  std::size_t        n_in;
  if (_p.n_samples_target > 0) {
    n_in = static_cast<std::size_t>(_p.n_samples_target);
    samples.assign(n_in, 0.0f);
    std::size_t copy_n = std::min(n_in, n_samples);
    std::copy_n(pcm, copy_n, samples.begin());
  } else {
    samples.assign(pcm, pcm + n_samples);
    n_in = n_samples;
  }

  if (n_in < static_cast<std::size_t>(n_fft)) {
    return 0;
  }

  // 2. Reflection-pad both sides by n_fft/2 so frame-centers align on
  // a 0-indexed hop grid (librosa STFT with center=True).
  std::vector<float> padded(n_in + 2 * half_w);
  for (int i = 0; i < half_w; ++i) {
    padded[i] = samples[half_w - i];                              // reflect left
    padded[padded.size() - 1 - i] = samples[n_in - 2 - i];        // reflect right
  }
  std::copy(samples.begin(), samples.end(), padded.begin() + half_w);

  // 3. Frame + window. Build a [n_frames, n_fft] matrix of windowed
  // frames, then do ONE batched real FFT -> [n_frames, n_freqs]
  // power spectrum. Pocketfft's r2c handles the axis=1 sweep with
  // SIMD-friendly inner loops; this replaces what used to be an
  // O(n_frames * n_fft^2) per-frame DFT (216 ms warm on a 6 s clip)
  // with an O(n_frames * n_fft * log(n_fft)) batched call (~5-10 ms).
  //
  // The Whisper feature extractor uses center=True padding (which we
  // performed above) then drops the LAST frame so that a 30 s 16 kHz
  // input lands on exactly 3000 frames at hop=160 -- the value the
  // model's positional embeddings size for. The math:
  //   total_frames_centered = 1 + n_samples / hop = 3001 at 30 s
  //   reported            = total - 1            = 3000
  const std::size_t total_centered =
      (padded.size() - n_fft) / hop + 1;
  const std::size_t n_frames =
      (total_centered > 0) ? (total_centered - 1) : 0;
  if (n_frames == 0) {
    return 0;
  }
  std::vector<float> windowed(n_frames * n_fft);
  for (std::size_t t = 0; t < n_frames; ++t) {
    const float* base = padded.data() + t * hop;
    float*       dst  = windowed.data() + t * n_fft;
    for (int i = 0; i < n_fft; ++i) {
      dst[i] = base[i] * _window[i];
    }
  }
  std::vector<float> power(n_frames * n_freqs);
  batched_rfft_power_(windowed.data(), n_frames, n_fft, power.data());

  // 4. Mel projection: [n_frames, n_freqs] @ [n_freqs, n_mels]^T
  //    = [n_frames, n_mels], then transpose -> [n_mels, n_frames].
  // We materialise mel-major directly to avoid the explicit
  // transpose; cache pressure is fine at these sizes.
  out->assign(static_cast<std::size_t>(n_mels) * n_frames, 0.0f);
  for (std::size_t t = 0; t < n_frames; ++t) {
    const float* p = power.data() + t * n_freqs;
    for (int m = 0; m < n_mels; ++m) {
      const float* f = _mel_filters.data()
          + static_cast<std::size_t>(m) * n_freqs;
      float s = 0.0f;
      for (int k = 0; k < n_freqs; ++k) {
        s += p[k] * f[k];
      }
      (*out)[static_cast<std::size_t>(m) * n_frames + t] = s;
    }
  }

  // 5. log10 + clip + normalise (Whisper recipe).
  float max_lm = -std::numeric_limits<float>::infinity();
  for (float& v : *out) {
    float clipped = std::max(v, 1e-10f);
    float lm = std::log10(clipped);
    v = lm;
    if (lm > max_lm) { max_lm = lm; }
  }
  const float floor_v = max_lm - 8.0f;
  for (float& v : *out) {
    v = std::max(v, floor_v);
    v = (v + 4.0f) * 0.25f;
  }
  return n_frames;
}

}  // namespace vpipe::genai
