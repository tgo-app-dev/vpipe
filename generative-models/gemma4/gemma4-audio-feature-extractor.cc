#include "generative-models/gemma4/gemma4-audio-feature-extractor.h"

#include "3rd-party/pocketfft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <vector>

namespace vpipe::genai {

namespace {

// HTK mel scale (matches the reference _mel_filter_bank / HF htk).
float hz_to_mel_htk_(float hz)
{
  return 2595.0f * std::log10(1.0f + hz / 700.0f);
}
float mel_to_hz_htk_(float mel)
{
  return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// HTK triangular mel filterbank [n_mel, n_freqs], norm=None. Bin
// frequencies are k * sr/(2*(n_freqs-1)) (== k*sr/fft_length), which is
// the rfft bin grid; centers are mel-linspaced between min/max.
std::vector<float>
build_mel_filters_(int fft_length, int n_mel, int sample_rate,
                   float f_min, float f_max)
{
  const int n_freqs = fft_length / 2 + 1;
  std::vector<float> filt(static_cast<std::size_t>(n_mel) * n_freqs, 0.0f);

  std::vector<float> all_freqs(n_freqs);
  const float df = static_cast<float>(sample_rate)
      / static_cast<float>(2 * (n_freqs - 1));
  for (int k = 0; k < n_freqs; ++k) {
    all_freqs[k] = static_cast<float>(k) * df;
  }

  const float mel_min = hz_to_mel_htk_(f_min);
  const float mel_max = hz_to_mel_htk_(f_max);
  std::vector<float> freq_pts(n_mel + 2);
  for (int i = 0; i < n_mel + 2; ++i) {
    const float mel = mel_min + (mel_max - mel_min)
        * static_cast<float>(i) / static_cast<float>(n_mel + 1);
    freq_pts[i] = mel_to_hz_htk_(mel);
  }

  for (int m = 0; m < n_mel; ++m) {
    const float lower  = freq_pts[m];
    const float center = freq_pts[m + 1];
    const float upper  = freq_pts[m + 2];
    const float dl = std::max(center - lower, 1e-10f);
    const float du = std::max(upper - center, 1e-10f);
    for (int k = 0; k < n_freqs; ++k) {
      const float rising  = (all_freqs[k] - lower) / dl;
      const float falling = (upper - all_freqs[k]) / du;
      filt[static_cast<std::size_t>(m) * n_freqs + k] =
          std::max(0.0f, std::min(rising, falling));
    }
  }
  return filt;
}

// Periodic Hann over `frame_length`: 0.5 - 0.5*cos(2*pi*n/frame_length).
std::vector<float>
build_periodic_hann_(int frame_length)
{
  std::vector<float> w(frame_length);
  for (int i = 0; i < frame_length; ++i) {
    w[i] = 0.5f - 0.5f * std::cos(2.0f
        * static_cast<float>(std::numbers::pi)
        * static_cast<float>(i) / static_cast<float>(frame_length));
  }
  return w;
}

// Batched rfft MAGNITUDE of a [n_frames, fft_length] f32 matrix ->
// [n_frames, n_freqs] f32 (|c|, not |c|^2).
void
batched_rfft_mag_(const float* frames, std::size_t n_frames, int fft_length,
                  float* out)
{
  const std::size_t n_freqs =
      static_cast<std::size_t>(fft_length / 2 + 1);
  std::vector<std::complex<float>> spec(n_frames * n_freqs);
  pocketfft::shape_t shape_in{n_frames,
                              static_cast<std::size_t>(fft_length)};
  pocketfft::stride_t stride_in{
      static_cast<ptrdiff_t>(fft_length * sizeof(float)),
      static_cast<ptrdiff_t>(sizeof(float))};
  pocketfft::stride_t stride_out{
      static_cast<ptrdiff_t>(n_freqs * sizeof(std::complex<float>)),
      static_cast<ptrdiff_t>(sizeof(std::complex<float>))};
  pocketfft::r2c<float>(shape_in, stride_in, stride_out, /*axis=*/1,
                        /*forward=*/true, frames, spec.data(), /*fct=*/1.0f);
  for (std::size_t i = 0; i < spec.size(); ++i) {
    out[i] = std::hypot(spec[i].real(), spec[i].imag());
  }
}

}  // namespace

Gemma4AudioFeatureExtractor::Gemma4AudioFeatureExtractor()
  : Gemma4AudioFeatureExtractor(Params{})
{}

Gemma4AudioFeatureExtractor::Gemma4AudioFeatureExtractor(const Params& p)
  : _p(p)
  , _window(build_periodic_hann_(p.frame_length))
  , _mel_filters(build_mel_filters_(p.fft_length, p.n_mel, p.sample_rate,
                                    p.min_freq, p.max_freq))
{}

std::size_t
Gemma4AudioFeatureExtractor::n_frames_for(std::size_t n_samples)
    const noexcept
{
  const std::size_t padded = n_samples + (std::size_t)(_p.frame_length / 2);
  const std::size_t unfold = (std::size_t)_p.frame_length + 1;
  if (padded < unfold) { return 0; }
  return (padded - unfold) / (std::size_t)_p.hop_length + 1;
}

std::size_t
Gemma4AudioFeatureExtractor::extract(const float*        pcm,
                                     std::size_t         n_samples,
                                     std::vector<float>* out) const
{
  if (!out) { return 0; }
  out->clear();
  if (!pcm || n_samples == 0) { return 0; }

  const int F = _p.frame_length;
  const int hop = _p.hop_length;
  const int nfft = _p.fft_length;
  const int n_mel = _p.n_mel;
  const int n_freqs = nfft / 2 + 1;
  const int pad_left = F / 2;

  // Semicausal left-pad of frame_length/2 zeros (centers the first frame
  // at t=0), then unfold(size=F+1, step=hop) and use the first F samples
  // of each window (preemphasis is 0 -> the +1 sample is dropped).
  std::vector<float> padded((std::size_t)pad_left + n_samples, 0.0f);
  std::copy_n(pcm, n_samples, padded.begin() + pad_left);
  const std::size_t unfold = (std::size_t)F + 1;
  if (padded.size() < unfold) { return 0; }
  const std::size_t n_frames =
      (padded.size() - unfold) / (std::size_t)hop + 1;
  if (n_frames == 0) { return 0; }

  // Windowed + zero-padded frames [n_frames, fft_length].
  std::vector<float> windowed((std::size_t)n_frames * nfft, 0.0f);
  for (std::size_t t = 0; t < n_frames; ++t) {
    const float* base = padded.data() + t * hop;
    float* dst = windowed.data() + t * nfft;
    for (int i = 0; i < F; ++i) { dst[i] = base[i] * _window[i]; }
  }
  std::vector<float> mag((std::size_t)n_frames * n_freqs);
  batched_rfft_mag_(windowed.data(), n_frames, nfft, mag.data());

  // Mel projection [n_frames, n_freqs] @ [n_freqs, n_mel]^T + log floor.
  // Frame-major output: out[t*n_mel + m].
  out->assign((std::size_t)n_frames * n_mel, 0.0f);
  for (std::size_t t = 0; t < n_frames; ++t) {
    const float* mg = mag.data() + t * n_freqs;
    float* dst = out->data() + t * n_mel;
    for (int m = 0; m < n_mel; ++m) {
      const float* fb = _mel_filters.data()
          + (std::size_t)m * n_freqs;
      float s = 0.0f;
      for (int k = 0; k < n_freqs; ++k) { s += mg[k] * fb[k]; }
      dst[m] = std::log(s + _p.mel_floor);
    }
  }
  return n_frames;
}

}  // namespace vpipe::genai
