#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_AUDIO_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_AUDIO_H

// MAGI-2-preview's soundtrack, as the release's inference/pipeline/
// audio_decoder.py decodes it (read, not run): the DiT's 64-wide audio
// latents run at 25 a second, and Stable Audio Open's autoencoder (its
// stable-audio-open-1.0/, an Oobleck decoder: MetalOobleckDecoder reads it)
// decodes one to 2048 stereo samples -- 51,200 a second, which the release
// does not resample but TIME STRETCHES by 441/512 to 44.1 kHz with
// scipy.signal.resample (an FFT resample over the whole clip).

#include <vector>

namespace vpipe {
namespace genai {
namespace magi2 {

// Audio latents a second (the DiT's), and the waveform's rate after the
// stretch.
inline constexpr int kAudioLatentRate = 25;
inline constexpr int kAudioSampleRate = 44100;

// The samples n decoded samples stretch to: Python's int(n * (441 / 512)).
inline int
stretched_length(long n)
{
  return (int)((double)n * (441.0 / 512.0));
}

// scipy.signal.resample(x, num) of each channel of PLANAR [channels, n]
// f32, as scipy 1.12 computes it (the real FFT, the spectrum truncated or
// zero-padded, the Nyquist bin split or joined, the inverse scaled by
// num / n) -- in double. Returns planar [channels, num].
std::vector<float> fft_resample(const float* planar, int channels, long n,
                                long num);

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_AUDIO_H
