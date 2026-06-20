#ifndef VPIPE_GENERATIVE_MODELS_GEMMA4_AUDIO_FEATURE_EXTRACTOR_H
#define VPIPE_GENERATIVE_MODELS_GEMMA4_AUDIO_FEATURE_EXTRACTOR_H

// Gemma-4 (USM) audio feature extractor: raw 16 kHz mono PCM -> log-mel
// spectrogram [n_frames, n_mel] (frame-major), MLX-free. Mirrors
// mlx_vlm/models/gemma4/audio_feature_extractor.py
// (Gemma4AudioFeatureExtractor): semicausal left-pad of frame_length/2
// zeros, unfold(size=frame_length+1, step=hop) then drop the last sample
// (preemphasis is 0), periodic Hann over frame_length, rfft zero-padded
// to fft_length, MAGNITUDE (not power), HTK mel filterbank (no norm),
// log(mel + mel_floor). No per-bin normalisation. Distinct from
// WhisperFeatureExtractor (Slaney mel, power, reflect-pad, mel-major).

#include <cstddef>
#include <vector>

namespace vpipe::genai {

class Gemma4AudioFeatureExtractor {
public:
  struct Params {
    int   sample_rate  = 16000;
    int   frame_length = 320;     // 20 ms window
    int   hop_length   = 160;     // 10 ms hop
    int   fft_length   = 512;     // 2^ceil(log2(frame_length))
    int   n_mel        = 128;
    float min_freq     = 0.0f;
    float max_freq     = 8000.0f;
    float mel_floor    = 1.0e-3f;
  };

  Gemma4AudioFeatureExtractor();   // default Params
  explicit Gemma4AudioFeatureExtractor(const Params& p);

  // Extract a [n_frames * n_mel] row-major (frame-major) log-mel buffer.
  // Returns n_frames (0 on failure / too-short input).
  std::size_t extract(const float* pcm, std::size_t n_samples,
                      std::vector<float>* out) const;

  std::size_t n_frames_for(std::size_t n_samples) const noexcept;

  const Params& params() const noexcept { return _p; }

private:
  Params             _p;
  std::vector<float> _window;       // [frame_length] periodic Hann
  std::vector<float> _mel_filters;  // [n_mel, fft_length/2 + 1]
};

}  // namespace vpipe::genai

#endif
