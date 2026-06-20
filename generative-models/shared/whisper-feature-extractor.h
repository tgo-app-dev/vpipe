#ifndef VPIPE_GENERATIVE_MODELS_WHISPER_FEATURE_EXTRACTOR_H
#define VPIPE_GENERATIVE_MODELS_WHISPER_FEATURE_EXTRACTOR_H

#include <cstddef>
#include <vector>

namespace vpipe::genai {

// ---- Whisper-style log-mel feature extractor ------------------------
//
// 16 kHz mono PCM -> [n_mel_bins, n_frames] log-mel spectrogram. Pure
// CPU (pocketfft + std only) -- no MLX -- so both the MLX audio encoder
// and the metal-compute audio encoder reuse it. Parameters match the HF
// WhisperFeatureExtractor defaults Qwen3-ASR ships:
//   n_fft        = 400   (25 ms window at 16 kHz)
//   hop_length   = 160   (10 ms hop)
//   n_mel_bins   = 128
//   sample_rate  = 16000
//
// The result is normalised the same way as Whisper / Qwen3-ASR:
//   log_mel = max(log10(max(mel_power, 1e-10)), max(...) - 8.0)
//   log_mel = (log_mel + 4.0) / 4.0
// Output values are roughly in [-1, 1].
//
// `pcm` is left-padded / right-truncated to `n_samples_target` samples
// before STFT when the input length differs (Whisper's 30 s window =
// 480000 samples at 16 kHz). Pass `n_samples_target = 0` to skip the
// pad/truncate step (the caller has already arranged the right length).
class WhisperFeatureExtractor {
public:
  struct Params {
    int n_fft         = 400;
    int hop_length    = 160;
    int n_mel_bins    = 128;
    int sample_rate   = 16000;
    int n_samples_target = 0;  // 0 = no pad/truncate
    // Frequency range for the mel filterbank.
    float f_min       = 0.0f;
    float f_max       = 8000.0f;
  };

  explicit WhisperFeatureExtractor(const Params& p);

  // Compute the log-mel spectrogram. `out` is filled with
  // [n_mel_bins * n_frames] f32 values in row-major order
  // (mel-major, frame-minor: out[m * n_frames + t]).
  // Returns the number of frames produced (== ceil((n_samples_eff -
  // n_fft) / hop_length) + 1 with reflection padding semantics, or
  // 0 on empty input).
  std::size_t extract(const float*        pcm,
                      std::size_t         n_samples,
                      std::vector<float>* out) const;

  const Params& params() const noexcept { return _p; }

  // Number of frames the extractor would produce for a given input
  // length (after the target-length pad/truncate). Useful for
  // pre-sizing buffers without running the extractor.
  std::size_t n_frames_for(std::size_t n_samples) const noexcept;

private:
  Params               _p;
  // Hann window of length n_fft, applied to each STFT frame.
  std::vector<float>   _window;
  // Mel filterbank, shape [n_mel_bins, n_fft/2 + 1] in row-major.
  std::vector<float>   _mel_filters;
};

}  // namespace vpipe::genai

#endif
