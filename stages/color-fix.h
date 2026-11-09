#ifndef VPIPE_STAGES_COLOR_FIX_H
#define VPIPE_STAGES_COLOR_FIX_H

#include <cstddef>
#include <vector>

namespace vpipe::color_fix {

// THE COLOUR FIXES A RESTORER'S RELEASE APPLIES TO ITS OUTPUT against the
// picture it restored, on the host, one plane at a time. Samples are in
// 0..1 and the result is clamped to 0..1, as the releases' are.
//
// These are VOSR's (`inference_vosr_onestep.py`, `--align_method`):
//
//   gaussian  `wavelet_color_fix`: x = clamp(low(ref) + x - low(x)), with
//             `low` OpenCV's GaussianBlur at `sigma` (5 there), ksize 0 --
//             so cvRound(8 sigma + 1) | 1 taps and reflect-101 edges. It
//             is computed as x + low(ref - x), which is the same sum by
//             linearity and needs one blur and one scratch plane.
//   adain     `adain_color_fix`: each channel's mean and spread moved to
//             the reference's, (x - m_x) / (s_x + 1e-5) * (s_r + 1e-5) +
//             m_r, the spreads torch.std's (unbiased).
//
// FlashVSR+'s wavelet fix (a stack of dilated 3x3 passes) is a different
// low-pass and lives in wavelet-color-fix.h.

// OpenCV's taps for a FLOAT image at `sigma` with ksize 0 (the kernel
// getGaussianKernel builds for CV_32F: exp(-d^2 / (2 sigma^2)) over
// d = -r..r, normalised in double, stored as float).
std::vector<float> gaussian_taps(double sigma);

// `dst` = `src` blurred by `taps` (odd count), rows then columns, edges
// reflect-101 as OpenCV's default border. `tmp` is a scratch plane of
// H * W. `dst` may not alias `src` or `tmp`.
void gaussian_blur(const float* src, float* dst, float* tmp, int H, int W,
                   const std::vector<float>& taps);

// VOSR's `wavelet_color_fix` on one plane: x <- clamp(x + low(ref - x)).
// `tmp` is a scratch plane of H * W.
void gaussian(float* x, const float* ref, float* tmp, int H, int W,
              const std::vector<float>& taps);

// VOSR's `adain_color_fix` on one plane of `n` samples.
void adain(float* x, const float* ref, std::size_t n);

}  // namespace vpipe::color_fix

#endif
