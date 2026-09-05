#include "generative-models/minimax-h3/vdn-delta-solve.h"

#include <cmath>
#include <vector>

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace vdn {

bool
cholesky_lower(const float* a, int d, float* l)
{
  if (a == nullptr || l == nullptr || d <= 0) { return false; }
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) { l[(std::size_t)i * d + j] = 0.0f; }
  }
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j <= i; ++j) {
      // M = I + A, lower triangle only.
      double sum = (double)a[(std::size_t)i * d + j] + (i == j ? 1.0 : 0.0);
      for (int k = 0; k < j; ++k) {
        sum -= (double)l[(std::size_t)i * d + k]
               * (double)l[(std::size_t)j * d + k];
      }
      if (i == j) {
        if (!(sum > 0.0)) { return false; }
        l[(std::size_t)i * d + j] = (float)std::sqrt(sum);
      } else {
        const double diag = (double)l[(std::size_t)j * d + j];
        if (diag == 0.0) { return false; }
        l[(std::size_t)i * d + j] = (float)(sum / diag);
      }
    }
  }
  return true;
}

void
cholesky_solve(const float* l, int d, const float* c, int cols, float* x)
{
  std::vector<double> y((std::size_t)d);
  for (int col = 0; col < cols; ++col) {
    for (int i = 0; i < d; ++i) {          // L y = c
      double sum = (double)c[(std::size_t)i * cols + col];
      for (int k = 0; k < i; ++k) {
        sum -= (double)l[(std::size_t)i * d + k] * y[(std::size_t)k];
      }
      y[(std::size_t)i] = sum / (double)l[(std::size_t)i * d + i];
    }
    for (int i = d - 1; i >= 0; --i) {     // L^T x = y
      double sum = y[(std::size_t)i];
      for (int k = i + 1; k < d; ++k) {
        sum -= (double)l[(std::size_t)k * d + i]
               * (double)x[(std::size_t)k * cols + col];
      }
      x[(std::size_t)i * cols + col] =
          (float)(sum / (double)l[(std::size_t)i * d + i]);
    }
  }
}

void
cholesky_inverse(const float* l, int d, float* inv)
{
  // L^-1 by forward substitution against the identity, then L^-T L^-1.
  // The reference takes the same two steps (one triangular solve and a
  // product rather than two solves), so this is the same association.
  std::vector<double> linv((std::size_t)d * d, 0.0);
  for (int col = 0; col < d; ++col) {
    for (int i = col; i < d; ++i) {
      double sum = (i == col) ? 1.0 : 0.0;
      for (int k = col; k < i; ++k) {
        sum -= (double)l[(std::size_t)i * d + k]
               * linv[(std::size_t)k * d + col];
      }
      linv[(std::size_t)i * d + col] =
          sum / (double)l[(std::size_t)i * d + i];
    }
  }
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) {
      double sum = 0.0;
      // (L^-T L^-1)[i,j] = sum_k L^-1[k,i] L^-1[k,j], k >= max(i,j)
      const int k0 = i > j ? i : j;
      for (int k = k0; k < d; ++k) {
        sum += linv[(std::size_t)k * d + i] * linv[(std::size_t)k * d + j];
      }
      inv[(std::size_t)i * d + j] = (float)sum;
    }
  }
}

bool
factor_apply(const float* a, const float* b, const float* alpha,
             int frames, int heads, int d_k, int d_v, float* transition,
             float* injection)
{
  if (frames <= 0 || heads <= 0 || d_k <= 0 || d_v <= 0) { return false; }
  const std::size_t kk = (std::size_t)d_k * d_k;
  const std::size_t vk = (std::size_t)d_v * d_k;
  std::vector<float> l(kk), inv(kk);
  for (int f = 0; f < frames; ++f) {
    for (int h = 0; h < heads; ++h) {
      const std::size_t idx = (std::size_t)f * heads + h;
      const float* af = a + idx * kk;
      if (!cholesky_lower(af, d_k, l.data())) { return false; }
      cholesky_inverse(l.data(), d_k, inv.data());

      const float* al = alpha + idx * (std::size_t)d_k;
      float* tr = transition + idx * kk;
      for (int i = 0; i < d_k; ++i) {
        const float s = al[i];
        for (int j = 0; j < d_k; ++j) {
          tr[(std::size_t)i * d_k + j] = s * inv[(std::size_t)i * d_k + j];
        }
      }
      const float* bf = b + idx * vk;
      float* inj = injection + idx * vk;
      for (int i = 0; i < d_v; ++i) {
        for (int j = 0; j < d_k; ++j) {
          double sum = 0.0;
          for (int k = 0; k < d_k; ++k) {
            sum += (double)bf[(std::size_t)i * d_k + k]
                   * (double)inv[(std::size_t)k * d_k + j];
          }
          inj[(std::size_t)i * d_k + j] = (float)sum;
        }
      }
    }
  }
  return true;
}

bool
solve_state(const float* a, const float* alpha, const float* state_in,
            const float* b, int d_k, int d_v, float* state_out)
{
  if (d_k <= 0 || d_v <= 0) { return false; }
  std::vector<float> l((std::size_t)d_k * d_k);
  if (!cholesky_lower(a, d_k, l.data())) { return false; }
  // C = S_in Diag(alpha) + B, then solve X (I+A) = C as M X^T = C^T.
  std::vector<float> ct((std::size_t)d_k * d_v);
  for (int i = 0; i < d_v; ++i) {
    for (int j = 0; j < d_k; ++j) {
      const float s = state_in == nullptr
                          ? 0.0f
                          : state_in[(std::size_t)i * d_k + j] * alpha[j];
      ct[(std::size_t)j * d_v + i] = s + b[(std::size_t)i * d_k + j];
    }
  }
  std::vector<float> xt((std::size_t)d_k * d_v);
  cholesky_solve(l.data(), d_k, ct.data(), d_v, xt.data());
  for (int i = 0; i < d_v; ++i) {
    for (int j = 0; j < d_k; ++j) {
      state_out[(std::size_t)i * d_k + j] = xt[(std::size_t)j * d_v + i];
    }
  }
  return true;
}

}  // namespace vdn
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
