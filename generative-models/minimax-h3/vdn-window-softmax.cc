#include "generative-models/minimax-h3/vdn-window-softmax.h"

#include <cmath>
#include <vector>

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace vdn {

void
window_softmax(const float* q, const float* k, const float* v,
               const WindowMask& mask, int heads, int head_dim, float scale,
               float* out)
{
  const int seq = mask.seq_len;
  if (seq <= 0 || heads <= 0 || head_dim <= 0) { return; }
  const std::size_t row_stride = (std::size_t)heads * head_dim;
  std::vector<int> allowed;
  std::vector<double> logit;
  allowed.reserve((std::size_t)seq);
  logit.reserve((std::size_t)seq);

  for (int qi = 0; qi < seq; ++qi) {
    // The allowed set is a property of the ROW, not of the head, so it
    // is built once and reused across all of them.
    allowed.clear();
    for (int ki = 0; ki < seq; ++ki) {
      if (mask.allows(qi, ki)) { allowed.push_back(ki); }
    }
    for (int h = 0; h < heads; ++h) {
      const float* qp = q + (std::size_t)qi * row_stride
                        + (std::size_t)h * head_dim;
      float* op = out + (std::size_t)qi * row_stride
                  + (std::size_t)h * head_dim;
      for (int i = 0; i < head_dim; ++i) { op[i] = 0.0f; }
      if (allowed.empty()) { continue; }   // zero, never NaN

      logit.clear();
      double maxv = -1e30;
      for (int ki : allowed) {
        const float* kp = k + (std::size_t)ki * row_stride
                          + (std::size_t)h * head_dim;
        double dot = 0.0;
        for (int i = 0; i < head_dim; ++i) { dot += (double)qp[i] * kp[i]; }
        dot *= (double)scale;
        logit.push_back(dot);
        if (dot > maxv) { maxv = dot; }
      }
      double denom = 0.0;
      for (std::size_t a = 0; a < logit.size(); ++a) {
        logit[a] = std::exp(logit[a] - maxv);
        denom += logit[a];
      }
      const double inv = denom > 0.0 ? 1.0 / denom : 0.0;
      std::vector<double> acc((std::size_t)head_dim, 0.0);
      for (std::size_t a = 0; a < allowed.size(); ++a) {
        const float* vp = v + (std::size_t)allowed[a] * row_stride
                          + (std::size_t)h * head_dim;
        const double w = logit[a] * inv;
        for (int i = 0; i < head_dim; ++i) { acc[(std::size_t)i] += w * vp[i]; }
      }
      for (int i = 0; i < head_dim; ++i) { op[i] = (float)acc[(std::size_t)i]; }
    }
  }
}

void
apply_softmax_gate(const float* attn, const float* x, int seq_len, int heads,
                   int head_dim, int hidden, const float* gate_w,
                   const float* gate_b, float* out)
{
  const std::size_t row_stride = (std::size_t)heads * head_dim;
  for (int t = 0; t < seq_len; ++t) {
    const float* xp = x + (std::size_t)t * hidden;
    for (int h = 0; h < heads; ++h) {
      double logit = gate_b != nullptr ? (double)gate_b[h] : 0.0;
      const float* w = gate_w + (std::size_t)h * hidden;
      for (int i = 0; i < hidden; ++i) { logit += (double)w[i] * xp[i]; }
      const float g = 1.0f / (1.0f + std::exp(-(float)logit));
      const float* ap = attn + (std::size_t)t * row_stride
                        + (std::size_t)h * head_dim;
      float* op = out + (std::size_t)t * row_stride
                  + (std::size_t)h * head_dim;
      for (int i = 0; i < head_dim; ++i) { op[i] = ap[i] * g; }
    }
  }
}

}  // namespace vdn
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
