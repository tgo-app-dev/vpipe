#include "generative-models/minimax-h3/vdn-linear-branch.h"

#include "generative-models/minimax-h3/vdn-delta-solve.h"

#include <cmath>
#include <cstring>

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace vdn {

namespace {

float
sigmoid_(float v)
{
  return 1.0f / (1.0f + std::exp(-v));
}

// log(1 + exp(v)), the numerically stable branch torch takes.
double
softplus_(double v)
{
  return v > 20.0 ? v : std::log1p(std::exp(v));
}

// y = W x, W [out, in] row-major, no bias.
void
linear_(const float* x, const float* w, int out_dim, int in_dim, double* y)
{
  for (int o = 0; o < out_dim; ++o) {
    double sum = 0.0;
    const float* row = w + (std::size_t)o * in_dim;
    for (int i = 0; i < in_dim; ++i) { sum += (double)row[i] * x[i]; }
    y[o] = sum;
  }
}

}  // namespace

void
branch_beta(const float* x, int rows, const LinearBranchConfig& cfg,
            const float* beta_proj, float* beta)
{
  std::vector<double> y((std::size_t)cfg.heads);
  for (int r = 0; r < rows; ++r) {
    linear_(x + (std::size_t)r * cfg.hidden, beta_proj, cfg.heads, cfg.hidden,
            y.data());
    for (int h = 0; h < cfg.heads; ++h) {
      beta[(std::size_t)r * cfg.heads + h] = sigmoid_((float)y[h]);
    }
  }
}

void
frame_statistics(const float* key, const float* value, const float* beta,
                 int frames, int tokens_per_frame, int heads, int head_dim,
                 float* a, float* b)
{
  const std::size_t mat = (std::size_t)head_dim * head_dim;
  const int d = head_dim;
  for (int f = 0; f < frames; ++f) {
    for (int h = 0; h < heads; ++h) {
      float* af = a + ((std::size_t)f * heads + h) * mat;
      float* bf = b + ((std::size_t)f * heads + h) * mat;
      std::memset(af, 0, mat * sizeof(float));
      std::memset(bf, 0, mat * sizeof(float));
      for (int s = 0; s < tokens_per_frame; ++s) {
        const std::size_t row =
            ((std::size_t)f * tokens_per_frame + s) * heads + h;
        const float* k = key + row * d;
        const float* v = value + row * d;
        const float bt = beta[((std::size_t)f * tokens_per_frame + s) * heads
                              + h];
        for (int i = 0; i < d; ++i) {
          const float kb = k[i] * bt;
          const float vb = v[i] * bt;
          for (int j = 0; j < d; ++j) {
            af[(std::size_t)i * d + j] += kb * k[j];
            bf[(std::size_t)i * d + j] += vb * k[j];
          }
        }
      }
      // SYMMETRISE. A is (k*beta)^T k, so (i,j) and (j,i) multiply
      // differently-rounded operands; the Cholesky reads one triangle.
      for (int i = 0; i < d; ++i) {
        for (int j = i + 1; j < d; ++j) {
          const float m = 0.5f * (af[(std::size_t)i * d + j]
                                  + af[(std::size_t)j * d + i]);
          af[(std::size_t)i * d + j] = m;
          af[(std::size_t)j * d + i] = m;
        }
      }
    }
  }
}

void
branch_alpha(const float* x, int frames, int tokens_per_frame,
             const LinearBranchConfig& cfg, const LinearBranchWeights& w,
             float* frame_mean, float* alpha)
{
  const int H = cfg.heads, d = cfg.head_dim, hid = cfg.hidden;
  std::vector<float> mean_local;
  float* mean = frame_mean;
  if (mean == nullptr) {
    mean_local.resize((std::size_t)frames * hid);
    mean = mean_local.data();
  }
  // fp32 BEFORE any downcast: rounding the mean first cannot be undone
  // by promoting afterwards.
  for (int f = 0; f < frames; ++f) {
    float* m = mean + (std::size_t)f * hid;
    for (int i = 0; i < hid; ++i) { m[i] = 0.0f; }
    for (int s = 0; s < tokens_per_frame; ++s) {
      const float* row =
          x + ((std::size_t)f * tokens_per_frame + s) * hid;
      for (int i = 0; i < hid; ++i) { m[i] += row[i]; }
    }
    for (int i = 0; i < hid; ++i) { m[i] /= (float)tokens_per_frame; }
  }

  std::vector<double> lo((std::size_t)d), hi((std::size_t)H * d);
  for (int f = 0; f < frames; ++f) {
    linear_(mean + (std::size_t)f * hid, w.alpha_down, d, hid, lo.data());
    std::vector<float> lof(lo.begin(), lo.end());
    linear_(lof.data(), w.alpha_up, H * d, d, hi.data());
    for (int h = 0; h < H; ++h) {
      // A_log is per HEAD; dt_bias is per CHANNEL. Both, and the two
      // nested exponentials, in fp32.
      const double scale = std::exp((double)w.alpha_a_log[h]);
      for (int i = 0; i < d; ++i) {
        const std::size_t c = (std::size_t)h * d + i;
        const double delta = hi[c] + (double)w.alpha_dt_bias[c];
        alpha[((std::size_t)f * H + h) * d + i] =
            (float)std::exp(-scale * softplus_(delta));
      }
    }
  }
}

bool
branch_text_state(const float* text_x, const float* text_k_raw,
                  const float* text_v_raw, int text_len,
                  const LinearBranchConfig& cfg, const LinearBranchWeights& w,
                  float* text_state)
{
  const int H = cfg.heads, d = cfg.head_dim;
  const std::size_t rows = (std::size_t)text_len * H;
  std::vector<float> key(rows * d), value(rows * d);
  // No conv on text: the short conv is a (t, h, w) stencil and the
  // prompt has no such grid. Everything else is the video path's.
  linear_features(text_k_raw, 1, 1, text_len, H, d, nullptr, true,
                  key.data());
  linear_features(text_v_raw, 1, 1, text_len, H, d, nullptr, false,
                  value.data());
  std::vector<float> beta(rows);
  branch_beta(text_x, text_len, cfg, w.beta_proj, beta.data());

  const std::size_t mat = (std::size_t)d * d;
  std::vector<float> a((std::size_t)H * mat), b((std::size_t)H * mat);
  // ONE chunk over all L rows, exactly as a frame is one chunk over S.
  frame_statistics(key.data(), value.data(), beta.data(), 1, text_len, H, d,
                   a.data(), b.data());
  std::vector<float> ones((std::size_t)H * d, 1.0f);
  std::vector<float> transition((std::size_t)H * mat);
  if (!factor_apply(a.data(), b.data(), ones.data(), 1, H, d, d,
                    transition.data(), text_state)) {
    return false;
  }
  // Both scans start from HALF of it: each direction carries the whole
  // prompt and the gather adds them.
  for (std::size_t i = 0; i < (std::size_t)H * mat; ++i) {
    text_state[i] *= cfg.text_state_scale;
  }
  return true;
}

bool
branch_scans(const float* a, const float* b, const float* alpha, int frames,
             int heads, int head_dim, const float* text_state, float* prefix,
             float* suffix)
{
  const std::size_t mat = (std::size_t)head_dim * head_dim;
  const std::size_t per = (std::size_t)heads * mat;
  std::vector<float> transition((std::size_t)frames * per);
  std::vector<float> injection((std::size_t)frames * per);
  if (!factor_apply(a, b, alpha, frames, heads, head_dim, head_dim,
                    transition.data(), injection.data())) {
    return false;
  }
  std::vector<float> state(per);
  auto start = [&]() {
    if (text_state != nullptr) {
      std::memcpy(state.data(), text_state, per * sizeof(float));
    } else {
      std::memset(state.data(), 0, per * sizeof(float));
    }
  };
  auto step = [&](int f, float* dst) {
    for (int h = 0; h < heads; ++h) {
      const float* tr = transition.data() + ((std::size_t)f * heads + h) * mat;
      const float* in = injection.data() + ((std::size_t)f * heads + h) * mat;
      const float* st = state.data() + (std::size_t)h * mat;
      float* o = dst + (std::size_t)h * mat;
      for (int i = 0; i < head_dim; ++i) {
        for (int j = 0; j < head_dim; ++j) {
          double sum = in[(std::size_t)i * head_dim + j];
          for (int k = 0; k < head_dim; ++k) {
            sum += (double)st[(std::size_t)i * head_dim + k]
                   * (double)tr[(std::size_t)k * head_dim + j];
          }
          o[(std::size_t)i * head_dim + j] = (float)sum;
        }
      }
    }
    std::memcpy(state.data(), dst, per * sizeof(float));
  };
  start();
  for (int f = 0; f < frames; ++f) { step(f, prefix + (std::size_t)f * per); }
  start();
  for (int f = frames - 1; f >= 0; --f) {
    step(f, suffix + (std::size_t)f * per);
  }
  return true;
}

void
branch_readout(const float* query, const float* state, const float* x,
               int frames, int tokens_per_frame,
               const LinearBranchConfig& cfg, const LinearBranchWeights& w,
               float* out)
{
  const int H = cfg.heads, d = cfg.head_dim, hid = cfg.hidden;
  const std::size_t mat = (std::size_t)d * d;
  std::vector<double> lo((std::size_t)d), hi((std::size_t)H * d);
  std::vector<float> row((std::size_t)d);
  for (int f = 0; f < frames; ++f) {
    for (int s = 0; s < tokens_per_frame; ++s) {
      const std::size_t token = (std::size_t)f * tokens_per_frame + s;
      // The gate: sigmoid(up(down(x))), per (token, head, channel).
      linear_(x + token * hid, w.gate_down, d, hid, lo.data());
      std::vector<float> lof(lo.begin(), lo.end());
      linear_(lof.data(), w.gate_up_w, H * d, d, hi.data());
      for (int h = 0; h < H; ++h) {
        const float* q = query + (token * H + h) * d;
        const float* st = state + ((std::size_t)f * H + h) * mat;
        // readout[v] = sum_k state[v, k] q[k]
        double ms = 0.0;
        for (int v = 0; v < d; ++v) {
          double sum = 0.0;
          for (int k = 0; k < d; ++k) {
            sum += (double)st[(std::size_t)v * d + k] * (double)q[k];
          }
          row[v] = (float)sum;
          ms += sum * sum;
        }
        // RMSNorm over head_dim, second moment in fp32, eps inside.
        const double inv = 1.0 / std::sqrt(ms / (double)d + cfg.norm_eps);
        float* o = out + token * (std::size_t)H * d + (std::size_t)h * d;
        for (int v = 0; v < d; ++v) {
          const std::size_t c = (std::size_t)h * d + v;
          const float gate =
              sigmoid_((float)(hi[c] + (double)w.gate_up_b[c]));
          o[v] = (float)((double)row[v] * inv * (double)w.norm[v]) * gate;
        }
      }
    }
  }
}

bool
linear_branch_forward(const float* x, const float* q_raw, const float* k_raw,
                      const float* v_raw, int frames, int grid_h, int grid_w,
                      const std::vector<Bound>& bounds, const float* text_x,
                      const float* text_k_raw, const float* text_v_raw,
                      int text_len, const LinearBranchConfig& cfg,
                      const LinearBranchWeights& w, float* out)
{
  const int H = cfg.heads, d = cfg.head_dim;
  const int S = grid_h * grid_w;
  const std::size_t wide = (std::size_t)H * d;
  std::memset(out, 0, (std::size_t)frames * S * wide * sizeof(float));
  if (frames <= 0 || S <= 0) { return true; }

  // With anchor_frames "both" the two anchor frames leave the INPUT --
  // their rows stay exactly zero above, and the window rebases by one.
  int f0 = 0, fn = frames;
  std::vector<Bound> use = bounds;
  if (cfg.skip_ends) {
    if (frames <= 2) { return true; }     // the anchors ARE the clip
    f0 = 1;
    fn = frames - 2;
    use = rebase_for_anchor_skip(bounds, frames);
  }
  const std::size_t off = (std::size_t)f0 * S;
  const float* xin = x + off * cfg.hidden;
  const std::size_t qoff = off * wide;

  std::vector<float> q((std::size_t)fn * S * wide);
  std::vector<float> k(q.size()), v(q.size());
  linear_features(q_raw + qoff, fn, grid_h, grid_w, H, d, nullptr, true,
                  q.data());
  linear_features(k_raw + qoff, fn, grid_h, grid_w, H, d, &w.k_conv, true,
                  k.data());
  linear_features(v_raw + qoff, fn, grid_h, grid_w, H, d, &w.v_conv, false,
                  v.data());

  std::vector<float> beta((std::size_t)fn * S * H);
  branch_beta(xin, fn * S, cfg, w.beta_proj, beta.data());

  const std::size_t mat = (std::size_t)d * d;
  std::vector<float> a((std::size_t)fn * H * mat), b(a.size());
  frame_statistics(k.data(), v.data(), beta.data(), fn, S, H, d, a.data(),
                   b.data());

  std::vector<float> alpha((std::size_t)fn * H * d);
  branch_alpha(xin, fn, S, cfg, w, nullptr, alpha.data());

  std::vector<float> text_state((std::size_t)H * mat, 0.0f);
  bool have_text = false;
  if (cfg.enable_text_state && text_x != nullptr && text_len > 0) {
    if (!branch_text_state(text_x, text_k_raw, text_v_raw, text_len, cfg, w,
                           text_state.data())) {
      return false;
    }
    have_text = true;
  }

  std::vector<float> prefix((std::size_t)fn * H * mat), suffix(prefix.size());
  if (!branch_scans(a.data(), b.data(), alpha.data(), fn, H, d,
                    have_text ? text_state.data() : nullptr, prefix.data(),
                    suffix.data())) {
    return false;
  }

  const GatherIndex idx = gather_indices(use, fn);
  std::vector<float> state(prefix.size());
  gather_linear_state(prefix.data(), suffix.data(), alpha.data(), idx, fn, H,
                      d, d, cfg.bridge_alpha,
                      have_text ? text_state.data() : nullptr, state.data());

  branch_readout(q.data(), state.data(), xin, fn, S, cfg, w,
                 out + qoff);
  return true;
}

}  // namespace vdn
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
