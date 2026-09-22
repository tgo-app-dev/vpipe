#include "generative-models/z-image/z-image-layout.h"

#include <cstddef>

namespace vpipe {
namespace genai {
namespace zimage {

namespace {

int round_up_(int n, int mult)
{
  if (mult <= 0) { return n; }
  const int r = n % mult;
  return r == 0 ? n : n + (mult - r);
}

}  // namespace

bool
build_layout(int grid_h, int grid_w, int cap_len, Layout* out,
             std::string* err)
{
  auto fail = [&](const char* why) {
    if (err != nullptr) { *err = why; }
    return false;
  };
  if (out == nullptr) { return fail("no output layout"); }
  if (grid_h <= 0 || grid_w <= 0) {
    return fail("patch grid must be positive on both axes");
  }
  if (cap_len <= 0) { return fail("caption must have at least one token"); }
  if (cap_len > kMaxCaptionTokens) {
    // The reference TRUNCATES rather than refusing, but it does so in
    // the tokenizer, before anything here can see it. Reaching this
    // with a longer caption means the caller skipped that step, and
    // silently keeping the extra rows would condition on a sequence
    // the model was never trained to see.
    return fail("caption exceeds the 512-token limit; truncate first");
  }

  Layout L;
  L.grid_h = grid_h;
  L.grid_w = grid_w;
  L.img_tokens = grid_h * grid_w;
  L.img_total = round_up_(L.img_tokens, kSeqMultiple);
  L.cap_len = cap_len;
  L.cap_total = round_up_(cap_len, kSeqMultiple);
  L.joint_len = L.img_total + L.cap_total;

  L.pos_t.assign((std::size_t)L.joint_len, 0);
  L.pos_h.assign((std::size_t)L.joint_len, 0);
  L.pos_w.assign((std::size_t)L.joint_len, 0);

  // The image half. Every real image row shares ONE axis-0 coordinate,
  // one past the caption's last: the reference's position grid is
  // (frames, height, width) with frames == 1 for a still, started at
  // cap_total + 1. So axis 0 separates the two halves and axes 1/2
  // carry the picture. Pad rows sit at the origin.
  const std::int32_t img_t = (std::int32_t)(L.cap_total + 1);
  for (int h = 0; h < grid_h; ++h) {
    for (int w = 0; w < grid_w; ++w) {
      const std::size_t r = (std::size_t)(h * grid_w + w);
      L.pos_t[r] = img_t;
      L.pos_h[r] = (std::int32_t)h;
      L.pos_w[r] = (std::int32_t)w;
    }
  }
  // [img_tokens, img_total) stays (0, 0, 0) from the assign above.

  // The caption half: positions 1..cap_total on axis 0, zero on the
  // other two, and the padding CONTINUES the run rather than falling
  // back to the origin. See the header -- this asymmetry is real.
  for (int i = 0; i < L.cap_total; ++i) {
    L.pos_t[(std::size_t)(L.img_total + i)] = (std::int32_t)(i + 1);
  }

  *out = std::move(L);
  return true;
}

void
pack_latents(const float* z, int channels, int lat_h, int lat_w, int patch,
             float* rows)
{
  if (z == nullptr || rows == nullptr || patch <= 0) { return; }
  const int gh = lat_h / patch, gw = lat_w / patch;
  const int rw = patch * patch * channels;
  for (int ht = 0; ht < gh; ++ht) {
    for (int wt = 0; wt < gw; ++wt) {
      float* dst = rows + (std::size_t)(ht * gw + wt) * rw;
      for (int ph = 0; ph < patch; ++ph) {
        for (int pw = 0; pw < patch; ++pw) {
          const int h = ht * patch + ph, w = wt * patch + pw;
          float* d = dst + (std::size_t)(ph * patch + pw) * channels;
          for (int c = 0; c < channels; ++c) {
            d[c] = z[((std::size_t)c * lat_h + h) * lat_w + w];
          }
        }
      }
    }
  }
}

void
unpack_latents(const float* rows, int channels, int lat_h, int lat_w,
               int patch, float* z)
{
  if (z == nullptr || rows == nullptr || patch <= 0) { return; }
  const int gh = lat_h / patch, gw = lat_w / patch;
  const int rw = patch * patch * channels;
  for (int ht = 0; ht < gh; ++ht) {
    for (int wt = 0; wt < gw; ++wt) {
      const float* src = rows + (std::size_t)(ht * gw + wt) * rw;
      for (int ph = 0; ph < patch; ++ph) {
        for (int pw = 0; pw < patch; ++pw) {
          const int h = ht * patch + ph, w = wt * patch + pw;
          const float* s = src + (std::size_t)(ph * patch + pw) * channels;
          for (int c = 0; c < channels; ++c) {
            z[((std::size_t)c * lat_h + h) * lat_w + w] = s[c];
          }
        }
      }
    }
  }
}

std::string
prompt_template(const std::string& text)
{
  return "<|im_start|>user\n" + text +
         "<|im_end|>\n<|im_start|>assistant\n";
}

}  // namespace zimage
}  // namespace genai
}  // namespace vpipe
