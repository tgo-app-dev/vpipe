// HiddenStateEncoder: the per-layer hidden-state tap the diffusion text
// encoders condition on.
//
// Two halves. The REGISTRY half needs no model and pins the contract a
// plugin sees: first-wins, a named failure when an architecture is
// unregistered, and the built-in Gemma factory actually being reachable
// (a self-registering TU the linker drops is the failure this catches).
//
// The MODEL half needs VPIPE_GEMMA4_TEST_MODEL_PATH and pins the part
// that is easy to get subtly wrong: the HuggingFace index convention.
// Index n_layers is the residual AFTER the final norm, not the raw one,
// and the two are the same shape -- so nothing but a numeric check
// separates a correct tap from the plausible wrong one.

#include "minitest.h"

#include "generative-models/hidden-state-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

std::string
gemma_dir_()
{
  const char* p = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  return (p != nullptr) ? std::string(p) : std::string();
}

// Row rms of slot `s`, row `r`, reading the compute dtype out by hand.
double
row_rms_(const HiddenTapResult& r, int slot, int row)
{
  const std::size_t off =
      ((std::size_t)slot * r.tokens + (std::size_t)row) * r.hidden;
  const auto* u = static_cast<const std::uint16_t*>(r.data.contents());
  double acc = 0.0;
  for (int i = 0; i < r.hidden; ++i) {
    const std::uint16_t bits = u[off + (std::size_t)i];
    float v = 0.0f;
    if (r.dtype == "bf16") {
      const std::uint32_t w = (std::uint32_t)bits << 16;
      std::memcpy(&v, &w, 4);
    } else {
      // f16 -> f32 by hand; the test must not depend on _Float16 support.
      const std::uint32_t sign = (std::uint32_t)(bits >> 15) << 31;
      const std::uint32_t exp  = (bits >> 10) & 0x1f;
      const std::uint32_t man  = bits & 0x3ff;
      std::uint32_t w = 0;
      if (exp == 0) {
        w = sign;                                  // (sub)normals -> 0
      } else if (exp == 0x1f) {
        w = sign | 0x7f800000u | (man << 13);
      } else {
        w = sign | ((exp + 112) << 23) | (man << 13);
      }
      std::memcpy(&v, &w, 4);
    }
    acc += (double)v * (double)v;
  }
  return std::sqrt(acc / (double)r.hidden);
}

}  // namespace

TEST(hidden_state_encoder, builtin_gemma_is_registered) {
  auto& reg = HiddenStateEncoderRegistry::get();
  // The whole point of referencing register_builtin_hidden_state_encoders
  // from the registry TU rather than self-registering: if the linker had
  // dropped the adaptor, this is empty and every LTX-style encoder
  // silently has no backend.
  EXPECT_TRUE(reg.contains("gemma4_unified"));
  EXPECT_TRUE(reg.contains("Gemma4UnifiedForConditionalGeneration"));
  EXPECT_TRUE(!reg.architectures().empty());
}

TEST(hidden_state_encoder, first_registration_wins) {
  auto& reg = HiddenStateEncoderRegistry::get();
  auto f = [](const HiddenStateEncoderArgs&)
      -> std::unique_ptr<HiddenStateEncoder> { return nullptr; };
  const std::string arch = "test-arch-hse-first-wins";
  EXPECT_TRUE(reg.register_arch(arch, f));
  EXPECT_TRUE(!reg.register_arch(arch, f));   // second is refused
  EXPECT_TRUE(reg.contains(arch));
}

TEST(hidden_state_encoder, unknown_architecture_is_named) {
  auto& reg = HiddenStateEncoderRegistry::get();
  HiddenStateEncoderArgs args;
  args.dir  = "/nonexistent/checkpoint";
  args.arch = "NotAnArchitectureAnyoneRegistered";
  std::string err;
  auto enc = reg.open(args, &err);
  EXPECT_TRUE(enc == nullptr);
  // The message must name BOTH what was found and what is available --
  // "no encoder" and "an encoder this build lacks" are different bugs.
  EXPECT_TRUE(err.find("NotAnArchitectureAnyoneRegistered") !=
              std::string::npos);
  EXPECT_TRUE(err.find("gemma4_unified") != std::string::npos);
}

TEST(hidden_state_encoder, unidentifiable_checkpoint_is_refused) {
  auto& reg = HiddenStateEncoderRegistry::get();
  HiddenStateEncoderArgs args;
  args.dir = "/nonexistent/checkpoint";       // no arch, no config.json
  std::string err;
  EXPECT_TRUE(reg.open(args, &err) == nullptr);
  EXPECT_TRUE(!err.empty());
}

TEST(hidden_state_encoder, all_indices_covers_the_stack) {
  const HiddenTapRequest r = HiddenStateEncoder::all_indices(48);
  EXPECT_TRUE(r.indices.size() == 49);        // L+1 states, not L
  EXPECT_TRUE(r.indices.front() == 0);
  EXPECT_TRUE(r.indices.back() == 48);
}

// ---- the model half ----------------------------------------------------

TEST(hidden_state_encoder, gemma_taps_the_whole_stack) {
  const std::string dir = gemma_dir_();
  if (dir.empty()) { return; }
  vpipe::Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  HiddenStateEncoderArgs args;
  args.dir     = dir;
  args.metal   = mc;
  args.session = &sess;
  std::string err;
  auto enc = HiddenStateEncoderRegistry::get().open(args, &err);
  if (enc == nullptr) {
    // A checkpoint this build cannot open is an environment problem, not
    // a contract failure -- but say so rather than passing quietly.
    std::printf("  [SKIP] %s\n", err.c_str());
    return;
  }

  // The stack this model can actually serve per token. e4b has a
  // KV-shared tail computed for the last position only, so its limit is
  // BELOW n_layers; the 12B unified encoder LTX-2.5 uses has no tail and
  // reports the full stack.
  const int L = enc->max_tap_index();
  const int H = enc->hidden_dim();
  ASSERT_TRUE(L > 0 && H > 0);
  ASSERT_TRUE(L <= enc->n_layers());
  std::printf("  n_layers=%d max_tap_index=%d hidden=%d\n",
              enc->n_layers(), L, H);

  // A short synthetic prompt: this checks the tap, not the tokenizer.
  std::vector<std::int32_t> ids{2, 105, 236742, 1596, 236764, 1902, 236743};
  const int n = (int)ids.size();

  HiddenTapResult res;
  const bool ok = enc->encode(ids, enc->available_indices(), &res, &err);
  if (!ok) { std::printf("  encode failed: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  ASSERT_TRUE(res.valid());
  EXPECT_TRUE(res.slots == L + 1);
  EXPECT_TRUE(res.tokens == n);
  EXPECT_TRUE(res.hidden == H);
  EXPECT_TRUE(res.dtype == "bf16" || res.dtype == "f16");

  // Every state finite, and none of them all-zero (an untouched slot --
  // what a tap that never fired for that layer leaves behind).
  int zero_slots = 0, bad = 0;
  for (int s = 0; s <= L; ++s) {
    double m = 0.0;
    for (int r = 0; r < n; ++r) {
      const double v = row_rms_(res, s, r);
      if (!std::isfinite(v)) { ++bad; }
      m += v;
    }
    if (m == 0.0) { ++zero_slots; std::printf("  slot %d is all zero\n", s); }
  }
  EXPECT_TRUE(bad == 0);
  EXPECT_TRUE(zero_slots == 0);

  // THE INDEX CONVENTION. Index n_layers has been through the final
  // RMSNorm and index n_layers-1 has not. Gemma's residual stream grows
  // through the stack, so the normed state is much smaller -- a tap that
  // returned the raw residual there would land near its neighbour
  // instead. Only checkable when the whole stack is available; a model
  // with a KV-shared tail never reaches the final norm per token.
  double last_normed = 0.0;
  if (L == enc->n_layers()) {
    const double last_raw = row_rms_(res, L - 1, n - 1);
    last_normed           = row_rms_(res, L,     n - 1);
    std::printf("  rms[L-1]=%.4f rms[L]=%.4f\n", last_raw, last_normed);
    EXPECT_TRUE(last_normed < last_raw);
  } else {
    last_normed = row_rms_(res, L, n - 1);
    std::printf("  KV-shared tail: final norm not reachable per token\n");
  }

  // Index 0 is the EMBEDDING output, so it must differ from layer 0's
  // output -- equal ones would mean the tap fired twice at the same
  // point.
  EXPECT_TRUE(std::abs(row_rms_(res, 0, n - 1) - row_rms_(res, 1, n - 1)) >
              1e-4);

  // Deterministic: the same ids on a fresh context give the same states.
  HiddenTapResult res2;
  ASSERT_TRUE(enc->encode(ids, enc->available_indices(), &res2, &err));
  EXPECT_TRUE(std::abs(row_rms_(res2, L, n - 1) - last_normed) < 1e-6);
}

TEST(hidden_state_encoder, gemma_refuses_what_it_cannot_do) {
  const std::string dir = gemma_dir_();
  if (dir.empty()) { return; }
  vpipe::Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  HiddenStateEncoderArgs args;
  args.dir     = dir;
  args.metal   = mc;
  args.session = &sess;
  std::string err;
  auto enc = HiddenStateEncoderRegistry::get().open(args, &err);
  if (enc == nullptr) { return; }

  const int L = enc->n_layers();
  std::vector<std::int32_t> ids{2, 105, 1596};
  HiddenTapResult res;

  // Out of range: REFUSED, not clamped. A clamped index reads a
  // different layer than the caller asked for and nothing downstream can
  // tell.
  err.clear();
  EXPECT_TRUE(!enc->encode(ids, {{L + 1}, 0}, &res, &err));
  EXPECT_TRUE(!err.empty());

  err.clear();
  EXPECT_TRUE(!enc->encode(ids, {{-1}, 0}, &res, &err));
  EXPECT_TRUE(!err.empty());

  // Inside the KV-shared tail: refused, and the message must say WHY --
  // a caller that gets "out of range" for an index that is plainly in
  // range would go looking in the wrong place.
  if (enc->max_tap_index() < L) {
    err.clear();
    EXPECT_TRUE(!enc->encode(ids, {{L}, 0}, &res, &err));
    EXPECT_TRUE(err.find("KV-shared") != std::string::npos);
    // The boundary itself IS available: the tap is taken before the
    // collapse, so index max_tap_index() is still full width.
    err.clear();
    EXPECT_TRUE(enc->encode(ids, {{enc->max_tap_index()}, 0}, &res, &err));
  }

  // A duplicate index would leave one of the two slots stale.
  err.clear();
  EXPECT_TRUE(!enc->encode(ids, {{0, 1, 0}, 0}, &res, &err));
  EXPECT_TRUE(!err.empty());

  // No prefix key-mask on this model: honouring key_valid_len silently
  // would condition the states on the padding.
  err.clear();
  EXPECT_TRUE(!enc->encode(ids, {{0}, 1}, &res, &err));
  EXPECT_TRUE(err.find("key_valid_len") != std::string::npos);

  // Nothing to encode.
  err.clear();
  EXPECT_TRUE(!enc->encode({}, {{0}, 0}, &res, &err));
  EXPECT_TRUE(!err.empty());
}

// A QUANTIZED backbone under a FULL-PRECISION embedding table.
//
// `model-quantize` treats an embedding as a lookup table rather than a
// linear and leaves it dense, so this shape falls out of quantizing any
// comfy-style encoder pack -- LTX-2.5's Gemma-4 12B is the one that
// prompted it. The model then has affine layers and a bf16
// `embed_tokens.weight` with no `.scales` beside it.
//
// Worth its own test because the failure was SILENT and total. The
// prefill has its own embed dispatch, separate from the decode step's,
// and it took the affine branch -- reading three buffers the load never
// filled, writing zeros, and carrying zeros through every layer. Nothing
// errored. MEASURED at the time: conditioning at 6e-5 of its norm, and
// BYTE-IDENTICAL between a 4-bit and an 8-bit backbone, which is what a
// result that does not depend on the weights looks like.
//
// The all-zero-slot assertion in gemma_taps_the_whole_stack would have
// caught it; what was missing was a checkpoint of this shape to run it
// against.
TEST(hidden_state_encoder, gemma_quantized_backbone_with_a_dense_embed) {
  const char* p = std::getenv("VPIPE_GEMMA_DENSE_EMBED_TEST_MODEL_PATH");
  if (p == nullptr || *p == '\0') { return; }
  vpipe::Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  HiddenStateEncoderArgs args;
  args.dir     = p;
  args.metal   = mc;
  args.session = &sess;
  std::string err;
  auto enc = HiddenStateEncoderRegistry::get().open(args, &err);
  if (enc == nullptr) { std::printf("  [SKIP] %s\n", err.c_str()); return; }

  std::vector<std::int32_t> ids{2, 105, 236742, 1596, 236764, 1902, 236743};
  HiddenTapResult res;
  ASSERT_TRUE(enc->encode(ids, enc->available_indices(), &res, &err));
  ASSERT_TRUE(res.valid());

  // Slot 0 is the EMBEDDING output -- the gather's own result, before any
  // layer runs. If the dense table was not reached, this is the slot that
  // is zero, and everything after it inherits that.
  const auto* h = static_cast<const std::uint16_t*>(res.data.contents());
  const std::size_t per_slot = (std::size_t)res.tokens * res.hidden;
  // DTYPE-CORRECT, because the buffer is bf16 on one model and f16 on
  // another and reading either as the other gives a number that is
  // non-zero for the wrong reason -- which would pass this test while
  // measuring nothing.
  const bool is_bf16 = (res.dtype == "bf16");
  auto slot_norm = [&](int s) {
    double acc = 0.0;
    for (std::size_t i = 0; i < per_slot; ++i) {
      const std::uint16_t b = h[(std::size_t)s * per_slot + i];
      float f;
      if (is_bf16) {
        const std::uint32_t u = (std::uint32_t)b << 16;
        std::memcpy(&f, &u, 4);
      } else {
        _Float16 hf;
        std::memcpy(&hf, &b, 2);
        f = (float)hf;
      }
      acc += (double)f * (double)f;
    }
    return std::sqrt(acc);
  };
  const double n0 = slot_norm(0);
  const double nl = slot_norm(res.slots - 1);
  std::printf("  dtype=%s embedding-slot norm %.3f, last-slot norm %.3f\n",
              res.dtype.c_str(), n0, nl);
  EXPECT_TRUE(n0 > 1e-2);
  EXPECT_TRUE(nl > 1e-2);
}
