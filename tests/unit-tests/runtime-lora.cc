// shared/runtime-lora.h: the part of "attach an adapter" that every DiT
// family does the same way.
//
// Worth its own tests because it was EXTRACTED from a working
// implementation, and because the H3 LoRA suite -- which looks like it
// covers this -- does not: those tests either open the adapter file
// directly and check kernels against its shapes, or need a 33B base to
// reach bind_lora_ at all. Nothing exercised the binder cheaply, which
// is exactly the gap that lets an extraction regress silently.
//
// The fixtures are written here, in F32, so every assertion is about
// arithmetic this file can predict.

#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "common/session.h"
#include "generative-models/shared/runtime-lora.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <utility>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
namespace fs = std::filesystem;

namespace {

struct T {
  std::string               name;
  std::vector<std::int64_t> shape;
  std::vector<float>        v;
};

// A minimal F32 safetensors: 8-byte header length, JSON header, data.
bool
write_st(const fs::path& p, const std::vector<T>& ts,
         const std::vector<std::pair<std::string, std::string>>& meta = {})
{
  std::string hdr = "{";
  std::size_t off = 0;
  if (!meta.empty()) {
    hdr += "\"__metadata__\":{";
    for (std::size_t i = 0; i < meta.size(); ++i) {
      if (i) { hdr += ","; }
      hdr += "\"" + meta[i].first + "\":\"" + meta[i].second + "\"";
    }
    hdr += "},";
  }
  for (std::size_t i = 0; i < ts.size(); ++i) {
    const std::size_t n = ts[i].v.size() * 4;
    if (i) { hdr += ","; }
    hdr += "\"" + ts[i].name + "\":{\"dtype\":\"F32\",\"shape\":[";
    for (std::size_t d = 0; d < ts[i].shape.size(); ++d) {
      if (d) { hdr += ","; }
      hdr += std::to_string(ts[i].shape[d]);
    }
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," +
           std::to_string(off + n) + "]}";
    off += n;
  }
  hdr += "}";
  while (hdr.size() % 8 != 0) { hdr += " "; }
  std::ofstream f(p, std::ios::binary);
  if (!f) { return false; }
  const std::uint64_t hl = hdr.size();
  f.write(reinterpret_cast<const char*>(&hl), 8);
  f.write(hdr.data(), (std::streamsize)hdr.size());
  for (const auto& t : ts) {
    f.write(reinterpret_cast<const char*>(t.v.data()),
            (std::streamsize)t.v.size() * 4);
  }
  return (bool)f;
}

std::vector<float>
ramp(std::size_t n, float base, float step)
{
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) { v[i] = base + step * (float)i; }
  return v;
}

float
bf16_at(const metal_compute::SharedBuffer& b, std::size_t i)
{
  const auto* p = static_cast<const std::uint16_t*>(b.contents());
  const std::uint32_t u = (std::uint32_t)p[i] << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

fs::path scratch() { return fs::temp_directory_path() / "vpipe-ut-lora"; }

}  // namespace

TEST(runtime_lora, binds_either_publisher_spelling)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::error_code ec;
  fs::create_directories(scratch(), ec);
  const fs::path p = scratch() / "spellings.safetensors";

  const int R = 4, K = 8, N = 6;
  ASSERT_TRUE(write_st(p, {
      {"attn.to_q.lora_A.weight", {R, K}, ramp(R * K, 1.0f, 0.0f)},
      {"attn.to_q.lora_B.weight", {N, R}, ramp(N * R, 2.0f, 0.0f)},
      // The SAME module under the ComfyUI container prefix.
      {"diffusion_model.attn.to_k.lora_A.weight", {R, K},
       ramp(R * K, 3.0f, 0.0f)},
      {"diffusion_model.attn.to_k.lora_B.weight", {N, R},
       ramp(N * R, 4.0f, 0.0f)},
  }));

  std::string err;
  auto ad = genai::lora::Adapter::open(p.string(), mc, &err);
  ASSERT_TRUE(ad != nullptr);
  if (!ad) { return; }

  genai::lora::Factors q, k, absent;
  EXPECT_TRUE(ad->bind("attn.to_q", N, K, &q));
  EXPECT_TRUE(ad->bind("attn.to_k", N, K, &k));   // via the prefix
  // A module the file does not carry is NOT an error and NOT a skip: an
  // adapter is free to touch some projections and not others.
  EXPECT_FALSE(ad->bind("attn.to_v", N, K, &absent));

  EXPECT_TRUE(q.rank == R && k.rank == R);
  EXPECT_TRUE(absent.empty());
  EXPECT_TRUE(ad->modules() == 2);
  EXPECT_TRUE(ad->skipped() == 0);
  EXPECT_TRUE(ad->max_rank() == R);
  fs::remove(p, ec);
}

// alpha/rank is folded into A at bind, because it is a property of the
// FILE. Getting it wrong changes every output silently, so the values
// are checked rather than just the counts.
TEST(runtime_lora, alpha_folds_into_a_at_alpha_over_rank)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::error_code ec;
  fs::create_directories(scratch(), ec);
  const fs::path p = scratch() / "alpha.safetensors";

  const int R = 4, K = 8, N = 6;
  ASSERT_TRUE(write_st(p, {
      {"m.lora_A.weight", {R, K}, ramp(R * K, 1.0f, 0.0f)},   // all 1.0
      {"m.lora_B.weight", {N, R}, ramp(N * R, 1.0f, 0.0f)},
      {"m.alpha", {1}, {2.0f}},                        // alpha/rank = 0.5
      {"n.lora_A.weight", {R, K}, ramp(R * K, 1.0f, 0.0f)},
      {"n.lora_B.weight", {N, R}, ramp(N * R, 1.0f, 0.0f)},   // no alpha
  }));

  std::string err;
  auto ad = genai::lora::Adapter::open(p.string(), mc, &err);
  ASSERT_TRUE(ad != nullptr);
  if (!ad) { return; }

  genai::lora::Factors withA, withoutA;
  ASSERT_TRUE(ad->bind("m", N, K, &withA));
  ASSERT_TRUE(ad->bind("n", N, K, &withoutA));

  // A is scaled by alpha/rank = 2/4; B is untouched either way.
  EXPECT_TRUE(bf16_at(withA.a, 0) == 0.5f);
  EXPECT_TRUE(bf16_at(withA.b, 0) == 1.0f);
  // Absent alpha means "already at strength", i.e. the same as
  // alpha == rank -- NOT zero, and not a skip.
  EXPECT_TRUE(bf16_at(withoutA.a, 0) == 1.0f);
  EXPECT_TRUE(bf16_at(withoutA.b, 0) == 1.0f);
  fs::remove(p, ec);
}

// A shape that does not fit means the adapter was trained against a
// DIFFERENT model. Counted separately from absence, and the factors are
// left empty rather than half-applied.
TEST(runtime_lora, a_wrong_shape_is_skipped_not_applied)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::error_code ec;
  fs::create_directories(scratch(), ec);
  const fs::path p = scratch() / "shape.safetensors";

  const int R = 4, K = 8, N = 6;
  ASSERT_TRUE(write_st(p, {
      {"bad_k.lora_A.weight", {R, K + 1}, ramp(R * (K + 1), 1.0f, 0.0f)},
      {"bad_k.lora_B.weight", {N, R}, ramp(N * R, 1.0f, 0.0f)},
      {"bad_n.lora_A.weight", {R, K}, ramp(R * K, 1.0f, 0.0f)},
      {"bad_n.lora_B.weight", {N + 1, R}, ramp((N + 1) * R, 1.0f, 0.0f)},
      {"bad_r.lora_A.weight", {R, K}, ramp(R * K, 1.0f, 0.0f)},
      {"bad_r.lora_B.weight", {N, R + 1}, ramp(N * (R + 1), 1.0f, 0.0f)},
      {"good.lora_A.weight", {R, K}, ramp(R * K, 1.0f, 0.0f)},
      {"good.lora_B.weight", {N, R}, ramp(N * R, 1.0f, 0.0f)},
  }));

  std::string err;
  auto ad = genai::lora::Adapter::open(p.string(), mc, &err);
  ASSERT_TRUE(ad != nullptr);
  if (!ad) { return; }

  genai::lora::Factors f;
  EXPECT_FALSE(ad->bind("bad_k", N, K, &f));   // A's K disagrees
  EXPECT_TRUE(f.empty());
  EXPECT_FALSE(ad->bind("bad_n", N, K, &f));   // B's N disagrees
  EXPECT_TRUE(f.empty());
  EXPECT_FALSE(ad->bind("bad_r", N, K, &f));   // A and B disagree on rank
  EXPECT_TRUE(f.empty());
  EXPECT_TRUE(ad->bind("good", N, K, &f));
  EXPECT_TRUE(ad->modules() == 1);
  EXPECT_TRUE(ad->skipped() == 3);
  fs::remove(p, ec);
}

// The header-only probe that decides how BLOCKS are built, before there
// is a model to bind to.
TEST(runtime_lora, file_touches_reads_only_the_header)
{
  std::error_code ec;
  fs::create_directories(scratch(), ec);
  const fs::path p = scratch() / "touch.safetensors";
  ASSERT_TRUE(write_st(p, {
      {"blocks.0.mlp.fc1.lora_A.weight", {2, 2}, ramp(4, 1.0f, 0.0f)},
      {"blocks.0.mlp.fc1.lora_B.weight", {2, 2}, ramp(4, 1.0f, 0.0f)},
  }));
  EXPECT_TRUE(genai::lora::Adapter::file_touches(p.string(),
                                                 ".mlp.fc1.lora_"));
  EXPECT_FALSE(genai::lora::Adapter::file_touches(p.string(), ".ff.gate."));
  // An unreadable file answers NO rather than throwing: the real error
  // belongs to the bind that follows, and refusing a fusion over a file
  // that turns out to be unreadable would be a slowdown chosen for a
  // reason that never materialized.
  EXPECT_FALSE(genai::lora::Adapter::file_touches(
      (scratch() / "nope.safetensors").string(), ".mlp.fc1.lora_"));
  fs::remove(p, ec);
}

// ---- the ai-toolkit / ComfyUI names ---------------------------------

// A stand-in for a family's name map: the same SHAPE as the real one in
// shared/lora-names.h -- gated on a container prefix, rewriting the
// container and the projection -- without dragging a model's topology
// into a test about the mechanism.
std::string
demo_rename(std::string m)
{
  const std::string pre = "vendor.";
  if (m.compare(0, pre.size(), pre) != 0) { return {}; }
  m = m.substr(pre.size());
  const std::size_t p = m.find(".wq");
  if (p != std::string::npos) { m.replace(p, 3, ".to_q"); }
  return "blk." + m;
}

// The whole point of the rename: an adapter published under another
// convention names NOTHING the model has, and a plain lookup then
// reports success while binding zero modules.
TEST(runtime_lora, binds_the_other_convention_through_a_rename)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::error_code ec;
  fs::create_directories(scratch(), ec);
  const fs::path p = scratch() / "rename.safetensors";

  const int R = 4, K = 8, N = 6;
  ASSERT_TRUE(write_st(p, {
      {"vendor.0.attn.wq.lora_A.weight", {R, K}, ramp(R * K, 1.0f, 0.0f)},
      {"vendor.0.attn.wq.lora_B.weight", {N, R}, ramp(N * R, 2.0f, 0.0f)},
  }));
  // The model's name for that module.
  const std::string mine = "blk.0.attn.to_q";

  std::string err;
  {   // WITHOUT the map: nothing binds, and nothing is an error either.
    auto ad = genai::lora::Adapter::open(p.string(), mc, &err);
    ASSERT_TRUE(ad != nullptr);
    if (ad) {
      genai::lora::Factors f;
      EXPECT_FALSE(ad->bind(mine, N, K, &f));
      EXPECT_TRUE(ad->modules() == 0);
      EXPECT_TRUE(ad->skipped() == 0);
    }
  }
  {   // WITH it: bound, and counted as having come the other way.
    auto ad = genai::lora::Adapter::open(p.string(), mc, &err, &demo_rename);
    ASSERT_TRUE(ad != nullptr);
    if (!ad) { return; }
    genai::lora::Factors f;
    EXPECT_TRUE(ad->bind(mine, N, K, &f));
    EXPECT_TRUE(f.rank == R);
    EXPECT_TRUE(ad->modules() == 1);
    EXPECT_TRUE(ad->renamed() == 1);
    // The factors are the ones the file holds, read through the file's
    // own name -- not a name the map invented.
    EXPECT_TRUE(bf16_at(f.a, 0) == 1.0f);
    EXPECT_TRUE(bf16_at(f.b, 0) == 2.0f);
    // A module neither spelling carries is still simply absent.
    genai::lora::Factors g;
    EXPECT_FALSE(ad->bind("blk.0.attn.to_k", N, K, &g));
    EXPECT_TRUE(ad->skipped() == 0);
  }
  fs::remove(p, ec);
}

// A file already in the model's own spelling must be unaffected: the
// direct lookup wins, and nothing is reported as renamed.
TEST(runtime_lora, a_native_file_is_untouched_by_the_rename)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::error_code ec;
  fs::create_directories(scratch(), ec);
  const fs::path p = scratch() / "native.safetensors";
  const int R = 4, K = 8, N = 6;
  ASSERT_TRUE(write_st(p, {
      {"blk.0.attn.to_q.lora_A.weight", {R, K}, ramp(R * K, 5.0f, 0.0f)},
      {"blk.0.attn.to_q.lora_B.weight", {N, R}, ramp(N * R, 7.0f, 0.0f)},
  }));
  std::string err;
  auto ad = genai::lora::Adapter::open(p.string(), mc, &err, &demo_rename);
  ASSERT_TRUE(ad != nullptr);
  if (!ad) { return; }
  genai::lora::Factors f;
  EXPECT_TRUE(ad->bind("blk.0.attn.to_q", N, K, &f));
  EXPECT_TRUE(ad->modules() == 1);
  EXPECT_TRUE(ad->renamed() == 0);
  EXPECT_TRUE(bf16_at(f.a, 0) == 5.0f);
  fs::remove(p, ec);
}

// The fusion gate asks the FILE a question phrased in the MODEL's
// names, so it has to see through the rename as well. Without this an
// adapter would bind its pre-activation projections and then meet a
// fused kernel that drops the delta -- the exact failure the gate
// exists to prevent, reached by the other spelling.
TEST(runtime_lora, file_touches_sees_through_the_rename)
{
  std::error_code ec;
  fs::create_directories(scratch(), ec);
  const fs::path p = scratch() / "touch-rename.safetensors";
  ASSERT_TRUE(write_st(p, {
      {"vendor.0.attn.wq.lora_A.weight", {2, 2}, ramp(4, 1.0f, 0.0f)},
      {"vendor.0.attn.wq.lora_B.weight", {2, 2}, ramp(4, 1.0f, 0.0f)},
  }));
  // The model's spelling of what the file carries.
  EXPECT_FALSE(genai::lora::Adapter::file_touches(p.string(),
                                                  ".attn.to_q.lora_"));
  EXPECT_TRUE(genai::lora::Adapter::file_touches(p.string(),
                                                 ".attn.to_q.lora_",
                                                 &demo_rename));
  // And a needle NEITHER spelling carries still answers no, so the map
  // widens the question rather than blurring it.
  EXPECT_FALSE(genai::lora::Adapter::file_touches(p.string(),
                                                  ".attn.to_v.lora_",
                                                  &demo_rename));
  fs::remove(p, ec);
}

TEST(runtime_lora, an_unreadable_adapter_reports_rather_than_crashes)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto ad = genai::lora::Adapter::open(
      (scratch() / "does-not-exist.safetensors").string(), mc, &err);
  EXPECT_TRUE(ad == nullptr);
  EXPECT_FALSE(err.empty());
}

// ---- the peft spelling, and the file-level alpha ---------------------

// peft writes `<module>.lora_A.<adapter>.weight`. The adapter name is
// usually "default" but is not fixed, so the spelling is DISCOVERED
// from the file rather than guessed at.
TEST(runtime_lora, binds_the_peft_adapter_name_spelling)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::error_code ec;
  fs::create_directories(scratch(), ec);
  const fs::path p = scratch() / "peft.safetensors";
  const int R = 4, K = 8, N = 6;
  // Not "default": the point is that nothing is hard-coded.
  ASSERT_TRUE(write_st(p, {
      {"m.lora_A.turbo_v1.weight", {R, K}, ramp(R * K, 3.0f, 0.0f)},
      {"m.lora_B.turbo_v1.weight", {N, R}, ramp(N * R, 5.0f, 0.0f)},
  }));
  std::string err;
  auto ad = genai::lora::Adapter::open(p.string(), mc, &err);
  ASSERT_TRUE(ad != nullptr);
  if (!ad) { return; }
  genai::lora::Factors f;
  EXPECT_TRUE(ad->has("m"));
  EXPECT_TRUE(ad->bind("m", N, K, &f));
  EXPECT_TRUE(f.rank == R);
  EXPECT_TRUE(bf16_at(f.a, 0) == 3.0f);
  EXPECT_TRUE(bf16_at(f.b, 0) == 5.0f);
  fs::remove(p, ec);
}

// A diffusers/peft export states its alpha ONCE, in __metadata__, and
// writes no per-module `.alpha`. Missing it is not a subtle error: the
// lightx2v adapters are alpha 8 at rank 128, so reading them as
// "already at strength" applies them 16x too hard, with every module
// bound and nothing in the log to say so.
TEST(runtime_lora, the_file_level_alpha_is_read_when_no_module_has_one)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::error_code ec;
  fs::create_directories(scratch(), ec);
  const int R = 4, K = 8, N = 6;

  {   // metadata alpha alone: folds as alpha/rank = 2/4.
    const fs::path p = scratch() / "meta-alpha.safetensors";
    ASSERT_TRUE(write_st(p, {
        {"m.lora_A.default.weight", {R, K}, ramp(R * K, 1.0f, 0.0f)},
        {"m.lora_B.default.weight", {N, R}, ramp(N * R, 1.0f, 0.0f)},
    }, {{"alpha", "2"}, {"format", "pt"}}));
    std::string err;
    auto ad = genai::lora::Adapter::open(p.string(), mc, &err);
    ASSERT_TRUE(ad != nullptr);
    if (ad) {
      EXPECT_TRUE(ad->metadata_alpha() == 2.0f);
      genai::lora::Factors f;
      EXPECT_TRUE(ad->bind("m", N, K, &f));
      EXPECT_TRUE(bf16_at(f.a, 0) == 0.5f);   // 2/4
      EXPECT_TRUE(bf16_at(f.b, 0) == 1.0f);   // B is never rescaled
    }
    fs::remove(p, ec);
  }
  {   // A per-module alpha WINS: it is the more specific statement, and
      // a converted file carries both.
    const fs::path p = scratch() / "both-alpha.safetensors";
    ASSERT_TRUE(write_st(p, {
        {"m.lora_A.weight", {R, K}, ramp(R * K, 1.0f, 0.0f)},
        {"m.lora_B.weight", {N, R}, ramp(N * R, 1.0f, 0.0f)},
        {"m.alpha", {1}, {1.0f}},                     // 1/4
    }, {{"alpha", "2"}}));
    std::string err;
    auto ad = genai::lora::Adapter::open(p.string(), mc, &err);
    ASSERT_TRUE(ad != nullptr);
    if (ad) {
      genai::lora::Factors f;
      EXPECT_TRUE(ad->bind("m", N, K, &f));
      EXPECT_TRUE(bf16_at(f.a, 0) == 0.25f);
    }
    fs::remove(p, ec);
  }
  {   // NEITHER still means "already at strength" -- unchanged, and the
      // reason the metadata read is a fallback and not a default.
    const fs::path p = scratch() / "no-alpha.safetensors";
    ASSERT_TRUE(write_st(p, {
        {"m.lora_A.weight", {R, K}, ramp(R * K, 1.0f, 0.0f)},
        {"m.lora_B.weight", {N, R}, ramp(N * R, 1.0f, 0.0f)},
    }));
    std::string err;
    auto ad = genai::lora::Adapter::open(p.string(), mc, &err);
    ASSERT_TRUE(ad != nullptr);
    if (ad) {
      EXPECT_TRUE(ad->metadata_alpha() == 0.0f);
      genai::lora::Factors f;
      EXPECT_TRUE(ad->bind("m", N, K, &f));
      EXPECT_TRUE(bf16_at(f.a, 0) == 1.0f);
    }
    fs::remove(p, ec);
  }
}

// ---- fusing separate projections into one -------------------------

// A = the parts stacked on the rank axis; B = each part's rows
// scattered, everything else ZERO. The values are checked, not just the
// shapes: a B that came back with the parts in the wrong bands is a
// well-shaped adapter that adds one projection's delta to another's.
TEST(runtime_lora, bind_fused_stacks_a_and_block_diagonalises_b)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::error_code ec;
  fs::create_directories(scratch(), ec);
  const fs::path p = scratch() / "fused.safetensors";
  const int R = 2, K = 3, NP = 4;          // per part
  ASSERT_TRUE(write_st(p, {
      {"q.lora_A.weight", {R, K}, ramp(R * K, 1.0f, 1.0f)},   // 1..6
      {"q.lora_B.weight", {NP, R}, ramp(NP * R, 10.0f, 1.0f)},
      {"k.lora_A.weight", {R, K}, ramp(R * K, 100.0f, 1.0f)},
      {"k.lora_B.weight", {NP, R}, ramp(NP * R, 200.0f, 1.0f)},
  }));
  std::string err;
  auto ad = genai::lora::Adapter::open(p.string(), mc, &err);
  ASSERT_TRUE(ad != nullptr);
  if (!ad) { return; }
  genai::lora::Factors f;
  const int N = 2 * NP;
  // DIRTY THE ALLOCATOR FIRST. Two thirds of a block-diagonal B is
  // zero, and "is zero" is only an assertion about bind_fused if the
  // memory it gets was not already zero -- a fresh Metal buffer usually
  // is, so without this the off-band checks below pass whether or not
  // anything zero-filled them. Small buffers are heap sub-allocations,
  // which is where a released one comes back.
  {
    std::vector<metal_compute::SharedBuffer> dirt;
    for (int i = 0; i < 16; ++i) {
      metal_compute::SharedBuffer b =
          mc->make_shared_buffer((std::size_t)N * 2 * R * 2);
      if (!b.empty()) {
        std::memset(b.contents(), 0xFF, b.byte_size());
        dirt.push_back(std::move(b));
      }
    }
  }
  // Flat: part 0's rows first, then part 1's.
  ASSERT_TRUE(ad->bind_fused({"q", "k"}, N, K, &f,
                             [](int pt, int r) { return pt * NP + r; }));
  EXPECT_TRUE(f.rank == 2 * R);

  // A is [2R, K]: part 0's rows then part 1's, unchanged.
  bool a_ok = true;
  for (int i = 0; i < R * K; ++i) {
    a_ok = a_ok && bf16_at(f.a, (std::size_t)i) == 1.0f + (float)i;
    a_ok = a_ok && bf16_at(f.a, (std::size_t)(R * K + i)) == 100.0f + (float)i;
  }
  EXPECT_TRUE(a_ok);

  // B is [N, 2R]: part p's rows carry its values in columns
  // [p*R, (p+1)*R) and ZERO in the other part's columns.
  bool b_ok = true;
  int nonzero = 0;
  for (int row = 0; row < N; ++row) {
    const int part = row / NP, sub = row % NP;
    const float base = part == 0 ? 10.0f : 200.0f;
    for (int col = 0; col < 2 * R; ++col) {
      const float got = bf16_at(f.b, (std::size_t)row * 2 * R + col);
      const bool in_band = col / R == part;
      const float want = in_band ? base + (float)(sub * R + col % R) : 0.0f;
      if (got != want) { b_ok = false; }
      if (got != 0.0f) { ++nonzero; }
    }
  }
  EXPECT_TRUE(b_ok);
  // Exactly half of a two-part block-diagonal is populated.
  EXPECT_TRUE(nonzero == N * R);
  EXPECT_TRUE(ad->modules() == 1);   // ONE projection, from two files

  // A row map that reorders rather than concatenates -- the shape of
  // the SwiGLU half swap. One part, same rank, permuted rows.
  genai::lora::Factors g;
  ASSERT_TRUE(ad->bind_fused({"q"}, NP, K, &g,
                             [](int, int r) { return NP - 1 - r; }));
  bool g_ok = true;
  for (int row = 0; row < NP; ++row) {
    for (int col = 0; col < R; ++col) {
      const float want = 10.0f + (float)((NP - 1 - row) * R + col);
      g_ok = g_ok && bf16_at(g.b, (std::size_t)row * R + col) == want;
    }
  }
  EXPECT_TRUE(g_ok);

  // A part the file does not carry leaves the WHOLE projection unbound:
  // half a fused adapter is not a weaker one, it is a wrong one.
  genai::lora::Factors h;
  EXPECT_FALSE(ad->bind_fused({"q", "v"}, N, K, &h,
                              [](int pt, int r) { return pt * NP + r; }));
  EXPECT_TRUE(h.empty());
  fs::remove(p, ec);
}

// ---- the Applier: does the adapter reach the GPU at all? ------------

// Everything above tests the READER. This tests the half that computes,
// and it is the half that had no test: an Applier whose kernels do not
// load returns early from every apply(), so the adapter binds, reports
// its modules, and changes NOTHING -- which is indistinguishable, from
// the outside, from an adapter that was never configured.
//
// y += scale * (x A^T) B^T, against a CPU reference.
TEST(runtime_lora, applier_computes_x_a_b_on_the_gpu)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const int M = 5, K = 6, R = 3, N = 4;
  const float scale = 0.5f;

  auto bf16 = [](float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  };
  auto buf = [&](const std::vector<float>& v) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < v.size(); ++i) { d[i] = bf16(v[i]); }
    return b;
  };
  // Small, exactly-representable values: bf16 has 8 mantissa bits, so
  // the reference and the GPU agree to the bit on these and the test
  // needs no tolerance argument.
  auto seq = [](std::size_t n, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = 0.25f + step * (float)(i % 7);
    }
    return v;
  };
  const std::vector<float> xv = seq((std::size_t)M * K, 0.5f);
  const std::vector<float> av = seq((std::size_t)R * K, 0.25f);
  const std::vector<float> bv = seq((std::size_t)N * R, 0.5f);
  const std::vector<float> yv = seq((std::size_t)M * N, 1.0f);

  genai::lora::Factors f;
  f.rank = R;
  f.a = buf(av);
  f.b = buf(bv);
  metal_compute::SharedBuffer x = buf(xv);
  metal_compute::SharedBuffer y = buf(yv);
  ASSERT_TRUE(!f.a.empty() && !f.b.empty() && !x.empty() && !y.empty());

  genai::lora::Applier ap;
  // THE FIRST THING THIS ANSWERS: do the kernels load? A false here is
  // an adapter that can never apply, on every family that uses this.
  ASSERT_TRUE(ap.init(mc, /*allow_mma=*/false));
  EXPECT_TRUE(ap.valid());
  ASSERT_TRUE(ap.ensure_scratch((std::size_t)M * R));

  {
    metal_compute::CommandStream st = mc->make_command_stream();
    metal_compute::ComputeEncoder enc = st.begin_compute();
    ap.apply(enc, x, 0, f, y, 0, M, N, K, scale, /*mma_min_m=*/1 << 30);
    enc.end();
    std::string gerr;
    ASSERT_TRUE(st.commit().wait_ok(&gerr));
  }

  // Reference in f32 over the bf16-rounded inputs.
  auto rd = [&](const metal_compute::SharedBuffer& b, std::size_t i) {
    const std::uint32_t u =
        (std::uint32_t)static_cast<const std::uint16_t*>(b.contents())[i] << 16;
    float o;
    std::memcpy(&o, &u, 4);
    return o;
  };
  int wrong = 0;
  double worst = 0.0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int r = 0; r < R; ++r) {
        double t = 0.0;
        for (int k = 0; k < K; ++k) {
          t += (double)rd(x, (std::size_t)m * K + k) *
               (double)rd(f.a, (std::size_t)r * K + k);
        }
        acc += t * (double)rd(f.b, (std::size_t)n * R + r);
      }
      const double want =
          (double)rd(y, 0) * 0.0 + (double)yv[(std::size_t)m * N + n]
          + (double)scale * acc;
      const double got = rd(y, (std::size_t)m * N + n);
      const double rel =
          std::fabs(got - want) / std::max(1e-6, std::fabs(want));
      worst = std::max(worst, rel);
      if (rel > 1e-2) { ++wrong; }
    }
  }
  if (wrong != 0) {
    std::printf("[runtime_lora] applier: %d of %d cells wrong, "
                "worst rel %.4f\n", wrong, M * N, worst);
  }
  EXPECT_TRUE(wrong == 0);

  // Scale 0 must be EXACTLY off -- not a pair of roundings that cancel.
  metal_compute::SharedBuffer y2 = buf(yv);
  {
    metal_compute::CommandStream st = mc->make_command_stream();
    metal_compute::ComputeEncoder enc = st.begin_compute();
    ap.apply(enc, x, 0, f, y2, 0, M, N, K, 0.0f, 1 << 30);
    enc.end();
    std::string gerr;
    ASSERT_TRUE(st.commit().wait_ok(&gerr));
  }
  bool untouched = true;
  for (int i = 0; i < M * N; ++i) {
    untouched = untouched &&
                rd(y2, (std::size_t)i) == rd(buf(yv), (std::size_t)i);
  }
  EXPECT_TRUE(untouched);
}

// ---- telling a CONVERTED fusion from a TRAINED one ------------------

// The structural question behind the qkv-grouping check.
//
// A fusion CONVERTED from separate q/k/v is block-diagonal by
// construction, and its bands are a claim about which rows of the base
// are q, k and v -- a claim that is wrong on a base grouped the other
// way. One trained DIRECTLY on the fused projection is dense and makes
// no claim at all. Asking the FACTORS is what tells them apart; asking
// "is it fused" cannot, and a check that did warned about correct
// pairings.
TEST(runtime_lora, block_diagonal_tells_a_converted_fusion_from_a_trained_one)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::error_code ec;
  fs::create_directories(scratch(), ec);
  const int R = 2, K = 3, NP = 4, N = 3 * NP;   // 3 parts

  // CONVERTED: built by bind_fused, so block-diagonal by construction.
  const fs::path pc = scratch() / "converted.safetensors";
  ASSERT_TRUE(write_st(pc, {
      {"q.lora_A.weight", {R, K}, ramp(R * K, 1.0f, 1.0f)},
      {"q.lora_B.weight", {NP, R}, ramp(NP * R, 1.0f, 1.0f)},
      {"k.lora_A.weight", {R, K}, ramp(R * K, 2.0f, 1.0f)},
      {"k.lora_B.weight", {NP, R}, ramp(NP * R, 2.0f, 1.0f)},
      {"v.lora_A.weight", {R, K}, ramp(R * K, 3.0f, 1.0f)},
      {"v.lora_B.weight", {NP, R}, ramp(NP * R, 3.0f, 1.0f)},
  }));
  std::string err;
  auto ac = genai::lora::Adapter::open(pc.string(), mc, &err);
  ASSERT_TRUE(ac != nullptr);
  if (!ac) { return; }
  genai::lora::Factors conv;
  ASSERT_TRUE(ac->bind_fused({"q", "k", "v"}, N, K, &conv,
                             [](int pt, int r) { return pt * NP + r; }));
  EXPECT_TRUE(genai::lora::Adapter::block_diagonal_b(conv, 3, N));

  // TRAINED: one dense B over the whole fused width, which is what
  // larryvrh's Turbo adapter is. Same shape, different provenance.
  const fs::path pt = scratch() / "trained.safetensors";
  ASSERT_TRUE(write_st(pt, {
      {"qkv.lora_A.weight", {3 * R, K}, ramp(3 * R * K, 1.0f, 1.0f)},
      {"qkv.lora_B.weight", {N, 3 * R}, ramp(N * 3 * R, 1.0f, 1.0f)},
  }));
  auto at = genai::lora::Adapter::open(pt.string(), mc, &err);
  ASSERT_TRUE(at != nullptr);
  if (!at) { return; }
  genai::lora::Factors trained;
  // rank is 3R (A's rows); K is A's columns, as for any other module.
  ASSERT_TRUE(at->bind("qkv", N, K, &trained));
  EXPECT_FALSE(genai::lora::Adapter::block_diagonal_b(trained, 3, N));

  // A rank that does not divide by the part count is not one of these
  // either -- and must not be read as one.
  EXPECT_FALSE(genai::lora::Adapter::block_diagonal_b(conv, 4, N));

  // ---- and the repair -------------------------------------------
  // Reversing the flat band order is a permutation, so every row keeps
  // its values and only its position changes.
  std::vector<float> before;
  for (int i = 0; i < N * 3 * R; ++i) {
    before.push_back(bf16_at(conv.b, (std::size_t)i));
  }
  ASSERT_TRUE(genai::lora::Adapter::permute_b_rows(
      &conv, N, [](int, int r) { return N - 1 - r; }));
  bool moved = true;
  for (int row = 0; row < N; ++row) {
    for (int c = 0; c < 3 * R; ++c) {
      const float got = bf16_at(conv.b, (std::size_t)row * 3 * R + c);
      const float want = before[(std::size_t)(N - 1 - row) * 3 * R + c];
      if (got != want) { moved = false; }
    }
  }
  EXPECT_TRUE(moved);

  // A map that is NOT a permutation leaves the factors ALONE. A
  // half-scrambled B would be worse than the order already there.
  std::vector<float> now;
  for (int i = 0; i < N * 3 * R; ++i) {
    now.push_back(bf16_at(conv.b, (std::size_t)i));
  }
  EXPECT_FALSE(genai::lora::Adapter::permute_b_rows(
      &conv, N, [](int, int) { return 0; }));       // every row -> 0
  bool untouched = true;
  for (int i = 0; i < N * 3 * R; ++i) {
    if (bf16_at(conv.b, (std::size_t)i) != now[(std::size_t)i]) {
      untouched = false;
    }
  }
  EXPECT_TRUE(untouched);
  fs::remove(pc, ec); fs::remove(pt, ec);
}
