// Qwen-Image-2.1's NEGATIVE prompt, end to end: the conditioner's beat
// and the DiT's sequence.
//
// The negative is a sequence of its own, as the reference's is: its
// conditioning goes through the same template (and the same references,
// grounded), its slot mask is its own (`negative_image_pad_mask`), the
// target's slots are appended to it, and the transformer runs it over
// the same image blocks with that mask -- its text another length, its
// image slots at other rows.
//
// Both halves were wrong at once, and neither shows as a crash. The
// conditioner kept ONE mask and published the positive after encoding
// the negative, so the positive went out with the negative's mask (and
// generate-image refused it whenever the two prompts differed in
// length); and generate-image ran the negative through the POSITIVE's
// layout, reading its rows where the positive's text and slots lie.
//
// Env: VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH, the repo root (text_encoder/
// for the first test, transformer/ for the second).

#include "tests/minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/qwen-image/metal-qwen-image21-transformer.h"
#include "generative-models/gen-input.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/generate-image-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;

#define QN_REQUIRE(cond)                                                   \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::printf("[qwen_image_21_negative] REQUIRE failed: %s\n", #cond); \
      EXPECT_TRUE(false);                                                  \
      return;                                                              \
    }                                                                      \
  } while (0)

namespace {

std::string
root_(const char* sub)
{
  const char* p = std::getenv("VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH");
  if (p == nullptr || *p == '\0') { return {}; }
  std::error_code ec;
  if (!std::filesystem::exists(
          std::filesystem::path(p) / sub / "config.json", ec)) {
    return {};
  }
  return p;
}

std::uint16_t
bf16_bits_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

// One prompt per beat, then the end of the stream.
class SourceText : public TypedStage<SourceText> {
public:
  static constexpr const char* kTypeName = "ut-source-text-qi21-neg";
  SourceText(const SessionContextIntf* s, std::string id,
             std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::vector<std::string> prompts;
  std::size_t next = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (next < prompts.size()) {
      co_await ctx.write(0, std::make_unique<FlexDataPayload>(
                                FlexData::make_string(prompts[next++])));
      co_return;
    }
    ctx.signal_done();
    co_return;
  }
};

// One planar RGB picture, then the end of the stream (so it is held for
// every prompt).
class SourcePicture : public TypedStage<SourcePicture> {
public:
  static constexpr const char* kTypeName = "ut-source-picture-qi21-neg";
  SourcePicture(const SessionContextIntf* s, std::string id,
                std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  int side = 256;
  bool sent = false;
  Job
  process(RuntimeContext& ctx) override
  {
    if (!sent) {
      sent = true;
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::U8;
      tb->shape = {3, side, side};
      tb->resize_contiguous((std::size_t)3 * side * side);
      auto* d = tb->as_u8();
      for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < side; ++y) {
          for (int x = 0; x < side; ++x) {
            // A lit disc on a gradient: something for the tower to see.
            const int dx = x - side / 2, dy = y - side / 3;
            const bool disc = dx * dx + dy * dy < side * side / 16;
            d[((std::size_t)c * side + y) * side + x] = (std::uint8_t)(
                disc ? 230 - 60 * c : (x * (c + 1) + y * (3 - c)) % 256);
          }
        }
      }
      co_await ctx.write(0, std::move(tb));
      co_return;
    }
    ctx.signal_done();
    co_return;
  }
};

class SinkCapture : public TypedStage<SinkCapture> {
public:
  static constexpr const char* kTypeName = "ut-sink-capture-qi21-neg";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> captured;
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

// A conditioning beat as the conditioner sends it: rows, bytes, and the
// sideband's slot mask and reference grids.
struct Cond {
  int rows = 0;
  std::vector<std::uint8_t> bytes;
  bool masked = false;               // the sideband carries a slot mask
  std::vector<int> slots, grid_h, grid_w;
  int ones() const
  {
    int n = 0;
    for (int v : slots) { n += v != 0; }
    return n;
  }
  bool
  operator==(const Cond& o) const
  {
    return rows == o.rows && bytes == o.bytes && masked == o.masked &&
           slots == o.slots && grid_h == o.grid_h && grid_w == o.grid_w;
  }
};

Cond
cond_of_(const BeatPayloadIntf* b)
{
  Cond c;
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(b);
  if (tb == nullptr || tb->shape.size() != 2) { return c; }
  c.rows = (int)tb->shape[0];
  const auto* p = static_cast<const std::uint8_t*>((const void*)tb->as_u8());
  c.bytes.assign(p, p + tb->byte_size());
  if (!tb->sideband.is_object()) { return c; }
  FlexData sb = tb->sideband;          // as_object() is a view: keep it
  auto o = sb.as_object();
  auto ints = [&](std::string_view k, std::vector<int>& dst) {
    if (!o.contains(k)) { return false; }
    const FlexData f = o.at(k);
    if (!f.is_array()) { return false; }
    auto a = f.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      dst.push_back((int)a.at(i).as_int(0));
    }
    return true;
  };
  c.masked = ints(genai::cond_sideband::kImgSlots, c.slots);
  ints(genai::cond_sideband::kRefGridH, c.grid_h);
  ints(genai::cond_sideband::kRefGridW, c.grid_w);
  return c;
}

// One graph through the conditioner: `prompt`, optionally a negative and
// a reference picture. Returns false when it did not run as asked.
bool
condition_(Session& sess, const std::string& root, const std::string& prompt,
           const std::string* negative, bool picture, Cond* pos, Cond* neg)
{
  auto pl = std::make_unique<Pipeline>("qi21-negative-cond", &sess);
  auto su = std::make_unique<SourceText>(&sess, "src", std::vector<InEdge>{},
                                         FlexData::make_object());
  su->prompts = {prompt};
  auto* src = pl->insert_stage(std::move(su));
  // The conditioner's ports: prompt, negative, model, ref_image.
  std::vector<InEdge> in{{src, 0}};
  if (negative != nullptr || picture) {
    Vertex* ns = nullptr;
    if (negative != nullptr) {
      auto nu = std::make_unique<SourceText>(
          &sess, "neg-src", std::vector<InEdge>{}, FlexData::make_object());
      nu->prompts = {*negative};
      ns = pl->insert_stage(std::move(nu));
    }
    in.push_back(ns != nullptr ? InEdge{ns, 0} : InEdge{nullptr, 0});
  }
  if (picture) {
    auto* ps = pl->insert_stage(std::make_unique<SourcePicture>(
        &sess, "pic", std::vector<InEdge>{}, FlexData::make_object()));
    in.push_back(InEdge{nullptr, 0});        // model
    in.push_back(InEdge{ps, 0});             // ref_image
  }
  FlexData c = FlexData::make_object();
  c.as_object().insert("hf_dir", FlexData::make_string(root));
  auto* cond = pl->insert_stage(std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::move(in), std::move(c)));
  auto* sp = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "pos",
                                    std::vector<InEdge>{{cond, 0}},
                                    FlexData::make_object())));
  auto* sn = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "neg",
                                    std::vector<InEdge>{{cond, 1}},
                                    FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  const bool launched = rt.launch();
  if (launched) { rt.wait_idle(); }
  rt.stop();
  if (!launched || sp->captured.size() != 1 ||
      sn->captured.size() != (negative != nullptr ? 1u : 0u)) {
    std::printf("[qwen_image_21_negative] conditioning ran %d, %zu "
                "positive(s), %zu negative(s)\n", (int)launched,
                sp->captured.size(), sn->captured.size());
    return false;
  }
  *pos = cond_of_(sp->captured[0].get());
  if (neg != nullptr && !sn->captured.empty()) {
    *neg = cond_of_(sn->captured[0].get());
  }
  return true;
}

}  // namespace

// THE NEGATIVE'S BEAT IS ITS OWN ENCODE. A prompt and its negative: the
// positive carries its own mask (every row of it), and the negative is
// exactly the conditioning -- bytes and mask -- of a run whose PROMPT is
// the negative's text. Text alone, then grounded on one reference, where
// both masks hold the picture's slots at different rows. ~16 GB of
// encoder, read four times.
TEST(qwen_image_21_negative, the_negative_carries_its_own_mask)
{
  const std::string root = root_("text_encoder");
  if (root.empty()) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const std::string p = "a red fox asleep in fresh snow, morning light";
  const std::string n =
      "blurry, low quality, watermark, oversaturated, extra limbs, text";
  for (const bool picture : {false, true}) {
    Cond pos, neg, alone;
    QN_REQUIRE(condition_(sess, root, p, &n, picture, &pos, &neg));
    QN_REQUIRE(condition_(sess, root, n, nullptr, picture, &alone, nullptr));
    std::printf("[qwen_image_21_negative] %s: prompt %d rows (mask %zu, %d "
                "slots), negative %d rows (mask %zu, %d slots), negative "
                "as a prompt %d rows -- %s\n",
                picture ? "one reference" : "text alone", pos.rows,
                pos.slots.size(), pos.ones(), neg.rows, neg.slots.size(),
                neg.ones(), alone.rows,
                neg == alone ? "the same beat" : "DIFFERENT");
    // The two prompts differ in length, or this proves nothing.
    QN_REQUIRE(pos.rows != neg.rows);
    EXPECT_TRUE(pos.masked && (int)pos.slots.size() == pos.rows);
    EXPECT_TRUE(neg.masked && (int)neg.slots.size() == neg.rows);
    EXPECT_TRUE(neg == alone);
    if (picture) {
      // The same picture's slots in both, and its grid published on both.
      EXPECT_TRUE(pos.ones() > 0 && pos.ones() == neg.ones());
      EXPECT_TRUE(pos.grid_h.size() == 1 && pos.grid_h == neg.grid_h &&
                  pos.grid_w == neg.grid_w);
    } else {
      EXPECT_TRUE(pos.ones() == 0 && neg.ones() == 0);
    }
  }
}

namespace {

// Conditioning beats straight onto generate-image's ports, each
// generation an optional negative (sent FIRST, as the conditioner sends
// it) and its positive -- no encoder.
class SourceCond : public TypedStage<SourceCond> {
public:
  static constexpr const char* kTypeName = "ut-source-cond-qi21-neg";
  SourceCond(const SessionContextIntf* s, std::string id,
             std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(2); }
  struct Beat {
    std::vector<float> rows;          // [n, cols]
    std::vector<int> slots;           // empty: no mask on the beat
    std::vector<int> grid_h, grid_w;
  };
  struct Gen {
    Beat pos;
    bool has_neg = false;
    Beat neg;
  };
  std::vector<Gen> gens;
  int cols = 0;
  std::size_t next = 0;
  std::unique_ptr<TensorBeatPayload>
  beat_(const Beat& b) const
  {
    auto tb = std::make_unique<TensorBeatPayload>();
    tb->dtype = TensorBeat::DType::Bf16;
    tb->shape = {(int)(b.rows.size() / (std::size_t)cols), cols};
    tb->resize_contiguous(b.rows.size());
    auto* d = reinterpret_cast<std::uint16_t*>((void*)tb->as_u8());
    for (std::size_t i = 0; i < b.rows.size(); ++i) {
      d[i] = bf16_bits_(b.rows[i]);
    }
    if (!b.slots.empty()) {
      FlexData sb = FlexData::make_object();
      auto o = sb.as_object();
      auto arr = [](const std::vector<int>& v) {
        FlexData a = FlexData::make_array();
        auto av = a.as_array();
        for (int x : v) { av.push_back(FlexData::make_int(x)); }
        return a;
      };
      o.insert_or_assign(genai::cond_sideband::kImgSlots, arr(b.slots));
      o.insert_or_assign(genai::cond_sideband::kRefGridH, arr(b.grid_h));
      o.insert_or_assign(genai::cond_sideband::kRefGridW, arr(b.grid_w));
      tb->sideband = std::move(sb);
    }
    return tb;
  }
  Job
  process(RuntimeContext& ctx) override
  {
    if (next < gens.size()) {
      const Gen& g = gens[next++];
      if (g.has_neg) { co_await ctx.write(1, beat_(g.neg)); }
      co_await ctx.write(0, beat_(g.pos));
      co_return;
    }
    ctx.signal_done();
    co_return;
  }
};

// One reference latent [C, h, w], then the end of the stream (held).
class SourceLatent : public TypedStage<SourceLatent> {
public:
  static constexpr const char* kTypeName = "ut-source-latent-qi21-neg";
  SourceLatent(const SessionContextIntf* s, std::string id,
               std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  int C = 64, h = 0, w = 0;
  bool sent = false;
  Job
  process(RuntimeContext& ctx) override
  {
    if (!sent) {
      sent = true;
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::F32;
      tb->shape = {C, h, w};
      tb->resize_contiguous((std::size_t)C * h * w);
      std::mt19937 rng(11);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      float* d = tb->as_f32();
      for (std::size_t i = 0; i < (std::size_t)C * h * w; ++i) {
        d[i] = nd(rng);
      }
      co_await ctx.write(0, std::move(tb));
      co_return;
    }
    ctx.signal_done();
    co_return;
  }
};

std::vector<float>
randn_(std::size_t n, unsigned seed)
{
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) { x = nd(rng); }
  return v;
}

// A mask of text and slot runs: {3, -4, 4} is 3 text rows, 4 slots, 4
// text rows.
std::vector<int>
runs_(std::initializer_list<int> r)
{
  std::vector<int> m;
  for (int k : r) { m.insert(m.end(), (std::size_t)std::abs(k), k < 0); }
  return m;
}

// The generations' latents, in order, from one graph.
std::vector<std::vector<float>>
generate_(Session& sess, const std::string& root, SourceCond::Gen* gens,
          int ngen, int cols, int ref_h, int ref_w, double guidance)
{
  auto pl = std::make_unique<Pipeline>("qi21-negative-gen", &sess);
  auto su = std::make_unique<SourceCond>(&sess, "src", std::vector<InEdge>{},
                                         FlexData::make_object());
  su->gens.assign(gens, gens + ngen);
  su->cols = cols;
  auto* src = pl->insert_stage(std::move(su));
  // generate-image's ports: conditioning, neg_conditioning, model,
  // sampler, scheduler, ref_latent0.
  std::vector<InEdge> in{{src, 0}, {src, 1}};
  if (ref_h > 0) {
    auto lu = std::make_unique<SourceLatent>(
        &sess, "ref", std::vector<InEdge>{}, FlexData::make_object());
    lu->h = ref_h;
    lu->w = ref_w;
    auto* ls = pl->insert_stage(std::move(lu));
    in.push_back(InEdge{nullptr, 0});
    in.push_back(InEdge{nullptr, 0});
    in.push_back(InEdge{nullptr, 0});
    in.push_back(InEdge{ls, 0});
  }
  FlexData gc = FlexData::make_object();
  auto go = gc.as_object();
  go.insert("hf_dir", FlexData::make_string(root));
  go.insert("width", FlexData::make_int(256));
  go.insert("height", FlexData::make_int(256));
  go.insert("seed", FlexData::make_int(7));
  go.insert("seed_sequence", FlexData::make_string("keep"));
  // ONE step: the latent is then the noise plus a fixed multiple of the
  // velocity, so a guided step is EXACTLY the unguided ones combined.
  go.insert("steps", FlexData::make_int(1));
  go.insert("guidance_scale", FlexData::make_real(guidance));
  auto* gen = static_cast<GenerateImageStage*>(pl->insert_stage(
      std::make_unique<GenerateImageStage>(&sess, "gen", std::move(in),
                                           std::move(gc))));
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "sink",
                                    std::vector<InEdge>{{gen, 0}},
                                    FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  const bool launched = rt.launch();
  if (launched) { rt.wait_idle(); }
  rt.stop();
  std::vector<std::vector<float>> out;
  if (!launched || !gen->config_error().empty()) { return out; }
  for (const auto& b : sink->captured) {
    const auto* t = dynamic_cast<const TensorBeatPayload*>(b.get());
    if (t == nullptr || t->dtype != TensorBeat::DType::F32) { continue; }
    out.emplace_back(t->as_f32(), t->as_f32() + t->element_count());
  }
  return out;
}

}  // namespace

// THE NEGATIVE RUNS AS ITS OWN SEQUENCE. One step from one seed makes
// the latent x0 + dt * v, so the guided step's latent is the two
// unguided ones combined -- lat(neg) + g (lat(pos) - lat(neg)) -- where
// each unguided run took that conditioning as its PROMPT, its own
// sequence by construction. Run with the positive's layout, the
// negative's rows are read where the positive's text and slots lie and
// the guided latent is another picture. Hand-made conditioning (no
// encoder) of different lengths: text alone, then on one reference
// latent with its slots at different rows. A negative whose beat
// carries no mask is refused, as the positive is. ~13 GB of DiT.
TEST(qwen_image_21_negative, cfg_runs_the_negative_through_its_own_sequence)
{
  const std::string root = root_("transformer");
  if (root.empty()) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  genai::MetalQwenImage21Transformer::Config dc;
  QN_REQUIRE(genai::MetalQwenImage21Transformer::Config::read_dims(
      (std::filesystem::path(root) / "transformer").string(), &dc));
  const int cols = dc.txt_dim;
  const double g = 4.0;
  for (const bool ref : {false, true}) {
    // A 4x4 reference latent is 16 tokens: four slots.
    const std::vector<int> pm = ref ? runs_({3, -4, 4}) : runs_({7});
    const std::vector<int> nm = ref ? runs_({6, -4, 2}) : runs_({12});
    const std::vector<int> grid = ref ? std::vector<int>{4}
                                      : std::vector<int>{};
    SourceCond::Beat P{randn_(pm.size() * cols, 1), pm, grid, grid};
    SourceCond::Beat N{randn_(nm.size() * cols, 2), nm, grid, grid};
    SourceCond::Beat bare = N;
    bare.slots.clear();
    // The guided generation FIRST, then the refused one: a negative is
    // taken off its port when its positive arrives, so a later
    // generation's negative must not already be waiting there for an
    // earlier one that has none.
    SourceCond::Gen gens[] = {
        {P, true, N},        // guided
        {P, true, bare},     // a negative with no mask: refused
        {P, false, {}},      // the positive alone
        {N, false, {}},      // the negative as a prompt
    };
    const auto lat = generate_(sess, root, gens, 4, cols, ref ? 4 : 0,
                               ref ? 4 : 0, g);
    QN_REQUIRE(lat.size() == 3);
    const auto& c = lat[0];
    const auto& a = lat[1];
    const auto& b = lat[2];
    QN_REQUIRE(!c.empty() && a.size() == c.size() && b.size() == c.size());
    double e2 = 0.0, r2 = 0.0, d2 = 0.0;
    for (std::size_t k = 0; k < c.size(); ++k) {
      const double want = (double)b[k] + g * ((double)a[k] - (double)b[k]);
      e2 += (c[k] - want) * (c[k] - want);
      r2 += want * want;
      d2 += ((double)a[k] - b[k]) * ((double)a[k] - b[k]);
    }
    const double rel = std::sqrt(e2 / std::max(r2, 1e-30));
    const double moved = std::sqrt(d2 / std::max(r2, 1e-30));
    std::printf("[qwen_image_21_negative] %s: guided latent %.3g from the "
                "unguided pair combined (the prompts move it %.3g)\n",
                ref ? "one reference" : "text alone", rel, moved);
    // The two prompts must move the latent, or the combination is
    // vacuous.
    EXPECT_TRUE(moved > 1e-2);
    EXPECT_TRUE(rel < 1e-4);
  }
}
