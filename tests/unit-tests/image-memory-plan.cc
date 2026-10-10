// Does a streaming image DiT declare a floor the RESOURCE PHASE can see?
//
// There are two ledgers and they take the same number for different
// purposes. StageMemory::hold(source, preload, floor) is what a run
// REPORTS; ResourceClaim::floor_bytes is what phase_footprint_floor()
// and phase_peak() read, and those are the only place a graph is turned
// away. A DiT claimed without one is judged at its full on-disk size
// however carefully the plan describes its floor.
//
// Gated on VPIPE_FLUX2_TEST_MODEL_PATH because it needs a real
// checkpoint's tensor table, but nothing here is FLUX.2-specific: point
// it at any generate-image family and the same assertions hold. It
// loads no weights.
//
// Env: VPIPE_FLUX2_TEST_MODEL_PATH = an image model root (a directory
// holding transformer/).

#include "minitest.h"

// MiniMax-H3 is a VIDEO family and this file is the image plan's, but the
// ANE claim is one contract and it broke the same way in both: the qkv
// tier's module booked at zero. The arithmetic for both lives beside the
// image family's here rather than in the H3 suite, whose tests are gated
// on a 33B checkpoint and skip on most boxes -- which is how the gap
// survived in the first place.
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include <iterator>
#include "pipeline/resource-plan.h"
#include "generative-models/generative-model-manager.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "pipeline/pipeline.h"
#include "generative-models/image-model-registry.h"
#include "generative-models/vosr/metal-vosr-transformer.h"
#include "stages/generate-image-stage.h"
#include "stages/model-memory.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vpipe;

// THE DiT'S STREAMING FLOOR HAS TO REACH THE LEDGER THAT REFUSES.
//
// There are two, and they take the same number for different purposes.
// declare_memory()'s floor is what a run REPORTS. The floor on a
// ResourceClaim is what phase_footprint_floor() and phase_peak() read,
// and those are the only place a graph is turned away. A DiT claimed
// without one is judged at its full on-disk size -- so "everything
// streamable at its floor" reads identically to "everything preloaded",
// and a 17 GB checkpoint that runs on a couple of blocks is weighed as
// 17 GB against the box.
//
// Both are asserted here, and asserted EQUAL, because two ledgers fed
// from two copies of a block-stem list is exactly how they drift.
// Neither loads anything: both are answered from the tensor table.
TEST(image_memory_plan, the_dit_is_claimed_at_its_streaming_floor)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  namespace fs = std::filesystem;
  const std::string dit = (fs::path(root) / "transformer").string();
  const std::size_t whole = model_memory::dir_weights_bytes(dit);
  if (whole == 0) { return; }

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{},
                           std::move(cfg));
  ASSERT_TRUE(stage.config_error().empty());

  std::size_t claim_floor = 0;
  bool found = false;
  for (const ResourceClaim& c : stage.declare_resources()) {
    if (c.key == dit) { claim_floor = c.floor_bytes; found = true; }
  }
  ASSERT_TRUE(found);
  std::printf("[image_memory_plan] DiT %zu MB on disk, claimed floor %zu MB\n",
              whole >> 20, claim_floor >> 20);
  // A floor of zero is not "unknown" to the planner -- it is "this
  // cannot be reduced", which is the whole bug.
  EXPECT_TRUE(claim_floor > 0);
  EXPECT_TRUE(claim_floor < whole);

  std::size_t plan_floor = 0;
  for (const auto& h : stage.declare_memory().holdings) {
    if (h.source == dit) { plan_floor = h.floor; }
  }
  EXPECT_TRUE(plan_floor == claim_floor);
}

// The ANE tier's memory is CoreML's, so the only thing that can keep it
// honest is the claim. Two halves, and both have to hold: the stage
// claims when the tier is on, and does NOT when it is off -- a claim for
// bytes nothing allocates is the same lie pointing the other way.
TEST(image_memory_plan, the_ane_tier_claims_coreml_residency)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;

  auto claims_of = [&](bool ane) {
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("hf_dir", FlexData::make_string(root));
    if (ane) {
      cfg.as_object().insert("ane_ffn", FlexData::make_bool(true));
    }
    GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{},
                             std::move(cfg));
    std::size_t bytes = 0;
    int units = 0;
    for (const ResourceClaim& c : stage.declare_resources()) {
      if (c.kind != model_memory::kCoreMLKind) { continue; }
      // "<label>|<unit_bytes>|<units>"
      const std::size_t b2 = c.key.rfind('|');
      const std::size_t b1 = c.key.rfind('|', b2 - 1);
      bytes = (std::size_t)std::stoull(c.key.substr(b1 + 1, b2 - b1 - 1));
      units = std::stoi(c.key.substr(b2 + 1));
      // Charged to the denoise: the modules go when the DiT does, so a
      // claim held through the decode would put them in a moment they
      // are not in.
      EXPECT_TRUE(c.phase == model_memory::kPhaseDenoise);
    }
    return std::make_pair(bytes, units);
  };

  const auto off = claims_of(false);
  EXPECT_TRUE(off.second == 0);

  const auto on = claims_of(true);
  std::printf("[image_memory_plan] ANE claim: %d unit(s) x %zu MB\n",
              on.second, on.first >> 20);
  // ONE unit: the feed-forward runs on a single shared runtime-weight
  // module, so its cost does not scale with the number of blocks.
  EXPECT_TRUE(on.second == 1);
  const genai::MetalKrea2Transformer::Config kc;
  EXPECT_TRUE(on.first ==
              genai::MetalKrea2Transformer::ane_runtime_bytes(
                  kc, genai::MetalKrea2Transformer::ane_plan_seq(0, 0)));
  // At least the IOSurface weight slots themselves, which hold a full
  // fp16 copy of one block's feed-forward.
  const std::size_t slots =
      3ull * (std::size_t)kc.hidden * (std::size_t)kc.ffn * 2ull;
  EXPECT_TRUE(on.first > slots);
}

// THE ARITHMETIC, with no checkpoint in sight.
//
// ane_runtime_bytes() is static and takes a Config, so the accounting
// can be checked on any box -- which matters here more than usual: the
// stage-level test below needs a Krea-2 checkpoint and skips without
// one, and this bug (the qkv module booked at zero) survived precisely
// because nothing exercised the path. A test that runs everywhere is
// what keeps it fixed.
TEST(image_memory_plan, ane_runtime_bytes_counts_both_modules) {
  // The env form ORs into the booking and would make both arms equal.
  if (std::getenv("VPIPE_KREA2_ANE_QKV") != nullptr) { return; }
  namespace k = genai;
  k::MetalKrea2Transformer::Config ff;
  k::MetalKrea2Transformer::Config both;
  both.ane_qkv = true;
  const int seq = k::MetalKrea2Transformer::ane_plan_seq(0, 0);
  const std::size_t a = k::MetalKrea2Transformer::ane_runtime_bytes(ff, seq);
  const std::size_t b = k::MetalKrea2Transformer::ane_runtime_bytes(both, seq);
  std::printf("[image_memory_plan] krea2 ane bytes: ff %zu MB, ff+qkv %zu MB\n",
              a >> 20, b >> 20);
  EXPECT_TRUE(a > 0);
  // The qkv module is not free, and before this it was booked as if it
  // were: the two figures were identical.
  EXPECT_TRUE(b > a);

  // MiniMax-H3 had the same gap, and from the same cause -- the video
  // preflight charged for the qkv tier while the CLAIM did not, so the
  // gate refused against memory the plan had never granted.
  k::MetalMiniMaxH3Transformer::Config h3ff;
  k::MetalMiniMaxH3Transformer::Config h3both;
  h3both.ane_qkv = true;
  if (std::getenv("VPIPE_H3_ANE_QKV") == nullptr) {
    const std::size_t c =
        k::MetalMiniMaxH3Transformer::ane_runtime_bytes(h3ff, 19456);
    const std::size_t d =
        k::MetalMiniMaxH3Transformer::ane_runtime_bytes(h3both, 19456);
    std::printf("[image_memory_plan] h3 ane bytes: ff %zu MB, ff+qkv %zu MB\n",
                c >> 20, d >> 20);
    EXPECT_TRUE(c > 0);
    EXPECT_TRUE(d > c);
  }
}

// `ane_qkv` is a SECOND module, and the claim has to say so.
//
// It did not. ane_runtime_bytes() booked the feed-forward alone, so a
// graph that turned the qkv tier on allocated a module no ledger knew
// about -- on a box where the plan had already promised that memory to
// something else. The tier was reachable only through
// VPIPE_KREA2_ANE_QKV at the time, which is why it went unnoticed: no
// pipeline spec could ask for it, so nothing exercised the path.
TEST(image_memory_plan, ane_qkv_is_claimed_beside_the_feed_forward) {
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  // The env form ORs into the booking, so it would make the two arms
  // below identical and the test vacuous rather than failing.
  if (std::getenv("VPIPE_KREA2_ANE_QKV") != nullptr) { return; }
  Session sess;

  auto claim_bytes = [&](bool qkv) {
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("hf_dir", FlexData::make_string(root));
    cfg.as_object().insert("ane_ffn", FlexData::make_bool(true));
    if (qkv) { cfg.as_object().insert("ane_qkv", FlexData::make_bool(true)); }
    GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{},
                             std::move(cfg));
    std::size_t bytes = 0;
    for (const ResourceClaim& c : stage.declare_resources()) {
      if (c.kind != model_memory::kCoreMLKind) { continue; }
      const std::size_t b2 = c.key.rfind('|');
      const std::size_t b1 = c.key.rfind('|', b2 - 1);
      bytes = (std::size_t)std::stoull(c.key.substr(b1 + 1, b2 - b1 - 1));
    }
    return bytes;
  };

  const std::size_t ff_only  = claim_bytes(false);
  const std::size_t with_qkv = claim_bytes(true);
  std::printf("[image_memory_plan] ANE claim: ff %zu MB, ff+qkv %zu MB\n",
              ff_only >> 20, with_qkv >> 20);
  // STRICTLY more, which is the whole point: the qkv module's weight
  // slots and staging are not free, and a claim that did not grow is a
  // claim that is not covering them.
  EXPECT_TRUE(ff_only > 0);
  EXPECT_TRUE(with_qkv > ff_only);

  // And exactly what the family will build, so the plan grants the same
  // shape the model allocates.
  genai::MetalKrea2Transformer::Config kq;
  kq.ane_qkv = true;
  EXPECT_TRUE(with_qkv ==
              genai::MetalKrea2Transformer::ane_runtime_bytes(
                  kq, genai::MetalKrea2Transformer::ane_plan_seq(0, 0)));
}

// A QUANTIZED checkpoint claims the ANE module exactly as a dense one does:
// its feed-forward is dequantized into the module's fp16 inputs per block,
// so the module costs the same. Built as a copy of the real checkpoint's
// layout that differs only in transformer/config.json carrying a
// quantization block -- what the loader keys on -- so the claim is checked
// against the same geometry either way.
TEST(image_memory_plan, a_quantized_checkpoint_claims_the_ane_module)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  namespace fs = std::filesystem;
  const fs::path src(root);
  const fs::path q = fs::temp_directory_path() / "vpipe-krea2-quantized-plan";
  std::error_code ec;
  fs::remove_all(q, ec);
  fs::create_directories(q / "transformer", ec);
  for (const auto& e : fs::directory_iterator(src, ec)) {
    if (e.path().filename() == "transformer") { continue; }
    fs::create_symlink(e.path(), q / e.path().filename(), ec);
  }
  for (const auto& e : fs::directory_iterator(src / "transformer", ec)) {
    if (e.path().filename() == "config.json") { continue; }
    fs::create_symlink(e.path(), q / "transformer" / e.path().filename(), ec);
  }
  {
    std::ifstream in(src / "transformer" / "config.json");
    std::string j((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
    const std::size_t brace = j.find('{');
    ASSERT_TRUE(brace != std::string::npos);
    j.insert(brace + 1,
             "\n  \"quantization\": {\"bits\": 4, \"group_size\": 64},");
    std::ofstream out(q / "transformer" / "config.json");
    out << j;
  }

  Session sess;
  auto ane_units = [&](const std::string& r) {
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("hf_dir", FlexData::make_string(r));
    cfg.as_object().insert("ane_ffn", FlexData::make_bool(true));
    GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{},
                             std::move(cfg));
    int n = 0;
    for (const ResourceClaim& c : stage.declare_resources()) {
      if (c.kind == model_memory::kCoreMLKind) { ++n; }
    }
    return n;
  };
  EXPECT_TRUE(ane_units(src.string()) == 1);
  EXPECT_TRUE(ane_units(q.string()) == 1);
  fs::remove_all(q, ec);
}

namespace {

// A real safetensors file of F32 zeros: an 8-byte little-endian header
// length, the JSON table, the data. Enough for the tensor table the
// planner reads, and nothing is ever loaded from it.
void
write_safetensors_(const std::filesystem::path& p,
                   const std::vector<std::pair<std::string, std::size_t>>& t)
{
  std::string hdr = "{";
  std::size_t off = 0;
  for (std::size_t i = 0; i < t.size(); ++i) {
    if (i != 0) { hdr += ","; }
    hdr += "\"" + t[i].first + "\":{\"dtype\":\"F32\",\"shape\":[" +
           std::to_string(t[i].second) + "],\"data_offsets\":[" +
           std::to_string(off * 4) + "," +
           std::to_string((off + t[i].second) * 4) + "]}";
    off += t[i].second;
  }
  hdr += "}";
  while (hdr.size() % 8 != 0) { hdr += ' '; }
  std::ofstream f(p, std::ios::binary);
  const std::uint64_t len = hdr.size();
  f.write(reinterpret_cast<const char*>(&len), sizeof(len));
  f.write(hdr.data(), (std::streamsize)hdr.size());
  const std::vector<float> zeros(off, 0.0f);
  f.write(reinterpret_cast<const char*>(zeros.data()),
          (std::streamsize)(zeros.size() * sizeof(float)));
}

}  // namespace

// VOSR IS NOT CLAIMED STREAMABLE, because it cannot stream.
//
// Its checkpoint names its blocks `blocks.N.`, which the image stage's
// single-stack stem matches, so a floor measured from the names says
// "trunk plus two blocks" -- 417 MB for VOSR 2.0 -- while
// MetalVosrTransformer copies all 36 blocks in, ~2659 MB. A floor is a
// promise, and that one would admit graphs the box could not hold. The
// published `ema_model.safetensors` escaped by accident (the directory
// opens nothing, so the floor measured 0 -- see the real-checkpoint arm
// below); a checkpoint named `model.safetensors`, which the loader also
// accepts, did not. Both ledgers must read "cannot be reduced": a claim
// floor of zero and a holding floor of zero, the holding at full size.
//
// No model needed: a two-block VOSR-shaped root in a temp directory is
// enough, and the test first proves the stem WOULD have matched it.
TEST(image_memory_plan, vosr_is_not_claimed_streamable)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-vosr-floor";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root / "checkpoints", ec);
  {
    std::ofstream a(root / "args.json");
    a << "{\"ae_type\": \"qwen\", \"distill_type\": \"onestep\"}";
  }
  // Named model.safetensors so the tensor table opens: open_model() does
  // not open a directory holding only a freely-named file (the published
  // `ema_model.safetensors`), which is why the real checkpoint measures a
  // floor of 0 today by accident rather than by declaration.
  write_safetensors_(root / "checkpoints" / "model.safetensors",
                     {{"blocks.0.attn.weight", 4096},
                      {"blocks.1.attn.weight", 4096},
                      {"x_embedder.weight", 1024}});
  const std::string dit = (root / "checkpoints").string();
  const std::size_t whole = model_memory::dir_weights_bytes(dit);
  ASSERT_TRUE(whole > 0);
  // Not vacuous: measured by names alone, this checkpoint has a floor.
  const std::vector<std::string_view> stems = {"blocks."};
  EXPECT_TRUE(model_memory::streaming_floor_bytes(dit, stems) > 0);

  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(root.string()));
  GenerateImageStage stage(&sess, "restore", std::vector<InEdge>{},
                           std::move(cfg));

  bool claimed = false;
  std::size_t claim_floor = 1;
  for (const ResourceClaim& c : stage.declare_resources()) {
    if (c.key == dit) { claimed = true; claim_floor = c.floor_bytes; }
  }
  EXPECT_TRUE(claimed);
  EXPECT_TRUE(claim_floor == 0);

  bool held = false;
  std::size_t plan_floor = 1, plan_preload = 0;
  for (const auto& h : stage.declare_memory().holdings) {
    if (h.source == dit) {
      held = true;
      plan_floor = h.floor;
      plan_preload = h.preload;
    }
  }
  EXPECT_TRUE(held);
  EXPECT_TRUE(plan_floor == 0);
  EXPECT_TRUE(plan_preload == whole);
  std::printf("[image_memory_plan] VOSR-shaped root: %zu B on disk, claim "
              "floor %zu, holding floor %zu\n", whole, claim_floor,
              plan_floor);
  fs::remove_all(root, ec);
}

// The same question on the REAL checkpoint, printed rather than assumed.
// VPIPE_VOSR_TEST_MODEL_PATH = the VOSR root (holding VOSR2/).
TEST(image_memory_plan, vosr_real_checkpoint_floor)
{
  const char* root = std::getenv("VPIPE_VOSR_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  namespace fs = std::filesystem;
  std::string vroot = root;
  if (fs::is_directory(fs::path(root) / "VOSR2")) {
    vroot = (fs::path(root) / "VOSR2").string();
  }
  const std::string dit = genai::MetalVosrTransformer::weights_dir(vroot);
  ASSERT_TRUE(!dit.empty());
  if (dit.empty()) { return; }
  const std::vector<std::string_view> stems = {"blocks."};
  const std::size_t by_names = model_memory::streaming_floor_bytes(dit, stems);
  const std::size_t whole = model_memory::dir_weights_bytes(dit);

  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(vroot));
  GenerateImageStage stage(&sess, "restore", std::vector<InEdge>{},
                           std::move(cfg));
  std::size_t claim_floor = 1, plan_floor = 1;
  for (const ResourceClaim& c : stage.declare_resources()) {
    if (c.key == dit) { claim_floor = c.floor_bytes; }
  }
  for (const auto& h : stage.declare_memory().holdings) {
    if (h.source == dit) { plan_floor = h.floor; }
  }
  std::printf("[image_memory_plan] real VOSR %s: %zu MB on disk, floor by "
              "block names %zu MB, declared claim floor %zu, holding floor "
              "%zu\n", dit.c_str(), whole >> 20, by_names >> 20, claim_floor,
              plan_floor);
  EXPECT_TRUE(claim_floor == 0);
  EXPECT_TRUE(plan_floor == 0);
}


// Does a real Krea-2 graph get its ANE module, box by box?
//
// The claims come from the actual stage at 1024x1024 and go through the
// actual registered planners, with VPIPE_RAM_LIMIT_MB standing in for the
// box. The generate-image stage's claims are most of the graph -- the DiT
// floor, the text encoder, the decode arena and the ANE unit; a VAE
// decode stage would add its ~250 MB of weights -- so this is a slight
// UNDER-estimate of the peak, and a refusal here is a refusal for real.
TEST(image_memory_plan, ane_module_grant_by_box)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  for (int gb : {16, 24, 32, 64}) {
    ::setenv("VPIPE_RAM_LIMIT_MB", std::to_string(gb * 1024).c_str(), 1);
    Session sess;
    auto* mgr = sess.generative_model_manager();
    if (mgr == nullptr) { break; }
    mgr->clear_declarations();
    mgr->clear_scratch();
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("hf_dir", FlexData::make_string(root));
    cfg.as_object().insert("ane_ffn", FlexData::make_bool(true));
    cfg.as_object().insert("width", FlexData::make_int(1024));
    cfg.as_object().insert("height", FlexData::make_int(1024));
    GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{},
                             std::move(cfg));
    const std::vector<ResourceClaim> claims = stage.declare_resources();
    const auto planners = ResourcePlannerRegistry::get().all();
    for (auto* p : planners) { p->begin_plan(&sess); }
    mgr->set_phase_order({std::string(model_memory::kPhaseCondition),
                          std::string(model_memory::kPhaseDenoise),
                          std::string(model_memory::kPhaseDecodeAudio),
                          std::string(model_memory::kPhaseDecode)});
    std::string label;
    std::size_t unit = 0;
    for (const ResourceClaim& c : claims) {
      if (auto* p = ResourcePlannerRegistry::get().find(c.kind)) {
        p->claim(&sess, c.key, c.phase, c.last_phase, c.floor_bytes);
      }
      if (c.kind == model_memory::kCoreMLKind) {
        const std::size_t b2 = c.key.rfind('|');
        const std::size_t b1 = c.key.rfind('|', b2 - 1);
        label = c.key.substr(0, b1);
        unit = (std::size_t)std::stoull(c.key.substr(b1 + 1, b2 - b1 - 1));
      }
    }
    const std::size_t peak = mgr->phase_peak();
    const std::size_t flat = mgr->phase_footprint(std::string()) +
                             mgr->scratch_bytes(std::string());
    for (auto* p : planners) { p->end_plan(&sess); }
    const int granted = model_memory::coreml_grant(&sess, label, unit, 1);
    std::vector<std::pair<std::string, std::size_t>> by;
    const std::size_t peak_after = mgr->phase_peak(&by);
    std::string phases;
    for (const auto& ph : by) {
      phases += " " + ph.first + "=" + std::to_string(ph.second >> 20);
    }
    std::printf("[image_memory_plan] %2d GB box: module %zu MB -> %s | "
                "per phase after grant%s | peak %zu -> %zu MB (full-size "
                "flat %zu)\n", gb,
                unit >> 20, granted > 0 ? "GRANTED" : "refused",
                phases.c_str(), peak >> 20, peak_after >> 20, flat >> 20);
    // Granted on every box, 16 GB included: the module lives in the
    // denoise, which stays under the decode's peak.
    EXPECT_TRUE(granted == 1);
    EXPECT_TRUE(peak_after == peak);
    mgr->clear_scratch();
    mgr->clear_declarations();
  }
  ::unsetenv("VPIPE_RAM_LIMIT_MB");
}

// A REGISTERED FAMILY THAT CLAIMS THE ENCODER ANSWERS FOR IT.
//
// generate-image books the conditioning encoder (`mllm/` or
// `text_encoder/`) on the conditioner's behalf, unphased -- the safe
// default, since a hole in the ledger reads as room. But a family whose
// conditioner is its OWN stage claims the encoder itself, phased to
// `condition` when that stage lets it go after the prompt, and the
// stage's unphased claim then booked it through the denoise anyway:
// MEASURED with a plugin family, a released 32 GB MLLM still pushed its
// 12 GB DiT into streaming on a 64 GB box. Both halves: a family that claims
// the encoder suppresses the stage's claim (in both ledgers), and one
// that does not still gets it.
namespace {

class EncoderClaimingFamily final : public genai::ImageModelFamily {
public:
  EncoderClaimingFamily(std::string tag, std::string root, bool claim_enc)
      : _tag(std::move(tag)), _root(std::move(root)), _claim(claim_enc) {}
  std::string_view tag() const noexcept override { return _tag; }
  bool claims(const std::string& root, const std::string&) const override
  {
    return root == _root;
  }
  std::vector<ResourceClaim>
  declare_resources(const std::string& root) const override
  {
    namespace fs = std::filesystem;
    std::vector<ResourceClaim> out = model_memory::weight_claims(
        {(fs::path(root) / "transformer").string()});
    if (_claim) {
      for (auto& c : model_memory::weight_claims_in_phase(
               {(fs::path(root) / "mllm").string()},
               model_memory::kPhaseCondition)) {
        out.push_back(std::move(c));
      }
    }
    return out;
  }
  std::unique_ptr<genai::ImageGenerator>
  load(const genai::ImageModelCreateArgs&) override
  {
    return nullptr;
  }

private:
  std::string _tag, _root;
  bool _claim;
};

}  // namespace

TEST(image_memory_plan, a_family_that_claims_the_encoder_answers_for_it)
{
  namespace fs = std::filesystem;
  Session sess;
  for (const bool claim : {true, false}) {
    const fs::path root = fs::temp_directory_path() /
        ("vpipe-encclaim-" + std::to_string((long)::getpid()) +
         (claim ? "-a" : "-b"));
    std::error_code ec;
    fs::create_directories(root / "mllm", ec);
    fs::create_directories(root / "transformer", ec);
    // Weight bytes for the encoder to weigh: a zero-byte holding is not
    // a holding at all, and would pass the second half vacuously.
    std::ofstream(root / "mllm" / "model.safetensors")
        << std::string(4096, 'x');
    const std::string enc = (root / "mllm").string();
    static int n = 0;
    ASSERT_TRUE(genai::ImageModelRegistry::get().add(
        std::make_unique<EncoderClaimingFamily>(
            "test-encclaim-" + std::to_string(++n), root.string(), claim)));
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("hf_dir", FlexData::make_string(root.string()));
    GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{},
                             std::move(cfg));
    int unphased = 0, phased = 0;
    bool arena = false;
    for (const ResourceClaim& c : stage.declare_resources()) {
      // A registered family's decode arena is declared too: without it
      // vae-decode's per-beat revision has no label to correct.
      if (c.kind == model_memory::kScratchKind &&
          c.key.rfind("vae-decode", 0) == 0) {
        arena = true;
      }
      if (c.key != enc) { continue; }
      if (c.phase.empty()) { ++unphased; } else { ++phased; }
    }
    EXPECT_TRUE(arena);
    bool held = false;
    for (const auto& h : stage.declare_memory().holdings) {
      if (h.source == enc) { held = true; }
    }
    if (claim) {
      // The family's condition-phase claim, and nothing unphased.
      EXPECT_TRUE(phased == 1 && unphased == 0);
      EXPECT_FALSE(held);
    } else {
      EXPECT_TRUE(unphased == 1 && phased == 0);
      EXPECT_TRUE(held);
    }
    fs::remove_all(root, ec);
  }
}

// A REGISTERED FAMILY'S CoreML CLAIM IS THE STAGE'S TO BOOK.
//
// A family implementing `ane_ffn` claims its ANE module unconditionally
// (it cannot see the graph's accel bag at plan time) under a label of
// its own. generate-image passed that through as-is: booked even when
// the graph did not ask for the tier -- peers then sized against a
// module that never exists -- and under a label two stages would share.
// Now it is kept only with `ane_ffn`, relabelled to the stage, as
// generate-video does.
namespace {

class CoreMLClaimingFamily final : public genai::ImageModelFamily {
public:
  CoreMLClaimingFamily(std::string tag, std::string root)
      : _tag(std::move(tag)), _root(std::move(root)) {}
  std::string_view tag() const noexcept override { return _tag; }
  bool claims(const std::string& root, const std::string&) const override
  {
    return root == _root;
  }
  std::vector<ResourceClaim>
  declare_resources(const std::string&) const override
  {
    return model_memory::coreml_claims("family-own-label", 640u << 20, 1,
                                       model_memory::kPhaseDenoise);
  }
  std::unique_ptr<genai::ImageGenerator>
  load(const genai::ImageModelCreateArgs&) override
  {
    return nullptr;
  }

private:
  std::string _tag, _root;
};

}  // namespace

TEST(image_memory_plan, a_family_coreml_claim_follows_the_stage)
{
  namespace fs = std::filesystem;
  Session sess;
  const fs::path root = fs::temp_directory_path() /
      ("vpipe-coreml-" + std::to_string((long)::getpid()));
  std::error_code ec;
  fs::create_directories(root / "transformer", ec);
  static int n = 0;
  ASSERT_TRUE(genai::ImageModelRegistry::get().add(
      std::make_unique<CoreMLClaimingFamily>(
          "test-coreml-" + std::to_string(++n), root.string())));
  for (const bool ane : {false, true}) {
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("hf_dir", FlexData::make_string(root.string()));
    if (ane) { cfg.as_object().insert("ane_ffn", FlexData::make_bool(true)); }
    GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{},
                             std::move(cfg));
    int coreml = 0;
    std::string label;
    for (const ResourceClaim& c : stage.declare_resources()) {
      std::string l;
      std::size_t ub = 0;
      int un = 0;
      if (model_memory::parse_coreml_claim(c, &l, &ub, &un)) {
        ++coreml;
        label = l;
      }
    }
    if (ane) {
      EXPECT_TRUE(coreml == 1);
      EXPECT_TRUE(label == "ane-ffn/t2i");
    } else {
      EXPECT_TRUE(coreml == 0);
    }
  }
  fs::remove_all(root, ec);
}
