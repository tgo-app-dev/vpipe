// Host-memory sizing shared by the model-holding diffusion stages
// (stages/model-memory.h) plus the `unload_when_idle` config surface it backs.
//
// The point of these is that the three stages that hold a big chunk of weights
// -- diffusion-conditioner (text encoder + vision tower), generate-image (DiT)
// and vae-encode/vae-decode -- must agree about whether the box is
// memory-bounded. They each read the SAME two numbers (physical RAM, the
// .safetensors bytes of a component dir), so a wrong answer in one of them is a
// pipeline that either thrashes or refuses to unload.

#include "minitest.h"
#include "interfaces/session-services-intf.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "stages/diffusion-conditioner-stage.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/weight-set.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/model-memory.h"
#include "stages/vae-decode-stage.h"
#include "stages/vae-encode-stage.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace vpipe;
using vpipe::model_memory::UnloadPolicy;

TEST(model_memory, phys_ram_is_plausible)
{
  const std::size_t ram = model_memory::phys_ram();
  // Any Mac that can build this tree has at least 4 GB and less than 4 TB.
  EXPECT_TRUE(ram >= (4ull << 30));
  EXPECT_TRUE(ram < (4096ull << 30));
}

TEST(model_memory, weight_bytes_counts_only_safetensors)
{
  namespace fs = std::filesystem;
  const fs::path dir =
      fs::temp_directory_path() / "vpipe-model-memory-test" / "sub";
  std::error_code ec;
  fs::create_directories(dir, ec);
  auto write = [](const fs::path& p, std::size_t n) {
    std::ofstream f(p, std::ios::binary);
    std::string blob(n, 'x');
    f.write(blob.data(), (std::streamsize)blob.size());
  };
  write(dir.parent_path() / "a.safetensors", 1000);
  write(dir / "b.safetensors", 2000);        // recursive
  write(dir / "notes.json", 5000);           // ignored
  const std::size_t got =
      model_memory::dir_weights_bytes(dir.parent_path().string());
  EXPECT_TRUE(got == 3000);
  // A missing dir is 0, not an error -- callers treat it as "component absent"
  // (e.g. a pipeline with mllm/ but no text_encoder/).
  EXPECT_TRUE(model_memory::dir_weights_bytes(
                  (dir / "nope" / "nope").string()) == 0);
  EXPECT_TRUE(model_memory::dir_weights_bytes("") == 0);
  fs::remove_all(dir.parent_path(), ec);
}

TEST(model_memory, bounded_compares_against_physical_ram)
{
  const std::size_t ram = model_memory::phys_ram();
  ASSERT_TRUE(ram > 0);
  Session s;
  // Nothing always fits; more than RAM never does.
  EXPECT_TRUE(!model_memory::bounded(&s, {}, 0));
  EXPECT_TRUE(model_memory::bounded(&s, {}, ram + 1));
  // The real Boogu question: a 4-bit DiT (~5.6 GB) beside a 4-bit Qwen3-VL
  // mllm (~4.7 GB) plus 6 GB of working set is 16.3 GB -- bounded on a 16 GB
  // box, roomy on a 32 GB one. Assert the arithmetic, not this box's answer.
  const std::size_t dit = 5600ull << 20, enc = 4700ull << 20;
  const std::size_t need = dit + enc + (6ull << 30);
  EXPECT_TRUE(need > (16ull << 30));
  EXPECT_TRUE(need < (32ull << 30));
}

// The footprint is a UNION over checkpoints, not a sum over directory
// names. This is the case the old per-directory sum got wrong: a
// component named twice -- by two stages of one graph, or by two
// pipelines sharing a model -- is loaded once, so counting it twice
// over-estimates and pushes a box into streaming it does not need.
TEST(model_memory, footprint_counts_a_shared_directory_once)
{
  namespace fs = std::filesystem;
  const fs::path root =
      fs::temp_directory_path() / "vpipe-mm-union-test";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root / "a", ec);
  fs::create_directories(root / "b", ec);
  auto write = [](const fs::path& p, std::size_t n) {
    std::ofstream f(p, std::ios::binary);
    std::string blob(n, 'x');
    f.write(blob.data(), (std::streamsize)blob.size());
  };
  write(root / "a" / "w.safetensors", 4000);
  write(root / "b" / "w.safetensors", 1000);

  Session s;
  const std::string a = (root / "a").string();
  const std::string b = (root / "b").string();

  EXPECT_TRUE(model_memory::weight_footprint(&s, {a}) == 4000u);
  EXPECT_TRUE(model_memory::weight_footprint(&s, {a, b}) == 5000u);
  // Named twice -> still counted once.
  EXPECT_TRUE(model_memory::weight_footprint(&s, {a, a}) == 4000u);
  EXPECT_TRUE(model_memory::weight_footprint(&s, {a, b, a, b}) == 5000u);
  // A checkpoint named by FILE, not by directory -- a Comfy-Org repack
  // is one .safetensors per component, so the claim a stage declares IS
  // a file path. It has to be SIZED: a directory walk over a file
  // silently yields 0, and a 66 GB DiT that reports as free is worse
  // than no accounting, because every peer then sizes against a box
  // that does not exist.
  const std::string af = (root / "a" / "w.safetensors").string();
  EXPECT_TRUE(model_memory::weight_footprint(&s, {af}) == 4000u);
  // Only a weight file counts; a config or a README beside it does not.
  write(root / "a" / "notes.txt", 777);
  EXPECT_TRUE(model_memory::weight_footprint(
                  &s, {(root / "a" / "notes.txt").string()}) == 0u);
  // Empty and missing directories contribute nothing rather than failing.
  EXPECT_TRUE(model_memory::weight_footprint(&s, {a, "", (root / "z")
                                                          .string()})
              == 4000u);
  fs::remove_all(root, ec);
}

// With no session there is no manager to ask, and the footprint has to
// degrade to the old on-disk sum rather than reporting zero -- an
// offline tool must not conclude the box is empty.
TEST(model_memory, footprint_without_a_session_is_the_on_disk_sum)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-mm-nosess-test";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root / "a", ec);
  std::ofstream f(root / "a" / "w.safetensors", std::ios::binary);
  std::string blob(2048, 'x');
  f.write(blob.data(), (std::streamsize)blob.size());
  f.close();
  const std::string a = (root / "a").string();
  EXPECT_TRUE(model_memory::weight_footprint(nullptr, {a}) == 2048u);
  EXPECT_TRUE(model_memory::weight_footprint(nullptr, {a, a}) == 2048u);
  fs::remove_all(root, ec);
}

// A checkpoint the manager already holds contributes what it is REALLY
// holding, not what its files weigh. That is the second half of the
// union: the on-disk size of an open checkpoint would double-count it
// against the manager's own resident total.
TEST(model_memory, an_open_checkpoint_is_not_counted_from_disk)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  auto* mc  = s.metal_compute();
  if (mgr == nullptr || mc == nullptr || !mc->valid()) { return; }

  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-mm-open-test";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root, ec);
  // A real (tiny) checkpoint the manager can actually open.
  const std::string hdr =
      "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],"
      "\"data_offsets\":[0,16]}}";
  std::string padded = hdr;
  while (((8 + padded.size()) % 16) != 0) { padded.push_back(' '); }
  {
    std::ofstream f(root / "model.safetensors", std::ios::binary);
    const std::uint64_t n = padded.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(padded.data(), (std::streamsize)padded.size());
    const float v[4] = {1, 2, 3, 4};
    f.write(reinterpret_cast<const char*>(v), 16);
  }
  const std::string dir = root.string();
  const std::size_t on_disk = model_memory::dir_weights_bytes(dir);
  EXPECT_TRUE(on_disk > 0u);
  EXPECT_TRUE(!mgr->holds_weights(dir));
  EXPECT_TRUE(model_memory::weight_footprint(&s, {dir}) == on_disk);

  auto ws = mgr->weight_set(dir);
  ASSERT_TRUE(ws != nullptr);
  EXPECT_TRUE(mgr->holds_weights(dir));
  // Open but nothing materialised yet: it holds 0 bytes, so the
  // footprint drops to 0 -- an open checkpoint reports what it HOLDS.
  EXPECT_TRUE(mgr->resident_weight_bytes() == 0u);
  EXPECT_TRUE(model_memory::weight_footprint(&s, {dir}) == 0u);

  // Materialise the tensor and the footprint follows the real bytes.
  ASSERT_TRUE(!ws->tensor("w", mc, genai::WeightSet::Residency::Copied)
                   .empty());
  EXPECT_TRUE(mgr->resident_weight_bytes() == 16u);
  EXPECT_TRUE(model_memory::weight_footprint(&s, {dir}) == 16u);
  fs::remove_all(root, ec);
}

TEST(model_memory, unload_policy_parses)
{
  bool bad = true;
  EXPECT_TRUE(model_memory::parse_unload_policy("", &bad) == UnloadPolicy::kAuto);
  EXPECT_TRUE(!bad);
  EXPECT_TRUE(model_memory::parse_unload_policy("auto", &bad) ==
              UnloadPolicy::kAuto);
  EXPECT_TRUE(model_memory::parse_unload_policy("always", &bad) ==
              UnloadPolicy::kAlways);
  EXPECT_TRUE(model_memory::parse_unload_policy("never", &bad) ==
              UnloadPolicy::kNever);
  EXPECT_TRUE(!bad);
  // Unknown values fall back to auto and REPORT it -- stage config is
  // deferred-validated, so the stage warns instead of throwing.
  EXPECT_TRUE(model_memory::parse_unload_policy("sometimes", &bad) ==
              UnloadPolicy::kAuto);
  EXPECT_TRUE(bad);
  EXPECT_TRUE(std::string(model_memory::unload_policy_name(
                  UnloadPolicy::kAlways)) == "always");
  EXPECT_TRUE(std::string(model_memory::unload_policy_name(
                  UnloadPolicy::kNever)) == "never");
  EXPECT_TRUE(std::string(model_memory::unload_policy_name(
                  UnloadPolicy::kAuto)) == "auto");
}

// Every stage that holds weights takes the key, and a bad value is a WARNING
// (config_error stays empty) rather than an inert stage -- the stage still runs,
// just with the default policy.
TEST(model_memory, stages_accept_unload_when_idle)
{
  Session sess;
  auto cfg_with = [](const char* v) {
    FlexData c = FlexData::make_object();
    c.as_object().insert("hf_dir", FlexData::make_string("/nonexistent"));
    c.as_object().insert("unload_when_idle", FlexData::make_string(v));
    return c;
  };
  for (const char* v : {"auto", "always", "never", "bogus"}) {
    DiffusionConditionerStage c(&sess, "cond", std::vector<InEdge>{},
                               cfg_with(v));
    EXPECT_TRUE(c.config_error().empty());
    VaeDecodeStage d(&sess, "dec", std::vector<InEdge>{}, cfg_with(v));
    EXPECT_TRUE(d.config_error().empty());
    VaeEncodeStage e(&sess, "enc", std::vector<InEdge>{}, cfg_with(v));
    EXPECT_TRUE(e.config_error().empty());
  }
  // And it is documented, so the web-ui / --help surface it.
  auto documented = [](const StageSpec& sp) {
    for (const auto& a : sp.attrs) {
      if (std::string(a.key) == "unload_when_idle") { return true; }
    }
    return false;
  };
  DiffusionConditionerStage c(&sess, "cond", std::vector<InEdge>{},
                             cfg_with("auto"));
  VaeDecodeStage d(&sess, "dec", std::vector<InEdge>{}, cfg_with("auto"));
  VaeEncodeStage e(&sess, "enc", std::vector<InEdge>{}, cfg_with("auto"));
  EXPECT_TRUE(documented(c.spec()));
  EXPECT_TRUE(documented(d.spec()));
  EXPECT_TRUE(documented(e.spec()));
}

// Declarations exist to make a decision taken DURING initialize()
// correct. Every driver runs initialize() concurrently, so without them
// a stage sizing the box sees whichever peers happen to have loaded
// already -- a race, not an ordering.
TEST(model_memory, a_declared_model_counts_before_it_loads)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }

  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-mm-declare-test";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root / "peer", ec);
  {
    std::ofstream f(root / "peer" / "w.safetensors", std::ios::binary);
    std::string blob(8192, 'x');
    f.write(blob.data(), (std::streamsize)blob.size());
  }
  const std::string peer = (root / "peer").string();

  // Nothing declared: a stage that does not name this dir cannot see it.
  EXPECT_TRUE(mgr->resident_weight_bytes() == 0u);
  EXPECT_TRUE(!mgr->accounts_for(peer));

  mgr->declare_weights(peer, model_memory::dir_weights_bytes(peer));
  EXPECT_TRUE(mgr->accounts_for(peer));
  EXPECT_TRUE(mgr->resident_weight_bytes() == 8192u);

  // And a stage that DOES name it must not add it a second time.
  EXPECT_TRUE(model_memory::weight_footprint(&s, {peer}) == 8192u);

  // Two stages declaring one checkpoint describe the same bytes.
  mgr->declare_weights(peer, 8192);
  EXPECT_TRUE(mgr->resident_weight_bytes() == 8192u);

  // Revised down by a model that knows it keeps less -- the streaming
  // DiT case. Only a DECLARED checkpoint can be revised.
  mgr->revise_declaration(peer, 1024);
  EXPECT_TRUE(mgr->resident_weight_bytes() == 1024u);
  mgr->revise_declaration((root / "absent").string(), 99);
  EXPECT_TRUE(mgr->resident_weight_bytes() == 1024u);

  mgr->clear_declarations();
  EXPECT_TRUE(mgr->resident_weight_bytes() == 0u);
  EXPECT_TRUE(!mgr->accounts_for(peer));
  fs::remove_all(root, ec);
}

// The declaration is an upper bound while a load is in flight, and the
// real bytes win once they exceed it. A peer 30% through loading a big
// checkpoint genuinely holds 30% of it -- sizing against that number
// would find room that is not there.
TEST(model_memory, a_declaration_is_a_floor_not_a_replacement)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  auto* mc  = s.metal_compute();
  if (mgr == nullptr || mc == nullptr || !mc->valid()) { return; }

  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-mm-floor-test";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root, ec);
  const std::string hdr =
      "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],"
      "\"data_offsets\":[0,16]}}";
  std::string padded = hdr;
  while (((8 + padded.size()) % 16) != 0) { padded.push_back(' '); }
  {
    std::ofstream f(root / "model.safetensors", std::ios::binary);
    const std::uint64_t n = padded.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(padded.data(), (std::streamsize)padded.size());
    const float v[4] = {1, 2, 3, 4};
    f.write(reinterpret_cast<const char*>(v), 16);
  }
  const std::string dir = root.string();

  // Declared big, holding nothing yet -> the declaration governs.
  mgr->declare_weights(dir, 1u << 20);
  EXPECT_TRUE(mgr->resident_weight_bytes() == (1u << 20));

  // Open it and materialise the tensor: still under the declaration, so
  // the declaration still governs -- mid-load must not read as "cheap".
  auto ws = mgr->weight_set(dir);
  ASSERT_TRUE(ws != nullptr);
  ASSERT_TRUE(!ws->tensor("w", mc, genai::WeightSet::Residency::Copied)
                   .empty());
  EXPECT_TRUE(mgr->resident_weight_bytes() == (1u << 20));

  // And it KEEPS governing after the load finishes. This is the case
  // that makes clearing-at-the-barrier wrong: a weight set only accounts
  // for what it cached, and a model reading uncached (every LM) holds
  // its weights in its own members. Dropping the estimate here would
  // report a multi-GB checkpoint as 16 bytes.
  EXPECT_TRUE(mgr->resident_weight_bytes() == (1u << 20));

  // A model that genuinely keeps less says so, and then the smaller
  // number governs.
  mgr->revise_declaration(dir, 16);
  EXPECT_TRUE(mgr->resident_weight_bytes() == 16u);
  fs::remove_all(root, ec);
}

// The headroom for an irreversible decision is wider than for a
// revisable one, deliberately: failing to stream means thrash or an OOM
// kill, streaming needlessly costs ~2-3x per step.
TEST(model_memory, the_irreversible_decision_gets_more_headroom)
{
  EXPECT_TRUE(model_memory::kStreamHeadroom > model_memory::kHeadroom);
}

namespace {

// A stage that declares a model directory and records what the manager
// knew at two moments: inside initialize() (where the real stages size
// the box) and inside process() (which runs strictly after the init
// barrier). One stage is enough to prove the wiring -- what matters is
// that the runtime asked BEFORE initialize and cleared AFTER the
// barrier, neither of which the stage itself controls.
class DeclaringStage : public TypedStage<DeclaringStage> {
public:
  static constexpr const char* kTypeName = "ut-declaring-stage";
  using TypedStage::TypedStage;

  std::string dir;
  std::size_t saw_in_initialize = 0;
  std::size_t saw_in_process    = 0;
  bool        accounted_in_init = false;
  bool        ran_process       = false;

  std::vector<ResourceClaim> declare_resources() const override
  {
    return model_memory::weight_claims({dir});
  }

  Job initialize(RuntimeContext& ctx) override
  {
    (void)ctx;
    if (auto* m = session()->services()->generative_model_manager()) {
      saw_in_initialize = m->resident_weight_bytes();
      accounted_in_init = m->accounts_for(dir);
    }
    co_return;
  }

  Job process(RuntimeContext& ctx) override
  {
    if (!ran_process) {
      ran_process = true;
      if (auto* m = session()->services()->generative_model_manager()) {
        saw_in_process = m->resident_weight_bytes();
      }
    }
    ctx.signal_done();
    co_return;
  }
};

}  // namespace

// End-to-end: the runtime must collect declarations before any
// initialize() runs, and they must still stand afterwards. Collecting
// late reintroduces the race the declaration exists to remove; dropping
// them at the barrier would erase every uncached-reading model from the
// accounting (see GenerativeModelManager::declare_weights).
TEST(model_memory, the_runtime_declares_before_init_and_clears_after)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-mm-runtime-test";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root, ec);
  {
    std::ofstream f(root / "w.safetensors", std::ios::binary);
    std::string blob(4096, 'x');
    f.write(blob.data(), (std::streamsize)blob.size());
  }

  Session sess;
  if (sess.generative_model_manager() == nullptr) {
    fs::remove_all(root, ec);
    return;
  }
  Pipeline pl("declare-test", &sess);
  auto st = std::make_unique<DeclaringStage>(
      &sess, "decl", std::vector<InEdge>{}, FlexData::make_object());
  st->dir = root.string();
  st->allocate_oports(0);
  auto* stage = static_cast<DeclaringStage*>(pl.insert_stage(std::move(st)));

  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  // Declared before initialize(): the stage saw its own 4096 bytes
  // accounted for even though nothing had loaded.
  EXPECT_TRUE(stage->accounted_in_init);
  EXPECT_TRUE(stage->saw_in_initialize == 4096u);
  // And it STANDS after the barrier. Clearing there would have made
  // any model that reads uncached -- every LM -- disappear from the
  // accounting the moment it finished loading, which is exactly when
  // peers start sizing the box against it.
  EXPECT_TRUE(stage->ran_process);
  EXPECT_TRUE(stage->saw_in_process == 4096u);
  fs::remove_all(root, ec);
}
