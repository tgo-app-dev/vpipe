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
#include "interfaces/ui-delegate-intf.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "stages/diffusion-conditioner-stage.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/shared/wired-pool.h"
#include "generative-models/weight-set.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/resource-plan.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "stages/vae-decode-stage.h"
#include "stages/vae-encode-stage.h"

#include <cstdlib>
#include <functional>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
  // The canonical vocabulary is destroy/park/keep. "always" and "never"
  // are still ACCEPTED (existing pipeline JSON says them) but they are
  // aliases, so they no longer name themselves back.
  EXPECT_TRUE(model_memory::parse_unload_policy("destroy", &bad) ==
              UnloadPolicy::kDestroy);
  EXPECT_TRUE(model_memory::parse_unload_policy("park", &bad) ==
              UnloadPolicy::kPark);
  EXPECT_TRUE(model_memory::parse_unload_policy("keep", &bad) ==
              UnloadPolicy::kKeep);
  EXPECT_TRUE(!bad);
  // The legacy spellings are the SAME states, not neighbouring ones --
  // an alias that drifted would silently change what a shipped pipeline
  // does to its weights.
  EXPECT_TRUE(model_memory::parse_unload_policy("always", &bad) ==
              UnloadPolicy::kDestroy);
  EXPECT_TRUE(model_memory::parse_unload_policy("never", &bad) ==
              UnloadPolicy::kKeep);
  EXPECT_TRUE(std::string(model_memory::unload_policy_name(
                  UnloadPolicy::kDestroy)) == "destroy");
  EXPECT_TRUE(std::string(model_memory::unload_policy_name(
                  UnloadPolicy::kPark)) == "park");
  EXPECT_TRUE(std::string(model_memory::unload_policy_name(
                  UnloadPolicy::kKeep)) == "keep");
  EXPECT_TRUE(std::string(model_memory::unload_policy_name(
                  UnloadPolicy::kAuto)) == "auto");
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

// A claim on a directory that CONTAINS another claim is counted ONCE.
//
// dir_weights_bytes() is recursive but the manager keys declarations by
// exact path, so a stage claiming a repo ROOT and a peer claiming the
// DiT inside it described the same bytes twice. MEASURED on a Comfy-Org
// MiniMax-H3 repack: 230781 MB declared against a true 118452 MB, which
// every peer then sized itself against.
TEST(model_memory, a_claim_inside_another_claim_counts_once)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }

  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-mm-nested-test";
  std::error_code ec;
  fs::remove_all(root, ec);
  auto put = [&](const fs::path& rel, std::size_t n) {
    fs::create_directories((root / rel).parent_path(), ec);
    std::ofstream f(root / rel, std::ios::binary);
    std::string blob(n, 'x');
    f.write(blob.data(), (std::streamsize)blob.size());
  };
  // The repack shape: components in siblings, no weights at the root.
  put("diffusion_models/dit.safetensors", 6000);
  put("text_encoders/enc.safetensors", 3000);
  put("vae/vae.safetensors", 1000);

  const std::string r   = root.string();
  const std::string dit = (root / "diffusion_models").string();
  const std::string enc = (root / "text_encoders").string();

  // The root's own recursive scan already IS the sum of the three.
  EXPECT_TRUE(model_memory::dir_weights_bytes(r) == 10000u);

  mgr->declare_weights(r, model_memory::dir_weights_bytes(r));
  EXPECT_TRUE(mgr->resident_weight_bytes() == 10000u);

  // Two peers naming components inside it must not add their bytes again.
  mgr->declare_weights(dit, model_memory::dir_weights_bytes(dit));
  mgr->declare_weights(enc, model_memory::dir_weights_bytes(enc));
  EXPECT_TRUE(mgr->resident_weight_bytes() == 10000u);

  // The inner claim is the one kept, so a PHASE on it still applies --
  // folding it into an unphased root would hold the encoder through a
  // denoise it is not resident for.
  mgr->set_declaration_phase(enc,
                             std::string(model_memory::kPhaseCondition));
  EXPECT_TRUE(mgr->phase_footprint(
                  std::string(model_memory::kPhaseDenoise)) == 7000u);

  // Nested three deep -- root, component dir, and the file named inside
  // it, all legal claims on a repack. The file's bytes are inside the
  // component, which is inside the root, so subtracting it from BOTH
  // would report less than the checkpoint weighs.
  const std::string dit_file =
      (root / "diffusion_models" / "dit.safetensors").string();
  mgr->declare_weights(dit_file, model_memory::dir_weights_bytes(dit_file));
  EXPECT_TRUE(mgr->resident_weight_bytes() == 10000u);

  // A sibling that merely shares a prefix is NOT inside it.
  const fs::path other = fs::temp_directory_path() /
                         "vpipe-mm-nested-test-sibling";
  fs::remove_all(other, ec);
  fs::create_directories(other, ec);
  {
    std::ofstream f(other / "w.safetensors", std::ios::binary);
    std::string blob(500, 'x');
    f.write(blob.data(), (std::streamsize)blob.size());
  }
  mgr->declare_weights(other.string(),
                       model_memory::dir_weights_bytes(other.string()));
  EXPECT_TRUE(mgr->resident_weight_bytes() == 10500u);

  mgr->clear_declarations();
  fs::remove_all(root, ec);
  fs::remove_all(other, ec);
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

namespace {

// The two halves of a `model-select` graph, reduced to what the
// planning phase cares about.
//
// A constant source: no iports, one oport, a beat that is a pure
// function of configuration -- so it is knowable before anything runs.
class ConstModelSource : public TypedStage<ConstModelSource> {
public:
  static constexpr const char* kTypeName = "ut-const-model-source";
  using TypedStage::TypedStage;

  std::string dir;
  // The negative control's switch. With this false the stage behaves
  // exactly as every source did before folding existed: it emits the
  // same beat at run time and answers nothing before.
  bool        constant = true;
  bool        emitted  = false;

  FlexData beat() const
  {
    FlexData fd = FlexData::make_object();
    auto     o  = fd.as_object();
    o.insert_or_assign("hf_dir", FlexData::make_string(dir));
    return fd;
  }

  std::optional<FlexData> constant_output(unsigned oport) const override
  {
    if (oport != 0 || !constant) { return std::nullopt; }
    return beat();
  }

  Job process(RuntimeContext& ctx) override
  {
    if (emitted) { ctx.signal_done(); co_return; }
    emitted = true;
    co_await ctx.write(0, make_payload<FlexDataPayload>(beat()));
    ctx.signal_done();
  }
};

// The consumer: takes its model from an iport, as the diffusion stages
// take theirs, and records what it knew when the runtime asked it to
// declare. That question is the whole test -- a stage asked before the
// constant arrives has nothing to declare and is invisible to every
// peer sizing the box.
class LatchingDeclarer : public TypedStage<LatchingDeclarer> {
public:
  static constexpr const char* kTypeName = "ut-latching-declarer";
  using TypedStage::TypedStage;

  std::string         dir;              // empty until a constant lands
  mutable std::string dir_at_declare;
  mutable unsigned    declares          = 0;
  unsigned            applied           = 0;
  bool                accounted_in_init = false;

  void apply_constant(unsigned iport, const FlexData& b) override
  {
    if (iport != 0) { return; }
    ++applied;
    apply_model_select_beat(b, dir);
  }

  std::vector<ResourceClaim> declare_resources() const override
  {
    ++declares;
    dir_at_declare = dir;
    if (dir.empty()) { return {}; }     // the shipped stages' guard
    return model_memory::weight_claims({dir});
  }

  Job initialize(RuntimeContext& ctx) override
  {
    (void)ctx;
    if (auto* m = session()->services()->generative_model_manager()) {
      accounted_in_init = !dir.empty() && m->accounts_for(dir);
    }
    co_return;
  }

  Job process(RuntimeContext& ctx) override
  {
    auto t = co_await ctx.read(0);
    if (!t) { ctx.signal_done(); }
  }
};

struct FoldFixture {
  std::filesystem::path       root;
  std::unique_ptr<Session>    sess;
  std::unique_ptr<Pipeline>   pl;
  ConstModelSource*           src  = nullptr;
  LatchingDeclarer*           cons = nullptr;
};

// src -> consumer, with a real 4096-byte checkpoint on disk so the
// planner has something to size.
FoldFixture
build_fold_graph(const char* tag, bool constant)
{
  namespace fs = std::filesystem;
  FoldFixture f;
  f.root = fs::temp_directory_path() / (std::string("vpipe-fold-") + tag);
  std::error_code ec;
  fs::remove_all(f.root, ec);
  fs::create_directories(f.root, ec);
  {
    std::ofstream o(f.root / "w.safetensors", std::ios::binary);
    std::string   blob(4096, 'x');
    o.write(blob.data(), (std::streamsize)blob.size());
  }
  f.sess = std::make_unique<Session>();
  f.pl   = std::make_unique<Pipeline>(tag, f.sess.get());

  auto s = std::make_unique<ConstModelSource>(
      f.sess.get(), "sel", std::vector<InEdge>{}, FlexData::make_object());
  s->dir      = f.root.string();
  s->constant = constant;
  s->allocate_oports(1);
  f.src = static_cast<ConstModelSource*>(f.pl->insert_stage(std::move(s)));

  auto c = std::make_unique<LatchingDeclarer>(
      f.sess.get(), "dit", std::vector<InEdge>{{f.src, 0}},
      FlexData::make_object());
  c->allocate_oports(0);
  f.cons = static_cast<LatchingDeclarer*>(f.pl->insert_stage(std::move(c)));
  return f;
}

}  // namespace

// The point of the whole mechanism: a graph that names its model once,
// in a source, must still have that model on the manager's books before
// any stage initializes.
//
// Before folding this failed on every shipped video pipeline. The four
// model-holding stages of docs/pipelines/minimax-h3-text-to-video.vpipeline
// carry no hf_dir of their own -- `model-select` holds the only copy --
// so each of them hit its `if (_hf_dir.empty()) { return {}; }` guard and
// declared NOTHING. The planning phase ran, bracketed every planner, and
// put zero bytes on the books, which is the sizing race it exists to
// prevent.
TEST(model_memory, a_constant_source_reaches_the_planning_phase)
{
  namespace fs = std::filesystem;
  FoldFixture f = build_fold_graph("reaches", /*constant=*/true);
  std::error_code ec;
  if (f.sess->generative_model_manager() == nullptr) {
    fs::remove_all(f.root, ec);
    return;
  }
  PipelineRuntime rt(f.pl.get(), f.sess.get());
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  // Delivered exactly once, and BEFORE the runtime asked what this
  // stage intended to acquire.
  EXPECT_TRUE(f.cons->applied == 1u);
  EXPECT_TRUE(f.cons->declares >= 1u);
  EXPECT_TRUE(f.cons->dir_at_declare == f.root.string());
  // ...and the claim it could then make actually landed on the books,
  // which is what a peer sizing the box reads.
  EXPECT_TRUE(f.cons->accounted_in_init);
  fs::remove_all(f.root, ec);
}

// The negative control, and the reason the test above means anything.
//
// Same graph, same beat at run time, same everything -- except the
// source declines to answer constant_output. This is the pre-folding
// behaviour, and it must still produce an empty declaration: if this
// passed too, the assertions above would be measuring the beat that
// arrives in process(), not the fold.
TEST(model_memory, without_a_constant_the_planning_phase_sees_nothing)
{
  namespace fs = std::filesystem;
  FoldFixture f = build_fold_graph("nofold", /*constant=*/false);
  std::error_code ec;
  if (f.sess->generative_model_manager() == nullptr) {
    fs::remove_all(f.root, ec);
    return;
  }
  PipelineRuntime rt(f.pl.get(), f.sess.get());
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(f.cons->applied == 0u);
  EXPECT_TRUE(f.cons->dir_at_declare.empty());
  EXPECT_TRUE(!f.cons->accounted_in_init);
  // The beat still arrives at run time -- folding is an analysis, not a
  // rewrite, so the data path is identical either way.
  EXPECT_TRUE(f.src->emitted);
  fs::remove_all(f.root, ec);
}

namespace {

// A checkpoint of a known size, so the phase arithmetic below is
// exact rather than approximate.
std::string
make_ckpt(const std::filesystem::path& p, std::size_t bytes)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories(p, ec);
  std::ofstream f(p / "w.safetensors", std::ios::binary);
  std::string   blob(bytes, 'x');
  f.write(blob.data(), (std::streamsize)blob.size());
  return p.string();
}

// Records every warning, so the release audit can be asserted on rather
// than described. It is the UI delegate and not the log delegate
// because Session::warn routes there -- a warning is something the user
// is meant to see.
struct CapturingUi : public UiDelegateIntf {
  std::vector<std::string> lines;
  void error(const VpipeFormat& f) override { lines.push_back(f()); }
  void warn (const VpipeFormat& f) override { lines.push_back(f()); }
  void info (const VpipeFormat&) override {}
  UiInputStatus getline(const VpipeFormat&, std::string&,
                        const std::function<bool()>&) override
  {
    return UiInputStatus::Eof;
  }
  std::unique_ptr<UiTextStream> open_text_stream() override
  {
    return std::make_unique<NullUiTextStream>();
  }
  bool contains(std::string_view needle) const
  {
    for (const std::string& l : lines) {
      if (l.find(needle) != std::string::npos) { return true; }
    }
    return false;
  }
};

}  // namespace

// The arithmetic that makes phases worth having: claims in different
// phases are never resident together, so the box has to survive their
// MAXIMUM, not their sum.
//
// The numbers are the shape of a real video graph -- a text encoder
// that is done before the DiT starts, a DiT that runs for minutes, and
// a VAE that only matters after -- scaled down. On MiniMax-H3 the same
// shape is 94 GB summed against 78 GB peaked, which is the difference
// between streaming a 33B DiT and holding it.
TEST(model_memory, phases_peak_rather_than_sum)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-phase-arith";
  std::error_code ec;
  fs::remove_all(root, ec);
  const std::string enc = make_ckpt(root / "enc", 3000);
  const std::string dit = make_ckpt(root / "dit", 9000);
  const std::string vae = make_ckpt(root / "vae", 1000);

  Session sess;
  auto*   mgr = sess.generative_model_manager();
  if (mgr == nullptr) { fs::remove_all(root, ec); return; }

  mgr->clear_declarations();
  mgr->declare_weights(enc, 3000, std::string(model_memory::kPhaseCondition));
  mgr->declare_weights(dit, 9000, std::string(model_memory::kPhaseDenoise));
  mgr->declare_weights(vae, 1000);          // persistent: no phase

  // Each phase sees the persistent weights plus only its own.
  EXPECT_TRUE(mgr->phase_footprint(std::string(model_memory::kPhaseCondition))
              == 4000u);
  EXPECT_TRUE(mgr->phase_footprint(std::string(model_memory::kPhaseDenoise))
              == 10000u);
  // A phase nobody claimed gets the persistent weights and nothing else.
  EXPECT_TRUE(mgr->phase_footprint(std::string(model_memory::kPhaseDecode))
              == 1000u);
  // Unnamed: the box-level peak -- persistent plus the WIDEST phase.
  EXPECT_TRUE(mgr->phase_footprint(std::string()) == 10000u);
  // And the unphased total is untouched, because that is the number
  // bounded() asks and it must stay the no-release worst case. If this
  // ever equals the peak above, a stage deciding whether to release is
  // reading a figure that already assumes it did.
  EXPECT_TRUE(mgr->resident_weight_bytes() == 13000u);

  mgr->clear_declarations();
  fs::remove_all(root, ec);
}

// Two stages disagreeing about one checkpoint's lifetime: the WIDER
// answer has to win. Believing the shorter one subtracts bytes that the
// other stage is still holding.
TEST(model_memory, the_wider_lifetime_wins)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-phase-widen";
  std::error_code ec;
  fs::remove_all(root, ec);
  const std::string d = make_ckpt(root / "m", 5000);

  Session sess;
  auto*   mgr = sess.generative_model_manager();
  if (mgr == nullptr) { fs::remove_all(root, ec); return; }

  // Phase first, then persistent: the persistent claim must widen it.
  mgr->clear_declarations();
  mgr->declare_weights(d, 5000, std::string(model_memory::kPhaseCondition));
  mgr->declare_weights(d, 5000);
  EXPECT_TRUE(mgr->phase_footprint(std::string(model_memory::kPhaseDenoise))
              == 5000u);

  // And the other order, which must reach the same answer -- otherwise
  // the result depends on which stage the runtime happened to ask first.
  mgr->clear_declarations();
  mgr->declare_weights(d, 5000);
  mgr->declare_weights(d, 5000, std::string(model_memory::kPhaseCondition));
  EXPECT_TRUE(mgr->phase_footprint(std::string(model_memory::kPhaseDenoise))
              == 5000u);

  mgr->clear_declarations();
  fs::remove_all(root, ec);
}

// A phase claim is a promise, and nothing else in the system can
// falsify it: if the encoder is never dropped the run just thrashes,
// with no line in the log connecting the thrash to the claim that
// caused it. So the promise is audited at the end of the run.
TEST(model_memory, an_unkept_phase_promise_is_reported)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-phase-audit";
  std::error_code ec;
  fs::remove_all(root, ec);
  const std::string kept   = make_ckpt(root / "kept", 2000);
  const std::string broken = make_ckpt(root / "broken", 2000);

  Session sess;
  auto    ui_owned = std::make_unique<CapturingUi>();
  auto*   log      = ui_owned.get();
  sess.set_ui_delegate(std::move(ui_owned));
  auto* mgr = sess.generative_model_manager();
  if (mgr == nullptr) { fs::remove_all(root, ec); return; }

  mgr->clear_declarations();
  log->lines.clear();
  mgr->declare_weights(kept, 2000, std::string(model_memory::kPhaseCondition));
  mgr->declare_weights(broken, 2000,
                       std::string(model_memory::kPhaseCondition));
  mgr->note_phase_released(kept);          // this one kept its word
  mgr->clear_declarations();               // end of run: audit here

  EXPECT_TRUE(log->contains("broken"));
  EXPECT_TRUE(log->contains("never reported being released"));
  // The one that released is NOT named -- an audit that warns about
  // every phase claim is one nobody reads.
  EXPECT_FALSE(log->contains("kept'"));

  fs::remove_all(root, ec);
}

namespace {

// A stage that DECLARES a checkpoint and, in the second pass, DECIDES
// whether to phase-limit it by asking how big the graph is. That
// question is the one the two passes exist to make order-free: asked
// mid-declaration it gets a different answer depending on where the
// flattener put this stage.
class DecidingStage : public TypedStage<DecidingStage> {
public:
  static constexpr const char* kTypeName = "ut-deciding-stage";
  using TypedStage::TypedStage;

  std::string dir;
  std::string phase;                       // empty => decide nothing
  // What the whole graph weighed when this stage was asked to decide.
  mutable std::size_t saw_at_decide = 0;

  std::vector<ResourceClaim> declare_resources() const override
  {
    return model_memory::weight_claims({dir});
  }

  std::vector<ResourceClaim> decide_resources() const override
  {
    saw_at_decide = model_memory::weight_footprint(session(), {});
    if (phase.empty()) { return {}; }
    return model_memory::weight_claims_in_phase({dir}, phase);
  }

  Job process(RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};
VPIPE_REGISTER_STAGE(DecidingStage)

// Build a graph of N unconnected declaring/deciding stages, in the
// given insertion order, and launch it.
//
// `pl` is the CALLER's, because the Pipeline owns the stages: built on
// this frame it would take them with it on return and leave the
// returned pointers dangling.
std::vector<DecidingStage*>
run_deciding_graph(Session& sess, std::unique_ptr<Pipeline>& pl,
                   const char* id,
                   const std::vector<std::pair<std::string, std::string>>& spec)
{
  pl = std::make_unique<Pipeline>(id, &sess);
  std::vector<DecidingStage*> out;
  for (std::size_t i = 0; i < spec.size(); ++i) {
    auto s = std::make_unique<DecidingStage>(
        &sess, "s" + std::to_string(i), std::vector<InEdge>{},
        FlexData::make_object());
    s->dir   = spec[i].first;
    s->phase = spec[i].second;
    s->allocate_oports(0);
    out.push_back(static_cast<DecidingStage*>(pl->insert_stage(std::move(s))));
  }
  PipelineRuntime rt(pl.get(), &sess);
  if (rt.launch()) {
    rt.wait_idle();
    rt.stop();
  }
  return out;
}

}  // namespace

// THE PROPERTY THE SPLIT EXISTS FOR: every stage deciding sees the same
// box, whatever order the runtime visits them in.
//
// With declaration and decision in one pass this is false by
// construction -- the first stage asked sees an empty graph and the last
// sees all of it, so the same graph sizes itself differently depending
// on where the flattener happened to emit each stage. That is the exact
// race the planning phase was built to remove, reintroduced inside it.
TEST(model_memory, every_stage_decides_against_the_same_box)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-decide-order";
  std::error_code ec;
  fs::remove_all(root, ec);
  const std::string a = make_ckpt(root / "a", 4000);
  const std::string b = make_ckpt(root / "b", 8000);
  const std::string c = make_ckpt(root / "c", 2000);

  Session sess;
  if (sess.generative_model_manager() == nullptr) {
    fs::remove_all(root, ec);
    return;
  }
  std::unique_ptr<Pipeline> pl;
  auto st = run_deciding_graph(sess, pl, "order",
                               {{a, ""}, {b, ""}, {c, ""}});
  ASSERT_TRUE(st.size() == 3);
  // All three saw the FULL 14000, including the stages asked first.
  EXPECT_TRUE(st[0]->saw_at_decide == 14000u);
  EXPECT_TRUE(st[1]->saw_at_decide == 14000u);
  EXPECT_TRUE(st[2]->saw_at_decide == 14000u);
  fs::remove_all(root, ec);
}

// And a refinement one stage makes must not be visible to another's
// decision -- otherwise the second pass has merely moved the ordering
// problem rather than removed it.
//
// It takes TWO phase claimants and an observer ordered after both to
// see this. The peak is persistent + the widest single phase, so moving
// one checkpoint out of persistent and into a phase where it is the
// widest changes nothing; only the SECOND phase, overlapping the first,
// makes the total drop. An earlier version of this test used one
// claimant, could not tell buffered from immediate, and passed against
// a planner that applied refinements on arrival.
TEST(model_memory, a_refinement_is_invisible_to_other_decisions)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-decide-buffer";
  std::error_code ec;
  fs::remove_all(root, ec);
  const std::string a = make_ckpt(root / "a", 4000);   // persistent
  const std::string b = make_ckpt(root / "b", 8000);   // condition
  const std::string c = make_ckpt(root / "c", 8000);   // decode
  const std::string d = make_ckpt(root / "d", 1000);   // persistent, LAST

  Session sess;
  auto*   mgr = sess.generative_model_manager();
  if (mgr == nullptr) { fs::remove_all(root, ec); return; }

  std::unique_ptr<Pipeline> pl;
  auto st = run_deciding_graph(
      sess, pl, "buffer",
      {{a, ""},
       {b, std::string(model_memory::kPhaseCondition)},
       {c, std::string(model_memory::kPhaseDecode)},
       {d, ""}});
  ASSERT_TRUE(st.size() == 4);
  // Every stage saw the whole 21000, including `d`, which decides after
  // both refinements. Applied on arrival it would have seen persistent
  // (a + d = 5000) plus the widest phase (8000) = 13000 -- a smaller
  // box than its peers were given, decided by position alone.
  EXPECT_TRUE(st[0]->saw_at_decide == 21000u);
  EXPECT_TRUE(st[3]->saw_at_decide == 21000u);
  // The refinements DID land once the pass was over: the two phases now
  // overlap, so the peak is persistent (5000) + the widest (8000).
  EXPECT_TRUE(mgr->phase_footprint(std::string()) == 13000u);
  fs::remove_all(root, ec);
}

// The second thing that has to fit, and until now the one nothing
// declared. A VAE's WEIGHTS are small -- FLUX.2's are 160 MB on disk --
// and its decode ARENA is gigabytes, so a graph accounted purely by
// weights reads as roomy right up to the allocation that does not fit.
TEST(model_memory, a_decode_arena_is_estimated_from_the_vae_config)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-scratch-est";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root / "vae", ec);
  auto write_vae = [&](const char* cls, const char* blocks) {
    std::ofstream f(root / "vae" / "config.json");
    f << "{\"_class_name\": \"" << cls
      << "\", \"block_out_channels\": " << blocks << "}";
  };

  // FLUX.2: ~7 full-res base-channel buffers at 2 B/elt -> h*w*base*14.
  // base is max(block_out[0], block_out[1]) -- block_out[0] alone
  // under-modelled the peak 2x, which is what the 2K overflow fix found.
  write_vae("AutoencoderKLFlux2", "[128, 256, 512, 512]");
  const std::size_t flux = model_memory::vae_decode_scratch_bytes(
      root.string(), 1024, 1024);
  EXPECT_TRUE(flux == (std::size_t)1024 * 1024 * 256 * 14);

  // Qwen-Image / Krea-2 spell their width `base_dim` and ship NO
  // block_out_channels -- every quantized Krea-2 and Qwen-Image-Edit
  // pack has the key absent. Read as block_out it fell to the 128
  // default against a real 96, over-declaring 1.33x.
  {
    std::ofstream f(root / "vae" / "config.json");
    f << "{\"_class_name\": \"AutoencoderKLQwenImage\", "
         "\"base_dim\": 96}";
  }
  EXPECT_TRUE(model_memory::vae_decode_scratch_bytes(root.string(), 1024, 1024)
              == (std::size_t)1024 * 1024 * 96 * 27);
  // ...and the default when even base_dim is absent is 96, not the
  // AutoencoderKL 128: a pack with neither key is still a Qwen-Image
  // VAE, and its width is not the other family's.
  write_vae("AutoencoderKLQwenImage", "[128, 256, 512, 512]");
  const std::size_t qwen = model_memory::vae_decode_scratch_bytes(
      root.string(), 1024, 1024);
  EXPECT_TRUE(qwen == (std::size_t)1024 * 1024 * 96 * 27);

  // Boogu ships the plain AutoencoderKL and this tree decodes it through
  // MetalFlux2Vae -- same code, same peak. Left at the conservative
  // default it over-declared Boogu 1.93x.
  write_vae("AutoencoderKL", "[128, 256, 512, 512]");
  EXPECT_TRUE(model_memory::vae_decode_scratch_bytes(root.string(), 1024, 1024)
              == flux);

  // An UNRECOGNISED VAE gets the LARGER multiplier on the block_out
  // width it does carry. Under-declaring an arena reads as room that is
  // not there, and the stage that believed it has already decided
  // something it cannot undo.
  write_vae("AutoencoderKLSomethingNew", "[128, 256, 512, 512]");
  EXPECT_TRUE(model_memory::vae_decode_scratch_bytes(root.string(), 1024, 1024)
              == (std::size_t)1024 * 1024 * 256 * 27);

  // No readable config at all is 0, not a guess: a caller that gets 0
  // falls back to its old behaviour rather than to a number nobody
  // computed.
  fs::remove_all(root / "vae", ec);
  EXPECT_TRUE(model_memory::vae_decode_scratch_bytes(root.string(), 1024, 1024)
              == 0u);
  EXPECT_TRUE(model_memory::vae_decode_scratch_bytes(root.string(), 0, 1024)
              == 0u);
  fs::remove_all(root, ec);
}

// The accounting has to use the geometry the model will actually
// produce, not the one config asked for.
//
// Every video model constrains its latent shape and rounds UP, so an
// estimate over the requested numbers describes a clip nobody will
// make. H3 is the sharpest case: its VAE takes 17-frame chunks and
// keeps 5 latents from each, so a 9-frame request becomes 22 and an
// unrounded arena is 2.4x short -- and under-declaring reads as room
// that is not there.
TEST(model_memory, the_arena_is_sized_from_the_rounded_geometry)
{
  // The rule the estimate depends on. If this ever stops being 22,
  // GenerateVideoStage::planned_geometry_ is describing a different
  // clip from the one MiniMax-H3 produces.
  EXPECT_TRUE(genai::minimax_h3::align_num_frames(9, 17, 5) == 22);

  const std::size_t asked =
      model_memory::video_decode_scratch_bytes(256, 256, 9);
  const std::size_t made =
      model_memory::video_decode_scratch_bytes(256, 256, 22);
  // 9 bytes per output pixel, so the ratio is exactly the frame ratio.
  EXPECT_TRUE(asked == (std::size_t)256 * 256 * 9 * 9);
  EXPECT_TRUE(made  == (std::size_t)256 * 256 * 22 * 9);
  EXPECT_TRUE(made > asked * 2);        // the 2.4x, in one assertion
}

// Scratch is phased like weights, and for the same reason: an arena
// exists only while its stage runs, so a decode's does not belong in the
// DiT's sizing -- but it does belong in the box-level peak.
TEST(model_memory, scratch_is_phased_and_kept_apart_from_weights)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-scratch-ledger";
  std::error_code ec;
  fs::remove_all(root, ec);
  const std::string w = make_ckpt(root / "w", 4000);

  Session sess;
  auto*   mgr = sess.generative_model_manager();
  if (mgr == nullptr) { fs::remove_all(root, ec); return; }

  mgr->clear_declarations();
  mgr->clear_scratch();
  mgr->declare_weights(w, 4000);
  mgr->declare_scratch("vae-decode", 9000,
                       std::string(model_memory::kPhaseDecode));
  mgr->declare_scratch("cond-arena", 1000,
                       std::string(model_memory::kPhaseCondition));

  EXPECT_TRUE(mgr->scratch_bytes(std::string(model_memory::kPhaseDecode))
              == 9000u);
  EXPECT_TRUE(mgr->scratch_bytes(std::string(model_memory::kPhaseCondition))
              == 1000u);
  // The denoise sees NEITHER arena -- that is the whole point.
  EXPECT_TRUE(mgr->scratch_bytes(std::string(model_memory::kPhaseDenoise))
              == 0u);
  // Unnamed: the peak, which is the widest single phase.
  EXPECT_TRUE(mgr->scratch_bytes(std::string()) == 9000u);
  // And it stays OUT of the weight ledger. Folding the two together
  // would make resident_weight_bytes() stop meaning what it says, and
  // every existing caller of it is asking about checkpoints.
  EXPECT_TRUE(mgr->resident_weight_bytes() == 4000u);
  EXPECT_TRUE(mgr->phase_footprint(std::string(model_memory::kPhaseDecode))
              == 4000u);

  // Two claims under one label are one arena named twice: counted once,
  // at the larger.
  mgr->declare_scratch("vae-decode", 5000,
                       std::string(model_memory::kPhaseDecode));
  EXPECT_TRUE(mgr->scratch_bytes(std::string(model_memory::kPhaseDecode))
              == 9000u);

  mgr->clear_scratch();
  EXPECT_TRUE(mgr->scratch_bytes(std::string()) == 0u);
  mgr->clear_declarations();
  fs::remove_all(root, ec);
}

// An arena's size is a function of the BEAT, not of configuration.
//
// This is what the plan cannot hold: a video decode's transient scales
// with the pixel frame count, which is the VAE's expansion of the
// latent, and an image EDIT's geometry comes from the reference image
// and is in no config at all. So the plan declares what it can compute
// -- or kUnknownArena when it can compute nothing -- and the first beat
// supplies the truth.
TEST(model_memory, a_declared_arena_is_revised_to_the_beats_real_size)
{
  Session sess;
  auto*   mgr = sess.generative_model_manager();
  if (mgr == nullptr) { return; }

  mgr->clear_scratch();
  mgr->declare_scratch("vae-decode", 1000,
                       std::string(model_memory::kPhaseDecode));

  // Up for a longer clip...
  mgr->revise_scratch("vae-decode", 9000);
  EXPECT_TRUE(mgr->scratch_bytes(std::string(model_memory::kPhaseDecode))
              == 9000u);
  // ...and back DOWN for a short one. SET, not max: a ledger that only
  // grew would describe the largest beat of the run forever, which for
  // a mixed-size edit sequence is wrong for every frame after the big
  // one.
  mgr->revise_scratch("vae-decode", 200);
  EXPECT_TRUE(mgr->scratch_bytes(std::string(model_memory::kPhaseDecode))
              == 200u);
  // The phase survives revision: a corrected arena is still a decode
  // arena, so the denoise still does not pay for it.
  EXPECT_TRUE(mgr->scratch_bytes(std::string(model_memory::kPhaseDenoise))
              == 0u);

  // A label nobody declared cannot be revised into existence -- the same
  // rule revise_declaration follows for weights. The plan stays
  // authoritative about WHAT exists; runtime only supplies magnitudes,
  // and a mistyped label is a no-op rather than a phantom entry.
  mgr->revise_scratch("never-declared", 5000);
  EXPECT_TRUE(mgr->scratch_bytes(std::string()) == 200u);
  mgr->clear_scratch();
}

// THE EDIT CASE, which is what kUnknownArena exists for. A graph with no
// config geometry declares a presence marker, and the first beat's real
// figure replaces it. Declaring nothing instead would leave the runtime
// truth with no entry to correct, and the ledger reporting 0 for an
// allocation about to happen.
TEST(model_memory, an_unsized_arena_is_declared_then_replaced)
{
  Session sess;
  auto*   mgr = sess.generative_model_manager();
  if (mgr == nullptr) { return; }

  mgr->clear_scratch();
  // The marker is negligible by construction: it cannot move a sizing
  // decision on any box this runs on.
  EXPECT_TRUE(model_memory::kUnknownArena < (1u << 20));
  mgr->declare_scratch("vae-decode", model_memory::kUnknownArena,
                       std::string(model_memory::kPhaseDecode));
  EXPECT_TRUE(mgr->scratch_bytes(std::string(model_memory::kPhaseDecode))
              == model_memory::kUnknownArena);

  // The first beat replaces it outright -- the marker is not a floor.
  mgr->revise_scratch("vae-decode", 234881024);
  EXPECT_TRUE(mgr->scratch_bytes(std::string(model_memory::kPhaseDecode))
              == 234881024u);
  // And it is still phase-limited, so the denoise never pays for it.
  EXPECT_TRUE(mgr->scratch_bytes(std::string(model_memory::kPhaseDenoise))
              == 0u);
  mgr->clear_scratch();
}

// The idle-unload rule, re-asked per beat in BOTH directions.
//
// The case that settles the direction: a run of image edits at mixed
// sizes. Frame 2 is large and forces the VAE out; frames 3-5 are small
// and must be allowed to bring it back, or every one of them pays a
// reload it did not need. Nothing here is on the critical path -- the
// answer decides only whether to hold the weights until the next beat.
TEST(model_memory, the_idle_rule_follows_the_beat_both_ways)
{
  const std::size_t ram   = 16000;
  const std::size_t peers = 10000;

  // Small beat on a roomy box: keep, whatever we were doing before.
  EXPECT_TRUE(!model_memory::resolve_idle_unload(ram, peers, 1000, false));
  EXPECT_TRUE(!model_memory::resolve_idle_unload(ram, peers, 1000, true));
  // Large beat that does not fit: unload, whatever we were doing before.
  EXPECT_TRUE(model_memory::resolve_idle_unload(ram, peers, 9000, false));
  EXPECT_TRUE(model_memory::resolve_idle_unload(ram, peers, 9000, true));

  // The five-frame edit run, in order. The large frame tightens; the
  // small ones after it must loosen again.
  bool unloading = false;
  const std::size_t sizes[5] = {1000, 9000, 1200, 800, 1000};
  bool seen[5] = {};
  for (int i = 0; i < 5; ++i) {
    unloading = model_memory::resolve_idle_unload(ram, peers, sizes[i],
                                                  unloading);
    seen[i] = unloading;
  }
  EXPECT_TRUE(!seen[0]);
  EXPECT_TRUE(seen[1]);           // the big frame forces the VAE out
  EXPECT_TRUE(!seen[2]);          // ...and the small ones bring it back
  EXPECT_TRUE(!seen[3]);
  EXPECT_TRUE(!seen[4]);

  // THE BAND. An arena sitting on the threshold holds its current
  // answer rather than flipping every beat -- a flip is a real reload
  // (MiniMax-H3's video VAE is 10.4 GB), so churn there is not free.
  // ram == peers + arena exactly: fits, but with no room to spare.
  EXPECT_TRUE(model_memory::resolve_idle_unload(ram, peers, 6000, true));
  EXPECT_TRUE(!model_memory::resolve_idle_unload(ram, peers, 6000, false));
  // Comfortably under (arena/8 = 625 of slack needed): decides `keep`
  // even from `unload`.
  EXPECT_TRUE(!model_memory::resolve_idle_unload(ram, peers, 5000, true));

  // An unknown box never churns: whatever was in force stays.
  EXPECT_TRUE(model_memory::resolve_idle_unload(0, peers, 9000, true));
  EXPECT_TRUE(!model_memory::resolve_idle_unload(0, peers, 9000, false));
}

// THE REMOVABLE POOL, which outlives a launch.
//
// A checkpoint a stage is finished with is kept alive and purgeable
// rather than dropped, so a relaunch over the same model pays no reload
// -- and under pressure it goes first, because it is spare capacity by
// construction where a live model's weights are not.
// THE LIMIT IS A COMPOSITION, and each layer has to be able to bind.
//
// ask (absolute, or pct% of RAM) -> capped by what the GPU can keep
// resident -> capped by whatever the box turned out to grant. Reported
// as one number, which is why the pieces are worth pinning separately:
// a run that streams because its pool is small has no other way to say
// WHICH of the three decided that.
TEST(model_memory, the_wired_pool_limit_composes_ask_and_device_cap)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }

  // The absolute form REPLACES the percentage rather than combining
  // with it, so a tiny ask wins over a large share of a large box.
  mgr->set_wired_pool_pct(75);
  mgr->set_wired_pool_bytes(64ull << 20);
  EXPECT_TRUE(mgr->wired_pool_bytes() == (64ull << 20));
  EXPECT_TRUE(mgr->wired_pool_limit() == (64ull << 20));

  // ...and setting the percentage clears it again, so which one is in
  // force never depends on the order they were set in.
  mgr->set_wired_pool_pct(50);
  EXPECT_TRUE(mgr->wired_pool_bytes() == 0u);
  EXPECT_TRUE(mgr->wired_pool_limit() > (64ull << 20));

  // 0 turns wiring off outright -- a setting, not a failure.
  mgr->set_wired_pool_pct(0);
  EXPECT_TRUE(mgr->wired_pool_limit() == 0u);

  // THE DEVICE CAP. An ask far above what the GPU can keep resident is
  // clamped rather than attempted: wiring past that point does not buy
  // residency, it buys a working set the driver pages against. Skipped
  // where there is no device to ask, which is every headless build.
  const std::size_t devmax = mgr->wired_pool_device_max();
  if (devmax > 0) {
    mgr->set_wired_pool_bytes(devmax * 4);
    EXPECT_TRUE(mgr->wired_pool_limit() == devmax);
    // And an ask BELOW the cap is left alone -- the cap is a ceiling,
    // not a target.
    mgr->set_wired_pool_bytes(devmax / 2);
    EXPECT_TRUE(mgr->wired_pool_limit() == devmax / 2);
  }
  mgr->set_wired_pool_pct(0);
}

// THE POOL IS A HARD CEILING, AND THE RETRY MUST NOT LIFT IT.
//
// Wired memory is the one allocation the kernel cannot take back, so an
// accounting bug here does not degrade throughput -- it panics the box.
// reopen_wired_pool() exists so a momentarily busy machine gets asked
// again, and the thing to prove about it is that it restores the
// CONFIGURED ask and nothing more.
TEST(model_memory, reopening_the_wired_pool_never_lifts_the_ceiling)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  auto* mc = s.services()->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  // Small enough to sit under RLIMIT_MEMLOCK on any box, and under the
  // device maximum so the ask is what binds.
  const std::size_t kAsk = 8ull << 20;
  mgr->set_wired_pool_bytes(kAsk);
  const std::size_t lim0 = mgr->wired_pool_limit();
  EXPECT_TRUE(lim0 == kAsk);

  // Fill it. Each buffer is charged, and the total never passes the ask.
  std::vector<metal_compute::SharedBuffer> held;
  for (int i = 0; i < 16; ++i) {
    auto b = mc->make_shared_buffer(1ull << 20);
    if (b.empty()) { break; }
    mgr->wire_into_pool(b);
    held.push_back(std::move(b));
    EXPECT_TRUE(mgr->wired_pool_used() <= mgr->wired_pool_limit());
  }
  const std::size_t used = mgr->wired_pool_used();
  EXPECT_TRUE(used <= kAsk);

  // Whatever happened above -- the pool filling, or the box refusing and
  // collapsing the ceiling to what it granted -- reopening restores the
  // ASK and never more than it.
  mgr->reopen_wired_pool();
  EXPECT_TRUE(mgr->wired_pool_limit() == kAsk);
  EXPECT_TRUE(mgr->wired_pool_used() == used);   // reopen wires nothing

  // And the gate still holds afterwards: nothing may be taken past the
  // ceiling, which is what stops a retry loop from growing without
  // bound.
  const std::size_t room = kAsk > used ? kAsk - used : 0;
  EXPECT_FALSE(mgr->wired_pool_can_take(room + (1ull << 20)));

  // Repeated reopens are idempotent -- a per-forward retry on a box that
  // never frees anything must not accumulate.
  for (int i = 0; i < 8; ++i) {
    mgr->reopen_wired_pool();
    EXPECT_TRUE(mgr->wired_pool_limit() == kAsk);
    EXPECT_TRUE(mgr->wired_pool_used() == used);
  }

  for (auto& b : held) { mgr->unwire_from_pool(b); }
  EXPECT_TRUE(mgr->wired_pool_used() == 0u);
  mgr->set_wired_pool_pct(0);
}

// A BUFFER THAT IS FREED WHILE WIRED LEAKS THE POOL'S BUDGET.
//
// Destroying a wired SharedBuffer unwires it in the kernel, so the
// machine recovers -- but `_pool_used` is decremented only by
// unwire_from_pool(), so the pool goes on believing those bytes are
// held. It then wires less of whatever comes next, and the resident set
// shrinks for a reason nothing in the log names.
//
// This pins the property so the accounting cannot drift back: the two
// models that reallocate wired scratch on a geometry change
// (MiniMax-H3's ensure_scratch_, LTX-2.5's set_geometry) both unwire
// first, and this is what says why they must.
TEST(model_memory, freeing_a_wired_buffer_without_unwiring_leaks_the_pool)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  auto* mc = s.services()->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  mgr->set_wired_pool_bytes(8ull << 20);
  EXPECT_TRUE(mgr->wired_pool_used() == 0u);

  // THE CORRECT ORDER: unwire, then drop. The pool comes back.
  {
    auto b = mc->make_shared_buffer(1ull << 20);
    if (b.empty() || mgr->wire_into_pool(b) == 0) {
      mgr->set_wired_pool_pct(0);
      return;                       // RLIMIT_MEMLOCK: self-skip
    }
    EXPECT_TRUE(mgr->wired_pool_used() == (1ull << 20));
    mgr->unwire_from_pool(b);
  }
  EXPECT_TRUE(mgr->wired_pool_used() == 0u);

  // THE MISTAKE: dropped while still wired. The counter does NOT come
  // back, which is the leak -- asserted rather than wished away, because
  // a test that pretended otherwise would pass on a fix that never
  // happened.
  {
    auto b = mc->make_shared_buffer(1ull << 20);
    if (b.empty() || mgr->wire_into_pool(b) == 0) {
      mgr->set_wired_pool_pct(0);
      return;
    }
    EXPECT_TRUE(mgr->wired_pool_used() == (1ull << 20));
  }
  EXPECT_TRUE(mgr->wired_pool_used() == (1ull << 20));   // still charged

  // And the budget really is gone: what is left is the pool minus the
  // phantom, so a caller sizing against it gets less than it asked for.
  EXPECT_FALSE(mgr->wired_pool_can_take(8ull << 20));
  mgr->set_wired_pool_pct(0);
}

// The device maximum outranks the ask, and reopening does not escape it
// either. Wiring past what the GPU can keep resident is the case that
// takes the machine down rather than merely slowing it.
TEST(model_memory, reopening_still_respects_the_device_cap)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  const std::size_t devmax = mgr->wired_pool_device_max();
  if (devmax == 0) { return; }              // no device to ask

  mgr->set_wired_pool_bytes(devmax * 4);
  EXPECT_TRUE(mgr->wired_pool_limit() == devmax);
  mgr->reopen_wired_pool();
  EXPECT_TRUE(mgr->wired_pool_limit() == devmax);
  EXPECT_FALSE(mgr->wired_pool_can_take(devmax + 1));
  mgr->set_wired_pool_pct(0);
}

TEST(model_memory, a_pooled_checkpoint_survives_its_last_model)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-pool-test";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root, ec);
  {
    std::ofstream f(root / "w.safetensors", std::ios::binary);
    // A minimal safetensors: 8-byte header length, then the header.
    const std::string hdr = "{\"__metadata__\":{}}";
    std::uint64_t n = hdr.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(hdr.data(), (std::streamsize)hdr.size());
  }
  const std::string dir = root.string();

  auto ws = mgr->weight_set(dir);
  if (!ws) { fs::remove_all(root, ec); return; }
  const genai::WeightSet* raw = ws.get();

  // Dropped by its last holder, WITHOUT being handed over first: the
  // manager owns every checkpoint it opened, so there is nothing to hand
  // over. This is the case the old weak-reference cache re-read from
  // disk, however comfortably the model fitted in RAM.
  ws.reset();
  auto again = mgr->weight_set(dir);
  EXPECT_TRUE(again != nullptr);
  EXPECT_TRUE(again.get() == raw);        // the SAME set, not a reopen

  // And a BORROWED set is not spare capacity: an eviction must not drop
  // pages a model is reading.
  EXPECT_TRUE(mgr->pooled_bytes() == 0u);

  fs::remove_all(root, ec);
}

// A set specialised to a run's parameters is DROPPED rather than pooled:
// handing it to a launch that does not share them would give that launch
// weights which are silently wrong for it.
TEST(model_memory, an_unrecyclable_checkpoint_is_not_pooled)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-pool-norecycle";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root, ec);
  {
    std::ofstream f(root / "w.safetensors", std::ios::binary);
    const std::string hdr = "{\"__metadata__\":{}}";
    std::uint64_t n = hdr.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(hdr.data(), (std::streamsize)hdr.size());
  }
  const std::string dir = root.string();

  auto ws = mgr->weight_set(dir);
  if (!ws) { fs::remove_all(root, ec); return; }
  EXPECT_TRUE(ws->recyclable());          // a plain checkpoint is
  ws->set_not_recyclable("AdaLN baked for a 5-step schedule");
  EXPECT_TRUE(!ws->recyclable());
  EXPECT_TRUE(!ws->unrecyclable_reason().empty());

  // The manager keeps what nobody is borrowing -- EXCEPT this. Dropped
  // outright, so the next launch opens the checkpoint afresh rather than
  // inheriting a set baked for a run it knows nothing about.
  ws.reset();
  mgr->pool_weights(dir);                   // settle it now
  EXPECT_TRUE(mgr->weight_set_count() == 0u);

  // Re-opening gives a FRESH set, which is what "not recyclable" has to
  // mean: the specialisation did not come back with it. Checked by the
  // flag rather than by address, since a new allocation may reuse the
  // old one's.
  auto again = mgr->weight_set(dir);
  ASSERT_TRUE(again != nullptr);
  if (again != nullptr) {
    EXPECT_TRUE(again->recyclable());
    EXPECT_TRUE(again->unrecyclable_reason().empty());
  }
  fs::remove_all(root, ec);
}

// EITHER ORDER WORKS, which it did not used to.
//
// pool_weights() found the set through a WEAK reference, so a caller
// that dropped its model first had nothing left to pool -- the call
// returned having done neither the pooling nor the "not recyclable"
// report. That shipped once: generate-video reset its DiT and then
// pooled, and the silence looked exactly like a checkpoint the pool had
// refused. The manager owns the set now, so the order is free; what the
// call still does is ask for the checkpoint to be settled NOW rather
// than whenever the manager is next doing work.
TEST(model_memory, pooling_after_the_last_holder_is_a_no_op)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-pool-order";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root, ec);
  {
    std::ofstream f(root / "w.safetensors", std::ios::binary);
    const std::string hdr = "{\"__metadata__\":{}}";
    std::uint64_t n = hdr.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(hdr.data(), (std::streamsize)hdr.size());
  }
  const std::string dir = root.string();

  // DROP, THEN POOL: settles immediately, and the set is still there.
  {
    auto ws = mgr->weight_set(dir);
    if (!ws) { fs::remove_all(root, ec); return; }
    const genai::WeightSet* raw = ws.get();
    ws.reset();
    mgr->pool_weights(dir);
    auto again = mgr->weight_set(dir);
    EXPECT_TRUE(again != nullptr);
    EXPECT_TRUE(again != nullptr && again.get() == raw);
  }
  // POOL, THEN DROP: the caller is still borrowing at the call, so
  // nothing settles there; the set is kept regardless and settles on the
  // next manager call.
  {
    auto ws = mgr->weight_set(dir);
    if (!ws) { fs::remove_all(root, ec); return; }
    const genai::WeightSet* raw = ws.get();
    mgr->pool_weights(dir);
    ws.reset();
    auto again = mgr->weight_set(dir);
    EXPECT_TRUE(again != nullptr);
    EXPECT_TRUE(again.get() == raw);   // recycled, not reopened
  }
  fs::remove_all(root, ec);
}


// ---- WiredPool: the per-model window onto the pool -----------------------
//
// The state machine every streamed DiT now shares (MiniMax-H3, LTX-2.5,
// FLUX.2, Krea-2, Qwen-Image-Edit/Mage-Flow, Boogu-Image). What it
// decides is whether a block may be KEPT at all, so the properties worth
// pinning are the ones that bound memory rather than the ones that make
// it fast: the budget is never exceeded, a refusal STOPS admission, and
// the counter can neither underflow nor drift upward on its own.
//
// Wired memory is the one allocation the kernel cannot reclaim, so
// "uncapped growth" here is not a slow run, it is a panicked box.
TEST(model_memory, the_wire_budget_bounds_what_a_model_will_keep)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  auto* mc = s.services()->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  const std::size_t kAsk = 8ull << 20;
  mgr->set_wired_pool_bytes(kAsk);

  genai::WiredPool w;
  w.open(mc);
  if (!w.on()) { mgr->set_wired_pool_pct(0); return; }   // pool unavailable
  EXPECT_TRUE(w.budget() <= mgr->wired_pool_limit());

  // A block that fits is admissible; one that does not is refused BEFORE
  // it is kept -- which is the whole point. Growth that outruns the pool
  // leaves blocks resident but unprotected, and those are exactly the
  // pages the compressor takes first.
  const std::size_t budget = w.budget();
  EXPECT_TRUE(w.wirable(budget));
  EXPECT_FALSE(w.wirable(budget + 1));

  // Wire real buffers through it, one "block" at a time, and check the
  // two invariants after every one: this model never claims more than
  // its budget, and the process never passes the pool's ceiling.
  std::vector<metal_compute::SharedBuffer> held;
  for (int i = 0; i < 24; ++i) {
    const std::size_t nb = 1ull << 20;
    if (!w.wirable(nb)) { break; }
    auto b = mc->make_shared_buffer(nb);
    if (b.empty()) { break; }
    const std::size_t got = w.wire_one(mc, b, true);
    w.note_wired(mc, got, nb);
    held.push_back(std::move(b));
    EXPECT_TRUE(w.wired_bytes() <= w.budget());
    EXPECT_TRUE(mgr->wired_pool_used() <= mgr->wired_pool_limit());
  }
  EXPECT_TRUE(w.wired_bytes() <= kAsk);

  // Giving blocks back is what an eviction does, and it must land on
  // zero rather than wrapping: note_unwired takes a byte count, and a
  // count larger than what is held is the ordinary case when a partly
  // wired block is dropped whole.
  for (auto& b : held) { w.note_unwired(w.wire_one(mc, b, false)); }
  w.note_unwired(1ull << 40);                    // deliberate over-give
  EXPECT_TRUE(w.wired_bytes() == 0u);
  held.clear();
  EXPECT_TRUE(mgr->wired_pool_used() == 0u);

  mgr->set_wired_pool_pct(0);
}

// A REFUSAL COLLAPSES THE BUDGET, and the retry cannot raise it past the
// pool. The first half is what stops a run admitting blocks nothing can
// protect; the second is what stops the retry becoming a way around the
// ceiling.
TEST(model_memory, a_refused_wiring_stops_growth_and_the_retry_is_capped)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  auto* mc = s.services()->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  mgr->set_wired_pool_bytes(8ull << 20);
  genai::WiredPool w;
  w.open(mc);
  if (!w.on() || w.budget() == 0) { mgr->set_wired_pool_pct(0); return; }

  // A partial grant -- the box gave less than the block asked for.
  const std::size_t want = 4ull << 20, got = 1ull << 20;
  w.note_wired(mc, got, want);
  EXPECT_TRUE(w.wired_bytes() == got);
  // Budget is now exactly what was granted, so nothing further is
  // admissible: growth stops on the spot rather than one failed mlock
  // per block for the rest of the schedule.
  EXPECT_TRUE(w.budget() == got);
  EXPECT_FALSE(w.wirable(1));

  // The retry is gated on the box having freed a block's worth SINCE the
  // refusal, so asking with a huge block never fires -- a genuinely full
  // box is never asked, because reopening the ceiling would let the mlock
  // behind it fail and leave a block resident but unwired.
  EXPECT_FALSE(w.retry(mc, (std::size_t)1 << 60));
  EXPECT_TRUE(w.budget() == got);

  // And however it goes, the budget never passes the pool's own limit.
  w.retry(mc, 0);
  EXPECT_TRUE(w.budget() <= mgr->wired_pool_limit());
  EXPECT_TRUE(w.wired_bytes() <= w.budget());

  mgr->set_wired_pool_pct(0);
}

// WITH THE POOL OFF, NOTHING CHANGES. The gate has to be transparent:
// wirable() is what sits in front of every admission, so a false here
// would turn the pool being unavailable into a model that streams
// everything.
TEST(model_memory, a_closed_pool_gates_nothing)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  auto* mc = s.services()->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  mgr->set_wired_pool_pct(0);
  genai::WiredPool w;
  w.open(mc);
  EXPECT_FALSE(w.on());
  EXPECT_TRUE(w.wirable((std::size_t)1 << 60));   // admission unaffected
  EXPECT_FALSE(w.retry(mc, 0));
  EXPECT_TRUE(w.wired_bytes() == 0u);
}

// A MAPPED WEIGHT CAN BE NEITHER WIRED NOR PARKED, so the residency
// choice and the pool are one decision, not two. This is the rule every
// DiT loader now shares; getting it backwards is silent (the load
// succeeds, the mlock refuses, and the run reads as merely slow).
TEST(model_memory, mapping_is_off_whenever_streaming_or_wiring)
{
  EXPECT_TRUE(genai::weights_may_be_mapped(false, false));
  EXPECT_FALSE(genai::weights_may_be_mapped(true, false));    // streaming
  EXPECT_FALSE(genai::weights_may_be_mapped(false, true));    // pool on
  EXPECT_FALSE(genai::weights_may_be_mapped(true, true));
}

// A PHASE ON A DECLARATION IS HONOURED, and the planner must not say
// otherwise.
//
// `ResourceClaim::phase` is a declaration-time field -- "Optional
// LIFETIME", empty meaning the whole run -- and every VAE stage sets it
// to `decode`, because a VAE is loaded when a latent arrives and dropped
// after: charging its whole weight to the conditioning and denoise
// columns is what turns a graph that runs into a graph that reads as too
// big.
//
// The weights planner warned that a declared phase was "ignored" and
// told callers to move it to decide_resources. That was true of the
// commit that added the warning and stopped being true three days later,
// when claim() started forwarding `phase` to declare_weights -- so the
// warning outlived its subject and fired on every graph with a
// vae-decode in it, advising a move that was never needed. The two-pass
// split exists to stop a stage READING accumulating manager state in
// declare_resources; it was never about describing a lifetime.
//
// This goes through the PLANNER rather than declare_weights directly,
// because the planner is the part that was dropping it, and a test that
// called the manager would have passed throughout.
TEST(model_memory, a_declared_phase_survives_the_planner)
{
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  ResourcePlanner* pl =
      ResourcePlannerRegistry::get().find(model_memory::kWeightsKind);
  ASSERT_TRUE(pl != nullptr);
  if (pl == nullptr) { return; }

  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-mm-declphase";
  std::error_code ec;
  fs::remove_all(root, ec);
  auto put = [&](const fs::path& rel, std::size_t n) {
    fs::create_directories((root / rel).parent_path(), ec);
    std::ofstream f(root / rel, std::ios::binary);
    std::string blob(n, 'x');
    f.write(blob.data(), (std::streamsize)blob.size());
  };
  put("diffusion_models/dit.safetensors", 8000);
  put("vae/vae.safetensors", 2000);
  const std::string dit = (root / "diffusion_models").string();
  const std::string vae = (root / "vae").string();

  // Exactly what the runtime does: begin, every stage's claims, end.
  // The DiT is unphased (held throughout); the VAE names `decode` the
  // way VaeDecodeStage::declare_resources does.
  pl->begin_plan(&s);
  pl->claim(&s, dit, "", "", 0);
  pl->claim(&s, vae, std::string(model_memory::kPhaseDecode), "", 0);
  pl->end_plan(&s);

  // The denoise does not carry the VAE; the decode carries both.
  EXPECT_TRUE(mgr->phase_footprint(
                  std::string(model_memory::kPhaseDenoise)) == 8000u);
  EXPECT_TRUE(mgr->phase_footprint(
                  std::string(model_memory::kPhaseDecode)) == 10000u);
  // And the peak is the widest phase, not the sum of every claim ever
  // made -- which is the number an ignored phase would have produced.
  EXPECT_TRUE(mgr->phase_peak(nullptr) == 10000u);

  fs::remove_all(root, ec);
}

namespace {

// RAII for VPIPE_RAM_LIMIT_MB, which every memory decision reads to
// decide what it BELIEVES the box has.
class RamLimit {
public:
  explicit RamLimit(const char* mb) { ::setenv("VPIPE_RAM_LIMIT_MB", mb, 1); }
  ~RamLimit()                       { ::unsetenv("VPIPE_RAM_LIMIT_MB"); }
};

}   // namespace

// A PEAK OVER BELIEVED RAM REPORTS; IT DOES NOT REFUSE -- and the advice
// it gives names the free fix first.
//
// The message used to end "It does not fit at any setting", which is a
// verdict, and the graph it was aimed at ran to completion. These models
// hold their weights in mlock-WIRED buffers and the pool has a ceiling:
// past it WiredPool::wirable() gates ADMISSION rather than merely
// wiring, so a block nothing can protect is not kept at all and the
// resident set falls back toward its streaming floor. That costs a
// re-read per forward -- speed -- not correctness, and nothing fails to
// allocate: a refused buffer is still allocated, just unwired.
//
// So the contract this pins is (a) end_plan does NOT refuse, and (b) the
// remedy an operator reads first is `unload_when_idle`, ahead of the two
// that trade output for room. Asserting the ORDER rather than the
// wording, so the prose can be edited without breaking the test but the
// priority cannot be silently inverted.
TEST(model_memory, over_believed_ram_reports_and_advises_unload_first)
{
  // The UI delegate, not the log delegate: Session::warn routes there.
  auto  ui_owner = std::make_unique<CapturingUi>();
  auto* ui       = ui_owner.get();
  Session s;
  s.set_ui_delegate(std::move(ui_owner));
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  ResourcePlanner* pl =
      ResourcePlannerRegistry::get().find(model_memory::kWeightsKind);
  ASSERT_TRUE(pl != nullptr);
  if (pl == nullptr) { return; }

  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-mm-overram";
  std::error_code ec;
  fs::remove_all(root, ec);
  auto put = [&](const fs::path& rel, std::size_t n) {
    fs::create_directories((root / rel).parent_path(), ec);
    std::ofstream f(root / rel, std::ios::binary);
    std::string blob(n, 'x');
    f.write(blob.data(), (std::streamsize)blob.size());
  };
  // Comfortably over the 1 MB box below, so `need` clears it after the
  // >> 20 the message prints in.
  put("diffusion_models/dit.safetensors", 6u << 20);
  put("vae/vae.safetensors", 3u << 20);

  RamLimit tiny("1");
  pl->begin_plan(&s);
  pl->claim(&s, (root / "diffusion_models").string(), "", "", 0);
  pl->claim(&s, (root / "vae").string(),
            std::string(model_memory::kPhaseDecode), "", 0);
  // REPORTED, NOT REFUSED. wired_pool_enforce is off by default, and a
  // graph over believed RAM is exactly the case that still runs.
  EXPECT_TRUE(pl->end_plan(&s));

  std::string msg;
  for (const std::string& l : ui->lines) {
    if (l.find("estimated peak") != std::string::npos) { msg = l; break; }
  }
  EXPECT_TRUE(!msg.empty());
  if (msg.empty()) { fs::remove_all(root, ec); return; }
  // The old verdict is gone.
  EXPECT_TRUE(msg.find("does not fit at any setting") == std::string::npos);
  // Streaming is already inside the number, so recommending it would be
  // advice that cannot be taken.
  const std::size_t unload = msg.find("unload_when_idle");
  const std::size_t geom   = msg.find("geometry");
  const std::size_t quant  = msg.find("quantized");
  EXPECT_TRUE(unload != std::string::npos);
  EXPECT_TRUE(geom != std::string::npos);
  EXPECT_TRUE(quant != std::string::npos);
  // The free fix is offered BEFORE the two that cost output quality.
  EXPECT_TRUE(unload < geom && unload < quant);
  // And it names the policy that actually shortens a lifetime.
  EXPECT_TRUE(msg.find("destroy") != std::string::npos);
  EXPECT_TRUE(msg.find("park") != std::string::npos);

  fs::remove_all(root, ec);
}

// EVERY PHASE IN THE RUNNING ORDER IS DECIDABLE.
//
// The planner's decide() hand-listed condition/denoise/decode and so
// rejected `decode-audio`, which has been a first-class phase since the
// two decodes were split apart ("an audio VAE and a video VAE are loaded
// and dropped independently, and summing them sizes a moment that does
// not happen"). A stage deciding it correctly was told its phase was
// unknown and had its claim counted resident for the WHOLE RUN -- an
// over-count that lands in every other phase's column, which is how a
// graph starts reading as too big for its box.
//
// Driven from kPhasesInOrder rather than from a list written here, so
// the test cannot desync the way the code did: adding a phase adds a
// case automatically.
TEST(model_memory, every_phase_in_the_order_can_be_decided)
{
  auto  ui_owner = std::make_unique<CapturingUi>();
  auto* ui       = ui_owner.get();
  Session s;
  s.set_ui_delegate(std::move(ui_owner));
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  ResourcePlanner* pl =
      ResourcePlannerRegistry::get().find(model_memory::kWeightsKind);
  ASSERT_TRUE(pl != nullptr);
  if (pl == nullptr) { return; }

  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-mm-allphases";
  std::error_code ec;
  fs::remove_all(root, ec);
  const int n = model_memory::phase_count();
  EXPECT_TRUE(n > 0);
  std::vector<std::string> dirs;
  for (int i = 0; i < n; ++i) {
    const fs::path d = root / ("c" + std::to_string(i));
    fs::create_directories(d, ec);
    std::ofstream f(d / "w.safetensors", std::ios::binary);
    std::string blob(1000, 'x');
    f.write(blob.data(), (std::streamsize)blob.size());
    dirs.push_back(d.string());
  }

  pl->begin_plan(&s);
  for (const std::string& d : dirs) { pl->claim(&s, d, "", "", 0); }
  // One component per phase, each deciding the phase it really runs in.
  for (int i = 0; i < n; ++i) {
    pl->decide(&s, dirs[(std::size_t)i],
               std::string(model_memory::phase_name(i)));
  }
  pl->end_plan(&s);

  // Not one of them may be called unknown.
  const bool any_unknown = ui->contains("decides unknown phase");
  if (any_unknown) {
    for (const std::string& l : ui->lines) {
      std::printf("[all-phases] %s\n", l.c_str());
    }
  }
  EXPECT_FALSE(any_unknown);
  // And each really landed in its own phase: every phase carries its own
  // 1000 bytes and nothing else, so the peak is one component, not four.
  for (int i = 0; i < n; ++i) {
    EXPECT_TRUE(mgr->phase_footprint(
                    std::string(model_memory::phase_name(i))) == 1000u);
  }
  EXPECT_TRUE(mgr->phase_peak(nullptr) == 1000u);

  fs::remove_all(root, ec);
}
