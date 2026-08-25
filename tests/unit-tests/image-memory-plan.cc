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

#include "common/flex-data.h"
#include "common/session.h"
#include "pipeline/pipeline.h"
#include "stages/generate-image-stage.h"
#include "stages/model-memory.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
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
