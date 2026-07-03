// Env-driven driver to (a) quantize a raw bf16 LM to a PERSISTENT dir and
// (b) benchmark any loadable LM dir at chosen contexts. Used for the
// Gemma-4 peak-path perf sweep. All tests skip vacuously without env.
//
//   Quantize:  VPIPE_GBENCH_SRC=<raw dir> VPIPE_GBENCH_OUT=<persistent dir>
//              VPIPE_GBENCH_BITS=4|8  (skip_existing=true -> cheap re-run)
//   Benchmark: VPIPE_GBENCH_MODEL=<dir>
//              VPIPE_GBENCH_CONTEXTS=1024,2048,4096 (default)
//              VPIPE_GBENCH_DECODE=64 (default)  VPIPE_GBENCH_WARMUP=1

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "stages/model-benchmark-stage.h"
#include "stages/model-quantize-stage.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;

TEST(gemma_bench_driver, quantize)
{
  const char* src = std::getenv("VPIPE_GBENCH_SRC");
  const char* out = std::getenv("VPIPE_GBENCH_OUT");
  if (src == nullptr || *src == '\0') { return; }
  if (out == nullptr || *out == '\0') { return; }
  const char* bits_s = std::getenv("VPIPE_GBENCH_BITS");
  const int bits = (bits_s && *bits_s) ? std::atoi(bits_s) : 8;
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("src_model", FlexData::make_string(src));
    o.insert("output_name", FlexData::make_string(out));
    o.insert("bits", FlexData::make_int(bits));
    o.insert("skip_existing", FlexData::make_bool(true));
  }
  ModelQuantizeStage q(&sess, "mq", std::vector<InEdge>{}, std::move(cfg));
  ASSERT_TRUE(q.config_error().empty());
  EXPECT_TRUE(q.quantize_once());
}

TEST(gemma_bench_driver, benchmark)
{
  const char* m = std::getenv("VPIPE_GBENCH_MODEL");
  if (m == nullptr || *m == '\0') { return; }
  if (!std::filesystem::exists(
          std::filesystem::path(m) / "config.json")) {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  const char* ctx = std::getenv("VPIPE_GBENCH_CONTEXTS");
  const char* dec = std::getenv("VPIPE_GBENCH_DECODE");
  const char* wu = std::getenv("VPIPE_GBENCH_WARMUP");
  const char* cd = std::getenv("VPIPE_GBENCH_COOLDOWN");

  FlexData c = FlexData::make_object();
  {
    auto o = c.as_object();
    o.insert("model", FlexData::make_string(m));
    o.insert("contexts", FlexData::make_string(
        (ctx && *ctx) ? ctx : "1024,2048,4096"));
    o.insert("decode_tokens",
             FlexData::make_int((dec && *dec) ? std::atoi(dec) : 64));
    o.insert("warmup",
             FlexData::make_int((wu && *wu) ? std::atoi(wu) : 1));
    o.insert("cooldown_s",
             FlexData::make_real((cd && *cd) ? std::atof(cd) : 0.0));
  }
  ModelBenchmarkStage s(&sess, "bm", std::vector<InEdge>{}, std::move(c));
  ASSERT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.benchmark_once());
}
