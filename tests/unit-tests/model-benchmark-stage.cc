// ModelBenchmarkStage: always-on config-validation tests, plus an
// env-gated real benchmark (VPIPE_QWEN35_TEST_MODEL_PATH = a loadable
// LM dir).

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "stages/model-benchmark-stage.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {
class CerrSilencer {
public:
  CerrSilencer() : _saved(std::cerr.rdbuf()) { std::cerr.rdbuf(&_null); }
  ~CerrSilencer() { std::cerr.rdbuf(_saved); }
private:
  struct NullBuf : std::streambuf {
    int overflow(int c) override { return c; }
  };
  std::streambuf* _saved;
  NullBuf         _null;
};
}  // namespace

TEST(model_benchmark_stage, type_is_registered)
{
  EXPECT_TRUE(std::string_view(ModelBenchmarkStage::kTypeName)
              == "model-benchmark");
}

TEST(model_benchmark_stage, config_defaults)
{
  Session sess;
  CerrSilencer hush;
  FlexData c = FlexData::make_object();
  c.as_object().insert("model", FlexData::make_string("/tmp/m"));
  ModelBenchmarkStage s(&sess, "bm", std::vector<InEdge>{},
                        std::move(c));
  EXPECT_TRUE(s.model() == "/tmp/m");
  EXPECT_TRUE(s.contexts().size() == 3);
  EXPECT_TRUE(s.decode_tokens() == 128);
  EXPECT_TRUE(s.prefill_chunk() == 1024);
  EXPECT_TRUE(s.config_error().empty());
}

TEST(model_benchmark_stage, contexts_parse_sorted_unique)
{
  Session sess;
  CerrSilencer hush;
  FlexData c = FlexData::make_object();
  auto o = c.as_object();
  o.insert("model", FlexData::make_string("/tmp/m"));
  o.insert("contexts", FlexData::make_string("4096, 1024,1024 ,2048"));
  ModelBenchmarkStage s(&sess, "bm", std::vector<InEdge>{},
                        std::move(c));
  ASSERT_TRUE(s.contexts().size() == 3);
  EXPECT_TRUE(s.contexts()[0] == 1024);
  EXPECT_TRUE(s.contexts()[1] == 2048);
  EXPECT_TRUE(s.contexts()[2] == 4096);
}

TEST(model_benchmark_stage, missing_model_deferred)
{
  Session sess;
  CerrSilencer hush;
  FlexData c = FlexData::make_object();
  ModelBenchmarkStage s(&sess, "bm", std::vector<InEdge>{},
                        std::move(c));
  EXPECT_FALSE(s.config_error().empty());
}

// End-to-end benchmark on a real LM. Skips unless the model dir is
// provided. Small budgets keep it quick.
TEST(model_benchmark_stage, real_benchmark)
{
  const char* m = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (m == nullptr || *m == '\0') { return; }
  if (!std::filesystem::exists(
          std::filesystem::path(m) / "config.json")) {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  FlexData c = FlexData::make_object();
  {
    auto o = c.as_object();
    o.insert("model", FlexData::make_string(m));
    o.insert("contexts", FlexData::make_string("256,512"));
    o.insert("decode_tokens", FlexData::make_int(16));
    o.insert("warmup", FlexData::make_int(0));
  }
  ModelBenchmarkStage s(&sess, "bm", std::vector<InEdge>{},
                        std::move(c));
  ASSERT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.benchmark_once());
}

// Cooperative stop: a predicate that trips after a couple of checks must
// break the run out early (not grind through all 9 tests of a 3-context x
// 3-phase sweep with 64-token decodes) and still return a partial report.
TEST(model_benchmark_stage, stop_is_responsive)
{
  const char* m = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (m == nullptr || *m == '\0') { return; }
  if (!std::filesystem::exists(
          std::filesystem::path(m) / "config.json")) {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  FlexData c = FlexData::make_object();
  {
    auto o = c.as_object();
    o.insert("model", FlexData::make_string(m));
    o.insert("contexts", FlexData::make_string("256,512,1024"));
    o.insert("decode_tokens", FlexData::make_int(64));
    o.insert("warmup", FlexData::make_int(0));
  }
  ModelBenchmarkStage s(&sess, "bm", std::vector<InEdge>{},
                        std::move(c));
  ASSERT_TRUE(s.config_error().empty());

  int calls = 0;
  EXPECT_TRUE(s.benchmark_once([&] { return ++calls >= 3; }));
}

// MTP speculative-decode benchmark. Skips unless the OptiQ model dir is
// provided (VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH = a model carrying an MTP
// head). Validates that the mtp=true decode path runs end-to-end and the
// report is produced.
TEST(model_benchmark_stage, real_benchmark_mtp)
{
  const char* m = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (m == nullptr || *m == '\0') { return; }
  if (!std::filesystem::exists(
          std::filesystem::path(m) / "config.json")) {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  FlexData c = FlexData::make_object();
  {
    auto o = c.as_object();
    o.insert("model", FlexData::make_string(m));
    o.insert("contexts", FlexData::make_string("256,512"));
    o.insert("decode_tokens", FlexData::make_int(32));
    o.insert("warmup", FlexData::make_int(0));
    o.insert("mtp", FlexData::make_bool(true));
  }
  ModelBenchmarkStage s(&sess, "bm", std::vector<InEdge>{},
                        std::move(c));
  ASSERT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.benchmark_once());
}
