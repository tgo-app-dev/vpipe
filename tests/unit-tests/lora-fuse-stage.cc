// LoraFuseStage: config-validation + the trigger/summary ports that let it
// cascade into a preparation recipe (model-fetch -> model-quantize ->
// lora-fuse). The fusion math itself is covered by krea2_lora.

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "stages/lora-fuse-stage.h"

#include <iostream>
#include <streambuf>
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

FlexData
basic_cfg_()
{
  FlexData cfg = FlexData::make_object();
  auto o = cfg.as_object();
  o.insert("base_model", FlexData::make_string("/tmp/base/transformer"));
  o.insert("lora", FlexData::make_string("/tmp/lora.safetensors"));
  o.insert("output_name", FlexData::make_string("/tmp/fused"));
  return cfg;
}

}  // namespace

TEST(lora_fuse_stage, type_is_registered)
{
  EXPECT_TRUE(std::string_view(LoraFuseStage::kTypeName) == "lora-fuse");
}

// One trigger iport (any beat type) + one FlexData summary oport, matching
// model-fetch / model-quantize so the three cascade into a setup recipe.
TEST(lora_fuse_stage, trigger_and_summary_ports)
{
  Session sess;
  CerrSilencer hush;
  LoraFuseStage s(&sess, "lf", std::vector<InEdge>{}, basic_cfg_());
  const StageSpec& sp = s.spec();
  ASSERT_TRUE(sp.iports.size() == 1u);
  ASSERT_TRUE(sp.oports.size() == 1u);
  EXPECT_TRUE(std::string_view(sp.iports[0].name) == "trigger");
  EXPECT_TRUE(sp.iports[0].type == nullptr);          // any beat type
  EXPECT_TRUE(std::string_view(sp.oports[0].name) == "summary");
  // By mangled name, not typeid pointer (stage in libvpipe vs test image).
  ASSERT_TRUE(sp.oports[0].type != nullptr);
  EXPECT_TRUE(std::string_view(sp.oports[0].type->name())
              == typeid(FlexDataPayload).name());
}

TEST(lora_fuse_stage, config_defaults)
{
  Session sess;
  CerrSilencer hush;
  LoraFuseStage s(&sess, "lf", std::vector<InEdge>{}, basic_cfg_());
  EXPECT_TRUE(s.base_model() == "/tmp/base/transformer");
  EXPECT_TRUE(s.scale() == 1.0);                       // default fusion strength
}

// Config is deferred-validated: a missing required field records a config
// error (the runtime skips the stage at launch) rather than throwing.
TEST(lora_fuse_stage, missing_required_deferred)
{
  Session sess;
  CerrSilencer hush;
  const char* cfgs[] = {
    R"({"lora":"/l","output_name":"/o"})",             // no base_model
    R"({"base_model":"/b","output_name":"/o"})",       // no lora
    R"({"base_model":"/b","lora":"/l"})",              // no output_name
  };
  for (const char* json : cfgs) {
    LoraFuseStage s(&sess, "lf", std::vector<InEdge>{}, FlexData::from_json(json));
    EXPECT_FALSE(s.config_error().empty());
  }
}
