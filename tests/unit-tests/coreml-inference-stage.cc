// CoreMLInferenceStage end-to-end test.
//
// Most paths require a real .mlmodelc on disk; supply one via
//
//     VPIPE_TEST_COREML_MODEL=/path/to/tiny.mlmodelc
//     VPIPE_TEST_COREML_INPUT=name_of_input_feature
//     VPIPE_TEST_COREML_OUTPUT=name_of_output_feature
//
// When VPIPE_TEST_COREML_MODEL is unset, all model-loading tests
// pass with a single check that the stage type is registered.

#include "minitest.h"

#include "stages/coreml-inference-stage.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "pipeline/stage-registry.h"

#include <cstdlib>
#include <string>

namespace {

const char* env_or_null_(const char* name)
{
  const char* v = std::getenv(name);
  return (v && *v) ? v : nullptr;
}

}

TEST(coreml_inference_stage, type_is_registered)
{
  // The TypedStage<>/VPIPE_REGISTER_STAGE pair must register the
  // stage type by name at static-init time.
  auto& reg = vpipe::StageRegistry::get();
  EXPECT_TRUE(reg.find_id("coreml-inference") !=
              vpipe::StageTypeId::unknown);
}

// Construction must succeed for any config (so a graph can be built/
// edited before required fields are supplied); the missing field is
// recorded in config_error() and deferred to launch.
TEST(coreml_inference_stage, missing_model_path_deferred)
{
  vpipe::FlexData cfg = vpipe::FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "input_feature_name", vpipe::FlexData::make_string("x"));
  vpipe::FlexData outs = vpipe::FlexData::make_array();
  outs.as_array().push_back(vpipe::FlexData::make_string("y"));
  cfg.as_object().insert_or_assign("output_feature_names",
                                   std::move(outs));
  // model_path is omitted.

  bool threw = false;
  vpipe::Session sess;
  try {
    vpipe::CoreMLInferenceStage stage(
        &sess, "infer", {}, std::move(cfg));
    EXPECT_FALSE(stage.config_error().empty());
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_FALSE(threw);
}

TEST(coreml_inference_stage, missing_input_feature_deferred)
{
  vpipe::FlexData cfg = vpipe::FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "model_path",
      vpipe::FlexData::make_string("/tmp/x.mlmodelc"));
  vpipe::FlexData outs = vpipe::FlexData::make_array();
  outs.as_array().push_back(vpipe::FlexData::make_string("y"));
  cfg.as_object().insert_or_assign("output_feature_names",
                                   std::move(outs));

  bool threw = false;
  vpipe::Session sess;
  try {
    vpipe::CoreMLInferenceStage stage(
        &sess, "infer", {}, std::move(cfg));
    EXPECT_FALSE(stage.config_error().empty());
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_FALSE(threw);
}

TEST(coreml_inference_stage, model_path_env_skipped_or_loaded)
{
  // Soft check: if an env model is set, exercise the ctor;
  // otherwise pass trivially. End-to-end pipeline test would sit
  // in tests/api but is skipped here to keep this TU dependency
  // light.
  const char* model = env_or_null_("VPIPE_TEST_COREML_MODEL");
  if (!model) {
    return;
  }
  const char* in_name  = env_or_null_("VPIPE_TEST_COREML_INPUT");
  const char* out_name = env_or_null_("VPIPE_TEST_COREML_OUTPUT");
  if (!in_name || !out_name) {
    return;
  }

  vpipe::FlexData cfg = vpipe::FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "model_path", vpipe::FlexData::make_string(model));
  cfg.as_object().insert_or_assign(
      "input_feature_name", vpipe::FlexData::make_string(in_name));
  vpipe::FlexData outs = vpipe::FlexData::make_array();
  outs.as_array().push_back(vpipe::FlexData::make_string(out_name));
  cfg.as_object().insert_or_assign("output_feature_names",
                                   std::move(outs));

  bool threw = false;
  try {
    vpipe::Session sess;
    vpipe::CoreMLInferenceStage stage(
        &sess, "infer", {}, std::move(cfg));
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_FALSE(threw);
}
