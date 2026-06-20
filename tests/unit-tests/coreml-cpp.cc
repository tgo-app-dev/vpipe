// Tests for the CoreML C++ wrapper. None of these tests require an
// .mlmodelc file; they exercise the wrapper plumbing only.
//
// The inference-stage end-to-end test that actually loads a model
// lives in coreml-inference-stage.cc and skips when no model path is
// supplied via VPIPE_TEST_COREML_MODEL.

#include "minitest.h"

#include "apple-silicon/coreml/coreml-cpp/CoreML.hpp"

#include <cstring>

namespace {

// RAII pool around each test body. CoreML and Foundation factories
// hand out autoreleased objects; the pool drains them on test exit.
struct AutoPool {
  NS::AutoreleasePool* p;
  AutoPool() : p(NS::AutoreleasePool::alloc()->init()) {}
  ~AutoPool() { p->release(); }
};

}

TEST(coreml_cpp, model_configuration_round_trip)
{
  AutoPool pool;

  auto* cfg = CML::ModelConfiguration::alloc()->init();
  ASSERT_TRUE(cfg != nullptr);

  cfg->setComputeUnits(CML::ComputeUnitsCPUOnly);
  EXPECT_TRUE(cfg->computeUnits() == CML::ComputeUnitsCPUOnly);

  cfg->setComputeUnits(CML::ComputeUnitsCPUAndNeuralEngine);
  EXPECT_TRUE(cfg->computeUnits() ==
              CML::ComputeUnitsCPUAndNeuralEngine);

  cfg->setAllowLowPrecisionAccumulationOnGPU(true);
  EXPECT_TRUE(cfg->allowLowPrecisionAccumulationOnGPU());

  cfg->setAllowLowPrecisionAccumulationOnGPU(false);
  EXPECT_FALSE(cfg->allowLowPrecisionAccumulationOnGPU());

  cfg->release();
}

TEST(coreml_cpp, prediction_options_round_trip)
{
  AutoPool pool;

  auto* opts = CML::PredictionOptions::alloc()->init();
  ASSERT_TRUE(opts != nullptr);

  opts->setUsesCPUOnly(true);
  EXPECT_TRUE(opts->usesCPUOnly());

  opts->setUsesCPUOnly(false);
  EXPECT_FALSE(opts->usesCPUOnly());

  opts->release();
}

TEST(coreml_cpp, multi_array_alloc_and_data_path)
{
  AutoPool pool;

  // shape = [1, 3, 2, 2]; 12 float32 elements.
  const NS::Object* dims[4] = {
    NS::Number::number(1ll),
    NS::Number::number(3ll),
    NS::Number::number(2ll),
    NS::Number::number(2ll),
  };
  NS::Array* shape = NS::Array::array(dims, 4);

  NS::Error* err = nullptr;
  auto* multi = CML::MultiArray::alloc()->initWithShape(
      shape, CML::MultiArrayDataTypeFloat32, &err);
  ASSERT_TRUE(multi != nullptr);
  ASSERT_TRUE(err == nullptr);

  EXPECT_TRUE(multi->dataType() == CML::MultiArrayDataTypeFloat32);
  EXPECT_TRUE(multi->shape()->count() == 4);

  // Stomp known values into the buffer; read them back.
  float* data = static_cast<float*>(multi->dataPointer());
  ASSERT_TRUE(data != nullptr);
  for (int i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i) + 0.5f;
  }
  for (int i = 0; i < 12; ++i) {
    EXPECT_TRUE(data[i] == static_cast<float>(i) + 0.5f);
  }

  multi->release();
}

TEST(coreml_cpp, feature_value_int_double_string)
{
  AutoPool pool;

  auto* fv_i = CML::FeatureValue::featureValueWithInt64(42);
  ASSERT_TRUE(fv_i != nullptr);
  EXPECT_TRUE(fv_i->type() == CML::FeatureTypeInt64);
  EXPECT_TRUE(fv_i->int64Value() == 42);

  auto* fv_d = CML::FeatureValue::featureValueWithDouble(3.5);
  ASSERT_TRUE(fv_d != nullptr);
  EXPECT_TRUE(fv_d->type() == CML::FeatureTypeDouble);
  EXPECT_TRUE(fv_d->doubleValue() == 3.5);

  auto* nstr = NS::String::string("hello", NS::UTF8StringEncoding);
  auto* fv_s = CML::FeatureValue::featureValueWithString(nstr);
  ASSERT_TRUE(fv_s != nullptr);
  EXPECT_TRUE(fv_s->type() == CML::FeatureTypeString);
}

TEST(coreml_cpp, feature_value_with_multi_array)
{
  AutoPool pool;

  const NS::Object* dims[2] = {
    NS::Number::number(2ll),
    NS::Number::number(3ll),
  };
  NS::Array* shape = NS::Array::array(dims, 2);

  NS::Error* err = nullptr;
  auto* multi = CML::MultiArray::alloc()->initWithShape(
      shape, CML::MultiArrayDataTypeFloat32, &err);
  ASSERT_TRUE(multi != nullptr);

  auto* fv = CML::FeatureValue::featureValueWithMultiArray(multi);
  ASSERT_TRUE(fv != nullptr);
  EXPECT_TRUE(fv->type() == CML::FeatureTypeMultiArray);

  auto* round = fv->multiArrayValue();
  ASSERT_TRUE(round != nullptr);
  EXPECT_TRUE(round->shape()->count() == 2);

  multi->release();
}

TEST(coreml_cpp, dictionary_feature_provider_round_trip)
{
  AutoPool pool;

  // {"x": <Int64 7>}
  auto* fv  = CML::FeatureValue::featureValueWithInt64(7);
  auto* key = NS::String::string("x", NS::UTF8StringEncoding);

  const NS::Object* objs[1] = { fv };
  const NS::Object* keys[1] = { key };
  NS::Dictionary* dict = NS::Dictionary::dictionary(objs, keys, 1);

  NS::Error* err = nullptr;
  auto* dfp = CML::DictionaryFeatureProvider::alloc()
                  ->initWithDictionary(dict, &err);
  ASSERT_TRUE(dfp != nullptr);
  ASSERT_TRUE(err == nullptr);

  auto* names = dfp->featureNames();
  ASSERT_TRUE(names != nullptr);
  EXPECT_TRUE(names->count() == 1);

  auto* got = dfp->featureValueForName(key);
  ASSERT_TRUE(got != nullptr);
  EXPECT_TRUE(got->int64Value() == 7);

  dfp->release();
}

TEST(coreml_cpp, model_load_bogus_url_returns_error)
{
  AutoPool pool;

  // /tmp/no-such-model.mlmodelc -- never present.
  auto* path = NS::String::string("/tmp/no-such-model.mlmodelc",
                                  NS::UTF8StringEncoding);
  auto* url  = NS::URL::fileURLWithPath(path);

  NS::Error* err = nullptr;
  auto* model = CML::Model::modelWithContentsOfURL(url, &err);

  EXPECT_TRUE(model == nullptr);
  EXPECT_TRUE(err   != nullptr);
}
