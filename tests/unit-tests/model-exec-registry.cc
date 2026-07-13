// ModelExecRegistry: the LM-model plugin extension point. Register a stub
// arch -> ModelExec factory and verify contains / create / first-wins /
// unknown-arch, without any real weights. The dispatch consult itself
// lives in LoadedLanguageModel (a 3-line lookup before the built-in arch
// cascade); this isolates the registry behavior.

#include "minitest.h"

#include "generative-models/model-exec.h"
#include "generative-models/model-exec-registry.h"
#include "generative-models/model-loader.h"

#include <cstdint>
#include <memory>
#include <span>

using namespace vpipe::genai;

namespace {

// Minimal ModelExec: the 4 pure-virtuals + owns_kv()==true (as every live
// metal exec is), enough for the registry to hand one back.
class StubExec : public ModelExec {
public:
  std::int32_t prefill(ContextId, std::span<const std::int32_t>) override
  {
    return 7;
  }
  std::int32_t decode_one(ContextId, std::int32_t) override { return 7; }
  bool valid() const noexcept override { return true; }
  bool owns_kv() const noexcept override { return true; }
  void set_eval_per_layer(bool) noexcept override {}
};

}  // namespace

TEST(model_exec_registry, register_contains_create)
{
  ModelExecRegistry& reg = ModelExecRegistry::get();

  // Unknown until registered.
  EXPECT_FALSE(reg.contains("StubArchForTest"));

  const bool first = reg.register_arch(
      "StubArchForTest", [](const ModelExecCreateArgs&) {
        return std::unique_ptr<ModelExec>(new StubExec());
      });
  EXPECT_TRUE(first);
  EXPECT_TRUE(reg.contains("StubArchForTest"));

  // First-wins: a second registration for the same arch is ignored.
  const bool second = reg.register_arch(
      "StubArchForTest",
      [](const ModelExecCreateArgs&) { return std::unique_ptr<ModelExec>(); });
  EXPECT_FALSE(second);

  // create() returns the stub for the registered arch.
  ModelConfig cfg;
  cfg.architecture = "StubArchForTest";
  const ModelExecCreateArgs args{
      "/tmp/nonexistent", cfg, nullptr, nullptr, 256u, 16u, false};
  auto exec = reg.create("StubArchForTest", args);
  ASSERT_TRUE(exec != nullptr);
  EXPECT_TRUE(exec->valid());
  EXPECT_TRUE(exec->owns_kv());

  // Unknown arch -> nullptr.
  EXPECT_TRUE(reg.create("NoSuchArchXyz", args) == nullptr);
}

TEST(model_exec_registry, empty_arch_or_null_factory_rejected)
{
  ModelExecRegistry& reg = ModelExecRegistry::get();
  EXPECT_FALSE(reg.register_arch("", [](const ModelExecCreateArgs&) {
    return std::unique_ptr<ModelExec>(new StubExec());
  }));
  EXPECT_FALSE(reg.register_arch("SomeArch", ModelExecFactory{}));
}
