// unclaimed-checkpoint.cc -- a checkpoint nothing here runs is REFUSED.
//
// generate-image and diffusion-conditioner both fall through to "krea2"
// when nothing names a family, and Krea-2's text encoder is a Qwen3-VL
// several families share -- so a checkpoint whose family lives in a
// plugin nobody loaded would load the Krea-2 path at full cost and run a
// different model. Each stage therefore refuses a transformer
// `_class_name` that neither its own table nor a registered family
// claims, and says which class it was, so the fix (load the plugin that
// provides it) is in the message. Model-free: the refusal happens before
// any weight is read.
//
//   vpipe_test --filter 'unclaimed_checkpoint*'

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "interfaces/ui-delegate-intf.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/generate-image-stage.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

const char* const kClass = "UtUnclaimedTransformer2DModel";

struct Errors : public UiDelegateIntf {
  std::mutex               mu;
  std::vector<std::string> errors;
  void
  error(const VpipeFormat& f) override
  {
    std::lock_guard<std::mutex> lk(mu);
    errors.push_back(f());
  }
  void warn(const VpipeFormat&) override {}
  void info(const VpipeFormat&) override {}
  UiInputStatus
  getline(const VpipeFormat&, std::string&,
          const std::function<bool()>&) override
  {
    return UiInputStatus::Eof;
  }
  std::unique_ptr<UiTextStream>
  open_text_stream() override
  {
    return std::make_unique<NullUiTextStream>();
  }
};

std::filesystem::path
root_()
{
  const auto root =
      std::filesystem::temp_directory_path() / "vpipe-ut-unclaimed";
  std::filesystem::create_directories(root / "transformer");
  std::ofstream(root / "transformer" / "config.json")
      << "{\"_class_name\": \"" << kClass << "\"}";
  return root;
}

// Whether the stage refused naming the class. Launches a one-stage
// pipeline: the refusal is in initialize(), which the launch runs.
template <class StageT>
bool
refused_(const std::filesystem::path& root)
{
  Session sess;
  auto ui = std::make_unique<Errors>();
  Errors* errs = ui.get();
  sess.set_ui_delegate(std::move(ui));
  auto pl = std::make_unique<Pipeline>("p", &sess);
  FlexData c = FlexData::make_object();
  c.as_object().insert_or_assign("hf_dir",
                                 FlexData::make_string(root.string()));
  pl->insert_stage(std::make_unique<StageT>(&sess, "s",
                                            std::vector<InEdge>{},
                                            std::move(c)));
  {
    PipelineRuntime rt(pl.get(), &sess);
    if (rt.launch()) {
      rt.wait_idle();
      rt.stop();
    }
  }
  std::lock_guard<std::mutex> lk(errs->mu);
  for (const std::string& e : errs->errors) {
    if (e.find(kClass) != std::string::npos &&
        e.find("no loaded plugin claims") != std::string::npos) {
      return true;
    }
  }
  for (const std::string& e : errs->errors) {
    std::printf("[unclaimed_checkpoint] error: %s\n", e.c_str());
  }
  return false;
}

}  // namespace

TEST(unclaimed_checkpoint, generate_image_refuses_an_unknown_class)
{
  const auto root = root_();
  EXPECT_TRUE(refused_<GenerateImageStage>(root));
  std::filesystem::remove_all(root);
}

TEST(unclaimed_checkpoint, the_conditioner_refuses_an_unknown_class)
{
  const auto root = root_();
  EXPECT_TRUE(refused_<DiffusionConditionerStage>(root));
  std::filesystem::remove_all(root);
}
