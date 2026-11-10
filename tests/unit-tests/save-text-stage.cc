// Unit + pipeline tests for the save-text sink stage. Self-contained (no
// model): a tiny FlexData source feeds beats, save-text saves the extracted
// field to a temp file, and the file content is checked per newline policy.

#include "minitest.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/save-text-stage.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Emits a fixed list of FlexData beats (in order) then signals done.
class FlexSource : public TypedStage<FlexSource> {
public:
  static constexpr const char* kTypeName = "ut-flex-source";
  using TypedStage::TypedStage;
  std::vector<FlexData> beats;
  Job process(RuntimeContext& ctx) override
  {
    if (_i >= beats.size()) { ctx.signal_done(); co_return; }
    FlexData fd = beats[_i++];
    co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(fd)));
  }
private:
  std::size_t _i = 0;
};

FlexData text_obj_(const std::string& key, const std::string& val)
{
  FlexData d = FlexData::make_object();
  d.as_object().insert(key, FlexData::make_string(val));
  return d;
}

std::string tmp_path_(const char* name)
{
  return (std::filesystem::temp_directory_path() / name).string();
}

// Run a source -> save-text pipeline to completion and return the file text.
std::string run_(Session& sess, std::vector<FlexData> beats,
                 const std::string& path, const char* newline,
                 const char* key = "text", bool append = false,
                 const char* format = "text")
{
  Pipeline pl("tw", &sess);
  auto src = make_unique<FlexSource>(&sess, "src", vector<InEdge>{},
                                     FlexData::make_object());
  src->beats = std::move(beats);
  src->allocate_oports(1);
  auto* s = static_cast<FlexSource*>(pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("path", FlexData::make_string(path));
    o.insert("key", FlexData::make_string(key));
    o.insert("newline", FlexData::make_string(newline));
    o.insert("append", FlexData::make_bool(append));
    o.insert("format", FlexData::make_string(format));
  }
  auto tw = make_unique<SaveTextStage>(&sess, "tw",
                                        vector<InEdge>{ { s, 0 } },
                                        std::move(cfg));
  pl.insert_stage(std::move(tw));

  PipelineRuntime rt(&pl, &sess);
  if (!rt.launch()) { return "<launch-failed>"; }
  rt.wait_idle();
  rt.stop();
  std::ifstream in(path, std::ios::binary);
  std::ostringstream buf; buf << in.rdbuf();
  return buf.str();
}

}  // namespace

TEST(save_text_stage, type_is_registered) {
  Session sess;
  Pipeline pl("p", &sess);
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("path", FlexData::make_string("/tmp/x.txt"));
  auto* s = pl.insert_stage("save-text", "w", {}, std::move(cfg));
  EXPECT_TRUE(s != nullptr);
}

TEST(save_text_stage, missing_path_deferred) {
  Session sess;
  SaveTextStage s(&sess, "w", vector<InEdge>{}, FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());   // path required
}

TEST(save_text_stage, bad_newline_deferred) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("path", FlexData::make_string("/tmp/x.txt"));
    o.insert("newline", FlexData::make_string("sideways"));
  }
  SaveTextStage s(&sess, "w", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(save_text_stage, newline_after_one_entry_per_line) {
  Session sess;
  const std::string p = tmp_path_("vpipe-tw-after.txt");
  std::error_code ec; std::filesystem::remove(p, ec);
  const std::string got =
      run_(sess, { text_obj_("text", "hello"), text_obj_("text", "world") },
           p, "after");
  EXPECT_TRUE(got == "hello\nworld\n");
  std::filesystem::remove(p, ec);
}

TEST(save_text_stage, newline_before_no_leading_newline) {
  Session sess;
  const std::string p = tmp_path_("vpipe-tw-before.txt");
  std::error_code ec; std::filesystem::remove(p, ec);
  const std::string got =
      run_(sess, { text_obj_("text", "a"), text_obj_("text", "b"),
                   text_obj_("text", "c") }, p, "before");
  EXPECT_TRUE(got == "a\nb\nc");
  std::filesystem::remove(p, ec);
}

TEST(save_text_stage, newline_none_concatenates) {
  Session sess;
  const std::string p = tmp_path_("vpipe-tw-none.txt");
  std::error_code ec; std::filesystem::remove(p, ec);
  const std::string got =
      run_(sess, { text_obj_("text", "foo"), text_obj_("text", "bar") },
           p, "none");
  EXPECT_TRUE(got == "foobar");
  std::filesystem::remove(p, ec);
}

TEST(save_text_stage, custom_key_and_empty_skipped) {
  Session sess;
  const std::string p = tmp_path_("vpipe-tw-key.txt");
  std::error_code ec; std::filesystem::remove(p, ec);
  // key = "msg"; a beat missing "msg" and a beat with an empty "msg" are
  // both skipped (no blank lines).
  const std::string got =
      run_(sess, { text_obj_("msg", "one"), text_obj_("text", "ignored"),
                   text_obj_("msg", ""), text_obj_("msg", "two") },
           p, "after", /*key=*/"msg");
  EXPECT_TRUE(got == "one\ntwo\n");
  std::filesystem::remove(p, ec);
}

TEST(save_text_stage, plain_string_payload_written_whole) {
  Session sess;
  const std::string p = tmp_path_("vpipe-tw-str.txt");
  std::error_code ec; std::filesystem::remove(p, ec);
  // A plain-string FlexData payload is written whole, regardless of `key`.
  const std::string got =
      run_(sess, { FlexData::make_string("bare line") }, p, "after");
  EXPECT_TRUE(got == "bare line\n");
  std::filesystem::remove(p, ec);
}

// format json: the field `key` as compact JSON when the beat has it, else
// the whole beat -- one JSON value a line, read back as written.
TEST(save_text_stage, json_writes_the_field_or_the_beat) {
  Session sess;
  const std::string p = tmp_path_("vpipe-tw-json.txt");
  std::error_code ec; std::filesystem::remove(p, ec);
  FlexData whole = FlexData::make_object();
  {
    auto o = whole.as_object();
    FlexData a = FlexData::make_array();
    a.as_array().push_back(FlexData::make_real(0.5));
    a.as_array().push_back(FlexData::make_real(1.25));
    o.insert("beats", std::move(a));
    o.insert("tempo", FlexData::make_real(117.45));
  }
  FlexData field = FlexData::make_object();
  {
    FlexData inner = FlexData::make_object();
    inner.as_object().insert("n", FlexData::make_int(7));
    field.as_object().insert("text", std::move(inner));
    field.as_object().insert("other", FlexData::make_int(1));
  }
  const std::string got =
      run_(sess, { whole, field }, p, "after", "text", false, "json");
  std::istringstream lines(got);
  std::string l1, l2, l3;
  std::getline(lines, l1);
  std::getline(lines, l2);
  const bool more = (bool)std::getline(lines, l3);
  std::istringstream j1(l1), j2(l2);
  const FlexData r1 = FlexData::from_json(j1);
  const FlexData r2 = FlexData::from_json(j2);
  EXPECT_TRUE(!more);
  EXPECT_TRUE(r1.is_object() && r1.as_object().contains("tempo") &&
              r1.as_object().at("tempo").as_real(0.0) == 117.45);
  EXPECT_TRUE(r1.as_object().at("beats").as_array().size() == 2);
  EXPECT_TRUE(r2.is_object() && r2.as_object().contains("n") &&
              !r2.as_object().contains("other"));
  std::filesystem::remove(p, ec);

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("path", FlexData::make_string("/tmp/x.txt"));
  cfg.as_object().insert("format", FlexData::make_string("yaml"));
  SaveTextStage bad(&sess, "w", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(bad.config_error().empty());
}
