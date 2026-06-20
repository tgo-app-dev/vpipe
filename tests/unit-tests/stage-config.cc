#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "pipeline/stage-config.h"
#include "stages/chrono-stage.h"
#include "stages/vision/detection-overlay-stage.h"

#include <iostream>
#include <span>
#include <streambuf>
#include <string>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

class CerrSilencer {
public:
  CerrSilencer() : _saved(cerr.rdbuf()) { cerr.rdbuf(&_null); }
  ~CerrSilencer() { cerr.rdbuf(_saved); }
private:
  struct NullBuf : public streambuf {
    int overflow(int c) override { return c; }
  };
  streambuf* _saved;
  NullBuf    _null;
};

const ConfigParam*
find_(const vector<ConfigParam>& v, string_view key)
{
  for (const ConfigParam& p : v) {
    if (p.key == key) { return &p; }
  }
  return nullptr;
}

}  // namespace

// ----- free-function layer --------------------------------------------

TEST(stage_config, type_names_are_stable)
{
  EXPECT_TRUE(config_type_name(ConfigType::Bool)   == "bool");
  EXPECT_TRUE(config_type_name(ConfigType::Int)    == "int");
  EXPECT_TRUE(config_type_name(ConfigType::Uint)   == "uint");
  EXPECT_TRUE(config_type_name(ConfigType::Real)   == "real");
  EXPECT_TRUE(config_type_name(ConfigType::String) == "string");
  EXPECT_TRUE(config_type_name(ConfigType::Array)  == "array");
  EXPECT_TRUE(config_type_name(ConfigType::Object) == "object");
  EXPECT_TRUE(config_type_name(ConfigType::Any)    == "any");
}

TEST(stage_config, resolve_uses_default_when_key_absent)
{
  constexpr ConfigKey spec[] = {
    {.key = "n",    .type = ConfigType::Int,    .def_int = 7},
    {.key = "name", .type = ConfigType::String, .def_str = "hi"},
    {.key = "on",   .type = ConfigType::Bool,   .def_bool = true},
  };
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("n", FlexData::make_int(42));

  auto params = resolve_config_params(spec, cfg);
  EXPECT_TRUE(params.size() == 3u);

  const ConfigParam* n = find_(params, "n");
  EXPECT_TRUE(n != nullptr);
  EXPECT_TRUE(n->type == ConfigType::Int);
  EXPECT_TRUE(n->default_value.as_int() == 7);
  EXPECT_TRUE(n->current_value.as_int() == 42);   // overridden

  const ConfigParam* name = find_(params, "name");
  EXPECT_TRUE(name != nullptr);
  EXPECT_TRUE(name->current_value.as_string() == "hi");   // default

  const ConfigParam* on = find_(params, "on");
  EXPECT_TRUE(on != nullptr);
  EXPECT_TRUE(on->current_value.as_bool() == true);       // default
}

TEST(stage_config, resolve_non_object_config_yields_all_defaults)
{
  constexpr ConfigKey spec[] = {
    {.key = "n", .type = ConfigType::Int, .def_int = 3},
  };
  FlexData cfg = FlexData::make_null();
  auto params = resolve_config_params(spec, cfg);
  EXPECT_TRUE(params.size() == 1u);
  EXPECT_TRUE(params[0].current_value.as_int() == 3);
}

TEST(stage_config, required_key_defaults_to_null)
{
  constexpr ConfigKey spec[] = {
    {.key = "path", .type = ConfigType::String, .required = true},
  };
  auto params = resolve_config_params(spec, FlexData::make_object());
  EXPECT_TRUE(params.size() == 1u);
  EXPECT_TRUE(params[0].required);
  EXPECT_TRUE(params[0].default_value.is_null());
  EXPECT_TRUE(params[0].current_value.is_null());
}

TEST(stage_config, params_to_flex_shape)
{
  constexpr ConfigKey spec[] = {
    {.key = "n", .type = ConfigType::Int, .doc = "a count", .def_int = 1},
    {.key = "q", .type = ConfigType::String, .required = true},
  };
  FlexData arr = config_params_to_flex(
      resolve_config_params(spec, FlexData::make_object()));
  EXPECT_TRUE(arr.is_array());
  auto av = arr.as_array();
  EXPECT_TRUE(av.size() == 2u);

  FlexData e0 = av.at(0);
  auto o0 = e0.as_object();
  EXPECT_TRUE(o0.at("key").as_string() == "n");
  EXPECT_TRUE(o0.at("type").as_string() == "int");
  EXPECT_TRUE(o0.at("required").as_bool() == false);
  EXPECT_TRUE(o0.contains("doc"));
  EXPECT_TRUE(o0.at("doc").as_string() == "a count");
  EXPECT_TRUE(o0.at("default").as_int() == 1);
  EXPECT_TRUE(o0.at("current").as_int() == 1);

  FlexData e1 = av.at(1);
  auto o1 = e1.as_object();
  EXPECT_TRUE(o1.at("required").as_bool() == true);
  EXPECT_TRUE(!o1.contains("doc"));     // empty doc omitted
  EXPECT_TRUE(o1.at("current").is_null());
}

// The suggest_db UI hint flows from the schema into the serialized
// config (so the editor can offer a key-suggestion dropdown); it is
// omitted when unset.
TEST(stage_config, suggest_db_surfaces_in_schema)
{
  constexpr ConfigKey spec[] = {
    {.key = "model", .type = ConfigType::String, .suggest_db = "models"},
    {.key = "plain", .type = ConfigType::String},
  };
  FlexData arr = config_params_to_flex(
      resolve_config_params(spec, FlexData::make_object()));
  auto av = arr.as_array();
  EXPECT_TRUE(av.size() == 2u);

  FlexData e0 = av.at(0);
  EXPECT_TRUE(e0.as_object().contains("suggest_db"));
  EXPECT_TRUE(e0.as_object().at("suggest_db").as_string() == "models");

  FlexData e1 = av.at(1);
  EXPECT_TRUE(!e1.as_object().contains("suggest_db"));   // unset -> omit
}

// suggest_db_type (the model_type the suggestion dropdown filters on)
// surfaces alongside suggest_db, and is omitted when unset.
TEST(stage_config, suggest_db_type_surfaces_in_schema)
{
  constexpr ConfigKey spec[] = {
    {.key = "vis", .type = ConfigType::String, .suggest_db = "models",
     .suggest_db_type = "yolo"},
    {.key = "any", .type = ConfigType::String, .suggest_db = "models"},
  };
  FlexData arr = config_params_to_flex(
      resolve_config_params(spec, FlexData::make_object()));
  auto av = arr.as_array();
  EXPECT_TRUE(av.size() == 2u);

  FlexData e0 = av.at(0);
  EXPECT_TRUE(e0.as_object().contains("suggest_db_type"));
  EXPECT_TRUE(e0.as_object().at("suggest_db_type").as_string() == "yolo");

  FlexData e1 = av.at(1);
  EXPECT_TRUE(!e1.as_object().contains("suggest_db_type"));  // unset -> omit
}

// ----- real stages ----------------------------------------------------

TEST(stage_config, chrono_reports_keys_and_current_values)
{
  CerrSilencer hush;
  Session sess;

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("period_seconds", FlexData::make_real(2.0));
  cfg.as_object().insert("count", FlexData::make_int(5));

  ChronoStage st(&sess, "c", {}, std::move(cfg));
  auto params = st.config_params();
  EXPECT_TRUE(params.size() == 6u);

  const ConfigParam* ps = find_(params, "period_seconds");
  EXPECT_TRUE(ps != nullptr);
  EXPECT_TRUE(ps->type == ConfigType::Real);
  EXPECT_TRUE(ps->current_value.as_real() == 2.0);
  EXPECT_TRUE(ps->present == true);

  const ConfigParam* cnt = find_(params, "count");
  EXPECT_TRUE(cnt != nullptr);
  EXPECT_TRUE(cnt->current_value.as_int() == 5);
  EXPECT_TRUE(cnt->present == true);

  // A key the stage understands but the instance did not set: the
  // resolver fills `current_value` from the declared default, and the
  // `present` flag is false so the web-ui editor can tell apart "I
  // explicitly set 0" from "I never set this key" (essential for
  // chrono's mutually-exclusive frequency_hz vs period_* validation).
  const ConfigParam* hz = find_(params, "frequency_hz");
  EXPECT_TRUE(hz != nullptr);
  EXPECT_TRUE(hz->current_value.as_real() == 0.0);
  EXPECT_TRUE(hz->present == false);
}

TEST(stage_config, detection_overlay_defaults_and_overrides)
{
  CerrSilencer hush;
  Session sess;

  // Empty config -> every key resolves to its declared default.
  {
    DetectionOverlayStage st(&sess, "o", {}, FlexData::make_object());
    auto params = st.config_params();
    EXPECT_TRUE(!params.empty());
    const ConfigParam* top_n = find_(params, "top_n");
    EXPECT_TRUE(top_n != nullptr);
    EXPECT_TRUE(top_n->current_value.as_int() == 10);
    const ConfigParam* dl = find_(params, "draw_label");
    EXPECT_TRUE(dl != nullptr);
    EXPECT_TRUE(dl->current_value.as_bool() == true);
  }

  // Override a couple of keys.
  {
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("top_n", FlexData::make_int(3));
    cfg.as_object().insert("draw_label", FlexData::make_bool(false));
    DetectionOverlayStage st(&sess, "o", {}, std::move(cfg));
    auto params = st.config_params();
    EXPECT_TRUE(find_(params, "top_n")->current_value.as_int() == 3);
    EXPECT_TRUE(find_(params, "draw_label")->current_value.as_bool()
                == false);
  }
}

TEST(stage_config, schema_json_round_trips)
{
  CerrSilencer hush;
  Session sess;
  DetectionOverlayStage st(&sess, "o", {}, FlexData::make_object());
  string js = st.config_schema().to_json();
  EXPECT_TRUE(!js.empty());
  FlexData parsed = FlexData::from_json(js);
  EXPECT_TRUE(parsed.is_array());
  EXPECT_TRUE(parsed.as_array().size() == st.config_spec().size());
}
