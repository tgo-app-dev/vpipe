#include "stages/save-text-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <ios>
#include <string>
#include <utility>

using namespace std;

namespace vpipe {

SaveTextStage::SaveTextStage(const SessionContextIntf* s,
                               string                    id,
                               vector<InEdge>            iports,
                               FlexData                  config)
  : TypedStage<SaveTextStage>(s, std::move(id), std::move(iports),
                               std::move(config))
{
  // Config problems are recorded via fail_config and deferred to launch (the
  // ctor must succeed for any config so a graph can be built/edited).
  _path   = attr_path("path", true);
  _key    = attr_str("key");
  _append = attr_bool("append");
  const string nl = attr_str("newline");
  if (nl == "before") {
    _newline = Newline::Before;
  } else if (nl == "none") {
    _newline = Newline::None;
  } else {
    _newline = Newline::After;   // default + explicit "after"
  }

  if (_path.empty()) {
    fail_config(fmt(
        "SaveTextStage('{}'): config.path is required (non-empty file path)",
        this->id()));
  }
  if (!nl.empty() && nl != "after" && nl != "before" && nl != "none") {
    fail_config(fmt(
        "SaveTextStage('{}'): newline must be one of after|before|none "
        "(got '{}')", this->id(), nl));
  }
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "path", .type = ConfigType::String, .required = true,
   .doc = "output text file path",
   .is_path = true, .path_write = true, .path_filter = "text"},
  {.key = "key", .type = ConfigType::String,
   .doc = "FlexData field to write (a plain-string payload is written whole)",
   .def_str = "text"},
  {.key = "newline", .type = ConfigType::String,
   .doc = "entry separator policy: after (default) | before | none",
   .def_str = "after"},
  {.key = "append", .type = ConfigType::Bool,
   .doc = "append to the file (default); false truncates it at start",
   .def_bool = true},
};
const PortSpec kIports[] = {
  {.name = "text", .doc = "FlexData carrying the text to save (see `key`)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "save-text",
  .doc       = "Sink: appends a text field (config `key`, default \"text\") "
               "from each FlexData beat to a text file, one entry per beat "
               "under the newline policy. EOS ends the stage.",
  .display_name = "Save Text",
  .category  = StageCategory::Text,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};

// The text to write from a FlexData beat: a plain-string payload is used
// whole; an object payload uses field `key` (as a string). Empty when the
// field is missing / not a string.
std::string extract_text_(const FlexData& d, const std::string& key)
{
  if (d.is_string()) { return std::string(d.as_string("")); }
  if (d.is_object() && !key.empty()) {
    auto o = d.as_object();
    if (o.contains(key)) { return std::string(o.at(key).as_string("")); }
  }
  return {};
}
}  // namespace

const StageSpec&
SaveTextStage::spec() const noexcept
{
  return kSpec;
}

Job
SaveTextStage::initialize(RuntimeContext& /*ctx*/)
{
  if (_path.empty()) { co_return; }   // ctor already recorded the config error
  const auto mode =
      std::ios::out | (_append ? std::ios::app : std::ios::trunc);
  _out.open(_path, mode);
  if (!_out.is_open()) {
    session()->error(fmt(
        "SaveTextStage('{}'): cannot open '{}' for writing; the stage is "
        "inert", this->id(), _path));
    co_return;
  }
  const char* nl = _newline == Newline::After ? "newline-after"
                 : _newline == Newline::Before ? "newline-before"
                 : "no-newline";
  session()->info(fmt(
      "SaveTextStage('{}'): writing to '{}' (key='{}', {}, {})",
      this->id(), _path, _key.empty() ? "<whole>" : _key,
      _append ? "append" : "truncate", nl));
  co_return;
}

Job
SaveTextStage::process(RuntimeContext& ctx)
{
  auto t = co_await ctx.read(0);
  if (!t) {
    ctx.signal_done();
    co_return;
  }
  if (!_out.is_open()) { co_return; }   // inert: the file never opened

  const auto* fdp = dynamic_cast<const FlexDataPayload*>(t.get());
  if (fdp == nullptr) {
    session()->warn(fmt(
        "SaveTextStage('{}'): expected FlexDataPayload on in-port 0, got {}; "
        "dropping beat", this->id(), t->describe()));
    co_return;
  }
  const std::string text = extract_text_(fdp->data, _key);
  if (text.empty()) { co_return; }   // nothing to write

  switch (_newline) {
    case Newline::After:
      _out << text << '\n';
      break;
    case Newline::Before:
      if (_wrote_any) { _out << '\n'; }
      _out << text;
      break;
    case Newline::None:
      _out << text;
      break;
  }
  _out.flush();
  _wrote_any = true;
  ++_entries_written;
}

VPIPE_REGISTER_STAGE(SaveTextStage)
VPIPE_REGISTER_SPEC(SaveTextStage, kSpec)

}
