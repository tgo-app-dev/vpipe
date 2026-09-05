#include "generative-models/minimax-h3/vdn-config.h"

#include "common/flex-data.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace vdn {

namespace {

bool
fail_(std::string* err, const std::string& what)
{
  if (err != nullptr) { *err = what; }
  return false;
}

}  // namespace

bool
Config::supported(std::string* why) const
{
  if (delta_rule != "vdn_solve") {
    return fail_(why, "delta_rule '" + delta_rule + "' is not implemented; "
                      "only 'vdn_solve' is. The others are a different "
                      "operator, not a different accuracy");
  }
  if (linear_head_dim <= 0 || linear_head_dim > 128) {
    return fail_(why, "linear_head_dim " + std::to_string(linear_head_dim)
                      + " is outside the 1..128 the kernels are built for");
  }
  return true;
}

int
Adapter::rank_for(const std::string& module) const
{
  auto it = rank_pattern.find(module);
  return it == rank_pattern.end() ? rank : it->second;
}

double
Adapter::alpha_for(const std::string& module) const
{
  auto it = alpha_pattern.find(module);
  return it == alpha_pattern.end() ? alpha : it->second;
}

double
Adapter::scale_for(const std::string& module) const
{
  const int r = rank_for(module);
  return r > 0 ? alpha_for(module) / (double)r : 0.0;
}

bool
parse_config(const FlexData& transform, Config* out, std::string* err)
{
  if (out == nullptr) { return fail_(err, "null out"); }
  FlexData owner = transform;
  if (!owner.is_object()) { return fail_(err, "transform is not an object"); }
  auto t = owner.as_object();
  if (!t.contains("type") || !t.contains("config")) {
    return fail_(err, "transform has no type/config");
  }
  const std::string type(t.at("type").as_string(""));
  if (type != "hybrid_attention") {
    return fail_(err, "transform type '" + type + "' is not hybrid_attention");
  }
  // The version gates the SHAPE of what follows. A future version that
  // renamed a field would otherwise be read with this one's meanings.
  const int version = t.contains("version")
                          ? (int)t.at("version").as_int(0) : 0;
  if (version != 2) {
    return fail_(err, "hybrid_attention version " + std::to_string(version)
                      + " is not 2");
  }
  FlexData cowner = t.at("config");
  if (!cowner.is_object()) { return fail_(err, "config is not an object"); }
  auto c = cowner.as_object();

  if (!c.contains("anchor_frames")
      || !parse_anchor_frames(std::string(c.at("anchor_frames").as_string("")),
                              &out->anchors)) {
    return fail_(err, "anchor_frames missing or unrecognised");
  }
  out->enable_softmax_gate =
      c.contains("enable_softmax_gate")
          ? c.at("enable_softmax_gate").as_bool(true) : true;

  if (!c.contains("softmax_attention")) {
    return fail_(err, "no softmax_attention block");
  }
  FlexData sowner = c.at("softmax_attention");
  auto sa = sowner.as_object();
  out->chunk  = (int)sa.at("chunk").as_int(0);
  out->radius = (int)sa.at("radius").as_int(-1);
  if (out->radius < 0) { return fail_(err, "softmax_attention.radius"); }

  if (!c.contains("linear_attention")) {
    return fail_(err, "no linear_attention block");
  }
  FlexData lowner = c.at("linear_attention");
  auto la = lowner.as_object();
  out->a_fp32 = la.contains("a_fp32") ? la.at("a_fp32").as_bool(true) : true;
  out->enable_text_state =
      la.contains("enable_text_state")
          ? la.at("enable_text_state").as_bool(false) : false;
  out->linear_head_dim =
      la.contains("linear_head_dim")
          ? (int)la.at("linear_head_dim").as_int(0) : 0;
  if (out->linear_head_dim <= 0) {
    return fail_(err, "linear_attention.linear_head_dim");
  }
  const std::string bridge(la.contains("bridge")
                               ? la.at("bridge").as_string("") : "");
  if      (bridge == "alpha") { out->bridge_alpha = true; }
  else if (bridge == "none")  { out->bridge_alpha = false; }
  else { return fail_(err, "linear_attention.bridge '" + bridge + "'"); }
  out->delta_rule =
      std::string(la.contains("delta_rule")
                      ? la.at("delta_rule").as_string("") : "");
  // The three rules the FORMAT defines. Parsing is about whether the
  // file was understood; whether THIS port implements the rule is a
  // separate question and lives in supported(), so that a checkpoint
  // using a rule we cannot run gets a message naming the rule rather
  // than "unparseable".
  if (out->delta_rule != "vdn_solve" && out->delta_rule != "vdn_scaled"
      && out->delta_rule != "sana_scaled") {
    return fail_(err, "linear_attention.delta_rule '" + out->delta_rule
                      + "' is not one of vdn_solve, vdn_scaled, "
                        "sana_scaled");
  }

  // short_conv names the projections the depthwise conv runs on. ABSENT
  // means none -- a conv-less checkpoint is a different parameter set,
  // not a default -- and an unknown name is refused rather than ignored.
  out->conv_q = out->conv_k = out->conv_v = false;
  if (la.contains("short_conv")) {
    FlexData scowner = la.at("short_conv");
    auto sc = scowner.as_object();
    if (sc.contains("targets")) {
      FlexData towner = sc.at("targets");
      auto ta = towner.as_array();
      for (std::size_t i = 0; i < ta.size(); ++i) {
        const std::string n(ta.at(i).as_string(""));
        if      (n == "q") { out->conv_q = true; }
        else if (n == "k") { out->conv_k = true; }
        else if (n == "v") { out->conv_v = true; }
        else { return fail_(err, "short_conv target '" + n + "'"); }
      }
    }
  }
  return true;
}

bool
load_config(const std::string& dir, Config* out, std::string* err)
{
  namespace fs = std::filesystem;
  const fs::path base(dir);
  // Either spelling of "this model": the stage directory or the weights
  // beside it.
  const fs::path direct = base / "config.json";
  const fs::path nested = base / "linear_branch" / "config.json";
  const fs::path spec   = base / "model_spec.json";

  if (fs::exists(direct) || fs::exists(nested)) {
    std::ifstream in(fs::exists(nested) ? nested : direct);
    if (!in) { return fail_(err, "cannot open the linear_branch config"); }
    return parse_config(FlexData::from_json(in), out, err);
  }
  if (!fs::exists(spec)) {
    // THE REPO ROOT IS THE LIKELY MISTAKE, so say so rather than only
    // that nothing was found. OpenVDN publishes both stages from one
    // repo, so a checkout has stage-b-step-2000/ and stage-dmd-step-250/
    // under it and the root carries no config of its own -- and a
    // registry key resolves to that root, since both records share it.
    // Naming the stages that ARE here turns a dead end into the answer.
    std::string found;
    std::error_code ec;
    for (const auto& de : fs::directory_iterator(base, ec)) {
      if (ec) { break; }
      if (!de.is_directory(ec)) { continue; }
      const fs::path sub = de.path();
      if (fs::exists(sub / "model_spec.json", ec)
          || fs::exists(sub / "linear_branch" / "config.json", ec)) {
        if (!found.empty()) { found += ", "; }
        found += sub.filename().string();
      }
    }
    std::string msg = "no config.json, linear_branch/config.json or "
                      "model_spec.json under '" + dir + "'";
    if (!found.empty()) {
      msg += " -- but a stage is one level down (" + found
             + "); name that directory, or a registry key whose record "
               "pins it";
    }
    return fail_(err, msg);
  }
  std::ifstream in(spec);
  if (!in) { return fail_(err, "cannot open model_spec.json"); }
  FlexData doc = FlexData::from_json(in);
  if (!doc.is_object()) { return fail_(err, "model_spec.json is not JSON"); }
  auto root = doc.as_object();
  if (!root.contains("transforms")) {
    return fail_(err, "model_spec.json has no transforms");
  }
  FlexData towner = root.at("transforms");
  auto ta = towner.as_array();
  for (std::size_t i = 0; i < ta.size(); ++i) {
    FlexData e = ta.at(i);
    if (!e.is_object()) { continue; }
    auto eo = e.as_object();
    if (eo.contains("type")
        && std::string(eo.at("type").as_string("")) == "hybrid_attention") {
      return parse_config(e, out, err);
    }
  }
  return fail_(err, "model_spec.json has no hybrid_attention transform");
}

bool
load_adapter(const std::string& dir, Adapter* out, std::string* err)
{
  namespace fs = std::filesystem;
  if (out == nullptr) { return fail_(err, "null out"); }
  const fs::path p = fs::path(dir) / "adapter_config.json";
  std::ifstream in(p);
  if (!in) { return fail_(err, "cannot open " + p.string()); }
  FlexData doc = FlexData::from_json(in);
  if (!doc.is_object()) { return fail_(err, "adapter_config is not JSON"); }
  auto root = doc.as_object();
  if (!root.contains("type")
      || std::string(root.at("type").as_string("")) != "lora") {
    return fail_(err, "adapter type is not lora");
  }
  if (!root.contains("config")) { return fail_(err, "adapter has no config"); }
  FlexData cowner = root.at("config");
  if (!cowner.is_object()) { return fail_(err, "adapter config"); }
  auto c = cowner.as_object();

  out->name = fs::path(dir).filename().string();
  if (c.contains("name")) {
    out->name = std::string(c.at("name").as_string(out->name.c_str()));
  }
  out->rank  = c.contains("rank") ? (int)c.at("rank").as_int(0) : 0;
  out->alpha = c.contains("alpha") ? c.at("alpha").as_real(0.0) : 0.0;
  if (out->rank <= 0) { return fail_(err, "adapter rank"); }
  out->exact_targets =
      c.contains("exact_targets") ? c.at("exact_targets").as_bool(false)
                                  : false;
  if (c.contains("targets")) {
    FlexData towner = c.at("targets");
    auto ta = towner.as_array();
    out->targets.reserve(ta.size());
    for (std::size_t i = 0; i < ta.size(); ++i) {
      out->targets.push_back(std::string(ta.at(i).as_string("")));
    }
  }
  if (c.contains("rank_pattern")) {
    FlexData rp = c.at("rank_pattern");
    auto ro = rp.as_object();
    for (auto e : ro) {
      out->rank_pattern[std::string(e.first)] = (int)e.second.as_int(0);
    }
  }
  if (c.contains("alpha_pattern")) {
    FlexData ap = c.at("alpha_pattern");
    auto ao = ap.as_object();
    for (auto e : ao) {
      out->alpha_pattern[std::string(e.first)] = e.second.as_real(0.0);
    }
  }
  return true;
}

std::vector<std::string>
list_adapters(const std::string& stage_dir)
{
  namespace fs = std::filesystem;
  std::vector<std::string> out;
  std::error_code ec;
  const fs::path root = fs::path(stage_dir) / "adapters";
  if (!fs::is_directory(root, ec)) { return out; }
  for (const auto& e : fs::directory_iterator(root, ec)) {
    if (!e.is_directory()) { continue; }
    if (fs::exists(e.path() / "adapter_config.json")) {
      out.push_back(e.path().filename().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace vdn
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
