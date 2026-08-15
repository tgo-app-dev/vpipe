#include "generative-models/hidden-state-encoder.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

using std::string;
using std::string_view;

namespace vpipe::genai {

// The built-in architectures. DECLARED here and DEFINED in the per-family
// adapter, so referencing it from the registry is what drags that
// translation unit out of the static library -- a self-registering static
// initialiser in an unreferenced TU is the classic thing a linker drops,
// and it fails as an architecture that is simply absent at run time.
// Takes the registry rather than calling get(): get() is what invokes
// this, so reaching back through it would recurse into a std::call_once
// that is still running -- a deadlock, not a compile error.
void register_builtin_hidden_state_encoders(HiddenStateEncoderRegistry& r);

HiddenStateEncoderRegistry&
HiddenStateEncoderRegistry::get() noexcept
{
  static HiddenStateEncoderRegistry* r = [] {
    auto* p = new HiddenStateEncoderRegistry();
    return p;
  }();
  // Outside the constructor: a builtin's factory may touch the registry,
  // and doing that from inside get()'s init would recurse into a
  // half-built singleton.
  static std::once_flag once;
  std::call_once(once, [p = r] { register_builtin_hidden_state_encoders(*p); });
  return *r;
}

bool
HiddenStateEncoderRegistry::register_arch(string arch,
                                          HiddenStateEncoderFactory f)
{
  if (arch.empty() || !f) { return false; }
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& e : _map) {
    if (e.first == arch) { return false; }   // first wins
  }
  _map.emplace_back(std::move(arch), std::move(f));
  return true;
}

bool
HiddenStateEncoderRegistry::contains(string_view arch) const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& e : _map) {
    if (e.first == arch) { return true; }
  }
  return false;
}

std::vector<string>
HiddenStateEncoderRegistry::architectures() const
{
  std::lock_guard<std::mutex> lk(_mu);
  std::vector<string> out;
  out.reserve(_map.size());
  for (const auto& e : _map) { out.push_back(e.first); }
  return out;
}

std::unique_ptr<HiddenStateEncoder>
HiddenStateEncoderRegistry::create(string_view                   arch,
                                   const HiddenStateEncoderArgs& args) const
{
  HiddenStateEncoderFactory f;
  {
    std::lock_guard<std::mutex> lk(_mu);
    for (const auto& e : _map) {
      if (e.first == arch) { f = e.second; break; }
    }
  }
  if (!f) { return nullptr; }
  // A factory opens multi-GB checkpoints; it is allowed to fail, and a
  // throw here would take down a whole pipeline launch.
  try {
    return f(args);
  } catch (const std::exception& e) {
    if (args.session != nullptr) {
      args.session->warn(fmt("hidden-state encoder '{}': {}", string(arch),
                             e.what()));
    }
  } catch (...) {
    if (args.session != nullptr) {
      args.session->warn(fmt("hidden-state encoder '{}': unknown exception",
                             string(arch)));
    }
  }
  return nullptr;
}

namespace {

// `architectures[0]`, else `model_type`. Both are HuggingFace's, and a
// checkpoint that carries neither is one this cannot identify.
string
arch_of_(const FlexData& cfg)
{
  if (!cfg.is_object()) { return {}; }
  const auto o = cfg.as_object();
  if (o.contains("architectures")) {
    const FlexData& a = o.at("architectures");
    if (a.is_array()) {
      const auto arr = a.as_array();
      if (arr.size() > 0) { return string(arr.at(0).as_string("")); }
    }
  }
  if (o.contains("model_type")) {
    return string(o.at("model_type").as_string(""));
  }
  return {};
}

}  // namespace

std::unique_ptr<HiddenStateEncoder>
HiddenStateEncoderRegistry::open(const HiddenStateEncoderArgs& args,
                                 string*                       err) const
{
  auto fail = [&](string m) -> std::unique_ptr<HiddenStateEncoder> {
    if (err != nullptr) { *err = std::move(m); }
    return nullptr;
  };

  string arch = args.arch;
  if (arch.empty()) { arch = arch_of_(args.config); }
  if (arch.empty()) {
    // Only now go to disk: a caller that handed a config in has already
    // said everything, and a comfy pack has no config.json to read.
    const std::filesystem::path p =
        std::filesystem::path(args.dir) / "config.json";
    std::ifstream in(p);
    if (in) {
      std::ostringstream ss;
      ss << in.rdbuf();
      const FlexData cfg = FlexData::from_json(ss.str());
      arch = arch_of_(cfg);
    }
  }
  if (arch.empty()) {
    return fail("could not identify the architecture of '" + args.dir +
                "': no `arch`, no `architectures`/`model_type` in the "
                "supplied config, and no readable config.json");
  }
  if (!contains(arch)) {
    // NAMING what was found and what is available, because "no encoder"
    // and "an encoder this build was not compiled with" are different
    // problems and only one of them is the caller's to fix.
    string have;
    for (const auto& a : architectures()) {
      have += have.empty() ? "" : ", ";
      have += a;
    }
    return fail("'" + args.dir + "' is a '" + arch + "', which no "
                "hidden-state encoder is registered for (registered: " +
                (have.empty() ? "none" : have) + ")");
  }

  HiddenStateEncoderArgs a = args;
  a.arch = arch;
  std::string ferr;
  if (a.err == nullptr) { a.err = &ferr; }
  auto enc = create(arch, a);
  if (!enc) {
    return fail("the '" + arch + "' hidden-state encoder failed to open '" +
                args.dir + "'" +
                (a.err->empty() ? std::string() : ": " + *a.err));
  }
  return enc;
}

}  // namespace vpipe::genai
