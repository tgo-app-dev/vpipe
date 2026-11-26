#include "generative-models/magi2/magi2-layout.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>

namespace vpipe {
namespace genai {
namespace magi2 {

namespace fs = std::filesystem;

namespace {

bool
resolve_(const std::string& root, Layout* out)
{
  std::error_code ec;
  const fs::path r(root);
  if (root.empty() || !fs::is_directory(r, ec)) { return false; }
  const fs::path index = r / "preview" / "model.safetensors.index.json";
  if (!fs::is_regular_file(index, ec) ||
      !fs::is_regular_file(r / "turbo_vae" / "checkpoint.ckpt", ec) ||
      !fs::is_regular_file(r / "stable-audio-open-1.0" / "model_config.json",
                           ec) ||
      !fs::is_regular_file(r / "text_encoder" / "config.json", ec)) {
    return false;
  }
  // The index names the preview's tensors: its pre-adapter and its
  // multi-head MoE's router.
  std::ifstream in(index);
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string s = ss.str();
  if (s.find("pre_adapter.video_embedder.weight") == std::string::npos ||
      s.find("mlp.moe_mlp.router.expert_bias_ema") == std::string::npos) {
    return false;
  }
  if (out != nullptr) {
    Layout l;
    l.root = r.string();
    l.preview = (r / "preview").string();
    l.text_encoder = (r / "text_encoder").string();
    l.turbo = (r / "turbo_vae").string();
    l.audio = (r / "stable-audio-open-1.0").string();
    if (fs::is_regular_file(r / "refiner" / "model.safetensors.index.json",
                            ec)) {
      l.refiner = (r / "refiner").string();
    }
    if (fs::is_regular_file(r / "vae" / "Wan2.2_VAE.pth", ec)) {
      l.wan_vae = (r / "vae" / "Wan2.2_VAE.pth").string();
    }
    *out = std::move(l);
  }
  return true;
}

}  // namespace

bool
resolve_layout(const std::string& root, Layout* out)
{
  // Memoized: every family's claims() runs on every model resolve.
  static std::mutex mu;
  static std::map<std::string, bool> seen;
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = seen.find(root);
    if (it != seen.end() && !it->second) { return false; }
    if (it != seen.end() && out == nullptr) { return true; }
  }
  const bool ok = resolve_(root, out);
  std::lock_guard<std::mutex> lk(mu);
  seen[root] = ok;
  return ok;
}

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe
