#include "generative-models/shared/comfy-checkpoint.h"

#include "common/flex-data.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace vpipe {
namespace genai {
namespace comfy {
namespace {

namespace fs = std::filesystem;

// The safetensors header is capped so a corrupt or non-safetensors file
// cannot ask for a multi-gigabyte allocation. Real ones here are ~100 KB
// (the MiniMax-H3 DiT's is 57 KB over 535 tensors).
constexpr std::uint64_t kMaxHeader = 64ull << 20;

std::string
lower_(std::string s)
{
  for (char& c : s) {
    c = (char)std::tolower((unsigned char)c);
  }
  return s;
}

bool
fail_(std::string* err, std::string m)
{
  if (err != nullptr) { *err = std::move(m); }
  return false;
}

}  // namespace

bool
read_metadata(const std::string& file, FlexData& out, std::string* err)
{
  std::error_code ec;
  if (!fs::is_regular_file(fs::path(file), ec) || ec) {
    return fail_(err, file + " is not a file");
  }
  std::ifstream in(file, std::ios::binary);
  if (!in) { return fail_(err, "cannot open " + file); }
  std::uint64_t len = 0;
  if (!in.read(reinterpret_cast<char*>(&len), 8)) {
    return fail_(err, file + " is too short to be safetensors");
  }
  if (len == 0 || len > kMaxHeader) {
    return fail_(err, file + " has an implausible safetensors header");
  }
  std::string header((std::size_t)len, '\0');
  if (!in.read(header.data(), (std::streamsize)len)) {
    return fail_(err, file + ": truncated safetensors header");
  }
  FlexData j;
  try {
    j = FlexData::from_json(header);
  } catch (...) {
    return fail_(err, file + ": cannot parse the safetensors header");
  }
  if (!j.is_object()) { return fail_(err, file + ": header is not an object"); }
  auto o = j.as_object();
  if (!o.contains("__metadata__")) {
    return fail_(err, file + " carries no __metadata__");
  }
  FlexData md = o.at("__metadata__");
  if (!md.is_object()) {
    return fail_(err, file + ": __metadata__ is not an object");
  }
  out = std::move(md);
  return true;
}

bool
metadata_json(const std::string& file, const std::string& key, FlexData& out,
              std::string* err)
{
  FlexData md;
  if (!read_metadata(file, md, err)) { return false; }
  auto o = md.as_object();
  if (!o.contains(key)) {
    return fail_(err, file + ": no __metadata__ key '" + key + "'");
  }
  // Bind the entry before taking a string_view of it: as_string() is a
  // VIEW into the FlexData, so chaining off at() would dangle.
  const FlexData v = o.at(key);
  if (!v.is_string()) {
    return fail_(err, file + ": __metadata__['" + key + "'] is not a string");
  }
  const std::string text(v.as_string(""));
  FlexData parsed;
  try {
    parsed = FlexData::from_json(text);
  } catch (...) {
    return fail_(err, file + ": __metadata__['" + key + "'] is not JSON");
  }
  out = std::move(parsed);
  return true;
}

bool
is_component(const std::string& path, const std::string& key)
{
  std::error_code ec;
  const fs::path p(path);
  if (!fs::is_regular_file(p, ec) || ec) { return false; }
  if (lower_(p.extension().string()) != ".safetensors") { return false; }
  FlexData md;
  if (!read_metadata(path, md, nullptr)) { return false; }
  return md.as_object().contains(key);
}

bool
unsupported_quant(const std::string& file)
{
  const std::string n = lower_(fs::path(file).filename().string());
  static const char* kMarkers[] = {"int8_convrot", "fp8_scaled", "fp8_e4m3",
                                   "fp8_e5m2",     "nvfp4",      "int4",
                                   "gguf"};
  for (const char* m : kMarkers) {
    if (n.find(m) != std::string::npos) { return true; }
  }
  return false;
}

std::string
resolve_component(const std::string& path, const std::string& subdir,
                  const std::string& meta_key,
                  const std::vector<std::string>& prefer)
{
  std::error_code ec;
  const fs::path p(path);
  // Handed the file itself -- the only case that accepts an unsupported
  // packing's NAME, because an explicit path is a deliberate choice and
  // the metadata check below still has to pass.
  if (fs::is_regular_file(p, ec) && !ec) {
    return is_component(path, meta_key) ? path : std::string();
  }
  if (!fs::is_directory(p, ec) || ec) { return {}; }

  // The component subdirectory, then the repo root holding it. A root
  // whose subdir is absent is not a Comfy-Org repo.
  std::vector<fs::path> dirs;
  if (p.filename() == subdir) { dirs.push_back(p); }
  const fs::path sub = p / subdir;
  if (fs::is_directory(sub, ec) && !ec) { dirs.push_back(sub); }
  if (dirs.empty()) { return {}; }

  // Rank: preferred-substring index first (earliest wins), then the
  // SHORTEST filename. The length tiebreak is what picks the plain
  // conversion over a variant of it -- `minimax_h3_fl2va_bf16` over
  // `minimax_h3_fl2va_pruned_bf16` -- without this file having to know
  // every adjective Comfy-Org might add.
  std::string best;
  std::size_t best_rank = prefer.size() + 1;
  for (const fs::path& dir : dirs) {
    std::error_code lec;
    for (const auto& de : fs::directory_iterator(dir, lec)) {
      const fs::path& f = de.path();
      if (lower_(f.extension().string()) != ".safetensors") { continue; }
      const std::string name = lower_(f.filename().string());
      if (unsupported_quant(name)) { continue; }
      std::size_t rank = prefer.size();
      for (std::size_t i = 0; i < prefer.size(); ++i) {
        if (name.find(lower_(prefer[i])) != std::string::npos) {
          rank = i;
          break;
        }
      }
      if (rank > best_rank) { continue; }
      if (rank == best_rank && !best.empty() &&
          fs::path(best).filename().string().size() <= name.size()) {
        continue;
      }
      // The metadata read is last: it opens the file, and every cheap
      // rejection above has already run.
      if (!is_component(f.string(), meta_key)) { continue; }
      best = f.string();
      best_rank = rank;
    }
  }
  return best;
}

std::vector<Component>
scan_repo(const std::string& root)
{
  // The subdirectories Comfy-Org publishes under. Fixed rather than "any
  // subdir with .safetensors in it": the role IS the directory name, so
  // accepting an unknown one would invent a role no caller can act on.
  static const char* kRoles[] = {"diffusion_models", "text_encoders", "vae",
                                 "clip_vision", "loras"};
  std::vector<Component> out;
  std::error_code ec;
  const fs::path p(root);
  if (!fs::is_directory(p, ec) || ec) { return out; }
  for (const char* role : kRoles) {
    const fs::path dir = p / role;
    if (!fs::is_directory(dir, ec) || ec) { continue; }
    std::vector<fs::path> files;
    std::error_code lec;
    for (const auto& de : fs::directory_iterator(dir, lec)) {
      const fs::path& f = de.path();
      if (lower_(f.extension().string()) != ".safetensors") { continue; }
      if (unsupported_quant(f.string())) { continue; }
      files.push_back(f);
    }
    std::sort(files.begin(), files.end());
    for (const fs::path& f : files) {
      FlexData md;
      if (!read_metadata(f.string(), md, nullptr)) { continue; }
      // A component is identified by its metadata key. A file with an
      // empty `__metadata__` is not one -- Comfy-Org always writes it,
      // and a LoRA or a stray checkpoint without one has nothing here
      // that could name the model it belongs to.
      for (auto e : md.as_object()) {
        out.push_back({role, f.string(), std::string(e.first)});
        break;
      }
    }
  }
  return out;
}

}  // namespace comfy
}  // namespace genai
}  // namespace vpipe
