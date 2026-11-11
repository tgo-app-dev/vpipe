#ifndef VPIPE_STAGES_MINIMAX_H3_IMAGE_HOST_H
#define VPIPE_STAGES_MINIMAX_H3_IMAGE_HOST_H

// minimax-h3-image-host.h -- what the stages share about MiniMax-H3-Image
// (generative-models/minimax-h3/minimax-h3-image.h) that needs the models
// DB. Host-only, not installed.

#include "generative-models/minimax-h3/minimax-h3-image.h"
#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"
#include "stages/model-registry.h"

#include <filesystem>
#include <string>
#include <system_error>

namespace vpipe {

// MiniMax-H3-Image's TEXT ENCODER, which its repository does not carry:
// model_index.json names MiniMax-H3's `text_encoder/`. Found, in order,
// as `named` says (a stage's `encoder_dir`: the encoder, a partition or
// a repository root, a path or a models-DB key), in the root itself (a
// copy put there), then through the models DB under the repository the
// manifest names and the records published from it (MiniMax-H3's FL2VA
// and Ref2VA pipelines each carry the same 32B encoder). Empty when no
// local copy answers -- the caller says what to fetch.
inline std::string
minimax_h3_image_encoder_dir(const SessionContextIntf* session,
                             const std::string& root,
                             const std::string& named = {})
{
  namespace fs = std::filesystem;
  std::error_code ec;
  auto encoder_in = [&](const std::string& dir) -> std::string {
    if (dir.empty() || !fs::exists(dir, ec)) { return {}; }
    const std::string e = genai::MiniMaxH3TextEncoder::resolve_encoder_dir(dir);
    if (e.empty()) { return {}; }
    if (fs::is_directory(e, ec) && !fs::exists(fs::path(e) / "config.json")) {
      return {};
    }
    return e;
  };
  if (!named.empty()) {
    const std::string e = encoder_in(resolve_model_dir(session, named));
    if (!e.empty()) { return e; }
  }
  if (const std::string e = encoder_in((fs::path(root) / "text_encoder")
                                           .string());
      !e.empty()) {
    return e;
  }
  genai::h3i::Manifest man;
  (void)genai::h3i::read_manifest(root, &man, nullptr);
  const std::string repo = man.text_encoder_repo;
  for (const std::string& key :
       {repo + "-text-encoder", repo, repo + "-FL2VA", repo + "-Ref2VA"}) {
    const ResolvedModel rm = resolve_model(session, key);
    if (!rm.from_registry) { continue; }
    if (const std::string e = encoder_in(rm.dir); !e.empty()) { return e; }
  }
  return {};
}

}  // namespace vpipe

#endif
