#ifndef GENERATIVE_MODELS_MINIMAX_H3_VDN_CONFIG_H
#define GENERATIVE_MODELS_MINIMAX_H3_VDN_CONFIG_H

// What a VDN-H3 checkpoint says about itself.
//
// A stage directory holds `model_spec.json` (the whole recipe: the base
// it derives from, the hybrid-attention transform, and the adapters),
// `linear_branch/{config.json,model.safetensors}` and
// `adapters/<name>/{adapter_config.json,adapter_model.safetensors}`.
// The transform block in model_spec.json and linear_branch/config.json
// carry the SAME object, so either file answers the question.
//
// EVERY FIELD IS REFUSED RATHER THAN DEFAULTED when it is not
// recognised, and that is the whole design of this file. The values name
// ALGORITHMS -- which delta rule, which bridge, how the anchor frames sit
// in the mask -- and each has a plausible sibling that runs to
// completion and produces a normal-looking video:
//
//   delta_rule    `sana_scaled` is `vdn_solve`'s first-order truncation.
//                 With L2-normed keys trace(A) = sum(beta) exactly, so at
//                 production geometry A is large and the two are
//                 different operators, not different accuracies.
//   bridge        `none` skips the decay through the window and makes the
//                 output gate relearn a gain it cannot absorb, since the
//                 bridge is not constant across frames.
//   anchor_frames `both` is what lets the linear branch DROP frames 0 and
//                 F-1; under any other mode it must keep covering them,
//                 so a wrong reading double-counts or drops two frames.
//
// None of those would raise. Hence: no default, no fallback, a name.

#include "generative-models/minimax-h3/vdn-geometry.h"

#include <map>
#include <string>
#include <vector>

namespace vpipe {
class FlexData;
namespace genai {
namespace minimax_h3 {
namespace vdn {

struct Config {
  AnchorFrames anchors        = AnchorFrames::kBoth;
  bool  enable_softmax_gate   = true;
  bool  a_fp32                = true;
  bool  bridge_alpha          = true;    // "alpha" | "none"
  bool  enable_text_state     = true;
  int   linear_head_dim       = 128;
  bool  conv_q = false, conv_k = true, conv_v = true;
  int   chunk = 5, radius = 1;
  std::string delta_rule = "vdn_solve";

  // Only `vdn_solve` is implemented here. The other two rules exist in
  // the reference as a control arm and a cheaper approximation; a
  // checkpoint asking for one is a checkpoint this code cannot run, and
  // saying so is better than running the rule it has.
  bool supported(std::string* why) const;
};

struct Adapter {
  std::string name;
  int rank = 0;
  double alpha = 0.0;
  bool exact_targets = false;
  std::vector<std::string> targets;
  // The turbo adapter varies rank and alpha per module; a module absent
  // from the map takes the top-level value.
  std::map<std::string, int>    rank_pattern;
  std::map<std::string, double> alpha_pattern;

  int    rank_for(const std::string& module) const;
  double alpha_for(const std::string& module) const;
  // alpha / rank, the scale a runtime application multiplies by.
  double scale_for(const std::string& module) const;
};

// Parse the `{type, version, config}` transform object that both
// model_spec.json (inside "transforms") and linear_branch/config.json
// carry. False with `err` set on an unknown type, an unknown version, or
// any unrecognised enum value.
bool parse_config(const FlexData& transform, Config* out, std::string* err);

// `dir` may be the STAGE directory (stage-dmd-step-250) or its
// `linear_branch` subdirectory; both are accepted because a caller
// pointing at the weights is pointing at the same model.
bool load_config(const std::string& dir, Config* out, std::string* err);

// `dir` is an `adapters/<name>` directory.
bool load_adapter(const std::string& dir, Adapter* out, std::string* err);

// Adapter names under `<stage>/adapters`, sorted. Empty when the stage
// has none.
std::vector<std::string> list_adapters(const std::string& stage_dir);

}  // namespace vdn
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
