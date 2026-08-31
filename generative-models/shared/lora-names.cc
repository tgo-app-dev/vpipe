#include "generative-models/shared/lora-names.h"

namespace vpipe {
namespace genai {
namespace lora {

namespace {

void
replace_all(std::string& s, const std::string& from, const std::string& to)
{
  if (from.empty()) { return; }
  std::size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) {
    s.replace(p, from.size(), to);
    p += to.size();
  }
}

}  // namespace

std::string
remap_ai_toolkit_module(std::string m)
{
  const std::string kPre = "diffusion_model.";
  if (m.size() < kPre.size() || m.compare(0, kPre.size(), kPre) != 0) {
    return {};
  }
  m = m.substr(kPre.size());
  if (m.compare(0, 7, "blocks.") == 0) {
    m = "transformer_" + m;              // blocks.N -> transformer_blocks.N
  } else {
    replace_all(m, "txtfusion.", "text_fusion.");   // layerwise/refiner_blocks
  }
  replace_all(m, ".mlp.", ".ff.");        // mlp.{gate,up,down} -> ff.{...}
  replace_all(m, ".attn.wq", ".attn.to_q");
  replace_all(m, ".attn.wk", ".attn.to_k");
  replace_all(m, ".attn.wv", ".attn.to_v");
  replace_all(m, ".attn.wo", ".attn.to_out.0");
  replace_all(m, ".attn.gate", ".attn.to_gate");
  return m;
}

}  // namespace lora
}  // namespace genai
}  // namespace vpipe
