#include "generative-models/shared/grounded-encode-params.h"

#include <string>

namespace vpipe {
namespace genai {

GroundedEncodeParams
GroundedEncodeParams::for_family(const std::string& family)
{
  GroundedEncodeParams p;
  if (family == "krea2") {
    // The ComfyUI-Krea2Edit grounding node's default.
    p.long_edge = 768;
    return p;
  }
  if (family == "mage-flow") {
    // pipeline.py `vl_cond_long_edge`. The processor bounds come from
    // preprocessor_config.json, whose shortest_edge is 65536 against the
    // Qwen default of 3136 -- so a small or very wide reference is
    // UPSCALED before patching, which the default bound would skip.
    p.long_edge  = 384;
    p.min_pixels = 65536;
    p.max_pixels = 16777216;
    return p;
  }
  if (family == "boogu-image") {
    // BooguImagePipeline's VLM preprocessing: max side 768, area capped
    // at 384x384. BOTH, not either -- which is why the long edge here is
    // Mage-Flow's doubled rather than equal to it.
    p.long_edge    = 768;
    p.pixel_budget = (std::size_t)384 * 384;
    p.min_pixels   = 65536;
    p.max_pixels   = 16777216;
    return p;
  }
  if (family == "qwen-image-edit") {
    // Qwen2.5-VL takes a pixel budget directly and does its own
    // smart-resize from it, so there is no separate long-edge cap.
    p.pixel_budget = (std::size_t)384 * 384;
    return p;
  }
  return p;   // no grounded path: no capping, tower defaults
}

void
GroundedEncodeParams::merge_flex(const FlexData& fd, std::string* err)
{
  if (!fd.is_object()) {
    if (err != nullptr) { *err = "not a JSON object; keeping the defaults"; }
    return;
  }
  auto o = fd.as_object();
  auto sz = [&](const char* k, std::size_t& dst) {
    if (!o.contains(k)) { return; }
    const std::int64_t v = o.at(k).as_int((std::int64_t)dst);
    dst = v > 0 ? (std::size_t)v : 0;
  };
  if (o.contains("vl_long_edge")) {
    const std::int64_t v = o.at("vl_long_edge").as_int(long_edge);
    long_edge = v > 0 ? (int)v : 0;
  }
  sz("vl_pixel_budget", pixel_budget);
  sz("vl_min_pixels", min_pixels);
  sz("vl_max_pixels", max_pixels);
}

}  // namespace genai
}  // namespace vpipe
