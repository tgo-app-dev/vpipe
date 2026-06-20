#include "pipeline/stage-spec.h"

namespace vpipe {

std::string_view
stage_category_name(StageCategory c) noexcept
{
  switch (c) {
    case StageCategory::Generic:  return "generic";
    case StageCategory::Audio:    return "audio";
    case StageCategory::Video:    return "video";
    case StageCategory::Text:     return "text";
    case StageCategory::Control:  return "control";
    case StageCategory::Database: return "database";
    case StageCategory::Network:  return "network";
  }
  return "generic";
}

}
