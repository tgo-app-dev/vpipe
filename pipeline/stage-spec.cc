#include "pipeline/stage-spec.h"

#include <vector>

namespace vpipe {

namespace {

// Trim ASCII whitespace from both ends of a view.
std::string_view
trim_tag_(std::string_view s) noexcept
{
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string_view::npos) {
    return {};
  }
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// Split a comma-separated tag list into its non-empty, trimmed tags.
std::vector<std::string_view>
split_tags_(std::string_view list)
{
  std::vector<std::string_view> out;
  size_t i = 0;
  while (i <= list.size()) {
    const size_t c = list.find(',', i);
    const size_t end = (c == std::string_view::npos) ? list.size() : c;
    const std::string_view t = trim_tag_(list.substr(i, end - i));
    if (!t.empty()) {
      out.push_back(t);
    }
    if (c == std::string_view::npos) {
      break;
    }
    i = c + 1;
  }
  return out;
}

}  // namespace

bool
port_tags_compatible(std::string_view produced,
                     std::string_view accepted) noexcept
{
  const std::vector<std::string_view> p = split_tags_(produced);
  const std::vector<std::string_view> a = split_tags_(accepted);
  // No constraint on either side -> compatible (type alone decides).
  if (p.empty() || a.empty()) {
    return true;
  }
  // OR semantics: compatible iff the two sets share at least one tag.
  for (const std::string_view pt : p) {
    for (const std::string_view at : a) {
      if (pt == at) {
        return true;
      }
    }
  }
  return false;
}

std::string_view
stage_category_name(StageCategory c) noexcept
{
  switch (c) {
    case StageCategory::Generic:  return "generic";
    case StageCategory::Audio:    return "audio";
    case StageCategory::Visual:   return "visual";
    case StageCategory::Vision:   return "vision";
    case StageCategory::Generative: return "generative";
    case StageCategory::Text:     return "text";
    case StageCategory::Control:  return "control";
    case StageCategory::Database: return "database";
    case StageCategory::Network:  return "network";
    case StageCategory::Preparation: return "preparation";
  }
  return "generic";
}

}
