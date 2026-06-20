#include "common/session-config.h"
#include <fstream>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vpipe {

namespace {

bool
is_inline_json(std::string_view s) noexcept
{
  for (char c : s) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      continue;
    }
    return c == '{' || c == '[';
  }
  return false;
}

bool
all_whitespace(std::string_view s) noexcept
{
  for (char c : s) {
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
      return false;
    }
  }
  return true;
}

std::string
read_file(std::string_view path)
{
  std::ifstream in(std::string(path), std::ios::binary);
  if (!in) {
    throw std::runtime_error("session config: cannot open '" +
                             std::string(path) + "'");
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  if (in.bad()) {
    throw std::runtime_error("session config: read error on '" +
                             std::string(path) + "'");
  }
  return ss.str();
}

bool
has_flex_binary_magic(const std::string& bytes) noexcept
{
  return bytes.size() >= 4
      && static_cast<unsigned char>(bytes[0]) == 'V'
      && static_cast<unsigned char>(bytes[1]) == 'P'
      && static_cast<unsigned char>(bytes[2]) == 'F'
      && static_cast<unsigned char>(bytes[3]) == 'D';
}

}

FlexData
parse_session_config(std::string_view src)
{
  if (all_whitespace(src)) {
    return FlexData::make_object();
  }
  if (is_inline_json(src)) {
    return FlexData::from_json(src);
  }
  // Otherwise treat as filesystem path.
  std::string bytes = read_file(src);
  if (has_flex_binary_magic(bytes)) {
    return FlexData::from_binary(bytes);
  }
  return FlexData::from_json(bytes);
}

}
