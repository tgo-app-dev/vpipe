#include "common/i18n.h"

#include <cctype>
#include <unordered_map>

namespace vpipe {

namespace {

// Locale slots in the catalogue's per-key string array.
enum { kEN = 0, kZH_CN = 1, kZH_TW = 2, kNLocales = 3 };

struct Entry {
  const char* key;
  const char* s[kNLocales];   // [en-us, zh-cn, zh-tw]
};

// Initial application string catalogue. Add keys here. Every key MUST
// carry an en-us string (the fallback); a zh-* slot may be "" to defer a
// translation (lookup then falls back to en-us). Keep this in sync with
// the browser catalogue in apps/web-ui/web/js/i18n.js.
constexpr Entry kCatalog[] = {
  {"app.name",          {"VPIPE", "VPIPE", "VPIPE"}},
  {"nav.pipelines",     {"Pipelines", "管道", "管線"}},
  {"nav.profiler",      {"Profiler", "性能分析", "效能分析"}},
  {"nav.io",            {"User I/O", "用户输入输出", "使用者輸入輸出"}},
  {"nav.database",      {"Database", "数据库", "資料庫"}},
  {"nav.settings",      {"Settings", "设置", "設定"}},
  {"settings.title",    {"Settings", "设置", "設定"}},
  {"settings.language", {"Language", "语言", "語言"}},
  {"settings.theme",    {"Color theme", "颜色主题", "色彩主題"}},
  {"common.save",       {"Save", "保存", "儲存"}},
  {"common.dismiss",    {"Dismiss", "关闭", "關閉"}},
  {"startup.title",     {"Startup checks", "启动检查", "啟動檢查"}},
};

const std::unordered_map<std::string_view, const Entry*>&
catalog_index()
{
  static const std::unordered_map<std::string_view, const Entry*> idx = [] {
    std::unordered_map<std::string_view, const Entry*> m;
    for (const auto& e : kCatalog) { m.emplace(e.key, &e); }
    return m;
  }();
  return idx;
}

int
locale_slot_(std::string_view norm)
{
  if (norm == "zh-cn") { return kZH_CN; }
  if (norm == "zh-tw") { return kZH_TW; }
  return kEN;
}

}  // namespace

const std::vector<std::string>&
supported_languages()
{
  static const std::vector<std::string> v = {"en-us", "zh-cn", "zh-tw"};
  return v;
}

std::string_view
default_language()
{
  return "en-us";
}

std::string
normalize_language(std::string_view tag)
{
  // Lowercase, normalizing '_' to '-'.
  std::string t;
  t.reserve(tag.size());
  for (char c : tag) {
    char lc = (c == '_') ? '-'
                         : static_cast<char>(
                               std::tolower(static_cast<unsigned char>(c)));
    t.push_back(lc);
  }
  // Split into primary subtag + remainder (script / region).
  std::string primary = t;
  std::string rest;
  if (auto pos = t.find('-'); pos != std::string::npos) {
    primary = t.substr(0, pos);
    rest = t.substr(pos + 1);
  }
  if (primary == "en") { return "en-us"; }
  if (primary == "zh") {
    // Traditional-Chinese scripts / regions; everything else (incl.
    // Hans / CN / SG / bare "zh") maps to simplified.
    if (rest.find("hant") != std::string::npos
        || rest.find("tw") != std::string::npos
        || rest.find("hk") != std::string::npos
        || rest.find("mo") != std::string::npos) {
      return "zh-tw";
    }
    return "zh-cn";
  }
  return "";
}

bool
is_supported_language(std::string_view tag)
{
  return !normalize_language(tag).empty();
}

std::string
localize(std::string_view lang, std::string_view key)
{
  std::string norm = normalize_language(lang);
  if (norm.empty()) { norm = "en-us"; }
  const int slot = locale_slot_(norm);

  const auto& idx = catalog_index();
  auto it = idx.find(key);
  if (it == idx.end()) {
    return std::string(key);   // unknown key -> visible key
  }
  const Entry* e = it->second;
  if (e->s[slot] != nullptr && e->s[slot][0] != '\0') {
    return std::string(e->s[slot]);
  }
  // Requested locale has no string for this key: fall back to en-us,
  // then to the key itself.
  if (e->s[kEN] != nullptr && e->s[kEN][0] != '\0') {
    return std::string(e->s[kEN]);
  }
  return std::string(key);
}

}  // namespace vpipe
