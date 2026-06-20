#include "minitest.h"

#include "common/i18n.h"
#include "common/session.h"

#include <string>

using namespace vpipe;

TEST(i18n, normalize_language_maps_variants) {
  EXPECT_TRUE(normalize_language("en") == "en-us");
  EXPECT_TRUE(normalize_language("EN") == "en-us");
  EXPECT_TRUE(normalize_language("en-US") == "en-us");
  EXPECT_TRUE(normalize_language("en_us") == "en-us");
  EXPECT_TRUE(normalize_language("zh") == "zh-cn");
  EXPECT_TRUE(normalize_language("zh-CN") == "zh-cn");
  EXPECT_TRUE(normalize_language("zh-Hans") == "zh-cn");
  EXPECT_TRUE(normalize_language("zh-TW") == "zh-tw");
  EXPECT_TRUE(normalize_language("zh-Hant") == "zh-tw");
  EXPECT_TRUE(normalize_language("zh-HK") == "zh-tw");
  EXPECT_TRUE(normalize_language("fr").empty());
  EXPECT_TRUE(normalize_language("").empty());
}

TEST(i18n, supported_languages_list) {
  const auto& v = supported_languages();
  EXPECT_TRUE(v.size() == 3);
  EXPECT_TRUE(v[0] == "en-us");
  EXPECT_TRUE(is_supported_language("zh-Hant"));
  EXPECT_TRUE(!is_supported_language("de"));
}

TEST(i18n, localize_lookup_and_fallback) {
  // Known key, each locale.
  EXPECT_TRUE(localize("en-us", "nav.settings") == "Settings");
  EXPECT_TRUE(localize("zh-cn", "nav.settings") == std::string("设置"));
  EXPECT_TRUE(localize("zh-tw", "nav.settings") == std::string("設定"));
  // Tag normalization happens inside localize.
  EXPECT_TRUE(localize("ZH-Hant", "nav.settings") == std::string("設定"));
  // Unsupported locale falls back to en-us.
  EXPECT_TRUE(localize("fr", "nav.settings") == "Settings");
  // Unknown key falls back to the key itself.
  EXPECT_TRUE(localize("en-us", "no.such.key") == "no.such.key");
}

TEST(i18n, session_language_get_set) {
  Session sess;
  EXPECT_TRUE(sess.language() == "en-us");           // default locale
  EXPECT_TRUE(sess.tr("nav.settings") == "Settings");
  EXPECT_TRUE(sess.set_language("zh-TW").code == 0);  // normalized
  EXPECT_TRUE(sess.language() == "zh-tw");
  EXPECT_TRUE(sess.tr("nav.settings") == std::string("設定"));
  EXPECT_TRUE(sess.set_language("xx").code != 0);     // rejected
  EXPECT_TRUE(sess.language() == "zh-tw");            // unchanged
}

TEST(i18n, session_language_from_config) {
  Session sess(R"({"language":"zh-cn"})");
  EXPECT_TRUE(sess.language() == "zh-cn");
  EXPECT_TRUE(sess.tr("nav.settings") == std::string("设置"));
}
