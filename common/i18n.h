#ifndef VPIPE_COMMON_I18N_H
#define VPIPE_COMMON_I18N_H

#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

// Lightweight UI/message localization for the application layer (web-ui
// chrome, user-facing Session messages). This is NOT the model/ASR
// "language hint" used by transcription stages -- that stays separate.
//
// Locales are IETF BCP-47 tags, normalized to lowercase "lang-region"
// (e.g. "en-us", "zh-cn", "zh-tw"). The string catalogue lives in
// i18n.cc; add keys there. Lookups fall back in order: the requested
// locale -> "en-us" -> the key itself, so a missing translation degrades
// to English and then to a visible key (never an empty string).

// Supported locales, in display order. The first entry ("en-us") is the
// fallback locale.
const std::vector<std::string>& supported_languages();

// Default / fallback locale tag ("en-us").
std::string_view default_language();

// True if `tag` names a supported locale (after normalization).
bool is_supported_language(std::string_view tag);

// Normalize an IETF tag to a supported locale tag, or "" if none match.
// Case-insensitive; accepts '_' or '-' separators; maps script/region
// variants onto the supported set:
//   "en", "EN", "en-US", "en_us"        -> "en-us"
//   "zh", "zh-CN", "zh-Hans", "zh_hans" -> "zh-cn"
//   "zh-TW", "zh-Hant", "zh-HK"         -> "zh-tw"
std::string normalize_language(std::string_view tag);

// Look up a localized string by `key` for `lang`. `lang` may be any tag
// (it is normalized internally; an unsupported tag uses en-us). Falls
// back to en-us, then to `key` itself.
std::string localize(std::string_view lang, std::string_view key);

}  // namespace vpipe

#endif
