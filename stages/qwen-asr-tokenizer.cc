#include "stages/qwen-asr-tokenizer.h"
#include "common/flex-data.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace fs = std::filesystem;

namespace vpipe {

namespace {

// The Qwen2/Qwen3 byte-level pre-tokenizer regex. Functionally vpipe
// only needs it to contain \p{L} and \p{N} (that pair routes encoding
// to the hand-coded Qwen/Llama-3 scanner), but we emit the real pattern
// so the file is a faithful tokenizer.json.
constexpr const char* kQwenPreRegex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| )"
    R"(?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

FlexData
str_(std::string_view v)
{
  return FlexData::make_string(v);
}

// merges.txt -> FlexData array of "a b" strings (skip the "#version"
// header and blank lines).
FlexData
parse_merges_(const std::string& merges_txt)
{
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  size_t i = 0;
  while (i < merges_txt.size()) {
    size_t nl = merges_txt.find('\n', i);
    std::string line =
        merges_txt.substr(i, (nl == std::string::npos ? merges_txt.size()
                                                      : nl) - i);
    i = (nl == std::string::npos) ? merges_txt.size() : nl + 1;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty() || line[0] == '#') {
      continue;
    }
    a.push_back(str_(line));
  }
  return arr;
}

// tokenizer_config.json's added_tokens_decoder (idstr -> {content,
// special, ...}) -> the fast-format `added_tokens` array
// [{id, content, special, ...}]. Empty when the field is absent.
FlexData
parse_added_tokens_(const std::string& cfg_json)
{
  FlexData arr = FlexData::make_array();
  if (cfg_json.empty()) {
    return arr;
  }
  FlexData cfg;
  try {
    cfg = FlexData::from_json(cfg_json);
  } catch (const std::exception&) {
    return arr;
  }
  if (!cfg.is_object()) {
    return arr;
  }
  auto co = cfg.as_object();
  if (!co.contains("added_tokens_decoder")) {
    return arr;
  }
  FlexData dec = co.at("added_tokens_decoder");
  if (!dec.is_object()) {
    return arr;
  }
  auto deco = dec.as_object();
  auto a = arr.as_array();
  for (auto it = deco.begin(); it != deco.end(); ++it) {
    auto entry = *it;
    const std::string idstr(entry.first);
    FlexData v = entry.second;
    if (!v.is_object()) {
      continue;
    }
    char* endp = nullptr;
    const long id = std::strtol(idstr.c_str(), &endp, 10);
    if (endp == idstr.c_str() || *endp != '\0' || id < 0) {
      continue;
    }
    auto vo = v.as_object();
    std::string content = vo.contains("content")
        ? std::string(vo.at("content").as_string("")) : "";
    if (content.empty()) {
      continue;
    }
    const bool special = vo.contains("special")
        ? vo.at("special").as_bool(true) : true;

    FlexData t = FlexData::make_object();
    auto to = t.as_object();
    to.insert_or_assign("id", FlexData::make_int(id));
    to.insert_or_assign("content", str_(content));
    to.insert_or_assign("single_word", FlexData::make_bool(false));
    to.insert_or_assign("lstrip", FlexData::make_bool(false));
    to.insert_or_assign("rstrip", FlexData::make_bool(false));
    to.insert_or_assign("normalized", FlexData::make_bool(false));
    to.insert_or_assign("special", FlexData::make_bool(special));
    a.push_back(std::move(t));
  }
  return arr;
}

bool
read_file_(const fs::path& p, std::string& out)
{
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    return false;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

}  // namespace

std::string
build_qwen_asr_tokenizer_json(const std::string& vocab_json,
                              const std::string& merges_txt,
                              const std::string& tokenizer_config_json,
                              std::string&       err)
{
  FlexData vocab;
  try {
    vocab = FlexData::from_json(vocab_json);
  } catch (const std::exception& e) {
    err = std::string("vocab.json parse error: ") + e.what();
    return {};
  }
  if (!vocab.is_object()) {
    err = "vocab.json is not a JSON object";
    return {};
  }

  FlexData merges = parse_merges_(merges_txt);
  FlexData added  = parse_added_tokens_(tokenizer_config_json);

  // model: byte-level BPE (vpipe requires type=="BPE" + vocab; merges
  // optional but always present here).
  FlexData model = FlexData::make_object();
  {
    auto m = model.as_object();
    m.insert_or_assign("type", str_("BPE"));
    m.insert_or_assign("dropout", FlexData::make_null());
    m.insert_or_assign("unk_token", FlexData::make_null());
    m.insert_or_assign("continuing_subword_prefix", FlexData::make_null());
    m.insert_or_assign("end_of_word_suffix", FlexData::make_null());
    m.insert_or_assign("fuse_unk", FlexData::make_bool(false));
    m.insert_or_assign("byte_fallback", FlexData::make_bool(false));
    m.insert_or_assign("vocab", std::move(vocab));
    m.insert_or_assign("merges", std::move(merges));
  }

  // pre_tokenizer: Sequence[ Split(<qwen regex>), ByteLevel ].
  FlexData split = FlexData::make_object();
  {
    auto so = split.as_object();
    so.insert_or_assign("type", str_("Split"));
    FlexData pat = FlexData::make_object();
    pat.as_object().insert_or_assign("Regex", str_(kQwenPreRegex));
    so.insert_or_assign("pattern", std::move(pat));
    so.insert_or_assign("behavior", str_("Isolated"));
    so.insert_or_assign("invert", FlexData::make_bool(false));
  }
  FlexData byte_level = FlexData::make_object();
  {
    auto bo = byte_level.as_object();
    bo.insert_or_assign("type", str_("ByteLevel"));
    bo.insert_or_assign("add_prefix_space", FlexData::make_bool(false));
    bo.insert_or_assign("trim_offsets", FlexData::make_bool(true));
    bo.insert_or_assign("use_regex", FlexData::make_bool(false));
  }
  FlexData pre = FlexData::make_object();
  {
    auto po = pre.as_object();
    po.insert_or_assign("type", str_("Sequence"));
    FlexData pts = FlexData::make_array();
    {
      auto a = pts.as_array();
      a.push_back(std::move(split));
      a.push_back(std::move(byte_level));
    }
    po.insert_or_assign("pretokenizers", std::move(pts));
  }

  FlexData decoder = FlexData::make_object();
  {
    auto d = decoder.as_object();
    d.insert_or_assign("type", str_("ByteLevel"));
    d.insert_or_assign("add_prefix_space", FlexData::make_bool(true));
    d.insert_or_assign("trim_offsets", FlexData::make_bool(true));
    d.insert_or_assign("use_regex", FlexData::make_bool(true));
  }

  FlexData root = FlexData::make_object();
  {
    auto r = root.as_object();
    r.insert_or_assign("version", str_("1.0"));
    r.insert_or_assign("added_tokens", std::move(added));
    // No metaspace normaliser -> vpipe keeps GPT-2 byte-level encoding.
    r.insert_or_assign("normalizer", FlexData::make_null());
    r.insert_or_assign("pre_tokenizer", std::move(pre));
    r.insert_or_assign("decoder", std::move(decoder));
    r.insert_or_assign("model", std::move(model));
  }

  return root.to_json(/*pretty=*/false);
}

bool
prepare_qwen_asr_tokenizer_json(const std::string& model_dir,
                                std::string&       err)
{
  const fs::path dir(model_dir);
  std::string vocab, merges, cfg;
  if (!read_file_(dir / "vocab.json", vocab)) {
    err = "vocab.json not found in " + model_dir;
    return false;
  }
  if (!read_file_(dir / "merges.txt", merges)) {
    err = "merges.txt not found in " + model_dir;
    return false;
  }
  read_file_(dir / "tokenizer_config.json", cfg);   // optional

  const std::string out =
      build_qwen_asr_tokenizer_json(vocab, merges, cfg, err);
  if (out.empty()) {
    return false;
  }
  std::ofstream o(dir / "tokenizer.json",
                  std::ios::binary | std::ios::trunc);
  if (!o) {
    err = "cannot open tokenizer.json for writing";
    return false;
  }
  o << out;
  if (!o.good()) {
    err = "writing tokenizer.json failed";
    return false;
  }
  return true;
}

}
