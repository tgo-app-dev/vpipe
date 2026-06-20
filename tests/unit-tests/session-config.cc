#include "minitest.h"
#include "common/flex-data.h"
#include "common/session-config.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>

using namespace std;
using namespace vpipe;

namespace {

string make_tempdir() {
  auto base = filesystem::temp_directory_path() /
              "vpipe_session_config_test_XXXXXX";
  string tmpl = base.string();
  if (!mkdtemp(tmpl.data())) {
    throw runtime_error("mkdtemp failed");
  }
  return tmpl;
}

struct TempDir {
  string path;
  TempDir() : path(make_tempdir()) {}
  ~TempDir() {
    error_code ec;
    filesystem::remove_all(path, ec);
  }
};

void
write_file(const string& path, const string& contents)
{
  ofstream out(path, ios::binary);
  out.write(contents.data(),
            static_cast<streamsize>(contents.size()));
}

}

TEST(session_config, empty_string_yields_empty_object) {
  auto v = parse_session_config("");
  EXPECT_TRUE(v.is_object());
  EXPECT_TRUE(v.as_object().empty());
}

TEST(session_config, whitespace_only_yields_empty_object) {
  auto v = parse_session_config("  \t\n  ");
  EXPECT_TRUE(v.is_object());
  EXPECT_TRUE(v.as_object().empty());
}

TEST(session_config, inline_json_object_parses) {
  auto v = parse_session_config(
    R"({"log":{"delegate":"stdout","level":"verbose"}})");
  EXPECT_TRUE(v.is_object());
  auto log = v.as_object().at("log");
  EXPECT_TRUE(log.as_object().at("delegate").get_string() == "stdout");
  EXPECT_TRUE(log.as_object().at("level").get_string()    == "verbose");
}

TEST(session_config, inline_json5_with_comments_parses) {
  auto v = parse_session_config(R"(
    {
      // top-level comment
      log: { delegate: 'db', level: 'debug' },
    }
  )");
  EXPECT_TRUE(v.is_object());
  auto log = v.as_object().at("log");
  EXPECT_TRUE(log.as_object().at("delegate").get_string() == "db");
  EXPECT_TRUE(log.as_object().at("level").get_string()    == "debug");
}

TEST(session_config, file_path_json_parses) {
  TempDir dir;
  string path = dir.path + "/cfg.json";
  write_file(path, R"({"log":{"level":"warn"}})");
  auto v = parse_session_config(path);
  EXPECT_TRUE(v.as_object().at("log")
                .as_object().at("level").get_string() == "warn");
}

TEST(session_config, file_path_binary_flexdata_parses) {
  TempDir dir;
  string path = dir.path + "/cfg.bin";
  // Build a FlexData object, write its native binary, parse back.
  auto src = FlexData::make_object();
  src.as_object().insert("kind",
                         FlexData::make_string("binary-cfg"));
  src.as_object().insert("count", FlexData::make_uint(42));
  write_file(path, src.to_binary());
  auto v = parse_session_config(path);
  EXPECT_TRUE(v.is_object());
  EXPECT_TRUE(v.as_object().at("kind").get_string() == "binary-cfg");
  EXPECT_TRUE(v.as_object().at("count").get_uint()  == 42u);
}

TEST(session_config, missing_file_throws) {
  bool threw = false;
  try {
    parse_session_config("/this/path/does/not/exist/cfg.json");
  } catch (const exception&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(session_config, malformed_inline_json_throws) {
  bool threw = false;
  try {
    parse_session_config(R"({not-valid)");
  } catch (const exception&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}
