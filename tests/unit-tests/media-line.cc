#include "minitest.h"
#include "common/media-line.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace vpipe;
namespace ml = vpipe::media_line;

namespace {
std::vector<std::uint8_t>
bytes_(std::initializer_list<int> v)
{
  std::vector<std::uint8_t> out;
  for (int b : v) {
    out.push_back(static_cast<std::uint8_t>(b));
  }
  return out;
}
}

TEST(media_line, base64_round_trip_known_vectors) {
  // RFC 4648 vectors.
  EXPECT_TRUE(ml::base64_encode({}) == "");
  const std::string foobar = "foobar";
  std::vector<std::uint8_t> fb(foobar.begin(), foobar.end());
  EXPECT_TRUE(ml::base64_encode(
      std::span<const std::uint8_t>(fb.data(), 1)) == "Zg==");
  EXPECT_TRUE(ml::base64_encode(
      std::span<const std::uint8_t>(fb.data(), 2)) == "Zm8=");
  EXPECT_TRUE(ml::base64_encode(
      std::span<const std::uint8_t>(fb.data(), 3)) == "Zm9v");
  EXPECT_TRUE(ml::base64_encode(
      std::span<const std::uint8_t>(fb.data(), 6)) == "Zm9vYmFy");
  // Decode: padded, unpadded, whitespace-tolerant.
  auto d = ml::base64_decode("Zm9vYmFy");
  ASSERT_TRUE(d.has_value());
  EXPECT_TRUE(std::string(d->begin(), d->end()) == "foobar");
  d = ml::base64_decode("Zm8=");
  ASSERT_TRUE(d.has_value() && d->size() == 2);
  d = ml::base64_decode("Zm8");
  ASSERT_TRUE(d.has_value() && d->size() == 2);
  d = ml::base64_decode("Zm9v\nYmFy");
  ASSERT_TRUE(d.has_value() && d->size() == 6);
  // Invalid: bad byte, data after padding, dangling 6-bit group.
  EXPECT_TRUE(!ml::base64_decode("Zm9v!").has_value());
  EXPECT_TRUE(!ml::base64_decode("Zm8=Zg").has_value());
  EXPECT_TRUE(!ml::base64_decode("Z").has_value());
  // Binary round trip incl. 0x00/0xff.
  auto bin = bytes_({0, 1, 2, 0xfe, 0xff, 0x80, 0x7f});
  auto rt = ml::base64_decode(ml::base64_encode(
      std::span<const std::uint8_t>(bin.data(), bin.size())));
  ASSERT_TRUE(rt.has_value());
  EXPECT_TRUE(*rt == bin);
}

TEST(media_line, parse_plain_text_single_segment) {
  EXPECT_TRUE(!ml::has_media_marker("hello world"));
  auto segs = ml::parse("hello world");
  ASSERT_TRUE(segs.size() == 1);
  EXPECT_TRUE(segs[0].kind == ml::Segment::Kind::Text);
  EXPECT_TRUE(segs[0].text == "hello world");
}

TEST(media_line, parse_fs_and_base64_interleaved) {
  auto payload = bytes_({1, 2, 3, 4, 5});
  const std::string line =
      "look at " +
      ml::make_fs_marker(ml::Modality::Image, "/tmp/pic.jpg") +
      " and listen to " +
      ml::make_base64_marker(
          ml::Modality::Audio,
          std::span<const std::uint8_t>(payload.data(), payload.size())) +
      " please";
  EXPECT_TRUE(ml::has_media_marker(line));
  std::vector<std::string> errs;
  auto segs = ml::parse(line, &errs);
  EXPECT_TRUE(errs.empty());
  ASSERT_TRUE(segs.size() == 5);
  EXPECT_TRUE(segs[0].kind == ml::Segment::Kind::Text);
  EXPECT_TRUE(segs[0].text == "look at ");
  EXPECT_TRUE(segs[1].kind == ml::Segment::Kind::FsPath);
  EXPECT_TRUE(segs[1].modality == ml::Modality::Image);
  EXPECT_TRUE(segs[1].text == "/tmp/pic.jpg");
  EXPECT_TRUE(segs[2].kind == ml::Segment::Kind::Text);
  EXPECT_TRUE(segs[2].text == " and listen to ");
  EXPECT_TRUE(segs[3].kind == ml::Segment::Kind::Bytes);
  EXPECT_TRUE(segs[3].modality == ml::Modality::Audio);
  EXPECT_TRUE(segs[3].bytes == payload);
  EXPECT_TRUE(segs[4].kind == ml::Segment::Kind::Text);
  EXPECT_TRUE(segs[4].text == " please");
}

TEST(media_line, parse_malformed_markers_drop_with_errors) {
  std::vector<std::string> errs;
  // LENGTH mismatch.
  auto segs = ml::parse(
      "a <|__vpipe_base64_im_start__|>9,Zm9v<|__vpipe_base64_im_end__|> b",
      &errs);
  ASSERT_TRUE(segs.size() == 1);
  EXPECT_TRUE(segs[0].kind == ml::Segment::Kind::Text);
  EXPECT_TRUE(segs[0].text == "a  b");
  EXPECT_TRUE(errs.size() == 1);

  // Unterminated marker: tail dropped, not leaked.
  errs.clear();
  segs = ml::parse("hi <|__vpipe_fs_im_start__|>/no/end", &errs);
  ASSERT_TRUE(segs.size() == 1);
  EXPECT_TRUE(segs[0].text == "hi ");
  EXPECT_TRUE(errs.size() == 1);

  // Unknown marker prefix kept literally.
  errs.clear();
  segs = ml::parse("x <|__vpipe_fs_im_end__|> y", &errs);
  ASSERT_TRUE(segs.size() == 1);
  EXPECT_TRUE(segs[0].text.find("<|__vpipe_") != std::string::npos);
  EXPECT_TRUE(errs.size() == 1);

  // Empty fs path dropped.
  errs.clear();
  segs = ml::parse(
      "p <|__vpipe_fs_au_start__|><|__vpipe_fs_au_end__|> q", &errs);
  ASSERT_TRUE(segs.size() == 1);
  EXPECT_TRUE(segs[0].text == "p  q");
  EXPECT_TRUE(errs.size() == 1);

  // Bad base64 payload dropped.
  errs.clear();
  segs = ml::parse(
      "<|__vpipe_base64_au_start__|>3,!!!<|__vpipe_base64_au_end__|>",
      &errs);
  EXPECT_TRUE(segs.empty());
  EXPECT_TRUE(errs.size() == 1);
}

TEST(media_line, think_markers_render_plain) {
  const std::string in = std::string(ml::kThinkStart)
      + "let me reason" + std::string(ml::kThinkEnd) + "Answer: 9.";
  const std::string out = ml::render_think_markers_plain(in);
  EXPECT_TRUE(out == "\xE2\x9F\xA6think\xE2\x9F\xA7" "let me reason"
                     "\xE2\x9F\xA6/think\xE2\x9F\xA7" "Answer: 9.");
  // Unterminated block and plain text both pass through sanely.
  EXPECT_TRUE(ml::render_think_markers_plain(
                  std::string(ml::kThinkStart) + "cut off")
              == "\xE2\x9F\xA6think\xE2\x9F\xA7" "cut off");
  EXPECT_TRUE(ml::render_think_markers_plain("no markers")
              == "no markers");
}

TEST(media_line, to_display_compresses_markers) {
  auto payload = bytes_({9, 9, 9});
  const std::string line =
      "see " + ml::make_fs_marker(ml::Modality::Image, "/tmp/a.png") +
      " hear " +
      ml::make_base64_marker(
          ml::Modality::Audio,
          std::span<const std::uint8_t>(payload.data(), payload.size()));
  const std::string disp = ml::to_display(line);
  EXPECT_TRUE(disp.find("/tmp/a.png") != std::string::npos);
  EXPECT_TRUE(disp.find("3 bytes") != std::string::npos);
  // No marker text and no base64 data in the display form.
  EXPECT_TRUE(disp.find("<|__vpipe_") == std::string::npos);
  EXPECT_TRUE(disp.find("CQkJ") == std::string::npos);
  // Plain text passes through untouched.
  EXPECT_TRUE(ml::to_display("just text") == "just text");
}
