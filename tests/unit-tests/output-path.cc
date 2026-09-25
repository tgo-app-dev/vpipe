#include "minitest.h"

#include "stages/output-path.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace std;
using namespace vpipe;
using vpipe::outpath::SeqPicker;
using vpipe::outpath::Tokens;

namespace {

// minitest's ASSERT_TRUE does NOT abort, so anything
// that would run against a state the assertion just disproved gets its
// own guard rather than trusting the line above it.
#define REQUIRE(cond) do { if (!(cond)) { EXPECT_TRUE(cond); return; } } \
                      while (0)

// A wall clock the tests can name, so an expectation is a string and
// not a re-implementation of strftime. 2026-09-24 11:15:30 local.
time_t fixed_time() {
  tm t{};
  t.tm_year = 2026 - 1900;
  t.tm_mon  = 8;          // September
  t.tm_mday = 24;
  t.tm_hour = 11;
  t.tm_min  = 15;
  t.tm_sec  = 30;
  t.tm_isdst = -1;
  return mktime(&t);
}

string exp_(const string& tmpl, uint64_t seq, int pad = 6) {
  return outpath::expand(tmpl, seq, pad, fixed_time());
}

string tmp_dir_() {
  static int n = 0;
  string d = "/tmp/vpipe-outpath-" + to_string((long)getpid())
           + "-" + to_string(n++);
  error_code ec;
  filesystem::remove_all(d, ec);
  filesystem::create_directories(d, ec);
  return d;
}

void touch_(const string& p) {
  ofstream(p, ios::binary) << "x";
}

}   // namespace

// ---- the expander -------------------------------------------------

TEST(output_path, a_plain_name_is_left_alone) {
  EXPECT_TRUE(exp_("out/frame.png", 0) == "out/frame.png");
}

TEST(output_path, an_integer_conversion_takes_the_sequence_number) {
  EXPECT_TRUE(exp_("f-%d.png", 7)    == "f-7.png");
  EXPECT_TRUE(exp_("f-%04d.png", 7)  == "f-0007.png");
  EXPECT_TRUE(exp_("f-%4d.png", 7)   == "f-   7.png");
  EXPECT_TRUE(exp_("f-%-4d.png", 7)  == "f-7   .png");
  EXPECT_TRUE(exp_("f-%x.png", 255)  == "f-ff.png");
  EXPECT_TRUE(exp_("f-%X.png", 255)  == "f-FF.png");
  EXPECT_TRUE(exp_("f-%#04x.png", 10) == "f-0x0a.png");
}

// The old code cast the index through `(int)(unsigned)`, so a long run
// wrapped negative at 2^31. Nothing formats through printf now.
TEST(output_path, a_sequence_number_past_2_31_is_not_wrapped) {
  EXPECT_TRUE(exp_("f-%d.png", 3000000000ull) == "f-3000000000.png");
}

TEST(output_path, a_doubled_percent_is_a_literal_one) {
  EXPECT_TRUE(exp_("100%%-%02d.png", 3) == "100%-03.png");
}

// A path is data. Handing it to snprintf (which is what this replaced)
// made "%s" read a pointer off the stack; now it is just a filename.
TEST(output_path, an_unknown_conversion_is_copied_verbatim) {
  EXPECT_TRUE(exp_("f-%s-%02d.png", 4) == "f-%s-04.png");
  EXPECT_TRUE(exp_("50%off.png", 0)    == "50%off.png");
}

TEST(output_path, every_conversion_gets_the_same_number) {
  EXPECT_TRUE(exp_("%02d/f-%04d.png", 5) == "05/f-0005.png");
}

TEST(output_path, t_stamps_the_local_time) {
  EXPECT_TRUE(exp_("clip-%t.mp4", 0) == "clip-20260924-111530.mp4");
}

TEST(output_path, t_takes_an_explicit_strftime_format) {
  EXPECT_TRUE(exp_("clip-%t{%Y-%m-%d}.mp4", 0) == "clip-2026-09-24.mp4");
  EXPECT_TRUE(exp_("%t{%H%M}-x.mp4", 0)        == "1115-x.mp4");
}

TEST(output_path, the_time_and_the_number_compose) {
  EXPECT_TRUE(exp_("s-%t-%03d.png", 12)
              == "s-20260924-111530-012.png");
}

// The fallback: a template with no conversion of its own still cannot
// clobber itself, and the suffix goes BEFORE the extension.
TEST(output_path, a_template_with_no_conversion_gets_a_suffix) {
  EXPECT_TRUE(exp_("out/f.png", 0)    == "out/f.png");
  EXPECT_TRUE(exp_("out/f.png", 1)    == "out/f-000001.png");
  EXPECT_TRUE(exp_("out/f.png", 1, 3) == "out/f-001.png");
}

// A dot in a DIRECTORY is not an extension.
TEST(output_path, the_suffix_skips_a_dot_in_a_directory_name) {
  EXPECT_TRUE(exp_("out.d/f", 2) == "out.d/f-000002");
}

// The fallback applies only when the template did not number itself.
TEST(output_path, a_numbered_template_gets_no_extra_suffix) {
  EXPECT_TRUE(exp_("f-%02d.png", 3) == "f-03.png");
}

// ---- check() ------------------------------------------------------

TEST(output_path, check_reports_which_tokens_a_template_carries) {
  Tokens t;
  EXPECT_TRUE(outpath::check("plain.png", &t).empty());
  EXPECT_TRUE(!t.sequence && !t.time);
  EXPECT_TRUE(outpath::check("f-%04d.png", &t).empty());
  EXPECT_TRUE(t.sequence && !t.time);
  EXPECT_TRUE(outpath::check("f-%t.png", &t).empty());
  EXPECT_TRUE(!t.sequence && t.time);
  EXPECT_TRUE(outpath::check("f-%t-%d.png", &t).empty());
  EXPECT_TRUE(t.sequence && t.time);
}

TEST(output_path, check_refuses_an_unterminated_time_format) {
  EXPECT_TRUE(!outpath::check("f-%t{%Y.png", nullptr).empty());
}

// A '/' would name a directory nothing creates, and would leave the
// confined prefix the stage already resolved.
TEST(output_path, check_refuses_a_separator_in_a_time_format) {
  EXPECT_TRUE(!outpath::check("f-%t{%Y/%m}.png", nullptr).empty());
  EXPECT_TRUE(!outpath::check("f-%t{%Y\\%m}.png", nullptr).empty());
}

TEST(output_path, check_refuses_an_absurd_field_width) {
  EXPECT_TRUE(!outpath::check("f-%999d.png", nullptr).empty());
  EXPECT_TRUE(outpath::check("f-%32d.png", nullptr).empty());
}

TEST(output_path, a_doubled_percent_is_not_a_token) {
  Tokens t;
  EXPECT_TRUE(outpath::check("100%%done.png", &t).empty());
  EXPECT_TRUE(!t.sequence && !t.time);
}

// ---- the picker ---------------------------------------------------

TEST(output_path, the_picker_counts_up_and_overwrites_by_default) {
  const string d = tmp_dir_();
  SeqPicker p;
  p.configure(false);
  const string tmpl = d + "/f-%02d.png";
  const string a = p.take(tmpl, 6);
  touch_(a);
  const string b = p.take(tmpl, 6);
  EXPECT_TRUE(a == d + "/f-00.png");
  EXPECT_TRUE(b == d + "/f-01.png");
  // A relaunch rewinds: the file the operator is watching is rewritten.
  p.reset();
  EXPECT_TRUE(p.take(tmpl, 6) == a);
  filesystem::remove_all(d);
}

TEST(output_path, no_overwrite_starts_past_the_names_already_on_disk) {
  const string d = tmp_dir_();
  const string tmpl = d + "/f-%02d.png";
  touch_(d + "/f-00.png");
  touch_(d + "/f-01.png");
  touch_(d + "/f-02.png");
  SeqPicker p;
  p.configure(true);
  const string got = p.take(tmpl, 6);
  EXPECT_TRUE(got == d + "/f-03.png");
  // And the scan is not repaid per file: the next one is simply 04.
  EXPECT_TRUE(p.take(tmpl, 6) == d + "/f-04.png");
  filesystem::remove_all(d);
}

TEST(output_path, no_overwrite_fills_a_hole_it_has_not_passed_yet) {
  const string d = tmp_dir_();
  const string tmpl = d + "/f-%02d.png";
  touch_(d + "/f-00.png");
  touch_(d + "/f-02.png");
  SeqPicker p;
  p.configure(true);
  EXPECT_TRUE(p.take(tmpl, 6) == d + "/f-01.png");
  // 02 is taken, so the one after the hole is 03.
  EXPECT_TRUE(p.take(tmpl, 6) == d + "/f-03.png");
  filesystem::remove_all(d);
}

// The cursor is MONOTONIC: a file deleted behind the cursor is not
// re-used, so two writes in one run can never collide.
TEST(output_path, the_cursor_never_rewinds_within_a_run) {
  const string d = tmp_dir_();
  const string tmpl = d + "/f-%02d.png";
  SeqPicker p;
  p.configure(true);
  const string a = p.take(tmpl, 6);
  touch_(a);
  const string b = p.take(tmpl, 6);
  touch_(b);
  remove(a.c_str());                  // a viewer cleans up behind us
  EXPECT_TRUE(p.take(tmpl, 6) == d + "/f-02.png");
  filesystem::remove_all(d);
}

// A relaunch under no_overwrite resets the cursor, and the scan then
// lands past what the previous run wrote. This is the issue-38 case:
// run twice, keep both clips.
TEST(output_path, a_relaunch_under_no_overwrite_keeps_the_previous_run) {
  const string d = tmp_dir_();
  const string tmpl = d + "/clip-%03d.mp4";
  SeqPicker p;
  p.configure(true);
  const string r1 = p.take(tmpl, 6);
  touch_(r1);
  p.reset();                           // stop + relaunch
  const string r2 = p.take(tmpl, 6);
  EXPECT_TRUE(r1 == d + "/clip-000.mp4");
  EXPECT_TRUE(r2 == d + "/clip-001.mp4");
  filesystem::remove_all(d);
}

// Without a conversion the fallback suffix is what varies, so the scan
// still terminates and still never overwrites.
TEST(output_path, no_overwrite_works_through_the_fallback_suffix) {
  const string d = tmp_dir_();
  const string tmpl = d + "/only.wav";
  touch_(tmpl);
  SeqPicker p;
  p.configure(true);
  EXPECT_TRUE(p.take(tmpl, 3) == d + "/only-001.wav");
  filesystem::remove_all(d);
}

// A network URL has nothing to stat; the scan must not try, and must
// not loop forever deciding.
TEST(output_path, a_network_url_is_never_scanned) {
  SeqPicker p;
  p.configure(true);
  const string u = "rtmp://example.invalid/live/key";
  EXPECT_TRUE(p.take(u, 6) == u);
  filesystem::path();                  // (no filesystem access expected)
}
