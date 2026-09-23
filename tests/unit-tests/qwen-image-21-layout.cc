// Qwen-Image-2.1's joint text/image sequence layout: qi21::build_layout.
//
// Pure integer arithmetic, so it needs no checkpoint and no GPU -- which
// is the point of having it as its own module. Two tiers:
//
//   invariants   self-contained. Shapes, the four-fold slot expansion,
//                block boundaries that do NOT come from runs of image
//                rows, the centred rotary grid, the segment cover, and
//                the refusals.
//   golden       against the reference's own build_token_metadata and
//                QwenImage21Rope, dumped by
//                tools/dump_qwen_image21_golden.py --what layout.
//                VPIPE_QWEN_IMAGE21_GOLDEN; skips unset.

#include "minitest.h"

#include "common/flex-data.h"
#include "generative-models/qwen-image/qwen-image21-layout.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using namespace vpipe::genai::qi21;

namespace {

// `cond` is evaluated ONCE. An ASSERT_TRUE(cond) followed by
// if (!(cond)) evaluates it twice, which is invisible for a
// comparison and doubles the work for a call -- a model loaded, or a
// forward run, twice per use.
#define Q21L_REQUIRE(cond)                                                 \
  do {                                                                     \
    if (!(cond)) {                                                         \
      report_expect_true_failure(#cond, __FILE__, __LINE__);               \
      return;                                                              \
    }                                                                      \
  } while (0)

// A slot mask with `text_runs` text tokens between successive images.
std::vector<std::uint8_t>
make_slots_(const std::vector<int>& text_runs,
            const std::vector<int>& slots_per_image)
{
  std::vector<std::uint8_t> s;
  for (std::size_t i = 0; i < slots_per_image.size(); ++i) {
    const int t = i < text_runs.size() ? text_runs[i] : 0;
    for (int k = 0; k < t; ++k) { s.push_back(0); }
    for (int k = 0; k < slots_per_image[i]; ++k) { s.push_back(1); }
  }
  return s;
}

std::string
golden_dir_()
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  return (g != nullptr && *g != '\0') ? std::string(g) : std::string();
}

std::vector<std::int32_t>
read_i32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<std::int32_t> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

}  // namespace

TEST(qwen_image_21_layout, expands_slots_four_fold_and_labels_rows)
{
  // One condition image of 2x2 latent tokens (so one encoder slot) and a
  // target of 4x4 (four slots), with text around them.
  const std::vector<std::uint8_t> slot = make_slots_({3, 2}, {1, 4});
  const std::vector<ImgBlock> blocks = {{1, 2, 2}, {1, 4, 4}};
  Layout L;
  std::string err;
  Q21L_REQUIRE(build_layout(slot, {}, blocks, &L, &err));

  // 5 text tokens stay 1 row each; 5 slots become 4 rows each.
  EXPECT_TRUE(L.joint_len == 5 + 5 * 4);
  EXPECT_TRUE(L.text_len == 5);
  EXPECT_TRUE(L.image_len == 4 + 16);
  EXPECT_TRUE(L.target_len == 16);
  EXPECT_TRUE(L.prefix_len == L.joint_len - 16);
  EXPECT_TRUE(L.image_row.size() == 20);

  // Exactly the target block's rows are marked, and they are the tail.
  int targets = 0;
  for (int p = 0; p < L.joint_len; ++p) {
    targets += L.is_target[(std::size_t)p] != 0 ? 1 : 0;
    if (p >= L.prefix_len) { EXPECT_TRUE(L.is_target[(std::size_t)p] != 0); }
  }
  EXPECT_TRUE(targets == 16);
  // Every image row carries its block, every text row carries none.
  for (int p = 0; p < L.joint_len; ++p) {
    if (L.is_image[(std::size_t)p] != 0) {
      EXPECT_TRUE(L.block_id[(std::size_t)p] >= 0);
    } else {
      EXPECT_TRUE(L.block_id[(std::size_t)p] == -1);
    }
  }
}

// The boundary between two images comes from the TOKEN COUNTS, not from
// runs of image rows -- two adjacent condition images with no text
// between them are one run and must stay two blocks, or they would
// attend to each other bidirectionally.
TEST(qwen_image_21_layout, adjacent_images_stay_separate_blocks)
{
  const std::vector<std::uint8_t> slot = make_slots_({2, 0, 0}, {1, 1, 4});
  const std::vector<ImgBlock> blocks = {{1, 2, 2}, {1, 2, 2}, {1, 4, 4}};
  Layout L;
  std::string err;
  Q21L_REQUIRE(build_layout(slot, {}, blocks, &L, &err));

  // The two condition images are contiguous in the joint sequence --
  // rows 2..5 are the first (one slot, four tokens) and 6..9 the second,
  // with no text row between them...
  EXPECT_TRUE(L.is_image[2] != 0 && L.is_image[9] != 0);
  // ...and still carry different block ids.
  EXPECT_TRUE(L.block_id[2] == 0);
  EXPECT_TRUE(L.block_id[5] == 0);
  EXPECT_TRUE(L.block_id[6] == 1);
  EXPECT_TRUE(L.block_id[9] == 1);
  EXPECT_TRUE(L.block_id[10] == 2);
  // The segment cover must split them rather than merging the run.
  int image_segs = 0;
  for (const Segment& s : L.segments) {
    if (!s.is_text) { ++image_segs; }
  }
  EXPECT_TRUE(image_segs == 3);
  // Segments tile [0, joint_len) exactly, in order.
  int at = 0;
  for (const Segment& s : L.segments) {
    EXPECT_TRUE(s.start == at);
    EXPECT_TRUE(s.end > s.start);
    at = s.end;
  }
  EXPECT_TRUE(at == L.joint_len);
}

// The rotary grid is centred on zero with the EXTRA row on the negative
// side, and the frame axis freezes across a block then jumps by
// max(h, w) -- not by the token count.
TEST(qwen_image_21_layout, rotary_grid_is_centred_and_frame_jumps)
{
  EXPECT_TRUE(centred_lo(4) == -2);     // [-2,-1,0,1]
  EXPECT_TRUE(centred_lo(5) == -3);     // [-3,-2,-1,0,1]
  EXPECT_TRUE(centred_lo(1) == -1);     // [-1]
  EXPECT_TRUE(centred_lo(2) == -1);     // [-1,0]

  // 2 text, a 2x2 image, 1 text, a 2x2 target. Every image is a whole
  // number of encoder slots by construction -- a slot IS four latent
  // tokens -- so the geometries here are all multiples of four.
  const std::vector<std::uint8_t> slot = {0, 0, 1, 0, 1};
  const std::vector<ImgBlock> blocks = {{1, 2, 2}, {1, 2, 2}};
  Layout L;
  std::string err;
  Q21L_REQUIRE(build_layout(slot, {}, blocks, &L, &err));

  // rows: t t I I I I t I I I I
  EXPECT_TRUE(L.joint_len == 11);
  // Text advances the shared position one per row.
  EXPECT_TRUE(L.pos_t[0] == 0 && L.pos_t[1] == 1);
  EXPECT_TRUE(L.pos_h[0] == 0 && L.pos_w[1] == 1);
  // The first image freezes the frame axis at 2 for all four rows.
  for (int p = 2; p < 6; ++p) { EXPECT_TRUE(L.pos_t[(std::size_t)p] == 2); }
  // ...and lays out a centred 2x2: (h,w) over {-1,0} x {-1,0}.
  EXPECT_TRUE(L.pos_h[2] == -1 && L.pos_w[2] == -1);
  EXPECT_TRUE(L.pos_h[3] == -1 && L.pos_w[3] == 0);
  EXPECT_TRUE(L.pos_h[4] == 0 && L.pos_w[4] == -1);
  EXPECT_TRUE(L.pos_h[5] == 0 && L.pos_w[5] == 0);
  // After the block the shared position jumps by max(2,2) = 2, so the
  // next text row is at 2 + 2 = 4, NOT at 2 + 4 tokens.
  EXPECT_TRUE(L.pos_t[6] == 4);
  // The second image freezes at 5.
  for (int p = 7; p < 11; ++p) { EXPECT_TRUE(L.pos_t[(std::size_t)p] == 5); }
}

// Left padding must be excluded as a KEY and kept as a QUERY, so the
// mask rides the joint sequence rather than being sliced off a prefix.
TEST(qwen_image_21_layout, padding_rides_the_joint_sequence)
{
  const std::vector<std::uint8_t> slot = {0, 0, 1, 0, 1};
  const std::vector<std::uint8_t> valid = {0, 1, 1, 1, 1};   // left-padded
  const std::vector<ImgBlock> blocks = {{1, 2, 2}, {1, 2, 2}};
  Layout L;
  std::string err;
  Q21L_REQUIRE(build_layout(slot, valid, blocks, &L, &err));
  Q21L_REQUIRE(L.key_valid.size() == (std::size_t)L.joint_len);
  EXPECT_TRUE(L.key_valid[0] == 0);
  for (int p = 1; p < L.joint_len; ++p) {
    EXPECT_TRUE(L.key_valid[(std::size_t)p] != 0);
  }
}

TEST(qwen_image_21_layout, refuses_a_geometry_that_does_not_add_up)
{
  std::string err;
  Layout L;
  // Three slots = 12 image rows, but the blocks account for 8.
  EXPECT_FALSE(build_layout({1, 1, 1}, {}, {{1, 2, 2}, {1, 2, 2}}, &L, &err));
  EXPECT_FALSE(err.empty());
  err.clear();
  EXPECT_FALSE(build_layout({1}, {}, {}, &L, &err));
  EXPECT_FALSE(err.empty());
  err.clear();
  EXPECT_FALSE(build_layout({1}, {}, {{1, 0, 4}}, &L, &err));
  EXPECT_FALSE(err.empty());
  err.clear();
  // A valid mask of the wrong length is a caller bug, not something to
  // pad around.
  EXPECT_FALSE(build_layout({1, 0}, {1}, {{1, 2, 2}}, &L, &err));
  EXPECT_FALSE(err.empty());
}

// ---- against the reference ------------------------------------------

TEST(qwen_image_21_layout, matches_reference_metadata_and_rope)
{
  const std::string gd = golden_dir_();
  if (gd.empty()) { return; }
  std::ifstream mf(gd + "/layout.json");
  if (!mf) { return; }
  FlexData man;
  try {
    man = FlexData::from_json(mf);
  } catch (...) {
    return;
  }
  if (!man.is_object()) { return; }
  auto o = man.as_object();

  auto arr_i32 = [&](const char* k) {
    std::vector<std::int32_t> v;
    if (!o.contains(k)) { return v; }
    const FlexData f = o.at(k);
    if (!f.is_array()) { return v; }
    auto a = f.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      v.push_back((std::int32_t)a.at(i).as_int(0));
    }
    return v;
  };

  const std::vector<std::int32_t> slot_i = arr_i32("slot");
  const std::vector<std::int32_t> valid_i = arr_i32("valid");
  const std::vector<std::int32_t> shapes = arr_i32("img_shapes");
  Q21L_REQUIRE(!slot_i.empty());
  Q21L_REQUIRE(shapes.size() % 3 == 0);

  std::vector<std::uint8_t> slot(slot_i.begin(), slot_i.end());
  std::vector<std::uint8_t> valid(valid_i.begin(), valid_i.end());
  std::vector<ImgBlock> blocks;
  for (std::size_t i = 0; i + 2 < shapes.size(); i += 3) {
    blocks.push_back(ImgBlock{shapes[i], shapes[i + 1], shapes[i + 2]});
  }

  Layout L;
  std::string err;
  Q21L_REQUIRE(build_layout(slot, valid, blocks, &L, &err));

  const std::vector<std::int32_t> ids = read_i32_(gd + "/lay_image_ids.i32");
  const std::vector<std::int32_t> tgt = read_i32_(gd + "/lay_target.i32");
  const std::vector<std::int32_t> ft = read_i32_(gd + "/lay_pos_t.i32");
  const std::vector<std::int32_t> fh = read_i32_(gd + "/lay_pos_h.i32");
  const std::vector<std::int32_t> fw = read_i32_(gd + "/lay_pos_w.i32");
  Q21L_REQUIRE(!ids.empty());
  Q21L_REQUIRE(ids.size() == (std::size_t)L.joint_len);

  int bad_id = 0, bad_tgt = 0, bad_t = 0, bad_h = 0, bad_w = 0;
  for (int p = 0; p < L.joint_len; ++p) {
    const std::size_t i = (std::size_t)p;
    if (ids[i] != L.block_id[i]) { ++bad_id; }
    if (!tgt.empty() && (tgt[i] != 0) != (L.is_target[i] != 0)) { ++bad_tgt; }
    if (!ft.empty() && ft[i] != L.pos_t[i]) { ++bad_t; }
    if (!fh.empty() && fh[i] != L.pos_h[i]) { ++bad_h; }
    if (!fw.empty() && fw[i] != L.pos_w[i]) { ++bad_w; }
  }
  std::printf("[qwen_image_21_layout] joint=%d prefix=%d target=%d; "
              "mismatches id=%d target=%d t=%d h=%d w=%d\n",
              L.joint_len, L.prefix_len, L.target_len, bad_id, bad_tgt,
              bad_t, bad_h, bad_w);
  EXPECT_TRUE(bad_id == 0);
  EXPECT_TRUE(bad_tgt == 0);
  EXPECT_TRUE(bad_t == 0);
  EXPECT_TRUE(bad_h == 0);
  EXPECT_TRUE(bad_w == 0);
  if (o.contains("prefix_len")) {
    EXPECT_TRUE(L.prefix_len == (int)o.at("prefix_len").as_int(-1));
  }
}

// ---- the conditioning prompt ----------------------------------------

// Checked as a STRING against the reference's own rendered template,
// because that is what decides the tokenization and every part of it is
// silent when wrong: this model's system prompt is neither the
// "Describe the image by detailing..." of Krea-2 / Mage-Flow /
// Qwen-Image-2512 nor the "Describe the key features of the input
// image" of Qwen-Image-Edit / Boogu, it keeps a trailing generation
// turn that Boogu drops, and it labels references `<imageN>` rather
// than "Picture N: ".
//
// Golden from tools/dump_qwen_image21_golden.py --what cond, which
// needs only the 11 MB tokenizer -- not the 17.5 GB encoder.
TEST(qwen_image_21_layout, prompt_matches_the_reference_template)
{
  const std::string gd = golden_dir_();
  if (gd.empty()) { return; }
  std::ifstream f(gd + "/cond.json");
  if (!f) { return; }
  FlexData man;
  try {
    man = FlexData::from_json(f);
  } catch (...) {
    return;
  }
  if (!man.is_object()) { return; }
  auto o = man.as_object();
  auto str = [&](const char* k) {
    return o.contains(k) ? std::string(o.at(k).as_string("")) : std::string();
  };

  const std::string prompt = str("prompt");
  Q21L_REQUIRE(!prompt.empty());
  EXPECT_TRUE(std::string(system_prompt()) == str("sys_prompt"));
  EXPECT_TRUE(system_prefix() == str("sys_prefix"));

  // Text-to-image, then one, two and three references. The multi-ref
  // cases are where the leading space between blocks lives.
  const std::string got_t2i = prompt_template(0, prompt);
  if (got_t2i != str("t2i_rendered")) {
    std::printf("[qwen_image_21_layout] t2i mismatch\n  got  %s\n  want %s\n",
                got_t2i.c_str(), str("t2i_rendered").c_str());
  }
  EXPECT_TRUE(got_t2i == str("t2i_rendered"));
  for (int n = 1; n <= 3; ++n) {
    const std::string key = "ti2i_" + std::to_string(n) + "_rendered";
    const std::string want = str(key.c_str());
    if (want.empty()) { continue; }
    const std::string got = prompt_template(n, prompt);
    if (got != want) {
      std::printf("[qwen_image_21_layout] %d-ref mismatch\n  got  %s\n"
                  "  want %s\n", n, got.c_str(), want.c_str());
    }
    EXPECT_TRUE(got == want);
  }
  // The DROP is derived, not hardcoded: the reference counts the tokens
  // of the system turn, and this renders the same string for the same
  // tokenizer to count. So the string matching is what makes the counts
  // agree -- but assert the reference's own number is self-consistent
  // too, since a change to either would otherwise pass here silently.
  if (o.contains("drop_idx") && o.contains("sys_prefix_ids")) {
    const FlexData ids = o.at("sys_prefix_ids");
    if (ids.is_array()) {
      const int nid = (int)ids.as_array().size();
      EXPECT_TRUE((int)o.at("drop_idx").as_int(-1) == nid);
    }
  }
  std::printf("[qwen_image_21_layout] prompt template matches the reference "
              "(t2i + 1/2/3 refs); drop %d tokens\n",
              (int)(o.contains("drop_idx") ? o.at("drop_idx").as_int(-1) : -1));
}
