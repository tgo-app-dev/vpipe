#include "generative-models/qwen-image/qwen-image21-layout.h"

#include <algorithm>

namespace vpipe {
namespace genai {
namespace qi21 {

namespace {

// Each encoder-side image slot stands for a 2x2 group of latent tokens.
constexpr int kImgTokensPerSlot = 4;

}  // namespace

bool
build_layout(const std::vector<std::uint8_t>& slot,
             const std::vector<std::uint8_t>& valid,
             const std::vector<ImgBlock>& blocks, Layout* out,
             std::string* err)
{
  auto fail = [&](const char* m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (out == nullptr) { return fail("no output"); }
  *out = Layout{};
  if (blocks.empty()) { return fail("no image blocks"); }
  if (!valid.empty() && valid.size() != slot.size()) {
    return fail("valid mask length does not match the slot mask");
  }

  int slots = 0;
  for (std::uint8_t s : slot) { slots += (s != 0) ? 1 : 0; }
  long long want = 0;
  for (const ImgBlock& b : blocks) {
    if (b.frame <= 0 || b.grid_h <= 0 || b.grid_w <= 0) {
      return fail("an image block has a non-positive extent");
    }
    want += b.tokens();
  }
  // The reference raises here rather than truncating, and so does this:
  // a mismatch means the caller's slot bookkeeping and its geometry
  // disagree, and either one could be the wrong half.
  if (want != (long long)slots * kImgTokensPerSlot) {
    return fail("img_shapes and the slot mask account for different "
                "numbers of image tokens");
  }

  const int joint = (int)slot.size() + slots * (kImgTokensPerSlot - 1);
  out->joint_len = joint;
  out->is_image.assign((std::size_t)joint, 0);
  out->is_target.assign((std::size_t)joint, 0);
  out->block_id.assign((std::size_t)joint, -1);
  out->pos_t.assign((std::size_t)joint, 0);
  out->pos_h.assign((std::size_t)joint, 0);
  out->pos_w.assign((std::size_t)joint, 0);
  out->image_row.reserve((std::size_t)want);
  out->text_row.reserve(slot.size());
  out->text_src.reserve(slot.size());
  if (!valid.empty()) { out->key_valid.assign((std::size_t)joint, 1); }

  // ---- expand the encoder sequence, and label every row --------------
  //
  // Block boundaries come from the token COUNTS in `blocks`, never from
  // runs of image rows: two condition images with no text between them
  // form one run and must still be two blocks, or they would attend to
  // each other bidirectionally.
  int jr = 0;                       // joint row
  std::size_t bi = 0;               // which block we are filling
  int filled = 0;                   // tokens placed into block bi
  for (std::size_t e = 0; e < slot.size(); ++e) {
    const int reps = (slot[e] != 0) ? kImgTokensPerSlot : 1;
    for (int r = 0; r < reps; ++r, ++jr) {
      if (!valid.empty()) { out->key_valid[(std::size_t)jr] = valid[e]; }
      if (slot[e] == 0) {
        out->text_row.push_back(jr);
        out->text_src.push_back((std::int32_t)e);
        continue;
      }
      while (bi < blocks.size() && filled >= blocks[bi].tokens()) {
        ++bi;
        filled = 0;
      }
      if (bi >= blocks.size()) { return fail("ran out of image blocks"); }
      out->is_image[(std::size_t)jr] = 1;
      out->block_id[(std::size_t)jr] = (std::int32_t)bi;
      out->image_row.push_back(jr);
      ++filled;
    }
  }
  if (jr != joint) { return fail("joint length bookkeeping is wrong"); }

  out->image_len = (int)want;
  out->text_len = joint - out->image_len;
  const std::size_t last = blocks.size() - 1;
  out->target_len = blocks[last].tokens();
  for (int p = joint - 1, n = 0; p >= 0 && n < out->target_len; --p) {
    if (out->is_image[(std::size_t)p] != 0 &&
        out->block_id[(std::size_t)p] == (std::int32_t)last) {
      out->is_target[(std::size_t)p] = 1;
      ++n;
    }
  }
  out->prefix_len = 0;
  for (int p = 0; p < joint; ++p) {
    if (out->is_target[(std::size_t)p] != 0) { break; }
    ++out->prefix_len;
  }

  // ---- rotary coordinates --------------------------------------------
  //
  // Text advances a SHARED position on all three axes, one per row. An
  // image block FREEZES the frame axis at the position the preceding
  // text reached and lays its tokens on a height/width grid centred on
  // zero -- so a block's spatial coordinates do not depend on where it
  // sits in the sequence. After a block the shared position jumps by
  // max(h, w), not by the token count.
  {
    int cursor = 0, position = 0;
    for (const ImgBlock& b : blocks) {
      int start = cursor;
      while (start < joint && out->is_image[(std::size_t)start] == 0) {
        ++start;
      }
      if (start >= joint) { return fail("fewer image runs than blocks"); }
      for (int p = cursor; p < start; ++p) {
        out->pos_t[(std::size_t)p] = position;
        out->pos_h[(std::size_t)p] = position;
        out->pos_w[(std::size_t)p] = position;
        ++position;
      }
      const int n = b.tokens();
      const int hlo = centred_lo(b.grid_h);
      const int wlo = centred_lo(b.grid_w);
      for (int i = 0; i < n; ++i) {
        const int p = start + i;
        if (p >= joint) { return fail("an image block runs past the end"); }
        out->pos_t[(std::size_t)p] = position;
        // Row-major over (h, w) within the block, which is the order the
        // reference builds and the order the latents are packed in.
        out->pos_h[(std::size_t)p] = hlo + (i / b.grid_w) % b.grid_h;
        out->pos_w[(std::size_t)p] = wlo + (i % b.grid_w);
      }
      cursor = start + n;
      position += std::max(b.grid_h, b.grid_w);
    }
    for (int p = cursor; p < joint; ++p) {
      out->pos_t[(std::size_t)p] = position;
      out->pos_h[(std::size_t)p] = position;
      out->pos_w[(std::size_t)p] = position;
      ++position;
    }
  }

  // ---- segments -------------------------------------------------------
  //
  // One run per maximal stretch of equal block_id. A text run is causal;
  // an image run is one bidirectional block.
  {
    int s = 0;
    while (s < joint) {
      int e = s + 1;
      while (e < joint &&
             out->block_id[(std::size_t)e] == out->block_id[(std::size_t)s]) {
        ++e;
      }
      Segment seg;
      seg.start = s;
      seg.end = e;
      seg.block_id = out->block_id[(std::size_t)s];
      seg.is_text = seg.block_id < 0;
      out->segments.push_back(seg);
      s = e;
    }
  }
  return true;
}

const char*
system_prompt()
{
  return "Comprehend and analyze the provided prompt.";
}

std::string
system_prefix()
{
  return std::string("<|im_start|>system\n") + system_prompt() +
         "<|im_end|>\n";
}

std::string
prompt_template(int n_ref, const std::string& text)
{
  std::string out = system_prefix() + "<|im_start|>user\n";
  for (int i = 0; i < n_ref; ++i) {
    // A LEADING SPACE before every block after the first; the reference
    // builds its repeat that way and the tokenization differs without
    // it.
    if (i > 0) { out += " "; }
    out += "<image" + std::to_string(i + 1) + ">";
    out += "<|vision_start|><|image_pad|><|vision_end|>";
  }
  out += text;
  out += "<|im_end|>\n<|im_start|>assistant\n";
  return out;
}

}  // namespace qi21
}  // namespace genai
}  // namespace vpipe
