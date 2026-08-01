// Gaussian-Shading watermark: bit-compatibility with Microsoft's detector.
//
// The detector recovers the initial noise and reads each entry's SIGN against
// an expected half-bit derived from the key, so the pad / pos arrays that
// `numpy.random.default_rng(key)` produces are a hard interface -- get them
// wrong and the mark is unreadable even though the image is fine. This checks
// vpipe's reimplementation of numpy's SeedSequence + PCG64 + Lemire bounded
// integers against a real numpy dump (scratchpad/dump_wm_golden.py), for an
// integer key, a passphrase (SHA-256 -> 256-bit int) and a >64-bit decimal.
//
// Env: VPIPE_MAGE_WM_GOLDEN = that dump dir. Skips vacuously if unset -- but
// the self-consistency and distribution checks below need no golden and
// always run.

#include "minitest.h"

#include "generative-models/mage/mage-watermark.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe::genai;

namespace {

constexpr std::size_t kN = 4096;   // must match the dump script

template <typename T>
std::vector<T>
read_bin_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<T> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / sizeof(T));
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

std::string
golden_dir_()
{
  const char* g = std::getenv("VPIPE_MAGE_WM_GOLDEN");
  if (g == nullptr || *g == '\0') { return ""; }
  if (!std::filesystem::exists(std::string(g) + "/msg.u8")) { return ""; }
  return g;
}

}  // namespace

// The 256-bit payload expansion (SHA-256, LSB-first per byte).
TEST(mage_watermark, payload_bits_match_numpy)
{
  const std::string G = golden_dir_();
  if (G.empty()) { return; }
  const std::vector<std::uint8_t> g = read_bin_<std::uint8_t>(G + "/msg.u8");
  const std::vector<std::uint8_t> v =
      mage_wm::payload_bits(mage_wm::kDefaultPayload);
  ASSERT_TRUE(g.size() == (std::size_t)mage_wm::kMsgBits);
  ASSERT_TRUE(v.size() == g.size());
  std::size_t bad = 0;
  for (std::size_t i = 0; i < g.size(); ++i) { bad += (v[i] != g[i]); }
  std::printf("[mage_watermark] payload bits mismatched = %zu / %zu\n", bad,
              g.size());
  EXPECT_TRUE(bad == 0);
}

// The key-seeded pad + message-index map: numpy SeedSequence -> PCG64 ->
// Lemire. This is the part that must be bit-exact.
TEST(mage_watermark, pad_and_pos_match_numpy)
{
  const std::string G = golden_dir_();
  if (G.empty()) { return; }
  struct Case { const char* tag; const char* key; };
  const Case cases[] = {
    {"int",    "20260720"},
    {"phrase", "correct horse battery"},
    {"big",    "123456789012345678901234567890"},
  };
  for (const Case& c : cases) {
    const std::vector<std::uint8_t> gpad =
        read_bin_<std::uint8_t>(G + "/pad_" + c.tag + ".u8");
    const std::vector<std::uint16_t> gpos =
        read_bin_<std::uint16_t>(G + "/pos_" + c.tag + ".u16");
    if (gpad.size() != kN || gpos.size() != kN) {
      std::printf("[mage_watermark] %s: golden missing; skipping\n", c.tag);
      continue;
    }
    std::vector<std::uint8_t> pad;
    std::vector<std::uint16_t> pos;
    mage_wm::pad_and_pos(kN, c.key, pad, pos);
    ASSERT_TRUE(pad.size() == kN && pos.size() == kN);
    std::size_t bpad = 0, bpos = 0;
    for (std::size_t i = 0; i < kN; ++i) {
      bpad += (pad[i] != gpad[i]);
      bpos += (pos[i] != gpos[i]);
    }
    std::printf("[mage_watermark] %-7s pad mismatched=%zu pos mismatched=%zu "
                "(of %zu)\n", c.tag, bpad, bpos, kN);
    EXPECT_TRUE(bpad == 0);
    EXPECT_TRUE(bpos == 0);
  }
}

// The same reproduction, but far enough into the PCG64 stream to matter. The
// case above stops at 4096 entries; a 1536x1024 edit draws 128*96*64 = 786432
// pad values and then as many pos values, so a bounded-integer path that only
// agrees with numpy for the first few thousand draws (numpy buffers bits for
// the range-2 `pad` draw and uses Lemire rejection for `pos`) would pass the
// small golden and still de-randomize the watermark at real resolutions.
TEST(mage_watermark, pad_and_pos_match_numpy_at_edit_scale)
{
  const std::string G = golden_dir_();
  if (G.empty()) { return; }
  constexpr std::size_t kBig = 128u * 96u * 64u;
  const std::vector<std::uint8_t> gpad =
      read_bin_<std::uint8_t>(G + "/pad_n786432.u8");
  const std::vector<std::uint16_t> gpos =
      read_bin_<std::uint16_t>(G + "/pos_n786432.u16");
  if (gpad.size() != kBig || gpos.size() != kBig) {
    std::printf("[mage_watermark] large-n golden missing; skipping\n");
    return;
  }
  std::vector<std::uint8_t> pad;
  std::vector<std::uint16_t> pos;
  mage_wm::pad_and_pos(kBig, "20260720", pad, pos);
  ASSERT_TRUE(pad.size() == kBig && pos.size() == kBig);
  std::size_t bpad = 0, bpos = 0, first = kBig;
  for (std::size_t i = 0; i < kBig; ++i) {
    if (pad[i] != gpad[i]) { bpad++; if (i < first) { first = i; } }
    if (pos[i] != gpos[i]) { bpos++; if (i < first) { first = i; } }
  }
  std::printf("[mage_watermark] n=%zu pad mismatched=%zu pos mismatched=%zu"
              " (first divergence at %zu)\n", kBig, bpad, bpos, first);
  EXPECT_TRUE(bpad == 0);
  EXPECT_TRUE(bpos == 0);
}

// End to end on our own side: the encoded noise must be detectable (raw
// accuracy ~1) under the same key, undetectable (~0.5) under a different one,
// and still look like standard normal noise.
TEST(mage_watermark, encoded_noise_is_detectable_and_normal)
{
  const int C = 128, H = 16, W = 16;
  const std::size_t n = (std::size_t)C * H * W;
  const std::vector<float> tok =
      mage_wm::encode_noise(C, H, W, mage_wm::kDefaultKey, 42);
  ASSERT_TRUE(tok.size() == n);

  // encode_noise returns TOKEN-major; the detector reads channel-first.
  const std::size_t hw = (std::size_t)H * W;
  std::vector<float> chw(n);
  for (std::size_t t = 0; t < hw; ++t) {
    for (int c = 0; c < C; ++c) {
      chw[(std::size_t)c * hw + t] = tok[t * (std::size_t)C + c];
    }
  }
  const double hit = mage_wm::raw_accuracy(chw, mage_wm::kDefaultKey);
  const double miss = mage_wm::raw_accuracy(chw, "99999999");
  std::printf("[mage_watermark] raw accuracy: same key %.4f, wrong key %.4f\n",
              hit, miss);
  EXPECT_TRUE(hit > 0.999);            // the sign IS the payload
  EXPECT_TRUE(miss > 0.42 && miss < 0.58);   // chance

  // Distribution preserved: the watermark only chooses which half-plane each
  // entry lands in, so the sample must still be ~N(0,1).
  double mean = 0.0;
  for (const float v : tok) { mean += (double)v; }
  mean /= (double)n;
  double var = 0.0;
  double amax = 0.0;
  for (const float v : tok) {
    const double d = (double)v - mean;
    var += d * d;
    amax = std::max(amax, std::fabs((double)v));
  }
  const double sd = std::sqrt(var / (double)n);
  std::printf("[mage_watermark] noise mean=%.4f std=%.4f absmax=%.2f\n", mean,
              sd, amax);
  EXPECT_TRUE(std::fabs(mean) < 0.05);
  EXPECT_TRUE(sd > 0.95 && sd < 1.05);
  EXPECT_TRUE(std::isfinite(amax) && amax < 6.0);
}
