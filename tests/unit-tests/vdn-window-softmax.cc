// VDN-H3's windowed softmax against the reference's own implementation.
//
// All FOUR anchor modes are checked, not just the released "both". The
// modes differ only in which of two extra clauses is OR'd into the mask,
// so a port that implements one of them and silently falls back for the
// rest looks correct on the configuration it was tested with -- and the
// mode is a CROSS-BRANCH fact: "both" is what lets the linear branch
// drop frames 0 and F-1, so getting it wrong double-counts them.
//
// Regenerate: ~/dock/dump/vpipe-test/vdn/gen_goldens.py softmax

#include "minitest.h"

#include "common/flex-data.h"
#include "generative-models/minimax-h3/vdn-geometry.h"
#include "generative-models/minimax-h3/vdn-window-softmax.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai::minimax_h3;

namespace {

bool
read_bin_(const std::string& path, std::vector<float>* out)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) { return false; }
  const std::streamsize n = in.tellg();
  if (n <= 0) { return false; }
  in.seekg(0);
  out->resize((std::size_t)n / sizeof(float));
  return (bool)in.read((char*)out->data(), n);
}

double
rel_l2_(const std::vector<float>& got, const std::vector<float>& want)
{
  if (got.size() != want.size() || got.empty()) { return -1.0; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = (double)got[i] - (double)want[i];
    num += d * d;
    den += (double)want[i] * (double)want[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

TEST(vdn_window_softmax, all_four_anchor_modes_match_the_reference)
{
  std::string dir;
  if (const char* e = std::getenv("VPIPE_VDN_SOFTMAX_GOLDEN")) { dir = e; }
  else if (const char* h = std::getenv("HOME")) {
    dir = std::string(h) + "/dock/dump/vpipe-test/vdn/softmax";
  } else {
    return;
  }
  std::ifstream mi(dir + "/meta.json");
  if (!mi) { return; }        // dock-side artifact; see gen_goldens.py
  FlexData meta = FlexData::from_json(mi);
  ASSERT_TRUE(meta.is_object());
  if (!meta.is_object()) { return; }
  auto m = meta.as_object();
  const int seq = (int)m.at("seq_len").as_int(0);
  const int H   = (int)m.at("heads").as_int(0);
  const int d   = (int)m.at("head_dim").as_int(0);
  const float scale = (float)m.at("scale").as_real(0.0);

  vdn::WindowMask mask;
  mask.seq_len          = seq;
  mask.video_start      = (int)m.at("video_start").as_int(0);
  mask.num_frames       = (int)m.at("num_frames").as_int(0);
  mask.tokens_per_frame = (int)m.at("tokens_per_frame").as_int(0);
  mask.bounds = vdn::window_bounds(mask.num_frames,
                                   (int)m.at("radius").as_int(0),
                                   (int)m.at("chunk").as_int(0));
  {
    // The bounds are also in the golden; disagreeing with them would
    // make every mode wrong in the same way and look like an attention
    // bug rather than a geometry one.
    FlexData bo = m.at("bounds");
    auto ba = bo.as_array();
    ASSERT_TRUE(ba.size() == mask.bounds.size());
    bool same = ba.size() == mask.bounds.size();
    for (std::size_t i = 0; same && i < ba.size(); ++i) {
      FlexData p = ba.at(i);
      auto pa = p.as_array();
      same = pa.size() == 2 && (int)pa.at(0).as_int(0) == mask.bounds[i].lo
             && (int)pa.at(1).as_int(0) == mask.bounds[i].hi;
    }
    EXPECT_TRUE(same);
  }

  std::vector<float> q, k, v, x;
  ASSERT_TRUE(read_bin_(dir + "/q.bin", &q));
  ASSERT_TRUE(read_bin_(dir + "/k.bin", &k));
  ASSERT_TRUE(read_bin_(dir + "/v.bin", &v));
  ASSERT_TRUE(read_bin_(dir + "/x.bin", &x));
  if (q.size() != (std::size_t)seq * H * d) { return; }

  const char* modes[] = {"none", "columns", "rows", "both"};
  int ran = 0;
  for (const char* mode : modes) {
    std::vector<float> want;
    if (!read_bin_(dir + "/out_" + std::string(mode) + ".bin", &want)) {
      EXPECT_TRUE(false);
      continue;
    }
    ASSERT_TRUE(vdn::parse_anchor_frames(mode, &mask.anchors));
    std::vector<float> got(q.size());
    vdn::window_softmax(q.data(), k.data(), v.data(), mask, H, d, scale,
                        got.data());
    const double e = rel_l2_(got, want);
    const bool ok = e >= 0.0 && e < 1e-6;
    EXPECT_TRUE(ok);
    if (!ok) { std::printf("[vdn_softmax] %s rel-L2 %.3e\n", mode, e); }
    ++ran;
  }
  EXPECT_TRUE(ran == 4);

  // The modes must actually DIFFER, or the loop above proves nothing:
  // four identical goldens would pass against an implementation that
  // ignored the mode entirely.
  std::vector<float> a, b;
  ASSERT_TRUE(read_bin_(dir + "/out_none.bin", &a));
  ASSERT_TRUE(read_bin_(dir + "/out_both.bin", &b));
  EXPECT_TRUE(rel_l2_(a, b) > 1e-3);

  // --- the gate
  std::vector<float> gw, gb, want_gate, want_gated, attn_both;
  ASSERT_TRUE(read_bin_(dir + "/w_softmax_gate_up_weight.bin", &gw));
  ASSERT_TRUE(read_bin_(dir + "/w_softmax_gate_up_bias.bin", &gb));
  ASSERT_TRUE(read_bin_(dir + "/gated.bin", &want_gated));
  ASSERT_TRUE(read_bin_(dir + "/out_both.bin", &attn_both));
  std::vector<float> gated(attn_both.size());
  vdn::apply_softmax_gate(attn_both.data(), x.data(), seq, H, d,
                          (int)m.at("hidden").as_int(0), gw.data(), gb.data(),
                          gated.data());
  const double eg = rel_l2_(gated, want_gated);
  const bool gok = eg >= 0.0 && eg < 1e-6;
  EXPECT_TRUE(gok);
  if (!gok) { std::printf("[vdn_softmax] gate rel-L2 %.3e\n", eg); }
}

TEST(vdn_window_softmax, a_global_row_stays_dense_both_ways)
{
  // The prompt and the soundtrack are read exactly, in BOTH directions:
  // a global query sees the whole sequence and a global key is visible
  // to every query. The linear branch assumes it -- it covers only what
  // the window cannot see among VIDEO rows -- so a window that clipped
  // the globals would lose them from both halves at once.
  vdn::WindowMask mask;
  mask.seq_len = 40;
  mask.video_start = 4;         // 4 text rows, then 6 frames of 5, then 6
  mask.num_frames = 6;
  mask.tokens_per_frame = 5;
  mask.anchors = vdn::AnchorFrames::kNone;
  mask.bounds = vdn::window_bounds(6, 0, 0);   // the tightest window there is

  bool dense = true;
  for (int g = 0; g < mask.seq_len; ++g) {
    const bool g_global = g < mask.video_start || g >= mask.video_end();
    if (!g_global) { continue; }
    for (int o = 0; o < mask.seq_len; ++o) {
      dense = dense && mask.allows(g, o) && mask.allows(o, g);
    }
  }
  EXPECT_TRUE(dense);

  // ...while a video query with radius 0 sees only its own frame.
  int seen = 0;
  const int qi = mask.video_start + 3 * mask.tokens_per_frame;
  for (int o = mask.video_start; o < mask.video_end(); ++o) {
    if (mask.allows(qi, o)) { ++seen; }
  }
  EXPECT_TRUE(seen == mask.tokens_per_frame);
}

// ---- the mask as spans, and the kernel that consumes them ------------

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <cstring>

using namespace vpipe::metal_compute;

TEST(vdn_window_softmax, spans_agree_with_the_predicate)
{
  // Two representations of one mask: `allows(q, k)` is what the goldens
  // pin, and the CSR spans are what a kernel can walk. Checking them
  // against each other over EVERY pair is the whole safety of the
  // second, because nothing downstream can tell a wrong span from a
  // wrong model -- attention over the wrong key set still returns a
  // normalised distribution.
  //
  // All four anchor modes and both window shapes, including geometries
  // where the window is wider than the clip (every mode collapses to
  // dense) and where it is tighter than one chunk.
  for (int mode_i = 0; mode_i < 4; ++mode_i) {
    for (int chunk = 0; chunk <= 5; chunk += 5) {
      for (int radius = 0; radius <= 2; ++radius) {
        for (int F : {1, 2, 3, 17}) {
          vdn::WindowMask mask;
          mask.video_start = 4;
          mask.num_frames = F;
          mask.tokens_per_frame = 3;
          mask.seq_len = 4 + F * 3 + 2;
          mask.anchors = (vdn::AnchorFrames)mode_i;
          mask.bounds = vdn::window_bounds(F, radius, chunk);

          const vdn::WindowSpans sp = vdn::build_window_spans(mask);
          ASSERT_TRUE((int)sp.off.size() == sp.groups + 1);
          if ((int)sp.off.size() != sp.groups + 1) { return; }

          bool ok = true;
          for (int q = 0; q < mask.seq_len && ok; ++q) {
            const int ve = mask.video_end();
            const int grp = (q < mask.video_start || q >= ve)
                                ? 0
                                : 1 + (q - mask.video_start)
                                          / mask.tokens_per_frame;
            // The spans must be sorted and DISJOINT, or the online
            // softmax counts a key twice.
            int prev_end = -1;
            for (int s = sp.off[grp]; s < sp.off[grp + 1] && ok; ++s) {
              ok = sp.start[s] >= prev_end && sp.end[s] > sp.start[s];
              prev_end = sp.end[s];
            }
            for (int k = 0; k < mask.seq_len && ok; ++k) {
              bool in_span = false;
              for (int s = sp.off[grp]; s < sp.off[grp + 1]; ++s) {
                if (k >= sp.start[s] && k < sp.end[s]) { in_span = true; }
              }
              ok = in_span == mask.allows(q, k);
            }
          }
          EXPECT_TRUE(ok);
          if (!ok) {
            std::printf("[vdn_softmax] spans != predicate: mode %d chunk %d "
                        "radius %d F %d\n", mode_i, chunk, radius, F);
          }
        }
      }
    }
  }
}

TEST(vdn_window_softmax, metal_spans_kernel_matches_the_cpu_reference)
{
  std::string dir;
  if (const char* e = std::getenv("VPIPE_VDN_SOFTMAX_GOLDEN")) { dir = e; }
  else if (const char* h = std::getenv("HOME")) {
    dir = std::string(h) + "/dock/dump/vpipe-test/vdn/softmax";
  } else {
    return;
  }
  std::ifstream mi(dir + "/meta.json");
  if (!mi) { return; }
  FlexData meta = FlexData::from_json(mi);
  if (!meta.is_object()) { return; }
  auto m = meta.as_object();
  const int seq = (int)m.at("seq_len").as_int(0);
  const int H   = (int)m.at("heads").as_int(0);
  const int d   = (int)m.at("head_dim").as_int(0);
  const float scale = (float)m.at("scale").as_real(0.0);

  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  ComputeLibrary lib = mc->load_library("sdpa");
  ComputeFunction fn = lib.function("sdpa_spans_f16");
  ASSERT_TRUE(fn.valid());
  if (!fn.valid()) { return; }

  vdn::WindowMask mask;
  mask.seq_len          = seq;
  mask.video_start      = (int)m.at("video_start").as_int(0);
  mask.num_frames       = (int)m.at("num_frames").as_int(0);
  mask.tokens_per_frame = (int)m.at("tokens_per_frame").as_int(0);
  mask.bounds = vdn::window_bounds(mask.num_frames,
                                   (int)m.at("radius").as_int(0),
                                   (int)m.at("chunk").as_int(0));

  std::vector<float> q, k, v;
  if (!read_bin_(dir + "/q.bin", &q) || !read_bin_(dir + "/k.bin", &k)
      || !read_bin_(dir + "/v.bin", &v)) {
    EXPECT_TRUE(false);
    return;
  }

  // ROUND THE INPUTS TO f16 FIRST and give the CPU reference the same
  // rounded tensors. Otherwise the comparison is dominated by the
  // kernel's f16 storage and a genuine mask error -- which is what this
  // test is for -- would sit under the same 1e-3 as the precision.
  auto to_half = [](std::vector<float> x) {
    for (float& e : x) { e = (float)(__fp16)e; }
    return x;
  };
  const std::vector<float> qh = to_half(q), kh = to_half(k), vh = to_half(v);

  const std::size_t n = (std::size_t)seq * H * d;
  // The kernel reads [head, token, dim]; the goldens are [token, head,
  // dim]. Transposing here rather than in the kernel keeps it the same
  // shape as every other sdpa entry point in this file.
  auto pack = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(n * sizeof(__fp16));
    __fp16* p = (__fp16*)b.contents();
    for (int t = 0; t < seq; ++t) {
      for (int h = 0; h < H; ++h) {
        for (int i = 0; i < d; ++i) {
          p[((std::size_t)h * seq + t) * d + i] =
              (__fp16)src[((std::size_t)t * H + h) * d + i];
        }
      }
    }
    return b;
  };
  SharedBuffer qb = pack(qh), kb = pack(kh), vb = pack(vh);
  SharedBuffer ob = mc->make_shared_buffer(n * sizeof(__fp16));
  if (qb.empty() || ob.empty()) {
    EXPECT_TRUE(false);
    return;
  }

  const char* modes[] = {"none", "columns", "rows", "both"};
  for (const char* mode : modes) {
    ASSERT_TRUE(vdn::parse_anchor_frames(mode, &mask.anchors));
    const vdn::WindowSpans sp = vdn::build_window_spans(mask);
    auto ibuf = [&](const std::vector<int>& x) {
      SharedBuffer b = mc->make_shared_buffer(x.size() * sizeof(int));
      std::memcpy(b.contents(), x.data(), x.size() * sizeof(int));
      return b;
    };
    SharedBuffer offb = ibuf(sp.off), sb = ibuf(sp.start), eb = ibuf(sp.end);
    {
      CommandStream stream = mc->make_command_stream();
      ComputeEncoder enc = stream.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, qb);
      enc.set_buffer(1, kb);
      enc.set_buffer(2, vb);
      enc.set_buffer(3, ob);
      enc.set_constant(4, scale);
      enc.set_constant(5, seq);
      enc.set_constant(6, d);
      enc.set_constant(7, H);
      enc.set_constant(8, H);
      enc.set_constant(9, seq);
      enc.set_constant(10, seq);
      enc.set_buffer(11, offb);
      enc.set_buffer(12, sb);
      enc.set_buffer(13, eb);
      enc.set_constant(14, mask.video_start);
      enc.set_constant(15, mask.tokens_per_frame);
      enc.set_constant(16, mask.num_frames);
      enc.dispatch({32, (unsigned)H, (unsigned)seq}, {32, 1, 1});
      enc.end();
      std::string err;
      ASSERT_TRUE(stream.commit().wait_ok(&err));
    }
    std::vector<float> want(n);
    vdn::window_softmax(qh.data(), kh.data(), vh.data(), mask, H, d, scale,
                        want.data());
    std::vector<float> got(n);
    const __fp16* p = (const __fp16*)ob.contents();
    for (int t = 0; t < seq; ++t) {
      for (int h = 0; h < H; ++h) {
        for (int i = 0; i < d; ++i) {
          got[((std::size_t)t * H + h) * d + i] =
              (float)p[((std::size_t)h * seq + t) * d + i];
        }
      }
    }
    const double e = rel_l2_(got, want);
    // Same inputs, same key sets: what is left is f16 STORAGE of the
    // output and the accumulation order, not the mask.
    const bool ok = e >= 0.0 && e < 2e-3;
    EXPECT_TRUE(ok);
    if (!ok) { std::printf("[vdn_softmax] metal %s rel-L2 %.3e\n", mode, e); }
  }
}
