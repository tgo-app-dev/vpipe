// RelayVSR (generative-models/relayvsr/) against an fp32 restatement of the
// release's inference code.
//
// Goldens come from tools/dump_relayvsr_golden.py, which writes both
// networks and the collaborative stream as refvsr_inference/ defines them
// (read, not run), in fp32 on the CPU -- the truth -- and in bf16, the
// release's default dtype, which is the bar.
//
//   VPIPE_RELAYVSR_TEST_MODEL_PATH  the model directory (wan.safetensors
//       and flash.safetensors beside each other; kopper/RelayVSR).
//   VPIPE_RELAYVSR_GOLDEN           the dumper's --out ROOT: one directory
//       a case (each with its meta.json). Every case found is checked.
//
// window_attention_matches_reference needs neither: it holds the windowed
// attention kernel to a direct f64 evaluation of both window rules.

#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/vpipe-format.h"
#include "generative-models/relayvsr/metal-relayvsr-decoder.h"
#include "generative-models/relayvsr/metal-relayvsr-wan.h"
#include "generative-models/relayvsr/relayvsr-ops.h"
#include "generative-models/relayvsr/relayvsr-stream.h"
#include "generative-models/shared/ane-ffn.h"
#include "generative-models/weight-set.h"
#include "apple-silicon/tensor-beat.h"
#include "common/flex-bag.h"
#include "common/job.h"
#include "generative-models/generative-model-manager.h"
#include "interfaces/session-services-intf.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/relayvsr-upscale-stage.h"
#include "stages/model-memory.h"

#include <malloc/malloc.h>

#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
using genai::MetalRelayVsrDecoder;
using genai::MetalRelayVsrWan;
using genai::RelayVsrStream;

namespace fs = std::filesystem;

namespace {

const char*
env_(const char* k)
{
  const char* v = std::getenv(k);
  return v != nullptr && *v != '\0' ? v : nullptr;
}

template <typename T>
bool
read_bin_(const fs::path& p, std::size_t n, std::vector<T>* out)
{
  std::ifstream f(p, std::ios::binary);
  if (!f) { return false; }
  out->resize(n);
  f.read(reinterpret_cast<char*>(out->data()),
         (std::streamsize)(n * sizeof(T)));
  return (std::size_t)f.gcount() == n * sizeof(T);
}

FlexData
read_json_(const fs::path& p)
{
  std::ifstream f(p);
  if (!f) { return FlexData{}; }
  return FlexData::from_json(f);
}

int
int_(const FlexData& m, const char* k)
{
  auto o = m.as_object();
  auto it = o.find(k);
  return it == o.end() ? 0 : (int)(*it).second.as_int();
}

std::vector<double>
reals_(const FlexData& m, const char* k)
{
  std::vector<double> v;
  auto o = m.as_object();
  auto it = o.find(k);
  if (it == o.end()) { return v; }
  const FlexData a = (*it).second;
  for (std::size_t i = 0; i < a.as_array().size(); ++i) {
    v.push_back(a.as_array()[i].as_real());
  }
  return v;
}

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// FNV-1a over bytes: a run's outputs compared bit for bit across builds.
std::uint64_t
fnv_(const void* p, std::size_t n, std::uint64_t h = 1469598103934665603ull)
{
  const auto* b = (const std::uint8_t*)p;
  for (std::size_t i = 0; i < n; ++i) {
    h = (h ^ b[i]) * 1099511628211ull;
  }
  return h;
}

std::vector<fs::path>
golden_cases_(const char* root)
{
  std::vector<fs::path> out;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(root, ec)) {
    if (fs::exists(e.path() / "meta.json")) { out.push_back(e.path()); }
  }
  std::sort(out.begin(), out.end());
  return out;
}

// The release's windows over n frames: (frames, first emitted offset).
std::vector<std::pair<std::vector<int>, int>>
plan_(int n, int window)
{
  std::vector<std::pair<std::vector<int>, int>> out;
  int start = 0;
  bool first = true;
  while (true) {
    std::vector<int> idx;
    for (int i = start; i < std::min(start + window, n); ++i) {
      idx.push_back(i);
    }
    if (idx.size() == 1 && !first) { break; }
    out.push_back({idx, first ? 0 : 1});
    first = false;
    if ((int)idx.size() < window) { break; }
    start = idx.back();
  }
  return out;
}

struct Model {
  std::shared_ptr<genai::WeightSet> ws_wan, ws_flash;
  std::unique_ptr<MetalRelayVsrWan> wan;
  std::unique_ptr<MetalRelayVsrDecoder> dec;
};

// The acceleration a test loads both models with: an int8 tier (-1 none,
// 0 accurate .. 3 token_plain, as the stage passes it before
// effective_i8_tier) and the ANE feed-forward with its share (0 balanced).
// The tuners are off, so two runs of one arm are the same run.
struct Accel {
  int i8 = -1;
  bool ane = false;
  float ane_rows = 0.0f;
  int sage = 0;                  // 1 SageAttention's int8 QK, 2 + P*V
  bool wan = true, dec = true;   // which model takes it
  // The accurate route's split-K tuner (the stage's default; off for
  // runs that must repeat).
  bool autotune = false;
};

const char*
tier_name_(int t)
{
  return t == 0 ? "accurate" : t == 1 ? "fast" : t == 2 ? "token"
         : t == 3 ? "token_plain" : "none";
}

bool
load_(MetalCompute* mc, bool want_wan, Model* m, const Accel& a = {})
{
  const char* root = env_("VPIPE_RELAYVSR_TEST_MODEL_PATH");
  if (root == nullptr) { return false; }
  genai::RelayVsrFiles files;
  std::string err;
  if (!genai::resolve_relayvsr_files(root, &files, &err)) {
    std::printf("[relayvsr] %s\n", err.c_str());
    return false;
  }
  MetalRelayVsrDecoder::Config dc;
  if (!MetalRelayVsrDecoder::config_from_checkpoint(files.flash, &dc,
                                                    &err)) {
    std::printf("[relayvsr] %s\n", err.c_str());
    return false;
  }
  const int tier = genai::relayvsr::effective_i8_tier(
      a.i8, mc->supports_matrix_cores());
  const bool mx = mc->supports_matrix_cores();
  if (a.dec) {
    dc.i8_tier = tier;
    dc.ane_ffn = a.ane;
    dc.ane_rows = a.ane_rows;
    dc.sage_attn = mx && a.sage >= 1;
    dc.sage_pv8 = mx && a.sage >= 2;
  }
  dc.autotune = a.autotune;
  m->ws_flash = genai::WeightSet::open(files.flash, nullptr);
  m->dec = MetalRelayVsrDecoder::load(m->ws_flash, mc, dc, &err);
  if (!m->dec) {
    std::printf("[relayvsr] decoder: %s\n", err.c_str());
    return false;
  }
  if (want_wan) {
    MetalRelayVsrWan::Config wc;
    if (!MetalRelayVsrWan::config_from_checkpoint(files.wan, &wc, &err)) {
      std::printf("[relayvsr] %s\n", err.c_str());
      return false;
    }
    if (a.wan) {
      wc.i8_tier = tier;
      wc.ane_ffn = a.ane;
      wc.ane_rows = a.ane_rows;
      wc.sage_attn = mx && a.sage >= 1;
      wc.sage_pv8 = mx && a.sage >= 2;
    }
    wc.autotune = a.autotune;
    m->ws_wan = genai::WeightSet::open(files.wan, nullptr);
    m->wan = MetalRelayVsrWan::load(m->ws_wan, mc, wc, &err);
    if (!m->wan) {
      std::printf("[relayvsr] keyframe model: %s\n", err.c_str());
      return false;
    }
  }
  return true;
}

// A frame the stream gave out, copied.
struct OutFrame {
  std::int64_t index = 0;
  std::vector<std::uint8_t> px;
};

// Every frame of the stream's ready window, in order -- copied into `out`
// when given, counted in *n when given.
bool
drain_(RelayVsrStream& st, std::vector<OutFrame>* out, std::size_t* n,
       std::string* err)
{
  RelayVsrStream::Frame f;
  while (st.next(&f, err)) {
    if (out != nullptr) {
      const auto* p = (const std::uint8_t*)f.px;
      out->push_back({f.index, std::vector<std::uint8_t>(p, p + f.bytes)});
    }
    if (n != nullptr) { ++*n; }
  }
  return err->empty() && !st.ready();
}

// The padded channel-last upload of a u8 frame, through the stream's own
// kernel (the same Ops a stream loads).
SharedBuffer
upload_(MetalCompute* mc, const genai::relayvsr::Ops& op,
        const std::vector<std::uint8_t>& px, int h, int w)
{
  const int ph = h + (8 - h % 8) % 8, pw = w + (8 - w % 8) % 8;
  SharedBuffer src = mc->make_shared_buffer(px.size());
  std::memcpy(src.contents(), px.data(), px.size());
  SharedBuffer dst = mc->make_shared_buffer((std::size_t)ph * pw * 3 * 2);
  auto cs = mc->make_command_stream();
  ComputeEncoder enc = cs.begin_compute();
  enc.set_function(op.upload);
  enc.set_buffer(0, src); enc.set_buffer(1, dst);
  enc.set_constant(2, h); enc.set_constant(3, w);
  enc.set_constant(4, ph); enc.set_constant(5, pw);
  enc.set_constant(6, 0);
  enc.dispatch({(unsigned)pw, (unsigned)ph, 1}, {32, 8, 1});
  enc.end();
  cs.commit().wait();
  return dst;
}

}  // namespace

// THE WINDOWED ATTENTION KERNEL against a direct f64 evaluation: both window
// rules (shifted inward / centred, half-open and truncated) on a grid that
// is a multiple of neither tile side, three planes at different strides,
// GQA, head dims 64 and 128, f16 and bf16.
TEST(relayvsr, window_attention_matches_reference)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const int H = 13, W = 19, n = H * W;
  struct Arm { bool bf16; int hd, Hq, Hkv; };
  for (const Arm& arm : {Arm{false, 64, 8, 2}, Arm{true, 64, 8, 2},
                         Arm{false, 128, 4, 4}, Arm{true, 128, 4, 4}}) {
    genai::relayvsr::Ops op;
    std::string err;
    ASSERT_TRUE(op.load(mc, arm.bf16, &err));
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    auto rnd = [&](std::size_t cnt, float s) {
      std::vector<float> v(cnt);
      for (float& x : v) {
        x = genai::relayvsr::round_elt(s * u(rng), arm.bf16);
      }
      return v;
    };
    const int qd = arm.Hq * arm.hd, kd = arm.Hkv * arm.hd;
    // Strides wider than the rows: q and o have 4 spare columns a token,
    // plane 1's key 8.
    const int q_tok = qd + 4, o_tok = qd + 4;
    const int k_tok[3] = {kd, kd + 8, kd};
    const struct { int h, w; bool sh; } win[3] = {
        {6, 10, true}, {8, 8, false}, {4, 6, false}};
    const std::vector<float> q = rnd((std::size_t)n * q_tok, 2.0f);
    std::vector<std::vector<float>> k(3), v(3);
    std::vector<SharedBuffer> kb(3), vb(3);
    for (int p = 0; p < 3; ++p) {
      k[(std::size_t)p] = rnd((std::size_t)n * k_tok[p], 1.5f);
      v[(std::size_t)p] = rnd((std::size_t)n * kd, 1.0f);
      kb[(std::size_t)p] =
          genai::relayvsr::upload_elt(mc, k[(std::size_t)p], arm.bf16);
      vb[(std::size_t)p] =
          genai::relayvsr::upload_elt(mc, v[(std::size_t)p], arm.bf16);
    }
    SharedBuffer qb = genai::relayvsr::upload_elt(mc, q, arm.bf16);
    SharedBuffer ob = mc->make_shared_buffer((std::size_t)n * o_tok * 2);
    std::vector<genai::relayvsr::WindowPlane> planes(3);
    for (int p = 0; p < 3; ++p) {
      auto& pl = planes[(std::size_t)p];
      pl.k = &kb[(std::size_t)p]; pl.k_tok = k_tok[p];
      pl.v = &vb[(std::size_t)p]; pl.v_tok = kd;
      pl.win_h = win[p].h; pl.win_w = win[p].w; pl.shifted = win[p].sh;
    }
    // The f64 truth, once: [n, Hq, hd].
    std::vector<double> ref((std::size_t)n * arm.Hq * arm.hd);
    const double scale = 1.0 / std::sqrt((double)arm.hd);
    for (int qi = 0; qi < n; ++qi) {
      const int qy = qi / W, qx = qi % W;
      for (int h = 0; h < arm.Hq; ++h) {
        const int kvh = h / (arm.Hq / arm.Hkv);
        std::vector<double> s;
        std::vector<const float*> vv;
        for (int p = 0; p < 3; ++p) {
          int t, b, l, r;
          if (win[p].sh) {
            const int wh = std::min(win[p].h, H), ww = std::min(win[p].w, W);
            t = std::clamp(qy - wh / 2, 0, H - wh); b = t + wh;
            l = std::clamp(qx - ww / 2, 0, W - ww); r = l + ww;
          } else {
            t = qy - win[p].h / 2; b = qy + win[p].h / 2;
            l = qx - win[p].w / 2; r = qx + win[p].w / 2;
          }
          for (int ky = std::max(t, 0); ky < std::min(b, H); ++ky) {
            for (int kx = std::max(l, 0); kx < std::min(r, W); ++kx) {
              const int ki = ky * W + kx;
              const float* kr = k[(std::size_t)p].data() +
                                (std::size_t)ki * k_tok[p] + kvh * arm.hd;
              const float* qr = q.data() + (std::size_t)qi * q_tok +
                                h * arm.hd;
              double d = 0.0;
              for (int e = 0; e < arm.hd; ++e) {
                d += (double)qr[e] * (double)kr[e];
              }
              s.push_back(d * scale);
              vv.push_back(v[(std::size_t)p].data() + (std::size_t)ki * kd +
                           kvh * arm.hd);
            }
          }
        }
        double mx = -1e300, sum = 0.0;
        for (double x : s) { mx = std::max(mx, x); }
        for (double& x : s) { x = std::exp(x - mx); sum += x; }
        for (int e = 0; e < arm.hd; ++e) {
          double o = 0.0;
          for (std::size_t j = 0; j < s.size(); ++j) {
            o += s[j] * (double)vv[j][e];
          }
          o /= sum;
          ref[((std::size_t)qi * arm.Hq + h) * arm.hd + e] = o;
        }
      }
    }
    // Every route this box runs: the ALU kernel, and on the matrix cores
    // the NAX kernel in f16 / bf16, with SageAttention's int8 QK, and with
    // its int8 P*V too.
    const bool nax = op.nax_attention();
    // 1-3 the staged matrix-core kernel, 4-6 the direct one.
    const char* names[7] = {"alu", "nax", "nax int8 qk", "nax int8 qk+pv",
                            "nax direct", "nax direct int8 qk",
                            "nax direct int8 qk+pv"};
    for (int route = 0; route < (nax ? 7 : 1); ++route) {
      op.set_nax_attention(route > 0);
      op.set_window_staged(route <= 3);
      const int mode = route == 0 ? 0 : (route - 1) % 3;
      std::memset(ob.contents(), 0, (std::size_t)n * o_tok * 2);
      {
        auto cs = mc->make_command_stream();
        ComputeEncoder enc = cs.begin_compute();
        op.window_attention(enc, qb, 0, q_tok, planes, ob, 0, o_tok, H, W,
                            arm.Hq, arm.Hkv, arm.hd, mode);
        enc.end();
        cs.commit().wait();
      }
      const auto* og = (const std::uint16_t*)ob.contents();
      double num = 0.0, den = 0.0;
      for (int qi = 0; qi < n; ++qi) {
        for (int c = 0; c < arm.Hq * arm.hd; ++c) {
          const std::uint16_t g = og[(std::size_t)qi * o_tok + c];
          const double got = arm.bf16 ? genai::relayvsr::from_bf16(g)
                                      : genai::relayvsr::from_f16(g);
          const double o = ref[(std::size_t)qi * arm.Hq * arm.hd + c];
          num += (got - o) * (got - o);
          den += o * o;
        }
      }
      const double rel = std::sqrt(num / den);
      std::printf("[relayvsr] window attention %s hd %d (%d/%d heads), %s: "
                  "rel-L2 %.2e\n", arm.bf16 ? "bf16" : "f16", arm.hd,
                  arm.Hq, arm.Hkv, names[route], rel);
      // Every route's measured figure with room (ALU f16 2.1e-4, bf16
      // 1.7e-3; NAX f16 2.6e-4 -- 3.9e-4 with P left unrounded, which
      // this bar refuses; int8 QK 5.3-5.7e-3, + P.V 8.5-8.8e-3).
      const double bar = mode == 0 ? (arm.bf16 ? 2e-3 : 3.2e-4)
                       : mode == 1 ? 8e-3 : 1.2e-2;
      EXPECT_TRUE(rel < bar);
    }
    op.set_nax_attention(nax);
    op.set_window_staged(true);
  }
}

// THE DECODER, fed the reference's fp32 keyframe latents (teacher-forced),
// over every window of every golden case: each frame no further from the
// fp32 truth than the release's own bf16 decoder is.
TEST(relayvsr, decoder_matches_reference)
{
  const char* golden = env_("VPIPE_RELAYVSR_GOLDEN");
  if (golden == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Model m;
  if (!load_(mc, false, &m)) { return; }
  genai::relayvsr::Ops op16;
  std::string err;
  ASSERT_TRUE(op16.load(mc, false, &err));
  int frames = 0;
  for (const fs::path& c : golden_cases_(golden)) {
    const FlexData meta = read_json_(c / "meta.json");
    const int h = int_(meta, "height"), w = int_(meta, "width");
    const int ph = int_(meta, "pad_height"), pw = int_(meta, "pad_width");
    const int count = int_(meta, "count"), window = int_(meta, "window");
    const std::vector<double> bar = reals_(meta, "bar_out_tf");
    ASSERT_TRUE((int)bar.size() == count);
    std::map<int, MetalRelayVsrDecoder::Condition> cond;
    FlexData mm = meta;
    const FlexData kfs = mm.as_object().at("keyframes");
    for (const FlexData& kf : kfs.as_array()) {
      const int k = (int)kf.as_int();
      std::vector<float> lat;
      ASSERT_TRUE(read_bin_(c / fmt("latent_{}_f32.bin", k)(),
                            (std::size_t)48 * (ph / 4) * (pw / 4), &lat));
      SharedBuffer lb = genai::relayvsr::upload_f32(mc, lat);
      ASSERT_TRUE(m.dec->prepare(lb, ph / 4, pw / 4, &cond[k], &err));
    }
    std::vector<double> got((std::size_t)count, -1.0);
    SharedBuffer out;
    for (const auto& [idx, emit_from] : plan_(count, window)) {
      m.dec->begin_window();
      for (std::size_t off = 0; off < idx.size(); ++off) {
        const int i = idx[off];
        std::vector<std::uint8_t> px;
        ASSERT_TRUE(read_bin_(c / fmt("lr_{}.u8", i)(),
                              (std::size_t)3 * h * w, &px));
        SharedBuffer lr = upload_(mc, op16, px, h, w);
        ASSERT_TRUE(m.dec->decode(lr, ph, pw, cond[idx.front()],
                                  cond[idx.back()], 4 * h, 4 * w, true, &out,
                                  &err));
        if ((int)off < emit_from) { continue; }
        std::vector<float> ref;
        const std::size_t ne = (std::size_t)3 * 16 * h * w;
        ASSERT_TRUE(read_bin_(c / fmt("out_{}_f32.bin", i)(), ne, &ref));
        std::vector<float> ours(ne);
        const auto* o = (const float*)out.contents();
        for (std::size_t e = 0; e < ne; ++e) { ours[e] = 2.0f * o[e] - 1.0f; }
        got[(std::size_t)i] = rel_l2_(ours.data(), ref.data(), ne);
      }
    }
    double worst_ratio = 0.0, mean = 0.0, mean_bar = 0.0;
    for (int i = 0; i < count; ++i) {
      ASSERT_TRUE(got[(std::size_t)i] >= 0.0);
      worst_ratio = std::max(worst_ratio,
                             got[(std::size_t)i] / bar[(std::size_t)i]);
      mean += got[(std::size_t)i] / count;
      mean_bar += bar[(std::size_t)i] / count;
      ++frames;
      EXPECT_TRUE(got[(std::size_t)i] <= bar[(std::size_t)i]);
    }
    std::printf("[relayvsr] decoder %s: %d frames, rel-L2 mean %.4f "
                "(reference bf16 %.4f), worst ratio %.2f\n",
                c.filename().c_str(), count, mean, mean_bar, worst_ratio);
  }
  EXPECT_TRUE(frames > 0);
}

// THE KEYFRAME DiT over a golden's keyframes in order -- its ring filling,
// wrapping, and (past 8 keyframes) RoPE's origin moving -- from the noise
// the stream draws (asserted to be the reference's draw, bit for bit):
// each latent no further from fp32 than the reference's bf16.
TEST(relayvsr, keyframes_match_reference)
{
  const char* golden = env_("VPIPE_RELAYVSR_GOLDEN");
  if (golden == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Model m;
  if (!load_(mc, true, &m)) { return; }
  int checked = 0;
  for (const fs::path& c : golden_cases_(golden)) {
    const FlexData meta = read_json_(c / "meta.json");
    const int h = int_(meta, "height"), w = int_(meta, "width");
    const int ph = int_(meta, "pad_height"), pw = int_(meta, "pad_width");
    const std::vector<double> bar = reals_(meta, "bar_latent");
    RelayVsrStream::Options o;
    o.window = int_(meta, "window");
    o.seed = (std::uint64_t)int_(meta, "seed");
    RelayVsrStream st(m.wan.get(), m.dec.get(), mc, o);
    std::vector<double> got;
    bool noise_exact = true;
    std::uint64_t digest = 1469598103934665603ull;
    m.wan->keep_debug(true);
    std::vector<float> lq0, blk0;
    st.on_keyframe = [&](std::int64_t f, const std::vector<float>& noise,
                         const SharedBuffer& lat) {
      const std::size_t ne = (std::size_t)48 * (ph / 4) * (pw / 4);
      std::vector<float> rn, rl;
      if (!read_bin_(c / fmt("noise_{}.f32", f)(), ne, &rn) ||
          !read_bin_(c / fmt("latent_{}_f32.bin", f)(), ne, &rl)) {
        got.push_back(1e9);
        return;
      }
      noise_exact = noise_exact &&
                    std::memcmp(rn.data(), noise.data(), ne * 4) == 0;
      got.push_back(rel_l2_((const float*)lat.contents(), rl.data(), ne));
      digest = fnv_(lat.contents(), ne * 4, digest);
      if (f == 0) {
        lq0 = m.wan->debug_lq();
        blk0 = m.wan->debug_block0();
      }
    };
    std::string err;
    const int count = int_(meta, "count");
    for (int i = 0; i < count; ++i) {
      std::vector<std::uint8_t> px;
      ASSERT_TRUE(read_bin_(c / fmt("lr_{}.u8", i)(), (std::size_t)3 * h * w,
                            &px));
      ASSERT_TRUE(st.push(px.data(), RelayVsrStream::In::kU8, h, w, &err));
      ASSERT_TRUE(drain_(st, nullptr, nullptr, &err));
    }
    ASSERT_TRUE(st.finish(&err));
    ASSERT_TRUE(drain_(st, nullptr, nullptr, &err));
    EXPECT_TRUE(noise_exact);
    ASSERT_TRUE(got.size() == bar.size());
    {
      const std::size_t n = (std::size_t)(ph / 8) * (pw / 8) * 3072;
      std::vector<float> r;
      if (read_bin_(c / "lq_0_f32.bin", n, &r) && lq0.size() == n) {
        std::printf("[relayvsr]   keyframe 0 source rows %.4f, after block "
                    "0 ", rel_l2_(lq0.data(), r.data(), n));
        if (read_bin_(c / "blk0_0_f32.bin", n, &r) && blk0.size() == n) {
          std::printf("%.4f\n", rel_l2_(blk0.data(), r.data(), n));
        } else {
          std::printf("n/a\n");
        }
      }
    }
    std::string line;
    for (std::size_t k = 0; k < got.size(); ++k) {
      line += fmt(" {:.4f}/{:.4f}", got[k], bar[k])();
      EXPECT_TRUE(got[k] <= 1.1 * bar[k]);
      ++checked;
    }
    std::printf("[relayvsr] keyframes %s (ours/reference bf16):%s; latents "
                "digest %016llx\n", c.filename().c_str(), line.c_str(),
                (unsigned long long)digest);
  }
  EXPECT_TRUE(checked > 0);
}

// THE WHOLE STREAM, as the stage runs it: every frame out, in order, once,
// each no further from the fp32 stream than the reference's bf16 stream is.
TEST(relayvsr, stream_matches_reference)
{
  const char* golden = env_("VPIPE_RELAYVSR_GOLDEN");
  if (golden == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Model m;
  if (!load_(mc, true, &m)) { return; }
  for (const fs::path& c : golden_cases_(golden)) {
    const FlexData meta = read_json_(c / "meta.json");
    const int h = int_(meta, "height"), w = int_(meta, "width");
    const int count = int_(meta, "count");
    const std::vector<double> bar = reals_(meta, "bar_out");
    RelayVsrStream::Options o;
    o.window = int_(meta, "window");
    o.seed = (std::uint64_t)int_(meta, "seed");
    o.f32_out = true;
    RelayVsrStream st(m.wan.get(), m.dec.get(), mc, o);
    int keys = 0;
    st.on_keyframe = [&](std::int64_t, const std::vector<float>&,
                         const SharedBuffer&) { ++keys; };
    std::string err;
    std::vector<OutFrame> out;
    for (int i = 0; i < count; ++i) {
      std::vector<std::uint8_t> px;
      ASSERT_TRUE(read_bin_(c / fmt("lr_{}.u8", i)(), (std::size_t)3 * h * w,
                            &px));
      ASSERT_TRUE(st.push(px.data(), RelayVsrStream::In::kU8, h, w, &err));
      ASSERT_TRUE(drain_(st, &out, nullptr, &err));
    }
    ASSERT_TRUE(st.finish(&err));
    ASSERT_TRUE(drain_(st, &out, nullptr, &err));
    ASSERT_TRUE((int)out.size() == count);
    double mean = 0.0, mean_bar = 0.0, worst = 0.0;
    std::uint64_t digest = 1469598103934665603ull;
    for (int i = 0; i < count && i < (int)out.size(); ++i) {
      EXPECT_TRUE(out[(std::size_t)i].index == i);
      digest = fnv_(out[(std::size_t)i].px.data(),
                    out[(std::size_t)i].px.size(), digest);
      const std::size_t ne = (std::size_t)3 * 16 * h * w;
      std::vector<float> ref;
      ASSERT_TRUE(read_bin_(c / fmt("out_{}_f32.bin", i)(), ne, &ref));
      const auto* p = (const float*)out[(std::size_t)i].px.data();
      std::vector<float> ours(ne);
      for (std::size_t e = 0; e < ne; ++e) { ours[e] = 2.0f * p[e] - 1.0f; }
      const double r = rel_l2_(ours.data(), ref.data(), ne);
      mean += r / count;
      mean_bar += bar[(std::size_t)i] / count;
      worst = std::max(worst, r / bar[(std::size_t)i]);
      EXPECT_TRUE(r <= 1.1 * bar[(std::size_t)i]);
    }
    std::printf("[relayvsr] stream %s: %d frames, %d keyframes, rel-L2 mean "
                "%.4f (reference bf16 %.4f), worst ratio %.2f; frames "
                "digest %016llx\n", c.filename().c_str(), count, keys, mean,
                mean_bar, worst, (unsigned long long)digest);
  }
}

// VPIPE_RELAYVSR_BENCH=WxH (the INPUT size): one keyframe and one decoded
// frame, timed apart, three rounds each after a warm-up.
TEST(relayvsr, bench)
{
  const char* spec = env_("VPIPE_RELAYVSR_BENCH");
  if (spec == nullptr) { return; }
  int w = 0, h = 0;
  if (std::sscanf(spec, "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  // VPIPE_RELAYVSR_BENCH_TIER = accurate | fast | token | token_plain,
  // VPIPE_RELAYVSR_BENCH_ANE = 1 (+ _ANE_ROWS, a share): the arm.
  Accel acc;
  if (const char* t = env_("VPIPE_RELAYVSR_BENCH_TIER")) {
    const std::string ts(t);
    acc.i8 = ts == "fast" ? 1 : ts == "token" ? 2 : ts == "token_plain" ? 3
             : ts == "accurate" ? 0 : -1;
  }
  acc.ane = env_("VPIPE_RELAYVSR_BENCH_ANE") != nullptr;
  // VPIPE_RELAYVSR_BENCH_SAGE = 1 (int8 QK) | 2 (+ int8 P*V).
  if (const char* sg = env_("VPIPE_RELAYVSR_BENCH_SAGE")) {
    acc.sage = std::atoi(sg);
  }
  if (const char* r = env_("VPIPE_RELAYVSR_BENCH_ANE_ROWS")) {
    acc.ane_rows = (float)std::atof(r);
  }
  Model m;
  if (!load_(mc, true, &m, acc)) { return; }
  genai::relayvsr::Ops op16, op16b;
  std::string err;
  ASSERT_TRUE(op16.load(mc, false, &err));
  ASSERT_TRUE(op16b.load(mc, true, &err));
  const int ph = h + (8 - h % 8) % 8, pw = w + (8 - w % 8) % 8;
  std::mt19937 rng(1);
  std::vector<std::uint8_t> px((std::size_t)3 * h * w);
  for (auto& b : px) { b = (std::uint8_t)(rng() & 0xff); }
  SharedBuffer lr16 = upload_(mc, op16, px, h, w);
  SharedBuffer lrbf = upload_(mc, op16b, px, h, w);
  std::normal_distribution<float> nd;
  std::vector<float> noise((std::size_t)48 * (ph / 4) * (pw / 4));
  for (float& v : noise) { v = nd(rng); }
  SharedBuffer lat, out;
  MetalRelayVsrDecoder::Condition c0, c1;
  auto ms = [](auto t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
  };
  double tk = 1e30, tp = 1e30, td = 1e30, td2 = 1e30;
  const int rounds =
      env_("VPIPE_RELAYVSR_BENCH_ROUNDS") ? std::atoi(env_(
                                                "VPIPE_RELAYVSR_BENCH_ROUNDS"))
                                          : 4;
  for (int r = 0; r < rounds; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(m.wan->keyframe(lrbf, ph, pw, noise, &lat, &err));
    if (r > 0) { tk = std::min(tk, ms(t0)); }
    t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(m.dec->prepare(lat, ph / 4, pw / 4, &c0, &err));
    if (r > 0) { tp = std::min(tp, ms(t0)); }
    ASSERT_TRUE(m.dec->prepare(lat, ph / 4, pw / 4, &c1, &err));
    m.dec->begin_window();
    t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(m.dec->decode(lr16, ph, pw, c0, c1, 4 * h, 4 * w, false,
                              &out, &err));
    if (r > 0) { td = std::min(td, ms(t0)); }
    t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(m.dec->decode(lr16, ph, pw, c0, c1, 4 * h, 4 * w, false,
                              &out, &err));
    if (r > 0) { td2 = std::min(td2, ms(t0)); }
  }
  std::printf("[relayvsr] bench %dx%d -> %dx%d (%s, int8 %s, ANE %s, "
              "sage %d, window attention on %s, cross on %s): "
              "keyframe %.0f ms, prime %.0f ms, frame %.0f ms (first of a "
              "window) / %.0f ms (with history); a 16-frame window ~%.3f s "
              "a frame; int8 GEMMs %ld / %ld, ANE rows %ld / %ld\n",
              w, h, 4 * w, 4 * h,
              m.wan->uses_matrix_cores() ? "matrix cores" : "steel",
              tier_name_(m.wan->config().i8_tier), acc.ane ? "on" : "off",
              m.wan->config().sage_mode(),
              m.wan->window_on_matrix_cores() ? "matrix cores" : "alu",
              m.wan->cross_on_matrix_cores() ? "matrix cores" : "steel", tk,
              tp,
              td, td2, (tk + tp + td + 14 * td2) / 15.0 / 1000.0,
              m.wan->i8_gemms(), m.dec->i8_gemms(), m.wan->ane_rows(),
              m.dec->ane_rows());
}

namespace {

template <typename Fn>
std::size_t
device_peak_of_(MetalCompute* mc, Fn&& fn)
{
  const std::size_t base = mc->memory_budget_settled().allocated;
  std::atomic<std::size_t> peak{base};
  std::atomic<bool> done{false};
  std::thread sampler([&] {
    while (!done.load()) {
      const std::size_t a = mc->memory_budget().allocated;
      if (a > peak.load()) { peak.store(a); }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });
  fn();
  done.store(true);
  sampler.join();
  const std::size_t top =
      std::max(peak.load(), mc->memory_budget().allocated);
  return top > base ? top - base : 0;
}

}  // namespace

// THE PLAN BOOKS WHAT A RUN HOLDS: the device's peak allocation over a load
// and a stream long enough to fill and wrap the keyframe ring (window 4,
// 14 frames: 5 keyframes) against the stage's booking -- both models'
// weights as held, and RelayVsrStream::run_bytes.
TEST(relayvsr, allocates_what_it_books)
{
  if (env_("VPIPE_RELAYVSR_TEST_MODEL_PATH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  // Every acceleration arm the box runs: the int8 tiers that move the
  // booking (accurate's per-call scratch, token's held layout) where there
  // are matrix cores, and the ANE pinned at half -- its modules booked as
  // CoreML claims, whose host rows are Metal buffers.
  std::vector<Accel> arms(1);
  if (mc->supports_matrix_cores()) {
    arms.push_back(Accel{});
    arms.back().i8 = 0;
    arms.push_back(Accel{});
    arms.back().i8 = 2;
    // ...and token with its tuner, the stage's default: the decoder's
    // down, 1408 deep, is no layout's and runs the accurate route there --
    // its per-call weight, the tuner's rows and its split planes.
    arms.push_back(Accel{});
    arms.back().i8 = 2;
    arms.back().autotune = true;
  }
  arms.push_back(Accel{});
  arms.back().ane = true;
  arms.back().ane_rows = 0.5f;
  // ...and SageAttention's int8 QK and P*V: the windows' prologue, and
  // the cross-attention's (which by default takes it only from 4096 rows,
  // past both sizes here: lowered for this arm, so it is booked and held).
  if (mc->supports_matrix_cores()) {
    arms.push_back(Accel{});
    arms.back().sage = 2;
  }
  for (const Accel& arm : arms)
  for (const auto& [h, w] : {std::pair<int, int>{136, 248},
                             std::pair<int, int>{270, 480}}) {
    if (arm.sage > 0) {
      setenv("VPIPE_RELAYVSR_XSAGE_MIN_ROWS", "512", 1);
    } else {
      unsetenv("VPIPE_RELAYVSR_XSAGE_MIN_ROWS");
    }
    std::mt19937 rng(3);
    std::vector<std::uint8_t> px((std::size_t)3 * h * w);
    Model m;
    MetalRelayVsrWan::Config wc;
    MetalRelayVsrDecoder::Config dc;
    std::size_t n_out = 0;
    const std::size_t peak = device_peak_of_(mc, [&] {
      ASSERT_TRUE(load_(mc, true, &m, arm));
      if (!m.wan) { return; }
      wc = m.wan->config();
      dc = m.dec->config();
      RelayVsrStream::Options o;
      o.window = 4;
      RelayVsrStream st(m.wan.get(), m.dec.get(), mc, o);
      std::string err;
      for (int i = 0; i < 14; ++i) {
        for (auto& b : px) { b = (std::uint8_t)(rng() & 0xff); }
        ASSERT_TRUE(st.push(px.data(), RelayVsrStream::In::kU8, h, w, &err));
        if (st.ready()) {
          // A window not yet drained refuses the next frame, by name.
          std::string e2;
          EXPECT_FALSE(st.push(px.data(), RelayVsrStream::In::kU8, h, w,
                               &e2));
          EXPECT_FALSE(e2.empty());
        }
        ASSERT_TRUE(drain_(st, nullptr, &n_out, &err));
      }
      ASSERT_TRUE(st.finish(&err));
      ASSERT_TRUE(drain_(st, nullptr, &n_out, &err));
    });
    EXPECT_TRUE(n_out == 14);
    unsetenv("VPIPE_RELAYVSR_XSAGE_MIN_ROWS");
    if (!m.wan) { continue; }
    std::size_t wb = MetalRelayVsrWan::weight_bytes(wc) +
                     MetalRelayVsrDecoder::weight_bytes(dc);
    // The loads hold what the plan books (the stage revises to held_bytes,
    // which is this wherever the context ran the tier the rule names).
    EXPECT_TRUE(m.wan->held_bytes() == MetalRelayVsrWan::weight_bytes(wc));
    EXPECT_TRUE(m.dec->held_bytes() ==
                MetalRelayVsrDecoder::weight_bytes(dc));
    const std::size_t rb = RelayVsrStream::run_bytes(wc, dc, h, w, 4);
    // An armed module's whole booking, at the chunk it was built at: its
    // host rows are the Metal part.
    const std::size_t ab = m.wan->ane_held_bytes() + m.dec->ane_held_bytes();
    EXPECT_TRUE(ab <= MetalRelayVsrWan::ane_runtime_bytes(wc) +
                          MetalRelayVsrDecoder::ane_runtime_bytes(dc));
    wb += ab;
    // The windows' int8 prologue: each buffer made ONCE (sized at the
    // model's planes from the first call -- grown as the ring filled, the
    // keyframe's made K and V four times), and within its booking.
    for (const auto* op : {&m.wan->ops(), &m.dec->ops()}) {
      const long want = arm.sage == 0 ? 0 : arm.sage == 1 ? 7 : 9;
      EXPECT_TRUE(op->window_int8_allocs() == want);
    }
    const std::size_t w8 = m.wan->ops().window_int8_held() +
                           m.dec->ops().window_int8_held();
    const std::size_t w8_booked =
        genai::relayvsr::Ops::window_int8_bytes(
            wc.context_frames, wc.heads, wc.heads,
            ((h + 7) / 8) * ((w + 7) / 8), wc.head_dim, wc.sage_mode()) +
        genai::relayvsr::Ops::window_int8_bytes(
            4, dc.heads, dc.kv_heads, ((h + 7) / 8 * 2) * ((w + 7) / 8 * 2),
            dc.head_dim, dc.sage_mode());
    EXPECT_TRUE(w8 <= w8_booked + (18u << 14));   // a page a buffer
    std::printf("[relayvsr] %dx%d, int8 %s%s, ANE %s, sage %d: device "
                "+%.1f MB, booked %.1f MB (weights %.1f, run %.1f, ANE "
                "%.1f; int8 attention %.1f of %.1f)\n",
                w, h, tier_name_(wc.i8_tier),
                arm.autotune ? " (tuned)" : "", arm.ane ? "on" : "off",
                arm.sage,
                peak / 1048576.0, (wb + rb) / 1048576.0,
                (wb - ab) / 1048576.0, rb / 1048576.0, ab / 1048576.0,
                w8 / 1048576.0, w8_booked / 1048576.0);
    // BOOKED AT LEAST WHAT THE DEVICE HELD (it had allowed 32 MB short,
    // which hid the tuned decoder's int8 scratch).
    EXPECT_TRUE(peak <= wb + rb);
    // The ANE arm books CoreML's own copy too, which no Metal figure sees.
    EXPECT_TRUE(peak + (256u << 20) + ab >= wb + rb);
  }
}

namespace {

// The host heap in use (every malloc zone). Metal's buffers -- weights,
// working sets, the stream's frames -- are not in it.
std::size_t
heap_in_use_()
{
  malloc_statistics_t st{};
  malloc_zone_statistics(nullptr, &st);
  return st.size_in_use;
}

// The heap's peak over its level when `fn` starts, sampled from a thread.
template <typename Fn>
std::size_t
heap_peak_of_(Fn&& fn)
{
  const std::size_t base = heap_in_use_();
  std::atomic<std::size_t> peak{base};
  std::atomic<bool> done{false};
  std::thread sampler([&] {
    while (!done.load()) {
      const std::size_t a = heap_in_use_();
      if (a > peak.load()) { peak.store(a); }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });
  fn();
  done.store(true);
  sampler.join();
  return peak.load() > base ? peak.load() - base : 0;
}

}  // namespace

// THE LOAD WIDENS NOTHING LARGE ON THE HOST. macOS keeps a freed large
// heap block in the process's footprint, so a loader that widens a
// tensor into a host f32 vector leaves it there for the run: the
// keyframe model's timestep projection is 226 MB as f32 and its 3x3 conv
// 85 MB, twice for the transpose. Both are now read where the file's
// bytes land (a Metal buffer) and converted element by element into the
// buffer that keeps them. The heap's peak over a load of both models.
TEST(relayvsr, loading_keeps_no_large_host_copy)
{
  if (env_("VPIPE_RELAYVSR_TEST_MODEL_PATH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Model m;
  bool ok = false;
  const std::size_t peak = heap_peak_of_([&] { ok = load_(mc, true, &m); });
  ASSERT_TRUE(ok);
  std::printf("[relayvsr] load: host heap peak +%.1f MB\n",
              peak / 1048576.0);
  EXPECT_TRUE(peak < (32u << 20));
}

namespace {

std::atomic<bool> g_rv_arm{false}, g_rv_armed{false};

// `count` u8 frames [3, h, w], each tagged with its index in the sideband.
class RvSource : public TypedStage<RvSource> {
public:
  static constexpr const char* kTypeName = "ut-relayvsr-source";
  using TypedStage::TypedStage;
  int h = 0, w = 0, count = 0, n = 0;

  void reset_run_state() override { n = 0; }

  Job
  process(RuntimeContext& ctx) override
  {
    if (n >= count) { ctx.signal_done(); co_return; }
    if (n == 0) {
      // The models are loaded (initialize ran): the heap's level now is
      // the baseline.
      g_rv_arm.store(true);
      while (!g_rv_armed.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    auto b = std::make_unique<TensorBeatPayload>();
    b->dtype = TensorBeat::DType::U8;
    b->shape = {3, h, w};
    const std::size_t px = (std::size_t)3 * h * w;
    b->resize_contiguous(px);
    std::uint8_t* d = b->as_u8();
    for (std::size_t i = 0; i < px; ++i) {
      d[i] = (std::uint8_t)((i * 2654435761u >> 13) + i / 509 + 7 * n);
    }
    FlexData sb = FlexData::make_object();
    bag::set_integer(sb, "i", n);
    b->sideband = std::move(sb);
    ++n;
    co_await ctx.write(0, std::move(b));
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "frames", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

// Counts the 4x frames, checks their order and shape, drops them, and
// reads the stage's arena as booked when the first arrives.
class RvSink : public TypedStage<RvSink> {
public:
  static constexpr const char* kTypeName = "ut-relayvsr-sink";
  using TypedStage::TypedStage;
  int h = 0, w = 0, frames = 0;
  bool ordered = true, shaped = true;
  std::size_t booked = 0;
  // At the second frame (the stage has settled its ANE bookings after the
  // first): the two checkpoints as planned and the ANE modules' bookings.
  genai::RelayVsrFiles files;
  std::size_t weights = 0, ane_k = 0, ane_d = 0;

  void
  reset_run_state() override
  {
    frames = 0;
    ordered = shaped = true;
    booked = weights = ane_k = ane_d = 0;
  }

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get());
    shaped = shaped && tb != nullptr && tb->dtype == TensorBeat::DType::F32 &&
             tb->shape.size() == 3 && tb->shape[0] == 3 &&
             tb->shape[1] == 4 * h && tb->shape[2] == 4 * w;
    ordered = ordered && tb != nullptr &&
              bag::integer(tb->sideband, "i", -1) == frames;
    if (frames == 0) {
      if (auto* mgr = session()->services()->generative_model_manager()) {
        booked = mgr->declared_scratch("relayvsr-upscale:rv");
      }
    }
    if (frames == 1 && !files.wan.empty()) {
      if (auto* mgr = session()->services()->generative_model_manager()) {
        weights = mgr->planned_weight_bytes(files.wan) +
                  mgr->planned_weight_bytes(files.flash);
        ane_k = mgr->declared_scratch("coreml:ane-ffn/rv/keyframe");
        ane_d = mgr->declared_scratch("coreml:ane-ffn/rv/decoder");
      }
    }
    ++frames;
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "frames", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

// The arena a stage's launch plan books: its scratch claim's bytes.
std::size_t
planned_arena_(const RelayVsrUpscaleStage& st)
{
  for (const ResourceClaim& c : st.declare_resources()) {
    if (c.key.rfind("relayvsr-upscale:", 0) != 0) { continue; }
    const auto bar = c.key.rfind('|');
    if (bar != std::string::npos) {
      return (std::size_t)std::stoull(c.key.substr(bar + 1));
    }
  }
  return 0;
}

}  // namespace

// THE STAGE, LAUNCHED TWICE: frames out in order and at 4x, a complete
// window handed on ONE FRAME AT A TIME (the host's peak over the stream is
// a frame or two, where holding the window -- twice, as it once did --
// was 32 frames: 354 MB at this size in f32); the arena booked at the
// run's frame size once it starts, each launch afresh; everything booked
// given back at the end of the stream; and every launch planned at the
// stage's own default size, never the last run's.
TEST(relayvsr, the_stage_hands_on_a_frame_at_a_time_and_gives_back)
{
  const char* root = env_("VPIPE_RELAYVSR_TEST_MODEL_PATH");
  if (root == nullptr) { return; }
  genai::RelayVsrFiles files;
  std::string err;
  ASSERT_TRUE(genai::resolve_relayvsr_files(root, &files, &err));
  MetalRelayVsrWan::Config wc;
  MetalRelayVsrDecoder::Config dc;
  ASSERT_TRUE(MetalRelayVsrWan::config_from_checkpoint(files.wan, &wc, &err));
  ASSERT_TRUE(
      MetalRelayVsrDecoder::config_from_checkpoint(files.flash, &dc, &err));
  constexpr int H = 180, W = 320, N = 20;
  const std::size_t frame = (std::size_t)3 * 16 * H * W * 4;   // f32 out
  const std::size_t run = RelayVsrStream::run_bytes(wc, dc, H, W, 16);
  const std::size_t plan = RelayVsrStream::run_bytes(wc, dc, 544, 960, 16);
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  auto* mgr = sess.services()->generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src_u = std::make_unique<RvSource>(&sess, "src", std::vector<InEdge>{},
                                          FlexData::make_object());
  src_u->h = H;
  src_u->w = W;
  src_u->count = N;
  src_u->allocate_oports(1);
  auto* src = pl->insert_stage(std::move(src_u));
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(root));
  cfg.as_object().insert_or_assign("output_dtype",
                                   FlexData::make_string("f32"));
  auto* rv = static_cast<RelayVsrUpscaleStage*>(pl->insert_stage(
      std::make_unique<RelayVsrUpscaleStage>(
          &sess, "rv", std::vector<InEdge>{{src, 0}}, cfg)));
  auto sink_u = std::make_unique<RvSink>(&sess, "sink",
                                         std::vector<InEdge>{{rv, 0}},
                                         FlexData::make_object());
  sink_u->h = H;
  sink_u->w = W;
  auto* sink = static_cast<RvSink*>(pl->insert_stage(std::move(sink_u)));
  EXPECT_TRUE(planned_arena_(*rv) == plan);
  {
    // Its output size given: the launch planned at the run's frame.
    FlexData sized = FlexData::make_object();
    sized.as_object().insert_or_assign("hf_dir", FlexData::make_string(root));
    sized.as_object().insert_or_assign("output_width",
                                       FlexData::make_int(4 * W));
    sized.as_object().insert_or_assign("output_height",
                                       FlexData::make_int(4 * H));
    RelayVsrUpscaleStage st(&sess, "sized", std::vector<InEdge>{},
                            std::move(sized));
    EXPECT_TRUE(st.config_error().empty());
    EXPECT_TRUE(planned_arena_(st) == run);
    EXPECT_TRUE(run < plan);
  }
  for (int launch = 0; launch < 2; ++launch) {
    g_rv_arm.store(false);
    g_rv_armed.store(false);
    std::atomic<std::size_t> base{0}, peak{0};
    std::atomic<bool> going{true};
    std::thread sampler([&] {
      while (going.load()) {
        if (g_rv_arm.load() && !g_rv_armed.load()) {
          base.store(heap_in_use_());
          peak.store(base.load());
          g_rv_armed.store(true);
        }
        const std::size_t now = heap_in_use_();
        if (g_rv_armed.load() && now > peak.load()) { peak.store(now); }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });
    bool ok = false;
    {
      PipelineRuntime rt(pl.get(), &sess);
      ok = rt.launch();
      if (ok) { rt.wait_idle(); }
      going.store(false);
      sampler.join();
      if (ok) { rt.stop(); }
    }
    ASSERT_TRUE(ok);
    const std::size_t grew = peak.load() > base.load()
                                 ? peak.load() - base.load()
                                 : 0;
    const std::size_t held_w = mgr->planned_weight_bytes(files.wan) +
                               mgr->planned_weight_bytes(files.flash);
    const std::size_t held_s = mgr->declared_scratch("relayvsr-upscale:rv");
    std::printf("[relayvsr] stage launch %d: %d frames (in order %d, 4x "
                "f32 %d); host heap peak +%.1f MB (%.1f frames of %.1f "
                "MB); arena booked %.1f MB at the first frame (run %.1f); "
                "after the stream weights %.1f MB, arena %.1f MB; the next "
                "launch planned at %.1f MB (default size %.1f)\n",
                launch, sink->frames, (int)sink->ordered, (int)sink->shaped,
                grew / 1048576.0, (double)grew / (double)frame,
                frame / 1048576.0, sink->booked / 1048576.0,
                run / 1048576.0, held_w / 1048576.0, held_s / 1048576.0,
                planned_arena_(*rv) / 1048576.0, plan / 1048576.0);
    EXPECT_TRUE(sink->frames == N);
    EXPECT_TRUE(sink->ordered && sink->shaped);
    EXPECT_TRUE(rv->emitted() == (std::uint64_t)N);
    EXPECT_TRUE(grew < 4 * frame);
    EXPECT_TRUE(sink->booked == run);
    EXPECT_TRUE(held_w == 0 && held_s == 0);
    EXPECT_TRUE(planned_arena_(*rv) == plan);
  }
}

// THE STAGE REFUSES A RUN THAT CANNOT FIT, before it allocates, and gives
// back what the plan booked for it. On a 16 GB box as believed
// (VPIPE_RAM_LIMIT_MB 16384), whatever this machine is:
//   1. an output size whose run alone is over the GPU's working set --
//      3840 x 2160, 21.8 GB with the weights -- is refused AT THE LAUNCH,
//      before the weights are read;
//   2. that decision does not move with what is allocated then: a small
//      output size is admitted beside a buffer holding most of the
//      working set (a peer that loaded first), where reading the room
//      left would have refused it;
//   3. no size given, 960 x 540 frames are refused AT THE FIRST FRAME
//      (13.4 GB of ring beside 8.4 GB of loaded weights).
// Each refusal: no frames out, the pipeline ends, a "memory" report (step
// "upscale", `at` the moment, the input frame, its parts summing to
// need, the gates), and every booking withdrawn.
TEST(relayvsr, the_stage_refuses_a_run_that_cannot_fit)
{
  const char* root = env_("VPIPE_RELAYVSR_TEST_MODEL_PATH");
  if (root == nullptr) { return; }
  genai::RelayVsrFiles files;
  std::string err;
  ASSERT_TRUE(genai::resolve_relayvsr_files(root, &files, &err));
  MetalRelayVsrWan::Config wc;
  MetalRelayVsrDecoder::Config dc;
  ASSERT_TRUE(MetalRelayVsrWan::config_from_checkpoint(files.wan, &wc, &err));
  ASSERT_TRUE(
      MetalRelayVsrDecoder::config_from_checkpoint(files.flash, &dc, &err));
  const std::uint64_t weights = MetalRelayVsrWan::weight_bytes(wc) +
                                MetalRelayVsrDecoder::weight_bytes(dc);
  const std::uint64_t big = RelayVsrStream::run_bytes(wc, dc, 540, 960, 16);
  const std::uint64_t small = RelayVsrStream::run_bytes(wc, dc, 88, 160, 16);
  // The believed box, as it was on the way out.
  struct RamLimit {
    std::string old;
    bool had = false;
    RamLimit()
    {
      if (const char* e = std::getenv("VPIPE_RAM_LIMIT_MB")) {
        old = e;
        had = true;
      }
      setenv("VPIPE_RAM_LIMIT_MB", "16384", 1);
    }
    ~RamLimit()
    {
      if (had) {
        setenv("VPIPE_RAM_LIMIT_MB", old.c_str(), 1);
      } else {
        unsetenv("VPIPE_RAM_LIMIT_MB");
      }
    }
  } limit;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto* mgr = sess.services()->generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  auto stage_cfg = [&](int out_w, int out_h) {
    FlexData c = FlexData::make_object();
    c.as_object().insert_or_assign("hf_dir", FlexData::make_string(root));
    if (out_w > 0) {
      c.as_object().insert_or_assign("output_width",
                                     FlexData::make_int(out_w));
      c.as_object().insert_or_assign("output_height",
                                     FlexData::make_int(out_h));
    }
    return c;
  };
  struct Out {
    bool ok = false;
    int frames = 0, loads = 0;
    std::size_t ws = 0, weights_after = 0, arena_after = 0;
    FlexData report;                // rv's "memory" report, if any
  };
  // src (count frames of h x w) -> rv -> sink, one launch.
  auto run = [&](int out_w, int out_h, int h, int w, int count) {
    Out r;
    auto pl = std::make_unique<Pipeline>("p", &sess);
    auto src_u = std::make_unique<RvSource>(
        &sess, "src", std::vector<InEdge>{}, FlexData::make_object());
    src_u->h = h;
    src_u->w = w;
    src_u->count = count;
    src_u->allocate_oports(1);
    auto* src = pl->insert_stage(std::move(src_u));
    auto* rv = static_cast<RelayVsrUpscaleStage*>(pl->insert_stage(
        std::make_unique<RelayVsrUpscaleStage>(
            &sess, "rv", std::vector<InEdge>{{src, 0}},
            stage_cfg(out_w, out_h))));
    auto sink_u = std::make_unique<RvSink>(
        &sess, "sink", std::vector<InEdge>{{rv, 0}}, FlexData::make_object());
    sink_u->h = h;
    sink_u->w = w;
    auto* sink = static_cast<RvSink*>(pl->insert_stage(std::move(sink_u)));
    const std::uint64_t v0 = sess.reports_version();
    g_rv_arm.store(false);
    g_rv_armed.store(true);         // nothing samples the heap here
    {
      PipelineRuntime rt(pl.get(), &sess);
      r.ok = rt.launch();
      if (r.ok) {
        rt.wait_idle();
        rt.stop();
      }
    }
    g_rv_armed.store(false);
    r.frames = sink->frames;
    r.loads = rv->loads();
    r.ws = rv->gpu_working_set();
    r.weights_after = mgr->planned_weight_bytes(files.wan) +
                      mgr->planned_weight_bytes(files.flash);
    r.arena_after = mgr->declared_scratch("relayvsr-upscale:rv");
    const FlexData doc = sess.reports();
    const FlexData items = doc.as_object().at("items");
    for (std::size_t i = 0; i < items.as_array().size(); ++i) {
      const FlexData it = items.as_array()[i];
      if (it.as_object().at("id").as_uint() <= v0 ||
          it.as_object().at("kind").as_string() != "memory") {
        continue;
      }
      const FlexData data = it.as_object().at("data");
      if (data.as_object().at("stage").as_string() == "rv") {
        r.report = data;
      }
    }
    return r;
  };
  // A refusal's report: the moment, the input frame, the parts summing
  // to `need`, the working-set gate short of what the stage says it has.
  auto refused = [](const Out& r, const char* at, std::uint64_t need,
                    std::size_t gates, std::size_t ws_have) {
    if (!r.report.is_object()) { return false; }
    const auto o = r.report.as_object();
    const FlexData parts = o.at("parts");
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < parts.as_array().size(); ++i) {
      const FlexData p = parts.as_array()[i];
      sum += p.as_object().at("bytes").as_uint();
    }
    const FlexData g = o.at("gates");
    if (g.as_array().size() != gates) { return false; }
    const FlexData g0 = g.as_array()[0];
    const auto go = g0.as_object();
    return o.at("step").as_string() == "upscale" &&
           o.at("at").as_string() == at &&
           o.at("width").as_uint() == 960 && o.at("height").as_uint() == 540 &&
           o.at("need").as_uint() == need && sum == need &&
           go.at("name").as_string() == "gpu_working_set" &&
           !go.at("ok").as_bool() && go.at("have").as_uint() <= ws_have;
  };

  // 1. Refused at the launch: never loaded, nothing left booked.
  const Out a = run(3840, 2160, 540, 960, 3);
  std::printf("[relayvsr] refuse at launch: working set %.1f MB, the run "
              "%.1f MB; frames %d, loads %d, after: weights %.1f MB, arena "
              "%.1f MB, reported %d\n",
              a.ws / 1048576.0, (weights + big) / 1048576.0, a.frames,
              a.loads, a.weights_after / 1048576.0, a.arena_after / 1048576.0,
              (int)a.report.is_object());
  EXPECT_TRUE(a.ok);
  EXPECT_TRUE(a.frames == 0);
  EXPECT_TRUE(a.loads == 0);
  EXPECT_TRUE(a.weights_after == 0 && a.arena_after == 0);
  EXPECT_TRUE(refused(a, "launch", weights + big, 1, a.ws));
  if (a.report.is_object()) {
    const FlexData g = a.report.as_object().at("gates");
    const FlexData g0 = g.as_array()[0];
    EXPECT_TRUE(g0.as_object().at("have").as_uint() == a.ws);
  }

  // 2. Admitted at the launch beside a peer holding most of the working
  // set: the run alone fits, and that is the launch's question.
  {
    RelayVsrUpscaleStage probe(&sess, "probe", std::vector<InEdge>{},
                               stage_cfg(640, 352));
    const std::size_t ws = probe.gpu_working_set();
    const std::size_t want = (std::size_t)((double)(weights + small) / 0.95);
    if (ws < want + (1ull << 30)) {
      std::printf("[relayvsr] refuse: SKIPPED the peer arm (working set "
                  "%.1f MB under the small run's %.1f MB + 1 GB)\n",
                  ws / 1048576.0, want / 1048576.0);
    } else {
      // Past what the room left would hold beside the run.
      const std::size_t peer_bytes = ws - want + (512ull << 20);
      SharedBuffer peer = mc->make_shared_buffer(peer_bytes);
      ASSERT_TRUE(!peer.empty());
      const Out b = run(640, 352, 88, 160, 0);
      std::printf("[relayvsr] admit beside a peer: peer %.1f MB of %.1f MB, "
                  "the run %.1f MB; loads %d, reported %d, after: weights "
                  "%.1f MB, arena %.1f MB\n",
                  peer_bytes / 1048576.0, ws / 1048576.0,
                  (weights + small) / 1048576.0, b.loads,
                  (int)b.report.is_object(), b.weights_after / 1048576.0,
                  b.arena_after / 1048576.0);
      EXPECT_TRUE(b.ok);
      EXPECT_TRUE(b.loads == 1);
      EXPECT_FALSE(b.report.is_object());
      EXPECT_TRUE(b.weights_after == 0 && b.arena_after == 0);
    }
  }

  // 3. No size given: loaded, then refused at the first frame and let go.
  const Out c = run(0, 0, 540, 960, 3);
  std::printf("[relayvsr] refuse at the first frame: the ring %.1f MB; "
              "frames %d, loads %d, after: weights %.1f MB, arena %.1f MB, "
              "reported %d\n",
              big / 1048576.0, c.frames, c.loads, c.weights_after / 1048576.0,
              c.arena_after / 1048576.0, (int)c.report.is_object());
  EXPECT_TRUE(c.ok);
  EXPECT_TRUE(c.frames == 0);
  EXPECT_TRUE(c.loads == 1);
  EXPECT_TRUE(c.weights_after == 0 && c.arena_after == 0);
  EXPECT_TRUE(refused(c, "first_frame", big, 2, c.ws));
}

// THE STAGE TAKES ITS ACCELERATION: a launch with `ane_ffn` (a pinned
// half) and, where there are matrix cores, `i8_gemm` at 'token'. Its frames
// out as ever; its two checkpoints PLANNED at what the loads hold (the
// projections' int8 layout on the token tier); each ANE module's CoreML
// booking, once the window has run, at what it holds. At 320x180 the
// keyframe's 920 rows give a pinned half less than one 512-row chunk, so
// that module DECLINES (booked 0), while the decoder's 3680 rows take a
// 1024-row chunk (booked at that chunk, under its 2048-row claim). On the
// token tier every feed-forward is held int8, so neither is CLAIMED.
// Everything withdrawn at the end.
TEST(relayvsr, the_stage_takes_its_acceleration)
{
  const char* root = env_("VPIPE_RELAYVSR_TEST_MODEL_PATH");
  if (root == nullptr) { return; }
  genai::RelayVsrFiles files;
  std::string err;
  ASSERT_TRUE(genai::resolve_relayvsr_files(root, &files, &err));
  MetalRelayVsrWan::Config wc;
  MetalRelayVsrDecoder::Config dc;
  ASSERT_TRUE(MetalRelayVsrWan::config_from_checkpoint(files.wan, &wc, &err));
  ASSERT_TRUE(
      MetalRelayVsrDecoder::config_from_checkpoint(files.flash, &dc, &err));
  constexpr int H = 180, W = 320, N = 18;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const bool i8 = mc->supports_matrix_cores();
  wc.i8_tier = dc.i8_tier = i8 ? 2 : -1;
  // As the stage is configured below: untuned (the decoder's down runs the
  // accurate route on 'token', whose tuner the run would book).
  wc.autotune = dc.autotune = false;
  // SageAttention's int8 QK and P*V, which the configs take on matrix
  // cores alone (and the run books: its prologue's scratch).
  wc.sage_attn = dc.sage_attn = wc.sage_pv8 = dc.sage_pv8 = i8;
  auto* mgr = sess.services()->generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src_u = std::make_unique<RvSource>(&sess, "src", std::vector<InEdge>{},
                                          FlexData::make_object());
  src_u->h = H;
  src_u->w = W;
  src_u->count = N;
  src_u->allocate_oports(1);
  auto* src = pl->insert_stage(std::move(src_u));
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("hf_dir", FlexData::make_string(root));
    o.insert_or_assign("output_dtype", FlexData::make_string("f32"));
    o.insert_or_assign("ane_ffn", FlexData::make_bool(true));
    o.insert_or_assign("ane_rows", FlexData::make_real(0.5));
    o.insert_or_assign("autotune", FlexData::make_bool(false));
    o.insert_or_assign("sage_attn", FlexData::make_bool(true));
    o.insert_or_assign("sage_pv8", FlexData::make_bool(true));
    if (i8) {
      o.insert_or_assign("i8_gemm", FlexData::make_bool(true));
      o.insert_or_assign("i8_gemm_tier", FlexData::make_string("token"));
    }
  }
  auto* rv = static_cast<RelayVsrUpscaleStage*>(pl->insert_stage(
      std::make_unique<RelayVsrUpscaleStage>(
          &sess, "rv", std::vector<InEdge>{{src, 0}}, cfg)));
  // A module is claimed only where one can be built.
  int coreml_claims = 0;
  for (const ResourceClaim& c : rv->declare_resources()) {
    coreml_claims += c.kind == model_memory::kCoreMLKind ? 1 : 0;
  }
  EXPECT_TRUE(coreml_claims == (i8 ? 0 : 2));
  auto sink_u = std::make_unique<RvSink>(&sess, "sink",
                                         std::vector<InEdge>{{rv, 0}},
                                         FlexData::make_object());
  sink_u->h = H;
  sink_u->w = W;
  sink_u->files = files;
  auto* sink = static_cast<RvSink*>(pl->insert_stage(std::move(sink_u)));
  // No heap sampler here: the source need not wait for one.
  g_rv_arm.store(false);
  g_rv_armed.store(true);
  bool ok = false;
  {
    PipelineRuntime rt(pl.get(), &sess);
    ok = rt.launch();
    if (ok) { rt.wait_idle(); }
    if (ok) { rt.stop(); }
  }
  ASSERT_TRUE(ok);
  const std::size_t want_w = MetalRelayVsrWan::weight_bytes(wc) +
                             MetalRelayVsrDecoder::weight_bytes(dc);
  const std::size_t ane_k = MetalRelayVsrWan::ane_runtime_bytes(wc);
  const std::size_t ane_d = MetalRelayVsrDecoder::ane_runtime_bytes(dc);
  std::printf("[relayvsr] stage, int8 %s + ANE: %d frames; weights planned "
              "%.1f MB (held by the tier %.1f); ANE booked keyframe %.1f / "
              "decoder %.1f MB (a module %.1f / %.1f); after the stream "
              "%.1f / %.1f\n", i8 ? "token" : "none", sink->frames,
              sink->weights / 1048576.0, want_w / 1048576.0,
              sink->ane_k / 1048576.0, sink->ane_d / 1048576.0,
              ane_k / 1048576.0, ane_d / 1048576.0,
              mgr->declared_scratch("coreml:ane-ffn/rv/keyframe") /
                  1048576.0,
              mgr->declared_scratch("coreml:ane-ffn/rv/decoder") /
                  1048576.0);
  EXPECT_TRUE(sink->frames == N && sink->ordered && sink->shaped);
  EXPECT_TRUE(sink->weights == want_w);
  // The run as booked at the first frame: with Sage's scratch on matrix
  // cores, none elsewhere (the M4 stays f16 / bf16 there).
  const std::size_t want_run = RelayVsrStream::run_bytes(wc, dc, H, W, 16);
  wc.sage_attn = dc.sage_attn = wc.sage_pv8 = dc.sage_pv8 = false;
  const std::size_t run_f16 = RelayVsrStream::run_bytes(wc, dc, H, W, 16);
  std::printf("[relayvsr] stage, sage %s: run booked %.1f MB (%.1f MB "
              "without)\n", i8 ? "on" : "off (no matrix cores)",
              sink->booked / 1048576.0, run_f16 / 1048576.0);
  EXPECT_TRUE(sink->booked == want_run);
  EXPECT_TRUE(!i8 || want_run > run_f16);
  // The keyframe's module declines (its rows); the decoder's arms, at the
  // chunk its frame's rows take -- but on the token tier every
  // feed-forward is held int8, and neither is claimed.
  const std::size_t ane_d_built = genai::AneFeedForward::runtime_bytes(
      dc.dim, dc.ffn, 0,
      genai::relayvsr::ane_chunk(((H + 7) / 8 * 2) * ((W + 7) / 8 * 2)));
  EXPECT_TRUE(sink->ane_k == 0);
  EXPECT_TRUE(sink->ane_d == (i8 ? 0 : ane_d_built));
  EXPECT_TRUE(ane_d_built < ane_d);
  EXPECT_TRUE(mgr->declared_scratch("coreml:ane-ffn/rv/keyframe") == 0);
  EXPECT_TRUE(mgr->declared_scratch("coreml:ane-ffn/rv/decoder") == 0);
  EXPECT_TRUE(mgr->planned_weight_bytes(files.wan) == 0);
}

// The stage's surface: what it refuses at construction, and its ports.
TEST(relayvsr, the_stage_checks_its_config)
{
  Session s;
  auto cfg = [](std::initializer_list<std::pair<const char*, FlexData>> kv) {
    FlexData c = FlexData::make_object();
    for (const auto& [k, v] : kv) { c.as_object().insert_or_assign(k, v); }
    return c;
  };
  auto make = [&](FlexData c) {
    return std::string(RelayVsrUpscaleStage(&s, "x", std::vector<InEdge>{},
                                            std::move(c))
                           .config_error());
  };
  const FlexData dir = FlexData::make_string("kopper/RelayVSR");
  // No model named: nothing to run.
  EXPECT_FALSE(make(cfg({})).empty());
  EXPECT_TRUE(make(cfg({{"hf_dir", dir}})).empty());
  for (int w : {1, 0, 65}) {
    EXPECT_FALSE(make(cfg({{"hf_dir", dir},
                           {"window", FlexData::make_int(w)}}))
                     .empty());
  }
  for (int w : {2, 16, 64}) {
    EXPECT_TRUE(make(cfg({{"hf_dir", dir}, {"window", FlexData::make_int(w)}}))
                    .empty());
  }
  EXPECT_FALSE(make(cfg({{"hf_dir", dir},
                         {"output_dtype", FlexData::make_string("f16")}}))
                   .empty());
  for (const char* d : {"auto", "u8", "f32"}) {
    EXPECT_TRUE(make(cfg({{"hf_dir", dir},
                          {"output_dtype", FlexData::make_string(d)}}))
                    .empty());
  }
  // The int8 tier, by name, read only where i8_gemm is on.
  const FlexData on = FlexData::make_bool(true);
  EXPECT_FALSE(make(cfg({{"hf_dir", dir}, {"i8_gemm", on},
                         {"i8_gemm_tier", FlexData::make_string("int4")}}))
                   .empty());
  EXPECT_TRUE(make(cfg({{"hf_dir", dir},
                        {"i8_gemm_tier", FlexData::make_string("int4")}}))
                  .empty());
  for (const char* t : {"accurate", "fast", "token", "token_plain"}) {
    EXPECT_TRUE(make(cfg({{"hf_dir", dir}, {"i8_gemm", on},
                          {"i8_gemm_tier", FlexData::make_string(t)}}))
                    .empty());
  }
  // The output size the launch plans at: both or neither, multiples of 4
  // (the frames are 4x the input's), 32 .. 32768.
  auto sized = [&](int w, int h) {
    return make(cfg({{"hf_dir", dir}, {"output_width", FlexData::make_int(w)},
                     {"output_height", FlexData::make_int(h)}}));
  };
  EXPECT_TRUE(sized(0, 0).empty());
  EXPECT_TRUE(sized(1408, 960).empty());
  EXPECT_TRUE(sized(32768, 32).empty());
  EXPECT_FALSE(sized(1408, 0).empty());
  EXPECT_FALSE(sized(0, 960).empty());
  EXPECT_FALSE(sized(16, 960).empty());
  EXPECT_FALSE(sized(1408, 32800).empty());
  EXPECT_FALSE(sized(1410, 960).empty());
  EXPECT_FALSE(sized(1408, 962).empty());
  EXPECT_TRUE(make(cfg({{"hf_dir", dir}, {"ane_ffn", on},
                        {"ane_rows", FlexData::make_real(0.5)}}))
                  .empty());
  RelayVsrUpscaleStage st(&s, "y", std::vector<InEdge>{},
                          cfg({{"hf_dir", dir}}));
  const StageSpec& sp = st.spec();
  EXPECT_TRUE(std::string(sp.type_name) == "relayvsr-upscale");
  EXPECT_TRUE(sp.iports.size() == 1 && sp.oports.size() == 1);
  // A frame leaves once its window is whole: the oport is a clock of its
  // own, not the iport's.
  EXPECT_TRUE(st.iport_clock_group(0) != st.oport_clock_group(0));
  EXPECT_TRUE(st.window() == 16);
}

// THE MATRIX-CORE GEMM ROUNDS ONCE WITH ITS BIAS (M5): the bias-epilogue
// tiles against the old route (the product rounded, then the bias added and
// rounded again), each against x W^T + b in f64 -- the share of outputs that
// are the exact value correctly rounded, and the rel-L2. Ragged M and N, the
// square and the wide tile, bf16 and f16.
TEST(relayvsr, matrix_core_bias_rounds_once)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  struct Shape { int M, N, K; };
  for (const bool bf16 : {true, false}) {
    for (const Shape sh : {Shape{200, 328, 3072}, Shape{136, 200, 14336}}) {
      std::mt19937 rng(11);
      std::normal_distribution<float> nd;
      auto rnd = [&](std::size_t n, float s) {
        std::vector<float> v(n);
        for (float& e : v) {
          e = genai::relayvsr::round_elt(s * nd(rng), bf16);
        }
        return v;
      };
      const auto x = rnd((std::size_t)sh.M * sh.K, 1.0f);
      const auto w = rnd((std::size_t)sh.N * sh.K, 0.02f);
      const auto b = rnd((std::size_t)sh.N, 3.0f);
      SharedBuffer xb = genai::relayvsr::upload_elt(mc, x, bf16);
      SharedBuffer wb = genai::relayvsr::upload_elt(mc, w, bf16);
      SharedBuffer bb = genai::relayvsr::upload_elt(mc, b, bf16);
      std::vector<double> ref((std::size_t)sh.M * sh.N);
      for (int m = 0; m < sh.M; ++m) {
        for (int n = 0; n < sh.N; ++n) {
          double acc = b[(std::size_t)n];
          for (int k = 0; k < sh.K; ++k) {
            acc += (double)x[(std::size_t)m * sh.K + k] *
                   (double)w[(std::size_t)n * sh.K + k];
          }
          ref[(std::size_t)m * sh.N + n] = acc;
        }
      }
      double rel[2] = {0, 0}, exact[2] = {0, 0};
      for (int arm = 0; arm < 2; ++arm) {
        if (arm == 1) { setenv("VPIPE_RELAYVSR_BIAS_PASS", "1", 1); }
        genai::relayvsr::Ops op;
        std::string err;
        const bool ok = op.load(mc, bf16, &err);
        unsetenv("VPIPE_RELAYVSR_BIAS_PASS");
        ASSERT_TRUE(ok && op.matrix_cores());
        SharedBuffer yb = mc->make_shared_buffer((std::size_t)sh.M * sh.N * 2);
        auto cs = mc->make_command_stream();
        ComputeEncoder enc = cs.begin_compute();
        op.gemm(enc, xb, 0, wb, &bb, yb, 0, sh.M, sh.N, sh.K);
        enc.end();
        cs.commit().wait();
        const auto* y = (const std::uint16_t*)yb.contents();
        double num = 0, den = 0;
        std::size_t hit = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
          const float got = bf16 ? genai::relayvsr::from_bf16(y[i])
                                 : genai::relayvsr::from_f16(y[i]);
          num += (got - ref[i]) * (got - ref[i]);
          den += ref[i] * ref[i];
          hit += got == genai::relayvsr::round_elt((float)ref[i], bf16);
        }
        rel[arm] = std::sqrt(num / den);
        exact[arm] = (double)hit / (double)ref.size();
      }
      std::printf("[relayvsr] matmul2d %s %dx%dx%d: bias fused rel-L2 %.3e, "
                  "%.1f%% correctly rounded; bias pass %.3e, %.1f%%\n",
                  bf16 ? "bf16" : "f16", sh.M, sh.N, sh.K, rel[0],
                  100 * exact[0], rel[1], 100 * exact[1]);
      EXPECT_TRUE(rel[0] <= rel[1]);
      EXPECT_TRUE(exact[0] > exact[1]);
    }
  }
}

namespace {

// One golden case run through the stream: each keyframe's latent (and its
// rel-L2 from the fp32 truth), each frame's rel-L2 from it, and the frames'
// digest.
struct StreamRun {
  std::vector<std::vector<float>> latents;
  std::vector<double> kf, frames, bar_kf, bar_frames;
  std::uint64_t digest = 1469598103934665603ull;
};

bool
run_stream_(MetalCompute* mc, Model& m, const fs::path& c, StreamRun* r)
{
  const FlexData meta = read_json_(c / "meta.json");
  const int h = int_(meta, "height"), w = int_(meta, "width");
  const int ph = int_(meta, "pad_height"), pw = int_(meta, "pad_width");
  const int count = int_(meta, "count");
  r->bar_kf = reals_(meta, "bar_latent");
  r->bar_frames = reals_(meta, "bar_out");
  RelayVsrStream::Options o;
  o.window = int_(meta, "window");
  o.seed = (std::uint64_t)int_(meta, "seed");
  o.f32_out = true;
  RelayVsrStream st(m.wan.get(), m.dec.get(), mc, o);
  const std::size_t nl = (std::size_t)48 * (ph / 4) * (pw / 4);
  st.on_keyframe = [&](std::int64_t f, const std::vector<float>&,
                       const SharedBuffer& lat) {
    std::vector<float> ref;
    const auto* p = (const float*)lat.contents();
    r->latents.emplace_back(p, p + nl);
    r->kf.push_back(read_bin_(c / fmt("latent_{}_f32.bin", f)(), nl, &ref)
                        ? rel_l2_(p, ref.data(), nl)
                        : 1e9);
  };
  std::string err;
  std::vector<OutFrame> out;
  for (int i = 0; i < count; ++i) {
    std::vector<std::uint8_t> px;
    if (!read_bin_(c / fmt("lr_{}.u8", i)(), (std::size_t)3 * h * w, &px) ||
        !st.push(px.data(), RelayVsrStream::In::kU8, h, w, &err) ||
        !drain_(st, &out, nullptr, &err)) {
      std::printf("[relayvsr] %s\n", err.c_str());
      return false;
    }
  }
  if (!st.finish(&err) || !drain_(st, &out, nullptr, &err) ||
      (int)out.size() != count) {
    return false;
  }
  const std::size_t ne = (std::size_t)3 * 16 * h * w;
  std::vector<float> ours(ne), ref;
  for (const OutFrame& f : out) {
    r->digest = fnv_(f.px.data(), f.px.size(), r->digest);
    if (!read_bin_(c / fmt("out_{}_f32.bin", f.index)(), ne, &ref)) {
      return false;
    }
    const auto* p = (const float*)f.px.data();
    for (std::size_t e = 0; e < ne; ++e) { ours[e] = 2.0f * p[e] - 1.0f; }
    r->frames.push_back(rel_l2_(ours.data(), ref.data(), ne));
  }
  return true;
}

double
mean_(const std::vector<double>& v)
{
  double s = 0.0;
  for (double x : v) { s += x; }
  return v.empty() ? 0.0 : s / (double)v.size();
}

// The decoder alone, teacher-forced on the reference's fp32 keyframes over
// one golden case: each emitted frame's rel-L2 from the fp32 truth.
bool
decode_tf_(MetalCompute* mc, Model& m, const genai::relayvsr::Ops& op16,
           const fs::path& c, std::vector<double>* got,
           std::vector<double>* bar)
{
  const FlexData meta = read_json_(c / "meta.json");
  const int h = int_(meta, "height"), w = int_(meta, "width");
  const int ph = int_(meta, "pad_height"), pw = int_(meta, "pad_width");
  const int count = int_(meta, "count"), window = int_(meta, "window");
  *bar = reals_(meta, "bar_out_tf");
  std::map<int, MetalRelayVsrDecoder::Condition> cond;
  FlexData mm = meta;
  std::string err;
  const FlexData kfs = mm.as_object().at("keyframes");
  for (const FlexData& kf : kfs.as_array()) {
    const int k = (int)kf.as_int();
    std::vector<float> lat;
    if (!read_bin_(c / fmt("latent_{}_f32.bin", k)(),
                   (std::size_t)48 * (ph / 4) * (pw / 4), &lat)) {
      return false;
    }
    SharedBuffer lb = genai::relayvsr::upload_f32(mc, lat);
    if (!m.dec->prepare(lb, ph / 4, pw / 4, &cond[k], &err)) { return false; }
  }
  got->assign((std::size_t)count, -1.0);
  SharedBuffer out;
  const std::size_t ne = (std::size_t)3 * 16 * h * w;
  std::vector<float> ours(ne), ref;
  for (const auto& [idx, emit_from] : plan_(count, window)) {
    m.dec->begin_window();
    for (std::size_t off = 0; off < idx.size(); ++off) {
      const int i = idx[off];
      std::vector<std::uint8_t> px;
      if (!read_bin_(c / fmt("lr_{}.u8", i)(), (std::size_t)3 * h * w,
                     &px)) {
        return false;
      }
      SharedBuffer lr = upload_(mc, op16, px, h, w);
      if (!m.dec->decode(lr, ph, pw, cond[idx.front()], cond[idx.back()],
                         4 * h, 4 * w, true, &out, &err)) {
        return false;
      }
      if ((int)off < emit_from) { continue; }
      if (!read_bin_(c / fmt("out_{}_f32.bin", i)(), ne, &ref)) {
        return false;
      }
      const auto* o = (const float*)out.contents();
      for (std::size_t e = 0; e < ne; ++e) { ours[e] = 2.0f * o[e] - 1.0f; }
      (*got)[(std::size_t)i] = rel_l2_(ours.data(), ref.data(), ne);
    }
  }
  return true;
}

}  // namespace

// THE INT8 TIERS (matrix cores only; a SKIP without them), each ENGAGED --
// every keyframe-block projection on the int8 route (240 a keyframe), the
// decoder's where its depth allows (accurate: the 1408-deep down alone;
// fast and token: q, k, v, o and gate|up held, the down accurate) -- and
// what it costs held to a GROSS bar against the fp32 truth. int8 is a
// numerics choice, opt-in and lossy, so the bar is a multiple of the
// reference's own bf16 distance, not the distance itself. The decoder
// alone, teacher-forced on the reference's latents, gives its share.
TEST(relayvsr, int8_tiers_stay_close)
{
  const char* golden = env_("VPIPE_RELAYVSR_GOLDEN");
  if (golden == nullptr || env_("VPIPE_RELAYVSR_TEST_MODEL_PATH") == nullptr) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[relayvsr] int8 tiers: no matrix cores here; SKIPPED\n");
    return;
  }
  genai::relayvsr::Ops op16;
  std::string err;
  ASSERT_TRUE(op16.load(mc, false, &err));
  MetalRelayVsrWan::Config wc0;
  MetalRelayVsrDecoder::Config dc0;
  for (int tier = 0; tier <= 3; ++tier) {
    Accel a;
    a.i8 = tier;
    Model m;
    ASSERT_TRUE(load_(mc, true, &m, a));
    if (!m.wan) { continue; }
    EXPECT_TRUE(m.wan->config().i8_tier == tier);
    EXPECT_TRUE(m.dec->config().i8_tier == tier);
    wc0 = m.wan->config();
    dc0 = m.dec->config();
    wc0.i8_tier = dc0.i8_tier = -1;
    const double held =
        (double)(MetalRelayVsrWan::weight_bytes(m.wan->config()) +
                 MetalRelayVsrDecoder::weight_bytes(m.dec->config()));
    const double dense = (double)(MetalRelayVsrWan::weight_bytes(wc0) +
                                  MetalRelayVsrDecoder::weight_bytes(dc0));
    if (tier >= 1) { EXPECT_TRUE(held < 0.6 * dense); }
    for (const fs::path& c : golden_cases_(golden)) {
      StreamRun r;
      ASSERT_TRUE(run_stream_(mc, m, c, &r));
      ASSERT_TRUE(r.kf.size() == r.bar_kf.size());
      ASSERT_TRUE(r.frames.size() == r.bar_frames.size());
      EXPECT_TRUE(m.wan->i8_gemms() == 8 * m.wan->config().layers);
      EXPECT_TRUE(m.dec->i8_gemms() ==
                  (tier == 0 ? 1 : 6) * m.dec->config().depth);
      double kw = 0.0, fw = 0.0;
      for (std::size_t k = 0; k < r.kf.size(); ++k) {
        kw = std::max(kw, r.kf[k] / r.bar_kf[k]);
      }
      for (std::size_t k = 0; k < r.frames.size(); ++k) {
        fw = std::max(fw, r.frames[k] / r.bar_frames[k]);
      }
      std::vector<double> tf, tf_bar;
      ASSERT_TRUE(decode_tf_(mc, m, op16, c, &tf, &tf_bar));
      double tw = 0.0;
      for (std::size_t k = 0; k < tf.size(); ++k) {
        tw = std::max(tw, tf[k] / tf_bar[k]);
      }
      std::printf("[relayvsr] int8 %s, %s: keyframes rel-L2 mean %.4f "
                  "(reference bf16 %.4f, worst ratio %.2f); frames %.4f "
                  "(%.4f, worst %.2f); the decoder alone %.4f (%.4f, worst "
                  "%.2f); weights %.0f MB against %.0f\n",
                  tier_name_(tier), c.filename().c_str(), mean_(r.kf),
                  mean_(r.bar_kf), kw, mean_(r.frames), mean_(r.bar_frames),
                  fw, mean_(tf), mean_(tf_bar), tw, held / 1048576.0,
                  dense / 1048576.0);
      // Over what each tier measured on an M5 Max (worst keyframe / frame
      // / decoder alone: accurate 1.31 / 1.12 / 0.32, fast 1.35 / 1.18 /
      // 1.20, token 1.39 / 1.17 / 1.19, token_plain 3.89 / 3.04 / 1.62).
      const bool plain = tier == 3;
      EXPECT_TRUE(kw <= (plain ? 4.5 : 1.6));
      EXPECT_TRUE(fw <= (plain ? 3.5 : 1.35));
      EXPECT_TRUE(tw <= (plain ? 1.9 : 1.4));
    }
  }
}

// THE NEURAL ENGINE TIER: both models' feed-forward split by rows with the
// ANE at a PINNED half (so every block splits, whichever engine is faster
// here) -- the keyframe DiT's in fp16 there beside bf16 on the GPU, the
// decoder's fp16 both ways. ENGAGED (rows on the ANE in both), and the
// stream as close to the fp32 truth as the GPU-only one is: the ANE's rows
// are the same arithmetic in fp16. A SKIP where the ANE tier cannot be
// built (no CoreML emitter).
TEST(relayvsr, ane_feed_forward_stays_close)
{
  const char* golden = env_("VPIPE_RELAYVSR_GOLDEN");
  if (golden == nullptr || env_("VPIPE_RELAYVSR_TEST_MODEL_PATH") == nullptr) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  for (const fs::path& c : golden_cases_(golden)) {
    StreamRun gpu, ane;
    {
      Model m;
      ASSERT_TRUE(load_(mc, true, &m));
      if (!m.wan) { return; }
      ASSERT_TRUE(run_stream_(mc, m, c, &gpu));
    }
    Accel a;
    a.ane = true;
    a.ane_rows = 0.5f;
    Model m;
    ASSERT_TRUE(load_(mc, true, &m, a));
    ASSERT_TRUE(run_stream_(mc, m, c, &ane));
    if (m.wan->ane_rows() == 0 && m.dec->ane_rows() == 0) {
      std::printf("[relayvsr] ANE tier: not built here; SKIPPED\n");
      return;
    }
    EXPECT_TRUE(m.wan->ane_rows() > 0);
    EXPECT_TRUE(m.dec->ane_rows() > 0);
    ASSERT_TRUE(ane.latents.size() == gpu.latents.size());
    double lat = 0.0;
    for (std::size_t k = 0; k < ane.latents.size(); ++k) {
      lat = std::max(lat, rel_l2_(ane.latents[k].data(),
                                  gpu.latents[k].data(),
                                  gpu.latents[k].size()));
    }
    double kw = 0.0, fw = 0.0;
    for (std::size_t k = 0; k < ane.kf.size(); ++k) {
      kw = std::max(kw, ane.kf[k] / ane.bar_kf[k]);
    }
    for (std::size_t k = 0; k < ane.frames.size(); ++k) {
      fw = std::max(fw, ane.frames[k] / ane.bar_frames[k]);
    }
    std::printf("[relayvsr] ANE %s: %ld / %ld rows a keyframe / a frame on "
                "the ANE; latents %.4f from the GPU's (worst); keyframes "
                "rel-L2 mean %.4f (GPU %.4f, reference bf16 %.4f, worst "
                "ratio %.2f); frames %.4f (GPU %.4f, %.4f, worst %.2f)\n",
                c.filename().c_str(), m.wan->ane_rows(), m.dec->ane_rows(),
                lat, mean_(ane.kf), mean_(gpu.kf), mean_(ane.bar_kf), kw,
                mean_(ane.frames), mean_(gpu.frames), mean_(ane.bar_frames),
                fw);
    EXPECT_TRUE(kw <= 1.1 && fw <= 1.1);
  }
}

// VPIPE_RELAYVSR_WATTN_BENCH=1: the windowed attention alone at both
// models' 480x270 geometry -- the decoder's (68 x 120 tokens, four planes:
// 48 x 48 truncated twice, 24 x 24 twice; 8 / 2 heads of 64, f16) and the
// keyframe DiT's (34 x 60, four 22 x 40 shifted planes, 24 heads of 128,
// bf16) -- on every route, the int8 ones with and without their prologue.
TEST(relayvsr, window_attention_bench)
{
  if (env_("VPIPE_RELAYVSR_WATTN_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  struct Geo {
    const char* name;
    bool bf16;
    int H, W, hd, Hq, Hkv;
    int win[4][2];
    bool sh;
  };
  const Geo geos[2] = {
      {"decoder", false, 68, 120, 64, 8, 2,
       {{48, 48}, {48, 48}, {24, 24}, {24, 24}}, false},
      {"keyframe", true, 34, 60, 128, 24, 24,
       {{22, 40}, {22, 40}, {22, 40}, {22, 40}}, true}};
  for (const Geo& g : geos) {
    genai::relayvsr::Ops op;
    std::string err;
    ASSERT_TRUE(op.load(mc, g.bf16, &err));
    const int n = g.H * g.W, qd = g.Hq * g.hd, kd = g.Hkv * g.hd;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd;
    auto rnd = [&](std::size_t c) {
      std::vector<float> v(c);
      for (float& x : v) { x = nd(rng); }
      return genai::relayvsr::upload_elt(mc, v, g.bf16);
    };
    SharedBuffer q = rnd((std::size_t)n * qd);
    SharedBuffer o = mc->make_shared_buffer((std::size_t)n * qd * 2);
    std::vector<SharedBuffer> kb, vb;
    std::vector<genai::relayvsr::WindowPlane> planes(4);
    for (int p = 0; p < 4; ++p) {
      kb.push_back(rnd((std::size_t)n * kd));
      vb.push_back(rnd((std::size_t)n * kd));
    }
    for (int p = 0; p < 4; ++p) {
      auto& pl = planes[(std::size_t)p];
      pl.k = &kb[(std::size_t)p]; pl.k_tok = kd;
      pl.v = &vb[(std::size_t)p]; pl.v_tok = kd;
      pl.win_h = g.win[p][0]; pl.win_w = g.win[p][1]; pl.shifted = g.sh;
    }
    auto time_ms = [&](int mode, bool prologue) {
      constexpr int R = 20;
      double best = 1e30;
      for (int round = 0; round < 3; ++round) {
        op.set_window_prologue(true);
        {
          auto cs = mc->make_command_stream();
          ComputeEncoder enc = cs.begin_compute();
          op.window_attention(enc, q, 0, qd, planes, o, 0, qd, g.H, g.W,
                              g.Hq, g.Hkv, g.hd, mode);
          enc.end();
          cs.commit().wait();
        }
        op.set_window_prologue(prologue);
        auto cs = mc->make_command_stream();
        const auto t0 = std::chrono::steady_clock::now();
        ComputeEncoder enc = cs.begin_compute();
        for (int r = 0; r < R; ++r) {
          op.window_attention(enc, q, 0, qd, planes, o, 0, qd, g.H, g.W,
                              g.Hq, g.Hkv, g.hd, mode);
        }
        enc.end();
        cs.commit().wait();
        best = std::min(best, std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                      .count() / R);
      }
      op.set_window_prologue(true);
      return best;
    };
    const bool nax = op.nax_attention();
    op.set_nax_attention(false);
    const double alu = time_ms(0, true);
    op.set_nax_attention(nax);
    if (!nax) {
      std::printf("[relayvsr] window attention %s: alu %.2f ms (no matrix "
                  "cores)\n", g.name, alu);
      continue;
    }
    for (const bool staged : {false, true}) {
      op.set_window_staged(staged);
      const double f = time_ms(0, true);
      const double q1 = time_ms(1, true), q1k = time_ms(1, false);
      const double q2 = time_ms(2, true), q2k = time_ms(2, false);
      std::printf("[relayvsr] window attention %s, %s: alu %.2f ms, nax "
                  "%.2f, int8 qk %.2f (kernel %.2f), int8 qk+pv %.2f "
                  "(kernel %.2f)\n", g.name, staged ? "staged" : "direct",
                  alu, f, q1, q1k, q2, q2k);
    }
    op.set_window_staged(true);
  }
}

// SAGEATTENTION (matrix cores only; a SKIP without them): the int8 QK,
// and the int8 QK and P*V, in both models' windowed attention and the
// keyframe's cross-attention -- ENGAGED (the matrix-core kernels, the mode
// asked; the cross-attention's from 4096 rows by default, lowered here so
// the goldens' few hundred reach it) -- with the stream and the decoder
// alone held to a bar against the fp32 truth over the reference's own
// bf16 distance, as the int8 GEMM tiers are. MEASURED (M5 Max), worst
// ratios keyframes / frames / decoder alone: off 1.05 / 0.95 / 0.20, QK
// 1.19 / 1.10 / 1.15, QK and P*V 1.73 / 1.31 / 1.15.
TEST(relayvsr, sage_attention_stays_close)
{
  const char* golden = env_("VPIPE_RELAYVSR_GOLDEN");
  if (golden == nullptr || env_("VPIPE_RELAYVSR_TEST_MODEL_PATH") == nullptr) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[relayvsr] sage: no matrix cores here; SKIPPED\n");
    return;
  }
  genai::relayvsr::Ops op16;
  std::string err;
  ASSERT_TRUE(op16.load(mc, false, &err));
  for (int sage = 0; sage <= 2; ++sage) {
    Accel a;
    a.sage = sage;
    Model m;
    ASSERT_TRUE(load_(mc, true, &m, a));
    if (!m.wan) { continue; }
    EXPECT_TRUE(m.wan->window_on_matrix_cores());
    EXPECT_TRUE(m.wan->cross_on_matrix_cores());
    EXPECT_TRUE(m.dec->window_on_matrix_cores());
    EXPECT_TRUE(m.wan->config().sage_mode() == sage);
    EXPECT_TRUE(m.dec->config().sage_mode() == sage);
    if (sage > 0) {
      setenv("VPIPE_RELAYVSR_XSAGE_MIN_ROWS", "0", 1);
    }
    for (const fs::path& c : golden_cases_(golden)) {
      StreamRun r;
      ASSERT_TRUE(run_stream_(mc, m, c, &r));
      ASSERT_TRUE(r.kf.size() == r.bar_kf.size());
      ASSERT_TRUE(r.frames.size() == r.bar_frames.size());
      double kw = 0.0, fw = 0.0;
      for (std::size_t k = 0; k < r.kf.size(); ++k) {
        kw = std::max(kw, r.kf[k] / r.bar_kf[k]);
      }
      for (std::size_t k = 0; k < r.frames.size(); ++k) {
        fw = std::max(fw, r.frames[k] / r.bar_frames[k]);
      }
      std::vector<double> tf, tf_bar;
      ASSERT_TRUE(decode_tf_(mc, m, op16, c, &tf, &tf_bar));
      double tw = 0.0;
      for (std::size_t k = 0; k < tf.size(); ++k) {
        tw = std::max(tw, tf[k] / tf_bar[k]);
      }
      std::printf("[relayvsr] sage %s, %s: keyframes rel-L2 mean %.4f "
                  "(reference bf16 %.4f, worst ratio %.2f); frames %.4f "
                  "(%.4f, worst %.2f); the decoder alone %.4f (%.4f, worst "
                  "%.2f)\n",
                  sage == 0 ? "off (f16 matrix cores)"
                  : sage == 1 ? "int8 qk" : "int8 qk+pv",
                  c.filename().c_str(), mean_(r.kf), mean_(r.bar_kf), kw,
                  mean_(r.frames), mean_(r.bar_frames), fw, mean_(tf),
                  mean_(tf_bar), tw);
      const double bk = sage == 0 ? 1.1 : sage == 1 ? 1.4 : 2.0;
      const double bf = sage == 0 ? 1.1 : sage == 1 ? 1.3 : 1.5;
      const double bt = sage == 0 ? 0.5 : 1.4;
      EXPECT_TRUE(kw <= bk && fw <= bf && tw <= bt);
    }
    unsetenv("VPIPE_RELAYVSR_XSAGE_MIN_ROWS");
  }
}
