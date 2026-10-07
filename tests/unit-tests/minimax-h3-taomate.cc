// TaoMate-H3's host arithmetic (minimax-h3-taomate.h), against upstream.
//
// Two tiers. The first runs everywhere: the request plans and the four
// schedules are short tables, copied here from upstream's own functions
// (tools/dump_taomate_h3_host_golden.py prints them), so a regression in
// the arithmetic that decides WHICH frames a chunk owns or WHICH sigmas
// it steps through fails on any box.
//
// The second compares the full layouts -- every row's rotary
// coordinates on the run's global timeline, for chunks of three
// requests at two canvases -- and the request noise, bit for bit, with
// the golden that script writes from upstream's code:
//   VPIPE_H3_TAOMATE_HOST_GOLDEN = its output dir (host.json, noise.bin).
// Unset, those tests skip.

#include "minitest.h"

#include "common/flex-data.h"
#include "generative-models/minimax-h3/minimax-h3-scheduler.h"
#include "generative-models/minimax-h3/minimax-h3-taomate.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
namespace tmh = vpipe::genai::minimax_h3::taomate;

namespace {

// upstream: [index, groups, frame_start, frame_stop, video_start,
// video_stop, audio_start, audio_stop] per phase.
struct PhaseRow { int v[8]; };

bool
phase_is(const tmh::Phase& p, const PhaseRow& r)
{
  return p.index == r.v[0] && p.group_count == r.v[1] &&
         p.frame_start == r.v[2] && p.frame_stop == r.v[3] &&
         p.video_latent_start == r.v[4] && p.video_latent_stop == r.v[5] &&
         p.audio_latent_start == r.v[6] && p.audio_latent_stop == r.v[7];
}

const char*
golden_dir()
{
  const char* g = std::getenv("VPIPE_H3_TAOMATE_HOST_GOLDEN");
  return (g != nullptr && *g != '\0') ? g : nullptr;
}

bool
load_json(const std::string& path, FlexData* out)
{
  std::ifstream in(path);
  if (!in) { return false; }
  std::stringstream ss;
  ss << in.rdbuf();
  try {
    *out = FlexData::from_json(ss.str());
  } catch (...) {
    return false;
  }
  return out->is_object();
}

std::vector<double>
reals(const FlexData& arr)
{
  std::vector<double> v;
  const auto a = arr.as_array();
  v.reserve(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) { v.push_back(a[i].as_real()); }
  return v;
}

}  // namespace

TEST(minimax_h3_taomate, plans_match_upstream)
{
  // direct_5s_plan and the first two canonical continuations.
  const PhaseRow r0[4] = {{{0, 2, 0, 39, 0, 12, 0, 65}},
                          {{1, 2, 39, 73, 12, 22, 65, 122}},
                          {{2, 2, 73, 107, 22, 32, 122, 178}},
                          {{3, 1, 107, 124, 32, 37, 178, 207}}};
  const PhaseRow r1[4] = {{{0, 2, 0, 34, 0, 10, 0, 56}},
                          {{1, 2, 34, 68, 10, 20, 56, 113}},
                          {{2, 2, 68, 102, 20, 30, 113, 170}},
                          {{3, 1, 102, 119, 30, 35, 170, 198}}};
  const PhaseRow r2[4] = {{{0, 2, 0, 34, 0, 10, 0, 57}},
                          {{1, 2, 34, 68, 10, 20, 57, 113}},
                          {{2, 2, 68, 102, 20, 30, 113, 170}},
                          {{3, 1, 102, 119, 30, 35, 170, 198}}};
  const PhaseRow* want[3] = {r0, r1, r2};
  const int frames[3] = {124, 119, 119};
  for (int r = 0; r < 3; ++r) {
    const tmh::Plan p = tmh::request_plan(r);
    ASSERT_TRUE(p.phases.size() == 4);
    if (p.phases.size() != 4) { continue; }
    EXPECT_TRUE(p.native_frames == frames[r]);
    for (int i = 0; i < 4; ++i) {
      if (!phase_is(p.phases[(std::size_t)i], want[r][i])) {
        std::printf("[minimax_h3_taomate] request %d phase %d differs\n", r,
                    i);
        EXPECT_TRUE(false);
      }
    }
  }
  // Half-to-even, not half-up: frame 3 is 5.0 latents, frame 39 is 65.0,
  // and the tie at frame 9 (15.0) is not a tie -- the first real tie is
  // a frame with frame*40 % 24 == 12, i.e. frame 3*k + 1.5, which never
  // happens, so the rounding mode shows only through ODD multiples of
  // 0.5: frame*5/3 ends in .5 never; it ends in .333 / .667. Check the
  // two near-ties around them instead.
  EXPECT_TRUE(tmh::audio_latent_boundary(124) == 207);   // 206.67
  EXPECT_TRUE(tmh::audio_latent_boundary(243) == 405);   // 405.0
  EXPECT_TRUE(tmh::audio_latent_boundary(158) == 263);   // 263.33
}

TEST(minimax_h3_taomate, schedules_match_upstream)
{
  // select_time_shift_sigmas(num_steps=50, shift, (0, 16, 33, 49)) and
  // time_shift_sigmas(num_steps=10, shift), as upstream returns them.
  const float cv[4] = {1.0f, 0.9611650705337524f, 0.8533332943916321f, 0.0f};
  const float ca[4] = {1.0f, 0.8608695268630981f, 0.5925925970077515f, 0.0f};
  const float tv[10] = {1.0f, 0.9896907806396484f, 0.9767442941665649f,
                        0.9599999189376831f, 0.9375f, 0.9056603908538818f,
                        0.8571428060531616f, 0.7741935849189758f,
                        0.5999999642372131f, 0.0f};
  const float ta[10] = {1.0f, 0.9600000381469727f, 0.9130435585975647f,
                        0.8571428060531616f, 0.7894737124443054f,
                        0.7058823704719543f, 0.5999999642372131f,
                        0.46153849363327026f, 0.27272728085517883f, 0.0f};
  MiniMaxH3Scheduler sv(tmh::kVideoShift), sa(tmh::kAudioShift);
  ASSERT_TRUE(sv.set_sigmas(tmh::chunk_raw_grid()));
  ASSERT_TRUE(sa.set_sigmas(tmh::chunk_raw_grid()));
  ASSERT_TRUE(sv.sigmas().size() == 4 && sa.sigmas().size() == 4);
  for (int i = 0; i < 4 && sv.sigmas().size() == 4; ++i) {
    if (sv.sigmas()[(std::size_t)i] != cv[i] ||
        sa.sigmas()[(std::size_t)i] != ca[i]) {
      std::printf("[minimax_h3_taomate] chunk sigma %d: %.9g / %.9g vs "
                  "upstream %.9g / %.9g\n", i, sv.sigmas()[(std::size_t)i],
                  sa.sigmas()[(std::size_t)i], cv[i], ca[i]);
      EXPECT_TRUE(false);
    }
  }
  MiniMaxH3Scheduler tsv(tmh::kVideoShift), tsa(tmh::kAudioShift);
  ASSERT_TRUE(tsv.set_timesteps(tmh::kTeacherGridPoints));
  ASSERT_TRUE(tsa.set_timesteps(tmh::kTeacherGridPoints));
  ASSERT_TRUE(tsv.sigmas().size() == 10 && tsa.sigmas().size() == 10);
  // To ONE float32 ulp here, not exactly: torch's CPU linspace kernel is
  // vectorized, and a lane takes `base + k * step` from the start of its
  // VECTOR rather than the scalar path's halfway split -- so the
  // reference grid depends on the vector width of the box that ran it
  // (this golden: an arm64 Mac; upstream's: x86). Point 3 of ten lands an
  // ulp apart, 6e-8 relative on a timestep.
  auto ulp1 = [](float a, float b) {
    return a == b || std::fabs(a - b) <= 1.2e-7f * std::fabs(b);
  };
  for (int i = 0; i < 10 && tsv.sigmas().size() == 10; ++i) {
    if (!ulp1(tsv.sigmas()[(std::size_t)i], tv[i]) ||
        !ulp1(tsa.sigmas()[(std::size_t)i], ta[i])) {
      std::printf("[minimax_h3_taomate] teacher sigma %d: %.9g / %.9g vs "
                  "upstream %.9g / %.9g\n", i, tsv.sigmas()[(std::size_t)i],
                  tsa.sigmas()[(std::size_t)i], tv[i], ta[i]);
      EXPECT_TRUE(false);
    }
  }
  // The teacher's kept states land where the chunk's steps do -- the
  // pairing the audio injection rests on. Not equal (two different
  // grids), but within a step of the 10-point one.
  EXPECT_TRUE(std::fabs(tv[3] - cv[1]) < 0.01f);
  EXPECT_TRUE(std::fabs(tv[6] - cv[2]) < 0.01f);
}

TEST(minimax_h3_taomate, video_noise_seeds_and_renorm)
{
  EXPECT_TRUE(tmh::video_noise_seed(8301, 0) == 8301u);
  EXPECT_TRUE(tmh::video_noise_seed(8301, 1) == 8301u + 1000003u);
  // mod 2^63, not a wrap at 2^64.
  EXPECT_TRUE(tmh::video_noise_seed((1ull << 63) - 1, 1) == 1000002u);

  // The first chunk is the anchor and is left alone; a later one takes
  // its per-channel mean and population std.
  tmh::VideoRenorm rn;
  std::vector<float> a = {1, 10, 3, 30};          // 2 rows x 2 ch
  rn.apply(a.data(), 2, 2);
  EXPECT_TRUE(a[0] == 1 && a[1] == 10 && a[2] == 3 && a[3] == 30);
  std::vector<float> b = {0, 0, 4, 2};            // mean 2/1, std 2/1
  rn.apply(b.data(), 2, 2);
  // channel 0: anchor mean 2, std 1 -> (0-2)/2*1+2 = 1, (4-2)/2+2 = 3
  // channel 1: anchor mean 20, std 10 -> (0-1)/1*10+20 = 10, 30.
  EXPECT_TRUE(b[0] == 1.0f && b[2] == 3.0f);
  EXPECT_TRUE(b[1] == 10.0f && b[3] == 30.0f);
}

TEST(minimax_h3_taomate, layouts_match_upstream)
{
  const char* gd = golden_dir();
  if (gd == nullptr) { return; }
  FlexData j;
  ASSERT_TRUE(load_json(std::string(gd) + "/host.json", &j));
  if (!j.is_object()) { return; }
  const auto top = j.as_object();

  // Plans for 14 requests: the continuation's global audio cut.
  {
    const FlexData plans = top.at("plans");
    const auto pa = plans.as_array();
    for (std::size_t r = 0; r < pa.size(); ++r) {
      const FlexData pr = pa[r];
      const tmh::Plan p = tmh::request_plan((int)r);
      EXPECT_TRUE(p.native_frames ==
                  (int)pr.as_object().at("native_frames").as_int());
      const FlexData phs = pr.as_object().at("phases");
      const auto phv = phs.as_array();
      ASSERT_TRUE(phv.size() == p.phases.size());
      for (std::size_t i = 0; i < phv.size() && i < p.phases.size(); ++i) {
        const FlexData row = phv[i];
        PhaseRow want{};
        const auto rv = row.as_array();
        for (int k = 0; k < 8; ++k) {
          want.v[k] = (int)rv[(std::size_t)k].as_int();
        }
        if (!phase_is(p.phases[i], want)) {
          std::printf("[minimax_h3_taomate] plan %zu phase %zu differs\n", r,
                      i);
          EXPECT_TRUE(false);
        }
      }
    }
  }

  // Chunk layouts. Positions compared EXACTLY in double: upstream rounds
  // each one once (Fraction -> float), and so must the port.
  int chunks = 0, rows_checked = 0, mismatched = 0;
  {
    const FlexData cs = top.at("chunks");
    const auto ca = cs.as_array();
    for (std::size_t k = 0; k < ca.size(); ++k) {
      const FlexData c = ca[k];
      const auto o = c.as_object();
      const int r = (int)o.at("request").as_int();
      const int ph = (int)o.at("phase").as_int();
      const int tl = (int)o.at("text_len").as_int();
      const tmh::Plan plan = tmh::request_plan(r);
      minimax_h3::PackedLayout L;
      const bool ok = tmh::build_chunk_layout(
          std::vector<int>((std::size_t)tl, minimax_h3::kTextTag),
          plan.phases[(std::size_t)ph], (int)o.at("latent_h").as_int(),
          (int)o.at("latent_w").as_int(), (int)o.at("media_origin").as_int(),
          (int)o.at("video_offset").as_int(),
          (int)o.at("audio_offset").as_int(), &L);
      ASSERT_TRUE(ok);
      if (!ok) { continue; }
      const std::vector<double> want = reals(o.at("position_ids"));
      // Upstream pads the sequence to its packing alignment; vpipe does
      // not. The pad rows sit at the END and carry zeros, and nothing
      // attends to them in the streaming hook (their own document).
      ASSERT_TRUE((int)want.size() >= L.seq_len * 3);
      for (int i = 0; i < L.seq_len * 3 && i < (int)want.size(); ++i) {
        if (L.position_ids[(std::size_t)i] != want[(std::size_t)i]) {
          if (mismatched < 5) {
            std::printf("[minimax_h3_taomate] chunk r%d p%d row %d axis %d: "
                        "%.17g vs upstream %.17g\n", r, ph, i / 3, i % 3,
                        L.position_ids[(std::size_t)i],
                        want[(std::size_t)i]);
          }
          ++mismatched;
        }
      }
      const FlexData tags = o.at("token_tags");
      const auto ta = tags.as_array();
      for (int i = 0; i < L.seq_len; ++i) {
        if (L.token_tags[(std::size_t)i] != (int)ta[(std::size_t)i].as_int()) {
          ++mismatched;
        }
      }
      rows_checked += L.seq_len;
      ++chunks;
    }
  }
  std::printf("[minimax_h3_taomate] %d chunk layouts, %d rows: %d "
              "mismatches\n", chunks, rows_checked, mismatched);
  EXPECT_TRUE(chunks == 24);
  EXPECT_TRUE(mismatched == 0);

  // Teacher layouts: fresh, then two frozen-prefix continuations.
  int tmis = 0, tn = 0;
  {
    const FlexData ts = top.at("teachers");
    const auto ta = ts.as_array();
    for (std::size_t k = 0; k < ta.size(); ++k) {
      const FlexData t = ta[k];
      const auto o = t.as_object();
      const int tl = (int)o.at("text_len").as_int();
      minimax_h3::PackedLayout L;
      const bool ok = tmh::build_teacher_layout(
          std::vector<int>((std::size_t)tl, minimax_h3::kTextTag),
          (int)o.at("ref").as_int(), (int)o.at("audio_t").as_int(),
          (int)o.at("latent_h").as_int(), (int)o.at("latent_w").as_int(),
          (int)o.at("ref_time_start").as_int(),
          (int)o.at("target_time_start").as_int(), &L);
      ASSERT_TRUE(ok);
      if (!ok) { continue; }
      const std::vector<double> want = reals(o.at("position_ids"));
      ASSERT_TRUE((int)want.size() >= L.seq_len * 3);
      for (int i = 0; i < L.seq_len * 3 && i < (int)want.size(); ++i) {
        if (L.position_ids[(std::size_t)i] != want[(std::size_t)i]) {
          if (tmis < 5) {
            std::printf("[minimax_h3_taomate] teacher %zu row %d axis %d: "
                        "%.17g vs %.17g\n", k, i / 3, i % 3,
                        L.position_ids[(std::size_t)i],
                        want[(std::size_t)i]);
          }
          ++tmis;
        }
      }
      ++tn;
    }
  }
  std::printf("[minimax_h3_taomate] %d teacher layouts: %d mismatches\n", tn,
              tmis);
  EXPECT_TRUE(tn == 6);
  EXPECT_TRUE(tmis == 0);
}

TEST(minimax_h3_taomate, noise_matches_upstream)
{
  const char* gd = golden_dir();
  if (gd == nullptr) { return; }
  FlexData j;
  ASSERT_TRUE(load_json(std::string(gd) + "/host.json", &j));
  if (!j.is_object()) { return; }
  std::ifstream bin(std::string(gd) + "/noise.bin", std::ios::binary);
  ASSERT_TRUE((bool)bin);
  const FlexData ns = j.as_object().at("noise");
  const auto na = ns.as_array();
  int cases = 0, bad = 0;
  for (std::size_t k = 0; k < na.size(); ++k) {
    const FlexData n = na[k];
    const auto o = n.as_object();
    const std::uint64_t seed = (std::uint64_t)o.at("seed").as_int();
    const int lh = (int)o.at("latent_h").as_int();
    const int lw = (int)o.at("latent_w").as_int();
    const int vr = (int)o.at("video_rows").as_int();
    const int ar = (int)o.at("audio_rows").as_int();
    std::vector<float> wv((std::size_t)vr * 96), wa((std::size_t)ar * 32);
    bin.read(reinterpret_cast<char*>(wv.data()),
             (std::streamsize)(wv.size() * 4));
    bin.read(reinterpret_cast<char*>(wa.data()),
             (std::streamsize)(wa.size() * 4));
    const std::vector<float> gv = tmh::video_noise_rows(seed, 24, 37, lh, lw);
    const std::vector<float> ga = tmh::audio_noise_rows(seed, 24, lh, lw, 32);
    ASSERT_TRUE(gv.size() == wv.size() && ga.size() == wa.size());
    if (gv.size() != wv.size() || ga.size() != wa.size()) { continue; }
    if (std::memcmp(gv.data(), wv.data(), wv.size() * 4) != 0) { ++bad; }
    if (std::memcmp(ga.data(), wa.data(), wa.size() * 4) != 0) { ++bad; }
    ++cases;
  }
  std::printf("[minimax_h3_taomate] %d noise cases, %d tensors differ\n",
              cases, bad);
  EXPECT_TRUE(cases == 3);
  EXPECT_TRUE(bad == 0);
}

// ---- the streaming forwards, against upstream's own model code ---------
//
// tools/dump_taomate_h3_golden.py runs UPSTREAM's MiniMaxH3DiT on the CPU
// (one rank, FlashAttention-3 as SDPA, the first N blocks plus the whole
// token refiner) with the TaoMate adapter, through upstream's layouts,
// timestep plans, H3StreamingSession and CleanAVKVCache. This replays
// the same forwards here:
//
//   teacher0  base model (adapter at 0), audio-only, request-0 layout
//   teacher1  the same with 40 frozen reference latents
//   chunk0    chunk 0, step 0, empty cache
//   chunk3    chunk 3, step 1, after chunks 0..2 were COMMITTED -- the
//             cache trimmed once, chunk 0 aged to its video sink
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (the FL2VA dir),
// VPIPE_H3_TAOMATE_LORA (the adapter's safetensors), and
// VPIPE_H3_TAOMATE_FWD_GOLDEN (the script's output dir).

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-stream-kv.h"

namespace {

std::vector<float>
read_f32_tm(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<float> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

metal_compute::SharedBuffer
bf16_buf_tm(metal_compute::MetalCompute* mc, const std::vector<float>& v)
{
  metal_compute::SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
  if (b.empty()) { return b; }
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::uint32_t u;
    std::memcpy(&u, &v[i], 4);
    d[i] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }
  return b;
}

double
rel_l2_tm(const metal_compute::SharedBuffer& got, const std::vector<float>& ref)
{
  const auto* g = static_cast<const std::uint16_t*>(got.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const std::uint32_t u = (std::uint32_t)g[i] << 16;
    float f;
    std::memcpy(&f, &u, 4);
    const double d = (double)f - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : 1.0;
}

}  // namespace

TEST(minimax_h3_taomate, forwards_match_upstream)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const char* lora = std::getenv("VPIPE_H3_TAOMATE_LORA");
  const char* gd = std::getenv("VPIPE_H3_TAOMATE_FWD_GOLDEN");
  if (root == nullptr || lora == nullptr || gd == nullptr || *root == '\0' ||
      *lora == '\0' || *gd == '\0') {
    return;
  }
  const std::string g(gd);
  FlexData meta;
  ASSERT_TRUE(load_json(g + "/meta.json", &meta));
  if (!meta.is_object()) { return; }
  const auto mo = meta.as_object();
  const int n_layers = (int)mo.at("n_layers").as_int();
  const int n_text = (int)mo.at("text_len").as_int();
  const int lh = (int)mo.at("latent_h").as_int();
  const int lw = (int)mo.at("latent_w").as_int();
  const int tstep = (int)mo.at("teacher_step").as_int();

  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr));
  cfg.n_layers = n_layers;
  MetalMiniMaxH3Transformer::LoraSpec spec;
  spec.path = lora;
  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg, false, {spec});
  ASSERT_TRUE(m != nullptr);
  if (!m) { return; }
  // ENGAGEMENT before accuracy: four projections a block, the refiner's
  // two blocks included. An adapter that bound nothing would leave the
  // base model, which is what the teacher compares against -- the
  // chunks would then fail for a reason the numbers could not name.
  std::printf("[minimax_h3_taomate] adapter bound %d modules over %d+2 "
              "blocks\n", m->lora_modules(), n_layers);
  EXPECT_TRUE(m->lora_modules() == 4 * (n_layers + 2));

  const metal_compute::SharedBuffer text =
      bf16_buf_tm(mc, read_f32_tm(g + "/text_in.f32"));
  ASSERT_TRUE(!text.empty());
  const std::vector<int> tags((std::size_t)n_text, minimax_h3::kTextTag);

  MiniMaxH3Scheduler tv(tmh::kVideoShift), ta(tmh::kAudioShift);
  MiniMaxH3Scheduler cv(tmh::kVideoShift), ca(tmh::kAudioShift);
  ASSERT_TRUE(tv.set_timesteps(tmh::kTeacherGridPoints) &&
              ta.set_timesteps(tmh::kTeacherGridPoints) &&
              cv.set_sigmas(tmh::chunk_raw_grid()) &&
              ca.set_sigmas(tmh::chunk_raw_grid()));

  auto run = [&](const minimax_h3::PackedLayout& L,
                 const std::vector<float>& vin, const std::vector<float>& ain,
                 float t_v, float t_a, minimax_h3::StreamKv* kv, bool commit,
                 MetalMiniMaxH3Transformer::Velocity* out) {
    const metal_compute::SharedBuffer vb =
        vin.empty() ? metal_compute::SharedBuffer{} : bf16_buf_tm(mc, vin);
    const metal_compute::SharedBuffer ab = bf16_buf_tm(mc, ain);
    std::vector<float> uniq;
    std::vector<int> ri;
    minimax_h3::build_row_timesteps(L, t_v, t_a, 1.0f, &uniq, &ri, 1.0f);
    MetalMiniMaxH3Transformer::Step st;
    st.video = vin.empty() ? nullptr : &vb;
    st.audio = &ab;
    st.text = &text;
    st.layout = &L;
    st.timesteps = &uniq;
    st.row_timestep_index = &ri;
    st.stream_kv = kv;
    st.stream_commit = commit;
    std::string ferr;
    *out = m->forward(st, &ferr);
    if (out->empty()) {
      std::printf("[minimax_h3_taomate] forward: %s\n", ferr.c_str());
    }
    return !out->empty();
  };

  // ---- the teacher: base model, audio only -----------------------------
  m->set_lora_scale(0, 0.0f);
  {
    minimax_h3::PackedLayout L;
    ASSERT_TRUE(tmh::build_teacher_layout(tags, 0, 207, lh, lw, 0, 0, &L));
    MetalMiniMaxH3Transformer::Velocity v;
    ASSERT_TRUE(run(L, {}, read_f32_tm(g + "/teacher0_audio_in.f32"),
                    tv.timesteps()[(std::size_t)tstep],
                    ta.timesteps()[(std::size_t)tstep], nullptr, false, &v));
    if (!v.empty()) {
      const double r =
          rel_l2_tm(v.audio, read_f32_tm(g + "/teacher0_audio_out.f32"));
      std::printf("[minimax_h3_taomate] teacher0 audio rel-L2 %.5f\n", r);
      EXPECT_TRUE(r < 0.03);
    }
  }
  {
    const FlexData t1 = mo.at("teacher1");
    const auto o = t1.as_object();
    minimax_h3::PackedLayout L;
    ASSERT_TRUE(tmh::build_teacher_layout(
        tags, (int)o.at("ref").as_int(), (int)o.at("audio_t").as_int(), lh,
        lw, (int)o.at("ref_time_start").as_int(),
        (int)o.at("target_time_start").as_int(), &L));
    MetalMiniMaxH3Transformer::Velocity v;
    ASSERT_TRUE(run(L, {}, read_f32_tm(g + "/teacher1_audio_in.f32"),
                    tv.timesteps()[(std::size_t)tstep],
                    ta.timesteps()[(std::size_t)tstep], nullptr, false, &v));
    if (!v.empty()) {
      const double r =
          rel_l2_tm(v.audio, read_f32_tm(g + "/teacher1_audio_out.f32"));
      std::printf("[minimax_h3_taomate] teacher1 (frozen reference) audio "
                  "rel-L2 %.5f\n", r);
      EXPECT_TRUE(r < 0.03);
    }
  }

  // ---- the chunks: the adapter, the clean-KV cache --------------------
  m->set_lora_scale(0, 1.0f);
  minimax_h3::StreamKv kv;
  std::string kerr;
  ASSERT_TRUE(kv.init(mc, n_layers, cfg.n_heads, cfg.head_dim,
                      tmh::kv_capacity_rows(lh, lw), &kerr));
  const tmh::Plan plan = tmh::request_plan(0);
  double r_chunk[4] = {0, 0, 0, 0};
  for (const tmh::Phase& ph : plan.phases) {
    minimax_h3::PackedLayout L;
    ASSERT_TRUE(tmh::build_chunk_layout(tags, ph, lh, lw, n_text, 0, 0, &L));
    const std::string c = g + "/chunk" + std::to_string(ph.index);
    if (ph.index == 0 || ph.index == 3) {
      const int k = ph.index == 0 ? 0 : 1;
      MetalMiniMaxH3Transformer::Velocity v;
      ASSERT_TRUE(run(L, read_f32_tm(c + "_video_in.f32"),
                      read_f32_tm(c + "_audio_in.f32"),
                      cv.timesteps()[(std::size_t)k],
                      ca.timesteps()[(std::size_t)k], &kv, false, &v));
      if (!v.empty()) {
        const double r = rel_l2_tm(v.video, read_f32_tm(c + "_video_out.f32"));
        std::printf("[minimax_h3_taomate] chunk %d (step %d, cache %d rows, "
                    "%d audio) video rel-L2 %.5f\n", ph.index, k, kv.rows(),
                    kv.audio_rows(), r);
        EXPECT_TRUE(r < 0.03);
        r_chunk[ph.index] = r;
      }
    }
    if (ph.index == 3) { break; }
    MetalMiniMaxH3Transformer::Velocity v;
    ASSERT_TRUE(run(L, read_f32_tm(c + "_commit_video.f32"),
                    read_f32_tm(c + "_commit_audio.f32"), 1.0f, 1.0f, &kv,
                    true, &v));
  }
  // What upstream's cache held when chunk 3 ran: chunk 0's video, and
  // chunks 1 and 2 whole.
  EXPECT_TRUE(kv.rows() == 994);

  // CONTROLS. The bars above are only evidence if the things they
  // exercise MOVE the answer by well over them: chunk 3 through an EMPTY
  // cache, and chunk 0 with the adapter OFF, must both land several times
  // further from the golden than the real run. Relative, not absolute:
  // at a depth of two blocks the adapter and the cache move the velocity
  // by ~5%, and that grows with depth.
  //   MEASURED at depth 2: no cache 0.0547 against 0.0059 (9x); adapter
  //   off 0.0433 against 0.0063 (7x).
  {
    minimax_h3::StreamKv empty;
    ASSERT_TRUE(empty.init(mc, n_layers, cfg.n_heads, cfg.head_dim,
                           tmh::kv_capacity_rows(lh, lw), &kerr));
    minimax_h3::PackedLayout L;
    ASSERT_TRUE(tmh::build_chunk_layout(tags, plan.phases[3], lh, lw,
                                        n_text, 0, 0, &L));
    const std::string c = g + "/chunk3";
    MetalMiniMaxH3Transformer::Velocity v;
    ASSERT_TRUE(run(L, read_f32_tm(c + "_video_in.f32"),
                    read_f32_tm(c + "_audio_in.f32"), cv.timesteps()[1],
                    ca.timesteps()[1], &empty, false, &v));
    if (!v.empty()) {
      const double r = rel_l2_tm(v.video, read_f32_tm(c + "_video_out.f32"));
      std::printf("[minimax_h3_taomate] control: chunk 3 WITHOUT the cache "
                  "rel-L2 %.5f\n", r);
      EXPECT_TRUE(r > 4.0 * r_chunk[3]);
    }
  }
  {
    m->set_lora_scale(0, 0.0f);
    minimax_h3::StreamKv empty;
    ASSERT_TRUE(empty.init(mc, n_layers, cfg.n_heads, cfg.head_dim,
                           tmh::kv_capacity_rows(lh, lw), &kerr));
    minimax_h3::PackedLayout L;
    ASSERT_TRUE(tmh::build_chunk_layout(tags, plan.phases[0], lh, lw,
                                        n_text, 0, 0, &L));
    const std::string c = g + "/chunk0";
    MetalMiniMaxH3Transformer::Velocity v;
    ASSERT_TRUE(run(L, read_f32_tm(c + "_video_in.f32"),
                    read_f32_tm(c + "_audio_in.f32"), cv.timesteps()[0],
                    ca.timesteps()[0], &empty, false, &v));
    if (!v.empty()) {
      const double r = rel_l2_tm(v.video, read_f32_tm(c + "_video_out.f32"));
      std::printf("[minimax_h3_taomate] control: chunk 0 with the adapter "
                  "OFF rel-L2 %.5f\n", r);
      EXPECT_TRUE(r > 4.0 * r_chunk[0]);
    }
  }
}

// ---- the clean-KV cache's bookkeeping, row by row ----------------------
//
// Upstream's retention is covered against its own code above (the 994
// rows chunk 3 attended). What that cannot reach is the AUDIO RESET --
// every 12 requests -- and the lazy way it is applied here: the rows
// stay in the buffer until the next commit compacts them, and the
// gather skips them meanwhile. Synthetic, exact: every key is the small
// integer fwd * 32 + row, which bf16 holds exactly.
#include "apple-silicon/metal-compute/command-stream.h"

TEST(minimax_h3_taomate, stream_kv_retains_and_drops_audio)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  metal_compute::ComputeFunction copy_rect =
      mc->load_library("llm_elementwise_bf16").function("copy_rect_f16");
  ASSERT_TRUE(copy_rect.valid());
  const int H = 2, D = 4;
  minimax_h3::StreamKv kv;
  std::string err;
  ASSERT_TRUE(kv.init(mc, 1, H, D, 40, &err));

  auto bf = [](float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  };
  auto fl = [](std::uint16_t h) {
    const std::uint32_t u = (std::uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  // One "forward": its head-major keys/values, gathered, then (maybe)
  // committed. Returns the staging keys of head 1, column 3, per row.
  auto step = [&](int fwd, int n_text, int audio, int video, bool commit,
                  std::vector<float>* staged) {
    const int seq = n_text + audio + video;
    metal_compute::SharedBuffer k =
        mc->make_shared_buffer((std::size_t)H * seq * D * 2);
    metal_compute::SharedBuffer v =
        mc->make_shared_buffer((std::size_t)H * seq * D * 2);
    auto* kp = static_cast<std::uint16_t*>(k.contents());
    auto* vp = static_cast<std::uint16_t*>(v.contents());
    for (int h = 0; h < H; ++h) {
      for (int r = 0; r < seq; ++r) {
        for (int d = 0; d < D; ++d) {
          kp[((std::size_t)h * seq + r) * D + d] = bf((float)(fwd * 32 + r));
          vp[((std::size_t)h * seq + r) * D + d] = bf((float)(fwd * 32 + r));
        }
      }
    }
    if (!kv.ensure_staging(seq)) { return false; }
    minimax_h3::StreamKv::CommitPlan plan;
    if (commit && !kv.plan_commit(n_text, audio, video, &plan, &err)) {
      return false;
    }
    const int SR = kv.rows() + seq;
    {
      metal_compute::CommandStream st = mc->make_command_stream();
      metal_compute::ComputeEncoder enc = st.begin_compute();
      kv.encode_gather(enc, copy_rect, 0, k, v, seq);
      if (commit) { kv.encode_commit(enc, copy_rect, 0, plan, seq); }
      enc.end();
      std::string gerr;
      if (!st.commit().wait_ok(&gerr)) { return false; }
    }
    if (staged != nullptr) {
      staged->clear();
      const auto* s =
          static_cast<const std::uint16_t*>(kv.staging_k().contents());
      for (int r = 0; r < SR; ++r) {
        staged->push_back(fl(s[((std::size_t)1 * SR + r) * D + 3]));
      }
    }
    if (commit) { kv.finish_commit(plan); }
    return true;
  };
  // Three chunks, [text 2 | audio a | video v] each.
  ASSERT_TRUE(step(1, 2, 4, 6, true, nullptr));
  ASSERT_TRUE(step(2, 2, 3, 5, true, nullptr));
  EXPECT_TRUE(kv.rows() == 10 + 8);                 // both whole
  ASSERT_TRUE(step(3, 2, 2, 4, true, nullptr));
  // From three on: chunk 0 aged to its video, the two latest whole.
  EXPECT_TRUE(kv.commits() == 3);
  EXPECT_TRUE(kv.rows() == 6 + 8 + 6);
  EXPECT_TRUE(kv.audio_rows() == 3 + 2);

  // The reset. Nothing moves; the gather simply stops seeing audio.
  EXPECT_TRUE(kv.drop_audio() == 5);
  EXPECT_TRUE(kv.rows() == 6 + 5 + 4);
  std::vector<float> s;
  ASSERT_TRUE(step(4, 2, 1, 2, false, &s));
  std::vector<float> want;
  for (int r = 6; r < 12; ++r) { want.push_back((float)(1 * 32 + r)); }
  for (int r = 5; r < 10; ++r) { want.push_back((float)(2 * 32 + r)); }
  for (int r = 4; r < 8; ++r) { want.push_back((float)(3 * 32 + r)); }
  for (int r = 0; r < 5; ++r) { want.push_back((float)(4 * 32 + r)); }
  EXPECT_TRUE(s == want);

  // A commit after the reset compacts: [chunk 0 video | chunk 2 video
  // (its audio gone) | the new chunk's audio and video].
  ASSERT_TRUE(step(4, 2, 1, 2, true, nullptr));
  EXPECT_TRUE(kv.commits() == 3);
  EXPECT_TRUE(kv.rows() == 6 + 4 + 3);
  EXPECT_TRUE(kv.audio_rows() == 1);
  ASSERT_TRUE(step(5, 1, 1, 1, false, &s));
  want.clear();
  for (int r = 6; r < 12; ++r) { want.push_back((float)(1 * 32 + r)); }
  for (int r = 4; r < 8; ++r) { want.push_back((float)(3 * 32 + r)); }
  for (int r = 2; r < 5; ++r) { want.push_back((float)(4 * 32 + r)); }
  for (int r = 0; r < 3; ++r) { want.push_back((float)(5 * 32 + r)); }
  EXPECT_TRUE(s == want);
  if (s != want) {
    for (float x : s) { std::printf("%g ", x); }
    std::printf("\n");
  }
  // A retained set past the capacity is refused, not overflowed.
  minimax_h3::StreamKv::CommitPlan p;
  EXPECT_FALSE(kv.plan_commit(1, 20, 20, &p, &err));
}
