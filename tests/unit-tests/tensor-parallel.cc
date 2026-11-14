#include "minitest.h"
#include "minimax-h3-block-run.h"
#include "residency-reserve.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-bag.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/shared/tp-overlap.h"
#include "generative-models/tensor-parallel.h"
#include "pipeline/resource-claim-delivery.h"
#include "pipeline/resource-plan.h"
#include "stages/generate-video-stage.h"
#include "stages/model-memory-host.h"
#include "stages/model-memory.h"
#include "stages/vae-decode-stage.h"
#include "apple-silicon/tensor-beat.h"
#include "common/job.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using namespace vpipe::metal_compute;
namespace h3 = vpipe::genai::minimax_h3;

namespace {

// bf16 of a small integer, exact.
std::uint16_t
bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)(u >> 16);
}

float
f32_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// A peer that answers each partial with itself, a little late: the sum
// that comes back is the partial, so what the GPU adds doubles it. It
// checks the sequence it is handed and can fail -- or throw -- at a given
// exchange; it keeps what it was handed and what it was told to forget,
// in order ("x<seq>" an exchange answered, "f" a forget).
struct EchoChannel final : TpChannel {
  std::atomic<std::uint64_t> next{0};
  std::atomic<int>           out_of_step{0};
  std::atomic<int>           not_bf16{0};
  std::uint64_t              fail_at = UINT64_MAX;
  std::uint64_t              throw_at = UINT64_MAX;
  int                        delay_us = 300;
  std::mutex                 mu;
  std::vector<std::string>   log;
  std::set<const void*>      handed;     // every base exchanged
  std::set<const void*>      forgotten;
  bool exchange(const void* partial, void* others, std::size_t bytes,
                TpDType dtype, std::uint64_t seq, std::string* err) override
  {
    if (seq != next.fetch_add(1)) { out_of_step.fetch_add(1); }
    if (dtype != TpDType::kBF16) { not_bf16.fetch_add(1); }
    if (seq == throw_at) { throw std::runtime_error("a socket gave up"); }
    if (seq == fail_at) {
      if (err != nullptr) { *err = "the peer went away"; }
      return false;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    std::memcpy(others, partial, bytes);
    std::lock_guard<std::mutex> lk(mu);
    handed.insert(partial);
    handed.insert(others);
    log.push_back("x" + std::to_string(seq));
    return true;
  }
  void forget(const void* base, std::size_t) override
  {
    std::lock_guard<std::mutex> lk(mu);
    forgotten.insert(base);
    log.push_back("f");
  }
};

// A transport that cannot name itself: the registry must not take it.
struct NamelessTransport final : TpTransport {
  std::string_view name() const override
  {
    throw std::runtime_error("no name");
  }
  std::shared_ptr<TpChannel> open(const FlexData&, const TpShare&,
                                  std::string*) override
  {
    return nullptr;
  }
};

struct Rig {
  Session                sess;
  MetalCompute*          mc = nullptr;
  ComputeLibrary         lib;
  ComputeFunction        add;
  bool ok()
  {
    mc = sess.metal_compute();
    if (mc == nullptr || !mc->valid()) { return false; }
    lib = mc->load_library("llm_elementwise_bf16");
    add = lib.function("residual_add_f16");
    return add.valid();
  }
  // y = x + sub, n elements.
  void residual(ComputeEncoder& enc, const SharedBuffer& x,
                const SharedBuffer& sub, const SharedBuffer& y, int n)
  {
    enc.set_function(add);
    enc.set_buffer(0, x);
    enc.set_buffer(1, sub);
    enc.set_buffer(2, y);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  }
};

}  // namespace

TEST(tensor_parallel, a_share_is_even_unless_given)
{
  TpShare s;
  std::string err;
  FlexData fd = FlexData::from_json(R"({"rank": 1, "ranks": 2})");
  EXPECT_TRUE(TpShare::from_flex(fd, 56, 14336, &s, &err));
  EXPECT_TRUE(s.head0 == 28 && s.head1 == 56);
  EXPECT_TRUE(s.ffn0 == 7168 && s.ffn1 == 14336);
  FlexData given = FlexData::from_json(
      R"({"rank": 1, "ranks": 2, "heads": [44, 56], "ffn": [11264, 14336]})");
  EXPECT_TRUE(TpShare::from_flex(given, 56, 14336, &s, &err));
  EXPECT_TRUE(s.heads() == 12 && s.ffn() == 3072 && s.active());
  FlexData past =
      FlexData::from_json(R"({"rank": 0, "ranks": 2, "heads": [0, 57]})");
  EXPECT_FALSE(TpShare::from_flex(past, 56, 14336, &s, &err));
  EXPECT_TRUE(!err.empty());
}

// A RANK'S BANDS: an eighth of the sequence, and none but the last under
// 1,024 rows -- or under the share's own floor (tp_extra::kMinBandRows,
// both ranks alike), which a host states for a link that carries a
// band's partial in well under its compute: an 832 x 480 x 73 clip's
// 8,843 rows in four bands of 2,304, not eight of 1,152; a sequence whose
// eighths are taller keeps them. Where an eighth is shorter than the
// floor, balancing bands of the floor came out UNDER it (4,700 rows: three
// of 1,664), so the rank takes fewer, each at least the floor.
TEST(tensor_parallel, a_ranks_bands_take_the_floor_it_is_given)
{
  if (std::getenv("VPIPE_H3_TP_BANDS") != nullptr) { return; }
  using genai::MetalMiniMaxH3Transformer;
  MetalMiniMaxH3Transformer::Config c;
  std::string err;
  TpShare s;
  ASSERT_TRUE(TpShare::from_flex(
      FlexData::from_json(R"({"rank": 0, "ranks": 2})"), 56, 14336, &s,
      &err));
  c.tp = s;
  EXPECT_TRUE(MetalMiniMaxH3Transformer::band_rows(c, 8843) == 1152);
  // Three of 768 before the floor held.
  EXPECT_TRUE(MetalMiniMaxH3Transformer::band_rows(c, 2100) == 1152);
  // Under two floors: whole blocks.
  EXPECT_TRUE(MetalMiniMaxH3Transformer::band_rows(c, 2000) == 0);
  ASSERT_TRUE(TpShare::from_flex(
      FlexData::from_json(
          R"({"rank": 0, "ranks": 2, "min_band_rows": 2304})"),
      56, 14336, &s, &err));
  c.tp = s;
  EXPECT_TRUE(MetalMiniMaxH3Transformer::band_rows(c, 8843) == 2304);
  EXPECT_TRUE(MetalMiniMaxH3Transformer::band_rows(c, 20480) == 2560);
  EXPECT_TRUE(MetalMiniMaxH3Transformer::band_rows(c, 4700) == 2432);
  // Every band but the last at least the floor, at every length that
  // holds two (and one band only where the floor was given back: none).
  int under = 0;
  for (int seq = 4608; seq <= 40000; seq += 37) {
    const int b = MetalMiniMaxH3Transformer::band_rows(c, seq);
    under += b < 2304;
  }
  EXPECT_TRUE(under == 0);
  // The key a host states it by.
  EXPECT_TRUE(s.extra.is_object() &&
              s.extra.as_object().contains(tp_extra::kMinBandRows));
  // A floor that is not one.
  EXPECT_FALSE(TpShare::from_flex(
      FlexData::from_json(R"({"rank": 0, "ranks": 2, "min_band_rows": 7})"),
      56, 14336, &s, &err));
}

// THE HANDOFF, PIPELINED AS A SPLIT DiT RUNS IT: each step writes a
// partial on the GPU and signals it, the worker hands it to the channel,
// and the GPU adds the answer behind a wait -- step k's add encoded
// after step k + 1's write, two slots in turn, the encoder never
// waiting. Every sum is the partial doubled only if each signal follows
// its write, each wait precedes its read, and a slot's next write waits
// for its last read.
TEST(tensor_parallel, partials_cross_while_the_gpu_works_on)
{
  Rig r;
  if (!r.ok()) { return; }
  auto ch = std::make_shared<EchoChannel>();
  TpOverlap ov(r.mc, ch, 2);
  EXPECT_TRUE(ov.valid());
  const int n = 1 << 20, steps = 24;
  const std::size_t bytes = (std::size_t)n * 2;
  std::vector<SharedBuffer> src, out;
  SharedBuffer zero = r.mc->make_shared_buffer(bytes);
  std::memset(zero.contents(), 0, bytes);
  for (int k = 0; k < steps; ++k) {
    src.push_back(r.mc->make_shared_buffer(bytes));
    out.push_back(r.mc->make_shared_buffer(bytes));
    auto* p = static_cast<std::uint16_t*>(src.back().contents());
    for (int i = 0; i < n; ++i) { p[i] = bf16_((float)(1 + (k + i) % 100)); }
  }
  SharedBuffer part[2] = {r.mc->make_shared_buffer(bytes),
                          r.mc->make_shared_buffer(bytes)};
  SharedBuffer recv[2] = {r.mc->make_shared_buffer(bytes),
                          r.mc->make_shared_buffer(bytes)};
  CommandStream st = r.mc->make_command_stream();
  ComputeEncoder enc = st.begin_compute();
  std::vector<std::uint64_t> v((std::size_t)steps);
  std::uint64_t seq = 0;
  auto write = [&](int k) {
    // A long write ahead of the partial's, so the worker is there first.
    for (int w = 0; w < 4; ++w) { r.residual(enc, zero, src[k], out[k], n); }
    r.residual(enc, zero, src[k], part[k & 1], n);
    const TpOverlap::Ticket t =
        ov.post(k & 1, part[k & 1], recv[k & 1], bytes, seq++);
    v[(std::size_t)k] = t.done_at;
    enc.end();
    st.encode_signal(*t.ready, t.ready_at);
    enc = st.begin_compute();
  };
  auto add = [&](int k) {
    enc.end();
    st.encode_wait(ov.done(), v[(std::size_t)k]);
    enc = st.begin_compute();
    r.residual(enc, part[k & 1], recv[k & 1], out[k], n);
  };
  for (int i = 0; i < steps + 1; ++i) {
    if (i < steps) { write(i); }
    if (i >= 1) { add(i - 1); }
  }
  enc.end();
  std::string why;
  EXPECT_TRUE(st.commit().wait_ok(&why));
  EXPECT_FALSE(ov.failed());
  EXPECT_FALSE(ov.pending());
  EXPECT_TRUE(ch->out_of_step.load() == 0);
  int wrong = 0;
  for (int k = 0; k < steps; ++k) {
    const auto* s = static_cast<const std::uint16_t*>(src[k].contents());
    const auto* o = static_cast<const std::uint16_t*>(out[k].contents());
    for (int i = 0; i < n; i += 997) {
      if (f32_(o[i]) != 2.0f * f32_(s[i])) { ++wrong; }
    }
  }
  EXPECT_TRUE(wrong == 0);
  const TpOverlap::Stats ts = ov.take_stats();
  EXPECT_TRUE(ts.exchanges == (std::uint64_t)steps);
  EXPECT_TRUE(ts.bytes == (std::uint64_t)steps * bytes);
  // In line once the GPU has drained, after what was posted.
  std::vector<std::uint16_t> a(64, bf16_(3.0f)), b(64, 0);
  EXPECT_TRUE(ov.exchange_now(a.data(), b.data(), 128, seq++, &why));
  EXPECT_TRUE(f32_(b[63]) == 3.0f);
}

// A link that fails releases every wait: the GPU finishes, wrong, and
// the reason is kept for the forward to fail with.
TEST(tensor_parallel, a_failed_exchange_releases_the_gpu)
{
  Rig r;
  if (!r.ok()) { return; }
  auto ch = std::make_shared<EchoChannel>();
  ch->fail_at = 2;
  TpOverlap ov(r.mc, ch, 1);
  const int n = 4096;
  const std::size_t bytes = (std::size_t)n * 2;
  SharedBuffer p = r.mc->make_shared_buffer(bytes);
  SharedBuffer q = r.mc->make_shared_buffer(bytes);
  CommandStream st = r.mc->make_command_stream();
  for (int k = 0; k < 6; ++k) {
    ComputeEncoder enc = st.begin_compute();
    r.residual(enc, p, p, p, n);
    const TpOverlap::Ticket t = ov.post(0, p, q, bytes, (std::uint64_t)k);
    enc.end();
    st.encode_signal(*t.ready, t.ready_at);
    st.encode_wait(ov.done(), t.done_at);
  }
  std::string why;
  EXPECT_TRUE(st.commit().wait_ok(&why));
  EXPECT_TRUE(ov.failed());
  EXPECT_TRUE(ov.error().find("went away") != std::string::npos);
}

// A forward that stops after posting, its signals never committed: the
// worker would wait on the GPU for ever. Abandoned, nothing is pending
// and a later wait passes at once.
TEST(tensor_parallel, an_abandoned_post_is_let_go)
{
  Rig r;
  if (!r.ok()) { return; }
  auto ch = std::make_shared<EchoChannel>();
  TpOverlap ov(r.mc, ch, 1);
  SharedBuffer p = r.mc->make_shared_buffer(256);
  SharedBuffer q = r.mc->make_shared_buffer(256);
  const std::uint64_t v = ov.post(0, p, q, 256, 0).done_at;
  EXPECT_TRUE(ov.pending());
  ov.abandon("the forward ended early");
  EXPECT_FALSE(ov.pending());
  EXPECT_TRUE(ov.failed());
  EXPECT_TRUE(ov.done().signaled_value() >= v);
}

// A PARTIAL IS SENT ONLY ONCE IT IS WRITTEN, whatever finishes first. The
// first is written at the end of a long chain, the second by one short
// dispatch on buffers of its own, each in a command buffer of its own:
// the GPU may finish the second first. A ready signal standing for every
// partial posted before it would let the first go out unwritten.
TEST(tensor_parallel, a_partial_is_sent_only_once_written)
{
  Rig r;
  if (!r.ok()) { return; }
  auto ch = std::make_shared<EchoChannel>();
  TpOverlap ov(r.mc, ch, 2);
  const int big = 1 << 22, small = 256;
  SharedBuffer zero = r.mc->make_shared_buffer((std::size_t)big * 2);
  SharedBuffer one  = r.mc->make_shared_buffer((std::size_t)big * 2);
  std::memset(zero.contents(), 0, (std::size_t)big * 2);
  auto* o = static_cast<std::uint16_t*>(one.contents());
  for (int i = 0; i < big; ++i) { o[i] = bf16_(1.0f); }
  SharedBuffer acc  = r.mc->make_shared_buffer((std::size_t)big * 2);
  SharedBuffer p0   = r.mc->make_shared_buffer((std::size_t)big * 2);
  SharedBuffer q0   = r.mc->make_shared_buffer((std::size_t)big * 2);
  SharedBuffer p1   = r.mc->make_shared_buffer((std::size_t)small * 2);
  SharedBuffer q1   = r.mc->make_shared_buffer((std::size_t)small * 2);
  SharedBuffer z1   = r.mc->make_shared_buffer((std::size_t)small * 2);
  SharedBuffer o1   = r.mc->make_shared_buffer((std::size_t)small * 2);
  std::memset(z1.contents(), 0, (std::size_t)small * 2);
  std::memcpy(o1.contents(), one.contents(), (std::size_t)small * 2);
  std::memset(p0.contents(), 0, (std::size_t)big * 2);
  CommandStream st = r.mc->make_command_stream();
  // The long chain: acc counts to 32, then p0 = acc.
  {
    ComputeEncoder enc = st.begin_compute();
    r.residual(enc, zero, one, acc, big);
    for (int w = 0; w < 31; ++w) { r.residual(enc, acc, one, acc, big); }
    r.residual(enc, zero, acc, p0, big);
  }
  const TpOverlap::Ticket t0 =
      ov.post(0, p0, q0, (std::size_t)big * 2, 0);
  st.encode_signal(*t0.ready, t0.ready_at);
  CommandStream::Fence f0 = st.commit();
  {
    ComputeEncoder enc = st.begin_compute();
    r.residual(enc, z1, o1, p1, small);
  }
  const TpOverlap::Ticket t1 =
      ov.post(1, p1, q1, (std::size_t)small * 2, 1);
  st.encode_signal(*t1.ready, t1.ready_at);
  st.encode_wait(ov.done(), t1.done_at);
  std::string why;
  EXPECT_TRUE(st.commit().wait_ok(&why));
  EXPECT_TRUE(f0.wait_ok(&why));
  const auto* q = static_cast<const std::uint16_t*>(q0.contents());
  int wrong = 0;
  for (int i = 0; i < big; i += 4099) {
    if (f32_(q[i]) != 32.0f) { ++wrong; }
  }
  EXPECT_TRUE(wrong == 0);
  EXPECT_FALSE(ov.failed());
}

// THE HOST CATCHES WHAT A PLUGIN THROWS. A channel's exchange that throws
// is the run's failure, kept with its reason -- on the worker, where an
// exception would have ended the process, and in line -- and a transport
// whose name throws is never registered.
TEST(tensor_parallel, what_a_plugin_throws_is_its_failure)
{
  EXPECT_FALSE(register_tp_transport(std::make_unique<NamelessTransport>()));
  EXPECT_TRUE(find_tp_transport("") == nullptr);
  Rig r;
  if (!r.ok()) { return; }
  {
    auto ch = std::make_shared<EchoChannel>();
    ch->throw_at = 1;
    TpOverlap ov(r.mc, ch, 1);
    SharedBuffer p = r.mc->make_shared_buffer(256);
    SharedBuffer q = r.mc->make_shared_buffer(256);
    CommandStream st = r.mc->make_command_stream();
    for (int k = 0; k < 3; ++k) {
      const TpOverlap::Ticket t = ov.post(0, p, q, 256, (std::uint64_t)k);
      st.encode_signal(*t.ready, t.ready_at);
      st.encode_wait(ov.done(), t.done_at);
    }
    std::string why;
    EXPECT_TRUE(st.commit().wait_ok(&why));
    EXPECT_TRUE(ov.failed());
    EXPECT_TRUE(ov.error().find("a socket gave up") != std::string::npos);
  }
  {
    auto ch = std::make_shared<EchoChannel>();
    ch->throw_at = 0;
    TpOverlap ov(r.mc, ch, 1);
    std::vector<std::uint16_t> a(64), b(64);
    std::string why;
    EXPECT_FALSE(ov.exchange_now(a.data(), b.data(), 128, 0, &why));
    EXPECT_TRUE(why.find("threw") != std::string::npos);
  }
}

// AN EXCHANGE BUFFER IS FORGOTTEN ONLY ONCE ITS EXCHANGE IS ANSWERED. A
// partial posted and its GPU signal committed, then the buffer forgotten
// at once, on the encoding thread: the channel hears the forget after the
// exchange, never beside it -- a transport may drop a device's
// registration of the buffer then.
TEST(tensor_parallel, a_buffer_is_forgotten_after_its_exchange)
{
  Rig r;
  if (!r.ok()) { return; }
  auto ch = std::make_shared<EchoChannel>();
  ch->delay_us = 50000;
  TpOverlap ov(r.mc, ch, 1);
  SharedBuffer p = r.mc->make_shared_buffer(4096);
  SharedBuffer q = r.mc->make_shared_buffer(4096);
  CommandStream st = r.mc->make_command_stream();
  const TpOverlap::Ticket t = ov.post(0, p, q, 4096, 0);
  st.encode_signal(*t.ready, t.ready_at);
  CommandStream::Fence f = st.commit();
  ov.forget(q);
  std::string why;
  EXPECT_TRUE(f.wait_ok(&why));
  std::lock_guard<std::mutex> lk(ch->mu);
  EXPECT_TRUE(ch->log.size() == 2 && ch->log[0] == "x0" && ch->log[1] == "f");
  EXPECT_TRUE(ch->forgotten.count(q.contents()) == 1);
}

// A RANK'S SCRATCH IS ITS SHARE'S, AND ITS EXCHANGE BUFFERS ARE FORGOTTEN.
// Two blocks of the real checkpoint loaded as rank 0 of 2 -- 11 of 56
// heads, 2,816 of 14,336 feed-forward rows -- over a channel that echoes,
// a sequence long enough to band. What its forward allocates is what
// scratch_bytes says for rank_config(cfg) at its band (the figure a stage
// books), and less than the unsplit model's at the checkpoint's widths;
// every partial went as bf16, and when the model goes every buffer the
// channel was handed is forgotten.
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (the FL2VA transformer).
TEST(tensor_parallel, a_ranks_scratch_is_its_shares)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using T = MetalMiniMaxH3Transformer;
  T::Config cfg;
  std::string err;
  ASSERT_TRUE(T::config_from_json(root, cfg, &err));
  cfg.n_layers = 2;
  FlexData tp = FlexData::from_json(
      R"({"rank": 0, "ranks": 2, "heads": [0, 11], "ffn": [0, 2816]})");
  ASSERT_TRUE(TpShare::from_flex(tp, cfg.n_heads, cfg.ffn, &cfg.tp, &err));
  auto ch = std::make_shared<EchoChannel>();
  ch->delay_us = 0;

  // 9 latent frames of a 24 x 40 latent canvas: 2,160 video rows, past
  // the two bands of 1,024 a rank needs to band.
  h3::PackedLayout L;
  const std::vector<int> tags(8, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, 9, 24, 40, 8, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels, {},
                                        &L));
  std::vector<float> uniq;
  std::vector<int> row_idx;
  h3::build_row_timesteps(L, 0.3125f, 0.5f, 0.75f, &uniq, &row_idx);
  auto bf16_of = [&](std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = bf16_(0.01f); }
    return b;
  };
  const SharedBuffer vb = bf16_of(L.video_indices.size() *
                                  (std::size_t)cfg.video_patch_elems());
  const SharedBuffer ab = bf16_of((std::size_t)L.num_audio_rows *
                                  (std::size_t)cfg.audio_channels);
  const SharedBuffer tb = bf16_of(tags.size() * (std::size_t)cfg.text_dim);

  std::size_t est = 0, act = 0, whole = 0;
  int band = 0;
  {
    auto m = T::load(root, mc, cfg);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    std::string berr;
    ASSERT_TRUE(m->bake_adaln({uniq}, &berr));
    ASSERT_TRUE(m->begin_tp_generation(ch, &berr));
    T::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    step.schedule_index = 0;
    std::string ferr;
    const auto v = m->forward(step, &ferr);
    if (v.empty()) { std::printf("[tensor_parallel] %s\n", ferr.c_str()); }
    ASSERT_TRUE(!v.empty());
    band = m->last_band_rows();
    est = T::scratch_bytes(T::rank_config(cfg), L.seq_len, (int)tags.size(),
                           (int)uniq.size(), false, m->ff_scratch_narrow(),
                           0, band, 0);
    act = m->scratch_resident_bytes();
    whole = T::scratch_bytes(cfg, L.seq_len, (int)tags.size(),
                             (int)uniq.size(), false, m->ff_scratch_narrow(),
                             0, band, 0);
  }
  std::printf("[tensor_parallel] rank of %d rows, band %d: scratch est %zu, "
              "allocated %zu, at the checkpoint's widths %zu\n", L.seq_len,
              band, est, act, whole);
  EXPECT_TRUE(band > 0);
  EXPECT_TRUE(est == act);
  EXPECT_TRUE(act < whole);
  EXPECT_TRUE(ch->out_of_step.load() == 0);
  EXPECT_TRUE(ch->not_bf16.load() == 0);
  std::lock_guard<std::mutex> lk(ch->mu);
  EXPECT_TRUE(!ch->handed.empty());
  int unforgotten = 0;
  for (const void* b : ch->handed) {
    if (ch->forgotten.count(b) == 0) { ++unforgotten; }
  }
  EXPECT_TRUE(unforgotten == 0);
}

namespace {

// TWO RANKS IN ONE PROCESS. Each exchange leaves this rank's partial
// under its `seq` and takes the peer's of the same `seq` -- with two
// ranks, the sum of the others'. Bytes that differ in length under one
// `seq` are ranks out of step; a peer that never comes fails the
// exchange after a minute rather than hanging the suite.
struct PairLink {
  std::mutex              mu;
  std::condition_variable cv;
  struct Entry {
    std::vector<std::uint8_t> part[2];
    bool has[2] = {false, false};
    int  taken = 0;
  };
  std::map<std::uint64_t, Entry> at;
  std::atomic<int>               mismatched{0};
};

struct PairChannel final : TpChannel {
  std::shared_ptr<PairLink>  link;
  int                        me = 0;
  std::atomic<std::uint64_t> next{0};
  std::atomic<int>           out_of_step{0};
  std::atomic<std::uint64_t> exchanges{0};
  PairChannel(std::shared_ptr<PairLink> l, int r) : link(std::move(l)), me(r)
  {
  }
  bool exchange(const void* partial, void* others, std::size_t bytes,
                TpDType, std::uint64_t seq, std::string* err) override
  {
    if (seq != next.fetch_add(1)) { out_of_step.fetch_add(1); }
    const auto* p = static_cast<const std::uint8_t*>(partial);
    std::unique_lock<std::mutex> lk(link->mu);
    PairLink::Entry& e = link->at[seq];
    e.part[me].assign(p, p + bytes);
    e.has[me] = true;
    link->cv.notify_all();
    if (!link->cv.wait_for(lk, std::chrono::seconds(60),
                           [&]() { return e.has[1 - me]; })) {
      if (err != nullptr) { *err = "the peer never came"; }
      return false;
    }
    if (e.part[1 - me].size() != bytes) {
      link->mismatched.fetch_add(1);
      if (err != nullptr) { *err = "the ranks are out of step"; }
      return false;
    }
    std::memcpy(others, e.part[1 - me].data(), bytes);
    exchanges.fetch_add(1);
    if (++e.taken == 2) { link->at.erase(seq); }
    return true;
  }
};

using T = MetalMiniMaxH3Transformer;

// The modulation baked for the one schedule BlockRun's inputs carry: a
// rank holds no AdaLN projection, so its forward reads the bake's
// tables (and the unsplit one is run the same way).
void
baked_step_(T::Step& s)
{
  s.schedule_index = 0;
}

// A model of `cfg`'s two blocks, baked for `run`'s schedule, its GEMMs
// pinned to steel's 32-row tile (no tuner deciding per model); `stream`,
// its blocks streamed (else held from the load: the VPIPE_PRELOAD A/B).
std::unique_ptr<T>
load_two_blocks_(MetalCompute* mc, const BlockRun& run, const T::Config& cfg,
                 const std::vector<T::LoraSpec>& loras = {},
                 bool stream = false)
{
  auto m = T::load(run.root, mc, cfg, stream, loras);
  if (m == nullptr) { return nullptr; }
  m->set_gemm_route(T::GemmRoute::kSteelBm32);
  std::string berr;
  if (!m->bake_adaln({run.uniq}, &berr)) {
    std::printf("  bake: %s\n", berr.c_str());
    return nullptr;
  }
  return m;
}

// The unsplit model's velocity.
bool
whole_forward_(MetalCompute* mc, BlockRun& run, const T::Config& cfg,
               std::vector<float>* out,
               const std::vector<T::LoraSpec>& loras = {})
{
  auto m = load_two_blocks_(mc, run, cfg, loras);
  double ms = 0.0;
  return m != nullptr && run.forward_on(*m, out, &ms, baked_step_);
}

// Two ranks of `cfg` -- heads [0, h) and [h, n), feed-forward rows [0, f)
// and [f, F) -- over one PairLink, their forwards run at once on two
// threads; each rank's velocity back.
struct TwoRanks {
  std::vector<float> v[2];
  int band = 0;
  int out_of_step = 0, mismatched = 0;
  std::uint64_t exchanges = 0;
  // Rank 0's projections in the int8 layout: main blocks', refiner's.
  int main_i8 = 0, refiner_i8 = 0;
};
bool
split_forward_(MetalCompute* mc, BlockRun& run, const T::Config& cfg,
               int h, int f, TwoRanks* out,
               const std::vector<T::LoraSpec>& loras = {})
{
  auto link = std::make_shared<PairLink>();
  std::shared_ptr<PairChannel> ch[2];
  std::unique_ptr<T> m[2];
  for (int r = 0; r < 2; ++r) {
    T::Config c = cfg;
    c.tp = TpShare{};
    c.tp.rank = r;
    c.tp.ranks = 2;
    c.tp.n_heads = cfg.n_heads;
    c.tp.n_ffn = cfg.ffn;
    c.tp.head0 = r == 0 ? 0 : h;
    c.tp.head1 = r == 0 ? h : cfg.n_heads;
    c.tp.ffn0 = r == 0 ? 0 : f;
    c.tp.ffn1 = r == 0 ? f : cfg.ffn;
    m[r] = load_two_blocks_(mc, run, c, loras);
    if (m[r] == nullptr) { return false; }
    ch[r] = std::make_shared<PairChannel>(link, r);
    std::string err;
    if (!m[r]->begin_tp_generation(ch[r], &err)) {
      std::printf("  begin: %s\n", err.c_str());
      return false;
    }
  }
  bool ok1 = false;
  std::thread t([&]() {
    double ms = 0.0;
    ok1 = run.forward_on(*m[1], &out->v[1], &ms, baked_step_);
  });
  double ms = 0.0;
  const bool ok0 = run.forward_on(*m[0], &out->v[0], &ms, baked_step_);
  t.join();
  out->band = m[0]->last_band_rows();
  out->main_i8 = m[0]->int8_layouts(false);
  out->refiner_i8 = m[0]->int8_layouts(true);
  out->out_of_step = ch[0]->out_of_step.load() + ch[1]->out_of_step.load();
  out->mismatched = link->mismatched.load();
  out->exchanges = ch[0]->exchanges.load();
  return ok0 && ok1;
}

// Rank `r` of a two-rank split of `cfg`: heads [0, h) and [h, n),
// feed-forward rows [0, f) and [f, F).
T::Config
rank_of_(const T::Config& cfg, int r, int h, int f)
{
  T::Config c = cfg;
  c.tp = TpShare{};
  c.tp.rank = r;
  c.tp.ranks = 2;
  c.tp.n_heads = cfg.n_heads;
  c.tp.n_ffn = cfg.ffn;
  c.tp.head0 = r == 0 ? 0 : h;
  c.tp.head1 = r == 0 ? h : cfg.n_heads;
  c.tp.ffn0 = r == 0 ? 0 : f;
  c.tp.ffn1 = r == 0 ? f : cfg.ffn;
  return c;
}

// Two ranks of `cfg` STREAMED, their forwards run at once: `forwards` of
// them -- and with `keep` (residency may keep blocks; else none is), on
// until a forward has run with every block kept, its admissions being
// capped per forward. Each rank's last velocity, the blocks it had kept
// before that forward, and what it holds after it: weights (held_bytes),
// adapter factors, blocks kept.
struct StreamedRanks {
  std::vector<float> v[2];
  std::size_t held[2] = {}, lora[2] = {};
  int kept[2] = {}, kept_before[2] = {};
  int forwards = 0;
  int out_of_step = 0, mismatched = 0;
};
bool
split_streamed_(MetalCompute* mc, BlockRun& run, const T::Config& cfg,
                int h, int f, bool keep, int forwards, StreamedRanks* out,
                const std::vector<T::LoraSpec>& loras = {})
{
  auto link = std::make_shared<PairLink>();
  std::shared_ptr<PairChannel> ch[2];
  std::unique_ptr<T> m[2];
  for (int r = 0; r < 2; ++r) {
    m[r] = load_two_blocks_(mc, run, rank_of_(cfg, r, h, f), loras, true);
    if (m[r] == nullptr) { return false; }
    if (keep) {
      m[r]->set_residency_reserve(test_residency_reserve(2ull << 30));
    }
    ch[r] = std::make_shared<PairChannel>(link, r);
    std::string err;
    if (!m[r]->begin_tp_generation(ch[r], &err)) {
      std::printf("  begin: %s\n", err.c_str());
      return false;
    }
  }
  const int n = run.cfg.n_layers;
  for (out->forwards = 0; out->forwards < 8;) {
    for (int r = 0; r < 2; ++r) {
      out->kept_before[r] = m[r]->resident_block_count();
    }
    bool ok1 = false;
    std::thread t([&]() {
      double ms = 0.0;
      ok1 = run.forward_on(*m[1], &out->v[1], &ms, baked_step_);
    });
    double ms = 0.0;
    const bool ok0 = run.forward_on(*m[0], &out->v[0], &ms, baked_step_);
    t.join();
    if (!ok0 || !ok1) { return false; }
    ++out->forwards;
    const bool all =
        out->kept_before[0] == n && out->kept_before[1] == n;
    if (out->forwards >= forwards && (!keep || all)) { break; }
  }
  for (int r = 0; r < 2; ++r) {
    out->held[r] = m[r]->held_bytes();
    out->lora[r] = m[r]->lora_bytes();
    out->kept[r] = m[r]->resident_block_count();
  }
  out->out_of_step = ch[0]->out_of_step.load() + ch[1]->out_of_step.load();
  out->mismatched = link->mismatched.load();
  return true;
}

bool
same_bits_(const std::vector<float>& a, const std::vector<float>& b)
{
  return a.size() == b.size() && !a.empty() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// The environment a forward reads, set for one scope.
struct EnvScope {
  std::string k;
  EnvScope(const char* key, const char* v) : k(key)
  {
    ::setenv(key, v, 1);
  }
  ~EnvScope() { ::unsetenv(k.c_str()); }
};

}  // namespace

// A SPLIT FORWARD IS THE UNSPLIT ONE. Two ranks of two real blocks --
// 11 of 56 heads and 2,816 of 14,336 feed-forward rows on rank 0, the
// rest on rank 1, the cut a real two-Mac deployment takes -- exchange
// their partial sums over an in-process link, and each rank's velocity
// is held to the unsplit model's. Arms, each against its own unsplit
// forward:
//   - per-head q|k|v (the release's), banded: 2,184 rows in bands of 768,
//     each band's exchanges overlapped (VPIPE_H3_TP_LAG 1) -- and run in
//     turn (lag 0), the same arithmetic BIT FOR BIT; and whole
//     (VPIPE_H3_TP_BANDS=1: every block's partial exchanged in line);
//   - FLAT q|k|v (ComfyUI's grouping; Config::qkv_per_head false over
//     the same bytes, so the unsplit model reads the same layout): its
//     cut is [q | k | v] each at the share's heads;
//   - an AFFINE pack (VPIPE_MINIMAX_H3_QUANT_DIT, a model-quantize'd
//     DiT): codes, scales and biases cut by rows, and by whole groups of
//     columns for the out projection and fc2.
// The two ranks' velocities are IDENTICAL (two ranks add a + b and
// b + a); against the unsplit model the split differs by the bf16
// rounding of two partial sums in place of one -- measured below.
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (FL2VA), optionally
// VPIPE_MINIMAX_H3_QUANT_DIT.
TEST(tensor_parallel, a_split_forward_is_the_unsplit_one)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  // The bar: two bf16 roundings of a partial sum where the unsplit model
  // rounds once, twice a block, through two blocks and the final layer.
  // MEASURED on the M4 Pro (printed): 4.4e-3 (the pack) .. 5.7e-3 (flat
  // q|k|v). MUTATIONS, each failing it: gate | up halves swapped
  // 0.39-0.47 (every arm), a flat q|k|v cut by the per-head rule 0.20
  // (flat only), an affine pack's biases cut from the share's first
  // column 0.43 (the pack only), a band adding the other band's sum
  // 0.016-0.33 and the ranks parting (banded arms only).
  constexpr double kBar = 1e-2;

  auto arm = [&](const char* name, const std::string& dir, bool per_head,
                 bool banded_expected, bool lag0_too) {
    BlockRun run;
    ASSERT_TRUE(run.setup(mc, dir, 2, 9, 24, 40, 8));
    run.cfg.autotune = false;
    run.cfg.qkv_per_head = per_head && run.cfg.qkv_per_head;
    std::vector<float> whole;
    ASSERT_TRUE(whole_forward_(mc, run, run.cfg, &whole));
    TwoRanks s;
    ASSERT_TRUE(split_forward_(mc, run, run.cfg, 11, 2816, &s));
    const double d = rel_l2_(s.v[0], whole);
    std::printf("  %-26s %d rows, band %d: split vs unsplit rel-L2 %.3e, "
                "ranks %s, %llu exchanges\n", name, run.L.seq_len, s.band,
                d, same_bits_(s.v[0], s.v[1]) ? "IDENTICAL" : "DIFFER",
                (unsigned long long)s.exchanges);
    EXPECT_TRUE(s.out_of_step == 0 && s.mismatched == 0);
    EXPECT_TRUE(same_bits_(s.v[0], s.v[1]));
    EXPECT_TRUE(d < kBar);
    EXPECT_TRUE((s.band > 0) == banded_expected);
    if (lag0_too) {
      EnvScope lag("VPIPE_H3_TP_LAG", "0");
      TwoRanks s0;
      ASSERT_TRUE(split_forward_(mc, run, run.cfg, 11, 2816, &s0));
      std::printf("  %-26s band after band: %s\n", name,
                  same_bits_(s0.v[0], s.v[0]) ? "IDENTICAL to overlapped"
                                              : "DIFFERS from overlapped");
      EXPECT_TRUE(same_bits_(s0.v[0], s.v[0]));
      EXPECT_TRUE(same_bits_(s0.v[0], s0.v[1]));
    }
  };
  arm("bf16, per-head q|k|v", root, true, true, true);
  {
    EnvScope whole("VPIPE_H3_TP_BANDS", "1");
    arm("bf16, unbanded", root, true, false, false);
  }
  arm("bf16, flat q|k|v", root, false, true, false);
  const std::string pack = env_("VPIPE_MINIMAX_H3_QUANT_DIT");
  if (!pack.empty()) {
    arm("affine pack", pack, true, true, false);
  } else {
    std::printf("  (VPIPE_MINIMAX_H3_QUANT_DIT unset: no affine arm)\n");
  }
}

// A SPLIT FORWARD UNDER AN ADAPTER AND AN INT8 TIER is the unsplit one
// under them. Each rank binds the adapter at the checkpoint's widths and
// cuts its factors as the weights are cut (tp_cut_lora_): B's rows of
// q|k|v and fc1, A's columns of the out projection and fc2; and holds
// its share in its tier's int8 layout, quantized within the share's own
// columns (tp_fast_). Arms, each against the unsplit model under the
// same adapter and tier:
//   - the Turbo LoRA (VPIPE_MINIMAX_H3_TURBO_LORA) at 1.0, bf16 -- the
//     bar the plain split's, two partial sums' rounding;
//   - the adapter's strength at -0.5 (a delta the ranks must agree on in
//     sign too);
//   - int8 FAST (rotated, groups of 512) and PER TOKEN (rotated by 256)
//     at 12 heads (whole groups on both ranks), and Per Token at 11
//     heads, where the out projection's share is 1,408 wide -- no
//     rotated group divides it, so it stays on the accurate tier there.
//     An int8 tier's activations are quantized over a rank's columns
//     where the unsplit model quantizes over all of them, so the bar is
//     the tier's own: the split within half again of what the tier is
//     from bf16 (whole against whole), and the plain split's bar at
//     least;
//   - the Turbo LoRA under Per Token.
// The two ranks' velocities are IDENTICAL in every arm, and a rank holds
// the tier's layout on its main blocks only: its refiner's share stays
// as cut, as the unsplit model's refiner does.
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (FL2VA), VPIPE_MINIMAX_H3_TURBO_LORA
// (an FL2VA adapter; its arms skipped without it).
TEST(tensor_parallel, a_split_forward_takes_adapters_and_int8_tiers)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  constexpr double kBar = 1e-2;
  constexpr double kDeltaBar = 0.15;
  BlockRun run;
  ASSERT_TRUE(run.setup(mc, root, 2, 9, 24, 40, 8));
  run.cfg.autotune = false;
  std::vector<float> bf16_whole;
  ASSERT_TRUE(whole_forward_(mc, run, run.cfg, &bf16_whole));

  // The plain split and whole forwards at a tier, kept for the adapter
  // arms' deltas.
  std::map<std::pair<int, int>, std::pair<std::vector<float>,
                                          std::vector<float>>> plain;
  auto plain_of = [&](int tier, int heads) {
    auto it = plain.find({tier, heads});
    if (it != plain.end()) { return it->second; }
    T::Config cfg = run.cfg;
    cfg.i8_gemm = tier > 0;
    cfg.i8_tier = tier;
    std::vector<float> w;
    TwoRanks s;
    const bool ok =
        whole_forward_(mc, run, cfg, &w) &&
        split_forward_(mc, run, cfg, heads, cfg.ffn / cfg.n_heads * heads,
                       &s);
    if (!ok) { w.clear(); }
    return plain[{tier, heads}] = {w, s.v[0]};
  };
  auto minus = [](const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> d(a.size());
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
      d[i] = a[i] - b[i];
    }
    return d;
  };
  auto arm = [&](const char* name, int tier, int heads,
                 const std::vector<T::LoraSpec>& loras) {
    T::Config cfg = run.cfg;
    cfg.i8_gemm = tier > 0;
    cfg.i8_tier = tier;
    std::vector<float> whole;
    ASSERT_TRUE(whole_forward_(mc, run, cfg, &whole, loras));
    TwoRanks s;
    const int f = cfg.ffn / cfg.n_heads * heads;
    ASSERT_TRUE(split_forward_(mc, run, cfg, heads, f, &s, loras));
    const double d = rel_l2_(s.v[0], whole);
    // What the tier (and the adapter) is from the plain bf16 model.
    const double own = rel_l2_(whole, bf16_whole);
    const double bar = tier > 0 ? std::max(kBar, 1.5 * own) : kBar;
    std::printf("  %-30s %d heads: split vs unsplit rel-L2 %.3e (bar "
                "%.3e; unsplit from plain bf16 %.3e), ranks %s; int8 "
                "layouts %d main, %d refiner\n", name, heads, d, bar, own,
                same_bits_(s.v[0], s.v[1]) ? "IDENTICAL" : "DIFFER",
                s.main_i8, s.refiner_i8);
    EXPECT_TRUE(s.out_of_step == 0 && s.mismatched == 0);
    // The tier's layout on the main blocks, never on the refiner: its
    // text rows run under the int8 row floor, where a fast weight is
    // only widened back -- and the unsplit model keeps it dense.
    if (tier > 0 && mc->supports_matrix_cores()) {
      EXPECT_TRUE(s.main_i8 > 0);
    }
    EXPECT_TRUE(s.refiner_i8 == 0);
    EXPECT_TRUE(same_bits_(s.v[0], s.v[1]));
    EXPECT_TRUE(d < bar);
    if (loras.empty() || std::fabs(loras[0].scale) < 4.0f) { return; }
    // THE ADAPTER'S DELTA, split against whole, at a strength where it
    // stands above a split's rounding (at 1.0 that rounding is a third of
    // it, two blocks deep): what it adds to the split model held to what
    // it adds to the unsplit one. MEASURED 5.1e-2 at +-8; fc1's B cut to
    // the gate rows twice 0.30 (and the velocity 3.8e-2). A factor's cut
    // is held exactly by a_ranks_adapter_factors_are_its_share.
    const auto& [pw, ps] = plain_of(tier, heads);
    ASSERT_TRUE(!pw.empty() && !ps.empty());
    const double dd = rel_l2_(minus(s.v[0], ps), minus(whole, pw));
    std::printf("  %-30s its delta, split vs unsplit: rel-L2 %.3e\n", "",
                dd);
    EXPECT_TRUE(dd < kDeltaBar);
  };
  const std::string lora = env_("VPIPE_MINIMAX_H3_TURBO_LORA");
  if (!lora.empty()) {
    arm("Turbo LoRA 1.0", 0, 11, {{lora, 1.0f}});
    arm("Turbo LoRA -0.5", 0, 11, {{lora, -0.5f}});
    arm("Turbo LoRA 8.0", 0, 11, {{lora, 8.0f}});
    arm("Turbo LoRA -8.0", 0, 11, {{lora, -8.0f}});
  } else {
    std::printf("  (VPIPE_MINIMAX_H3_TURBO_LORA unset: no adapter arms)\n");
  }
  arm("int8 fast", 1, 12, {});
  arm("int8 per token", 2, 12, {});
  arm("int8 per token, out accurate", 2, 11, {});
  if (!lora.empty()) {
    arm("Turbo LoRA under per token", 2, 12, {{lora, 1.0f}});
  }
  // AN AFFINE PACK under Per Token: a rank's cut w8 share takes the
  // tier's layout as the unsplit model's whole one does (tp_fast_affine_)
  // -- left as cut, a compressed checkpoint's rank ran every projection
  // dequantized at bf16's GEMM, ~100 TF/s on the M5 Ultra against ~160.
  const std::string pack = env_("VPIPE_MINIMAX_H3_QUANT_DIT");
  if (pack.empty()) {
    std::printf("  (VPIPE_MINIMAX_H3_QUANT_DIT unset: no affine arm)\n");
    return;
  }
  BlockRun prun;
  ASSERT_TRUE(prun.setup(mc, pack, 2, 9, 24, 40, 8));
  prun.cfg.autotune = false;
  T::Config pc = prun.cfg;
  pc.i8_gemm = true;
  pc.i8_tier = 2;
  std::vector<float> pw, pb;
  ASSERT_TRUE(whole_forward_(mc, prun, prun.cfg, &pb));
  ASSERT_TRUE(whole_forward_(mc, prun, pc, &pw));
  TwoRanks ps;
  ASSERT_TRUE(split_forward_(mc, prun, pc, 12, pc.ffn / pc.n_heads * 12,
                             &ps));
  const double pd = rel_l2_(ps.v[0], pw);
  const double pown = rel_l2_(pw, pb);
  const double pbar = std::max(kBar, 1.5 * pown);
  std::printf("  %-30s 12 heads: split vs unsplit rel-L2 %.3e (bar %.3e; "
              "unsplit from the pack's plain %.3e), ranks %s; int8 "
              "layouts %d main, %d refiner\n", "affine pack, per token", pd,
              pbar, pown, same_bits_(ps.v[0], ps.v[1]) ? "IDENTICAL"
                                                       : "DIFFER",
              ps.main_i8, ps.refiner_i8);
  EXPECT_TRUE(ps.out_of_step == 0 && ps.mismatched == 0);
  if (mc->supports_matrix_cores()) { EXPECT_TRUE(ps.main_i8 > 0); }
  EXPECT_TRUE(ps.refiner_i8 == 0);
  EXPECT_TRUE(same_bits_(ps.v[0], ps.v[1]));
  EXPECT_TRUE(pd < pbar);
}

// A RANK'S ADAPTER FACTORS ARE ITS SHARE OF THE WHOLE ONES, exactly.
// Each rank binds the adapter at the checkpoint's widths and cuts it
// (tp_cut_lora_); for a row of input the four projections' deltas,
// worked on the host from the bound factors (lora_delta_for_test), are
// held to the unsplit model's: q|k|v's and fc1's, the whole delta's rows
// of the rank's heads and feed-forward rows; the out projection's and
// fc2's, from the rank's columns of the input, summing over the two
// ranks to the whole delta. The forward arms above cannot see a factor
// cut from the wrong columns -- two blocks of a strength-1 adapter move
// the velocity less than a split's rounding does; this sees every
// factor. Arms: the release's per-head q|k|v and the flat grouping
// (a banded q|k|v B regrouped). Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH
// (FL2VA), VPIPE_MINIMAX_H3_TURBO_LORA.
TEST(tensor_parallel, a_ranks_adapter_factors_are_its_share)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const std::string lora = env_("VPIPE_MINIMAX_H3_TURBO_LORA");
  if (root.empty() || lora.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH / "
                "VPIPE_MINIMAX_H3_TURBO_LORA unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  for (const bool per_head : {true, false}) {
    T::Config cfg;
    std::string cerr;
    ASSERT_TRUE(T::config_from_json(root, cfg, &cerr));
    cfg.n_layers = 2;
    cfg.autotune = false;
    cfg.qkv_per_head = per_head && cfg.qkv_per_head;
    const std::vector<T::LoraSpec> loras = {{lora, 1.0f}};
    auto whole = T::load(root, mc, cfg, false, loras);
    ASSERT_TRUE(whole != nullptr);
    if (whole == nullptr) { return; }
    const int HD = cfg.head_dim, NH = cfg.n_heads, F = cfg.ffn;
    const int I = NH * HD, H = cfg.hidden;
    const int h = 11, f = F / NH * h;
    std::unique_ptr<T> rank[2];
    for (int r = 0; r < 2; ++r) {
      T::Config c = cfg;
      c.tp.rank = r;
      c.tp.ranks = 2;
      c.tp.n_heads = NH;
      c.tp.n_ffn = F;
      c.tp.head0 = r == 0 ? 0 : h;
      c.tp.head1 = r == 0 ? h : NH;
      c.tp.ffn0 = r == 0 ? 0 : f;
      c.tp.ffn1 = r == 0 ? f : F;
      rank[r] = T::load(root, mc, c, false, loras);
      ASSERT_TRUE(rank[r] != nullptr);
      if (rank[r] == nullptr) { return; }
    }
    // The rows of the whole output a rank's share holds, in its order.
    auto rows_of = [&](int which, int r) {
      std::vector<int> out;
      const int a = r == 0 ? 0 : h, b = r == 0 ? h : NH;
      const int fa = r == 0 ? 0 : f, fb = r == 0 ? f : F;
      if (which == 0) {
        if (cfg.qkv_per_head) {
          for (int i = a * 3 * HD; i < b * 3 * HD; ++i) { out.push_back(i); }
        } else {
          for (int part = 0; part < 3; ++part) {
            for (int i = a * HD; i < b * HD; ++i) {
              out.push_back(part * I + i);
            }
          }
        }
      } else {
        for (int half = 0; half < 2; ++half) {
          for (int i = fa; i < fb; ++i) { out.push_back(half * F + i); }
        }
      }
      return out;
    };
    std::uint32_t seed = 12345;
    auto rnd = [&]() {
      seed = seed * 1664525u + 1013904223u;
      return (float)((seed >> 8) & 0xffff) / 32768.0f - 1.0f;
    };
    double worst = 0.0;
    for (int blk = 0; blk < 2; ++blk) {
      for (int which = 0; which < 4; ++which) {
        const bool cols = which == 1 || which == 3;
        const int K = which == 1 ? I : which == 3 ? F : H;
        std::vector<float> x((std::size_t)K);
        for (float& v : x) { v = rnd(); }
        const auto y = whole->lora_delta_for_test(0, blk, which, x);
        ASSERT_TRUE(!y.empty());
        double err = 0.0, norm = 0.0;
        if (cols) {
          std::vector<double> sum(y.size(), 0.0);
          for (int r = 0; r < 2; ++r) {
            const int k0 = which == 1 ? (r == 0 ? 0 : h * HD)
                                      : (r == 0 ? 0 : f);
            const int k1 = which == 1 ? (r == 0 ? h * HD : I)
                                      : (r == 0 ? f : F);
            const std::vector<float> xr(x.begin() + k0, x.begin() + k1);
            const auto yr = rank[r]->lora_delta_for_test(0, blk, which, xr);
            ASSERT_TRUE(yr.size() == y.size());
            for (std::size_t i = 0; i < yr.size(); ++i) { sum[i] += yr[i]; }
          }
          for (std::size_t i = 0; i < y.size(); ++i) {
            err += (sum[i] - y[i]) * (sum[i] - y[i]);
            norm += (double)y[i] * y[i];
          }
        } else {
          for (int r = 0; r < 2; ++r) {
            const auto yr = rank[r]->lora_delta_for_test(0, blk, which, x);
            const auto at = rows_of(which, r);
            ASSERT_TRUE(yr.size() == at.size());
            for (std::size_t i = 0; i < yr.size(); ++i) {
              const double d = (double)yr[i] - y[(std::size_t)at[i]];
              err += d * d;
              norm += (double)y[(std::size_t)at[i]] * y[(std::size_t)at[i]];
            }
          }
        }
        ASSERT_TRUE(norm > 0.0);
        worst = std::max(worst, std::sqrt(err / norm));
      }
    }
    std::printf("  %s q|k|v: the ranks' deltas from the whole one's, worst "
                "rel-L2 %.3e\n", per_head ? "per-head" : "flat", worst);
    EXPECT_TRUE(worst < 1e-5);
  }
}

// A MODEL KEPT WARM KEEPS ITS ADALN TABLES for the schedule they were
// baked for: baked again for the same timesteps (the next clip at the
// same steps and shift) nothing is read -- a split rank read its 25 GB
// of AdaLN projections each clip -- and its forward is the first one's to
// the bit; baked for another schedule, then this one again, the tables
// are made anew and the forward is still the first one's to the bit.
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (FL2VA).
TEST(tensor_parallel, a_kept_models_tables_stay_for_their_schedule)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  BlockRun run;
  ASSERT_TRUE(run.setup(mc, root, 2, 9, 24, 40, 8));
  run.cfg.autotune = false;
  auto m = load_two_blocks_(mc, run, run.cfg);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  std::vector<float> v1, v2, v3;
  double ms = 0.0;
  ASSERT_TRUE(run.forward_on(*m, &v1, &ms, baked_step_));
  std::string err;
  const auto t0 = std::chrono::steady_clock::now();
  ASSERT_TRUE(m->bake_adaln({run.uniq}, &err));
  const double kept_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
  ASSERT_TRUE(run.forward_on(*m, &v2, &ms, baked_step_));
  std::vector<float> other = run.uniq;
  for (float& t : other) { t *= 0.5f; }
  ASSERT_TRUE(m->bake_adaln({other}, &err));
  const auto t1 = std::chrono::steady_clock::now();
  ASSERT_TRUE(m->bake_adaln({run.uniq}, &err));
  const double rebake_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t1)
                               .count();
  ASSERT_TRUE(run.forward_on(*m, &v3, &ms, baked_step_));
  std::printf("  the same schedule again: %.2f ms (kept), after another "
              "%.2f ms (baked anew); forwards %s, %s\n", kept_ms, rebake_ms,
              same_bits_(v1, v2) ? "IDENTICAL" : "DIFFER",
              same_bits_(v1, v3) ? "IDENTICAL" : "DIFFER");
  EXPECT_TRUE(same_bits_(v1, v2));
  EXPECT_TRUE(same_bits_(v1, v3));
  EXPECT_TRUE(kept_ms * 10.0 < rebake_ms);
  // Kept warm, its forward's SCRATCH goes (release_forward_scratch): none
  // left, and the next forward builds it again to the same result.
  const std::size_t freed = m->release_forward_scratch();
  std::vector<float> v4;
  const std::size_t left =
      m->scratch_resident_bytes() + m->dequant_scratch_bytes();
  ASSERT_TRUE(run.forward_on(*m, &v4, &ms, baked_step_));
  std::printf("  its scratch let go: %zu MB, %zu left; the next forward %s\n",
              freed >> 20, left, same_bits_(v1, v4) ? "IDENTICAL" : "DIFFERS");
  EXPECT_TRUE(freed > 0 && left == 0);
  EXPECT_TRUE(same_bits_(v1, v4));
}

// A RANK HOLDS WHAT THE PLAN BOOKS. The plan's figure for a rank
// (MetalMiniMaxH3Transformer::tp_held_bytes, off the checkpoint's headers
// before anything loads: the trunk and norms as stored, each projection's
// share in the layout the load leaves it, the bake's tables) against what
// the loaded rank holds (held_bytes: its weight set's entries, its cut
// projections, its tables) -- and its adapter's footprint at the rank's
// cut (lora_footprint with the share) against the factors it binds. Each
// tier, both ranks of an 11-of-56-heads split, two real blocks and the
// refiner. The int8 tiers engage on matrix cores alone; elsewhere the
// share stays bf16, and the plan's figure follows the same context. Before,
// the plan booked the bf16 share at every tier and the adapter's factors
// at the checkpoint's width, and the final layer's AdaLN projection --
// which the bake's reset could not free, a cached entry -- was held for
// the run. An AFFINE PACK's share takes its tier's layout on a main
// block (tp_fast_affine_), a w4 one only where Config::i8_hold_w4 holds
// it -- so the pack runs with the hold off and on, and what the hold
// adds (the GROWTH generate-video weighs a rank's hold by, and books)
// is the plan's difference as much as the load's: more than nothing
// exactly where a main-block projection is w4 and the layout engages.
// The layout engages on matrix cores alone, so on any other GPU the hold
// arm checks only that it changes nothing; the w4 branches need an M5
// and a w4 or mixed pack.
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (FL2VA), optionally
// VPIPE_MINIMAX_H3_TURBO_LORA, VPIPE_MINIMAX_H3_QUANT_DIT (an affine
// pack's DiT).
TEST(tensor_parallel, a_rank_holds_what_the_plan_books)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string lora = env_("VPIPE_MINIMAX_H3_TURBO_LORA");
  auto arm = [&](const char* name, const std::string& dir,
                 std::initializer_list<int> tiers,
                 const std::vector<T::LoraSpec>& loras, bool holds) {
    BlockRun run;
    ASSERT_TRUE(run.setup(mc, dir, 2, 9, 24, 40, 8));
    run.cfg.autotune = false;
    const int NH = run.cfg.n_heads, F = run.cfg.ffn;
    const int h = 11, f = F / NH * h;
    // The main blocks' w4 projections (codes K/8 words a row), which
    // alone a hold changes.
    int w4 = 0;
    if (holds) {
      const auto w = MetalLlamaWeights::open_model(
          T::resolve_dit_dir(dir, run.cfg.partition));
      for (int b = 0; w && b < run.cfg.n_layers; ++b) {
        for (const char* p : {"attn.qkv_proj", "attn.out_proj", "mlp.fc1",
                              "mlp.fc2"}) {
          const std::string nm =
              "blocks." + std::to_string(b) + "." + p;
          const auto* ci = w->info(nm + ".weight");
          const auto* si = w->info(nm + ".scales");
          if (ci != nullptr && si != nullptr && ci->shape.size() == 2 &&
              si->shape.size() == 2 &&
              ci->shape[1] * 32 == si->shape[1] * 64 * 4) {
            ++w4;
          }
        }
      }
    }
    for (const int tier : tiers) {
      std::size_t booked_off[2] = {}, held_off[2] = {};
      for (const bool hold : {false, true}) {
        if (hold && (!holds || tier == 0)) { continue; }
        for (int r = 0; r < 2; ++r) {
          T::Config c = run.cfg;
          c.i8_gemm = tier > 0;
          c.i8_tier = tier;
          c.i8_hold_w4 = hold;
          c.tp = TpShare{};
          c.tp.rank = r;
          c.tp.ranks = 2;
          c.tp.n_heads = NH;
          c.tp.n_ffn = F;
          c.tp.head0 = r == 0 ? 0 : h;
          c.tp.head1 = r == 0 ? h : NH;
          c.tp.ffn0 = r == 0 ? 0 : f;
          c.tp.ffn1 = r == 0 ? f : F;
          auto m = load_two_blocks_(mc, run, c, loras);
          ASSERT_TRUE(m != nullptr);
          if (m == nullptr) { return; }
          const std::size_t booked =
              T::tp_held_bytes(dir, c, (int)run.uniq.size(), mc);
          const std::size_t held = m->held_bytes();
          std::printf("  %s, tier %d%s, rank %d: booked %zu, held %zu bytes "
                      "(%s); int8 layouts %d main, %d refiner\n", name, tier,
                      hold ? ", w4 held" : "", r, booked, held,
                      booked == held ? "EQUAL" : "DIFFER",
                      m->int8_layouts(false), m->int8_layouts(true));
          EXPECT_TRUE(booked == held);
          if (tier > 0 && mc->supports_matrix_cores()) {
            EXPECT_TRUE(m->int8_layouts(false) > 0);
          }
          if (!hold) {
            booked_off[r] = booked;
            held_off[r] = held;
          } else {
            // The growth the stage weighs and books, plan and load alike.
            const std::size_t gb = booked - booked_off[r];
            const std::size_t gh = held - held_off[r];
            const bool engages = mc->supports_matrix_cores();
            std::printf("    w4 hold's growth: booked %zu, held %zu bytes "
                        "(%d w4 main-block projections)\n", gb, gh, w4);
            EXPECT_TRUE(booked >= booked_off[r] && held >= held_off[r]);
            EXPECT_TRUE(gb == gh);
            EXPECT_TRUE((gb > 0) == (w4 > 0 && engages));
          }
          if (!loras.empty()) {
            const auto fp =
                T::lora_footprint(loras[0].path, c.n_layers, nullptr, &c.tp);
            const auto whole = T::lora_footprint(loras[0].path, c.n_layers);
            std::printf("    adapter: booked %zu, held %zu (%s; the whole "
                        "width's %zu)\n", fp.whole, m->lora_bytes(),
                        fp.whole == m->lora_bytes() ? "EQUAL" : "DIFFER",
                        whole.whole);
            EXPECT_TRUE(fp.whole == m->lora_bytes());
            EXPECT_TRUE(fp.streamed < fp.whole);
            EXPECT_TRUE(fp.whole < whole.whole);
          }
        }
      }
    }
  };
  std::vector<T::LoraSpec> loras;
  if (!lora.empty()) { loras.push_back({lora, 1.0f}); }
  arm("bf16", root, {0, 1, 2}, loras, false);
  // An affine pack's share keeps its codes, scales and biases -- on a
  // main block under an int8 tier, its layout in their place, converted
  // (tp_fast_affine_), a w4 one only where the hold is on.
  const std::string pack = env_("VPIPE_MINIMAX_H3_QUANT_DIT");
  if (!pack.empty()) {
    arm("affine pack", pack, {0, 1, 2}, {}, true);
  } else {
    std::printf("  (VPIPE_MINIMAX_H3_QUANT_DIT unset: no affine arm)\n");
  }
}

// A RANK STREAMS, THEN KEEPS ITS SHARE, as the whole model does: each
// streamed block is its share cut afresh (the per-block path -- a slot
// refills the checkpoint's tensors whole, which a cut is not), the next
// one's read under this one's GPU work, and residency keeps what the box
// allows. Three real blocks, an 11-of-56-heads split, both ranks, each
// arm against the same split PRELOADED (the VPIPE_PRELOAD A/B):
//   - streamed, nothing kept (no residency reserve), two forwards: the
//     velocity BIT-IDENTICAL, no block resident, and the rank holding
//     its plan floor (tp_held_bytes' `floor`) less the two blocks a
//     forward has in the air -- the blocks are uniform, so one block's
//     share is the figure with all three kept less the floor;
//   - streamed and kept: forwards promote blocks (a capped number a
//     forward) until one runs with all three kept: BIT-IDENTICAL, and the
//     rank holds what the plan books with every block kept. (A streamed
//     block takes the split SwiGLU, a kept one of a pack the fused one --
//     woven at its promotion, as the whole model's are -- and the two
//     round differently, so a pack's streamed forward is not the
//     preloaded one bit for bit: it is held to a rounding bar.)
// With an adapter, its factors spilled to the streamed set (lora_
// footprint's `streamed`, cut to the share) and taken back with the
// blocks (its `whole`). An affine pack when given, on the token tier
// (its int8 layout on matrix cores alone).
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (FL2VA), optionally
// VPIPE_MINIMAX_H3_TURBO_LORA, VPIPE_MINIMAX_H3_QUANT_DIT.
TEST(tensor_parallel, a_rank_streams_then_keeps_its_share)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  // `fused`: an affine pack's fc1 takes the fused SwiGLU kept (no int8
  // layout converted it, VPIPE_H3_NO_FUSED_FF unset, no adapter on it).
  auto arm = [&](const char* name, const std::string& dir, int tier,
                 const std::vector<T::LoraSpec>& loras, bool fused) {
    BlockRun run;
    ASSERT_TRUE(run.setup(mc, dir, 3, 9, 24, 40, 8));
    if (run.cfg.n_layers != 3) { return; }
    run.cfg.autotune = false;
    T::Config c = run.cfg;
    c.i8_gemm = tier > 0;
    c.i8_tier = tier;
    const int h = 11, f = c.ffn / c.n_heads * h;
    TwoRanks pre;
    StreamedRanks none, kept;
    ASSERT_TRUE(split_forward_(mc, run, c, h, f, &pre, loras));
    ASSERT_TRUE(split_streamed_(mc, run, c, h, f, false, 2, &none, loras));
    ASSERT_TRUE(split_streamed_(mc, run, c, h, f, true, 2, &kept, loras));
    for (int r = 0; r < 2; ++r) {
      const T::Config rc = rank_of_(c, r, h, f);
      std::size_t floor = 0;
      const std::size_t booked =
          T::tp_held_bytes(dir, rc, (int)run.uniq.size(), mc, &floor);
      const std::size_t block = booked - floor;
      std::printf("  %s, tier %d, rank %d: booked %zu, floor %zu; "
                  "streamed holds %zu (+ two blocks %zu: %s), %d kept; "
                  "kept holds %zu (%s), %d kept after %d forwards\n", name,
                  tier, r, booked, floor, none.held[r], 2 * block,
                  none.held[r] + 2 * block == floor ? "THE FLOOR" : "OFF",
                  none.kept[r], kept.held[r],
                  kept.held[r] == booked ? "EQUAL" : "DIFFER",
                  kept.kept_before[r], kept.forwards);
      EXPECT_TRUE(booked > floor && floor > 2 * block);
      EXPECT_TRUE(none.kept[r] == 0 && kept.kept_before[r] == 3);
      EXPECT_TRUE(none.held[r] + 2 * block == floor);
      EXPECT_TRUE(kept.held[r] == booked);
      if (!loras.empty()) {
        const auto fp = T::lora_footprint(loras[0].path, rc.n_layers,
                                          nullptr, &rc.tp);
        std::printf("    adapter: streamed %zu (booked %zu), kept %zu "
                    "(booked %zu)\n", none.lora[r], fp.streamed,
                    kept.lora[r], fp.whole);
        EXPECT_TRUE(fp.streamed < fp.whole);
        EXPECT_TRUE(none.lora[r] == fp.streamed);
        EXPECT_TRUE(kept.lora[r] == fp.whole);
      }
    }
    std::printf("    velocity: streamed %s, kept %s the preloaded split; "
                "ranks %s / %s\n",
                same_bits_(none.v[0], pre.v[0]) ? "IDENTICAL to" : "OFF",
                same_bits_(kept.v[0], pre.v[0]) ? "IDENTICAL to" : "OFF",
                same_bits_(none.v[0], none.v[1]) ? "IDENTICAL" : "DIFFER",
                same_bits_(kept.v[0], kept.v[1]) ? "IDENTICAL" : "DIFFER");
    if (fused) {
      const double d = rel_l2_(none.v[0], pre.v[0]);
      std::printf("    streamed (split SwiGLU) vs preloaded (fused): "
                  "rel-L2 %.3e\n", d);
      EXPECT_TRUE(d < 1e-2);
    } else {
      EXPECT_TRUE(same_bits_(none.v[0], pre.v[0]));
    }
    EXPECT_TRUE(same_bits_(kept.v[0], pre.v[0]));
    EXPECT_TRUE(same_bits_(none.v[0], none.v[1]));
    EXPECT_TRUE(same_bits_(kept.v[0], kept.v[1]));
    EXPECT_TRUE(none.out_of_step == 0 && none.mismatched == 0);
    EXPECT_TRUE(kept.out_of_step == 0 && kept.mismatched == 0);
  };
  arm("bf16", root, 0, {}, false);
  const std::string lora = env_("VPIPE_MINIMAX_H3_TURBO_LORA");
  if (!lora.empty()) {
    arm("bf16 + Turbo LoRA", root, 0, {{lora, 1.0f}}, false);
  }
  const std::string pack = env_("VPIPE_MINIMAX_H3_QUANT_DIT");
  if (!pack.empty()) {
    arm("affine pack", pack, 2, {}, !mc->supports_matrix_cores());
    // ...and with the fusion off, every forward the preloaded one.
    EnvScope no_fuse("VPIPE_H3_NO_FUSED_FF", "1");
    arm("affine pack, no fused SwiGLU", pack, 2, {}, false);
  }
}

// A CLIP AFTER A STOPPED ONE STARTS AFRESH. A rank stopped part way
// through a forward has posted exchanges the GPU never ran: they are
// abandoned and the generation's worker fails, as it must -- the ranks
// are out of step from there. The model's weights stay (a stage keeps
// them, by this host's memory); the next clip opens a new channel and
// begins at seq 0, and its forward is the clean one's to the bit. Before
// a generation owned the link, the channel, the failed worker and the
// count were the MODEL's: every later clip on a kept rank failed ("the
// forward ended early"), and a peer that had reloaded met seq N.
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (FL2VA).
TEST(tensor_parallel, a_clip_after_a_stopped_one_starts_afresh)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  BlockRun run;
  ASSERT_TRUE(run.setup(mc, root, 2, 9, 24, 40, 8));
  run.cfg.autotune = false;
  T::Config c = run.cfg;
  std::string err;
  FlexData tp = FlexData::from_json(
      R"({"rank": 0, "ranks": 2, "heads": [0, 11], "ffn": [0, 2816]})");
  ASSERT_TRUE(TpShare::from_flex(tp, c.n_heads, c.ffn, &c.tp, &err));
  auto m = load_two_blocks_(mc, run, c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  double ms = 0.0;
  std::vector<float> clean, again, none;
  // No generation open: a rank's forward is refused, not run alone.
  EXPECT_TRUE(!run.forward_on(*m, &none, &ms, baked_step_));

  auto ch1 = std::make_shared<EchoChannel>();
  ch1->delay_us = 0;
  ASSERT_TRUE(m->begin_tp_generation(ch1, &err));
  ASSERT_TRUE(run.forward_on(*m, &clean, &ms, baked_step_));
  // Stopped at block 1's check: block 0's band exchanges were posted and
  // never committed.
  int calls = 0;
  m->set_stream_stop([&calls]() { return ++calls >= 2; });
  EXPECT_TRUE(!run.forward_on(*m, &none, &ms, baked_step_));
  m->set_stream_stop({});
  // ...so this generation is out of step for good.
  EXPECT_TRUE(!run.forward_on(*m, &none, &ms, baked_step_));
  m->end_tp_generation();
  EXPECT_TRUE(!m->tp_generation_open());

  auto ch2 = std::make_shared<EchoChannel>();
  ch2->delay_us = 0;
  ASSERT_TRUE(m->begin_tp_generation(ch2, &err));
  ASSERT_TRUE(run.forward_on(*m, &again, &ms, baked_step_));
  m->end_tp_generation();
  std::printf("  stopped at block %d; next clip: %llu exchanges from seq 0, "
              "%d out of step, velocity %s the first clip's\n", calls - 1,
              (unsigned long long)ch2->next.load(), ch2->out_of_step.load(),
              same_bits_(again, clean) ? "IDENTICAL to" : "DIFFERS from");
  // (MUTATION: the count carried over -- 16 of 16 out of step.)
  EXPECT_TRUE(ch2->out_of_step.load() == 0);
  EXPECT_TRUE(ch2->next.load() > 0);
  EXPECT_TRUE(same_bits_(again, clean));
  // Every buffer the first channel was handed was forgotten before it
  // went, as the second's were.
  for (EchoChannel* ch : {ch1.get(), ch2.get()}) {
    std::lock_guard<std::mutex> lk(ch->mu);
    int unforgotten = 0;
    for (const void* b : ch->handed) {
      if (ch->forgotten.count(b) == 0) { ++unforgotten; }
    }
    EXPECT_TRUE(!ch->handed.empty() && unforgotten == 0);
  }
}

// A RANK IS BOOKED AT ITS SHARE. generate-video's claim for a rank's DiT
// -- the ledger that refuses graphs and every peer sizes against -- is
// what the rank holds with every block kept (its share of the four
// projections, the trunk, the bake's tables; declare_memory()'s figure),
// streamable to the RANK's floor (two blocks' shares in place of all);
// the weights planner books exactly that. Claimed as the unsplit DiT's
// -- the whole checkpoint, streamable to its floor -- an 11-head rank of
// FL2VA read as 66 GB. The same stage unsplit keeps the whole model's
// claim.
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (FL2VA); configs and sizes only.
TEST(tensor_parallel, a_rank_is_booked_at_its_share)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root.empty()) { return; }
  Session sess;
  auto cfg = [&](bool split) {
    FlexData c = FlexData::make_object();
    auto o = c.as_object();
    o.insert("hf_dir", FlexData::make_string(root));
    o.insert("width", FlexData::make_int(832));
    o.insert("height", FlexData::make_int(480));
    o.insert("frames", FlexData::make_int(124));
    o.insert("steps", FlexData::make_int(8));
    if (split) {
      o.insert("tensor_parallel", FlexData::from_json(
          R"({"rank": 0, "ranks": 2, "heads": [0, 11], "ffn": [0, 2816],
              "transport": "nowhere"})"));
    }
    return c;
  };
  struct Seen {
    std::string dit;
    std::int64_t held = 0;
    std::size_t floor = 0;
    ResourceClaim claim;
  };
  auto scan = [&](bool split) {
    GenerateVideoStage st(&sess, "gv", std::vector<InEdge>{}, cfg(split));
    Seen s;
    for (const ResourceClaim& c : st.declare_resources()) {
      if (c.kind != model_memory::kWeightsKind ||
          c.key.find("transformer") == std::string::npos) {
        continue;
      }
      s.dit = c.key;
      s.held = bag::integer(c.extra, model_memory::kClaimHeldBytes, 0);
      s.floor = c.floor_bytes;
      s.claim = c;
    }
    return s;
  };
  const Seen r = scan(true);
  const Seen u = scan(false);
  ASSERT_TRUE(!r.dit.empty() && r.dit == u.dit);
  const std::size_t whole = model_memory::dir_weights_bytes(r.dit);
  // Through the weights planner, as a launch delivers it.
  auto* mgr = sess.services()->generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  auto& reg = ResourcePlannerRegistry::get();
  for (std::string_view k :
       {model_memory::kWeightsKind, model_memory::kScratchKind}) {
    if (ResourcePlanner* p = reg.find(k)) { p->begin_plan(&sess); }
  }
  if (ResourcePlanner* p = reg.find(r.claim.kind)) {
    DeliveringResourceClaim d(r.claim);
    p->claim(&sess, r.claim.key, r.claim.phase, r.claim.last_phase,
             r.claim.floor_bytes);
  }
  const std::size_t planned = mgr->planned_weight_bytes(r.dit);
  std::printf("  FL2VA %zu MB; rank of 11 heads booked %lld MB (floor %zu "
              "MB), the planner %zu MB; unsplit: floor %zu MB, held %s\n",
              whole >> 20, (long long)(r.held >> 20), r.floor >> 20,
              planned >> 20, u.floor >> 20, u.held > 0 ? "SET" : "unset");
  // (MUTATION: the claim without its held bytes -- the planner books
  // the checkpoint, 63,209 MB.)
  EXPECT_TRUE(r.held > 0 && (std::size_t)r.held < whole / 4);
  EXPECT_TRUE(r.floor > 0 && r.floor < (std::size_t)r.held);
  EXPECT_TRUE(planned == (std::size_t)r.held);
  EXPECT_TRUE(u.held == 0 && u.floor < whole);
}

// A SPLIT DECODE'S HEADERS (vae-decode's exchange 0): both parts are the
// whole clip's buffer, whatever either rank decoded, so ranks that would
// not add to one clip cross the same bytes -- the header is what tells.
// Two shares tiling the units agree, in either order and with one empty;
// a gap, an overlap, another tiling or another clip does not.
TEST(tensor_parallel, a_split_decodes_headers_refuse_a_mismatch)
{
  using vae_decode_split::Header;
  using vae_decode_split::agree;
  auto h = [](int rank, int u0, int u1) {
    Header x;
    x.rank = rank;
    x.ranks = 2;
    x.units = 40;
    x.u0 = u0;
    x.u1 = u1;
    x.frames = 73;
    x.h = 480;
    x.w = 832;
    x.lt = 19;
    return x;
  };
  std::string why;
  EXPECT_TRUE(agree(h(0, 0, 18), h(1, 18, 40), &why));
  EXPECT_TRUE(agree(h(1, 18, 40), h(0, 0, 18), &why));
  EXPECT_TRUE(agree(h(0, 22, 40), h(1, 0, 22), &why));
  EXPECT_TRUE(agree(h(0, 0, 40), h(1, 40, 40), &why));
  EXPECT_FALSE(agree(h(0, 0, 18), h(1, 20, 40), &why));    // a gap
  EXPECT_TRUE(why.find("tile") != std::string::npos);
  EXPECT_FALSE(agree(h(0, 0, 22), h(1, 18, 40), &why));    // an overlap
  EXPECT_FALSE(agree(h(0, 0, 18), h(0, 18, 40), &why));    // one rank twice
  Header t = h(1, 18, 40);
  t.units = 48;                                            // another tiling
  EXPECT_FALSE(agree(h(0, 0, 18), t, &why));
  Header f = h(1, 18, 40);
  f.frames = 72;                                           // another clip
  EXPECT_FALSE(agree(h(0, 0, 18), f, &why));
  Header m = h(1, 18, 40);
  m.magic = 0;
  EXPECT_FALSE(agree(h(0, 0, 18), m, &why));
}

namespace {

// bf16 of a float, rounded to nearest even, as the merge rounds a sum.
std::uint16_t
rne_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

}  // namespace

// A SPLIT DECODE'S PARTS CROSS AND ADD (vae_decode_split::cross): rank 1's
// frames alone cross, a piece at a time, and rank 0 copies them where it
// wrote nothing and adds them where both wrote (the frame they fade
// across), leaving its own frames as they are; rank 1's clip is left.
// Pieces cut across frames' planes and channels' runs (a 500-element
// piece over 143-element planes), and rank 0 holds the first share in
// one run, the second in the other.
TEST(tensor_parallel, a_split_decodes_parts_cross_and_add)
{
  constexpr int C = 3, F = 7;
  constexpr std::size_t plane = 13 * 11;
  const std::size_t n = (std::size_t)C * F * plane;
  // Share A wrote frames [0, 4), share B [3, 7); both wrote frame 3.
  const std::pair<int, int> fa{0, 4}, fb{3, 7};
  std::vector<std::uint16_t> a(n, 0), b(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const int f = (int)((i / plane) % F);
    const float v = std::sin(0.013f * (float)i) * 3.0f;
    if (f >= fa.first && f < fa.second) { a[i] = rne_(v); }
    if (f >= fb.first && f < fb.second) { b[i] = rne_(0.37f * v - 0.5f); }
  }
  for (const bool zero_holds_a : {true, false}) {
    std::vector<std::uint16_t> r0 = zero_holds_a ? a : b;
    std::vector<std::uint16_t> r1 = zero_holds_a ? b : a;
    const auto mine0 = zero_holds_a ? fa : fb;
    const auto theirs = zero_holds_a ? fb : fa;
    auto link = std::make_shared<PairLink>();
    PairChannel c0(link, 0), c1(link, 1);
    constexpr std::size_t piece = 1000;   // bytes
    std::vector<std::uint8_t> s0(piece), q0(piece), s1(piece), q1(piece);
    bool ok0 = false, ok1 = false;
    std::string e0, e1;
    std::thread t([&]() {
      ok1 = vae_decode_split::cross(c1, 1, r1.data(), C, F, plane, theirs,
                                    mine0, s1.data(), q1.data(), piece, 0,
                                    &e1);
    });
    ok0 = vae_decode_split::cross(c0, 0, r0.data(), C, F, plane, theirs,
                                  mine0, s0.data(), q0.data(), piece, 0,
                                  &e0);
    t.join();
    EXPECT_TRUE(ok0 && ok1);
    const std::size_t crossing =
        (std::size_t)C * (theirs.second - theirs.first) * plane;
    EXPECT_TRUE(c0.exchanges.load() == (crossing + 499) / 500);
    EXPECT_TRUE(c0.out_of_step.load() == 0 && link->mismatched.load() == 0);
    const std::vector<std::uint16_t>& rank1_own = zero_holds_a ? b : a;
    std::size_t off = 0, off1 = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const int f = (int)((i / plane) % F);
      const bool in_a = f >= fa.first && f < fa.second;
      const bool in_b = f >= fb.first && f < fb.second;
      std::uint16_t want = 0;
      if (in_a && in_b) {
        want = rne_(f32_(a[i]) + f32_(b[i]));
      } else if (in_a) {
        want = a[i];
      } else if (in_b) {
        want = b[i];
      }
      off += r0[i] != want;
      off1 += r1[i] != rank1_own[i];
    }
    std::printf("[tensor_parallel] split decode, rank 0 holding %s: %zu "
                "of %zu off on rank 0, %zu changed on rank 1, %llu "
                "exchanges\n", zero_holds_a ? "the first share" :
                "the second", off, n, off1,
                (unsigned long long)c0.exchanges.load());
    EXPECT_TRUE(off == 0);
    EXPECT_TRUE(off1 == 0);
  }
}

namespace {

// One latent beat, then the end (vae_decode_clip's, under its own name).
class TpLatentSource : public TypedStage<TpLatentSource> {
public:
  static constexpr const char* kTypeName = "ut-tp-latent-source";
  using TypedStage::TypedStage;
  TensorBeat beat;
  bool sent = false;
  Job
  process(RuntimeContext& ctx) override
  {
    if (sent) { ctx.signal_done(); co_return; }
    sent = true;
    TensorBeat tb = beat;
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
    co_return;
  }
};

// Every beat's bytes.
class TpCollector : public TypedStage<TpCollector> {
public:
  static constexpr const char* kTypeName = "ut-tp-collector";
  using TypedStage::TypedStage;
  std::vector<std::vector<std::uint8_t>> bytes;
  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      const std::uint8_t* p = tb->bytes_();
      bytes.emplace_back(p, p + tb->byte_size());
    }
    co_return;
  }
};

// Pairs the two ranks' channels of one clip over a PairLink, and keeps
// the shares it was handed.
struct StagePairTransport final : TpTransport {
  std::mutex                                mu;
  std::shared_ptr<PairLink>                 link;
  int                                       opened = 0;
  std::vector<TpShare>                      shares;
  std::vector<std::shared_ptr<PairChannel>> channels;
  std::string_view name() const override { return "ut-tp-stage-pair"; }
  std::shared_ptr<TpChannel> open(const FlexData&, const TpShare& share,
                                  std::string*) override
  {
    std::lock_guard<std::mutex> lk(mu);
    if (!link || opened == 2) {
      link = std::make_shared<PairLink>();
      opened = 0;
    }
    ++opened;
    shares.push_back(share);
    auto ch = std::make_shared<PairChannel>(link, share.rank);
    channels.push_back(ch);
    return ch;
  }
  void reset()
  {
    std::lock_guard<std::mutex> lk(mu);
    link.reset();
    opened = 0;
    shares.clear();
    channels.clear();
  }
};

StagePairTransport*
stage_pair_transport_()
{
  static StagePairTransport* t = []() {
    auto u = std::make_unique<StagePairTransport>();
    StagePairTransport* p = u.get();
    register_tp_transport(std::move(u));
    return p;
  }();
  return t;
}

// One vae-decode of `lat` in a session of its own: the frame beats it
// emitted, and whether the session's manager still holds a weight set.
struct DecodeRun {
  std::vector<std::vector<std::uint8_t>> frames;
  std::size_t sets_after = 0;
};

void
decode_run_(const char* root, const TensorBeat& lat, const FlexData& tp,
            DecodeRun* out)
{
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src_u = std::make_unique<TpLatentSource>(
      &sess, "src", std::vector<InEdge>{}, FlexData::make_object());
  src_u->beat = lat;
  auto* src = static_cast<TpLatentSource*>(pl->insert_stage(std::move(src_u)));
  src->allocate_oports(1);
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(root));
  cfg.as_object().insert_or_assign("unload_when_idle",
                                   FlexData::make_string("destroy"));
  if (tp.is_object()) {
    cfg.as_object().insert_or_assign("tensor_parallel", tp);
  }
  auto vd_u = std::make_unique<VaeDecodeStage>(
      &sess, "vd", std::vector<InEdge>{{src, 0}}, cfg);
  auto* vd = static_cast<VaeDecodeStage*>(pl->insert_stage(std::move(vd_u)));
  auto col_u = std::make_unique<TpCollector>(
      &sess, "frames", std::vector<InEdge>{{vd, 0}}, FlexData::make_object());
  auto* col = static_cast<TpCollector*>(pl->insert_stage(std::move(col_u)));
  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return; }
  rt.wait_idle();
  rt.stop();
  out->frames = col->bytes;
  out->sets_after =
      sess.services()->generative_model_manager()->weight_set_count();
}

FlexData
tp_cfg_(int rank, const char* share)
{
  std::string j = "{\"rank\": " + std::to_string(rank) +
                  ", \"ranks\": 2, \"transport\": \"ut-tp-stage-pair\"";
  if (share != nullptr) { j += std::string(", \"decode_share\": ") + share; }
  return FlexData::from_json(j + "}");
}

}  // namespace

// A SPLIT DECODE THROUGH THE STAGE: two vae-decode ranks, each in a
// session of its own, over one pair of channels. Rank 0 emits the clip
// the whole decode does -- each frame the same bytes, or a level off
// where the ranks fade across each other -- and rank 1 emits nothing;
// both channels said they were a decode's; the headers crossed first;
// and each rank let its VAE go after the clip, the box asking for it
// (`unload_when_idle: destroy`) -- rank 1 too, which returns early.
// Shares that do not tile the units are refused on both ranks.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (a partition holding video_vae/).
TEST(tensor_parallel, a_split_decode_runs_through_the_stage)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  StagePairTransport* tr = stage_pair_transport_();
  tr->reset();
  // Several chunks of a small canvas: a unit is a chunk here, and the
  // ranks fade across the chunk where their shares meet.
  const int Cz = 24, LT = 14, lh = 4, lw = 4;
  TensorBeat lat;
  lat.dtype = TensorBeat::DType::F32;
  lat.shape = {Cz, LT, lh, lw};
  lat.resize_contiguous((std::size_t)Cz * LT * lh * lw * sizeof(float));
  float* z = reinterpret_cast<float*>(lat.data.data());
  for (std::size_t i = 0; i < (std::size_t)Cz * LT * lh * lw; ++i) {
    z[i] = 0.9f * std::sin(0.37f * (float)i);
  }
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign("fps", FlexData::make_real(24.0));
  lat.sideband = std::move(sb);

  DecodeRun whole;
  decode_run_(root, lat, FlexData(), &whole);
  ASSERT_TRUE(!whole.frames.empty());
  if (whole.frames.empty()) { return; }

  DecodeRun r0, r1;
  std::thread t1([&]() { decode_run_(root, lat, tp_cfg_(1, nullptr), &r1); });
  decode_run_(root, lat, tp_cfg_(0, nullptr), &r0);
  t1.join();
  std::size_t same = 0, off = 0;
  int worst = 0;
  ASSERT_TRUE(r0.frames.size() == whole.frames.size());
  for (std::size_t f = 0; f < r0.frames.size() && f < whole.frames.size();
       ++f) {
    const auto& a = r0.frames[f];
    const auto& b = whole.frames[f];
    if (a.size() != b.size()) { ++off; continue; }
    same += a == b;
    for (std::size_t i = 0; i < a.size(); ++i) {
      worst = std::max(worst, std::abs((int)a[i] - (int)b[i]));
    }
  }
  std::uint64_t crossed = 0;
  bool decode_kind = tr->shares.size() == 2;
  for (const TpShare& sh : tr->shares) {
    decode_kind = decode_kind && sh.heads() == 0 && sh.extra.is_object() &&
                  sh.extra.as_object().contains(tp_extra::kChannel) &&
                  sh.extra.as_object().at(tp_extra::kChannel).as_string("")
                      == tp_extra::kChannelDecode;
  }
  for (const auto& ch : tr->channels) {
    crossed = std::max(crossed, ch->exchanges.load());
  }
  std::printf("[tensor_parallel] split decode through the stage: %zu "
              "frames, %zu the whole decode's bytes, worst %d level(s); "
              "rank 1 emitted %zu; %llu exchanges; sets held after %zu / "
              "%zu\n", r0.frames.size(), same, worst, r1.frames.size(),
              (unsigned long long)crossed, r0.sets_after, r1.sets_after);
  EXPECT_TRUE(off == 0);
  EXPECT_TRUE(worst <= 1);
  EXPECT_TRUE(same * 2 >= r0.frames.size());
  EXPECT_TRUE(r1.frames.empty());
  EXPECT_TRUE(decode_kind);
  EXPECT_TRUE(crossed >= 2);   // the headers, then rank 1's frames
  EXPECT_TRUE(r0.sets_after == 0 && r1.sets_after == 0);

  // Shares that leave a gap: both ranks refuse, rank 0 emits nothing.
  tr->reset();
  DecodeRun g0, g1;
  std::thread t2(
      [&]() { decode_run_(root, lat, tp_cfg_(1, "[0.75, 1.0]"), &g1); });
  decode_run_(root, lat, tp_cfg_(0, "[0.0, 0.25]"), &g0);
  t2.join();
  std::printf("[tensor_parallel] shares with a gap: rank 0 emitted %zu, "
              "rank 1 %zu\n", g0.frames.size(), g1.frames.size());
  EXPECT_TRUE(g0.frames.empty() && g1.frames.empty());
}
