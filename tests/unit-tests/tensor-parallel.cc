#include "minitest.h"
#include "minimax-h3-block-run.h"
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

#include <atomic>
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
// pinned to steel's 32-row tile (no tuner deciding per model).
std::unique_ptr<T>
load_two_blocks_(MetalCompute* mc, const BlockRun& run, const T::Config& cfg)
{
  auto m = T::load(run.root, mc, cfg, false, {});
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
               std::vector<float>* out)
{
  auto m = load_two_blocks_(mc, run, cfg);
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
};
bool
split_forward_(MetalCompute* mc, BlockRun& run, const T::Config& cfg,
               int h, int f, TwoRanks* out)
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
    m[r] = load_two_blocks_(mc, run, c);
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
  out->out_of_step = ch[0]->out_of_step.load() + ch[1]->out_of_step.load();
  out->mismatched = link->mismatched.load();
  out->exchanges = ch[0]->exchanges.load();
  return ok0 && ok1;
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
// what the rank holds (its share of the four projections, the trunk, the
// bake's tables; declare_memory()'s figure), not streamable below it;
// the weights planner books exactly that. Claimed as the unsplit DiT's
// -- the whole checkpoint, streamable to its floor -- an 11-head rank of
// FL2VA read as 66 GB that could shrink to a few. The same stage
// unsplit keeps the streaming claim.
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
  EXPECT_TRUE(r.floor == (std::size_t)r.held);
  EXPECT_TRUE(planned == (std::size_t)r.held);
  EXPECT_TRUE(u.held == 0 && u.floor < whole);
}
