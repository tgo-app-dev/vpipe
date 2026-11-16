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
#include "generative-models/flux3/metal-flux3-transformer.h"
#include "generative-models/moss-image/metal-moss-image-transformer.h"
#include "generative-models/krea2/metal-krea2-transformer.h"
#include "generative-models/qwen-image/metal-qwen-image21-transformer.h"
#include "generative-models/qwen-image/qwen-image21-layout.h"
#include "generative-models/z-image/metal-z-image-transformer.h"
#include "generative-models/fibo2/fibo2-layout.h"
#include "generative-models/fibo2/metal-fibo2-transformer.h"
#include "generative-models/z-image/z-image-layout.h"
#include "generative-models/quantize/safetensors-writer.h"
#include "generative-models/shared/tp-overlap.h"
#include "generative-models/tensor-parallel.h"
#include "pipeline/resource-claim-delivery.h"
#include "pipeline/resource-plan.h"
#include "stages/generate-image-stage.h"
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
#include <filesystem>
#include <fstream>
#include <functional>
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
namespace fs = std::filesystem;
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

// A PROMPT LONGER THAN A RANK'S BAND (a Ref2VA presentation, an
// in-context refine's draft) bands only where the share says both ranks
// may (tp_extra::kPromptBands): an older rank runs such a prompt's
// blocks whole, and a pair must cut the same bands. Unsaid -- or false
// -- the rank keeps that rule; said, it bands as an unsplit model does.
// A prompt within the band, and an unsplit model, band either way. The
// key is a boolean or the share is refused.
TEST(tensor_parallel, a_long_prompt_bands_only_where_both_ranks_may)
{
  if (std::getenv("VPIPE_H3_TP_BANDS") != nullptr) { return; }
  using genai::MetalMiniMaxH3Transformer;
  using T = MetalMiniMaxH3Transformer;
  std::string err;
  auto share = [&](const char* json) {
    T::Config c;
    TpShare s;
    if (!TpShare::from_flex(FlexData::from_json(json), 56, 14336, &s,
                            &err)) {
      return T::Config{};
    }
    c.tp = s;
    return c;
  };
  const T::Config unsaid = share(R"({"rank": 0, "ranks": 2})");
  const T::Config said =
      share(R"({"rank": 0, "ranks": 2, "prompt_bands": true})");
  const T::Config no =
      share(R"({"rank": 1, "ranks": 2, "prompt_bands": false})");
  ASSERT_TRUE(said.tp.active() && no.tp.active());
  EXPECT_TRUE(said.tp.extra.is_object() &&
              said.tp.extra.as_object()
                  .at(tp_extra::kPromptBands)
                  .as_bool(false));
  const int seq = 8843;
  const int band = T::band_rows(unsaid, seq);
  ASSERT_TRUE(band == 1152);
  // Within the band: banded whatever the share says.
  EXPECT_TRUE(T::band_for(unsaid, seq, 512) == band);
  EXPECT_TRUE(T::band_for(said, seq, 512) == band);
  // Past it: whole unless said.
  EXPECT_TRUE(T::band_for(unsaid, seq, 2600) == 0);
  EXPECT_TRUE(T::band_for(no, seq, 2600) == 0);
  EXPECT_TRUE(T::band_for(said, seq, 2600) == band);
  EXPECT_TRUE(T::text_band_rows(said, 2600, band) > 0 &&
              T::text_band_rows(said, 2600, band) <= band);
  // An unsplit model bands any prompt.
  T::Config one;
  const int whole_band = T::band_rows(one, 400000);
  EXPECT_TRUE(whole_band > 0 && T::band_for(one, 400000, 20000) ==
                                    whole_band);
  // Not a boolean: refused.
  TpShare s;
  EXPECT_FALSE(TpShare::from_flex(
      FlexData::from_json(R"({"rank": 0, "ranks": 2, "prompt_bands": 1})"),
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
  // 9 latent frames of a 24 x 40 latent canvas: 2,160 video rows, past
  // the two bands of 1,024 a rank needs to band -- behind an 8-row
  // prompt, and a 2,600-row one past the band: given `prompt_bands`, the
  // refiner banded, its partials exchanged a band at a time; unsaid,
  // every block whole. A channel each: a model counts its exchanges
  // from 0.
  struct Case { int text; bool prompt_bands; };
  for (const Case cs : {Case{8, false}, Case{2600, true}, Case{2600, false}}) {
    const int text = cs.text;
    FlexData tp = FlexData::from_json(
        cs.prompt_bands
            ? R"({"rank": 0, "ranks": 2, "heads": [0, 11], "ffn": [0, 2816],
                 "prompt_bands": true})"
            : R"({"rank": 0, "ranks": 2, "heads": [0, 11], "ffn": [0, 2816]})");
    ASSERT_TRUE(TpShare::from_flex(tp, cfg.n_heads, cfg.ffn, &cfg.tp, &err));
    auto ch = std::make_shared<EchoChannel>();
    ch->delay_us = 0;
    h3::PackedLayout L;
    const std::vector<int> tags((std::size_t)text, h3::kTextTag);
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
    int band = 0, tband = 0;
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
      tband = m->last_text_band_rows();
      est = T::scratch_bytes(T::rank_config(cfg), L.seq_len, (int)tags.size(),
                             (int)uniq.size(), false, m->ff_scratch_narrow(),
                             0, band, 0);
      act = m->scratch_resident_bytes();
      whole = T::scratch_bytes(cfg, L.seq_len, (int)tags.size(),
                               (int)uniq.size(), false, m->ff_scratch_narrow(),
                               0, band, 0);
    }
    std::printf("[tensor_parallel] rank of %d rows (a %d-row prompt), band "
                "%d, the prompt's %d: scratch est %zu, allocated %zu, at "
                "the checkpoint's widths %zu\n", L.seq_len, text, band,
                tband, est, act, whole);
    EXPECT_TRUE((band > 0) == (text == 8 || cs.prompt_bands));
    EXPECT_TRUE((tband > 0) == (band > 0 && text > band));
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
  // Whether a stop was ever set, and whether one is set now.
  std::atomic<bool>          stop_seen{false}, stop_now{false};
  void set_stop(std::function<bool()> stop) override
  {
    stop_now = (bool)stop;
    if (stop) { stop_seen = true; }
  }
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
  // Both ranks' ANE feed-forward armed; rank 0's int8 ANE modules; the
  // rows each rank's feed-forward and qkv tiers ran ON THE ANE; rank 1's
  // band.
  bool ane = false;
  int ane_int8 = 0;
  std::int64_t ane_rows[2] = {}, qkv_rows[2] = {};
  int band1 = 0;
  // Rank 0's text refiner's band (0: whole).
  int tband = 0;
};
bool
split_forward_(MetalCompute* mc, BlockRun& run, const T::Config& cfg,
               int h, int f, TwoRanks* out,
               const std::vector<T::LoraSpec>& loras = {},
               int drop_ane = -1, bool prompt_bands = false)
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
    // `prompt_bands`: stated alike to both ranks, as a host states it.
    if (prompt_bands) {
      c.tp.extra = FlexData::make_object();
      c.tp.extra.as_object().insert_or_assign(tp_extra::kPromptBands,
                                              FlexData::make_bool(true));
    }
    m[r] = load_two_blocks_(mc, run, c, loras);
    if (m[r] == nullptr) { return false; }
    // `drop_ane`: that rank's ANE tiers taken off for the clip, as
    // generate-video's preflight takes them off a clip that fits only
    // without them.
    if (r == drop_ane) { m[r]->disable_ane(); }
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
  out->ane = m[0]->ane_armed() && m[1]->ane_armed();
  out->ane_int8 = m[0]->ane_int8_modules();
  for (int r = 0; r < 2; ++r) {
    out->ane_rows[r] = m[r]->ane_rows_run(false);
    out->qkv_rows[r] = m[r]->ane_rows_run(true);
  }
  out->band1 = m[1]->last_band_rows();
  out->tband = m[0]->last_text_band_rows();
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
//   - a 2,600-row prompt, past the rank's band of 1,024: given
//     `prompt_bands`, the text refiner in bands of its own, each band's
//     partials exchanged; unsaid, every block whole (an older rank's
//     rule, which its peer must share);
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
                 bool banded_expected, bool lag0_too, int text = 8,
                 bool prompt_bands = false) {
    BlockRun run;
    ASSERT_TRUE(run.setup(mc, dir, 2, 9, 24, 40, 8, text));
    run.cfg.autotune = false;
    run.cfg.qkv_per_head = per_head && run.cfg.qkv_per_head;
    std::vector<float> whole;
    ASSERT_TRUE(whole_forward_(mc, run, run.cfg, &whole));
    TwoRanks s;
    ASSERT_TRUE(split_forward_(mc, run, run.cfg, 11, 2816, &s, {}, -1,
                               prompt_bands));
    const double d = rel_l2_(s.v[0], whole);
    std::printf("  %-26s %d rows, band %d (the prompt's %d): split vs "
                "unsplit rel-L2 %.3e, ranks %s, %llu exchanges\n", name,
                run.L.seq_len, s.band, s.tband, d,
                same_bits_(s.v[0], s.v[1]) ? "IDENTICAL" : "DIFFER",
                (unsigned long long)s.exchanges);
    EXPECT_TRUE(s.out_of_step == 0 && s.mismatched == 0);
    EXPECT_TRUE(same_bits_(s.v[0], s.v[1]));
    EXPECT_TRUE(d < kBar);
    EXPECT_TRUE((s.band > 0) == banded_expected);
    // A prompt longer than a band runs its refiner in bands of its own,
    // each band's partial sums exchanged.
    EXPECT_TRUE((s.tband > 0) == (s.band > 0 && text > s.band));
    if (lag0_too) {
      EnvScope lag("VPIPE_H3_TP_LAG", "0");
      TwoRanks s0;
      ASSERT_TRUE(split_forward_(mc, run, run.cfg, 11, 2816, &s0, {}, -1,
                                 prompt_bands));
      std::printf("  %-26s band after band: %s\n", name,
                  same_bits_(s0.v[0], s.v[0]) ? "IDENTICAL to overlapped"
                                              : "DIFFERS from overlapped");
      EXPECT_TRUE(same_bits_(s0.v[0], s.v[0]));
      EXPECT_TRUE(same_bits_(s0.v[0], s0.v[1]));
    }
  };
  arm("bf16, per-head q|k|v", root, true, true, true);
  // A prompt past a band (an in-context refine's): 2,600 rows, the
  // refiner's bands 896, 896 and 808 in the rank's bands of 1,024 --
  // where the share says both ranks may (prompt_bands); unsaid, every
  // block whole, as an older rank runs it.
  arm("bf16, a prompt past a band", root, true, true, true, 2600, true);
  arm("bf16, past a band, unsaid", root, true, false, false, 2600);
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

// A RANK UNDER THE ANE TIER BANDS AS ONE WITHOUT IT. Each rank's
// feed-forward (and qkv) modules are its share's widths, staged from its
// cut projections, and each BAND's rows are split between the engines --
// the qkv's in the band's first pass, the feed-forward's straight into the
// band's exchange buffer -- so its bands, and the exchanges crossing under
// the next band's work, are those of a rank without the tier. Two real
// FL2VA blocks, 2,184 rows (two bands of ~1,100), held to the unsplit
// model under the same tier (whole: a whole model's blocks stay whole
// under the ANE), the bar the tier's own distance from bf16, twice over:
//   - the feed-forward; + qkv at 12 heads and at 11; int8 + qkv: banded as
//     the plain split bands, its exchanges as many, BOTH tiers' rows run
//     on the ANE on BOTH ranks (counted -- being armed does not say), the
//     ranks identical, and band after band (VPIPE_H3_TP_LAG=0) the same
//     bits;
//   - ROWS LOST (VPIPE_ANE_FAIL=1:1): each rank's second split, a band,
//     loses its ANE rows from the second chunk on -- the GPU takes them
//     back inside the band, the modules stay on the GPU from there (fewer
//     ANE rows than the arm before counted), and the bands, exchanges and
//     sums are unmoved;
//   - whole blocks (VPIPE_H3_TP_BANDS=1);
//   - the chunk the band chose (ane_chunk_for_, no VPIPE_H3_ANE_CHUNK pin)
//     and the qkv share the balancer's (no VPIPE_H3_ANE_QKV_ROWS: probes,
//     band by band): the ANE still takes feed-forward rows of a
//     ~1,100-row band (at 2,048 it took none);
//   - ONE RANK'S ANE DROPPED for the clip (disable_ane, as generate-
//     video's preflight drops it on a host short of room): the same bands
//     on both ranks, in step, within the bar. When the ANE kept a rank's
//     blocks whole, this rank banded beside a whole-block peer and both
//     forwards failed at the first exchange;
//   - the Turbo LoRA under the tier (VPIPE_MINIMAX_H3_TURBO_LORA): each
//     rank's cut factors folded into its staged rows.
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (FL2VA), optionally
// VPIPE_MINIMAX_H3_TURBO_LORA.
TEST(tensor_parallel, a_split_forward_runs_the_ane_tier)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  // The unsplit model runs its 2,184 rows WHOLE, where the default
  // 2,048-row chunk leaves the ANE none: a 256-row chunk for every model,
  // and the qkv module's share pinned at half (the feed-forward's is,
  // Config::ane_rows) -- balanced from measured times, two runs of one
  // forward can give the ANE other rows, and band after band would not
  // be the same bits -- but where an arm says otherwise.
  struct Pin {
    const char* key;
    bool had;
    std::string v;
    Pin(const char* k, const char* to) : key(k)
    {
      const char* e = std::getenv(k);
      had = e != nullptr;
      v = had ? e : "";
      ::setenv(k, to, 1);
    }
    ~Pin()
    {
      if (had) { ::setenv(key, v.c_str(), 1); }
      else     { ::unsetenv(key); }
    }
  };
  const Pin chunk_pin("VPIPE_H3_ANE_CHUNK", "256");
  const Pin qkv_pin("VPIPE_H3_ANE_QKV_ROWS", "0.5");
  constexpr double kBar = 1e-2;
  BlockRun run;
  ASSERT_TRUE(run.setup(mc, root, 2, 9, 24, 40, 8));
  run.cfg.autotune = false;
  std::vector<float> bf16_whole;
  ASSERT_TRUE(whole_forward_(mc, run, run.cfg, &bf16_whole));
  // The pattern a rank without the tier runs: its bands and exchanges.
  TwoRanks plain;
  {
    const int f = run.cfg.ffn / run.cfg.n_heads * 12;
    ASSERT_TRUE(split_forward_(mc, run, run.cfg, 12, f, &plain));
  }
  // The plan books a rank's modules at its share's widths.
  {
    T::Config c = run.cfg;
    c.ane_ffn = c.ane_qkv = true;
    const int f = c.ffn / c.n_heads * 12;
    const std::size_t whole = T::ane_runtime_bytes(c, 4096);
    const std::size_t r0 = T::ane_runtime_bytes(rank_of_(c, 0, 12, f), 4096);
    const std::size_t r1 = T::ane_runtime_bytes(rank_of_(c, 1, 12, f), 4096);
    std::printf("  ANE modules booked: whole %zu MB, ranks %zu + %zu MB\n",
                whole >> 20, r0 >> 20, r1 >> 20);
    EXPECT_TRUE(r0 > 0 && r0 < r1 && r1 < whole);
  }
  struct Opt {
    bool qkv = false, i8 = false;
    int heads = 12;
    bool lag0 = false;        // band after band too
    bool whole = false;       // VPIPE_H3_TP_BANDS=1
    // The ranks' chunk their bands' choice, the qkv share the balancer's.
    bool unpinned = false;
    int drop = -1;            // a rank whose ANE is dropped
    // VPIPE_ANE_FAIL on the ranks' modules: a split loses its rows and
    // the GPU takes them (and every later block's) back.
    const char* fail = nullptr;
    std::vector<T::LoraSpec> loras;
  };
  TwoRanks last;              // the arm before's
  auto arm = [&](const char* name, const Opt& o) {
    T::Config cfg = run.cfg;
    cfg.ane_ffn = true;
    cfg.ane_rows = 0.5f;
    cfg.ane_qkv = o.qkv;
    cfg.ane_int8 = o.i8;
    std::vector<float> whole;
    ASSERT_TRUE(whole_forward_(mc, run, cfg, &whole, o.loras));
    std::vector<float> base;
    if (!o.loras.empty()) {
      ASSERT_TRUE(whole_forward_(mc, run, run.cfg, &base, o.loras));
    }
    const int f = cfg.ffn / cfg.n_heads * o.heads;
    TwoRanks s;
    {
      std::unique_ptr<EnvScope> bands;
      if (o.whole) {
        bands = std::make_unique<EnvScope>("VPIPE_H3_TP_BANDS", "1");
      }
      std::unique_ptr<EnvScope> fail;
      if (o.fail != nullptr) {
        fail = std::make_unique<EnvScope>("VPIPE_ANE_FAIL", o.fail);
      }
      if (o.unpinned) {
        ::unsetenv("VPIPE_H3_ANE_CHUNK");
        ::unsetenv("VPIPE_H3_ANE_QKV_ROWS");
      }
      const bool ok =
          split_forward_(mc, run, cfg, o.heads, f, &s, o.loras, o.drop);
      if (o.unpinned) {
        ::setenv("VPIPE_H3_ANE_CHUNK", "256", 1);
        ::setenv("VPIPE_H3_ANE_QKV_ROWS", "0.5", 1);
      }
      ASSERT_TRUE(ok);
    }
    const double d = rel_l2_(s.v[0], whole);
    // The tier's own distance: from bf16, or from the adapter's bf16.
    const double own = rel_l2_(whole, o.loras.empty() ? bf16_whole : base);
    const double bar = std::max(kBar, 2.0 * own);
    std::printf("  %-30s %d heads: split vs unsplit rel-L2 %.3e (bar "
                "%.3e; unsplit from bf16 %.3e), band %d / %d, %llu "
                "exchanges, ANE rows ff %lld / %lld qkv %lld / %lld, %d "
                "int8 module(s), ranks %s\n", name, o.heads, d, bar, own,
                s.band, s.band1, (unsigned long long)s.exchanges,
                (long long)s.ane_rows[0], (long long)s.ane_rows[1],
                (long long)s.qkv_rows[0], (long long)s.qkv_rows[1],
                s.ane_int8,
                same_bits_(s.v[0], s.v[1]) ? "IDENTICAL" : "DIFFER");
    EXPECT_TRUE(s.out_of_step == 0 && s.mismatched == 0);
    EXPECT_TRUE(same_bits_(s.v[0], s.v[1]));
    EXPECT_TRUE(d < bar);
    // The bands a rank without the tier cuts, on both ranks -- or whole.
    EXPECT_TRUE(s.band == s.band1);
    EXPECT_TRUE(o.whole ? s.band == 0
                        : s.band == plain.band &&
                              s.exchanges == plain.exchanges);
    // The ANE's rows really ran -- on each rank whose ANE was not dropped.
    for (int r = 0; r < 2; ++r) {
      if (r == o.drop) {
        EXPECT_TRUE(s.ane_rows[r] == 0 && s.qkv_rows[r] == 0);
        continue;
      }
      EXPECT_TRUE(s.ane_rows[r] > 0);
      // (Unpinned, the balancer may keep the qkv on the GPU.)
      if (o.qkv && !o.unpinned) { EXPECT_TRUE(s.qkv_rows[r] > 0); }
      // A split that lost rows left fewer on the ANE than the arm before,
      // which ran the same blocks and bands whole on it.
      if (o.fail != nullptr) {
        EXPECT_TRUE(s.ane_rows[r] < last.ane_rows[r]);
        if (o.qkv) { EXPECT_TRUE(s.qkv_rows[r] < last.qkv_rows[r]); }
      }
    }
    last = s;
    if (o.lag0) {
      EnvScope lag("VPIPE_H3_TP_LAG", "0");
      TwoRanks s0;
      ASSERT_TRUE(split_forward_(mc, run, cfg, o.heads, f, &s0, o.loras));
      std::printf("  %-30s band after band: %s\n", name,
                  same_bits_(s0.v[0], s.v[0]) ? "IDENTICAL to overlapped"
                                              : "DIFFERS from overlapped");
      EXPECT_TRUE(same_bits_(s0.v[0], s.v[0]));
    }
  };
  std::printf("  without the tier: band %d / %d, %llu exchanges\n",
              plain.band, plain.band1, (unsigned long long)plain.exchanges);
  EXPECT_TRUE(plain.band > 0);
  {
    Opt o;
    o.lag0 = true;
    arm("ANE feed-forward", o);
  }
  {
    Opt o;
    o.qkv = true;
    o.lag0 = true;
    arm("ANE feed-forward + qkv", o);
    o.lag0 = false;
    // Both ranks' second split loses its rows from the second chunk on:
    // the GPU recomputes them inside the band, and every module stays
    // on the GPU from there -- the bands, and the exchanges, unmoved.
    o.fail = "1:1";
    arm("ANE ff + qkv, rows lost", o);
    o.fail = nullptr;
    o.heads = 11;
    arm("ANE feed-forward + qkv", o);
  }
  {
    Opt o;
    o.qkv = o.i8 = true;
    arm("ANE int8 feed-forward + qkv", o);
  }
  {
    Opt o;
    o.qkv = o.whole = true;
    arm("ANE ff + qkv, whole blocks", o);
  }
  {
    Opt o;
    o.qkv = o.unpinned = true;
    arm("ANE ff + qkv, the band's chunk", o);
  }
  {
    Opt o;
    o.qkv = true;
    o.drop = 1;
    arm("ANE ff + qkv, rank 1's dropped", o);
  }
  const std::string lora = env_("VPIPE_MINIMAX_H3_TURBO_LORA");
  if (!lora.empty()) {
    Opt o;
    o.qkv = true;
    o.heads = 11;
    o.loras = {{lora, 1.0f}};
    arm("ANE ff + qkv, Turbo LoRA", o);
  } else {
    std::printf("  (VPIPE_MINIMAX_H3_TURBO_LORA unset: no adapter arm)\n");
  }
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

// ---- FLUX.3: one partial a block ---------------------------------------

namespace {

// A small random FLUX.3 DiT on disk (Flux3Transformer2DModel's names and
// config): `heads` heads of 128, `modes` per-mode blocks and `joint`
// joint blocks, every projection bf16 -- or, `w8`, the block projections
// an MLX affine 8-bit pack in groups of 64 (a compressed copy's
// layout), the model-level ones dense.
struct F3Small {
  int heads = 6, modes = 2, joint = 3, jd = 256;
  fs::path dir;
};

bool
write_flux3_small_(const F3Small& s, bool w8, bool flow_map = true)
{
  std::error_code ec;
  fs::remove_all(s.dir, ec);
  fs::create_directories(s.dir, ec);
  const int H = s.heads * 128, F = 3 * H, TF = 256, IC = 128;
  {
    std::ofstream o(s.dir / "config.json");
    o << "{\"_class_name\":\"Flux3Transformer2DModel\","
      << "\"attention_head_dim\":128,\"axes_dims_rope\":[32,32,32,32],"
      << "\"cond_in_channels\":128,\"eps\":1e-06,\"flow_map\":true,"
      << "\"in_channels\":128,\"joint_attention_dim\":" << s.jd << ","
      << "\"mlp_ratio\":3.0,\"num_attention_heads\":" << s.heads << ","
      << "\"num_layers\":" << s.modes << ",\"num_single_layers\":"
      << s.joint << ",\"rope_theta\":10000,\"timestep_channels\":256}";
  }
  genai::SafetensorsWriter wr(s.dir.string());
  std::uint64_t st = 0x9E3779B97F4A7C15ull ^ (w8 ? 1 : 0);
  auto rnd = [&]() {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    return (float)((double)(st >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
  };
  auto dense = [&](const std::string& n, int N, int K, float sc) {
    std::vector<std::uint16_t> v((std::size_t)N * K);
    for (auto& x : v) { x = bf16_(rnd() * sc); }
    return wr.add(n + ".weight", "BF16", {N, K}, v.data(), v.size() * 2);
  };
  // MLX affine: 4 codes a word, element i in bits 8i; a scale and a bias
  // a group of 64, bf16.
  auto packed = [&](const std::string& n, int N, int K, float sc) {
    const int g = 64;
    std::vector<std::uint32_t> codes((std::size_t)N * K / 4);
    std::vector<std::uint16_t> scales((std::size_t)N * K / g);
    std::vector<std::uint16_t> biases(scales.size());
    std::vector<float> row((std::size_t)K);
    for (int r = 0; r < N; ++r) {
      for (auto& x : row) { x = rnd() * sc; }
      for (int g0 = 0; g0 < K; g0 += g) {
        float lo = row[(std::size_t)g0], hi = lo;
        for (int i = g0; i < g0 + g; ++i) {
          lo = std::min(lo, row[(std::size_t)i]);
          hi = std::max(hi, row[(std::size_t)i]);
        }
        const float s8 = f32_(bf16_(std::max(hi - lo, 1e-6f) / 255.0f));
        const float b8 = f32_(bf16_(lo));
        const std::size_t gi = ((std::size_t)r * K + g0) / g;
        scales[gi] = bf16_(s8);
        biases[gi] = bf16_(b8);
        for (int i = g0; i < g0 + g; ++i) {
          const int q = std::clamp(
              (int)std::lround((row[(std::size_t)i] - b8) / s8), 0, 255);
          const std::size_t e = (std::size_t)r * K + i;
          codes[e / 4] |= (std::uint32_t)q << (8 * (e % 4));
        }
      }
    }
    return wr.add(n + ".weight", "U32", {N, K / 4}, codes.data(),
                  codes.size() * 4) &&
           wr.add(n + ".scales", "BF16", {N, K / g}, scales.data(),
                  scales.size() * 2) &&
           wr.add(n + ".biases", "BF16", {N, K / g}, biases.data(),
                  biases.size() * 2);
  };
  auto proj = [&](const std::string& n, int N, int K, float sc) {
    return w8 ? packed(n, N, K, sc) : dense(n, N, K, sc);
  };
  bool ok = dense("x_embedders.image", H, IC, 0.08f) &&
            dense("x_embedders.image_cond", H, IC, 0.08f) &&
            dense("context_embedder", H, s.jd, 0.06f) &&
            dense("timestep_embedder.linear_1", H, TF, 0.06f) &&
            dense("timestep_embedder.linear_2", H, H, 0.03f) &&
            // A flow map's target embedder: its tensors are what make
            // the checkpoint one.
            (!flow_map ||
             (dense("target_timestep_embedder.linear_1", H, TF, 0.06f) &&
              dense("target_timestep_embedder.linear_2", H, H, 0.03f))) &&
            dense("norm_out.linear", 2 * H, H, 0.03f) &&
            dense("proj_out", IC, H, 0.03f);
  for (const char* m : {"image", "image_cond", "txt"}) {
    ok = ok &&
         dense(std::string("early_stream_modulations.") + m + ".linear",
               3 * H, H, 0.03f) &&
         dense(std::string("single_stream_modulations.") + m + ".linear",
               3 * H, H, 0.03f);
  }
  std::vector<std::string> blocks;
  for (int i = 0; i < s.modes; ++i) {
    blocks.push_back("mode_blocks.image." + std::to_string(i) + ".");
    blocks.push_back("txt_mode_blocks." + std::to_string(i) + ".");
    blocks.push_back("mode_blocks.image_cond." + std::to_string(i) + ".");
  }
  for (int i = 0; i < s.joint; ++i) {
    blocks.push_back("single_transformer_blocks." + std::to_string(i) +
                     ".");
  }
  for (const std::string& p : blocks) {
    std::vector<std::uint16_t> norm(128);
    for (auto& x : norm) { x = bf16_(1.0f + 0.1f * rnd()); }
    ok = ok && proj(p + "attn.to_q", H, H, 0.04f) &&
         proj(p + "attn.to_k", H, H, 0.04f) &&
         proj(p + "attn.to_v", H, H, 0.04f) &&
         proj(p + "attn.to_out.0", H, H, 0.04f) &&
         proj(p + "ff.linear_in", 2 * F, H, 0.04f) &&
         proj(p + "ff.linear_out", H, F, 0.03f) &&
         wr.add(p + "attn.norm_q.weight", "BF16", {128}, norm.data(), 256) &&
         wr.add(p + "attn.norm_k.weight", "BF16", {128}, norm.data(), 256);
  }
  return wr.close() && ok;
}

// One forward of `m`: a 16 x 16 image, 40 text rows, one 8 x 8 reference,
// all random but fixed; its velocity as f32.
bool
flux3_forward_(MetalCompute* mc, MetalFlux3Transformer& m, int jd,
               std::vector<float>* out, std::string* err)
{
  std::uint64_t st = 12345;
  auto rnd = [&]() {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    return (float)((double)(st >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
  };
  auto buf = [&](std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = bf16_(rnd()); }
    return b;
  };
  const int gh = 16, gw = 16, txt = 40;
  SharedBuffer lat = buf((std::size_t)gh * gw * 128);
  SharedBuffer text = buf((std::size_t)txt * jd);
  SharedBuffer ref = buf((std::size_t)8 * 8 * 128);
  MetalFlux3Transformer::Request req;
  req.latents = &lat;
  req.grid_h = gh;
  req.grid_w = gw;
  req.text = &text;
  req.text_rows = txt;
  req.refs.push_back({&ref, 8, 8});
  req.t = 0.75f;
  req.t_next = 0.5f;
  SharedBuffer v = m.forward(req, err);
  if (v.empty()) { return false; }
  const auto* p = static_cast<const std::uint16_t*>(v.contents());
  out->resize((std::size_t)gh * gw * 128);
  for (std::size_t i = 0; i < out->size(); ++i) { (*out)[i] = f32_(p[i]); }
  return true;
}

}  // namespace

namespace {

using F3 = MetalFlux3Transformer;

// Rank `r` of a two-rank split of `cfg`: heads [0, h) and [h, n) and the
// matching feed-forward rows; `floor` > 0 gives the share that band
// floor (tp_extra::kMinBandRows), as a host's bag would.
F3::Config
f3_rank_(const F3::Config& cfg, int r, int h, int floor)
{
  F3::Config c = cfg;
  const int f = cfg.mlp_hidden / cfg.n_heads * h;
  c.tp = TpShare{};
  c.tp.rank = r;
  c.tp.ranks = 2;
  c.tp.n_heads = cfg.n_heads;
  c.tp.n_ffn = cfg.mlp_hidden;
  c.tp.head0 = r == 0 ? 0 : h;
  c.tp.head1 = r == 0 ? h : cfg.n_heads;
  c.tp.ffn0 = r == 0 ? 0 : f;
  c.tp.ffn1 = r == 0 ? f : cfg.mlp_hidden;
  if (floor > 0) {
    c.tp.extra = FlexData::make_object();
    c.tp.extra.as_object().insert_or_assign(
        tp_extra::kMinBandRows, FlexData::make_int(floor));
  }
  return c;
}

// Two ranks of `cfg` over one PairLink, preloaded or STREAMED (residency
// may keep every block), their forwards run at once `forwards` times.
// Each rank's last velocity, rank 0's exchanges, what it holds after
// (resident_bytes) against what the plan books for it (tp_held_bytes),
// and the blocks it kept.
struct F3Split {
  std::vector<float> v[2];
  std::uint64_t exchanges = 0;
  int out_of_step = 0, mismatched = 0;
  std::size_t held = 0, booked = 0;
  int kept = 0;
};
bool
flux3_split_(MetalCompute* mc, const F3Small& s, const F3::Config& cfg,
             int h, bool stream, int forwards, int floor, F3Split* out)
{
  auto link = std::make_shared<PairLink>();
  std::shared_ptr<PairChannel> ch[2];
  std::unique_ptr<F3> m[2];
  for (int r = 0; r < 2; ++r) {
    const F3::Config c = f3_rank_(cfg, r, h, floor);
    m[r] = F3::load(s.dir.string(), mc, c, stream);
    if (m[r] == nullptr) { return false; }
    if (stream) {
      m[r]->set_residency_reserve(test_residency_reserve(1ull << 30));
      m[r]->set_residency_schedule(forwards);
    }
    ch[r] = std::make_shared<PairChannel>(link, r);
    std::string err;
    if (!m[r]->begin_tp_generation(ch[r], &err)) {
      std::printf("  begin: %s\n", err.c_str());
      return false;
    }
  }
  for (int k = 0; k < forwards; ++k) {
    std::string e0, e1;
    bool ok1 = false;
    std::thread t(
        [&]() { ok1 = flux3_forward_(mc, *m[1], s.jd, &out->v[1], &e1); });
    const bool ok0 = flux3_forward_(mc, *m[0], s.jd, &out->v[0], &e0);
    t.join();
    if (!ok0 || !ok1) {
      std::printf("  rank 0: %s / rank 1: %s\n", e0.c_str(), e1.c_str());
      return false;
    }
  }
  // The exchange buffers: held through the picture, gone with its
  // channel (end_tp_generation). MUTATION: kept -- fails.
  const std::size_t exch_held = m[0]->tp_exchange_bytes();
  for (auto& mm : m) { mm->end_tp_generation(); }
  const std::size_t exch_left =
      m[0]->tp_exchange_bytes() + m[1]->tp_exchange_bytes();
  if (exch_held == 0 || exch_left != 0) {
    std::printf("  the exchange buffers: %zu bytes through the picture, "
                "%zu after its channels went (want some, then none)\n",
                exch_held, exch_left);
    return false;
  }
  out->exchanges = ch[0]->exchanges.load();
  out->out_of_step = ch[0]->out_of_step.load() + ch[1]->out_of_step.load();
  out->mismatched = link->mismatched.load();
  out->held = (std::size_t)m[0]->resident_bytes();
  out->booked =
      F3::tp_held_bytes(s.dir.string(), f3_rank_(cfg, 0, h, floor));
  out->kept = m[0]->resident_block_count();
  return true;
}

// The exchanges a split forward of flux3_forward_'s geometry makes: a
// block's bands (tp_row_bands) over its stack's rows -- the image's, the
// text's, the reference's, then the joint's.
std::uint64_t
flux3_exchanges_(const F3Small& s, int floor, int forced)
{
  const int img = 16 * 16, txt = 40, ref = 8 * 8;
  auto n = [&](int rows) {
    return (std::uint64_t)tp_row_bands(rows, floor, forced).size();
  };
  return (std::uint64_t)s.modes * (n(img) + n(txt) + n(ref)) +
         (std::uint64_t)s.joint * n(txt + img + ref);
}

}  // namespace

// A FLUX.3 DiT SPLIT between two ranks is the unsplit one: each holds
// its share of every block's heads (q / k / v by their rows, the out
// projection by its columns) and feed-forward rows (linear_in's gate and
// up halves alike, linear_out by its columns), and a block's attention
// and feed-forward -- one norm, one gate over their sum -- leave ONE
// partial a block, written straight into an exchange buffer and posted
// (the encoder never waits), and added before the gate. A random small
// model, dense bf16 and an 8-bit affine pack, 4 of 6 heads on rank 0:
//   - preloaded, whole blocks: both ranks' velocities identical, the
//     split within the two roundings of a partial where the unsplit
//     model rounds once (MEASURED on the M5 Max: 2.1e-3 bf16, 2.2e-3
//     the pack), nine exchanges for nine blocks; and what the rank holds
//     is what the plan books for it (tp_held_bytes), to the byte;
//   - STREAMED, then kept: the same velocity BIT FOR BIT, every block
//     kept by the second forward, the same bytes held;
//   - its share's rows READ alone (q / k / v and linear_in) against the
//     whole weights read and cut (VPIPE_TP_WHOLE_READ): the same bits;
//   - BANDED (a floor of 100 rows on the share): each block's rows in
//     bands, a band's partial crossing under the next band's work --
//     within the bar, one exchange a band; and overlapped against band
//     after band (VPIPE_FLUX3_TP_LAG=0), the same bits.
// MUTATIONS, each failing it: no exchange 2.8e-2 (the ranks differ too),
// the up rows cut from the gate's half 9.5e-2, the out projection cut by
// its rows 1.8e-2.
TEST(tensor_parallel, a_split_flux3_forward_is_the_unsplit_one)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  F3Small s;
  s.dir = fs::temp_directory_path() / "vpipe_flux3_tp_small";
  constexpr double kBar = 1e-2;
  const int h = 4;
  for (const bool w8 : {false, true}) {
    ASSERT_TRUE(write_flux3_small_(s, w8));
    F3::Config cfg;
    ASSERT_TRUE(F3::Config::read_dims(s.dir.string(), &cfg));
    cfg.autotune = false;
    std::vector<float> whole;
    {
      auto m = F3::load(s.dir.string(), mc, cfg, false);
      ASSERT_TRUE(m != nullptr);
      std::string err;
      ASSERT_TRUE(flux3_forward_(mc, *m, s.jd, &whole, &err));
    }
    const char* arm = w8 ? "w8 pack" : "bf16";
    auto check = [&](const char* what, const F3Split& x,
                     std::uint64_t want) {
      const double d = rel_l2_(x.v[0], whole);
      std::printf("  flux3 %s, %s: split vs unsplit rel-L2 %.3e, ranks %s, "
                  "%llu exchanges (want %llu)\n", arm, what, d,
                  same_bits_(x.v[0], x.v[1]) ? "IDENTICAL" : "DIFFER",
                  (unsigned long long)x.exchanges,
                  (unsigned long long)want);
      EXPECT_TRUE(x.out_of_step == 0 && x.mismatched == 0);
      EXPECT_TRUE(x.exchanges == want);
      EXPECT_TRUE(same_bits_(x.v[0], x.v[1]));
      EXPECT_TRUE(d < kBar);
    };
    // Whole blocks, preloaded.
    F3Split pre;
    ASSERT_TRUE(flux3_split_(mc, s, cfg, h, false, 1, 0, &pre));
    check("preloaded", pre, flux3_exchanges_(s, 0, 0));
    std::printf("  flux3 %s: rank 0 holds %zu bytes, booked %zu\n", arm,
                pre.held, pre.booked);
    EXPECT_TRUE(pre.held == pre.booked && pre.booked > 0);
    // Streamed, then kept.
    F3Split str;
    ASSERT_TRUE(flux3_split_(mc, s, cfg, h, true, 2, 0, &str));
    std::printf("  flux3 %s, streamed: %s the preloaded rank's; %d of %d "
                "blocks kept, %zu bytes held\n", arm,
                same_bits_(str.v[0], pre.v[0]) ? "IDENTICAL to" : "DIFFERS "
                "from", str.kept, 3 * s.modes + s.joint, str.held);
    EXPECT_TRUE(same_bits_(str.v[0], pre.v[0]));
    EXPECT_TRUE(str.kept == 3 * s.modes + s.joint);
    EXPECT_TRUE(str.held == pre.booked);
    // The whole weights read and cut: the same share.
    {
      EnvScope e("VPIPE_TP_WHOLE_READ", "1");
      F3Split wr;
      ASSERT_TRUE(flux3_split_(mc, s, cfg, h, false, 1, 0, &wr));
      std::printf("  flux3 %s, whole weights read: %s the row reads\n", arm,
                  same_bits_(wr.v[0], pre.v[0]) ? "IDENTICAL to"
                                                : "DIFFERS from");
      EXPECT_TRUE(same_bits_(wr.v[0], pre.v[0]));
    }
    // Banded, overlapped and in turn.
    F3Split bd;
    ASSERT_TRUE(flux3_split_(mc, s, cfg, h, false, 1, 100, &bd));
    check("banded (floor 100)", bd, flux3_exchanges_(s, 100, 0));
    {
      EnvScope e("VPIPE_FLUX3_TP_LAG", "0");
      F3Split b0;
      ASSERT_TRUE(flux3_split_(mc, s, cfg, h, false, 1, 100, &b0));
      std::printf("  flux3 %s, banded in turn: %s the overlapped bands\n",
                  arm, same_bits_(b0.v[0], bd.v[0]) ? "IDENTICAL to"
                                                    : "DIFFERS from");
      EXPECT_TRUE(same_bits_(b0.v[0], bd.v[0]));
    }
  }
  std::error_code ec;
  fs::remove_all(s.dir, ec);
}

// ---- Qwen-Image-2.1: two partials a block ------------------------------

namespace {

// A small random Qwen-Image-2.1 DiT on disk (QwenImage21Transformer2DModel's
// names and config): 8 heads of 128, two blocks, every projection bf16 --
// or, `w8`, the block projections an MLX affine 8-bit pack in groups of
// 64, the model-level ones dense.
bool
write_qi21_small_(const fs::path& dir, bool w8)
{
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  const int H = 1024, F = 3 * H, C = 256, IN = 64, L = 2;
  {
    std::ofstream o(dir / "config.json");
    o << "{\"_class_name\": \"QwenImage21Transformer2DModel\", "
      << "\"attention_head_dim\": 128, \"axes_dims_rope\": [16, 56, 56], "
      << "\"context_in_dim\": " << C << ", \"in_channels\": " << IN
      << ", \"num_attention_heads\": 8, \"num_layers\": " << L
      << ", \"out_channels\": " << IN << ", \"patch_size\": 1, "
      << "\"mlp_ratio\": 3, \"eps\": 1e-06, \"causal_condition\": true}";
  }
  genai::SafetensorsWriter wr(dir.string());
  std::uint64_t st = 0xD1B54A32D192ED03ull ^ (w8 ? 1 : 0);
  auto rnd = [&]() {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    return (float)((double)(st >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
  };
  auto dense = [&](const std::string& n, std::vector<std::int64_t> shape,
                   float sc, float centre = 0.0f) {
    std::size_t k = 1;
    for (auto d : shape) { k *= (std::size_t)d; }
    std::vector<std::uint16_t> v(k);
    for (auto& x : v) { x = bf16_(centre + rnd() * sc); }
    return wr.add(n, "BF16", shape, v.data(), v.size() * 2);
  };
  auto packed = [&](const std::string& n, int N, int K, float sc) {
    const int g = 64;
    std::vector<std::uint32_t> codes((std::size_t)N * K / 4);
    std::vector<std::uint16_t> scales((std::size_t)N * K / g);
    std::vector<std::uint16_t> biases(scales.size());
    std::vector<float> row((std::size_t)K);
    for (int r = 0; r < N; ++r) {
      for (auto& x : row) { x = rnd() * sc; }
      for (int g0 = 0; g0 < K; g0 += g) {
        float lo = row[(std::size_t)g0], hi = lo;
        for (int i = g0; i < g0 + g; ++i) {
          lo = std::min(lo, row[(std::size_t)i]);
          hi = std::max(hi, row[(std::size_t)i]);
        }
        const float s8 = f32_(bf16_(std::max(hi - lo, 1e-6f) / 255.0f));
        const float b8 = f32_(bf16_(lo));
        const std::size_t gi = ((std::size_t)r * K + g0) / g;
        scales[gi] = bf16_(s8);
        biases[gi] = bf16_(b8);
        for (int i = g0; i < g0 + g; ++i) {
          const int q = std::clamp(
              (int)std::lround((row[(std::size_t)i] - b8) / s8), 0, 255);
          const std::size_t e = (std::size_t)r * K + i;
          codes[e / 4] |= (std::uint32_t)q << (8 * (e % 4));
        }
      }
    }
    return wr.add(n + ".weight", "U32", {N, K / 4}, codes.data(),
                  codes.size() * 4) &&
           wr.add(n + ".scales", "BF16", {N, K / g}, scales.data(),
                  scales.size() * 2) &&
           wr.add(n + ".biases", "BF16", {N, K / g}, biases.data(),
                  biases.size() * 2);
  };
  auto proj = [&](const std::string& n, int N, int K, float sc) {
    return w8 ? packed(n, N, K, sc) : dense(n + ".weight", {N, K}, sc);
  };
  bool ok = dense("img_in.weight", {H, IN}, 0.05f) &&
            dense("norm_out.linear.weight", {H, H}, 0.02f) &&
            dense("proj_out.weight", {IN, H}, 0.05f) &&
            dense("time_text_embed.timestep_embedder.linear_1.weight",
                  {H, 256}, 0.05f) &&
            dense("time_text_embed.timestep_embedder.linear_2.weight",
                  {H, H}, 0.05f) &&
            dense("txt_in.in_layer.weight", {H, C}, 0.05f) &&
            dense("txt_in.out_layer.weight", {H, H}, 0.05f) &&
            dense("txt_in.text_norm.weight", {C}, 0.1f) &&
            dense("modulation.1.weight", {4 * H, H}, 0.02f);
  for (int l = 0; l < L; ++l) {
    const std::string p = "transformer_blocks." + std::to_string(l) + ".";
    ok = ok && dense(p + "attn.norm_q.weight", {128}, 0.1f, 1.0f) &&
         dense(p + "attn.norm_k.weight", {128}, 0.1f, 1.0f) &&
         proj(p + "attn.to_q", H, H, 0.04f) &&
         proj(p + "attn.to_k", H, H, 0.04f) &&
         proj(p + "attn.to_v", H, H, 0.04f) &&
         proj(p + "attn.to_out.0", H, H, 0.04f) &&
         proj(p + "img_mlp.gate_layer", F, H, 0.04f) &&
         proj(p + "img_mlp.proj", F, H, 0.04f) &&
         proj(p + "img_mlp.out", H, F, 0.03f);
  }
  return wr.close() && ok;
}

// A picture's first two steps of `m`: the prefix's K / V kept on the
// first (kFill) and read on the second (kCached); each step's target
// rows as f32. Text, one 8 x 8 condition image, text, a 16 x 16 target.
// `neg`: classifier-free guidance's NEGATIVE beside it, as
// generate_qwen_image21_ runs one -- a sequence of its own (other text
// lengths over the same image blocks and latents) with a K / V cache of
// its own, its forward after the positive's each step.
struct Qi21Steps {
  std::vector<float> v[2];
  std::vector<float> n[2];   // the negative's, with `neg`
};
// The positive's and the negative's text slots and layouts.
std::vector<std::uint8_t>
qi21_slots_(bool neg)
{
  const int t0 = neg ? 2 : 5, t1 = neg ? 6 : 3;
  std::vector<std::uint8_t> slot((std::size_t)t0, 0);
  slot.insert(slot.end(), 16, 1);
  slot.insert(slot.end(), (std::size_t)t1, 0);
  slot.insert(slot.end(), 64, 1);
  return slot;
}
bool
qi21_layouts_(qi21::Layout* lay, qi21::Layout* lay_neg, std::string* err)
{
  const std::vector<qi21::ImgBlock> blocks = {{1, 8, 8}, {1, 16, 16}};
  return qi21::build_layout(qi21_slots_(false), {}, blocks, lay, err) &&
         qi21::build_layout(qi21_slots_(true), {}, blocks, lay_neg, err);
}
bool
qi21_steps_(MetalCompute* mc, MetalQwenImage21Transformer& m, Qi21Steps* out,
            std::string* err, bool neg = false)
{
  using T = MetalQwenImage21Transformer;
  const std::vector<std::uint8_t> slot = qi21_slots_(false);
  const std::vector<std::uint8_t> slot_neg = qi21_slots_(true);
  qi21::Layout lay, lay_neg;
  if (!qi21_layouts_(&lay, &lay_neg, err)) { return false; }
  std::uint64_t st = 777;
  auto buf = [&](std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      st ^= st << 13; st ^= st >> 7; st ^= st << 17;
      d[i] = bf16_((float)((double)(st >> 11) / 9007199254740992.0) - 0.5f);
    }
    return b;
  };
  const T::Config& c = m.config();
  SharedBuffer lat = buf((std::size_t)lay.image_len * c.in_channels);
  SharedBuffer txt = buf(slot.size() * (std::size_t)c.txt_dim);
  SharedBuffer txt_neg = buf(slot_neg.size() * (std::size_t)c.txt_dim);
  T::KvCache kv, kv_neg;
  // [target_len, out_channels] bf16, as the forward returns it (and the
  // stage reads it).
  auto run = [&](const SharedBuffer& tx, const qi21::Layout& ly,
                 T::KvCache& cache, int k, std::vector<float>* v) {
    T::Request req;
    req.latents = &lat;
    req.txt = &tx;
    req.layout = &ly;
    req.timestep = k == 0 ? 0.9f : 0.6f;
    req.kv = &cache;
    req.kv_mode = k == 0 ? T::KvMode::kFill : T::KvMode::kCached;
    SharedBuffer o = m.forward(req, err);
    if (o.empty()) { return false; }
    const auto* p = static_cast<const std::uint16_t*>(o.contents());
    const std::size_t n = (std::size_t)ly.target_len * c.out_channels;
    v->resize(n);
    for (std::size_t i = 0; i < n; ++i) { (*v)[i] = f32_(p[i]); }
    return true;
  };
  for (int k = 0; k < 2; ++k) {
    if (!run(txt, lay, kv, k, &out->v[k])) { return false; }
    if (neg && !run(txt_neg, lay_neg, kv_neg, k, &out->n[k])) {
      return false;
    }
  }
  return true;
}

using QI = MetalQwenImage21Transformer;

// Rank `r` of a two-rank split of `cfg`: heads [0, h) and [h, n) and the
// matching feed-forward rows; `floor` > 0 the share's band floor.
QI::Config
qi21_rank_(const QI::Config& cfg, int r, int h, int floor)
{
  QI::Config c = cfg;
  const int F = cfg.hidden * cfg.mlp_ratio;
  const int f = F / cfg.n_heads * h;
  c.tp = TpShare{};
  c.tp.rank = r;
  c.tp.ranks = 2;
  c.tp.n_heads = cfg.n_heads;
  c.tp.n_ffn = F;
  c.tp.head0 = r == 0 ? 0 : h;
  c.tp.head1 = r == 0 ? h : cfg.n_heads;
  c.tp.ffn0 = r == 0 ? 0 : f;
  c.tp.ffn1 = r == 0 ? f : F;
  if (floor > 0) {
    c.tp.extra = FlexData::make_object();
    c.tp.extra.as_object().insert_or_assign(
        tp_extra::kMinBandRows, FlexData::make_int(floor));
  }
  return c;
}

// Two ranks of `cfg` over one PairLink, preloaded or STREAMED (residency
// may keep both blocks), each running qi21_steps_ at once -- twice when
// streamed, the second from what residency kept. Each rank's steps,
// rank 0's exchanges and int8-layout projections, what it holds after
// (resident_bytes) against what the plan books for it (tp_held_bytes),
// and the blocks it kept.
struct Qi21Split {
  Qi21Steps s[2];
  std::uint64_t exchanges = 0;
  int out_of_step = 0, mismatched = 0;
  int fast = 0, kept = 0;
  std::size_t held = 0, booked = 0;
};
bool
qi21_split_(MetalCompute* mc, const fs::path& dir, const QI::Config& cfg,
            int h, bool stream, int floor, bool neg, Qi21Split* out)
{
  auto link = std::make_shared<PairLink>();
  std::shared_ptr<PairChannel> ch[2];
  std::unique_ptr<QI> m[2];
  const int runs = stream ? 2 : 1;
  for (int r = 0; r < 2; ++r) {
    m[r] = QI::load(dir.string(), mc, qi21_rank_(cfg, r, h, floor), stream);
    if (m[r] == nullptr) { return false; }
    if (stream) {
      m[r]->set_residency_reserve(test_residency_reserve(1ull << 30));
      m[r]->set_residency_schedule(2 * runs * (neg ? 2 : 1));
    }
    ch[r] = std::make_shared<PairChannel>(link, r);
    std::string err;
    if (!m[r]->begin_tp_generation(ch[r], &err)) {
      std::printf("  begin: %s\n", err.c_str());
      return false;
    }
  }
  for (int k = 0; k < runs; ++k) {
    std::string e0, e1;
    bool ok1 = false;
    std::thread t(
        [&]() { ok1 = qi21_steps_(mc, *m[1], &out->s[1], &e1, neg); });
    const bool ok0 = qi21_steps_(mc, *m[0], &out->s[0], &e0, neg);
    t.join();
    if (!ok0 || !ok1) {
      std::printf("  rank 0: %s / rank 1: %s\n", e0.c_str(), e1.c_str());
      return false;
    }
  }
  out->fast = m[0]->fast_weights(8);
  // The exchange buffers: held through the picture, gone with its
  // channel (end_tp_generation). MUTATION: kept -- fails.
  const std::size_t exch_held = m[0]->tp_exchange_bytes();
  for (auto& mm : m) { mm->end_tp_generation(); }
  const std::size_t exch_left =
      m[0]->tp_exchange_bytes() + m[1]->tp_exchange_bytes();
  if (exch_held == 0 || exch_left != 0) {
    std::printf("  the exchange buffers: %zu bytes through the picture, "
                "%zu after its channels went (want some, then none)\n",
                exch_held, exch_left);
    return false;
  }
  out->exchanges = ch[0]->exchanges.load();
  out->out_of_step = ch[0]->out_of_step.load() + ch[1]->out_of_step.load();
  out->mismatched = link->mismatched.load();
  out->held = (std::size_t)m[0]->resident_bytes();
  out->booked =
      QI::tp_held_bytes(dir.string(), qi21_rank_(cfg, 0, h, floor), mc);
  out->kept = m[0]->resident_block_count();
  return true;
}

}  // namespace

// A QWEN-IMAGE-2.1 DiT SPLIT between two ranks is the unsplit one: each
// holds its share of every block's heads (to_q / to_k / to_v by their
// rows, to_out by its columns) and feed-forward rows (gate_layer and
// proj by their rows, out by its columns), and its two sublayers --
// gated apart -- leave TWO partials a block, each written straight into
// an exchange buffer, posted, and added before its gate. The picture's
// first two steps -- the prefix's K / V kept, then read -- on a random
// small model, 4 of 8 heads on rank 0, with classifier-free guidance's
// NEGATIVE beside them (its own sequence and K / V cache):
//   - bf16, and an 8-bit affine pack, preloaded, whole blocks: the split
//     within the bf16 rounding of two partials where the unsplit model
//     rounds one (MEASURED on the M4 Pro: 2.6-2.8e-3 bf16, 2.4-2.6e-3
//     the pack, positive and negative alike), and what the rank holds
//     what the plan books for it, to the byte. (THE OUTPUT IS bf16.
//     Read as IEEE f16, as this test first read it, bf16's rounding
//     noise read 6-10x too large -- 2.1-2.6% -- and a uniform 20% error
//     in magnitude 2%; the bar was then set from a "reordering floor"
//     that was the misreading.) MUTATIONS 20-44% (no exchange, the up
//     rows cut as the heads', the out projection cut by its rows);
//   - STREAMED, then kept: the same steps BIT FOR BIT, both blocks kept;
//   - the share's rows READ alone against the whole weights read and
//     cut (VPIPE_TP_WHOLE_READ): the same bits;
//   - BANDED (a floor of 100 rows): two exchanges a band, within the
//     bar; overlapped against band after band (VPIPE_QWEN_IMAGE21_TP_LAG
//     =0), the same bits;
//   - the same under int8 tiers (Per Token on the bf16, Fast on the
//     pack; matrix cores), a rank's share laid out for the tier within
//     its own columns: its activations are then quantized over its
//     columns where the whole model quantizes over all of them, so the
//     bar is the tier's own -- the split within half again of what the
//     tier is from the plain model, and the plain bar at least.
// Both ranks' outputs identical in every arm, every step; eight
// exchanges a step whole (two blocks, two sublayers, the positive and
// the negative).
TEST(tensor_parallel, a_split_qwen_image_21_forward_is_the_unsplit_one)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const fs::path dir = fs::temp_directory_path() / "vpipe_qi21_tp_small";
  constexpr double kBar = 1e-2;
  const int h = 4;
  struct Arm { const char* name; bool w8; int tier; };
  const Arm arms[] = {{"bf16", false, -1}, {"w8 pack", true, -1},
                      {"bf16, int8 Per Token", false, 2},
                      {"w8 pack, int8 Fast", true, 1}};
  bool written_w8 = false, written = false;
  Qi21Steps plain[2];   // [w8], the whole model untiered
  for (const Arm& a : arms) {
    if (!written || written_w8 != a.w8) {
      ASSERT_TRUE(write_qi21_small_(dir, a.w8));
      written = true;
      written_w8 = a.w8;
    }
    QI::Config cfg;
    ASSERT_TRUE(QI::Config::read_dims(dir.string(), &cfg));
    cfg.autotune = false;
    if (a.tier >= 0) {
      if (!mc->supports_matrix_cores()) {
        std::printf("  %-22s SKIP: no matrix cores\n", a.name);
        continue;
      }
      cfg.i8_gemm = true;
      cfg.i8_tier = a.tier;
    }
    Qi21Steps whole;
    {
      auto m = QI::load(dir.string(), mc, cfg, false);
      ASSERT_TRUE(m != nullptr);
      if (a.tier >= 0) { EXPECT_TRUE(m->fast_tier()); }
      std::string err;
      ASSERT_TRUE(qi21_steps_(mc, *m, &whole, &err, /*neg=*/true));
    }
    if (a.tier < 0) { plain[a.w8] = whole; }
    // The arm's bar for one output: the plain bar, or under a tier the
    // tier's own distance from the plain model, half again.
    auto bar_for = [&](const std::vector<float>& w,
                       const std::vector<float>& p) {
      return a.tier < 0 ? kBar : std::max(kBar, 1.5 * rel_l2_(w, p));
    };
    auto check = [&](const char* what, const Qi21Split& x,
                     std::uint64_t want) {
      for (int k = 0; k < 2; ++k) {
        const struct {
          const char* side;
          const std::vector<float>* got[2];
          const std::vector<float>* w;
          const std::vector<float>* p;
        } sides[] = {{"positive", {&x.s[0].v[k], &x.s[1].v[k]},
                      &whole.v[k], &plain[a.w8].v[k]},
                     {"negative", {&x.s[0].n[k], &x.s[1].n[k]},
                      &whole.n[k], &plain[a.w8].n[k]}};
        for (const auto& sd : sides) {
          const double d = rel_l2_(*sd.got[0], *sd.w);
          const double bar = bar_for(*sd.w, *sd.p);
          std::printf("  qi21 %-22s %-18s step %d (%s) %s: split vs "
                      "unsplit rel-L2 %.3e (bar %.3e), ranks %s\n", a.name,
                      what, k, k == 0 ? "fill" : "cached", sd.side, d, bar,
                      same_bits_(*sd.got[0], *sd.got[1]) ? "IDENTICAL"
                                                         : "DIFFER");
          EXPECT_TRUE(same_bits_(*sd.got[0], *sd.got[1]));
          EXPECT_TRUE(d < bar);
        }
      }
      std::printf("  qi21 %-22s %-18s %llu exchanges (want %llu); rank 0 "
                  "holds %d int8-layout projections\n", a.name, what,
                  (unsigned long long)x.exchanges,
                  (unsigned long long)want, x.fast);
      EXPECT_TRUE(x.exchanges == want);
      EXPECT_TRUE(x.out_of_step == 0 && x.mismatched == 0);
      // Under a tier every projection of the share takes its layout: 7
      // a block, 2 blocks.
      if (a.tier >= 0) { EXPECT_TRUE(x.fast == 14); }
    };
    auto same = [](const Qi21Split& x, const Qi21Split& y) {
      bool ok = true;
      for (int k = 0; k < 2; ++k) {
        ok = ok && same_bits_(x.s[0].v[k], y.s[0].v[k]) &&
             same_bits_(x.s[0].n[k], y.s[0].n[k]);
      }
      return ok;
    };
    // Two steps, positive and negative: the forwards' bands (tp_row_
    // bands) over their rows -- the joint's filling, the target's cached.
    qi21::Layout lay, lay_neg;
    {
      std::string lerr;
      ASSERT_TRUE(qi21_layouts_(&lay, &lay_neg, &lerr));
    }
    auto exchanges = [&](int floor) {
      std::uint64_t n = 0;
      for (const int rows : {lay.joint_len, lay.target_len,
                             lay_neg.joint_len, lay_neg.target_len}) {
        n += 2 * (std::uint64_t)cfg.n_layers *
             tp_row_bands(rows, floor).size();
      }
      return n;
    };
    // Whole blocks, preloaded.
    Qi21Split pre;
    ASSERT_TRUE(qi21_split_(mc, dir, cfg, h, false, 0, true, &pre));
    check("preloaded", pre, exchanges(0));
    std::printf("  qi21 %-22s rank 0 holds %zu bytes, booked %zu\n",
                a.name, pre.held, pre.booked);
    EXPECT_TRUE(pre.held == pre.booked && pre.booked > 0);
    // Streamed, then kept.
    Qi21Split str;
    ASSERT_TRUE(qi21_split_(mc, dir, cfg, h, true, 0, true, &str));
    std::printf("  qi21 %-22s streamed: %s the preloaded rank's; %d of %d "
                "blocks kept, %zu bytes held\n", a.name,
                same(str, pre) ? "IDENTICAL to" : "DIFFERS from", str.kept,
                cfg.n_layers, str.held);
    EXPECT_TRUE(same(str, pre));
    EXPECT_TRUE(str.kept == cfg.n_layers);
    EXPECT_TRUE(str.held == pre.booked);
    // The whole weights read and cut: the same share.
    {
      EnvScope e("VPIPE_TP_WHOLE_READ", "1");
      Qi21Split wr;
      ASSERT_TRUE(qi21_split_(mc, dir, cfg, h, false, 0, true, &wr));
      std::printf("  qi21 %-22s whole weights read: %s the row reads\n",
                  a.name, same(wr, pre) ? "IDENTICAL to" : "DIFFERS from");
      EXPECT_TRUE(same(wr, pre));
    }
    // Banded, overlapped and in turn.
    Qi21Split bd;
    ASSERT_TRUE(qi21_split_(mc, dir, cfg, h, false, 100, true, &bd));
    check("banded (floor 100)", bd, exchanges(100));
    {
      EnvScope e("VPIPE_QWEN_IMAGE21_TP_LAG", "0");
      Qi21Split b0;
      ASSERT_TRUE(qi21_split_(mc, dir, cfg, h, false, 100, true, &b0));
      std::printf("  qi21 %-22s banded in turn: %s the overlapped bands\n",
                  a.name, same(b0, bd) ? "IDENTICAL to" : "DIFFERS from");
      EXPECT_TRUE(same(b0, bd));
    }
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ---- Krea-2: two partials a block, a share whole kv groups ---------------

namespace {

using K2 = MetalKrea2Transformer;

// A small Krea-2 DiT's shape: 8 heads of 128 over 2 kv heads (GQA by
// four, as the release's 48 over 12), two main blocks, a 2,048-row
// feed-forward; the text-fusion tower its projector and txt_in alone (no
// layerwise or refiner block: forward_dit never reads them).
K2::Config
k2_small_cfg_()
{
  K2::Config c;
  c.hidden = 1024;
  c.n_heads = 8;
  c.n_kv_heads = 2;
  c.head_dim = 128;
  c.ffn = 2048;
  c.n_layers = 2;
  c.text_hidden = 128;
  c.n_text_layers = 2;
  c.n_layerwise = 0;
  c.n_refiner = 0;
  c.autotune = false;
  return c;
}

// That model on disk under Krea2Transformer2DModel's names, every
// projection bf16 -- or, `w8`, the main blocks' projections an MLX affine
// 8-bit pack in groups of 64, its scales and biases F16 as a model-
// quantize'd pack stores them (config.json's `quantization`), the rest
// dense. Norms zero-centred, as the checkpoint stores them.
bool
write_krea2_small_(const fs::path& dir, bool w8)
{
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  const K2::Config c = k2_small_cfg_();
  const int H = c.hidden, F = c.ffn, IC = c.in_channels, TD = c.timestep_dim;
  const int qd = c.n_heads * c.head_dim, kd = c.n_kv_heads * c.head_dim;
  const int TH = c.text_hidden;
  {
    std::ofstream o(dir / "config.json");
    o << "{\"_class_name\": \"Krea2Transformer2DModel\"";
    if (w8) { o << ", \"quantization\": {\"bits\": 8, \"group_size\": 64}"; }
    o << "}";
  }
  genai::SafetensorsWriter wr(dir.string());
  std::uint64_t st = 0x9E3779B97F4A7C15ull ^ (w8 ? 1 : 0);
  auto rnd = [&]() {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    return (float)((double)(st >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
  };
  auto dense = [&](const std::string& n, std::vector<std::int64_t> shape,
                   float sc) {
    std::size_t k = 1;
    for (auto d : shape) { k *= (std::size_t)d; }
    std::vector<std::uint16_t> v(k);
    for (auto& x : v) { x = bf16_(rnd() * sc); }
    return wr.add(n, "BF16", shape, v.data(), v.size() * 2);
  };
  auto f16 = [](float f) {
    const _Float16 h = (_Float16)f;
    std::uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
  };
  auto packed = [&](const std::string& n, int N, int K, float sc) {
    const int g = 64;
    std::vector<std::uint32_t> codes((std::size_t)N * K / 4);
    std::vector<std::uint16_t> scales((std::size_t)N * K / g);
    std::vector<std::uint16_t> biases(scales.size());
    std::vector<float> row((std::size_t)K);
    for (int r = 0; r < N; ++r) {
      for (auto& x : row) { x = rnd() * sc; }
      for (int g0 = 0; g0 < K; g0 += g) {
        float lo = row[(std::size_t)g0], hi = lo;
        for (int i = g0; i < g0 + g; ++i) {
          lo = std::min(lo, row[(std::size_t)i]);
          hi = std::max(hi, row[(std::size_t)i]);
        }
        const float s8 = (float)(_Float16)(std::max(hi - lo, 1e-6f) / 255.0f);
        const float b8 = (float)(_Float16)lo;
        const std::size_t gi = ((std::size_t)r * K + g0) / g;
        scales[gi] = f16(s8);
        biases[gi] = f16(b8);
        for (int i = g0; i < g0 + g; ++i) {
          const int q = std::clamp(
              (int)std::lround((row[(std::size_t)i] - b8) / s8), 0, 255);
          const std::size_t e = (std::size_t)r * K + i;
          codes[e / 4] |= (std::uint32_t)q << (8 * (e % 4));
        }
      }
    }
    return wr.add(n + ".weight", "U32", {N, K / 4}, codes.data(),
                  codes.size() * 4) &&
           wr.add(n + ".scales", "F16", {N, K / g}, scales.data(),
                  scales.size() * 2) &&
           wr.add(n + ".biases", "F16", {N, K / g}, biases.data(),
                  biases.size() * 2);
  };
  auto proj = [&](const std::string& n, int N, int K, float sc) {
    return w8 ? packed(n, N, K, sc) : dense(n + ".weight", {N, K}, sc);
  };
  bool ok =
      dense("text_fusion.projector.weight", {1, c.n_text_layers}, 0.5f) &&
      dense("txt_in.norm.weight", {TH}, 0.1f) &&
      dense("txt_in.linear_1.weight", {H, TH}, 0.05f) &&
      dense("txt_in.linear_1.bias", {H}, 0.02f) &&
      dense("txt_in.linear_2.weight", {H, H}, 0.03f) &&
      dense("txt_in.linear_2.bias", {H}, 0.02f) &&
      dense("img_in.weight", {H, IC}, 0.08f) &&
      dense("img_in.bias", {H}, 0.02f) &&
      dense("time_embed.linear_1.weight", {H, TD}, 0.05f) &&
      dense("time_embed.linear_1.bias", {H}, 0.02f) &&
      dense("time_embed.linear_2.weight", {H, H}, 0.03f) &&
      dense("time_embed.linear_2.bias", {H}, 0.02f) &&
      dense("time_mod_proj.weight", {6 * H, H}, 0.02f) &&
      dense("time_mod_proj.bias", {6 * H}, 0.02f) &&
      dense("final_layer.scale_shift_table", {2, H}, 0.1f) &&
      dense("final_layer.norm.weight", {H}, 0.1f) &&
      dense("final_layer.linear.weight", {IC, H}, 0.03f) &&
      dense("final_layer.linear.bias", {IC}, 0.02f);
  for (int l = 0; l < c.n_layers; ++l) {
    const std::string p = "transformer_blocks." + std::to_string(l) + ".";
    ok = ok && dense(p + "scale_shift_table", {6, H}, 0.1f) &&
         dense(p + "norm1.weight", {H}, 0.1f) &&
         dense(p + "norm2.weight", {H}, 0.1f) &&
         dense(p + "attn.norm_q.weight", {c.head_dim}, 0.1f) &&
         dense(p + "attn.norm_k.weight", {c.head_dim}, 0.1f) &&
         proj(p + "attn.to_q", qd, H, 0.04f) &&
         proj(p + "attn.to_k", kd, H, 0.04f) &&
         proj(p + "attn.to_v", kd, H, 0.04f) &&
         proj(p + "attn.to_gate", qd, H, 0.04f) &&
         proj(p + "attn.to_out.0", H, qd, 0.04f) &&
         proj(p + "ff.gate", F, H, 0.04f) &&
         proj(p + "ff.up", F, H, 0.04f) &&
         proj(p + "ff.down", H, F, 0.03f);
  }
  return wr.close() && ok;
}

// One forward of `m` over random f16 text and latents (a 16 x 16 latent
// grid beside 40 prompt rows), velocity as f32 -- or, `tap` >= 0, the
// joint residual after that block (a real checkpoint's few blocks).
struct K2Geom {
  int text = 40, grid = 16;
};
bool
krea2_forward_(MetalCompute* mc, K2& m, const K2Geom& gm, int tap,
               std::vector<float>* out, std::string* err)
{
  const K2::Config& c = m.config();
  const int img = gm.grid * gm.grid;
  std::uint64_t st = 4242;
  auto buf = [&](std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      st ^= st << 13; st ^= st >> 7; st ^= st << 17;
      d[i] = (_Float16)((float)((double)(st >> 11) / 9007199254740992.0) -
                        0.5f);
    }
    return b;
  };
  SharedBuffer txt = buf((std::size_t)gm.text * c.hidden);
  SharedBuffer lat = buf((std::size_t)img * c.in_channels);
  SharedBuffer o = m.forward_dit(txt, gm.text, lat, img, gm.grid, gm.grid,
                                 0.6f, tap);
  if (o.empty()) {
    if (err != nullptr) { *err = "the forward returned nothing"; }
    return false;
  }
  const auto* p = static_cast<const _Float16*>(o.contents());
  out->resize(o.byte_size() / 2);
  for (std::size_t i = 0; i < out->size(); ++i) { (*out)[i] = (float)p[i]; }
  return true;
}

// Rank `r` of a two-rank split of `cfg`: heads [0, h) and [h, n) and the
// matching feed-forward rows; `floor` > 0 the share's band floor.
K2::Config
k2_rank_(const K2::Config& cfg, int r, int h, int floor)
{
  K2::Config c = cfg;
  const int f = cfg.ffn / cfg.n_heads * h;
  c.tp = TpShare{};
  c.tp.rank = r;
  c.tp.ranks = 2;
  c.tp.n_heads = cfg.n_heads;
  c.tp.n_ffn = cfg.ffn;
  c.tp.head0 = r == 0 ? 0 : h;
  c.tp.head1 = r == 0 ? h : cfg.n_heads;
  c.tp.ffn0 = r == 0 ? 0 : f;
  c.tp.ffn1 = r == 0 ? f : cfg.ffn;
  if (floor > 0) {
    c.tp.extra = FlexData::make_object();
    c.tp.extra.as_object().insert_or_assign(
        tp_extra::kMinBandRows, FlexData::make_int(floor));
  }
  return c;
}

// Two ranks of `cfg` over one PairLink, preloaded or STREAMED (residency
// may keep both blocks), their forwards run at once `forwards` times.
// Each rank's last output, rank 0's exchanges and int8-layout
// projections, what it holds after (resident_bytes) against what the
// plan books for it (tp_held_bytes), and the blocks it kept.
struct K2Split {
  std::vector<float> v[2];
  std::uint64_t exchanges = 0;
  int out_of_step = 0, mismatched = 0;
  int fast = 0, kept = 0;
  std::size_t held = 0, booked = 0;
};
bool
krea2_split_(MetalCompute* mc, const fs::path& dir, const K2::Config& cfg,
             int h, bool stream, int forwards, int floor, const K2Geom& gm,
             int tap, K2Split* out)
{
  auto link = std::make_shared<PairLink>();
  std::shared_ptr<PairChannel> ch[2];
  std::unique_ptr<K2> m[2];
  for (int r = 0; r < 2; ++r) {
    m[r] = K2::load(dir.string(), mc, k2_rank_(cfg, r, h, floor), stream);
    if (m[r] == nullptr) { return false; }
    if (stream && tap < 0) {
      m[r]->set_residency_reserve(test_residency_reserve(1ull << 30));
      m[r]->set_residency_schedule(forwards);
    }
    ch[r] = std::make_shared<PairChannel>(link, r);
    std::string err;
    if (!m[r]->begin_tp_generation(ch[r], &err)) {
      std::printf("  begin: %s\n", err.c_str());
      return false;
    }
  }
  for (int k = 0; k < forwards; ++k) {
    std::string e0, e1;
    bool ok1 = false;
    std::thread t([&]() {
      ok1 = krea2_forward_(mc, *m[1], gm, tap, &out->v[1], &e1);
    });
    const bool ok0 = krea2_forward_(mc, *m[0], gm, tap, &out->v[0], &e0);
    t.join();
    if (!ok0 || !ok1) {
      std::printf("  rank 0: %s / rank 1: %s\n", e0.c_str(), e1.c_str());
      return false;
    }
  }
  out->fast = m[0]->fast_weights(8);
  // The exchange buffers: held through the picture, gone with its
  // channel (end_tp_generation). MUTATION: kept -- fails.
  const std::size_t exch_held = m[0]->tp_exchange_bytes();
  for (auto& mm : m) { mm->end_tp_generation(); }
  const std::size_t exch_left =
      m[0]->tp_exchange_bytes() + m[1]->tp_exchange_bytes();
  if (exch_held == 0 || exch_left != 0) {
    std::printf("  the exchange buffers: %zu bytes through the picture, "
                "%zu after its channels went (want some, then none)\n",
                exch_held, exch_left);
    return false;
  }
  out->exchanges = ch[0]->exchanges.load();
  out->out_of_step = ch[0]->out_of_step.load() + ch[1]->out_of_step.load();
  out->mismatched = link->mismatched.load();
  out->held = (std::size_t)m[0]->resident_bytes();
  out->booked =
      K2::tp_held_bytes(dir.string(), k2_rank_(cfg, 0, h, floor), mc);
  out->kept = m[0]->resident_block_count();
  return true;
}

}  // namespace

// A KREA-2 DiT SPLIT between two ranks is the unsplit one: each holds its
// share of every main block's heads -- to_q and to_gate (the sigmoid gate
// over the attention's output) by their rows, to_k / to_v by the rows of
// the kv heads those query heads read, to_out by its columns -- and
// feed-forward rows (ff.gate and ff.up by their rows, ff.down by its
// columns). Its attention is GROUPED, four query heads to a kv head, so a
// share is whole groups; any other is refused at load. The gated
// attention residual and the gated feed-forward residual are apart, so a
// block leaves TWO partials, each written straight into an exchange
// buffer, posted, and added before its gate. A random small model (8
// heads over 2 kv heads), 4 heads -- one kv group -- on rank 0:
//   - bf16, and an 8-bit affine pack (F16 scales, read and widened to
//     bf16 as the rows are read), preloaded, whole blocks: the split
//     within the bar of the unsplit model, both ranks bit-identical, and
//     what the rank holds what the plan books for it, to the byte;
//   - STREAMED, then kept: the same output BIT FOR BIT, both blocks kept
//     by the second forward;
//   - the share's rows READ alone against the whole weights read and cut
//     (VPIPE_TP_WHOLE_READ): the same bits;
//   - BANDED (a floor of 100 rows: two bands of the 296-row joint),
//     two exchanges a band, within the bar; overlapped against band after
//     band (VPIPE_KREA2_TP_LAG=0), the same bits;
//   - the same under int8 tiers (Per Token on the bf16, Fast on the pack;
//     matrix cores), a rank's share laid out for the tier within its own
//     columns: its activations are then quantized over its columns where
//     the whole model quantizes over all of them, so the bar is the
//     tier's own -- half again its distance from the plain model, and the
//     plain bar at least.
// THE BAR is the model's own reordering floor: the unsplit model on the
// steel kernels against the matrix-core ones (VPIPE_KREA2_NO_MMA2, the
// same sums in another order, rounded to bf16 alike) -- twice it, and 1%
// at least (a box with no matrix cores has no second order: 1%).
// MEASURED on the M5 Max: the floor 5.1e-3, the bar 1.0e-2; the split
// 3.0e-3 (bf16, the pack), 4.2e-3 at Per Token, 3.0e-3 at Fast (each
// tier 5.1-5.3e-3 from the plain model); banded the same as whole, to the
// bit. MUTATIONS, each failing it: no others' sum added after the out
// projection 2.9e-2 (the ranks differ too), rank 1 reading kv head 0 in
// place of its own 4.0e-2, rank 1 gating with rank 0's gate rows 1.4e-2
// -- the last under a bar of four floors (2.0e-2), which is why it is
// two.
// Env, optional: VPIPE_KREA2_TEST_MODEL_PATH -- the real Krea-2-Turbo's
// root (its transformer/), two of its blocks split 24 / 24 heads and
// streamed, the joint residual after them held to the same 1% (MEASURED:
// 3.9e-3, both ranks identical, four exchanges).
TEST(tensor_parallel, a_split_krea2_forward_is_the_unsplit_one)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const fs::path dir = fs::temp_directory_path() / "vpipe_krea2_tp_small";
  constexpr double kBar = 1e-2;
  const int h = 4;
  const K2Geom gm;
  const int seq = gm.text + gm.grid * gm.grid;
  struct Arm { const char* name; bool w8; int tier; };
  const Arm arms[] = {{"bf16", false, -1}, {"w8 pack", true, -1},
                      {"bf16, int8 Per Token", false, 2},
                      {"w8 pack, int8 Fast", true, 1}};
  bool written_w8 = false, written = false;
  std::vector<float> plain[2];   // [w8], the whole model untiered
  double floor_of[2] = {0.0, 0.0};
  for (const Arm& a : arms) {
    if (!written || written_w8 != a.w8) {
      ASSERT_TRUE(write_krea2_small_(dir, a.w8));
      written = true;
      written_w8 = a.w8;
    }
    K2::Config cfg = k2_small_cfg_();
    if (a.tier >= 0) {
      if (!mc->supports_matrix_cores()) {
        std::printf("  %-22s SKIP: no matrix cores\n", a.name);
        continue;
      }
      cfg.i8_gemm = true;
      cfg.i8_tier = a.tier;
    }
    std::vector<float> whole;
    {
      auto m = K2::load(dir.string(), mc, cfg, false);
      ASSERT_TRUE(m != nullptr);
      if (a.tier >= 0) { EXPECT_TRUE(m->fast_tier()); }
      std::string err;
      ASSERT_TRUE(krea2_forward_(mc, *m, gm, -1, &whole, &err));
    }
    if (a.tier < 0) {
      plain[a.w8] = whole;
      // The reordering floor: the same model, the same sums, the other
      // kernels.
      EnvScope e("VPIPE_KREA2_NO_MMA2", "1");
      auto m = K2::load(dir.string(), mc, cfg, false);
      ASSERT_TRUE(m != nullptr);
      std::vector<float> steel;
      std::string err;
      ASSERT_TRUE(krea2_forward_(mc, *m, gm, -1, &steel, &err));
      floor_of[a.w8] = rel_l2_(steel, whole);
    }
    const double bar =
        a.tier < 0 ? std::max(kBar, 2.0 * floor_of[a.w8])
                   : std::max(kBar, 1.5 * rel_l2_(whole, plain[a.w8]));
    std::printf("  krea2 %-22s bar %.3e (reordering floor %.3e; the tier "
                "from plain %.3e)\n", a.name, bar, floor_of[a.w8],
                a.tier < 0 ? 0.0 : rel_l2_(whole, plain[a.w8]));
    auto check = [&](const char* what, const K2Split& x,
                     std::uint64_t want) {
      const double d = rel_l2_(x.v[0], whole);
      std::printf("  krea2 %-22s %-18s split vs unsplit rel-L2 %.3e, ranks "
                  "%s, %llu exchanges (want %llu), rank 0 holds %d int8-"
                  "layout projections\n", a.name, what, d,
                  same_bits_(x.v[0], x.v[1]) ? "IDENTICAL" : "DIFFER",
                  (unsigned long long)x.exchanges,
                  (unsigned long long)want, x.fast);
      EXPECT_TRUE(same_bits_(x.v[0], x.v[1]));
      EXPECT_TRUE(d < bar);
      EXPECT_TRUE(x.exchanges == want);
      EXPECT_TRUE(x.out_of_step == 0 && x.mismatched == 0);
      // Under a tier every projection of the share takes its layout: 8
      // a block, 2 blocks.
      if (a.tier >= 0) { EXPECT_TRUE(x.fast == 16); }
    };
    auto exchanges = [&](int floor, int forwards) {
      return (std::uint64_t)forwards * 2 * (std::uint64_t)cfg.n_layers *
             tp_row_bands(seq, floor).size();
    };
    // A share that is not whole kv groups is refused by name.
    if (a.tier < 0 && !a.w8) {
      // (A session's error throws: refused either way.)
      std::string why = "null";
      std::unique_ptr<K2> bad;
      try {
        bad = K2::load(dir.string(), mc, k2_rank_(cfg, 0, 2, 0), false);
      } catch (const std::exception& e) {
        why = e.what();
      }
      std::printf("  krea2 %-22s 2 of 8 heads (half a kv group): %s (%s)\n",
                  a.name, bad == nullptr ? "refused" : "LOADED", why.c_str());
      EXPECT_TRUE(bad == nullptr);
    }
    // Whole blocks, preloaded.
    K2Split pre;
    ASSERT_TRUE(krea2_split_(mc, dir, cfg, h, false, 1, 0, gm, -1, &pre));
    check("preloaded", pre, exchanges(0, 1));
    std::printf("  krea2 %-22s rank 0 holds %zu bytes, booked %zu\n",
                a.name, pre.held, pre.booked);
    EXPECT_TRUE(pre.held == pre.booked && pre.booked > 0);
    // Streamed, then kept.
    K2Split str;
    ASSERT_TRUE(krea2_split_(mc, dir, cfg, h, true, 2, 0, gm, -1, &str));
    std::printf("  krea2 %-22s streamed: %s the preloaded rank's; %d of %d "
                "blocks kept, %zu bytes held\n", a.name,
                same_bits_(str.v[0], pre.v[0]) ? "IDENTICAL to"
                                               : "DIFFERS from",
                str.kept, cfg.n_layers, str.held);
    EXPECT_TRUE(same_bits_(str.v[0], pre.v[0]));
    EXPECT_TRUE(str.kept == cfg.n_layers);
    EXPECT_TRUE(str.held == pre.booked);
    // The whole weights read and cut: the same share.
    {
      EnvScope e("VPIPE_TP_WHOLE_READ", "1");
      K2Split wr;
      ASSERT_TRUE(krea2_split_(mc, dir, cfg, h, false, 1, 0, gm, -1, &wr));
      std::printf("  krea2 %-22s whole weights read: %s the row reads\n",
                  a.name, same_bits_(wr.v[0], pre.v[0]) ? "IDENTICAL to"
                                                        : "DIFFERS from");
      EXPECT_TRUE(same_bits_(wr.v[0], pre.v[0]));
    }
    // Banded, overlapped and in turn.
    K2Split bd;
    ASSERT_TRUE(krea2_split_(mc, dir, cfg, h, false, 1, 100, gm, -1, &bd));
    check("banded (floor 100)", bd, exchanges(100, 1));
    {
      EnvScope e("VPIPE_KREA2_TP_LAG", "0");
      K2Split b0;
      ASSERT_TRUE(
          krea2_split_(mc, dir, cfg, h, false, 1, 100, gm, -1, &b0));
      std::printf("  krea2 %-22s banded in turn: %s the overlapped bands\n",
                  a.name, same_bits_(b0.v[0], bd.v[0]) ? "IDENTICAL to"
                                                       : "DIFFERS from");
      EXPECT_TRUE(same_bits_(b0.v[0], bd.v[0]));
    }
  }
  std::error_code ec;
  fs::remove_all(dir, ec);

  // The real checkpoint, two of its blocks: its own widths (48 heads over
  // 12 kv heads, 16,384 rows), its bf16 rows read as the share.
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') {
    std::printf("  krea2 real checkpoint: SKIP (VPIPE_KREA2_TEST_MODEL_PATH "
                "unset)\n");
    return;
  }
  const fs::path tdir = fs::path(root) / "transformer";
  K2::Config rc;
  rc.autotune = false;
  const K2Geom rg{64, 16};
  std::vector<float> whole;
  {
    auto m = K2::load(tdir.string(), mc, rc, /*stream=*/true);
    ASSERT_TRUE(m != nullptr);
    // A model that did not load stops here, not read through null.
    if (m == nullptr) { return; }
    std::string err;
    ASSERT_TRUE(krea2_forward_(mc, *m, rg, 1, &whole, &err));
  }
  K2Split rs;
  ASSERT_TRUE(krea2_split_(mc, tdir, rc, 24, true, 1, 0, rg, 1, &rs));
  const double d = rel_l2_(rs.v[0], whole);
  std::printf("  krea2 real, 2 blocks, 24 / 48 heads: split vs unsplit "
              "rel-L2 %.3e, ranks %s, %llu exchanges (want 4)\n", d,
              same_bits_(rs.v[0], rs.v[1]) ? "IDENTICAL" : "DIFFER",
              (unsigned long long)rs.exchanges);
  EXPECT_TRUE(same_bits_(rs.v[0], rs.v[1]));
  EXPECT_TRUE(d < kBar);
  EXPECT_TRUE(rs.exchanges == 4);
}

namespace {

// A small random MOSS-Image DiT on disk (MOSSImageTransformer's names and
// config): 8 heads of 128, two main blocks and the two text blocks,
// every projection bf16 -- or, `w8`, the main blocks' projections an
// MLX affine 8-bit pack in groups of 64, the rest dense. The fused
// ffn_up_gate is [hidden | gate], hidden FIRST, as the release lays it.
// The checkpoint's own `rope.bands` table rides along (F32, never
// loaded): a rank's plan must not book it.
bool
write_moss_small_(const fs::path& dir, bool w8)
{
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  const int H = 1024, F = 3072, TD = 256, LC = 64, L = 2;
  {
    std::ofstream o(dir / "config.json");
    o << "{\"_class_name\": \"MOSSImageTransformer\", \"dim\": " << H
      << ", \"ffn_dim\": " << F << ", \"freq_dim\": 256, "
      << "\"latent_channels\": " << LC << ", \"num_heads\": 8, "
      << "\"num_layers\": " << L << ", \"rope_base\": 10000.0, "
      << "\"text_dim\": " << TD << "}";
  }
  genai::SafetensorsWriter wr(dir.string());
  std::uint64_t st = 0x9E3779B97F4A7C15ull ^ (w8 ? 1 : 0);
  auto rnd = [&]() {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    return (float)((double)(st >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
  };
  auto dense = [&](const std::string& n, std::vector<std::int64_t> shape,
                   float sc, float centre = 0.0f) {
    std::size_t k = 1;
    for (auto d : shape) { k *= (std::size_t)d; }
    std::vector<std::uint16_t> v(k);
    for (auto& x : v) { x = bf16_(centre + rnd() * sc); }
    return wr.add(n, "BF16", shape, v.data(), v.size() * 2);
  };
  auto packed = [&](const std::string& n, int N, int K, float sc) {
    const int g = 64;
    std::vector<std::uint32_t> codes((std::size_t)N * K / 4);
    std::vector<std::uint16_t> scales((std::size_t)N * K / g);
    std::vector<std::uint16_t> biases(scales.size());
    std::vector<float> row((std::size_t)K);
    for (int r = 0; r < N; ++r) {
      for (auto& x : row) { x = rnd() * sc; }
      for (int g0 = 0; g0 < K; g0 += g) {
        float lo = row[(std::size_t)g0], hi = lo;
        for (int i = g0; i < g0 + g; ++i) {
          lo = std::min(lo, row[(std::size_t)i]);
          hi = std::max(hi, row[(std::size_t)i]);
        }
        const float s8 = f32_(bf16_(std::max(hi - lo, 1e-6f) / 255.0f));
        const float b8 = f32_(bf16_(lo));
        const std::size_t gi = ((std::size_t)r * K + g0) / g;
        scales[gi] = bf16_(s8);
        biases[gi] = bf16_(b8);
        for (int i = g0; i < g0 + g; ++i) {
          const int q = std::clamp(
              (int)std::lround((row[(std::size_t)i] - b8) / s8), 0, 255);
          const std::size_t e = (std::size_t)r * K + i;
          codes[e / 4] |= (std::uint32_t)q << (8 * (e % 4));
        }
      }
    }
    return wr.add(n + ".weight", "U32", {N, K / 4}, codes.data(),
                  codes.size() * 4) &&
           wr.add(n + ".scales", "BF16", {N, K / g}, scales.data(),
                  scales.size() * 2) &&
           wr.add(n + ".biases", "BF16", {N, K / g}, biases.data(),
                  biases.size() * 2);
  };
  auto proj = [&](const std::string& n, int N, int K, float sc) {
    return w8 ? packed(n, N, K, sc) : dense(n + ".weight", {N, K}, sc);
  };
  auto linear = [&](const std::string& n, int N, int K, float sc) {
    return dense(n + ".weight", {N, K}, sc) &&
           dense(n + ".bias", {N}, 0.02f);
  };
  bool ok = linear("video_embedder", H, LC, 0.1f) &&
            linear("text_embedder.proj", H, TD, 0.06f) &&
            linear("time_embedding.0", H, 256, 0.06f) &&
            linear("time_embedding.2", H, H, 0.03f) &&
            linear("shared_block_modulation", 6 * H, H, 0.03f) &&
            linear("video_head_modulation", 2 * H, H, 0.03f) &&
            dense("final_norm_video.weight", {H}, 0.1f, 1.0f) &&
            dense("final_linear_video.weight", {LC, H}, 0.04f);
  {
    const std::vector<float> bands(8, 1.0f);
    ok = ok && wr.add("rope.bands", "F32", {8}, bands.data(),
                      bands.size() * 4);
  }
  for (int i = 0; i < 2; ++i) {
    const std::string p = "text_embedder.blocks." + std::to_string(i) + ".";
    ok = ok && linear(p + "qkv", 3 * H, H, 0.04f) &&
         linear(p + "proj", H, H, 0.04f) &&
         linear(p + "ffn_up", F, H, 0.04f) &&
         linear(p + "ffn_gate", F, H, 0.04f) &&
         linear(p + "ffn_down", H, F, 0.03f) &&
         dense(p + "norm1.weight", {H}, 0.1f, 1.0f) &&
         dense(p + "norm2.weight", {H}, 0.1f, 1.0f) &&
         dense(p + "q_norm.weight", {128}, 0.1f, 1.0f) &&
         dense(p + "k_norm.weight", {128}, 0.1f, 1.0f);
  }
  for (int l = 0; l < L; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    ok = ok && dense(p + "norm1.weight", {H}, 0.1f, 1.0f) &&
         dense(p + "norm2.weight", {H}, 0.1f, 1.0f) &&
         dense(p + "q_norm.weight", {128}, 0.1f, 1.0f) &&
         dense(p + "k_norm.weight", {128}, 0.1f, 1.0f) &&
         proj(p + "proj.q", H, H, 0.04f) &&
         proj(p + "proj.k", H, H, 0.04f) &&
         proj(p + "proj.v", H, H, 0.04f) &&
         proj(p + "proj.o", H, H, 0.04f) &&
         proj(p + "proj.ffn_up_gate", 2 * F, H, 0.04f) &&
         proj(p + "proj.ffn_down", H, F, 0.03f);
  }
  return wr.close() && ok;
}

using MI = MetalMossImageTransformer;

// A picture's first two steps of `m` as generate_moss_image_ runs them:
// each step the POSITIVE (every block) then classifier-free guidance's
// NEGATIVE -- its own, shorter prompt, block `skip` left out as the
// reference's negative leaves block 22 out -- over one 16 x 16 latent.
// Each forward's image rows as f32 (the forward returns bf16).
struct MossSteps {
  std::vector<float> v[2];
  std::vector<float> n[2];
};
constexpr int kMossTxt = 24, kMossTxtNeg = 17, kMossGrid = 16;
bool
moss_steps_(MetalCompute* mc, MI& m, MossSteps* out, std::string* err,
            int skip = 1)
{
  const MI::Config& c = m.config();
  std::uint64_t st = 4242;
  auto buf = [&](std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      st ^= st << 13; st ^= st >> 7; st ^= st << 17;
      d[i] = bf16_((float)((double)(st >> 11) / 9007199254740992.0) - 0.5f);
    }
    return b;
  };
  const int img = kMossGrid * kMossGrid;
  SharedBuffer lat = buf((std::size_t)img * c.latent_channels);
  SharedBuffer txt = buf((std::size_t)kMossTxt * c.text_dim);
  SharedBuffer neg = buf((std::size_t)kMossTxtNeg * c.text_dim);
  auto run = [&](const SharedBuffer& tx, int rows, std::vector<int> sk,
                 float t, std::vector<float>* v) {
    MI::Request req;
    req.latents = &lat;
    req.grid_h = kMossGrid;
    req.grid_w = kMossGrid;
    req.text = &tx;
    req.text_rows = rows;
    req.timestep = t;
    req.skip_layers = std::move(sk);
    SharedBuffer o = m.forward(req, err);
    if (o.empty()) { return false; }
    const auto* p = static_cast<const std::uint16_t*>(o.contents());
    const std::size_t n = (std::size_t)img * c.latent_channels;
    v->resize(n);
    for (std::size_t i = 0; i < n; ++i) { (*v)[i] = f32_(p[i]); }
    return true;
  };
  for (int k = 0; k < 2; ++k) {
    const float t = k == 0 ? 900.0f : 500.0f;
    if (!run(txt, kMossTxt, {}, t, &out->v[k])) { return false; }
    if (!run(neg, kMossTxtNeg, {skip}, t, &out->n[k])) { return false; }
  }
  return true;
}

// Rank `r` of a two-rank split of `cfg`: heads [0, h) and [h, n) and the
// matching feed-forward rows; `floor` > 0 the share's band floor.
MI::Config
moss_rank_(const MI::Config& cfg, int r, int h, int floor)
{
  MI::Config c = cfg;
  const int F = cfg.ffn_inner;
  const int f = F / cfg.n_heads * h;
  c.tp = TpShare{};
  c.tp.rank = r;
  c.tp.ranks = 2;
  c.tp.n_heads = cfg.n_heads;
  c.tp.n_ffn = F;
  c.tp.head0 = r == 0 ? 0 : h;
  c.tp.head1 = r == 0 ? h : cfg.n_heads;
  c.tp.ffn0 = r == 0 ? 0 : f;
  c.tp.ffn1 = r == 0 ? f : F;
  if (floor > 0) {
    c.tp.extra = FlexData::make_object();
    c.tp.extra.as_object().insert_or_assign(
        tp_extra::kMinBandRows, FlexData::make_int(floor));
  }
  return c;
}

// Two ranks of `cfg` over one PairLink, preloaded or STREAMED (residency
// may keep both blocks), each running moss_steps_ at once -- twice when
// streamed, the second from what residency kept. Each rank's steps,
// rank 0's exchanges and int8-layout projections, what it holds after
// (resident_bytes) against what the plan books for it (tp_held_bytes),
// and the blocks it kept.
struct MossSplit {
  MossSteps s[2];
  std::uint64_t exchanges = 0;
  int out_of_step = 0, mismatched = 0;
  int fast = 0, kept = 0;
  std::size_t held = 0, booked = 0;
};
bool
moss_split_(MetalCompute* mc, const fs::path& dir, const MI::Config& cfg,
            int h, bool stream, int floor, MossSplit* out, int skip = 1)
{
  auto link = std::make_shared<PairLink>();
  std::shared_ptr<PairChannel> ch[2];
  std::unique_ptr<MI> m[2];
  const int runs = stream ? 2 : 1;
  for (int r = 0; r < 2; ++r) {
    m[r] = MI::load(dir.string(), mc, moss_rank_(cfg, r, h, floor), stream);
    if (m[r] == nullptr) { return false; }
    if (stream) {
      m[r]->set_residency_reserve(test_residency_reserve(1ull << 30));
      m[r]->set_residency_schedule(2 * runs * 2);
    }
    ch[r] = std::make_shared<PairChannel>(link, r);
    std::string err;
    if (!m[r]->begin_tp_generation(ch[r], &err)) {
      std::printf("  begin: %s\n", err.c_str());
      return false;
    }
  }
  for (int k = 0; k < runs; ++k) {
    std::string e0, e1;
    bool ok1 = false;
    std::thread t(
        [&]() { ok1 = moss_steps_(mc, *m[1], &out->s[1], &e1, skip); });
    const bool ok0 = moss_steps_(mc, *m[0], &out->s[0], &e0, skip);
    t.join();
    if (!ok0 || !ok1) {
      std::printf("  rank 0: %s / rank 1: %s\n", e0.c_str(), e1.c_str());
      return false;
    }
  }
  out->fast = m[0]->fast_weights(8);
  // The exchange buffers: held through the picture, gone with its
  // channel (end_tp_generation). MUTATION: kept -- fails.
  const std::size_t exch_held = m[0]->tp_exchange_bytes();
  for (auto& mm : m) { mm->end_tp_generation(); }
  const std::size_t exch_left =
      m[0]->tp_exchange_bytes() + m[1]->tp_exchange_bytes();
  if (exch_held == 0 || exch_left != 0) {
    std::printf("  the exchange buffers: %zu bytes through the picture, "
                "%zu after its channels went (want some, then none)\n",
                exch_held, exch_left);
    return false;
  }
  out->exchanges = ch[0]->exchanges.load();
  out->out_of_step = ch[0]->out_of_step.load() + ch[1]->out_of_step.load();
  out->mismatched = link->mismatched.load();
  out->held = (std::size_t)m[0]->resident_bytes();
  out->booked =
      MI::tp_held_bytes(dir.string(), moss_rank_(cfg, 0, h, floor), mc);
  out->kept = m[0]->resident_block_count();
  return true;
}

}  // namespace

// A MOSS-IMAGE DiT SPLIT between two ranks is the unsplit one: each holds
// its share of every main block's heads (proj.q / k / v by their rows,
// proj.o by its columns) and feed-forward rows (ffn_up_gate by its rows
// -- the share's rows of the hidden half AND the same rows of the gate
// half after it -- ffn_down by its columns), and its two sublayers --
// gated apart by the one shared modulation -- leave TWO partials a
// block, each written straight into an exchange buffer, posted, and
// added before its gate. The two text blocks run whole on both ranks.
// The picture's first two steps on a random small model, 4 of 8 heads on
// rank 0, each step the positive and classifier-free guidance's
// NEGATIVE (its own prompt, block 1 skipped: no exchange for it on
// either rank):
//   - THE BAR is the model's own: its reordering floor -- the unsplit
//     model through the steel tiles (VPIPE_MOSS_IMAGE_NO_MMA2) against
//     the matrix-core GEMMs, the same arithmetic summed in another order
//     -- four times over, 1e-2 at least, as the other families' (a box
//     with one route, an M4, has a floor of 0: at 2.5e-3 the split's
//     2.2-3.0e-3 failed there);
//   - bf16, and an 8-bit affine pack, preloaded, whole blocks: the split
//     within the bar, and what the rank holds what the plan books for it,
//     to the byte;
//   - STREAMED, then kept: the same steps BIT FOR BIT, both blocks kept;
//   - the share's rows READ alone against the whole weights read and
//     cut (VPIPE_TP_WHOLE_READ): the same bits;
//   - BANDED (a floor of 100 rows): two exchanges a band, within the
//     bar; overlapped against band after band (VPIPE_MOSS_IMAGE_TP_LAG
//     =0), the same bits;
//   - the same under int8 tiers (Per Token on the bf16, Fast on the
//     pack; matrix cores), a rank's share laid out for the tier within
//     its own columns: its activations are then quantized over its
//     columns where the whole model quantizes over all of them, so the
//     bar is the tier's own -- the split within half again of what the
//     tier is from the plain model, and the plain bar at least.
// Both ranks' outputs identical in every arm, every step. MEASURED on an
// M5 Max: floor 2.5-2.6e-3 (bar ~1e-2), the split 2.2-3.0e-3 plain and
// 3.1-4.2e-3 under Per Token. MUTATIONS, each tried and taken out, each
// failing it: ffn_up_gate cut as if the gate came first 8.6e-2 to
// 1.3e-1, the attention's partial not summed 3.1-4.9e-2 (ranks apart),
// a band's feed-forward gate added at the block's first rows 1.3-2.0e-1
// (banded arms alone).
TEST(tensor_parallel, a_split_moss_image_forward_is_the_unsplit_one)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const fs::path dir = fs::temp_directory_path() / "vpipe_moss_tp_small";
  const int h = 4;
  struct Arm { const char* name; bool w8; int tier; };
  const Arm arms[] = {{"bf16", false, -1}, {"w8 pack", true, -1},
                      {"bf16, int8 Per Token", false, 2},
                      {"w8 pack, int8 Fast", true, 1}};
  bool written_w8 = false, written = false;
  MossSteps plain[2];   // [w8], the whole model untiered
  double bar0[2] = {0.0, 0.0};
  for (const Arm& a : arms) {
    if (!written || written_w8 != a.w8) {
      ASSERT_TRUE(write_moss_small_(dir, a.w8));
      written = true;
      written_w8 = a.w8;
    }
    MI::Config cfg;
    ASSERT_TRUE(MI::Config::read_dims(dir.string(), &cfg));
    cfg.autotune = false;
    if (a.tier >= 0) {
      if (!mc->supports_matrix_cores()) {
        std::printf("  %-22s SKIP: no matrix cores\n", a.name);
        continue;
      }
      cfg.i8_gemm = true;
      cfg.i8_tier = a.tier;
    }
    MossSteps whole;
    {
      auto m = MI::load(dir.string(), mc, cfg, false);
      ASSERT_TRUE(m != nullptr);
      if (a.tier >= 0) { EXPECT_TRUE(m->fast_tier()); }
      std::string err;
      ASSERT_TRUE(moss_steps_(mc, *m, &whole, &err));
    }
    if (a.tier < 0) {
      plain[a.w8] = whole;
      // The model's own reordering floor: the same forwards through the
      // other GEMM route.
      MossSteps other;
      {
        EnvScope e("VPIPE_MOSS_IMAGE_NO_MMA2", "1");
        auto m = MI::load(dir.string(), mc, cfg, false);
        ASSERT_TRUE(m != nullptr);
        std::string err;
        ASSERT_TRUE(moss_steps_(mc, *m, &other, &err));
      }
      double floor = 0.0;
      for (int k = 0; k < 2; ++k) {
        floor = std::max({floor, rel_l2_(other.v[k], whole.v[k]),
                          rel_l2_(other.n[k], whole.n[k])});
      }
      bar0[a.w8] = std::max(1e-2, 4.0 * floor);
      std::printf("  moss %-22s reordering floor %.3e -> bar %.3e\n",
                  a.name, floor, bar0[a.w8]);
    }
    // The arm's bar for one output: the plain bar, or under a tier the
    // tier's own distance from the plain model, half again.
    auto bar_for = [&](const std::vector<float>& w,
                       const std::vector<float>& p) {
      return a.tier < 0 ? bar0[a.w8]
                        : std::max(bar0[a.w8], 1.5 * rel_l2_(w, p));
    };
    auto check = [&](const char* what, const MossSplit& x,
                     std::uint64_t want) {
      for (int k = 0; k < 2; ++k) {
        const struct {
          const char* side;
          const std::vector<float>* got[2];
          const std::vector<float>* w;
          const std::vector<float>* p;
        } sides[] = {{"positive", {&x.s[0].v[k], &x.s[1].v[k]},
                      &whole.v[k], &plain[a.w8].v[k]},
                     {"negative", {&x.s[0].n[k], &x.s[1].n[k]},
                      &whole.n[k], &plain[a.w8].n[k]}};
        for (const auto& sd : sides) {
          const double d = rel_l2_(*sd.got[0], *sd.w);
          const double bar = bar_for(*sd.w, *sd.p);
          std::printf("  moss %-22s %-18s step %d %s: split vs unsplit "
                      "rel-L2 %.3e (bar %.3e), ranks %s\n", a.name, what, k,
                      sd.side, d, bar,
                      same_bits_(*sd.got[0], *sd.got[1]) ? "IDENTICAL"
                                                         : "DIFFER");
          EXPECT_TRUE(same_bits_(*sd.got[0], *sd.got[1]));
          EXPECT_TRUE(d < bar);
        }
      }
      std::printf("  moss %-22s %-18s %llu exchanges (want %llu); rank 0 "
                  "holds %d int8-layout projections\n", a.name, what,
                  (unsigned long long)x.exchanges,
                  (unsigned long long)want, x.fast);
      EXPECT_TRUE(x.exchanges == want);
      EXPECT_TRUE(x.out_of_step == 0 && x.mismatched == 0);
      // Under a tier every projection of the share takes its layout: 6
      // a block, 2 blocks.
      if (a.tier >= 0) { EXPECT_TRUE(x.fast == 12); }
    };
    auto same = [](const MossSplit& x, const MossSplit& y) {
      bool ok = true;
      for (int k = 0; k < 2; ++k) {
        ok = ok && same_bits_(x.s[0].v[k], y.s[0].v[k]) &&
             same_bits_(x.s[0].n[k], y.s[0].n[k]);
      }
      return ok;
    };
    // Two steps: the positive over every block, the negative over all but
    // the skipped one -- each block's bands (tp_row_bands) over its rows.
    const int img = kMossGrid * kMossGrid;
    auto exchanges = [&](int floor) {
      const std::uint64_t pos = 2 * (std::uint64_t)cfg.n_layers *
                                tp_row_bands(img + kMossTxt, floor).size();
      const std::uint64_t neg =
          2 * (std::uint64_t)(cfg.n_layers - 1) *
          tp_row_bands(img + kMossTxtNeg, floor).size();
      return 2 * (pos + neg);
    };
    // Whole blocks, preloaded.
    MossSplit pre;
    ASSERT_TRUE(moss_split_(mc, dir, cfg, h, false, 0, &pre));
    check("preloaded", pre, exchanges(0));
    std::printf("  moss %-22s rank 0 holds %zu bytes, booked %zu\n", a.name,
                pre.held, pre.booked);
    EXPECT_TRUE(pre.held == pre.booked && pre.booked > 0);
    // Streamed, then kept.
    MossSplit str;
    ASSERT_TRUE(moss_split_(mc, dir, cfg, h, true, 0, &str));
    std::printf("  moss %-22s streamed: %s the preloaded rank's; %d of %d "
                "blocks kept, %zu bytes held\n", a.name,
                same(str, pre) ? "IDENTICAL to" : "DIFFERS from", str.kept,
                cfg.n_layers, str.held);
    EXPECT_TRUE(same(str, pre));
    EXPECT_TRUE(str.kept == cfg.n_layers);
    EXPECT_TRUE(str.held == pre.booked);
    // The whole weights read and cut: the same share.
    {
      EnvScope e("VPIPE_TP_WHOLE_READ", "1");
      MossSplit wr;
      ASSERT_TRUE(moss_split_(mc, dir, cfg, h, false, 0, &wr));
      std::printf("  moss %-22s whole weights read: %s the row reads\n",
                  a.name, same(wr, pre) ? "IDENTICAL to" : "DIFFERS from");
      EXPECT_TRUE(same(wr, pre));
    }
    // Banded, overlapped and in turn.
    MossSplit bd;
    ASSERT_TRUE(moss_split_(mc, dir, cfg, h, false, 100, &bd));
    check("banded (floor 100)", bd, exchanges(100));
    {
      EnvScope e("VPIPE_MOSS_IMAGE_TP_LAG", "0");
      MossSplit b0;
      ASSERT_TRUE(moss_split_(mc, dir, cfg, h, false, 100, &b0));
      std::printf("  moss %-22s banded in turn: %s the overlapped bands\n",
                  a.name, same(b0, bd) ? "IDENTICAL to" : "DIFFERS from");
      EXPECT_TRUE(same(b0, bd));
    }
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ---- Z-Image's ranks -------------------------------------------------

namespace {

// A small random Z-Image DiT on disk (ZImageTransformer2DModel's names
// and config): 12 heads of 128, one block in each refiner and two in
// the main stack, its SwiGLU int(dim / 3 * 8) = 4096 wide. `dt` the
// dtype every float tensor is written in ("BF16", or "F32" as Turbo
// ships); `w8`, the block projections an MLX affine 8-bit pack in
// groups of 64, the rest dense.
bool
write_zi_small_(const fs::path& dir, const char* dt, bool w8)
{
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  const int H = 1536, F = 4096, C = 256, PD = 64, L = 2, R = 1;
  {
    std::ofstream o(dir / "config.json");
    o << "{\"_class_name\": \"ZImageTransformer2DModel\", \"dim\": " << H
      << ", \"n_heads\": 12, \"n_kv_heads\": 12, \"n_layers\": " << L
      << ", \"n_refiner_layers\": " << R << ", \"in_channels\": 16, "
      << "\"cap_feat_dim\": " << C << ", \"axes_dims\": [32, 48, 48], "
      << "\"axes_lens\": [1536, 512, 512], \"all_patch_size\": [2], "
      << "\"all_f_patch_size\": [1], \"qk_norm\": true}";
  }
  genai::SafetensorsWriter wr(dir.string());
  std::uint64_t st = 0x9E3779B97F4A7C15ull ^ (w8 ? 1 : 0) ^
                     (std::strcmp(dt, "F32") == 0 ? 2 : 0);
  auto rnd = [&]() {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    return (float)((double)(st >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
  };
  const bool f32 = std::strcmp(dt, "F32") == 0;
  auto dense = [&](const std::string& n, std::vector<std::int64_t> shape,
                   float sc, float centre = 0.0f) {
    std::size_t k = 1;
    for (auto d : shape) { k *= (std::size_t)d; }
    if (f32) {
      std::vector<float> v(k);
      for (auto& x : v) { x = centre + rnd() * sc; }
      return wr.add(n, "F32", shape, v.data(), v.size() * 4);
    }
    std::vector<std::uint16_t> v(k);
    for (auto& x : v) { x = bf16_(centre + rnd() * sc); }
    return wr.add(n, "BF16", shape, v.data(), v.size() * 2);
  };
  auto packed = [&](const std::string& n, int N, int K, float sc) {
    const int g = 64;
    std::vector<std::uint32_t> codes((std::size_t)N * K / 4);
    std::vector<std::uint16_t> scales((std::size_t)N * K / g);
    std::vector<std::uint16_t> biases(scales.size());
    std::vector<float> row((std::size_t)K);
    for (int r = 0; r < N; ++r) {
      for (auto& x : row) { x = rnd() * sc; }
      for (int g0 = 0; g0 < K; g0 += g) {
        float lo = row[(std::size_t)g0], hi = lo;
        for (int i = g0; i < g0 + g; ++i) {
          lo = std::min(lo, row[(std::size_t)i]);
          hi = std::max(hi, row[(std::size_t)i]);
        }
        const float s8 = f32_(bf16_(std::max(hi - lo, 1e-6f) / 255.0f));
        const float b8 = f32_(bf16_(lo));
        const std::size_t gi = ((std::size_t)r * K + g0) / g;
        scales[gi] = bf16_(s8);
        biases[gi] = bf16_(b8);
        for (int i = g0; i < g0 + g; ++i) {
          const int q = std::clamp(
              (int)std::lround((row[(std::size_t)i] - b8) / s8), 0, 255);
          const std::size_t e = (std::size_t)r * K + i;
          codes[e / 4] |= (std::uint32_t)q << (8 * (e % 4));
        }
      }
    }
    return wr.add(n + ".weight", "U32", {N, K / 4}, codes.data(),
                  codes.size() * 4) &&
           wr.add(n + ".scales", "BF16", {N, K / g}, scales.data(),
                  scales.size() * 2) &&
           wr.add(n + ".biases", "BF16", {N, K / g}, biases.data(),
                  biases.size() * 2);
  };
  auto proj = [&](const std::string& n, int N, int K, float sc) {
    return w8 ? packed(n, N, K, sc) : dense(n + ".weight", {N, K}, sc);
  };
  auto block = [&](const std::string& p, bool ada) {
    bool ok = proj(p + "attention.to_q", H, H, 0.03f) &&
              proj(p + "attention.to_k", H, H, 0.03f) &&
              proj(p + "attention.to_v", H, H, 0.03f) &&
              proj(p + "attention.to_out.0", H, H, 0.03f) &&
              dense(p + "attention.norm_q.weight", {128}, 0.1f, 1.0f) &&
              dense(p + "attention.norm_k.weight", {128}, 0.1f, 1.0f) &&
              proj(p + "feed_forward.w1", F, H, 0.03f) &&
              proj(p + "feed_forward.w3", F, H, 0.03f) &&
              proj(p + "feed_forward.w2", H, F, 0.02f) &&
              dense(p + "attention_norm1.weight", {H}, 0.1f, 1.0f) &&
              dense(p + "attention_norm2.weight", {H}, 0.1f, 1.0f) &&
              dense(p + "ffn_norm1.weight", {H}, 0.1f, 1.0f) &&
              dense(p + "ffn_norm2.weight", {H}, 0.1f, 1.0f);
    if (ada) {
      ok = ok && dense(p + "adaLN_modulation.0.weight", {4 * H, 256},
                       0.05f) &&
           dense(p + "adaLN_modulation.0.bias", {4 * H}, 0.2f);
    }
    return ok;
  };
  bool ok = dense("all_x_embedder.2-1.weight", {H, PD}, 0.1f) &&
            dense("all_x_embedder.2-1.bias", {H}, 0.05f) &&
            dense("cap_embedder.0.weight", {C}, 0.1f, 1.0f) &&
            dense("cap_embedder.1.weight", {H, C}, 0.06f) &&
            dense("cap_embedder.1.bias", {H}, 0.05f) &&
            dense("t_embedder.mlp.0.weight", {1024, 256}, 0.06f) &&
            dense("t_embedder.mlp.0.bias", {1024}, 0.05f) &&
            dense("t_embedder.mlp.2.weight", {256, 1024}, 0.03f) &&
            dense("t_embedder.mlp.2.bias", {256}, 0.05f) &&
            dense("x_pad_token", {1, H}, 0.5f) &&
            dense("cap_pad_token", {1, H}, 0.5f) &&
            dense("all_final_layer.2-1.adaLN_modulation.1.weight",
                  {H, 256}, 0.05f) &&
            dense("all_final_layer.2-1.adaLN_modulation.1.bias", {H},
                  0.05f) &&
            dense("all_final_layer.2-1.linear.weight", {PD, H}, 0.03f) &&
            dense("all_final_layer.2-1.linear.bias", {PD}, 0.05f);
  for (int i = 0; i < R; ++i) {
    ok = ok && block("context_refiner." + std::to_string(i) + ".", false) &&
         block("noise_refiner." + std::to_string(i) + ".", true);
  }
  for (int l = 0; l < L; ++l) {
    ok = ok && block("layers." + std::to_string(l) + ".", true);
  }
  return wr.close() && ok;
}

using ZI = MetalZImageTransformer;

// The sizes every Z-Image forward here runs: a 32 x 32 patch grid (1,024
// image rows, the int8 tiers' floor) beside a 40-token caption (64 rows
// padded) -- three row counts, one a block group.
bool
zi_layout_(zimage::Layout* lay, std::string* err)
{
  return zimage::build_layout(32, 32, 40, lay, err);
}

// Two forwards of `m` (sigma 0.9, then 0.6) on fixed random inputs; each
// one's image rows as f32 (the forward returns bf16).
struct ZiSteps {
  std::vector<float> v[2];
};
bool
zi_steps_(MetalCompute* mc, ZI& m, ZiSteps* out, std::string* err)
{
  zimage::Layout lay;
  if (!zi_layout_(&lay, err)) { return false; }
  std::uint64_t st = 4242;
  auto buf = [&](std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      st ^= st << 13; st ^= st >> 7; st ^= st << 17;
      d[i] = bf16_((float)((double)(st >> 11) / 9007199254740992.0) - 0.5f);
    }
    return b;
  };
  const ZI::Config& c = m.config();
  SharedBuffer lat = buf((std::size_t)lay.img_tokens * c.patch_dim());
  SharedBuffer cap = buf((std::size_t)lay.cap_len * c.cap_dim);
  for (int k = 0; k < 2; ++k) {
    ZI::Request req;
    req.latents = &lat;
    req.cap = &cap;
    req.layout = &lay;
    req.timestep = k == 0 ? 0.9f : 0.6f;
    SharedBuffer o = m.forward(req, err);
    if (o.empty()) { return false; }
    const auto* p = static_cast<const std::uint16_t*>(o.contents());
    const std::size_t n = (std::size_t)lay.img_tokens * c.patch_dim();
    out->v[k].resize(n);
    for (std::size_t i = 0; i < n; ++i) { out->v[k][i] = f32_(p[i]); }
  }
  return true;
}

// Rank `r` of a two-rank split of `cfg`: heads [0, h) and [h, n) and the
// matching feed-forward rows; `floor` > 0 the share's band floor.
ZI::Config
zi_rank_(const ZI::Config& cfg, int r, int h, int floor)
{
  ZI::Config c = cfg;
  const int F = cfg.ffn_inner;
  const int f = F * h / cfg.n_heads;
  c.tp = TpShare{};
  c.tp.rank = r;
  c.tp.ranks = 2;
  c.tp.n_heads = cfg.n_heads;
  c.tp.n_ffn = F;
  c.tp.head0 = r == 0 ? 0 : h;
  c.tp.head1 = r == 0 ? h : cfg.n_heads;
  c.tp.ffn0 = r == 0 ? 0 : f;
  c.tp.ffn1 = r == 0 ? f : F;
  if (floor > 0) {
    c.tp.extra = FlexData::make_object();
    c.tp.extra.as_object().insert_or_assign(
        tp_extra::kMinBandRows, FlexData::make_int(floor));
  }
  return c;
}

// Two ranks of `cfg` over one PairLink, preloaded or STREAMED (residency
// may keep every block), each running zi_steps_ at once -- twice when
// streamed, the second from what residency kept. Each rank's steps,
// rank 0's exchanges and int8-layout projections, what it holds after
// (resident_bytes) against what the plan books for it (tp_held_bytes),
// and the blocks it kept.
struct ZiSplit {
  ZiSteps s[2];
  std::uint64_t exchanges = 0;
  int out_of_step = 0, mismatched = 0;
  int fast = 0, kept = 0;
  std::size_t held = 0, booked = 0;
};
bool
zi_split_(MetalCompute* mc, const fs::path& dir, const ZI::Config& cfg,
          int h, bool stream, int floor, ZiSplit* out)
{
  auto link = std::make_shared<PairLink>();
  std::shared_ptr<PairChannel> ch[2];
  std::unique_ptr<ZI> m[2];
  const int runs = stream ? 2 : 1;
  for (int r = 0; r < 2; ++r) {
    m[r] = ZI::load(dir.string(), mc, zi_rank_(cfg, r, h, floor), stream);
    if (m[r] == nullptr) { return false; }
    if (stream) {
      m[r]->set_residency_reserve(test_residency_reserve(1ull << 30));
      m[r]->set_residency_schedule(2 * runs);
    }
    ch[r] = std::make_shared<PairChannel>(link, r);
    std::string err;
    if (!m[r]->begin_tp_generation(ch[r], &err)) {
      std::printf("  begin: %s\n", err.c_str());
      return false;
    }
  }
  for (int k = 0; k < runs; ++k) {
    std::string e0, e1;
    bool ok1 = false;
    std::thread t([&]() { ok1 = zi_steps_(mc, *m[1], &out->s[1], &e1); });
    const bool ok0 = zi_steps_(mc, *m[0], &out->s[0], &e0);
    t.join();
    if (!ok0 || !ok1) {
      std::printf("  rank 0: %s / rank 1: %s\n", e0.c_str(), e1.c_str());
      return false;
    }
  }
  out->fast = m[0]->fast_weights(8);
  // The exchange buffers: held through the picture, gone with its
  // channel (end_tp_generation). MUTATION: kept -- fails.
  const std::size_t exch_held = m[0]->tp_exchange_bytes();
  for (auto& mm : m) { mm->end_tp_generation(); }
  const std::size_t exch_left =
      m[0]->tp_exchange_bytes() + m[1]->tp_exchange_bytes();
  if (exch_held == 0 || exch_left != 0) {
    std::printf("  the exchange buffers: %zu bytes through the picture, "
                "%zu after its channels went (want some, then none)\n",
                exch_held, exch_left);
    return false;
  }
  out->exchanges = ch[0]->exchanges.load();
  out->out_of_step = ch[0]->out_of_step.load() + ch[1]->out_of_step.load();
  out->mismatched = link->mismatched.load();
  out->held = (std::size_t)m[0]->resident_bytes();
  out->booked =
      ZI::tp_held_bytes(dir.string(), zi_rank_(cfg, 0, h, floor), mc);
  out->kept = m[0]->resident_block_count();
  return true;
}

}  // namespace

// A Z-IMAGE DiT SPLIT between two ranks is the unsplit one. Every block
// of the one shape -- the context refiner's, the noise refiner's and the
// main stack's -- is a rank's share: its heads (to_q / to_k / to_v by
// their rows, to_out.0 by its columns) and feed-forward rows (w1 and w3
// by their rows, w2 by its columns). The block is SANDWICH-normed: the
// attention's and the feed-forward's outputs are RMS-normed inside the
// residual, under a tanh gate (the context refiner's ungated), so each
// partial is completed by the exchange BEFORE that post-norm -- two
// exchanges a block. Two forwards on a random small model, 6 of 12 heads
// on rank 0; arms:
//   - bf16, F32 (as Turbo ships: rows read F32 and narrowed) and an
//     8-bit affine pack, preloaded, whole blocks, against a bar of four
//     times the model's own REORDERING FLOOR (the unsplit model against
//     itself on another GEMM route, VPIPE_Z_IMAGE_NO_MMA2: what bf16
//     rounding in another order does to it), 1e-2 at least; what the
//     rank holds is what the plan books, to the byte;
//   - STREAMED, then kept: the same forwards BIT FOR BIT, every block
//     kept;
//   - the share's rows READ alone against the whole weights read and cut
//     (VPIPE_TP_WHOLE_READ): the same bits;
//   - BANDED (a floor of 100: 8 bands of the main stack's 1,088 rows, 8
//     of the noise refiner's 1,024, the caption's 64 whole): two
//     exchanges a band, within the bar; overlapped against band after
//     band (VPIPE_Z_IMAGE_TP_LAG=0), the same bits;
//   - int8 tiers (Per Token on the bf16, Fast on the pack; matrix
//     cores), a rank's share laid out for the tier within its own
//     columns: the bar the tier's own distance from the plain model,
//     half again, and the plain bar at least.
// Both ranks' outputs identical in every arm, every forward.
TEST(tensor_parallel, a_split_z_image_forward_is_the_unsplit_one)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const fs::path dir = fs::temp_directory_path() / "vpipe_zi_tp_small";
  const int h = 6;
  struct Arm { const char* name; const char* dt; bool w8; int tier; };
  const Arm arms[] = {{"bf16", "BF16", false, -1},
                      {"f32", "F32", false, -1},
                      {"w8 pack", "BF16", true, -1},
                      {"bf16, int8 Per Token", "BF16", false, 2},
                      {"w8 pack, int8 Fast", "BF16", true, 1}};
  std::string written;
  // By checkpoint: the whole model untiered, and its bar.
  std::map<std::string, ZiSteps> plain;
  std::map<std::string, double> floor_bar;
  zimage::Layout lay;
  {
    std::string lerr;
    ASSERT_TRUE(zi_layout_(&lay, &lerr));
  }
  for (const Arm& a : arms) {
    const std::string tag = std::string(a.dt) + (a.w8 ? "w8" : "");
    if (written != tag) {
      ASSERT_TRUE(write_zi_small_(dir, a.dt, a.w8));
      written = tag;
    }
    ZI::Config cfg;
    ASSERT_TRUE(ZI::Config::read_dims(dir.string(), &cfg));
    cfg.autotune = false;
    if (a.tier >= 0) {
      if (!mc->supports_matrix_cores()) {
        std::printf("  %-22s SKIP: no matrix cores\n", a.name);
        continue;
      }
      cfg.i8_gemm = true;
      cfg.i8_tier = a.tier;
    }
    ZiSteps whole;
    {
      auto m = ZI::load(dir.string(), mc, cfg, false);
      ASSERT_TRUE(m != nullptr);
      if (a.tier >= 0) { EXPECT_TRUE(m->fast_tier()); }
      std::string err;
      ASSERT_TRUE(zi_steps_(mc, *m, &whole, &err));
    }
    if (a.tier < 0) {
      // THE REORDERING FLOOR: the same model on the steel tiles.
      ZiSteps other;
      {
        EnvScope e("VPIPE_Z_IMAGE_NO_MMA2", "1");
        auto m = ZI::load(dir.string(), mc, cfg, false);
        ASSERT_TRUE(m != nullptr);
        std::string err;
        ASSERT_TRUE(zi_steps_(mc, *m, &other, &err));
      }
      const double fl = std::max(rel_l2_(other.v[0], whole.v[0]),
                                 rel_l2_(other.v[1], whole.v[1]));
      floor_bar[tag] = std::max(1e-2, 4.0 * fl);
      std::printf("  zi %-22s reordering floor %.3e -> bar %.3e\n", a.name,
                  fl, floor_bar[tag]);
      plain[tag] = whole;
    }
    auto bar_for = [&](int k) {
      const double b = floor_bar[tag];
      return a.tier < 0
                 ? b
                 : std::max(b, 1.5 * rel_l2_(whole.v[k], plain[tag].v[k]));
    };
    auto check = [&](const char* what, const ZiSplit& x, std::uint64_t want) {
      for (int k = 0; k < 2; ++k) {
        const double d = rel_l2_(x.s[0].v[k], whole.v[k]);
        std::printf("  zi %-22s %-18s forward %d: split vs unsplit rel-L2 "
                    "%.3e (bar %.3e), ranks %s\n", a.name, what, k, d,
                    bar_for(k),
                    same_bits_(x.s[0].v[k], x.s[1].v[k]) ? "IDENTICAL"
                                                         : "DIFFER");
        EXPECT_TRUE(same_bits_(x.s[0].v[k], x.s[1].v[k]));
        EXPECT_TRUE(d < bar_for(k));
      }
      std::printf("  zi %-22s %-18s %llu exchanges (want %llu); rank 0 "
                  "holds %d int8-layout projections\n", a.name, what,
                  (unsigned long long)x.exchanges,
                  (unsigned long long)want, x.fast);
      EXPECT_TRUE(x.exchanges == want);
      EXPECT_TRUE(x.out_of_step == 0 && x.mismatched == 0);
      // Under a tier every modulated block's projection takes its
      // layout: 7 a block, 3 blocks (the context refiner stays accurate,
      // as the whole model keeps it).
      if (a.tier >= 0 && a.w8) { EXPECT_TRUE(x.fast == 21); }
    };
    auto same = [](const ZiSplit& x, const ZiSplit& y) {
      return same_bits_(x.s[0].v[0], y.s[0].v[0]) &&
             same_bits_(x.s[0].v[1], y.s[0].v[1]);
    };
    // Two forwards, each block's bands over its rows: the context
    // refiner's caption, the noise refiner's image, the main stack's
    // joint sequence.
    auto exchanges = [&](int floor) {
      const std::uint64_t per =
          (std::uint64_t)cfg.n_refiner *
              (tp_row_bands(lay.cap_total, floor).size() +
               tp_row_bands(lay.img_total, floor).size()) +
          (std::uint64_t)cfg.n_layers *
              tp_row_bands(lay.joint_len, floor).size();
      return 2 * 2 * per;
    };
    ZiSplit pre;
    ASSERT_TRUE(zi_split_(mc, dir, cfg, h, false, 0, &pre));
    check("preloaded", pre, exchanges(0));
    std::printf("  zi %-22s rank 0 holds %zu bytes, booked %zu\n", a.name,
                pre.held, pre.booked);
    EXPECT_TRUE(pre.held == pre.booked && pre.booked > 0);
    ZiSplit str;
    ASSERT_TRUE(zi_split_(mc, dir, cfg, h, true, 0, &str));
    std::printf("  zi %-22s streamed: %s the preloaded rank's; %d of %d "
                "blocks kept, %zu bytes held\n", a.name,
                same(str, pre) ? "IDENTICAL to" : "DIFFERS from", str.kept,
                cfg.n_mod_blocks(), str.held);
    EXPECT_TRUE(same(str, pre));
    EXPECT_TRUE(str.kept == cfg.n_mod_blocks());
    EXPECT_TRUE(str.held == pre.booked);
    {
      EnvScope e("VPIPE_TP_WHOLE_READ", "1");
      ZiSplit wr;
      ASSERT_TRUE(zi_split_(mc, dir, cfg, h, false, 0, &wr));
      std::printf("  zi %-22s whole weights read: %s the row reads\n",
                  a.name, same(wr, pre) ? "IDENTICAL to" : "DIFFERS from");
      EXPECT_TRUE(same(wr, pre));
    }
    ZiSplit bd;
    ASSERT_TRUE(zi_split_(mc, dir, cfg, h, false, 100, &bd));
    check("banded (floor 100)", bd, exchanges(100));
    {
      EnvScope e("VPIPE_Z_IMAGE_TP_LAG", "0");
      ZiSplit b0;
      ASSERT_TRUE(zi_split_(mc, dir, cfg, h, false, 100, &b0));
      std::printf("  zi %-22s banded in turn: %s the overlapped bands\n",
                  a.name, same(b0, bd) ? "IDENTICAL to" : "DIFFERS from");
      EXPECT_TRUE(same(b0, bd));
    }
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ---- Fibo 2.0's ranks ------------------------------------------------

namespace {

// A small random Fibo 2.0 DiT on disk (BriaFibo2Transformer2DModel's
// names and config): 12 heads of 128, one noise-refiner block and two
// main ones -- the second an INJECTION block, cross-attending its own
// text bundle -- a one-layer Perceiver, its SwiGLU (dim * 8 // 3) // 64
// * 64 = 4096 wide. `w8`: the stack's block projections an MLX affine
// 8-bit pack in groups of 64, the text path (the Perceiver, the
// injection's cross-attention) and the rest dense, as model-quantize
// leaves them.
bool
write_fibo2_small_(const fs::path& dir, bool w8)
{
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  const int H = 1536, F = 4096, CAP = 512, PD = 128, L = 2, R = 1;
  {
    std::ofstream o(dir / "config.json");
    o << "{\"_class_name\": \"BriaFibo2Transformer2DModel\", \"dim\": " << H
      << ", \"n_heads\": 12, \"n_layers\": " << L
      << ", \"n_refiner_layers\": " << R << ", \"in_channels\": 32, "
      << "\"patch_size\": 2, \"cap_feat_dim\": " << CAP
      << ", \"axes_dims\": [32, 48, 48], \"axes_lens\": [256, 512, 512], "
      << "\"perceiver_num_layers\": 1, \"min_num_gist_tokens\": 64, "
      << "\"max_num_gist_tokens\": 128, \"gist_step\": 64, "
      << "\"gist_min_text_len\": 16, \"gist_max_text_len\": 64, "
      << "\"injection_layer_ids\": [1]}";
  }
  genai::SafetensorsWriter wr(dir.string());
  std::uint64_t st = 0xC2B2AE3D27D4EB4Full ^ (w8 ? 1 : 0);
  auto rnd = [&]() {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    return (float)((double)(st >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
  };
  auto dense = [&](const std::string& n, std::vector<std::int64_t> shape,
                   float sc, float centre = 0.0f) {
    std::size_t k = 1;
    for (auto d : shape) { k *= (std::size_t)d; }
    std::vector<std::uint16_t> v(k);
    for (auto& x : v) { x = bf16_(centre + rnd() * sc); }
    return wr.add(n, "BF16", shape, v.data(), v.size() * 2);
  };
  auto packed = [&](const std::string& n, int N, int K, float sc) {
    const int g = 64;
    std::vector<std::uint32_t> codes((std::size_t)N * K / 4);
    std::vector<std::uint16_t> scales((std::size_t)N * K / g);
    std::vector<std::uint16_t> biases(scales.size());
    std::vector<float> row((std::size_t)K);
    for (int r = 0; r < N; ++r) {
      for (auto& x : row) { x = rnd() * sc; }
      for (int g0 = 0; g0 < K; g0 += g) {
        float lo = row[(std::size_t)g0], hi = lo;
        for (int i = g0; i < g0 + g; ++i) {
          lo = std::min(lo, row[(std::size_t)i]);
          hi = std::max(hi, row[(std::size_t)i]);
        }
        const float s8 = f32_(bf16_(std::max(hi - lo, 1e-6f) / 255.0f));
        const float b8 = f32_(bf16_(lo));
        const std::size_t gi = ((std::size_t)r * K + g0) / g;
        scales[gi] = bf16_(s8);
        biases[gi] = bf16_(b8);
        for (int i = g0; i < g0 + g; ++i) {
          const int q = std::clamp(
              (int)std::lround((row[(std::size_t)i] - b8) / s8), 0, 255);
          const std::size_t e = (std::size_t)r * K + i;
          codes[e / 4] |= (std::uint32_t)q << (8 * (e % 4));
        }
      }
    }
    return wr.add(n + ".weight", "U32", {N, K / 4}, codes.data(),
                  codes.size() * 4) &&
           wr.add(n + ".scales", "BF16", {N, K / g}, scales.data(),
                  scales.size() * 2) &&
           wr.add(n + ".biases", "BF16", {N, K / g}, biases.data(),
                  biases.size() * 2);
  };
  auto proj = [&](const std::string& n, int N, int K, float sc) {
    return w8 ? packed(n, N, K, sc) : dense(n + ".weight", {N, K}, sc);
  };
  // A cross-attention: queries from the hidden width, keys and values
  // from the text's.
  auto cross = [&](const std::string& p) {
    return dense(p + "to_q.weight", {H, H}, 0.03f) &&
           dense(p + "to_k.weight", {H, CAP}, 0.05f) &&
           dense(p + "to_v.weight", {H, CAP}, 0.05f) &&
           dense(p + "to_out.0.weight", {H, H}, 0.03f) &&
           dense(p + "norm_q.weight", {128}, 0.1f, 1.0f) &&
           dense(p + "norm_k.weight", {128}, 0.1f, 1.0f);
  };
  auto block = [&](const std::string& p) {
    return proj(p + "attention.to_q", H, H, 0.03f) &&
           proj(p + "attention.to_k", H, H, 0.03f) &&
           proj(p + "attention.to_v", H, H, 0.03f) &&
           proj(p + "attention.to_out.0", H, H, 0.03f) &&
           dense(p + "attention.norm_q.weight", {128}, 0.1f, 1.0f) &&
           dense(p + "attention.norm_k.weight", {128}, 0.1f, 1.0f) &&
           proj(p + "feed_forward.w1", F, H, 0.03f) &&
           proj(p + "feed_forward.w3", F, H, 0.03f) &&
           proj(p + "feed_forward.w2", H, F, 0.02f) &&
           dense(p + "attention_norm1.weight", {H}, 0.1f, 1.0f) &&
           dense(p + "attention_norm2.weight", {H}, 0.1f, 1.0f) &&
           dense(p + "ffn_norm1.weight", {H}, 0.1f, 1.0f) &&
           dense(p + "ffn_norm2.weight", {H}, 0.1f, 1.0f) &&
           dense(p + "adaLN_modulation.0.weight", {4 * H, 256}, 0.05f) &&
           dense(p + "adaLN_modulation.0.bias", {4 * H}, 0.2f);
  };
  bool ok = dense("x_embedder.weight", {H, PD}, 0.08f) &&
            dense("x_embedder.bias", {H}, 0.05f) &&
            dense("t_embedder.mlp.0.weight", {1024, 256}, 0.06f) &&
            dense("t_embedder.mlp.0.bias", {1024}, 0.05f) &&
            dense("t_embedder.mlp.2.weight", {256, 1024}, 0.03f) &&
            dense("t_embedder.mlp.2.bias", {256}, 0.05f) &&
            dense("final_layer.adaLN_modulation.1.weight", {H, 256},
                  0.05f) &&
            dense("final_layer.adaLN_modulation.1.bias", {H}, 0.05f) &&
            dense("final_layer.linear.weight", {PD, H}, 0.03f) &&
            dense("final_layer.linear.bias", {PD}, 0.05f) &&
            dense("perceiver.latent_queries", {128, H}, 0.5f) &&
            dense("perceiver.norm_out.weight", {H}, 0.1f, 1.0f);
  {
    const std::string p = "perceiver.layers.0.";
    ok = ok && dense(p + "cross_attn_norm.weight", {H}, 0.1f, 1.0f) &&
         dense(p + "self_attn_norm.weight", {H}, 0.1f, 1.0f) &&
         dense(p + "ffn_norm.weight", {H}, 0.1f, 1.0f) &&
         cross(p + "cross_attention.") &&
         dense(p + "self_attention.to_q.weight", {H, H}, 0.03f) &&
         dense(p + "self_attention.to_k.weight", {H, H}, 0.03f) &&
         dense(p + "self_attention.to_v.weight", {H, H}, 0.03f) &&
         dense(p + "self_attention.to_out.0.weight", {H, H}, 0.03f) &&
         dense(p + "self_attention.norm_q.weight", {128}, 0.1f, 1.0f) &&
         dense(p + "self_attention.norm_k.weight", {128}, 0.1f, 1.0f) &&
         dense(p + "ffn.w1.weight", {F, H}, 0.03f) &&
         dense(p + "ffn.w3.weight", {F, H}, 0.03f) &&
         dense(p + "ffn.w2.weight", {H, F}, 0.02f);
  }
  for (int i = 0; i < R; ++i) {
    ok = ok && block("noise_refiner." + std::to_string(i) + ".");
  }
  for (int l = 0; l < L; ++l) {
    ok = ok && block("layers." + std::to_string(l) + ".");
  }
  {
    const std::string p = "layers.1.";
    const std::uint16_t g = bf16_(0.7f);
    ok = ok && dense(p + "cross_attn_norm1.weight", {H}, 0.1f, 1.0f) &&
         dense(p + "cross_attn_norm2.weight", {H}, 0.1f, 1.0f) &&
         cross(p + "cross_attention.") &&
         wr.add(p + "cross_attn_gate", "BF16", {1}, &g, 2);
  }
  return wr.close() && ok;
}

using FB = MetalFibo2Transformer;

// Every Fibo 2.0 forward here: a 32 x 32 patch grid (1,024 image rows,
// the int8 tiers' floor) and one 16 x 16 CONTEXT picture after them --
// an edit's reference, which enters as rows of its own -- then the gist
// of a 16-token text (64 rows).
constexpr int kFbText = 16;
bool
fb_layout_(const FB& m, fibo2::Layout* lay, std::string* err)
{
  const FB::Config& c = m.config();
  const int g = fibo2::gist_count(kFbText, c.min_gist, c.max_gist,
                                  c.gist_step, c.gist_min_text,
                                  c.gist_max_text);
  return fibo2::build_layout(32, 32, {{16, 16}}, g, c.max_gist, lay, err);
}

// The text, then two forwards of `m` (sigma 0.9, then 0.6) on fixed
// random inputs -- the first runs the text's pass (the Perceiver, the
// injection's keys and values); each one's image rows as f32.
struct FbSteps {
  std::vector<float> v[2];
};
bool
fb_steps_(MetalCompute* mc, FB& m, FbSteps* out, std::string* err)
{
  fibo2::Layout lay;
  if (!fb_layout_(m, &lay, err)) { return false; }
  std::uint64_t st = 9191;
  auto buf = [&](std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      st ^= st << 13; st ^= st >> 7; st ^= st << 17;
      d[i] = bf16_((float)((double)(st >> 11) / 9007199254740992.0) - 0.5f);
    }
    return b;
  };
  const FB::Config& c = m.config();
  SharedBuffer text =
      buf((std::size_t)c.n_bundles() * kFbText * (std::size_t)c.cap_dim);
  if (!m.set_text(text, kFbText, err)) { return false; }
  SharedBuffer lat = buf((std::size_t)lay.refiner_len() * c.patch_dim());
  for (int k = 0; k < 2; ++k) {
    FB::Request req;
    req.latents = &lat;
    req.layout = &lay;
    req.timestep = k == 0 ? 0.9f : 0.6f;
    SharedBuffer o = m.forward(req, err);
    if (o.empty()) { return false; }
    const auto* p = static_cast<const std::uint16_t*>(o.contents());
    const std::size_t n = (std::size_t)lay.img_tokens * c.patch_dim();
    out->v[k].resize(n);
    for (std::size_t i = 0; i < n; ++i) { out->v[k][i] = f32_(p[i]); }
  }
  return true;
}

FB::Config
fb_rank_(const FB::Config& cfg, int r, int h, int floor)
{
  FB::Config c = cfg;
  const int F = cfg.ffn_inner;
  // In whole groups of 64, as a pack's columns are cut.
  const int f = F * h / cfg.n_heads / 64 * 64;
  c.tp = TpShare{};
  c.tp.rank = r;
  c.tp.ranks = 2;
  c.tp.n_heads = cfg.n_heads;
  c.tp.n_ffn = F;
  c.tp.head0 = r == 0 ? 0 : h;
  c.tp.head1 = r == 0 ? h : cfg.n_heads;
  c.tp.ffn0 = r == 0 ? 0 : f;
  c.tp.ffn1 = r == 0 ? f : F;
  if (floor > 0) {
    c.tp.extra = FlexData::make_object();
    c.tp.extra.as_object().insert_or_assign(
        tp_extra::kMinBandRows, FlexData::make_int(floor));
  }
  return c;
}

// As zi_split_, for Fibo 2.0: what the rank holds read with its text let
// go (the text's pass is the prompt's, not the weights').
struct FbSplit {
  FbSteps s[2];
  std::uint64_t exchanges = 0;
  int out_of_step = 0, mismatched = 0;
  int fast = 0, kept = 0;
  std::size_t held = 0, booked = 0;
};
bool
fb_split_(MetalCompute* mc, const fs::path& dir, const FB::Config& cfg,
          int h, bool stream, int floor, FbSplit* out)
{
  auto link = std::make_shared<PairLink>();
  std::shared_ptr<PairChannel> ch[2];
  std::unique_ptr<FB> m[2];
  const int runs = stream ? 2 : 1;
  for (int r = 0; r < 2; ++r) {
    m[r] = FB::load(dir.string(), mc, fb_rank_(cfg, r, h, floor), stream);
    if (m[r] == nullptr) { return false; }
    if (stream) {
      m[r]->set_residency_reserve(test_residency_reserve(1ull << 30));
      m[r]->set_residency_schedule(2 * runs);
    }
    ch[r] = std::make_shared<PairChannel>(link, r);
    std::string err;
    if (!m[r]->begin_tp_generation(ch[r], &err)) {
      std::printf("  begin: %s\n", err.c_str());
      return false;
    }
  }
  for (int k = 0; k < runs; ++k) {
    std::string e0, e1;
    bool ok1 = false;
    std::thread t([&]() { ok1 = fb_steps_(mc, *m[1], &out->s[1], &e1); });
    const bool ok0 = fb_steps_(mc, *m[0], &out->s[0], &e0);
    t.join();
    if (!ok0 || !ok1) {
      std::printf("  rank 0: %s / rank 1: %s\n", e0.c_str(), e1.c_str());
      return false;
    }
  }
  out->fast = m[0]->fast_weights(8);
  // The exchange buffers: held through the picture, gone with its
  // channel (end_tp_generation). MUTATION: kept -- fails.
  const std::size_t exch_held = m[0]->tp_exchange_bytes();
  for (auto& mm : m) { mm->end_tp_generation(); }
  const std::size_t exch_left =
      m[0]->tp_exchange_bytes() + m[1]->tp_exchange_bytes();
  if (exch_held == 0 || exch_left != 0) {
    std::printf("  the exchange buffers: %zu bytes through the picture, "
                "%zu after its channels went (want some, then none)\n",
                exch_held, exch_left);
    return false;
  }
  out->exchanges = ch[0]->exchanges.load();
  out->out_of_step = ch[0]->out_of_step.load() + ch[1]->out_of_step.load();
  out->mismatched = link->mismatched.load();
  m[0]->clear_text(0);
  out->held = (std::size_t)m[0]->resident_bytes();
  out->booked =
      FB::tp_held_bytes(dir.string(), fb_rank_(cfg, 0, h, floor), mc);
  out->kept = m[0]->resident_block_count();
  return true;
}

}  // namespace

// A FIBO 2.0 DiT SPLIT between two ranks is the unsplit one. Its block
// is Z-Image's -- sandwich-normed, two partials a block, each completed
// before its post-norm and gate -- and its INJECTION blocks cross-attend
// a text bundle after the attention: their cross-attention is cut by the
// same heads (to_q, to_k / to_v over the text, by their rows; to_out.0 by
// its columns), the text's keys and values its heads' alone, and its
// partial completed before cross_attn_norm2 and the tanh gate -- a third
// exchange. The Perceiver runs whole on every rank. An edit's reference
// enters as CONTEXT rows after the image's, modulated at t = 0 (the
// per-token modulation): a band of a split block may straddle the two.
// Arms as Z-Image's (a_split_z_image_forward_is_the_unsplit_one) -- bf16
// and an 8-bit pack, int8 Per Token and Fast; preloaded, streamed, the
// rows read alone, banded (a floor of 100: the refiner's 1,280 rows and
// the main stack's 1,344 in 8 bands each, the image's last row inside
// one) overlapped and in turn -- against four times the model's own
// reordering floor, 1e-2 at least. Both ranks identical in every arm.
TEST(tensor_parallel, a_split_fibo2_forward_is_the_unsplit_one)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const fs::path dir = fs::temp_directory_path() / "vpipe_fibo2_tp_small";
  const int h = 6;
  struct Arm { const char* name; bool w8; int tier; };
  const Arm arms[] = {{"bf16", false, -1}, {"w8 pack", true, -1},
                      {"bf16, int8 Per Token", false, 2},
                      {"w8 pack, int8 Fast", true, 1}};
  int written = -1;
  std::map<int, FbSteps> plain;
  std::map<int, double> floor_bar;
  for (const Arm& a : arms) {
    if (written != (int)a.w8) {
      ASSERT_TRUE(write_fibo2_small_(dir, a.w8));
      written = (int)a.w8;
    }
    FB::Config cfg;
    ASSERT_TRUE(FB::Config::read_dims(dir.string(), &cfg));
    cfg.autotune = false;
    if (a.tier >= 0) {
      if (!mc->supports_matrix_cores()) {
        std::printf("  %-22s SKIP: no matrix cores\n", a.name);
        continue;
      }
      cfg.i8_gemm = true;
      cfg.i8_tier = a.tier;
    }
    FbSteps whole;
    fibo2::Layout lay;
    {
      auto m = FB::load(dir.string(), mc, cfg, false);
      ASSERT_TRUE(m != nullptr);
      if (a.tier >= 0) { EXPECT_TRUE(m->fast_tier()); }
      std::string err;
      ASSERT_TRUE(fb_steps_(mc, *m, &whole, &err));
      ASSERT_TRUE(fb_layout_(*m, &lay, &err));
    }
    if (a.tier < 0) {
      FbSteps other;
      {
        EnvScope e("VPIPE_FIBO2_NO_MMA2", "1");
        auto m = FB::load(dir.string(), mc, cfg, false);
        ASSERT_TRUE(m != nullptr);
        std::string err;
        ASSERT_TRUE(fb_steps_(mc, *m, &other, &err));
      }
      const double fl = std::max(rel_l2_(other.v[0], whole.v[0]),
                                 rel_l2_(other.v[1], whole.v[1]));
      floor_bar[a.w8] = std::max(1e-2, 4.0 * fl);
      std::printf("  fibo2 %-22s reordering floor %.3e -> bar %.3e\n",
                  a.name, fl, floor_bar[a.w8]);
      plain[a.w8] = whole;
    }
    auto bar_for = [&](int k) {
      const double b = floor_bar[a.w8];
      return a.tier < 0
                 ? b
                 : std::max(b, 1.5 * rel_l2_(whole.v[k], plain[a.w8].v[k]));
    };
    auto check = [&](const char* what, const FbSplit& x, std::uint64_t want) {
      for (int k = 0; k < 2; ++k) {
        const double d = rel_l2_(x.s[0].v[k], whole.v[k]);
        std::printf("  fibo2 %-22s %-18s forward %d: split vs unsplit "
                    "rel-L2 %.3e (bar %.3e), ranks %s\n", a.name, what, k, d,
                    bar_for(k),
                    same_bits_(x.s[0].v[k], x.s[1].v[k]) ? "IDENTICAL"
                                                         : "DIFFER");
        EXPECT_TRUE(same_bits_(x.s[0].v[k], x.s[1].v[k]));
        EXPECT_TRUE(d < bar_for(k));
      }
      std::printf("  fibo2 %-22s %-18s %llu exchanges (want %llu); rank 0 "
                  "holds %d int8-layout projections\n", a.name, what,
                  (unsigned long long)x.exchanges,
                  (unsigned long long)want, x.fast);
      EXPECT_TRUE(x.exchanges == want);
      EXPECT_TRUE(x.out_of_step == 0 && x.mismatched == 0);
      // Under a tier every block's projection takes its layout: 7 a
      // block, 3 blocks.
      if (a.tier >= 0 && a.w8) { EXPECT_TRUE(x.fast == 21); }
    };
    auto same = [](const FbSplit& x, const FbSplit& y) {
      return same_bits_(x.s[0].v[0], y.s[0].v[0]) &&
             same_bits_(x.s[0].v[1], y.s[0].v[1]);
    };
    // Two forwards: the refiner's two partials a band over its rows, the
    // plain main block's two and the injection block's three over the
    // joint sequence's.
    auto exchanges = [&](int floor) {
      const std::uint64_t per =
          2 * (std::uint64_t)cfg.n_refiner *
              tp_row_bands(lay.refiner_len(), floor).size() +
          (2 * ((std::uint64_t)cfg.n_layers - 1) + 3) *
              tp_row_bands(lay.joint_len(), floor).size();
      return 2 * per;
    };
    FbSplit pre;
    ASSERT_TRUE(fb_split_(mc, dir, cfg, h, false, 0, &pre));
    check("preloaded", pre, exchanges(0));
    std::printf("  fibo2 %-22s rank 0 holds %zu bytes, booked %zu\n",
                a.name, pre.held, pre.booked);
    EXPECT_TRUE(pre.held == pre.booked && pre.booked > 0);
    FbSplit str;
    ASSERT_TRUE(fb_split_(mc, dir, cfg, h, true, 0, &str));
    std::printf("  fibo2 %-22s streamed: %s the preloaded rank's; %d of %d "
                "blocks kept, %zu bytes held\n", a.name,
                same(str, pre) ? "IDENTICAL to" : "DIFFERS from", str.kept,
                cfg.n_mod_blocks(), str.held);
    EXPECT_TRUE(same(str, pre));
    EXPECT_TRUE(str.kept == cfg.n_mod_blocks());
    EXPECT_TRUE(str.held == pre.booked);
    {
      EnvScope e("VPIPE_TP_WHOLE_READ", "1");
      FbSplit wr;
      ASSERT_TRUE(fb_split_(mc, dir, cfg, h, false, 0, &wr));
      std::printf("  fibo2 %-22s whole weights read: %s the row reads\n",
                  a.name, same(wr, pre) ? "IDENTICAL to" : "DIFFERS from");
      EXPECT_TRUE(same(wr, pre));
    }
    FbSplit bd;
    ASSERT_TRUE(fb_split_(mc, dir, cfg, h, false, 100, &bd));
    check("banded (floor 100)", bd, exchanges(100));
    {
      EnvScope e("VPIPE_FIBO2_TP_LAG", "0");
      FbSplit b0;
      ASSERT_TRUE(fb_split_(mc, dir, cfg, h, false, 100, &b0));
      std::printf("  fibo2 %-22s banded in turn: %s the overlapped bands\n",
                  a.name, same(b0, bd) ? "IDENTICAL to" : "DIFFERS from");
      EXPECT_TRUE(same(b0, bd));
    }
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// The same on a REAL checkpoint, where one is given: Fibo-2.0-edit-turbo
// (an 8-bit pack, model-quantize's, or the release's bf16) at
// VPIPE_FIBO2_TP_TEST_MODEL_PATH -- its transformer/ directory, or the
// root holding one. Two forwards of the split (14 of 28 heads a rank,
// the 44 + 2 blocks, five injection blocks), preloaded and banded,
// against the whole model's, on the small test's geometry and random
// text; the bar four times the checkpoint's own reordering floor, 1e-2
// at least -- a gross 1e-1 on a box with one route (MEASURED on the M4
// Pro, the release's bf16: 2.1-3.2e-2, held == booked). The whole model
// and the two ranks are loaded one after the other.
TEST(tensor_parallel, a_split_real_fibo2_forward_is_the_unsplit_one)
{
  const char* env = std::getenv("VPIPE_FIBO2_TP_TEST_MODEL_PATH");
  if (env == nullptr || *env == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  std::error_code dec;
  const std::string tdir =
      fs::is_directory(fs::path(env) / "transformer", dec)
          ? (fs::path(env) / "transformer").string()
          : std::string(env);
  const char* dir = tdir.c_str();
  FB::Config cfg;
  const bool dims = FB::Config::read_dims(dir, &cfg);
  ASSERT_TRUE(dims);
  if (!dims) { return; }
  cfg.autotune = false;
  FbSteps whole, other;
  for (int pass = 0; pass < 2; ++pass) {
    std::unique_ptr<EnvScope> e;
    if (pass == 1) {
      e = std::make_unique<EnvScope>("VPIPE_FIBO2_NO_MMA2", "1");
    }
    auto m = FB::load(dir, mc, cfg, false);
    ASSERT_TRUE(m != nullptr);
    // A model that did not load stops here, not read through null.
    if (m == nullptr) { return; }
    std::string err;
    ASSERT_TRUE(fb_steps_(mc, *m, pass == 0 ? &whole : &other, &err));
  }
  const double fl = std::max(rel_l2_(other.v[0], whole.v[0]),
                             rel_l2_(other.v[1], whole.v[1]));
  // One route (no matrix cores: the floor reads 0): a GROSS bar, as the
  // real MOSS-Image test's -- the split moves 46 blocks of random text
  // 2.1-3.2e-2 there (M4 Pro), as it does beside a floor on an M5.
  const double bar = fl > 0.0 ? std::max(1e-2, 4.0 * fl) : 1e-1;
  std::printf("  fibo2 real: reordering floor %.3e -> bar %.3e\n", fl, bar);
  const int h = cfg.n_heads / 2;
  for (const int floor : {0, 512}) {
    FbSplit x;
    ASSERT_TRUE(fb_split_(mc, fs::path(dir), cfg, h, false, floor, &x));
    for (int k = 0; k < 2; ++k) {
      const double d = rel_l2_(x.s[0].v[k], whole.v[k]);
      std::printf("  fibo2 real floor %d forward %d: split vs unsplit "
                  "rel-L2 %.3e (bar %.3e), ranks %s; %llu exchanges\n",
                  floor, k, d, bar,
                  same_bits_(x.s[0].v[k], x.s[1].v[k]) ? "IDENTICAL"
                                                       : "DIFFER",
                  (unsigned long long)x.exchanges);
      EXPECT_TRUE(same_bits_(x.s[0].v[k], x.s[1].v[k]));
      EXPECT_TRUE(d < bar);
    }
    std::printf("  fibo2 real floor %d: rank 0 holds %zu bytes, booked "
                "%zu\n", floor, x.held, x.booked);
    EXPECT_TRUE(x.held == x.booked);
    EXPECT_TRUE(x.out_of_step == 0 && x.mismatched == 0);
  }
}

// ---- generate-image's ranks ------------------------------------------

namespace {

// A FLUX.3 model ROOT generate-image opens: the small DiT (dense bf16)
// under transformer/ -- a flow map, or the undistilled shape the stage
// refuses at its first picture -- and the checkpoint's sampler.
fs::path
flux3_root_(const char* tag, bool flow_map, F3Small* s)
{
  const fs::path root = fs::temp_directory_path() / tag;
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root, ec);
  s->dir = root / "transformer";
  if (!write_flux3_small_(*s, false, flow_map)) { return {}; }
  std::ofstream o(root / "config.json");
  o << "{\"sampler\": {\"steps\": 2}}";
  return root;
}

// `count` conditioning beats, the FLUX.3 conditioner's shape: [rows,
// joint_dim] bf16, fixed.
class TpCondSource : public TypedStage<TpCondSource> {
public:
  static constexpr const char* kTypeName = "ut-tp-cond-source";
  using TypedStage::TypedStage;
  TensorBeat beat;
  int count = 1, sent = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (sent >= count) { ctx.signal_done(); co_return; }
    ++sent;
    TensorBeat tb = beat;
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
    co_return;
  }
};

TensorBeat
flux3_cond_(int rows, int jd)
{
  TensorBeat b;
  b.dtype = TensorBeat::DType::Bf16;
  b.shape = {rows, jd};
  b.resize_contiguous((std::size_t)rows * jd * 2);
  auto* d = reinterpret_cast<std::uint16_t*>(b.data.data());
  std::uint64_t st = 99;
  for (std::size_t i = 0; i < (std::size_t)rows * jd; ++i) {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    d[i] = bf16_((float)((double)(st >> 11) / 9007199254740992.0) - 0.5f);
  }
  return b;
}

// One generate-image launch in `sess`: `prompts` pictures at 256^2, two
// steps, the seed kept. Its latents' bytes, whether the stage took a DiT
// kept warm, and what the plan counts for the DiT once the pictures are
// made (the stage holds it until the launch stops). `idle`, when given,
// runs then too -- the stage and its DiT still there.
struct ImageRun {
  std::vector<std::vector<std::uint8_t>> latents;
  bool taken_warm = false;
  std::size_t planned = 0;
};
// `cond`, when given, is the conditioning each picture reads (another
// family's shape); else FLUX.3's.
void
flux3_image_run_(Session& sess, const fs::path& root, int prompts,
                 const FlexData& tp, double keep_loaded, ImageRun* out,
                 const std::function<void()>& idle = {},
                 const TensorBeat* cond = nullptr)
{
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src_u = std::make_unique<TpCondSource>(
      &sess, "src", std::vector<InEdge>{}, FlexData::make_object());
  src_u->beat = cond != nullptr ? *cond : flux3_cond_(40, F3Small{}.jd);
  src_u->count = prompts;
  auto* src = static_cast<TpCondSource*>(pl->insert_stage(std::move(src_u)));
  src->allocate_oports(1);
  FlexData cfg = FlexData::make_object();
  auto o = cfg.as_object();
  o.insert_or_assign("hf_dir", FlexData::make_string(root.string()));
  o.insert_or_assign("width", FlexData::make_int(256));
  o.insert_or_assign("height", FlexData::make_int(256));
  o.insert_or_assign("steps", FlexData::make_int(2));
  o.insert_or_assign("seed", FlexData::make_int(7));
  o.insert_or_assign("seed_sequence", FlexData::make_string("keep"));
  o.insert_or_assign("autotune", FlexData::make_bool(false));
  if (keep_loaded > 0.0) {
    o.insert_or_assign("keep_loaded", FlexData::make_real(keep_loaded));
  }
  if (tp.is_object()) { o.insert_or_assign("tensor_parallel", tp); }
  auto gen_u = std::make_unique<GenerateImageStage>(
      &sess, "gen", std::vector<InEdge>{{src, 0}}, cfg);
  auto* gen = static_cast<GenerateImageStage*>(
      pl->insert_stage(std::move(gen_u)));
  auto col_u = std::make_unique<TpCollector>(
      &sess, "latents", std::vector<InEdge>{{gen, 0}},
      FlexData::make_object());
  auto* col = static_cast<TpCollector*>(pl->insert_stage(std::move(col_u)));
  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return; }
  rt.wait_idle();
  out->planned = sess.services()->generative_model_manager()
                     ->planned_weight_bytes((root / "transformer").string());
  if (idle) { idle(); }
  rt.stop();
  out->latents = col->bytes;
  out->taken_warm = gen->dit_taken_warm();
}

FlexData
image_tp_cfg_(int rank, const char* heads, const char* transport,
              int floor = 0)
{
  std::string j = "{\"rank\": " + std::to_string(rank) +
                  ", \"ranks\": 2, \"heads\": " + heads +
                  ", \"transport\": \"" + transport + "\"";
  if (floor > 0) { j += ", \"min_band_rows\": " + std::to_string(floor); }
  return FlexData::from_json(j + "}");
}

std::vector<float>
latent_f32_(const std::vector<std::uint8_t>& b)
{
  std::vector<float> v(b.size() / 4);
  std::memcpy(v.data(), b.data(), v.size() * 4);
  return v;
}

}  // namespace

// A SPLIT FLUX.3 THROUGH generate-image: two ranks, each a stage in a
// session of its own, over the stage pair transport, against the whole
// model through the same stage. Two pictures a launch (the seed kept):
//   - both ranks' latents the same bytes, each within the bar of the
//     whole model's; a channel opened per picture per rank, each its own
//     exchange count from 0 (none out of step), each given the stop and
//     LET GO once its picture is made, the stop cleared; the shares the
//     bag named (heads [0, 4) / [4, 6), the feed-forward even);
//   - what the plan counts for the rank's DiT once its pictures are made
//     is what the rank holds -- its share, every block kept -- as the
//     plan booked it (tp_held_bytes), not the checkpoint;
//   - the WARM KEY: a rank's DiT kept warm (`keep_loaded`) is not handed
//     to the whole model the same session launches next, nor the whole
//     model to a rank after it; a rank of the same share takes its own;
//   - a picture that THROWS after its channel opened -- the undistilled
//     shape, which generate_flux3_ refuses through Session::error -- still
//     lets the channel go and clears its stop.
// MUTATIONS, each failing it: the picture scope removed (the throw keeps
// the channel), the claim without the share's held bytes (the plan
// counts the checkpoint), the share left out of the warm key (the whole
// model takes the rank's DiT and fails its forwards).
TEST(tensor_parallel, a_split_flux3_runs_through_the_stage)
{
  {
    Session probe;
    if (probe.metal_compute() == nullptr) { return; }
  }
  StagePairTransport* tr = stage_pair_transport_();
  tr->reset();
  F3Small s;
  const fs::path root = flux3_root_("vpipe_flux3_tp_stage", true, &s);
  ASSERT_TRUE(!root.empty());
  const std::string dit = (root / "transformer").string();
  constexpr double kBar = 1e-2;

  ImageRun whole;
  {
    Session sess;
    flux3_image_run_(sess, root, 2, FlexData(), 0.0, &whole);
  }
  ASSERT_TRUE(whole.latents.size() == 2);
  if (whole.latents.size() != 2) { return; }

  const char* kH0 = "[0, 4]";
  const char* kH1 = "[4, 6]";
  ImageRun r0, r1;
  bool idle_let_go = true;
  {
    Session s0, s1;
    std::thread t([&]() {
      flux3_image_run_(s1, root, 2, image_tp_cfg_(1, kH1, tr->name().data()),
                       0.0, &r1);
    });
    flux3_image_run_(s0, root, 2, image_tp_cfg_(0, kH0, tr->name().data()),
                     0.0, &r0, [&]() {
                       // Rank 0's two pictures' channels, its DiT still
                       // there: let go as each picture ended.
                       for (const auto& ch : tr->channels) {
                         if (ch->me == 0) {
                           idle_let_go = idle_let_go && ch.use_count() == 1;
                         }
                       }
                     });
    t.join();
  }
  ASSERT_TRUE(r0.latents.size() == 2 && r1.latents.size() == 2);
  if (r0.latents.size() != 2 || r1.latents.size() != 2) { return; }
  // Per forward: the image and text stacks' blocks, then the joint's,
  // one exchange each; two steps a picture.
  const std::uint64_t per_picture =
      2 * ((std::uint64_t)s.modes * 2 + (std::uint64_t)s.joint);
  bool let_go = true, stopped = true, counts = true, kinds = true;
  int out_of_step = 0;
  for (const auto& ch : tr->channels) {
    let_go = let_go && ch.use_count() == 1;
    stopped = stopped && ch->stop_seen.load() && !ch->stop_now.load();
    counts = counts && ch->exchanges.load() == per_picture;
    out_of_step += ch->out_of_step.load();
  }
  for (const TpShare& sh : tr->shares) {
    kinds = kinds && sh.ranks == 2 && sh.n_heads == 6 &&
            sh.heads() == (sh.rank == 0 ? 4 : 2) &&
            sh.ffn() == sh.n_ffn / 2;
  }
  // Rank 0's share as its bag reads (heads given, the feed-forward
  // split evenly), and what the plan books for it.
  F3::Config c0;
  ASSERT_TRUE(F3::Config::read_dims(dit, &c0));
  ASSERT_TRUE(TpShare::from_flex(image_tp_cfg_(0, kH0, tr->name().data()),
                                 c0.n_heads, c0.mlp_hidden, &c0.tp,
                                 nullptr));
  std::size_t floor = 0;
  const std::size_t booked = F3::tp_held_bytes(dit, c0, &floor);
  // What it holds once its pictures are made: every block they ran,
  // kept -- not the reference stack's, which a picture with no
  // reference never reads (floor = trunk + two blocks).
  const int nb = 3 * s.modes + s.joint;
  const std::size_t blk = (booked - floor) / (std::size_t)(nb - 2);
  const std::size_t held = booked - (std::size_t)s.modes * blk;
  for (int k = 0; k < 2; ++k) {
    const double d = rel_l2_(latent_f32_(r0.latents[(std::size_t)k]),
                             latent_f32_(whole.latents[(std::size_t)k]));
    std::printf("[tensor_parallel] flux3 through the stage, picture %d: "
                "split vs whole rel-L2 %.3e, ranks %s\n", k, d,
                r0.latents[(std::size_t)k] == r1.latents[(std::size_t)k]
                    ? "IDENTICAL" : "DIFFER");
    EXPECT_TRUE(r0.latents[(std::size_t)k] == r1.latents[(std::size_t)k]);
    EXPECT_TRUE(d < kBar);
  }
  std::printf("[tensor_parallel] %zu channels (want 4), %s, stop %s, %s "
              "%llu exchanges a picture, %d out of step; rank 0's DiT "
              "planned at %zu bytes once its pictures were made (its "
              "share less the reference stack: %zu), booked at %zu (the "
              "checkpoint %zu)\n", tr->channels.size(),
              let_go ? "every one let go" : "SOME KEPT",
              stopped ? "set and cleared" : "NOT CLEARED",
              counts ? "each" : "NOT each",
              (unsigned long long)per_picture, out_of_step, r0.planned,
              held, booked, model_memory::dir_weights_bytes(dit));
  EXPECT_TRUE(tr->channels.size() == 4);
  EXPECT_TRUE(let_go && stopped && counts && kinds && out_of_step == 0);
  EXPECT_TRUE(idle_let_go);
  EXPECT_TRUE(booked > 0 && r0.planned == held);

  // THE WARM KEY, in one session for rank 0 (its peer in another): a
  // rank kept warm, then the whole model, then the rank again -- and the
  // rank once more, which takes the one kept before it.
  tr->reset();
  {
    Session s0, s1;
    auto rank_launch = [&](ImageRun* a) {
      ImageRun b;
      std::thread t([&]() {
        flux3_image_run_(s1, root, 1,
                         image_tp_cfg_(1, kH1, tr->name().data()), 60.0, &b);
      });
      flux3_image_run_(s0, root, 1, image_tp_cfg_(0, kH0, tr->name().data()),
                       60.0, a);
      t.join();
    };
    ImageRun a, w, c, d;
    rank_launch(&a);
    flux3_image_run_(s0, root, 1, FlexData(), 60.0, &w);
    rank_launch(&c);
    rank_launch(&d);
    std::printf("[tensor_parallel] warm key: rank (warm %d), whole (warm "
                "%d, %s the whole run's), rank (warm %d), rank (warm %d)\n",
                a.taken_warm, w.taken_warm,
                w.latents.size() == 1 && w.latents[0] == whole.latents[0]
                    ? "IDENTICAL to" : "NOT",
                c.taken_warm, d.taken_warm);
    EXPECT_TRUE(!a.taken_warm && !w.taken_warm && !c.taken_warm);
    EXPECT_TRUE(d.taken_warm);
    EXPECT_TRUE(w.latents.size() == 1 && w.latents[0] == whole.latents[0]);
    EXPECT_TRUE(c.latents.size() == 1 && d.latents.size() == 1 &&
                c.latents[0] == r0.latents[0] &&
                d.latents[0] == r0.latents[0]);
  }

  // A picture that throws once its channel is open.
  tr->reset();
  {
    F3Small s2;
    const fs::path base = flux3_root_("vpipe_flux3_tp_base", false, &s2);
    ASSERT_TRUE(!base.empty());
    // Read while the stage and its DiT are still there: the DiT's end
    // lets the channel go too, so after the launch nothing could tell.
    Session sess;
    ImageRun t;
    bool opened = false, gone = false, cleared = false;
    flux3_image_run_(sess, base, 1, image_tp_cfg_(0, kH0, tr->name().data()),
                     0.0, &t, [&]() {
                       opened = tr->channels.size() == 1;
                       gone = opened && tr->channels[0].use_count() == 1;
                       cleared = opened &&
                                 tr->channels[0]->stop_seen.load() &&
                                 !tr->channels[0]->stop_now.load();
                     });
    std::printf("[tensor_parallel] a picture refused once its channel was "
                "open: %zu latents, the channel %s, its stop %s\n",
                t.latents.size(), gone ? "let go" : "KEPT",
                cleared ? "cleared" : "NOT CLEARED");
    EXPECT_TRUE(t.latents.empty() && opened && gone && cleared);
    std::error_code ec;
    fs::remove_all(base, ec);
  }
  tr->reset();
  std::error_code ec;
  fs::remove_all(root, ec);
}

// A RANK'S PLAN IS ITS SHARE, for every split family (FLUX.3, Qwen-Image-
// 2.1, Z-Image, Fibo 2.0): generate-image's DiT
// claim books the rank at what it holds with every block kept (the
// claim's held bytes: the transformers' tp_held_bytes) and streamable to
// its floor -- the trunk and two blocks' shares -- the report says the
// same, no Neural Engine module is claimed for a rank (it runs none),
// and the denoise's activations are the rank's -- its widths and its
// exchange buffers, less than the whole model's. Nothing loads.
TEST(tensor_parallel, a_ranks_plan_is_its_share)
{
  Session sess;
  struct Fam { const char* name; fs::path root; std::string dit; };
  F3Small fs3;
  const fs::path froot = flux3_root_("vpipe_flux3_tp_plan", true, &fs3);
  ASSERT_TRUE(!froot.empty());
  const fs::path qroot = fs::temp_directory_path() / "vpipe_qi21_tp_plan";
  {
    std::error_code ec;
    fs::remove_all(qroot, ec);
    fs::create_directories(qroot, ec);
    ASSERT_TRUE(write_qi21_small_(qroot / "transformer", false));
  }
  const fs::path mroot = fs::temp_directory_path() / "vpipe_moss_tp_plan";
  {
    std::error_code ec;
    fs::remove_all(mroot, ec);
    fs::create_directories(mroot, ec);
    ASSERT_TRUE(write_moss_small_(mroot / "transformer", false));
  }
  const fs::path zroot = fs::temp_directory_path() / "vpipe_zi_tp_plan";
  const fs::path broot = fs::temp_directory_path() / "vpipe_fibo2_tp_plan";
  {
    std::error_code ec;
    for (const fs::path& r : {zroot, broot}) {
      fs::remove_all(r, ec);
      fs::create_directories(r, ec);
    }
    ASSERT_TRUE(write_zi_small_(zroot / "transformer", "BF16", false));
    ASSERT_TRUE(write_fibo2_small_(broot / "transformer", false));
  }
  std::vector<Fam> fams = {{"flux3", froot,
                            (froot / "transformer").string()},
                           {"qwen-image-21", qroot,
                            (qroot / "transformer").string()},
                           {"moss-image", mroot,
                            (mroot / "transformer").string()},
                           {"z-image", zroot,
                            (zroot / "transformer").string()},
                           {"fibo2", broot,
                            (broot / "transformer").string()}};
  // Krea-2's widths are Turbo's alone (its Config reads none): the real
  // checkpoint's headers, where one is given -- nothing loads.
  if (const char* kr = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
      kr != nullptr && *kr != '\0') {
    fams.push_back({"krea2", fs::path(kr),
                    (fs::path(kr) / "transformer").string()});
  } else {
    std::printf("[tensor_parallel] krea2 plan: SKIP "
                "(VPIPE_KREA2_TEST_MODEL_PATH unset)\n");
  }
  for (const Fam& f : fams) {
    auto plan = [&](bool split, std::size_t* held, std::size_t* floor,
                    std::size_t* report, int* coreml,
                    std::size_t* scratch) {
      FlexData cfg = FlexData::make_object();
      auto o = cfg.as_object();
      o.insert_or_assign("hf_dir", FlexData::make_string(f.root.string()));
      o.insert_or_assign("width", FlexData::make_int(1024));
      o.insert_or_assign("height", FlexData::make_int(1024));
      o.insert_or_assign("ane_ffn", FlexData::make_bool(true));
      if (split) {
        o.insert_or_assign("tensor_parallel",
                           image_tp_cfg_(0, "[0, 4]", "none"));
      }
      GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{}, cfg);
      *held = *floor = *report = *scratch = 0;
      *coreml = 0;
      for (const ResourceClaim& c : stage.declare_resources()) {
        std::size_t ub = 0;
        int un = 0;
        if (model_memory::parse_coreml_claim(c, nullptr, &ub, &un)) {
          ++*coreml;
        }
        if (c.key == f.dit) {
          *floor = c.floor_bytes;
          *held = (std::size_t)bag::integer(
              c.extra, model_memory::kClaimHeldBytes, 0);
        }
        // A scratch claim's key is "label|bytes".
        if (c.kind == model_memory::kScratchKind &&
            c.key.find("-dit:t2i|") != std::string::npos) {
          *scratch = std::max(
              *scratch, (std::size_t)std::stoull(
                            c.key.substr(c.key.rfind('|') + 1)));
        }
      }
      for (const auto& h : stage.declare_memory().holdings) {
        if (h.source == f.dit) { *report = h.preload; }
      }
    };
    std::size_t held = 0, floor = 0, report = 0, scratch = 0;
    std::size_t wheld = 0, wfloor = 0, wreport = 0, wscratch = 0;
    int coreml = 0, wcoreml = 0;
    plan(true, &held, &floor, &report, &coreml, &scratch);
    plan(false, &wheld, &wfloor, &wreport, &wcoreml, &wscratch);
    // The share as the bag reads: heads given, the feed-forward even.
    const FlexData bag = image_tp_cfg_(0, "[0, 4]", "none");
    std::size_t want = 0, want_floor = 0;
    if (std::string(f.name) == "flux3") {
      F3::Config c;
      ASSERT_TRUE(F3::Config::read_dims(f.dit, &c));
      ASSERT_TRUE(TpShare::from_flex(bag, c.n_heads, c.mlp_hidden, &c.tp,
                                     nullptr));
      want = F3::tp_held_bytes(f.dit, c, &want_floor);
    } else if (std::string(f.name) == "krea2") {
      K2::Config c;
      ASSERT_TRUE(TpShare::from_flex(bag, c.n_heads, c.ffn, &c.tp, nullptr));
      want = K2::tp_held_bytes(f.dit, c, sess.metal_compute(), &want_floor);
    } else if (std::string(f.name) == "moss-image") {
      MI::Config c;
      ASSERT_TRUE(MI::Config::read_dims(f.dit, &c));
      ASSERT_TRUE(TpShare::from_flex(bag, c.n_heads, c.ffn_inner, &c.tp,
                                     nullptr));
      want = MI::tp_held_bytes(f.dit, c, sess.metal_compute(), &want_floor);
    } else if (std::string(f.name) == "z-image") {
      ZI::Config c;
      ASSERT_TRUE(ZI::Config::read_dims(f.dit, &c));
      ASSERT_TRUE(TpShare::from_flex(bag, c.n_heads, c.ffn_inner, &c.tp,
                                     nullptr));
      want = ZI::tp_held_bytes(f.dit, c, sess.metal_compute(), &want_floor);
    } else if (std::string(f.name) == "fibo2") {
      FB::Config c;
      ASSERT_TRUE(FB::Config::read_dims(f.dit, &c));
      ASSERT_TRUE(TpShare::from_flex(bag, c.n_heads, c.ffn_inner, &c.tp,
                                     nullptr));
      want = FB::tp_held_bytes(f.dit, c, sess.metal_compute(), &want_floor);
    } else {
      QI::Config c;
      ASSERT_TRUE(QI::Config::read_dims(f.dit, &c));
      ASSERT_TRUE(TpShare::from_flex(bag, c.n_heads,
                                     c.hidden * c.mlp_ratio, &c.tp,
                                     nullptr));
      want = QI::tp_held_bytes(f.dit, c, sess.metal_compute(), &want_floor);
    }
    std::printf("[tensor_parallel] %s plan: rank held %zu floor %zu "
                "(want %zu / %zu), reported %zu, %d CoreML claim(s), "
                "activations %zu; whole model %zu on disk, %d CoreML "
                "claim(s), activations %zu\n", f.name, held, floor, want,
                want_floor, report, coreml, scratch,
                model_memory::dir_weights_bytes(f.dit), wcoreml, wscratch);
    EXPECT_TRUE(want > 0 && held == want && floor == want_floor);
    EXPECT_TRUE(report == want);
    EXPECT_TRUE(held < model_memory::dir_weights_bytes(f.dit));
    EXPECT_TRUE(coreml == 0 && wcoreml == 1);
    EXPECT_TRUE(scratch > 0 && scratch < wscratch);
  }
  std::error_code ec;
  fs::remove_all(froot, ec);
  fs::remove_all(qroot, ec);
  fs::remove_all(zroot, ec);
  fs::remove_all(broot, ec);
  fs::remove_all(mroot, ec);
}

// THE REAL CHECKPOINT SPLIT: MOSS-Image's own DiT (24 heads, 42 blocks,
// bf16), 12 heads and half the feed-forward rows a rank, the picture's
// first two steps on a 16 x 16 latent with random text -- the positive,
// and the negative without block 22 as the reference runs it -- whole
// blocks and banded (a floor of 128), against the unsplit model. The bar
// is the model's own reordering floor (the unsplit model through the
// steel tiles against the matrix-core GEMMs) four times over, at least
// 1e-2 -- a gross 1e-1 on a box with one route; the ranks identical, the
// exchanges counted, what a rank holds what the plan books for it.
// MEASURED on the M4 Pro (one route): 1.6-4.3e-2, held == booked
// (5,405,179,904 bytes). Env: VPIPE_MOSS_IMAGE_TEST_MODEL_PATH, the
// repo ROOT as every MOSS-Image test reads it (its transformer/; the
// folder itself is taken too). Heavy: the whole model, its floor and two
// ranks load one after another (~10 GB each at most).
TEST(tensor_parallel, a_split_real_moss_image_forward_is_the_unsplit_one)
{
  const char* path = std::getenv("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  if (path == nullptr || *path == '\0') {
    std::printf("  SKIP: VPIPE_MOSS_IMAGE_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  std::error_code dec;
  const fs::path dir = fs::is_directory(fs::path(path) / "transformer", dec)
                           ? fs::path(path) / "transformer"
                           : fs::path(path);
  MI::Config cfg;
  const bool dims = MI::Config::read_dims(dir.string(), &cfg);
  ASSERT_TRUE(dims);
  if (!dims) { return; }
  cfg.autotune = false;
  const int skip = 22;
  MossSteps whole, other;
  // ASSERT_TRUE records and goes on: a model that did not load stops the
  // test here rather than being read through a null pointer.
  for (int pass = 0; pass < 2; ++pass) {
    std::unique_ptr<EnvScope> e;
    if (pass == 1) {
      e = std::make_unique<EnvScope>("VPIPE_MOSS_IMAGE_NO_MMA2", "1");
    }
    auto m = MI::load(dir.string(), mc, cfg, false);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    std::string err;
    ASSERT_TRUE(moss_steps_(mc, *m, pass == 0 ? &whole : &other, &err,
                            skip));
  }
  double floor = 0.0;
  for (int k = 0; k < 2; ++k) {
    floor = std::max({floor, rel_l2_(other.v[k], whole.v[k]),
                      rel_l2_(other.n[k], whole.n[k])});
  }
  // A box with ONE route (no matrix cores: the floor reads 0) has no
  // second order to calibrate on, and 42 blocks of random text are
  // chaotic -- the split's own roundings move them 1.6-4.3e-2 there (M4
  // Pro). So a GROSS bar: a wrong cut reads O(1) (MUTATION, ffn_up_gate
  // cut gate-first: 0.91-1.12), and the arithmetic is the small model's
  // test to hold.
  const double bar = floor > 0.0 ? std::max(1e-2, 4.0 * floor) : 1e-1;
  std::printf("  moss real: reordering floor %.3e -> bar %.3e\n", floor,
              bar);
  const int h = cfg.n_heads / 2;
  const int img = kMossGrid * kMossGrid;
  for (const int band_floor : {0, 128}) {
    MossSplit x;
    ASSERT_TRUE(moss_split_(mc, dir, cfg, h, false, band_floor, &x, skip));
    for (int k = 0; k < 2; ++k) {
      const double dp = rel_l2_(x.s[0].v[k], whole.v[k]);
      const double dn = rel_l2_(x.s[0].n[k], whole.n[k]);
      std::printf("  moss real (floor %d) step %d: split vs unsplit "
                  "rel-L2 %.3e positive, %.3e negative, ranks %s\n",
                  band_floor, k, dp, dn,
                  same_bits_(x.s[0].v[k], x.s[1].v[k]) &&
                          same_bits_(x.s[0].n[k], x.s[1].n[k])
                      ? "IDENTICAL" : "DIFFER");
      EXPECT_TRUE(dp < bar && dn < bar);
      EXPECT_TRUE(same_bits_(x.s[0].v[k], x.s[1].v[k]) &&
                  same_bits_(x.s[0].n[k], x.s[1].n[k]));
    }
    const std::uint64_t want =
        2 * (2 * (std::uint64_t)cfg.n_layers *
                 tp_row_bands(img + kMossTxt, band_floor).size() +
             2 * (std::uint64_t)(cfg.n_layers - 1) *
                 tp_row_bands(img + kMossTxtNeg, band_floor).size());
    std::printf("  moss real (floor %d): %llu exchanges (want %llu); rank 0 "
                "holds %zu bytes, booked %zu (the checkpoint %zu)\n",
                band_floor, (unsigned long long)x.exchanges,
                (unsigned long long)want, x.held, x.booked,
                model_memory::dir_weights_bytes(dir.string()));
    EXPECT_TRUE(x.exchanges == want);
    EXPECT_TRUE(x.out_of_step == 0 && x.mismatched == 0);
    EXPECT_TRUE(x.held == x.booked && x.booked > 0);
  }
}

namespace {

// Rank `rank`'s channels in the stage pair transport: how many it opened
// (`n`), and whether every one is LET GO -- held by the transport alone
// -- with its stop cleared. Asked AT IDLE, the stage still holding its
// DiT: the DiT's destructor lets a channel go and clears its stop too, so
// the same check made after the launch stops cannot fail.
bool
rank_channels_let_go_(StagePairTransport* tr, int rank, int* n)
{
  std::lock_guard<std::mutex> lk(tr->mu);
  bool ok = true;
  *n = 0;
  for (const auto& ch : tr->channels) {
    if (ch->me != rank) { continue; }
    ++*n;
    ok = ok && ch.use_count() == 1 && !ch->stop_now.load();
  }
  return ok;
}

// A FAMILY'S SPLIT DiT THROUGH generate-image: two ranks, each a stage in
// a session of its own, over the stage pair transport, against the whole
// model through the same stage. Two pictures a launch (the seed kept),
// two steps each (no negative wired: no guidance):
//   - both ranks' latents the same bytes, each within 1e-2 of the whole
//     model's; a channel opened per picture per rank, each its own
//     exchange count from 0 (none out of step), `per_picture` a picture,
//     each given the stop; rank 0's two LET GO with their stop cleared AT
//     IDLE, the stage still holding its DiT -- as each picture ended
//     (DitPictureScope); the shares the bag named;
//   - what the plan counts for rank 0's DiT once its pictures are made is
//     `booked` -- its share, every block kept -- not the checkpoint; and
//     `held_beyond` more where the family's holding counts a prompt's
//     state beside its weights (Fibo 2.0's, as the whole model's does);
//   - the WARM KEY: a rank's DiT kept warm is not handed to the whole
//     model the same session launches next; a rank of the same share
//     takes its own.
// MUTATIONS, each failing it: the picture's scope taken out of the
// family's branch of process() (rank 0's channels still held at idle),
// the share left out of the warm key (the whole model takes the rank's
// DiT, its picture not the whole run's).
struct StageSplitCase {
  const char* name = "";
  fs::path root;                  // its transformer/ beneath
  const TensorBeat* cond = nullptr;
  const char* h0 = "";            // rank 0's heads, as the bag gives them
  const char* h1 = "";            // rank 1's
  int n_heads = 0, heads0 = 0;    // the model's heads, rank 0's count
  std::uint64_t per_picture = 0;  // a rank's exchanges a picture
  std::size_t booked = 0;         // rank 0's tp_held_bytes
  std::size_t held_beyond = 0;    // its holding's prompt state, if any
};

// True when every check held; each that did not printed with its line.
// (minitest's EXPECT_TRUE belongs to a TEST's body, so a helper reports.)
bool
split_runs_through_the_stage_(const StageSplitCase& c)
{
  bool ok = true;
#define VPIPE_TP_CHECK(cond)                                              \
  if (!(cond)) {                                                          \
    std::printf("  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
    ok = false;                                                           \
  }
  constexpr double kBar = 1e-2;
  StagePairTransport* tr = stage_pair_transport_();
  tr->reset();
  const char* tn = tr->name().data();
  ImageRun whole;
  {
    Session sess;
    flux3_image_run_(sess, c.root, 2, FlexData(), 0.0, &whole, {}, c.cond);
  }
  VPIPE_TP_CHECK(whole.latents.size() == 2);
  if (whole.latents.size() != 2) { return false; }

  ImageRun r0, r1;
  int idle_n = 0;
  bool idle_let_go = false;
  {
    Session s0, s1;
    std::thread t([&]() {
      flux3_image_run_(s1, c.root, 2, image_tp_cfg_(1, c.h1, tn), 0.0, &r1,
                       {}, c.cond);
    });
    flux3_image_run_(s0, c.root, 2, image_tp_cfg_(0, c.h0, tn), 0.0, &r0,
                     [&]() {
                       idle_let_go = rank_channels_let_go_(tr, 0, &idle_n);
                     },
                     c.cond);
    t.join();
  }
  VPIPE_TP_CHECK(r0.latents.size() == 2 && r1.latents.size() == 2);
  if (r0.latents.size() != 2 || r1.latents.size() != 2) { return false; }
  bool let_go = true, stopped = true, counts = true, kinds = true;
  int out_of_step = 0;
  {
    std::lock_guard<std::mutex> lk(tr->mu);
    for (const auto& ch : tr->channels) {
      let_go = let_go && ch.use_count() == 1;
      stopped = stopped && ch->stop_seen.load() && !ch->stop_now.load();
      counts = counts && ch->exchanges.load() == c.per_picture;
      out_of_step += ch->out_of_step.load();
    }
    for (const TpShare& sh : tr->shares) {
      kinds = kinds && sh.ranks == 2 && sh.n_heads == c.n_heads &&
              sh.heads() == (sh.rank == 0 ? c.heads0
                                          : c.n_heads - c.heads0) &&
              sh.ffn() == sh.n_ffn / 2;
    }
  }
  for (int k = 0; k < 2; ++k) {
    const double d = rel_l2_(latent_f32_(r0.latents[(std::size_t)k]),
                             latent_f32_(whole.latents[(std::size_t)k]));
    std::printf("[tensor_parallel] %s through the stage, picture %d: split "
                "vs whole rel-L2 %.3e, ranks %s\n", c.name, k, d,
                r0.latents[(std::size_t)k] == r1.latents[(std::size_t)k]
                    ? "IDENTICAL" : "DIFFER");
    VPIPE_TP_CHECK(r0.latents[(std::size_t)k] == r1.latents[(std::size_t)k]);
    VPIPE_TP_CHECK(d < kBar);
  }
  const std::string dit = (c.root / "transformer").string();
  std::printf("[tensor_parallel] %s: %zu channels (want 4), at idle rank "
              "0's %d %s; after the launches %s, stop %s, %s %llu "
              "exchanges a picture, %d out of step; rank 0's DiT planned "
              "at %zu bytes once its pictures were made, booked at %zu "
              "+ %zu of prompt state (the checkpoint %zu)\n", c.name,
              tr->channels.size(), idle_n,
              idle_let_go ? "let go" : "STILL HELD",
              let_go ? "every one let go" : "SOME KEPT",
              stopped ? "set and cleared" : "NOT CLEARED",
              counts ? "each" : "NOT each",
              (unsigned long long)c.per_picture, out_of_step, r0.planned,
              c.booked, c.held_beyond, model_memory::dir_weights_bytes(dit));
  VPIPE_TP_CHECK(tr->channels.size() == 4);
  VPIPE_TP_CHECK(idle_n == 2 && idle_let_go);
  VPIPE_TP_CHECK(let_go && stopped && counts && kinds && out_of_step == 0);
  VPIPE_TP_CHECK(c.booked > 0 &&
                 r0.planned == c.booked + c.held_beyond);

  // THE WARM KEY, in one session for rank 0 (its peer in another).
  tr->reset();
  {
    Session s0, s1;
    auto rank_launch = [&](ImageRun* a) {
      ImageRun b;
      std::thread t([&]() {
        flux3_image_run_(s1, c.root, 1, image_tp_cfg_(1, c.h1, tn), 60.0,
                         &b, {}, c.cond);
      });
      flux3_image_run_(s0, c.root, 1, image_tp_cfg_(0, c.h0, tn), 60.0, a,
                       {}, c.cond);
      t.join();
    };
    ImageRun a, w, x, y;
    rank_launch(&a);
    flux3_image_run_(s0, c.root, 1, FlexData(), 60.0, &w, {}, c.cond);
    rank_launch(&x);
    rank_launch(&y);
    std::printf("[tensor_parallel] %s warm key: rank (warm %d), whole (warm "
                "%d, %s the whole run's), rank (warm %d), rank (warm %d)\n",
                c.name, a.taken_warm, w.taken_warm,
                w.latents.size() == 1 && w.latents[0] == whole.latents[0]
                    ? "IDENTICAL to" : "NOT",
                x.taken_warm, y.taken_warm);
    VPIPE_TP_CHECK(!a.taken_warm && !w.taken_warm && !x.taken_warm);
    VPIPE_TP_CHECK(y.taken_warm);
    VPIPE_TP_CHECK(w.latents.size() == 1 && w.latents[0] == whole.latents[0]);
    VPIPE_TP_CHECK(x.latents.size() == 1 && y.latents.size() == 1 &&
                x.latents[0] == r0.latents[0] &&
                y.latents[0] == r0.latents[0]);
  }
  tr->reset();
#undef VPIPE_TP_CHECK
  return ok;
}

// A fresh directory with a small model's transformer/ written by `write`.
fs::path
stage_root_(const char* name, const std::function<bool(const fs::path&)>&
                                  write)
{
  const fs::path root = fs::temp_directory_path() / name;
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root, ec);
  return write(root / "transformer") ? root : fs::path();
}

}  // namespace

// MOSS-Image through generate-image (split_runs_through_the_stage_): 5
// of its 8 heads on rank 0; two exchanges a main block, its two text
// blocks whole.
TEST(tensor_parallel, a_split_moss_image_runs_through_the_stage)
{
  {
    Session probe;
    if (probe.metal_compute() == nullptr) { return; }
  }
  const fs::path root = stage_root_("vpipe_moss_tp_stage", [](auto& d) {
    return write_moss_small_(d, false);
  });
  ASSERT_TRUE(!root.empty());
  if (root.empty()) { return; }
  const std::string dit = (root / "transformer").string();
  MI::Config c0;
  ASSERT_TRUE(MI::Config::read_dims(dit, &c0));
  const TensorBeat cond = flux3_cond_(30, c0.text_dim);
  StageSplitCase sc;
  sc.name = "moss-image";
  sc.root = root;
  sc.cond = &cond;
  sc.h0 = "[0, 5]";
  sc.h1 = "[5, 8]";
  sc.n_heads = c0.n_heads;
  sc.heads0 = 5;
  // Two steps a picture, one forward a step, two exchanges a main block.
  sc.per_picture = 2 * 2 * (std::uint64_t)c0.n_layers;
  ASSERT_TRUE(TpShare::from_flex(
      image_tp_cfg_(0, sc.h0, stage_pair_transport_()->name().data()),
      c0.n_heads, c0.ffn_inner, &c0.tp, nullptr));
  {
    Session plan_sess;
    sc.booked = MI::tp_held_bytes(dit, c0, plan_sess.metal_compute());
  }
  EXPECT_TRUE(split_runs_through_the_stage_(sc));
  std::error_code ec;
  fs::remove_all(root, ec);
}

// Z-Image through generate-image: 8 of its 12 heads on rank 0; two
// exchanges a block, the caption refiner's, the noise refiner's and the
// main stack's alike.
TEST(tensor_parallel, a_split_z_image_runs_through_the_stage)
{
  {
    Session probe;
    if (probe.metal_compute() == nullptr) { return; }
  }
  const fs::path root = stage_root_("vpipe_zi_tp_stage", [](auto& d) {
    return write_zi_small_(d, "BF16", false);
  });
  ASSERT_TRUE(!root.empty());
  if (root.empty()) { return; }
  const std::string dit = (root / "transformer").string();
  ZI::Config c0;
  ASSERT_TRUE(ZI::Config::read_dims(dit, &c0));
  const TensorBeat cond = flux3_cond_(30, c0.cap_dim);
  StageSplitCase sc;
  sc.name = "z-image";
  sc.root = root;
  sc.cond = &cond;
  sc.h0 = "[0, 8]";
  sc.h1 = "[8, 12]";
  sc.n_heads = c0.n_heads;
  sc.heads0 = 8;
  // Two steps a picture, one forward a step: every block -- the caption
  // refiner's, then the noise refiner's and the main stack's -- two.
  sc.per_picture =
      2 * 2 * (std::uint64_t)(c0.n_refiner + c0.n_mod_blocks());
  ASSERT_TRUE(TpShare::from_flex(
      image_tp_cfg_(0, sc.h0, stage_pair_transport_()->name().data()),
      c0.n_heads, c0.ffn_inner, &c0.tp, nullptr));
  {
    Session plan_sess;
    sc.booked = ZI::tp_held_bytes(dit, c0, plan_sess.metal_compute());
  }
  EXPECT_TRUE(split_runs_through_the_stage_(sc));
  std::error_code ec;
  fs::remove_all(root, ec);
}

// Fibo 2.0 through generate-image: 8 of its 12 heads on rank 0; two
// exchanges a block and a third in the injection block, the
// conditioner's beat [bundles, rows, cap] (rows its second axis).
TEST(tensor_parallel, a_split_fibo2_runs_through_the_stage)
{
  {
    Session probe;
    if (probe.metal_compute() == nullptr) { return; }
  }
  const fs::path root = stage_root_("vpipe_fibo2_tp_stage", [](auto& d) {
    return write_fibo2_small_(d, false);
  });
  ASSERT_TRUE(!root.empty());
  if (root.empty()) { return; }
  const std::string dit = (root / "transformer").string();
  FB::Config c0;
  ASSERT_TRUE(FB::Config::read_dims(dit, &c0));
  TensorBeat cond = flux3_cond_(c0.n_bundles() * 30, c0.cap_dim);
  cond.shape = {c0.n_bundles(), 30, c0.cap_dim};
  StageSplitCase sc;
  sc.name = "fibo2";
  sc.root = root;
  sc.cond = &cond;
  sc.h0 = "[0, 8]";
  sc.h1 = "[8, 12]";
  sc.n_heads = c0.n_heads;
  sc.heads0 = 8;
  // Two steps a picture, one forward a step: two a block, and the
  // injection blocks' cross-attention a third.
  sc.per_picture =
      2 * (2 * (std::uint64_t)c0.n_mod_blocks() +
           (std::uint64_t)c0.injection_ids.size());
  ASSERT_TRUE(TpShare::from_flex(
      image_tp_cfg_(0, sc.h0, stage_pair_transport_()->name().data()),
      c0.n_heads, c0.ffn_inner, &c0.tp, nullptr));
  {
    Session plan_sess;
    sc.booked = FB::tp_held_bytes(dit, c0, plan_sess.metal_compute());
  }
  // Its holding counts the prompt's state beside the weights, as the
  // whole model's does (the safe side; the plan books the weights): the
  // gist rows [gist, hidden] and each injection block's keys and values
  // over the text, at rank 0's heads.
  const std::size_t gist = (std::size_t)fibo2::gist_count(
      30, c0.min_gist, c0.max_gist, c0.gist_step, c0.gist_min_text,
      c0.gist_max_text);
  sc.held_beyond = gist * (std::size_t)c0.hidden * 2 +
                   c0.injection_ids.size() * 2 * 30 *
                       (std::size_t)(sc.heads0 * c0.head_dim) * 2;
  EXPECT_TRUE(split_runs_through_the_stage_(sc));
  std::error_code ec;
  fs::remove_all(root, ec);
}
