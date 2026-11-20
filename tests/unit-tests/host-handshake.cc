// host_handshake: a GPU <-> host handshake INSIDE a running command buffer
// through the system memory scope (generative-models/gpu-host-handshake.h,
// gpu-kernels/metal/ops/host_handshake.metal). Model-free: the protocol a
// MoE model's expert-cache turn stands on (DeepSeek-V4.1, the Qwen family).
#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/gpu-host-handshake.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::ComputeLibrary;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return (v != nullptr && *v != '\0') ? std::string(v) : std::string();
}

}  // namespace

// Every Apple GPU measured sees the host through the system scope (M4 Pro,
// base M5, M5 Ultra), so every MoE model with an expert cache hands its
// layers' picks over inside the running GPU work. This fails where a box
// does not: those models then fall back to a commit and wait a layer
// (~20% of a cached decode), which nothing else would say -- the Ultra's
// device-scope handshake went that way with an OS update, unnoticed for
// days. VPIPE_HOST_HANDSHAKE_EXPECT=0: not checked (a box known not to).
TEST(host_handshake, this_gpu_takes_the_system_scope)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  const bool built = mc->load_library(host_handshake::kLibrary)
                         .function(host_handshake::kIo)
                         .valid();
  const std::uint32_t spins = host_handshake::probe(mc);
  std::printf("[host_handshake] the system-scope library %s; the probe %s "
              "(%u loads a give-up)\n", built ? "built" : "DID NOT BUILD",
              spins > 0 ? "passed" : "FAILED", spins);
  if (env_("VPIPE_HOST_HANDSHAKE_EXPECT") == "0") { return; }
  ASSERT_TRUE(built);
  ASSERT_TRUE(spins > 0);
}

// The protocol's data both ways, every round: the GPU's words reach the
// host through the mailbox (written by a dispatch before the turn), and
// the host's answer reaches the dispatches after it. Plain stores both
// ways are counted too, for the record; nothing relies on them (on the M4
// Pro and the base M5 neither is ever seen). The rounds go a buffer of 16
// at a time, each answered before the next is encoded: a host that encodes
// far ahead of its answers can block on the queue's buffer limit while the
// GPU waits for it. VPIPE_HOST_HANDSHAKE_ROUNDS (2000).
TEST(host_handshake, a_mid_buffer_turn_carries_data)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  if (host_handshake::probe(mc) == 0) {
    std::printf("[host_handshake] this GPU does not see the host "
                "mid-buffer; not checked here\n");
    return;
  }
  ComputeLibrary lib = mc->load_library(host_handshake::kLibrary);
  ComputeFunction fw = lib.function("host_handshake_study_write");
  ComputeFunction fh = lib.function(host_handshake::kIo);
  ComputeFunction fc = lib.function("host_handshake_study_check");
  ASSERT_TRUE(fw.valid() && fh.valid() && fc.valid());
  if (!fw.valid() || !fh.valid() || !fc.valid()) { return; }
  const std::string rv = env_("VPIPE_HOST_HANDSHAKE_ROUNDS");
  const unsigned rounds = rv.empty() ? 2000u : (unsigned)std::stoi(rv);
  const unsigned NO = 3168, NI = 600, chunk = 16;
  auto words = [&](unsigned n) {
    SharedBuffer b = mc->make_shared_buffer((std::size_t)n * 4);
    std::memset(b.contents(), 0, (std::size_t)n * 4);
    return b;
  };
  SharedBuffer f = words(16), src = words(NO), bo = words(NO + 1);
  SharedBuffer bi = words(NI + 1), dst = words(NI), raw = words(NI);
  auto* fw32 = static_cast<std::uint32_t*>(f.contents());
  auto* box = static_cast<const std::uint32_t*>(bo.contents());
  auto* plain = static_cast<const std::uint32_t*>(src.contents());
  auto* in = static_cast<std::uint32_t*>(bi.contents());
  auto* rw = static_cast<std::uint32_t*>(raw.contents());
  auto word = [&](int k) { return std::atomic_ref<std::uint32_t>(fw32[k]); };
  unsigned long bad_box = 0, bad_plain = 0, bad_sums = 0;
  bool lost = false;
  CommandStream st = mc->make_command_stream();
  std::string why;
  const std::uint32_t spins = 1u << 24;
  const auto t0 = std::chrono::steady_clock::now();
  for (unsigned r0 = 1; r0 <= rounds && !lost; r0 += chunk) {
    const unsigned r1 = std::min(rounds, r0 + chunk - 1);
    ComputeEncoder enc = st.begin_compute();
    for (unsigned r = r0; r <= r1; ++r) {
      enc.set_function(fw);
      enc.set_buffer(0, src);
      enc.set_constant(1, r);
      enc.set_constant(2, NO);
      enc.dispatch({NO, 1, 1}, {256, 1, 1});
      enc.set_function(fh);
      enc.set_buffer(0, f);
      enc.set_buffer(1, src);
      enc.set_buffer(2, src);
      enc.set_buffer(3, src);
      enc.set_buffer(4, bo);
      enc.set_buffer(5, bi);
      enc.set_buffer(6, dst);
      enc.set_buffer(7, dst);
      enc.set_buffer(8, dst);
      const std::uint32_t nv[4] = {NO, 0, 0, r};
      const std::uint32_t mv[4] = {NI, 0, 0, spins};
      enc.set_constant_bytes(9, nv, sizeof nv);
      enc.set_constant_bytes(10, mv, sizeof mv);
      enc.dispatch({256, 1, 1}, {256, 1, 1});
      enc.set_function(fc);
      enc.set_buffer(0, dst);
      enc.set_buffer(1, raw);
      enc.set_buffer(2, f);
      enc.set_constant(3, r);
      enc.set_constant(4, NI);
      enc.dispatch({NI, 1, 1}, {256, 1, 1});
    }
    enc.end();
    auto fence = st.commit();
    for (unsigned r = r0; r <= r1 && !lost; ++r) {
      const auto ts = std::chrono::steady_clock::now();
      while (word(host_handshake::kPosted).load(std::memory_order_acquire) <
             r) {
        word(host_handshake::kBeat).fetch_add(1, std::memory_order_relaxed);
        if (word(host_handshake::kGaveUp).load(std::memory_order_relaxed) !=
                0 ||
            std::chrono::steady_clock::now() - ts > std::chrono::seconds(2)) {
          lost = true;
          break;
        }
      }
      if (lost) { break; }
      // the mailbox once its checksum holds (as the models read it)
      const auto tc = std::chrono::steady_clock::now();
      while (host_handshake::sum(box, NO, r) !=
             host_handshake::load_now(box[NO])) {
        if (std::chrono::steady_clock::now() - tc > std::chrono::seconds(1)) {
          ++bad_sums;
          break;
        }
      }
      for (unsigned k = 0; k < NO; ++k) {
        bad_box += box[k] != r * 4096u + k;
        bad_plain += plain[k] != r * 4096u + k;
      }
      for (unsigned k = 0; k < NI; ++k) {
        in[k] = r * 7u + k;
        rw[k] = r * 7u + k;
      }
      in[NI] = host_handshake::sum(in, NI, r);
      word(host_handshake::kAnswered).store(r, std::memory_order_release);
    }
    if (lost) {   // let the buffer finish: every later wait passes
      word(host_handshake::kAnswered).store(~0u, std::memory_order_release);
    }
    ASSERT_TRUE(fence.wait_ok(&why));
    // the GPU's word that it took each answer
    ASSERT_TRUE(lost || word(host_handshake::kAcked).load() == r1);
  }
  const double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0).count();
  std::printf("[host_handshake] %u rounds, %.1f us a round; mailbox %lu / "
              "%u bad out (%lu sums), %u bad in; plain %lu bad out, %u bad "
              "in; %u gave up%s\n", rounds, us / rounds, bad_box,
              rounds * NO, bad_sums, fw32[5], bad_plain, fw32[6], fw32[2],
              lost ? " (LOST)" : "");
  ASSERT_TRUE(!lost);
  ASSERT_TRUE(fw32[2] == 0);
  ASSERT_TRUE(bad_box == 0 && bad_sums == 0);
  ASSERT_TRUE(fw32[5] == 0);
}

// A turn nobody answers: the GPU gives up once the heartbeat stops, counts
// it, acknowledges the turn, and the waits after it in the same forward do
// not wait (a host that is gone costs one wait, not one a layer). The host
// sees all of it once the buffer is done -- what a model's redo stands on.
TEST(host_handshake, an_unanswered_turn_gives_up_once)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  const std::uint32_t spins = host_handshake::probe(mc);
  if (spins == 0) {
    std::printf("[host_handshake] no handshake on this GPU\n");
    return;
  }
  ComputeFunction fh =
      mc->load_library(host_handshake::kLibrary).function(host_handshake::kIo);
  ASSERT_TRUE(fh.valid());
  SharedBuffer f = mc->make_shared_buffer(host_handshake::kControlBytes);
  SharedBuffer v = mc->make_shared_buffer(64), bo = mc->make_shared_buffer(64);
  SharedBuffer bi = mc->make_shared_buffer(64);
  std::memset(f.contents(), 0, host_handshake::kControlBytes);
  std::memset(v.contents(), 0, 64);
  auto* fw32 = static_cast<std::uint32_t*>(f.contents());
  CommandStream st = mc->make_command_stream();
  ComputeEncoder enc = st.begin_compute();
  for (std::uint32_t r = 1; r <= 3; ++r) {
    enc.set_function(fh);
    enc.set_buffer(0, f);
    for (unsigned b = 1; b <= 3; ++b) { enc.set_buffer(b, v); }
    enc.set_buffer(4, bo);
    enc.set_buffer(5, bi);
    for (unsigned b = 6; b <= 8; ++b) { enc.set_buffer(b, v, 32); }
    const std::uint32_t nv[4] = {1, 0, 0, r};
    const std::uint32_t mv[4] = {1, 0, 0, spins};
    enc.set_constant_bytes(9, nv, sizeof nv);
    enc.set_constant_bytes(10, mv, sizeof mv);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
  }
  enc.end();
  const auto t0 = std::chrono::steady_clock::now();
  std::string why;
  ASSERT_TRUE(st.commit().wait_ok(&why));
  const double s = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
  std::printf("[host_handshake] 3 turns unanswered: %.2f s, gave up %u, "
              "acknowledged %u\n", s, fw32[host_handshake::kGaveUp],
              fw32[host_handshake::kAcked]);
  EXPECT_TRUE(fw32[host_handshake::kGaveUp] == 1);
  EXPECT_TRUE(fw32[host_handshake::kAcked] == 3);
  EXPECT_TRUE(fw32[host_handshake::kPosted] == 3);
  // one give-up's wait (~kGiveUpSeconds), not three
  EXPECT_TRUE(s < 2.5 * host_handshake::kGiveUpSeconds + 0.5);
}

// What a running dispatch sees of the host's stores through the SYSTEM
// scope (VPIPE_HOST_HANDSHAKE_STUDY=1; the study kernel in
// host_handshake.metal): the answer on the post's line or its own, and 32
// words of data a round on a line of their own -- read first after the
// answer, read by an earlier command buffer (warm 1) or an earlier
// dispatch of this one (warm 2), by this thread before its post (touch 4),
// or in another buffer (touch 5), the host's heartbeat beside the answer
// or not. What the handshake's layout rests on (see the .metal); deepseek_
// v41.handshake_visibility_probe is the device scope's.
TEST(host_handshake, system_scope_study)
{
  if (env_("VPIPE_HOST_HANDSHAKE_STUDY").empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  auto lib = mc->load_library(host_handshake::kLibrary);
  ComputeFunction fp = lib.function("host_handshake_study");
  if (!fp.valid()) {
    std::printf("[host_handshake] the library did not build\n");
    return;
  }
  const std::uint32_t rounds = 200;
  const std::size_t fbytes = (1024 + 32 * (rounds + 2)) * 4;
  SharedBuffer f = mc->make_shared_buffer(fbytes);
  auto* w = static_cast<std::uint32_t*>(f.contents());
  auto word = [w](int k) { return std::atomic_ref<std::uint32_t>(w[k]); };
  CommandStream st = mc->make_command_stream();
  ComputeFunction fwarm = lib.function("host_handshake_warm_read");
  SharedBuffer sink = mc->make_shared_buffer(64);
  // warm 0: the data lines first read in the handshake; 1: read in an
  // earlier command buffer; 2: read earlier in the same buffer. The waits
  // and the post always fenced (the handshake's way).
  // beat 1: the host adds to word 3 (the post's line) as it waits for
  // each post; 2: a thread of its own adds to it every 50 us all along;
  // 3: as 1 by plain load + store. touch 5: the data in another buffer
  // (sep 1) or this one bound again (sep 0).
  SharedBuffer other = mc->make_shared_buffer(fbytes);
  struct Case { std::uint32_t warm, touch, at, beat, sep; };
  const Case cases[] = {
      {0, 0, 1, 0, 0}, {0, 0, 64, 0, 0}, {0, 3, 1, 0, 0}, {1, 3, 1, 0, 0},
      {2, 3, 1, 0, 0}, {0, 4, 1, 0, 0},  {0, 3, 1, 1, 0}, {0, 3, 1, 2, 0},
      {0, 3, 1, 3, 0}, {0, 5, 1, 0, 0},  {0, 5, 1, 0, 1}, {0, 5, 1, 2, 1}};
  for (const Case& cs : cases) {
    std::memset(other.contents(), 0, fbytes);
    auto* dw = static_cast<std::uint32_t*>(
        cs.touch == 5 && cs.sep != 0 ? other.contents() : f.contents());
    std::atomic<bool> beating{cs.beat == 2};
    std::thread beater([&] {
      while (beating.load()) {
        word(3).fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    });
    std::memset(f.contents(), 0, fbytes);
    auto warm_up = [&](ComputeEncoder& e) {
      e.set_function(fwarm);
      e.set_buffer(0, f);
      e.set_buffer(1, sink);
      e.set_constant(2, 1024u);
      e.set_constant(3, 32u * (rounds + 2));
      e.dispatch({1, 1, 1}, {32, 1, 1});
    };
    if (cs.warm == 1) {
      ComputeEncoder e0 = st.begin_compute();
      warm_up(e0);
      e0.end();
      std::string why0;
      ASSERT_TRUE(st.commit().wait_ok(&why0));
    }
    ComputeEncoder enc = st.begin_compute();
    if (cs.warm == 2) { warm_up(enc); }
    enc.set_function(fp);
    enc.set_buffer(0, f);
    enc.set_buffer(1, f);
    enc.set_constant(2, rounds);
    enc.set_constant(3, 1u << 22);
    enc.set_constant(4, cs.at);
    enc.set_constant(5, 1u);
    enc.set_constant(6, 1u);
    enc.set_constant(7, cs.touch);
    enc.set_buffer(8, cs.sep != 0 ? other : f);
    enc.dispatch({1, 1, 1}, {32, 1, 1});
    enc.end();
    auto fc = st.commit();
    std::uint32_t seen = 0;
    double worst = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint32_t r = 1; r <= rounds; ++r) {
      const auto ts = std::chrono::steady_clock::now();
      bool ok = true;
      while (word(0).load(std::memory_order_acquire) < r) {
        if (cs.beat == 1) {
          word(3).fetch_add(1, std::memory_order_relaxed);
        } else if (cs.beat == 3) {
          w[3] = w[3] + 1;
        }
        if (word(2).load(std::memory_order_relaxed) != 0 ||
            std::chrono::steady_clock::now() - ts >
                std::chrono::seconds(2)) {
          ok = false;
          break;
        }
      }
      if (!ok) { break; }
      worst = std::max(worst, std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - ts)
                                  .count());
      seen = r;
      if (cs.touch >= 3) {
        for (std::uint32_t k = 0; k < 32; ++k) {
          dw[1024 + 32 * r + k] = r * 7u + k;
        }
      }
      word((int)cs.at).store(r, std::memory_order_release);
    }
    word((int)cs.at).store(~0u, std::memory_order_release);
    std::string why;
    ASSERT_TRUE(fc.wait_ok(&why));
    beating = false;
    beater.join();
    const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - t0)
                          .count() / rounds;
    std::printf("[host_handshake] study: warm %u, touch %u, beat %u, sep %u, "
                "answer %s: %u of %u rounds seen, %u gave up, %u data words "
                "wrong, %.2f us a round (worst %.1f)\n", cs.warm, cs.touch,
                cs.beat, cs.sep,
                cs.at == 1 ? "on the post's line" : "on its own line", seen,
                rounds, w[2], w[6], us, worst);
  }
}
