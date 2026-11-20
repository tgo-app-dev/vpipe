#include "generative-models/gpu-host-handshake.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <string>

namespace vpipe::genai::host_handshake {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::ComputeLibrary;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

std::uint32_t
load_now(const std::uint32_t& x)
{
  return std::atomic_ref<std::uint32_t>(const_cast<std::uint32_t&>(x))
      .load(std::memory_order_acquire);
}

std::uint32_t
sum(const std::uint32_t* w, std::uint32_t n, std::uint32_t seq)
{
  std::uint32_t s = 0;
  for (std::uint32_t k = 0; k < n; ++k) {
    const std::uint32_t v = load_now(w[k]);
    s += v * (2u * k + 1u) + (v >> 16) * 0x9E3779B9u;
  }
  return s + seq * 2654435761u;
}

namespace {

std::uint32_t
probe_(MetalCompute* mc, const char* lib, const char* io)
{
  ComputeFunction hs = mc->load_library(lib).function(io);
  SharedBuffer f = mc->make_shared_buffer(kControlBytes);
  SharedBuffer v = mc->make_shared_buffer(64), d = mc->make_shared_buffer(64);
  SharedBuffer bo = mc->make_shared_buffer(64), bi = mc->make_shared_buffer(64);
  if (!hs.valid() || f.empty() || v.empty() || d.empty() || bo.empty() ||
      bi.empty()) {
    return 0;
  }
  constexpr std::uint32_t kRounds = 8, kCal = 1u << 20;
  auto* hw = static_cast<std::uint32_t*>(f.contents());
  auto word = [hw](int k) { return std::atomic_ref<std::uint32_t>(hw[k]); };
  auto* vw = static_cast<std::uint32_t*>(v.contents());
  std::memset(f.contents(), 0, kControlBytes);
  std::memset(d.contents(), 0, 64);
  for (std::uint32_t r = 0; r <= kRounds; ++r) { vw[r] = 1000 + r; }
  // round `seq` carries v[seq] out and the host's word into d[4]
  auto turn = [&](ComputeEncoder& enc, std::uint32_t seq, std::uint32_t nw,
                  std::uint32_t spins) {
    enc.set_function(hs);
    enc.set_buffer(0, f);
    enc.set_buffer(1, v, (std::size_t)seq * 4);
    enc.set_buffer(2, v);
    enc.set_buffer(3, v);
    enc.set_buffer(4, bo);
    enc.set_buffer(5, bi);
    for (unsigned b = 6; b <= 8; ++b) { enc.set_buffer(b, d, 16); }
    const std::uint32_t nv[4] = {nw, 0, 0, seq};
    const std::uint32_t mv[4] = {nw, 0, 0, spins};
    enc.set_constant_bytes(9, nv, sizeof nv);
    enc.set_constant_bytes(10, mv, sizeof mv);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
  };
  CommandStream st = mc->make_command_stream();
  ComputeEncoder enc = st.begin_compute();
  for (std::uint32_t r = 1; r <= kRounds; ++r) { turn(enc, r, 1, kCal); }
  enc.end();
  auto fence = st.commit();
  bool ok = true;
  const auto* out = static_cast<const std::uint32_t*>(bo.contents());
  auto* in = static_cast<std::uint32_t*>(bi.contents());
  for (std::uint32_t r = 1; r <= kRounds && ok; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    while (word(kPosted).load(std::memory_order_acquire) < r) {
      word(kBeat).fetch_add(1, std::memory_order_relaxed);
      if (word(kGaveUp).load(std::memory_order_relaxed) != 0 ||
          std::chrono::steady_clock::now() - t0 > std::chrono::seconds(1)) {
        ok = false;
        break;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    while (ok && sum(out, 1, r) != load_now(out[1])) {
      ok = std::chrono::steady_clock::now() - t1 < std::chrono::seconds(1);
    }
    ok = ok && out[0] == 1000 + r;
    in[0] = 2000 + r;
    in[1] = sum(in, 1, r);
    // a failed round lets every later wait pass, so the buffer ends
    word(kAnswered).store(ok ? r : ~0u, std::memory_order_release);
  }
  std::string why;
  ok = fence.wait_ok(&why) && ok && word(kGaveUp).load() == 0 &&
       static_cast<const std::uint32_t*>(d.contents())[4] == 2000 + kRounds;
  if (!ok) { return 0; }
  // the loads one unanswered wait makes, timed
  std::memset(f.contents(), 0, kControlBytes);
  ComputeEncoder e2 = st.begin_compute();
  turn(e2, 1, 0, kCal);
  e2.end();
  auto f2 = st.commit();
  if (!f2.wait_ok(&why) || word(kGaveUp).load() != 1) { return 0; }
  const double t = f2.gpu_times().gpu_s;
  return (std::uint32_t)std::clamp(
      t > 0.0 ? kCal / t * kGiveUpSeconds : (double)kCal, (double)kCal,
      (double)(1u << 30));
}

}  // namespace

std::uint32_t
probe(MetalCompute* mc, const char* lib, const char* io)
{
  static std::mutex mu;
  static std::map<std::string, std::uint32_t> known;
  if (mc == nullptr || lib == nullptr || io == nullptr) { return 0; }
  std::lock_guard<std::mutex> lk(mu);
  const std::string key = std::string(lib) + "/" + io;
  if (auto it = known.find(key); it != known.end()) { return it->second; }
  const std::uint32_t s = probe_(mc, lib, io);
  known.emplace(key, s);
  return s;
}

Heartbeat::~Heartbeat()
{
  if (_t.joinable()) {
    {
      std::lock_guard<std::mutex> lk(_mu);
      _quit = true;
    }
    _cv.notify_all();
    _t.join();
  }
}

void
Heartbeat::set(bool on)
{
  if (on && !_t.joinable()) {
    _t = std::thread([this] {
      std::unique_lock<std::mutex> lk(_mu);
      while (!_quit) {
        if (!_on) {
          _cv.wait(lk);
          continue;
        }
        lk.unlock();
        std::atomic_ref<std::uint32_t>(_w[kBeat]).fetch_add(
            1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        lk.lock();
      }
    });
  }
  {
    std::lock_guard<std::mutex> lk(_mu);
    _on = on;
  }
  _cv.notify_all();
}

}  // namespace vpipe::genai::host_handshake
