// gpu-host-handshake.h -- the host's side of a GPU <-> host handshake INSIDE
// a running command buffer (gpu-kernels/metal/ops/host_handshake.metal,
// library `host_handshake`, where the rules it stands on are written).
//
// A model's step hands the host something it just computed and needs the
// host's answer to go on -- a MoE layer's expert picks, answered with the
// expert cache's slots. A commit and wait at that point costs ~0.3 ms a
// time; the handshake costs a few microseconds. The GPU copies its words
// to a mailbox, posts the turn's sequence number, and one thread waits for
// the host's answer; the host answers each turn as it is posted, encoding
// a few layers ahead of its last answer. The caller owns the protocol's
// discipline (DeepSeek-V4.1 and the Qwen family's expert cache use it):
//   * encode at most a few turns past the last one answered -- a queue
//     holds 64 command buffers, and a host blocked on the 65th while the
//     GPU waits for it is a deadlock that outlives the process;
//   * keep the heartbeat beating while any turn may be waited on (a
//     waiting GPU gives up once it stops: a host that is gone costs one
//     wait, not one a layer);
//   * check the checksums both ways, and REDO a forward the GPU gave up
//     on (word kGaveUp): its later layers ran on a stale answer. The redo
//     must find the state the forward found -- a forward that rewrites a
//     recurrent state in place cannot take the handshake.
// Host-only: nothing here is installed (a plugin has no expert cache).
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>

namespace vpipe::metal_compute {
class MetalCompute;
}  // namespace vpipe::metal_compute

namespace vpipe::genai::host_handshake {

// The system-scope library and its turn (deepseek_v41.metal keeps a
// device-scope twin of the turn, dsv41_handshake_io, with the same
// buffers).
inline constexpr const char* kLibrary = "host_handshake";
inline constexpr const char* kIo = "host_handshake_io";

// The control words, one 64-byte buffer (one cache line): what the GPU
// posted, what the host answered, the GPU's give-ups, the host's
// heartbeat, and the last turn the GPU took its answer for (the
// system-scope turn's alone).
enum Word : int { kPosted = 0, kAnswered = 1, kGaveUp = 2, kBeat = 3,
                  kAcked = 4 };
inline constexpr std::size_t kControlBytes = 64;
// A turn's wait gives up once the heartbeat has not moved in this long.
inline constexpr double kGiveUpSeconds = 0.25;

// Whether a running command buffer and this host see each other's stores
// through library `lib`'s turn `io`: rounds of a word carried out and one
// carried in, each a dispatch of its own answered as it is posted, then
// the loads a wait nobody answers makes before it gives up, timed -- the
// loads in kGiveUpSeconds (the `spins` a turn waits), 0 where they do
// not see each other (or the library did not build). Probed once a
// process for each library.
std::uint32_t probe(metal_compute::MetalCompute* mc,
                    const char* lib = kLibrary, const char* io = kIo);

// A word the GPU writes while this host reads it, as it is now.
std::uint32_t load_now(const std::uint32_t& x);

// A mailbox's checksum: its `n` words for turn `seq` (the GPU's is
// host_handshake_mix_; it sits in the word after them).
std::uint32_t sum(const std::uint32_t* w, std::uint32_t n,
                  std::uint32_t seq);

// A thread that moves a control buffer's kBeat word every 50 us while on.
class Heartbeat {
public:
  explicit Heartbeat(std::uint32_t* words) : _w(words) {}
  ~Heartbeat();
  Heartbeat(const Heartbeat&) = delete;
  Heartbeat& operator=(const Heartbeat&) = delete;
  void set(bool on);

private:
  std::uint32_t* _w;
  std::thread _t;
  std::mutex _mu;
  std::condition_variable _cv;
  bool _on = false, _quit = false;
};

}  // namespace vpipe::genai::host_handshake
