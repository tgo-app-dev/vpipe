// host_handshake.metal -- a GPU <-> host handshake INSIDE a running command
// buffer, through the SYSTEM memory scope (generative-models/
// gpu-host-handshake.h has the host's side).
//
// A model's step hands the host something it just computed (a MoE layer's
// expert picks) and needs the host's answer (the cache slots holding
// them) to go on. A commit and wait a layer costs ~0.3 ms; this costs a
// few microseconds: the GPU posts, one thread waits for the host's answer
// and reads it, and the command buffers after it run on. It needs each
// side to see the other's stores while a buffer runs. Through the device
// scope only an M5 Ultra under macOS 27.0 did -- 27.2 took that away, and
// an M4 Pro or a base M5 never did. Through the system scope all three
// do, by these rules (MEASURED on an M4 Pro and a base M5 under 26.6, an
// M5 Ultra under 27.2.1; host_handshake.system_scope_study, 200 rounds a
// case, and host_handshake.a_mid_buffer_turn_carries_data, 2000 rounds of
// 3168 words out and 600 in):
//   * GPU -> host: words reach the host only when STORED through the
//     system scope (a volatile coherent(system) pointer). Device-scope
//     atomics followed by a system fence stayed in the GPU's caches on the
//     M4 Pro: the host saw not one of 3168 words a round in 64 rounds (the
//     post alone got out -- its line the GPU then read through the system
//     scope), nor any plain store an earlier dispatch made;
//   * host -> GPU: the host's answer word is seen through the system scope
//     on the 128-byte line the GPU posted to -- on an M4 Pro and a base M5
//     ONLY there (a word on a line of its own: 1 round of 200), on the
//     Ultra anywhere -- 2-8 us a round, the host's heartbeat stores on that
//     line or not;
//   * data on other lines (another buffer too) is seen once that answer
//     is, after a system fence -- lines an earlier dispatch or command
//     buffer read included, and lines this thread read before its post on
//     the M4 Pro and the base M5. (On the Ultra that last case -- the data
//     read, then its sum added to a word on the post's line by a
//     device-scope atomic, then the post -- lost the answer: the handshake
//     reads nothing of the mailbox before the answer and makes no
//     read-modify-write on that line before it posts.)
// So every word either side writes goes through the system scope, the
// control words sit on one line (`f`, 64 bytes), the mailbox `bi` is read
// only after the answer, and the checksums stay: a stale or partial
// mailbox is a give-up (and the caller redoes its forward with commits),
// never a wrong answer.
//
// The words of `f`: 0 posted (the turn's seq), 1 answered (the host's), 2
// gave up (a count), 3 the host's heartbeat, 4 acknowledged (the last turn
// whose answer this dispatch took, or gave up on).
//
// The system scope (thread_scope 3) is not in Metal's public headers; it
// takes a private pragma, which is why this file is a library of its own:
// a compiler that refuses it leaves this library invalid, and its users on
// a commit and wait a layer (the probe decides, host_handshake::probe).
//
// No element type: one library serves the f16 and bf16 models.

#pragma METAL internals : enable
#include <metal_stdlib>
using namespace metal;

#include "ops/deepseek_v41_route.h"

constexpr constant thread_scope kHostScope = static_cast<thread_scope>(3);

inline void host_handshake_fence_()
{
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      kHostScope);
}

// The mailbox checksum's term for word `k` holding `v` (the host's is
// host_handshake::sum, and deepseek_v41.metal's device-scope twin's
// dsv41_hs_mix_ is the same).
inline uint host_handshake_mix_(uint k, uint v)
{
  return v * (2u * k + 1u) + (v >> 16) * 0x9E3779B9u;
}

// The host's turn inside a running buffer, with data both ways: the words
// of s0..s2 (n0..n2 each: what the GPU wrote before) go to `bo`, then
// their checksum, a system fence, then word 0 of `f` is set to `seq`;
// thread 0 waits for word 1 to reach `seq`, giving up when word 3 (the
// host's heartbeat) has not moved in `spins` loads (counted in word 2;
// once one has given up, later ones do not wait: a host that is gone
// costs one wait, not one a layer). Then -- and only then -- `bi` (the
// host's answer, m0..m2 words and its checksum) is read into d0..d2 until
// its checksum holds, and word 4 set to `seq`. grid {256}, tg {256}.
inline void host_handshake_body_(
    volatile coherent(system) device uint* f, const device uint* s0,
    const device uint* s1, const device uint* s2,
    volatile coherent(system) device uint* bo,
    volatile coherent(system) device const uint* bi, device uint* d0,
    device uint* d1, device uint* d2, const uint4 n, const uint4 m,
    const uint t, const uint nt, const uint sg, const uint lane,
    threadgroup uint* part, threadgroup uint& verdict, threadgroup uint& gave)
{
  const uint no = n.x + n.y + n.z, ni = m.x + m.y + m.z, seq = n.w;
  // the group's sum of what each thread holds
  auto total = [&](uint mine) {
    const uint s = simd_sum(mine);
    if (lane == 0) { part[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint all = 0;
    for (uint g = 0; g < (nt + 31) / 32; ++g) { all += part[g]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return all;
  };
  uint mine = 0;
  for (uint k = t; k < no; k += nt) {
    const uint v = k < n.x ? s0[k] : k < n.x + n.y ? s1[k - n.x]
                                                    : s2[k - n.x - n.y];
    bo[k] = v;
    mine += host_handshake_mix_(k, v);
  }
  const uint so = total(mine) + seq * 2654435761u;
  if (t == 0) { bo[no] = so; }
  host_handshake_fence_();
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) {
    f[0] = seq;
    host_handshake_fence_();
    uint k = 0;
    uint hb = f[3];
    bool mine_up = false;
    for (;;) {
      host_handshake_fence_();
      if (f[2] != 0 || f[1] >= seq) { break; }
      if (++k < m.w) { continue; }
      const uint h = f[3];
      if (h == hb) {
        f[2] = f[2] + 1u;
        mine_up = true;
        break;
      }
      hb = h;
      k = 0;
    }
    host_handshake_fence_();
    gave = mine_up || f[2] != 0 ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_device);
  // the answer, read until its checksum holds (or the wait gave up)
  for (uint tries = 0;; ++tries) {
    host_handshake_fence_();
    uint got = 0;
    for (uint k = t; k < ni; k += nt) {
      const uint v = bi[k];
      got += host_handshake_mix_(k, v);
      if (k < m.x) {
        d0[k] = v;
      } else if (k < m.x + m.y) {
        d1[k - m.x] = v;
      } else {
        d2[k - m.x - m.y] = v;
      }
    }
    const uint sum = total(got) + seq * 2654435761u;
    if (t == 0) {
      const bool ok = bi[ni] == sum;
      verdict = ok || gave != 0 ? 1u : tries + 1 >= 4096u ? 2u : 0u;
      if (verdict == 2u) { f[2] = f[2] + 1u; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (verdict != 0u) { break; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0) {
    host_handshake_fence_();
    f[4] = seq;
    host_handshake_fence_();
  }
}

// grid {256}, tg {256}.
kernel void host_handshake_io(
    volatile coherent(system) device uint* f [[buffer(0)]],
    const device uint*  s0    [[buffer(1)]],
    const device uint*  s1    [[buffer(2)]],
    const device uint*  s2    [[buffer(3)]],
    volatile coherent(system) device uint* bo [[buffer(4)]],
    volatile coherent(system) device const uint* bi [[buffer(5)]],
    device uint*        d0    [[buffer(6)]],
    device uint*        d1    [[buffer(7)]],
    device uint*        d2    [[buffer(8)]],
    constant uint4&     n     [[buffer(9)]],    // n0, n1, n2, seq
    constant uint4&     m     [[buffer(10)]],   // m0, m1, m2, spins
    uint t    [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  threadgroup uint part[32];
  threadgroup uint verdict, gave;
  host_handshake_body_(f, s0, s1, s2, bo, bi, d0, d1, d2, n, m, t, nt, sg,
                       lane, part, verdict, gave);
}

// DeepSeek-V4.1's decode step: its route (dsv41_route_tg_, by the whole
// group) for its one row, then host_handshake_io with the picks as s0 --
// the route's dispatch folded into the handshake's (deepseek_v41.metal
// keeps the device-scope twin under the same name). grid {256}, tg {256}.
kernel void dsv41_handshake_route(
    volatile coherent(system) device uint* f [[buffer(0)]],
    const device uint*  s1      [[buffer(2)]],
    const device uint*  s2      [[buffer(3)]],
    volatile coherent(system) device uint* bo [[buffer(4)]],
    volatile coherent(system) device const uint* bi [[buffer(5)]],
    device uint*        d0      [[buffer(6)]],
    device uint*        d1      [[buffer(7)]],
    device uint*        d2      [[buffer(8)]],
    constant uint4&     n       [[buffer(9)]],    // n0, n1, n2, seq
    constant uint4&     m       [[buffer(10)]],   // m0, m1, m2, spins
    const device float* sc      [[buffer(11)]],   // [rows, E]
    const device float* bias    [[buffer(12)]],   // [E]
    const device float* bias_vl [[buffer(13)]],   // [E]
    const device uchar* vl      [[buffer(14)]],   // [rows] (has_vl)
    device int*         eid     [[buffer(15)]],   // [rows, k]
    device float*       wt      [[buffer(16)]],   // [rows, k]
    constant int&       E       [[buffer(17)]],
    constant int&       topk    [[buffer(18)]],
    constant float&     rscale  [[buffer(19)]],
    constant int&       norm    [[buffer(20)]],
    constant int&       has_vl  [[buffer(21)]],
    uint t    [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  threadgroup uint part[32];
  threadgroup uint verdict, gave;
  threadgroup float val[1024], sel[1024];
  threadgroup float rv[8];
  threadgroup int ri[8];
  threadgroup int pick[16];
  dsv41_route_tg_(sc, bias, bias_vl, vl, eid, wt, E, topk, rscale, norm,
                  has_vl, 0, t, sg, lane, val, sel, rv, ri, pick);
  threadgroup_barrier(mem_flags::mem_device);
  host_handshake_body_(f, reinterpret_cast<const device uint*>(eid), s1, s2,
                       bo, bi, d0, d1, d2, n, m, t, nt, sg, lane, part,
                       verdict, gave);
}

// ---- the visibility study (host_handshake.system_scope_study) ----------

// `rounds` handshakes in one dispatch: post r at word 0, then wait for word
// `at` to reach r, read through the system scope (`g`, the same buffer as
// `f`) with a system fence before each load (fence 1) or none (fence 0);
// post_fence 1 fences the post too. Word 2 counts a give-up. `touch`:
// 1 a store to the answer's line (the word after it) before the post,
// 2 before every load too; 3 then 32 words of data on a line read for the
// first time (word 1024 + 32 r ...), written by the host before its
// answer, checked (wrong ones counted in word 6); 4 as 3, the round's data
// read by this thread before the post; 5 as 3, the data in `h` (another
// buffer, or the same one bound again). grid {1}.
kernel void host_handshake_study(
    device atomic_uint* f [[buffer(0)]],
    volatile coherent(system) device const uint* g [[buffer(1)]],
    constant uint& rounds     [[buffer(2)]],
    constant uint& spins      [[buffer(3)]],
    constant uint& at         [[buffer(4)]],
    constant uint& fence      [[buffer(5)]],
    constant uint& post_fence [[buffer(6)]],
    constant uint& touch      [[buffer(7)]],
    volatile coherent(system) device const uint* h [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
  if (i != 0) { return; }
  for (uint r = 1; r <= rounds; ++r) {
    if (touch == 1 || touch == 2) {
      atomic_store_explicit(&f[at + 1], r, memory_order_relaxed);
    }
    if (touch == 4) {
      uint s = 0;
      for (uint j = 0; j < 32; ++j) { s += g[1024 + 32 * r + j]; }
      atomic_fetch_add_explicit(&f[4], s, memory_order_relaxed);
    }
    atomic_store_explicit(&f[0], r, memory_order_relaxed);
    if (post_fence != 0) { host_handshake_fence_(); }
    uint k = 0;
    while (++k < spins) {
      if (touch == 2) {
        atomic_store_explicit(&f[at + 1], r, memory_order_relaxed);
      }
      if (fence != 0) { host_handshake_fence_(); }
      if (g[at] >= r) { break; }
    }
    if (k >= spins) {
      atomic_fetch_add_explicit(&f[2], 1u, memory_order_relaxed);
      return;
    }
    if (touch >= 3) {
      volatile coherent(system) device const uint* d = touch == 5 ? h : g;
      host_handshake_fence_();
      for (uint j = 0; j < 32; ++j) {
        if (d[1024 + 32 * r + j] != r * 7u + j) {
          atomic_fetch_add_explicit(&f[6], 1u, memory_order_relaxed);
        }
      }
    }
  }
}

// The data test's GPU side (host_handshake.a_mid_buffer_turn_carries_data):
// `w` set to seq * 4096 + k before a turn; after it, `d` (the answer,
// copied) and `raw` (the host's plain stores) checked against seq * 7 + k,
// mismatches counted in words 5 (copied) and 6 (plain) of `f`. grid {n}.
kernel void host_handshake_study_write(
    device uint*        w   [[buffer(0)]],
    constant uint&      seq [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint k [[thread_position_in_grid]])
{
  if (k < n) { w[k] = seq * 4096u + k; }
}

kernel void host_handshake_study_check(
    const device uint*  d   [[buffer(0)]],
    const device uint*  raw [[buffer(1)]],
    device atomic_uint* f   [[buffer(2)]],
    constant uint&      seq [[buffer(3)]],
    constant uint&      n   [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
  if (k >= n) { return; }
  if (d[k] != seq * 7u + k) {
    atomic_fetch_add_explicit(&f[5], 1u, memory_order_relaxed);
  }
  if (raw[k] != seq * 7u + k) {
    atomic_fetch_add_explicit(&f[6], 1u, memory_order_relaxed);
  }
}

// Words [from, from + n) summed into out[0]: warms the GPU's caches with
// the lines a later handshake reads. grid {1}.
kernel void host_handshake_warm_read(
    device const uint* g    [[buffer(0)]],
    device uint*       out  [[buffer(1)]],
    constant uint&     from [[buffer(2)]],
    constant uint&     n    [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
  if (i != 0) { return; }
  uint s = 0;
  for (uint k = 0; k < n; ++k) { s += g[from + k]; }
  out[0] = s;
}
