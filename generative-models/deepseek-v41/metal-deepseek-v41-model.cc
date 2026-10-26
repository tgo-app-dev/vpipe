#include "generative-models/deepseek-v41/metal-deepseek-v41-model.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/moe-expert-cache.h"
#include "generative-models/safetensors-ranges.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/weight-set.h"

#include <dispatch/dispatch.h>

#include <chrono>
#include <cstdio>

#include <algorithm>
#include <bit>
#include <functional>
#include <atomic>
#include <cmath>
#include <cstring>
#include <future>
#include <filesystem>
#include <format>

namespace vpipe::genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::ComputeLibrary;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// matmul2d's tile only pays once M amortizes it (the 64-row floor).
constexpr int kMmaMinRows = 64;
// The f32-output GEMV serves a few rows; the tiled kernel takes more.
constexpr int kGemv32Rows = 8;
constexpr int kSortExperts = 512;        // dsv41_moe_sort's histogram
// Window rings hold this many slots past the window: a speculative verify
// writes rows it may not keep, and a rejected row must not take a slot a
// later query still reads (a verify is at most 1 + 7 rows).
constexpr int kRingSlack = 8;
// A rollback (truncate) takes back at most this many rows -- a
// speculative verify's -- of any forward: each compressor keeps its last
// inputs by position (so the open group can be rebuilt), the window rings
// hold this many slots past the window, and a CED prefill runs its decoder
// on this many rows past the window (so a rollback into the prompt still
// finds a whole window computed: DeepInfer's KEEP_TOKENS >= window +
// backtracking).
constexpr int kKeepRows = 8;
// The handshake: the host answers at most this many layers behind the one
// it has encoded (so the buffers in flight stay few: a host that blocks on
// the queue's limit while the GPU waits for it is a deadlock), and a wait
// gives up once the host's heartbeat has not moved in kHsGiveUp (the
// probe turns that into loads: an M4 Pro polls far faster than an M5).
constexpr int kHsAhead = 4;
// dsv41_attn_split's keys a chunk (and its threadgroup's heads)
constexpr int kAttnChunk = 16, kAttnHeads = 16;
// The two-phase top-k's chunk (DSV41_TOPK_CHUNK) and threads, and the
// fewest chunks a row takes it from. MEASURED on an M5 Ultra: 512 of 16K
// 62 us two-phase against 45 on one threadgroup, of 64K 61 against 93, of
// 128K 64 against 145; and the top 2048 (a chunk keeping half of itself)
// no quicker at any width the indexer reaches.
constexpr int kTopkChunk = 4096, kTopkTwoMin = 8, kTopkThreads = 256;
constexpr double kHsGiveUp = 0.25;   // seconds
// The hyper-connection mix's stream is summed in at most this many
// slices, each a threadgroup (_hc_slices of them).
constexpr int kHcSlices = 64;
// A forward's index scores [rows, compressed positions] f32 are made a
// chunk of rows at a time under this (score_rows_). VPIPE_DSV41_SCORE_
// CHUNKS=0 scores a piece's rows at once and cuts the pieces instead
// (piece_rows_: under the budget, no fewer than kMinPiece rows).
constexpr std::size_t kScoreBudget = std::size_t(1) << 30;
constexpr int kMinPiece = 2048;

bool
score_chunks_()
{
  static const bool on = [] {
    const char* e = std::getenv("VPIPE_DSV41_SCORE_CHUNKS");
    return e == nullptr || std::atoi(e) != 0;
  }();
  return on;
}
// A forward of this many rows reads its later engram slots on workers.
constexpr int kEngramAsyncRows = 64;

// The sorted rows the grouped expert GEMMs may need for `np` pairs over
// `E` experts: each expert's run rounded up to 32.
std::size_t
grouped_cap_(std::size_t np, std::size_t E)
{
  return (np + 31 * std::min(E, np) + 31) / 32 * 32;
}

// The E4M3 code of `f` when f is exactly an E4M3 value (none past 448, no
// NaN), else -1.
int
e4m3_exact_(float f)
{
  if (f == 0.0f) { return std::signbit(f) ? 0x80 : 0; }
  const int sign = f < 0.0f ? 0x80 : 0;
  const float a = std::fabs(f);
  if (!(a <= 448.0f)) { return -1; }
  int ex = 0;
  const float m = std::frexp(a, &ex);   // a = m 2^ex, m in [0.5, 1)
  const int E = ex + 6;                 // (1 + k / 8) 2^(E - 7)
  if (E >= 1) {
    const float k = (m * 2.0f - 1.0f) * 8.0f;
    if (k != std::floor(k) || (E == 15 && k == 7.0f)) { return -1; }
    return sign | E << 3 | (int)k;
  }
  const float k = a * 512.0f;           // k 2^-9, k in 1..7
  if (k != std::floor(k) || k < 1.0f || k > 7.0f) { return -1; }
  return sign | (int)k;
}

// A BF16 [N, K] that is the release's FP8 widened (MLX's wo_a) back into
// E4M3 codes with an E8M0 scale per 32 of a row: each 32 takes the least
// exponent that keeps its largest value within 448, at which a widened
// E4M3 value is again one (a smaller exponent only shifts its bits), so
// every code times its scale IS the BF16 value. False, `w` untouched, when
// any value is not.
bool
bf16_to_fp8_codes_(MetalCompute* mc, SharedBuffer* w, SharedBuffer* s,
                   int N, int K)
{
  if (K % 256 != 0 || w->byte_size() < (std::size_t)N * K * 2) {
    return false;
  }
  SharedBuffer codes = mc->make_shared_buffer((std::size_t)N * K);
  SharedBuffer sc = mc->make_shared_buffer((std::size_t)N * (K / 32));
  if (codes.empty() || sc.empty()) { return false; }
  const auto* v = static_cast<const std::uint16_t*>(w->contents());
  auto* cd = static_cast<std::uint8_t*>(codes.contents());
  auto* sd = static_cast<std::uint8_t*>(sc.contents());
  constexpr int kRowsAJob = 64;
  std::atomic<bool> ok_v{true};
  std::atomic<bool>* const okp = &ok_v;   // a block captures by const copy
  dispatch_apply((std::size_t)(N + kRowsAJob - 1) / kRowsAJob,
                 dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                 ^(std::size_t job) {
    const int r1 = std::min(N, (int)(job + 1) * kRowsAJob);
    std::atomic<bool>& ok = *okp;
    for (int r = (int)job * kRowsAJob; r < r1 && ok.load(); ++r) {
      for (int k0 = 0; k0 < K; k0 += 32) {
        float x[32], amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
          const std::uint32_t u =
              (std::uint32_t)v[(std::size_t)r * K + k0 + j] << 16;
          std::memcpy(&x[j], &u, 4);
          amax = std::max(amax, std::fabs(x[j]));
        }
        int e = 0;
        if (amax > 0.0f) {
          e = std::ilogb(amax) - 8;
          while (amax > std::ldexp(448.0f, e)) { ++e; }
          while (amax <= std::ldexp(448.0f, e - 1)) { --e; }
        }
        if (e + 127 < 0 || e + 127 > 254) {
          ok = false;
          return;
        }
        sd[(std::size_t)r * (K / 32) + k0 / 32] = (std::uint8_t)(e + 127);
        for (int j = 0; j < 32; ++j) {
          const int c = e4m3_exact_(std::ldexp(x[j], -e));
          if (c < 0) {
            ok = false;
            return;
          }
          cd[(std::size_t)r * K + k0 + j] = (std::uint8_t)c;
        }
      }
    }
  });
  if (!ok_v.load()) { return false; }
  *w = std::move(codes);
  *s = std::move(sc);
  return true;
}

// A word the GPU writes while this host reads it, as it is now.
std::uint32_t
load_now_(const std::uint32_t& x)
{
  return std::atomic_ref<std::uint32_t>(const_cast<std::uint32_t&>(x))
      .load(std::memory_order_acquire);
}

// dsv41_handshake_io's checksum of a mailbox's `n` words for round `seq`
// (the word after them holds it).
std::uint32_t
hs_sum_(const std::uint32_t* w, std::uint32_t n, std::uint32_t seq)
{
  std::uint32_t s = 0;
  for (std::uint32_t k = 0; k < n; ++k) {
    const std::uint32_t v = load_now_(w[k]);
    s += v * (2u * k + 1u) + (v >> 16) * 0x9E3779B9u;
  }
  return s + seq * 2654435761u;
}

// The key chunks of a row's split attention: its window and its picks.
std::size_t
attn_chunks_(const DeepseekV41Config& c)
{
  return (std::size_t)(c.window + c.index_topk + kAttnChunk - 1) /
         kAttnChunk;
}

// The two-phase top-k's top R a chunk for K over G chunks: as many keys,
// over them all, as K and a half (dsv41_topk_merge holds 3072).
int
topk_two_r_(int K, int G)
{
  return std::clamp(3 * K / 2 / std::max(G, 1), 1, std::min(K, 3072 / G));
}

// The two-phase top-k's candidates a row (each chunk's top K) and chunks a
// row, at their widest over the indexer's selections at C compressed
// positions: the top index_topk of C, the candidate source's blocks.
std::size_t
topk_two_slots_(const DeepseekV41Config& c, std::size_t C)
{
  const std::size_t CH = kTopkChunk;
  std::size_t w = (C + CH - 1) / CH * (std::size_t)c.index_topk;
  if (c.candidate_block > 0 && c.candidate_blocks > 0) {
    const std::size_t nblk = (C + c.candidate_block - 1) / c.candidate_block;
    w = std::max(w, (nblk + CH - 1) / CH * (std::size_t)c.candidate_blocks);
  }
  return w;
}

float
bf16_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t
f32_bf16_(float f)
{
  std::uint32_t b;
  std::memcpy(&b, &f, 4);
  if ((b & 0x7fffffffu) > 0x7f800000u) { return 0x7fc0; }
  b += 0x7fffu + ((b >> 16) & 1u);
  return (std::uint16_t)(b >> 16);
}

std::uint16_t
f32_elt_(float f, bool bf16)
{
  if (bf16) { return f32_bf16_(f); }
  const _Float16 h = (_Float16)f;
  std::uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

// E4M3 (fn: no infinities, 0x7F/0xFF NaN) as f32.
const float*
e4m3_table_()
{
  static const std::vector<float> t = [] {
    std::vector<float> v(256);
    for (int c = 0; c < 256; ++c) {
      const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
      float x;
      if (e == 0) {
        x = std::ldexp((float)m / 8.0f, -6);
      } else if (e == 15 && m == 7) {
        x = NAN;
      } else {
        x = std::ldexp(1.0f + (float)m / 8.0f, e - 7);
      }
      v[(std::size_t)c] = s != 0 ? -x : x;
    }
    return v;
  }();
  return t.data();
}

float
e8m0_(std::uint8_t e)
{
  return e == 255 ? NAN : std::ldexp(1.0f, (int)e - 127);
}

// Run f(lo, hi) over [0, n) in chunks on the system's worker pool.
template <typename F>
void
parallel_(std::size_t n, F&& f)
{
  if (n < 2) { f((std::size_t)0, n); return; }
  struct Ctx {
    F* f;
    std::size_t n, chunks;
  } ctx{&f, n, std::min<std::size_t>(n, 64)};
  dispatch_apply_f(ctx.chunks,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   &ctx, [](void* p, std::size_t i) {
                     const Ctx& c = *static_cast<const Ctx*>(p);
                     const std::size_t lo = i * c.n / c.chunks;
                     const std::size_t hi = (i + 1) * c.n / c.chunks;
                     if (lo < hi) { (*c.f)(lo, hi); }
                   });
}

}  // namespace

std::string
MetalDeepseekV41Model::tensor_name(const std::string& release, bool mlx)
{
  if (!mlx) { return release; }
  if (release.rfind("layers.", 0) == 0 || release.rfind("mtp.", 0) == 0 ||
      release == "embed.weight" || release == "head.weight" ||
      release == "norm.weight") {
    return "language_model." + release;
  }
  return release;
}

MetalDeepseekV41Model::~MetalDeepseekV41Model()
{
  if (_hs_beat.joinable()) {
    {
      std::lock_guard<std::mutex> lk(_hs_mu);
      _hs_quit = true;
    }
    _hs_cv.notify_all();
    _hs_beat.join();
  }
}

void
MetalDeepseekV41Model::hs_beating_(bool on)
{
  if (on && !_hs_beat.joinable()) {
    _hs_beat = std::thread([this] {
      auto* w = static_cast<std::uint32_t*>(_hsf.contents());
      std::unique_lock<std::mutex> lk(_hs_mu);
      while (!_hs_quit) {
        if (!_hs_beating) {
          _hs_cv.wait(lk);
          continue;
        }
        lk.unlock();
        std::atomic_ref<std::uint32_t>(w[3]).fetch_add(
            1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        lk.lock();
      }
    });
  }
  {
    std::lock_guard<std::mutex> lk(_hs_mu);
    _hs_beating = on;
  }
  _hs_cv.notify_all();
}

std::uint32_t
MetalDeepseekV41Model::handshake_works(MetalCompute* mc)
{
  // once a process: rounds of a word the GPU wrote carried out and one the
  // host wrote carried in, each answered as it is posted; then the loads a
  // wait nobody answers makes before it gives up, timed
  static std::mutex mu;
  static std::uint32_t known = 0;
  static bool tried = false;
  std::lock_guard<std::mutex> lk(mu);
  if (tried) { return known; }
  tried = true;
  if (mc == nullptr) { return 0; }
  ComputeLibrary lib = mc->load_library("deepseek_v41_bf16");
  ComputeFunction fill = lib.function("dsv41_fill32");
  ComputeFunction hs = lib.function("dsv41_handshake_io");
  SharedBuffer f = mc->make_shared_buffer(64), v = mc->make_shared_buffer(64);
  SharedBuffer bo = mc->make_shared_buffer(64), bi = mc->make_shared_buffer(64);
  if (!fill.valid() || !hs.valid() || f.empty() || v.empty() || bo.empty() ||
      bi.empty()) {
    return 0;
  }
  constexpr std::uint32_t kRounds = 8, kCal = 1u << 20;
  auto* hw = static_cast<std::uint32_t*>(f.contents());
  auto word = [hw](int k) { return std::atomic_ref<std::uint32_t>(hw[k]); };
  std::memset(f.contents(), 0, 64);
  std::memset(v.contents(), 0, 64);
  auto turn = [&](ComputeEncoder& enc, std::uint32_t seq, std::uint32_t nw,
                  std::uint32_t spins) {
    enc.set_function(hs);
    enc.set_buffer(0, f);
    for (unsigned b = 1; b <= 3; ++b) { enc.set_buffer(b, v); }
    enc.set_buffer(4, bo);
    enc.set_buffer(5, bi);
    for (unsigned b = 6; b <= 8; ++b) { enc.set_buffer(b, v, 16); }
    const std::uint32_t nv[4] = {nw, 0, 0, seq};
    const std::uint32_t mv[4] = {nw, 0, 0, spins};
    enc.set_constant_bytes(9, nv, sizeof nv);
    enc.set_constant_bytes(10, mv, sizeof mv);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
  };
  CommandStream st = mc->make_command_stream();
  ComputeEncoder enc = st.begin_compute();
  for (std::uint32_t r = 1; r <= kRounds; ++r) {
    enc.set_function(fill);
    enc.set_buffer(0, v);
    enc.set_constant(1, (int)(1000 + r));
    enc.set_constant(2, 1);
    enc.dispatch({1, 1, 1}, {32, 1, 1});
    turn(enc, r, 1, kCal);
  }
  enc.end();
  auto fence = st.commit();
  bool ok = true;
  const auto* out = static_cast<const std::uint32_t*>(bo.contents());
  auto* in = static_cast<std::uint32_t*>(bi.contents());
  for (std::uint32_t r = 1; r <= kRounds && ok; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    while (word(0).load(std::memory_order_acquire) < r) {
      word(3).fetch_add(1, std::memory_order_relaxed);   // the heartbeat
      if (word(2).load(std::memory_order_relaxed) != 0 ||
          std::chrono::steady_clock::now() - t0 > std::chrono::seconds(1)) {
        ok = false;
        break;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    while (ok && hs_sum_(out, 1, r) != load_now_(out[1])) {
      ok = std::chrono::steady_clock::now() - t1 < std::chrono::seconds(1);
    }
    ok = ok && out[0] == 1000 + r;
    in[0] = 2000 + r;
    in[1] = hs_sum_(in, 1, r);
    word(1).store(ok ? r : ~0u, std::memory_order_release);
  }
  std::string why;
  ok = fence.wait_ok(&why) && ok && word(2).load() == 0 &&
       static_cast<const std::uint32_t*>(v.contents())[4] == 2000 + kRounds;
  if (!ok) { return 0; }
  std::memset(f.contents(), 0, 64);
  ComputeEncoder e2 = st.begin_compute();
  turn(e2, 1, 0, kCal);
  e2.end();
  auto f2 = st.commit();
  if (!f2.wait_ok(&why) || word(2).load() != 1) { return 0; }
  const double t = f2.gpu_times().gpu_s;
  known = (std::uint32_t)std::clamp(
      t > 0.0 ? kCal / t * kHsGiveUp : (double)kCal, (double)kCal,
      (double)(1u << 30));
  return known;
}

std::unique_ptr<MetalDeepseekV41Model>
MetalDeepseekV41Model::load(const std::shared_ptr<WeightSet>& ws,
                            MetalCompute* mc, const Options& opt,
                            std::string* err)
{
  if (ws == nullptr || mc == nullptr) {
    if (err != nullptr) { *err = "no checkpoint or no metal device"; }
    return nullptr;
  }
  auto m = std::unique_ptr<MetalDeepseekV41Model>(new MetalDeepseekV41Model);
  m->_mc = mc;
  m->_opt = opt;
  m->_bf16 = opt.use_bf16;
  if (!m->init_(ws, err)) { return nullptr; }
  return m;
}

bool
MetalDeepseekV41Model::init_(const std::shared_ptr<WeightSet>& ws,
                             std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr && err->empty()) {
      *err = "[deepseek-v4.1] " + std::move(m);
    }
    return false;
  };
  _ws = ws;
  const std::string dir = ws->dir();
  std::string e;
  if (!DeepseekV41Config::from_dir(dir, &_cfg, &e)) { return fail(e); }
  const DeepseekV41Config& c = _cfg;
  if (!c.fp8) {
    return fail("only the released FP8 / FP4 checkpoint layout is read");
  }
  // the collapses: dsv41_hc_pre_norm keeps a row in registers, 8 values of
  // 1024 threads; dsv41_hc_collapse's last group norms 4 pairs a thread
  if (c.dim > 8192) { return fail("a hidden size past 8192"); }
  if (!_engram.init(c, (std::filesystem::path(dir) / "tokenizer.json")
                           .string(), &e)) {
    return fail(e);
  }
  MetalCompute* mc = _mc;
  const char* sfx = _bf16 ? "_bf16" : "";
  _lib = mc->load_library(std::string("deepseek_v41") + sfx);
  _lib_dense = mc->load_library(std::string("dense_gemm") + sfx);
  struct NF {
    metal_compute::ComputeFunction* fn;
    const char* name;
  };
  for (const NF& f : std::initializer_list<NF>{
           {&_fn_embed, "dsv41_embed_hc"}, {&_fn_rows_hc, "dsv41_rows_hc"},
           {&_fn_hc_mix, "dsv41_hc_mix"}, {&_fn_hc_pre, "dsv41_hc_pre_norm"},
           {&_fn_hc_post, "dsv41_hc_post"}, {&_fn_rms, "dsv41_rms_norm"},
           {&_fn_hc_post_pre, "dsv41_hc_post_pre"},
           {&_fn_hc_post_pre_moe, "dsv41_hc_post_pre_moe"},
           {&_fn_kv_norm_rope, "dsv41_kv_norm_rope"},
           {&_fn_hc_coll, "dsv41_hc_collapse"},
           {&_fn_rope, "dsv41_rope"}, {&_fn_ring, "dsv41_ring_write"},
           {&_fn_attn, "dsv41_attn"}, {&_fn_ungroup, "dsv41_ungroup"},
           {&_fn_attn_split, "dsv41_attn_split"},
           {&_fn_attn_merge, "dsv41_attn_merge"},
           {&_fn_pool, "dsv41_compress_pool"},
           {&_fn_keep, "dsv41_compress_keep"},
           {&_fn_iscore, "dsv41_index_score"},
           {&_fn_iscore_mma, "dsv41_index_score_mma"},
           {&_fn_iscore_kept, "dsv41_index_score_kept"},
           {&_fn_iunpack, "dsv41_index_unpack"},
           {&_fn_bscore, "dsv41_block_scores"},
           {&_fn_bmask, "dsv41_block_mask"}, {&_fn_topk, "dsv41_topk"},
           {&_fn_topk_par, "dsv41_topk_par"},
           {&_fn_topk_chunk, "dsv41_topk_chunk"},
           {&_fn_topk_merge, "dsv41_topk_merge"},
           {&_fn_route, "dsv41_route"},
           {&_fn_swiglu4, "dsv41_moe_swiglu_fp4"},
           {&_fn_down4, "dsv41_moe_down_fp4"},
           {&_fn_swiglu4r[0], "dsv41_moe_swiglu_fp4_rows2"},
           {&_fn_swiglu4r[1], "dsv41_moe_swiglu_fp4_rows4"},
           {&_fn_swiglu4r[2], "dsv41_moe_swiglu_fp4_rows8"},
           {&_fn_down4r[0], "dsv41_moe_down_fp4_rows2"},
           {&_fn_down4r[1], "dsv41_moe_down_fp4_rows4"},
           {&_fn_down4r[2], "dsv41_moe_down_fp4_rows8"},
           {&_fn_msort, "dsv41_moe_sort"},
           {&_fn_gu4g, "dsv41_moe_gu_fp4_grouped"},
           {&_fn_down4g, "dsv41_moe_down_fp4_grouped"},
           {&_fn_combine, "dsv41_moe_combine"},
           {&_fn_swiglu, "dsv41_swiglu_clamp"},
           {&_fn_egate, "dsv41_engram_gate"},
           {&_fn_gemm32, "dsv41_gemm_f32"},
           {&_fn_fill, "dsv41_fill32"}, {&_fn_rows_at, "dsv41_rows_at_hc"},
           {&_fn_copy32, "dsv41_copy32"},
           {&_fn_hc_part, "dsv41_hc_mix_part"},
           {&_fn_gv[0][0][0], "dsv41_gemv_f32"},
           {&_fn_gv[0][1][0], "dsv41_gemv_f32_rows"},
           {&_fn_gv[0][0][1], "dsv41_gemv_f32_k4"},
           {&_fn_gv[0][1][1], "dsv41_gemv_f32_rows_k4"},
           {&_fn_gv[1][0][0], "dsv41_gemv_one"},
           {&_fn_gv[1][1][0], "dsv41_gemv_rows"},
           {&_fn_gv[1][0][1], "dsv41_gemv_one_k4"},
           {&_fn_gv[1][1][1], "dsv41_gemv_rows_k4"},
           {&_fn_gv[2][0][0], "dsv41_gemv_fp8"},
           {&_fn_gv[2][1][0], "dsv41_gemv_fp8_rows"},
           {&_fn_gv[2][0][1], "dsv41_gemv_fp8_k4"},
           {&_fn_gv[2][1][1], "dsv41_gemv_fp8_rows_k4"},
           {&_fn_gvr[0][0][0], "dsv41_gemv_f32_r8"},
           {&_fn_gvr[0][1][0], "dsv41_gemv_f32_r8_rows"},
           {&_fn_gvr[0][0][1], "dsv41_gemv_f32_r8_k4"},
           {&_fn_gvr[0][1][1], "dsv41_gemv_f32_r8_rows_k4"},
           {&_fn_gvr[1][0][0], "dsv41_gemv_one_r8"},
           {&_fn_gvr[1][1][0], "dsv41_gemv_rows_r8"},
           {&_fn_gvr[1][0][1], "dsv41_gemv_one_r8_k4"},
           {&_fn_gvr[1][1][1], "dsv41_gemv_rows_r8_k4"},
           {&_fn_gvr[2][0][0], "dsv41_gemv_fp8_r8"},
           {&_fn_gvr[2][1][0], "dsv41_gemv_fp8_r8_rows"},
           {&_fn_gvr[2][0][1], "dsv41_gemv_fp8_r8_k4"},
           {&_fn_gvr[2][1][1], "dsv41_gemv_fp8_r8_rows_k4"},
           {&_fn_hc_mean, "dsv41_hc_mean"},
           {&_fn_spark_attn, "dsv41_spark_attn"},
           {&_fn_gather_tok, "dsv41_gather_tok"},
           {&_fn_add_argmax, "dsv41_add_argmax"},
           {&_fn_dq8, "dsv41_dequant_fp8"}}) {
    *f.fn = _lib.function(f.name);
    if (!f.fn->valid()) {
      return fail(std::format("kernel {} did not load", f.name));
    }
  }
  _fn_gemv = _lib_dense.function("dense_gemv_t_f16");
  _fn_gemm = _lib_dense.function("dense_gemm_t_f16");
  _fn_gemm64 = _lib_dense.function("dense_gemm_t_bm64_f16");
  if (!_fn_gemv.valid() || !_fn_gemm.valid() || !_fn_gemm64.valid()) {
    return fail("a dense GEMM kernel did not load");
  }
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_DSV41_NO_MMA") == nullptr) {
    _lib_mma = mc->load_library(std::string("dense_gemm_mma") + sfx);
    _fn_mma = _lib_mma.function("dense_gemm_mma_t_n128_f16");
    _fn_mma_deep = _lib_mma.function("dense_gemm_mma_t_n128x256_f16");
    _mma = _fn_mma.valid() && _fn_mma_deep.valid();
    // the grouped expert GEMMs' matrix-core twins (VPIPE_DSV41_MOE_NAX=0
    // keeps the f32 simdgroup kernels) and a prefill's attention on them
    // (VPIPE_DSV41_ATTN_NAX=0 keeps a simdgroup a head)
    _lib_nax = mc->load_library(std::string("deepseek_v41_nax") + sfx);
    const char* nx = std::getenv("VPIPE_DSV41_MOE_NAX");
    if (_lib_nax.valid() && (nx == nullptr || std::atoi(nx) != 0)) {
      _fn_gu4n = _lib_nax.function("dsv41_moe_gu_nax_fp4");
      _fn_down4n = _lib_nax.function("dsv41_moe_down_nax_fp4");
      _moe_nax = _fn_gu4n.valid() && _fn_down4n.valid();
    }
    const char* na = std::getenv("VPIPE_DSV41_ATTN_NAX");
    if (_lib_nax.valid() && (na == nullptr || std::atoi(na) != 0)) {
      _fn_attn_nax = _lib_nax.function("dsv41_attn_nax");
      _attn_nax = _fn_attn_nax.valid() && c.head_dim == 512;
    }
    // the indexer's dense scores (VPIPE_DSV41_INDEX_NAX=0 keeps the f32
    // simdgroup matrices: other bits)
    const char* ni = std::getenv("VPIPE_DSV41_INDEX_NAX");
    if (_lib_nax.valid() && (ni == nullptr || std::atoi(ni) != 0)) {
      _fn_iscore_nax = _lib_nax.function("dsv41_index_score_nax");
      _index_nax = _fn_iscore_nax.valid() && c.index_heads <= 32 &&
                   c.index_dim == 128;
    }
    // its P to the element type in two parts (VPIPE_DSV41_ATTN_PSPLIT=0:
    // one, faster and an f16 run's attention twice as far from exact)
    if (const char* e = std::getenv("VPIPE_DSV41_ATTN_PSPLIT")) {
      _attn_psplit = std::atoi(e) != 0;
    }
  }
  // The routed experts as grouped GEMMs from this many pairs an expert (a
  // tile of 32 rows reads its expert once; the pair kernels once a row).
  // VPIPE_DSV41_MOE_GROUPED_MIN pins it, 0 never.
  if (const char* e = std::getenv("VPIPE_DSV41_MOE_GROUPED_MIN")) {
    _grouped_ppe = std::atof(e);
  }
  // A verify's few rows read each expert once for every row that picked
  // it, up to _rows_mm of them at a time (bit-identical to the pairs);
  // VPIPE_DSV41_MOE_ROWS=<2|4|8> pins the width, 0 off.
  if (const char* e = std::getenv("VPIPE_DSV41_MOE_ROWS")) {
    const int v = std::atoi(e);
    _rows_mm = v >= 8 ? 8 : v >= 4 ? 4 : v >= 2 ? 2 : 0;
  }
  // The few-row GEMVs 8 lanes a row; VPIPE_DSV41_GEMV_R8=0 keeps a lane's
  // chunk a step across the row.
  if (const char* e = std::getenv("VPIPE_DSV41_GEMV_R8")) {
    _gemv_r8 = std::atoi(e) != 0;
  }
  // The top-k with its serial steps made parallel (the same picks);
  // VPIPE_DSV41_TOPK_PAR=0 keeps one thread at them.
  if (const char* e = std::getenv("VPIPE_DSV41_TOPK_PAR")) {
    _topk_par = std::atoi(e) != 0;
  }
  // A decode or verify step's top-k across the GPU's cores (the same
  // picks); VPIPE_DSV41_TOPK_TWO=0 keeps a row on one threadgroup, =2 takes
  // it at every width (tests).
  if (const char* e = std::getenv("VPIPE_DSV41_TOPK_TWO")) {
    _topk_two = std::atoi(e);
  }
  // The indexer's scores on simdgroup matrices; VPIPE_DSV41_INDEX_MMA=0
  // keeps a thread a key.
  if (const char* e = std::getenv("VPIPE_DSV41_INDEX_MMA")) {
    _index_mma = std::atoi(e) != 0;
  }
  // A layer past the candidate source scores only the positions of the
  // blocks it kept (the same picks); VPIPE_DSV41_INDEX_KEPT=0 scores every
  // position under the blocks' mask.
  if (const char* e = std::getenv("VPIPE_DSV41_INDEX_KEPT")) {
    _index_kept = std::atoi(e) != 0;
  }
  // A sublayer's post taken with the next collapse in one pass (the same
  // bits); VPIPE_DSV41_HC_FUSE=0 runs them apart.
  if (const char* e = std::getenv("VPIPE_DSV41_HC_FUSE")) {
    _hc_fuse = std::atoi(e) != 0;
  }
  // CED prefill (Options::ced; VPIPE_DSV41_CED=0|1 overrides).
  _ced = _opt.ced;
  if (const char* e = std::getenv("VPIPE_DSV41_CED")) {
    _ced = std::atoi(e) != 0;
  }
  _ced_mid = c.ced_mid();
  // The collapse over several threadgroups a row; VPIPE_DSV41_HC_MT=0
  // keeps one a row (the old kernels: other bits, the same arithmetic).
  if (const char* e = std::getenv("VPIPE_DSV41_HC_MT")) {
    _hc_mt = std::atoi(e) != 0;
  }
  // The mix's slices (A/B; other bits): VPIPE_DSV41_HC_SLICES=1..64.
  if (const char* e = std::getenv("VPIPE_DSV41_HC_SLICES")) {
    _hc_slices = std::clamp(std::atoi(e), 1, kHcSlices);
  }
  // A decode step's route inside the handshake's dispatch (the same
  // code); VPIPE_DSV41_HS_ROUTE=0 keeps it its own.
  if (const char* e = std::getenv("VPIPE_DSV41_HS_ROUTE")) {
    _hs_route = std::atoi(e) != 0;
  }
  // A decode or verify step's independent dispatches in concurrent
  // groups; VPIPE_DSV41_CONCURRENT=0 runs them in order.
  if (const char* e = std::getenv("VPIPE_DSV41_CONCURRENT")) {
    _concurrent = std::atoi(e) != 0;
  }
  // MLX's bf16 wo_a back to FP8 codes where exact; VPIPE_DSV41_WO_A_BF16=1
  // keeps it as stored.
  if (const char* e = std::getenv("VPIPE_DSV41_WO_A_BF16")) {
    _fp8_wo_a = std::atoi(e) == 0;
  }
  // A decode or verify step's attention split by keys; the A/B is
  // VPIPE_DSV41_SPLIT_ATTN=0 (one simdgroup a head over every key).
  if (const char* e = std::getenv("VPIPE_DSV41_SPLIT_ATTN")) {
    _split_attn = std::atoi(e) != 0;
  }

  // ---- weights ----------------------------------------------------------
  WeightSet& wts = *ws;
  const auto& src = wts.src();
  const bool bf16 = _bf16;
  // Two layouts of one model. DeepSeek's release: names as inference/
  // model.py spells them, FP8 projections with an E8M0 scale per 32x32
  // block, an expert's FP4 tensors of its own, FP8 n-gram tables. MLX's
  // (oMLX's oQ packs): the language model under `language_model.`, MXFP8
  // projections -- the same E4M3 codes, an E8M0 scale per 32 of a ROW --
  // a layer's experts stacked [E, rows, K/8] in MXFP4 (the release's codes
  // and scales, laid end to end), wo_a already bf16, the n-gram tables
  // affine at 4 bits per 32. Every name below is the release's; `tn` maps.
  _mlx = src.info("language_model.embed.weight") != nullptr;
  const bool mlx = _mlx;
  auto tn = [mlx](const std::string& r) { return tensor_name(r, mlx); };
  auto shape_is = [&](const std::string& nm0, std::vector<std::int64_t> sh,
                      const char* dtype) {
    const auto* ti = src.info(tn(nm0));
    return ti != nullptr && ti->dtype == dtype && ti->shape == sh;
  };
  std::string bad;
  auto note = [&](const std::string& nm) {
    if (bad.empty()) { bad = nm; }
    return SharedBuffer{};
  };
  // A BF16 tensor in the compute dtype.
  auto elt = [&](const std::string& nm,
                 std::vector<std::int64_t> sh) -> SharedBuffer {
    if (!shape_is(nm, sh, "BF16")) { return note(nm); }
    SharedBuffer raw = wts.read(tn(nm), _mc, WeightSet::Residency::Copied);
    if (raw.empty()) { return note(nm); }
    if (bf16) { return raw; }
    const std::size_t n = raw.byte_size() / 2;
    SharedBuffer out = _mc->make_shared_buffer(n * 2);
    if (out.empty()) { return note(nm); }
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    auto* d = static_cast<std::uint16_t*>(out.contents());
    parallel_(n, [&](std::size_t lo, std::size_t hi) {
      for (std::size_t i = lo; i < hi; ++i) {
        d[i] = f32_elt_(bf16_f32_(s[i]), false);
      }
    });
    return out;
  };
  // An F32 tensor as stored.
  auto f32 = [&](const std::string& nm,
                 std::vector<std::int64_t> sh) -> SharedBuffer {
    if (!shape_is(nm, sh, "F32")) { return note(nm); }
    SharedBuffer raw = wts.read(tn(nm), _mc, WeightSet::Residency::Copied);
    if (raw.empty()) { return note(nm); }
    return raw;
  };
  // An E4M3 [N, K] projection -- E8M0 scales per 32x32 block (release) or
  // per 32 of a row (MXFP8) -- dequantized into the compute dtype (exact in
  // bf16: 3 mantissa bits times a power of two), into `out` at row `row0`
  // (fused matrices) or a new buffer.
  auto fp8_into = [&](const std::string& p, std::int64_t N, std::int64_t K,
                      SharedBuffer* out, std::int64_t row0) -> bool {
    const std::string wn = p + ".weight";
    const std::string sn = p + (mlx ? ".scales" : ".scale");
    const std::int64_t NB = (N + 31) / 32, KB = (K + 31) / 32;
    const bool ok = mlx ? shape_is(wn, {N, K / 4}, "U32") &&
                              shape_is(sn, {N, KB}, "U8")
                        : shape_is(wn, {N, K}, "F8_E4M3") &&
                              shape_is(sn, {NB, KB}, "F8_E8M0");
    if (!ok) {
      note(wn);
      return false;
    }
    SharedBuffer w = wts.read(tn(wn), _mc, WeightSet::Residency::Copied);
    SharedBuffer s = wts.read(tn(sn), _mc, WeightSet::Residency::Copied);
    if (w.empty() || s.empty()) { note(wn); return false; }
    if (out->empty()) {
      *out = _mc->make_shared_buffer((std::size_t)N * (std::size_t)K * 2);
      if (out->empty()) { note(wn); return false; }
    }
    const auto* wc = static_cast<const std::uint8_t*>(w.contents());
    const auto* sc = static_cast<const std::uint8_t*>(s.contents());
    auto* d = static_cast<std::uint16_t*>(out->contents()) +
              (std::size_t)row0 * (std::size_t)K;
    const float* lut = e4m3_table_();
    parallel_((std::size_t)N, [&](std::size_t lo, std::size_t hi) {
      for (std::size_t r = lo; r < hi; ++r) {
        const std::uint8_t* sr = sc + (mlx ? r : r / 32) * (std::size_t)KB;
        for (std::int64_t k = 0; k < K; ++k) {
          const float v = lut[wc[r * (std::size_t)K + (std::size_t)k]] *
                          e8m0_(sr[k / 32]);
          d[r * (std::size_t)K + (std::size_t)k] = f32_elt_(v, bf16);
        }
      }
    });
    return true;
  };
  // An oQ pack's affine module `p` [N, K]: .weight U32 [N, K * bits / 32],
  // .scales and .biases BF16 [N, K / gs]. Its width and group, or false
  // when `p` is stored otherwise.
  auto affine_of = [&](const std::string& p, std::int64_t N, std::int64_t K,
                       int* bits, int* gs) -> bool {
    if (!mlx) { return false; }
    const auto* wi = src.info(tn(p + ".weight"));
    const auto* si = src.info(tn(p + ".scales"));
    const auto* bi = src.info(tn(p + ".biases"));
    if (wi == nullptr || si == nullptr || bi == nullptr ||
        wi->dtype != "U32" || si->dtype != "BF16" || bi->dtype != "BF16" ||
        wi->shape.size() != 2 || si->shape.size() != 2 ||
        si->shape != bi->shape || wi->shape[0] != N || si->shape[0] != N ||
        si->shape[1] <= 0 || K % si->shape[1] != 0 ||
        wi->shape[1] * 32 % K != 0) {
      return false;
    }
    const int b = (int)(wi->shape[1] * 32 / K);
    const std::int64_t g = K / si->shape[1];
    if (qidx_(b) < 0 || (g != 32 && g != 64 && g != 128)) { return false; }
    *bits = b;
    *gs = (int)g;
    return true;
  };
  // An affine module's values in the compute dtype into rows [row0, row0 +
  // N) of `out` (s * q + b in f32, rounded once).
  auto affine_dense_into = [&](const std::string& p, std::int64_t N,
                               std::int64_t K, int bits, int gs,
                               SharedBuffer* out,
                               std::int64_t row0) -> bool {
    SharedBuffer w = wts.read(tn(p + ".weight"), _mc,
                              WeightSet::Residency::Copied);
    SharedBuffer s = wts.read(tn(p + ".scales"), _mc,
                              WeightSet::Residency::Copied);
    SharedBuffer b = wts.read(tn(p + ".biases"), _mc,
                              WeightSet::Residency::Copied);
    if (w.empty() || s.empty() || b.empty()) { note(p); return false; }
    const auto* wc = static_cast<const std::uint32_t*>(w.contents());
    const auto* sc = static_cast<const std::uint16_t*>(s.contents());
    const auto* bc = static_cast<const std::uint16_t*>(b.contents());
    auto* d = static_cast<std::uint16_t*>(out->contents()) +
              (std::size_t)row0 * (std::size_t)K;
    const std::size_t rw = (std::size_t)K * bits / 32, G = K / gs;
    const std::uint32_t mask = (1u << bits) - 1u;
    parallel_((std::size_t)N, [&](std::size_t lo, std::size_t hi) {
      for (std::size_t r = lo; r < hi; ++r) {
        for (std::int64_t k = 0; k < K; ++k) {
          const std::size_t bit = (std::size_t)k * bits;
          const std::size_t wd = r * rw + bit / 32, sh = bit % 32;
          std::uint32_t v = wc[wd] >> sh;
          if (sh + bits > 32) { v |= wc[wd + 1] << (32 - sh); }
          const std::size_t g = r * G + (std::size_t)k / gs;
          const float x = std::fma(bf16_f32_(sc[g]), (float)(v & mask),
                                   bf16_f32_(bc[g]));
          d[r * (std::size_t)K + (std::size_t)k] = f32_elt_(x, bf16);
        }
      }
    });
    return true;
  };
  // A module read whole into the compute dtype at rows [row0, row0 + N)
  // of `out`: BF16 as stored, or an oQ pack's affine codes widened (the
  // small ones a decode reads as rows: the compressor, the indexer's
  // weights_proj and wk).
  auto dense_into = [&](const std::string& p, std::int64_t N,
                        std::int64_t K, SharedBuffer* out,
                        std::int64_t row0) -> bool {
    int qb = 0, qg = 0;
    if (affine_of(p, N, K, &qb, &qg)) {
      return affine_dense_into(p, N, K, qb, qg, out, row0);
    }
    SharedBuffer b = elt(p + ".weight", {N, K});
    if (b.empty()) { return false; }
    std::memcpy(static_cast<std::uint8_t*>(out->contents()) +
                    (std::size_t)row0 * (std::size_t)K * 2,
                b.contents(), (std::size_t)N * (std::size_t)K * 2);
    return true;
  };
  auto dense = [&](const std::string& p, std::int64_t N,
                   std::int64_t K) -> SharedBuffer {
    SharedBuffer out = _mc->make_shared_buffer((std::size_t)N * K * 2);
    if (out.empty() || !dense_into(p, N, K, &out, 0)) { return note(p); }
    return out;
  };
  // The same projection(s) KEPT as E4M3 codes with an E8M0 scale per 32
  // of a row (the release's block scales spread to their rows), several
  // stacked by rows into one (`Ntot` rows, this one's from `row0`); an oQ
  // pack's affine codes kept as stored, rows stacked alike (each piece at
  // the same width) -- widened where the GEMV does not take them (K not a
  // multiple of 256) or widened projections were asked for.
  const bool keep_fp8 = keeps_fp8_projections();
  auto proj_into = [&](const std::string& p, std::int64_t N, std::int64_t K,
                       std::int64_t Ntot, Proj* out,
                       std::int64_t row0) -> bool {
    int qb = 0, qg = 0;
    if (affine_of(p, N, K, &qb, &qg)) {
      const bool keep = keep_fp8 && K % 256 == 0;
      if (out->w.empty()) {
        out->N = (int)Ntot;
        out->K = (int)K;
        if (keep) {
          out->bits = qb;
          out->gs = qg;
          const std::size_t rb = (std::size_t)K * qb / 8;
          out->w = _mc->make_shared_buffer((std::size_t)Ntot * rb);
          out->s = _mc->make_shared_buffer((std::size_t)Ntot * (K / qg) * 2);
          out->b = _mc->make_shared_buffer((std::size_t)Ntot * (K / qg) * 2);
          if (out->w.empty() || out->s.empty() || out->b.empty()) {
            note(p);
            return false;
          }
        } else {
          out->w = _mc->make_shared_buffer((std::size_t)Ntot * K * 2);
          if (out->w.empty()) { note(p); return false; }
        }
      } else if (out->fp8 || out->bits != (keep ? qb : 0) ||
                 out->gs != (keep ? qg : 0)) {
        note(p);                        // fused pieces stored apart
        return false;
      }
      if (!keep) { return affine_dense_into(p, N, K, qb, qg, &out->w, row0); }
      SharedBuffer w = wts.read(tn(p + ".weight"), _mc,
                                WeightSet::Residency::Copied);
      SharedBuffer s = wts.read(tn(p + ".scales"), _mc,
                                WeightSet::Residency::Copied);
      SharedBuffer b = wts.read(tn(p + ".biases"), _mc,
                                WeightSet::Residency::Copied);
      if (w.empty() || s.empty() || b.empty()) { note(p); return false; }
      const std::size_t rb = (std::size_t)K * qb / 8;
      const std::size_t rs = (std::size_t)(K / qg) * 2;
      std::memcpy(static_cast<std::uint8_t*>(out->w.contents()) + row0 * rb,
                  w.contents(), (std::size_t)N * rb);
      std::memcpy(static_cast<std::uint8_t*>(out->s.contents()) + row0 * rs,
                  s.contents(), (std::size_t)N * rs);
      std::memcpy(static_cast<std::uint8_t*>(out->b.contents()) + row0 * rs,
                  b.contents(), (std::size_t)N * rs);
      return true;
    }
    if (!out->w.empty() && out->affine()) {
      note(p);                          // fused pieces stored apart
      return false;
    }
    if (out->w.empty()) {
      out->N = (int)Ntot;
      out->K = (int)K;
      out->fp8 = keep_fp8;
    }
    if (!keep_fp8) {
      if (out->w.empty()) {
        out->w = _mc->make_shared_buffer((std::size_t)Ntot * K * 2);
        if (out->w.empty()) { note(p); return false; }
      }
      return fp8_into(p, N, K, &out->w, row0);
    }
    const std::string wn = p + ".weight";
    const std::string sn = p + (mlx ? ".scales" : ".scale");
    const std::int64_t NB = (N + 31) / 32, KB = K / 32;
    const bool ok = K % 256 == 0 &&
                    (mlx ? shape_is(wn, {N, K / 4}, "U32") &&
                               shape_is(sn, {N, KB}, "U8")
                         : shape_is(wn, {N, K}, "F8_E4M3") &&
                               shape_is(sn, {NB, KB}, "F8_E8M0"));
    if (!ok) { note(wn); return false; }
    SharedBuffer w = wts.read(tn(wn), _mc, WeightSet::Residency::Copied);
    SharedBuffer sc = wts.read(tn(sn), _mc, WeightSet::Residency::Copied);
    if (w.empty() || sc.empty()) { note(wn); return false; }
    if (out->w.empty()) {
      out->w = _mc->make_shared_buffer((std::size_t)Ntot * K);
      out->s = _mc->make_shared_buffer((std::size_t)Ntot * KB);
      if (out->w.empty() || out->s.empty()) { note(wn); return false; }
    }
    std::memcpy(static_cast<std::uint8_t*>(out->w.contents()) +
                    (std::size_t)row0 * K,
                w.contents(), (std::size_t)N * K);
    auto* d = static_cast<std::uint8_t*>(out->s.contents()) +
              (std::size_t)row0 * KB;
    const auto* ss = static_cast<const std::uint8_t*>(sc.contents());
    for (std::int64_t r = 0; r < N; ++r) {
      std::memcpy(d + r * KB, ss + (mlx ? r : r / 32) * KB, (std::size_t)KB);
    }
    return true;
  };
  auto proj = [&](const std::string& p, std::int64_t N,
                  std::int64_t K) -> Proj {
    Proj out;
    if (!proj_into(p, N, K, N, &out, 0)) { return {}; }
    return out;
  };

  const std::int64_t H = c.dim, D = c.head_dim, NH = c.n_heads;
  const std::int64_t QL = c.q_lora, OL = c.o_lora, OG = c.o_groups;
  const std::int64_t I = c.moe_inter, E = c.n_experts, V = c.vocab;
  const std::int64_t IH = c.index_heads, ID = c.index_dim;
  const std::int64_t HC = c.hc, MIX = c.mix_hc();
  _embed = elt("embed.weight", {V, H});
  _head = elt("head.weight", {V, H});
  _norm = elt("norm.weight", {H});
  // What a backbone layer and a DSpark stage share: the attention's
  // projections and norms, the hyper-connections, the router over `ne`
  // experts and the shared expert.
  auto load_block = [&](const std::string& p, Layer& ly, std::int64_t ne) {
    ly.attn_norm = elt(p + "attn_norm.weight", {H});
    ly.ffn_norm = elt(p + "ffn_norm.weight", {H});
    ly.wq_a = proj(p + "attn.wq_a", QL, H);
    ly.q_norm = elt(p + "attn.q_norm.weight", {QL});
    ly.wq_b = proj(p + "attn.wq_b", NH * D, QL);
    ly.wkv = proj(p + "attn.wkv", D, H);
    ly.kv_norm = elt(p + "attn.kv_norm.weight", {D});
    // wo_a: the release's FP8; MLX's already widened to bf16 (as the
    // release's own convert.py widens it) -- or, were a pack to requantize
    // it, affine codes
    int wq_bits = 0, wq_gs = 0;
    if (mlx && affine_of(p + "attn.wo_a", OG * OL, NH * D / OG, &wq_bits,
                         &wq_gs)) {
      ly.wo_a = proj(p + "attn.wo_a", OG * OL, NH * D / OG);
    } else if (mlx) {
      ly.wo_a.w = elt(p + "attn.wo_a.weight", {OG * OL, NH * D / OG});
      ly.wo_a.N = (int)(OG * OL);
      ly.wo_a.K = (int)(NH * D / OG);
      // back to the release's codes where it is the release's FP8 widened
      // (exact; half the reads), unless widened projections were asked for
      if (keep_fp8 && _bf16 && _fp8_wo_a &&
          bf16_to_fp8_codes_(_mc, &ly.wo_a.w, &ly.wo_a.s, ly.wo_a.N,
                             ly.wo_a.K)) {
        ly.wo_a.fp8 = true;
        ++_wo_a_codes;
      }
    } else {
      ly.wo_a = proj(p + "attn.wo_a", OG * OL, NH * D / OG);
    }
    ly.wo_b = proj(p + "attn.wo_b", H, OG * OL);
    ly.sink = f32(p + "attn.attn_sink", {NH});
    // hyper-connections: fn [24, 4H] stored, kept transposed [4H, 24]
    for (int s = 0; s < 2; ++s) {
      const std::string q = p + (s == 0 ? "hc_attn_" : "hc_ffn_");
      SharedBuffer fn = f32(q + "fn", {MIX, HC * H});
      SharedBuffer& fnT = s == 0 ? ly.ha_fn : ly.hf_fn;
      if (!fn.empty()) {
        fnT = _mc->make_shared_buffer((std::size_t)MIX * HC * H * 4);
        if (!fnT.empty()) {
          const auto* a = static_cast<const float*>(fn.contents());
          auto* b = static_cast<float*>(fnT.contents());
          for (std::int64_t j = 0; j < MIX; ++j) {
            for (std::int64_t k = 0; k < HC * H; ++k) {
              b[k * MIX + j] = a[j * HC * H + k];
            }
          }
        }
      }
      (s == 0 ? ly.ha_scale : ly.hf_scale) = f32(q + "scale", {3});
      (s == 0 ? ly.ha_base : ly.hf_base) = f32(q + "base", {MIX});
    }
    ly.gate_w = elt(p + "ffn.gate.weight", {ne, H});
    ly.gate_b = f32(p + "ffn.gate.bias", {ne});
    if (src.info(tn(p + "ffn.gate.bias_vl")) != nullptr) {
      ly.gate_bvl = f32(p + "ffn.gate.bias_vl", {ne});
    }
    if (!proj_into(p + "ffn.shared_experts.w1", I, H, 2 * I, &ly.sh_w13, 0) ||
        !proj_into(p + "ffn.shared_experts.w3", I, H, 2 * I, &ly.sh_w13, I)) {
      note(p + "ffn.shared_experts");
    }
    ly.sh_w2 = proj(p + "ffn.shared_experts.w2", H, I);
  };
  _layers.resize((std::size_t)c.n_layers);
  for (int L = 0; L < c.n_layers; ++L) {
    Layer& ly = _layers[(std::size_t)L];
    const DeepseekV41Config::Plan& pl = c.plan[(std::size_t)L];
    const std::string p = std::format("layers.{}.", L);
    load_block(p, ly, E);
    if (pl.kv_source) {
      const std::int64_t rows = pl.ratio > 1 ? 2 * D : D;
      ly.c_w = _mc->make_shared_buffer((std::size_t)rows * H * 2);
      if (ly.c_w.empty() ||
          !dense_into(p + "attn.compressor.wkv", D, H, &ly.c_w, 0) ||
          (pl.ratio > 1 &&
           !dense_into(p + "attn.compressor.wgate", D, H, &ly.c_w, D))) {
        note(p + "attn.compressor");
      }
      ly.c_norm = elt(p + "attn.compressor.norm.weight", {D});
    }
    if (pl.index_source) {
      ly.i_wq_b = proj(p + "attn.indexer.wq_b", IH * ID, QL);
      ly.i_wp = dense(p + "attn.indexer.weights_proj", IH, H);
      if (pl.owns_index_keys) {
        ly.i_wk = dense(p + "attn.indexer.wk", ID, D);
        ly.i_knorm = elt(p + "attn.indexer.k_norm.weight", {ID});
      }
    }
    const int es = c.engram_slot(L);
    if (es >= 0) {
      const std::int64_t cols = c.engram_cols() * (std::int64_t)c.engram_dim;
      ly.e_wkv = proj(p + "engram.wkv", H * (HC + 1), cols);
      SharedBuffer qw = elt(p + "engram.q_weight", {HC, H});
      SharedBuffer kw = elt(p + "engram.k_weight", {HC, H});
      if (!qw.empty() && !kw.empty()) {
        ly.e_qk = _mc->make_shared_buffer((std::size_t)HC * H * 4);
        auto* d = static_cast<float*>(ly.e_qk.contents());
        const auto* a = static_cast<const std::uint16_t*>(qw.contents());
        const auto* b = static_cast<const std::uint16_t*>(kw.contents());
        for (std::int64_t k = 0; k < HC * H; ++k) {
          const float x = bf16 ? bf16_f32_(a[k])
                               : (float)reinterpret_cast<const _Float16*>(
                                     a)[k];
          const float y = bf16 ? bf16_f32_(b[k])
                               : (float)reinterpret_cast<const _Float16*>(
                                     b)[k];
          d[k] = x * y;
        }
      }
      const std::int64_t R = c.engram_rows[(std::size_t)es];
      const std::int64_t ED = c.engram_dim;
      if (mlx && shape_is(p + "engram.embed.weight", {R, ED}, "F8_E4M3")) {
        // an oQ -ngram pack's: the release's FP8 rows under MLX's names
        ly.e_rows = tn(p + "engram.embed.weight");
        ly.e_scales = tn(p + "engram.embed.scales");
        if (!shape_is(p + "engram.embed.scales", {R, ED / 32}, "F8_E8M0")) {
          note(ly.e_rows);
        }
      } else if (mlx) {
        // affine 4-bit, a bf16 scale and bias per 32
        ly.e_rows = tn(p + "engram.embed.weight");
        ly.e_scales = tn(p + "engram.embed.scales");
        ly.e_biases = tn(p + "engram.embed.biases");
        if (!shape_is(p + "engram.embed.weight", {R, ED / 8}, "U32") ||
            !shape_is(p + "engram.embed.scales", {R, ED / 32}, "BF16") ||
            !shape_is(p + "engram.embed.biases", {R, ED / 32}, "BF16")) {
          note(ly.e_rows);
        }
      } else {
        ly.e_rows = p + "engram.embed.weight";
        ly.e_scales = p + "engram.embed.scale";
        if (!shape_is(ly.e_rows, {R, ED}, "F8_E4M3") ||
            !shape_is(ly.e_scales, {R, ED / 32}, "F8_E8M0")) {
          note(ly.e_rows);
        }
      }
    }
    if (!bad.empty()) {
      return fail("tensor " + bad + " is missing or misshapen");
    }
  }
  if (!bad.empty()) {
    return fail("tensor " + bad + " is missing or misshapen");
  }

  // ---- DSpark, the draft head (mtp.*), when asked for --------------------
  if (_opt.dspark && c.n_mtp > 0 && c.dspark_block > 1) {
    const std::int64_t SE = c.dspark_experts;
    const std::int64_t NT = (std::int64_t)c.dspark_targets.size();
    const std::int64_t MR = c.dspark_markov;
    // an expert's parts, as the cache lays them: w1, its scales, w3, its
    // scales, w2, its scales
    const std::size_t pw = (std::size_t)I * H / 2, ps = (std::size_t)I * H / 32;
    for (int sg = 0; sg < c.n_mtp; ++sg) {
      auto stage = std::make_unique<Stage>();
      const std::string p = std::format("mtp.{}.", sg);
      load_block(p, stage->ly, SE);
      static const char* kW[3] = {"w1", "w3", "w2"};
      for (int k = 0; k < 3; ++k) {
        const std::int64_t N = k == 2 ? H : I, K = k == 2 ? I : H;
        SharedBuffer& w = stage->ex[2 * k];
        SharedBuffer& sc = stage->ex[2 * k + 1];
        if (mlx) {
          // stacked [E, N, K/8] codes and [E, N, K/32] scales: the pools
          const std::string b = p + "ffn.experts." + kW[k];
          if (!shape_is(b + ".weight", {SE, N, K / 8}, "U32") ||
              !shape_is(b + ".scales", {SE, N, K / 32}, "U8")) {
            note(b);
            continue;
          }
          w = wts.read(tn(b + ".weight"), _mc, WeightSet::Residency::Copied);
          sc = wts.read(tn(b + ".scales"), _mc,
                        WeightSet::Residency::Copied);
          if (w.empty() || sc.empty()) { note(b); }
          continue;
        }
        w = _mc->make_shared_buffer((std::size_t)SE * pw);
        sc = _mc->make_shared_buffer((std::size_t)SE * ps);
        if (w.empty() || sc.empty()) { note(p + "ffn.experts"); continue; }
        for (std::int64_t ex = 0; ex < SE; ++ex) {
          const std::string b =
              std::format("{}ffn.experts.{}.{}", p, ex, kW[k]);
          if (!shape_is(b + ".weight", {N, K / 2}, "I8") ||
              !shape_is(b + ".scale", {N, K / 32}, "F8_E8M0")) {
            note(b);
            break;
          }
          SharedBuffer rw = wts.read(b + ".weight", _mc,
                                     WeightSet::Residency::Copied);
          SharedBuffer rs = wts.read(b + ".scale", _mc,
                                     WeightSet::Residency::Copied);
          if (rw.empty() || rs.empty()) { note(b); break; }
          std::memcpy(static_cast<std::uint8_t*>(w.contents()) + ex * pw,
                      rw.contents(), pw);
          std::memcpy(static_cast<std::uint8_t*>(sc.contents()) + ex * ps,
                      rs.contents(), ps);
        }
      }
      _stages.push_back(std::move(stage));
    }
    _main_proj = proj("mtp.0.main_proj", H, NT * H);
    _main_norm = elt("mtp.0.main_norm.weight", {H});
    const std::string q = std::format("mtp.{}.", c.n_mtp - 1);
    _spark_norm = elt(q + "norm.weight", {H});
    _markov_embed = elt(q + "markov_head.embed.weight", {V, MR});
    _markov_head = elt(q + "markov_head.head.weight", {V, MR});
    // the confidence head is run in f32 (the reference keeps it so)
    if (shape_is(q + "confidence_head.proj.weight", {1, H + MR}, "BF16")) {
      SharedBuffer cw = wts.read(tn(q + "confidence_head.proj.weight"), _mc,
                                 WeightSet::Residency::Copied);
      const auto* u = static_cast<const std::uint16_t*>(cw.contents());
      _conf_w.resize((std::size_t)(H + MR));
      for (std::size_t k = 0; k < _conf_w.size() && u != nullptr; ++k) {
        _conf_w[k] = bf16_f32_(u[k]);
      }
    } else {
      note(q + "confidence_head.proj.weight");
    }
    if (MR % 256 != 0 || c.dspark_top_k > c.top_k ||
        SE > (std::int64_t)E) {
      return fail("a DSpark head this port does not run");
    }
    if (!bad.empty()) {
      return fail("DSpark tensor " + bad + " is missing or misshapen");
    }
  }

  // ---- the n-gram tables and the experts, read by range -----------------
  if (!c.engram_layers.empty()) {
    // UNCACHED: a lookup reads three scattered pieces of a 61 GB table, a
    // page each, and the page cache keeps nothing worth its memory -- a
    // cached hit costs a read what a miss does (concurrent preads of
    // cached pages serialize: MEASURED, 590K of them 0.96 s warm against
    // 0.69-0.75 s cold; one row read 590K times 5.3 s). What is worth
    // keeping is kept by the model (VPIPE_DSV41_ENGRAM_CACHED=1: through
    // the page cache).
    const char* ec = std::getenv("VPIPE_DSV41_ENGRAM_CACHED");
    _rows_file = SafetensorsRanges::open(
        dir, src, &e, /*cached=*/ec != nullptr && std::atoi(ec) != 0,
        [](std::string_view n) {
          return n.find(".engram.embed.") != std::string_view::npos;
        });
    if (_rows_file == nullptr) { return fail("engram tables: " + e); }
  }
  _expert_file = SafetensorsRanges::open(
      dir, src, &e, /*cached=*/false, [](std::string_view n) {
        return n.find(".ffn.experts.") != std::string_view::npos;
      });
  if (_expert_file == nullptr) { return fail("experts: " + e); }
  // A layer's experts: the release's E2M1 / E8M0, or an oQ pack's affine
  // codes ([E, N, K * bits / 32] with [E, N, K / gs] bf16 scales and
  // biases), w1 and w3 at one width, w2 at its own, layer by layer.
  int n_affine = 0;
  for (int L = 0; L < c.n_layers; ++L) {
    ExpertFmt& xf = _layers[(std::size_t)L].ex;
    int qb[3] = {0, 0, 0}, qg[3] = {0, 0, 0};
    for (int w = 1; w <= 3; ++w) {
      const std::int64_t N = w == 2 ? H : I, K = w == 2 ? I : H;
      const std::string p =
          mlx ? std::format("layers.{}.ffn.experts.w{}.", L, w)
              : std::format("layers.{}.ffn.experts.0.w{}.", L, w);
      if (mlx) {
        const auto* wi = src.info(tn(p + "weight"));
        const auto* si = src.info(tn(p + "scales"));
        const auto* bi = src.info(tn(p + "biases"));
        if (wi != nullptr && si != nullptr && bi != nullptr &&
            wi->dtype == "U32" && si->dtype == "BF16" &&
            bi->dtype == "BF16" && wi->shape.size() == 3 &&
            si->shape == bi->shape && si->shape.size() == 3 &&
            wi->shape[0] == E && wi->shape[1] == N && si->shape[0] == E &&
            si->shape[1] == N && si->shape[2] > 0 &&
            K % si->shape[2] == 0 && wi->shape[2] * 32 % K == 0) {
          const int b = (int)(wi->shape[2] * 32 / K);
          const std::int64_t g = K / si->shape[2];
          if (qidx_(b) < 0 || (g != 32 && g != 64 && g != 128)) {
            return fail(std::format("routed expert {} is affine at {} bits "
                                    "in groups of {}", p, b, g));
          }
          qb[w - 1] = b;
          qg[w - 1] = (int)g;
          continue;
        }
      }
      const bool ok =
          mlx ? shape_is(p + "weight", {E, N, K / 8}, "U32") &&
                    shape_is(p + "scales", {E, N, K / 32}, "U8")
              : shape_is(p + "weight", {N, K / 2}, "I8") &&
                    shape_is(p + "scale", {N, K / 32}, "F8_E8M0");
      if (!ok) {
        return fail("routed expert " + p +
                    " is neither E2M1 / E8M0 nor affine codes");
      }
    }
    if (qb[0] != qb[2] || qg[0] != qg[2] || (qb[0] == 0) != (qb[1] == 0)) {
      return fail(std::format("layer {}'s routed experts mix formats", L));
    }
    if (qb[0] > 0) {
      ++n_affine;
      xf.b13 = qb[0];
      xf.b2 = qb[1];
      // the swiglu takes one group size for w1 / w3, the down its own
      xf.gs = qg[0];
      if (qg[1] != qg[0]) {
        return fail(std::format("layer {}'s w2 groups ({}) are not w1's "
                                "({})", L, qg[1], qg[0]));
      }
    }
  }
  if (n_affine != 0 && n_affine != c.n_layers) {
    return fail("some layers' routed experts are FP4 and some affine");
  }
  _xaffine = n_affine > 0;
  MoeExpertCache::Options co;
  co.layers = c.n_layers;
  co.experts = (int)E;
  const std::size_t wb = (std::size_t)I * H / 2, sb = (std::size_t)I * H / 32;
  if (!_xaffine) {
    co.part_bytes = {wb, sb, wb, sb, wb, sb};
  } else {
    // A slot holds any layer's expert: each part as wide as the widest
    // layer's (an oQ2.5e pack keeps a few layers' w1 / w3 at 3 bits, the
    // rest at 2). Parts: w1 codes, scales, biases; w3's; w2's.
    std::size_t c13 = 0, s13 = 0, c2 = 0, s2 = 0;
    const std::size_t uI = (std::size_t)I, uH = (std::size_t)H;
    for (const Layer& ly : _layers) {
      const std::size_t g = (std::size_t)ly.ex.gs;
      c13 = std::max(c13, uI * uH * (std::size_t)ly.ex.b13 / 8);
      s13 = std::max(s13, uI * (uH / g) * 2);
      c2 = std::max(c2, uH * uI * (std::size_t)ly.ex.b2 / 8);
      s2 = std::max(s2, uH * (uI / g) * 2);
    }
    co.part_bytes = {c13, s13, s13, c13, s13, s13, c2, s2, s2};
    _xcw13 = (long)c13;
    _xcs13 = (long)(s13 / 2);
    _xcw2 = (long)c2;
    _xcs2 = (long)(s2 / 2);
  }
  // every affine width a module or an expert is stored at: its kernels
  for (const Layer& ly : _layers) {
    for (int b : {ly.ex.b13, ly.ex.b2, ly.wq_a.bits, ly.wq_b.bits,
                  ly.wkv.bits, ly.wo_a.bits, ly.wo_b.bits, ly.i_wq_b.bits,
                  ly.sh_w13.bits, ly.sh_w2.bits, ly.e_wkv.bits}) {
      if (b > 0 && !load_q_(b, &e)) { return fail(e); }
    }
  }
  for (const auto& sg : _stages) {
    const Layer& ly = sg->ly;
    for (int b : {ly.wq_a.bits, ly.wq_b.bits, ly.wkv.bits, ly.wo_a.bits,
                  ly.wo_b.bits, ly.sh_w13.bits, ly.sh_w2.bits,
                  _main_proj.bits}) {
      if (b > 0 && !load_q_(b, &e)) { return fail(e); }
    }
  }
  std::size_t slot = 0;
  for (std::size_t pb : co.part_bytes) { slot += pb; }
  co.transient = (int)E;
  const std::size_t all = (std::size_t)c.n_layers * (std::size_t)E;
  if (_opt.expert_cache_bytes == 0) {
    co.resident = (int)all;
  } else {
    const std::size_t n = _opt.expert_cache_bytes / slot;
    co.resident = (int)std::clamp<std::size_t>(
        n > (std::size_t)E ? n - (std::size_t)E : 0, (std::size_t)E, all);
  }
  // Holding every expert, nothing is ever read for a step alone (each
  // miss is admitted while there is room, and once all are in nothing is
  // missed or read ahead): a few spares, not a layer's worth -- 384 of
  // them were 4-7 GB held for nothing.
  if ((std::size_t)co.resident >= all) { co.transient = c.top_k; }
  // A decode step reads ahead what the next layer will likely pick, as
  // many as the cache measures pays (0, half or all of top-k) -- and a
  // verify's few rows as many for each; VPIPE_MOE_EXPERT_AHEAD=<n> pins it
  // (0 off).
  co.ahead_widths = {0, std::max(1, c.top_k / 2), c.top_k};
  co.ahead_rows = kKeepRows;
  if (const char* v = std::getenv("VPIPE_MOE_EXPERT_AHEAD")) {
    co.ahead_widths = {std::clamp(std::atoi(v), 0, (int)E)};
  }
  _ahead_max =
      *std::max_element(co.ahead_widths.begin(), co.ahead_widths.end());
  _ahead_sc = _mc->make_shared_buffer(E * 4 * kKeepRows);
  _ahead_rank.resize(E);
  _mgrp = _mc->make_shared_buffer((std::size_t)kKeepRows * c.top_k * 10 * 4);
  if (_ahead_sc.empty() || _mgrp.empty()) { return fail("out of memory"); }
  {
    const std::size_t np = (std::size_t)kKeepRows * c.top_k;
    _hsf = _mc->make_shared_buffer(64);
    _hsbo = _mc->make_shared_buffer((np + (std::size_t)kKeepRows * E + 1) *
                                    4);
    _hsbi = _mc->make_shared_buffer((np + 1) * 4);
    if (_hsf.empty() || _hsbo.empty() || _hsbi.empty()) {
      return fail("out of memory");
    }
    const char* v = std::getenv("VPIPE_DSV41_HANDSHAKE");
    _fn_hs = _lib.function("dsv41_handshake_io");
    _fn_hs_route = _lib.function("dsv41_handshake_route");
    if (!_fn_hs_route.valid()) { _hs_route = false; }
    if ((v == nullptr || std::atoi(v) != 0) && _fn_hs.valid()) {
      _hs_spins = handshake_works(_mc);
      _hs_ok = _hs_spins > 0;
    }
  }
  SafetensorsRanges* xf = _expert_file.get();
  // An expert's six tensors (parts: w1, w1 scale, w3, w3 scale, w2, w2
  // scale). The release stores a layer's scales together and its codes
  // together, so an expert is two reads of three neighbours each; MLX's
  // stacked tensors give an expert one range of each.
  // VPIPE_DSV41_SERIAL_READS=1: an expert's ranges one after another (A/B)
  const bool serial = std::getenv("VPIPE_DSV41_SERIAL_READS") != nullptr;
  std::vector<ExpertFmt> xfmt;
  for (const Layer& ly : _layers) { xfmt.push_back(ly.ex); }
  auto fill = [xf, mlx, H, I, serial, xfmt](int L, int ex,
                                            std::uint8_t* const* dst,
                                            std::string* why) {
    static const char* kW[3] = {"w1", "w2", "w3"};
    if (mlx) {
      // its ranges (three tensors' codes and scales, and an affine
      // layer's biases) read at once: a layer's one miss is then the
      // drive's wide reads, not one stream (M5 Ultra internal SSD: 1.70 ms
      // an expert against 2.51 in turn). An affine layer's parts are its
      // own sizes, at the start of parts as wide as the widest layer's.
      const ExpertFmt& f = xfmt[(std::size_t)L];
      const bool aff = f.b13 > 0;
      const int per = aff ? 3 : 2;              // tensors a weight
      std::atomic<int> bad{-1};
      std::atomic<int>* const badp = &bad;
      auto one = ^(std::size_t j) {
        const int k = (int)j / per, t = (int)j % per;
        const std::size_t N = k == 1 ? (std::size_t)H : (std::size_t)I;
        const std::size_t K = k == 1 ? (std::size_t)I : (std::size_t)H;
        std::size_t nb = 0;
        if (aff) {
          const int bits = k == 1 ? f.b2 : f.b13;
          nb = t == 0 ? N * K * bits / 8 : N * (K / f.gs) * 2;
        } else {
          nb = t == 0 ? N * K / 2 : N * K / 32;
        }
        static const char* kT[3] = {"weight", "scales", "biases"};
        const std::string nm = std::format(
            "language_model.layers.{}.ffn.experts.{}.{}", L, kW[k], kT[t]);
        // parts by weight: w1 at 0, w3 next, w2 last
        const int w_at = k == 0 ? 0 : k == 2 ? 1 : 2;
        if (!xf->read(nm, (std::size_t)ex * nb, nb, dst[w_at * per + t])) {
          badp->store(k);
        }
      };
      if (serial) {
        for (std::size_t j = 0; j < (std::size_t)(3 * per); ++j) { one(j); }
      } else {
        dispatch_apply((std::size_t)(3 * per),
                       dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,
                                                 0), one);
      }
      if (bad.load() >= 0) {
        if (why != nullptr) {
          *why = std::format("cannot read layer {} expert {}'s {}", L, ex,
                             kW[bad.load()]);
        }
        return false;
      }
      return true;
    }
    for (int scale = 0; scale < 2; ++scale) {
      std::vector<std::string> names;
      for (const char* w : kW) {
        names.push_back(std::format("layers.{}.ffn.experts.{}.{}.{}", L, ex,
                                    w, scale != 0 ? "scale" : "weight"));
      }
      thread_local std::vector<std::uint8_t> stage;
      std::vector<const std::uint8_t*> at;
      const bool together = xf->read_together(names, stage, at);
      for (int k = 0; k < 3; ++k) {
        const std::string& nm = names[(std::size_t)k];
        static const int kPartOfW[3] = {0, 4, 2};   // codes; scale +1
        std::uint8_t* d = dst[kPartOfW[k] + scale];
        if (together) {
          std::memcpy(d, at[(std::size_t)k], xf->nbytes(nm));
        } else if (!xf->read(nm, 0, xf->nbytes(nm), d)) {
          if (why != nullptr) { *why = "cannot read " + nm; }
          return false;
        }
      }
    }
    return true;
  };
  _cache = MoeExpertCache::create(_mc, co, fill, &e);
  if (_cache == nullptr) { return fail("expert cache: " + e); }
  if (const char* v = std::getenv("VPIPE_DSV41_AHEAD_STUDY")) {
    AheadStudy& a = _study;
    a.k = std::clamp(std::atoi(v), 0, 8);
    const std::size_t NL = (std::size_t)c.n_layers;
    a.early = _mc->make_shared_buffer(
        std::max<std::size_t>(NL * a.k * E * 4, 16));
    a.cur.assign(NL * E, 0.0f);
    a.prev = a.cur;
    a.picks.assign(NL, {});
    a.blocked.assign(NL, {});
    const char* sk = std::getenv("VPIPE_DSV41_AHEAD_STUDY_SKIP");
    a.skip = sk != nullptr ? std::atoi(sk) : 128;
    a.named.assign(NL * (std::size_t)(a.k + 1), {});
    a.fresh_at = std::vector<std::vector<std::uint8_t>>(a.named.size());
    a.pick_hit.assign((std::size_t)(2 * a.k + 1), {});
    a.block_hit = a.fresh = a.pick_hit;
    a.picks_n.assign(a.pick_hit.size(), 0.0);
    a.blocked_n = a.picks_n;
    a.blocked_at.assign(NL, 0.0);
    _cache->record_steps(a.k > 0);
  }
  return true;
}

// ---- rotary ------------------------------------------------------------

bool
MetalDeepseekV41Model::ensure_rope_(int positions)
{
  if (positions <= _rope_cap) { return true; }
  const DeepseekV41Config& c = _cfg;
  int cap = std::max(4096, _rope_cap);
  while (cap < positions) { cap *= 2; }
  const int half = c.rope_dim / 2;
  // precompute_freqs_cis in f32, as the reference forms it.
  auto freqs = [&](float base, int orig) {
    std::vector<float> f((std::size_t)half);
    const int dim = c.rope_dim;
    for (int i = 0; i < half; ++i) {
      const float ex = (float)(2 * i) / (float)dim;
      f[(std::size_t)i] = 1.0f / (float)std::pow((double)base, (double)ex);
    }
    if (orig > 0) {
      auto corrected = [&](double rot) {
        return dim * std::log(orig / (rot * 2 * M_PI)) /
               (2 * std::log((double)base));
      };
      const double lo = std::max(std::floor(corrected(c.beta_fast)), 0.0);
      const double hi = std::min(std::ceil(corrected(c.beta_slow)),
                                 (double)(dim - 1));
      for (int i = 0; i < half; ++i) {
        float ramp = ((float)i - (float)lo) /
                     (float)std::max(hi - lo, 1e-3);
        ramp = std::clamp(ramp, 0.0f, 1.0f);
        const float smooth = 1.0f - ramp;
        const float fi = f[(std::size_t)i];
        f[(std::size_t)i] =
            fi / c.rope_factor * (1.0f - smooth) + fi * smooth;
      }
    }
    return f;
  };
  const std::vector<float> fp = freqs(c.rope_theta, 0);
  const std::vector<float> fc = freqs(c.compress_rope_theta, c.orig_seq);
  SharedBuffer tp = _mc->make_shared_buffer((std::size_t)cap * half * 8);
  SharedBuffer tc = _mc->make_shared_buffer((std::size_t)cap * half * 8);
  if (tp.empty() || tc.empty()) { return false; }
  auto* a = static_cast<float*>(tp.contents());
  auto* b = static_cast<float*>(tc.contents());
  parallel_((std::size_t)cap, [&](std::size_t lo, std::size_t hi) {
    for (std::size_t p = lo; p < hi; ++p) {
      for (int i = 0; i < half; ++i) {
        const float x = (float)p * fp[(std::size_t)i];
        const float y = (float)p * fc[(std::size_t)i];
        const std::size_t o = (p * (std::size_t)half + (std::size_t)i) * 2;
        a[o] = (float)std::cos((double)x);
        a[o + 1] = (float)std::sin((double)x);
        b[o] = (float)std::cos((double)y);
        b[o + 1] = (float)std::sin((double)y);
      }
    }
  });
  _rope_plain = std::move(tp);
  _rope_comp = std::move(tc);
  _rope_cap = cap;
  return true;
}

// ---- state ---------------------------------------------------------------

bool
MetalDeepseekV41Model::ensure_ctx_(Ctx& cx, int end_pos, std::string* err)
{
  const DeepseekV41Config& c = _cfg;
  const std::size_t D = (std::size_t)c.head_dim;
  const std::size_t esz = 2;
  if (cx.L.empty()) {
    cx.L.resize((std::size_t)c.n_layers);
    for (int L = 0; L < c.n_layers; ++L) {
      LayerState& s = cx.L[(std::size_t)L];
      s.ring = _mc->make_shared_buffer(
          (std::size_t)(c.window + kRingSlack) * D * esz);
      const auto& pl = c.plan[(std::size_t)L];
      if (pl.kv_source && pl.ratio > 1) {
        s.pend = _mc->make_shared_buffer((std::size_t)pl.ratio * 2 * D * 4);
        s.keep = _mc->make_shared_buffer(
            (std::size_t)(kKeepRows + pl.ratio - 1) * 2 * D * 4);
        if (s.pend.empty() || s.keep.empty()) { break; }
      }
      if (s.ring.empty()) { break; }
    }
  }
  if (cx.spark_ring.size() != _stages.size()) {
    cx.spark_ring.clear();
    for (std::size_t k = 0; k < _stages.size(); ++k) {
      cx.spark_ring.push_back(_mc->make_shared_buffer(
          (std::size_t)(c.window + kRingSlack) * D * esz));
      if (cx.spark_ring.back().empty()) {
        if (err != nullptr) { *err = "out of memory for DSpark's window"; }
        return false;
      }
    }
  }
  for (int L = 0; L < c.n_layers; ++L) {
    LayerState& s = cx.L[(std::size_t)L];
    if (s.ring.empty()) {
      if (err != nullptr) { *err = "out of memory for the window"; }
      return false;
    }
    const auto& pl = c.plan[(std::size_t)L];
    if (!pl.kv_source) { continue; }
    const int need = end_pos / pl.ratio + 1;
    if (need <= s.cap) { continue; }
    int cap = std::max(1024, s.cap);
    while (cap < need) { cap *= 2; }
    SharedBuffer nc = _mc->make_shared_buffer((std::size_t)cap * D * esz);
    SharedBuffer nk;
    if (pl.owns_index_keys) {
      nk = _mc->make_shared_buffer((std::size_t)cap * c.index_dim * esz);
    }
    if (nc.empty() || (pl.owns_index_keys && nk.empty())) {
      if (err != nullptr) { *err = "out of memory for the compressed KV"; }
      return false;
    }
    if (s.cap > 0) {
      std::memcpy(nc.contents(), s.comp.contents(),
                  (std::size_t)s.cap * D * esz);
      if (pl.owns_index_keys) {
        std::memcpy(nk.contents(), s.keys.contents(),
                    (std::size_t)s.cap * c.index_dim * esz);
      }
    }
    s.comp = std::move(nc);
    s.keys = std::move(nk);
    s.cap = cap;
  }
  return true;
}

int
MetalDeepseekV41Model::seq_len(CtxId ctx) const
{
  auto it = _ctx.find(ctx);
  return it == _ctx.end() ? 0 : it->second.pos;
}

bool
MetalDeepseekV41Model::branch(CtxId parent, CtxId child)
{
  auto it = _ctx.find(parent);
  if (it == _ctx.end()) { _ctx.erase(child); return true; }
  const Ctx& p = it->second;
  Ctx c;
  c.pos = p.pos;
  c.hist = p.hist;
  c.back_lo = p.back_lo;
  c.L.resize(p.L.size());
  auto dup = [&](const SharedBuffer& b) {
    if (b.empty()) { return SharedBuffer{}; }
    SharedBuffer n = _mc->make_shared_buffer(b.byte_size());
    if (!n.empty()) { std::memcpy(n.contents(), b.contents(), b.byte_size()); }
    return n;
  };
  for (std::size_t L = 0; L < p.L.size(); ++L) {
    c.L[L].ring = dup(p.L[L].ring);
    c.L[L].comp = dup(p.L[L].comp);
    c.L[L].pend = dup(p.L[L].pend);
    c.L[L].keys = dup(p.L[L].keys);
    c.L[L].keep = dup(p.L[L].keep);
    c.L[L].cap = p.L[L].cap;
    if ((!p.L[L].ring.empty() && c.L[L].ring.empty()) ||
        (!p.L[L].comp.empty() && c.L[L].comp.empty())) {
      return false;
    }
  }
  for (const SharedBuffer& r : p.spark_ring) {
    c.spark_ring.push_back(dup(r));
    if (c.spark_ring.back().empty()) { return false; }
  }
  _ctx[child] = std::move(c);
  return true;
}

void
MetalDeepseekV41Model::release(CtxId ctx)
{
  _ctx.erase(ctx);
}

std::size_t
MetalDeepseekV41Model::kv_bytes() const
{
  std::size_t b = 0;
  for (const auto& [id, cx] : _ctx) {
    for (const LayerState& s : cx.L) {
      b += s.ring.byte_size() + s.comp.byte_size() + s.pend.byte_size() +
           s.keys.byte_size() + s.keep.byte_size();
    }
    for (const SharedBuffer& r : cx.spark_ring) { b += r.byte_size(); }
  }
  return b;
}

int
MetalDeepseekV41Model::preload_experts(std::string* err)
{
  if (_cache == nullptr) { return 0; }
  std::vector<std::pair<int, int>> keys;
  keys.reserve((std::size_t)_cfg.n_layers * _cfg.n_experts);
  for (int L = 0; L < _cfg.n_layers; ++L) {
    for (int e = 0; e < _cfg.n_experts; ++e) { keys.push_back({L, e}); }
  }
  return _cache->preload(keys, err);
}

std::size_t
MetalDeepseekV41Model::resident_bytes() const
{
  std::size_t b = _embed.byte_size() + _head.byte_size() + _norm.byte_size();
  for (const Layer& ly : _layers) {
    for (const SharedBuffer* s :
         {&ly.attn_norm, &ly.ffn_norm, &ly.q_norm, &ly.kv_norm, &ly.sink,
          &ly.c_w, &ly.c_norm, &ly.i_wp, &ly.i_wk, &ly.i_knorm,
          &ly.ha_fn, &ly.ha_scale, &ly.ha_base, &ly.hf_fn, &ly.hf_scale,
          &ly.hf_base, &ly.gate_w, &ly.gate_b, &ly.gate_bvl, &ly.e_qk}) {
      b += s->byte_size();
    }
    for (const Proj* p : {&ly.wq_a, &ly.wq_b, &ly.wkv, &ly.wo_a, &ly.wo_b,
                          &ly.i_wq_b, &ly.sh_w13, &ly.sh_w2, &ly.e_wkv}) {
      b += p->bytes();
    }
  }
  if (_cache != nullptr) { b += _cache->bytes(); }
  return b;
}

std::vector<std::size_t>
MetalDeepseekV41Model::scratch_sizes_(const DeepseekV41Config& c, int n,
                                      int cols, bool spark)
{
  const std::size_t N = (std::size_t)n, H = (std::size_t)c.dim;
  const std::size_t D = (std::size_t)c.head_dim;
  const std::size_t NH = (std::size_t)c.n_heads;
  const std::size_t QL = (std::size_t)c.q_lora;
  const std::size_t OG = (std::size_t)c.o_groups, OL = (std::size_t)c.o_lora;
  const std::size_t I = (std::size_t)c.moe_inter, E = (std::size_t)c.n_experts;
  const std::size_t K = (std::size_t)c.top_k;
  const std::size_t IH = (std::size_t)c.index_heads;
  const std::size_t ID = (std::size_t)c.index_dim;
  const std::size_t C = (std::size_t)cols;
  const std::size_t nblk =
      c.candidate_block > 0 ? (C + c.candidate_block - 1) / c.candidate_block
                            : 1;
  const std::size_t KT = (std::size_t)c.index_topk;
  const std::size_t KB = (std::size_t)std::max(c.candidate_blocks, 1);
  // the scores and block scores a chunk of rows (score_rows_), the widest
  // of the dense and kept-blocks passes
  const std::size_t KBB = KB * (std::size_t)std::max(c.candidate_block, 1);
  const std::size_t SR = (std::size_t)score_rows_(n, (int)C);
  const std::size_t SRk = (std::size_t)score_rows_(n, (int)KBB);
  const std::size_t SC = std::max(SR * C, std::min(SRk, N) * KBB);
  const std::size_t ec = (std::size_t)c.engram_cols() * c.engram_dim;
  // the pairs, or the grouped GEMMs' sorted rows (each expert's run padded
  // to a 32-row tile)
  const std::size_t cap = grouped_cap_(N * K, E);
  // in ensure_scratch_'s order
  std::vector<std::size_t> sz = {
      N * 4, N * 4 * H * 2, N * H * 2, N * QL * 2, N * QL * 2,
      N * NH * D * 2, N * D * 2, N * D * 2, N * 2 * D * 4, (N + 1) * D * 4,
      (N + 1) * ID * 2, N * IH * ID * 2, N * IH * 2, SC * 4,
      SR * nblk * 4, N * KB * 4, ((N * nblk + 3) / 4) * 4, N * KT * 4,
      N * NH * D * 2, N * OG * OL * 2, N * OG * OL * 2, N * H * 2,
      N * E * 4, N * K * 4, N * K * 4, N * K * 4, N * 2 * I * 2, N * I * 2,
      N * H * 2, cap * I * 2, cap * 4, cap * 4, cap * 4, cap / 32 * 4,
      N * K * H * 2, N * H * 2, N * 5 * H * 2, N * 16 * 4, N * 16 * 4,
      N * 16 * 4, N * 16 * 4, N * 16 * 4, N * 16 * 4, N * 4 * 4, N * 4 * 4,
      // DSpark's main hidden and main_x (when it is loaded)
      spark ? N * c.dspark_targets.size() * H * 2 : 0,
      spark ? N * H * 2 : 0,
      // the hyper-connection mix's slice sums
      std::max<std::size_t>(N, 8) * kHcSlices * 25 * 4,
      // a decode or verify step's split attention: partials, max and sum
      std::min<std::size_t>(N, kKeepRows) * NH * attn_chunks_(c) * D * 4,
      std::min<std::size_t>(N, kKeepRows) * NH * attn_chunks_(c) * 8,
      // and its two-phase top-k: the chunks' keys, positions, K-th keys
      // and top R (at most 3072 a row)
      std::min<std::size_t>(N, kKeepRows) * topk_two_slots_(c, C) * 4,
      std::min<std::size_t>(N, kKeepRows) * topk_two_slots_(c, C) * 4,
      std::min<std::size_t>(N, kKeepRows) * ((C + kTopkChunk - 1) /
                                             kTopkChunk) * 4,
      std::min<std::size_t>(N, kKeepRows) * 3072 * 4,
      // the collapse's groups: their sums of squares, a row's count
      std::max<std::size_t>(N, 8) * ((H + 1023) / 1024) * 4,
      std::max<std::size_t>(N, 8) * 4};
  for (std::size_t k = 0; k < c.engram_layers.size(); ++k) {
    sz.push_back(N * ec * 2);
  }
  return sz;
}

std::size_t
MetalDeepseekV41Model::run_bytes(const DeepseekV41Config& c, int rows,
                                 int ctx, bool spark)
{
  rows = std::max(rows, 1);
  ctx = std::max(ctx, std::min(rows, kMinPiece));
  // The scratch as ensure_scratch_ grows it: each buffer to the most a
  // piece needs, over the pieces piece_rows_ cuts -- every piece `rows`
  // wide, its scores a chunk of rows at a time; with the chunks off, the
  // widest pieces up to the context where their scores reach
  // kScoreBudget, the narrower ones from there to `ctx` (the indexer's
  // width grown in steps).
  const int cols = index_cols_(c, ctx) + 4096;
  const int rows_b = piece_rows_(c, rows, std::max(ctx - rows, 0), rows);
  int cols_a = cols;
  if (rows > kMinPiece && !score_chunks_()) {
    cols_a = std::min<int>(cols, (int)(kScoreBudget /
                                       ((std::size_t)rows * 4)) + 4096);
  }
  const auto za = scratch_sizes_(c, rows, cols_a, spark);
  const auto zb = scratch_sizes_(c, rows_b, cols, spark);
  std::size_t b = 0;
  for (std::size_t i = 0; i < za.size(); ++i) {
    b += std::max<std::size_t>(std::max(za[i], zb[i]), 16);
  }
  // a GEMM of many rows widens an FP8 projection first (gemm_p_)
  if (rows > kGemv32Rows) {
    const std::size_t H = (std::size_t)c.dim, D = (std::size_t)c.head_dim;
    const std::size_t QL = (std::size_t)c.q_lora;
    const std::size_t NHD = (std::size_t)c.n_heads * D;
    const std::size_t OG = (std::size_t)c.o_groups;
    const std::size_t OL = (std::size_t)c.o_lora;
    const std::size_t I = (std::size_t)c.moe_inter;    // the shared expert
    std::size_t w = std::max({QL * H, NHD * QL, D * H, OL * NHD / OG,
                              H * OG * OL,
                              (std::size_t)c.index_heads * c.index_dim * QL,
                              2 * I * H, H * I});
    if (!c.engram_layers.empty()) {
      w = std::max(w, 5 * H * (std::size_t)c.engram_cols() * c.engram_dim);
    }
    b += w * 2;
  }
  // one context's state at `ctx` and the rotary tables
  const std::size_t D = (std::size_t)c.head_dim;
  b += context_bytes(c, ctx);
  std::size_t rc = 4096;
  while (rc < (std::size_t)ctx) { rc *= 2; }
  b += 2 * rc * (std::size_t)(c.rope_dim / 2) * 8;
  // DSpark: each stage's window, a draft's block (residual, tokens,
  // logits, Markov rows and bias, the collapse)
  if (spark) {
    const std::size_t B = (std::size_t)std::max(c.dspark_block, 1);
    const std::size_t H = (std::size_t)c.dim;
    b += (std::size_t)c.n_mtp * (c.window + kRingSlack) * D * 2;
    b += B * 4 * H * 2 + (B + 1) * 4 + B * c.vocab * 4 +
         B * (std::size_t)c.dspark_markov * 2 + (std::size_t)c.vocab * 4 +
         B * H * 2;
  }
  // a long prefill's next piece's engram rows, read while this one runs
  // (forward(): one set beside _erow)
  if (!c.engram_layers.empty() && rows >= kEngramAsyncRows && ctx > rows) {
    b += c.engram_layers.size() * (std::size_t)rows *
         (std::size_t)c.engram_cols() * (std::size_t)c.engram_dim * 2;
  }
  // the last row's logits, a picture's rows and masks, a verify's groups,
  // the handshake's words and mailboxes
  b += (std::size_t)c.vocab * 4;
  b += (std::size_t)kKeepRows * c.top_k * 10 * 4;
  b += 64 + ((std::size_t)kKeepRows * (2 * c.top_k + c.n_experts) + 2) * 4;
  b += (std::size_t)rows * ((std::size_t)c.dim * 2 + 4 + 2) + 32;
  return b;
}

std::size_t
MetalDeepseekV41Model::context_bytes(const DeepseekV41Config& c, int ctx)
{
  // the window, the compressed latents and index keys at their doubled
  // capacity, the compressors' open rows and the inputs kept for a
  // rollback (ensure_ctx_ and the sources' growth)
  const std::size_t D = (std::size_t)c.head_dim;
  std::size_t b = 0;
  for (const auto& pl : c.plan) {
    b += (std::size_t)(c.window + kRingSlack) * D * 2;
    if (!pl.kv_source) { continue; }
    if (pl.ratio > 1) {
      b += (std::size_t)(2 * pl.ratio + kKeepRows - 1) * 2 * D * 4;
    }
    std::size_t cap = 1024;
    while (cap < (std::size_t)(std::max(ctx, 0) / std::max(pl.ratio, 1) + 1)) {
      cap *= 2;
    }
    b += cap * D * 2;
    if (pl.owns_index_keys) { b += cap * (std::size_t)c.index_dim * 2; }
  }
  // DSpark's windows are the forward's (run_bytes)
  return b;
}

std::size_t
MetalDeepseekV41Model::run_held() const
{
  std::size_t b = kv_bytes() + _wdq.byte_size() + _logits.byte_size() +
                  _rope_plain.byte_size() + _rope_comp.byte_size() +
                  _mrows.byte_size() + _mpos.byte_size() +
                  _tmask.byte_size() + _imask.byte_size();
  for (const SharedBuffer* s :
       {&_ids, &_X, &_xa, &_qa, &_qr, &_q, &_kvr, &_kv, &_ks, &_pooled,
        &_kk, &_qi, &_wp, &_S, &_B, &_bpick, &_cmask, &_idx, &_o, &_ot, &_og,
        &_ao, &_sc, &_eid, &_wt, &_slot, &_gu, &_sact, &_shout, &_act,
        &_srow, &_sdst, &_swt, &_tile2e, &_part, &_fo, &_ekv, &_mixA[0],
        &_mixA[1], &_mixA[2], &_mixF[0], &_mixF[1], &_mixF[2], &_pm[0],
        &_pm[1], &_mh, &_mx, &_hcpart, &_sX, &_stok, &_slog, &_memb,
        &_mbias, &_xpre, &_mgrp, &_apart, &_aml, &_tkk, &_tkp, &_tkt,
        &_tkr, &_hsf, &_hsbo, &_hsbi, &_hcps, &_hccnt}) {
    b += s->byte_size();
  }
  for (const auto& r : _erow) { b += r.byte_size(); }
  return b;
}

bool
MetalDeepseekV41Model::ensure_scratch_(int n, int ctx_end)
{
  const DeepseekV41Config& c = _cfg;
  int cols = index_cols_(c, ctx_end);
  if (cols > _cols) {
    // the compressed width grows in steps, not per token
    cols = std::max(cols, std::min(_cols * 2, cols + 4096));
  } else {
    cols = _cols;
  }
  // Each buffer grows to the most any one piece has needed: a piece's
  // rows shrink as the context grows (piece_rows_), so the rows-by-
  // columns buffers peak at neither the most rows nor the most columns.
  const std::vector<std::size_t> sz =
      scratch_sizes_(c, n, cols, !_stages.empty());
  SharedBuffer* const dst[] = {
      &_ids, &_X, &_xa, &_qa, &_qr, &_q, &_kvr, &_kv, &_ks, &_pooled, &_kk,
      &_qi, &_wp, &_S, &_B, &_bpick, &_cmask, &_idx, &_o, &_ot, &_og, &_ao,
      &_sc, &_eid, &_wt, &_slot, &_gu, &_sact, &_shout, &_act, &_srow,
      &_sdst, &_swt, &_tile2e, &_part, &_fo, &_ekv, &_mixA[0], &_mixA[1],
      &_mixA[2], &_mixF[0], &_mixF[1], &_mixF[2], &_pm[0], &_pm[1], &_mh,
      &_mx, &_hcpart, &_apart, &_aml, &_tkk, &_tkp, &_tkt, &_tkr, &_hcps,
      &_hccnt};
  constexpr std::size_t kNamed = sizeof(dst) / sizeof(dst[0]);
  if (sz.size() != kNamed + c.engram_layers.size()) { return false; }
  _erow.resize(c.engram_layers.size());
  bool ok = true;
  for (std::size_t i = 0; i < sz.size(); ++i) {
    SharedBuffer& b = i < kNamed ? *dst[i] : _erow[i - kNamed];
    const std::size_t want = std::max<std::size_t>(sz[i], 16);
    if (!b.empty() && b.byte_size() >= want) { continue; }
    b = _mc->make_shared_buffer(want);
    ok = ok && !b.empty();
    // the collapse's counts start at zero (and are set back after each
    // use)
    if (&b == &_hccnt && !b.empty()) {
      std::memset(b.contents(), 0, b.byte_size());
    }
  }
  if (!ok) { _rows = 0; _cols = 0; return false; }
  _rows = std::max(_rows, n);
  _cols = cols;
  return true;
}

// The rows of a forward's `n` scored at once, each [width] f32: all of
// them under kScoreBudget, else a multiple of 8 (the candidate mask's
// rows stay word-aligned) that keeps a chunk under it.
int
MetalDeepseekV41Model::score_rows_(int n, int width)
{
  const std::size_t row = (std::size_t)std::max(width, 1) * 4;
  if (!score_chunks_() || (std::size_t)n * row <= kScoreBudget) { return n; }
  return std::max<int>(8, (int)(kScoreBudget / row) / 8 * 8);
}

// The compressed positions the index sources hold at `ctx_end`.
int
MetalDeepseekV41Model::index_cols_(const DeepseekV41Config& c, int ctx_end)
{
  int cols = 1;
  for (const auto& pl : c.plan) {
    if (pl.index_source) { cols = std::max(cols, ctx_end / pl.ratio + 1); }
  }
  return cols;
}

// A prefill piece's rows from p0 with `left` to go: the most a forward
// takes (max_rows) -- its index scores made in chunks of rows -- or, with
// the chunks off, fewer where its scores [rows, columns] would pass
// kScoreBudget, never fewer than kMinPiece.
int
MetalDeepseekV41Model::piece_rows_(const DeepseekV41Config& c, int max_rows,
                                   int p0, int left)
{
  int rows = std::max(1, std::min(max_rows, left));
  for (int it = 0; it < 3 && rows > kMinPiece && !score_chunks_(); ++it) {
    const std::size_t cols = (std::size_t)index_cols_(c, p0 + rows);
    if ((std::size_t)rows * cols * 4 <= kScoreBudget) { break; }
    rows = std::max<int>(kMinPiece, (int)(kScoreBudget / (cols * 4)));
  }
  return rows;
}

// ---- encoders --------------------------------------------------------------

void
MetalDeepseekV41Model::gemm_(ComputeEncoder& enc, const SharedBuffer& w,
                             const SharedBuffer& x, std::size_t xoff,
                             const SharedBuffer& y, std::size_t yoff, int K,
                             int N, int M, std::size_t woff)
{
  // y[M, N] = x[M, K] @ w[N, K]^T; byte offsets. A few rows take one
  // GEMV, the weights read once for all of them and each row summed as a
  // decode step's one row is: a speculative verify's rows are its
  // decode's.
  if (M <= kGemv32Rows) {
    gemv_(enc, 1, w, woff, nullptr, 0, x, xoff, y, yoff, K, N, M);
    return;
  }
  if (_mma && M >= kMmaMinRows) {
    const bool deep = mma_use_wide_tile(N, K);
    const int BN = deep ? 256 : 128;
    const int band = mma_row_band(N, K);
    for (int r0 = 0; r0 < M; r0 += band) {
      const int m = std::min(band, M - r0);
      enc.set_function(deep ? _fn_mma_deep : _fn_mma);
      enc.set_buffer(0, x, xoff + (std::size_t)r0 * K * 2);
      enc.set_buffer(1, w, woff);
      enc.set_buffer(2, w, woff);
      enc.set_buffer(3, y, yoff + (std::size_t)r0 * N * 2);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.set_constant(6, m);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                    (unsigned)((m + 127) / 128), 1}, {256, 1, 1});
    }
    return;
  }
  const bool wide = M >= 128;
  const int bm = wide ? 64 : 32;
  enc.set_function(wide ? _fn_gemm64 : _fn_gemm);
  enc.set_buffer(0, x, xoff);
  enc.set_buffer(1, w, woff);
  enc.set_buffer(2, w, woff);
  enc.set_buffer(3, y, yoff);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});
}

bool
MetalDeepseekV41Model::keeps_fp8_projections()
{
  static const bool keep = std::getenv("VPIPE_DSV41_DENSE_PROJ") == nullptr;
  return keep;
}

int
MetalDeepseekV41Model::qidx_(int bits)
{
  for (int i = 0; i < 6; ++i) {
    if (kQBits[i] == bits) { return i; }
  }
  return -1;
}

bool
MetalDeepseekV41Model::load_q_(int bits, std::string* err)
{
  const int qi = qidx_(bits);
  if (qi < 0) {
    if (err != nullptr) { *err = std::format("no {}-bit kernels", bits); }
    return false;
  }
  QFns& q = _qf[qi];
  if (q.loaded) { return true; }
  const std::string b = std::to_string(bits);
  struct NF {
    metal_compute::ComputeFunction* fn;
    std::string name;
  };
  const NF fns[] = {
      {&q.gv[0][0], "dsv41_gemv_q" + b},
      {&q.gv[1][0], "dsv41_gemv_q" + b + "_rows"},
      {&q.gv[0][1], "dsv41_gemv_q" + b + "_k4"},
      {&q.gv[1][1], "dsv41_gemv_q" + b + "_rows_k4"},
      {&q.dq, "dsv41_dequant_q" + b},
      {&q.sw[0], "dsv41_moe_swiglu_q" + b},
      {&q.sw[1], "dsv41_moe_swiglu_q" + b + "_rows2"},
      {&q.sw[2], "dsv41_moe_swiglu_q" + b + "_rows4"},
      {&q.sw[3], "dsv41_moe_swiglu_q" + b + "_rows8"},
      {&q.dn[0], "dsv41_moe_down_q" + b},
      {&q.dn[1], "dsv41_moe_down_q" + b + "_rows2"},
      {&q.dn[2], "dsv41_moe_down_q" + b + "_rows4"},
      {&q.dn[3], "dsv41_moe_down_q" + b + "_rows8"},
      {&q.gu, "dsv41_moe_gu_q" + b + "_grouped"},
      {&q.dg, "dsv41_moe_down_q" + b + "_grouped"}};
  for (const NF& f : fns) {
    *f.fn = _lib.function(f.name);
    if (!f.fn->valid()) {
      if (err != nullptr) {
        *err = "[deepseek-v4.1] kernel " + f.name + " did not load";
      }
      return false;
    }
  }
  if (_lib_nax.valid()) {
    q.gun = _lib_nax.function("dsv41_moe_gu_nax_q" + b);
    q.dgn = _lib_nax.function("dsv41_moe_down_nax_q" + b);
  }
  q.loaded = true;
  return true;
}

// The GEMV of M <= 8 rows over an affine projection's rows [row0, row0 +
// N) (dsv41_gemv_q*: 8 lanes a row; K split across 4 simdgroups as gemv_
// splits it).
void
MetalDeepseekV41Model::gemv_q_(ComputeEncoder& enc, const Proj& p,
                               int row0, int N, const SharedBuffer& x,
                               std::size_t xoff, const SharedBuffer& y,
                               std::size_t yoff, int M)
{
  static const int pinned = [] {
    const char* e = std::getenv("VPIPE_DSV41_GEMV_KS");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  const int K = p.K;
  const int ks = pinned == 1 || pinned == 4
                     ? pinned
                     : ((N + 7) / 8 < 512 && K >= 1024 ? 4 : 1);
  const QFns& q = _qf[qidx_(p.bits)];
  const std::size_t rb = (std::size_t)K * p.bits / 8;
  const std::size_t rs = (std::size_t)(K / p.gs) * 2;
  enc.set_function(q.gv[M == 1 ? 0 : 1][ks == 4 ? 1 : 0]);
  enc.set_buffer(0, p.w, (std::size_t)row0 * rb);
  enc.set_buffer(1, p.s, (std::size_t)row0 * rs);
  enc.set_buffer(2, p.b, (std::size_t)row0 * rs);
  enc.set_buffer(3, x, xoff);
  enc.set_buffer(4, y, yoff);
  enc.set_constant(5, K);
  enc.set_constant(6, N);
  enc.set_constant(7, M);
  enc.set_constant(8, p.gs);
  enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2 * ks), 1},
               {32, (unsigned)(2 * ks), 1});
}

// The GEMV of M <= 8 rows: kind 0 f32 out, 1 the element type, 2 FP8
// codes with their scales (`s`). K is split across 4 simdgroups where the
// matrix has too few row blocks to fill the GPU (a property of the matrix
// alone, so a verify's rows sum as its decode's; VPIPE_DSV41_GEMV_KS=1|4
// pins it). `batches` > 1 runs that many on grid.z, x / W / y advancing by
// their element strides.
void
MetalDeepseekV41Model::gemv_(ComputeEncoder& enc, int kind,
                             const SharedBuffer& w, std::size_t woff,
                             const SharedBuffer* s, std::size_t soff,
                             const SharedBuffer& x, std::size_t xoff,
                             const SharedBuffer& y, std::size_t yoff, int K,
                             int N, int M, int batches, std::int64_t xb,
                             std::int64_t wb, std::int64_t yb)
{
  static const int pinned = [] {
    const char* e = std::getenv("VPIPE_DSV41_GEMV_KS");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  const int ks = pinned == 1 || pinned == 4
                     ? pinned
                     : ((N + 7) / 8 < 512 && K >= 1024 ? 4 : 1);
  // 8 lanes a row where K allows (dsv41_gemv_r8_impl_ and its FP8 twin)
  const bool r8 = _gemv_r8 && K % (kind == 2 ? 128 : 64) == 0;
  enc.set_function((r8 ? _fn_gvr : _fn_gv)[kind][M == 1 ? 0 : 1]
                                         [ks == 4 ? 1 : 0]);
  int i = 0;
  if (kind == 2) {
    enc.set_buffer(i++, w, woff);
    enc.set_buffer(i++, *s, soff);
    enc.set_buffer(i++, x, xoff);
  } else {
    enc.set_buffer(i++, x, xoff);
    enc.set_buffer(i++, w, woff);
  }
  enc.set_buffer(i++, y, yoff);
  enc.set_constant(i++, K);
  enc.set_constant(i++, N);
  enc.set_constant(i++, M);
  enc.set_constant(i++, xb);
  enc.set_constant(i++, wb);
  enc.set_constant(i++, yb);
  enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2 * ks), (unsigned)batches},
               {32, (unsigned)(2 * ks), 1});
}

// wo_a's groups: group g's rows of x [M, K] (at g * M * K) through its
// `rows` of P (from g * rows) into y at g * M * rows. A few rows: one
// dispatch for every group.
void
MetalDeepseekV41Model::gemm_groups_(ComputeEncoder& enc, const Proj& p,
                                    const SharedBuffer& x,
                                    const SharedBuffer& y, int M, int G,
                                    int rows)
{
  const int K = p.K;
  if (M <= kGemv32Rows && !p.affine()) {
    gemv_(enc, p.fp8 ? 2 : 1, p.w, 0, p.fp8 ? &p.s : nullptr, 0, x, 0, y, 0,
          K, rows, M, G, (std::int64_t)M * K, (std::int64_t)rows * K,
          (std::int64_t)M * rows);
    return;
  }
  // (affine codes, or many rows: a group at a time)
  for (int g = 0; g < G; ++g) {
    gemm_p_(enc, p, x, (std::size_t)g * M * K * 2, y,
            (std::size_t)g * M * rows * 2, M, g * rows, rows);
  }
}

void
MetalDeepseekV41Model::gemm_p_(ComputeEncoder& enc, const Proj& p,
                               const SharedBuffer& x, std::size_t xoff,
                               const SharedBuffer& y, std::size_t yoff, int M,
                               int row0, int nrows)
{
  const int K = p.K, N = nrows < 0 ? p.N - row0 : nrows;
  if (p.affine()) {
    if (M <= kGemv32Rows) {
      gemv_q_(enc, p, row0, N, x, xoff, y, yoff, M);
      return;
    }
    // a GEMM of many rows: widen the codes once, then the dense routes
    const std::size_t need = (std::size_t)N * K * 2;
    if (_wdq.byte_size() < need) {
      _wdq = _mc->make_shared_buffer(need);
      if (_wdq.empty()) { return; }
    }
    const std::size_t rb = (std::size_t)K * p.bits / 8;
    const std::size_t rs = (std::size_t)(K / p.gs) * 2;
    enc.set_function(_qf[qidx_(p.bits)].dq);
    enc.set_buffer(0, p.w, (std::size_t)row0 * rb);
    enc.set_buffer(1, p.s, (std::size_t)row0 * rs);
    enc.set_buffer(2, p.b, (std::size_t)row0 * rs);
    enc.set_buffer(3, _wdq);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, p.gs);
    enc.dispatch({(unsigned)(K / 32), (unsigned)N, 1}, {32, 1, 1});
    gemm_(enc, _wdq, x, xoff, y, yoff, K, N, M);
    return;
  }
  if (!p.fp8) {
    gemm_(enc, p.w, x, xoff, y, yoff, K, N, M, (std::size_t)row0 * K * 2);
    return;
  }
  const std::size_t woff = (std::size_t)row0 * K, soff = woff / 32;
  if (M <= kGemv32Rows) {
    gemv_(enc, 2, p.w, woff, &p.s, soff, x, xoff, y, yoff, K, N, M);
    return;
  }
  // a GEMM of many rows: widen the codes once, then the dense routes
  const std::size_t need = (std::size_t)N * K * 2;
  if (_wdq.byte_size() < need) {
    _wdq = _mc->make_shared_buffer(need);
    if (_wdq.empty()) { return; }
  }
  enc.set_function(_fn_dq8);
  enc.set_buffer(0, p.w, woff);
  enc.set_buffer(1, p.s, soff);
  enc.set_buffer(2, _wdq);
  enc.set_constant(3, K);
  enc.set_constant(4, N);
  enc.dispatch({(unsigned)(K / 8), (unsigned)N, 1}, {64, 1, 1});
  gemm_(enc, _wdq, x, xoff, y, yoff, K, N, M);
}

void
MetalDeepseekV41Model::gemm_f32_(ComputeEncoder& enc, const SharedBuffer& w,
                                 const SharedBuffer& x, std::size_t xoff,
                                 const SharedBuffer& y, std::size_t yoff,
                                 int K, int N, int M)
{
  if (M <= kGemv32Rows) {
    gemv_(enc, 0, w, 0, nullptr, 0, x, xoff, y, yoff, K, N, M);
    return;
  }
  enc.set_function(_fn_gemm32);
  enc.set_buffer(0, x, xoff);
  enc.set_buffer(1, w);
  enc.set_buffer(2, y, yoff);
  enc.set_constant(3, K);
  enc.set_constant(4, N);
  enc.set_constant(5, M);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
}

void
MetalDeepseekV41Model::hc_mix_(ComputeEncoder& enc, const SharedBuffer& X,
                               int n, const SharedBuffer& fn,
                               const SharedBuffer& sc, const SharedBuffer& bs,
                               SharedBuffer* mix)
{
  hc_mix_part_(enc, X, n, fn);
  hc_mix_finish_(enc, n, sc, bs, mix);
}

void
MetalDeepseekV41Model::hc_mix_part_(ComputeEncoder& enc, const SharedBuffer& X,
                                    int n, const SharedBuffer& fn)
{
  const DeepseekV41Config& c = _cfg;
  enc.set_function(_fn_hc_part);
  enc.set_buffer(0, X);
  enc.set_buffer(1, fn);
  enc.set_buffer(2, _hcpart);
  enc.set_constant(3, 4 * c.dim);
  enc.set_constant(4, _hc_slices);
  enc.dispatch({(unsigned)(256 * _hc_slices), (unsigned)n, 1}, {256, 1, 1});
}

// (Fusing the two -- the last slice of a row running the rest -- cost a
// decode token 0.25 ms of GPU time on the M5 Ultra: the rest ran beside
// wo_a, off the critical path.)
void
MetalDeepseekV41Model::hc_mix_finish_(ComputeEncoder& enc, int n,
                                      const SharedBuffer& sc,
                                      const SharedBuffer& bs,
                                      SharedBuffer* mix)
{
  const DeepseekV41Config& c = _cfg;
  enc.set_function(_fn_hc_mix);
  enc.set_buffer(0, _hcpart);
  enc.set_buffer(1, sc);
  enc.set_buffer(2, bs);
  enc.set_buffer(3, mix[0]);
  enc.set_buffer(4, mix[1]);
  enc.set_buffer(5, mix[2]);
  enc.set_constant(6, 4 * c.dim);
  enc.set_constant(7, c.eps);
  enc.set_constant(8, c.hc_eps);
  enc.set_constant(9, c.hc_iters);
  enc.set_constant(10, _hc_slices);
  enc.set_constant(11, n);
  enc.dispatch({(unsigned)(32 * n), 1, 1}, {32, 1, 1});
}

void
MetalDeepseekV41Model::hc_pre_(ComputeEncoder& enc, const SharedBuffer& X,
                               int n, const SharedBuffer& pre,
                               const SharedBuffer* w, const SharedBuffer& y)
{
  hc_coll_(enc, X, n, pre, w, y, nullptr, nullptr, nullptr, 0);
}

void
MetalDeepseekV41Model::hc_coll_(ComputeEncoder& enc, const SharedBuffer& X,
                                int n, const SharedBuffer& pre,
                                const SharedBuffer* w, const SharedBuffer& y,
                                const SharedBuffer* a,
                                const SharedBuffer* post_mix,
                                const SharedBuffer* sh, int K)
{
  const int H = _cfg.dim, G = (H + 1023) / 1024;
  const bool mt = _hc_mt && _hccnt.byte_size() >= (std::size_t)n * 4 &&
                  _hcps.byte_size() >= (std::size_t)n * G * 4;
  if (!mt && post_mix == nullptr) {
    enc.set_function(_fn_hc_pre);
    enc.set_buffer(0, X);
    enc.set_buffer(1, pre);
    enc.set_buffer(2, w != nullptr ? *w : y);
    enc.set_buffer(3, y);
    enc.set_constant(4, H);
    enc.set_constant(5, _cfg.eps);
    enc.set_constant(6, w != nullptr ? 1 : 0);
    enc.dispatch({(unsigned)(1024 * n), 1, 1}, {1024, 1, 1});
    return;
  }
  if (!mt) {
    enc.set_function(K > 0 ? _fn_hc_post_pre_moe : _fn_hc_post_pre);
    enc.set_buffer(0, *a);
    enc.set_buffer(1, X);
    enc.set_buffer(2, pre);
    enc.set_buffer(3, post_mix[1]);
    enc.set_buffer(4, post_mix[2]);
    enc.set_buffer(5, *w);
    enc.set_buffer(6, y);
    enc.set_constant(7, H);
    enc.set_constant(8, _cfg.eps);
    if (K > 0) {
      enc.set_buffer(9, *sh);
      enc.set_constant(10, K);
    }
    enc.dispatch({(unsigned)(1024 * n), 1, 1}, {1024, 1, 1});
    return;
  }
  enc.set_function(_fn_hc_coll);
  enc.set_buffer(0, a != nullptr ? *a : y);
  enc.set_buffer(1, X);
  enc.set_buffer(2, pre);
  enc.set_buffer(3, post_mix != nullptr ? post_mix[1] : pre);
  enc.set_buffer(4, post_mix != nullptr ? post_mix[2] : pre);
  enc.set_buffer(5, w != nullptr ? *w : y);
  enc.set_buffer(6, y);
  enc.set_constant(7, H);
  enc.set_constant(8, _cfg.eps);
  enc.set_buffer(9, sh != nullptr ? *sh : y);
  enc.set_constant(10, post_mix == nullptr ? -1 : K);
  enc.set_constant(11, w != nullptr ? 1 : 0);
  enc.set_buffer(12, _hcps);
  enc.set_buffer(13, _hccnt);
  enc.dispatch({(unsigned)(1024 * G), (unsigned)n, 1}, {1024, 1, 1});
}

// The study's naming at layer L's resolve, as a read-ahead would name: the
// early routers' best 48 for layers L + 1 .., and the step before's for
// L + 1 (by the biased score, as the router picks), each marked by whether
// the cache did not hold it then.
void
MetalDeepseekV41Model::study_name_(int L)
{
  static constexpr int kTop = 48;
  AheadStudy& a = _study;
  const int NL = _cfg.n_layers, E = _cfg.n_experts, K = a.k;
  const int top = std::min(kTop, E);
  const auto* early = static_cast<const float*>(a.early.contents());
  std::vector<std::int32_t> ord((std::size_t)E);
  std::vector<float> v((std::size_t)E);
  auto name = [&](int T, int p, const float* sc) {
    const auto* bias =
        static_cast<const float*>(_layers[(std::size_t)T].gate_b.contents());
    for (int e = 0; e < E; ++e) {
      const float x = sc[e];
      v[(std::size_t)e] =
          std::sqrt(x > 20.0f ? x : std::log1p(std::exp(x))) + bias[e];
      ord[(std::size_t)e] = e;
    }
    std::partial_sort(ord.begin(), ord.begin() + top, ord.end(),
                      [&](std::int32_t x, std::int32_t y) {
                        return v[(std::size_t)x] > v[(std::size_t)y];
                      });
    const std::size_t at = (std::size_t)T * (std::size_t)(K + 1) + p;
    a.named[at].assign(ord.begin(), ord.begin() + top);
    a.fresh_at[at].resize((std::size_t)top);
    for (int j = 0; j < top; ++j) {
      a.fresh_at[at][(std::size_t)j] = !_cache->holds(T, ord[(std::size_t)j]);
    }
  };
  for (int k = 1; k <= K && L + k < NL; ++k) {
    name(L + k, k - 1,
         early + ((std::size_t)L * K + (std::size_t)(k - 1)) * E);
  }
  if (a.have_prev && L + 1 < NL) {
    name(L + 1, K, a.prev.data() + (std::size_t)(L + 1) * E);
  }
}

// The study's step: each layer's picks and blockers against the experts
// each predictor named, at widths 6, 12, 24 and 48, and what naming them
// would have read (not held when named). The first `skip` steps (the
// cache's warm-up) are not counted.
void
MetalDeepseekV41Model::study_step_()
{
  static constexpr int kW[4] = {6, 12, 24, 48};
  AheadStudy& a = _study;
  const int NL = _cfg.n_layers, K = a.k;
  const auto& st = _cache->stats();
  static MoeExpertCache::Stats last;
  const bool count = a.steps >= (double)a.skip;
  if (count && a.seen > 0) {
    a.read_ms += st.read_ms - last.read_ms;
    a.wait_ms += st.wait_ms - last.wait_ms;
    a.reads += (double)(st.transient + st.admitted - last.transient -
                        last.admitted);
  }
  last = st;
  a.steps += 1.0;
  if (count) {
    a.seen += 1.0;
    for (int L = 0; L < NL; ++L) {
      std::vector<int> P = a.picks[(std::size_t)L];
      std::sort(P.begin(), P.end());
      P.erase(std::unique(P.begin(), P.end()), P.end());
      const std::vector<int>& B = a.blocked[(std::size_t)L];
      a.blocked_at[(std::size_t)L] += (double)B.size();
      const std::size_t row = (std::size_t)L * (std::size_t)(K + 1);
      for (int p = 0; p < 2 * K + 1; ++p) {
        // the lists this predictor stands on: one, or L - 1's and another
        std::vector<std::size_t> from;
        if (p <= K) {
          from = {row + (std::size_t)p};
        } else {
          from = {row, row + (std::size_t)(p - K)};
          if (p - K >= K) { continue; }
        }
        bool have = true;
        for (std::size_t f : from) { have = have && !a.named[f].empty(); }
        if (!have) { continue; }
        a.picks_n[(std::size_t)p] += (double)P.size();
        a.blocked_n[(std::size_t)p] += (double)B.size();
        for (int w = 0; w < 4; ++w) {
          std::vector<std::int32_t> set;
          std::size_t fresh = 0;
          for (std::size_t f : from) {
            const int top = std::min<int>(kW[w], (int)a.named[f].size());
            for (int j = 0; j < top; ++j) {
              const std::int32_t e = a.named[f][(std::size_t)j];
              if (std::find(set.begin(), set.end(), e) != set.end()) {
                continue;
              }
              set.push_back(e);
              fresh += a.fresh_at[f][(std::size_t)j];
            }
          }
          auto in = [&](int e) {
            return std::find(set.begin(), set.end(), e) != set.end();
          };
          auto& ph = a.pick_hit[(std::size_t)p][(std::size_t)w];
          auto& bh = a.block_hit[(std::size_t)p][(std::size_t)w];
          for (int e : P) { ph += in(e); }
          for (int e : B) { bh += in(e); }
          a.fresh[(std::size_t)p][(std::size_t)w] += (double)fresh;
        }
      }
    }
  }
  for (auto& n : a.named) { n.clear(); }
  a.prev.swap(a.cur);
  a.have_prev = true;
}

std::string
MetalDeepseekV41Model::ahead_study() const
{
  const AheadStudy& a = _study;
  if (a.k == 0 || a.seen < 2) { return {}; }
  double picks = 0, blocked = 0;
  for (double b : a.blocked_at) { blocked += b; }
  picks = a.picks_n[0];
  std::string r = std::format(
      "{:.0f} steps counted (after {}); a step's blockers {:.2f} (own reads "
      "{:.2f}), {:.1f} ms reading + {:.1f} ms waiting on reads ahead a "
      "step\n",
      a.seen, a.skip, blocked / a.seen, a.reads / (a.seen - 1),
      a.read_ms / (a.seen - 1), a.wait_ms / (a.seen - 1));
  r += "blockers by layer (tenths):";
  const int NL = (int)a.blocked_at.size();
  for (int t = 0; t < 10; ++t) {
    double s = 0;
    for (int L = t * NL / 10; L < (t + 1) * NL / 10; ++L) {
      s += a.blocked_at[(std::size_t)L];
    }
    r += std::format(" {:.0f}%", blocked > 0 ? 100.0 * s / blocked : 0.0);
  }
  r += "\npredictor: picks named / blockers named / new reads a step, "
       "at widths 6 12 24 48\n";
  (void)picks;
  for (int p = 0; p < 2 * a.k; ++p) {
    const std::size_t u = (std::size_t)p;
    r += p < a.k    ? std::format("  router of L on L-{}'s input:", p + 1)
         : p == a.k ? std::string("  the step before's own scores:")
                    : std::format("  L-1's and L-{}'s together:", p - a.k + 1);
    for (int w = 0; w < 4; ++w) {
      r += std::format(
          " {:.0f}%/{:.0f}%/{:.1f}",
          a.picks_n[u] > 0 ? 100.0 * a.pick_hit[u][(std::size_t)w] /
                                 a.picks_n[u] : 0.0,
          a.blocked_n[u] > 0 ? 100.0 * a.block_hit[u][(std::size_t)w] /
                                   a.blocked_n[u] : 0.0,
          a.fresh[u][(std::size_t)w] / a.seen);
    }
    r += "\n";
  }
  return r;
}

// Debug: `f` (rows * k) as the picks in _eid, weighed into _wt by this
// forward's own scores (_sc, rows * E) as dsv41_route weighs its own.
void
MetalDeepseekV41Model::force_picks_(const std::vector<std::int32_t>& f,
                                    int rows, int E, int K)
{
  const DeepseekV41Config& c = _cfg;
  const auto* sc = static_cast<const float*>(_sc.contents());
  auto* ew = static_cast<std::int32_t*>(_eid.contents());
  auto* wt = static_cast<float*>(_wt.contents());
  for (int t = 0; t < rows; ++t) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      const int e = f[(std::size_t)(t * K + k)];
      const float x = sc[(std::size_t)t * E + e];
      const float v =
          std::sqrt(x > 20.0f ? x : std::log(1.0f + std::exp(x)));
      ew[t * K + k] = e;
      wt[t * K + k] = v;
      sum += v;
    }
    const float dn = c.norm_topk && K > 1 ? sum + 1e-20f : 1.0f;
    for (int k = 0; k < K; ++k) {
      wt[t * K + k] = wt[t * K + k] / dn * c.route_scale;
    }
  }
}

void
MetalDeepseekV41Model::hc_post_(ComputeEncoder& enc, const SharedBuffer& a,
                                const SharedBuffer& X, int n,
                                SharedBuffer* mix)
{
  enc.set_function(_fn_hc_post);
  enc.set_buffer(0, a);
  enc.set_buffer(1, X);
  enc.set_buffer(2, mix[1]);
  enc.set_buffer(3, mix[2]);
  enc.set_constant(4, _cfg.dim);
  enc.dispatch({(unsigned)_cfg.dim, (unsigned)n, 1}, {256, 1, 1});
}

void
MetalDeepseekV41Model::rms_(ComputeEncoder& enc, const SharedBuffer& x,
                            std::size_t xoff, const SharedBuffer& w,
                            const SharedBuffer& y, std::size_t yoff, int C,
                            int rows, int ldx, int ldy, bool x_f32)
{
  enc.set_function(_fn_rms);
  enc.set_buffer(0, x, xoff);
  enc.set_buffer(1, w);
  enc.set_buffer(2, y, yoff);
  enc.set_constant(3, C);
  enc.set_constant(4, ldx);
  enc.set_constant(5, ldy);
  enc.set_constant(6, _cfg.eps);
  enc.set_constant(7, x_f32 ? 1 : 0);
  enc.dispatch({(unsigned)(256 * rows), 1, 1}, {256, 1, 1});
}

void
MetalDeepseekV41Model::rope_(ComputeEncoder& enc, const SharedBuffer& x,
                             std::size_t xoff, bool compressed, int rows,
                             int heads, int ld_row, int ld_head, int D,
                             int pos0, int step, bool inverse)
{
  const int rd = _cfg.rope_dim;
  enc.set_function(_fn_rope);
  enc.set_buffer(0, x, xoff);
  enc.set_buffer(1, compressed ? _rope_comp : _rope_plain);
  enc.set_constant(2, ld_row);
  enc.set_constant(3, ld_head);
  enc.set_constant(4, D);
  enc.set_constant(5, rd);
  enc.set_constant(6, pos0);
  enc.set_constant(7, step);
  enc.set_constant(8, inverse ? 1 : 0);
  enc.dispatch({(unsigned)(rd / 2), (unsigned)heads, (unsigned)rows},
               {(unsigned)(rd / 2), 1, 1});
}

void
MetalDeepseekV41Model::fill_(ComputeEncoder& enc, const SharedBuffer& b,
                             std::int32_t v, std::size_t words)
{
  enc.set_function(_fn_fill);
  enc.set_buffer(0, b);
  enc.set_constant(1, v);
  enc.set_constant(2, (int)words);
  enc.dispatch({(unsigned)words, 1, 1}, {256, 1, 1});
}

bool
MetalDeepseekV41Model::topk_(ComputeEncoder& enc, const SharedBuffer& s,
                             int rows, int C, int K, const SharedBuffer& out,
                             std::size_t ooff)
{
  const int G = (C + kTopkChunk - 1) / kTopkChunk;
  if (_topk_two > 0 && rows <= kKeepRows && G <= 3072 &&
      (_topk_two > 1 || (G >= kTopkTwoMin && 8 * K <= kTopkChunk))) {
    // a few rows: each chunk its own top K on its own core, then a row's
    // top K of theirs (the same picks)
    const int R = topk_two_r_(K, G);
    ++_two_passes;
    enc.set_function(_fn_topk_chunk);
    enc.set_buffer(0, s);
    enc.set_buffer(1, _tkk);
    enc.set_buffer(2, _tkp);
    enc.set_buffer(3, _tkt);
    enc.set_buffer(4, _tkr);
    enc.set_constant(5, C);
    enc.set_constant(6, K);
    enc.set_constant(7, R);
    enc.dispatch({(unsigned)(kTopkThreads * G), (unsigned)rows, 1},
                 {kTopkThreads, 1, 1});
    enc.set_function(_fn_topk_merge);
    enc.set_buffer(0, _tkk);
    enc.set_buffer(1, _tkp);
    enc.set_buffer(2, _tkt);
    enc.set_buffer(3, _tkr);
    enc.set_buffer(4, out, ooff);
    enc.set_constant(5, C);
    enc.set_constant(6, K);
    enc.set_constant(7, R);
    enc.dispatch({(unsigned)(kTopkThreads * rows), 1, 1},
                 {kTopkThreads, 1, 1});
    return true;
  }
  enc.set_function(_topk_par ? _fn_topk_par : _fn_topk);
  enc.set_buffer(0, s);
  enc.set_buffer(1, out, ooff);
  enc.set_constant(2, C);
  enc.set_constant(3, K);
  enc.dispatch({(unsigned)(1024 * rows), 1, 1}, {1024, 1, 1});
  return true;
}

bool
MetalDeepseekV41Model::engram_rows_(int slot,
                                    const std::vector<std::int64_t>& rows,
                                    int n, SharedBuffer& dst,
                                    std::string* err)
{
  // Engram's table rows for every (row, column), dequantized into the
  // compute dtype -- E4M3 x the E8M0 per 32 (exact in bf16: the release's
  // tables, and an oQ -ngram pack's), or an MLX pack's affine 4-bit codes
  // -- [n, cols * dim]. Each DISTINCT table row is read and dequantized
  // once, then copied to every lookup that names it: a row's reads (two or
  // three, each its own random read off the drive) are what a piece waits
  // for, and in text 40% of a piece's lookups repeat (MEASURED, a 17K-
  // token code prompt's first 8192 positions: 117011 distinct of 196608).
  // A distinct row lands in its first lookup's place in `dst`, the others
  // copied from there (no host copy of the rows: ~100 MB a slot at 8192
  // rows).
  const DeepseekV41Config& c = _cfg;
  const int L = c.engram_layers[(std::size_t)slot];
  const Layer& ly = _layers[(std::size_t)L];
  const int cols = c.engram_cols(), dim = c.engram_dim, ns = dim / 32;
  auto* out = static_cast<std::uint16_t*>(dst.contents());
  const float* lut = e4m3_table_();
  const bool bf16 = _bf16, mlx = !ly.e_biases.empty();
  std::atomic<bool> ok{true};
  const std::size_t total = (std::size_t)n * (std::size_t)cols;
  const std::size_t ns_ = c.engram_layers.size();
  // the slot's lookups by row id, then the distinct ids and each lookup's
  // (VPIPE_DSV41_ENGRAM_DEDUP=0: every lookup read, the A/B)
  static const bool dedup = [] {
    const char* e = std::getenv("VPIPE_DSV41_ENGRAM_DEDUP");
    return e == nullptr || std::atoi(e) != 0;
  }();
  std::vector<std::pair<std::int64_t, std::uint32_t>> key(total);
  for (std::size_t k = 0; k < total; ++k) {
    const std::size_t i = k / (std::size_t)cols, cc = k % (std::size_t)cols;
    key[k] = {rows[(i * ns_ + (std::size_t)slot) * (std::size_t)cols + cc],
              (std::uint32_t)k};
  }
  if (dedup) { std::sort(key.begin(), key.end()); }
  std::vector<std::int64_t> uniq;
  std::vector<std::uint32_t> where(total), first;
  uniq.reserve(total);
  first.reserve(total);
  for (const auto& [r, k] : key) {
    if (!dedup || uniq.empty() || uniq.back() != r) {
      uniq.push_back(r);
      first.push_back(k);
    }
    where[k] = (std::uint32_t)(uniq.size() - 1);
  }
  parallel_(uniq.size(), [&](std::size_t lo, std::size_t hi) {
    // release: E4M3 codes + E8M0 per 32; MLX: 4-bit codes (8 a word, low
    // nibble first) + a bf16 scale and bias per 32
    std::vector<std::uint8_t> w(mlx ? (std::size_t)dim / 2 : (std::size_t)dim);
    std::vector<std::uint8_t> s(mlx ? (std::size_t)ns * 2 : (std::size_t)ns);
    std::vector<std::uint8_t> bi(mlx ? (std::size_t)ns * 2 : 0);
    for (std::size_t u = lo; u < hi; ++u) {
      const std::int64_t r = uniq[u];
      bool got = _rows_file->read(ly.e_rows, (std::size_t)r * w.size(),
                                  w.size(), w.data()) &&
                 _rows_file->read(ly.e_scales, (std::size_t)r * s.size(),
                                  s.size(), s.data());
      if (mlx && got) {
        got = _rows_file->read(ly.e_biases, (std::size_t)r * bi.size(),
                               bi.size(), bi.data());
      }
      if (!got) {
        ok = false;
        return;
      }
      std::uint16_t* d = out + (std::size_t)first[u] * dim;
      if (mlx) {
        const auto* sc = reinterpret_cast<const std::uint16_t*>(s.data());
        const auto* bs = reinterpret_cast<const std::uint16_t*>(bi.data());
        for (int j = 0; j < dim; ++j) {
          const unsigned q = (w[(std::size_t)j / 2] >> (4 * (j & 1))) & 0xFu;
          const float v = bf16_f32_(sc[j / 32]) * (float)q +
                          bf16_f32_(bs[j / 32]);
          d[j] = f32_elt_(v, bf16);
        }
      } else {
        for (int j = 0; j < dim; ++j) {
          d[j] = f32_elt_(lut[w[(std::size_t)j]] *
                              e8m0_(s[(std::size_t)j / 32]), bf16);
        }
      }
    }
  });
  if (!ok) {
    if (err != nullptr) { *err = "an engram row could not be read"; }
    return false;
  }
  parallel_(total, [&](std::size_t lo, std::size_t hi) {
    for (std::size_t k = lo; k < hi; ++k) {
      const std::uint32_t f = first[where[k]];
      if (f == (std::uint32_t)k) { continue; }
      std::memcpy(out + k * (std::size_t)dim, out + (std::size_t)f * dim,
                  (std::size_t)dim * 2);
    }
  });
  return true;
}

bool
MetalDeepseekV41Model::engram_read_(const std::vector<std::int32_t>& hist,
                                    std::size_t hist0, int n,
                                    std::vector<SharedBuffer>& dst,
                                    std::string* err)
{
  const DeepseekV41Config& c = _cfg;
  const int nslots = (int)c.engram_layers.size(), cols = c.engram_cols();
  std::vector<std::int64_t> rows((std::size_t)n * nslots * cols);
  for (int i = 0; i < n; ++i) {
    _engram.hash(std::span<const std::int32_t>(hist.data(),
                                               hist0 + (std::size_t)i + 1),
                 rows.data() + (std::size_t)i * nslots * cols);
  }
  for (int s = 0; s < nslots; ++s) {
    if (!engram_rows_(s, rows, n, dst[(std::size_t)s], err)) { return false; }
  }
  return true;
}

// ---- the forward ---------------------------------------------------------

bool
MetalDeepseekV41Model::forward(CtxId ctx, std::span<const std::int32_t> ids,
                               std::vector<float>* logits, bool all_rows,
                               std::string* err, const Media* media)
{
  Ctx& cx = _ctx[ctx];
  if (logits != nullptr) { logits->clear(); }
  const int first = piece_rows_(_cfg, _opt.max_rows, cx.pos,
                                (int)ids.size());
  // forced picks are for one piece
  if (ids.size() > (std::size_t)first) { _force.clear(); }
  const std::size_t H = (std::size_t)_cfg.dim;
  // A long prefill reads each next piece's engram rows on a worker while
  // the GPU runs the piece before it (cur: this piece's, next: the one
  // being read); `ahead` is waited for on every way out.
  const int nslots = (int)_cfg.engram_layers.size();
  const std::size_t ec =
      (std::size_t)_cfg.engram_cols() * (std::size_t)_cfg.engram_dim * 2;
  EngramPrep cur, next;
  // (VPIPE_DSV41_ENGRAM_CONTEND=1: the next piece's reads started at once,
  // beside this one's -- the A/B)
  static const bool contend =
      std::getenv("VPIPE_DSV41_ENGRAM_CONTEND") != nullptr;
  std::future<bool> ahead;
  struct Join {
    std::future<bool>& f;
    ~Join() {
      if (f.valid()) { f.wait(); }
    }
  } join{ahead};
  std::vector<char> media_at;
  if (media != nullptr && media->rows != nullptr && nslots > 0) {
    media_at.assign(ids.size(), 0);
    for (int p : media->positions) {
      if (p >= 0 && p < (int)ids.size()) { media_at[(std::size_t)p] = 1; }
    }
  }
  std::vector<float> piece;
  std::size_t n = 0;
  for (std::size_t i = 0; i < ids.size(); i += n) {
    n = (std::size_t)piece_rows_(_cfg, _opt.max_rows, cx.pos,
                                 (int)(ids.size() - i));
    const bool last = i + n == ids.size();
    std::vector<float>* lg =
        logits != nullptr && (all_rows || last) ? &piece : nullptr;
    // this piece's media rows, its own positions
    std::vector<int> mpos;
    std::vector<float> mrows;
    if (media != nullptr && media->rows != nullptr) {
      for (std::size_t k = 0; k < media->positions.size(); ++k) {
        const int p = media->positions[k];
        if (p >= (int)i && p < (int)(i + n)) {
          mpos.push_back(p - (int)i);
          mrows.insert(mrows.end(), media->rows + k * H,
                       media->rows + (k + 1) * H);
        }
      }
    }
    // the next piece's engram rows, read while this one runs: hashed over
    // the history's last tokens (all a hash looks back at) and the two
    // pieces' compressed tokens -- started once this piece's own reads are
    // done (after_reads), not to compete with them for the drive
    const std::size_t i2 = i + n;
    std::function<void()> start_ahead;
    if (nslots > 0 && i2 < ids.size() && n >= (std::size_t)kEngramAsyncRows) {
      const int n2 = piece_rows_(_cfg, _opt.max_rows, cx.pos + (int)n,
                                 (int)(ids.size() - i2));
      const std::size_t keep = std::min<std::size_t>(cx.hist.size(), 16);
      auto hist = std::make_shared<std::vector<std::int32_t>>(
          cx.hist.end() - (std::ptrdiff_t)keep, cx.hist.end());
      for (std::size_t k = i; k < i2 + (std::size_t)n2; ++k) {
        hist->push_back(!media_at.empty() && media_at[k] != 0
                            ? DeepseekV41Engram::kDead
                            : _engram.compress(ids[k]));
      }
      next.pos = cx.pos + (int)n;
      next.n = n2;
      next.ok = false;
      next.err.clear();
      // Into the set this piece frees: once forward_piece_ has taken this
      // piece's rows read ahead (cur) into _erow, cur holds the set the
      // piece before ran on -- done with, a piece waiting for its GPU work
      // -- so only a first read-ahead allocates one (two sets beside
      // _erow otherwise; run_bytes books one).
      const bool reuse = !contend && cur.ok && cur.pos == cx.pos &&
                         cur.n == (int)n &&
                         cur.rows.size() == (std::size_t)nslots;
      bool room = true;
      if (!reuse) {
        next.rows.resize((std::size_t)nslots);
        for (auto& b : next.rows) {
          if (b.empty() || b.byte_size() < (std::size_t)n2 * ec) {
            b = _mc->make_shared_buffer((std::size_t)n2 * ec);
          }
          room = room && !b.empty();
        }
      }
      if (room) {
        const std::size_t h0 = keep + n;
        start_ahead = [this, hist, h0, n2, nslots, ec, &cur, &next,
                       &ahead] {
          if (next.rows.size() != (std::size_t)nslots) {
            next.rows.swap(cur.rows);
          }
          for (const auto& b : next.rows) {
            if (b.empty() || b.byte_size() < (std::size_t)n2 * ec) {
              return;   // the next piece reads its own
            }
          }
          ahead = std::async(std::launch::async, [this, hist, h0, n2, &next] {
            next.ok = engram_read_(*hist, h0, n2, next.rows, &next.err);
            return next.ok;
          });
        };
      }
    }
    if (contend && start_ahead) {
      start_ahead();
      start_ahead = nullptr;
    }
    if (!forward_piece_(cx, ids.subspan(i, n), lg, all_rows, err, mpos,
                        mrows.data(), &cur,
                        start_ahead ? &start_ahead : nullptr)) {
      return false;
    }
    if (lg != nullptr) {
      logits->insert(logits->end(), piece.begin(), piece.end());
    }
    if (ahead.valid()) {
      ahead.wait();
      ahead = std::future<bool>();
      std::swap(cur, next);
    } else {
      cur.ok = false;
    }
  }
  _force.clear();
  return true;
}

bool
MetalDeepseekV41Model::forward_piece_(
    Ctx& cx, std::span<const std::int32_t> ids, std::vector<float>* logits,
    bool all_rows, std::string* err, const std::vector<int>& mpos,
    const float* mrows, EngramPrep* prep,
    const std::function<void()>* after_reads)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[deepseek-v4.1] " + std::move(m); }
    _hccnt_dirty = true;
    return false;
  };
  const DeepseekV41Config& c = _cfg;
  int n = (int)ids.size();      // the tail's, past a CED cut
  if (n == 0) { return true; }
  int p0 = cx.pos;
  const int end = p0 + n;
  for (std::int32_t t : ids) {
    if (t < 0 || t >= c.vocab) { return fail("token id out of range"); }
  }
  std::string e;
  if (!ensure_ctx_(cx, end, &e)) { return fail(e); }
  if (!ensure_scratch_(n, end)) { return fail("out of memory for scratch"); }
  if (!ensure_rope_(end + 1)) { return fail("out of memory for rotary"); }
  const int H = c.dim, D = c.head_dim, NH = c.n_heads, QL = c.q_lora;
  const int OG = c.o_groups, OL = c.o_lora;
  const int I = c.moe_inter, E = c.n_experts, K = c.top_k;
  const int IH = c.index_heads, ID = c.index_dim, W = c.window;
  const int nslots = (int)c.engram_layers.size();

  // ---- media rows and masks (host) ----------------------------------------
  const int nm = (int)mpos.size();
  std::vector<char> is_media((std::size_t)n, 0);
  for (int p : mpos) {
    if (p < 0 || p >= n) { return fail("a media row outside the chunk"); }
    is_media[(std::size_t)p] = 1;
  }
  if (nm > 0) {
    if (_mrows.byte_size() < (std::size_t)nm * H * 2) {
      _mrows = _mc->make_shared_buffer((std::size_t)nm * H * 2);
      _mpos = _mc->make_shared_buffer((std::size_t)nm * 4);
    }
    if (_tmask.byte_size() < (std::size_t)n) {
      _tmask = _mc->make_shared_buffer((std::size_t)n + 16);
      _imask = _mc->make_shared_buffer((std::size_t)n + 16);
    }
    if (_mrows.empty() || _mpos.empty() || _tmask.empty() || _imask.empty()) {
      return fail("out of memory for media rows");
    }
    auto* d = static_cast<std::uint16_t*>(_mrows.contents());
    for (std::size_t k = 0; k < (std::size_t)nm * H; ++k) {
      d[k] = f32_elt_(mrows[k], _bf16);
    }
    std::memcpy(_mpos.contents(), mpos.data(), (std::size_t)nm * 4);
    auto* tm = static_cast<std::uint8_t*>(_tmask.contents());
    auto* im = static_cast<std::uint8_t*>(_imask.contents());
    for (int i = 0; i < n; ++i) {
      tm[i] = is_media[(std::size_t)i] ? 0 : 1;
      im[i] = is_media[(std::size_t)i] ? 1 : 0;
    }
  }

  // VPIPE_DSV41_PROFILE=1: where a forward's time goes, on stderr.
  static const int prof_level = [] {
    const char* e = std::getenv("VPIPE_DSV41_PROFILE");
    return e != nullptr ? std::max(1, std::atoi(e)) : 0;
  }();
  const bool prof = prof_level > 0;
  // Level 2 commits and waits at each section's end, so the GPU's time is
  // attributed to it (and the run is slower by the waits).
  enum { kHc, kAttnIn, kIndex, kAttn, kAttnOut, kRoute, kShared, kExperts,
         kSections };
  static const char* kSec[kSections] = {
      "hc+norm", "q/kv proj", "compress+index", "attention", "out proj",
      "router", "shared expert", "routed experts"};
  double t_sec[kSections] = {};
  using Clk = std::chrono::steady_clock;
  auto ms = [](Clk::time_point a, Clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const auto t_start = Clk::now();
  double t_gpu = 0, t_cache = 0;

  // ---- engram rows (host) -------------------------------------------------
  // A long prefill's piece has them read ahead (forward(): `prep`, on a
  // worker while the piece before it ran); else they are read here, a
  // many-row forward's on one worker, slot by slot, each waited for at its
  // layer (the GPU runs the layers before it meanwhile). `after_reads`
  // then runs -- forward() starts the next piece's reads there, so they do
  // not compete with this piece's for the drive (MEASURED on an M5 Ultra:
  // a prompt's first 8192 rows waited 2.3 s for their first slot beside
  // the next piece's reads, 1.2 s alone).
  const std::size_t hist0 = cx.hist.size();
  auto t_ehash = Clk::now();
  for (int i = 0; i < n; ++i) {
    cx.hist.push_back(is_media[(std::size_t)i]
                          ? DeepseekV41Engram::kDead
                          : _engram.compress(ids[(std::size_t)i]));
  }
  std::vector<std::int64_t> erows;
  std::vector<std::shared_future<bool>> eready((std::size_t)nslots);
  std::vector<std::string> eerr((std::size_t)nslots);
  std::future<void> etask;
  struct WaitTask {
    std::future<void>& f;
    ~WaitTask() {
      if (f.valid()) { f.wait(); }
    }
  } wait_task{etask};
  bool reads_done = true;
  if (nslots > 0) {
    if (prep != nullptr && prep->ok && prep->pos == p0 && prep->n == n &&
        prep->rows.size() == (std::size_t)nslots) {
      for (int s = 0; s < nslots; ++s) {
        std::swap(_erow[(std::size_t)s], prep->rows[(std::size_t)s]);
      }
      ++_engram_ahead;
    } else {
      const int cols = c.engram_cols();
      erows.resize((std::size_t)n * nslots * cols);
      for (int i = 0; i < n; ++i) {
        _engram.hash(std::span<const std::int32_t>(
                         cx.hist.data(), hist0 + (std::size_t)i + 1),
                     erows.data() + (std::size_t)i * nslots * cols);
      }
      t_ehash = Clk::now();
      if (n >= kEngramAsyncRows) {
        auto pr = std::make_shared<std::vector<std::promise<bool>>>(
            (std::size_t)nslots);
        for (int s = 0; s < nslots; ++s) {
          eready[(std::size_t)s] = (*pr)[(std::size_t)s].get_future().share();
        }
        reads_done = false;
        etask = std::async(std::launch::async,
                           [this, pr, &erows, n, &eerr, after_reads, nslots] {
          for (int s = 0; s < nslots; ++s) {
            const bool ok = engram_rows_(s, erows, n, _erow[(std::size_t)s],
                                         &eerr[(std::size_t)s]);
            (*pr)[(std::size_t)s].set_value(ok);
            if (!ok) {
              for (int t = s + 1; t < nslots; ++t) {
                (*pr)[(std::size_t)t].set_value(false);
              }
              return;
            }
          }
          if (after_reads != nullptr) { (*after_reads)(); }
        });
      } else {
        for (int s = 0; s < nslots; ++s) {
          if (!engram_rows_(s, erows, n, _erow[(std::size_t)s], &e)) {
            cx.hist.resize(hist0);
            return fail(e);
          }
        }
      }
    }
  }
  if (reads_done && after_reads != nullptr) { (*after_reads)(); }
  const auto t_engram = Clk::now();
  std::memcpy(_ids.contents(), ids.data(), (std::size_t)n * 4);
  {
    auto* pm = static_cast<float*>(_pm[0].contents());
    for (int i = 0; i < n; ++i) {
      pm[i * 4] = 1.0f;
      pm[i * 4 + 1] = pm[i * 4 + 2] = pm[i * 4 + 3] = 0.0f;
    }
  }
  _routes.assign((std::size_t)c.n_layers, {});
  _captured.clear();
  const bool spark = !_stages.empty();


  // one stream for the model's life: a new one is a new MTLCommandQueue,
  // whose first commit waits 1-5 ms before the GPU starts it (4.8 ms on
  // an M5 Ultra, every forward)
  if (!_stream.valid()) { _stream = _mc->make_command_stream(); }
  CommandStream& stream = _stream;
  ComputeEncoder enc = stream.begin_compute();
  if (_hccnt_dirty) {
    // a forward failed with a collapse's count open: set them back first
    fill_(enc, _hccnt, 0, _hccnt.byte_size() / 4);
    _hccnt_dirty = false;
  }
  long n_disp = 0;                // dispatches (VPIPE_DSV41_PROFILE)
  auto t_mark = Clk::now();
  double g_sec[kSections] = {};
  auto mark = [&](int sec) {
    if (prof_level < 2) { return; }
    // the section's own GPU time too (run with VPIPE_MC_CMDBUF_SPLIT=0,
    // so it is one buffer), apart from the wait's overhead
    n_disp += enc.dispatch_count();
    enc.end();
    auto f = stream.commit();
    std::string w;
    (void)f.wait_ok(&w);
    g_sec[sec] += f.gpu_times().gpu_s * 1e3;
    enc = stream.begin_compute();
    const auto now = Clk::now();
    t_sec[sec] += ms(t_mark, now);
    t_mark = now;
  };
  enc.set_function(_fn_embed);
  enc.set_buffer(0, _embed);
  enc.set_buffer(1, _ids);
  enc.set_buffer(2, _X);
  enc.set_constant(3, H);
  enc.set_constant(4, c.hc);
  enc.dispatch({(unsigned)H, (unsigned)n, 1}, {256, 1, 1});
  if (nm > 0) {
    enc.set_function(_fn_rows_at);
    enc.set_buffer(0, _mrows);
    enc.set_buffer(1, _mpos);
    enc.set_buffer(2, _X);
    enc.set_constant(3, H);
    enc.dispatch({(unsigned)H, (unsigned)nm, 1}, {256, 1, 1});
  }

  auto hc_mix = [&](const SharedBuffer& fn, const SharedBuffer& sc,
                    const SharedBuffer& bs, SharedBuffer* mix) {
    hc_mix_(enc, _X, n, fn, sc, bs, mix);
  };
  auto hc_pre = [&](const SharedBuffer& pre, const SharedBuffer& w,
                    const SharedBuffer& y) {
    hc_pre_(enc, _X, n, pre, &w, y);
  };
  auto hc_post = [&](const SharedBuffer& a, SharedBuffer* mix) {
    hc_post_(enc, a, _X, n, mix);
  };
  // A sublayer's post (its branch into the copies) waits for the next
  // collapse, which takes it in the same pass (dsv41_hc_post_pre, the same
  // bits; VPIPE_DSV41_HC_FUSE=0 runs the two); anything else that reads the
  // copies first has it flushed.
  const SharedBuffer* post_a = nullptr;
  SharedBuffer* post_mix = nullptr;
  // the MoE's branch is its K routed parts and the shared expert's output,
  // summed by dsv41_moe_combine -- or, fused, by the collapse that takes
  // the post (dsv41_hc_post_pre_moe, the same sum): one dispatch less
  bool post_moe = false;
  auto combine = [&] {
    enc.set_function(_fn_combine);
    enc.set_buffer(0, _part);
    enc.set_buffer(1, _shout);
    enc.set_buffer(2, _fo);
    enc.set_constant(3, H);
    enc.set_constant(4, K);
    enc.dispatch({(unsigned)H, (unsigned)n, 1}, {256, 1, 1});
  };
  auto defer_post = [&](const SharedBuffer& a, SharedBuffer* mix) {
    post_a = &a;
    post_mix = mix;
    post_moe = false;
    if (!_hc_fuse) {
      hc_post(a, mix);
      post_a = nullptr;
    }
  };
  auto defer_post_moe = [&](SharedBuffer* mix) {
    if (!_hc_fuse) {
      combine();
      hc_post(_fo, mix);
      return;
    }
    post_a = &_fo;
    post_mix = mix;
    post_moe = true;
  };
  auto flush_post = [&] {
    if (post_a != nullptr) {
      if (post_moe) { combine(); }
      hc_post(*post_a, post_mix);
      post_a = nullptr;
      post_moe = false;
    }
  };
  // the copies collapsed with `pre` and normed by `w` into y, the post
  // waiting before it taken in the same pass (its mix's pre is `pre`)
  auto collapse = [&](const SharedBuffer& pre, const SharedBuffer& w,
                      const SharedBuffer& y) {
    if (post_a == nullptr || &pre != &post_mix[0]) {
      flush_post();
      hc_pre(pre, w, y);
      return;
    }
    hc_coll_(enc, _X, n, post_mix[0], &w, y, post_moe ? &_part : post_a,
             post_mix, post_moe ? &_shout : nullptr, post_moe ? K : 0);
    post_a = nullptr;
    post_moe = false;
  };

  // the logits' buffer first: nothing may return early once a turn waits
  const int rows = all_rows ? n : 1;
  int r0 = n - rows;
  if (logits != nullptr &&
      _logits.byte_size() < (std::size_t)rows * c.vocab * 4) {
    _logits = _mc->make_shared_buffer((std::size_t)rows * c.vocab * 4);
    if (_logits.empty()) { return fail("out of memory for logits"); }
  }
  // ---- the handshake (_fn_hs): a decode or verify step's resolves -----
  const bool hs = _hs_ok && !_hs_redo && n <= kKeepRows && _force.empty() &&
                  _capture < 0 && prof_level < 2 && _study.k == 0;
  const bool study = _study.k > 0 && n == 1;
  // a few rows' independent dispatches side by side (VPIPE_DSV41_CONCURRENT)
  const bool conc = _concurrent && n <= kKeepRows && prof_level < 2;
  // a decode step's route inside the handshake's dispatch, by its whole
  // group (VPIPE_DSV41_HS_ROUTE); a verify's rows keep the route's own,
  // a group a row side by side
  const bool fold_route = hs && _hs_route && n == 1;
  // ---- CED: the decoder on the piece's last window ---------------------
  // The README's causal encoder-decoder: the decoder's global KV (layer
  // cmid's latents and index keys) is projected from the encoder's last
  // hidden states, so a prefill needs the decoder only for the rows the
  // decode attends -- its last `window` ("SWA bounded replay"), and
  // kKeepRows more, so that a rollback into the prompt (truncate) still
  // finds a whole window. At the first decoder layer: its collapse,
  // latents and keys over every row, then the copies and the collapse cut
  // to the tail; every later step on the tail, each decoder window from
  // the tail's start (nothing before it was computed there). The encoder,
  // the global KV and the index keys are the whole stack's bits; the
  // decoder's first tail rows see shorter windows. VPIPE_DSV41_CED=0 /
  // Options::ced runs the whole stack.
  const int cmid = _ced_mid, T = W + kKeepRows;
  const bool ced = _ced && cmid >= 0 && n > T && !all_rows && nm == 0 &&
                   _force.empty() && _capture < cmid && !hs && !study;
  if (ced) { ++_ced_pieces; }
  struct HsTurn { int L, aw; std::uint32_t seq; };
  std::vector<HsTurn> turns;
  std::vector<CommandStream::Fence> fences;
  std::size_t served = 0;
  bool hs_lost = false;
  std::string hs_err;
  auto* hw = static_cast<std::uint32_t*>(_hsf.contents());
  auto word = [hw](int k) { return std::atomic_ref<std::uint32_t>(hw[k]); };
  // the compressors' open groups (a ring of `ratio` rows each), kept on the
  // host: a forward the GPU gave up on may have written over the row
  // before it, and its redo needs that row back
  std::vector<std::vector<std::uint8_t>> pend0;
  if (hs) {
    hw[0] = hw[1] = hw[2] = 0;   // the GPU reads them from the next commit
    pend0.resize((std::size_t)c.n_layers);
    for (int L = 0; L < c.n_layers; ++L) {
      const SharedBuffer& b = cx.L[(std::size_t)L].pend;
      if (b.empty()) { continue; }
      const auto* q = static_cast<const std::uint8_t*>(b.contents());
      pend0[(std::size_t)L].assign(q, q + b.byte_size());
    }
    hs_beating_(true);
  }
  int np = n * K;
  // a verify's few rows read each expert once for all its pairs (not under
  // the handshake: its grid is encoded before the picks are known)
  const bool rows_groups = !hs && n > 1 && n <= kKeepRows && _rows_mm > 0;
  const int rk = _rows_mm >= 8 ? 2 : _rows_mm >= 4 ? 1 : 0;
  int ngroups = 0;
  // The host's part of a layer once its router has run: fp32's picks if
  // forced, the routes kept, the next layer's likeliest experts ranked,
  // and the cache's resolve (reading what it lacks) into the slots.
  auto service = [&](int L, int aw, std::string& swhy,
                     const std::int32_t* eid, const float* ahead,
                     std::int32_t* slot) -> bool {
    if ((std::size_t)L < _force.size() &&
        _force[(std::size_t)L].size() == (std::size_t)np) {
      force_picks_(_force[(std::size_t)L], n, E, K);
    }
    {
      _routes[(std::size_t)L].assign(eid, eid + np);
      MoeExpertCache::Ahead ah;
      if (aw > 0) {
        // each row's likeliest by the biased score, as the router picks;
        // the rows' firsts, then their seconds, ..., each expert once
        const Layer& nx = _layers[(std::size_t)L + 1];
        const auto* bias = static_cast<const float*>(nx.gate_b.contents());
        std::vector<float> v((std::size_t)E);
        std::vector<std::int32_t> top((std::size_t)(n * aw)), ord(E);
        for (int t = 0; t < n; ++t) {
          const float* sc = ahead + (std::size_t)t * E;
          for (int e = 0; e < E; ++e) {
            const float x = sc[e];
            v[(std::size_t)e] =
                std::sqrt(x > 20.0f ? x : std::log1p(std::exp(x))) +
                bias[e];
            ord[(std::size_t)e] = e;
          }
          std::partial_sort(ord.begin(), ord.begin() + aw, ord.end(),
                            [&](std::int32_t a, std::int32_t b) {
                              return v[(std::size_t)a] > v[(std::size_t)b];
                            });
          for (int k = 0; k < aw; ++k) {
            top[(std::size_t)(k * n + t)] = ord[(std::size_t)k];
          }
        }
        int na = 0;
        for (std::int32_t e : top) {
          if (std::find(_ahead_rank.begin(), _ahead_rank.begin() + na, e) ==
              _ahead_rank.begin() + na) {
            _ahead_rank[(std::size_t)na++] = e;
          }
        }
        ah = {L + 1, _ahead_rank.data(), na};
      }
      if (!_cache->resolve(L, eid, np, n, slot, &swhy,
                           aw > 0 ? &ah : nullptr)) {
        std::fill(slot, slot + np, 0);     // in bounds, at least
        return false;
      }
      if (study) {
        _study.picks[(std::size_t)L].assign(eid, eid + np);
        _study.blocked[(std::size_t)L] = _cache->last_blocked();
        std::memcpy(_study.cur.data() + (std::size_t)L * E, _sc.contents(),
                    (std::size_t)E * 4);
        study_name_(L);
      }
      if (rows_groups) {
        // the pairs by slot, in first-pick order, _rows_mm a group
        auto* g = static_cast<std::int32_t*>(_mgrp.contents());
        ngroups = 0;
        for (int q = 0; q < np; ++q) {
          int at = 0;
          while (at < ngroups && (g[at * 10] != slot[q] ||
                                  g[at * 10 + 1] >= _rows_mm)) {
            ++at;
          }
          if (at == ngroups) {
            g[at * 10] = slot[q];
            g[at * 10 + 1] = 0;
            ++ngroups;
          }
          g[at * 10 + 2 + g[at * 10 + 1]++] = q;
        }
      }
    }
    return true;
  };
  // Answer the turns the GPU has posted, and wait for (and answer) each
  // up to layer `upto`. A wait that sees the GPU give up (or a lost post)
  // releases every later one; the forward is then redone with commits.
  auto serve = [&](int upto) {
    while (served < turns.size() && !hs_lost) {
      const HsTurn& h = turns[served];
      if (word(0).load(std::memory_order_acquire) < h.seq) {
        if (h.L > upto) { return; }
        const auto t0 = Clk::now();
        while (word(0).load(std::memory_order_acquire) < h.seq) {
          if (word(2).load(std::memory_order_relaxed) != 0 ||
              Clk::now() - t0 > std::chrono::seconds(30)) {
            hs_lost = true;
            word(1).store(~0u, std::memory_order_release);
            return;
          }
        }
        t_gpu += ms(t0, Clk::now());
      }
      if (h.L == _hs_drop) {
        // debug: this turn abandoned, as by a host that is gone
        _hs_drop = -1;
        word(2).store(1, std::memory_order_relaxed);
        hs_lost = true;
        word(1).store(~0u, std::memory_order_release);
        return;
      }
      const auto t1 = Clk::now();
      const auto* bo = static_cast<const std::int32_t*>(_hsbo.contents());
      auto* bi = static_cast<std::int32_t*>(_hsbi.contents());
      // the GPU's words, once their checksum holds (they may land after
      // the post that announces them)
      const std::uint32_t no =
          (std::uint32_t)(np + (h.aw > 0 ? n * E : 0));
      const auto* bw = reinterpret_cast<const std::uint32_t*>(bo);
      while (hs_sum_(bw, no, h.seq) != load_now_(bw[no])) {
        if (word(2).load(std::memory_order_relaxed) != 0 ||
            Clk::now() - t1 > std::chrono::seconds(1)) {
          hs_lost = true;
          word(1).store(~0u, std::memory_order_release);
          return;
        }
      }
      std::string why;
      if (!service(h.L, h.aw, why, bo,
                   reinterpret_cast<const float*>(bo + np), bi)) {
        if (hs_err.empty()) { hs_err = "expert cache: " + why; }
        std::fill(bi, bi + np, 0);     // in bounds, at least
      }
      auto* biw = reinterpret_cast<std::uint32_t*>(bi);
      biw[np] = hs_sum_(biw, (std::uint32_t)np, h.seq);
      word(1).store(h.seq, std::memory_order_release);
      t_cache += ms(t1, Clk::now());
      ++served;
    }
  };

  // Layer 0's attention collapses the copies with the identity mix; every
  // later sublayer with the mix the sublayer before it computed (the
  // previous FFN's `pre` for an attention, the attention's for its FFN).
  for (int L = 0; L < c.n_layers; ++L) {
    const Layer& ly = _layers[(std::size_t)L];
    const DeepseekV41Config::Plan& pl = c.plan[(std::size_t)L];
    LayerState& st = cx.L[(std::size_t)L];
    // ---- engram --------------------------------------------------------
    const int es = c.engram_slot(L);
    if (es >= 0) {
      if (auto& f = eready[(std::size_t)es]; f.valid() && !f.get()) {
        cx.hist.resize(hist0);
        return fail(eerr[(std::size_t)es]);
      }
      flush_post();
      gemm_p_(enc, ly.e_wkv, _erow[(std::size_t)es], 0, _ekv, 0, n);
      enc.set_function(_fn_egate);
      enc.set_buffer(0, _X);
      enc.set_buffer(1, _ekv);
      enc.set_buffer(2, ly.e_qk);
      enc.set_buffer(3, nm > 0 ? _tmask : _ekv);
      enc.set_constant(4, H);
      enc.set_constant(5, c.eps);
      enc.set_constant(6, nm > 0 ? 1 : 0);
      enc.dispatch({(unsigned)(256 * n), 1, 1}, {256, 1, 1});
    }
    // ---- DSpark's main hidden: a target layer's input, its copies' mean
    if (spark) {
      for (std::size_t k = 0; k < c.dspark_targets.size(); ++k) {
        if (c.dspark_targets[k] != L) { continue; }
        flush_post();
        enc.set_function(_fn_hc_mean);
        enc.set_buffer(0, _X);
        enc.set_buffer(1, _mh);
        enc.set_constant(2, H);
        enc.set_constant(3, (int)c.dspark_targets.size() * H);
        enc.set_constant(4, (int)k * H);
        enc.dispatch({(unsigned)H, (unsigned)n, 1}, {256, 1, 1});
      }
    }
    // ---- attention -------------------------------------------------------
    mark(kExperts);
    const bool comp = pl.ratio > 0;
    // a few rows' independent dispatches side by side (each group's
    // outputs disjoint, read after it): the compressor's and the indexer's
    // projections of the collapse go with the q / kv chains
    const bool cw_early = conc && comp && pl.kv_source && pl.ratio > 1;
    const bool ix_early = conc && comp && pl.index_source;
    // A decode or verify step's few rows (at most the ring's slack): the
    // latent normed, rotated and written into the window ring in one
    // dispatch (dsv41_kv_norm_rope, ahead of the attention, which then
    // needs no ring write after it), and q's rotation done by the split
    // attention as it reads q -- two dispatches fewer a layer, the same
    // bits. A longer forward keeps the three kernels.
    // The compressor: latents for the groups this chunk completes (a kv
    // source), with the indexer's keys where it owns them.
    auto source_kv = [&] {
      const int r = pl.ratio;
      // The compressor: latents for the groups this chunk completes.
      const int g0 = p0 / r, g1 = end / r, ng = g1 - g0;
      if (r > 1) {
        if (!cw_early) {
          gemm_f32_(enc, ly.c_w, _xa, 0, _ks, 0, H, 2 * D, n);
        }
        {
          // the last rows' inputs by position, for a rollback's open group
          // (truncate): f32 rows of 2D as 4D 16-bit elements
          const int kr = kKeepRows + r - 1, kn = std::min(n, kr);
          enc.set_function(_fn_ring);
          enc.set_buffer(0, _ks, (std::size_t)(n - kn) * 2 * D * 4);
          enc.set_buffer(1, st.keep);
          enc.set_constant(2, kn);
          enc.set_constant(3, 4 * D);
          enc.set_constant(4, kr);
          enc.set_constant(5, end - kn);
          enc.dispatch({(unsigned)(4 * D), (unsigned)kn, 1}, {256, 1, 1});
        }
        if (ng > 0) {
          enc.set_function(_fn_pool);
          enc.set_buffer(0, _ks);
          enc.set_buffer(1, st.pend);
          enc.set_buffer(2, _pooled);
          enc.set_constant(3, D);
          enc.set_constant(4, r);
          enc.set_constant(5, p0);
          enc.set_constant(6, g0);
          enc.dispatch({(unsigned)D, (unsigned)ng, 1}, {256, 1, 1});
        }
        const int q0 = g1 * r;
        if (q0 < end) {
          enc.set_function(_fn_keep);
          enc.set_buffer(0, _ks);
          enc.set_buffer(1, st.pend);
          enc.set_constant(2, 2 * D);
          enc.set_constant(3, r);
          enc.set_constant(4, p0);
          enc.set_constant(5, q0);
          enc.dispatch({(unsigned)(2 * D), (unsigned)(end - q0), 1},
                       {256, 1, 1});
        }
        if (ng > 0) {
          rms_(enc, _pooled, 0, ly.c_norm, st.comp,
               (std::size_t)g0 * D * 2, D, ng, D, D, true);
        }
      } else {
        gemm_(enc, ly.c_w, _xa, 0, _kvr, 0, H, D, n);
        rms_(enc, _kvr, 0, ly.c_norm, st.comp, (std::size_t)g0 * D * 2, D,
             n, D, D, false);
      }
      if (ng > 0) {
        if (pl.owns_index_keys) {
          // The indexer's keys come from the latent BEFORE its rotary.
          gemm_(enc, ly.i_wk, st.comp, (std::size_t)g0 * D * 2, _kk, 0,
                D, ID, ng);
          rms_(enc, _kk, 0, ly.i_knorm, st.keys, (std::size_t)g0 * ID * 2,
               ID, ng, ID, ID, false);
          rope_(enc, st.keys, (std::size_t)g0 * ID * 2, true, ng, 1, ID,
                ID, ID, g0 * r, r, false);
        }
        // the cached latent rotated at its group's first position
        rope_(enc, st.comp, (std::size_t)g0 * D * 2, true, ng, 1, D, D, D,
              g0 * r, r, false);
      }
    };
    // The CED cut (above): the copies [n, 4, H] and the collapse [n, H]
    // cut to their last T rows, through _part (the collapse took the
    // MoE's partials), and the forward's rows those.
    auto cut = [&] {
      const int off = n - T;
      const int xw = T * 4 * H / 2, aw2 = T * H / 2;   // words
      auto copy = [&](const SharedBuffer& a, std::size_t ao,
                      const SharedBuffer& b, std::size_t bo, int words) {
        enc.set_function(_fn_copy32);
        enc.set_buffer(0, a, ao);
        enc.set_buffer(1, b, bo);
        enc.set_constant(2, words);
        enc.dispatch({(unsigned)words, 1, 1}, {256, 1, 1});
      };
      copy(_X, (std::size_t)off * 4 * H * 2, _part, 0, xw);
      copy(_xa, (std::size_t)off * H * 2, _part, (std::size_t)xw * 4, aw2);
      copy(_part, 0, _X, 0, xw);
      copy(_part, (std::size_t)xw * 4, _xa, 0, aw2);
      if (spark) {
        // DSpark's taps so far (a target at or before this layer): their
        // rows cut the same way, as the commit after the stack reads them
        const int nt = (int)c.dspark_targets.size();
        const int mw = T * nt * H / 2;
        copy(_mh, (std::size_t)off * nt * H * 2, _part, 0, mw);
        copy(_part, 0, _mh, 0, mw);
      }
      n = T;
      p0 = end - T;
      np = n * K;
      r0 = n - rows;
    };
    const bool few = n <= kKeepRows;
    const bool rope_in_attn = few && _split_attn;
    auto kv_tail = [&] {
      if (!few) {
        rms_(enc, _kvr, 0, ly.kv_norm, _kv, 0, D, n, D, D, false);
        rope_(enc, _kv, 0, comp, n, 1, D, D, D, p0, 1, false);
        return;
      }
      enc.set_function(_fn_kv_norm_rope);
      enc.set_buffer(0, _kvr);
      enc.set_buffer(1, ly.kv_norm);
      enc.set_buffer(2, _kv);
      enc.set_buffer(3, comp ? _rope_comp : _rope_plain);
      enc.set_buffer(4, st.ring);
      enc.set_constant(5, D);
      enc.set_constant(6, c.eps);
      enc.set_constant(7, c.rope_dim);
      enc.set_constant(8, p0);
      enc.set_constant(9, W + kRingSlack);
      enc.set_constant(10, 1);
      enc.dispatch({(unsigned)(256 * n), 1, 1}, {256, 1, 1});
    };
    auto rope_q = [&] {
      if (!rope_in_attn) {
        rope_(enc, _q, 0, comp, n, NH, NH * D, D, D, p0, 1, false);
      }
    };
    if (conc) {
      // what reads the collapse: the projections, beside the mix's slices
      // (after a fused post they read the copies it wrote); the mix's
      // finisher waits for wo_a, beside which it runs (its partials are
      // not written again before the FFN's slices)
      auto reads_xa = [&] {
        gemm_p_(enc, ly.wq_a, _xa, 0, _qa, 0, n);
        gemm_p_(enc, ly.wkv, _xa, 0, _kvr, 0, n);
        if (cw_early) { gemm_f32_(enc, ly.c_w, _xa, 0, _ks, 0, H, 2 * D, n); }
        if (ix_early) { gemm_(enc, ly.i_wp, _xa, 0, _wp, 0, H, IH, n); }
      };
      if (post_a != nullptr) {
        collapse(_mixF[0], ly.attn_norm, _xa);
        {
          auto cs = enc.concurrent_scope();
          hc_mix_part_(enc, _X, n, ly.ha_fn);
          reads_xa();
        }
      } else {
        {
          auto cs = enc.concurrent_scope();
          hc_mix_part_(enc, _X, n, ly.ha_fn);
          hc_pre(L == 0 ? _pm[0] : _mixF[0], ly.attn_norm, _xa);
        }
        {
          auto cs = enc.concurrent_scope();
          reads_xa();
        }
      }
      {
        auto cs = enc.concurrent_scope();
        rms_(enc, _qa, 0, ly.q_norm, _qr, 0, QL, n, QL, QL, false);
        kv_tail();
      }
      {
        auto cs = enc.concurrent_scope();
        gemm_p_(enc, ly.wq_b, _qr, 0, _q, 0, n);
        if (ix_early) { gemm_p_(enc, ly.i_wq_b, _qr, 0, _qi, 0, n); }
      }
      rope_q();
    } else {
      collapse(L == 0 ? _pm[0] : _mixF[0], ly.attn_norm, _xa);
      if (ced && L == cmid) {
        source_kv();       // over every row, from the encoder's output
        cut();
      }
      hc_mix(ly.ha_fn, ly.ha_scale, ly.ha_base, _mixA);
      mark(kHc);
      gemm_p_(enc, ly.wq_a, _xa, 0, _qa, 0, n);
      rms_(enc, _qa, 0, ly.q_norm, _qr, 0, QL, n, QL, QL, false);
      gemm_p_(enc, ly.wq_b, _qr, 0, _q, 0, n);
      rope_q();
      gemm_p_(enc, ly.wkv, _xa, 0, _kvr, 0, n);
      kv_tail();
    }
    mark(kAttnIn);
    int KC = 0;
    const SharedBuffer* cbuf = &st.ring;      // unused unless comp
    if (comp) {
      const int r = pl.ratio;
      if (pl.kv_source && !(ced && L == cmid)) { source_kv(); }
      LayerState& src = cx.L[(std::size_t)pl.kv_from];
      cbuf = &src.comp;
      if (pl.index_source) {
        const int C = end / r;
        LayerState& ksrc = cx.L[(std::size_t)pl.keys_from];
        if (!ix_early) {
          gemm_p_(enc, ly.i_wq_b, _qr, 0, _qi, 0, n);
        }
        rope_(enc, _qi, 0, true, n, IH, IH * ID, ID, ID, p0, 1, false);
        if (!ix_early) { gemm_(enc, ly.i_wp, _xa, 0, _wp, 0, H, IH, n); }
        const int blk = std::max(c.candidate_block, 1);
        const int nblk = (C + blk - 1) / blk;
        const int KB = c.candidate_blocks, KT = c.index_topk;
        const bool kept = C > 0 && pl.uses_candidates && _index_kept &&
                          _index_mma && nblk > KB;
        // The scores [rows, width] f32 a chunk of rows at a time, each
        // chunk's under kScoreBudget (DeepInfer's row chunks): a piece's
        // rows do not follow the context's length. Rows are scored alone,
        // so the picks are the same however they are chunked. The kept
        // blocks (_bpick) and their mask (_cmask) are every row's: later
        // layers read them.
        int SR = score_rows_(n, kept ? KB * blk : C);
        if (_opt.score_rows > 0) {
          SR = std::min(SR, std::max(8, _opt.score_rows / 8 * 8));
        }
        const float wscale =
            (float)(1.0 / std::sqrt((double)ID) / std::sqrt((double)IH));
        if (C > 0 && !kept && pl.candidate_source) {
          fill_(enc, _cmask, 0, ((std::size_t)n * nblk + 3) / 4);
        }
        for (int i0 = 0; i0 < n && C > 0; i0 += SR) {
          const int m = std::min(SR, n - i0);
          if (i0 > 0) { ++_score_chunks_n; }
          const std::size_t oq = (std::size_t)i0 * IH * ID * 2;
          const std::size_t ow = (std::size_t)i0 * IH * 2;
          const std::size_t ob = (std::size_t)i0 * KB * 4;
          const std::size_t oi = (std::size_t)i0 * KT * 4;
          const std::size_t oc = (std::size_t)i0 * nblk;   // SR % 8 == 0
          if (kept) {
            // only the kept blocks' positions, compacted in position order
            // into _S, then the top-k's slots back to positions
            ++_kept_passes;
            enc.set_function(_fn_iscore_kept);
            enc.set_buffer(0, _qi, oq);
            enc.set_buffer(1, ksrc.keys);
            enc.set_buffer(2, _wp, ow);
            enc.set_buffer(3, _bpick, ob);
            enc.set_buffer(4, _S);
            enc.set_constant(5, IH);
            enc.set_constant(6, C);
            enc.set_constant(7, p0 + i0);
            enc.set_constant(8, r);
            enc.set_constant(9, wscale);
            enc.set_constant(10, KB);
            enc.set_constant(11, blk);
            enc.dispatch({(unsigned)(128 * ((KB * blk + 63) / 64)),
                          (unsigned)m, 1}, {128, 1, 1});
            topk_(enc, _S, m, KB * blk, KT, _idx, oi);
            enc.set_function(_fn_iunpack);
            enc.set_buffer(0, _idx, oi);
            enc.set_buffer(1, _bpick, ob);
            enc.set_constant(2, KT);
            enc.set_constant(3, KB);
            enc.set_constant(4, blk);
            enc.dispatch({(unsigned)KT, (unsigned)m, 1}, {64, 1, 1});
            continue;
          }
          // (a prefill's rows: 4 a key tile; a decode or verify step's
          // few keep the simdgroup matrices, a row a tile)
          const bool inax = _index_nax && _index_mma && n > kKeepRows;
          if (inax) { ++_nax_index; }
          enc.set_function(inax    ? _fn_iscore_nax
                           : _index_mma ? _fn_iscore_mma
                                        : _fn_iscore);
          enc.set_buffer(0, _qi, oq);
          enc.set_buffer(1, ksrc.keys);
          enc.set_buffer(2, _wp, ow);
          enc.set_buffer(3, _cmask, oc);
          enc.set_buffer(4, _S);
          enc.set_constant(5, IH);
          enc.set_constant(6, C);
          enc.set_constant(7, p0 + i0);
          enc.set_constant(8, r);
          enc.set_constant(9, wscale);
          enc.set_constant(10, pl.uses_candidates ? 1 : 0);
          enc.set_constant(11, blk);
          enc.set_constant(12, nblk);
          if (inax) {
            enc.set_constant(13, m);
            enc.dispatch({(unsigned)(256 * ((C + 63) / 64)),
                          (unsigned)((m + 3) / 4), 1}, {256, 1, 1});
          } else if (_index_mma) {
            enc.dispatch({(unsigned)(128 * ((C + 63) / 64)), (unsigned)m, 1},
                         {128, 1, 1});
          } else {
            enc.dispatch({(unsigned)(256 * ((C + 255) / 256)), (unsigned)m,
                          1}, {256, 1, 1});
          }
          if (pl.candidate_source) {
            enc.set_function(_fn_bscore);
            enc.set_buffer(0, _S);
            enc.set_buffer(1, _B);
            enc.set_constant(2, C);
            enc.set_constant(3, blk);
            enc.set_constant(4, nblk);
            enc.set_constant(5, p0 + i0);
            enc.set_constant(6, r);
            enc.dispatch({(unsigned)nblk, (unsigned)m, 1}, {64, 1, 1});
            topk_(enc, _B, m, nblk, KB, _bpick, ob);
            enc.set_function(_fn_bmask);
            enc.set_buffer(0, _bpick, ob);
            enc.set_buffer(1, _B);
            enc.set_buffer(2, _cmask, oc);
            enc.set_constant(3, KB);
            enc.set_constant(4, nblk);
            enc.dispatch({(unsigned)KB, (unsigned)m, 1}, {32, 1, 1});
          }
          topk_(enc, _S, m, C, KT, _idx, oi);
        }
        if (C <= 0) { fill_(enc, _idx, -1, (std::size_t)n * KT); }
      }
      KC = c.index_topk;
    }
    mark(kIndex);
    // the sparse attention, the output rotated back, group-major: a few
    // rows split by keys (dsv41_attn_split), else a simdgroup a head
    const int CH = (W + KC + kAttnChunk - 1) / kAttnChunk;
    if (n <= kKeepRows && _split_attn) {
      enc.set_function(_fn_attn_split);
      enc.set_buffer(0, _q);
      enc.set_buffer(1, _kv);
      enc.set_buffer(2, st.ring);
      enc.set_buffer(3, *cbuf);
      enc.set_buffer(4, _idx);
      enc.set_buffer(5, _apart);
      enc.set_buffer(6, _aml);
      enc.set_constant(7, NH);
      enc.set_constant(8, W);
      enc.set_constant(9, p0);
      enc.set_constant(10, KC);
      enc.set_constant(11, std::max(pl.ratio, 1));
      enc.set_constant(12, (float)(1.0 / std::sqrt((double)D)));
      enc.set_constant(13, W + kRingSlack);
      enc.set_constant(14, CH);
      enc.set_buffer(15, comp ? _rope_comp : _rope_plain);
      enc.set_constant(16, rope_in_attn ? c.rope_dim : 0);
      enc.dispatch({(unsigned)(512 * ((NH + kAttnHeads - 1) / kAttnHeads)),
                    (unsigned)CH, (unsigned)n},
                   {512, 1, 1});
      enc.set_function(_fn_attn_merge);
      enc.set_buffer(0, _apart);
      enc.set_buffer(1, _aml);
      enc.set_buffer(2, ly.sink);
      enc.set_buffer(3, _o);
      enc.set_buffer(4, comp ? _rope_comp : _rope_plain);
      enc.set_constant(5, n);
      enc.set_constant(6, NH);
      enc.set_constant(7, p0);
      enc.set_constant(8, CH);
      enc.set_constant(9, OG);
      enc.set_constant(10, c.rope_dim);
      enc.dispatch({(unsigned)(256 * NH), (unsigned)n, 1}, {256, 1, 1});
    } else {
    const bool anax = _attn_nax;
    if (anax) { ++_nax_attn; }
    enc.set_function(anax ? _fn_attn_nax : _fn_attn);
    enc.set_buffer(0, _q);
    enc.set_buffer(1, _kv);
    enc.set_buffer(2, st.ring);
    enc.set_buffer(3, *cbuf);
    enc.set_buffer(4, _idx);
    enc.set_buffer(5, ly.sink);
    enc.set_buffer(6, _o);
    enc.set_buffer(7, comp ? _rope_comp : _rope_plain);
    enc.set_constant(8, n);
    enc.set_constant(9, NH);
    enc.set_constant(10, W);
    enc.set_constant(11, p0);
    enc.set_constant(12, KC);
    enc.set_constant(13, std::max(pl.ratio, 1));
    enc.set_constant(14, (float)(1.0 / std::sqrt((double)D)));
    enc.set_constant(15, OG);
    enc.set_constant(16, c.rope_dim);
    enc.set_constant(17, W + kRingSlack);
    enc.set_constant(18, ced && L >= cmid ? p0 : 0);
    if (anax) { enc.set_constant(19, _attn_psplit ? 1 : 0); }
    // the matrix cores' 32 heads a threadgroup, else 8
    const int hpt = anax ? 32 : 8;
    enc.dispatch({(unsigned)(256 * ((NH + hpt - 1) / hpt)), (unsigned)n, 1},
                 {256, 1, 1});
    }
    // this chunk's latents into the ring (after the attention read it; a
    // few rows' went in with their norm)
    if (!few) {
      enc.set_function(_fn_ring);
      enc.set_buffer(0, _kv);
      enc.set_buffer(1, st.ring);
      enc.set_constant(2, n);
      enc.set_constant(3, D);
      enc.set_constant(4, W + kRingSlack);
      enc.set_constant(5, p0);
      enc.dispatch({(unsigned)D, (unsigned)n, 1}, {256, 1, 1});
    }
    mark(kAttn);
    // wo_a per group of heads (beside it, a few rows' attention mix
    // finisher), then wo_b
    {
      auto cs = enc.concurrent_scope(conc);
      gemm_groups_(enc, ly.wo_a, _o, _ot, n, OG, OL);
      if (conc) { hc_mix_finish_(enc, n, ly.ha_scale, ly.ha_base, _mixA); }
    }
    const SharedBuffer* og = &_ot;
    if (n > 1) {
      enc.set_function(_fn_ungroup);
      enc.set_buffer(0, _ot);
      enc.set_buffer(1, _og);
      enc.set_constant(2, n);
      enc.set_constant(3, OL);
      enc.set_constant(4, OG);
      enc.dispatch({(unsigned)OL, (unsigned)n, (unsigned)OG}, {256, 1, 1});
      og = &_og;
    }
    gemm_p_(enc, ly.wo_b, *og, 0, _ao, 0, n);
    defer_post(_ao, _mixA);

    // ---- MoE -----------------------------------------------------------
    mark(kAttnOut);
    // READ AHEAD (a decode step, or a verify's few rows): the next layer's
    // router over this layer's FFN input -- the residual moves little from
    // one layer to the next -- ranks what that layer will pick; the cache
    // reads the likeliest it does not hold while the GPU runs this layer.
    const int aw = n <= kKeepRows && L + 1 < c.n_layers
                       ? std::min(_cache->ahead_width(), _ahead_max) : 0;
    auto route = [&] {
      enc.set_function(_fn_route);
      enc.set_buffer(0, _sc);
      enc.set_buffer(1, ly.gate_b);
      enc.set_buffer(2, ly.gate_bvl.empty() ? ly.gate_b : ly.gate_bvl);
      enc.set_buffer(3, nm > 0 ? _imask : _cmask);
      enc.set_buffer(4, _eid);
      enc.set_buffer(5, _wt);
      enc.set_constant(6, E);
      enc.set_constant(7, K);
      enc.set_constant(8, c.route_scale);
      enc.set_constant(9, c.norm_topk ? 1 : 0);
      enc.set_constant(10, nm > 0 && !ly.gate_bvl.empty() ? 1 : 0);
      enc.dispatch({(unsigned)(256 * n), 1, 1}, {256, 1, 1});
    };
    auto swiglu = [&] {
      enc.set_function(_fn_swiglu);
      enc.set_buffer(0, _gu);
      enc.set_buffer(1, _sact);
      enc.set_constant(2, I);
      enc.set_constant(3, c.swiglu_limit);
      enc.dispatch({(unsigned)I, (unsigned)n, 1}, {256, 1, 1});
    };
    auto ahead = [&] {
      if (aw > 0) {
        gemm_f32_(enc, _layers[(std::size_t)L + 1].gate_w, _xa, 0,
                  _ahead_sc, 0, H, E, n);
      }
    };
    if (conc) {
      auto reads_xa = [&] {
        gemm_f32_(enc, ly.gate_w, _xa, 0, _sc, 0, H, E, n);
        gemm_p_(enc, ly.sh_w13, _xa, 0, _gu, 0, n);
        ahead();
      };
      // the mix's finisher and the shared expert's w2 wait for the routed
      // experts, beside whose gate|up they run (expert_side below)
      if (post_a != nullptr) {
        collapse(_mixA[0], ly.ffn_norm, _xa);
        {
          auto cs = enc.concurrent_scope();
          hc_mix_part_(enc, _X, n, ly.hf_fn);
          reads_xa();
        }
      } else {
        {
          auto cs = enc.concurrent_scope();
          hc_mix_part_(enc, _X, n, ly.hf_fn);
          hc_pre(_mixA[0], ly.ffn_norm, _xa);
        }
        {
          auto cs = enc.concurrent_scope();
          reads_xa();
        }
      }
      // folded, the route runs in the handshake's dispatch, the shared
      // expert's activation beside it (below)
      if (!fold_route) {
        auto cs = enc.concurrent_scope();
        route();
        swiglu();
      }
    } else {
      collapse(_mixA[0], ly.ffn_norm, _xa);
      hc_mix(ly.hf_fn, ly.hf_scale, ly.hf_base, _mixF);
      mark(kHc);
      gemm_f32_(enc, ly.gate_w, _xa, 0, _sc, 0, H, E, n);
      if (!fold_route) { route(); }
      mark(kRoute);
      // the shared expert runs while the routed ones are read
      gemm_p_(enc, ly.sh_w13, _xa, 0, _gu, 0, n);
      swiglu();
      gemm_p_(enc, ly.sh_w2, _sact, 0, _shout, 0, n);
      ahead();
    }
    if (study) {
      // the study's routers of the layers after this one, on its input
      for (int k = 1; k <= _study.k && L + k < c.n_layers; ++k) {
        gemm_f32_(enc, _layers[(std::size_t)(L + k)].gate_w, _xa, 0,
                  _study.early,
                  (((std::size_t)L * _study.k + (std::size_t)(k - 1)) * E) * 4,
                  H, E, 1);
      }
    }
    mark(kShared);
    if (hs) {
      // the host's turn inside the running buffers: picks (and the next
      // layer's scores) out, slots in; this layer's buffer goes now, and
      // the host answers as the GPU posts, kHsAhead layers behind at most
      const std::uint32_t seq = (std::uint32_t)turns.size() + 1;
      {
        auto cs = enc.concurrent_scope(fold_route && conc);
        if (fold_route && conc) { swiglu(); }
        enc.set_function(fold_route ? _fn_hs_route : _fn_hs);
        if (fold_route) {
          enc.set_buffer(11, _sc);
          enc.set_buffer(12, ly.gate_b);
          enc.set_buffer(13, ly.gate_bvl.empty() ? ly.gate_b : ly.gate_bvl);
          enc.set_buffer(14, nm > 0 ? _imask : _cmask);
          enc.set_buffer(15, _eid);
          enc.set_buffer(16, _wt);
          enc.set_constant(17, E);
          enc.set_constant(18, K);
          enc.set_constant(19, c.route_scale);
          enc.set_constant(20, c.norm_topk ? 1 : 0);
          enc.set_constant(21, nm > 0 && !ly.gate_bvl.empty() ? 1 : 0);
        }
        enc.set_buffer(0, _hsf);
        enc.set_buffer(1, _eid);
        enc.set_buffer(2, aw > 0 ? _ahead_sc : _eid);
        enc.set_buffer(3, _eid);
        enc.set_buffer(4, _hsbo);
        enc.set_buffer(5, _hsbi);
        enc.set_buffer(6, _slot);
        enc.set_buffer(7, _slot);
        enc.set_buffer(8, _slot);
        const std::uint32_t nv[4] = {
            (std::uint32_t)np, aw > 0 ? (std::uint32_t)(n * E) : 0u, 0u, seq};
        const std::uint32_t mv[4] = {(std::uint32_t)np, 0u, 0u, _hs_spins};
        enc.set_constant_bytes(9, nv, sizeof nv);
        enc.set_constant_bytes(10, mv, sizeof mv);
        enc.dispatch({256, 1, 1}, {256, 1, 1});
      }
      n_disp += enc.dispatch_count();
      enc.end();
      fences.push_back(stream.commit());
      enc = stream.begin_compute();
      turns.push_back({L, aw, seq});
      serve(L - kHsAhead);
    } else {
      std::string why;
      const auto t0 = Clk::now();
      n_disp += enc.dispatch_count();
      if (!CommandStream::sync_and_reopen(enc, &why)) {
        cx.hist.resize(hist0);
        return fail("GPU: " + why);
      }
      const auto t1 = Clk::now();
      if (!service(L, aw, why,
                   static_cast<const std::int32_t*>(_eid.contents()),
                   static_cast<const float*>(_ahead_sc.contents()),
                   static_cast<std::int32_t*>(_slot.contents()))) {
        cx.hist.resize(hist0);
        return fail("expert cache: " + why);
      }
      const auto t2 = Clk::now();
      t_gpu += ms(t0, t1);
      t_cache += ms(t1, t2);
    }
    const bool aff = ly.ex.b13 > 0;       // an oQ pack's affine experts
    // a few rows' work that waits for no expert, beside their gate|up:
    // the shared expert's w2 and the FFN mix's finisher
    auto expert_side = [&] {
      if (!conc) { return; }
      gemm_p_(enc, ly.sh_w2, _sact, 0, _shout, 0, n);
      hc_mix_finish_(enc, n, ly.hf_scale, ly.hf_base, _mixF);
    };
    if (_grouped_ppe > 0.0 && E <= kSortExperts &&
        (double)np >= _grouped_ppe * E) {
      // sorted by expert into 32-row tiles: a tile reads its expert once
      const int cap = (int)grouped_cap_((std::size_t)np, (std::size_t)E);
      enc.set_function(_fn_msort);
      enc.set_buffer(0, _eid);
      enc.set_buffer(1, _slot);
      enc.set_buffer(2, _wt);
      enc.set_buffer(3, _srow);
      enc.set_buffer(4, _sdst);
      enc.set_buffer(5, _swt);
      enc.set_buffer(6, _tile2e);
      enc.set_constant(7, np);
      enc.set_constant(8, K);
      enc.set_constant(9, E);
      enc.set_constant(10, cap);
      enc.dispatch({1024, 1, 1}, {1024, 1, 1});
      const unsigned tiles = (unsigned)(cap / 32);
      // the matrix-core twins: 64 features a threadgroup
      const QFns& qf13 = _qf[std::max(qidx_(ly.ex.b13), 0)];
      const QFns& qf2 = _qf[std::max(qidx_(ly.ex.b2), 0)];
      const bool nax = _moe_nax && I % 64 == 0 && H % 64 == 0 &&
                       (aff ? qf13.gun.valid() && qf2.dgn.valid() : true);
      const unsigned gx = nax ? (unsigned)(I / 64 * 32) : (unsigned)I;
      const unsigned dx = nax ? (unsigned)(H / 64 * 32) : (unsigned)H;
      if (nax) { _nax_gemms += 2; }
      if (aff) {
        {
        auto cs = enc.concurrent_scope(conc);
        enc.set_function(nax ? qf13.gun : qf13.gu);
        for (int k = 0; k < 6; ++k) { enc.set_buffer(k, _cache->pool(k)); }
        enc.set_buffer(6, _xa);
        enc.set_buffer(7, _act);
        enc.set_buffer(8, _srow);
        enc.set_buffer(9, _swt);
        enc.set_buffer(10, _tile2e);
        enc.set_constant(11, H);
        enc.set_constant(12, I);
        enc.set_constant(13, c.swiglu_limit);
        enc.set_constant(14, ly.ex.gs);
        enc.set_constant(15, _xcw13);
        enc.set_constant(16, _xcs13);
        enc.dispatch({gx, tiles * 2, 2}, {32, 2, 2});
        expert_side();
        }
        enc.set_function(nax ? qf2.dgn : qf2.dg);
        for (int k = 0; k < 3; ++k) {
          enc.set_buffer(k, _cache->pool(6 + k));
        }
        enc.set_buffer(3, _act);
        enc.set_buffer(4, _part);
        enc.set_buffer(5, _sdst);
        enc.set_buffer(6, _tile2e);
        enc.set_constant(7, I);
        enc.set_constant(8, H);
        enc.set_constant(9, ly.ex.gs);
        enc.set_constant(10, _xcw2);
        enc.set_constant(11, _xcs2);
        enc.dispatch({dx, tiles * 2, 2}, {32, 2, 2});
        ++_grouped_steps;
      } else {
        {
        auto cs = enc.concurrent_scope(conc);
        enc.set_function(nax ? _fn_gu4n : _fn_gu4g);
        for (int k = 0; k < 4; ++k) { enc.set_buffer(k, _cache->pool(k)); }
        enc.set_buffer(4, _xa);
        enc.set_buffer(5, _act);
        enc.set_buffer(6, _srow);
        enc.set_buffer(7, _swt);
        enc.set_buffer(8, _tile2e);
        enc.set_constant(9, H);
        enc.set_constant(10, I);
        enc.set_constant(11, c.swiglu_limit);
        enc.dispatch({gx, tiles * 2, 2}, {32, 2, 2});
        expert_side();
        }
        enc.set_function(nax ? _fn_down4n : _fn_down4g);
        enc.set_buffer(0, _cache->pool(4));
        enc.set_buffer(1, _cache->pool(5));
        enc.set_buffer(2, _act);
        enc.set_buffer(3, _part);
        enc.set_buffer(4, _sdst);
        enc.set_buffer(5, _tile2e);
        enc.set_constant(6, I);
        enc.set_constant(7, H);
        enc.dispatch({dx, tiles * 2, 2}, {32, 2, 2});
        ++_grouped_steps;
      }
    } else if (aff) {
      // the affine twins: 8 lanes a row, so 8 features (outputs) a
      // threadgroup; a few rows' groups as their FP4 kernels take them
      const unsigned nz = (unsigned)(rows_groups ? ngroups : np);
      const int mi = rows_groups ? 1 + rk : 0;
      {
      auto cs = enc.concurrent_scope(conc);
      enc.set_function(_qf[qidx_(ly.ex.b13)].sw[mi]);
      for (int k = 0; k < 6; ++k) { enc.set_buffer(k, _cache->pool(k)); }
      enc.set_buffer(6, _xa);
      enc.set_buffer(7, _act);
      enc.set_buffer(8, rows_groups ? _mgrp : _slot);
      enc.set_buffer(9, _wt);
      enc.set_constant(10, H);
      enc.set_constant(11, I);
      enc.set_constant(12, K);
      enc.set_constant(13, c.swiglu_limit);
      enc.set_constant(14, ly.ex.gs);
      enc.set_constant(15, _xcw13);
      enc.set_constant(16, _xcs13);
      enc.dispatch({32, (unsigned)(((I + 7) / 8) * 2), nz}, {32, 2, 1});
      expert_side();
      }
      enc.set_function(_qf[qidx_(ly.ex.b2)].dn[mi]);
      for (int k = 0; k < 3; ++k) { enc.set_buffer(k, _cache->pool(6 + k)); }
      enc.set_buffer(3, _act);
      enc.set_buffer(4, _part);
      enc.set_buffer(5, rows_groups ? _mgrp : _slot);
      enc.set_constant(6, I);
      enc.set_constant(7, H);
      enc.set_constant(8, ly.ex.gs);
      enc.set_constant(9, _xcw2);
      enc.set_constant(10, _xcs2);
      enc.dispatch({32, (unsigned)(((H + 7) / 8) * 2), nz}, {32, 2, 1});
    } else {
      const unsigned nz = (unsigned)(rows_groups ? ngroups : np);
      {
      auto cs = enc.concurrent_scope(conc);
      enc.set_function(rows_groups ? _fn_swiglu4r[rk] : _fn_swiglu4);
      enc.set_buffer(0, _cache->pool(0));
      enc.set_buffer(1, _cache->pool(1));
      enc.set_buffer(2, _cache->pool(2));
      enc.set_buffer(3, _cache->pool(3));
      enc.set_buffer(4, _xa);
      enc.set_buffer(5, _act);
      enc.set_buffer(6, rows_groups ? _mgrp : _slot);
      enc.set_buffer(7, _wt);
      enc.set_constant(8, H);
      enc.set_constant(9, I);
      enc.set_constant(10, K);
      enc.set_constant(11, c.swiglu_limit);
      enc.dispatch({32, (unsigned)((I / 4) * 2), nz}, {32, 2, 1});
      expert_side();
      }
      enc.set_function(rows_groups ? _fn_down4r[rk] : _fn_down4);
      enc.set_buffer(0, _cache->pool(4));
      enc.set_buffer(1, _cache->pool(5));
      enc.set_buffer(2, _act);
      enc.set_buffer(3, _part);
      enc.set_buffer(4, rows_groups ? _mgrp : _slot);
      enc.set_constant(5, I);
      enc.set_constant(6, H);
      enc.dispatch({32, (unsigned)((H / 8) * 2), nz}, {32, 2, 1});
    }
    defer_post_moe(_mixF);
    if (_capture == L) {
      flush_post();
      std::string why;
      n_disp += enc.dispatch_count();
      if (!CommandStream::sync_and_reopen(enc, &why)) {
        return fail("GPU: " + why);
      }
      _captured.resize((std::size_t)n * 4 * H);
      const auto* x = static_cast<const std::uint16_t*>(_X.contents());
      for (std::size_t k = 0; k < _captured.size(); ++k) {
        _captured[k] = _bf16 ? bf16_f32_(x[k])
                             : (float)reinterpret_cast<const _Float16*>(x)[k];
      }
    }
  }

  // ---- DSpark: every row's main latent into each stage's window ---------
  // (DSparkAttention's main_kv; a verify's rows past what it keeps land on
  // slots the drafts never read, and are taken again when committed)
  if (spark) {
    gemm_p_(enc, _main_proj, _mh, 0, _ao, 0, n);
    rms_(enc, _ao, 0, _main_norm, _mx, 0, H, n, H, H, false);
    for (std::size_t k = 0; k < _stages.size(); ++k) {
      const Layer& sl = _stages[k]->ly;
      gemm_p_(enc, sl.wkv, _mx, 0, _kvr, 0, n);
      rms_(enc, _kvr, 0, sl.kv_norm, _kv, 0, D, n, D, D, false);
      rope_(enc, _kv, 0, false, n, 1, D, D, D, p0, 1, false);
      enc.set_function(_fn_ring);
      enc.set_buffer(0, _kv);
      enc.set_buffer(1, cx.spark_ring[k]);
      enc.set_constant(2, n);
      enc.set_constant(3, D);
      enc.set_constant(4, W + kRingSlack);
      enc.set_constant(5, p0);
      enc.dispatch({(unsigned)D, (unsigned)n, 1}, {256, 1, 1});
    }
  }

  // ---- the head ---------------------------------------------------------
  if (logits != nullptr) {
    collapse(_mixF[0], _norm, _xa);
    gemm_f32_(enc, _head, _xa, (std::size_t)r0 * H * 2, _logits, 0, H,
              c.vocab, rows);
  }
  flush_post();
  n_disp += enc.dispatch_count();
  enc.end();
  std::string why;
  const auto t_last = Clk::now();
  auto fence = stream.commit();
  if (hs) { serve(c.n_layers); }       // the last turns, the tail queued
  bool all_ok = fence.wait_ok(&why);
  for (CommandStream::Fence& f : fences) {   // done by now: in order
    std::string w;
    if (!f.wait_ok(&w) && all_ok) {
      all_ok = false;
      why = w;
    }
  }
  if (hs) {
    hs_beating_(false);
    if (hs_lost || word(2).load(std::memory_order_relaxed) != 0) {
      // the GPU stopped waiting for a turn, so its slots are stale: the
      // whole forward again, a commit and wait a layer, from the state the
      // forward found
      cx.hist.resize(hist0);
      for (int L = 0; L < c.n_layers; ++L) {
        const auto& q = pend0[(std::size_t)L];
        if (!q.empty()) {
          std::memcpy(cx.L[(std::size_t)L].pend.contents(), q.data(),
                      q.size());
        }
      }
      _hs_redo = true;
      ++_hs_redos;
      const bool ok = forward_piece_(cx, ids, logits, all_rows, err, mpos,
                                     mrows);
      _hs_redo = false;
      return ok;
    }
  }
  if (!all_ok) {
    cx.hist.resize(hist0);
    return fail("GPU: " + why);
  }
  if (!hs_err.empty()) {
    cx.hist.resize(hist0);
    return fail(hs_err);
  }
  if (study) { study_step_(); }
  if (prof_level >= 2) {
    std::string line;
    for (int k = 0; k < kSections; ++k) {
      line += std::format(" {} {:.1f}/{:.2f}", kSec[k], t_sec[k], g_sec[k]);
    }
    std::fprintf(stderr, "[dsv41] sections (wall/GPU ms):%s\n", line.c_str());
  }
  if (prof) {
    const auto t_end = Clk::now();
    std::fprintf(stderr,
                 "[dsv41] forward %d rows: %.1f ms -- engram rows %.1f "
                 "(hashed in %.1f), GPU "
                 "waits %.1f (+ last %.1f), expert reads %.1f, encode %.1f; "
                 "%ld dispatches\n",
                 (int)ids.size(), ms(t_start, t_end), ms(t_start, t_engram),
                 ms(t_start, t_ehash),
                 t_gpu,
                 ms(t_last, t_end), t_cache,
                 ms(t_engram, t_last) - t_gpu - t_cache, n_disp);
  }
  if (logits != nullptr) {
    const auto* lg = static_cast<const float*>(_logits.contents());
    logits->assign(lg, lg + (std::size_t)rows * c.vocab);
  }
  cx.pos = end;
  cx.back_lo = std::max(cx.back_lo, end - kKeepRows);
  return true;
}

bool
MetalDeepseekV41Model::ensure_spark_(int rows)
{
  if (!_sX.empty()) { return true; }
  const DeepseekV41Config& c = _cfg;
  const std::size_t B = (std::size_t)rows, H = (std::size_t)c.dim;
  auto b = [&](std::size_t bytes) {
    return _mc->make_shared_buffer(std::max<std::size_t>(bytes, 16));
  };
  _sX = b(B * 4 * H * 2);
  _stok = b((B + 1) * 4);
  _slog = b(B * c.vocab * 4);
  _memb = b(B * (std::size_t)c.dspark_markov * 2);
  _mbias = b((std::size_t)c.vocab * 4);
  _xpre = b(B * H * 2);
  return !_sX.empty() && !_stok.empty() && !_slog.empty() &&
         !_memb.empty() && !_mbias.empty() && !_xpre.empty();
}

bool
MetalDeepseekV41Model::dspark_draft(CtxId ctx, std::int32_t token,
                                    std::vector<std::int32_t>* drafts,
                                    std::vector<float>* conf,
                                    std::vector<float>* logits,
                                    std::string* err,
                                    const std::vector<std::int32_t>* given)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[deepseek-v4.1] DSpark: " + std::move(m); }
    _hccnt_dirty = true;
    return false;
  };
  if (_stages.empty()) { return fail("not loaded"); }
  auto it = _ctx.find(ctx);
  if (it == _ctx.end() || it->second.pos < 1) {
    return fail("a draft needs a committed position");
  }
  Ctx& cx = it->second;
  const DeepseekV41Config& c = _cfg;
  if (token < 0 || token >= c.vocab) { return fail("token out of range"); }
  const int B = c.dspark_block, p = cx.pos - 1;
  // The block's rows: [token, noise...] (DSpark's one pass), or a chained
  // pass's [token, given...] -- every row a real token, its next draft the
  // last row's (the rows attend each other whole, as the block's do).
  const int R = given != nullptr ? (int)given->size() + 1 : B;
  if (R > B) { return fail("a chain past the block"); }
  const int H = c.dim, D = c.head_dim, NH = c.n_heads, QL = c.q_lora;
  const int OG = c.o_groups, OL = c.o_lora;
  const int I = c.moe_inter, SE = c.dspark_experts, SK = c.dspark_top_k;
  const int W = c.window, V = c.vocab, MR = c.dspark_markov;
  std::string e;
  if (!ensure_scratch_(B, cx.pos) || !ensure_spark_(B) ||
      !ensure_rope_(p + B + 2)) {
    return fail("out of memory for a draft");
  }
  // the block [token, noise...] at positions p + 1 .., the identity mix
  {
    auto* id = static_cast<std::int32_t*>(_ids.contents());
    auto* stok = static_cast<std::int32_t*>(_stok.contents());
    id[0] = token;
    stok[0] = token;
    for (int i = 1; i < B; ++i) {
      const bool real = given != nullptr && i < R;
      id[i] = real ? (*given)[(std::size_t)i - 1] : c.dspark_noise;
      if (real) { stok[i] = id[i]; }
    }
    auto* pm = static_cast<float*>(_pm[0].contents());
    for (int i = 0; i < B; ++i) {
      pm[i * 4] = 1.0f;
      pm[i * 4 + 1] = pm[i * 4 + 2] = pm[i * 4 + 3] = 0.0f;
    }
  }
  // one stream for the model's life: a new one is a new MTLCommandQueue,
  // whose first commit waits 1-5 ms before the GPU starts it (4.8 ms on
  // an M5 Ultra, every forward)
  if (!_stream.valid()) { _stream = _mc->make_command_stream(); }
  CommandStream& stream = _stream;
  ComputeEncoder enc = stream.begin_compute();
  if (_hccnt_dirty) {
    // a forward failed with a collapse's count open: set them back first
    fill_(enc, _hccnt, 0, _hccnt.byte_size() / 4);
    _hccnt_dirty = false;
  }
  enc.set_function(_fn_embed);
  enc.set_buffer(0, _embed);
  enc.set_buffer(1, _ids);
  enc.set_buffer(2, _sX);
  enc.set_constant(3, H);
  enc.set_constant(4, c.hc);
  enc.dispatch({(unsigned)H, (unsigned)R, 1}, {256, 1, 1});
  const SharedBuffer* pre = &_pm[0];
  for (std::size_t sg = 0; sg < _stages.size(); ++sg) {
    const Stage& st = *_stages[sg];
    const Layer& ly = st.ly;
    // ---- the window-only attention over the committed latents + block
    hc_mix_(enc, _sX, R, ly.ha_fn, ly.ha_scale, ly.ha_base, _mixA);
    hc_pre_(enc, _sX, R, *pre, &ly.attn_norm, _xa);
    gemm_p_(enc, ly.wq_a, _xa, 0, _qa, 0, R);
    rms_(enc, _qa, 0, ly.q_norm, _qr, 0, QL, R, QL, QL, false);
    gemm_p_(enc, ly.wq_b, _qr, 0, _q, 0, R);
    rope_(enc, _q, 0, false, R, NH, NH * D, D, D, p + 1, 1, false);
    gemm_p_(enc, ly.wkv, _xa, 0, _kvr, 0, R);
    rms_(enc, _kvr, 0, ly.kv_norm, _kv, 0, D, R, D, D, false);
    rope_(enc, _kv, 0, false, R, 1, D, D, D, p + 1, 1, false);
    enc.set_function(_fn_spark_attn);
    enc.set_buffer(0, _q);
    enc.set_buffer(1, _kv);
    enc.set_buffer(2, cx.spark_ring[sg]);
    enc.set_buffer(3, ly.sink);
    enc.set_buffer(4, _o);
    enc.set_buffer(5, _rope_plain);
    enc.set_constant(6, R);
    enc.set_constant(7, NH);
    enc.set_constant(8, W);
    enc.set_constant(9, W + kRingSlack);
    enc.set_constant(10, p);
    enc.set_constant(11, (float)(1.0 / std::sqrt((double)D)));
    enc.set_constant(12, OG);
    enc.set_constant(13, c.rope_dim);
    enc.dispatch({(unsigned)(256 * ((NH + 7) / 8)), (unsigned)R, 1},
                 {256, 1, 1});
    gemm_groups_(enc, ly.wo_a, _o, _ot, R, OG, OL);
    enc.set_function(_fn_ungroup);
    enc.set_buffer(0, _ot);
    enc.set_buffer(1, _og);
    enc.set_constant(2, R);
    enc.set_constant(3, OL);
    enc.set_constant(4, OG);
    enc.dispatch({(unsigned)OL, (unsigned)R, (unsigned)OG}, {256, 1, 1});
    gemm_p_(enc, ly.wo_b, _og, 0, _ao, 0, R);
    hc_post_(enc, _ao, _sX, R, _mixA);
    // ---- its MoE: the stage's own experts, resident (slot = expert id)
    hc_mix_(enc, _sX, R, ly.hf_fn, ly.hf_scale, ly.hf_base, _mixF);
    hc_pre_(enc, _sX, R, _mixA[0], &ly.ffn_norm, _xa);
    gemm_f32_(enc, ly.gate_w, _xa, 0, _sc, 0, H, SE, R);
    enc.set_function(_fn_route);
    enc.set_buffer(0, _sc);
    enc.set_buffer(1, ly.gate_b);
    enc.set_buffer(2, ly.gate_b);
    enc.set_buffer(3, _cmask);
    enc.set_buffer(4, _eid);
    enc.set_buffer(5, _wt);
    enc.set_constant(6, SE);
    enc.set_constant(7, SK);
    enc.set_constant(8, c.route_scale);
    enc.set_constant(9, c.norm_topk ? 1 : 0);
    enc.set_constant(10, 0);
    enc.dispatch({(unsigned)(256 * R), 1, 1}, {256, 1, 1});
    if (sg < _force_draft.size() &&
        _force_draft[sg].size() == (std::size_t)(R * SK)) {
      // the given picks (debug), weighed by this draft's own scores
      std::string why;
      if (!CommandStream::sync_and_reopen(enc, &why)) {
        return fail("GPU: " + why);
      }
      force_picks_(_force_draft[sg], R, SE, SK);
    }
    gemm_p_(enc, ly.sh_w13, _xa, 0, _gu, 0, R);
    enc.set_function(_fn_swiglu);
    enc.set_buffer(0, _gu);
    enc.set_buffer(1, _sact);
    enc.set_constant(2, I);
    enc.set_constant(3, c.swiglu_limit);
    enc.dispatch({(unsigned)I, (unsigned)R, 1}, {256, 1, 1});
    gemm_p_(enc, ly.sh_w2, _sact, 0, _shout, 0, R);
    const int np = R * SK;
    enc.set_function(_fn_swiglu4);
    for (int k = 0; k < 4; ++k) { enc.set_buffer(k, st.ex[k]); }
    enc.set_buffer(4, _xa);
    enc.set_buffer(5, _act);
    enc.set_buffer(6, _eid);
    enc.set_buffer(7, _wt);
    enc.set_constant(8, H);
    enc.set_constant(9, I);
    enc.set_constant(10, SK);
    enc.set_constant(11, c.swiglu_limit);
    enc.dispatch({32, (unsigned)((I / 4) * 2), (unsigned)np}, {32, 2, 1});
    enc.set_function(_fn_down4);
    enc.set_buffer(0, st.ex[4]);
    enc.set_buffer(1, st.ex[5]);
    enc.set_buffer(2, _act);
    enc.set_buffer(3, _part);
    enc.set_buffer(4, _eid);
    enc.set_constant(5, I);
    enc.set_constant(6, H);
    enc.dispatch({32, (unsigned)((H / 8) * 2), (unsigned)np}, {32, 2, 1});
    enc.set_function(_fn_combine);
    enc.set_buffer(0, _part);
    enc.set_buffer(1, _shout);
    enc.set_buffer(2, _fo);
    enc.set_constant(3, H);
    enc.set_constant(4, SK);
    enc.dispatch({(unsigned)H, (unsigned)R, 1}, {256, 1, 1});
    hc_post_(enc, _fo, _sX, R, _mixF);
    pre = &_mixF[0];
  }
  // ---- forward_head: the collapse, the stage's norm, the shared head, then
  // each row's Markov bias from the token before it and its argmax
  hc_pre_(enc, _sX, R, *pre, nullptr, _xpre);
  rms_(enc, _xpre, 0, _spark_norm, _xa, 0, H, R, H, H, false);
  gemm_f32_(enc, _head, _xa, 0, _slog, 0, H, V, R);
  for (int i = given != nullptr ? R - 1 : 0; i < R; ++i) {
    enc.set_function(_fn_gather_tok);
    enc.set_buffer(0, _markov_embed);
    enc.set_buffer(1, _stok);
    enc.set_buffer(2, _memb);
    enc.set_constant(3, MR);
    enc.set_constant(4, i);
    enc.dispatch({(unsigned)MR, 1, 1}, {256, 1, 1});
    gemm_f32_(enc, _markov_head, _memb, (std::size_t)i * MR * 2, _mbias, 0,
              MR, V, 1);
    enc.set_function(_fn_add_argmax);
    enc.set_buffer(0, _slog);
    enc.set_buffer(1, _mbias);
    enc.set_buffer(2, _stok);
    enc.set_constant(3, V);
    enc.set_constant(4, i);
    enc.dispatch({1024, 1, 1}, {1024, 1, 1});
  }
  enc.end();
  std::string why;
  _force_draft.clear();
  if (!stream.commit().wait_ok(&why)) { return fail("GPU: " + why); }
  const auto* tk = static_cast<const std::int32_t*>(_stok.contents());
  const int r0 = given != nullptr ? R - 1 : 0;   // the rows reported
  if (drafts != nullptr) { drafts->assign(tk + 1 + r0, tk + 1 + R); }
  if (conf != nullptr) {
    // DSparkConfidenceHead: [collapse | Markov embedding] . w, in f32
    conf->assign((std::size_t)(R - r0), 0.0f);
    const auto* xp = static_cast<const std::uint16_t*>(_xpre.contents());
    const auto* mb = static_cast<const std::uint16_t*>(_memb.contents());
    auto to_f = [&](std::uint16_t u) {
      return _bf16 ? bf16_f32_(u)
                   : (float)std::bit_cast<_Float16>(u);
    };
    for (int i = r0; i < R; ++i) {
      float a = 0.0f;
      for (int d = 0; d < H; ++d) {
        a += _conf_w[(std::size_t)d] * to_f(xp[(std::size_t)i * H + d]);
      }
      for (int d = 0; d < MR; ++d) {
        a += _conf_w[(std::size_t)(H + d)] *
             to_f(mb[(std::size_t)i * MR + d]);
      }
      (*conf)[(std::size_t)(i - r0)] = a;
    }
  }
  if (logits != nullptr) {
    const auto* lg = static_cast<const float*>(_slog.contents());
    logits->assign(lg + (std::size_t)r0 * V, lg + (std::size_t)R * V);
  }
  return true;
}

std::size_t
MetalDeepseekV41Model::dspark_bytes(const std::string& model_dir)
{
  auto w = MetalLlamaWeights::open_model(model_dir);
  if (!w.has_value()) { return 0; }
  std::size_t b = 0;
  for (const std::string& nm : w->tensor_names()) {
    const std::string n = nm.rfind("language_model.", 0) == 0
                              ? nm.substr(15) : nm;
    if (n.rfind("mtp.", 0) != 0) { continue; }
    const auto* ti = w->info(nm);
    if (ti == nullptr) { continue; }
    // held as stored: FP4 experts, FP8 projections with a scale per 32 of
    // a row (the release's block scales spread), bf16 the rest; the MLX
    // pack's bf16 wo_a as it is
    b += ti->dtype == "F8_E8M0" && n.find(".experts.") == std::string::npos
             ? 32 * ti->nbytes
             : ti->nbytes;
  }
  return b;
}

bool
MetalDeepseekV41Model::truncate(CtxId ctx, int pos)
{
  auto it = _ctx.find(ctx);
  if (it == _ctx.end()) { return pos == 0; }
  Ctx& cx = it->second;
  if (pos == cx.pos) { return true; }
  // At most kKeepRows back, over forwards' boundaries: each compressor's
  // open group rebuilt from the inputs it keeps by position (the last
  // kKeepRows + ratio - 1 rows written, all of them from back_lo - ratio +
  // 1 on), the windows already holding what the next rows read.
  if (pos > cx.pos || pos < cx.back_lo) { return false; }
  const DeepseekV41Config& c = _cfg;
  const std::size_t D2 = (std::size_t)c.head_dim * 2;
  for (int L = 0; L < c.n_layers; ++L) {
    const auto& pl = c.plan[(std::size_t)L];
    LayerState& st = cx.L[(std::size_t)L];
    if (!pl.kv_source || pl.ratio <= 1 || st.pend.empty()) { continue; }
    const int r = pl.ratio, kr = kKeepRows + r - 1;
    for (int q = pos / r * r; q < pos; ++q) {
      std::memcpy(static_cast<float*>(st.pend.contents()) + (q % r) * D2,
                  static_cast<const float*>(st.keep.contents()) +
                      (std::size_t)(q % kr) * D2,
                  D2 * sizeof(float));
    }
  }
  cx.pos = pos;
  cx.hist.resize((std::size_t)pos);
  return true;
}

}  // namespace vpipe::genai
