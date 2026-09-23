#include "generative-models/shared/ane-ffn.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include "apple-silicon/coreml/ane-worker.h"
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "generative-models/generative-model-manager.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

namespace vpipe {
namespace genai {

namespace {

using Clk = std::chrono::steady_clock;

double
ms_since_(Clk::time_point t)
{
  return std::chrono::duration<double, std::milli>(Clk::now() - t).count();
}

// bf16 -> fp16 as a table. bf16 IS the top half of an f32, so the mapping
// is a pure function of its 16 bits: 131 KB, built once, and in L2. The
// narrowing is safe for a feed-forward because it is self-contained --
// only its output re-enters the bf16 residual stream, and that addition
// stays on the GPU.
const std::uint16_t*
bf16_to_f16_lut_()
{
  static const std::vector<std::uint16_t> lut = [] {
    std::vector<std::uint16_t> t(65536);
    for (std::uint32_t v = 0; v < 65536u; ++v) {
      const std::uint32_t bits = v << 16;
      float f = 0.0f;
      std::memcpy(&f, &bits, sizeof(f));
      const _Float16 hv = (_Float16)f;
      std::memcpy(&t[v], &hv, sizeof(hv));
    }
    return t;
  }();
  return lut.data();
}

float
f32_of_bf16_(std::uint16_t v)
{
  const std::uint32_t bits = (std::uint32_t)v << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

}  // namespace

bool
AneFfnSource::stageable() const noexcept
{
  if (!quantized) { return w != nullptr && !w->empty(); }
  return (bits == 4 || bits == 8) && codes != nullptr && !codes->empty() &&
         scales != nullptr && !scales->empty() && qbias != nullptr &&
         !qbias->empty();
}

std::unique_ptr<AneFeedForward>
AneFeedForward::create(const Options& o)
{
  const SessionContextIntf* ss = o.session;
  if (ss == nullptr || o.mc == nullptr || o.seq <= 0 || o.hidden <= 0 ||
      o.ffn <= 0) {
    return nullptr;
  }
  const std::string& tag = o.tag;
  if (ss->services() == nullptr ||
      ss->services()->coreml_model_manager() == nullptr) {
    ss->warn(fmt("{}: no CoreML model manager in this session; keeping the "
                 "GPU feed-forward", tag));
    return nullptr;
  }
  // The runtime-weight module is EMITTED -- there is no template form of
  // it -- so a machine where emission does not verify keeps the GPU path.
  if (!AneModule::emitter_usable(ss)) {
    ss->warn(fmt("{}: the ANE emitter did not verify on this machine; "
                 "keeping the GPU feed-forward", tag));
    return nullptr;
  }
  const int chunk = std::max(1, o.chunk);
  // Unset means AUTO, and auto opens at HALF. Not at the rate-implied
  // ~0.64: that is an overshoot at small seq -- at 1024^2 on Krea-2 it
  // rounds to 2 chunks, which leaves the GPU 64 rows and wastes a block
  // before the controller pulls it back. Half never starves either engine.
  double share = (double)o.rows;
  if (!(share > 0.0)) { share = 0.5; }
  if (share > 1.0) { share = 1.0; }
  // CHUNKED: the module is compiled at a FIXED chunk and the host calls it
  // as many times as the share needs. Rows are independent, so chunking
  // leaves no seam -- and it measured identical to an exact-M module
  // (95.72 ms against 96.71 for 2080 rows). The share rounds DOWN to whole
  // chunks and the GPU takes the remainder, so there is no padded tail.
  // AUTO rounds to the NEAREST chunk (at least one): a single tile of a
  // ViT decoder is 1797 rows, whose half floors to zero 1024-row chunks.
  // A pinned share still rounds down, as documented.
  const bool autos = !(o.rows > 0.0f);
  const double want = ((double)o.seq * share) / (double)chunk;
  const int chunks = autos ? std::max(1, (int)(want + 0.5)) : (int)want;
  int rows = chunks * chunk;
  if (rows >= o.seq) { rows = (chunks - 1) * chunk; }
  if (rows <= 0) {
    ss->info(fmt("{}: {} rows give the ANE less than one {}-row chunk; "
                 "keeping the GPU feed-forward", tag, o.seq, chunk));
    return nullptr;
  }

  AneGraphSpec spec;
  spec.kind = o.matmul ? AneGraphSpec::Kind::Matmul
            : o.gelu   ? AneGraphSpec::Kind::GeluFfn
                       : AneGraphSpec::Kind::SwiGluFfn;
  spec.M  = chunk;
  spec.K  = o.hidden;
  spec.N  = o.ffn;
  spec.bk = o.tile;
  spec.bn = o.tile;
  spec.runtime_weights = true;
  spec.weights_out_in  = true;    // staging copies rows as they are
  spec.biases          = o.biases && !o.matmul;
  if (!spec.valid()) {
    ss->warn(fmt("{}: a {}x{} feed-forward has no valid ANE graph at tile "
                 "{}; keeping the GPU feed-forward", tag, o.hidden, o.ffn,
                 o.tile));
    return nullptr;
  }
  char work[360];
  std::snprintf(work, sizeof(work), "%s/vpipe-ane-%s-rt-%dx%dx%d-t%d.mlmodelc",
                std::filesystem::temp_directory_path().c_str(),
                o.matmul ? "mm" : o.gelu ? "gelu" : "ffn", chunk, o.hidden,
                o.ffn, o.tile);
  // System-wide wired and the pool's own use BEFORE CoreML builds anything:
  // the growth at the first predict, less the pool's, is what the module
  // wired. See _ext_wired.
  const std::size_t wired0 = o.mc->memory_budget().wired;
  auto* mgr0 = ss->services()->generative_model_manager();
  const std::size_t pool0 = mgr0 != nullptr ? mgr0->wired_pool_used() : 0;
  const auto t0 = Clk::now();
  std::unique_ptr<AneFeedForward> f(new AneFeedForward());
  f->_wired0   = wired0;
  f->_pool0    = pool0;
  f->_estimate = o.matmul ? matmul_runtime_bytes(o.hidden, o.ffn, o.seq, chunk)
               : o.gelu   ? gelu_runtime_bytes(o.hidden, o.ffn, chunk)
                          : runtime_bytes(o.hidden, o.ffn, o.seq, chunk);
  f->_rt = AneModule::build_runtime(ss, spec, work);
  f->_w  = AneWeightSlots::create(spec);
  const double build_ms = ms_since_(t0);
  const std::size_t elems = (std::size_t)chunk * (std::size_t)o.hidden;
  const std::size_t oelems =
      (std::size_t)chunk * (std::size_t)(o.matmul ? o.ffn : o.hidden);
  if (f->_rt == nullptr || f->_w == nullptr ||
      f->_rt->input_elems() != elems || f->_rt->output_elems() != oelems) {
    ss->warn(fmt("{}: the ANE feed-forward module could not be built; "
                 "keeping the GPU feed-forward", tag));
    return nullptr;
  }
  f->_session = ss;
  f->_mc      = o.mc;
  f->_tag     = tag;
  f->_hidden  = o.hidden;
  f->_ffn     = o.ffn;
  f->_matmul  = o.matmul;
  f->_gelu    = o.gelu && !o.matmul;
  f->_out_w   = o.matmul ? o.ffn : o.hidden;
  f->_chunk   = chunk;
  f->_chunks  = rows / chunk;
  f->_rows_per_block = rows;
  f->_auto    = !(o.rows > 0.0f);
  f->_profile = o.profile;
  f->_check   = std::getenv("VPIPE_ANE_CHECK") != nullptr;
  ss->info(fmt("{}: ANE {} armed (runtime weights, {} MB IOSurface "
               "slots, module ready in {:.0f} ms), {} of {} rows per block "
               "in {} chunks of {} (GPU takes {})",
               tag, o.matmul ? "projection" : "feed-forward",
               f->_w->bytes() >> 20, build_ms, rows, o.seq,
               rows / chunk, chunk, o.seq - rows));
  return f;
}

AneFeedForward::~AneFeedForward()
{
  // A job may still be reading a block's weights or writing the ANE's
  // rows; neither may outlive the buffers it was handed.
  if (_session == nullptr) { return; }
  worker().join();
  // Give the wired pool back what this tier was charged: the host rows it
  // wired, and the bytes CoreML wired for the module.
  if (auto* mgr = _session->services()->generative_model_manager()) {
    mgr->unwire_from_pool(_in);
    mgr->unwire_from_pool(_out);
    mgr->release_external_wired(_ext_wired);
  }
}

AneWorker&
AneFeedForward::worker() const
{
  return _session->services()->coreml_model_manager()->ane_worker();
}

std::size_t
AneFeedForward::host_bytes() const noexcept
{
  return _in.byte_size() + _out.byte_size();
}

int AneFeedForward::chunk() const noexcept { return _chunk; }
int AneFeedForward::rows_per_block() const noexcept { return _rows_per_block; }
bool AneFeedForward::timing() const noexcept { return _auto || _profile; }
bool AneFeedForward::needs_barrier() const noexcept { return _barrier; }

std::size_t
AneFeedForward::gelu_runtime_bytes(int hidden, int ffn, int chunk) noexcept
{
  if (hidden <= 0 || ffn <= 0) { return 0; }
  const std::size_t h = (std::size_t)hidden, f = (std::size_t)ffn;
  const std::size_t c = (std::size_t)std::max(1, chunk);
  const std::size_t weights = (2 * h * f + f + h) * 2;         // slots
  const std::size_t staging = weights + c * (2 * f + 2 * h) * 2;
  const std::size_t host = c * h * 2 * 2;                      // in + out
  return weights + staging + host;
}

std::size_t
AneFeedForward::runtime_bytes(int hidden, int ffn, int seq, int chunk) noexcept
{
  if (hidden <= 0 || ffn <= 0) { return 0; }
  const std::size_t h = (std::size_t)hidden, f = (std::size_t)ffn;
  const std::size_t c = (std::size_t)std::max(1, chunk);
  (void)seq;       // the host term is one chunk whatever the sequence
  const std::size_t weights = 3 * h * f * 2;                  // slots
  const std::size_t staging = weights + c * (3 * f + 2 * h) * 2;
  const std::size_t host = c * h * 2 * 2;                     // in + out
  return weights + staging + host;
}

std::size_t
AneFeedForward::matmul_runtime_bytes(int in, int out, int seq,
                                     int chunk) noexcept
{
  if (in <= 0 || out <= 0) { return 0; }
  const std::size_t k = (std::size_t)in, n = (std::size_t)out;
  const std::size_t c = (std::size_t)std::max(1, chunk);
  (void)seq;       // the host term is one chunk whatever the sequence
  const std::size_t weights = k * n * 2;
  const std::size_t staging = weights + c * (k + n) * 2;
  // What CoreML holds past the slot and its staged copy: MEASURED 0.54x
  // the weights at 4096 -> 36864 (chunks 2048 and 3072 alike) and 0.58x at
  // 4096 -> 12288, independent of the chunk. Booked at 5/8.
  const std::size_t runtime = weights * 5 / 8;
  return weights + staging + runtime + c * (k + n) * 2;
}

int
AneFeedForward::chunk_rows(const char* env) noexcept
{
  if (env != nullptr) {
    if (const char* ce = std::getenv(env)) {
      const int v = std::atoi(ce);
      if (v >= 256 && v <= 16384) { return v; }
    }
  }
  return kChunkDefault;
}

void
AneFeedForward::stage(int layer, const AneFfnSource& gate,
                      const AneFfnSource& up, const AneFfnSource& down,
                      int group)
{
  stage_(layer, {gate, up, down}, group);
}

void
AneFeedForward::stage(int layer, const AneFfnSource& weight, int group)
{
  stage_(layer, {weight}, group);
}

void
AneFeedForward::stage(int layer, const std::vector<AneFfnSource>& parts,
                      int group)
{
  stage_(layer, parts, group);
}

void
AneFeedForward::stage_(int layer, std::vector<AneFfnSource> sources,
                       int group)
{
  AneWorker& w = worker();
  const std::size_t nsrc = sources.size();
  const std::size_t G = (std::size_t)std::max(1, group);
  w.dispatch([this, &w, src = std::move(sources), nsrc, layer, G]() {
    const auto t0 = Clk::now();
    // The module declares its weights [out,in] -- the checkpoint's own
    // layout -- so every source row maps to one slot row. MEASURED on one
    // 193 MB matrix: bf16 through a table 6.7 ms, 4-bit through a per-group
    // table of its 16 values 7.1 ms, 8-bit computed directly 16.7 ms, all
    // bit-exact -- against 116 ms for the per-element transposing convert
    // that first made staging ~500 ms a block.
    const std::uint16_t* lut = bf16_to_f16_lut_();
    bool ok = true;
    for (std::size_t i = 0; i < nsrc && ok; ++i) {
      const AneFfnSource& q = src[i];
      const std::size_t slot = q.slot >= 0 ? (std::size_t)q.slot : i;
      const std::size_t srow = q.slot_row;
      const std::size_t rows =
          q.rows > 0 ? q.rows
                     : (std::size_t)_w->rows(slot) -
                           std::min(srow, (std::size_t)_w->rows(slot));
      const std::size_t cols = (std::size_t)_w->cols(slot);
      // The up projection carries the fp16 headroom scale; see _up_scale.
      // A scaled or adapted row is built in f32 rather than through the
      // table.
      const float mul = (!_matmul && i == 1) ? 1.0f / _up_scale : 1.0f;
      const bool f32path = q.deltas > 0 || mul != 1.0f;
      ok = q.stageable() && rows > 0 && q.deltas >= 0 &&
           q.deltas <= AneFfnSource::kMaxDeltas &&
           slot < _w->count() &&
           srow + rows <= (std::size_t)_w->rows(slot);
      if (!ok) { break; }
      // The last source row read, plus one.
      const std::size_t srows = q.offset + (rows - 1) * q.stride + 1;
      if (!q.quantized) {
        ok = q.w->contents() != nullptr &&
             q.w->byte_size() >= srows * cols * 2;
      } else {
        const std::size_t per = 32 / (std::size_t)q.bits;
        ok = q.codes->contents() != nullptr && cols % per == 0 &&
             cols % G == 0 &&
             q.codes->byte_size() >= srows * (cols / per) * 4 &&
             q.scales->byte_size() >= srows * (cols / G) * 2 &&
             q.qbias->byte_size() >= srows * (cols / G) * 2;
      }
      // The adapters' A factors in f32, once per projection.
      std::vector<std::vector<float>> a32((std::size_t)q.deltas);
      for (int di = 0; di < q.deltas && ok; ++di) {
        const AneFfnSource::Delta& dl = q.delta[di];
        const std::size_t rk = (std::size_t)std::max(0, dl.rank);
        // A banded pair's A carries every part's rank.
        const std::size_t ark =
            rk * (std::size_t)(dl.group > 0 ? std::max(1, dl.parts) : 1);
        const std::size_t brows =
            dl.b_offset + (rows - 1) * std::max<std::size_t>(1, dl.b_stride) +
            1;
        ok = rk > 0 && dl.a != nullptr && dl.b != nullptr &&
             dl.a->contents() != nullptr && dl.b->contents() != nullptr &&
             dl.a->byte_size() >= ark * cols * 2 &&
             dl.b->byte_size() >= brows * rk * 2;
        if (!ok) { break; }
        a32[(std::size_t)di].resize(ark * cols);
        const auto* ap = static_cast<const std::uint16_t*>(dl.a->contents());
        for (std::size_t k = 0; k < ark * cols; ++k) {
          a32[(std::size_t)di][k] = f32_of_bf16_(ap[k]);
        }
      }
      if (!ok) { break; }
      const unsigned bits = (unsigned)std::max(1, q.bits);
      const std::size_t per = q.quantized ? 32 / bits : 1;
      const std::size_t groups = cols / G;
      const unsigned mask = q.quantized ? (1u << bits) - 1u : 0u;
      // Source row `sr` straight to fp16: bf16 through the table, 4-bit
      // through a per-group table of its 16 values, 8-bit computed directly.
      auto direct_row = [&](std::size_t sr, std::uint8_t* dr) {
        if (!q.quantized) {
          const std::uint16_t* s =
              static_cast<const std::uint16_t*>(q.w->contents()) + sr * cols;
          for (std::size_t cc = 0; cc < cols; ++cc) {
            const std::uint16_t hv = lut[s[cc]];
            std::memcpy(dr + cc * 2, &hv, sizeof(hv));
          }
          return;
        }
        // value = scale * q + bias, in float, narrowed to fp16 -- what the
        // GPU's affine_dequant computes.
        const std::uint32_t* cw =
            static_cast<const std::uint32_t*>(q.codes->contents()) +
            sr * (cols / per);
        const std::uint16_t* sc =
            static_cast<const std::uint16_t*>(q.scales->contents()) +
            sr * groups;
        const std::uint16_t* bi =
            static_cast<const std::uint16_t*>(q.qbias->contents()) +
            sr * groups;
        for (std::size_t g = 0; g < groups; ++g) {
          const float s  = f32_of_bf16_(sc[g]);
          const float bb = f32_of_bf16_(bi[g]);
          if (bits == 4) {
            std::uint16_t tbl[16];
            for (unsigned v = 0; v < 16; ++v) {
              const _Float16 hv = (_Float16)(s * (float)v + bb);
              std::memcpy(&tbl[v], &hv, sizeof(hv));
            }
            for (std::size_t wd = g * G / per; wd < (g + 1) * G / per; ++wd) {
              std::uint32_t word = cw[wd];
              for (std::size_t e = 0; e < per; ++e) {
                std::memcpy(dr + (wd * per + e) * 2, &tbl[word & mask],
                            sizeof(std::uint16_t));
                word >>= bits;
              }
            }
          } else {
            for (std::size_t k = g * G; k < (g + 1) * G; ++k) {
              const unsigned qv = (cw[k / per] >> (bits * (k % per))) & mask;
              const _Float16 hv = (_Float16)(s * (float)qv + bb);
              std::memcpy(dr + k * 2, &hv, sizeof(hv));
            }
          }
        }
      };
      // Source row `sr` into f32, for the rows that are scaled or adapted.
      auto decode_row = [&](std::size_t sr, float* acc) {
        if (!q.quantized) {
          const std::uint16_t* s =
              static_cast<const std::uint16_t*>(q.w->contents()) + sr * cols;
          for (std::size_t cc = 0; cc < cols; ++cc) {
            acc[cc] = f32_of_bf16_(s[cc]);
          }
          return;
        }
        const std::uint32_t* cw =
            static_cast<const std::uint32_t*>(q.codes->contents()) +
            sr * (cols / per);
        const std::uint16_t* sc =
            static_cast<const std::uint16_t*>(q.scales->contents()) +
            sr * groups;
        const std::uint16_t* bi =
            static_cast<const std::uint16_t*>(q.qbias->contents()) +
            sr * groups;
        for (std::size_t g = 0; g < groups; ++g) {
          const float s  = f32_of_bf16_(sc[g]);
          const float bb = f32_of_bf16_(bi[g]);
          for (std::size_t k = g * G; k < (g + 1) * G; ++k) {
            const unsigned qv = (cw[k / per] >> (bits * (k % per))) & mask;
            acc[k] = s * (float)qv + bb;
          }
        }
      };
      ok = _w->fill(slot, [&](void* base, std::size_t bpr) {
        auto* d = static_cast<std::uint8_t*>(base);
        // In row BANDS, so an adapter's delta is one BLAS product per band
        // -- scale * B[band] A, rows x rank x cols on the AMX -- rather than
        // rank row-axpys per row. MEASURED as the per-row loop: 270 ms a
        // MiniMax-H3 block on an M5 Pro and 355-395 on an M4 Pro, which on
        // the M5 is longer than the attention it is meant to hide under.
        constexpr std::size_t kBand = 1024;
        std::vector<float> delta, bsel;
        for (std::size_t b0 = 0; b0 < rows; b0 += kBand) {
          const std::size_t n = std::min(kBand, rows - b0);
          if (q.deltas > 0) {
            delta.assign(n * cols, 0.0f);
            for (int di = 0; di < q.deltas; ++di) {
              const AneFfnSource::Delta& dl = q.delta[di];
              const std::size_t rk = (std::size_t)dl.rank;
              const auto* bp =
                  static_cast<const std::uint16_t*>(dl.b->contents());
              bsel.resize(n * rk);
              for (std::size_t j = 0; j < n; ++j) {
                const std::size_t br =
                    (b0 + j) * std::max<std::size_t>(1, dl.b_stride) +
                    dl.b_offset;
                for (std::size_t k = 0; k < rk; ++k) {
                  bsel[j * rk + k] = f32_of_bf16_(bp[br * rk + k]);
                }
              }
              if (dl.group <= 0 || dl.parts <= 1) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (__LAPACK_int)n, (__LAPACK_int)cols,
                            (__LAPACK_int)rk, dl.scale, bsel.data(),
                            (__LAPACK_int)rk, a32[(std::size_t)di].data(),
                            (__LAPACK_int)cols, 1.0f, delta.data(),
                            (__LAPACK_int)cols);
                continue;
              }
              // Banded: one product per run of rows that share a part,
              // each against that part's rows of A. A run is at least a
              // group long except where the band cuts it.
              auto part_of = [&](std::size_t j) {
                const std::size_t br =
                    (b0 + j) * std::max<std::size_t>(1, dl.b_stride) +
                    dl.b_offset;
                return (br / (std::size_t)dl.group) % (std::size_t)dl.parts;
              };
              for (std::size_t j = 0; j < n;) {
                const std::size_t p = part_of(j);
                std::size_t j2 = j + 1;
                while (j2 < n && part_of(j2) == p) { ++j2; }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (__LAPACK_int)(j2 - j), (__LAPACK_int)cols,
                            (__LAPACK_int)rk, dl.scale,
                            bsel.data() + j * rk, (__LAPACK_int)rk,
                            a32[(std::size_t)di].data() + p * rk * cols,
                            (__LAPACK_int)cols, 1.0f,
                            delta.data() + j * cols, (__LAPACK_int)cols);
                j = j2;
              }
            }
          }
          // Row-granular slices over an ELEMENT count: each row goes to the
          // slice holding its first element, so every row is done once, and
          // the pool's inline threshold -- in elements -- is not tripped by
          // a small row count.
          w.parallel_for(n * cols, [&](std::size_t lo, std::size_t hi) {
            const std::size_t r0 = (lo + cols - 1) / cols;
            const std::size_t r1 = std::min(n, (hi + cols - 1) / cols);
            std::vector<float> acc(f32path ? cols : 0);
            for (std::size_t j = r0; j < r1; ++j) {
              const std::size_t r = b0 + j;
              const std::size_t sr = r * q.stride + q.offset;
              std::uint8_t* dr = d + (srow + r) * bpr;
              if (!f32path) {
                direct_row(sr, dr);
                continue;
              }
              decode_row(sr, acc.data());
              if (q.deltas > 0) {
                const float* dp = delta.data() + j * cols;
                for (std::size_t cc = 0; cc < cols; ++cc) { acc[cc] += dp[cc]; }
              }
              for (std::size_t cc = 0; cc < cols; ++cc) {
                const _Float16 hv = (_Float16)(acc[cc] * mul);
                std::memcpy(dr + cc * 2, &hv, sizeof(hv));
              }
            }
          });
        }
      });
    }
    // The biases, when the module has them: slot 3 + i, one row of the
    // projection's output width.
    for (std::size_t i = 0; i < nsrc && ok && _w->count() > nsrc; ++i) {
      const AneFfnSource& q = src[i];
      const std::size_t n = (std::size_t)_w->cols(nsrc + i);
      const float mul = (!_matmul && i == 1) ? 1.0f / _up_scale : 1.0f;
      const bool have = q.bias != nullptr && !q.bias->empty();
      ok = !have || (q.bias->contents() != nullptr &&
                     q.bias->byte_size() >=
                         (q.offset + (n - 1) * q.stride + 1) * 2);
      ok = ok && _w->fill(nsrc + i, [&](void* base, std::size_t) {
        auto* d = static_cast<std::uint8_t*>(base);
        const auto* bp =
            have ? static_cast<const std::uint16_t*>(q.bias->contents())
                 : nullptr;
        for (std::size_t cc = 0; cc < n; ++cc) {
          const float v =
              bp != nullptr ? f32_of_bf16_(bp[cc * q.stride + q.offset]) * mul
                            : 0.0f;
          const _Float16 hv = (_Float16)v;
          std::memcpy(d + cc * 2, &hv, sizeof(hv));
        }
      });
    }
    _staged   = ok ? layer : -1;
    if (ok) { _slot_scale = _up_scale; }
    _stage_ms = ms_since_(t0);
  });
}

AneFeedForward::Plan
AneFeedForward::plan_block()
{
  if (!_auto) { return Plan::kSplit; }
  const int n = _blocks_seen++;
  Plan p;
  // Four split blocks (the first two warm the module and the rates and are
  // not measured), then two probes, then whichever measured better.
  if (n < 4) {
    p = Plan::kSplit;
  } else if (n < 6) {
    p = Plan::kProbe;
  } else if (++_since_probe >= kReprobeBlocks) {
    _since_probe = 0;
    p = _gpu_mode ? Plan::kSplit : Plan::kProbe;   // re-measure the other
  } else {
    p = _gpu_mode ? Plan::kGpu : Plan::kSplit;
  }
  _barrier   = p != Plan::kGpu && _prev_plan == Plan::kGpu;
  _prev_plan = p;
  return p;
}

void
AneFeedForward::note_probe(double drain_ms, double gpu_ms)
{
  if (!_auto || gpu_ms <= 0.0) { return; }
  const double ms = drain_ms + gpu_ms;
  _probe_ms = push_median_(_probe_s, &_probe_n, ms);
  if (_profile) {
    _session->info(fmt("{}: ANE probe (GPU over every row) | drain {:.2f} | "
                       "gpu {:.2f} | section {:.2f}", _tag, drain_ms, gpu_ms,
                       ms));
  }
  decide_();
}

double
AneFeedForward::push_median_(double* ring, int* n, double v)
{
  ring[*n % kWindow] = v;
  ++*n;
  const int k = std::min(*n, kWindow);
  double tmp[kWindow];
  std::copy(ring, ring + k, tmp);
  std::sort(tmp, tmp + k);
  return (k % 2 == 1) ? tmp[k / 2] : 0.5 * (tmp[k / 2 - 1] + tmp[k / 2]);
}

void
AneFeedForward::decide_()
{
  if (!(_split_ms > 0.0) || !(_probe_ms > 0.0)) { return; }
  const bool was = _gpu_mode;
  if (!_gpu_mode && _probe_ms < 0.97 * _split_ms) { _gpu_mode = true; }
  else if (_gpu_mode && _split_ms < 0.97 * _probe_ms) { _gpu_mode = false; }
  if (was != _gpu_mode) {
    _session->info(fmt("{}: the ANE split measures {:.1f} ms a block section "
                       "against {:.1f} GPU-only -- {}", _tag, _split_ms,
                       _probe_ms,
                       _gpu_mode ? (_matmul ? "running the projection on "
                                              "the GPU"
                                            : "running the feed-forward on "
                                              "the GPU")
                                 : "splitting it with the ANE"));
  }
}

bool
AneFeedForward::join_stage(int layer)
{
  worker().join();
  return _staged == layer;
}

void
AneFeedForward::join()
{
  worker().join();
}

int
AneFeedForward::begin(const metal_compute::SharedBuffer& in,
                      const metal_compute::SharedBuffer& out, int seq)
{
  return begin(in, std::vector<OutSeg>{{&out, _out_w}}, seq);
}

int
AneFeedForward::begin(const metal_compute::SharedBuffer& in,
                      const std::vector<OutSeg>& outs, int seq)
{
  _ok = false;
  _t_cin = _t_pred = _t_cout = 0.0;
  // Whole chunks, and the GPU keeps at least one row: a later forward may
  // be shorter than the one the share was set from.
  const int max_rows = seq > 0 ? ((seq - 1) / _chunk) * _chunk : 0;
  const int a = std::min(_rows_per_block, max_rows);
  _a_rows = 0;
  if (a <= 0) { return 0; }
  // ONE CHUNK of host buffer at each width, reused chunk by chunk. Sized to
  // every ANE row instead, a MiniMax-H3 clip at 1344x768x328 (99k rows)
  // held ~1.2 GB for the feed-forward's fp16 rows and ~5.6 GB for the qkv
  // tier's 21504-wide output -- enough that a 24 GB M5 Pro had to run the
  // clip without the ANE at all. One chunk is ~44 MB and ~88 MB.
  const std::size_t cin_n  = (std::size_t)_chunk * (std::size_t)_hidden;
  const std::size_t cout_n = (std::size_t)_chunk * (std::size_t)_out_w;
  // WIRED THROUGH THE POOL, like a model's own scratch: these rows are read
  // and written every block. A buffer being replaced comes off the pool's
  // books first -- dropping a wired buffer unwires its pages but not the
  // pool's counter.
  auto* pool = _session->services()->generative_model_manager();
  auto grow = [&](metal_compute::SharedBuffer& b, std::size_t bytes) {
    if (b.byte_size() >= bytes) { return; }
    if (pool != nullptr) { pool->unwire_from_pool(b); }
    b = _mc->make_shared_buffer(bytes);
    if (pool != nullptr && !b.empty()) { (void)pool->wire_into_pool(b); }
  };
  grow(_in, cin_n * 2);
  grow(_out, cout_n * 2);
  const void* in_p = in.contents();
  // Each destination: its bf16 base, width, and first column in the
  // module's output row.
  struct Seg {
    std::uint16_t* p = nullptr;
    std::size_t    w = 0, c0 = 0;
    std::size_t    r0 = 0;       // OutSeg::row_offset
  };
  std::vector<Seg> segs;
  std::size_t wsum = 0;
  for (const OutSeg& o : outs) {
    if (o.buf == nullptr || o.buf->contents() == nullptr || o.width <= 0) {
      return 0;
    }
    segs.push_back({static_cast<std::uint16_t*>(o.buf->contents()),
                    (std::size_t)o.width, wsum, o.row_offset});
    wsum += (std::size_t)o.width;
  }
  if (_in.empty() || _out.empty() || in_p == nullptr || segs.empty() ||
      wsum != (std::size_t)_out_w) {
    return 0;
  }
  _a_rows = a;
  const bool tm = timing();
  const std::size_t off  = (std::size_t)(seq - a) * (std::size_t)_hidden;
  const std::size_t row0 = (std::size_t)(seq - a);   // first ANE row
  AneWorker& w = worker();
  w.dispatch([this, &w, in_p, segs, off, row0, cin_n, cout_n, a, tm]() {
    auto* in16 = static_cast<_Float16*>(_in.contents());
    const auto* o16 = static_cast<const _Float16*>(_out.contents());
    const auto* src = static_cast<const std::uint16_t*>(in_p);
    const std::size_t ow = (std::size_t)_out_w;
    auto non_finite = [&]() {
      std::atomic<std::size_t> bad{0};
      w.parallel_for(cout_n, [&](std::size_t lo, std::size_t hi) {
        std::size_t n = 0;
        for (std::size_t i = lo; i < hi; ++i) {
          if (!std::isfinite((float)o16[i])) { ++n; }
        }
        bad += n;
      });
      return bad.load();
    };
    // Divide the up slot (and its bias) by 4 in place: the fp16 headroom
    // retry. See _up_scale.
    auto rescale = [&]() {
      // Slot 1 is up in a SwiGLU {gate, up, down} and down in a GELU
      // {up, down}; either way its bias follows the projections at
      // nproj + 1.
      const std::size_t ur = (std::size_t)_w->rows(1);
      const std::size_t uc = (std::size_t)_w->cols(1);
      const std::size_t nproj = _gelu ? 2 : 3;
      if (_w->count() > nproj) {
        const std::size_t bs = nproj + 1;
        const std::size_t bn = (std::size_t)_w->cols(bs);
        (void)_w->fill(bs, [&](void* base, std::size_t) {
          auto* d = static_cast<std::uint8_t*>(base);
          for (std::size_t cc = 0; cc < bn; ++cc) {
            _Float16 v;
            std::memcpy(&v, d + cc * 2, sizeof(v));
            v = (_Float16)((float)v * 0.25f);
            std::memcpy(d + cc * 2, &v, sizeof(v));
          }
        });
      }
      return _w->fill(1, [&](void* base, std::size_t bpr) {
        auto* d = static_cast<std::uint8_t*>(base);
        w.parallel_for(ur * uc, [&](std::size_t lo, std::size_t hi) {
          const std::size_t r0 = (lo + uc - 1) / uc;
          const std::size_t r1 = std::min(ur, (hi + uc - 1) / uc);
          for (std::size_t r = r0; r < r1; ++r) {
            std::uint8_t* dr = d + r * bpr;
            for (std::size_t cc = 0; cc < uc; ++cc) {
              _Float16 v;
              std::memcpy(&v, dr + cc * 2, sizeof(v));
              v = (_Float16)((float)v * 0.25f);
              std::memcpy(dr + cc * 2, &v, sizeof(v));
            }
          }
        });
      });
    };
    _rescales = 0;
    float mx = 0.0f;
    std::size_t over = 0;
    const int chunks = a / _chunk;
    // Chunk by chunk: in, predict, out. When `in` and `out` are the same
    // buffer (a feed-forward writing where it read) this is still safe: a
    // chunk's rows are all converted in before its outputs land on them,
    // and later chunks read rows no earlier chunk has written.
    for (int ch = 0; ch < chunks; ++ch) {
      const std::size_t io = off + (std::size_t)ch * cin_n;
      const std::size_t rbase = row0 + (std::size_t)ch * (std::size_t)_chunk;
      const auto ta = Clk::now();
      // bf16 -> fp16 in. The DiTs run bf16 (their residual streams
      // overflow f16) while the module declares fp16, so handing the bits
      // over unconverted reinterprets them.
      w.parallel_for(cin_n, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
          in16[i] = (_Float16)f32_of_bf16_(src[io + i]);
        }
      });
      if (_check) {
        for (std::size_t i = 0; i < cin_n; ++i) {
          mx = std::max(mx, std::fabs(f32_of_bf16_(src[io + i])));
          if (!std::isfinite((float)in16[i])) { ++over; }
        }
      }
      if (tm) { _t_cin += ms_since_(ta); }
      const auto tp = Clk::now();
      bool ok = _rt->run(_in, 0, _out, 0, *_w);
      std::size_t bad = ok ? non_finite() : 0;
      // fp16 overflow inside the graph: rescale and predict THIS chunk
      // again. Earlier chunks were converted out at the scale they were
      // predicted with; the new one sticks for the rest.
      while (ok && !_matmul && bad > 0 && _slot_scale < kMaxUpScale) {
        if (!rescale()) { break; }
        _slot_scale *= 4.0f;
        _up_scale = _slot_scale;
        ++_rescales;
        ok = _rt->run(_in, 0, _out, 0, *_w);
        bad = ok ? non_finite() : 0;
      }
      if (tm) { _t_pred += ms_since_(tp); }
      if (!ok || bad > 0) {
        _out_bad = bad;
        return;
      }
      const auto tc = Clk::now();
      const float up = _slot_scale;          // undo the headroom scale
      // Row-granular slices; each row's columns go to their segments.
      w.parallel_for(cout_n, [&](std::size_t lo, std::size_t hi) {
        const std::size_t r0 = (lo + ow - 1) / ow;
        const std::size_t r1 = std::min((std::size_t)_chunk,
                                        (hi + ow - 1) / ow);
        for (std::size_t r = r0; r < r1; ++r) {
          const _Float16* orow = o16 + r * ow;
          for (const Seg& sg : segs) {
            std::uint16_t* d = sg.p + (sg.r0 + rbase + r) * sg.w;
            for (std::size_t cc = 0; cc < sg.w; ++cc) {
              const float f = (float)orow[sg.c0 + cc] * up;
              std::uint32_t bits = 0;
              std::memcpy(&bits, &f, sizeof(bits));
              d[cc] = (std::uint16_t)(bits >> 16);
            }
          }
        }
      });
      if (tm) { _t_cout += ms_since_(tc); }
    }
    _out_bad = 0;
    if (_check) {
      _in_max  = mx;
      _in_over = over;
    }
    _ok = true;
  });
  return a;
}

bool
AneFeedForward::finish(int layer, int seq, double gpu_ms, double drain_ms)
{
  if (_a_rows <= 0) { return true; }
  const auto t_j0 = Clk::now();
  worker().join();
  // The ANE time the GPU did NOT cover -- the only part that lengthens
  // the block.
  const double exposed = ms_since_(t_j0);
  // WHAT COREML WIRED FOR US, charged once, at the first predict that
  // worked -- that is when the IOSurface slots are pinned. System-wide
  // wired grew by the module's pages plus whatever the pool itself wired
  // meanwhile (the model's blocks, these host rows); the difference is the
  // module's. Other processes move the same figure, so it is bounded below
  // by the slots and above by this tier's own estimate.
  if (!_ext_measured && _ok) {
    _ext_measured = true;
    if (auto* pool = _session->services()->generative_model_manager()) {
      const std::size_t now  = _mc->memory_budget().wired;
      const std::size_t used = pool->wired_pool_used();
      const std::size_t grew = now > _wired0 ? now - _wired0 : 0;
      const std::size_t ours = used > _pool0 ? used - _pool0 : 0;
      const std::size_t lo   = _w->bytes();
      const std::size_t hi   = std::max(_estimate, lo);
      std::size_t ext = grew > ours ? grew - ours : 0;
      ext = std::clamp(ext, lo, hi);
      pool->charge_external_wired(ext);
      _ext_wired = ext;
      _session->info(fmt(
          "{}: CoreML wired ~{} MB for the ANE module (system wired +{} MB, "
          "the pool's own +{} MB); charged to the wired pool", _tag,
          ext >> 20, grew >> 20, ours >> 20));
    }
  }
  // The on/off measurement, past the two warm-up blocks: this block's
  // section from the split point on. See plan_block().
  // Not a block that re-predicted after an fp16 overflow: that is a one-off
  // cost of finding the scale, and averaged in it switched an M5 Pro run to
  // the GPU for good while its steady split was ~6% faster.
  if (_auto && _ok && _rescales == 0 && _blocks_seen > 2 && gpu_ms > 0.0) {
    _split_ms = push_median_(_split_s, &_split_n,
                             drain_ms + gpu_ms + exposed);
    decide_();
  }
  const int a_rows = _a_rows;
  _a_rows = 0;
  const int g_rows = seq - a_rows;
  const double ane_tot = _t_cin + _t_pred + _t_cout;
  // ---- rebalance ----------------------------------------------------
  //
  // PREDICT, then move. Stepping one chunk toward whichever engine waited
  // oscillated wherever the balance falls between two chunk counts -- on
  // Krea-2 at 1024^2 it swung 1 <-> 2 every block, exposing ~205 ms of ANE
  // time on each 2-chunk block. So keep a running rate per engine, pick
  // the chunk count whose predicted block time is lowest, and change only
  // for a clear (5%) win.
  if (_auto && _ok && g_rows > 0) {
    const int used = a_rows / _chunk;
    const int max_ch = (seq - 1) / _chunk;       // GPU keeps >= 1 row
    if (used > 0 && gpu_ms > 0.0) {
      const double gr = gpu_ms / (double)g_rows;
      const double ac = ane_tot / (double)used;
      _gpu_row_ms = _gpu_row_ms > 0.0 ? 0.5 * (_gpu_row_ms + gr) : gr;
      _chunk_ms   = _chunk_ms > 0.0 ? 0.5 * (_chunk_ms + ac) : ac;
      auto predicted = [&](int ch) {
        return std::max(_gpu_row_ms * (double)(seq - ch * _chunk),
                        _chunk_ms * (double)ch);
      };
      int best = _chunks;
      for (int ch = 1; ch <= max_ch; ++ch) {
        if (predicted(ch) < predicted(best)) { best = ch; }
      }
      if (best != _chunks && predicted(best) < predicted(_chunks) * 0.95) {
        _chunks = best;
      }
    }
    if (_chunks * _chunk != _rows_per_block) {
      _rows_per_block = _chunks * _chunk;
      if (_profile) {
        _session->info(fmt("{}: ANE rebalance -> {} chunks ({} of {} rows)",
                           _tag, _chunks, _rows_per_block, seq));
      }
    }
  }
  if (_profile) {
    _session->info(
        fmt("{}: ANE L{} stage {:.2f} (under the drain) | drain {:.2f} | "
            "gpu {:.2f} | ane {:.2f} (cin {:.2f} pred {:.2f} cout {:.2f}) | "
            "exposed {:.2f} ({:.0f}% hidden) | block-add {:.2f}",
            _tag, layer, _stage_ms, drain_ms, gpu_ms, ane_tot, _t_cin,
            _t_pred, _t_cout, exposed,
            ane_tot > 0.0 ? 100.0 * (1.0 - exposed / ane_tot) : 0.0,
            drain_ms + gpu_ms + exposed));
  }
  if (_check) {
    _session->info(fmt("{}: ANE L{} check: input max |x| {:.1f}, {} over "
                       "fp16; output {} non-finite of {}", _tag, layer,
                       _in_max, _in_over, _out_bad,
                       (std::size_t)a_rows * (std::size_t)_hidden));
  }
  if (_rescales > 0) {
    _session->info(fmt("{}: ANE L{} overflowed fp16; the up projection now "
                       "stages at 1/{:.0f} and the chunks were predicted "
                       "again ({} retr{})", _tag, layer, _slot_scale,
                       _rescales, _rescales == 1 ? "y" : "ies"));
  }
  if (!_ok) {
    // Both engines have already run; there is no falling back to a GPU
    // path for these rows, so this is loud.
    _session->warn(fmt("{}: ANE feed-forward failed on block {} ({} "
                       "non-finite outputs); those rows are not computed",
                       _tag, layer, _out_bad));
  }
  return _ok;
}

}  // namespace genai
}  // namespace vpipe
