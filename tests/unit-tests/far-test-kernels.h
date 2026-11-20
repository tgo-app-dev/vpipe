// far-test-kernels.h -- a far side for the tests: memory standing in for
// another machine's (generative-models/far-memory.h), and far-compute/2
// run by vpipe's own DeepSeek-V4.1 decode kernels on its bytes -- so a
// forward with far compute is the local forward bit for bit.
//
// RECORD AND REPLAY. A far side running kernels shares the test's GPU,
// which the GPU <-> host handshake's spin holds: such a forward runs a
// commit and wait a layer. To test the handshake's own path, a run that
// computes RECORDS each call's rows, keyed by what they are made of (the
// expert's far bytes, the row's activation, its weight), and a later run
// with the handshake on REPLAYS them -- no GPU work -- a call not recorded
// a false return (far compute then off: count `replayed`).
#pragma once

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/far-memory.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vpipe::genai::far_test {

// a ggml block's bytes for kFarExpertsDeepseekGguf's types (256 values)
inline std::uint64_t
gguf_block_bytes(std::uint32_t t)
{
  return t == 16 ? 66 : t == 17 ? 74 : t == 22 ? 82 : t == 18 ? 98
         : t == 29 ? 56 : 0;
}

// part k of an expert in form `f`, in bytes
inline std::uint64_t
part_bytes(const FarExpertForm& f, int k)
{
  const std::uint64_t cells = (std::uint64_t)f.hidden * f.inter;
  if (f.format == kFarExpertsDeepseekGguf) {
    if (k % 2 != 0) { return 0; }
    return cells / 256 * gguf_block_bytes((f.arg[2] >> (8 * (k / 2))) & 0xff);
  }
  return k % 2 == 0 ? cells / 2 : cells / 32;
}

// The recorded rows, by what they are made of.
struct Recording {
  std::mutex mu;
  std::map<std::uint64_t, std::vector<std::uint16_t>> rows;
};

class KernelFar final : public FarMemory, public FarCompute {
public:
  enum class Mode { kMemory, kCompute, kReplay };

  KernelFar(metal_compute::MetalCompute* mc, std::size_t bytes, Mode mode,
            std::shared_ptr<Recording> rec = nullptr)
    : _mc(mc), _mode(mode), _rec(std::move(rec))
  {
    _mem = mc->make_shared_buffer(bytes);
    _stream = mc->make_command_stream();
  }

  std::uint64_t bytes() const override { return _mem.byte_size(); }
  bool read(const FarPiece* p, std::size_t n, std::string*) override
  {
    auto* m = static_cast<std::uint8_t*>(_mem.contents());
    for (std::size_t i = 0; i < n; ++i) {
      std::memcpy(p[i].local, m + p[i].offset, p[i].bytes);
    }
    return true;
  }
  bool write(const FarPiece* p, std::size_t n, std::string*) override
  {
    auto* m = static_cast<std::uint8_t*>(_mem.contents());
    for (std::size_t i = 0; i < n; ++i) {
      std::memcpy(m + p[i].offset, p[i].local, p[i].bytes);
    }
    return true;
  }
  void* query_extension(std::string_view id) noexcept override
  {
    return _mode != Mode::kMemory && id == kFarComputeExtension
               ? static_cast<FarCompute*>(this)
               : nullptr;
  }

  bool experts(const FarExpertForm& f, const void* x,
               const FarExpertCall* calls, std::size_t n, void* out,
               std::string* err) override
  {
    const std::size_t H = f.hidden;
    auto* o = static_cast<std::uint16_t*>(out);
    if (_mode == Mode::kReplay) {
      std::lock_guard<std::mutex> lk(_rec->mu);
      for (std::size_t j = 0; j < n; ++j) {
        const auto it = _rec->rows.find(key_(f, x, calls[j]));
        if (it == _rec->rows.end()) {
          ++missed;
          if (err != nullptr) { *err = "a call not recorded"; }
          return false;
        }
        std::memcpy(o + j * H, it->second.data(), H * 2);
      }
      replayed.fetch_add((int)n);
      return true;
    }
    // (the model calls one at a time; the lock keeps a test honest if a
    // late call ever overlapped the next)
    std::lock_guard<std::mutex> run_lock(_run_mu);
    if (!run_(f, x, calls, n, out, err)) { return false; }
    if (_rec) {
      std::lock_guard<std::mutex> lk(_rec->mu);
      for (std::size_t j = 0; j < n; ++j) {
        _rec->rows[key_(f, x, calls[j])].assign(o + j * H, o + (j + 1) * H);
      }
    }
    ran.fetch_add((int)n);
    return true;
  }

  std::atomic<int> ran{0}, replayed{0}, missed{0};

private:
  // FNV-1a over the expert's far bytes, the row and the weight
  std::uint64_t key_(const FarExpertForm& f, const void* x,
                     const FarExpertCall& c) const
  {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* p, std::size_t b) {
      const auto* u = static_cast<const std::uint8_t*>(p);
      for (std::size_t i = 0; i < b; ++i) {
        h = (h ^ u[i]) * 1099511628211ull;
      }
    };
    const auto* m = static_cast<const std::uint8_t*>(_mem.contents());
    for (int k = 0; k < 6; ++k) {
      mix(m + c.part[k], (std::size_t)part_bytes(f, k));
    }
    mix(x, (std::size_t)f.hidden * 2);
    mix(&c.weight, 4);
    mix(&f.format, 4);
    return h;
  }

  bool run_(const FarExpertForm& f, const void* x,
            const FarExpertCall* calls, std::size_t n, void* out,
            std::string* err)
  {
    using metal_compute::ComputeEncoder;
    const bool gguf = f.format == kFarExpertsDeepseekGguf;
    if ((f.format != kFarExpertsDeepseekFp4 && !gguf) || f.dtype > 1) {
      if (err != nullptr) { *err = "a form this test does not run"; }
      return false;
    }
    if (!_fgu.valid()) {
      _lib = _mc->load_library(f.dtype == 1 ? "deepseek_v41_bf16"
                                            : "deepseek_v41");
      _fgu = _lib.function(gguf ? "dsv41_moe_swiglu_iq5_s4_r2"
                                : "dsv41_moe_swiglu_fp4");
      _fdn = _lib.function(gguf ? "dsv41_moe_down_iq5_s4_r2"
                                : "dsv41_moe_down_fp4");
      _x = _mc->make_shared_buffer((std::size_t)f.hidden * 2);
      _act = _mc->make_shared_buffer((std::size_t)f.inter * 2);
      _out = _mc->make_shared_buffer((std::size_t)8 * f.hidden * 2);
      _slot = _mc->make_shared_buffer(4);
      _wt = _mc->make_shared_buffer(4);
      *static_cast<std::int32_t*>(_slot.contents()) = 0;
    }
    const int H = (int)f.hidden, I = (int)f.inter;
    std::uint64_t tab = 0;
    if (gguf) {
      // the tables behind their head (version 1's)
      const std::uint64_t at =
          (std::uint64_t)f.arg[0] | ((std::uint64_t)f.arg[1] << 32);
      FarGgufTables head;
      std::memcpy(&head, static_cast<const std::uint8_t*>(_mem.contents()) +
                             at, sizeof head);
      if (head.magic != FarGgufTables{}.magic || head.version != 1 ||
          head.words != kFarGgufTablesV1Words) {
        if (err != nullptr) { *err = "the tables are not version 1's"; }
        return false;
      }
      tab = at + sizeof head;
    }
    const std::uint32_t t1 = f.arg[2] & 0xff, t3 = (f.arg[2] >> 8) & 0xff,
                        t2 = (f.arg[2] >> 16) & 0xff;
    std::memcpy(_x.contents(), x, (std::size_t)H * 2);
    for (std::size_t j = 0; j < n; ++j) {
      *static_cast<float*>(_wt.contents()) = calls[j].weight;
      ComputeEncoder e = _stream.begin_compute();
      e.set_function(_fgu);
      if (gguf) {
        // as the model's decode dispatches them (one pair, slot 0 of
        // the parts' own offsets)
        const long cw = 0;
        e.set_buffer(0, _mem, calls[j].part[0]);
        e.set_buffer(1, _mem, calls[j].part[2]);
        e.set_buffer(2, _mem, tab);
        e.set_buffer(3, _x);
        e.set_buffer(4, _act);
        e.set_buffer(5, _slot);
        e.set_buffer(6, _wt);
        e.set_constant(7, H);
        e.set_constant(8, I);
        e.set_constant(9, 1);
        e.set_constant(10, f.swiglu_limit);
        e.set_constant(11, t1);
        e.set_constant(12, t3);
        e.set_constant(13, cw);
        e.dispatch({32, (unsigned)(((I + 7) / 8) * 4), 1}, {32, 4, 1});
        e.set_function(_fdn);
        e.set_buffer(0, _mem, calls[j].part[4]);
        e.set_buffer(1, _mem, tab);
        e.set_buffer(2, _act);
        e.set_buffer(3, _out, j * (std::size_t)H * 2);
        e.set_buffer(4, _slot);
        e.set_constant(5, I);
        e.set_constant(6, H);
        e.set_constant(7, t2);
        e.set_constant(8, cw);
        e.dispatch({32, (unsigned)(((H + 7) / 8) * 4), 1}, {32, 4, 1});
      } else {
        for (int k = 0; k < 4; ++k) {
          e.set_buffer((unsigned)k, _mem, calls[j].part[k]);
        }
        e.set_buffer(4, _x);
        e.set_buffer(5, _act);
        e.set_buffer(6, _slot);
        e.set_buffer(7, _wt);
        e.set_constant(8, H);
        e.set_constant(9, I);
        e.set_constant(10, 1);
        e.set_constant(11, f.swiglu_limit);
        e.dispatch({32, (unsigned)((I / 4) * 2), 1}, {32, 2, 1});
        e.set_function(_fdn);
        e.set_buffer(0, _mem, calls[j].part[4]);
        e.set_buffer(1, _mem, calls[j].part[5]);
        e.set_buffer(2, _act);
        e.set_buffer(3, _out, j * (std::size_t)H * 2);
        e.set_buffer(4, _slot);
        e.set_constant(5, I);
        e.set_constant(6, H);
        e.dispatch({32, (unsigned)((H / 8) * 2), 1}, {32, 2, 1});
      }
      e.end();
      std::string why;
      if (!_stream.commit().wait_ok(&why)) {
        if (err != nullptr) { *err = why; }
        return false;
      }
    }
    std::memcpy(out, _out.contents(), n * (std::size_t)H * 2);
    return true;
  }

  metal_compute::MetalCompute* _mc;
  std::mutex _run_mu;
  Mode _mode;
  std::shared_ptr<Recording> _rec;
  metal_compute::SharedBuffer _mem, _x, _act, _out, _slot, _wt;
  metal_compute::CommandStream _stream;
  metal_compute::ComputeLibrary _lib;
  metal_compute::ComputeFunction _fgu, _fdn;
};

}  // namespace vpipe::genai::far_test
