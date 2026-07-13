#include "generative-models/shared/i8-gemm.h"

#include <cstdlib>

namespace vpipe {
namespace genai {

using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

I8GemmContext::I8GemmContext(MetalCompute* mc, bool want) : _mc(mc)
{
  bool on = want;
  if (const char* e = std::getenv("VPIPE_I8_GEMM")) {
    on = std::atoi(e) != 0;                    // env overrides either way
  }
  if (!on || mc == nullptr || !mc->supports_matrix_cores()) { return; }
  _lib_q = mc->load_library("affine_dequant");
  _lib_g = mc->load_library("dense_gemm_mma");
  _fn_quant = _lib_q.function("quant_f16_i8_row_g512");
  _fn_gemm = _lib_g.function("gemm_i8i8_sc_f16_n64_g512");
  _on = _fn_quant.valid() && _fn_gemm.valid();
  if (const char* e = std::getenv("VPIPE_I8_GEMM_MIN_M")) {
    _min_m = std::atoi(e);
  }
}

bool
I8GemmContext::gemm(ComputeEncoder& enc, const SharedBuffer& x,
                    std::size_t xe, const SharedBuffer& w,
                    const SharedBuffer& y, std::size_t ye,
                    int M, int N, int K)
{
  if (!accepts(M, N, K)) { return false; }
  const int G = K / 512;
  auto need = [&](SharedBuffer& b, std::size_t bytes) {
    if (b.empty() || b.byte_size() < bytes) {
      b = _mc->make_shared_buffer(bytes);
    }
    return !b.empty();
  };
  if (!need(_xq, (std::size_t)M * K) ||
      !need(_as, (std::size_t)M * G * 2) ||
      !need(_wq, (std::size_t)N * K) ||
      !need(_ws, (std::size_t)N * G * 2)) {
    return false;
  }
  // Activation rows -> i8 + per-(row, group) scales.
  enc.set_function(_fn_quant);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, _xq);
  enc.set_buffer(2, _as);
  enc.set_constant(3, K);
  enc.dispatch({256u, (unsigned)M, 1}, {256, 1, 1});
  // Weight rows (the [N,K] f16 dense weight) -> i8 + per-(chan, group)
  // scales. Re-quantized per call: one extra weight pass, no persistent
  // copy. Serial command streams + Metal's hazard tracking make the
  // shared scratches safe across back-to-back GEMMs.
  enc.set_function(_fn_quant);
  enc.set_buffer(0, w);
  enc.set_buffer(1, _wq);
  enc.set_buffer(2, _ws);
  enc.set_constant(3, K);
  enc.dispatch({256u, (unsigned)N, 1}, {256, 1, 1});
  // i8 x i8 GEMM, per-group f32 accumulate, f16 store.
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, _xq);
  enc.set_buffer(1, _wq);
  enc.set_buffer(2, _as);
  enc.set_buffer(3, _ws);
  enc.set_buffer(4, y, ye * 2);
  enc.set_constant(5, K);
  enc.set_constant(6, N);
  enc.set_constant(7, M);
  enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
  return true;
}

void
I8GemmContext::release_scratch()
{
  _xq = {};
  _as = {};
  _wq = {};
  _ws = {};
}

}  // namespace genai
}  // namespace vpipe
