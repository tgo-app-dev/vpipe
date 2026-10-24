#include "generative-models/quantize/affine-quantizer.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <string>

namespace vpipe::genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {
inline int bidx_(int bits)  { return bits == 8 ? 0 : 1; }
inline int gidx_(int group) { return group == 64 ? 0 : 1; }
}  // namespace

AffineQuantizer::AffineQuantizer(MetalCompute* mc) : _mc(mc)
{
  if (_mc == nullptr) { return; }
  // The affine_quant_* entry points live in the affine_dequant metallib;
  // the _bf16 twin compiles them with VPIPE_ELT=bfloat (bf16 weight input).
  _lib      = _mc->load_library("affine_dequant");
  _lib_bf16 = _mc->load_library("affine_dequant_bf16");
  if (!_lib.valid() || !_lib_bf16.valid()) { return; }

  struct { int bits, group; const char* name; } k[] = {
    {8, 64, "affine_quant_w8_g64"}, {4, 64, "affine_quant_w4_g64"},
    {8, 32, "affine_quant_w8_g32"}, {4, 32, "affine_quant_w4_g32"},
  };
  bool ok = true;
  for (const auto& e : k) {
    _fn[0][bidx_(e.bits)][gidx_(e.group)] = _lib.function(e.name);
    _fn[1][bidx_(e.bits)][gidx_(e.group)] = _lib_bf16.function(e.name);
    ok = ok && _fn[0][bidx_(e.bits)][gidx_(e.group)].valid() &&
         _fn[1][bidx_(e.bits)][gidx_(e.group)].valid();
    // The fitted twins, scales written f16 / bf16 (affine_quant_fit_*):
    // checked when asked for, so a library without them still serves kF16.
    const std::string fit =
        std::string("affine_quant_fit_") + (e.name + 13);   // "w8_g64"
    for (int sb = 0; sb < 2; ++sb) {
      const std::string nm = fit + (sb ? "_bf16s" : "_f16s");
      _fn_fit[0][bidx_(e.bits)][gidx_(e.group)][sb] =
          _lib.function(nm.c_str());
      _fn_fit[1][bidx_(e.bits)][gidx_(e.group)][sb] =
          _lib_bf16.function(nm.c_str());
    }
  }
  _ok = ok;
}

const ComputeFunction*
AffineQuantizer::fn_(bool bf16, int bits, int group, Scales sc) const
{
  if ((bits != 8 && bits != 4) || (group != 64 && group != 32)) {
    return nullptr;
  }
  if (sc == Scales::kF16) {
    return &_fn[bf16 ? 1 : 0][bidx_(bits)][gidx_(group)];
  }
  return &_fn_fit[bf16 ? 1 : 0][bidx_(bits)][gidx_(group)]
                 [sc == Scales::kFitBf16 ? 1 : 0];
}

bool
AffineQuantizer::quantize_linear(const SharedBuffer& w_in, bool src_bf16,
                                 int N, int K, int bits, int group, float clip,
                                 SharedBuffer& w_out, SharedBuffer& s_out,
                                 SharedBuffer& b_out, Scales sc) const
{
  if (!_ok || N <= 0 || K <= 0 || (K % group) != 0) { return false; }
  const ComputeFunction* fn = fn_(src_bf16, bits, group, sc);
  if (fn == nullptr || !fn->valid()) { return false; }

  w_out = _mc->make_shared_buffer(weight_bytes(N, K, bits));
  s_out = _mc->make_shared_buffer(scale_bytes(N, K, group));
  b_out = _mc->make_shared_buffer(scale_bytes(N, K, group));
  if (w_out.empty() || s_out.empty() || b_out.empty()) { return false; }

  CommandStream stream = _mc->make_command_stream();
  if (!stream.valid()) { return false; }
  {
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(*fn);
    enc.set_buffer(0, w_in);
    enc.set_buffer(1, w_out);
    enc.set_buffer(2, s_out);
    enc.set_buffer(3, b_out);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, clip);
    // One thread per group: grid {K/group, N, 1}.
    enc.dispatch({(unsigned)(K / group), (unsigned)N, 1}, {64, 1, 1});
  }
  stream.commit().wait();
  return true;
}

}  // namespace vpipe::genai
