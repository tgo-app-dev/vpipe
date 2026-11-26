// The kernel contract (apple-silicon/metal-compute/kernel-contract.h) held
// to the metallibs this build embeds.
//
// Every contracted entry point must RESOLVE in the library the contract
// names -- a name that does not exist hands back an invalid function,
// which is the failure class that once shipped two image families
// dispatching no kernel at all -- and the parameter block a plugin fills
// for the steel attention kernels must match the kernel's own struct
// field for field.

#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/kernel-contract.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/shared/sage-attention.h"
#include "interfaces/session-services-intf.h"

#include "gpu-kernels/metal/vendored/mlx/backend/metal/kernels/steel/attn/params.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

#define SAME_FIELD(f)                                                        \
  static_assert(offsetof(contract::SteelAttnParams, f) ==                   \
                    offsetof(mlx::steel::AttnParams, f) &&                   \
                sizeof(contract::SteelAttnParams::f) ==                      \
                    sizeof(mlx::steel::AttnParams::f),                       \
                "kernel contract: SteelAttnParams::" #f " moved")
SAME_FIELD(B);          SAME_FIELD(H);          SAME_FIELD(D);
SAME_FIELD(qL);         SAME_FIELD(kL);         SAME_FIELD(gqa_factor);
SAME_FIELD(scale);      SAME_FIELD(NQ);         SAME_FIELD(NK);
SAME_FIELD(NQ_aligned); SAME_FIELD(NK_aligned); SAME_FIELD(qL_rem);
SAME_FIELD(kL_rem);     SAME_FIELD(qL_off);     SAME_FIELD(Q_strides);
SAME_FIELD(K_strides);  SAME_FIELD(V_strides);  SAME_FIELD(O_strides);
#undef SAME_FIELD
static_assert(sizeof(contract::SteelAttnParams) ==
                  sizeof(mlx::steel::AttnParams),
              "kernel contract: SteelAttnParams size");
// The toolkit's Sage spells the constants it sets on the contracted
// flash entries itself (sage-attention.h); they must be the contract's.
static_assert(genai::sage::kQkInt8Constant == contract::kAttnQkInt8 &&
                  genai::sage::kPvInt8Constant == contract::kAttnPvInt8 &&
                  genai::sage::kQTokenScaleConstant ==
                      contract::kAttnQTokenScale &&
                  genai::sage::kSinkShiftConstant ==
                      contract::kAttnSinkShift,
              "kernel contract: Sage's function constants");

bool
is_steel_(std::string_view lib)
{
  return lib == "attn_steel" || lib == "attn_steel_nax";
}

}  // namespace

TEST(kernel_contract, every_contracted_kernel_resolves)
{
  Session sess;
  MetalCompute* mc = sess.services()->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  int checked = 0;
  for (const contract::Kernel& k : contract::kKernels) {
    // The NAX twin needs the matrix cores to BUILD; its absence on an M4
    // is the host's own business, not the contract's.
    if (k.library == "attn_steel_nax" && !mc->supports_matrix_cores()) {
      continue;
    }
    const ComputeLibrary lib = mc->load_library(k.library);
    EXPECT_TRUE(lib.valid());
    if (!lib.valid()) {
      std::printf("[contract] library '%s' did not load\n",
                  std::string(k.library).c_str());
      continue;
    }
    ComputeFunction fn;
    if (is_steel_(k.library)) {
      FunctionConstants fc;
      fc.set_bool(contract::kAttnAlignQ, true)
          .set_bool(contract::kAttnAlignK, true)
          .set_bool(contract::kAttnHasMask, false)
          .set_bool(contract::kAttnCausal, false)
          .set_bool(contract::kAttnSinks, false)
          .set_bool(contract::kAttnQkInt8, false);
      fn = lib.function(k.entry, fc);
    } else {
      fn = lib.function(k.entry);
    }
    EXPECT_TRUE(fn.valid());
    if (!fn.valid()) {
      std::printf("[contract] %s :: %s does not resolve\n",
                  std::string(k.library).c_str(),
                  std::string(k.entry).c_str());
    }
    ++checked;
    // The matrix-core entries take SageAttention's int8 QK and its int8
    // P*V too (kAttnPvInt8 / kAttnQTokenScale on top of kAttnQkInt8),
    // and a toolkit specialises them so: each must BUILD that way.
    if (k.library == "attn_steel_nax") {
      FunctionConstants fc;
      fc.set_bool(contract::kAttnAlignQ, true)
          .set_bool(contract::kAttnAlignK, true)
          .set_bool(contract::kAttnHasMask, false)
          .set_bool(contract::kAttnCausal, false)
          .set_bool(contract::kAttnSinks, false)
          .set_bool(contract::kAttnQkInt8, true)
          .set_bool(contract::kAttnPvInt8, true)
          .set_bool(contract::kAttnQTokenScale, true);
      const bool ok = lib.function(k.entry, fc).valid();
      EXPECT_TRUE(ok);
      if (!ok) {
        std::printf("[contract] %s :: %s does not build with the int8 "
                    "P*V\n", std::string(k.library).c_str(),
                    std::string(k.entry).c_str());
      }
      ++checked;
      // ...and an attention WITH SINKS under both: the sinks shifted as
      // the smoothed keys are (kAttnSinkShift), the int8 QK alone and
      // with its P*V.
      for (const bool pv8 : {false, true}) {
        FunctionConstants sf;
        sf.set_bool(contract::kAttnAlignQ, true)
            .set_bool(contract::kAttnAlignK, true)
            .set_bool(contract::kAttnHasMask, false)
            .set_bool(contract::kAttnCausal, false)
            .set_bool(contract::kAttnSinks, true)
            .set_bool(contract::kAttnQkInt8, true)
            .set_bool(contract::kAttnSinkShift, true)
            .set_bool(contract::kAttnPvInt8, pv8)
            .set_bool(contract::kAttnQTokenScale, pv8);
        const bool sok = lib.function(k.entry, sf).valid();
        EXPECT_TRUE(sok);
        if (!sok) {
          std::printf("[contract] %s :: %s does not build with sinks "
                      "under the int8 QK%s\n",
                      std::string(k.library).c_str(),
                      std::string(k.entry).c_str(), pv8 ? " and P*V" : "");
        }
        ++checked;
      }
    }
  }
  for (const contract::QuantFamily& q : contract::kQuantFamilies) {
    const ComputeLibrary lib = mc->load_library(q.library);
    EXPECT_TRUE(lib.valid());
    if (!lib.valid()) { continue; }
    for (int b : q.bits) {
      for (int g : q.groups) {
        std::string name = std::string(q.prefix) + "w" +
                           std::to_string(b) + "g" + std::to_string(g);
        std::vector<std::string> names = {name};
        if (!q.wide.empty() && g == q.wide_group) {
          names.push_back(name + std::string(q.wide));
        }
        for (const std::string& n : names) {
          const bool ok = lib.function(n).valid();
          EXPECT_TRUE(ok);
          if (!ok) {
            std::printf("[contract] %s :: %s does not resolve\n",
                        std::string(q.library).c_str(), n.c_str());
          }
          ++checked;
        }
      }
    }
  }
  for (const contract::ToolkitKernel& k : contract::kToolkitKernels) {
    std::vector<std::string> libs = {std::string(k.library)};
    if (k.twins) { libs.push_back(std::string(k.library) + "_bf16"); }
    for (const std::string& ln : libs) {
      const ComputeLibrary lib = mc->load_library(ln);
      const bool ok = lib.valid() && lib.function(k.entry).valid();
      EXPECT_TRUE(ok);
      if (!ok) {
        std::printf("[contract] toolkit %s :: %s does not resolve\n",
                    ln.c_str(), std::string(k.entry).c_str());
      }
      ++checked;
    }
  }
  std::printf("[contract] revision %d: %d entry points resolve\n",
              contract::kRevision, checked);
  EXPECT_TRUE(checked > 40);
}

// THE SINKS MARKER is the entry it says it is: dispatched, it writes the
// constant whose presence it marks (kAttnSinkShift), in both twins -- so
// a toolkit that finds the name finds this kernel, not a name that
// happened to resolve.
TEST(kernel_contract, the_sinks_marker_names_its_constant)
{
  Session sess;
  MetalCompute* mc = sess.services()->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  for (const char* ln : {"sage_quant", "sage_quant_bf16"}) {
    const ComputeLibrary lib = mc->load_library(ln);
    ComputeFunction fn;
    if (lib.valid()) { fn = lib.function("sage_sinks_marker"); }
    ASSERT_TRUE(fn.valid());
    if (!fn.valid()) { return; }
    SharedBuffer out = mc->make_shared_buffer(4);
    ASSERT_TRUE(!out.empty());
    if (out.empty()) { return; }
    *static_cast<std::uint32_t*>(out.contents()) = 0;
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      e.set_function(fn);
      e.set_buffer(0, out);
      e.dispatch({1, 1, 1}, {1, 1, 1});
    }
    std::string err;
    ASSERT_TRUE(st.commit().wait_ok(&err));
    const std::uint32_t got = *static_cast<std::uint32_t*>(out.contents());
    std::printf("[contract] %s :: sage_sinks_marker wrote %u\n", ln,
                (unsigned)got);
    EXPECT_TRUE(got == (std::uint32_t)contract::kAttnSinkShift);
  }
}
