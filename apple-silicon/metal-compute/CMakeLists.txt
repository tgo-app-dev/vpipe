# apple-silicon/metal-compute: CUDA-shape Metal compute kernel
# framework. Sibling of metal-runtime/ -- the five legacy inline
# kernels stay in MetalRuntime; new kernels declared via
# add_vpipe_metal_kernel() (CMake helper added in a later step)
# accumulate into MetalCompute as embedded metallib bytes.
#
# Apple-Silicon only; gated by VPIPE_BUILD_APPLE_SILICON in the
# parent apple-silicon/CMakeLists.txt.

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/vpipe-metal-kernel.cmake)

# The Metal kernel sources live in the top-level gpu-kernels/metal/ tree
# (grouped by function: attention/, quant/, gemm/, gated-delta/, encoders/,
# ops/, image-processing/, demos/), with the vendored MLX steel headers under
# gpu-kernels/metal/vendored/. add_vpipe_metal_kernel() resolves SRC/DEPENDS
# and its -I include roots against this dir.
set(VPIPE_METAL_KERNEL_DIR ${CMAKE_SOURCE_DIR}/gpu-kernels/metal)

set(LOCAL_SOURCES
  command-stream.cc
  compute-encoder.cc
  compute-library.cc
  event.cc
  fence.cc
  image-ops.cc
  metal-compute.cc
  shared-buffer.cc
  tensor-beat-bridge.cc
  texture.cc
  texture-cv-bridge.mm
)
set(LOCAL_HEADERS
  buffer-view.h
  command-stream.h
  compute-encoder.h
  compute-library.h
  event.h
  fence.h
  image-ops.h
  metal-compute.h
  shared-buffer.h
  tensor-beat-bridge.h
  texture.h
)

list(TRANSFORM LOCAL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND VPIPE_SOURCES ${LOCAL_SOURCES})
list(APPEND VPIPE_HEADERS ${LOCAL_HEADERS})

# Metal kernel TUs. Each call compiles a .metal source via xcrun and
# packages the generated <name>_embed.cc as an OBJECT library whose
# .o is linked into libvpipe at root CMakeLists time.
add_vpipe_metal_kernel(noop  SRC demos/noop.metal)
add_vpipe_metal_kernel(saxpy SRC demos/saxpy.metal
                             DEPENDS vpipe_tensor_view.metal)
add_vpipe_metal_kernel(texture_read_pixel
                       SRC demos/texture_read_pixel.metal)
add_vpipe_metal_kernel(specialized_add
                       SRC demos/specialized_add.metal)
add_vpipe_metal_kernel(fence_demo
                       SRC demos/fence_demo.metal)
add_vpipe_metal_kernel(nv12_read
                       SRC image-processing/nv12_read.metal)
add_vpipe_metal_kernel(concurrent_demo
                       SRC demos/concurrent_demo.metal)

# LLM inference kernels (procedural transformer on metal-compute).
add_vpipe_metal_kernel(affine_qmv
                       SRC quant/affine_qmv.metal)
add_vpipe_metal_kernel(affine_qmv_bf16
                       SRC quant/affine_qmv.metal DEFINES VPIPE_ELT=bfloat)
add_vpipe_metal_kernel(affine_qmm
                       SRC quant/affine_qmm.metal)
add_vpipe_metal_kernel(rms_norm SRC ops/rms_norm.metal)
add_vpipe_metal_kernel(rms_norm_bf16 SRC ops/rms_norm.metal
                       DEFINES VPIPE_ELT=bfloat)
add_vpipe_metal_kernel(llm_elementwise SRC ops/llm_elementwise.metal)
add_vpipe_metal_kernel(llm_elementwise_bf16 SRC ops/llm_elementwise.metal
                       DEFINES VPIPE_ELT=bfloat)
add_vpipe_metal_kernel(rope SRC ops/rope.metal)
add_vpipe_metal_kernel(rope_bf16 SRC ops/rope.metal
                       DEFINES VPIPE_ELT=bfloat)
add_vpipe_metal_kernel(sdpa SRC attention/sdpa.metal)
add_vpipe_metal_kernel(sdpa_bf16 SRC attention/sdpa.metal
                       DEFINES VPIPE_ELT=bfloat VPIPE_BF16=1)
add_vpipe_metal_kernel(qwen3_5_gated_delta
                       SRC gated-delta/qwen3_5_gated_delta.metal)
add_vpipe_metal_kernel(qwen3_5_gated_delta_bf16
                       SRC gated-delta/qwen3_5_gated_delta.metal
                       DEFINES VPIPE_ELT=bfloat)
add_vpipe_metal_kernel(qwen3_5_gdn_wy
                       SRC gated-delta/qwen3_5_gdn_wy.metal)
add_vpipe_metal_kernel(dense_gemm SRC gemm/dense_gemm.metal)
add_vpipe_metal_kernel(dense_gemm_bf16 SRC gemm/dense_gemm.metal
                       DEFINES VPIPE_ELT=bfloat)
# The matrix-core (NAX / matmul2d) kernels MISCOMPILE on M5 unless built with
# deployment target >= 26.2 (mlx #3622: "NAX requires MACOSX_DEPLOYMENT_TARGET=
# 26.2"). Without it the matmul2d codegen is wrong (e.g. attn_steel_nax was
# ~400% off the steel reference; the flag took it to ~0.6%). Gate the flag on
# the host SDK so boxes without a 26.2 SDK still build (their NAX path is
# capability-gated off at runtime anyway). Applied as FLAGS to every metal4.0
# kernel below.
execute_process(
    COMMAND xcrun -sdk macosx --show-sdk-version
    OUTPUT_VARIABLE _VPIPE_SDK_VER OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(_VPIPE_SDK_VER VERSION_GREATER_EQUAL 26.2)
  set(VPIPE_NAX_MIN_OS -mmacosx-version-min=26.2)
else()
  set(VPIPE_NAX_MIN_OS "")
endif()
message(STATUS
    "vpipe NAX kernels: deployment-target flag '${VPIPE_NAX_MIN_OS}' "
    "(macOS SDK ${_VPIPE_SDK_VER})")

# Matrix-core (M5+) dense GEMM via MetalPerformancePrimitives matmul2d.
# Needs -std=metal4.0 for the tensor / cooperative_tensor intrinsics. The
# metallib is embedded unconditionally but only bound when a runtime
# GPU-capability check reports matrix-core support (older GPUs keep the
# steel dense_gemm path), so the same source stays peak on every chip.
add_vpipe_metal_kernel(dense_gemm_mma SRC gemm/dense_gemm_mma.metal
                       STD metal4.0 FLAGS ${VPIPE_NAX_MIN_OS})
add_vpipe_metal_kernel(dense_gemm_mma_bf16 SRC gemm/dense_gemm_mma.metal
                       STD metal4.0 DEFINES VPIPE_ELT=bfloat
                       FLAGS ${VPIPE_NAX_MIN_OS})
add_vpipe_metal_kernel(qwen3_5_vision SRC encoders/qwen3_5_vision.metal)
add_vpipe_metal_kernel(audio_encoder SRC encoders/audio_encoder.metal)
set(STEEL_GEMM_HDRS
    vendored/mlx/backend/metal/kernels/steel/defines.h
    vendored/mlx/backend/metal/kernels/steel/utils.h
    vendored/mlx/backend/metal/kernels/steel/gemm/mma.h
    vendored/mlx/backend/metal/kernels/steel/gemm/loader.h
    vendored/mlx/backend/metal/kernels/steel/gemm/transforms.h
    vendored/mlx/backend/metal/kernels/steel/utils/integral_constant.h
    vendored/mlx/backend/metal/kernels/steel/utils/type_traits.h)
add_vpipe_metal_kernel(affine_qmm_steel
                       SRC quant/affine_qmm_steel.metal
                       DEPENDS ${STEEL_GEMM_HDRS})
add_vpipe_metal_kernel(affine_qmm_steel_bf16
                       SRC quant/affine_qmm_steel.metal
                       DEFINES VPIPE_ELT=bfloat
                       DEPENDS ${STEEL_GEMM_HDRS})
# Matrix-core (M5+) 4-bit quantized GEMM via MetalPerformancePrimitives
# matmul2d. Drop-in for affine_qmm_steel_w4g64; gated at runtime on
# matrix-core capability (older GPUs keep the steel path).
add_vpipe_metal_kernel(affine_qmm_mma
                       SRC quant/affine_qmm_mma.metal
                       STD metal4.0 FLAGS ${VPIPE_NAX_MIN_OS})
add_vpipe_metal_kernel(affine_qmm_mma_bf16
                       SRC quant/affine_qmm_mma.metal
                       STD metal4.0 DEFINES VPIPE_ELT=bfloat
                       FLAGS ${VPIPE_NAX_MIN_OS})
# 4-bit -> dense weight expansion feeding the matrix-core dense GEMM
# (dequant-once then dense_gemm_mma; faster than fused dequant-in-loop).
add_vpipe_metal_kernel(affine_dequant SRC quant/affine_dequant.metal)
add_vpipe_metal_kernel(affine_dequant_bf16 SRC quant/affine_dequant.metal
                       DEFINES VPIPE_ELT=bfloat)
# Matrix-core (M5+) flash attention, head_dim 256 (Qwen3.5). matmul2d
# QK^T / P*V with threadgroup-resident online softmax. Capability-gated.
add_vpipe_metal_kernel(sdpa_mma SRC attention/sdpa_mma.metal STD metal4.0
                       FLAGS ${VPIPE_NAX_MIN_OS})
add_vpipe_metal_kernel(sdpa_mma_bf16 SRC attention/sdpa_mma.metal
                       STD metal4.0 DEFINES VPIPE_ELT=bfloat
                       FLAGS ${VPIPE_NAX_MIN_OS})
# Matrix-core (M5+) batched matmul2d primitives for the Qwen3.5 GDN
# chunkwise prefill path (kkT/qkT/w@S/q@S/P@delta/delta^T@k). Capability-gated.
add_vpipe_metal_kernel(gdn_mma SRC gated-delta/gdn_mma.metal STD metal4.0
                       FLAGS ${VPIPE_NAX_MIN_OS})
add_vpipe_metal_kernel(gdn_mma_bf16 SRC gated-delta/gdn_mma.metal
                       STD metal4.0 DEFINES VPIPE_ELT=bfloat
                       FLAGS ${VPIPE_NAX_MIN_OS})
add_vpipe_metal_kernel(attn_steel
                       SRC attention/attn_steel.metal
                       DEPENDS vendored/mlx/backend/metal/kernels/steel/defines.h
                               vendored/mlx/backend/metal/kernels/steel/utils.h
                               vendored/mlx/backend/metal/kernels/steel/attn/attn.h
                               vendored/mlx/backend/metal/kernels/steel/attn/loader.h
                               vendored/mlx/backend/metal/kernels/steel/attn/mma.h
                               vendored/mlx/backend/metal/kernels/steel/attn/params.h
                               vendored/mlx/backend/metal/kernels/steel/attn/transforms.h
                               vendored/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h
                               vendored/mlx/backend/metal/kernels/steel/gemm/params.h
                               vendored/mlx/backend/metal/kernels/steel/utils/integral_constant.h
                               vendored/mlx/backend/metal/kernels/steel/utils/type_traits.h)

# MLX's MATRIX-CORE (NAX) steel flash attention (matmul2d). Same vendored
# headers as attn_steel + nax.h / steel_attention_nax.h. metal4.0 for the
# MPP tensor ops; #if __HAVE_TENSOR__ guards the real kernel (M5+).
add_vpipe_metal_kernel(attn_steel_nax
                       SRC attention/attn_steel_nax.metal
                       STD metal4.0
                       FLAGS ${VPIPE_NAX_MIN_OS}
                       DEPENDS vendored/mlx/backend/metal/kernels/steel/defines.h
                               vendored/mlx/backend/metal/kernels/steel/utils.h
                               vendored/mlx/backend/metal/kernels/steel/attn/nax.h
                               vendored/mlx/backend/metal/kernels/steel/attn/attn.h
                               vendored/mlx/backend/metal/kernels/steel/attn/loader.h
                               vendored/mlx/backend/metal/kernels/steel/attn/mma.h
                               vendored/mlx/backend/metal/kernels/steel/attn/params.h
                               vendored/mlx/backend/metal/kernels/steel/attn/transforms.h
                               vendored/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h
                               vendored/mlx/backend/metal/kernels/steel/gemm/params.h
                               vendored/mlx/backend/metal/kernels/steel/utils/integral_constant.h
                               vendored/mlx/backend/metal/kernels/steel/utils/type_traits.h)

# Ported from metal-runtime.cc -- the 5 legacy image/motion kernels.
add_vpipe_metal_kernel(nv12_to_planar_rgb_u8
                       SRC image-processing/nv12_to_planar_rgb_u8.metal)
add_vpipe_metal_kernel(letterbox_planar_u8_to_f32_chw
                       SRC image-processing/letterbox_planar_u8_to_f32_chw.metal)
add_vpipe_metal_kernel(letterbox_planar_u8_to_u8_chw
                       SRC image-processing/letterbox_planar_u8_to_u8_chw.metal)
add_vpipe_metal_kernel(letterbox_planar_u8_to_bgra_u8
                       SRC image-processing/letterbox_planar_u8_to_bgra_u8.metal)
add_vpipe_metal_kernel(motion_signature_u8
                       SRC image-processing/motion_signature_u8.metal)
add_vpipe_metal_kernel(motion_diff_u8
                       SRC image-processing/motion_diff_u8.metal)

set(VPIPE_SOURCES ${VPIPE_SOURCES} PARENT_SCOPE)
set(VPIPE_HEADERS ${VPIPE_HEADERS} PARENT_SCOPE)
set(VPIPE_MC_KERNEL_OBJS ${VPIPE_MC_KERNEL_OBJS} PARENT_SCOPE)
