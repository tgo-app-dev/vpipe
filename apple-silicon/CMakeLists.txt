# apple-silicon: Apple-only ML/inference subtrees. This file is a
# router that conditionally pulls in the CoreML and MLX backends and
# exposes the shared TensorBeat header used by both.

set(LOCAL_HEADERS
  aligned-allocator.h
  tensor-beat.h
  tensor-storage.h
)
set(LOCAL_SOURCES
  tensor-storage.cc
)
list(TRANSFORM LOCAL_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND VPIPE_HEADERS ${LOCAL_HEADERS})
list(APPEND VPIPE_SOURCES ${LOCAL_SOURCES})

if(VPIPE_BUILD_APPLE_SILICON)
    add_subdirectory(coreml)
    add_subdirectory(metal-compute)
    # The generative-model (LLM/VLM) subsystem moved to the top-level
    # generative-models/ tree (added from the root CMakeLists). The
    # Apple-only temporal-decimation stage moved to the top-level stages/
    # tree (gated there on VPIPE_BUILD_APPLE_SILICON).
endif()

set(VPIPE_SOURCES ${VPIPE_SOURCES} PARENT_SCOPE)
set(VPIPE_HEADERS ${VPIPE_HEADERS} PARENT_SCOPE)
# Bubble up the metal-compute kernel OBJECT-library list so the root
# CMakeLists can link the per-kernel .o into libvpipe.
set(VPIPE_MC_KERNEL_OBJS ${VPIPE_MC_KERNEL_OBJS} PARENT_SCOPE)
