set(LOCAL_TEST_SOURCES
  byte-track-stage.cc
  call-stage.cc
  chrono-shell-stages.cc
  save-text-stage.cc
  clock-domain.cc
  credential-store.cc
  detection-overlay-stage.cc
  feedback-pair.cc
  oport-buffer-retire.cc
  oport-buffer-peek.cc
  oport-buffer-thresholds.cc
  oport-buffer-wrap.cc
  ffmpeg-libraries.cc
  ffmpeg-log-tap.cc
  flex-data.cc
  graph-move-iport.cc
  load-image-stage.cc
  text-source-stages.cc
  save-image-stage.cc
  host-identity.cc
  i18n.cc
  library-handle.cc
  lmdb-wrapper.cc
  lmdb.cc
  log-delegate.cc
  media-decode.cc
  media-line.cc
  text-stream-chunk.cc
  mini-http.cc
  command-sandbox.cc
  path-sandbox.cc
  session-sandbox.cc
  model-catalog.cc
  model-remove-stage.cc
  plugin-abi.cc
  plugin-loader.cc
  onvif-soap.cc
  pipeline-handle.cc
  pipeline-runtime.cc
  pipeline-spec.cc
  pipeline.cc
  rest-client-stage.cc
  profiling.cc
  perf-buffer.cc
  hls-broadcast-stage.cc
  preview-stage.cc
  read-any.cc
  runtime-context-has-consumers.cc
  segment-writer.cc
  session-config.cc
  session.cc
  stage-config.cc
  stage-spec.cc
  stdout-log-delegate-async.cc
  tensor-beat.cc
  ui-delegate.cc
  tensor-storage.cc
  thread-pool.cc
  vertex-oport-policy.cc
  video-stages.cc
  video-to-rgb-stage.cc
  videos-db-cleanup-stage.cc
  ws-discovery.cc
)
if(APPLE)
    list(APPEND LOCAL_TEST_SOURCES audio-capture-stage.cc)
    list(APPEND LOCAL_TEST_SOURCES credential-cipher.cc)
    list(APPEND LOCAL_TEST_SOURCES onvif-refresh.cc)
endif()
if(APPLE AND VPIPE_BUILD_APPLE_SILICON)
    list(APPEND LOCAL_TEST_SOURCES
        save-audio-stage.cc
        audio-segment-stage.cc
        audio-tagging-stage.cc
        coreml-cpp.cc
        coreml-inference-stage.cc
        coreml-model-manager.cc
        metal-compute.cc
        metal-compute-gemm-mma.cc
        metal-compute-sdpa-mma.cc
        metal-compute-gdn-mma.cc
        metal-compute-binary-archive.cc
        metal-compute-concurrent.cc
        metal-compute-fence.cc
        metal-compute-function-constants.cc
        metal-compute-heap.cc
        metal-compute-letterbox-rgb-f16.cc
        metal-compute-library.cc
        metal-compute-nv12.cc
        metal-compute-residency.cc
        metal-compute-saxpy.cc
        metal-compute-shared-buffer.cc
        metal-compute-stream.cc
        metal-compute-tensor-beat-bridge.cc
        metal-compute-texture.cc
        audio-transcribe-stage.cc
        gemma4-unified-embedder.cc
        gguf-convert.cc
        metal-lm-smoke.cc
        model-quantize-roundtrip.cc
        model-quantize-stage.cc
        model-eval-stage.cc
        model-exec-registry.cc
        plugin-metal-library.cc
        model-benchmark-stage.cc
        gemma-bench-driver.cc
        moss-rt-driver.cc
        quant-arch-detect.cc
        moss-tts-local.cc
        krea2-encoder.cc
        krea2-dit.cc
        krea2-vae.cc
        krea2-t2i.cc
        krea2-dit-quant.cc
        krea2-lora.cc
        lora-fuse-stage.cc
        krea2-sampler.cc
        flux2-smoke.cc
        llm-chat-template.cc
        llm-sampler.cc
        llm-tokenizer.cc
        file-sandbox.cc
        mcp-tools.cc
        python-sandbox.cc
        shell-tool.cc
        url-fetch.cc
        metal-compute-image-ops.cc
        realtime-vqa-stage.cc
        temporal-decimation-stage.cc
        image-resample-stage.cc
        text-chat-stage.cc
        text-to-speech-stage.cc
        visual-qa-stage.cc
        yolo-detection-stage.cc)
endif()
set(LOCAL_TEST_HEADERS "")

list(TRANSFORM LOCAL_TEST_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_TEST_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# Internal tests may include anything: public headers, private headers,
# generated headers, and sources from common/.
if(LOCAL_TEST_SOURCES)
    add_library(vpipe_unit_tests OBJECT ${LOCAL_TEST_SOURCES})
    target_include_directories(vpipe_unit_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}
        ${VPIPE_FFMPEG_INCLUDE_DIRS}
    )
    target_link_libraries(vpipe_unit_tests PRIVATE lmdb pugixml)
    if(APPLE AND VPIPE_BUILD_APPLE_SILICON)
        target_link_libraries(vpipe_unit_tests PRIVATE
            metal_cpp_foundation)
        # Mirror the root-level definition so test TUs can predicate
        # on the apple-silicon backend being built.
        target_compile_definitions(vpipe_unit_tests PRIVATE
            VPIPE_BUILD_APPLE_SILICON)
    endif()
    list(APPEND VPIPETEST_SOURCES $<TARGET_OBJECTS:vpipe_unit_tests>)
    set(VPIPETEST_SOURCES ${VPIPETEST_SOURCES} PARENT_SCOPE)
endif()
