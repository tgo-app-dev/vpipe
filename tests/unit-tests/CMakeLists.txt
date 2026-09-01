set(LOCAL_TEST_SOURCES
  buffer-stage.cc
  byte-track-stage.cc
  call-stage.cc
  chrono-shell-stages.cc
  save-text-stage.cc
  clock-domain.cc
  console-writer.cc
  credential-store.cc
  denoise-progress.cc
  gpu-thermal.cc
  detection-overlay-stage.cc
  feedback-pair.cc
  stage-relaunch-sweep.cc
  weight-registry.cc
  weight-set.cc
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
  av-filter-graph.cc
  media-decode.cc
  media-line.cc
  text-stream-chunk.cc
  mini-http.cc
  qr-code.cc
  command-sandbox.cc
  path-sandbox.cc
  session-sandbox.cc
  model-catalog.cc
  model-source.cc
  model-source-live.cc
  block-residency.cc
  memory-plan.cc
  model-memory.cc
  model-register-stage.cc
  model-remove-stage.cc
  model-select-stage.cc
  sampler-select-stage.cc
  plugin-abi.cc
  plugin-loader.cc
  onvif-soap.cc
  pipeline-handle.cc
  pipeline-runtime.cc
  pipeline-spec.cc
  pipeline.cc
  rest-client-stage.cc
  resumable-fetch.cc
  xet-fetch.cc
  profiling.cc
  perf-buffer.cc
  hls-broadcast-stage.cc
  compare-image-stage.cc
  create-mask-stage.cc
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
  ui-view-registry.cc
  tensor-storage.cc
  thread-pool.cc
  vertex-oport-policy.cc
  image-metadata.cc
  video-stages.cc
  wan-video-stages.cc
  model-config-stages.cc
  video-to-rgb-stage.cc
  videos-db-cleanup-stage.cc
  ws-discovery.cc
)
if(APPLE)
    list(APPEND LOCAL_TEST_SOURCES audio-capture-stage.cc)
    list(APPEND LOCAL_TEST_SOURCES video-capture-stage.cc)
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
        optiq-block-perf.cc
        mlp-fuse-ab.cc
        metal-compute-sdpa-mma.cc
        metal-compute-gdn-mma.cc
        metal-compute-binary-archive.cc
        metal-compute-concurrent.cc
        metal-compute-fence.cc
        metal-compute-function-constants.cc
        metal-compute-heap.cc
        metal-compute-letterbox-rgb-f16.cc
        metal-compute-vision-bf16.cc
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
        metal-lm/attention.cc
        metal-lm/batched-decode.cc
        metal-lm/bench.cc
        metal-lm/gemma-gguf.cc
        metal-lm/gemma-text.cc
        metal-lm/kquant.cc
        metal-lm/manager.cc
        metal-lm/moss.cc
        metal-lm/multimodal-vqa.cc
        metal-lm/pdecode.cc
        metal-lm/quant-gemv-bench.cc
        metal-lm/quant-gemv.cc
        metal-lm/qwen-text.cc
        metal-lm/speculative-mtp.cc
        model-quantize-roundtrip.cc
        model-quantize-stage.cc
        model-eval-stage.cc
        model-exec-registry.cc
        vae-model-registry.cc
        video-model-registry.cc
        hidden-state-encoder.cc
        plugin-metal-library.cc
        model-benchmark-stage.cc
        gemma-bench-driver.cc
        moss-rt-driver.cc
        quant-arch-detect.cc
        moss-tts-local.cc
        krea2-encoder.cc
        krea2-dit.cc
        krea2-vae.cc
        wan-vae.cc
        wan-umt5.cc
        wan-unipc.cc
        wan-dit.cc
        wan-i2v.cc
        minimax-h3-blocks.cc
        minimax-h3-lora.cc
        minimax-h3-comfy.cc
        minimax-h3-layout.cc
        minimax-h3-dit.cc
        minimax-h3-vvae.cc
        minimax-h3-avae.cc
        audio-vae-encode.cc
        load-audio-stage.cc
        load-video-stage.cc
        vae-decode-clip.cc
        runtime-lora.cc
        minimax-h3-text-encoder.cc
        minimax-h3-refenc.cc
        mage-vae.cc
        mage-dit.cc
        mage-cond.cc
        mage-watermark.cc
        mage-t2i.cc
        mage-screen.cc
        krea2-t2i.cc
        krea2-dit-quant.cc
        krea2-lora.cc
        lora-fuse-stage.cc
        krea2-sampler.cc
        qwen-image-edit-sched.cc
        qwen-image-edit-vae.cc
        qwen-image-edit-text.cc
        qwen-image-edit-dit.cc
        qwen-image-edit-vision.cc
        qwen-image-edit-mm.cc
        qwen-image-edit-t2i.cc
        qwen-image-edit-awq.cc
        flux2-smoke.cc
        image-memory-plan.cc
        boogu-smoke.cc
        llm-chat-template.cc
        llm-sampler.cc
        llm-tokenizer.cc
        unigram-tokenizer.cc
        file-sandbox.cc
        mcp-tools.cc
        python-sandbox.cc
        shell-tool.cc
        url-fetch.cc
        metal-compute-image-ops.cc
        realtime-vqa-stage.cc
        temporal-decimation-stage.cc
        image-resample-stage.cc
        temporal-slice-stage.cc
        vae-encode-range.cc
        temporal-stack-stage.cc
        temporal-resample-stage.cc
        audio-temporal-resample-stage.cc
        text-chat-stage.cc
        text-to-speech-stage.cc
        visual-qa-stage.cc
        yolo-detection-stage.cc)
endif()
set(LOCAL_TEST_HEADERS "")

list(TRANSFORM LOCAL_TEST_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_TEST_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# The unit under test for qr-code.cc. It ships with vpipe-web-ui (its
# only consumer) rather than in libvpipe, and depends on nothing but the
# standard library -- so the test compiles it directly instead of the app
# growing a library for one file. Appended AFTER the TRANSFORM above,
# which would otherwise prepend this directory to an absolute path.
list(APPEND LOCAL_TEST_SOURCES
     "${CMAKE_SOURCE_DIR}/apps/web-ui/qr-code.cc")

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
