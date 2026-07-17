# stages: all pipeline stage implementations. Grouped on disk:
#   audio-video/  -- multimedia (ffmpeg) capture / decode / encode / stream
#   vision/       -- traditional computer vision (detection / tracking / overlay)
#   (root)        -- generic, LM-driven, and CoreML audio-ML stages
# Each stage keeps its original build gate (always / APPLE / VPIPE_BUILD_
# APPLE_SILICON); the gate, not the directory, decides where its sources go.

set(LOCAL_SOURCES
  # generic (always built)
  call-stage.cc
  chrono-stage.cc
  load-image-stage.cc
  load-text-stage.cc
  load-video-stage.cc
  lora-fuse-stage.cc
  sampler-select-stage.cc
  save-image-stage.cc
  save-video-stage.cc
  scheduler-select-stage.cc
  text-to-image-stage.cc
  vae-decode-stage.cc
  vae-encode-stage.cc
  model-catalog.cc
  model-fetch-stage.cc
  model-remove-stage.cc
  model-registry.cc
  passthrough-stage.cc
  qwen-asr-tokenizer.cc
  rest-client-stage.cc
  save-text-stage.cc
  shell-stage.cc
  text-input-stage.cc
  text-prompt-stage.cc
  videos-db-cleanup-stage.cc
  # vision (always built)
  vision/byte-track-stage.cc
  vision/detection-overlay-stage.cc
  # audio-video (always built; ffmpeg dlopen'd at runtime)
  audio-video/audio-to-pcm-stage.cc
  audio-video/hls-broadcast-stage.cc
  audio-video/preview-stage.cc
  audio-video/video-to-rgb-stage.cc
)
set(LOCAL_HEADERS
  # generic
  call-stage.h
  chrono-stage.h
  load-image-stage.h
  load-text-stage.h
  load-video-stage.h
  model-catalog.h
  model-fetch-stage.h
  model-remove-stage.h
  model-registry.h
  passthrough-stage.h
  qwen-asr-tokenizer.h
  rest-client-stage.h
  save-image-stage.h
  save-text-stage.h
  save-video-stage.h
  shell-stage.h
  text-input-stage.h
  text-prompt-stage.h
  trigger-beat.h
  videos-db-cleanup-stage.h
  # vision
  vision/byte-track-stage.h
  vision/detection-overlay-stage.h
  vision/detection-overlay-font5x7.h
  # audio-video
  audio-video/audio-to-pcm-stage.h
  audio-video/hls-broadcast-stage.h
  audio-video/preview-stage.h
  audio-video/video-to-rgb-stage.h
  audio-video/video-tokens.h
)
if(APPLE)
  # avfoundation device capture + ONVIF discovery (Apple-only APIs).
  list(APPEND LOCAL_SOURCES audio-video/audio-capture-stage.cc)
  list(APPEND LOCAL_HEADERS audio-video/audio-capture-stage.h)
  list(APPEND LOCAL_SOURCES audio-video/rtsp-capture-stage.cc)
  list(APPEND LOCAL_HEADERS audio-video/rtsp-capture-stage.h)
  list(APPEND LOCAL_SOURCES onvif-discovery-stage.cc)
  list(APPEND LOCAL_HEADERS onvif-discovery-stage.h)
endif()
# Stages on the apple-silicon axis (VPIPE_BUILD_APPLE_SILICON): the metal/
# no-MLX LM stages, the apple-silicon-typed AV stages, and the CoreML-driven
# stages (moved from apple-silicon/coreml/ -- they keep using that dir's
# coreml-model-manager / coreml-cpp library).
if(VPIPE_BUILD_APPLE_SILICON)
  # audio-video on the apple-silicon axis (TensorBeatPayload / metal image ops)
  list(APPEND LOCAL_SOURCES save-audio-stage.cc)
  list(APPEND LOCAL_HEADERS save-audio-stage.h)
  list(APPEND LOCAL_SOURCES audio-video/temporal-decimation-stage.cc)
  list(APPEND LOCAL_HEADERS audio-video/temporal-decimation-stage.h)
  list(APPEND LOCAL_SOURCES audio-video/image-resample-stage.cc)
  list(APPEND LOCAL_HEADERS audio-video/image-resample-stage.h)
  # LM-driven stages (self-contained metal/no-MLX path)
  list(APPEND LOCAL_SOURCES audio-transcribe-stage.cc)
  list(APPEND LOCAL_HEADERS audio-transcribe-stage.h)
  list(APPEND LOCAL_SOURCES realtime-vqa-stage.cc)
  list(APPEND LOCAL_HEADERS realtime-vqa-stage.h)
  list(APPEND LOCAL_SOURCES text-chat-stage.cc)
  list(APPEND LOCAL_HEADERS text-chat-stage.h)
  list(APPEND LOCAL_SOURCES text-to-speech-stage.cc)
  list(APPEND LOCAL_HEADERS text-to-speech-stage.h)
  list(APPEND LOCAL_SOURCES model-quantize-stage.cc)
  list(APPEND LOCAL_HEADERS model-quantize-stage.h)
  list(APPEND LOCAL_SOURCES model-eval-stage.cc)
  list(APPEND LOCAL_HEADERS model-eval-stage.h)
  list(APPEND LOCAL_SOURCES model-benchmark-stage.cc gpu-telemetry.mm)
  list(APPEND LOCAL_HEADERS model-benchmark-stage.h gpu-telemetry.h)
  list(APPEND LOCAL_SOURCES visual-qa-stage.cc)
  list(APPEND LOCAL_HEADERS visual-qa-stage.h)
  # CoreML-driven stages
  list(APPEND LOCAL_SOURCES audio-segment-stage.cc)
  list(APPEND LOCAL_HEADERS audio-segment-stage.h)
  list(APPEND LOCAL_SOURCES audio-tagging-stage.cc)
  list(APPEND LOCAL_HEADERS audio-tagging-stage.h ced-audioset-labels.h)
  list(APPEND LOCAL_SOURCES coreml-inference-stage.cc)
  list(APPEND LOCAL_HEADERS coreml-inference-stage.h)
  list(APPEND LOCAL_SOURCES vision/yolo-detection-stage.cc)
  list(APPEND LOCAL_HEADERS vision/yolo-detection-stage.h)
endif()

list(TRANSFORM LOCAL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND VPIPE_SOURCES ${LOCAL_SOURCES})
list(APPEND VPIPE_HEADERS ${LOCAL_HEADERS})
set(VPIPE_SOURCES ${VPIPE_SOURCES} PARENT_SCOPE)
set(VPIPE_HEADERS ${VPIPE_HEADERS} PARENT_SCOPE)
