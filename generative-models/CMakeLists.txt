# generative-models: the on-device generative-model (LLM/VLM) subsystem.
# Runs on the metal-compute backend with NO MLX in the forward pass; the
# whole subsystem (tokenizer, sampler, chat template, model loader,
# GenerativeModelManager, LoadedLanguageModel, the metal execs/encoders)
# builds under VPIPE_BUILD_APPLE_SILICON.
#
# Layout: top-level = public API + cross-model infrastructure;
#   shared/  = interfaces + cross-model encoders + format glue,
#   gemma4/  qwen3/  llama3/ = per-model-family sources.

set(LOCAL_SOURCES
  chat-template.cc
  context-manager.cc
  generative-model-manager.cc
  loaded-language-model.cc
  metal-token-muxer.cc
  model-exec-registry.cc
  model-loader.cc
  sampler.cc
  tokenizer.cc
  shared/gguf-convert.cc
  shared/gguf-file.cc
  shared/i8-gemm.cc
  shared/mcp/file-sandbox.cc
  shared/mcp/mcp-tools.cc
  shared/mcp/python-sandbox.cc
  shared/mcp/shell-tool.cc
  shared/mcp/url-fetch.cc
  shared/kernel-sets/decode-gqa-attn-set.cc
  shared/kernel-sets/prefill-gqa-attn-set.cc
  shared/whisper-feature-extractor.cc
  gemma4/gemma4-audio-feature-extractor.cc
  gemma4/gemma4-unified-embedder.cc
  gemma4/metal-gemma-model.cc
  gemma4/metal-gemma-model-exec.cc
  gemma4/metal-gemma4-audio.cc
  gemma4/metal-gemma4-vision.cc
  qwen3/metal-audio-encoder.cc
  qwen3/metal-qwen-model.cc
  qwen3/metal-qwen-model-exec.cc
  qwen3/metal-qwen-vision.cc
  llama3/metal-llama-model.cc
  llama3/metal-llama-model-exec.cc
  llama3/metal-llama-weights.cc
  moss/metal-moss-tts-model.cc
  moss/metal-moss-codec.cc
  moss/metal-moss-codec-v2.cc
  moss/moss-v15-processor.cc
  moss/metal-moss-local-transformer.cc
  moss/metal-moss-local-model.cc
  moss/metal-moss-v15-model.cc
  moss/moss-rt-processor.cc
  moss/metal-moss-rt-model.cc
  krea2/metal-krea2-transformer.cc
  krea2/metal-krea2-vae.cc
  krea2/metal-krea2-calibration.cc
  krea2/flow-sampler.cc
  qwen-image/metal-qwen-image-transformer.cc
  qwen-image/metal-qwen-image-calibration.cc
  qwen-image/metal-qwen25-vision.cc
  flux2/metal-flux2-transformer.cc
  flux2/metal-flux2-vae.cc
  flux2/metal-flux2-calibration.cc
  lora-fusion.cc
  quantize/affine-quantizer.cc
  quantize/arch-detect.cc
  quantize/calibration.cc
  quantize/calibration-corpus.cc
  quantize/model-quantizer.cc
  quantize/safetensors-writer.cc
  eval/lm-eval.cc
)
set(LOCAL_HEADERS
  chat-template.h
  context-manager.h
  generative-model-manager.h
  loaded-language-model.h
  metal-token-muxer.h
  model-exec.h
  model-exec-registry.h
  model-loader.h
  sampler.h
  token-muxer.h
  tokenizer.h
  shared/gguf-convert.h
  shared/gguf-file.h
  shared/mcp/file-sandbox.h
  shared/mcp/mcp-tools.h
  shared/mcp/python-sandbox.h
  shared/mcp/shell-tool.h
  shared/mcp/url-fetch.h
  shared/whisper-feature-extractor.h
  gemma4/gemma4-audio-feature-extractor.h
  gemma4/gemma4-unified-embedder.h
  gemma4/metal-gemma-model.h
  gemma4/metal-gemma-model-exec.h
  gemma4/metal-gemma4-audio.h
  gemma4/metal-gemma4-vision.h
  qwen3/metal-audio-encoder.h
  qwen3/metal-qwen-model.h
  qwen3/metal-qwen-model-exec.h
  qwen3/metal-qwen-vision.h
  llama3/metal-llama-model.h
  llama3/metal-llama-model-exec.h
  llama3/metal-llama-weights.h
  moss/metal-moss-tts-model.h
  moss/metal-moss-codec.h
  moss/metal-moss-codec-v2.h
  moss/moss-v15-processor.h
  moss/metal-moss-local-transformer.h
  moss/metal-moss-local-model.h
  moss/metal-moss-v15-model.h
  moss/moss-rt-processor.h
  moss/metal-moss-rt-model.h
  quantize/affine-quantizer.h
  quantize/arch-detect.h
  quantize/calibration.h
  quantize/model-quantizer.h
  quantize/safetensors-writer.h
  eval/lm-eval.h
)
# The CoreML-backed vision tower builds on the apple-silicon axis: its
# MLX-free host API (encode_host -> Result of host f32) + metal-compute
# letterbox preproc drive the metal LM vision path.
if(APPLE AND VPIPE_BUILD_APPLE_SILICON)
  list(APPEND LOCAL_SOURCES shared/coreml-vision-encoder.cc)
  list(APPEND LOCAL_HEADERS shared/coreml-vision-encoder.h)
endif()

list(TRANSFORM LOCAL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND VPIPE_SOURCES ${LOCAL_SOURCES})
list(APPEND VPIPE_HEADERS ${LOCAL_HEADERS})
set(VPIPE_SOURCES ${VPIPE_SOURCES} PARENT_SCOPE)
set(VPIPE_HEADERS ${VPIPE_HEADERS} PARENT_SCOPE)
