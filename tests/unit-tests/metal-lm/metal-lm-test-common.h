// metal-lm/ -- the metal-compute LM (language / vision-language /
// speech) test group. These tests compile and run in the
// VPIPE_BUILD_APPLE_SILICON test block and are the end-to-end proof
// that the language-model subsystem loads + generates on the metal
// backend. They never reference any other inference runtime.
//
// Env-gated: set VPIPE_METAL_LM_SMOKE_MODEL to a metal-supported
// model dir (Qwen3.5-4B / Llama / Qwen3-ASR text decoder); the
// per-theme files name the extra vars they need. Loads force
// VPIPE_LLM_BACKEND=metal for the duration.
//
// This header carries the include block and the `using namespace
// vpipe` every file in the group shares. It is test-only -- nothing
// outside tests/unit-tests/metal-lm/ may include it.

#ifndef VPIPE_TESTS_INTERNAL_METAL_LM_TEST_COMMON_H
#define VPIPE_TESTS_INTERNAL_METAL_LM_TEST_COMMON_H

#include "minitest.h"
#include "generative-models/chat-template.h"
#include "generative-models/gemma4/gemma4-unified-embedder.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/gemma4/metal-gemma4-vision.h"
#include "generative-models/shared/coreml-vision-encoder.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/moss/metal-moss-tts-model.h"
#include "generative-models/moss/metal-moss-codec.h"
#include "generative-models/model-loader.h"
#include "generative-models/sampler.h"
#include "generative-models/token-muxer.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"
#include "generative-models/shared/gguf-file.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/media-line.h"
#include "common/session.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <optional>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <sstream>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <mach/mach.h>

using namespace vpipe;

#endif  // VPIPE_TESTS_INTERNAL_METAL_LM_TEST_COMMON_H
