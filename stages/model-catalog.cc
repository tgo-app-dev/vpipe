#include "stages/model-catalog.h"
#include "common/flex-data.h"

#include <algorithm>
#include <cctype>

namespace vpipe {

const std::vector<ModelCatalogEntry>&
model_catalog()
{
  // ==================================================================
  // MODEL CATALOGUE -- single edit point. Append an entry to add a
  // model; the selection menu rebuilds itself from this table.
  // ==================================================================
  // The 18 non-asset files every Mage-Flow repo ships. All six repos share
  // ONE layout, so the list is named once here and reused by each entry
  // below; pinning it skips the ~20 sample images under assets/.
  static const std::vector<std::string> kMageFlowFiles = {
      "model_index.json",
      "transformer/config.json",
      "transformer/diffusion_pytorch_model.safetensors",
      "vae/config.json",
      "vae/diffusion_pytorch_model.safetensors",
      "text_encoder/config.json",
      "text_encoder/generation_config.json",
      "text_encoder/model.safetensors.index.json",
      "text_encoder/model-00001-of-00002.safetensors",
      "text_encoder/model-00002-of-00002.safetensors",
      "text_encoder/tokenizer.json",
      "text_encoder/tokenizer_config.json",
      "text_encoder/vocab.json",
      "text_encoder/merges.txt",
      "text_encoder/chat_template.json",
      "text_encoder/preprocessor_config.json",
      "text_encoder/video_preprocessor_config.json",
      "scheduler/scheduler_config.json"};

  // The Wan2.2-I2V-A14B (diffusers) files. `assets/` is the README's
  // showcase art and is skipped; everything else is load-bearing, and the
  // two 54 GB expert transformers dominate (they ship fp32 -- the
  // reference runs them in bf16, which is what our loader converts to).
  // ~126 GB total, so this is the largest catalogued model by some margin.
  static const std::vector<std::string> kWan22I2VFiles = {
      "model_index.json",
      "transformer/config.json",
      "transformer/diffusion_pytorch_model.safetensors.index.json",
      "transformer/diffusion_pytorch_model-00001-of-00012.safetensors",
      "transformer/diffusion_pytorch_model-00002-of-00012.safetensors",
      "transformer/diffusion_pytorch_model-00003-of-00012.safetensors",
      "transformer/diffusion_pytorch_model-00004-of-00012.safetensors",
      "transformer/diffusion_pytorch_model-00005-of-00012.safetensors",
      "transformer/diffusion_pytorch_model-00006-of-00012.safetensors",
      "transformer/diffusion_pytorch_model-00007-of-00012.safetensors",
      "transformer/diffusion_pytorch_model-00008-of-00012.safetensors",
      "transformer/diffusion_pytorch_model-00009-of-00012.safetensors",
      "transformer/diffusion_pytorch_model-00010-of-00012.safetensors",
      "transformer/diffusion_pytorch_model-00011-of-00012.safetensors",
      "transformer/diffusion_pytorch_model-00012-of-00012.safetensors",
      "transformer_2/config.json",
      "transformer_2/diffusion_pytorch_model.safetensors.index.json",
      "transformer_2/diffusion_pytorch_model-00001-of-00012.safetensors",
      "transformer_2/diffusion_pytorch_model-00002-of-00012.safetensors",
      "transformer_2/diffusion_pytorch_model-00003-of-00012.safetensors",
      "transformer_2/diffusion_pytorch_model-00004-of-00012.safetensors",
      "transformer_2/diffusion_pytorch_model-00005-of-00012.safetensors",
      "transformer_2/diffusion_pytorch_model-00006-of-00012.safetensors",
      "transformer_2/diffusion_pytorch_model-00007-of-00012.safetensors",
      "transformer_2/diffusion_pytorch_model-00008-of-00012.safetensors",
      "transformer_2/diffusion_pytorch_model-00009-of-00012.safetensors",
      "transformer_2/diffusion_pytorch_model-00010-of-00012.safetensors",
      "transformer_2/diffusion_pytorch_model-00011-of-00012.safetensors",
      "transformer_2/diffusion_pytorch_model-00012-of-00012.safetensors",
      "text_encoder/config.json",
      "text_encoder/model.safetensors.index.json",
      "text_encoder/model-00001-of-00003.safetensors",
      "text_encoder/model-00002-of-00003.safetensors",
      "text_encoder/model-00003-of-00003.safetensors",
      "tokenizer/tokenizer.json",
      "tokenizer/tokenizer_config.json",
      "tokenizer/special_tokens_map.json",
      "tokenizer/spiece.model",
      "vae/config.json",
      "vae/diffusion_pytorch_model.safetensors",
      "scheduler/scheduler_config.json"};

  // The MiniMax-H3 FL2VA files. The repo is PARTITIONED: `FL2VA/` and
  // `Ref2VA/` are each a complete pipeline (their transformer configs are
  // byte-identical -- only `model_index.json`'s `_minimax_h3.partition`
  // tells them apart), so pinning one partition halves the download and
  // still yields a runnable model root at `<repo>/FL2VA`. The reference
  // .py under video_vae/ and audio_vae/ is `trust_remote_code` plumbing
  // for diffusers, not architecture we read, and is skipped -- but their
  // config.json / config.yaml / metadata.json ARE the architecture, so
  // those stay. ~144 GB: the 33B DiT and the Qwen3-VL-32B encoder are
  // 66 GB EACH, which makes this the largest catalogued model, ahead of
  // Wan2.2-I2V-A14B's ~126 GB.
  static const std::vector<std::string> kMiniMaxH3FL2VAFiles = {
      "FL2VA/model_index.json",
      "FL2VA/transformer/config.json",
      "FL2VA/transformer/model.safetensors.index.json",
      "FL2VA/transformer/model-00001-of-00013.safetensors",
      "FL2VA/transformer/model-00002-of-00013.safetensors",
      "FL2VA/transformer/model-00003-of-00013.safetensors",
      "FL2VA/transformer/model-00004-of-00013.safetensors",
      "FL2VA/transformer/model-00005-of-00013.safetensors",
      "FL2VA/transformer/model-00006-of-00013.safetensors",
      "FL2VA/transformer/model-00007-of-00013.safetensors",
      "FL2VA/transformer/model-00008-of-00013.safetensors",
      "FL2VA/transformer/model-00009-of-00013.safetensors",
      "FL2VA/transformer/model-00010-of-00013.safetensors",
      "FL2VA/transformer/model-00011-of-00013.safetensors",
      "FL2VA/transformer/model-00012-of-00013.safetensors",
      "FL2VA/transformer/model-00013-of-00013.safetensors",
      "FL2VA/text_encoder/config.json",
      "FL2VA/text_encoder/model.safetensors.index.json",
      "FL2VA/text_encoder/model-00001-of-00014.safetensors",
      "FL2VA/text_encoder/model-00002-of-00014.safetensors",
      "FL2VA/text_encoder/model-00003-of-00014.safetensors",
      "FL2VA/text_encoder/model-00004-of-00014.safetensors",
      "FL2VA/text_encoder/model-00005-of-00014.safetensors",
      "FL2VA/text_encoder/model-00006-of-00014.safetensors",
      "FL2VA/text_encoder/model-00007-of-00014.safetensors",
      "FL2VA/text_encoder/model-00008-of-00014.safetensors",
      "FL2VA/text_encoder/model-00009-of-00014.safetensors",
      "FL2VA/text_encoder/model-00010-of-00014.safetensors",
      "FL2VA/text_encoder/model-00011-of-00014.safetensors",
      "FL2VA/text_encoder/model-00012-of-00014.safetensors",
      "FL2VA/text_encoder/model-00013-of-00014.safetensors",
      "FL2VA/text_encoder/model-00014-of-00014.safetensors",
      "FL2VA/text_encoder/tokenizer.json",
      "FL2VA/text_encoder/tokenizer_config.json",
      "FL2VA/text_encoder/vocab.json",
      "FL2VA/text_encoder/merges.txt",
      "FL2VA/text_encoder/chat_template.json",
      "FL2VA/text_encoder/preprocessor_config.json",
      "FL2VA/text_encoder/video_preprocessor_config.json",
      "FL2VA/video_vae/config.json",
      "FL2VA/video_vae/source/config.json",
      "FL2VA/video_vae/source/model.safetensors",
      "FL2VA/audio_vae/config.json",
      "FL2VA/audio_vae/config.yaml",
      "FL2VA/audio_vae/metadata.json",
      "FL2VA/audio_vae/model.safetensors",
      "FL2VA/processor/tokenizer.json",
      "FL2VA/processor/tokenizer_config.json",
      "FL2VA/processor/vocab.json",
      "FL2VA/processor/merges.txt",
      "FL2VA/processor/chat_template.json",
      "FL2VA/processor/preprocessor_config.json",
      "FL2VA/processor/video_preprocessor_config.json",
      "FL2VA/tokenizer/tokenizer.json",
      "FL2VA/tokenizer/tokenizer_config.json",
      "FL2VA/tokenizer/vocab.json",
      "FL2VA/tokenizer/merges.txt"};

  // All four Boogu-Image-0.1 repos are byte-identical in layout, so one pinned
  // list serves them. It SKIPS the assets/ showcase images and the two
  // `trust_remote_code` .py shims (compatibility re-exports that only point
  // diffusers at the upstream python package -- vpipe reads the configs
  // directly). The processor/ side carries the fast tokenizer + the chat
  // template the conditioner renders; mllm/ keeps its own tokenizer copy for
  // completeness. ~36 GB (10B DiT + 8B Qwen3-VL, bf16).
  static const std::vector<std::string> kBooguFiles = {
      "model_index.json",
      "transformer/config.json",
      "transformer/diffusion_pytorch_model.safetensors.index.json",
      "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
      "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
      "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
      "mllm/config.json",
      "mllm/generation_config.json",
      "mllm/model.safetensors.index.json",
      "mllm/model-00001-of-00004.safetensors",
      "mllm/model-00002-of-00004.safetensors",
      "mllm/model-00003-of-00004.safetensors",
      "mllm/model-00004-of-00004.safetensors",
      "mllm/preprocessor_config.json",
      "mllm/video_preprocessor_config.json",
      "mllm/tokenizer.json",
      "mllm/tokenizer_config.json",
      "mllm/chat_template.json",
      "vae/config.json",
      "vae/diffusion_pytorch_model.safetensors",
      "processor/tokenizer.json",
      "processor/tokenizer_config.json",
      "processor/preprocessor_config.json",
      "processor/video_preprocessor_config.json",
      "processor/chat_template.jinja",
      "processor/special_tokens_map.json",
      "processor/added_tokens.json",
      "processor/vocab.json",
      "processor/merges.txt",
      "scheduler/scheduler_config.json"};

  static const std::vector<ModelCatalogEntry> kCatalog = {
    // ---- Evaluation datasets (model-eval stage) ----------------------
    // Fetched on demand from the HuggingFace datasets-server /rows API and
    // registered in the models DB; the model-eval stage resolves the key to
    // the local dir and reads the rows-*.json pages. Kept OUT of the binary
    // so the dataset licenses (CC BY-SA) don't touch vpipe's Apache-2.0.
    {.family = "Datasets", .version = "eval", .param_class = "WikiText-2",
     .variant = "raw test (Salesforce/wikitext)",
     .hf_path = "vpipe-eval-datasets/wikitext-2-raw-test",
     .model_type = "eval-wikitext2",
     .name = "wikitext-2-raw-test",
     .dataset_files = {
       {"https://datasets-server.huggingface.co/rows?dataset=Salesforce/"
        "wikitext&config=wikitext-2-raw-v1&split=test&offset=0&length=100",
        "rows-0000.json"},
       {"https://datasets-server.huggingface.co/rows?dataset=Salesforce/"
        "wikitext&config=wikitext-2-raw-v1&split=test&offset=100&length=100",
        "rows-0100.json"},
       {"https://datasets-server.huggingface.co/rows?dataset=Salesforce/"
        "wikitext&config=wikitext-2-raw-v1&split=test&offset=200&length=100",
        "rows-0200.json"},
       {"https://datasets-server.huggingface.co/rows?dataset=Salesforce/"
        "wikitext&config=wikitext-2-raw-v1&split=test&offset=300&length=100",
        "rows-0300.json"}}},
    {.family = "Datasets", .version = "eval", .param_class = "ARC-Challenge",
     .variant = "test (allenai/ai2_arc)",
     .hf_path = "vpipe-eval-datasets/arc-challenge-test",
     .model_type = "eval-arc-challenge",
     .name = "arc-challenge-test",
     .dataset_files = {
       {"https://datasets-server.huggingface.co/rows?dataset=allenai/"
        "ai2_arc&config=ARC-Challenge&split=test&offset=0&length=100",
        "rows-0000.json"},
       {"https://datasets-server.huggingface.co/rows?dataset=allenai/"
        "ai2_arc&config=ARC-Challenge&split=test&offset=100&length=100",
        "rows-0100.json"},
       {"https://datasets-server.huggingface.co/rows?dataset=allenai/"
        "ai2_arc&config=ARC-Challenge&split=test&offset=200&length=100",
        "rows-0200.json"}}},
    // Qwen3.6-27B: a Qwen3.5-family hybrid VLM (model_type "qwen3_5",
    // full-attn + gated-DeltaNet, 64 layers, hidden 5120). bf16 source
    // (15 safetensors shards, ~54 GB) -- whole-repo fetch; quantize with
    // the model-quantize stage (AWQ supported: the GDN in-proj group folds
    // into input_layernorm).
    {.family = "Qwen", .version = "3.6", .param_class = "27B",
     .variant = "bf16 (Qwen)",
     .hf_path = "Qwen/Qwen3.6-27B",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    // The same 27B, pre-quantized OptiQ: mixed 4/8-bit per LINEAR over a
    // group-64 affine base (the per-tensor mixed path -- q/k/v/o, the GDN
    // in_proj group and the MLP each carry their own width; embed_tokens
    // and lm_head are w8). Whole-repo fetch: 4 shards plus the OptiQ
    // companions in optiq/ -- mtp.safetensors (the MTP draft head, read
    // when VPIPE_QWEN_CTX_MTP is set) and optiq_vision.safetensors (the
    // vision tower). The tree listing is recursive, so the subdir comes
    // along.
    {.family = "Qwen", .version = "3.6", .param_class = "27B",
     .variant = "MLX OptiQ 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.6-27B-OptiQ-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    // Qwen3.6-35B-A3B: a Qwen3.5-family MoE (SparseMoeBlock: routed experts +
    // shared expert, model_type "qwen3_5"). bf16 source (~70 GB) -- whole-repo
    // fetch; quantize with model-quantize (4-bit AWQ + on-device auto-calib;
    // the dense bf16 path covers the MoE experts so calibration can run).
    {.family = "Qwen", .version = "3.6", .param_class = "35B-A3B",
     .variant = "bf16 (Qwen)",
     .hf_path = "Qwen/Qwen3.6-35B-A3B",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    // The same 35B-A3B, pre-quantized OptiQ mixed 4/8-bit (256 routed
    // experts + shared expert; router / shared expert / embed / lm_head w8).
    // Whole-repo fetch (5 shards + the optiq/ companions). Verified
    // end-to-end. This is the mixed-precision MoE case: the routed experts
    // disagree across gate/up/down inside layers 1, 3 and 39, and the shared
    // expert sits at w8 on all 40 layers over a w4-global model, so every
    // expert dispatch takes its width per TENSOR. Layers whose gate and up
    // disagree cannot share the interleaved slab and run the mixed-width
    // fused SwiGLU (affine_gather_qmv_swiglu_w4w8g64 / _w8w4g64), which
    // reads each side at its native width.
    {.family = "Qwen", .version = "3.6", .param_class = "35B-A3B",
     .variant = "MLX OptiQ 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "9B",
     .variant = "MLX 4-bit (lmstudio-community)",
     .hf_path = "lmstudio-community/Qwen3.5-9B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "9B",
     .variant = "MLX 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-9B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "9B",
     .variant = "MLX OptiQ 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-9B-OptiQ-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "MLX 4-bit (lmstudio-community)",
     .hf_path = "lmstudio-community/Qwen3.5-4B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "MLX 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-4B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "MLX OptiQ 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-4B-OptiQ-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "MLX 8-bit (lmstudio-community)",
     .hf_path = "lmstudio-community/Qwen3.5-4B-MLX-8bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "MLX 8-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-4B-MLX-8bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "Q4_K_M GGUF +mmproj (unsloth)",
     .hf_path = "unsloth/Qwen3.5-4B-GGUF", .model_type = "qwen3.5",
     .files = {"Qwen3.5-4B-Q4_K_M.gguf",   // main quant (text)
               "mmproj-BF16.gguf"},        // BF16 multimodal projector
     .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "MTP Q4_K_M GGUF +mmproj +imatrix (unsloth)",
     .hf_path = "unsloth/Qwen3.5-4B-MTP-GGUF", .model_type = "qwen3.5",
     .files = {"Qwen3.5-4B-Q4_K_M.gguf",   // main quant (MTP)
               "mmproj-BF16.gguf",         // BF16 multimodal projector
               "imatrix_unsloth.gguf_file"},  // imatrix
     .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "2B",
     .variant = "MLX 4-bit (lmstudio-community)",
     .hf_path = "lmstudio-community/Qwen3.5-2B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "2B",
     .variant = "MLX 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-2B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "2B",
     .variant = "MLX OptiQ 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-2B-OptiQ-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "2B",
     .variant = "MLX 8-bit (lmstudio-community)",
     .hf_path = "lmstudio-community/Qwen3.5-2B-MLX-8bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "2B",
     .variant = "MLX 8-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-2B-MLX-8bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "2B",
     .variant = "Q4_K_M GGUF +mmproj (unsloth)",
     .hf_path = "unsloth/Qwen3.5-2B-GGUF", .model_type = "qwen3.5",
     .files = {"Qwen3.5-2B-Q4_K_M.gguf",   // main quant (text)
               "mmproj-BF16.gguf"},        // BF16 multimodal projector
     .needs_tokenizer_json = false},
    // ---- Qwen 3.6 (multimodal MTP, GGUF) -----------------------------
    {.family = "Qwen", .version = "3.6", .param_class = "27B",
     .variant = "Q4_K_M GGUF +mmproj +imatrix (unsloth)",
     .hf_path = "unsloth/Qwen3.6-27B-MTP-GGUF", .model_type = "qwen3.6",
     .files = {"Qwen3.6-27B-Q4_K_M.gguf",   // main quant (MTP)
               "mmproj-BF16.gguf",          // BF16 multimodal projector
               "imatrix_unsloth.gguf_file"},  // imatrix
     .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3-ASR", .param_class = "1.7B",
     .variant = "ASR MLX 8-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3-ASR-1.7B-8bit",
     .model_type = "qwen3-asr", .needs_tokenizer_json = true},
    {.family = "Qwen", .version = "3-ASR", .param_class = "0.6B",
     .variant = "ASR MLX 8-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3-ASR-0.6B-8bit",
     .model_type = "qwen3-asr", .needs_tokenizer_json = true},
    {.family = "Gemma", .version = "4", .param_class = "E4B",
     .variant = "MLX 4-bit (mlx-community)",
     .hf_path = "mlx-community/gemma-4-e4b-it-4bit",
     .model_type = "gemma4", .needs_tokenizer_json = false},
    {.family = "Gemma", .version = "4", .param_class = "12B",
     .variant = "GGUF QAT q4_0 (google, gated)",
     .hf_path = "google/gemma-4-12B-it-qat-q4_0-gguf",
     .model_type = "gemma4_unified", .needs_tokenizer_json = false},
    // Pre-quantized OptiQ: mixed 4/8-bit per LINEAR over a group-64 affine
    // base -- the per-tensor mixed path (`_mixed` in metal-gemma-model.cc,
    // detected by probing each layer's q/k/v + gate/up widths). Whole-repo
    // fetch; the recursive tree walk also picks up optiq/optiq_vision.
    {.family = "Gemma", .version = "4", .param_class = "12B",
     .variant = "MLX OptiQ 4-bit (mlx-community)",
     .hf_path = "mlx-community/gemma-4-12B-it-OptiQ-4bit",
     .model_type = "gemma4_unified", .needs_tokenizer_json = false},
    // The dense 31B, same per-layer tensor layout as the 12B OptiQ above
    // (60 layers, hidden 5376) -- it loads on the same mixed path.
    {.family = "Gemma", .version = "4", .param_class = "31B",
     .variant = "MLX OptiQ 4-bit (mlx-community)",
     .hf_path = "mlx-community/gemma-4-31B-it-OptiQ-4bit",
     .model_type = "gemma4_unified", .needs_tokenizer_json = false},
    // The 26B-A4B MoE (128 experts, 30 layers). Verified end-to-end. The
    // OptiQ pack differs from the model-quantize output in three ways the
    // loader now handles: the routed experts are named
    // experts.switch_glu.{gate,up,down}_proj, their width varies PER LAYER
    // (w8 on layers 0-4 + 29, w4 on the other 24 -- gate/up/down agree
    // within a layer), and router.proj arrives QUANTIZED (w8) where the
    // router GEMV is dense, so it is dequantized at load.
    {.family = "Gemma", .version = "4", .param_class = "26B-A4B",
     .variant = "MLX OptiQ 4-bit (mlx-community)",
     .hf_path = "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit",
     .model_type = "gemma4_unified", .needs_tokenizer_json = false},
    // Raw google bf16 Gemma-4-it releases (gated; whole-repo fetch --
    // quantize with model-quantize before running). E-series are the
    // gemma3n-style effective (PLE) models -> "gemma4"; the dense 12B/31B
    // -> "gemma4_unified"; 26B-A4B is a MoE (4B active).
    {.family = "Gemma", .version = "4", .param_class = "E2B",
     .variant = "bf16 (google, gated)",
     .hf_path = "google/gemma-4-E2B-it",
     .model_type = "gemma4", .needs_tokenizer_json = false},
    {.family = "Gemma", .version = "4", .param_class = "E4B",
     .variant = "bf16 (google, gated)",
     .hf_path = "google/gemma-4-E4B-it",
     .model_type = "gemma4", .needs_tokenizer_json = false},
    {.family = "Gemma", .version = "4", .param_class = "12B",
     .variant = "bf16 (google, gated)",
     .hf_path = "google/gemma-4-12B-it",
     .model_type = "gemma4_unified", .needs_tokenizer_json = false},
    {.family = "Gemma", .version = "4", .param_class = "31B",
     .variant = "bf16 (google, gated)",
     .hf_path = "google/gemma-4-31B-it",
     .model_type = "gemma4_unified", .needs_tokenizer_json = false},
    {.family = "Gemma", .version = "4", .param_class = "26B-A4B",
     .variant = "bf16 MoE (google, gated)",
     .hf_path = "google/gemma-4-26B-A4B-it",
     .model_type = "gemma4_unified", .needs_tokenizer_json = false},
    // ---- MOSS-TTS (text-to-speech: LM + audio codec) -----------------
    // The text-to-speech stage consumes TWO models: the MOSS-TTS LM
    // (the stage's hf_dir, model_type "moss-tts") and the
    // MOSS-Audio-Tokenizer codec (the stage's codec_dir, model_type
    // "moss-codec"). Both are whole-repo fetches (empty `files`):
    // sharded safetensors with no consolidated tokenizer.json to
    // synthesize. The stage's hf_dir/codec_dir fields filter the
    // suggestion dropdown on these two model_types.
    {.family = "MOSS", .version = "TTS", .param_class = "8B",
     .variant = "MLX 8-bit (mlx-community)",
     .hf_path = "mlx-community/MOSS-TTS-8B-8bit",
     .model_type = "moss-tts", .needs_tokenizer_json = false},
    {.family = "MOSS", .version = "Audio-Tokenizer", .param_class = "codec",
     .variant = "F32 (OpenMOSS-Team)",
     .hf_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer",
     .model_type = "moss-codec", .needs_tokenizer_json = false},
    // MOSS-TTS-Local-v1.5: the text-to-speech stage's v1.5 LM (hf_dir,
    // model_type "moss-tts-local"; quantize it with model-quantize before
    // use) + its 48 kHz stereo codec (codec_dir, "moss-codec-v2"). The
    // text-to-speech stage auto-detects this variant from config.json.
    {.family = "MOSS", .version = "TTS-Local", .param_class = "v1.5",
     .variant = "bf16 (OpenMOSS-Team)",
     .hf_path = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
     .model_type = "moss-tts-local", .needs_tokenizer_json = false},
    {.family = "MOSS", .version = "Audio-Tokenizer", .param_class = "codec-v2",
     .variant = "F32 (OpenMOSS-Team)",
     .hf_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
     .model_type = "moss-codec-v2", .needs_tokenizer_json = false},
    // MOSS-TTS-Realtime: the text-to-speech stage's realtime LM (hf_dir,
    // model_type "moss-tts-realtime"). Runs as-is (unquantized bf16) OR after
    // model-quantize to 8-bit (~2x faster, half the resident bytes). A
    // context-aware streaming TTS: a bf16
    // Qwen3-1.7B backbone drives a 4-layer Qwen3-style depth ("local")
    // transformer that autoregressively emits 16 RVQ codebooks per 12.5 Hz
    // frame. It reuses the 24 kHz MOSS-Audio-Tokenizer codec (codec_dir,
    // "moss-codec", above) -- decoded with the first 16 codebooks. The
    // text-to-speech stage auto-detects this variant from config.json.
    {.family = "MOSS", .version = "TTS-Realtime", .param_class = "1.7B",
     .variant = "bf16 (OpenMOSS-Team)",
     .hf_path = "OpenMOSS-Team/MOSS-TTS-Realtime",
     .model_type = "moss-tts-realtime", .needs_tokenizer_json = false},
    // ---- Krea (text-to-image diffusion) ------------------------------
    // Krea-2-Turbo: a flow-matching (rectified-flow) text-to-image model,
    // model_type "krea2". Diffusers-layout repo with per-component
    // subfolders -- three sub-models the generate-image stages consume:
    //   text_encoder/ = Qwen3-VL (model_type qwen3_vl, hidden 2560, 36L);
    //     the pipeline conditions on 12 SELECTED hidden layers
    //     (text_encoder_select_layers in model_index.json), not just the
    //     last state -- reuses the metal Qwen backbone.
    //   transformer/  = Krea2Transformer2DModel, a 12B dual-stream MMDiT
    //     (28 image blocks, head_dim 128, 48 q-heads GQA kv=12, 3D-RoPE
    //     axes [32,48,48]; in_channels 64 = 16 latent x 2x2 patch;
    //     interleaved 12-layer text tower). CFG-DISTILLED turbo: ~8 steps
    //     at guidance_scale 0 -- no classifier-free pass.
    //   vae/          = AutoencoderKLQwenImage (16 latent ch, 8x spatial,
    //     per-channel latents_mean/std whitening) -- the separate VAE stage.
    // The text/encoder->DiT stage reads text_encoder/ + transformer/ (+
    // scheduler/, tokenizer/, model_index.json); the VAE stage reads vae/.
    // `files` PINS the diffusers subfolders (sharded transformer via its
    // index.json) and SKIPS the redundant top-level turbo.safetensors (a
    // second copy of the transformer, ~26 GB) and the sample images/ --
    // fetching ~35.6 GB instead of the ~62 GB whole repo.
    {.family = "Krea", .version = "2", .param_class = "12B",
     .variant = "Turbo distilled bf16 (krea)",
     .hf_path = "krea/Krea-2-Turbo",
     .model_type = "krea2",
     .files = {"model_index.json",
               "transformer/config.json",
               "transformer/diffusion_pytorch_model.safetensors.index.json",
               "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
               "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
               "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
               "text_encoder/config.json",
               "text_encoder/model.safetensors",
               "vae/config.json",
               "vae/diffusion_pytorch_model.safetensors",
               "tokenizer/tokenizer.json",
               "tokenizer/tokenizer_config.json",
               "tokenizer/chat_template.jinja",
               "scheduler/scheduler_config.json"},
     .needs_tokenizer_json = false},
    // Krea-2-Raw: the NON-distilled sibling of Krea-2-Turbo -- same diffusers
    // layout + topology (Qwen3-VL text_encoder, Krea2Transformer2DModel, Qwen-
    // Image VAE), but NOT CFG-distilled, so it runs classifier-free guidance
    // (CFG>1). This is the model the identity-edit README prescribes for
    // DELETION edits ("Raw model at CFG 3, ~20 steps"); Turbo (CFG-1 only)
    // handles restyle/replace but not deletions. Same `files` pinning as Turbo
    // (SKIPS the redundant top-level raw.safetensors, ~26 GB).
    {.family = "Krea", .version = "2", .param_class = "12B",
     .variant = "Raw bf16 (krea)",
     .hf_path = "krea/Krea-2-Raw",
     .model_type = "krea2",
     .files = {"model_index.json",
               "transformer/config.json",
               "transformer/diffusion_pytorch_model.safetensors.index.json",
               "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
               "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
               "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
               "text_encoder/config.json",
               "text_encoder/model.safetensors",
               "vae/config.json",
               "vae/diffusion_pytorch_model.safetensors",
               "tokenizer/tokenizer.json",
               "tokenizer/tokenizer_config.json",
               "tokenizer/chat_template.jinja",
               "scheduler/scheduler_config.json"},
     .needs_tokenizer_json = false},
    // Krea-2 softwatercolor LoRA (adapts the Turbo DiT). Fuse into a DiT with
    // the lora-fuse stage (base = <Krea-2-Turbo>/transformer), then use the
    // fused DiT via the generate-image `dit_dir`. Trigger: "Art Deco watercolor
    // style". A single ~0.47 GB safetensors (lora_A/B pairs, rank 32).
    {.family = "Krea", .version = "2", .param_class = "LoRA",
     .variant = "softwatercolor LoRA (krea)",
     .hf_path = "krea/Krea-2-LoRA-softwatercolor",
     .model_type = "krea2-lora",
     .parent_model_type = "krea2",   // fuses into any Krea-2 DiT
     .files = {"softwatercolor.safetensors"},
     .needs_tokenizer_json = false},
    // M87 early-preview aesthetic LoRA (mgwr) for Krea-2 Turbo. A ~0.23 GB
    // standard low-rank LoRA (lora_A/B pairs, rank 32). Trigger: "--preview".
    // Uses the ai-toolkit / ComfyUI key convention
    // (diffusion_model.blocks.N.attn.{wq,wk,wv,wo,gate}, mlp.{gate,up,down});
    // the lora-fuse stage's name remap maps these to the diffusers base DiT
    // weights (all 256 modules fuse, verified base + B@A).
    {.family = "Krea", .version = "2", .param_class = "LoRA",
     .variant = "M87 aesthetic LoRA (mgwr)",
     .hf_path = "mgwr/M87",
     .model_type = "krea2-lora",
     .parent_model_type = "krea2",   // fuses into any Krea-2 DiT
     .files = {"m87_lora_v1.safetensors"},
     .needs_tokenizer_json = false},
    // Krea2-realism-V2 (RudySen) for Krea-2 Turbo. A ~1.56 GB LoKr (Kronecker)
    // adapter (lokr_w1/lokr_w2/alpha, full-matrix so scale=1), in the ai-toolkit
    // / ComfyUI key convention (diffusion_model.*). Trigger: "r3alism". The
    // lora-fuse stage reconstructs dW = kron(w1,w2) and name-remaps to the base
    // DiT (all 256 modules fuse, verified base + kron).
    {.family = "Krea", .version = "2", .param_class = "LoRA",
     .variant = "realism V2 LoKr (RudySen)",
     .hf_path = "RudySen/Krea2-realism-V2",
     .model_type = "krea2-lora",
     .parent_model_type = "krea2",   // fuses into any Krea-2 DiT
     .files = {"Krea2-realism-V2.safetensors"},
     .needs_tokenizer_json = false},
    // Krea-2 identity-edit LoRA (conradlocke). Unlike the style LoRAs above,
    // this is the adapter that ACTIVATES the in-context reference-edit path
    // (ComfyUI-Krea2Edit): the source image is kept as clean frame-1 tokens in
    // the DiT + the instruction is encoded image-grounded through Qwen3-VL. Wire
    // the source VAE latent to the generate-image `ref_latent0` iport (strength 0)
    // and the raw source image to the diffusion-conditioner `ref_image` iport.
    // Standard low-rank LoRA (lora_A/B), ai-toolkit / ComfyUI key convention
    // (diffusion_model.{blocks,txtfusion.{layerwise,refiner}_blocks}.N.attn.
    // {wq,wk,wv,wo,gate},mlp.{gate,up,down}); the lora-fuse name remap maps all
    // 512 modules onto the diffusers base DiT (28 main + 2+2 fusion blocks).
    // Trained on krea/Krea-2-Raw; the node + shared topology run on Turbo too.
    // v1_2 (~1.83 GB) recommended; r128/r64 are SVD rank-reduced (>99% energy).
    {.family = "Krea", .version = "2", .param_class = "LoRA",
     .variant = "identity-edit LoRA (conradlocke)",
     .hf_path = "conradlocke/krea2-identity-edit",
     .model_type = "krea2-lora",
     .parent_model_type = "krea2",   // fuses into any Krea-2 DiT
     .files = {"krea2_identity_edit_v1_2.safetensors",
               "krea2_identity_edit_v1_2_r128.safetensors",
               "krea2_identity_edit_v1_2_r64.safetensors"},
     .needs_tokenizer_json = false},
    // ---- Qwen-Image (text+image -> image editing diffusion) -----------
    // Qwen-Image-Edit-2511 (Qwen): a flow-matching multi-reference IMAGE
    // EDIT model, model_type "qwen-image-edit". Same diffusers split-stage
    // shape as Krea-2 (encoder->DiT stage + separate VAE stages) and the
    // SAME Qwen-Image VAE, but the base Qwen-Image topology (dual-stream)
    // rather than Krea's single-stream distill. Three sub-models:
    //   text_encoder/ = Qwen2.5-VL (Qwen2_5_VLForConditionalGeneration,
    //     hidden 3584, 28L, 28q/4kv, q/k/v attention BIAS, NO q/k-norm,
    //     mrope [16,24,24]) + a 32-layer vision tower (window attention,
    //     full-attn blocks [7,15,23,31]). Reference images are fed to the
    //     vision tower so the LM conditions on them; the pipeline uses the
    //     LAST hidden state -> txt_in (not a multi-layer tap).
    //   transformer/  = QwenImageTransformer2DModel, a 20B DUAL-STREAM
    //     MMDiT (60 blocks, 24 heads x head_dim 128 = 3072 hidden, in_ch
    //     64 = 16 latent x 2x2 patch, joint_attn_dim 3584, 3D-RoPE axes
    //     [16,56,56], guidance_embeds false). Reference latents are VAE-
    //     encoded and concatenated into the DiT sequence (RefImage tokens).
    //   vae/          = AutoencoderKLQwenImage (16 latent ch, 8x spatial,
    //     per-channel latents_mean/std) -- identical to Krea-2's VAE.
    // `files` PINS the diffusers subfolders (sharded transformer + text
    // encoder via their index.json) + the processor/ fast tokenizer +
    // image preprocessor + multimodal chat template; skips the README /
    // sample media. ~55 GB (20B DiT + 7B VL encoder, bf16).
    {.family = "Qwen-Image", .version = "Edit-2511", .param_class = "20B",
     .variant = "bf16 (Qwen)",
     .hf_path = "Qwen/Qwen-Image-Edit-2511",
     .model_type = "qwen-image-edit",
     .files = {"model_index.json",
               "transformer/config.json",
               "transformer/diffusion_pytorch_model.safetensors.index.json",
               "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
               "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
               "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
               "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
               "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
               "text_encoder/config.json",
               "text_encoder/model.safetensors.index.json",
               "text_encoder/model-00001-of-00004.safetensors",
               "text_encoder/model-00002-of-00004.safetensors",
               "text_encoder/model-00003-of-00004.safetensors",
               "text_encoder/model-00004-of-00004.safetensors",
               "vae/config.json",
               "vae/diffusion_pytorch_model.safetensors",
               "tokenizer/tokenizer_config.json",
               "tokenizer/vocab.json",
               "tokenizer/merges.txt",
               "tokenizer/special_tokens_map.json",
               "tokenizer/added_tokens.json",
               "processor/tokenizer.json",
               "processor/preprocessor_config.json",
               "processor/chat_template.jinja",
               "scheduler/scheduler_config.json"},
     .needs_tokenizer_json = false},
    // FLUX.2-klein-4B (black-forest-labs) -- a diffusers text-to-image
    // pipeline in the SAME split-stage shape as Krea-2 (encoder->DiT stage +
    // separate VAE stages), but the FLUX topology rather than Qwen-Image
    // MMDiT. Sub-models:
    //   text_encoder/ = Qwen3ForCausalLM (DENSE: 36 layers, hidden 2560,
    //     32 q-heads GQA kv=8, head_dim 128, tied embeds, rope theta 1e6).
    //     FLUX.2 taps hidden states from layers {10,20,30} and CONCATENATES
    //     them -> 3 x 2560 = 7680-dim prompt embeddings (max 512 tokens).
    //   transformer/ = Flux2Transformer2DModel, a 4B FLUX-topology DiT:
    //     5 double-stream (MMDiT joint) + 20 single-stream blocks, 24 heads
    //     x head_dim 128 = 3072 hidden, in_channels 128 (= 32 latent x 2x2
    //     patch), joint_attn_dim 7680, mlp_ratio 3.0, 4-axis RoPE
    //     [32,32,32,32] theta 2000. DISTILLED (guidance_embeds=false) -- ~no
    //     classifier-free pass.
    //   vae/          = AutoencoderKLFlux2 (32 latent ch, 8x spatial + [2,2]
    //     patch, block_out [128,256,512,512], mid-block attention) -- the
    //     separate VAE stages.
    // `files` PINS the diffusers subfolders (sharded text_encoder via its
    // index.json; single-file transformer) and SKIPS the redundant top-level
    // flux-2-klein-4b.safetensors (a second copy of the transformer, ~7.75
    // GB) and the sample images -- fetching ~16 GB instead of ~23.7 GB.
    {.family = "FLUX", .version = "2", .param_class = "4B",
     .variant = "klein distilled bf16 (black-forest-labs)",
     .hf_path = "black-forest-labs/FLUX.2-klein-4B",
     .model_type = "flux2",
     .files = {"model_index.json",
               "transformer/config.json",
               "transformer/diffusion_pytorch_model.safetensors",
               "text_encoder/config.json",
               "text_encoder/generation_config.json",
               "text_encoder/model.safetensors.index.json",
               "text_encoder/model-00001-of-00002.safetensors",
               "text_encoder/model-00002-of-00002.safetensors",
               "vae/config.json",
               "vae/diffusion_pytorch_model.safetensors",
               "tokenizer/tokenizer.json",
               "tokenizer/tokenizer_config.json",
               "tokenizer/chat_template.jinja",
               "scheduler/scheduler_config.json"},
     .needs_tokenizer_json = false},
    // FLUX.2-klein-9B (black-forest-labs) -- the larger klein sibling, same
    // split-stage FLUX topology as the 4B (all sub-model code is config-driven
    // off config.json, so one code path serves both sizes). Differences:
    //   text_encoder/ = an 8B Qwen3 (4 shards) rather than the 4B's ~4B Qwen3
    //     (still dense Qwen3ForCausalLM, tapped at layers {9,18,27}).
    //   transformer/ = a larger Flux2Transformer2DModel (2 shards), and it is
    //     GUIDANCE-DISTILLED (guidance_embeds=true): a guidance_embedder embeds
    //     the guidance scale into the timestep embedding (single forward pass;
    //     the distilled default runs ~4 steps at guidance_scale 1.0).
    //   vae/          = AutoencoderKLFlux2 (single file), same as the 4B.
    // `files` PINS the diffusers subfolders (sharded transformer + text_encoder
    // via their index.json) and SKIPS the redundant top-level
    // flux-2-klein-9b.safetensors (a second copy of the transformer) and the
    // sample images.
    {.family = "FLUX", .version = "2", .param_class = "9B",
     .variant = "klein guidance-distilled bf16 (black-forest-labs)",
     .hf_path = "black-forest-labs/FLUX.2-klein-9B",
     .model_type = "flux2",
     .files = {"model_index.json",
               "transformer/config.json",
               "transformer/diffusion_pytorch_model.safetensors.index.json",
               "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
               "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
               "text_encoder/config.json",
               "text_encoder/generation_config.json",
               "text_encoder/model.safetensors.index.json",
               "text_encoder/model-00001-of-00004.safetensors",
               "text_encoder/model-00002-of-00004.safetensors",
               "text_encoder/model-00003-of-00004.safetensors",
               "text_encoder/model-00004-of-00004.safetensors",
               "vae/config.json",
               "vae/diffusion_pytorch_model.safetensors",
               "tokenizer/tokenizer.json",
               "tokenizer/tokenizer_config.json",
               "tokenizer/chat_template.jinja",
               "scheduler/scheduler_config.json"},
     .needs_tokenizer_json = false},
    // FLUX.2-klein-9b-kv (black-forest-labs) -- the KV-cached sibling of
    // klein-9B. Same PARAMETER shapes (Klein9BParams is byte-identical: the
    // repo layout matches klein-9B file for file -- 2 transformer shards, the
    // same 4-shard 8B Qwen3 text_encoder, one VAE) and the same
    // guidance-distilled 4-step recipe, so the loader, quantizer and LoRA
    // fusion need nothing new. What differs is the ATTENTION TOPOLOGY, and it
    // is not a drop-in:
    //   - Token order is [text, refs, image] -- references BEFORE the
    //     generated tokens -- where plain klein-9B (and our forward_dit)
    //     appends references AFTER them.
    //   - Reference tokens SELF-ATTEND ONLY. In plain klein-9B they join the
    //     full joint attention; here they see neither text nor the generated
    //     image, which is exactly what makes their K/V independent of the
    //     denoising step and therefore cacheable.
    // The weights are distilled FOR that masking, so running this checkpoint
    // through the plain flux2 forward (or the reverse) is not merely slower,
    // it is wrong. The payoff is that a multi-reference edit computes the
    // reference K/V once at step 0 and reuses it for the remaining steps:
    // BFL measure 1.21-2.66x, largest with several references at small output
    // sizes.
    // GATED: the FLUX Non-Commercial License must be accepted on the repo
    // page and an HF token supplied -- an unauthenticated fetch 401s.
    // `files` mirrors the klein-9B pin: the diffusers subfolders, skipping the
    // redundant top-level flux-2-klein-9b-kv.safetensors (a second copy of the
    // transformer, 18.2 GB) and the three sample images (~6.3 MB).
    {.family = "FLUX", .version = "2", .param_class = "9B",
     .variant = "klein KV-cached guidance-distilled bf16 (black-forest-labs)",
     .hf_path = "black-forest-labs/FLUX.2-klein-9b-kv",
     .model_type = "flux2",
     .files = {"model_index.json",
               "transformer/config.json",
               "transformer/diffusion_pytorch_model.safetensors.index.json",
               "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
               "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
               "text_encoder/config.json",
               "text_encoder/generation_config.json",
               "text_encoder/model.safetensors.index.json",
               "text_encoder/model-00001-of-00004.safetensors",
               "text_encoder/model-00002-of-00004.safetensors",
               "text_encoder/model-00003-of-00004.safetensors",
               "text_encoder/model-00004-of-00004.safetensors",
               "vae/config.json",
               "vae/diffusion_pytorch_model.safetensors",
               "tokenizer/tokenizer.json",
               "tokenizer/tokenizer_config.json",
               "tokenizer/chat_template.jinja",
               "scheduler/scheduler_config.json"},
     .needs_tokenizer_json = false},
    // ---- Mage-Flow (microsoft) -- native-resolution t2i + image edit --
    // A 4B flow-matching family in the SAME split-stage diffusers shape as
    // Krea-2 / FLUX.2 (encoder -> DiT stage + separate VAE stages). Two
    // instantiations share ONE architecture and one code path: Mage-Flow
    // (text-to-image, model_type "mage-flow") and Mage-Flow-Edit
    // (instruction editing, "mage-flow-edit"); each ships Base, RL-aligned
    // and 4-step Turbo weights. All six repos are byte-identical in layout
    // (~17.5 GB bf16). Sub-models:
    //   text_encoder/ = Qwen3VLForConditionalGeneration (Qwen3-VL 4B: 36L,
    //     hidden 2560, 32q/8kv GQA head_dim 128, interleaved mrope
    //     [24,20,20] theta 5e6, tied embeds; plus a 24-layer ViT, patch 16,
    //     out_hidden 2560, deepstack taps [5,11,17]). The pipeline takes the
    //     LAST hidden state (a single tap, NOT a multi-layer concat like
    //     FLUX.2) and DROPS the templated system prefix -- 34 tokens for
    //     t2i, 64 for edit. Reference images ride the VL tower (long edge
    //     capped at 384) so the instruction is image-grounded.
    //   transformer/  = MageFlow, a 4B NR-MMDiT: 12 DUAL-STREAM joint
    //     blocks (no single-stream tail), 24 heads x head_dim 128 = 3072
    //     hidden, mlp_ratio 4 (GELU), in/out_channels 128 at patch_size 1
    //     (one token per latent pixel -- no 2x2 packing), context_in_dim
    //     2560, 3-axis RoPE [16,56,56] theta 10000 (frame/height/width,
    //     scale_rope) with the TEXT stream left UNROTATED. Distilled
    //     (guidance_embed=false). Edit conditioning matches Qwen-Image-Edit:
    //     each reference occupies its own RoPE frame band (frame = index+1)
    //     and stays clean while only the target tokens are stepped.
    //   vae/          = MageVAE -- NOT an AutoencoderKL, but a symmetric
    //     one-step diffusion codec (128 latent channels, 16x downsample, so
    //     the same bytes/pixel as FLUX.2-VAE's 32ch/8x). Encoder and decoder
    //     are DiCo conv trunks (1x1 + depthwise 3x3 + channel attention,
    //     adaLN) evaluated at a FIXED t=0, so every adaLN modulation is a
    //     constant that folds at load; the decoder ends in a per-pixel
    //     32-dim MLP head. ~0.35 GB.
    // Scheduler: FlowMatchEulerDiscreteScheduler, static shift 6.0. Turbo
    // runs 4 steps at cfg 1.0 (a single forward, no negative branch);
    // Base/RL run 30 / 20 steps at cfg 5.0.
    {.family = "Mage-Flow", .version = "Gen", .param_class = "4B",
     .variant = "base bf16 (microsoft)",
     .hf_path = "microsoft/Mage-Flow-Base",
     .model_type = "mage-flow",
     .files = kMageFlowFiles,
     .needs_tokenizer_json = false},
    {.family = "Mage-Flow", .version = "Gen", .param_class = "4B",
     .variant = "RL-aligned bf16 (microsoft)",
     .hf_path = "microsoft/Mage-Flow",
     .model_type = "mage-flow",
     .files = kMageFlowFiles,
     .needs_tokenizer_json = false},
    {.family = "Mage-Flow", .version = "Gen", .param_class = "4B",
     .variant = "Turbo 4-step distilled bf16 (microsoft)",
     .hf_path = "microsoft/Mage-Flow-Turbo",
     .model_type = "mage-flow",
     .files = kMageFlowFiles,
     .needs_tokenizer_json = false},
    {.family = "Mage-Flow", .version = "Edit", .param_class = "4B",
     .variant = "base bf16 (microsoft)",
     .hf_path = "microsoft/Mage-Flow-Edit-Base",
     .model_type = "mage-flow-edit",
     .files = kMageFlowFiles,
     .needs_tokenizer_json = false},
    {.family = "Mage-Flow", .version = "Edit", .param_class = "4B",
     .variant = "RL-aligned bf16 (microsoft)",
     .hf_path = "microsoft/Mage-Flow-Edit",
     .model_type = "mage-flow-edit",
     .files = kMageFlowFiles,
     .needs_tokenizer_json = false},
    {.family = "Mage-Flow", .version = "Edit", .param_class = "4B",
     .variant = "Turbo 4-step distilled bf16 (microsoft)",
     .hf_path = "microsoft/Mage-Flow-Edit-Turbo",
     .model_type = "mage-flow-edit",
     .files = kMageFlowFiles,
     .needs_tokenizer_json = false},
    // ---- Boogu-Image (text+image -> image, NextDiT lineage) -----------
    // Boogu-Image-0.1 (Boogu Team): a 10B flow-matching family in the same
    // diffusers split-stage shape as Krea-2 / FLUX.2 / Mage-Flow (encoder ->
    // DiT stage + separate VAE stages), but a NextDiT / Lumina-Image-2.0
    // topology rather than an MMDiT. Four repos share one architecture and one
    // code path: Base + Turbo are text-to-image ("boogu-image"), Edit +
    // Edit-Turbo take one reference image ("boogu-image-edit"); the two Turbo
    // variants are the SAME parameter count, DMD-distilled to 4 steps at
    // guidance 1. Sub-models:
    //   mllm/        = Qwen3VLForConditionalGeneration (Qwen3-VL 8B: 36L,
    //     hidden 4096, 32q/8kv GQA head_dim 128, interleaved mrope [24,20,20]
    //     theta 5e6, UNTIED embeds; plus a 27-layer ViT, patch 16, out_hidden
    //     4096, deepstack taps [8,16,24]). The pipeline takes the LAST hidden
    //     state and -- unlike every other family here -- DROPS NOTHING: the
    //     whole templated sequence, system prompt included, is conditioning.
    //     Reference images ride the VL tower (capped at 384x384) so the
    //     instruction is image-grounded.
    //   transformer/ = BooguImageTransformer2DModel, a 10B NextDiT: 3 refiner
    //     stacks (context / noise / ref-image, 2 blocks each) -> 8 DUAL-STREAM
    //     blocks (joint attention over [instruct; ref; noise] PLUS a second
    //     image-only self-attention per block) -> 32 single-stream blocks over
    //     the fused sequence, 28 heads x head_dim 120 = 3360 hidden, GQA kv=7,
    //     in_channels 16 at patch_size 2, instruction_feat_dim 4096, 3-axis
    //     RoPE [40,40,40] theta 10000. RMSNorm throughout with TANH-gated
    //     residuals (Lumina RMSNormZero).
    //   vae/         = the FLUX.1 AutoencoderKL (16 latent ch, 8x spatial, no
    //     quant/post-quant conv, scalar shift 0.1159 / scale 0.3611) -- served
    //     by the same code as the FLUX.2 VAE at patch 1.
    // Scheduler: FlowMatchEulerDiscrete with the v1 logistic time shift
    // (do_shift, static seq_len 4096) -- and note its sigma convention is
    // INVERTED (0 = noise, 1 = clean). Base/Edit run 25-50 steps at CFG
    // 2.0-5.0; the Turbo pair runs 4 DMD steps at CFG 1.0.
    {.family = "Boogu-Image", .version = "0.1", .param_class = "10B",
     .variant = "Base bf16 (Boogu)",
     .hf_path = "Boogu/Boogu-Image-0.1-Base",
     .model_type = "boogu-image",
     .inputs = {"text"}, .outputs = {"image"},
     .files = kBooguFiles,
     .needs_tokenizer_json = false},
    {.family = "Boogu-Image", .version = "0.1", .param_class = "10B",
     .variant = "Turbo 4-step distilled bf16 (Boogu)",
     .hf_path = "Boogu/Boogu-Image-0.1-Turbo",
     .model_type = "boogu-image",
     .inputs = {"text"}, .outputs = {"image"},
     .files = kBooguFiles,
     .needs_tokenizer_json = false},
    {.family = "Boogu-Image", .version = "0.1-Edit", .param_class = "10B",
     .variant = "Edit bf16 (Boogu)",
     .hf_path = "Boogu/Boogu-Image-0.1-Edit",
     .model_type = "boogu-image-edit",
     .inputs = {"text", "image"}, .outputs = {"image"},
     .files = kBooguFiles,
     .needs_tokenizer_json = false},
    {.family = "Boogu-Image", .version = "0.1-Edit", .param_class = "10B",
     .variant = "Edit-Turbo 4-step distilled bf16 (Boogu)",
     .hf_path = "Boogu/Boogu-Image-0.1-Edit-Turbo",
     .model_type = "boogu-image-edit",
     .inputs = {"text", "image"}, .outputs = {"image"},
     .files = kBooguFiles,
     .needs_tokenizer_json = false},
    // ---- Wan (text+image -> VIDEO diffusion) --------------------------
    // Wan2.2-I2V-A14B (Wan-AI): the first VIDEO model here, and the first
    // that outputs anything but text / images / audio. Same diffusers
    // split-stage shape as the image families (conditioner -> DiT stage +
    // separate VAE stages), but every tensor carries a TIME axis, so it is
    // served by `generate-video` rather than `generate-image`. Sub-models:
    //   text_encoder/ = UMT5EncoderModel (umT5-XXL encoder: 24L, d_model
    //     4096, d_ff 10240, 64 heads x d_kv 64, gated-GELU, RMSNorm, vocab
    //     256384). Encoder-only -- there is no decoder half in the repo.
    //     Unlike T5 it carries a relative-attention bias PER LAYER rather
    //     than sharing layer 0's. The pipeline feeds the DiT a fixed
    //     512-token context (padded / truncated), so `text_dim` 4096.
    //   transformer/ + transformer_2/ = TWO WanTransformer3DModel experts
    //     ("A14B" = 14B active): 40 blocks, 40 heads x head_dim 128 = 5120
    //     hidden, ffn 13824 (GELU-approx, ungated), patch (1,2,2) over a
    //     3D latent, 3-axis RoPE split 44/42/42 (t/h/w, interleaved),
    //     rms_norm_across_heads on q/k, and a cross-attention to the text
    //     context in every block. Modulation is a SHARED 6-way
    //     scale_shift_table added to the timestep projection, and the norms
    //     run in fp32. transformer/ is the HIGH-noise expert and
    //     transformer_2/ the LOW-noise one; the sampler switches at
    //     boundary_ratio 0.9 of the schedule (model_index.json), so only
    //     one 54 GB expert is resident at a time.
    //   vae/          = AutoencoderKLWan (16 latent ch, 8x spatial AND 4x
    //     TEMPORAL, per-channel latents_mean/std whitening). This is the
    //     net the Qwen-Image VAE (Krea-2 / Qwen-Image-Edit) is the
    //     single-frame specialization of -- same base_dim 96, dim_mult
    //     [1,2,4,4], 2 res blocks -- with the time axis restored: causal
    //     3D convs run over frame chunks with a feature cache, and two of
    //     the three resample stages also resample in time.
    // I2V conditioning is CHANNEL-WISE, not cross-attention: the DiT takes
    // 36 in_channels = 16 noise + 4 first-frame mask + 16 VAE latent of the
    // conditioning image. There is no CLIP image tower (image_dim null) --
    // Wan2.2 dropped the one Wan2.1-I2V had.
    // Scheduler: UniPCMultistepScheduler on flow sigmas, shift 3.0, order
    // 2 (bh2), ~40 steps at guidance 3.5 for both experts.
    {.family = "Wan", .version = "2.2-I2V", .param_class = "A14B",
     .variant = "MoE fp32 (Wan-AI, diffusers)",
     .hf_path = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
     .model_type = "wan-i2v",
     .inputs = {"text", "image"}, .outputs = {"video"},
     .files = kWan22I2VFiles,
     .needs_tokenizer_json = false},
    // MiniMax-H3 FL2VA (MiniMaxAI): the first model here that generates
    // VIDEO AND AUDIO from one denoise loop, and the first whose DiT is
    // too large to hold in bf16 on any box we have (33B = 66 GB), so
    // quantizing it is a precondition for running it at all rather than
    // an optimization. "FL2VA" = First-and-Last-frame to Video+Audio: it
    // takes zero, one or two images, which makes one checkpoint serve
    // t2va, first-frame, last-frame and first-and-last-frame. Sub-models:
    //   transformer/  = MiniMaxH3DiTModel, a 33B dense SINGLE-STREAM
    //     transformer over ONE packed sequence holding text + audio +
    //     video rows: 50 blocks, hidden 5376 but 56 heads x head_dim 128
    //     = 7168 attention inner width (WIDER than the residual stream,
    //     which is unusual), SwiGLU ffn 14336, patch (1,2,2). Attention
    //     is plain full self-attention -- there is no cross-attention
    //     anywhere -- and neither attention nor FFN is modality-specific.
    //     Modality enters only at the two input patch projections, the
    //     two output heads, and the AdaLN: each block projects the shared
    //     timestep embedding 2688 -> 96768 = 6 modulation vectors x 3
    //     modalities (0 video, 1 text, 2 audio), and every ROW picks its
    //     own by `timestep_index * 3 + tag`. That one projection is 260M
    //     parameters, so the AdaLN branches alone are ~13B of the 33B.
    //     One forward serves rows at DIFFERENT noise levels, which is how
    //     the keyframes stay pinned while the generated rows denoise.
    //   text_encoder/ = the full Qwen3-VL-32B (64L, hidden 5120), read
    //     for its layer-50 hidden states rather than for tokens, so it is
    //     an encoder here despite being a complete VLM. `text_dim` 5120.
    //   video_vae/    = MiniMaxH3VideoVAE, 16x SPATIAL and 4x temporal
    //     with 24 latent channels and per-channel mean/std whitening. At
    //     10.4 GB it is by far the largest VAE catalogued -- it carries a
    //     ViT alongside the CNN.
    //   audio_vae/    = MiniMaxH3AudioVAE, a DAC/BigVGAN stereo codec at
    //     32 kHz compressing to 32 latent channels at 40 Hz.
    // Scheduler: rectified-flow Euler with an exponential sigma shift,
    // run as TWO schedules per request (shift 12.0 video / 3.0 audio,
    // from model_index.json). Two traps versus every other flow model
    // here: the velocity is DATA-ward, so `x0 = x_t + sigma * v` rather
    // than minus, and timesteps are `t = 1 - sigma` in [0, 1] with t = 1
    // meaning clean -- the opposite direction from the usual 1000x sigma.
    // Output is 24 fps, 4-15 s, frame counts of the form 17n+5, on a
    // canvas whose axes are multiples of 32 (768 short edge by default).
    {.family = "MiniMax", .version = "H3-FL2VA", .param_class = "33B",
     .variant = "bf16 omni video+audio (MiniMaxAI, diffusers)",
     .hf_path = "MiniMaxAI/MiniMax-H3",
     .model_type = "minimax-h3-fl2va",
     .inputs = {"text", "image"}, .outputs = {"video", "audio"},
     .files = kMiniMaxH3FL2VAFiles,
     .needs_tokenizer_json = false},
    // The SAME model, from Comfy-Org's repack -- kept as a second entry
    // rather than replacing the one above because its value is being a
    // SECOND OPINION on the first. The two disagree about one thing that
    // no checkpoint states: MiniMaxAI groups the DiT's fused qkv
    // projection per head, Comfy-Org reorders it flat, under identical
    // names and shapes. Reading one as the other loads cleanly and
    // scrambles attention in all 50 blocks -- which is exactly the bug
    // that cost this bring-up, and having both copies on disk is what
    // turns "is our loader right?" into a diff instead of a guess.
    //
    // A COMPLETE pipeline, and the cheaper of the two: 123.6 GB against
    // the MiniMaxAI entry's 144.0 GB. The saving is entirely the text
    // encoder -- 51.5 GB here against 66.7 there, for the same 33B DiT --
    // because Comfy-Org's is TRUNCATED at the tap. Its header carries
    // layers 0..49 and says so
    // (`{"num_hidden_layers": 50, "output":
    // "unnormalized_hidden_after_layer_50"}`), where the released
    // checkpoint ships all 64 and we load 14 that never run.
    //
    // The one thing the repack does not ship is a tokenizer -- it is
    // weights-only, no configs of any kind -- so 11 MB of MiniMaxAI's
    // comes along as a companion. Without it a fetch of this entry
    // reports success and leaves a directory that cannot encode a
    // prompt. The encoder needs no config.json from either publisher:
    // its geometry is measured off the tensor shapes (comfy_config_ in
    // minimax-h3-text-encoder.cc), which is why only the tokenizer has
    // to be borrowed.
    //
    // Kept as a second entry rather than replacing the one above because
    // its other value is being a SECOND OPINION. The two disagree about
    // one thing that no checkpoint states: MiniMaxAI groups the DiT's
    // fused qkv projection per head, Comfy-Org reorders it flat, under
    // identical names and shapes. Reading one as the other loads cleanly
    // and scrambles attention in all 50 blocks -- which is exactly the
    // bug that cost this bring-up, and having both copies on disk is
    // what turns "is our loader right?" into a diff instead of a guess.
    //
    // Only the bf16 / fp16 / fp32 files are pinned, which is why `files`
    // is a whitelist rather than "fetch the repo": the same repo holds
    // 341.5 GB more in int8_convrot / fp8_scaled / nvfp4_awq packings
    // this build does not read, in `pruned` DiTs (a different model, not
    // a different packing of this one), and in the Ref2VA partition,
    // whose packed layout this tree does not implement. Same reason the
    // GGUF entries pin one quant instead of taking the whole repo.
    {.family = "MiniMax", .version = "H3-FL2VA", .param_class = "33B",
     .variant = "bf16/fp16 single-file (Comfy-Org)",
     .hf_path = "Comfy-Org/MiniMax-H3",
     .model_type = "minimax-h3-fl2va",
     .inputs = {"text", "image"}, .outputs = {"video", "audio"},
     .files = {"diffusion_models/minimax_h3_fl2va_bf16.safetensors",
               "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
               "vae/minimax_h3_video_vae_fp16.safetensors",
               "vae/minimax_h3_audio_vae_fp32.safetensors"},
     // `tokenizer/` at the repo root is one of the four places the
     // encoder looks (see the tokenizer search in its load()), so this
     // lands where it is already expected to be. tokenizer.json alone is
     // what the runtime reads -- it is self-contained, so vocab.json and
     // merges.txt would be 4.5 MB of nothing. tokenizer_config.json
     // comes along because it is 10 KB and makes the directory readable
     // by anything else that expects an HF tokenizer.
     .companion_files =
         {{.repo = "MiniMaxAI/MiniMax-H3",
           .file = "FL2VA/tokenizer/tokenizer.json",
           .dest = "tokenizer/tokenizer.json"},
          {.repo = "MiniMaxAI/MiniMax-H3",
           .file = "FL2VA/tokenizer/tokenizer_config.json",
           .dest = "tokenizer/tokenizer_config.json"}},
     .weight_format = "comfyui",
     .needs_tokenizer_json = false},
    // ---- Supplementary CoreML models (vpipe-supplement) --------------
    // One pre-converted *.mlpackage per .tar; all share ONE repo, so each
    // entry pins its archive + a distinct `name` (= registration key /
    // extract subdir) and sets extract_archive so the fetcher unpacks the
    // .tar and registers the contained .mlpackage. The model_type is the
    // compatibility hint the stages + web-ui filter on.
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "Vision tower CoreML 512x320 w8 (vpipe-supplement)",
     .hf_path = "tgo-app-dev/vpipe-supplement",
     .model_type = "qwen3.5-vision-encoder",
     .parent_model_type = "qwen3.5", .parent_param_class = "4B",
     .files = {"qwen3_5_mlx_4b_vision_vid_512x320_w8.tar"},
     .name = "qwen3_5_mlx_4b_vision_vid_512x320",
     .extract_archive = true},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "Vision tower CoreML 768x480 w8 (vpipe-supplement)",
     .hf_path = "tgo-app-dev/vpipe-supplement",
     .model_type = "qwen3.5-vision-encoder",
     .parent_model_type = "qwen3.5", .parent_param_class = "4B",
     .files = {"qwen3_5_mlx_4b_vision_vid_768x480_w8.tar"},
     .name = "qwen3_5_mlx_4b_vision_vid_768x480",
     .extract_archive = true},
    {.family = "Gemma", .version = "4", .param_class = "E4B",
     .variant = "Vision tower CoreML 768x480 w8 (vpipe-supplement)",
     .hf_path = "tgo-app-dev/vpipe-supplement",
     .model_type = "gemma4-vision-encoder",
     .parent_model_type = "gemma4", .parent_param_class = "E4B",
     .files = {"gemma4_mlx_e4b_vision_768x480_w8.tar"},
     .name = "gemma4_mlx_e4b_vision_768x480",
     .extract_archive = true},
    {.family = "YOLOX", .version = "L", .param_class = "1024x640",
     .variant = "CoreML w8 (vpipe-supplement)",
     .hf_path = "tgo-app-dev/vpipe-supplement",
     .model_type = "yolo",
     .files = {"yolox_l_1024x640_w8.tar"},
     .name = "yolox_l_1024x640",
     .extract_archive = true},
    {.family = "Silero", .version = "VAD v6", .param_class = "unified",
     .variant = "CoreML (vpipe-supplement)",
     .hf_path = "tgo-app-dev/vpipe-supplement",
     .model_type = "silero-vad",
     .files = {"silero-vad-unified-v6.tar"},
     .name = "silero_vad_unified_v6",
     .extract_archive = true},
    {.family = "BEATs", .version = "iter3+", .param_class = "AS2M",
     .variant = "Audio tagging CoreML 10s (vpipe-supplement)",
     .hf_path = "tgo-app-dev/vpipe-supplement",
     .model_type = "audio-tagging",
     .files = {"beats_as2m_10s.tar"},
     .name = "beats_as2m_10s",
     .extract_archive = true},
  };
  return kCatalog;
}

namespace {

// Append `v` to `out` if not already present (order-preserving dedupe).
void
push_unique_(std::vector<std::string>& out, const std::string& v)
{
  if (std::find(out.begin(), out.end(), v) == out.end()) {
    out.push_back(v);
  }
}

std::string
lower_(std::string s)
{
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

// Drop a case-insensitive prefix from `s` if present; returns whether it
// matched.
bool
strip_prefix_(std::string& s, const std::string& prefix)
{
  if (s.size() >= prefix.size()
      && lower_(s.substr(0, prefix.size())) == prefix) {
    s.erase(0, prefix.size());
    return true;
  }
  return false;
}

// Default input / output modalities by model_type, used when an entry
// does not record them explicitly (keeps I/O correct + DRY across the
// many same-type entries; an entry may still override via inputs/outputs).
// Published as catalog_default_io() below: model-register derives the
// same I/O for a model it detects on disk, so a registered model reads
// identically to a catalogued one.
void
default_io_(const std::string& mt, std::vector<std::string>& in,
            std::vector<std::string>& out)
{
  auto set = [&](std::initializer_list<const char*> i,
                 std::initializer_list<const char*> o) {
    for (const char* s : i) { in.emplace_back(s); }
    for (const char* s : o) { out.emplace_back(s); }
  };
  if (mt == "qwen3.5" || mt == "qwen3.6") {
    set({"text", "image", "video"}, {"text"});
  } else if (mt == "qwen3-asr") {
    set({"audio"}, {"text"});
  } else if (mt == "gemma4" || mt == "gemma4_unified") {
    // Both the effective (e4b) and the unified (12B/31B/26B-A4B) Gemma-4
    // models are multimodal-in / text-out.
    set({"text", "image", "audio", "video"}, {"text"});
  } else if (mt == "moss-tts" || mt == "moss-tts-local"
             || mt == "moss-tts-realtime") {
    set({"text"}, {"audio"});
  } else if (mt == "moss-codec" || mt == "moss-codec-v2") {
    set({"audio"}, {"audio"});
  } else if (mt == "krea2" || mt == "flux2" || mt == "qwen-image-edit"
             || mt == "mage-flow-edit") {
    set({"text", "image"}, {"image"});
  } else if (mt == "wan-i2v") {
    // The one family here that OUTPUTS video: a prompt plus a first-frame
    // image in, a clip out.
    set({"text", "image"}, {"video"});
  } else if (mt == "wan-t2v") {
    set({"text"}, {"video"});
  } else if (mt == "minimax-h3-fl2va") {
    // The only entry whose OUTPUT is two modalities: one denoise loop
    // over one packed sequence emits the clip and its soundtrack
    // together. Images are optional (zero, one or two keyframes), so the
    // text-only t2va task is the same checkpoint.
    set({"text", "image"}, {"video", "audio"});
  } else if (mt == "boogu-image-edit") {
    set({"text", "image"}, {"image"});
  } else if (mt == "boogu-image") {
    // The t2i instantiation takes no reference image (the edit
    // checkpoints, "boogu-image-edit", are the image-conditioned ones).
    set({"text"}, {"image"});
  } else if (mt == "mage-flow") {
    // The t2i instantiation takes no reference image (the edit checkpoints,
    // "mage-flow-edit", are the image-conditioned ones).
    set({"text"}, {"image"});
  } else if (mt == "yolo") {
    set({"image"}, {});
  } else if (mt == "silero-vad" || mt == "audio-tagging") {
    set({"audio"}, {});
  } else if (mt == "qwen3.5-vision-encoder"
             || mt == "gemma4-vision-encoder") {
    set({"image", "video"}, {});
  }
  // Datasets (eval-*) and unknown types keep empty I/O.
}

}  // namespace

std::vector<std::string>
catalog_families()
{
  std::vector<std::string> out;
  for (const auto& e : model_catalog()) {
    push_unique_(out, e.family);
  }
  return out;
}

std::vector<std::string>
catalog_versions(const std::string& family)
{
  std::vector<std::string> out;
  for (const auto& e : model_catalog()) {
    if (e.family == family) {
      push_unique_(out, e.version);
    }
  }
  return out;
}

std::vector<std::string>
catalog_param_classes(const std::string& family, const std::string& version)
{
  std::vector<std::string> out;
  for (const auto& e : model_catalog()) {
    if (e.family == family && e.version == version) {
      push_unique_(out, e.param_class);
    }
  }
  return out;
}

std::vector<std::string>
catalog_variants(const std::string& family, const std::string& version,
                 const std::string& param_class)
{
  std::vector<std::string> out;
  for (const auto& e : model_catalog()) {
    if (e.family == family && e.version == version
        && e.param_class == param_class) {
      push_unique_(out, e.variant);
    }
  }
  return out;
}

const ModelCatalogEntry*
catalog_find(const std::string& family, const std::string& version,
             const std::string& param_class, const std::string& variant)
{
  for (const auto& e : model_catalog()) {
    if (e.family == family && e.version == version
        && e.param_class == param_class && e.variant == variant) {
      return &e;
    }
  }
  return nullptr;
}

const ModelCatalogEntry*
catalog_by_path(const std::string& hf_path)
{
  for (const auto& e : model_catalog()) {
    if (e.hf_path == hf_path) {
      return &e;
    }
  }
  return nullptr;
}

const ModelCatalogEntry*
catalog_by_name(const std::string& name)
{
  if (name.empty()) {
    return nullptr;
  }
  for (const auto& e : model_catalog()) {
    if (e.name == name) {
      return &e;
    }
  }
  return nullptr;
}

std::string
catalog_category(const ModelCatalogEntry& e)
{
  if (!e.dataset_files.empty()) {
    return "dataset";
  }
  if (!e.parent_model_type.empty()) {
    return "supplement";
  }
  return "model";
}

void
catalog_default_io(const std::string& model_type,
                   std::vector<std::string>& in,
                   std::vector<std::string>& out)
{
  default_io_(model_type, in, out);
}

FlexData
catalog_entry_to_flex(const ModelCatalogEntry& e)
{
  FlexData doc = FlexData::make_object();
  auto o = doc.as_object();
  o.insert("family", FlexData::make_string(e.family));
  o.insert("version", FlexData::make_string(e.version));
  o.insert("param_class", FlexData::make_string(e.param_class));
  o.insert("variant", FlexData::make_string(e.variant));
  o.insert("hf_path", FlexData::make_string(e.hf_path));
  o.insert("model_type", FlexData::make_string(e.model_type));
  o.insert("category", FlexData::make_string(catalog_category(e)));
  if (!e.name.empty()) {
    o.insert("name", FlexData::make_string(e.name));
  }
  if (!e.parent_model_type.empty()) {
    o.insert("parent_model_type",
             FlexData::make_string(e.parent_model_type));
  }
  if (!e.parent_param_class.empty()) {
    o.insert("parent_param_class",
             FlexData::make_string(e.parent_param_class));
  }
  // Input / output modalities: explicit if recorded, else derived.
  std::vector<std::string> in = e.inputs, out = e.outputs;
  if (in.empty() && out.empty()) {
    default_io_(e.model_type, in, out);
  }
  FlexData ia = FlexData::make_array();
  {
    auto a = ia.as_array();
    for (const auto& s : in) {
      a.push_back(FlexData::make_string(s));
    }
  }
  FlexData oa = FlexData::make_array();
  {
    auto a = oa.as_array();
    for (const auto& s : out) {
      a.push_back(FlexData::make_string(s));
    }
  }
  o.insert("inputs", std::move(ia));
  o.insert("outputs", std::move(oa));
  return doc;
}

std::string
normalize_hf_path(const std::string& input)
{
  std::string s = input;
  // Trim surrounding whitespace.
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  if (s.empty()) {
    return {};
  }
  // Strip scheme + host so both full URLs and bare paths normalise.
  strip_prefix_(s, "https://");
  strip_prefix_(s, "http://");
  strip_prefix_(s, "www.");
  strip_prefix_(s, "huggingface.co/");
  // Drop query / fragment.
  s = s.substr(0, s.find_first_of("?#"));

  // Take the first two non-empty '/'-separated segments: owner/repo.
  std::string owner, repo;
  size_t i = 0;
  auto next_segment = [&](std::string& dst) {
    while (i < s.size() && s[i] == '/') { ++i; }
    size_t start = i;
    while (i < s.size() && s[i] != '/') { ++i; }
    dst = s.substr(start, i - start);
  };
  next_segment(owner);
  next_segment(repo);
  if (owner.empty() || repo.empty()) {
    return {};
  }
  return owner + "/" + repo;
}

std::vector<HfFile>
hf_tree_files(const FlexData& tree_json)
{
  std::vector<HfFile> out;
  if (!tree_json.is_array()) {
    return out;
  }
  auto arr = tree_json.as_array();
  for (std::size_t i = 0; i < arr.size(); ++i) {
    FlexData entry = arr.at(i);   // own a copy; views dangle off temporaries
    if (!entry.is_object()) {
      continue;
    }
    auto obj = entry.as_object();
    const std::string type = obj.contains("type")
        ? std::string(obj.at("type").as_string("")) : "";
    if (type != "file") {
      continue;
    }
    if (!obj.contains("path")) {
      continue;
    }
    std::string path(obj.at("path").as_string(""));
    if (path.empty()) {
      continue;
    }
    HfFile f;
    f.path = std::move(path);
    f.size = obj.contains("size") ? obj.at("size").as_uint(0) : 0;
    out.push_back(std::move(f));
  }
  return out;
}

}
