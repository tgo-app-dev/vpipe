#include "stages/model-catalog.h"

#include "generative-models/detect-profile.h"
#include "common/flex-data.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>

namespace vpipe {

namespace {

// The models that ship WITH vpipe. `model_catalog()` below returns this
// followed by whatever plugins contributed; see register_catalog_entries.
const std::vector<ModelCatalogEntry>&
builtin_catalog_()
{
  // ==================================================================
  // MODEL CATALOGUE -- single edit point. Append an entry to add a
  // model; the selection menu rebuilds itself from this table.
  // ==================================================================

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

  // The Ref2VA partition is the FL2VA list under the other prefix. The
  // two subtrees are MIRRORS -- same file names, same shard counts (13
  // DiT + 14 encoder), same ~133 GB -- because each is a COMPLETE
  // pipeline for its own task, encoder and both VAEs included.
  //
  // Derived rather than duplicated, so the two cannot drift: a 60-line
  // copy is one shard-count change away from a fetch that reports
  // success and leaves a directory missing a shard.
  //
  // It costs a user who wants BOTH partitions a second copy of the
  // encoder and the VAEs (~72 GB of the 133), which is the publisher's
  // layout and not something the catalogue can net out: unlike the
  // Comfy-Org repack, where both partitions name the SAME encoder file
  // and the fetcher skips what it already holds, these are distinct
  // paths. Pinning only the DiT would halve the download and produce a
  // directory that cannot encode a prompt.
  static const std::vector<std::string> kMiniMaxH3Ref2VAFiles = [] {
    constexpr std::size_t kPfx = sizeof("FL2VA/") - 1;
    std::vector<std::string> v;
    v.reserve(kMiniMaxH3FL2VAFiles.size());
    for (const std::string& f : kMiniMaxH3FL2VAFiles) {
      v.push_back("Ref2VA/" + f.substr(kPfx));
    }
    return v;
  }();

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
    // ---- Qwen 3.8 ------------------------------------------------------
    //
    // Qwen3.8-27B is the SAME ARCHITECTURE as Qwen3.6-27B below, retrained:
    // the two config.json files are identical field for field apart from
    // `transformers_version`, and the two checkpoints carry the same 1199
    // tensors under the same names for the same 55.56 GB. So it runs the
    // qwen3.5 family path unchanged -- hence `model_type = "qwen3.5"`, as
    // the 3.6 entries also use. The version here is a DISPLAY label
    // (from_catalog_ takes family/version from the entry, not from the
    // runtime tag), which is what keeps the drill-down honest without a
    // new runtime tag that every consumer would have to learn.
    //
    // ONE THING IS NOT THE SAME, and it is in the chat template rather
    // than the weights: 3.8 adds a `reasoning_effort` control
    // (xhigh | medium | low, DEFAULT xhigh) that prepends an instruction
    // paragraph to the system turn, and flips `preserve_thinking` to
    // default-true. See Qwen3ChatTemplate in chat-template.cc -- the
    // renderer is a hand-written ChatML scaffold, not a Jinja
    // interpreter, so it emits neither. A chat runs correctly; it just
    // runs without the reasoning-effort nudge the published template
    // sends by default.
    {.family = "Qwen", .version = "3.8", .param_class = "27B",
     .variant = "bf16 (Qwen)",
     .hf_path = "Qwen/Qwen3.8-27B",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    // The same 27B at MLX 4-bit (uniform group-affine, NOT OptiQ -- there
    // are no optiq/ companions in this repo, so no MTP draft head and no
    // separate vision tower file; the tower is in the three shards).
    {.family = "Qwen", .version = "3.8", .param_class = "27B",
     .variant = "MLX 4-bit (lmstudio-community)",
     .hf_path = "lmstudio-community/Qwen3.8-27B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    // GGUF. The repo ships ~23 quantization points plus two projectors,
    // so the entry pins ONE quant and the projector rather than pulling
    // all of them.
    //
    // The projector is the **F16** twin. It is not interchangeable with
    // the BF16 one at the loader: llama.cpp leaves `v.patch_embd.weight`
    // at F32 in the BF16 mmproj but narrows it to F16 in this one, and
    // the tower's patch-embed reader used to accept only F32 (it now
    // takes F32/F16/BF16 -- see patch_embed_gguf in
    // qwen3/metal-qwen-vision.cc). Either projector converts into the
    // run's own 2-byte compute dtype, so an F16 projector in a bf16 run
    // costs one register conversion at LOAD and then runs at 16-bit
    // width; nothing widens to f32.
    //
    // Two quantization points, one row each. Q4_K_M is all k-quant
    // (Q4_K 294 / Q5_K 48 / Q6_K 67) and passes through raw; Q8_0 is
    // repacked to affine 8-bit group 64, the only affine shape the metal
    // Qwen accepts (see GgufQwen35Converter). Both share the projector.
    //
    // Both land in the SAME directory (<base>/unsloth/Qwen3.8-27B-GGUF),
    // so a caller that fetches both and then points hf_dir at the
    // directory gets whichever sorts first -- Q4_K_M. Name the .gguf to
    // pick: hf_dir accepts a file.
    //
    // MEASURED, both against the same 27B quantized from the bf16
    // original to w8g64, layer-0 residual / final-layer residual:
    // Q8_0 4.3e-03 / 3.9e-02, Q4_K_M 1.1e-02 / 5.2e-02 at layer 32.
    // Q8_0 costs 29 GB against Q4_K_M's 17 GB, and its requant to
    // group 64 (Q8_0's own block is 32) puts it ~1.4x above the Q8_0
    // floor -- so it is the accuracy option, not the free one.
    {.family = "Qwen", .version = "3.8", .param_class = "27B",
     .variant = "Q4_K_M GGUF +mmproj (unsloth)",
     .hf_path = "unsloth/Qwen3.8-27B-GGUF", .model_type = "qwen3.5",
     .files = {"Qwen3.8-27B-Q4_K_M.gguf",  // main quant
               "mmproj-F16.gguf"},         // F16 multimodal projector
     .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.8", .param_class = "27B",
     .variant = "Q8_0 GGUF +mmproj (unsloth)",
     .hf_path = "unsloth/Qwen3.8-27B-GGUF", .model_type = "qwen3.5",
     .files = {"Qwen3.8-27B-Q8_0.gguf",    // main quant
               "mmproj-F16.gguf"},         // F16 multimodal projector
     .needs_tokenizer_json = false},
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
    // Krea2-HD VAE (wikeeyang): the Qwen-Image VAE fine-tuned alongside a
    // Krea-2-Turbo HD fine-tune of the DiT, and published as a
    // STANDALONE VAE -- one 507 MB safetensors carrying the full
    // autoencoder (decoder, encoder and both quant convs). It is a drop-in
    // for the stock `vae/` of any model on this latent space (the
    // publisher names Qwen-Image and Wan 2.1); the DiT beside it in the
    // repo is a separate model and is NOT pinned here.
    //
    // MEASURED against krea/Krea-2-Turbo's own VAE, tensor for tensor:
    // the DECODER is heavily retrained (median rel-L2 0.83) while the
    // ENCODER is barely touched (median 0.010, cosine 0.999999 on all 84
    // tensors). So it is a decoder fine-tune that ships its encoder along
    // -- worth selecting for vae-decode, ~neutral for vae-encode.
    //
    // TWO THINGS THIS ENTRY HAS TO CARRY. The weights are NATIVELY named
    // (decoder.upsamples.4.residual.2 rather than
    // decoder.up_blocks.1.resnets.0.conv1); the VAE loader maps them,
    // see generative-models/shared/wan-vae-names.h. And the repo ships no
    // config.json at all, so the per-channel `latents_mean`/`latents_std`
    // that un-whiten a latent come from a companion -- which is the right
    // source rather than a convenience: a drop-in replacement is a claim
    // about a LATENT SPACE, and guessing those numbers here would decode
    // plausible, wrongly-graded pixels.
    //
    // THE COMPANION IS THE FALLBACK, NOT THE PRIMARY ROUTE. A box that
    // already holds the model this VAE serves already holds the file:
    // resolve_vae_config_path borrows that model's vae/config.json in
    // place, and nothing is downloaded or copied. The companion is for
    // the other case -- somebody fetching the VAE on its own.
    //
    // AND IT COMES FROM Qwen/Qwen-Image, NOT FROM KREA-2, BECAUSE OF THE
    // GATE. This VAE is ungated, so nobody fetching it has
    // any reason to hold a Krea-2 licence token -- and krea/Krea-2-Turbo
    // answers an anonymous request with 401. Pointing an ungated 507 MB
    // download at a gated repo makes it fail for everyone who has not
    // separately accepted a licence for a model they are not fetching,
    // and it fails LATE: the weights land, the companion warns, and the
    // decode stage goes inert with a message about latents_mean.
    //
    // Qwen-Image is also the better authority on its own terms. The
    // publisher's claim is that this VAE serves "any model that used
    // Qwen-Image / Wan 2.1 VAE", so the latent space is Qwen-Image's;
    // Krea-2's own vae/config.json records `_name_or_path:
    // "Qwen/Qwen-Image"` and is a copy of it. VERIFIED byte for byte on
    // every field this tree reads -- _class_name, z_dim, base_dim,
    // num_res_blocks and all 16 entries of latents_mean / latents_std
    // are identical; the two differ only in _diffusers_version, an
    // unread `input_channels`, and that _name_or_path.
    {.family = "Krea", .version = "2", .param_class = "VAE",
     .variant = "HD fine-tuned VAE (wikeeyang)",
     .hf_path = "wikeeyang/Krea2-Turbo-HD-V1",
     .model_type = "krea2-vae",
     .parent_model_type = "krea2",   // any model on the Qwen-Image latent
     .files = {"Krea2-HD-vae.safetensors"},
     .companion_files = {{.repo = "Qwen/Qwen-Image",
                          .file = "vae/config.json",
                          .dest = "config.json"}},
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
    // Qwen-Image-2512 (Qwen): the TEXT-TO-IMAGE sibling of Edit-2511 and
    // architecturally the same model. VERIFIED against it tensor by
    // tensor over HTTP Range, not inferred from the config: the DiT is
    // 1933 tensors with identical names, dtypes and shapes (40.86 GB) and
    // the text encoder 729 (16.58 GB); text_encoder/, vae/ and scheduler/
    // configs are identical files. The transformer config differs in ONE
    // key, and it differs correctly -- Edit-2511 sets `zero_cond_t` true,
    // 2512 omits it, and that flag modulates REFERENCE tokens at timestep
    // 0, which a model with no references does not have. So the DiT, the
    // VAE and the encoder load it with nothing new.
    //
    // WHAT IS NOT SHARED IS THE RECIPE, which is why this carries its own
    // model_type instead of joining "qwen-image-edit". The two are
    // separate diffusers pipelines -- QwenImagePipeline here against
    // QwenImageEditPlusPipeline there -- and they condition through
    // DIFFERENT system prompts: the plain "Describe the image by
    // detailing ..." template (the one kPrefix / kDropPrefix already hold
    // for Krea-2 and Mage-Flow) against the edit template that opens
    // "Describe the key features of the input image". Typing this as the
    // edit model would load and run and quietly condition it on the wrong
    // prompt, which is the failure nothing downstream can see.
    //
    // The conditioner reads that recipe from the registry's model_type
    // (this field) or, for an unregistered directory, from the pipeline
    // model_index.json declares -- never from whether a reference image
    // is wired, which would have flipped the EDIT model's own text-only
    // conditioning and the golden that pins it.
    //
    // `needs_tokenizer_json` is TRUE because the repo ships no
    // consolidated tokenizer.json, only tokenizer/{vocab.json,
    // merges.txt, tokenizer_config.json}. The synthesis step now runs in
    // whichever directory holds vocab.json, so it writes
    // tokenizer/tokenizer.json here and the repo root for the ASR repos
    // it was built for.
    //
    // No processor/ subdir here, unlike Edit-2511: a model that takes no
    // reference image needs no image preprocessor, and the chat template
    // sits in tokenizer/ instead. ~58 GB (20B DiT + 7B VL encoder, bf16).
    {.family = "Qwen-Image", .version = "2512", .param_class = "20B",
     .variant = "bf16 (Qwen)",
     .hf_path = "Qwen/Qwen-Image-2512",
     .model_type = "qwen-image",
     .files = {"model_index.json",
               "transformer/config.json",
               "transformer/diffusion_pytorch_model.safetensors.index.json",
               "transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
               "transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
               "transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
               "transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
               "transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
               "transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
               "transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
               "transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
               "transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
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
               "tokenizer/chat_template.jinja",
               "scheduler/scheduler_config.json"},
     .needs_tokenizer_json = true},
    // Qwen-Image-2.1 (Qwen): one checkpoint that does BOTH text-to-image
    // and reference editing, and the first model here whose VAE is
    // natively RGBA. Despite the family name it shares almost nothing
    // with Edit-2511 / 2512 above -- read the three sub-models before
    // assuming any of that code applies.
    //   transformer/  = QwenImage21Transformer2DModel, a 7B SINGLE-stream
    //     DiT: 32 blocks, 32 heads x head_dim 128 = 4096 hidden, in/out
    //     channels 64, patch_size 1 (latents are consumed UNPATCHED),
    //     mlp_ratio 3 -> SwiGLU inner 12288, 3D-RoPE axes [16,56,56],
    //     causal_condition true. 297 tensors, every one a `.weight`:
    //     there is no bias anywhere in the model. Modulation is NOT per
    //     block -- a single shared 4096->16384 projection produces four
    //     4096-wide vectors (scale/gate for attention, scale/gate for
    //     the feed-forward) that all 32 blocks slice, and the gates go
    //     through tanh. Two mechanisms have no precedent in this tree:
    //     attention is BLOCK-CAUSAL (the joint text/image sequence is
    //     causal while each image block stays internally bidirectional),
    //     and the text + condition-image prefix is KV-CACHED across
    //     denoise steps -- which is only sound because causal_condition
    //     modulates that prefix from t=0, making it step-independent.
    //   text_encoder/ = Qwen3-VL 8B (Qwen3VLForConditionalGeneration, 36
    //     layers, hidden 4096, 32 q-heads GQA kv=8, head_dim 128, ffn
    //     12288, rope theta 5e6, UNTIED embeddings) wrapped as
    //     `model.language_model.` / `model.visual.`, with a 27-layer
    //     tower and deepstack taps [8,16,24]. That is the same encoder
    //     Boogu-Image drives, in the same layout. Reference images are
    //     fed to the tower and the DiT substitutes their VAE latents
    //     into the vision slots the encoder reserved, so editing rides
    //     the PROMPT sequence rather than appending tokens to the DiT
    //     stream the way Edit-2511 does. Up to 10 references.
    //   vae/          = AutoencoderKLQwenImage21: z_dim 64, 16x spatial
    //     (dim_mult [1,2,4,8,8], five levels), in_channels AND
    //     out_channels 4 -- RGBA -- and asymmetric, base_dim 96 for the
    //     encoder against decoder_base_dim 144. is_residual adds a
    //     parameter-free shortcut (group-average down, channel-repeat
    //     up) around each block; it carries no weights of its own.
    //     Stored F32.
    // Transparency is prompt-driven, not a mode: the card's recommended
    // wording ("This is an RGBA image with transparency ... the
    // background is transparent") is what produces a populated alpha
    // channel, and the VAE emits four channels either way.
    // Native resolution is 2048x2048 and the size grid is 32, not 16 --
    // the pipeline rounds the LATENT grid even, so a 16-multiple that is
    // not a 32-multiple is silently re-rounded.
    // `files` PINS the diffusers subfolders and processor/ (which holds
    // the consolidated tokenizer.json, hence needs_tokenizer_json false
    // and no tokenizer/ subdir at all), skipping only README/LICENSE.
    // ~33 GB (7B DiT + 8B VL encoder, bf16; 1.35 GB VAE).
    {.family = "Qwen-Image", .version = "2.1", .param_class = "7B",
     .variant = "bf16 (Qwen)",
     .hf_path = "Qwen/Qwen-Image-2.1",
     .model_type = "qwen-image-21",
     .files = {"model_index.json",
               "transformer/config.json",
               "transformer/diffusion_pytorch_model.safetensors.index.json",
               "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
               "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
               "text_encoder/config.json",
               "text_encoder/model.safetensors.index.json",
               "text_encoder/model-00001-of-00004.safetensors",
               "text_encoder/model-00002-of-00004.safetensors",
               "text_encoder/model-00003-of-00004.safetensors",
               "text_encoder/model-00004-of-00004.safetensors",
               "vae/config.json",
               "vae/diffusion_pytorch_model.safetensors",
               "processor/tokenizer.json",
               "processor/tokenizer_config.json",
               "processor/vocab.json",
               "processor/merges.txt",
               "processor/added_tokens.json",
               "processor/special_tokens_map.json",
               "processor/chat_template.jinja",
               "processor/preprocessor_config.json",
               "processor/video_preprocessor_config.json",
               "scheduler/scheduler_config.json"},
     .needs_tokenizer_json = false},
    // Z-Image (Tongyi-MAI): a 6B single-stream DiT in the Lumina/NextDiT
    // lineage, and the cheapest first-class image family in this tree --
    // two of its three sub-models are already supported code.
    //   transformer/  = ZImageTransformer2DModel. dim 3840, 30 heads x
    //     head_dim 128 (MHA, n_kv_heads == n_heads), 30 main layers PLUS
    //     two 2-layer REFINER stacks that run before the join, so 34
    //     blocks of identical shape. SwiGLU inner 10240 (dim/3*8), 3D
    //     RoPE axes [32,48,48] at theta 256 (not 10000), norm_eps 1e-5.
    //     Every norm is an RMSNorm and the block is SANDWICH-normed --
    //     attention_norm1 before the attention and attention_norm2 on
    //     its OUTPUT, inside the residual. Modulation is four chunks
    //     (scale_msa, gate_msa, scale_mlp, gate_mlp) off a per-block
    //     Linear(256 -> 4*dim) with NO activation in front of it, while
    //     the final layer's adaLN has a SiLU in front -- a difference
    //     visible only as the tensor index (.0 against .1).
    //   text_encoder/ = Qwen3-4B DENSE (Qwen3Model: 36 layers, hidden
    //     2560, 32 q-heads GQA kv=8, head_dim 128, ffn 9728, rope theta
    //     1e6, TIED embeddings) under a bare `model.` wrapper -- the
    //     same encoder FLUX.2-klein drives, and text-ONLY: there is no
    //     vision tower in this family. Conditioning taps HF's
    //     hidden_states[-2], i.e. the un-normed output of layer 34.
    //   vae/          = the plain diffusers AutoencoderKL that FLUX.1
    //     and Boogu-Image use, unchanged: 16 latent channels, 8x
    //     spatial, block_out [128,256,512,512], no quant/post-quant 1x1
    //     convs, scalar whitening (shift 0.1159, scale 0.3611).
    // The prompt is rendered through Qwen3's own chat template with a
    // generation turn and NO system message, and truncated at 512
    // tokens; there is no family system prompt to get wrong.
    // TWO CHECKPOINTS, ONE model_type, because their transformer
    // configs are byte-identical apart from a default: what differs is
    // the SCHEDULER shift (3.0 Turbo / 6.0 base) and whether CFG is
    // wanted, and both of those are read rather than inferred. Turbo is
    // distilled to ~8 steps at guidance 0; the base model wants 28-50
    // steps at guidance 3-5 and answers a negative prompt.
    // NOTE the Turbo transformer ships F32 (24.6 GB on disk against the
    // base's 12.3 GB bf16) -- same 6B of parameters, twice the bytes to
    // stream.
    {.family = "Z-Image", .version = "1", .param_class = "6B",
     .variant = "Turbo, 8-step distilled (Tongyi-MAI)",
     .hf_path = "Tongyi-MAI/Z-Image-Turbo",
     .model_type = "z-image",
     .files = {"model_index.json",
               "transformer/config.json",
               "transformer/diffusion_pytorch_model.safetensors.index.json",
               "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
               "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
               "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
               "text_encoder/config.json",
               "text_encoder/model.safetensors.index.json",
               "text_encoder/model-00001-of-00003.safetensors",
               "text_encoder/model-00002-of-00003.safetensors",
               "text_encoder/model-00003-of-00003.safetensors",
               "vae/config.json",
               "vae/diffusion_pytorch_model.safetensors",
               "tokenizer/tokenizer.json",
               "tokenizer/tokenizer_config.json",
               "tokenizer/vocab.json",
               "tokenizer/merges.txt",
               "scheduler/scheduler_config.json"},
     .needs_tokenizer_json = true},
    // Z-Image (base) -- the undistilled foundation model behind Turbo.
    // Same architecture, same text encoder, same VAE; scheduler shift
    // 6.0 rather than 3.0, and it is the variant that USES classifier-
    // free guidance and negative prompts. bf16 on disk.
    {.family = "Z-Image", .version = "1", .param_class = "6B",
     .variant = "base, CFG + negative prompt (Tongyi-MAI)",
     .hf_path = "Tongyi-MAI/Z-Image",
     .model_type = "z-image",
     .files = {"model_index.json",
               "transformer/config.json",
               "transformer/diffusion_pytorch_model.safetensors.index.json",
               "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
               "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
               "text_encoder/config.json",
               "text_encoder/model.safetensors.index.json",
               "text_encoder/model-00001-of-00003.safetensors",
               "text_encoder/model-00002-of-00003.safetensors",
               "text_encoder/model-00003-of-00003.safetensors",
               "vae/config.json",
               "vae/diffusion_pytorch_model.safetensors",
               "tokenizer/tokenizer.json",
               "tokenizer/tokenizer_config.json",
               "tokenizer/vocab.json",
               "tokenizer/merges.txt",
               "scheduler/scheduler_config.json"},
     .needs_tokenizer_json = true},
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
    // ---- Mage-Flow: NOT HERE, and that is the point ------------------
    // Its six records moved into the vpipe-mage-flow plugin, which
    // registers them with register_catalog_entries. A model whose CODE
    // is out of tree should have its catalogue row out of tree too, or
    // the browser offers a checkpoint nothing in the process can load.
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
    // ---- VOSR (image -> image, RESTORATION rather than generation) -----
    // VOSR 2.0 (PolyU / OPPO, CVPR 2026): a one-step, VISION-ONLY image
    // super-resolution model. It is the first entry here that conditions
    // on no text at all -- there is no prompt, no tokenizer and no text
    // encoder anywhere in it. What the DiT cross-attends to is a DINOv2
    // feature grid over the low-quality picture itself, which is why the
    // `diffusion-conditioner` on this family loads a vision tower and
    // nothing else. Sub-models:
    //   VOSR2/checkpoints/ = LightningDiT, 1.4B: 36 blocks at 1536 wide,
    //     24 heads x head_dim 64, patch 2, RMSNorm + qk-norm + SwiGLU,
    //     2D EVA-style RoPE whose positions are RESCALED into the 32x32
    //     grid it trained on rather than extrapolated. The patch
    //     projection reads 32 channels: the low-quality latent
    //     concatenated with the noise. Distilled to ONE flow step.
    //   Qwen-Image-vae-2d/ = the Qwen-Image VAE with its temporal axis
    //     removed (`AutoencoderKLQwenImage2D`) -- same 16 latent
    //     channels, same 8x spatial, same whitening statistics, one
    //     axis fewer on every conv. The stock 3D checkpoint decodes a
    //     single frame to the same numbers; this one just does not pay
    //     for the axis. Served by the same code either way.
    // The DINOv2 tower is NOT in this repo in a form anything here can
    // read (the release ships torch.hub's pickle); fetch the entry below.
    {.family = "VOSR", .version = "2.0", .param_class = "1.4B",
     .variant = "one-step fp32 (PolyU/OPPO)",
     .hf_path = "CSWRY/VOSR",
     .model_type = "vosr",
     .inputs = {"image"}, .outputs = {"image"},
     .files = {"VOSR2/args.json",
               "VOSR2/checkpoints/ema_model.safetensors",
               "Qwen-Image-vae-2d/config.json",
               "Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors"},
     .needs_tokenizer_json = false,
     .name = "VOSR-2.0"},
    // DINOv2 ViT-L/14 (Meta), the conditioner VOSR cross-attends to. A
    // SUPPLEMENT, not a model: it generates nothing on its own, and it
    // is listed apart from VOSR because the restorer's own repo ships it
    // only as a torch.hub pickle. These are the same weights in
    // safetensors, published by Meta.
    {.family = "DINOv2", .version = "1", .param_class = "L/14",
     .variant = "ViT-L/14 (Meta)",
     .hf_path = "facebook/dinov2-large",
     .model_type = "dinov2",
     .inputs = {"image"}, .outputs = {},
     .parent_model_type = "vosr",
     .files = {"config.json", "model.safetensors",
               "preprocessor_config.json"},
     .needs_tokenizer_json = false},
    // ---- FlashVSR (video -> video, streaming RESTORATION) -------------
    // FlashVSR-v1.1 (OpenImagingLab, arXiv 2510.12747): one-step streaming
    // 4x video super-resolution. Like VOSR it conditions on no text -- its
    // context is a constant the checkpoint carries -- and unlike VOSR it
    // runs on VIDEO, a chunk at a time against a key/value window.
    //   diffusion_pytorch_model_streaming_dmd.safetensors = the denoiser,
    //     Wan 2.1-T2V-1.3B tensor for tensor (30 blocks at 1536, 12 x 128
    //     heads), with block-sparse attention over (2,8,8) windows.
    //   LQ_proj_in.ckpt = the learned causal source projection, which
    //     reads the upscaled clip in PIXELS (the flashvsr-src-encoder
    //     stage runs it).
    //   Wan2.1_VAE.pth  = the Wan 2.1 VAE, natively named, no config.
    // Three of those are torch.save() files, READ IN PLACE: they map as
    // shards (shared/torch-zip.h), so nothing is converted.
    //
    // THE CONTEXT IS NOT ON THE MODEL PAGE. The reference keeps it in its
    // GitHub repo (examples/WanVSR/prompt_tensor/posi_prompt.pth), and a
    // companion can only name a HuggingFace repo -- so it comes from
    // deAPI-ai's copy of v1.1, which is BYTE-IDENTICAL to GitHub's
    // (sha256 4601107a11e4e11a...) as of 2026-09-12. 4 MB. TCDecoder.ckpt
    // (the reference's tiny decoder) is not fetched: nothing here runs it.
    {.family = "FlashVSR", .version = "1.1", .param_class = "1.3B",
     .variant = "one-step streaming 4x (OpenImagingLab)",
     .hf_path = "JunhaoZhuang/FlashVSR-v1.1",
     .model_type = "flashvsr",
     .inputs = {"video"}, .outputs = {"video"},
     .files = {"config.json", "model_index.json",
               "diffusion_pytorch_model_streaming_dmd.safetensors",
               "LQ_proj_in.ckpt", "Wan2.1_VAE.pth"},
     .companion_files = {{.repo = "deAPI-ai/flashvsr-v1-1",
                          .file = "posi_prompt.pth",
                          .dest = "posi_prompt.pth"}},
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
     .needs_tokenizer_json = false,
     // NAMED, because the Ref2VA entry below is published from this same
     // repo and lands in this same directory. Without a name both would
     // register under the repo path and the second fetch would overwrite
     // the first's record -- one key, one model_type, and no way for a
     // consumer to say which of the two partitions it meant. The
     // directory is still shared (local_dir is the repo path either
     // way), which is the point: 133 GB downloaded once per partition,
     // two records over it.
     .name = "MiniMaxAI/MiniMax-H3-FL2VA"},
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
    // this build does not read and in `pruned` DiTs (a different model,
    // not a different packing of this one). Same reason the GGUF
    // entries pin one quant instead of taking the whole repo. The
    // Ref2VA partition is its own entry below.
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
     .needs_tokenizer_json = false,
     .name = "Comfy-Org/MiniMax-H3-FL2VA"},
    // The Ref2VA partition: the OTHER half of MiniMax-H3.
    //
    // Same architecture as FL2VA down to the byte -- 535 tensors, same
    // names, same shapes, same embedded config -- and different WEIGHTS,
    // trained for a different task: text plus up to 9 reference images,
    // 3 video clips and 3 audio clips (12 files in total, and audio
    // never on its own), each packed as its own block ahead of the
    // generated rows.
    //
    // MiniMaxAI's release first, the repack after, mirroring how the
    // two FL2VA entries are ordered: the released weights are the
    // reference the repack is checked against, and a partition offered
    // in only one publisher's spelling reads as a partition that only
    // that publisher has.
    //
    // Its DiT is the per-head qkv grouping, like every MiniMaxAI file
    // and unlike the repack -- resolved from `Ref2VA/model_index.json`'s
    // `_minimax_h3.partition`, which is the only thing in the tree that
    // distinguishes the two: the transformer configs are identical.
    {.family = "MiniMax", .version = "H3-Ref2VA", .param_class = "33B",
     .variant = "bf16 omni video+audio (MiniMaxAI, diffusers)",
     .hf_path = "MiniMaxAI/MiniMax-H3",
     .model_type = "minimax-h3-ref2va",
     .inputs = {"text", "image", "video", "audio"},
     .outputs = {"video", "audio"},
     .files = kMiniMaxH3Ref2VAFiles,
     .needs_tokenizer_json = false,
     // The other half of the shared-repo pair -- see the FL2VA entry
     // above for why both are named. One directory, two records.
     .name = "MiniMaxAI/MiniMax-H3-Ref2VA"},
    // The same partition from Comfy-Org's repack, the second opinion on
    // the entry above for the same reason the FL2VA pair are two
    // entries: the two publishers disagree about the DiT's fused qkv
    // grouping, and having both on disk is what makes that a diff.
    //
    // It shares this repo's encoder and both VAEs with the FL2VA entry,
    // so a checkout that already has FL2VA pays only for the 66 GB DiT;
    // the fetcher skips files it already holds. Only the DiT is pinned
    // here for that reason -- the shared components are not repeated.
    {.family = "MiniMax", .version = "H3-Ref2VA", .param_class = "33B",
     .variant = "bf16 single-file (Comfy-Org)",
     .hf_path = "Comfy-Org/MiniMax-H3",
     .model_type = "minimax-h3-ref2va",
     .inputs = {"text", "image", "video", "audio"},
     .outputs = {"video", "audio"},
     .files = {"diffusion_models/minimax_h3_ref2va_bf16.safetensors",
               "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
               "vae/minimax_h3_video_vae_fp16.safetensors",
               "vae/minimax_h3_audio_vae_fp32.safetensors"},
     .companion_files =
         {{.repo = "MiniMaxAI/MiniMax-H3",
           .file = "Ref2VA/tokenizer/tokenizer.json",
           .dest = "tokenizer/tokenizer.json"},
          {.repo = "MiniMaxAI/MiniMax-H3",
           .file = "Ref2VA/tokenizer/tokenizer_config.json",
           .dest = "tokenizer/tokenizer_config.json"}},
     .weight_format = "comfyui",
     .needs_tokenizer_json = false,
     .name = "Comfy-Org/MiniMax-H3-Ref2VA"},
    // MiniMax-H3 Turbo: a few-step distillation LoRA for the FL2VA
    // partition. 4 steps instead of ~20 for joint video + synchronized
    // audio, and it keeps improving to about 8; past that it
    // over-sharpens, so 4-8 is the useful range at strength 1.0.
    //
    // Fuse it with the lora-fuse stage (base_model = the FL2VA DiT
    // FILE -- a Comfy-Org repack is one file per component and the
    // repo's diffusion_models/ holds BOTH partitions, so naming the
    // directory would merge two 66 GB models under one set of names),
    // then quantize the result and point generate-video at it.
    //
    // No name remap is needed and that is worth recording: this adapter
    // is keyed on the model's OWN module names (blocks.N.attn.qkv_proj,
    // mlp.fc1/fc2, adaln_proj.linear, token_refiner.blocks.N.*), so all
    // 259 modules resolve through lora-fuse's first rule. It carries no
    // `alpha` siblings either -- upstream states alpha = rank, i.e.
    // W + B@A with no rescaling -- which is exactly what a missing alpha
    // already means here.
    //
    // It is trained against Comfy-Org's repack, so it assumes the FLAT
    // qkv grouping. On MiniMaxAI's released weights the fused
    // `attn.qkv_proj` delta lands on the wrong channels -- one head's q
    // onto another head's k -- in every block, silently. The other four
    // adapted projections (out_proj, mlp.fc1/fc2, adaln_proj.linear)
    // carry no grouping, so ~207 of its 259 modules apply correctly
    // either way, which is why the mismatch reads as "Turbo, but not
    // quite" rather than as a broken model.
    //
    // The EVIDENCE is the repo, not the file: its README frontmatter is
    // `base_model: Comfy-Org/MiniMax-H3` with a `comfyui` tag, where
    // the safetensors' own `__metadata__.base_model` says only
    // "MiniMax-H3" -- the family, not the publisher. Reading the second
    // as the first is a mistake this note has already made once.
    //
    // lightx2v's ComfyUI copies are flat too, and there it was settled
    // WITHOUT trusting metadata: their B is exactly block-diagonal in
    // the [all q | all k | all v] sense. That is a property of a
    // CONVERTED fusion only; this adapter is trained directly on the
    // fused projection (rank 64, dense B), so nothing in its bytes
    // states a layout and only the repo does.
    // VDN-H3 (OpenVDN): NOT a LoRA, and the difference is the whole
    // point of listing it apart from the three above. A LoRA adds a
    // low-rank correction to weights the DiT already has; this adds a
    // SECOND PATHWAY -- a bidirectional delta-rule recurrence over
    // frames -- and narrows the softmax to a window of whole frames so
    // that the two PARTITION the keys. Attention is a different
    // function afterwards, which is why the branch ships its own 4.28 GB
    // checkpoint and why running the DiT without it, or it without the
    // window, is wrong rather than merely unadapted.
    //
    // It brings no projections of its own: the branch reads the
    // backbone's raw pre-QK-norm q/k/v, which is why one adapter on
    // attn.orig.to_{q,k,v} feeds both halves and why this attaches to
    // any MiniMax-H3 DiT rather than replacing one.
    //
    // TWO STAGES, and the choice between them is real. `stage-b` is the
    // 50-step model; `stage-dmd` is an 8-step distillation and ships an
    // extra `turbo` adapter -- the only adapter in this tree that
    // touches the ADALN projections, at rank 16 where the rest is 64.
    // Each pins its own subtree of one repo, so they need distinct
    // registration keys for the reason the two Turbo entries above do.
    //
    // Wired through the `linear_branch` key on the
    // `minimax-h3-model-config` source, not through a LoRA slot: it is
    // a checkpoint beside the DiT, not a correction to it.
    {.family = "MiniMax", .version = "H3-FL2VA", .param_class = "VDN",
     .variant = "VDN hybrid attention, stage-B 50-step (OpenVDN)",
     .hf_path = "OpenVDN/vdn-minimax-h3",
     .model_type = "minimax-h3-vdn",
     .parent_model_type = "minimax-h3-fl2va",
     .files = {"stage-b-step-2000/linear_branch/config.json",
               "stage-b-step-2000/linear_branch/model.safetensors",
               "stage-b-step-2000/adapters/default/adapter_config.json",
               "stage-b-step-2000/adapters/default/adapter_model.safetensors",
               "stage-b-step-2000/model_spec.json",
               "stage-b-step-2000/metadata.json"},
     .needs_tokenizer_json = false,
     .name = "OpenVDN/vdn-minimax-h3-stage-b"},
    {.family = "MiniMax", .version = "H3-FL2VA", .param_class = "VDN",
     .variant = "VDN hybrid attention, stage-DMD 8-step (OpenVDN)",
     .hf_path = "OpenVDN/vdn-minimax-h3",
     .model_type = "minimax-h3-vdn",
     .parent_model_type = "minimax-h3-fl2va",
     .files = {"stage-dmd-step-250/linear_branch/config.json",
               "stage-dmd-step-250/linear_branch/model.safetensors",
               "stage-dmd-step-250/adapters/default/adapter_config.json",
               "stage-dmd-step-250/adapters/default/"
               "adapter_model.safetensors",
               "stage-dmd-step-250/adapters/turbo/adapter_config.json",
               "stage-dmd-step-250/adapters/turbo/adapter_model.safetensors",
               "stage-dmd-step-250/model_spec.json",
               "stage-dmd-step-250/metadata.json"},
     .needs_tokenizer_json = false,
     .name = "OpenVDN/vdn-minimax-h3-stage-dmd"},
    {.family = "MiniMax", .version = "H3-FL2VA", .param_class = "LoRA",
     .variant = "Turbo few-step v4-600 EMA (larryvrh)",
     .hf_path = "larryvrh/MiniMax-H3-Turbo-Lora",
     .model_type = "minimax-h3-lora",
     .parent_model_type = "minimax-h3-fl2va",
     .files = {"minimax_h3_turbo_v4_step600_ema.safetensors"},
     .needs_tokenizer_json = false,
     // Both Turbo entries pin ONE file out of a repo holding eleven, so
     // they need distinct registration keys the way the vpipe-supplement
     // archives do -- without them the two would share the hf_path key
     // and a graph could not say which checkpoint it meant. Kept in the
     // `owner/name` shape every other key here has, so a reference reads
     // as a model wherever it appears and the browse flow and a scripted
     // fetch agree on one spelling. (`name` doubles as an extract subdir,
     // but only for `extract_archive` entries -- these are plain files.)
     .name = "larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema"},
    // The v1 line's last checkpoint, kept as its own entry because the
    // choice between them is real rather than a version bump: v4 is
    // better everywhere EXCEPT 4 steps with large, fast motion, where it
    // can trail/smear and this one does not. At 6-8 steps take v4.
    {.family = "MiniMax", .version = "H3-FL2VA", .param_class = "LoRA",
     .variant = "Turbo few-step v1-850 EMA (larryvrh)",
     .hf_path = "larryvrh/MiniMax-H3-Turbo-Lora",
     .model_type = "minimax-h3-lora",
     .parent_model_type = "minimax-h3-fl2va",
     .files = {"minimax_h3_turbo_4step_ema_ckpt850.safetensors"},
     .needs_tokenizer_json = false,
     .name = "larryvrh/MiniMax-H3-Turbo-Lora-v1-850-ema"},
    // lightx2v's Turbo line, the OTHER few-step distillation of this
    // model -- and the only one that covers BOTH partitions.
    //
    // The repo publishes each adapter TWICE and both spellings are
    // catalogued, because the difference is not cosmetic and the two
    // are not equally good.
    //
    // The diffusers copy is the original peft export: separate
    // to_q/to_k/to_v, an ff.net.0.proj in value-first order, factors
    // named `.lora_A.default.weight` and one alpha in `__metadata__`.
    // The ComfyUI copy is that file CONVERTED -- q/k/v stacked
    // block-diagonally into this model's fused qkv_proj, fc1's halves
    // swapped to gate-first, a per-module alpha written out. Both load
    // (see MetalMiniMaxH3Transformer::bind_lora_, whose conversion is
    // verified tensor-for-tensor against upstream's).
    //
    // PREFER THE DIFFUSERS COPY. A published fusion is built for ONE
    // qkv column grouping and is silently wrong on the other; fusing
    // from the split file happens against the DiT actually loaded, so
    // it is right on both publishers' weights. It is also the smaller
    // DOWNLOAD, 1.38 GB against 1.96 -- the difference being the two
    // thirds of a block-diagonal B that is zero. Not the smaller
    // model: the fusion is rebuilt at load either way.
    //
    // The ComfyUI copies key on `diffusion_model.<module>`, which both
    // the fuse and the runtime path resolve, and carry per-module alpha
    // (alpha == rank here, so no rescaling). Their fused qkv adapter is
    // rank 384 -- three rank-128 adapters stacked -- and its B is
    // block-diagonal in the [all q | all k | all v] sense, i.e. it
    // targets Comfy-Org's FLAT grouping and NOT the per-head release
    // that this repo's `base_model` tag names. On the released weights
    // it would add one head's q delta onto another head's k in all 50
    // blocks, silently.
    //
    // SCHEDULE: the 4-step was distilled at video shift 6 (and 768p),
    // where this model's default and every other adapter here is 12.
    // That is not a preference -- it is the grid the distillation was
    // fit to, so it belongs in the same `minimax-h3-model-config` beat
    // as the adapter itself.
    {.family = "MiniMax", .version = "H3-FL2VA", .param_class = "LoRA",
     .variant = "Turbo 4-step v1.0 768p, shift 6 (lightx2v)",
     .hf_path = "lightx2v/Minimax-h3-Turbo",
     .model_type = "minimax-h3-lora",
     .parent_model_type = "minimax-h3-fl2va",
     .files = {"minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16"
               ".safetensors"},
     .needs_tokenizer_json = false,
     .name = "lightx2v/Minimax-h3-Turbo-4step-768p"},
    // The 8-step, distilled at 544p on the checkpoint's own 12/3 shifts;
    // upstream runs it at 8 steps or 4.
    {.family = "MiniMax", .version = "H3-FL2VA", .param_class = "LoRA",
     .variant = "Turbo 8-step v1.0 544p (lightx2v)",
     .hf_path = "lightx2v/Minimax-h3-Turbo",
     .model_type = "minimax-h3-lora",
     .parent_model_type = "minimax-h3-fl2va",
     .files = {"minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"},
     .needs_tokenizer_json = false,
     .name = "lightx2v/Minimax-h3-Turbo-8step"},
    // The 8-step at 768p: the same distillation as the entry above run
    // at 1344x768 on the 4-step 768p's shifts (6 / 3), and upstream's
    // own default -- LightX2V Studio serves this one. Recommended at 8
    // steps, where the 544p 8-step is also usable at 4.
    {.family = "MiniMax", .version = "H3-FL2VA", .param_class = "LoRA",
     .variant = "Turbo 8-step v1.0 768p, shift 6 (lightx2v)",
     .hf_path = "lightx2v/Minimax-h3-Turbo",
     .model_type = "minimax-h3-lora",
     .parent_model_type = "minimax-h3-fl2va",
     .files = {"minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16"
               ".safetensors"},
     .needs_tokenizer_json = false,
     .name = "lightx2v/Minimax-h3-Turbo-8step-768p"},
    // The FIRST adapter here for the OTHER partition. Ref2VA's DiT is
    // the same architecture as FL2VA down to the tensor names, so the
    // adapter has the same 50+2 blocks and the same shapes -- and it is
    // trained for the other task, so the parent link is what keeps it
    // out of an FL2VA graph. Distilled at 544p on the checkpoint's own
    // 12 / 3 shifts, 4 steps.
    //
    // Its metadata names the FL2VA repack as `base_model`, which is a
    // stale string in upstream's conversion script rather than a claim
    // about the weights: the file's own module list is the Ref2VA
    // partition's, and the two partitions are byte-identical in shape.
    // What the string DOES get right is the publisher -- see the qkv
    // grouping note above, which applies to this file too.
    {.family = "MiniMax", .version = "H3-Ref2VA", .param_class = "LoRA",
     .variant = "Turbo 4-step v0.1 (lightx2v)",
     .hf_path = "lightx2v/Minimax-h3-Turbo",
     .model_type = "minimax-h3-lora",
     .parent_model_type = "minimax-h3-ref2va",
     .files = {"minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16"
               ".safetensors"},
     .needs_tokenizer_json = false,
     .name = "lightx2v/Minimax-h3-Turbo-ref2va-4step"},
    // The DIFFUSERS copies of the two entries above -- the same
    // adapters, split rather than fused, and the ones to reach for.
    // See the qkv-grouping paragraph at the head of this section.
    {.family = "MiniMax", .version = "H3-FL2VA", .param_class = "LoRA",
     .variant = "Turbo 8-step v1.0 768p, shift 6, split qkv (lightx2v)",
     .hf_path = "lightx2v/Minimax-h3-Turbo",
     .model_type = "minimax-h3-lora",
     .parent_model_type = "minimax-h3-fl2va",
     .files = {"minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors"},
     .needs_tokenizer_json = false,
     .name = "lightx2v/Minimax-h3-Turbo-8step-768p-split"},
    {.family = "MiniMax", .version = "H3-Ref2VA", .param_class = "LoRA",
     .variant = "Turbo 4-step v0.1, split qkv (lightx2v)",
     .hf_path = "lightx2v/Minimax-h3-Turbo",
     .model_type = "minimax-h3-lora",
     .parent_model_type = "minimax-h3-ref2va",
     .files = {"minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"},
     .needs_tokenizer_json = false,
     .name = "lightx2v/Minimax-h3-Turbo-ref2va-4step-split"},
    // v1.2 of the 4-step 768p line, both copies. The SPLIT one is
    // catalogued first because it is the one to reach for, and it is
    // named without a suffix for the same reason -- `-comfyui` marks the
    // fused copy here rather than `-split` marking the good one, which
    // is the naming the older entries above would have had if the
    // grouping trap had been understood when they were written.
    //
    // WHAT UPSTREAM SAYS ABOUT v1.1 AND v1.2 IS NOTHING. The model card
    // documents only the 8-step v1.0 it deploys, so everything below is
    // read off the FILENAME: fl2v, 4 steps, 768p, bf16. That is enough
    // for the fields here, and the one number it does not carry is the
    // SHIFT -- inherited from the rest of the 768p line at 6, because a
    // distillation is fit to the sigma grid it was trained on and every
    // 768p checkpoint in this repo was trained on that one. If a later
    // release breaks that pattern it will do so silently, so the shift
    // is the thing to re-check against upstream, not the step count.
    //
    // v1.1 is NOT catalogued. It is the same shape as v1.2 and nothing
    // published says what changed between them, so registering both
    // would offer a user a choice this file cannot describe -- and the
    // later one is the one a version number is for.
    {.family = "MiniMax", .version = "H3-FL2VA", .param_class = "LoRA",
     .variant = "Turbo 4-step v1.2 768p, shift 6, split qkv (lightx2v)",
     .hf_path = "lightx2v/Minimax-h3-Turbo",
     .model_type = "minimax-h3-lora",
     .parent_model_type = "minimax-h3-fl2va",
     .files = {"minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"},
     .needs_tokenizer_json = false,
     .name = "lightx2v/Minimax-h3-Turbo-4step-768p-v1.2"},
    {.family = "MiniMax", .version = "H3-FL2VA", .param_class = "LoRA",
     .variant = "Turbo 4-step v1.2 768p, shift 6, fused qkv (lightx2v)",
     .hf_path = "lightx2v/Minimax-h3-Turbo",
     .model_type = "minimax-h3-lora",
     .parent_model_type = "minimax-h3-fl2va",
     .files = {"minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16"
               ".safetensors"},
     .needs_tokenizer_json = false,
     .name = "lightx2v/Minimax-h3-Turbo-4step-768p-v1.2-comfyui"},
    // The Ref2VA 8-step at 768p, both copies -- the SECOND adapter for
    // the reference partition, where there was one.
    //
    // It is also the first Ref2VA adapter that does NOT run on the
    // checkpoint's own shifts. Every other reference-partition option
    // here was distilled at 544p on 12 / 3; this one is 768p, so it
    // takes the 6 that the whole 768p line takes -- and a Ref2VA graph
    // is exactly where that is easiest to miss, because the partition's
    // own documentation never had to mention a shift before.
    //
    // Same reading as the v1.2 pair above: the fields come off the
    // filename and the shift is inherited, because upstream's card
    // describes only the FL2VA 8-step it deploys.
    {.family = "MiniMax", .version = "H3-Ref2VA", .param_class = "LoRA",
     .variant = "Turbo 8-step v1.0 768p, shift 6, split qkv (lightx2v)",
     .hf_path = "lightx2v/Minimax-h3-Turbo",
     .model_type = "minimax-h3-lora",
     .parent_model_type = "minimax-h3-ref2va",
     .files = {"minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"},
     .needs_tokenizer_json = false,
     .name = "lightx2v/Minimax-h3-Turbo-ref2va-8step-768p"},
    {.family = "MiniMax", .version = "H3-Ref2VA", .param_class = "LoRA",
     .variant = "Turbo 8-step v1.0 768p, shift 6, fused qkv (lightx2v)",
     .hf_path = "lightx2v/Minimax-h3-Turbo",
     .model_type = "minimax-h3-lora",
     .parent_model_type = "minimax-h3-ref2va",
     .files = {"minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16"
               ".safetensors"},
     .needs_tokenizer_json = false,
     .name = "lightx2v/Minimax-h3-Turbo-ref2va-8step-768p-comfyui"},
    // The single-image VAE -- the one piece that lets an H3 latent
    // become a PICTURE.
    //
    // WHY IT IS NEEDED, and it is not a convenience. H3's encoder turns
    // a still into exactly 1 latent frame, and the chunked video decoder
    // cannot read that back: it spends `token_drop` latent frames
    // priming its temporal window, so it needs 7 before it emits
    // anything and refuses 1 outright. So the released model can encode
    // a picture it is unable to decode. This checkpoint's decoder was
    // fine-tuned to read one token on its own.
    //
    // VERIFIED against the released video VAE rather than taken from the
    // card: all 116 encoder tensors, `quant_conv` and both whitening
    // vectors are BYTE-IDENTICAL, while 439 of 440 decoder tensors and
    // `post_quant_conv` differ. The LATENT SPACE is therefore unchanged
    // and any H3 latent decodes here -- which is also why this attaches
    // to either partition and pins no parent: the same file serves
    // FL2VA and Ref2VA, and duplicating the entry to say so would
    // fetch 5.2 GB twice under two names.
    //
    // DO NOT PUT IT IN A VIDEO GRAPH. Upstream reports that its decoder
    // materially regresses multi-frame reconstruction -- patch-grid
    // ghosting and cross-frame mixing. Nothing about the weights
    // distinguishes the two files (same 562 names, same shapes), so the
    // two guards are the FILENAME, which loses to a `video_vae` one in
    // the component resolver's preference order, and the `h3_t1_direct`
    // flag in its metadata, which is what the decoder checks.
    {.family = "MiniMax", .version = "H3-Image-VAE", .param_class = "2.4B",
     .variant = "fp16 single-latent-frame decoder (Mamad8)",
     .hf_path = "Mamad8/MiniMax-H3-Image-VAE",
     .model_type = "minimax-h3-image-vae",
     .inputs = {"image"}, .outputs = {"image"},
     .files = {"minimax_h3_t1_image_vae_step1597.safetensors"},
     .weight_format = "comfyui",
     .needs_tokenizer_json = false,
     .name = "Mamad8/MiniMax-H3-Image-VAE"},
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

// The identity two entries must not share: the registration KEY (an
// explicit `name`, else the repo path) plus the file subset, because one
// repo legitimately publishes several models and they differ only in
// which files they pin (the two MiniMax-H3 partitions, the six
// vpipe-supplement archives).
std::string
catalog_identity_(const ModelCatalogEntry& e)
{
  std::string id = e.name.empty() ? e.hf_path : e.name;
  id += '\n';
  for (const auto& f : e.files) { id += f; id += '\x1f'; }
  return id;
}

// The published catalogue. Registration swaps in a new snapshot and
// RETAINS the old one, so a reference or element pointer handed out
// before the swap stays valid for the life of the process -- see the
// lifetime note on register_catalog_entries.
std::mutex&
catalog_mu_()
{
  static std::mutex mu;
  return mu;
}

const std::vector<ModelCatalogEntry>*& catalog_current_()   // guarded by mu
{
  static const std::vector<ModelCatalogEntry>* cur = nullptr;
  return cur;
}

std::vector<std::unique_ptr<const std::vector<ModelCatalogEntry>>>&
catalog_snapshots_()                                        // guarded by mu
{
  static std::vector<std::unique_ptr<const std::vector<ModelCatalogEntry>>> v;
  return v;
}

}  // namespace

const std::vector<ModelCatalogEntry>&
model_catalog()
{
  std::lock_guard<std::mutex> lk(catalog_mu_());
  const auto*& cur = catalog_current_();
  // Until a plugin contributes anything there is nothing to compose, so
  // the built-in table IS the catalogue -- no copy in the common case.
  if (cur == nullptr) { return builtin_catalog_(); }
  return *cur;
}

std::size_t
register_catalog_entries(std::vector<ModelCatalogEntry> entries)
{
  if (entries.empty()) { return 0; }
  std::lock_guard<std::mutex> lk(catalog_mu_());
  const auto*& cur = catalog_current_();
  const std::vector<ModelCatalogEntry>& base =
      cur != nullptr ? *cur : builtin_catalog_();

  std::unordered_set<std::string> seen;
  seen.reserve(base.size() * 2);
  for (const auto& e : base) { seen.insert(catalog_identity_(e)); }

  auto next = std::make_unique<std::vector<ModelCatalogEntry>>(base);
  std::size_t taken = 0;
  for (auto& e : entries) {
    if (e.hf_path.empty() && e.name.empty()) { continue; }
    if (!seen.insert(catalog_identity_(e)).second) { continue; }  // first-wins
    next->push_back(std::move(e));
    ++taken;
  }
  if (taken == 0) { return 0; }        // nothing new; keep the snapshot

  const std::vector<ModelCatalogEntry>* published = next.get();
  catalog_snapshots_().push_back(std::move(next));
  cur = published;
  return taken;
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
             ) {
    set({"text", "image"}, {"image"});
  } else if (mt == "qwen-image") {
    // The text-to-image half of the Qwen-Image family: no reference
    // image, which is the whole difference between it and the edit
    // model it is otherwise architecturally identical to. Saying
    // {"text", "image"} here would offer it to a stage that needs an
    // image input and hand that stage a checkpoint with nowhere to put
    // one.
    set({"text"}, {"image"});
  } else if (mt == "qwen-image-21") {
    // 2.1 is the case the split above does not have: ONE checkpoint
    // that answers both tasks, because a reference image occupies
    // vision slots the text encoder reserved rather than a separate
    // conditioning path. So it belongs in {"text", "image"} even though
    // it generates from a bare prompt too -- the image input is
    // optional, not absent, and splitting it into two model_types the
    // way qwen-image / qwen-image-edit are split would describe two
    // checkpoints where there is one.
    set({"text", "image"}, {"image"});
  } else if (mt == "z-image") {
    // Text-to-image ONLY. The family's editing checkpoint
    // (Z-Image-Edit) and its omni base are not released, and this
    // model has no vision tower and no reference path at all -- so
    // {"text", "image"} would offer it to a stage that needs an image
    // input and hand it a checkpoint with nowhere to put one.
    set({"text"}, {"image"});
  } else if (mt == "wan-i2v") {
    // The one family here that OUTPUTS video: a prompt plus a first-frame
    // image in, a clip out.
    set({"text", "image"}, {"video"});
  } else if (mt == "wan-t2v") {
    set({"text"}, {"video"});
  } else if (mt == "flashvsr") {
    // RESTORATION: a low-quality clip in, the same clip at 4x out. No
    // prompt -- the conditioning is a constant the checkpoint ships.
    set({"video"}, {"video"});
  } else if (mt == "minimax-h3-fl2va") {
    // The only entry whose OUTPUT is two modalities: one denoise loop
    // over one packed sequence emits the clip and its soundtrack
    // together. Images are optional (zero, one or two keyframes), so the
    // text-only t2va task is the same checkpoint.
    set({"text", "image"}, {"video", "audio"});
  } else if (mt == "minimax-h3-ref2va") {
    // The reference-conditioned partition of the same checkpoint: it
    // takes VIDEO and AUDIO references besides the prompt and stills.
    //
    // Absent from this table until now, and the gap only showed on a
    // model registered FROM DISK -- a fetched one carries the I/O its
    // catalogue entry records. So every locally quantized Ref2VA had no
    // modalities at all, and a picker that filters on need_inputs hid
    // it from exactly the stages that would use it.
    set({"text", "image", "video", "audio"}, {"video", "audio"});
  } else if (mt == "minimax-h3-image-vae") {
    // A VAE and nothing else: pixels in, pixels out, with no prompt and
    // no denoise anywhere in it. It is listed as its own model rather
    // than as a supplement because it is a COMPLETE autoencoder -- the
    // frozen H3 encoder plus a decoder retrained to read one latent
    // frame -- so it round-trips a picture on its own, and because the
    // same file serves both H3 partitions.
    set({"image"}, {"image"});
  } else if (mt == "boogu-image-edit") {
    set({"text", "image"}, {"image"});
  } else if (mt == "boogu-image") {
    // The t2i instantiation takes no reference image (the edit
    // checkpoints, "boogu-image-edit", are the image-conditioned ones).
    set({"text"}, {"image"});
  } else if (mt == "yolo") {
    set({"image"}, {});
  } else if (mt == "silero-vad" || mt == "audio-tagging") {
    set({"audio"}, {});
  } else if (mt == "qwen3.5-vision-encoder"
             || mt == "gemma4-vision-encoder") {
    set({"image", "video"}, {});
  } else if (const FlexData* p = genai::detect::for_model_type(mt)) {
    // A REGISTERED family's own modalities. A CATALOGUED model carries
    // its own on the entry; this is the other half -- a model detected
    // on disk, whose type is all there is to go on.
    for (std::string& s : genai::detect::strings(p, genai::detect::kInputs)) {
      in.push_back(std::move(s));
    }
    for (std::string& s :
         genai::detect::strings(p, genai::detect::kOutputs)) {
      out.push_back(std::move(s));
    }
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

std::vector<const ModelCatalogEntry*>
catalog_all_by_path(const std::string& hf_path)
{
  std::vector<const ModelCatalogEntry*> out;
  for (const auto& e : model_catalog()) {
    if (e.hf_path == hf_path) { out.push_back(&e); }
  }
  return out;
}

const ModelCatalogEntry*
catalog_pick_variant(const std::vector<const ModelCatalogEntry*>& cands,
                     const std::string& variant,
                     std::vector<const ModelCatalogEntry*>* matched)
{
  if (matched != nullptr) { matched->clear(); }
  if (cands.empty()) { return nullptr; }
  if (variant.empty()) {
    // Nothing to match on: only a single candidate is unambiguous.
    if (cands.size() != 1) { return nullptr; }
    if (matched != nullptr) { matched->push_back(cands.front()); }
    return cands.front();
  }
  auto lower = [](std::string v) {
    for (char& c : v) { c = (char)tolower((unsigned char)c); }
    return v;
  };
  const std::string w = lower(variant);
  std::vector<const ModelCatalogEntry*> exact, part;
  for (const ModelCatalogEntry* e : cands) {
    const std::string fields[] = {lower(e->version), lower(e->variant),
                                  lower(e->name), lower(e->model_type)};
    bool is_exact = false, is_part = false;
    for (const std::string& f : fields) {
      if (f.empty()) { continue; }
      if (f == w) { is_exact = true; }
      else if (f.find(w) != std::string::npos) { is_part = true; }
    }
    if (is_exact)     { exact.push_back(e); }
    else if (is_part) { part.push_back(e); }
  }
  // An exact hit anywhere outranks every substring hit; only within one
  // tier does a tie mean ambiguity.
  const std::vector<const ModelCatalogEntry*>& hits =
      exact.empty() ? part : exact;
  if (matched != nullptr) { *matched = hits; }
  return hits.size() == 1 ? hits.front() : nullptr;
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
    f.git_oid = obj.contains("oid")
        ? std::string(obj.at("oid").as_string("")) : "";
    f.xet_hash = obj.contains("xetHash")
        ? std::string(obj.at("xetHash").as_string("")) : "";
    // The LFS block is present only for the files git does not store
    // itself -- which is to say, exactly the multi-GB shards where a
    // truncated or resumed download is worth checking.
    if (obj.contains("lfs")) {
      FlexData lfs = obj.at("lfs");        // own it; the view dangles
      if (lfs.is_object()) {
        auto lo = lfs.as_object();
        if (lo.contains("oid")) {
          f.sha256 = std::string(lo.at("oid").as_string(""));
        }
        // The LFS size is the authority when both are given: `size` on
        // the outer entry is the pointer file's for some revisions.
        if (lo.contains("size")) {
          const std::uint64_t n = lo.at("size").as_uint(0);
          if (n > 0) { f.size = n; }
        }
      }
    }
    out.push_back(std::move(f));
  }
  return out;
}

}
