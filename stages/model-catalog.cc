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
    // Qwen3.6-35B-A3B: a Qwen3.5-family MoE (SparseMoeBlock: routed experts +
    // shared expert, model_type "qwen3_5"). bf16 source (~70 GB) -- whole-repo
    // fetch; quantize with model-quantize (4-bit AWQ + on-device auto-calib;
    // the dense bf16 path covers the MoE experts so calibration can run).
    {.family = "Qwen", .version = "3.6", .param_class = "35B-A3B",
     .variant = "bf16 (Qwen)",
     .hf_path = "Qwen/Qwen3.6-35B-A3B",
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
    // subfolders -- three sub-models the text-to-image stages consume:
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
    // Krea-2 softwatercolor LoRA (adapts the Turbo DiT). Fuse into a DiT with
    // the lora-fuse stage (base = <Krea-2-Turbo>/transformer), then use the
    // fused DiT via the text-to-image `dit_dir`. Trigger: "Art Deco watercolor
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
  } else if (mt == "krea2" || mt == "flux2" || mt == "qwen-image-edit") {
    set({"text", "image"}, {"image"});
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
