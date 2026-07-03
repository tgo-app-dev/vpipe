#ifndef VPIPE_GENAI_QUANTIZE_CALIBRATION_H
#define VPIPE_GENAI_QUANTIZE_CALIBRATION_H

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "generative-models/qwen3/metal-qwen-model.h"

namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class Tokenizer;

// The built-in calibration corpus text (a diversity-strided subset of the
// mlx-lm calibration_data_v5 set: prose / code / SQL / HTML / math /
// multilingual). Newline-separated documents; defined in the generated
// calibration-corpus.cc.
std::string_view builtin_calibration_text();

// Build token-id calibration sequences from the built-in corpus. The corpus
// text is grouped into up to `max_seqs` documents of roughly `seq_len` tokens
// each (each sequence truncated to `seq_len`). When `apply_chat_template` is
// set and the tokenizer carries chat control tokens (ChatML <|im_start|>/
// <|im_end|> or the Llama-3 header tokens), each document is wrapped as a user
// turn so the special control tokens are exercised in the calibration
// distribution; otherwise the raw text is encoded. Effective AWQ at 4-bit
// needs ~128 sequences of ~512 tokens. Returns the sequences (some may be
// shorter than seq_len for the tail document).
std::vector<std::vector<std::int32_t>>
build_builtin_calibration_corpus(const Tokenizer& tok, int max_seqs,
                                 int seq_len, bool apply_chat_template);

// On-device AWQ calibration: run the (already-quantized, ideally 8-bit for
// fidelity) backbone in `model_dir` over a token-id corpus and write the
// per-layer per-input-channel activation abs-max stats AWQ consumes --
// calib_qkv.f32 / calib_gateup.f32 [n_layers, hidden] + calib_down.f32
// [n_layers, ffn_inner], the same format as the offline HF script. Streams
// one prefill per sequence; peak extra RAM is the per-layer tap buffers. This
// replaces the dump/moss15_calib.py dependency for >memory self-calibration.
// Dense full-attention backbones only (the v1.5 Qwen3). Returns false + *err
// on failure.
bool collect_backbone_calibration(
    metal_compute::MetalCompute* mc, const std::string& model_dir,
    const MetalQwenModel::Config& cfg,
    const std::vector<std::vector<std::int32_t>>& corpus,
    const std::string& out_calib_dir, std::string* err,
    const std::function<bool()>& stop = [] { return false; });

// ---- Streaming layer-by-layer calibration (MoE, >memory-safe) -----------
// The MoE counterpart of collect_backbone_calibration. The dense
// collect_backbone_calibration loads the WHOLE backbone via
// MetalQwenModel::load (every layer's weights resident at once) -- fine for a
// few-GB dense model, fatal for the 35B MoE (~36 GB 8-bit / ~67 GB bf16:
// filling >RAM of mlock-wired UMA buffers hangs the box). This path NEVER
// holds more than ONE layer's weights: it mmaps `model_dir` via
// MetalLlamaWeights (pageable), embeds the corpus into a held residual stream
// (tens of MB), then for L = 0..n_layers-1 loads ONLY layer L's tensors into
// transient SharedBuffers, runs that layer over the residual, taps the
// per-input-channel |activation| (incl. PER-EXPERT gate/up + down stats for
// the MoE MLP), advances the residual, and FREES layer L before L+1. A host
// free-RAM guard (host_free_ram_bytes) aborts BEFORE loading a layer if free
// RAM < `min_free_bytes` so a regression that accidentally accumulates can
// never reach the panic threshold. Writes the same calib_qkv/gateup/down.f32
// PLUS per-expert calib_expert_gateup.f32 [L][E][H] / calib_expert_down.f32
// [L][E][I]. Returns false + *err on failure.
bool collect_backbone_calibration_streaming(
    metal_compute::MetalCompute* mc, const std::string& model_dir,
    const MetalQwenModel::Config& cfg,
    const std::vector<std::vector<std::int32_t>>& corpus,
    const std::string& out_calib_dir, std::string* err,
    std::uint64_t min_free_bytes = (std::uint64_t)8 << 30,
    const std::function<bool()>& stop = [] { return false; });

// Free physical RAM right now, in bytes (mach host_statistics64 free +
// inactive + purgeable pages). The streaming calibration polls this before
// each layer load and aborts if it falls below its guard threshold, so a
// full-load regression can never reach the panic point. 0 if the query fails.
std::uint64_t host_free_ram_bytes();

// Resident + wired bytes of THIS process (mach task_vm_info: phys_footprint
// and wired). For the periodic memory print the streaming calibration emits so
// an external vm_stat monitor can be cross-checked. Either is 0 on failure.
void process_memory_bytes(std::uint64_t* resident, std::uint64_t* wired);

}  // namespace vpipe::genai

#endif
