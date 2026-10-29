#ifndef VPIPE_GENERATIVE_MODELS_MINIMAX_MUSIC3_METAL_MUSIC3_AR_H
#define VPIPE_GENERATIVE_MODELS_MINIMAX_MUSIC3_METAL_MUSIC3_AR_H

// MetalMusic3Ar -- MiniMax-Music3's autoregressive half on metal-compute:
// the Qwen3-8B "global" LM that picks each 25 Hz frame's semantic code and
// the 0.6B "local" depth decoder that picks its seven residual codes, both
// under classifier-free guidance, and the per-frame hidden states the
// flow matching is conditioned on (music3-protocol.h has the protocol).
//
//   LM     MetalQwenModel, backbone-only, owns its paged K/V: two
//          contexts, the prompt and its guidance twin, stepped in
//          lockstep. Its head and embedding are this class's, and only
//          the rows the protocol uses are held: the head over the code
//          window (16389 of 200000 rows) and the 16384 code embeddings;
//          a prompt's own rows are read from the checkpoint by range.
//   DEPTH  four pre-norm blocks (16 heads of 256, NO rotary, a learned
//          position per step, SwiGLU 6144), the whole frame's K/V in a
//          16-position cache per guidance row. Step 1 runs positions 0
//          (the LM's hidden) and 1 (c0's LM embedding) together; step i
//          appends c_{i-1}'s own embedding at position i.
//
// Both are held as published (bf16) or, by default, with their
// projections as affine w8 group-64 built in memory as they load: every
// frame reads every weight of both (the LM once per guidance row, the
// depth decoder once per code), so the frame rate is the bytes.

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/minimax-music3/music3-protocol.h"
#include "generative-models/qwen3/metal-qwen-model.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::metal_compute {
class MetalCompute;
class ComputeEncoder;
}

namespace vpipe::genai {

class SafetensorsRanges;
class WeightSet;

class MetalMusic3Ar {
public:
  struct Config {
    // The global LM (language_model/config.json).
    int   layers     = 36;
    int   hidden     = 4096;
    int   heads      = 32;
    int   kv_heads   = 8;
    int   head_dim   = 128;
    int   ffn        = 12288;
    int   vocab      = 200000;
    float rope_theta = 1.0e6f;
    float rms_eps    = 1e-6f;
    int   context    = 10240;        // max_position_embeddings
    // The depth decoder (rvq_depth_decoder/config.json).
    int   d_layers   = 4;
    int   d_heads    = 16;
    int   d_ffn      = 6144;
    int   d_vocab    = 1024;
    int   codebooks  = 8;
    int   d_positions = 16;
    // How the projections are held: 8 = affine w8 group-64 built in
    // memory, 0 = bf16 as published. Not read from config.json.
    int   lm_quant_bits    = 8;
    int   depth_quant_bits = 8;
  };

  // language_model/config.json and rvq_depth_decoder/config.json under the
  // model root.
  static bool config_from_dir(const std::string& root, Config* out,
                              std::string* err = nullptr);

  // `lm` is language_model/'s set, `depth` rvq_depth_decoder/'s. The model
  // KEEPS both.
  static std::unique_ptr<MetalMusic3Ar> load(
      std::shared_ptr<WeightSet> lm, std::shared_ptr<WeightSet> depth,
      metal_compute::MetalCompute* mc, const Config& cfg,
      std::string* err = nullptr);
  ~MetalMusic3Ar();

  const Config& config() const { return _cfg; }
  bool lm_quantized() const;
  bool depth_quantized() const { return _depth_w8; }

  // What a model from `root` will hold, from the checkpoint's tensor
  // headers alone, as the loader builds it: the LM's layers (w8 built in
  // memory, or bf16) and the two windows of its tables
  // (language_model/), and the depth decoder (rvq_depth_decoder/). 0 when
  // a directory is unreadable.
  static std::size_t lm_weight_bytes(const std::string& root,
                                     int lm_quant_bits);
  static std::size_t depth_weight_bytes(const std::string& root,
                                        int depth_quant_bits);
  std::size_t resident_bytes() const;
  // ... of which the depth decoder's (rvq_depth_decoder/).
  std::size_t depth_bytes() const;
  // What a song of `prefix` prompt tokens and `frames` frames allocates
  // beside the weights, by the backbone's own figure
  // (MetalQwenModel::run_bytes: both contexts' paged K/V at the capacity
  // the pool's doubling can reach -- capped at two whole contexts -- a
  // prefill piece's activations, the prompt's embeddings, the decode
  // scratch) plus this class's: the depth decoder's and the batched
  // step's scratch. `matrix_cores` as the box has them.
  static std::size_t run_bytes(const Config& c, int prefix, int frames,
                               bool matrix_cores);
  // The song it hands on (Song::hiddens, [frames, 32768] f16): host
  // memory, alive until the flow matching has read it.
  static std::size_t handoff_bytes(const Config& c, int frames);
  // The K/V pool as it stands now (it keeps the capacity it grew to).
  std::size_t held_scratch_bytes() const;

  // ---- a song -----------------------------------------------------------
  struct Song {
    // [frames, 8 * 4096] f16: per kept frame, the LM's hidden that picked
    // its c0, then the depth decoder's that picked c1..c7.
    std::vector<_Float16> hiddens;
    // Every frame's codes, the first (not kept) included; an ending frame
    // is not one.
    std::vector<std::array<int, 8>> codes;
    int    frames = 0;               // kept frames
    bool   ended = false;            // the model picked <|audio_end|>
    bool   stopped = false;          // the callback asked to stop
    double prefill_seconds = 0.0;
    double seconds = 0.0;
  };
  // Called after every kept frame with the kept count. False stops.
  using FrameFn = std::function<bool(int frames)>;

  // Up to `max_frames` kept frames from the prompt `ids` (the conditional
  // prompt; its guidance twin is derived), drawing with `seed` under
  // guidance `cfg` (the protocol's 1.5).
  bool generate(const std::vector<std::int32_t>& ids, int max_frames,
                std::uint64_t seed, const FrameFn& on_frame, Song* out,
                std::string* err = nullptr, float cfg = music3::kArCfg);

  // Test hook: TEACHER-FORCED along `forced` (a frame's codes; c0 < 0 is
  // <|audio_end|>), recording each step's raw logits: the c0 window
  // ([steps, 2, kHeadRows], conditional row first) and each depth head's
  // ([steps, 7, 2, 1024]).
  struct Forced {
    std::vector<float> c0_logits;
    std::vector<float> depth_logits;
  };
  bool generate_forced(const std::vector<std::int32_t>& ids, int max_frames,
                       const std::vector<std::array<int, 8>>& forced,
                       Song* out, Forced* rec, std::string* err = nullptr);

private:
  MetalMusic3Ar() = default;

  // A projection, bf16 or affine w8 g64 (codes, scales, biases).
  struct Lin {
    metal_compute::SharedBuffer w, s, b;
    int n = 0, k = 0;
    bool w8() const { return !s.empty(); }
    bool empty() const { return w.empty(); }
  };
  struct DLayer {
    metal_compute::SharedBuffer ln1, ln2;
    Lin qkv, out, gu, down;          // gu: interleaved (gate f, up f)
  };

  bool init_(const std::shared_ptr<WeightSet>& lm,
             const std::shared_ptr<WeightSet>& depth,
             metal_compute::MetalCompute* mc, const Config& cfg,
             std::string* err);
  // The backbone's config, one for the load and the plan.
  static MetalQwenModel::Config qwen_config_(const Config& c);
  // The scratch init_ allocates (the depth decoder's rows and K/V, the
  // two LM hiddens, the head's logits, the frame's input row).
  static std::size_t own_scratch_bytes_(const Config& c);
  Lin lin_(WeightSet& ws, const std::string& key,
           const std::vector<std::string>& parts, int k, bool interleave,
           bool quantize);
  bool run_(const std::vector<std::int32_t>& ids, int max_frames,
            std::uint64_t seed, float cfg, const FrameFn& on_frame,
            const std::vector<std::array<int, 8>>* forced, Song* out,
            Forced* rec, std::string* err);
  // The prompt's embedding rows, read by range, bf16 [n, hidden].
  metal_compute::SharedBuffer embed_prompt_(
      const std::vector<std::int32_t>& ids, std::string* err);
  // The LM step of both contexts from the same input row (or a prefill),
  // leaving their final-normed hiddens in _h and the head's window logits
  // in _logits.
  bool lm_heads_(std::string* err);
  void encode_head_(metal_compute::ComputeEncoder& enc, int row);
  // Depth step i (1..7) of both guidance rows. Step 1 reads _h and c0;
  // a later one c_{i-1}. Leaves the head's logits [2, 1024] in _dlogits
  // and the conditional row's normed hidden at hiddens slot i.
  // The kept stream (VPIPE_MUSIC3_FRESH_STREAMS=1: a new one each call,
  // the A/B).
  metal_compute::CommandStream& stream_();
  bool depth_step_(int i, int prev_code, _Float16* frame_hidden,
                   std::string* err);
  void gemv_(metal_compute::ComputeEncoder& enc, const Lin& w,
             const metal_compute::SharedBuffer& x, std::size_t xe,
             const metal_compute::SharedBuffer& y, std::size_t ye, int rows);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::shared_ptr<WeightSet> _lm_ws, _depth_ws;
  // Every depth step and the prefill's head commit on this one stream: a
  // fresh queue's first commit costs ~1-2 ms, seven times a frame.
  metal_compute::CommandStream _stream;
  std::unique_ptr<MetalQwenModel> _lm;
  std::unique_ptr<SafetensorsRanges> _rows;   // the prompt's rows

  // LM, this class's half: the code embeddings and the head window, bf16.
  metal_compute::SharedBuffer _code_emb, _head;
  // Depth decoder.
  std::vector<DLayer> _d;
  Lin _proj;
  metal_compute::SharedBuffer _pos, _dnorm, _aemb, _dheads;
  bool _depth_w8 = false;

  // Scratch. _h: the two LM hiddens [2, H]; _logits: [2, kHeadRows];
  // _emb: an LM input row; the depth step's rows (at most 4).
  metal_compute::SharedBuffer _h, _logits, _emb, _din, _dx, _dn, _dqkv,
      _datt, _dtmp, _dgu, _dact, _dlogits, _kc, _vc;

  metal_compute::ComputeLibrary _lib_gemm, _lib_qmv, _lib_rms, _lib_elt,
      _lib_m3;
  metal_compute::ComputeFunction _fn_gemv_rows, _fn_qmv_batch,
      _fn_qmv_batch4, _fn_rms,
      _fn_add, _fn_glu, _fn_kv, _fn_attn, _fn_pos, _fn_copy;
};

}  // namespace vpipe::genai

#endif
