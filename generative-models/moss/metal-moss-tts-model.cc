#include "generative-models/moss/metal-moss-tts-model.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {
inline float
bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t bits = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}
inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  b += 0x7fffu + ((b >> 16) & 1u);   // round-to-nearest-even
  return (std::uint16_t)(b >> 16);
}
int
geti_(const FlexData::ConstObjectView& o, const char* k, int d)
{
  return o.contains(k) ? (int)o.at(k).as_int(d) : d;
}
double
getr_(const FlexData::ConstObjectView& o, const char* k, double d)
{
  return o.contains(k) ? o.at(k).as_real(d) : d;
}

// Host sampler over logit[0..n): temperature / top_k / top_p, with an optional
// repetition penalty against `seen` (HF style: scale a seen logit by 1/pen if
// >0 else *pen). `blocked` (small) excludes ids; temperature <= 0 => greedy
// argmax. `cand`/`prob` are caller scratch (reused across steps, no per-call
// alloc). Returns the chosen index.
int
sample_logits_(const float* logit, int n, const MossSampling& sp,
               const std::vector<int>& blocked,
               const std::unordered_set<int>* seen,
               std::vector<int>& cand, std::vector<float>& prob,
               std::mt19937_64& rng)
{
  auto is_blocked = [&](int i) {
    for (int b : blocked) { if (b == i) { return true; } }
    return false;
  };
  const bool rep = sp.repetition_penalty != 1.0f && seen != nullptr;
  auto adj = [&](int i) -> float {     // logit after repetition penalty
    float v = logit[i];
    if (rep && seen->count(i)) {
      v = (v > 0.0f) ? v / sp.repetition_penalty : v * sp.repetition_penalty;
    }
    return v;
  };
  if (sp.temperature <= 0.0f) {                       // greedy
    float bv = -std::numeric_limits<float>::infinity();
    int best = -1;
    for (int i = 0; i < n; ++i) {
      if (is_blocked(i)) { continue; }
      const float v = adj(i);
      if (v > bv) { bv = v; best = i; }
    }
    return best;
  }
  cand.clear();
  for (int i = 0; i < n; ++i) { if (!is_blocked(i)) { cand.push_back(i); } }
  if (cand.empty()) { return 0; }
  // top_k: select the k highest-adjusted-logit candidates, sorted desc.
  int k = (sp.top_k > 0 && sp.top_k < (int)cand.size())
              ? sp.top_k
              : (int)cand.size();
  auto gt = [&](int a, int b) { return adj(a) > adj(b); };
  if (k < (int)cand.size()) {
    std::nth_element(cand.begin(), cand.begin() + (k - 1), cand.end(), gt);
    cand.resize((std::size_t)k);
  }
  std::sort(cand.begin(), cand.end(), gt);
  // temperature softmax over the kept set, then top_p nucleus truncation.
  const float inv_t = 1.0f / sp.temperature;
  const float mx = adj(cand[0]) * inv_t;
  prob.resize(cand.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < cand.size(); ++i) {
    const float p = std::exp(adj(cand[i]) * inv_t - mx);
    prob[i] = p;
    sum += p;
  }
  std::size_t keep = cand.size();
  if (sp.top_p < 1.0f) {
    const double thresh = sp.top_p * sum;
    double cum = 0.0;
    keep = 0;
    for (std::size_t i = 0; i < cand.size(); ++i) {
      cum += prob[i];
      ++keep;
      if (cum >= thresh) { break; }
    }
  }
  double ksum = 0.0;
  for (std::size_t i = 0; i < keep; ++i) { ksum += prob[i]; }
  const double target =
      std::uniform_real_distribution<double>(0.0, 1.0)(rng) * ksum;
  double acc = 0.0;
  for (std::size_t i = 0; i < keep; ++i) {
    acc += prob[i];
    if (acc >= target) { return cand[i]; }
  }
  return cand[keep - 1];
}
}  // namespace

std::unique_ptr<MetalMossTtsModel>
MetalMossTtsModel::load(const std::string& model_dir,
                        metal_compute::MetalCompute* mc)
{
  if (mc == nullptr || !mc->valid()) {
    return nullptr;
  }
  auto self = std::unique_ptr<MetalMossTtsModel>(new MetalMossTtsModel());
  self->_mc = mc;
  Config cfg;

  // Backbone (dense Qwen3) shape, overridable from config.json/language_config.
  MetalQwenModel::Config bc;
  bc.n_layers   = 36;
  bc.hidden     = 4096;
  bc.n_heads    = 32;
  bc.n_kv_heads = 8;
  bc.head_dim   = 128;
  bc.ffn_inner  = 12288;
  bc.vocab      = 155648;
  bc.rope_theta = 1.0e6f;
  bc.rms_eps    = 1e-6f;

  {
    std::ifstream in(model_dir + "/config.json");
    if (in) {
      try {
        FlexData root = FlexData::from_json(in);
        if (root.is_object()) {
          const auto ro = root.as_object();
          cfg.n_vq            = geti_(ro, "n_vq", cfg.n_vq);
          cfg.audio_vocab     = geti_(ro, "audio_vocab_size", cfg.audio_vocab);
          cfg.audio_pad_code  = geti_(ro, "audio_pad_code", cfg.audio_vocab);
          cfg.sampling_rate   = geti_(ro, "sampling_rate", cfg.sampling_rate);
          cfg.audio_start     =
              geti_(ro, "audio_start_token_id", cfg.audio_start);
          cfg.audio_end       =
              geti_(ro, "audio_end_token_id", cfg.audio_end);
          cfg.audio_user_slot =
              geti_(ro, "audio_user_slot_token_id", cfg.audio_user_slot);
          cfg.audio_gen_slot  =
              geti_(ro, "audio_assistant_gen_slot_token_id",
                    cfg.audio_gen_slot);
          cfg.audio_delay_slot =
              geti_(ro, "audio_assistant_delay_slot_token_id",
                    cfg.audio_delay_slot);
          if (ro.contains("language_config")) {
            const FlexData lc = ro.at("language_config");
            if (lc.is_object()) {
              const auto lco = lc.as_object();
              bc.n_layers   = geti_(lco, "num_hidden_layers", bc.n_layers);
              bc.hidden     = geti_(lco, "hidden_size", bc.hidden);
              bc.n_heads    = geti_(lco, "num_attention_heads", bc.n_heads);
              bc.n_kv_heads = geti_(lco, "num_key_value_heads", bc.n_kv_heads);
              bc.head_dim   = geti_(lco, "head_dim", bc.head_dim);
              bc.ffn_inner  = geti_(lco, "intermediate_size", bc.ffn_inner);
              bc.vocab      = geti_(lco, "vocab_size", bc.vocab);
              bc.rope_theta = (float)getr_(lco, "rope_theta", bc.rope_theta);
              bc.rms_eps    = (float)getr_(lco, "rms_norm_eps", bc.rms_eps);
              cfg.pad_token = geti_(lco, "pad_token_id", cfg.pad_token);
            }
          }
        }
      } catch (...) {
        // Fall back to the 8B defaults baked above.
      }
    }
  }
  cfg.hidden = bc.hidden;
  cfg.vocab  = bc.vocab;

  bc.rotary_dim        = bc.head_dim;   // Qwen3 full rotary
  bc.full_attn_interval = 1;            // dense: every layer is full-attn
  bc.tie_embeddings    = false;
  bc.use_bf16          = true;          // match the reference dtype
  bc.quant_bits        = 8;
  bc.dense             = true;
  bc.attn_output_gate  = false;
  bc.weight_prefix     = "language_";   // -> language_model.layers / .norm
  bc.backbone_only     = true;
  bc.max_seq           = 16384;
  bc.page_tokens       = 256;

  self->_backbone = MetalQwenModel::load(model_dir, mc, bc);
  if (!self->_backbone) {
    return nullptr;
  }
  self->_cfg = cfg;

  // bf16 embedding tables + output heads (raw, unquantized in the ckpt).
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts) {
    return nullptr;
  }
  self->_embed_tokens = wts->load("language_model.embed_tokens.weight", mc);
  if (self->_embed_tokens.empty()) {
    return nullptr;
  }
  self->_emb_ext.resize((std::size_t)cfg.n_vq);
  for (int i = 0; i < cfg.n_vq; ++i) {
    const std::string nm = "emb_ext." + std::to_string(i) + ".weight";
    self->_emb_ext[(std::size_t)i] = wts->load(nm, mc);
    if (self->_emb_ext[(std::size_t)i].empty()) {
      return nullptr;
    }
  }
  self->_heads.resize((std::size_t)cfg.n_vq + 1);
  self->_head_out.resize((std::size_t)cfg.n_vq + 1);
  for (int i = 0; i <= cfg.n_vq; ++i) {
    const std::string nm = "lm_heads." + std::to_string(i) + ".weight";
    self->_heads[(std::size_t)i] = wts->load(nm, mc);
    if (self->_heads[(std::size_t)i].empty()) {
      return nullptr;
    }
    const auto* info = wts->info(nm);
    self->_head_out[(std::size_t)i] =
        (info != nullptr && !info->shape.empty())
            ? (int)info->shape[0]
            : (i == 0 ? cfg.vocab : cfg.audio_vocab + 1);
  }

  // Dense bf16 GEMV for the heads (same kernel the backbone uses post-dequant).
  self->_lib_dense = mc->load_library("dense_gemm_bf16");
  self->_fn_dense_gemv = self->_lib_dense.function("dense_gemv_t_f16");
  if (!self->_fn_dense_gemv.valid()) {
    return nullptr;
  }
  const std::size_t logit_elts =
      (std::size_t)cfg.vocab + (std::size_t)cfg.n_vq * (cfg.audio_vocab + 1);
  self->_logits_buf = mc->make_shared_buffer(logit_elts * 2);
  return self;
}

SharedBuffer
MetalMossTtsModel::assemble_embeds_(
    const std::vector<std::vector<std::int32_t>>& rows, int start, int n)
{
  const int H = _cfg.hidden, NV = _cfg.n_vq;
  SharedBuffer x = _mc->make_shared_buffer((std::size_t)n * H * 2);
  auto* xp = static_cast<std::uint16_t*>(x.contents());
  const auto* et = static_cast<const std::uint16_t*>(_embed_tokens.contents());
  // Match the reference's accumulation: inputs_embeds starts as the bf16
  // text embedding, then each audio-code embedding is added one at a time
  // with the result ROUNDED BACK TO BF16 after every add (MLX bf16 add).
  // A single f32 accumulation rounded once at the end diverges by ULPs and
  // flips occasional near-tie argmaxes.
  std::vector<std::uint16_t> cur((std::size_t)H);
  for (int r = 0; r < n; ++r) {
    const auto& row = rows[(std::size_t)(start + r)];
    const int text_id = row[0];
    const auto* trow = et + (std::size_t)text_id * H;
    for (int h = 0; h < H; ++h) { cur[(std::size_t)h] = trow[h]; }
    for (int cb = 0; cb < NV; ++cb) {
      const int code = row[(std::size_t)(1 + cb)];
      const auto* arow =
          static_cast<const std::uint16_t*>(_emb_ext[(std::size_t)cb].contents())
          + (std::size_t)code * H;
      for (int h = 0; h < H; ++h) {
        cur[(std::size_t)h] = f32_to_bf16_(
            bf16_to_f32_(cur[(std::size_t)h]) + bf16_to_f32_(arow[h]));
      }
    }
    auto* xr = xp + (std::size_t)r * H;
    for (int h = 0; h < H; ++h) { xr[h] = cur[(std::size_t)h]; }
  }
  return x;
}

void
MetalMossTtsModel::dispatch_heads_(ComputeEncoder& enc, const SharedBuffer& hn,
                                   bool need_full_text)
{
  const int H = _cfg.hidden, NV = _cfg.n_vq;
  const int V = _cfg.vocab, AC = _cfg.audio_vocab + 1;
  auto gemv = [&](const SharedBuffer& w, std::size_t yoff, int N) {
    enc.set_function(_fn_dense_gemv);
    enc.set_buffer(0, hn, 0);
    enc.set_buffer(1, w);
    enc.set_buffer(2, _logits_buf, yoff * 2);
    enc.set_constant(3, H);
    enc.set_constant(4, N);
    enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
  };
  // 32 audio heads always; the full text head only when needed.
  std::size_t off = (std::size_t)V;
  for (int cb = 0; cb < NV; ++cb) {
    gemv(_heads[(std::size_t)(1 + cb)], off, AC);
    off += (std::size_t)AC;
  }
  if (need_full_text) { gemv(_heads[0], 0, V); }
}

void
MetalMossTtsModel::read_heads_(const SharedBuffer& hn, bool need_full_text,
                               bool need_2slots,
                               std::vector<float>& text_logits,
                               std::vector<std::vector<float>>& audio_logits)
{
  const int H = _cfg.hidden, NV = _cfg.n_vq;
  const int V = _cfg.vocab, AC = _cfg.audio_vocab + 1;
  const auto* lb = static_cast<const std::uint16_t*>(_logits_buf.contents());
  audio_logits.assign((std::size_t)NV, {});
  std::size_t off = (std::size_t)V;
  for (int cb = 0; cb < NV; ++cb) {
    auto& al = audio_logits[(std::size_t)cb];
    al.resize((std::size_t)AC);
    for (int i = 0; i < AC; ++i) { al[(std::size_t)i] = bf16_to_f32_(lb[off + i]); }
    off += (std::size_t)AC;
  }
  if (need_full_text) {
    text_logits.resize((std::size_t)V);
    for (int i = 0; i < V; ++i) { text_logits[(std::size_t)i] = bf16_to_f32_(lb[i]); }
  } else if (need_2slots) {
    // Inside audio the text channel only ranks the gen/delay slot tokens:
    // two cheap host dot products hn . text_head_row[slot] (bf16 UMA reads).
    text_logits.resize((std::size_t)V);
    const auto* h = static_cast<const std::uint16_t*>(hn.contents());
    const auto* tw = static_cast<const std::uint16_t*>(_heads[0].contents());
    for (int slot : {_cfg.audio_gen_slot, _cfg.audio_delay_slot}) {
      const auto* row = tw + (std::size_t)slot * H;
      float acc = 0.0f;
      for (int k = 0; k < H; ++k) {
        acc += bf16_to_f32_(h[k]) * bf16_to_f32_(row[k]);
      }
      text_logits[(std::size_t)slot] = acc;
    }
  }
}

void
MetalMossTtsModel::head_logits_(const SharedBuffer& hn, bool need_full_text,
                                bool need_2slots,
                                std::vector<float>& text_logits,
                                std::vector<std::vector<float>>& audio_logits)
{
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    dispatch_heads_(enc, hn, need_full_text);
  }
  stream.commit().wait();
  read_heads_(hn, need_full_text, need_2slots, text_logits, audio_logits);
}

std::vector<std::vector<std::int32_t>>
MetalMossTtsModel::generate_delay(
    const std::vector<std::vector<std::int32_t>>& prompt, int max_new_tokens,
    const MossSampling& audio_sp, const MossSampling& text_sp,
    std::uint64_t seed)
{
  std::vector<std::vector<std::int32_t>> out;
  const int NV = _cfg.n_vq;
  const int seq = (int)prompt.size();
  if (seq <= 0) { return out; }
  ContextManager* cm = _backbone->context_manager();
  const ContextId cid = cm->acquire_root();

  // Sampler state: one RNG (seed 0 => nondeterministic), reused scratch, and
  // optional repetition-penalty history (per-codebook audio + text), allocated
  // only when the respective penalty is enabled.
  std::mt19937_64 rng(seed != 0 ? seed : std::random_device{}());
  std::vector<int> samp_cand;
  std::vector<float> samp_prob;
  const std::vector<int> kNoBlock;
  const bool audio_rep = audio_sp.repetition_penalty != 1.0f;
  const bool text_rep  = text_sp.repetition_penalty != 1.0f;
  std::vector<std::unordered_set<int>> audio_seen(
      audio_rep ? (std::size_t)NV : 0);
  std::unordered_set<int> text_seen;

  // Optional per-phase timing (VPIPE_MOSS_PROFILE).
  const bool kProf = std::getenv("VPIPE_MOSS_PROFILE") != nullptr;
  using Clk = std::chrono::steady_clock;
  auto ms = [](Clk::time_point a, Clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  double t_head = 0, t_fwd = 0, t_asm = 0, t_prefill = 0;
  int prof_steps = 0;

  // Prefill the prompt -> last position's normed hidden. Decode steps then
  // reuse the optimized qmv decode path (cur_hn tracks the live hidden:
  // the prefill buffer first, then the backbone's internal _d_hn scratch).
  const auto p0 = Clk::now();
  SharedBuffer x;
  SharedBuffer prefill_hn;
  {
    // text-prefill perf block (LLM lane). On the FIRST synthesis this also
    // absorbs the cold Metal pipeline-state compilation + first-touch weight
    // residency, so the block is huge on clip 1 and small thereafter.
    PerfAuxScope _pre(_session, kPerfLaneLLM, kGvidLlmPrefill,
                      kPerfLlmPrefillBegin, (std::uint64_t)seq);
    x = assemble_embeds_(prompt, 0, seq);
    prefill_hn = _backbone->forward_embeddings_hidden(cid, x, seq);
  }
  if (kProf) { t_prefill = ms(p0, Clk::now()); }
  if (prefill_hn.empty()) { cm->release(cid); return out; }
  const SharedBuffer* cur_hn = &prefill_hn;

  constexpr std::int64_t kSentinel = std::numeric_limits<std::int64_t>::max();
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  bool is_stopping = false;
  bool is_audio = false;
  int audio_lengths = 0;
  std::int64_t delayed_lengths = kSentinel;

  // Continuation bootstrap from the prompt's tail (matches the reference).
  const int last_text = prompt[(std::size_t)(seq - 1)][0];
  const bool is_continuation =
      (last_text == _cfg.audio_start || last_text == _cfg.audio_gen_slot);
  int audio_start_idx = -1;
  for (int i = 0; i < seq; ++i) {
    if (prompt[(std::size_t)i][0] == _cfg.audio_start) { audio_start_idx = i; }
  }
  is_audio = is_continuation && audio_start_idx != -1;
  if (is_audio) { audio_lengths = seq - audio_start_idx; }

  std::vector<float> text_logits;
  std::vector<std::vector<float>> audio_logits;
  // Head needs from the delay state: text samples only when delayed_lengths >
  // n_vq (else it is forced); inside audio it ranks just the 2 slot tokens,
  // outside audio it needs the full vocab head.
  auto need_full = [&] {
    return (!is_stopping && delayed_lengths > NV) && !is_audio;
  };
  auto need_2 = [&] {
    return (!is_stopping && delayed_lengths > NV) && is_audio;
  };
  // Step 0's heads run standalone on the prefill hidden; every later step's
  // heads are FUSED into the decode that produced its hidden (one commit).
  head_logits_(*cur_hn, need_full(), need_2(), text_logits, audio_logits);

  for (int t = 0; t < max_new_tokens; ++t) {
    if (kProf) { ++prof_steps; }
    // One text-decode perf event PER generated token (matches the LM /
    // realtime-vqa per-token decode timeline): brackets this step's token
    // selection + the fused decode forward that produces the next hidden.
    PerfAuxScope _dec(_session, kPerfLaneLLM, kGvidLlmDecode,
                      kPerfLlmDecodeBegin, 1);

    // ---- text channel (delay-pattern state machine) ------------------
    int next_text = _cfg.pad_token;
    if (!is_stopping && delayed_lengths < NV) {
      next_text = _cfg.audio_delay_slot;
    } else if (!is_stopping && delayed_lengths == NV) {
      next_text = _cfg.audio_end;
      is_audio = false;
    } else if (!is_stopping && delayed_lengths > NV) {
      if (is_audio) {
        // keep only {gen_slot, delay_slot}; t==0 also drops delay_slot.
        float bv = kNegInf;
        int best = -1;
        const int cand[2] = {_cfg.audio_gen_slot, _cfg.audio_delay_slot};
        for (int id : cand) {
          if (t == 0 && id == _cfg.audio_delay_slot) { continue; }
          if (text_logits[(std::size_t)id] > bv) {
            bv = text_logits[(std::size_t)id];
            best = id;
          }
        }
        next_text = best;
      } else {
        // Sample the free text token, excluding the control slots ({pad,
        // gen_slot, delay_slot, audio_end} + im_end until past the delay
        // window). temp<=0 reproduces the original argmax (lowest id on ties).
        std::vector<int> blk = {_cfg.pad_token, _cfg.audio_gen_slot,
                                _cfg.audio_delay_slot, _cfg.audio_end};
        if (t <= NV) { blk.push_back(_cfg.im_end); }
        const int best = sample_logits_(
            text_logits.data(), _cfg.vocab, text_sp, blk,
            text_rep ? &text_seen : nullptr, samp_cand, samp_prob, rng);
        next_text = best;
        if (text_rep && best >= 0) { text_seen.insert(best); }
      }
    }

    if (next_text == _cfg.audio_start) { is_audio = true; }
    if (next_text == _cfg.im_end) { is_stopping = true; }

    // ---- audio channels (per-codebook delay activation) --------------
    std::vector<std::int32_t> next_audio((std::size_t)NV, _cfg.audio_pad_code);
    for (int cb = 0; cb < NV; ++cb) {
      const bool pre_audio = audio_lengths > cb;
      const bool post_audio = (delayed_lengths == kSentinel)
                                  ? true
                                  : ((std::int64_t)cb > delayed_lengths - 1);
      if (!(pre_audio && post_audio)) { continue; }
      const auto& cl = audio_logits[(std::size_t)cb];
      const int best = sample_logits_(
          cl.data(), _cfg.audio_vocab, audio_sp, kNoBlock,
          audio_rep ? &audio_seen[(std::size_t)cb] : nullptr,
          samp_cand, samp_prob, rng);
      next_audio[(std::size_t)cb] = (best >= 0) ? best : 0;
      if (audio_rep && best >= 0) { audio_seen[(std::size_t)cb].insert(best); }
    }

    // ---- delay bookkeeping (order matches the reference) -------------
    if (next_text == _cfg.audio_start || next_text == _cfg.audio_gen_slot ||
        next_text == _cfg.audio_delay_slot) {
      ++audio_lengths;
    }
    if (next_text == _cfg.audio_end) { audio_lengths = 0; }
    if (delayed_lengths == kSentinel && next_text == _cfg.audio_delay_slot) {
      delayed_lengths = 0;
    }
    if (delayed_lengths != kSentinel) { ++delayed_lengths; }
    if (delayed_lengths > NV) { delayed_lengths = kSentinel; }

    std::vector<std::int32_t> row;
    row.reserve((std::size_t)(1 + NV));
    row.push_back(next_text);
    for (int cb = 0; cb < NV; ++cb) { row.push_back(next_audio[(std::size_t)cb]); }
    out.push_back(row);

    if (is_stopping) { break; }

    // Feed the new row back through the optimized qmv decode path, FUSING the
    // next step's heads (the state is now t+1) into the same command buffer.
    const bool nf1 = need_full(), n2_1 = need_2();
    const auto a0 = Clk::now();
    std::vector<std::vector<std::int32_t>> one(1, row);
    SharedBuffer xe = assemble_embeds_(one, 0, 1);
    const auto a1 = Clk::now();
    const std::function<void(ComputeEncoder&, const SharedBuffer&)> hook =
        [&](ComputeEncoder& enc, const SharedBuffer& d_hn) {
          dispatch_heads_(enc, d_hn, nf1);
        };
    cur_hn = _backbone->decode_embedding_hidden(cid, xe, -1, &hook);
    if (kProf) { t_fwd += ms(a1, Clk::now()); t_asm += ms(a0, a1); }
    if (cur_hn == nullptr) { break; }
    const auto rh0 = Clk::now();
    read_heads_(*cur_hn, nf1, n2_1, text_logits, audio_logits);
    if (kProf) { t_head += ms(rh0, Clk::now()); }
  }
  if (kProf && prof_steps > 0) {
    std::printf("[moss-prof] %d steps | prefill %.1f ms | per-step: "
                "decode-fwd %.1f ms + heads %.1f ms + embed %.2f ms "
                "(decode total %.1f ms/step)\n",
                prof_steps, t_prefill, t_fwd / prof_steps,
                t_head / prof_steps, t_asm / prof_steps,
                (t_fwd + t_head + t_asm) / prof_steps);
  }
  cm->release(cid);
  return out;
}

std::vector<MetalMossTtsModel::AudioMismatch>
MetalMossTtsModel::teacher_force_audio_mismatches(
    const std::vector<std::vector<std::int32_t>>& prompt,
    const std::vector<std::vector<std::int32_t>>& ref_rows)
{
  std::vector<AudioMismatch> out;
  const int NV = _cfg.n_vq;
  const int seq = (int)prompt.size();
  if (seq <= 0 || ref_rows.empty()) { return out; }
  ContextManager* cm = _backbone->context_manager();
  const ContextId cid = cm->acquire_root();
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();

  SharedBuffer x = assemble_embeds_(prompt, 0, seq);
  SharedBuffer prefill_hn = _backbone->forward_embeddings_hidden(cid, x, seq);
  if (prefill_hn.empty()) { cm->release(cid); return out; }
  const SharedBuffer* cur_hn = &prefill_hn;

  std::vector<float> text_logits;
  std::vector<std::vector<float>> audio_logits;
  for (int r = 0; r < (int)ref_rows.size(); ++r) {
    // Verification only inspects the audio heads; skip the text head entirely.
    head_logits_(*cur_hn, /*need_full_text=*/false, /*need_2slots=*/false,
                 text_logits, audio_logits);
    const auto& ref = ref_rows[(std::size_t)r];
    for (int cb = 0; cb < NV; ++cb) {
      const int ref_code = ref[(std::size_t)(1 + cb)];
      if (ref_code == _cfg.audio_pad_code) { continue; }   // inactive
      const auto& cl = audio_logits[(std::size_t)cb];
      float bv = kNegInf;
      int best = 0;
      for (int c = 0; c < _cfg.audio_vocab; ++c) {
        if (cl[(std::size_t)c] > bv) { bv = cl[(std::size_t)c]; best = c; }
      }
      if (best != ref_code) {
        out.push_back({r, cb, best, ref_code, bv,
                       cl[(std::size_t)ref_code]});
      }
    }
    std::vector<std::vector<std::int32_t>> one(1, ref);
    SharedBuffer xe = assemble_embeds_(one, 0, 1);
    cur_hn = _backbone->decode_embedding_hidden(cid, xe);
    if (cur_hn == nullptr) { break; }
  }
  cm->release(cid);
  return out;
}

}  // namespace vpipe::genai
