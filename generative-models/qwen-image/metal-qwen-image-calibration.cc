#include "generative-models/qwen-image/metal-qwen-image-calibration.h"

#include "common/vpipe-format.h"
#include "generative-models/context-manager.h"
#include "generative-models/krea2/flow-sampler.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/qwen-image/metal-qwen-image-transformer.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/tokenizer.h"
#include "interfaces/session-context-intf.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>

#include <sys/sysctl.h>

namespace vpipe {
namespace genai {

namespace {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

// The QwenImageEditPlus text-only template; the 64-token system prefix is
// dropped (matches the M3/M6/M7 conditioning path).
constexpr const char* kPrefix =
    "<|im_start|>system\nDescribe the key features of the input image (color, "
    "shape, size, texture, objects, background), then explain how the user's "
    "text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the "
    "original input where appropriate.<|im_end|>\n<|im_start|>user\n";
constexpr const char* kSuffix = "<|im_end|>\n<|im_start|>assistant\n";
constexpr int kDrop = 64;

std::uint16_t f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Total physical RAM (bytes), or 0 if unknown.
std::size_t physical_ram_()
{
  std::uint64_t mem = 0;
  std::size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) { return 0; }
  return (std::size_t)mem;
}

// Sum of the .safetensors bytes in a dir -- a close proxy for the wired f16
// footprint when preloaded (bf16 source -> f16 = same 2 bytes/elt).
std::size_t weights_bytes_(const std::filesystem::path& dir)
{
  namespace fs = std::filesystem;
  std::size_t total = 0;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(dir, ec), end; it != end;
       it.increment(ec)) {
    if (it->is_regular_file(ec) &&
        it->path().extension() == ".safetensors") {
      total += (std::size_t)it->file_size(ec);
    }
  }
  return total;
}

// In-place throttled progress bar (mirrors the Krea-2 collector / the
// model-quantizer's quant_progress_): redraw on a carriage-return only when the
// integer percentage changes, so long calibrations don't flood the log.
void calib_progress_(UiTextStream* bar, const char* tag, int done, int total,
                     int& last_pct)
{
  if (bar == nullptr || total <= 0) { return; }
  int pct = (int)((long)done * 100 / total);
  if (pct < 0) { pct = 0; } else if (pct > 100) { pct = 100; }
  if (pct == last_pct) { return; }
  last_pct = pct;
  constexpr int W = 24;
  const int fill = pct * W / 100;
  std::string b((std::size_t)fill, '#');
  b += std::string((std::size_t)(W - fill), '-');
  std::string line = fmt("\r[{}] {}% {} ({}/{})", b, pct, tag, done, total)();
  while (line.size() < 64) { line += ' '; }   // wipe stale tail
  bar->write(line);
}

// Qwen2.5-VL text backbone config (the M3/M7 encoder).
MetalQwenModel::Config encoder_config_()
{
  MetalQwenModel::Config c;
  c.n_layers = 28; c.hidden = 3584; c.n_heads = 28; c.n_kv_heads = 4;
  c.head_dim = 128; c.ffn_inner = 18944; c.vocab = 152064;
  c.rope_theta = 1.0e6f; c.rms_eps = 1e-6f; c.rotary_dim = 128;
  c.full_attn_interval = 1; c.tie_embeddings = false; c.use_bf16 = true;
  c.dense = true; c.zero_centered_norm = false; c.attn_output_gate = false;
  c.qk_norm = false; c.attention_bias = true; c.backbone_only = true;
  c.weight_prefix = ""; c.model_seg = "model."; c.max_seq = 1024;
  c.page_tokens = 256;
  return c;
}

// Special-token-aware encode: split at <|im_start|>/<|im_end|>, BPE each text
// run, splice the markers' ids (matches the HF fast tokenizer + M7 stage).
std::vector<std::int32_t>
encode_specials_(const Tokenizer& tok, const std::string& text)
{
  static const char* kMarkers[] = {"<|im_start|>", "<|im_end|>"};
  std::vector<std::int32_t> out;
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t best = std::string::npos;
    int which = -1;
    for (int mi = 0; mi < 2; ++mi) {
      const std::size_t f = text.find(kMarkers[mi], pos);
      if (f != std::string::npos && (best == std::string::npos || f < best)) {
        best = f; which = mi;
      }
    }
    if (which < 0) {
      const auto seg = tok.encode(text.substr(pos));
      out.insert(out.end(), seg.begin(), seg.end());
      break;
    }
    if (best > pos) {
      const auto seg = tok.encode(text.substr(pos, best - pos));
      out.insert(out.end(), seg.begin(), seg.end());
    }
    const std::int32_t sid = tok.special_token_id(kMarkers[which]);
    if (sid >= 0) { out.push_back(sid); }
    pos = best + std::strlen(kMarkers[which]);
  }
  return out;
}

}  // namespace

bool
collect_qwen_image_calibration(
    MetalCompute* mc, const std::string& model_root,
    const std::vector<std::string>& prompts, int steps, int height, int width,
    std::uint64_t seed, const std::string& out_dir, std::string* err,
    const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (mc == nullptr) { return fail("qie-calib: null metal-compute"); }
  if (steps <= 0) { steps = 8; }
  if ((height % 16) != 0 || (width % 16) != 0) {
    return fail("qie-calib: height/width must be multiples of 16");
  }
  const std::string enc_dir = (fs::path(model_root) / "text_encoder").string();
  const std::string dit_dir = (fs::path(model_root) / "transformer").string();
  std::string tok_path = (fs::path(model_root) / "processor" /
                          "tokenizer.json").string();
  if (!fs::exists(tok_path)) {
    tok_path = (fs::path(model_root) / "tokenizer" / "tokenizer.json").string();
  }
  const int H = 3584, IC = 64;
  const int gh = height / 16, gw = width / 16, img_seq = gh * gw;
  const int lh = height / 8, lw = width / 8;
  (void)lh; (void)lw;

  auto tok = Tokenizer::from_huggingface_json(tok_path, nullptr);
  if (!tok) { return fail("qie-calib: tokenizer load failed: " + tok_path); }

  auto* sess = mc->session();
  std::unique_ptr<UiTextStream> bar =
      sess ? sess->open_text_stream() : std::unique_ptr<UiTextStream>();
  int pct = -1;
  if (sess) {
    sess->log_debug(fmt(
        "qie-calib: {} prompts x {} steps @ {}x{} (img_seq {}, seed {}) -> {}",
        prompts.size(), steps, width, height, img_seq, seed, out_dir));
  }

  // ---- Phase 1: encode every prompt -> cached txt conditioning [n_real, H]. --
  std::vector<SharedBuffer> txt_cache;
  std::vector<int> nreal_cache;
  {
    auto enc = MetalQwenModel::load(enc_dir, mc, encoder_config_());
    if (!enc) { return fail("qie-calib: encoder load failed: " + enc_dir); }
    auto wts = MetalLlamaWeights::open_model(enc_dir);
    if (!wts.has_value()) { return fail("qie-calib: encoder weights"); }
    SharedBuffer emb = wts->load("model.embed_tokens.weight", mc);
    if (emb.empty()) { return fail("qie-calib: encoder embed table"); }
    std::vector<float> fnorm(H, 1.0f);
    {
      SharedBuffer nw = wts->load("model.norm.weight", mc);
      if (!nw.empty()) {
        const auto* p = static_cast<const std::uint16_t*>(nw.contents());
        for (int h = 0; h < H; ++h) { fnorm[h] = bf16_to_f32_(p[h]); }
      }
    }
    const int NL = 28;
    for (std::size_t pi = 0; pi < prompts.size(); ++pi) {
      const std::string& prompt = prompts[pi];
      if (stop()) { return fail("qie-calib: stopped"); }
      calib_progress_(bar.get(), "encode", (int)pi + 1, (int)prompts.size(),
                      pct);
      const std::vector<std::int32_t> ids =
          encode_specials_(*tok, std::string(kPrefix) + prompt + kSuffix);
      if ((int)ids.size() <= kDrop) {
        if (sess) { sess->log_debug(fmt("qie-calib: prompt {} too short "
                                        "({} <= {} dropped), skipped",
                                        pi, ids.size(), kDrop)); }
        continue;
      }
      const int n = (int)ids.size();
      const int n_real = n - kDrop;
      SharedBuffer x = mc->make_shared_buffer((std::size_t)n * H * 2);
      const auto* tbl = static_cast<const std::uint8_t*>(emb.contents());
      auto* xb = static_cast<std::uint8_t*>(x.contents());
      const std::size_t vocab = emb.byte_size() / ((std::size_t)H * 2);
      bool ok = true;
      for (int i = 0; i < n; ++i) {
        const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
        if (id >= vocab) { ok = false; break; }
        std::memcpy(xb + (std::size_t)i * H * 2,
                    tbl + (std::size_t)id * H * 2, (std::size_t)H * 2);
      }
      if (!ok) { continue; }
      ContextManager* cm = enc->context_manager();
      const ContextId cid = cm->acquire_root();
      SharedBuffer taps =
          enc->forward_embeddings_taps(cid, x, n, std::vector<int>{NL - 1});
      cm->release(cid);
      if (taps.empty()) {
        if (sess) { sess->log_debug(fmt("qie-calib: prompt {} encode produced "
                                        "no taps, skipped", pi)); }
        continue;
      }
      // Drop prefix + host final-RMSNorm -> [n_real, H] bf16.
      SharedBuffer txt = mc->make_shared_buffer((std::size_t)n_real * H * 2);
      const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
      auto* op = static_cast<std::uint16_t*>(txt.contents());
      for (int p = 0; p < n_real; ++p) {
        const auto* row = tp + (std::size_t)(p + kDrop) * H;
        double ss = 0.0;
        for (int h = 0; h < H; ++h) {
          const double v = bf16_to_f32_(row[h]); ss += v * v;
        }
        const double inv = 1.0 / std::sqrt(ss / (double)H + 1e-6);
        for (int h = 0; h < H; ++h) {
          op[(std::size_t)p * H + h] =
              f32_to_bf16_((float)(bf16_to_f32_(row[h]) * inv * fnorm[h]));
        }
      }
      txt_cache.push_back(std::move(txt));
      nreal_cache.push_back(n_real);
      if (sess) {
        sess->log_debug(fmt("qie-calib: encoded prompt {}/{} (n_real={}) '{}'",
                            pi + 1, prompts.size(), n_real, prompt));
      }
    }
    // encoder + embed table freed here (scope close) before the DiT loads.
  }
  if (txt_cache.empty()) { return fail("qie-calib: no prompts encoded"); }
  if (sess) {
    sess->log_debug(fmt(
        "qie-calib: encoded {} prompts, encoder freed; DiT denoising over {} "
        "forward passes", txt_cache.size(), txt_cache.size() * (std::size_t)steps));
  }

  // ---- Phase 2: DiT denoise trajectory with calib taps. ---------------------
  // Pick the DiT load strategy by available memory. PRELOAD (all 60 blocks
  // resident) is far faster -- weights are read once and reused across every
  // forward -- but needs the full 20B bf16 DiT (~40 GB) + encoder to fit; STREAM
  // loads each block per forward (re-read from the mmap, ~1 block resident),
  // essential on 16 GB boxes. Preload only when RAM comfortably clears the peak
  // footprint (DiT + encoder wired + scratch/OS headroom). The encoder is freed
  // above, but count it: page-cache + the just-touched pages still linger.
  // VPIPE_QIE_CALIB_STREAM=0/1 forces the choice.
  bool stream_blocks;
  {
    const std::size_t dit_b = weights_bytes_(dit_dir);
    const std::size_t enc_b = weights_bytes_(enc_dir);
    const std::size_t ram = physical_ram_();
    const std::size_t need = dit_b + enc_b + (8ull << 30);   // +8 GB headroom
    stream_blocks = (ram == 0) || (ram < need);
    if (const char* e = std::getenv("VPIPE_QIE_CALIB_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    if (mc->session() != nullptr) {
      mc->session()->log_debug(fmt(
          "qie-calib: dit {} GB + enc {} GB + 8 GB headroom vs {} GB RAM -> {}",
          dit_b >> 30, enc_b >> 30, ram >> 30,
          stream_blocks ? "STREAM blocks" : "PRELOAD (fast)"));
    }
  }
  // Pinned-prefix: in streaming mode, pin as many leading blocks as fit in 60%
  // of physical RAM (pinned + running) so they are read once and reused across
  // every forward, streaming only the tail. VPIPE_QIE_CALIB_PIN_FRAC overrides.
  double pin_frac = stream_blocks ? 0.60 : 0.0;
  if (const char* e = std::getenv("VPIPE_QIE_CALIB_PIN_FRAC")) {
    pin_frac = std::atof(e);
  }
  MetalQwenImageTransformer::Config qcfg;
  auto dit =
      MetalQwenImageTransformer::load(dit_dir, mc, qcfg, stream_blocks, pin_frac);
  if (!dit) { return fail("qie-calib: DiT load failed: " + dit_dir); }
  dit->set_stream_stop(stop);   // honor a pipeline stop within ~one block
  if (sess && stream_blocks) {
    sess->log_debug(fmt(
        "qie-calib: pinned {} of {} DiT blocks resident ({}% RAM budget), "
        "streaming the rest", dit->pinned_blocks(), qcfg.n_layers,
        (int)(pin_frac * 100)));
  }

  // Dynamic-shift sigmas (match inference; M2-verified).
  FlowSchedulerSpec sched;
  sched.steps = steps; sched.dynamic_shift = true; sched.base_shift = 0.5;
  sched.max_shift = 0.9; sched.shift_terminal = 0.02; sched.base_seq = 256;
  sched.max_seq = 8192; sched.num_train = 1000; sched.shift_type = "exponential";
  const std::vector<double> sig = sched.sigmas(img_seq);
  const int S = (int)sig.size() - 1;

  // Reference-conditioned (image-edit) calibration. QIE is an image-EDIT model:
  // in real use the DiT ALWAYS attends to one or more clean reference latents,
  // so the img-stream linears see clean reference tokens (sigma 0) alongside the
  // noisy generated tokens and the joint attention runs a longer sequence.
  // Calibrating on pure text-to-image (no reference) therefore samples an
  // off-distribution activation set. We roll the PREVIOUS prompt's fully
  // denoised latent in as the reference for the next prompt -- cost-neutral, as
  // it reuses each trajectory's own clean output as a plausible "previously
  // generated" image (exactly the packed [img_seq, IC] DiT-input layout a
  // RefImage wants). Prompt 0 runs reference-less to bootstrap (and covers the
  // rare no-reference path). VPIPE_QIE_CALIB_EDIT_FRAC in [0,1] selects what
  // fraction of prompts attach the reference (Bresenham selection); QIE defaults
  // to 1.0 (always edit).
  double edit_frac = 1.0;
  if (const char* e = std::getenv("VPIPE_QIE_CALIB_EDIT_FRAC")) {
    edit_frac = std::atof(e);
    edit_frac = edit_frac < 0.0 ? 0.0 : (edit_frac > 1.0 ? 1.0 : edit_frac);
  }
  SharedBuffer ref_buf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  bool ref_ready = false;   // ref_buf holds a valid generated latent
  double edit_acc = 0.0;
  int edit_used = 0;
  if (sess) {
    sess->log_debug(fmt(
        "qie-calib: image-edit reference calibration frac {} (rolling each "
        "prompt's generated latent in as the next prompt's reference)",
        edit_frac));
  }

  dit->calib_begin();
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  pct = -1;
  const int total_fwd = (int)txt_cache.size() * S;
  for (std::size_t e = 0; e < txt_cache.size(); ++e) {
    if (stop()) {
      if (bar) { bar->end(); }
      dit->calib_end();
      return fail("qie-calib: stopped");
    }
    // Attach the rolling reference (a previous generation) for this prompt?
    std::vector<MetalQwenImageTransformer::RefImage> refs;
    if (ref_ready && edit_frac > 0.0) {
      edit_acc += edit_frac;
      if (edit_acc >= 1.0) {
        edit_acc -= 1.0;
        MetalQwenImageTransformer::RefImage r;
        r.latents = ref_buf.subview(0, ref_buf.byte_size());
        r.seq = img_seq; r.grid_h = gh; r.grid_w = gw;
        refs.push_back(std::move(r));
        ++edit_used;
      }
    }
    if (sess) {
      sess->log_debug(fmt("qie-calib: denoise prompt {}/{} (n_real={}{})", e + 1,
                          txt_cache.size(), nreal_cache[e],
                          refs.empty() ? "" : ", +1 reference"));
    }
    std::mt19937_64 rng(seed + e);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> packed((std::size_t)img_seq * IC);
    for (auto& v : packed) { v = nd(rng); }
    for (int i = 0; i < S; ++i) {
      auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
      for (std::size_t k = 0; k < packed.size(); ++k) {
        lb[k] = f32_to_bf16_(packed[k]);
      }
      SharedBuffer vel = dit->forward(latbuf, img_seq, txt_cache[e],
                                      nreal_cache[e], gh, gw, (float)sig[i],
                                      refs);
      if (vel.empty()) {
        if (bar) { bar->end(); }
        dit->calib_end();
        // forward() returns empty when it bailed on the per-block stop hook.
        return fail(stop() ? "qie-calib: stopped" : "qie-calib: DiT step");
      }
      const auto* vp = static_cast<const std::uint16_t*>(vel.contents());
      const double dt = sig[(std::size_t)i + 1] - sig[(std::size_t)i];
      for (std::size_t k = 0; k < packed.size(); ++k) {
        packed[k] += (float)dt * bf16_to_f32_(vp[k]);
      }
      calib_progress_(bar.get(), "denoise", (int)e * S + i + 1, total_fwd, pct);
      if (sess) {
        sess->log_debug(fmt("qie-calib:   step {}/{} sigma {}", i + 1, S,
                            (float)sig[(std::size_t)i]));
      }
    }
    // Snapshot this prompt's now-clean generated latent as the reference for the
    // subsequent prompts (rolling; the image-edit calibration path above).
    if (edit_frac > 0.0) {
      auto* rb = static_cast<std::uint16_t*>(ref_buf.contents());
      for (std::size_t k = 0; k < packed.size(); ++k) {
        rb[k] = f32_to_bf16_(packed[k]);
      }
      ref_ready = true;
    }
  }
  if (bar) { bar->end(); }
  const auto stats = dit->calib_stats();
  dit->calib_end();
  if (sess) {
    for (const auto& kv : stats) {
      float m = 0.0f;
      for (float v : kv.second) { if (v > m) { m = v; } }
      sess->log_debug(fmt("qie-calib: tap '{}' abs-max {} over {} values",
                          kv.first, m, kv.second.size()));
    }
  }

  // ---- Write one raw-f32 file per group. ------------------------------------
  std::error_code ec;
  fs::create_directories(out_dir, ec);
  for (const auto& kv : stats) {
    std::ofstream out(fs::path(out_dir) / (kv.first + ".f32"), std::ios::binary);
    if (!out) { return fail("qie-calib: cannot write " + kv.first); }
    out.write(reinterpret_cast<const char*>(kv.second.data()),
              (std::streamsize)kv.second.size() * 4);
  }
  if (sess) {
    sess->log_debug(fmt(
        "qie-calib: wrote {} calib groups to {} ({} of {} prompts calibrated "
        "with an image-edit reference)", stats.size(), out_dir, edit_used,
        txt_cache.size()));
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
