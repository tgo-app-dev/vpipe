#include "generative-models/krea2/metal-krea2-calibration.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/vpipe-format.h"
#include "generative-models/context-manager.h"
#include "generative-models/krea2/metal-krea2-transformer.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/tokenizer.h"
#include "interfaces/session-context-intf.h"

#include <sys/sysctl.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

constexpr const char* kPrefix =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n";
constexpr const char* kSuffix = "<|im_end|>\n<|im_start|>assistant\n";
constexpr int kDropPrefix = 34;
const int kSelectLayers[12] = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35};

// The Krea-2 text encoder config (dense Qwen3-VL, raw bf16) -- matches the
// text-to-image stage's encoder_config_.
MetalQwenModel::Config
encoder_config_()
{
  MetalQwenModel::Config c;
  c.n_layers = 36; c.hidden = 2560; c.n_heads = 32; c.n_kv_heads = 8;
  c.head_dim = 128; c.ffn_inner = 9728; c.vocab = 151936;
  c.rope_theta = 5.0e6f; c.rms_eps = 1e-6f; c.rotary_dim = 128;
  c.full_attn_interval = 1; c.tie_embeddings = true; c.use_bf16 = true;
  c.dense = true; c.zero_centered_norm = false; c.attn_output_gate = false;
  c.backbone_only = true; c.weight_prefix = "language_model.";
  c.model_seg = ""; c.max_seq = 1024; c.page_tokens = 256;
  return c;
}

// Special-token-aware encode (mirrors the text-to-image stage's helper).
std::vector<std::int32_t>
encode_with_specials_(const Tokenizer& tok, const std::string& text)
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
      const std::vector<std::int32_t> seg = tok.encode(text.substr(pos));
      out.insert(out.end(), seg.begin(), seg.end());
      break;
    }
    if (best > pos) {
      const std::vector<std::int32_t> seg = tok.encode(text.substr(pos, best - pos));
      out.insert(out.end(), seg.begin(), seg.end());
    }
    const std::int32_t sid = tok.special_token_id(kMarkers[which]);
    if (sid >= 0) { out.push_back(sid); }
    pos = best + std::strlen(kMarkers[which]);
  }
  return out;
}

bool
write_calib_(const std::string& path,
             const std::vector<std::vector<float>>& stat)
{
  std::ofstream out(path, std::ios::binary);
  if (!out) { return false; }
  for (const auto& row : stat) {
    out.write(reinterpret_cast<const char*>(row.data()),
              (std::streamsize)row.size() * 4);
  }
  return (bool)out;
}

// In-place throttled progress bar (mirrors model-quantizer's quant_progress_):
// redraws on a carriage-return only when the integer percentage changes.
void
calib_progress_(UiTextStream* bar, const char* tag, int done, int total,
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

// Total physical RAM (bytes), or 0 if unknown.
std::size_t
physical_ram_()
{
  std::uint64_t mem = 0;
  std::size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) { return 0; }
  return (std::size_t)mem;
}

// Sum of the .safetensors bytes in a model dir -- a close proxy for the wired
// f16 footprint when the whole thing is preloaded (bf16 source -> f16 = same 2
// bytes/elt).
std::size_t
weights_bytes_(const std::filesystem::path& dir)
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

}  // namespace

std::vector<std::string>
default_dit_calibration_prompts()
{
  return {
      "a red fox standing in fresh snow",
      "a bustling city street at night with neon signs",
      "a bowl of ripe strawberries on a wooden table",
      "a snow-capped mountain range under a clear blue sky",
      "a portrait of an old fisherman with a weathered face",
      "an astronaut floating above the earth",
      "a watercolor painting of a quiet village",
      "a close-up of a dew-covered spider web at dawn",
      "a golden retriever puppy playing in a green meadow",
      "a futuristic sports car on a rain-slicked road",
      "a plate of sushi with soy sauce and wasabi",
      "an ancient stone temple overgrown with vines",
      "a hot air balloon drifting over rolling hills",
      "a steaming cup of coffee next to an open book",
      "a coral reef teeming with colorful tropical fish",
      "a lone lighthouse on a rocky cliff during a storm",
      "an abstract composition of bold geometric shapes",
      "a field of sunflowers stretching to the horizon",
      "a vintage typewriter on a cluttered desk",
      "a majestic eagle soaring over a canyon",
      "a cozy cabin in a pine forest with warm light",
      "a busy farmers market with fresh produce",
      "a ballerina mid-leap on a dim stage",
      "a bridge spanning a misty river at sunrise",
  };
}

bool
collect_dit_calibration(MetalCompute* mc, const std::string& model_root,
                        const std::vector<std::string>& prompts, int steps,
                        int height, int width, std::uint64_t seed,
                        const std::string& out_dir, std::string* err,
                        const std::function<bool()>& stop)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (mc == nullptr) { return fail("calib: null metal-compute"); }
  if (prompts.empty()) { return fail("calib: empty prompt corpus"); }
  if (steps <= 0) { steps = 8; }
  if (height % 16 != 0 || width % 16 != 0) {
    return fail("calib: height/width must be multiples of 16");
  }
  namespace fs = std::filesystem;

  const std::string enc_dir = (fs::path(model_root) / "text_encoder").string();
  const std::string dit_dir = (fs::path(model_root) / "transformer").string();
  const std::string tok_path =
      (fs::path(model_root) / "tokenizer" / "tokenizer.json").string();

  auto tok = Tokenizer::from_huggingface_json(tok_path, mc->session());
  if (!tok) { return fail("calib: tokenizer load failed: " + tok_path); }
  auto enc = MetalQwenModel::load(enc_dir, mc, encoder_config_());
  if (!enc) { return fail("calib: text encoder load failed: " + enc_dir); }
  SharedBuffer embed;
  {
    auto wts = MetalLlamaWeights::open_model(enc_dir);
    if (wts.has_value()) {
      embed = wts->load("language_model.embed_tokens.weight", mc);
    }
  }
  if (embed.empty()) { return fail("calib: encoder embed table load failed"); }

  // Pick the DiT load strategy by available memory. PRELOAD (all 28 blocks
  // resident) is far faster -- the weights are read once and reused across every
  // forward -- but needs the full bf16 DiT + encoder to fit; STREAM loads each
  // block per forward (re-read from disk, ~1 block resident), essential on 16 GB
  // boxes. Preload only when physical RAM comfortably clears the peak footprint
  // (DiT + encoder wired, + headroom for scratch/OS/page-cache). Env
  // VPIPE_KREA2_CALIB_STREAM=0/1 forces the choice.
  bool stream_blocks;
  {
    const std::size_t dit_b = weights_bytes_(dit_dir);
    const std::size_t enc_b = weights_bytes_(enc_dir);
    const std::size_t ram = physical_ram_();
    const std::size_t need = dit_b + enc_b + (8ull << 30);   // +8 GB headroom
    stream_blocks = (ram == 0) || (ram < need);
    if (const char* e = std::getenv("VPIPE_KREA2_CALIB_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    if (mc->session() != nullptr) {
      mc->session()->log_debug(fmt(
          "DiT calib: dit {} GB + enc {} GB + 8 GB headroom vs {} GB RAM -> {}",
          dit_b >> 30, enc_b >> 30, ram >> 30,
          stream_blocks ? "STREAM blocks" : "PRELOAD (fast)"));
    }
  }
  auto dit = MetalKrea2Transformer::load(dit_dir, mc,
                                         MetalKrea2Transformer::Config{},
                                         stream_blocks);
  if (!dit) { return fail("calib: DiT load failed: " + dit_dir); }

  const int EH = 2560, NL = 12;
  const int lh = height / 8, lw = width / 8;
  const int gh = height / 16, gw = width / 16;
  const int img_seq = gh * gw, IC = 64;

  // Turbo sigma schedule (distilled mu=1.15).
  std::vector<double> sig((std::size_t)steps + 1);
  const double mu = 1.15, emu = std::exp(mu);
  for (int i = 0; i < steps; ++i) {
    const double raw = (steps == 1) ? 1.0
        : 1.0 + (double)i * ((1.0 / (double)steps) - 1.0) / (double)(steps - 1);
    sig[(std::size_t)i] = emu / (emu + (1.0 / raw - 1.0));
  }
  sig[(std::size_t)steps] = 0.0;

  std::vector<int> tap_layers;
  for (int k : kSelectLayers) { tap_layers.push_back(k - 1); }

  // prompt -> fused text [n_real, hidden] (encoder 12-layer tap + text fusion).
  auto encode = [&](const std::string& prompt, int& n_real) -> SharedBuffer {
    std::vector<std::int32_t> ids =
        encode_with_specials_(*tok, std::string(kPrefix) + prompt + kSuffix);
    if ((int)ids.size() <= kDropPrefix) { return {}; }
    const int n = (int)ids.size();
    n_real = n - kDropPrefix;
    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
    const auto* tbl = static_cast<const std::uint8_t*>(embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    const std::size_t vocab = embed.byte_size() / ((std::size_t)EH * 2);
    for (int i = 0; i < n; ++i) {
      const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
      if (id >= vocab) { return {}; }
      std::memcpy(xb + (std::size_t)i * EH * 2,
                  tbl + (std::size_t)id * EH * 2, (std::size_t)EH * 2);
    }
    ContextManager* cm = enc->context_manager();
    const ContextId cid = cm->acquire_root();
    SharedBuffer taps = enc->forward_embeddings_taps(cid, x, n, tap_layers);
    cm->release(cid);
    if (taps.empty()) { return {}; }
    SharedBuffer ehs = mc->make_shared_buffer((std::size_t)n_real * NL * EH * 2);
    const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
    auto* ep = static_cast<_Float16*>(ehs.contents());
    for (int p = 0; p < n_real; ++p) {
      for (int j = 0; j < NL; ++j) {
        const std::size_t src = ((std::size_t)j * n + (kDropPrefix + p)) * EH;
        const std::size_t dst = ((std::size_t)p * NL + j) * EH;
        for (int h = 0; h < EH; ++h) {
          std::uint32_t u = (std::uint32_t)tp[src + h] << 16;
          float f; std::memcpy(&f, &u, 4);
          ep[dst + h] = (_Float16)f;
        }
      }
    }
    return dit->forward_text(ehs, n_real);
  };

  auto* sess = mc->session();
  std::unique_ptr<UiTextStream> bar =
      sess ? sess->open_text_stream() : std::unique_ptr<UiTextStream>();
  int pct = -1;
  if (sess) {
    sess->log_debug(fmt(
        "DiT calib: {} prompts x {} steps @ {}x{} (img_seq {}, streamed DiT), "
        "seed {}", prompts.size(), steps, width, height, img_seq, seed));
  }

  // Phase 1: encode every prompt (encoder + text-fusion tower resident) and
  // cache the fused text. The fused vectors are tiny ([n_real, hidden] f16, a
  // few hundred KB each), so all prompts fit; then we free the ~8 GB encoder so
  // the streamed DiT forward runs with headroom on 16 GB boxes.
  std::vector<SharedBuffer> fused_cache;
  std::vector<int> nreal_cache;
  std::vector<std::size_t> prompt_of;
  for (std::size_t pi = 0; pi < prompts.size(); ++pi) {
    if (stop()) { return fail("calib: stopped by request"); }
    int n_real = 0;
    SharedBuffer fused = encode(prompts[pi], n_real);
    calib_progress_(bar.get(), "encode", (int)pi + 1, (int)prompts.size(), pct);
    if (fused.empty()) {
      if (sess) { sess->log_debug(fmt("DiT calib: prompt {} encode failed, "
                                      "skipped", pi)); }
      continue;
    }
    if (sess) {
      sess->log_debug(fmt("DiT calib: encoded prompt {}/{} (n_real={}) '{}'",
                          pi + 1, prompts.size(), n_real, prompts[pi]));
    }
    fused_cache.push_back(std::move(fused));
    nreal_cache.push_back(n_real);
    prompt_of.push_back(pi);
  }
  enc.reset();     // free the text encoder (+ its context/KV)
  embed = {};      // free the embed table
  if (sess) {
    sess->log_debug(fmt("DiT calib: encoded {} prompts, encoder freed; "
                        "streamed denoising over {} forward passes",
                        fused_cache.size(), fused_cache.size() * (std::size_t)steps));
  }
  if (fused_cache.empty()) { return fail("calib: no prompt encoded successfully"); }

  // Phase 2: real denoising trajectory per cached prompt, tapping col-absmax per
  // block. forward_dit either preloads or streams the blocks per the decision
  // above; set_stream_stop only affects the streaming path.
  dit->calib_begin();
  dit->set_stream_stop(stop);   // forward_dit polls this per streamed block
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  std::vector<float> packed((std::size_t)img_seq * IC);
  int used = 0;
  pct = -1;
  const int total_fwd = (int)fused_cache.size() * steps;
  for (std::size_t e = 0; e < fused_cache.size(); ++e) {
    if (stop()) {
      if (bar) { bar->end(); }
      dit->calib_end();
      return fail("calib: stopped by request");
    }
    const int n_real = nreal_cache[e];
    if (sess) {
      sess->log_debug(fmt("DiT calib: denoise prompt {}/{} (n_real={})", e + 1,
                          fused_cache.size(), n_real));
    }
    // Independent noise per prompt so the calibration covers diverse latents.
    std::mt19937_64 rng(seed + prompt_of[e]);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
    for (int i = 0; i < steps; ++i) {
      // Per-step stop (covers the PRELOAD path, where forward_dit runs to
      // completion without the per-block stream-stop bail).
      if (stop()) {
        if (bar) { bar->end(); }
        dit->calib_end();
        return fail("calib: stopped by request");
      }
      auto* lb = static_cast<_Float16*>(latbuf.contents());
      for (std::size_t k = 0; k < packed.size(); ++k) { lb[k] = (_Float16)packed[k]; }
      SharedBuffer vel = dit->forward_dit(fused_cache[e], n_real, latbuf, img_seq,
                                          gh, gw, (float)sig[(std::size_t)i], -1);
      if (vel.empty()) {
        if (bar) { bar->end(); }
        dit->calib_end();
        // forward_dit returns empty when it bailed on the stop hook mid-block.
        return fail(stop() ? "calib: stopped by request"
                           : "calib: forward_dit failed");
      }
      const auto* vp = static_cast<const _Float16*>(vel.contents());
      const double dt = sig[(std::size_t)i + 1] - sig[(std::size_t)i];
      for (std::size_t k = 0; k < packed.size(); ++k) {
        packed[k] += (float)(dt * (double)(float)vp[k]);
      }
      calib_progress_(bar.get(), "denoise", (int)e * steps + i + 1, total_fwd,
                      pct);
      if (sess) {
        sess->log_debug(fmt("DiT calib:   step {}/{} sigma {}", i + 1, steps,
                            (float)sig[(std::size_t)i]));
      }
    }
    ++used;
  }
  if (bar) { bar->end(); }
  (void)lh; (void)lw;

  const auto cqkv = dit->calib_qkv();
  const auto co = dit->calib_o();
  const auto cgu = dit->calib_gateup();
  const auto cdn = dit->calib_down();
  dit->calib_end();
  if (used == 0) { return fail("calib: no prompt encoded successfully"); }
  if (sess) {
    auto amax = [](const std::vector<std::vector<float>>& s) {
      float m = 0.0f;
      for (const auto& row : s) { for (float v : row) { if (v > m) { m = v; } } }
      return m;
    };
    sess->log_debug(fmt(
        "DiT calib: tap abs-max over {} layers -> qkv {} o {} gateup {} down {}",
        cqkv.size(), amax(cqkv), amax(co), amax(cgu), amax(cdn)));
  }

  std::error_code ec;
  fs::create_directories(out_dir, ec);
  if (!write_calib_((fs::path(out_dir) / "calib_qkv.f32").string(), cqkv) ||
      !write_calib_((fs::path(out_dir) / "calib_o.f32").string(), co) ||
      !write_calib_((fs::path(out_dir) / "calib_gateup.f32").string(), cgu) ||
      !write_calib_((fs::path(out_dir) / "calib_down.f32").string(), cdn)) {
    return fail("calib: write failed to " + out_dir);
  }
  if (mc->session() != nullptr) {
    mc->session()->log_normal(fmt(
        "DiT calib: {} prompts x {} steps -> calib_*.f32 in {}", used, steps,
        out_dir));
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
