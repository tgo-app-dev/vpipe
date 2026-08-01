#include "generative-models/boogu/metal-boogu-calibration.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/boogu/metal-boogu-transformer.h"
#include "generative-models/context-manager.h"
#include "generative-models/krea2/flow-sampler.h"
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
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

inline std::uint16_t f32_to_bf16_(float f)
{
  std::uint32_t u; std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

// The Boogu mllm: a stock Qwen3VLForConditionalGeneration (the 10B ships an 8B
// Qwen3-VL: 36 layers, hidden 4096, 32q/8kv GQA head_dim 128, rope theta 5e6,
// UNTIED embeddings), checkpoint-wrapped as `model.language_model.` /
// `model.visual.` like Mage-Flow. Sized from mllm/config.json's text_config so
// one path serves any Boogu size. Mirrors the diffusion-conditioner's
// encoder_config_boogu_.
MetalQwenModel::Config
encoder_config_(const std::string& enc_dir)
{
  MetalQwenModel::Config c;
  c.n_layers = 36; c.hidden = 4096; c.n_heads = 32; c.n_kv_heads = 8;
  c.head_dim = 128; c.ffn_inner = 12288; c.vocab = 151936;
  c.rope_theta = 5.0e6f; c.rms_eps = 1e-6f; c.rotary_dim = 128;
  c.full_attn_interval = 1; c.tie_embeddings = false; c.use_bf16 = true;
  c.dense = true; c.zero_centered_norm = false; c.attn_output_gate = false;
  c.backbone_only = true; c.weight_prefix = "model.language_model.";
  c.model_seg = ""; c.max_seq = 1024; c.page_tokens = 256;
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(enc_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto root = fd.as_object();
      if (root.contains("text_config")) {
        FlexData tc = root.at("text_config");
        if (tc.is_object()) {
          auto o = tc.as_object();
          auto geti = [&](const char* k, int cur) {
            return o.contains(k) ? (int)o.at(k).as_int(cur) : cur; };
          auto getf = [&](const char* k, float cur) {
            return o.contains(k) ? (float)o.at(k).as_real(cur) : cur; };
          c.n_layers   = geti("num_hidden_layers", c.n_layers);
          c.hidden     = geti("hidden_size", c.hidden);
          c.n_heads    = geti("num_attention_heads", c.n_heads);
          c.n_kv_heads = geti("num_key_value_heads", c.n_kv_heads);
          c.head_dim   = geti("head_dim",
                              c.n_heads > 0 ? c.hidden / c.n_heads : c.head_dim);
          c.rotary_dim = c.head_dim;
          c.ffn_inner  = geti("intermediate_size", c.ffn_inner);
          c.vocab      = geti("vocab_size", c.vocab);
          c.rope_theta = getf("rope_theta", c.rope_theta);
          c.rms_eps    = getf("rms_norm_eps", c.rms_eps);
        }
      }
      if (root.contains("tie_word_embeddings")) {
        c.tie_embeddings =
            root.at("tie_word_embeddings").as_bool(c.tie_embeddings);
      }
    }
  }
  return c;
}

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
      const std::vector<std::int32_t> seg =
          tok.encode(text.substr(pos, best - pos));
      out.insert(out.end(), seg.begin(), seg.end());
    }
    const std::int32_t sid = tok.special_token_id(kMarkers[which]);
    if (sid >= 0) { out.push_back(sid); }
    pos = best + std::strlen(kMarkers[which]);
  }
  return out;
}

// Total physical RAM (bytes), 0 if unknown.
std::size_t
physical_ram_()
{
  std::uint64_t mem = 0;
  std::size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) { return 0; }
  return (std::size_t)mem;
}

// Sum of the .safetensors bytes in a dir -- a proxy for the wired footprint.
std::size_t
weights_bytes_(const std::filesystem::path& dir)
{
  namespace fs = std::filesystem;
  std::size_t total = 0;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(dir, ec), end; it != end;
       it.increment(ec)) {
    if (it->is_regular_file(ec) && it->path().extension() == ".safetensors") {
      total += (std::size_t)it->file_size(ec);
    }
  }
  return total;
}

// In-place throttled progress bar (mirrors the Krea-2 / FLUX.2 calibrations).
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

}  // namespace

bool
collect_boogu_calibration(MetalCompute* mc, const std::string& model_root,
                          const std::vector<std::string>& prompts, int steps,
                          int height, int width, std::uint64_t seed,
                          const std::string& out_dir, std::string* err,
                          const std::function<bool()>& stop)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (mc == nullptr) { return fail("boogu calib: null metal-compute"); }
  if (prompts.empty()) { return fail("boogu calib: empty prompt corpus"); }
  if (steps <= 0) { steps = 4; }             // the Turbo student's step count
  if (height % 16 != 0 || width % 16 != 0) {
    return fail("boogu calib: height/width must be multiples of 16");
  }
  namespace fs = std::filesystem;
  auto* sess = mc->session();

  const std::string enc_dir = (fs::path(model_root) / "mllm").string();
  const std::string dit_dir = (fs::path(model_root) / "transformer").string();
  const std::string tok_path =
      (fs::path(model_root) / "processor" / "tokenizer.json").string();

  // The t2i system prompt, verbatim from BooguImagePipeline (calibration runs
  // text-only, so it is the t2i one). Boogu drops NOTHING from the templated
  // sequence -- every token, system prompt included, is conditioning.
  static constexpr const char* kSysT2I =
      "<|im_start|>system\nYou are a helpful assistant that generates "
      "high-quality images based on user instructions. The instructions are as "
      "follows.<|im_end|>\n<|im_start|>user\n";
  static constexpr const char* kSuffix = "<|im_end|>\n";

  std::unique_ptr<UiTextStream> bar =
      sess ? sess->open_text_stream() : std::unique_ptr<UiTextStream>();
  int pct = -1;

  // ---- Phase 1: encoder resident -> cache each prompt's conditioning -------
  std::vector<SharedBuffer> ctx_cache;
  std::vector<int> n_cache;
  {
    auto tok = Tokenizer::from_huggingface_json(tok_path, sess);
    if (!tok) {
      return fail("boogu calib: tokenizer load failed: " + tok_path);
    }
    const MetalQwenModel::Config ecfg = encoder_config_(enc_dir);
    const int EH = ecfg.hidden;
    const int NL = ecfg.n_layers;
    auto enc = MetalQwenModel::load(enc_dir, mc, ecfg);
    if (!enc) { return fail("boogu calib: mllm load failed: " + enc_dir); }
    SharedBuffer embed;
    std::vector<float> fnorm((std::size_t)EH, 1.0f);
    {
      auto wts = MetalLlamaWeights::open_model(enc_dir);
      if (wts.has_value()) {
        embed = wts->load("model.language_model.embed_tokens.weight", mc);
        SharedBuffer nw = wts->load("model.language_model.norm.weight", mc);
        if (!nw.empty()) {
          const auto* p = static_cast<const std::uint16_t*>(nw.contents());
          for (int h = 0; h < EH; ++h) { fnorm[(std::size_t)h] = bf16_to_f32_(p[h]); }
        }
      }
    }
    if (embed.empty()) { return fail("boogu calib: embed table load failed"); }

    for (std::size_t pi = 0; pi < prompts.size(); ++pi) {
      if (stop()) { return fail("boogu calib: stopped by request"); }
      const std::string templated =
          std::string(kSysT2I) + prompts[pi] + kSuffix;
      std::vector<std::int32_t> ids = encode_with_specials_(*tok, templated);
      calib_progress_(bar.get(), "encode", (int)pi + 1, (int)prompts.size(),
                      pct);
      if (ids.empty()) { continue; }
      const int n = (int)ids.size();
      SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
      if (x.empty()) { continue; }
      const auto* tbl = static_cast<const std::uint8_t*>(embed.contents());
      auto* xb = static_cast<std::uint8_t*>(x.contents());
      const std::size_t vocab = embed.byte_size() / ((std::size_t)EH * 2);
      bool ok = true;
      for (int i = 0; i < n; ++i) {
        const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
        if (id >= vocab) { ok = false; break; }
        std::memcpy(xb + (std::size_t)i * EH * 2,
                    tbl + (std::size_t)id * EH * 2, (std::size_t)EH * 2);
      }
      if (!ok) { continue; }
      ContextManager* cm = enc->context_manager();
      const ContextId cid = cm->acquire_root();
      SharedBuffer taps =
          enc->forward_embeddings_taps(cid, x, n, std::vector<int>{NL - 1});
      cm->release(cid);
      if (taps.empty()) { continue; }
      // The tap is the last layer's PRE-final-norm output; the mllm's
      // last_hidden_state is post-norm, so apply the final RMSNorm on the host.
      SharedBuffer ctx = mc->make_shared_buffer((std::size_t)n * EH * 2);
      if (ctx.empty()) { continue; }
      const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
      auto* cp = static_cast<std::uint16_t*>(ctx.contents());
      for (int p = 0; p < n; ++p) {
        const auto* row = tp + (std::size_t)p * EH;
        double ss = 0.0;
        for (int h = 0; h < EH; ++h) {
          const double v = bf16_to_f32_(row[h]); ss += v * v;
        }
        const double inv = 1.0 / std::sqrt(ss / (double)EH + (double)ecfg.rms_eps);
        for (int h = 0; h < EH; ++h) {
          cp[(std::size_t)p * EH + h] = f32_to_bf16_(
              (float)(bf16_to_f32_(row[h]) * inv * fnorm[(std::size_t)h]));
        }
      }
      ctx_cache.push_back(std::move(ctx));
      n_cache.push_back(n);
    }
  }   // encoder + embed + tokenizer freed here
  if (ctx_cache.empty()) { return fail("boogu calib: no prompt encoded"); }
  if (sess) {
    sess->log_debug(fmt("boogu calib: encoded {} prompts; mllm freed",
                        ctx_cache.size()));
  }

  // ---- Phase 2: DiT resident -> denoise each cached prompt, tapping abs-max.
  bool stream_blocks;
  {
    const std::size_t dit_b = weights_bytes_(dit_dir);
    const std::size_t ram = physical_ram_();
    const std::size_t need = dit_b + (6ull << 30);   // +6 GB headroom
    stream_blocks = (ram == 0) || (ram < need);
    if (const char* e = std::getenv("VPIPE_BOOGU_CALIB_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    if (sess) {
      sess->log_debug(fmt(
          "boogu calib: dit {} GB + 6 GB headroom vs {} GB RAM -> {}",
          dit_b >> 30, ram >> 30, stream_blocks ? "STREAM" : "PRELOAD"));
    }
  }
  double pin_frac = stream_blocks ? 0.60 : 0.0;
  if (const char* e = std::getenv("VPIPE_BOOGU_CALIB_PIN_FRAC")) {
    pin_frac = std::atof(e);
  }
  auto dit = MetalBooguTransformer::load(
      dit_dir, mc, MetalBooguTransformer::Config{}, stream_blocks, pin_frac);
  if (!dit) { return fail("boogu calib: DiT load failed: " + dit_dir); }
  dit->set_stream_stop(stop);
  if (sess && stream_blocks) {
    sess->log_debug(fmt(
        "boogu calib: pinned {} of {} DiT blocks resident ({}% RAM budget), "
        "streaming the rest", dit->pinned_blocks(),
        dit->config().n_double + dit->config().n_single, (int)(pin_frac * 100)));
  }

  const int P = dit->config().patch;
  const int lat_h = height / 8, lat_w = width / 8;      // VAE 8x
  const int img_seq = (lat_h / P) * (lat_w / P);
  const int XIN = dit->config().x_in();

  // The DMD student's schedule -- ASCENDING sigma (0 = pure noise, 1 = clean),
  // linspace(conditioning_sigma, 1, steps+1)[:-1], with a renoise between
  // steps. This is the trajectory Edit-Turbo actually runs, so it is the one
  // the activation statistics must come from.
  std::vector<double> sig((std::size_t)steps);
  for (int i = 0; i < steps; ++i) {
    sig[(std::size_t)i] = (double)i / (double)steps;
  }

  // Reference-conditioned calibration. Boogu-Edit's image stream carries CLEAN
  // reference tokens through their own embedder + refiner, and the joint
  // attention runs a longer sequence -- a different activation distribution
  // than pure text-to-image. Roll the PREVIOUS prompt's denoised latent in as
  // the reference for a fraction of the prompts (cost-neutral: the trajectory's
  // clean output is already the packed [img_seq, XIN] layout a RefImage wants).
  // Edit-Turbo is primarily an EDIT model, so the default leans heavier than
  // FLUX.2's. VPIPE_BOOGU_CALIB_EDIT_FRAC overrides.
  double edit_frac = 0.75;
  if (const char* e = std::getenv("VPIPE_BOOGU_CALIB_EDIT_FRAC")) {
    edit_frac = std::atof(e);
    edit_frac = edit_frac < 0.0 ? 0.0 : (edit_frac > 1.0 ? 1.0 : edit_frac);
  }
  SharedBuffer ref_buf = mc->make_shared_buffer((std::size_t)img_seq * XIN * 2);
  bool ref_ready = false;
  double edit_acc = 0.0;
  int edit_used = 0;

  dit->calib_begin();
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * XIN * 2);
  if (latbuf.empty() || ref_buf.empty()) {
    dit->calib_end();
    return fail("boogu calib: latent scratch alloc failed");
  }
  std::vector<float> packed((std::size_t)img_seq * XIN);
  pct = -1;
  const int total_fwd = (int)ctx_cache.size() * steps;
  for (std::size_t e = 0; e < ctx_cache.size(); ++e) {
    if (stop()) {
      if (bar) { bar->end(); }
      dit->calib_end();
      return fail("boogu calib: stopped");
    }
    std::vector<MetalBooguTransformer::RefImage> refs;
    if (ref_ready && edit_frac > 0.0) {
      edit_acc += edit_frac;
      if (edit_acc >= 1.0) {
        edit_acc -= 1.0;
        MetalBooguTransformer::RefImage r;
        r.latents = ref_buf.subview(0, ref_buf.byte_size());
        r.seq = img_seq; r.grid_h = lat_h; r.grid_w = lat_w;
        refs.push_back(std::move(r));
        ++edit_used;
      }
    }
    std::mt19937_64 rng(seed + e);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
    for (int i = 0; i < steps; ++i) {
      if (stop()) {
        if (bar) { bar->end(); }
        dit->calib_end();
        return fail("boogu calib: stopped");
      }
      auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
      for (std::size_t k = 0; k < packed.size(); ++k) {
        lb[k] = f32_to_bf16_(packed[k]);
      }
      SharedBuffer vel = dit->forward_dit(ctx_cache[e], n_cache[e], latbuf,
                                          img_seq, lat_h, lat_w,
                                          (float)sig[(std::size_t)i], refs);
      if (vel.empty()) {
        if (bar) { bar->end(); }
        dit->calib_end();
        return fail("boogu calib: forward");
      }
      // DMD student step: x <- x + (1 - sigma) * v, then renoise toward the
      // next sigma (the last step leaves the clean latent).
      const auto* vp = static_cast<const std::uint16_t*>(vel.contents());
      const double s = sig[(std::size_t)i];
      for (std::size_t k = 0; k < packed.size(); ++k) {
        packed[k] += (float)((1.0 - s) * (double)bf16_to_f32_(vp[k]));
      }
      if (i + 1 < steps) {
        const double s1 = sig[(std::size_t)i + 1];
        std::mt19937_64 nrng(seed + e * 1000 + (std::uint64_t)i + 1);
        std::normal_distribution<float> nn(0.0f, 1.0f);
        for (auto& v : packed) {
          v = (float)((1.0 - s1) * (double)nn(nrng) + s1 * (double)v);
        }
      }
      calib_progress_(bar.get(), "denoise", (int)e * steps + i + 1, total_fwd,
                      pct);
    }
    if (edit_frac > 0.0) {
      auto* rb = static_cast<std::uint16_t*>(ref_buf.contents());
      for (std::size_t k = 0; k < packed.size(); ++k) {
        rb[k] = f32_to_bf16_(packed[k]);
      }
      ref_ready = true;
    }
    if (sess) {
      sess->log_debug(fmt("boogu calib: denoised prompt {}/{}{}", e + 1,
                          ctx_cache.size(),
                          refs.empty() ? "" : " (+1 reference)"));
    }
  }
  if (bar) { bar->end(); }
  const std::map<std::string, std::vector<float>> stats = dit->calib_stats();
  dit->calib_end();

  std::error_code ec;
  fs::create_directories(out_dir, ec);
  for (const auto& kv : stats) {
    std::ofstream out((fs::path(out_dir) / (kv.first + ".f32")).string(),
                      std::ios::binary);
    if (!out) { return fail("boogu calib: write failed: " + kv.first); }
    out.write(reinterpret_cast<const char*>(kv.second.data()),
              (std::streamsize)kv.second.size() * 4);
    if (!out) { return fail("boogu calib: write failed: " + kv.first); }
  }
  if (sess) {
    sess->log_normal(fmt(
        "boogu calib: {} prompts x {} steps -> {} group files in {} ({} prompts "
        "with a reference image)",
        ctx_cache.size(), steps, stats.size(), out_dir, edit_used));
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
