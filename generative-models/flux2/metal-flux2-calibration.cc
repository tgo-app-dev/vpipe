#include "generative-models/flux2/metal-flux2-calibration.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/context-manager.h"
#include "generative-models/flux2/metal-flux2-transformer.h"
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

// FLUX.2 klein text encoder (dense Qwen3, model. prefix) -- mirrors the
// text-to-image stage's encoder_config_flux2_, sized from text_encoder/
// config.json so both klein sizes calibrate (4B ~4B Qwen3, 9B an 8B Qwen3).
MetalQwenModel::Config
encoder_config_(const std::string& enc_dir)
{
  MetalQwenModel::Config c;
  c.n_layers = 36; c.hidden = 2560; c.n_heads = 32; c.n_kv_heads = 8;
  c.head_dim = 128; c.ffn_inner = 9728; c.vocab = 151936;
  c.rope_theta = 1.0e6f; c.rms_eps = 1e-6f; c.rotary_dim = 128;
  c.full_attn_interval = 1; c.tie_embeddings = true; c.use_bf16 = true;
  c.dense = true; c.zero_centered_norm = false; c.attn_output_gate = false;
  c.backbone_only = true; c.weight_prefix = "model.";
  c.model_seg = ""; c.max_seq = 1024; c.page_tokens = 256;
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(enc_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      auto geti = [&](const char* k, int cur) -> int {
        return o.contains(k) ? (int)o.at(k).as_int(cur) : cur;
      };
      auto getf = [&](const char* k, float cur) -> float {
        return o.contains(k) ? (float)o.at(k).as_real(cur) : cur;
      };
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
      if (o.contains("tie_word_embeddings")) {
        c.tie_embeddings = o.at("tie_word_embeddings").as_bool(c.tie_embeddings);
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
      const std::vector<std::int32_t> seg = tok.encode(text.substr(pos, best - pos));
      out.insert(out.end(), seg.begin(), seg.end());
    }
    const std::int32_t sid = tok.special_token_id(kMarkers[which]);
    if (sid >= 0) { out.push_back(sid); }
    pos = best + std::strlen(kMarkers[which]);
  }
  return out;
}

// FLUX.2 empirical flow-shift mu (diffusers Flux2Pipeline.compute_empirical_mu):
// resolution- AND step-dependent. Mirrors the copy in text-to-image-stage.cc.
double
flux2_empirical_mu_(int image_seq_len, int num_steps)
{
  const double a1 = 8.73809524e-05, b1 = 1.89833333;
  const double a2 = 0.00016927,     b2 = 0.45666666;
  const double n = (double)image_seq_len;
  if (n > 4300.0) { return a2 * n + b2; }
  const double m_200 = a2 * n + b2;
  const double m_10  = a1 * n + b1;
  const double a = (m_200 - m_10) / 190.0;
  const double b = m_200 - 200.0 * a;
  return a * (double)num_steps + b;
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

// Sum of the .safetensors bytes in a dir -- a proxy for the wired f16 footprint.
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

// In-place throttled progress bar (mirrors the Krea-2 calibration + the
// model-quantizer): redraws on a carriage-return only when the integer
// percentage changes, the frame space-padded so a shorter redraw overwrites a
// longer prior one.
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
collect_flux2_calibration(MetalCompute* mc, const std::string& model_root,
                          const std::vector<std::string>& prompts, int steps,
                          int height, int width, std::uint64_t seed,
                          const std::string& out_dir, std::string* err,
                          const std::function<bool()>& stop)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (mc == nullptr) { return fail("flux2 calib: null metal-compute"); }
  if (prompts.empty()) { return fail("flux2 calib: empty prompt corpus"); }
  if (steps <= 0) { steps = 8; }
  if (height % 16 != 0 || width % 16 != 0) {
    return fail("flux2 calib: height/width must be multiples of 16");
  }
  namespace fs = std::filesystem;
  auto* sess = mc->session();

  const std::string enc_dir = (fs::path(model_root) / "text_encoder").string();
  const std::string dit_dir = (fs::path(model_root) / "transformer").string();
  const std::string tok_path =
      (fs::path(model_root) / "tokenizer" / "tokenizer.json").string();

  static const int kTaps[3] = {9, 18, 27};   // output.hidden_states indices
  std::vector<int> tap_layers;
  for (int k : kTaps) { tap_layers.push_back(k - 1); }

  // In-place progress bar spanning both phases (encode, then denoise), painted
  // on the user-facing text stream like the Krea-2 calibration.
  std::unique_ptr<UiTextStream> bar =
      sess ? sess->open_text_stream() : std::unique_ptr<UiTextStream>();
  int pct = -1;

  // ---- Phase 1: encoder resident -> cache each prompt's context [n, 3*EH].
  std::vector<SharedBuffer> ctx_cache;
  std::vector<int> n_cache;
  {
    auto tok = Tokenizer::from_huggingface_json(tok_path, sess);
    if (!tok) {
      return fail("flux2 calib: tokenizer load failed: " + tok_path);
    }
    const MetalQwenModel::Config ecfg = encoder_config_(enc_dir);
    const int EH = ecfg.hidden;      // encoder hidden (4B 2560, 9B larger)
    const int JD = 3 * EH;           // 3 tapped layers concatenated
    auto enc = MetalQwenModel::load(enc_dir, mc, ecfg);
    if (!enc) { return fail("flux2 calib: encoder load failed: " + enc_dir); }
    SharedBuffer embed;
    {
      auto wts = MetalLlamaWeights::open_model(enc_dir);
      if (wts.has_value()) {
        embed = wts->load("model.embed_tokens.weight", mc);
      }
    }
    if (embed.empty()) { return fail("flux2 calib: embed table load failed"); }

    for (std::size_t pi = 0; pi < prompts.size(); ++pi) {
      if (stop()) { return fail("flux2 calib: stopped by request"); }
      const std::string templated = std::string("<|im_start|>user\n") +
                                    prompts[pi] +
                                    "<|im_end|>\n<|im_start|>assistant\n";
      std::vector<std::int32_t> ids = encode_with_specials_(*tok, templated);
      calib_progress_(bar.get(), "encode", (int)pi + 1, (int)prompts.size(),
                      pct);
      if (ids.empty()) { continue; }
      const int n = (int)ids.size();
      SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
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
      SharedBuffer taps = enc->forward_embeddings_taps(cid, x, n, tap_layers);
      cm->release(cid);
      if (taps.empty()) { continue; }
      SharedBuffer ctx = mc->make_shared_buffer((std::size_t)n * JD * 2);
      const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
      auto* cp = static_cast<_Float16*>(ctx.contents());
      for (int p = 0; p < n; ++p) {
        for (int j = 0; j < 3; ++j) {
          const std::size_t src = ((std::size_t)j * n + p) * EH;
          const std::size_t dst = (std::size_t)p * JD + (std::size_t)j * EH;
          for (int h = 0; h < EH; ++h) {
            std::uint32_t u = (std::uint32_t)tp[src + h] << 16;
            float f; std::memcpy(&f, &u, 4);
            cp[dst + h] = (_Float16)f;
          }
        }
      }
      ctx_cache.push_back(std::move(ctx));
      n_cache.push_back(n);
    }
  }   // encoder + embed + tokenizer freed here
  if (ctx_cache.empty()) { return fail("flux2 calib: no prompt encoded"); }
  if (sess) {
    sess->log_debug(fmt("flux2 calib: encoded {} prompts; encoder freed",
                        ctx_cache.size()));
  }

  // ---- Phase 2: DiT resident -> denoise each cached prompt, tapping abs-max.
  // Stream the DiT blocks on a box that can't hold the whole DiT (the encoder
  // is already freed here, so peak = DiT + scratch). VPIPE_FLUX2_CALIB_STREAM
  // forces the choice.
  bool stream_blocks;
  {
    const std::size_t dit_b = weights_bytes_(dit_dir);
    const std::size_t ram = physical_ram_();
    const std::size_t need = dit_b + (6ull << 30);   // +6 GB headroom
    stream_blocks = (ram == 0) || (ram < need);
    if (const char* e = std::getenv("VPIPE_FLUX2_CALIB_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    if (sess) {
      sess->log_debug(fmt(
          "flux2 calib: dit {} GB + 6 GB headroom vs {} GB RAM -> {}",
          dit_b >> 30, ram >> 30, stream_blocks ? "STREAM" : "PRELOAD"));
    }
  }
  auto dit = MetalFlux2Transformer::load(
      dit_dir, mc, MetalFlux2Transformer::Config{}, stream_blocks);
  if (!dit) { return fail("flux2 calib: DiT load failed: " + dit_dir); }
  dit->set_stream_stop(stop);

  const int gh = height / 16, gw = width / 16;
  const int img_seq = gh * gw;
  const int IC = dit->config().in_channels;

  // FlowMatchEuler sigma schedule. FLUX.2's flow-shift (mu) is resolution- AND
  // step-dependent (compute_empirical_mu); use it so the calibration samples the
  // same sigmas inference does (a fixed shift skews the tapped distribution).
  FlowSchedulerSpec sched;
  sched.type = "simple"; sched.steps = steps;
  sched.shift = flux2_empirical_mu_(img_seq, steps);
  sched.shift_type = "exponential";
  const std::vector<double> sig = sched.sigmas();

  dit->calib_begin();
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  std::vector<float> packed((std::size_t)img_seq * IC);
  pct = -1;
  const int total_fwd = (int)ctx_cache.size() * steps;
  for (std::size_t e = 0; e < ctx_cache.size(); ++e) {
    if (stop()) {
      if (bar) { bar->end(); }
      dit->calib_end();
      return fail("flux2 calib: stopped");
    }
    std::mt19937_64 rng(seed + e);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
    for (int i = 0; i < steps; ++i) {
      if (stop()) {
        if (bar) { bar->end(); }
        dit->calib_end();
        return fail("flux2 calib: stopped");
      }
      auto* lb = static_cast<_Float16*>(latbuf.contents());
      for (std::size_t k = 0; k < packed.size(); ++k) {
        lb[k] = (_Float16)packed[k];
      }
      SharedBuffer vel = dit->forward_dit(ctx_cache[e], n_cache[e], latbuf,
                                          img_seq, gh, gw,
                                          (float)sig[(std::size_t)i]);
      if (vel.empty()) {
        if (bar) { bar->end(); }
        dit->calib_end();
        return fail("flux2 calib: forward");
      }
      const auto* vp = static_cast<const _Float16*>(vel.contents());
      const double dt = sig[(std::size_t)i + 1] - sig[(std::size_t)i];
      for (std::size_t k = 0; k < packed.size(); ++k) {
        packed[k] += (float)(dt * (double)(float)vp[k]);
      }
      calib_progress_(bar.get(), "denoise", (int)e * steps + i + 1, total_fwd,
                      pct);
    }
    if (sess) {
      sess->log_debug(fmt("flux2 calib: denoised prompt {}/{}", e + 1,
                          ctx_cache.size()));
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
    if (!out) { return fail("flux2 calib: write failed: " + kv.first); }
    out.write(reinterpret_cast<const char*>(kv.second.data()),
              (std::streamsize)kv.second.size() * 4);
    if (!out) { return fail("flux2 calib: write failed: " + kv.first); }
  }
  if (sess) {
    sess->log_normal(fmt(
        "flux2 calib: {} prompts x {} steps -> {} group files in {}",
        ctx_cache.size(), steps, stats.size(), out_dir));
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
