#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

constexpr const char* kKey = "minimax-h3-avae/f16|";

// The Comfy-Org single-file audio VAE carries the latent statistics
// AND the decoder kwargs under this one `__metadata__` key.
constexpr const char* kComfyKey = "minimax_h3_audio_vae";

// The alias-free resamplers are fixed at ratio 2 / 12 taps everywhere in
// BigVGAN, and the Kaiser windows are stored as buffers, so the geometry
// below is a property of the architecture rather than of the checkpoint.
constexpr int kResampleTaps  = 12;
constexpr int kResampleRatio = 2;

// Read an F32 tensor into a host vector. These are all small (a filter, a
// per-channel alpha) or consumed and dropped (a convolution's weights,
// which the model keeps only in its folded f16 form).
bool
read_f32_(WeightSet& ws, MetalCompute* mc, const std::string& nm,
          std::vector<float>* out)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr || info->dtype != "F32") { return false; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty() || raw.byte_size() < n * 4) { return false; }
  out->assign(static_cast<const float*>(raw.contents()),
              static_cast<const float*>(raw.contents()) + n);
  return true;
}

}  // namespace

int
MetalMiniMaxH3AudioVae::Config::hop() const
{
  int h = 1;
  for (int r : up_rates) { h *= r; }
  return h;
}

int
MetalMiniMaxH3AudioVae::latent_frames_for_seconds(double seconds)
{
  if (seconds <= 0.0) { return 0; }
  const Config cfg;
  const int hop = cfg.hop();
  if (hop <= 0 || cfg.sample_rate <= 0) { return 0; }
  // The latent rate is sample_rate / hop -- 40 Hz at 32 kHz over an 800
  // sample hop. Rounded UP, because a partial frame is still a frame.
  const double rate = (double)cfg.sample_rate / (double)hop;
  return (int)(seconds * rate + 0.999);
}

void
MetalMiniMaxH3AudioVae::decode_cost(int latent_frames, std::size_t* pcm,
                                    std::size_t* arena)
{
  if (pcm != nullptr) { *pcm = 0; }
  if (arena != nullptr) { *arena = 0; }
  if (latent_frames <= 0) { return; }
  const Config cfg;
  const std::size_t hop = (std::size_t)cfg.hop();
  const std::size_t ch  = (std::size_t)cfg.stereo_channels;
  const std::size_t lf  = (std::size_t)latent_frames;
  if (hop == 0 || ch == 0) { return; }
  // Planar f32, the shape decode() documents: [stereo, frames * hop()].
  if (pcm != nullptr) { *pcm = ch * lf * hop * sizeof(float); }
  if (arena == nullptr) { return; }
  // Every upsample stage trades channels for time at a roughly constant
  // product, so each intermediate is about `width x latent_frames`
  // elements whatever stage it is -- the trunk starts at `latent_dim`
  // and the stages run at `decoder_dim`, so the widest is the bound.
  // Two are live at once, a stage's input and its output. f32
  // throughout, because this path runs F32 rather than bf16.
  const std::size_t width =
      (std::size_t)(cfg.latent_dim > cfg.decoder_dim ? cfg.latent_dim
                                                     : cfg.decoder_dim);
  *arena = 2 * width * lf * sizeof(float);
}

std::string
MetalMiniMaxH3AudioVae::resolve_vae_dir(const std::string& path)
{
  namespace fs = std::filesystem;
  fs::path p(path);
  // The Comfy-Org single file (see the video VAE for the shape of these
  // repacks). This one is fp32, i.e. the released weights verbatim; only
  // the config's envelope differs.
  {
    const std::string f =
        comfy::resolve_component(path, "vae", kComfyKey, {"audio_vae"});
    if (!f.empty()) { return f; }
  }
  if (!fs::is_directory(p)) { return path; }
  if (fs::exists(p / "model.safetensors") && fs::exists(p / "config.json") &&
      !fs::exists(p / "audio_vae")) {
    return p.string();                                   // already audio_vae/
  }
  if (fs::exists(p / "audio_vae" / "model.safetensors")) {
    return (p / "audio_vae").string();                   // a partition root
  }
  // Either partition's copy: both `FL2VA/` and `Ref2VA/` are complete
  // pipelines carrying the same codec, so a Ref2VA-only checkout has to
  // resolve here too and which copy answers does not matter.
  for (const char* part : {"FL2VA", "Ref2VA"}) {
    if (fs::exists(p / part / "audio_vae" / "model.safetensors")) {
      return (p / part / "audio_vae").string();
    }
  }
  return path;
}

bool
MetalMiniMaxH3AudioVae::config_from_json(const std::string& vae_dir,
                                         Config& out, std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const fs::path dir(resolve_vae_dir(vae_dir));
  // The released layout splits this across TWO files -- config.json for
  // the latent statistics, metadata.json for the decoder's shape. The
  // Comfy-Org single file carries both fields in one metadata blob, so
  // `fd` below is that blob and `kwargs` is read from it directly
  // instead of from metadata.metadata.kwargs.
  const bool is_comfy = comfy::is_component(dir.string(), kComfyKey);
  FlexData fd;
  if (is_comfy) {
    std::string cerr;
    if (!comfy::metadata_json(dir.string(), kComfyKey, fd, &cerr)) {
      return fail(cerr);
    }
  } else {
    std::ifstream in(dir / "config.json");
    if (!in) { return fail("no config.json under " + dir.string()); }
    try {
      fd = FlexData::from_json(in);
    } catch (const std::exception& e) {
      return fail(std::string("config.json: ") + e.what());
    }
  }
  if (!fd.is_object()) { return fail("the audio VAE config is not an object"); }
  auto o = fd.as_object();
  auto geti = [&](const char* k, int dflt) {
    return o.contains(k) ? (int)o.at(k).as_int(dflt) : dflt;
  };
  out.latent_channels = geti("latent_channels", out.latent_channels);
  out.stereo_channels = geti("output_channel", out.stereo_channels);
  out.sample_rate     = geti("sample_rate", out.sample_rate);
  auto reals = [&](const char* k, std::vector<float>* dst) {
    if (!o.contains(k)) { return; }
    FlexData v = o.at(k);
    if (!v.is_array()) { return; }
    for (auto x : v.as_real_span()) { dst->push_back((float)x); }
  };
  reals("latents_mean", &out.latents_mean);
  reals("latents_std", &out.latents_std);

  // The decoder's own shape lives in the source checkpoint's
  // metadata.json, not in the diffusers config. Absent, the defaults
  // above are the released ones -- but a checkpoint that changed the
  // rates and did not ship metadata would load and then produce a
  // waveform of the wrong LENGTH, so say so rather than assume.
  FlexData kw;
  if (is_comfy) {
    if (!o.contains("kwargs")) {
      return fail("the audio VAE metadata has no 'kwargs'");
    }
    kw = o.at("kwargs");
  } else {
    std::ifstream mj(dir / "metadata.json");
    if (!mj) { return fail("no metadata.json under " + dir.string()); }
    FlexData md;
    try {
      md = FlexData::from_json(mj);
    } catch (const std::exception& e) {
      return fail(std::string("metadata.json: ") + e.what());
    }
    if (!md.is_object()) { return fail("metadata.json is not an object"); }
    FlexData meta = md.as_object().contains("metadata")
                        ? md.as_object().at("metadata")
                        : FlexData::make_null();
    if (!meta.is_object() || !meta.as_object().contains("kwargs")) {
      return fail("metadata.json has no metadata.kwargs");
    }
    kw = meta.as_object().at("kwargs");
  }
  if (!kw.is_object()) { return fail("the audio VAE kwargs is not an object"); }
  auto ko = kw.as_object();
  if (ko.contains("decoder_dim")) {
    out.decoder_dim = (int)ko.at("decoder_dim").as_int(out.decoder_dim);
  }
  if (ko.contains("latent_dim")) {
    out.latent_dim = (int)ko.at("latent_dim").as_int(out.latent_dim);
  }
  if (ko.contains("vae_latent_channels")) {
    out.latent_channels =
        (int)ko.at("vae_latent_channels").as_int(out.latent_channels);
  }
  if (ko.contains("decoder_rates")) {
    FlexData dr = ko.at("decoder_rates");
    if (dr.is_array()) {
      std::vector<int> rates;
      for (auto v : dr.as_array()) { rates.push_back((int)v.as_int(0)); }
      if (!rates.empty()) { out.up_rates = std::move(rates); }
    }
  }
  if (out.latent_channels <= 0 || out.latent_dim <= 0 ||
      out.decoder_dim <= 0 || out.stereo_channels <= 0) {
    return fail("non-positive geometry in the audio VAE config");
  }
  for (int r : out.up_rates) {
    if (r <= 0) { return fail("non-positive decoder rate"); }
  }
  // `up_kernels` is filled from the checkpoint at load(); the default is
  // only there so a Config built by hand is usable.
  out.up_kernels.assign(out.up_rates.size(), 0);
  return true;
}

MetalMiniMaxH3AudioVae::~MetalMiniMaxH3AudioVae() = default;

SharedBuffer
MetalMiniMaxH3AudioVae::f16_(WeightSet& ws, const std::string& nm)
{
  if (ws.src().info(nm) == nullptr) { return {}; }
  return ws.derived(std::string(kKey) + nm, [&]() -> SharedBuffer {
    std::vector<float> v;
    if (!read_f32_(ws, _mc, nm, &v)) { return {}; }
    SharedBuffer out = _mc->make_shared_buffer(v.size() * 2);
    if (out.empty()) { return {}; }
    auto* d = static_cast<_Float16*>(out.contents());
    for (std::size_t i = 0; i < v.size(); ++i) { d[i] = (_Float16)v[i]; }
    return out;
  });
}

// Fold weight-norm, transpose into the GEMM's layout and narrow to f16, in
// one pass. PyTorch stores `weight_g` / `weight_v` with the norm taken over
// every dimension but the first, so `w = v * g / ||v||` -- and for a
// ConvTranspose1d that first dimension is the INPUT width, i.e. the
// contraction axis, not the output one.
MetalMiniMaxH3AudioVae::Conv1d
MetalMiniMaxH3AudioVae::conv1d_(WeightSet& ws, const std::string& nm,
                                bool transposed, int dilation, int rate,
                                int stride, int pad_override)
{
  Conv1d c;
  const bool wn = ws.src().info(nm + ".weight_v") != nullptr;
  const std::string wname = wn ? (nm + ".weight_v") : (nm + ".weight");
  const auto* info = ws.src().info(wname);
  if (info == nullptr || info->shape.size() != 3) { return c; }
  const int d0 = (int)info->shape[0];
  const int d1 = (int)info->shape[1];
  const int k  = (int)info->shape[2];
  c.cin      = transposed ? d0 : d1;
  c.cout     = transposed ? d1 : d0;
  c.k        = k;
  c.dilation = dilation;
  // Padding is DERIVED from the kernel the checkpoint actually stores,
  // never from config. Every ordinary convolution in this decoder is
  // "same"-padded (the reference spells it (k*d - d)/2, which is the
  // same thing), and a transposed one pads by (k - rate)/2. Taking k
  // from one place and the padding from another is how a decoder ends
  // up off by a sample per stage.
  c.pad = transposed ? (k - rate) / 2 : dilation * (k - 1) / 2;
  if (pad_override >= 0) { c.pad = pad_override; }
  c.stride = stride;
  c.b      = f16_(ws, nm + ".bias");            // may legitimately be empty

  const std::string key = std::string(kKey) + (transposed ? "ct|" : "c|") + nm;
  c.w = ws.derived(key, [&]() -> SharedBuffer {
    // Read UNCACHED: the folded f16 copy below is what the model keeps,
    // and caching the f32 original beside it would hold a second, four
    // times larger copy of every convolution in the decoder.
    std::vector<float> v, g;
    if (!read_f32_(ws, _mc, wname, &v)) { return {}; }
    if (wn && !read_f32_(ws, _mc, nm + ".weight_g", &g)) { return {}; }
    const std::size_t n = (std::size_t)d0 * d1 * k;
    if (v.size() < n) { return {}; }
    // One scale per dim-0 slice: g / ||v_slice||.
    std::vector<float> scale((std::size_t)d0, 1.0f);
    if (wn) {
      if (g.size() < (std::size_t)d0) { return {}; }
      for (int i = 0; i < d0; ++i) {
        double sq = 0.0;
        const float* row = v.data() + (std::size_t)i * d1 * k;
        for (std::size_t j = 0; j < (std::size_t)d1 * k; ++j) {
          sq += (double)row[j] * (double)row[j];
        }
        const double nrm = std::sqrt(sq);
        scale[(std::size_t)i] = nrm > 0.0 ? (float)((double)g[(std::size_t)i]
                                                    / nrm)
                                          : 0.0f;
      }
    }
    SharedBuffer out = _mc->make_shared_buffer(n * 2);
    if (out.empty()) { return {}; }
    auto* d = static_cast<_Float16*>(out.contents());
    if (!transposed) {
      // [cout, cin, k] -> [cout, k*cin], (k, cin) inside a row to pair
      // with im2col_1d_tc's column order.
      for (int co = 0; co < d0; ++co) {
        for (int ci = 0; ci < d1; ++ci) {
          for (int kk = 0; kk < k; ++kk) {
            d[((std::size_t)co * k + kk) * d1 + ci] =
                (_Float16)(v[((std::size_t)co * d1 + ci) * k + kk] *
                           scale[(std::size_t)co]);
          }
        }
      }
    } else {
      // [cin, cout, k] -> [k*cout, cin]: the GEMM produces every tap's
      // contribution side by side and col2im_1d_tc folds them.
      for (int ci = 0; ci < d0; ++ci) {
        for (int co = 0; co < d1; ++co) {
          for (int kk = 0; kk < k; ++kk) {
            d[((std::size_t)kk * d1 + co) * d0 + ci] =
                (_Float16)(v[((std::size_t)ci * d1 + co) * k + kk] *
                           scale[(std::size_t)ci]);
          }
        }
      }
    }
    return out;
  });
  return c;
}

MetalMiniMaxH3AudioVae::Snake
MetalMiniMaxH3AudioVae::snake_(WeightSet& ws, const std::string& nm)
{
  Snake s;
  std::vector<float> a, b;
  if (!read_f32_(ws, _mc, nm + ".alpha", &a) ||
      !read_f32_(ws, _mc, nm + ".beta", &b) || a.size() != b.size() ||
      a.empty()) {
    return s;
  }
  s.c = (int)a.size();
  // Both parameters are stored in LOG space, so evaluate the two
  // exponentials once per channel here rather than once per sample in the
  // kernel -- there are 127 of these activations and the last of them run
  // at the full sample rate.
  s.ealpha = ws.derived(std::string(kKey) + "ea|" + nm,
                        [&]() -> SharedBuffer {
    SharedBuffer o = _mc->make_shared_buffer(a.size() * 4);
    if (o.empty()) { return {}; }
    auto* d = static_cast<float*>(o.contents());
    for (std::size_t i = 0; i < a.size(); ++i) { d[i] = std::exp(a[i]); }
    return o;
  });
  s.rbeta = ws.derived(std::string(kKey) + "rb|" + nm,
                       [&]() -> SharedBuffer {
    SharedBuffer o = _mc->make_shared_buffer(b.size() * 4);
    if (o.empty()) { return {}; }
    auto* d = static_cast<float*>(o.contents());
    for (std::size_t i = 0; i < b.size(); ++i) {
      d[i] = 1.0f / (std::exp(b[i]) + 1e-9f);
    }
    return o;
  });
  if (s.rbeta.empty()) { s.ealpha = {}; }
  return s;
}

MetalMiniMaxH3AudioVae::Snake
MetalMiniMaxH3AudioVae::snake1d_(WeightSet& ws, const std::string& nm)
{
  Snake s;
  std::vector<float> a;
  if (!read_f32_(ws, _mc, nm + ".alpha", &a) || a.empty()) { return s; }
  s.c = (int)a.size();
  // Stored as [1, C, 1], which flattens to the per-channel vector the
  // kernel wants. Unlike the decoder's these are NOT log-space, and the
  // single parameter is both the frequency and the reciprocal gain:
  // `x + (alpha + 1e-9)^-1 * sin(alpha * x)^2`.
  s.ealpha = ws.derived(std::string(kKey) + "e1|" + nm,
                        [&]() -> SharedBuffer {
    SharedBuffer o = _mc->make_shared_buffer(a.size() * 4);
    if (o.empty()) { return {}; }
    auto* d = static_cast<float*>(o.contents());
    for (std::size_t i = 0; i < a.size(); ++i) { d[i] = a[i]; }
    return o;
  });
  s.rbeta = ws.derived(std::string(kKey) + "r1|" + nm,
                       [&]() -> SharedBuffer {
    SharedBuffer o = _mc->make_shared_buffer(a.size() * 4);
    if (o.empty()) { return {}; }
    auto* d = static_cast<float*>(o.contents());
    for (std::size_t i = 0; i < a.size(); ++i) {
      d[i] = 1.0f / (a[i] + 1e-9f);
    }
    return o;
  });
  if (s.rbeta.empty()) { s.ealpha = {}; }
  return s;
}

std::unique_ptr<MetalMiniMaxH3AudioVae>
MetalMiniMaxH3AudioVae::load(const std::string& vae_dir, MetalCompute* mc,
                             const Config& cfg)
{
  return load(WeightSet::open(resolve_vae_dir(vae_dir),
                              mc != nullptr ? mc->session() : nullptr),
              mc, cfg);
}

std::unique_ptr<MetalMiniMaxH3AudioVae>
MetalMiniMaxH3AudioVae::load(std::shared_ptr<WeightSet> ws_in, MetalCompute* mc,
                             const Config& cfg)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  WeightSet& ws = *ws_in;
  auto m =
      std::unique_ptr<MetalMiniMaxH3AudioVae>(new MetalMiniMaxH3AudioVae());
  m->_ws  = std::move(ws_in);
  m->_mc  = mc;
  m->_cfg = cfg;

  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_elt  = mc->load_library("llm_elementwise");
  m->_lib_avae = mc->load_library("audio_vae_1d");
  m->_lib_sdpa = mc->load_library("sdpa");
  // The encoder's posterior head. Looked up here rather than in
  // build_encoder_ so a missing entry point is a LOAD failure and not a
  // surprise on the first reference soundtrack; they are cheap handles.
  m->_fn_ln          = m->_lib_elt.function("layer_norm_affine_f16");
  m->_fn_transpose   = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_sdpa_causal = m->_lib_sdpa.function("sdpa_causal_f16");
  m->_fn_head_pool   = m->_lib_avae.function("attn_head_mean_pool_f16");
  m->_fn_geglu       = m->_lib_avae.function("geglu_tanh_f16");
  m->_fn_gemm     = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_im2col   = m->_lib_avae.function("im2col_1d_tc_f16");
  m->_fn_col2im   = m->_lib_avae.function("col2im_1d_tc_f16");
  m->_fn_snake    = m->_lib_avae.function("snake_beta_f16");
  m->_fn_up2      = m->_lib_avae.function("resample_up2_f16");
  m->_fn_down2    = m->_lib_avae.function("resample_down2_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_scale    = m->_lib_elt.function("scale_inplace_f16");
  m->_fn_clamp    = m->_lib_elt.function("clamp_f16");
  // M5 matrix cores. Gated on supports_matrix_cores() because matmul2d
  // EMULATES without them: on M4 and earlier this stays false and every
  // GEMM keeps taking dense_gemm_t_bm64_f16, the simdgroup-MMA kernel
  // that measures ~10 TFLOP/s there. VPIPE_H3_AVAE_NO_MMA2 forces that
  // same path on M5, which is what an A/B (and the M4-parity test) uses.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_H3_AVAE_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid() &&
                   m->_fn_bias_add.valid();
    if (const char* e = std::getenv("VPIPE_H3_AVAE_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
    if (const char* e = std::getenv("VPIPE_H3_AVAE_MMA_MIN_N")) {
      m->_mma_min_n = std::atoi(e);
    }
  }
  if (!m->_fn_gemm.valid() || !m->_fn_im2col.valid() ||
      !m->_fn_col2im.valid() || !m->_fn_snake.valid() || !m->_fn_up2.valid() ||
      !m->_fn_down2.valid() || !m->_fn_residual.valid() ||
      !m->_fn_scale.valid() || !m->_fn_clamp.valid() || !m->_fn_ln.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_sdpa_causal.valid() ||
      !m->_fn_head_pool.valid() || !m->_fn_geglu.valid()) {
    return nullptr;
  }

  // The two Kaiser windows. Every one of the 127 alias-free activations
  // builds its resamplers with the same (ratio 2, 12 taps) arguments, so
  // the filters are all bit-identical and one copy of each serves the
  // whole decoder -- see the audio-VAE test, which checks that against
  // every activation in the checkpoint rather than taking it on trust.
  m->_filt_up = ws.tensor("decoder.activation_post.upsample.filter", mc,
                          WeightSet::Residency::Copied);
  m->_filt_down = ws.tensor("decoder.activation_post.downsample.lowpass.filter",
                            mc, WeightSet::Residency::Copied);
  if (m->_filt_up.empty() || m->_filt_down.empty()) { return nullptr; }
  // The tap count is baked into the resampler kernels' inner loops, so a
  // checkpoint with a different window has to be rejected rather than
  // silently truncated to 12.
  if (m->_filt_up.byte_size() < (std::size_t)kResampleTaps * 4 ||
      m->_filt_down.byte_size() < (std::size_t)kResampleTaps * 4) {
    return nullptr;
  }

  m->_dec_in    = m->conv1d_(ws, "dec_in_proj", false, 1, 0);
  m->_conv_pre  = m->conv1d_(ws, "decoder.conv_pre", false, 1, 0);
  m->_conv_post = m->conv1d_(ws, "decoder.conv_post", false, 1, 0);
  if (m->_dec_in.empty() || m->_conv_pre.empty() || m->_conv_post.empty()) {
    return nullptr;
  }

  const int n_up = (int)m->_cfg.up_rates.size();
  const int n_kern = (int)m->_cfg.res_kernels.size();
  m->_cfg.up_kernels.assign((std::size_t)n_up, 0);
  m->_stages.resize((std::size_t)n_up);
  for (int i = 0; i < n_up; ++i) {
    UpStage& st = m->_stages[(std::size_t)i];
    const int rate = m->_cfg.up_rates[(std::size_t)i];
    const std::string un = "decoder.ups." + std::to_string(i) + ".0";
    // The transposed convolution's kernel comes from the checkpoint: the
    // released rates pair with kernels (9 for 5, 4 for 2) that no simple
    // rule reproduces, and guessing wrong changes the output LENGTH
    // rather than failing to load.
    // `.weight_v` only exists while the weight-norm parametrization is
    // still split (g, v); a checkpoint that has FOLDED it -- Comfy-Org's
    // repack of this component, and anything else exported after a
    // remove_weight_norm() -- carries the product as a plain `.weight`.
    // conv1d_ below already tries both, and this probe reading only the
    // parametrized name is what made a perfectly loadable audio VAE
    // report itself missing: the shape it wants is present either way.
    const auto* ui = ws.src().info(un + ".weight_v");
    if (ui == nullptr) { ui = ws.src().info(un + ".weight"); }
    if (ui == nullptr || ui->shape.size() != 3) { return nullptr; }
    const int k = (int)ui->shape[2];
    m->_cfg.up_kernels[(std::size_t)i] = k;
    st.rate = rate;
    st.up = m->conv1d_(ws, un, true, 1, rate);
    if (st.up.empty()) { return nullptr; }
    st.cout = st.up.cout;

    for (int j = 0; j < n_kern; ++j) {
      const int rb = i * n_kern + j;
      const std::string rn = "decoder.resblocks." + std::to_string(rb);
      const int kk = m->_cfg.res_kernels[(std::size_t)j];
      const auto& dil = m->_cfg.res_dilations[(std::size_t)j];
      if ((int)dil.size() != 3) { return nullptr; }
      AmpBlock& amp = st.amp[j];
      for (int d = 0; d < 3; ++d) {
        // convs1 is dilated; convs2 never is. Both pad to preserve
        // length, which conv1d_ derives from the stored kernel.
        amp.convs1[d] = m->conv1d_(ws, rn + ".convs1." + std::to_string(d),
                                   false, dil[(std::size_t)d], 0);
        amp.convs2[d] = m->conv1d_(ws, rn + ".convs2." + std::to_string(d),
                                   false, 1, 0);
        if (amp.convs1[d].empty() || amp.convs2[d].empty()) { return nullptr; }
        // `res_kernels` is the one piece of block geometry that is not
        // read from the checkpoint, so check it against what is: a
        // config that disagreed would otherwise pad by the wrong amount
        // and quietly shift the whole band.
        if (amp.convs1[d].k != kk || amp.convs2[d].k != kk) { return nullptr; }
      }
      for (int a = 0; a < 6; ++a) {
        amp.acts[a] =
            m->snake_(ws, rn + ".activations." + std::to_string(a) + ".act");
        if (amp.acts[a].empty()) { return nullptr; }
      }
    }
  }
  m->_act_post = m->snake_(ws, "decoder.activation_post.act");
  if (m->_act_post.empty()) { return nullptr; }

  if (mc->session() != nullptr) {
    mc->session()->log_debug(
        fmt("MetalMiniMaxH3AudioVae: BigVGAN decoder, {} stages, hop {}, "
            "{} Hz, latent {} -> {}",
            n_up, m->_cfg.hop(), m->_cfg.sample_rate, m->_cfg.latent_channels,
            m->_cfg.latent_dim));
  }
  return m;
}

int
MetalMiniMaxH3AudioVae::decoded_samples(int frames) const
{
  if (frames <= 0) { return 0; }
  int L = frames;
  for (std::size_t i = 0; i < _stages.size(); ++i) {
    const Conv1d& u = _stages[i].up;
    L = (L - 1) * _stages[i].rate - 2 * u.pad + u.k;
    if (L <= 0) { return 0; }
  }
  return L;
}

// The matrix-core route for gemm_. This decoder's weights are already
// dense f16 in the [N, K] order matmul2d wants, so unlike the quantized
// towers there is no dequant-once step -- the weight goes straight in.
//
// The shape gate is the whole design here. The BigVGAN trunk runs two
// very different families of GEMM: the resblock convolutions (M in the
// thousands, N 128-256, K ~1k-3k), which are compute-bound and carry
// ~75% of the decode's GEMM FLOPs, and the upsampling tail (M in the
// tens of thousands against N of 8-64), which is bandwidth-bound and
// would leave most of a 128-wide output tile idle. Only the first family
// is routed here; the second keeps the steel kernel, which tiles N at 32.
bool
MetalMiniMaxH3AudioVae::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& x,
                                  const Conv1d& c, const SharedBuffer& y,
                                  std::size_t y_off, int M, int N, int K)
{
  if (!_use_mma2 || M < _mma_min_m || N < _mma_min_n) { return false; }
  const bool wide = genai::mma_use_wide_tile(N, K);
  const int RN = wide ? 256 : 128;
  enc.set_function(wide ? _fn_dense_mma_deep : _fn_dense_mma);
  enc.set_buffer(0, x);
  enc.set_buffer(1, c.w);
  enc.set_buffer(2, c.w);          // bias slot, unread (has_bias = 0)
  enc.set_buffer(3, y, y_off * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  return true;
}

void
MetalMiniMaxH3AudioVae::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                              const Conv1d& c, const SharedBuffer& y,
                              std::size_t y_off, int M, int N, int K,
                              bool bias)
{
  if (gemm_mma_(enc, x, c, y, y_off, M, N, K)) {
    // The matmul2d kernel has no bias slot, so the fold is its own pass.
    if (bias) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, y_off * 2);
      enc.set_buffer(1, c.b);
      enc.set_constant(2, N);
      enc.set_constant(3, M * N);
      enc.dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x);
  enc.set_buffer(1, c.w);
  enc.set_buffer(2, bias ? c.b : c.w);
  enc.set_buffer(3, y, y_off * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias ? 1 : 0);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

void
MetalMiniMaxH3AudioVae::conv_(ComputeEncoder& enc, const Conv1d& c,
                              const SharedBuffer& in, const SharedBuffer& out,
                              int B, int Tin, int Tout)
{
  const int rows = B * Tout;
  const std::size_t per_row = (std::size_t)c.k * c.cin;
  std::size_t band = per_row > 0 ? _s.col_cap / per_row : (std::size_t)rows;
  band = std::min(std::max<std::size_t>(band, 1), (std::size_t)rows);
  const bool bias = !c.b.empty();
  for (std::size_t r0 = 0; r0 < (std::size_t)rows; r0 += band) {
    const int n = (int)std::min(band, (std::size_t)rows - r0);
    enc.set_function(_fn_im2col);
    enc.set_buffer(0, in);
    enc.set_buffer(1, _s.col);
    enc.set_constant(2, Tin);
    enc.set_constant(3, c.cin);
    enc.set_constant(4, c.k);
    enc.set_constant(5, c.dilation);
    enc.set_constant(6, c.pad);
    enc.set_constant(7, c.stride);
    enc.set_constant(8, Tout);
    enc.set_constant(9, (int)r0);
    enc.set_constant(10, n);
    enc.dispatch({(unsigned)per_row, (unsigned)n, 1}, {64, 1, 1});
    // The GEMM writes the band in place: `out` is offset by the rows
    // already done, in elements of the output width.
    gemm_(enc, _s.col, c, out, r0 * (std::size_t)c.cout, n, c.cout,
          (int)per_row, bias);
  }
}

void
MetalMiniMaxH3AudioVae::activate_(ComputeEncoder& enc, const Snake& s,
                                  const SharedBuffer& in,
                                  const SharedBuffer& out,
                                  const SharedBuffer& scratch, int B, int T,
                                  int C)
{
  const unsigned tx = (unsigned)std::min(C, 32);
  const unsigned ty = std::max(1u, 256u / tx);
  const int T2 = kResampleRatio * T;
  enc.set_function(_fn_up2);
  enc.set_buffer(0, in);
  enc.set_buffer(1, _filt_up);
  enc.set_buffer(2, scratch);
  enc.set_constant(3, T);
  enc.set_constant(4, C);
  enc.dispatch({(unsigned)C, (unsigned)(B * T2), 1}, {tx, ty, 1});

  const int n = B * T2 * C;
  enc.set_function(_fn_snake);
  enc.set_buffer(0, scratch);
  enc.set_buffer(1, s.ealpha);
  enc.set_buffer(2, s.rbeta);
  enc.set_buffer(3, scratch);
  enc.set_constant(4, C);
  enc.set_constant(5, n);
  enc.dispatch({(unsigned)C, (unsigned)(B * T2), 1}, {tx, ty, 1});

  // (2T - 1) / 2 + 1 == T, so the round trip is length-preserving.
  enc.set_function(_fn_down2);
  enc.set_buffer(0, scratch);
  enc.set_buffer(1, _filt_down);
  enc.set_buffer(2, out);
  enc.set_constant(3, T2);
  enc.set_constant(4, C);
  enc.set_constant(5, T);
  enc.dispatch({(unsigned)C, (unsigned)(B * T), 1}, {tx, ty, 1});
}

bool
MetalMiniMaxH3AudioVae::ensure_scratch_(int B, int T)
{
  if (_s.b == B && _s.t == T) { return true; }
  const Config& c = _cfg;
  // The widest thing any single buffer has to hold. Two candidates
  // compete: an activation (B * length * channels), and the transposed
  // convolutions' pre-fold GEMM output, which is `k` times an activation
  // at the INPUT length. The latter wins at every stage past the first.
  std::size_t need = (std::size_t)B * T * std::max(c.latent_dim,
                                                   c.decoder_dim);
  std::size_t act  = need;
  int L = T;
  for (std::size_t i = 0; i < _stages.size(); ++i) {
    const Conv1d& u = _stages[i].up;
    need = std::max(need, (std::size_t)B * L * u.k * u.cout);
    L = (L - 1) * _stages[i].rate - 2 * u.pad + u.k;
    const std::size_t a = (std::size_t)B * L * u.cout;
    need = std::max(need, a);
    act  = std::max(act, a);
  }
  Scratch s;
  s.b = B;
  s.t = T;
  // 16 MB of im2col band. The gather is `k` times the activation -- 11x
  // at the widest resblock -- so banding is what keeps it a fixed cost
  // instead of one that scales with the clip.
  s.col_cap = 8u << 20;
  s.x0  = _mc->make_shared_buffer(need * 2);
  s.acc = _mc->make_shared_buffer(need * 2);
  s.w   = _mc->make_shared_buffer(need * 2);
  s.t1  = _mc->make_shared_buffer(need * 2);
  s.t2  = _mc->make_shared_buffer(need * 2);
  s.up  = _mc->make_shared_buffer(act * (std::size_t)kResampleRatio * 2);
  s.col = _mc->make_shared_buffer(s.col_cap * 2);
  if (s.x0.empty() || s.acc.empty() || s.w.empty() || s.t1.empty() ||
      s.t2.empty() || s.up.empty() || s.col.empty()) {
    return false;
  }
  _s = std::move(s);
  return true;
}

void
MetalMiniMaxH3AudioVae::linear_(ComputeEncoder& enc, const SharedBuffer& x,
                                const SharedBuffer& w, const SharedBuffer& b,
                                const SharedBuffer& y, std::size_t y_off,
                                int M, int N, int K)
{
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, b.empty() ? w : b);
  enc.set_buffer(3, y, y_off * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, b.empty() ? 0 : 1);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

void
MetalMiniMaxH3AudioVae::layer_norm_(ComputeEncoder& enc, const SharedBuffer& x,
                                    const SharedBuffer& w,
                                    const SharedBuffer& b,
                                    const SharedBuffer& y, int rows, int C)
{
  enc.set_function(_fn_ln);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, b);
  enc.set_buffer(3, y);
  enc.set_constant(4, C);
  enc.set_constant(5, 1e-5f);
  enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
}

int
MetalMiniMaxH3AudioVae::encoded_frames(int samples) const
{
  const int h = _cfg.hop();
  if (samples <= 0 || h <= 0) { return 0; }
  return (samples + h - 1) / h;
}

// Build the encoder half. Split out of load() because `t2va` / `fl2va`
// never touch it, and it is ~110 tensors that would otherwise be read,
// folded and held for every generation that has no reference soundtrack.
bool
MetalMiniMaxH3AudioVae::build_encoder_(std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (_enc_loaded) { return true; }
  if (!_ws) { return fail("no weight set"); }
  WeightSet& ws = *_ws;
  // Probed the way conv1d_ RESOLVES it, not the way an un-normed
  // checkpoint would spell it. Every convolution in this VAE is weight-
  // normalized -- the released audio VAE stores `weight_g` / `weight_v`
  // for every convolution it has -- so asking for the folded name found
  // nothing and reported a checkpoint with a complete 147-tensor encoder
  // as carrying none. The decoder never noticed because it goes straight
  // through conv1d_, which has handled both spellings all along; only
  // this probe was written against the other one, and its cost was that
  // every `ref2va` reference with a soundtrack was silently skipped.
  if (ws.src().info("encoder.block.0.weight_v") == nullptr &&
      ws.src().info("encoder.block.0.weight") == nullptr) {
    return fail("this checkpoint carries no audio encoder");
  }
  const Config& c = _cfg;

  // The trunk. Widths double per stage from encoder_dim, and each stage
  // convolution is kernel 2*stride padded by ceil(stride/2) -- NOT the
  // "same" rule the rest of this file derives, so it is passed in.
  _enc_in = conv1d_(ws, "encoder.block.0", false, 1, 1);
  if (_enc_in.empty()) { return fail("encoder.block.0 failed to load"); }

  _enc_stages.clear();
  int width = c.encoder_dim;
  for (std::size_t i = 0; i < c.down_rates.size(); ++i) {
    const int rate = c.down_rates[i];
    if (rate <= 0) { return fail("non-positive encoder rate"); }
    const std::string base = "encoder.block." + std::to_string(i + 1);
    EncStage st;
    st.rate = rate;
    st.cin  = width;
    st.cout = width * 2;
    for (int u = 0; u < 3; ++u) {
      const std::string ub = base + ".block." + std::to_string(u) + ".block.";
      const int dil = u < (int)c.enc_dilations.size() ? c.enc_dilations[u] : 1;
      st.units[u].act1  = snake1d_(ws, ub + "0");
      st.units[u].conv1 = conv1d_(ws, ub + "1", false, dil, 1);
      st.units[u].act2  = snake1d_(ws, ub + "2");
      st.units[u].conv2 = conv1d_(ws, ub + "3", false, 1, 1);
      if (st.units[u].act1.empty() || st.units[u].act2.empty() ||
          st.units[u].conv1.empty() || st.units[u].conv2.empty()) {
        return fail("encoder residual unit " + ub + " failed to load");
      }
    }
    st.act  = snake1d_(ws, base + ".block.3");
    st.down = conv1d_(ws, base + ".block.4", false, 1, 1, rate,
                      (rate + 1) / 2);
    if (st.act.empty() || st.down.empty()) {
      return fail("encoder stage " + base + " failed to load");
    }
    if (st.down.k != 2 * rate) {
      return fail("encoder stage " + base + " has kernel " +
                  std::to_string(st.down.k) + ", expected " +
                  std::to_string(2 * rate));
    }
    width = st.down.cout;
    _enc_stages.push_back(std::move(st));
  }
  const int nb = (int)c.down_rates.size() + 1;
  _enc_act = snake1d_(ws, "encoder.block." + std::to_string(nb));
  _enc_out = conv1d_(ws, "encoder.block." + std::to_string(nb + 1), false, 1, 1);
  if (_enc_act.empty() || _enc_out.empty()) {
    return fail("the encoder trunk's output stage failed to load");
  }
  if (_enc_out.cout != c.latent_dim) {
    return fail("the encoder trunk emits " + std::to_string(_enc_out.cout) +
                " channels, but latent_dim is " +
                std::to_string(c.latent_dim));
  }

  // The posterior head. Its linears are stored [out, in], which is
  // already the [N, K] order the GEMM wants, so they load as f16 with no
  // transpose -- unlike the convolutions above.
  auto lin = [&](const char* nm) { return f16_(ws, nm); };
  _pre.n1w = lin("pre_block.norm1.weight");
  _pre.n1b = lin("pre_block.norm1.bias");
  _pre.n3w = lin("pre_block.norm3.weight");
  _pre.n3b = lin("pre_block.norm3.bias");
  _pre.n2w = lin("pre_block.norm2.weight");
  _pre.n2b = lin("pre_block.norm2.bias");
  _pre.aow = lin("pre_block.attn.proj.weight");
  _pre.aob = lin("pre_block.attn.proj.bias");
  _pre.pw  = lin("pre_block.proj.weight");
  _pre.pb  = lin("pre_block.proj.bias");
  _pre.mnw = lin("pre_block.mlp.norm.weight");
  _pre.mnb = lin("pre_block.mlp.norm.bias");
  _pre.w0  = lin("pre_block.mlp.w0.weight");
  _pre.b0  = lin("pre_block.mlp.w0.bias");
  _pre.w1  = lin("pre_block.mlp.w1.weight");
  _pre.b1  = lin("pre_block.mlp.w1.bias");
  _pre.w2  = lin("pre_block.mlp.w2.weight");
  _pre.b2  = lin("pre_block.mlp.w2.bias");
  _pre.qb  = lin("pre_block.attn.q_bias");
  _pre.vb  = lin("pre_block.attn.v_bias");
  if (_pre.n1w.empty() || _pre.n3w.empty() || _pre.n2w.empty() ||
      _pre.aow.empty() || _pre.pw.empty() || _pre.w0.empty() ||
      _pre.w1.empty() || _pre.w2.empty() || _pre.qb.empty() ||
      _pre.vb.empty()) {
    return fail("the encoder's posterior head failed to load");
  }
  {
    const auto* wi = ws.src().info("pre_block.mlp.w0.weight");
    if (wi == nullptr || wi->shape.size() != 2) {
      return fail("pre_block.mlp.w0.weight is not a matrix");
    }
    _pre.hidden = (int)wi->shape[0];
  }
  // The fused qkv is stored as ONE [3*latent_dim, latent_dim] matrix; it
  // is sliced into three so each is a plain linear the GEMM can take,
  // and because k's bias is a frozen ZERO buffer rather than a
  // parameter -- splitting is what lets k simply have no bias at all.
  {
    const std::string nm = "pre_block.attn.qkv.weight";
    const auto* wi = ws.src().info(nm);
    if (wi == nullptr || wi->shape.size() != 2) {
      return fail("pre_block.attn.qkv.weight is not a matrix");
    }
    const int rows = (int)wi->shape[0];
    const int cols = (int)wi->shape[1];
    if (rows != 3 * cols || cols != c.latent_dim) {
      return fail("pre_block.attn.qkv.weight is not [3*latent_dim, "
                  "latent_dim]");
    }
    SharedBuffer* dst[3] = {&_pre.qw, &_pre.kw, &_pre.vw};
    for (int part = 0; part < 3; ++part) {
      *dst[part] = ws.derived(
          std::string(kKey) + "qkv" + std::to_string(part) + "|" + nm,
          [&]() -> SharedBuffer {
            std::vector<float> v;
            if (!read_f32_(ws, _mc, nm, &v)) { return {}; }
            if (v.size() < (std::size_t)rows * cols) { return {}; }
            SharedBuffer o = _mc->make_shared_buffer((std::size_t)cols * cols * 2);
            if (o.empty()) { return {}; }
            auto* d = static_cast<_Float16*>(o.contents());
            const float* src = v.data() + (std::size_t)part * cols * cols;
            for (std::size_t i = 0; i < (std::size_t)cols * cols; ++i) {
              d[i] = (_Float16)src[i];
            }
            return o;
          });
      if (dst[part]->empty()) { return fail("qkv slice " +
                                            std::to_string(part) + " failed"); }
    }
  }
  _mean_proj = conv1d_(ws, "mean_proj", false, 1, 1);
  if (_mean_proj.empty()) { return fail("mean_proj failed to load"); }

  _enc_loaded = true;
  return true;
}

bool
MetalMiniMaxH3AudioVae::encode(const float* pcm, int stereo, int samples,
                               std::vector<float>* latents, int* frames_out,
                               std::string* err,
                               std::vector<std::vector<float>>* taps)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (pcm == nullptr || latents == nullptr) { return fail("null argument"); }
  if (stereo <= 0 || samples <= 0) { return fail("empty waveform"); }
  if (!build_encoder_(err)) { return false; }

  const Config& c = _cfg;
  const int B  = stereo;
  const int H  = c.hop();
  const int T  = encoded_frames(samples);
  const int S  = T * H;                 // the right-padded length
  const int ZC = c.latent_channels;
  const int LD = c.latent_dim;
  if (frames_out != nullptr) { *frames_out = T; }

  // The widest activation the trunk holds. Each stage halves time and
  // doubles width, so the product FALLS after the first stride block --
  // the input convolution's own output is the peak.
  std::size_t need = (std::size_t)B * S * c.encoder_dim;
  {
    int L = S, W = c.encoder_dim;
    for (const EncStage& st : _enc_stages) {
      need = std::max(need, (std::size_t)B * L * W);
      L = (L + 2 * st.down.pad - st.down.k) / st.rate + 1;
      W = st.cout;
      need = std::max(need, (std::size_t)B * L * W);
    }
    // Checked HERE rather than mid-stream: the strides multiply out to
    // the hop, so a checkpoint whose rates disagree with the decoder's
    // would silently emit a different number of latents than the layout
    // reserved rows for.
    if (L != T) {
      return fail("the encoder rates give " + std::to_string(L) +
                  " frames for " + std::to_string(S) +
                  " samples, but the hop gives " + std::to_string(T));
    }
  }
  // The posterior head widens back to 3 * latent_dim for the qkv slices.
  need = std::max(need, (std::size_t)B * T * LD);

  SharedBuffer cur = _mc->make_shared_buffer(need * 2);
  SharedBuffer t1  = _mc->make_shared_buffer(need * 2);
  SharedBuffer t2  = _mc->make_shared_buffer(need * 2);
  const std::size_t col_cap = 8u << 20;
  SharedBuffer col = _mc->make_shared_buffer(col_cap * 2);
  if (cur.empty() || t1.empty() || t2.empty() || col.empty()) {
    return fail("activation allocation failed (out of GPU memory)");
  }
  // conv_ bands through the shared scratch, so point it at ours for the
  // duration; the decoder's own scratch is sized in latent frames and
  // would be far too small for a waveform.
  Scratch saved = std::move(_s);
  _s.col        = std::move(col);
  _s.col_cap    = col_cap;

  // Restore on every exit -- an early return that left the encoder's
  // scratch installed would hand a decode a buffer sized for the wrong
  // thing.
  struct Restore {
    Scratch* slot;
    Scratch  saved;
    ~Restore() { *slot = std::move(saved); }
  } restore{&_s, std::move(saved)};

  // Mono in, one batch item per channel: planar [B, samples] -> the
  // time-major [B*S, 1] the convolution stack reads, zero-padded out to
  // the hop boundary.
  {
    auto* d = static_cast<_Float16*>(cur.contents());
    for (int b = 0; b < B; ++b) {
      for (int t = 0; t < S; ++t) {
        d[(std::size_t)b * S + t] =
            t < samples ? (_Float16)pcm[(std::size_t)b * samples + t]
                        : (_Float16)0.0f;
      }
    }
  }

  std::vector<std::size_t> tap_n;
  std::vector<SharedBuffer> tap_buf;
  auto tap = [&](const SharedBuffer& src, int n, ComputeEncoder& enc) {
    if (taps == nullptr) { return; }
    SharedBuffer dst = _mc->make_shared_buffer((std::size_t)n * 2);
    if (dst.empty()) { return; }
    enc.set_function(_fn_clamp);
    enc.set_buffer(0, src);
    enc.set_buffer(1, dst);
    enc.set_constant(2, n);
    enc.set_constant(3, -65504.0f);
    enc.set_constant(4, 65504.0f);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    tap_n.push_back((std::size_t)n);
    tap_buf.push_back(std::move(dst));
  };

  auto snake = [&](const Snake& s, const SharedBuffer& in,
                   const SharedBuffer& out, int n, int C,
                   ComputeEncoder& enc) {
    enc.set_function(_fn_snake);
    enc.set_buffer(0, in);
    enc.set_buffer(1, s.ealpha);
    enc.set_buffer(2, s.rbeta);
    enc.set_buffer(3, out);
    enc.set_constant(4, C);
    enc.set_constant(5, n);
    enc.dispatch({(unsigned)C, (unsigned)(n / C), 1}, {32, 8, 1});
  };
  auto add = [&](const SharedBuffer& a, const SharedBuffer& b_,
                 const SharedBuffer& o, int n, ComputeEncoder& enc) {
    enc.set_function(_fn_residual);
    enc.set_buffer(0, a);
    enc.set_buffer(1, b_);
    enc.set_buffer(2, o);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };

  CommandStream stream = _mc->make_command_stream();
  int L = S;
  int W = c.encoder_dim;
  {
    ComputeEncoder enc = stream.begin_compute();
    conv_(enc, _enc_in, cur, t1, B, S, S);
    std::swap(cur, t1);
    tap(cur, B * L * W, enc);

    for (const EncStage& st : _enc_stages) {
      for (int u = 0; u < 3; ++u) {
        const EncResUnit& ru = st.units[u];
        snake(ru.act1, cur, t1, B * L * W, W, enc);
        conv_(enc, ru.conv1, t1, t2, B, L, L);
        snake(ru.act2, t2, t1, B * L * W, W, enc);
        conv_(enc, ru.conv2, t1, t2, B, L, L);
        add(cur, t2, t1, B * L * W, enc);
        std::swap(cur, t1);
      }
      snake(st.act, cur, t1, B * L * W, W, enc);
      const int Lo = (L + 2 * st.down.pad - st.down.k) / st.rate + 1;
      conv_(enc, st.down, t1, t2, B, L, Lo);
      std::swap(cur, t2);
      L = Lo;
      W = st.cout;
      tap(cur, B * L * W, enc);
    }
    snake(_enc_act, cur, t1, B * L * W, W, enc);
    conv_(enc, _enc_out, t1, t2, B, L, L);
    std::swap(cur, t2);
    tap(cur, B * T * LD, enc);
  }

  // ---- the posterior head ------------------------------------------
  //
  // `cur` is [B*T, latent_dim], which is already the [B, T, C] the
  // reference's `pre_block` takes (it transposes because its trunk is
  // channel-first; this one is not).
  const int NH = c.enc_attn_heads;
  const int HD = NH > 0 ? LD / NH : 0;
  if (NH <= 0 || HD <= 0 || HD * NH != LD) {
    return fail("latent_dim is not divisible by the attention head count");
  }
  if (HD % 32 != 0 || HD > 512) {
    return fail("the attention head width " + std::to_string(HD) +
                " is outside what sdpa_causal supports");
  }
  SharedBuffer qh = _mc->make_shared_buffer((std::size_t)B * T * LD * 2);
  SharedBuffer kh = _mc->make_shared_buffer((std::size_t)B * T * LD * 2);
  SharedBuffer vh = _mc->make_shared_buffer((std::size_t)B * T * LD * 2);
  SharedBuffer ao = _mc->make_shared_buffer((std::size_t)B * T * LD * 2);
  SharedBuffer sm = _mc->make_shared_buffer((std::size_t)B * T *
                                            std::max(ZC, _pre.hidden) * 2);
  SharedBuffer sm2 = _mc->make_shared_buffer((std::size_t)B * T *
                                             std::max(ZC, _pre.hidden) * 2);
  SharedBuffer res = _mc->make_shared_buffer((std::size_t)B * T * ZC * 2);
  if (qh.empty() || kh.empty() || vh.empty() || ao.empty() || sm.empty() ||
      sm2.empty() || res.empty()) {
    return fail("posterior-head allocation failed");
  }
  {
    ComputeEncoder enc = stream.begin_compute();
    layer_norm_(enc, cur, _pre.n1w, _pre.n1b, t1, B * T, LD);
    // q and v carry a bias; k's is the checkpoint's frozen zero buffer,
    // so it simply has none.
    linear_(enc, t1, _pre.qw, _pre.qb, t2, 0, B * T, LD, LD);
    // Token-major [T, NH, HD] -> head-major [NH, T, HD], per batch item,
    // which is the layout sdpa_causal indexes.
    for (int b = 0; b < B; ++b) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, t2, (std::size_t)b * T * LD * 2);
      enc.set_buffer(1, qh, (std::size_t)b * T * LD * 2);
      enc.set_constant(2, T);
      enc.set_constant(3, NH);
      enc.set_constant(4, HD);
      enc.dispatch({(unsigned)HD, (unsigned)NH, (unsigned)T}, {32, 1, 1});
    }
    linear_(enc, t1, _pre.kw, {}, t2, 0, B * T, LD, LD);
    for (int b = 0; b < B; ++b) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, t2, (std::size_t)b * T * LD * 2);
      enc.set_buffer(1, kh, (std::size_t)b * T * LD * 2);
      enc.set_constant(2, T);
      enc.set_constant(3, NH);
      enc.set_constant(4, HD);
      enc.dispatch({(unsigned)HD, (unsigned)NH, (unsigned)T}, {32, 1, 1});
    }
    linear_(enc, t1, _pre.vw, _pre.vb, t2, 0, B * T, LD, LD);
    for (int b = 0; b < B; ++b) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, t2, (std::size_t)b * T * LD * 2);
      enc.set_buffer(1, vh, (std::size_t)b * T * LD * 2);
      enc.set_constant(2, T);
      enc.set_constant(3, NH);
      enc.set_constant(4, HD);
      enc.dispatch({(unsigned)HD, (unsigned)NH, (unsigned)T}, {32, 1, 1});
    }

    const float scale = 1.0f / std::sqrt((float)HD);
    for (int b = 0; b < B; ++b) {
      const std::size_t off = (std::size_t)b * T * LD * 2;
      enc.set_function(_fn_sdpa_causal);
      enc.set_buffer(0, qh, off);
      enc.set_buffer(1, kh, off);
      enc.set_buffer(2, vh, off);
      enc.set_buffer(3, ao, off);
      enc.set_constant(4, scale);
      enc.set_constant(5, T);
      enc.set_constant(6, HD);
      enc.set_constant(7, NH);
      enc.set_constant(8, NH);
      enc.set_constant(9, T);
      enc.set_constant(10, 0);
      enc.set_constant(11, T);
      enc.dispatch({32, (unsigned)NH, (unsigned)T}, {32, 1, 1});
      // Heads mean-pooled away, then the head width average-pooled down
      // to the latent width.
      enc.set_function(_fn_head_pool);
      enc.set_buffer(0, ao, off);
      enc.set_buffer(1, sm, (std::size_t)b * T * ZC * 2);
      enc.set_constant(2, NH);
      enc.set_constant(3, T);
      enc.set_constant(4, HD);
      enc.set_constant(5, ZC);
      enc.dispatch({(unsigned)ZC, (unsigned)T, 1}, {32, 8, 1});
    }
    linear_(enc, sm, _pre.aow, _pre.aob, sm2, 0, B * T, ZC, ZC);

    // The residual arm: a plain linear over its own norm of the INPUT.
    layer_norm_(enc, cur, _pre.n3w, _pre.n3b, t1, B * T, LD);
    linear_(enc, t1, _pre.pw, _pre.pb, res, 0, B * T, ZC, LD);
    add(res, sm2, res, B * T * ZC, enc);

    // The GeGLU MLP, over norm2 and then the MLP's own norm -- two
    // LayerNorms back to back, which is what the checkpoint stores.
    layer_norm_(enc, res, _pre.n2w, _pre.n2b, sm, B * T, ZC);
    layer_norm_(enc, sm, _pre.mnw, _pre.mnb, sm2, B * T, ZC);
    linear_(enc, sm2, _pre.w0, _pre.b0, t1, 0, B * T, _pre.hidden, ZC);
    linear_(enc, sm2, _pre.w1, _pre.b1, t2, 0, B * T, _pre.hidden, ZC);
    enc.set_function(_fn_geglu);
    enc.set_buffer(0, t1);
    enc.set_buffer(1, t2);
    enc.set_buffer(2, t1);
    enc.set_constant(3, B * T * _pre.hidden);
    enc.dispatch({(unsigned)(B * T * _pre.hidden), 1, 1}, {256, 1, 1});
    linear_(enc, t1, _pre.w2, _pre.b2, sm, 0, B * T, ZC, _pre.hidden);
    add(res, sm, res, B * T * ZC, enc);
    tap(res, B * T * ZC, enc);

    // mean_proj is a 1x1 convolution, i.e. a linear.
    linear_(enc, res, _mean_proj.w, _mean_proj.b, sm, 0, B * T, ZC, ZC);
  }

  std::string serr;
  if (!stream.commit().wait_ok(&serr)) {
    return fail("audio encode failed: " + serr);
  }

  latents->resize((std::size_t)B * T * ZC);
  {
    const auto* s = static_cast<const _Float16*>(sm.contents());
    for (std::size_t i = 0; i < latents->size(); ++i) {
      (*latents)[i] = (float)s[i];
    }
  }
  if (taps != nullptr) {
    taps->clear();
    for (std::size_t i = 0; i < tap_buf.size(); ++i) {
      std::vector<float> v(tap_n[i]);
      const auto* s = static_cast<const _Float16*>(tap_buf[i].contents());
      for (std::size_t j = 0; j < v.size(); ++j) { v[j] = (float)s[j]; }
      taps->push_back(std::move(v));
    }
  }
  return true;
}

bool
MetalMiniMaxH3AudioVae::decode(const float* z, int stereo, int frames,
                               std::vector<float>* pcm, std::string* err,
                               std::vector<std::vector<float>>* taps)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (z == nullptr || pcm == nullptr) { return fail("null argument"); }
  if (stereo <= 0 || frames <= 0) { return fail("empty latent"); }
  if (_stages.empty()) { return fail("no decoder stages"); }
  if (!ensure_scratch_(stereo, frames)) {
    return fail("activation allocation failed (out of GPU memory)");
  }
  const Config& c = _cfg;
  const int B = stereo, T = frames;
  const int ZC = c.latent_channels;

  // Channel-first [B, ZC, T] -> time-major [B*T, ZC]. One host pass over
  // the smallest tensor in the whole decode; the GPU alternative would be
  // a transpose kernel that exists only here.
  {
    auto* d = static_cast<_Float16*>(_s.t1.contents());
    for (int b = 0; b < B; ++b) {
      for (int t = 0; t < T; ++t) {
        for (int ch = 0; ch < ZC; ++ch) {
          d[((std::size_t)b * T + t) * ZC + ch] =
              (_Float16)z[((std::size_t)b * ZC + ch) * T + t];
        }
      }
    }
  }

  // The three interchangeable full-width buffers: `A` is the live stage
  // activation, `acc` sums the three parallel AMP blocks and `w` is the
  // one being built. Only pointers move.
  SharedBuffer* A   = &_s.x0;
  SharedBuffer* ACC = &_s.acc;
  SharedBuffer* W   = &_s.w;

  auto add = [&](const SharedBuffer& a, const SharedBuffer& b_,
                 const SharedBuffer& o, int n, ComputeEncoder& enc) {
    enc.set_function(_fn_residual);
    enc.set_buffer(0, a);
    enc.set_buffer(1, b_);
    enc.set_buffer(2, o);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };

  // Taps are COPIED out as they are produced. The three working buffers
  // rotate, so a pointer recorded at stage 0 would be holding stage 6's
  // activation by the time the stream is read back -- every tap would
  // look identical and the bisection would silently agree with itself.
  std::vector<std::size_t> tap_n;
  std::vector<SharedBuffer> tap_buf;
  auto tap = [&](const SharedBuffer& src, int n, ComputeEncoder& enc) {
    if (taps == nullptr) { return; }
    SharedBuffer dst = _mc->make_shared_buffer((std::size_t)n * 2);
    if (dst.empty()) { return; }
    // clamp at the f16 limits is an exact copy of any finite value, and
    // saves a copy kernel that would exist only for this test hook.
    enc.set_function(_fn_clamp);
    enc.set_buffer(0, src);
    enc.set_buffer(1, dst);
    enc.set_constant(2, n);
    enc.set_constant(3, -65504.0f);
    enc.set_constant(4, 65504.0f);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    tap_n.push_back((std::size_t)n);
    tap_buf.push_back(std::move(dst));
  };

  CommandStream stream = _mc->make_command_stream();
  int L = T;
  int C = c.decoder_dim;
  {
    ComputeEncoder enc = stream.begin_compute();
    // dec_in_proj is a 1x1 convolution, so it needs no gather at all.
    gemm_(enc, _s.t1, _dec_in, *A, 0, B * T, c.latent_dim, ZC, true);
    conv_(enc, _conv_pre, *A, *ACC, B, T, T);
    std::swap(A, ACC);
    tap(*A, B * L * C, enc);

    for (std::size_t si = 0; si < _stages.size(); ++si) {
      const UpStage& st = _stages[si];
      const Conv1d& u = st.up;
      const int Lo = (L - 1) * st.rate - 2 * u.pad + u.k;
      // The transposed convolution in two halves: a dense GEMM that
      // computes every tap's contribution, then the fold that scatters
      // them by stride and sums the overlaps (and adds the bias).
      gemm_(enc, *A, u, _s.t1, 0, B * L, u.k * u.cout, u.cin, false);
      {
        const unsigned tx = (unsigned)std::min(u.cout, 32);
        const unsigned ty = std::max(1u, 256u / tx);
        enc.set_function(_fn_col2im);
        enc.set_buffer(0, _s.t1);
        enc.set_buffer(1, u.b.empty() ? _s.t1 : u.b);
        enc.set_buffer(2, *ACC);
        enc.set_constant(3, L);
        enc.set_constant(4, u.cout);
        enc.set_constant(5, u.k);
        enc.set_constant(6, st.rate);
        enc.set_constant(7, u.pad);
        enc.set_constant(8, Lo);
        enc.set_constant(9, u.b.empty() ? 0 : 1);
        enc.dispatch({(unsigned)u.cout, (unsigned)(B * Lo), 1}, {tx, ty, 1});
      }
      std::swap(A, ACC);
      L = Lo;
      C = u.cout;
      const int n = B * L * C;

      // Three AMP blocks over the SAME input, averaged. They are
      // independent -- this is a parallel bank, not a stack.
      for (int j = 0; j < 3; ++j) {
        const AmpBlock& amp = st.amp[j];
        SharedBuffer* dst = (j == 0) ? ACC : W;
        for (int d = 0; d < 3; ++d) {
          const SharedBuffer& src = (d == 0) ? *A : *dst;
          activate_(enc, amp.acts[2 * d], src, _s.t2, _s.up, B, L, C);
          conv_(enc, amp.convs1[d], _s.t2, _s.t1, B, L, L);
          activate_(enc, amp.acts[2 * d + 1], _s.t1, _s.t2, _s.up, B, L, C);
          conv_(enc, amp.convs2[d], _s.t2, _s.t1, B, L, L);
          add(src, _s.t1, *dst, n, enc);
        }
        if (j > 0) { add(*ACC, *W, *ACC, n, enc); }
      }
      {
        enc.set_function(_fn_scale);
        enc.set_buffer(0, *ACC);
        enc.set_constant(1, n);
        enc.set_constant(2, 1.0f / 3.0f);
        enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
      }
      std::swap(A, ACC);
      tap(*A, n, enc);
    }

    activate_(enc, _act_post, *A, *ACC, _s.up, B, L, C);
    conv_(enc, _conv_post, *ACC, _s.t1, B, L, L);
    {
      const int n = B * L * _conv_post.cout;
      enc.set_function(_fn_clamp);
      enc.set_buffer(0, _s.t1);
      enc.set_buffer(1, _s.t1);
      enc.set_constant(2, n);
      enc.set_constant(3, -1.0f);
      enc.set_constant(4, 1.0f);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    }
  }
  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty()
                    ? std::string("MiniMax-H3 audio VAE decode failed")
                    : gpu_err);
  }

  // conv_post emits one channel, so [B*L, 1] is already the planar
  // [stereo, samples] the caller wants.
  {
    const auto* s = static_cast<const _Float16*>(_s.t1.contents());
    pcm->resize((std::size_t)B * L);
    for (std::size_t i = 0; i < pcm->size(); ++i) { (*pcm)[i] = (float)s[i]; }
  }
  if (taps != nullptr) {
    taps->clear();
    for (std::size_t i = 0; i < tap_buf.size(); ++i) {
      const auto* s = static_cast<const _Float16*>(tap_buf[i].contents());
      std::vector<float> v(tap_n[i]);
      for (std::size_t k = 0; k < tap_n[i]; ++k) { v[k] = (float)s[k]; }
      taps->push_back(std::move(v));
    }
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
