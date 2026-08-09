#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/comfy-checkpoint.h"
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
  if (fs::exists(p / "FL2VA" / "audio_vae" / "model.safetensors")) {
    return (p / "FL2VA" / "audio_vae").string();
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
                                bool transposed, int dilation, int rate)
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
  c.b   = f16_(ws, nm + ".bias");               // may legitimately be empty

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
  m->_fn_gemm     = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_im2col   = m->_lib_avae.function("im2col_1d_tc_f16");
  m->_fn_col2im   = m->_lib_avae.function("col2im_1d_tc_f16");
  m->_fn_snake    = m->_lib_avae.function("snake_beta_f16");
  m->_fn_up2      = m->_lib_avae.function("resample_up2_f16");
  m->_fn_down2    = m->_lib_avae.function("resample_down2_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_scale    = m->_lib_elt.function("scale_inplace_f16");
  m->_fn_clamp    = m->_lib_elt.function("clamp_f16");
  if (!m->_fn_gemm.valid() || !m->_fn_im2col.valid() ||
      !m->_fn_col2im.valid() || !m->_fn_snake.valid() || !m->_fn_up2.valid() ||
      !m->_fn_down2.valid() || !m->_fn_residual.valid() ||
      !m->_fn_scale.valid() || !m->_fn_clamp.valid()) {
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

void
MetalMiniMaxH3AudioVae::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                              const Conv1d& c, const SharedBuffer& y,
                              std::size_t y_off, int M, int N, int K,
                              bool bias)
{
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
                              int B, int T)
{
  const int rows = B * T;
  const std::size_t per_row = (std::size_t)c.k * c.cin;
  std::size_t band = per_row > 0 ? _s.col_cap / per_row : (std::size_t)rows;
  band = std::min(std::max<std::size_t>(band, 1), (std::size_t)rows);
  const bool bias = !c.b.empty();
  for (std::size_t r0 = 0; r0 < (std::size_t)rows; r0 += band) {
    const int n = (int)std::min(band, (std::size_t)rows - r0);
    enc.set_function(_fn_im2col);
    enc.set_buffer(0, in);
    enc.set_buffer(1, _s.col);
    enc.set_constant(2, T);
    enc.set_constant(3, c.cin);
    enc.set_constant(4, c.k);
    enc.set_constant(5, c.dilation);
    enc.set_constant(6, c.pad);
    enc.set_constant(7, 1);
    enc.set_constant(8, T);
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
    conv_(enc, _conv_pre, *A, *ACC, B, T);
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
          conv_(enc, amp.convs1[d], _s.t2, _s.t1, B, L);
          activate_(enc, amp.acts[2 * d + 1], _s.t1, _s.t2, _s.up, B, L, C);
          conv_(enc, amp.convs2[d], _s.t2, _s.t1, B, L);
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
    conv_(enc, _conv_post, *ACC, _s.t1, B, L);
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
