#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"

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
#include <cstdlib>
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

constexpr const char* kKey = "minimax-h3-vvae/bf16|";

// The Comfy-Org single-file video VAE keeps the wrapper config (and,
// nested under "source_config", the net's) under this `__metadata__`
// key; its presence identifies the file.
constexpr const char* kComfyKey = "minimax_h3_video_vae";

inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

inline float
bf16_to_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

SharedBuffer
to_bf16_(const MetalLlamaWeights& wts, MetalCompute* mc, const std::string& nm)
{
  const auto* info = wts.info(nm);
  if (info == nullptr || info->shape.empty()) { return {}; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { return {}; }
  if (info->dtype == "BF16") { return raw; }
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  auto* d = static_cast<std::uint16_t*>(out.contents());
  if (info->dtype == "F32") {
    const auto* s = static_cast<const float*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_(s[i]); }
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_((float)s[i]); }
  } else {
    return {};
  }
  return out;
}

std::string
blk_(int i, const char* rest)
{
  return "decoder.transformer_blocks." + std::to_string(i) + "." + rest;
}

}  // namespace

std::string
MetalMiniMaxH3VideoVae::resolve_vae_dir(const std::string& path)
{
  namespace fs = std::filesystem;
  fs::path p(path);
  // The Comfy-Org single file, probed first for the same reason as the
  // DiT's: it is self-describing, where a directory beside it is not.
  // Unlike the DiT, this conversion does NOT repack the tensors -- it is
  // the released net, cast to fp16 -- so the only thing that changes
  // here is where the config comes from.
  {
    const std::string f =
        comfy::resolve_component(path, "vae", kComfyKey, {"video_vae"});
    if (!f.empty()) { return f; }
  }
  if (!fs::is_directory(p)) { return path; }
  if (fs::exists(p / "model.safetensors") && fs::exists(p / "config.json") &&
      !fs::exists(p / "source")) {
    return p.string();                                 // already source/
  }
  if (fs::exists(p / "source" / "model.safetensors")) {
    return (p / "source").string();                    // video_vae/
  }
  if (fs::exists(p / "video_vae" / "source" / "model.safetensors")) {
    return (p / "video_vae" / "source").string();      // a partition root
  }
  if (fs::exists(p / "FL2VA" / "video_vae" / "source" / "model.safetensors")) {
    return (p / "FL2VA" / "video_vae" / "source").string();
  }
  return path;
}

bool
MetalMiniMaxH3VideoVae::config_from_json(const std::string& vae_dir,
                                         Config& out, std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  fs::path p(resolve_vae_dir(vae_dir));
  // `cfg` is the source net's config (diffusers `source/config.json`);
  // `wrap` is the wrapper's, which is where the clip/tile geometry and
  // the latent whitening live. The released layout keeps them in two
  // files one directory apart; the Comfy-Org file carries both in one
  // metadata blob, with the source config nested under "source_config".
  FlexData cfg, wrap;
  if (comfy::is_component(p.string(), kComfyKey)) {
    std::string cerr;
    if (!comfy::metadata_json(p.string(), kComfyKey, wrap, &cerr)) {
      return fail(cerr);
    }
    if (!wrap.is_object() || !wrap.as_object().contains("source_config")) {
      return fail(p.string() + ": __metadata__ has no 'source_config'");
    }
    cfg = wrap.as_object().at("source_config");
    if (!cfg.is_object()) {
      return fail(p.string() + ": 'source_config' is not an object");
    }
    // NOTE: vae_tile_size / vae_tile_overlap_min are NOT in this blob.
    // They fall back to the Config defaults, which are the released
    // 256 / 64 -- and they must, because a tile's rope is built from its
    // own extent, so a different tiling is a different function rather
    // than a different seam.
  } else {
    if (fs::is_directory(p)) { p = p / "config.json"; }
    std::ifstream f(p);
    if (!f) { return fail("cannot open " + p.string()); }
    try {
      cfg = FlexData::from_json(f);
    } catch (...) {
      return fail("cannot parse " + p.string());
    }
    if (!cfg.is_object()) { return fail(p.string() + " is not a JSON object"); }
    std::ifstream pf(p.parent_path().parent_path() / "config.json");
    if (pf) {
      try {
        wrap = FlexData::from_json(pf);
      } catch (...) {
        wrap = FlexData::make_null();
      }
    }
  }
  auto o = cfg.as_object();
  auto gi = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  out.z_channels   = gi("z_channels", 24);
  out.out_channels = gi("out_ch", 3);
  out.patch        = gi("vae_ratio", 16);
  out.patch_t      = gi("vae_ratio_t", 4);
  // The decoder MUST be the ViT one: the CNN decoder this class does not
  // implement reports the same class name and the same latent geometry,
  // so running it here would load a disjoint set of tensors and fail on
  // a missing name rather than on the thing that is actually wrong.
  if (o.contains("use_vit_decoder") && !o.at("use_vit_decoder").as_bool(true)) {
    return fail("this checkpoint has a CNN decoder (use_vit_decoder false), "
                "which is not implemented here");
  }
  if (o.contains("vit_decoder_kwargs")) {
    FlexData v = o.at("vit_decoder_kwargs");
    if (v.is_object()) {
      auto vo = v.as_object();
      auto vi = [&](const char* k, int d) {
        return vo.contains(k) ? (int)vo.at(k).as_int(d) : d;
      };
      out.n_layers  = vi("num_layers", 36);
      out.n_heads   = vi("heads", 32);
      out.head_dim  = vi("dim_head", 64);
      if (vo.contains("rope_theta")) {
        out.rope_theta = vo.at("rope_theta").as_real(100.0);
      }
      if (vo.contains("rope_dim_ratio")) {
        out.rope_dim_ratio = vo.at("rope_dim_ratio").as_real(0.75);
      }
      // The released config gates the feed-forward shape on two flags
      // rather than stating it. Anything but gated SiLU is a different
      // activation on the same weights.
      if (vo.contains("ffn_use_gated") && !vo.at("ffn_use_gated").as_bool(true)) {
        return fail("ffn_use_gated is false: this decoder's feed-forward is "
                    "not the gated one implemented here");
      }
      if (vo.contains("ffn_activation_fn")) {
        const std::string a(vo.at("ffn_activation_fn").as_string("silu"));
        if (a != "silu") {
          return fail("ffn_activation_fn '" + a + "' is not silu");
        }
      }
    }
  }
  // ---- the CNN encoder ------------------------------------------------
  out.in_channels = gi("in_channels", 3);
  out.base_ch     = gi("ch", 128);
  out.res_blocks  = gi("num_res_blocks", 2);
  auto ints = [&](const char* k, std::vector<int>& dst) {
    if (!o.contains(k)) { return; }
    FlexData v = o.at(k);
    if (!v.is_array()) { return; }
    auto a = v.as_array();
    std::vector<int> got;
    for (std::size_t i = 0; i < a.size(); ++i) {
      got.push_back((int)a.at(i).as_int(1));
    }
    if (!got.empty()) { dst = std::move(got); }
  };
  ints("ch_mult", out.ch_mult);
  ints("space_down", out.space_down);
  ints("time_down", out.time_down);
  if (out.ch_mult.size() != out.space_down.size() ||
      out.ch_mult.size() != out.time_down.size()) {
    return fail("ch_mult, space_down and time_down must have one entry per "
                "level");
  }
  // The per-level factors are the only statement of where a
  // downsampling convolution exists, so a product that disagrees with
  // the advertised compression ratio means the levels are being built
  // from the wrong list -- which loads a plausible but different set of
  // tensors and only shows up as a wrong latent.
  int sprod = 1, tprod = 1;
  for (std::size_t i = 0; i < out.ch_mult.size(); ++i) {
    sprod *= out.space_down[i];
    tprod *= out.time_down[i];
  }
  if (sprod != out.patch || tprod != out.patch_t) {
    return fail("space_down/time_down multiply out to " +
                std::to_string(sprod) + "x" + std::to_string(tprod) +
                ", not the configured " + std::to_string(out.patch) + "x" +
                std::to_string(out.patch_t));
  }
  // Three encoder behaviours are configurable in the reference and the
  // implementation here hard-codes all three. Each fails silently: zero
  // padding only darkens the frame border, whole-clip group norm is a
  // good approximation of the per-frame one, and a non-causal encoder
  // merely shifts the temporal alignment.
  if (o.contains("padding_mode")) {
    const std::string pm(o.at("padding_mode").as_string("reflect"));
    if (pm != "reflect") {
      return fail("encoder padding_mode '" + pm + "' is not reflect");
    }
  }
  if (o.contains("use_t_isolated_gn") &&
      !o.at("use_t_isolated_gn").as_bool(true)) {
    return fail("use_t_isolated_gn is false: this encoder's group norm is "
                "not the per-frame one implemented here");
  }
  if (o.contains("causal_encoder") && !o.at("causal_encoder").as_bool(true)) {
    return fail("causal_encoder is false: this encoder's temporal padding "
                "is not the causal one implemented here");
  }
  if (o.contains("use_3d_conv") && !o.at("use_3d_conv").as_bool(true)) {
    return fail("use_3d_conv is false: this encoder is not the 3D one "
                "implemented here");
  }
  // The clip length, the token drop and the latent whitening live in the
  // WRAPPER's config, not the source net's (see above for where each
  // layout keeps it).
  if (wrap.is_object()) {
    auto wo = wrap.as_object();
    if (wo.contains("vae_clip_length")) {
      out.clip_length = (int)wo.at("vae_clip_length").as_int(17);
    }
    if (wo.contains("vae_token_drop")) {
      out.token_drop = (int)wo.at("vae_token_drop").as_int(3);
    }
    if (wo.contains("vae_tile_size")) {
      out.tile_size = (int)wo.at("vae_tile_size").as_int(256);
    }
    if (wo.contains("vae_tile_overlap_min")) {
      out.tile_overlap_min =
          (int)wo.at("vae_tile_overlap_min").as_int(64);
    }
    auto reals = [&](const char* k, std::vector<float>& dst) {
      if (!wo.contains(k)) { return; }
      FlexData v = wo.at(k);
      if (!v.is_array()) { return; }
      auto a = v.as_array();
      std::vector<float> got;
      for (std::size_t i = 0; i < a.size(); ++i) {
        got.push_back((float)a.at(i).as_real(0.0));
      }
      if ((int)got.size() == out.z_channels) { dst = std::move(got); }
    };
    reals("latents_mean", out.latents_mean);
    reals("latents_std", out.latents_std);
  }

  out.dim = out.n_heads * out.head_dim;
  // FeedForward(dim, mult=4): the diffusers default this decoder is
  // constructed with, so the inner width is not in the config.
  out.ffn_inner = 4 * out.dim;
  if (out.head_dim != 64) {
    return fail("only dim_head 64 is supported (the steel flash-attention "
                "entry point this decoder runs on is bd64)");
  }
  if (out.rope_dim() % 6 != 0) {
    return fail("int(dim_head * rope_dim_ratio) must be divisible by 6");
  }
  return true;
}

SharedBuffer
MetalMiniMaxH3VideoVae::weight_(WeightSet& ws, const std::string& nm)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr) { return {}; }
  if (info->dtype == "BF16") {
    return ws.tensor(nm, _mc, WeightSet::Residency::Copied, _part);
  }
  return ws.derived(std::string(kKey) + nm, [&]() -> SharedBuffer {
    return to_bf16_(ws.src(), _mc, nm);
  }, _part);
}

MetalMiniMaxH3VideoVae::Linear
MetalMiniMaxH3VideoVae::linear_(WeightSet& ws, const std::string& nm)
{
  Linear l;
  l.b = weight_(ws, nm + ".bias");
  const MetalLlamaWeights& src = ws.src();
  const auto* si = src.info(nm + ".scales");
  const auto* ci = src.info(nm + ".weight");
  if (_quant_bits > 0 && si != nullptr && ci != nullptr &&
      si->shape.size() == 2 && ci->shape.size() == 2) {
    const long gcols = ci->shape[1];
    const long scols = si->shape[1];
    const long K = scols * (long)_quant_group;
    const int bits = K > 0 ? (int)(gcols * 32 / K) : 0;
    l.bits = (bits == 8) ? 8 : 4;
    l.codes  = ws.tensor(nm + ".weight", _mc, WeightSet::Residency::Copied,
                         _part);
    l.scales = weight_(ws, nm + ".scales");
    l.qbias  = weight_(ws, nm + ".biases");
    if (!l.codes.empty() && !l.scales.empty() && !l.qbias.empty()) {
      l.quantized = true;
      return l;
    }
    l.codes = {}; l.scales = {}; l.qbias = {};
  }
  l.w = weight_(ws, nm + ".weight");
  return l;
}

// A causal conv3d as a dense-GEMM weight. The 3x3x3 kernels flatten to
// [cout, 27*cin] ordered (kt, ky, kx, cin), which is the layout
// im2col_hwc_3x3x3_reflect_tiled emits; the 1x1x1 shortcut and quant
// convolutions are already a Linear and go through linear_() untouched.
MetalMiniMaxH3VideoVae::Conv3d
MetalMiniMaxH3VideoVae::conv3d_(WeightSet& ws, const std::string& nm)
{
  Conv3d c;
  const auto* info = ws.src().info(nm + ".weight");
  if (info == nullptr || info->shape.size() != 5) { return c; }
  const auto& sh = info->shape;
  c.cout = (int)sh[0];
  c.cin  = (int)sh[1];
  const int kt = (int)sh[2], kh = (int)sh[3], kw = (int)sh[4];
  if (kt == 1 && kh == 1 && kw == 1) {
    c.spatial = false;
    c.k = c.cin;
    c.l = linear_(ws, nm);
    return c;
  }
  if (kt != 3 || kh != 3 || kw != 3) { return Conv3d{}; }
  c.spatial = true;
  c.k = 27 * c.cin;
  const int cin = c.cin, cout = c.cout;
  c.l.b = weight_(ws, nm + ".bias");
  c.l.w = ws.derived(std::string(kKey) + "c3d|" + nm,
                     [&]() -> SharedBuffer {
    // Read UNCACHED: the permuted bf16 copy below is what the model
    // keeps, and caching the f32 original next to it would hold a
    // second, four-times-larger copy of every convolution.
    SharedBuffer raw =
        ws.read(nm + ".weight", _mc, WeightSet::Residency::Copied);
    if (raw.empty()) { return {}; }
    const std::size_t n = (std::size_t)cout * cin * 27;
    // The released checkpoint stores these f32; Comfy-Org's repack
    // stores the same net fp16. Read the dtype rather than assuming,
    // because the permutation below indexes the source directly -- an
    // f32 read of fp16 bytes does not fail, it walks off the end (or,
    // with a shorter tensor, silently permutes garbage).
    const auto* wi = ws.src().info(nm + ".weight");
    const std::string dt = wi != nullptr ? wi->dtype : std::string();
    const std::size_t esz =
        dt == "F32" ? 4u : (dt == "F16" || dt == "BF16" ? 2u : 0u);
    if (esz == 0 || raw.byte_size() < n * esz) { return {}; }
    const auto* s32 = static_cast<const float*>(raw.contents());
    const auto* s16 = static_cast<const _Float16*>(raw.contents());
    const auto* sb16 = static_cast<const std::uint16_t*>(raw.contents());
    auto src = [&](std::size_t i) -> float {
      if (dt == "F32")  { return s32[i]; }
      if (dt == "F16")  { return (float)s16[i]; }
      return bf16_to_f32_(sb16[i]);
    };
    SharedBuffer out = _mc->make_shared_buffer(n * 2);
    if (out.empty()) { return {}; }
    auto* d = static_cast<std::uint16_t*>(out.contents());
    for (int o = 0; o < cout; ++o) {
      for (int t = 0; t < 3; ++t) {
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            const int tap = (t * 3 + ky) * 3 + kx;
            for (int i = 0; i < cin; ++i) {
              const std::size_t si =
                  ((((std::size_t)o * cin + i) * 3 + t) * 3 + ky) * 3 + kx;
              d[((std::size_t)o * 27 + tap) * cin + i] =
                  f32_to_bf16_(src(si));
            }
          }
        }
      }
    }
    return out;
  }, _part);
  if (c.l.w.empty()) { return Conv3d{}; }
  return c;
}

bool
MetalMiniMaxH3VideoVae::load_encoder_(WeightSet& ws)
{
  const Config& c = _cfg;
  _enc_conv_in = conv3d_(ws, "encoder.conv_in");
  _enc_conv_out = conv3d_(ws, "encoder.conv_out");
  _quant_conv   = conv3d_(ws, "quant_conv");
  _enc_norm_w   = weight_(ws, "encoder.norm_out.weight");
  _enc_norm_b   = weight_(ws, "encoder.norm_out.bias");
  if (_enc_conv_in.empty() || _enc_conv_out.empty() || _quant_conv.empty() ||
      _enc_norm_w.empty() || _enc_norm_b.empty()) {
    return false;
  }
  const std::size_t levels = c.ch_mult.size();
  _enc_levels.clear();
  _enc_levels.resize(levels);
  int prev = c.base_ch;
  for (std::size_t i = 0; i < levels; ++i) {
    EncLevel& lv = _enc_levels[i];
    lv.space = c.space_down[i];
    lv.time  = c.time_down[i];
    lv.cin   = prev;
    lv.cout  = c.base_ch * c.ch_mult[i];
    const std::string base = "encoder.down." + std::to_string(i) + ".";
    for (int j = 0; j < c.res_blocks; ++j) {
      EncResnet r;
      const std::string p = base + "block." + std::to_string(j) + ".";
      r.cin  = (j == 0) ? lv.cin : lv.cout;
      r.cout = lv.cout;
      r.n1w = weight_(ws, p + "norm1.weight");
      r.n1b = weight_(ws, p + "norm1.bias");
      r.n2w = weight_(ws, p + "norm2.weight");
      r.n2b = weight_(ws, p + "norm2.bias");
      r.c1  = conv3d_(ws, p + "conv1");
      r.c2  = conv3d_(ws, p + "conv2");
      if (r.cin != r.cout) {
        r.skip = conv3d_(ws, p + "nin_shortcut");
        if (r.skip.empty()) { return false; }
      }
      if (r.n1w.empty() || r.n1b.empty() || r.n2w.empty() || r.n2b.empty() ||
          r.c1.empty() || r.c2.empty()) {
        return false;
      }
      lv.res.push_back(std::move(r));
    }
    if (lv.space * lv.time > 1) {
      lv.down = conv3d_(ws, base + "downsample.conv");
      if (lv.down.empty()) { return false; }
    }
    prev = lv.cout;
  }
  return true;
}

MetalMiniMaxH3VideoVae::~MetalMiniMaxH3VideoVae() = default;

std::unique_ptr<MetalMiniMaxH3VideoVae>
MetalMiniMaxH3VideoVae::load(const std::string& vae_dir, MetalCompute* mc,
                             const Config& cfg)
{
  return load(WeightSet::open(resolve_vae_dir(vae_dir), nullptr), mc, cfg);
}

std::unique_ptr<MetalMiniMaxH3VideoVae>
MetalMiniMaxH3VideoVae::load(std::shared_ptr<WeightSet> ws_in, MetalCompute* mc,
                             const Config& cfg)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  WeightSet& ws = *ws_in;
  auto m =
      std::unique_ptr<MetalMiniMaxH3VideoVae>(new MetalMiniMaxH3VideoVae());
  m->_ws = std::move(ws_in);
  m->_mc = mc;
  m->_cfg = cfg;

  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_fn_gemm      = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_rms       = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_rms_heads = m->_lib_rope.function("rms_norm_heads_strided_f16");
  m->_fn_trope = m->_lib_rope.function("transpose_rope_half_part_ftab_f16");
  m->_fn_gated     = m->_lib_elt.function("gated_residual_f16");
  // The decoder's FFN is `w2(silu(gate) * up)` with w1 emitting
  // [GATE | up] -- gate FIRST. (The value-first spelling is the other
  // convention for the same weights and silently halves the FFN into
  // nonsense rather than failing.)
  m->_fn_swiglu    = m->_lib_elt.function("swiglu_split_gate_first_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_ln        = m->_lib_elt.function("layer_norm_affine_f16");
  m->_fn_bias_add  = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_gn_frames = m->_lib_elt.function("group_norm_frames_f16");
  m->_fn_im2col_r  =
      m->_lib_elt.function("im2col_hwc_3x3x3_reflect_tiled_f16");
  m->_fn_residual  = m->_lib_elt.function("residual_add_f16");
  {
    metal_compute::ComputeLibrary sdpa = mc->load_library("sdpa_bf16");
    m->_fn_sdpa = sdpa.function("sdpa_full_f16");
  }
  if (!m->_fn_gemm.valid() || !m->_fn_rms.valid() ||
      !m->_fn_rms_heads.valid() || !m->_fn_trope.valid() ||
      !m->_fn_gated.valid() || !m->_fn_swiglu.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_ln.valid() ||
      !m->_fn_bias_add.valid() || !m->_fn_sdpa.valid() ||
      !m->_fn_gn_frames.valid() || !m->_fn_im2col_r.valid() ||
      !m->_fn_residual.valid()) {
    return nullptr;
  }
  m->_lib_attn = mc->load_library("attn_steel");
  m->_attn_p = mc->make_shared_buffer(sizeof(float) * 64);
  m->_steel_ok = m->_lib_attn.valid() && !m->_attn_p.empty() &&
                 std::getenv("VPIPE_H3_NO_STEEL_ATTN") == nullptr;
  // M5: the matrix-core (NAX) flash attention, same contract as the ALU
  // steel kernel with bq/bk 64/32 instead of 32/16. This tower is head_dim
  // 64 AND bf16, a combination NAX did not carry until it was instantiated
  // for this model -- the f16 bd64 and bf16 bd128 entries are the ones the
  // vision towers and the image DiTs use. VPIPE_H3_NO_ATTN_NAX forces ALU.
  if (m->_steel_ok && mc->supports_matrix_cores() &&
      std::getenv("VPIPE_H3_NO_ATTN_NAX") == nullptr) {
    m->_lib_attn_nax = mc->load_library("attn_steel_nax");
    m->_attn_nax = m->_lib_attn_nax.valid();
  }

  {
    std::ifstream qin(std::filesystem::path(ws.dir()) / "config.json");
    if (qin) {
      FlexData fd;
      try {
        fd = FlexData::from_json(qin);
      } catch (...) {
        fd = FlexData::make_null();
      }
      if (fd.is_object()) {
        auto o = fd.as_object();
        if (o.contains("quantization")) {
          FlexData q = o.at("quantization");
          if (q.is_object()) {
            auto qo = q.as_object();
            const int b = qo.contains("bits") ? (int)qo.at("bits").as_int(0) : 0;
            if (b == 4 || b == 8) { m->_quant_bits = b; }
            if (qo.contains("group_size")) {
              m->_quant_group = (int)qo.at("group_size").as_int(64);
            }
          }
        }
      }
    }
  }
  if (m->_quant_bits > 0) {
    const std::string g = "g" + std::to_string(m->_quant_group);
    m->_lib_qmm = mc->load_library("affine_qmm_steel_bf16");
    m->_fn_qmm4 = m->_lib_qmm.function("affine_qmm_steel_w4" + g);
    m->_fn_qmm8 = m->_lib_qmm.function("affine_qmm_steel_w8" + g);
    if (!m->_fn_qmm4.valid() || !m->_fn_qmm8.valid()) { return nullptr; }
  }

  // The same dequant-once + dense matmul2d route the DiT takes; see the
  // long note on MetalMiniMaxH3Transformer's loader for why the dequant is
  // a separate pass rather than fused into a quantized matmul2d.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_H3_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid();
    if (m->_use_mma2 && m->_quant_bits > 0) {
      const std::string g = "g" + std::to_string(m->_quant_group);
      m->_lib_dequant = mc->load_library("affine_dequant_bf16");
      m->_fn_dequant4 = m->_lib_dequant.function("affine_dequant_w4" + g);
      m->_fn_dequant8 = m->_lib_dequant.function("affine_dequant_w8" + g);
      if (!m->_fn_dequant4.valid() || !m->_fn_dequant8.valid()) {
        m->_use_mma2 = false;
      }
    }
    if (const char* e = std::getenv("VPIPE_H3_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
  }

  m->_post_quant = m->linear_(ws, "post_quant_conv");
  m->_proj_in    = m->linear_(ws, "decoder.x_embedder");
  m->_proj_out   = m->linear_(ws, "decoder.proj_out");
  m->_register_tokens = m->weight_(ws, "decoder.register_tokens");
  m->_norm_out_w = m->weight_(ws, "decoder.norm_out.weight");
  m->_norm_out_b = m->weight_(ws, "decoder.norm_out.bias");
  if (m->_post_quant.empty() || m->_proj_in.empty() || m->_proj_out.empty() ||
      m->_register_tokens.empty() || m->_norm_out_w.empty() ||
      m->_norm_out_b.empty()) {
    return nullptr;
  }

  m->_blocks.resize((std::size_t)cfg.n_layers);
  for (int i = 0; i < cfg.n_layers; ++i) {
    Block& b = m->_blocks[(std::size_t)i];
    b.n1  = m->weight_(ws, blk_(i, "norm1.weight"));
    b.n2  = m->weight_(ws, blk_(i, "norm2.weight"));
    b.s1  = m->weight_(ws, blk_(i, "scale1"));
    b.s2  = m->weight_(ws, blk_(i, "scale2"));
    b.qkv = m->linear_(ws, blk_(i, "attn.to_qkv"));
    b.out = m->linear_(ws, blk_(i, "attn.to_out"));
    b.w1  = m->linear_(ws, blk_(i, "ff.w1"));
    b.w2  = m->linear_(ws, blk_(i, "ff.w2"));
    if (b.n1.empty() || b.n2.empty() || b.s1.empty() || b.s2.empty() ||
        b.qkv.empty() || b.out.empty() || b.w1.empty() || b.w2.empty()) {
      return nullptr;
    }
  }
  if (mc->session() != nullptr) {
    mc->session()->log_debug(
        fmt("MetalMiniMaxH3VideoVae: ViT decoder, {} blocks, dim {}, "
            "{} heads x {}, patch {}x{}x{}",
            cfg.n_layers, cfg.dim, cfg.n_heads, cfg.head_dim, cfg.patch_t,
            cfg.patch, cfg.patch));
  }
  return m;
}

void
MetalMiniMaxH3VideoVae::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                              std::size_t x_off, const Linear& l,
                              const SharedBuffer& y, std::size_t y_off, int M,
                              int N, int K)
{
  const bool bias = !l.b.empty();
  if (gemm_mma_(enc, x, x_off, l, y, y_off, M, N, K)) {
    if (bias) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, y_off * 2);
      enc.set_buffer(1, l.b);
      enc.set_constant(2, N);
      enc.set_constant(3, M * N);
      enc.dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
    }
    return;
  }
  if (l.quantized) {
    enc.set_function(l.bits == 8 ? _fn_qmm8 : _fn_qmm4);
    enc.set_buffer(0, l.codes);
    enc.set_buffer(1, l.scales);
    enc.set_buffer(2, l.qbias);
    enc.set_buffer(3, x, x_off * 2);
    enc.set_buffer(4, y, y_off * 2);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.set_constant(7, M);
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32), (unsigned)(((M + 31) / 32) * 2),
                  2}, {32, 2, 2});
    if (bias) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, y_off * 2);
      enc.set_buffer(1, l.b);
      enc.set_constant(2, N);
      enc.set_constant(3, M * N);
      enc.dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x, x_off * 2);
  enc.set_buffer(1, l.w);
  enc.set_buffer(2, bias ? l.b : l.w);
  enc.set_buffer(3, y, y_off * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias ? 1 : 0);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

// The DiT's gemm_mma_ without the route enum: this tower has one hot shape
// per stage rather than four competing ones, and it runs once per clip, so
// the shared K-only tile rule is used directly instead of a tuner. Same
// scratch discipline -- one [N, K] bf16 buffer, grown on demand and reused,
// safe because the dequant/matmul pairs are encoded in order and Metal
// serializes the write-after-read on it.
bool
MetalMiniMaxH3VideoVae::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& x,
                                  std::size_t x_off, const Linear& l,
                                  const SharedBuffer& y, std::size_t y_off,
                                  int M, int N, int K)
{
  if (!_use_mma2 || M < _mma_min_m || N < 16) { return false; }
  const SharedBuffer* wdense = nullptr;
  if (l.quantized) {
    const metal_compute::ComputeFunction& dq =
        (l.bits == 8) ? _fn_dequant8 : _fn_dequant4;
    if (!dq.valid()) { return false; }
    // The dequant kernel's K contract -- see the note in the DiT's
    // route_ok_. A width that is not a whole number of packed words AND of
    // quant groups would under-write each row's tail, so it stays on steel.
    const int per_word = (l.bits == 8) ? 4 : 8;
    if (K % per_word != 0 || K % _quant_group != 0) { return false; }
    const std::size_t need = (std::size_t)N * (std::size_t)K * 2;
    if (_w_deq.empty() || _w_deq.byte_size() < need) {
      _w_deq = _mc->make_shared_buffer(need);
      if (_w_deq.empty()) { return false; }
    }
    enc.set_function(dq);
    enc.set_buffer(0, l.codes);
    enc.set_buffer(1, l.scales);
    enc.set_buffer(2, l.qbias);
    enc.set_buffer(3, _w_deq);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    const unsigned words = (unsigned)(l.bits == 8 ? (K / 4) : (K / 8));
    enc.dispatch({words, (unsigned)N, 1}, {64, 1, 1});
    wdense = &_w_deq;
  } else {
    wdense = &l.w;
  }
  const bool wide = mma_use_wide_tile(N, K);
  const int RN = wide ? 256 : 128;
  enc.set_function(wide ? _fn_dense_mma_deep : _fn_dense_mma);
  enc.set_buffer(0, x, x_off * 2);
  enc.set_buffer(1, *wdense);
  enc.set_buffer(2, *wdense);      // bias slot, unread (has_bias = 0)
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
MetalMiniMaxH3VideoVae::build_rope_(int T, int h, int w, SharedBuffer& cos_out,
                                    SharedBuffer& sin_out) const
{
  // Coordinates are length-normalized to [-1, 1) per axis -- CELL
  // CENTRES, `2*(i + 0.5)/size - 1` -- and the angle carries an explicit
  // 2*pi. So the grid depends on the tile's extent, not on absolute
  // position: the same voxel decoded inside a different tile size sees
  // different angles.
  const int F = _cfg.rope_freqs();
  const int half = 3 * F;
  const int voxels = T * h * w;
  const int rows = voxels + _cfg.n_register + 1;
  auto* cb = static_cast<float*>(cos_out.contents());
  auto* sb = static_cast<float*>(sin_out.contents());
  std::vector<double> inv((std::size_t)F);
  for (int i = 0; i < F; ++i) {
    // 1 / theta^arange(0, 1, 2*num_axes/rope_dim): the exponent grid is
    // built by STEP, not by count, so it is i * (6 / rope_dim).
    inv[(std::size_t)i] =
        1.0 / std::pow(_cfg.rope_theta,
                       (double)i * 6.0 / (double)_cfg.rope_dim());
  }
  // Diagnostic only: neutralize the rope (cos 1, sin 0) so a mismatch
  // can be attributed to the rope or to the rest of the block.
  const bool no_rope = std::getenv("VPIPE_H3_VVAE_NO_ROPE") != nullptr;
  const int sizes[3] = {T, h, w};
  for (int r = 0; r < rows; ++r) {
    double coord[3] = {0.0, 0.0, 0.0};
    if (r < voxels) {
      const int idx[3] = {r / (h * w), (r / w) % h, r % w};
      for (int a = 0; a < 3; ++a) {
        coord[a] = 2.0 * (((double)idx[a] + 0.5) / (double)sizes[a]) - 1.0;
      }
    }
    // The register tokens and the trailing zero token all sit at
    // position 0, which is what `coord` already is.
    for (int a = 0; a < 3; ++a) {
      for (int i = 0; i < F; ++i) {
        const double ang =
            2.0 * M_PI * coord[a] * inv[(std::size_t)i];
        const std::size_t o = (std::size_t)r * half + (std::size_t)(a * F + i);
        cb[o] = no_rope ? 1.0f : (float)std::cos(ang);
        sb[o] = no_rope ? 0.0f : (float)std::sin(ang);
      }
    }
  }
}

bool
MetalMiniMaxH3VideoVae::ensure_scratch_(int rows, int voxels)
{
  if (_s.rows == rows) { return true; }
  const Config& c = _cfg;
  const std::size_t R = (std::size_t)rows, D = (std::size_t)c.dim;
  auto mk = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  Scratch s;
  s.rows = rows;
  const std::size_t half = (std::size_t)(3 * c.rope_freqs());
  s.rcos = _mc->make_shared_buffer(R * half * sizeof(float));
  s.rsin = _mc->make_shared_buffer(R * half * sizeof(float));
  s.x   = mk(R * D);
  s.nm  = mk(R * D);
  // Sized for the WIDER of its two uses. s.qkv holds the fused q|k|v
  // (3*dim = 6144) and is then reused for the SwiGLU output (ffn_inner =
  // 8192), which is larger here -- the DiT's ordering is the other way
  // round, and carrying that assumption over writes past the end of the
  // buffer and feeds the second feed-forward GEMM garbage.
  s.qkv = mk(R * (std::size_t)std::max(3 * c.dim, c.ffn_inner));
  s.qh  = mk(R * D);
  s.kh  = mk(R * D);
  s.vh  = mk(R * D);
  s.oh  = mk(R * D);
  s.ob  = mk(R * D);
  s.ff  = mk(R * 2 * (std::size_t)c.ffn_inner);
  s.patches = mk(R * (std::size_t)c.patch_elems());
  s.zt  = mk((std::size_t)voxels * (std::size_t)c.z_channels);
  // The q/k norms have NO affine (elementwise_affine=False), so they
  // multiply by 1. A ones vector keeps the shared per-head RMS kernel
  // rather than forking it for a missing gamma.
  s.ones = mk((std::size_t)c.head_dim);
  if (s.rcos.empty() || s.rsin.empty() || s.x.empty() || s.nm.empty() ||
      s.qkv.empty() || s.qh.empty() || s.kh.empty() || s.vh.empty() ||
      s.oh.empty() || s.ob.empty() || s.ff.empty() || s.patches.empty() ||
      s.zt.empty() || s.ones.empty()) {
    return false;
  }
  auto* o = static_cast<std::uint16_t*>(s.ones.contents());
  for (int i = 0; i < c.head_dim; ++i) { o[i] = f32_to_bf16_(1.0f); }
  _s = std::move(s);
  return true;
}

SharedBuffer
MetalMiniMaxH3VideoVae::decode(const SharedBuffer& z, int T, int h, int w,
                               std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  const int D = c.dim, NH = c.n_heads, HD = c.head_dim, FF = c.ffn_inner;
  const int ZC = c.z_channels, PE = c.patch_elems();
  if (T <= 0 || h <= 0 || w <= 0) { return fail("empty latent grid"); }
  const int voxels = T * h * w;
  const int rows = voxels + c.n_register + 1;
  if (z.byte_size() < (std::size_t)ZC * voxels * 2) {
    return fail("latents are smaller than z_channels * T * h * w");
  }
  if (!ensure_scratch_(rows, voxels)) {
    return fail("activation allocation failed (out of GPU memory)");
  }
  Scratch& s = _s;
  build_rope_(T, h, w, s.rcos, s.rsin);

  // Channel-first [C, T, h, w] -> one row per voxel [T*h*w, C]. On the
  // host: it is a transpose of a few megabytes at tile sizes, and the
  // GPU alternative would be a kernel that exists only for this.
  {
    const auto* src = static_cast<const std::uint16_t*>(z.contents());
    auto* dst = static_cast<std::uint16_t*>(s.zt.contents());
    for (int v = 0; v < voxels; ++v) {
      for (int ch = 0; ch < ZC; ++ch) {
        dst[(std::size_t)v * ZC + ch] =
            src[(std::size_t)ch * voxels + (std::size_t)v];
      }
    }
  }

  const float scale = 1.0f / std::sqrt((float)HD);
  // As in the DiT: resolve the kernel BEFORE the tiles, because a param
  // block filled for one tile and a kernel compiled for the other is a
  // wrong answer, not a fallback. Dropping to the ALU steel kernel (rather
  // than to the scalar sdpa the validity check below would reach) keeps the
  // failure a small one.
  if (_attn_nax && _attn_rows != rows) {
    metal_compute::FunctionConstants probe;
    probe.set_bool(200, (rows % 64) == 0).set_bool(201, (rows % 32) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false);
    if (!_lib_attn_nax.function("attn_steel_nax_h_bd64_bf16", probe).valid()) {
      _attn_nax = false;
    }
  }
  const int A_BQ = _attn_nax ? 64 : 32;
  const int A_BK = _attn_nax ? 32 : 16;
  bool use_steel = _steel_ok;
  if (use_steel && _attn_rows != rows) {
    struct P {
      int B, H, D, qL, kL, gqa_factor;
      float scale;
      int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
      std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
    };
    auto* p = static_cast<P*>(_attn_p.contents());
    p->B = 1; p->H = NH; p->D = HD;
    p->qL = rows; p->kL = rows;
    p->gqa_factor = 1; p->scale = scale;
    p->NQ = (rows + A_BQ - 1) / A_BQ; p->NK = (rows + A_BK - 1) / A_BK;
    p->NQ_aligned = rows / A_BQ; p->NK_aligned = rows / A_BK;
    p->qL_rem = rows - p->NQ_aligned * A_BQ;
    p->kL_rem = rows - p->NK_aligned * A_BK;
    p->qL_off = 0;
    p->Q_strides[0] = (std::int64_t)NH * rows * HD;
    p->Q_strides[1] = (std::int64_t)rows * HD;
    p->Q_strides[2] = HD;
    for (int i = 0; i < 3; ++i) {
      p->K_strides[i] = p->Q_strides[i];
      p->V_strides[i] = p->Q_strides[i];
      p->O_strides[i] = p->Q_strides[i];
    }
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, (rows % A_BQ) == 0).set_bool(201, (rows % A_BK) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false);
    _fn_attn = _attn_nax
                   ? _lib_attn_nax.function("attn_steel_nax_h_bd64_bf16", fc)
                   : _lib_attn.function("attn_steel_h_bd64_bf16", fc);
    _attn_rows = rows;
  }
  if (use_steel) { use_steel = _fn_attn.valid(); }

  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // 1. post_quant_conv is a 1x1x1 convolution, i.e. a Linear over the
    // channel axis, then the patch embedding.
    gemm_(enc, s.zt, 0, _post_quant, s.nm, 0, voxels, ZC, ZC);
    gemm_(enc, s.nm, 0, _proj_in, s.x, 0, voxels, D, ZC);
    // 2. The suffix rows: `n_register` LEARNED register tokens and one
    // all-zero token. They never reach the patch projection, but they
    // are attended over by every real token, so they are part of the
    // function and not padding -- and s.x is reused across calls, so
    // leaving them uninitialized leaves the previous decode's tail in
    // the attention. Written on the host because the projection GEMM
    // above writes rows [0, voxels) only, so the ranges are disjoint and
    // the host store lands before the stream is committed.
    std::memcpy(static_cast<std::uint16_t*>(s.x.contents()) +
                    (std::size_t)voxels * D,
                _register_tokens.contents(),
                (std::size_t)c.n_register * D * 2);
    std::memset(static_cast<std::uint16_t*>(s.x.contents()) +
                    (std::size_t)(voxels + c.n_register) * D,
                0, (std::size_t)D * 2);

    for (int L = 0; L < c.n_layers; ++L) {
      const Block& b = _blocks[(std::size_t)L];
      enc.set_function(_fn_rms);
      enc.set_buffer(0, s.x); enc.set_buffer(1, b.n1); enc.set_buffer(2, s.nm);
      enc.set_constant(3, D); enc.set_constant(4, c.norm_eps);
      enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
      gemm_(enc, s.nm, 0, b.qkv, s.qkv, 0, rows, 3 * D, D);
      // Per-head RMS with NO affine, then the partial rotate-half rope
      // over 48 of the 64 head channels. Both read straight out of the
      // fused projection.
      // The fused to_qkv is grouped PER HEAD -- [h0(q,k,v) | h1(q,k,v) |
      // ...] -- so a head sits 3*HD apart and q/k/v are HD apart WITHIN
      // it. Reading it as [all q | all k | all v] gives each head a
      // blend of its own q, k and v: attention then relates no token to
      // its neighbours, and every 16x16 patch decodes on its own, which
      // reads as a tile grid rather than as noise.
      for (int qk = 0; qk < 2; ++qk) {
        enc.set_function(_fn_rms_heads);
        enc.set_buffer(0, s.qkv); enc.set_buffer(1, s.ones);
        enc.set_constant(2, rows); enc.set_constant(3, NH);
        enc.set_constant(4, HD); enc.set_constant(5, 3 * D);
        enc.set_constant(6, qk * HD); enc.set_constant(7, c.norm_eps);
        enc.set_constant(8, 3 * HD);
        enc.dispatch({32, (unsigned)(rows * NH), 1}, {32, 1, 1});
      }
      for (int i = 0; i < 3; ++i) {
        enc.set_function(_fn_trope);
        enc.set_buffer(0, s.qkv);
        enc.set_buffer(1, i == 0 ? s.qh : i == 1 ? s.kh : s.vh);
        enc.set_buffer(2, s.rcos); enc.set_buffer(3, s.rsin);
        enc.set_constant(4, NH); enc.set_constant(5, rows);
        enc.set_constant(6, HD);
        enc.set_constant(7, i < 2 ? c.rope_dim() : 0);
        enc.set_constant(8, 3 * D); enc.set_constant(9, i * HD);
        enc.set_constant(10, 3 * HD);   // per-head grouping, see above
        enc.dispatch({(unsigned)HD, (unsigned)rows, (unsigned)NH},
                     {(unsigned)HD, 1, 1});
      }
      if (use_steel) {
        enc.set_function(_fn_attn);
        enc.set_buffer(0, s.qh); enc.set_buffer(1, s.kh);
        enc.set_buffer(2, s.vh); enc.set_buffer(3, s.oh);
        enc.set_buffer(4, _attn_p);
        enc.dispatch({32 * (unsigned)((rows + A_BQ - 1) / A_BQ),
                      4 * (unsigned)NH, 1}, {32, 4, 1});
      } else {
        enc.set_function(_fn_sdpa);
        enc.set_buffer(0, s.qh); enc.set_buffer(1, s.kh);
        enc.set_buffer(2, s.vh); enc.set_buffer(3, s.oh);
        enc.set_constant(4, scale); enc.set_constant(5, rows);
        enc.set_constant(6, HD); enc.set_constant(7, NH);
        enc.set_constant(8, NH); enc.set_constant(9, rows);
        enc.set_constant(10, rows);
        enc.dispatch({32, (unsigned)NH, (unsigned)rows}, {32, 1, 1});
      }
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, s.oh); enc.set_buffer(1, s.ob);
      enc.set_constant(2, NH); enc.set_constant(3, rows);
      enc.set_constant(4, HD);
      enc.dispatch({(unsigned)HD, (unsigned)rows, (unsigned)NH},
                   {(unsigned)HD, 1, 1});
      gemm_(enc, s.ob, 0, b.out, s.nm, 0, rows, D, D);
      // The residual gate is a learned per-CHANNEL vector (zero-init at
      // training time), not a per-row modulation -- so the DiT's indexed
      // gate is the wrong kernel and the plain broadcast one is right.
      enc.set_function(_fn_gated);
      enc.set_buffer(0, s.x); enc.set_buffer(1, b.s1); enc.set_buffer(2, s.nm);
      enc.set_constant(3, D); enc.set_constant(4, rows * D);
      enc.dispatch({(unsigned)(rows * D), 1, 1}, {256, 1, 1});

      enc.set_function(_fn_rms);
      enc.set_buffer(0, s.x); enc.set_buffer(1, b.n2); enc.set_buffer(2, s.nm);
      enc.set_constant(3, D); enc.set_constant(4, c.norm_eps);
      enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
      gemm_(enc, s.nm, 0, b.w1, s.ff, 0, rows, 2 * FF, D);
      enc.set_function(_fn_swiglu);
      enc.set_buffer(0, s.ff); enc.set_buffer(1, s.qkv);
      enc.set_constant(2, rows); enc.set_constant(3, FF);
      enc.dispatch({(unsigned)(rows * FF), 1, 1}, {256, 1, 1});
      gemm_(enc, s.qkv, 0, b.w2, s.nm, 0, rows, D, FF);
      enc.set_function(_fn_gated);
      enc.set_buffer(0, s.x); enc.set_buffer(1, b.s2); enc.set_buffer(2, s.nm);
      enc.set_constant(3, D); enc.set_constant(4, rows * D);
      enc.dispatch({(unsigned)(rows * D), 1, 1}, {256, 1, 1});
    }

    // The output norm is a LAYER norm (mean subtracted), unlike every
    // norm inside the blocks.
    enc.set_function(_fn_ln);
    enc.set_buffer(0, s.x); enc.set_buffer(1, _norm_out_w);
    enc.set_buffer(2, _norm_out_b); enc.set_buffer(3, s.nm);
    enc.set_constant(4, D); enc.set_constant(5, c.norm_eps);
    enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
    // Only the voxel rows reach the patch projection; the register and
    // cls rows are dropped.
    gemm_(enc, s.nm, 0, _proj_out, s.patches, 0, voxels, PE, D);
  }
  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty() ? std::string("MiniMax-H3 video VAE decode "
                                              "failed")
                                : gpu_err);
  }

  // Unpatchify: each row is (out_channels, patch_t, patch, patch) with
  // the CHANNEL slowest, so the pixel index is
  // ((ch*pt + kt)*p + ky)*p + kx.
  const int PT = c.patch_t, P = c.patch, OC = c.out_channels;
  const int oh = h * P, ow = w * P, ot = T * PT;
  SharedBuffer out =
      _mc->make_shared_buffer((std::size_t)OC * ot * oh * ow * 2);
  if (out.empty()) { return fail("output allocation failed"); }
  {
    const auto* src = static_cast<const std::uint16_t*>(s.patches.contents());
    auto* dst = static_cast<std::uint16_t*>(out.contents());
    const std::size_t plane = (std::size_t)oh * ow;
    for (int t = 0; t < T; ++t) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const std::uint16_t* row =
              src + ((std::size_t)((t * h + y) * w + x)) * PE;
          for (int ch = 0; ch < OC; ++ch) {
            for (int kt = 0; kt < PT; ++kt) {
              for (int ky = 0; ky < P; ++ky) {
                for (int kx = 0; kx < P; ++kx) {
                  dst[((std::size_t)ch * ot + (t * PT + kt)) * plane +
                      (std::size_t)(y * P + ky) * ow + (x * P + kx)] =
                      row[((ch * PT + kt) * P + ky) * P + kx];
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

// ---- the CNN encoder ---------------------------------------------------

int
MetalMiniMaxH3VideoVae::encoded_frames(int T) const
{
  if (T <= 0) { return 0; }
  int t = T;
  for (std::size_t i = 0; i < _cfg.time_down.size(); ++i) {
    const int s = _cfg.time_down[i];
    // Causal padding prepends 2 frames, so a stride-s convolution over
    // T frames emits floor((T-1)/s)+1 -- 17 -> 9 -> 5, where a plain
    // division would say 17 -> 8 -> 4 and silently lose the last frame.
    if (s > 1) { t = (t - 1) / s + 1; }
  }
  return t;
}

void
MetalMiniMaxH3VideoVae::enc_gn_(ComputeEncoder& enc, const SharedBuffer& in,
                                const SharedBuffer& gamma,
                                const SharedBuffer& beta,
                                const SharedBuffer& out, int T, int rows,
                                int C, bool silu)
{
  enc.set_function(_fn_gn_frames);
  enc.set_buffer(0, in);
  enc.set_buffer(1, gamma);
  enc.set_buffer(2, beta);
  enc.set_buffer(3, out);
  enc.set_constant(4, rows);
  enc.set_constant(5, C);
  enc.set_constant(6, _cfg.gn_groups);
  enc.set_constant(7, _cfg.enc_norm_eps);
  enc.set_constant(8, silu ? 1 : 0);
  enc.dispatch({256, (unsigned)_cfg.gn_groups, (unsigned)T}, {256, 1, 1});
}

int
MetalMiniMaxH3VideoVae::enc_conv_(ComputeEncoder& enc, const Conv3d& c,
                                  const SharedBuffer& in, int Ti, int H, int W,
                                  const SharedBuffer& out, int stride_t,
                                  int stride_s)
{
  if (Ti <= 0 || H <= 0 || W <= 0) { return 0; }
  if (!c.spatial) {
    // 1x1x1: no spatial gather and no temporal reach, so it is a plain
    // Linear over every voxel of the clip at once.
    gemm_(enc, in, 0, c.l, out, 0, Ti * H * W, c.cout, c.cin);
    return Ti;
  }
  const int Ho = (stride_s == 2) ? (H - 2) / 2 + 1 : H;
  const int Wo = (stride_s == 2) ? (W - 2) / 2 + 1 : W;
  // A stride-2 convolution gets its padding from the reference's
  // separate asymmetric bottom/right pad, so it carries none of its own.
  const int pad_lo = (stride_s == 2) ? 0 : 1;
  const int To = (Ti - 1) / stride_t + 1;
  const std::size_t ohw = (std::size_t)Ho * Wo;
  const std::size_t ihw = (std::size_t)H * W;
  const std::size_t per_row = (std::size_t)27 * c.cin;
  std::size_t band = per_row > 0 ? _es.col_cap / per_row : ohw;
  band = std::min(std::max<std::size_t>(band, 1), ohw);
  band = std::min<std::size_t>(band, 4096);

  for (int t = 0; t < To; ++t) {
    // Causal taps: output frame t reads input frames t*stride - 2 ..
    // t*stride, and anything before the start of the clip reads as zero.
    int mask = 0;
    std::size_t off[3] = {0, 0, 0};
    for (int kt = 0; kt < 3; ++kt) {
      const int si = t * stride_t + kt - 2;
      if (si >= 0 && si < Ti) {
        mask |= 1 << kt;
        off[kt] = (std::size_t)si * ihw * (std::size_t)c.cin;
      }
    }
    for (std::size_t r0 = 0; r0 < ohw; r0 += band) {
      const int mrows = (int)std::min(band, ohw - r0);
      enc.set_function(_fn_im2col_r);
      for (int kt = 0; kt < 3; ++kt) {
        enc.set_buffer((unsigned)kt, in, off[kt] * 2);
      }
      enc.set_buffer(3, _es.col);
      enc.set_constant(4, H);
      enc.set_constant(5, W);
      enc.set_constant(6, c.cin);
      enc.set_constant(7, (int)r0);
      enc.set_constant(8, mrows);
      enc.set_constant(9, mask);
      enc.set_constant(10, Wo);
      enc.set_constant(11, stride_s);
      enc.set_constant(12, pad_lo);
      enc.dispatch({(unsigned)(27 * c.cin), (unsigned)mrows, 1}, {64, 1, 1});
      gemm_(enc, _es.col, 0, c.l, out,
            ((std::size_t)t * ohw + r0) * (std::size_t)c.cout, mrows, c.cout,
            27 * c.cin);
    }
  }
  return To;
}

bool
MetalMiniMaxH3VideoVae::ensure_enc_scratch_(int T, int H, int W)
{
  if (_es.T == T && _es.H == H && _es.W == W) { return true; }
  const Config& c = _cfg;
  // The widest activation the encode ever holds. It is almost always
  // level 0's -- the spatial extent falls by 4x per level while the
  // channel count only doubles -- but derive it rather than assume it,
  // because a checkpoint with a different `space_down` would move it.
  std::size_t need =
      (std::size_t)T * H * W * (std::size_t)std::max(c.in_channels, c.base_ch);
  int t = T, h = H, w = W;
  for (std::size_t i = 0; i < _enc_levels.size(); ++i) {
    const EncLevel& lv = _enc_levels[i];
    need = std::max(need, (std::size_t)t * h * w *
                              (std::size_t)std::max(lv.cin, lv.cout));
    if (lv.space > 1) { h = (h - 2) / 2 + 1; w = (w - 2) / 2 + 1; }
    if (lv.time > 1) { t = (t - 1) / lv.time + 1; }
    need = std::max(need, (std::size_t)t * h * w * (std::size_t)lv.cout);
  }
  need = std::max(need,
                  (std::size_t)t * h * w * (std::size_t)(2 * c.z_channels));
  EncScratch s;
  s.T = T; s.H = H; s.W = W;
  // 16 MB of im2col band. The gather is 27x the activation, so a full
  // frame of it would be larger than the whole encode's working set at
  // any real tile size; banding is what keeps it a fixed cost.
  s.col_cap = 8u << 20;
  s.a = _mc->make_shared_buffer(need * 2);
  s.b = _mc->make_shared_buffer(need * 2);
  s.c = _mc->make_shared_buffer(need * 2);
  s.col = _mc->make_shared_buffer(s.col_cap * 2);
  if (s.a.empty() || s.b.empty() || s.c.empty() || s.col.empty()) {
    return false;
  }
  _es = std::move(s);
  return true;
}

SharedBuffer
MetalMiniMaxH3VideoVae::encode(const SharedBuffer& x, int T, int H, int W,
                               std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  const int IC = c.in_channels;
  if (T <= 0 || H <= 0 || W <= 0) { return fail("empty clip"); }
  if (H % c.patch != 0 || W % c.patch != 0) {
    return fail("H and W must be multiples of " + std::to_string(c.patch));
  }
  if (x.byte_size() < (std::size_t)IC * T * H * W * 2) {
    return fail("input is smaller than in_channels * T * H * W");
  }
  if (!_ws) { return fail("no weight set"); }
  if (!_ws->ensure_part("encoder", [&]() {
        _part = "encoder";
        const bool ok = load_encoder_(*_ws);
        _part.clear();
        return ok;
      })) {
    return fail("the encoder half of this checkpoint failed to load");
  }
  if (!ensure_enc_scratch_(T, H, W)) {
    return fail("activation allocation failed (out of GPU memory)");
  }

  // Channel-first [C, T, H, W] -> per-frame channel-last [T, H*W, C],
  // the layout every kernel below reads. Done on the host for the same
  // reason as the decoder's: it is one pass over the input and the GPU
  // alternative is a kernel that exists only here.
  {
    const auto* src = static_cast<const std::uint16_t*>(x.contents());
    auto* dst = static_cast<std::uint16_t*>(_es.a.contents());
    const std::size_t hw = (std::size_t)H * W;
    for (int t = 0; t < T; ++t) {
      for (std::size_t p = 0; p < hw; ++p) {
        for (int ch = 0; ch < IC; ++ch) {
          dst[((std::size_t)t * hw + p) * IC + ch] =
              src[((std::size_t)ch * T + t) * hw + p];
        }
      }
    }
  }

  // Three interchangeable buffers. `cur` is the live activation; the
  // other two are spares, and every op writes to a spare so nothing is
  // ever read and written in the same dispatch.
  SharedBuffer* buf[3] = {&_es.a, &_es.b, &_es.c};
  int cur = 0;
  auto spare = [&](int n) { return (cur + 1 + n) % 3; };

  // Depth override, for bisecting a mismatch: run only the first N
  // levels and return that activation raw, with no output head. A
  // 0-level run is conv_in alone, which separates the padding and the
  // host transposes from everything the levels do.
  int levels = (int)_enc_levels.size();
  int out_ch = 2 * c.z_channels;
  if (const char* lv = std::getenv("VPIPE_H3_VVAE_ENC_LEVELS")) {
    const int n = std::atoi(lv);
    if (n >= 0 && n < levels) {
      levels = n;
      out_ch = (n == 0) ? c.base_ch : _enc_levels[(std::size_t)n - 1].cout;
    }
  }
  const bool head = out_ch == 2 * c.z_channels;
  const int ZC2o = out_ch;

  int t = T, h = H, w = W;
  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    t = enc_conv_(enc, _enc_conv_in, *buf[cur], t, h, w, *buf[spare(0)], 1, 1);
    cur = spare(0);
    if (t == 0) { return fail("conv_in produced no frames"); }

    for (std::size_t li = 0; li < (std::size_t)levels; ++li) {
      const EncLevel& lv = _enc_levels[li];
      for (const EncResnet& r : lv.res) {
        const int s1 = spare(0), s2 = spare(1);
        const std::size_t rows = (std::size_t)h * w;
        // norm -> silu -> conv1 -> norm -> silu -> conv2, then the skip.
        enc_gn_(enc, *buf[cur], r.n1w, r.n1b, *buf[s1], t, (int)rows, r.cin,
                true);
        enc_conv_(enc, r.c1, *buf[s1], t, h, w, *buf[s2], 1, 1);
        enc_gn_(enc, *buf[s2], r.n2w, r.n2b, *buf[s1], t, (int)rows, r.cout,
                true);
        enc_conv_(enc, r.c2, *buf[s1], t, h, w, *buf[s2], 1, 1);
        const int n = (int)((std::size_t)t * rows * (std::size_t)r.cout);
        if (r.skip.empty()) {
          enc.set_function(_fn_residual);
          enc.set_buffer(0, *buf[cur]);
          enc.set_buffer(1, *buf[s2]);
          enc.set_buffer(2, *buf[s1]);
          enc.set_constant(3, n);
          enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
        } else {
          // The 1x1x1 shortcut has to run BEFORE the add, and it reads
          // the resnet's input -- which is why three buffers are the
          // minimum here and not two.
          enc_conv_(enc, r.skip, *buf[cur], t, h, w, *buf[s1], 1, 1);
          enc.set_function(_fn_residual);
          enc.set_buffer(0, *buf[s1]);
          enc.set_buffer(1, *buf[s2]);
          enc.set_buffer(2, *buf[cur]);
          enc.set_constant(3, n);
          enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
          continue;                    // the sum landed back in `cur`
        }
        cur = s1;
      }
      if (!lv.down.empty()) {
        const int s1 = spare(0);
        const int nt =
            enc_conv_(enc, lv.down, *buf[cur], t, h, w, *buf[s1], lv.time,
                      lv.space);
        if (nt == 0) { return fail("a downsample produced no frames"); }
        if (lv.space > 1) { h = (h - 2) / 2 + 1; w = (w - 2) / 2 + 1; }
        t = nt;
        cur = s1;
      }
    }

    if (head) {
      const int s1 = spare(0), s2 = spare(1);
      enc_gn_(enc, *buf[cur], _enc_norm_w, _enc_norm_b, *buf[s1], t, h * w,
              _enc_levels.back().cout, true);
      enc_conv_(enc, _enc_conv_out, *buf[s1], t, h, w, *buf[s2], 1, 1);
      enc_conv_(enc, _quant_conv, *buf[s2], t, h, w, *buf[s1], 1, 1);
      cur = s1;
    }
  }
  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty()
                    ? std::string("MiniMax-H3 video VAE encode failed")
                    : gpu_err);
  }

  // Back to channel-first [2*z, To, h, w], matching what decode() takes.
  SharedBuffer out =
      _mc->make_shared_buffer((std::size_t)ZC2o * t * h * w * 2);
  if (out.empty()) { return fail("output allocation failed"); }
  {
    const auto* src = static_cast<const std::uint16_t*>(buf[cur]->contents());
    auto* dst = static_cast<std::uint16_t*>(out.contents());
    const std::size_t hw = (std::size_t)h * w;
    for (int f = 0; f < t; ++f) {
      for (std::size_t p = 0; p < hw; ++p) {
        for (int ch = 0; ch < ZC2o; ++ch) {
          dst[((std::size_t)ch * t + f) * hw + p] =
              src[((std::size_t)f * hw + p) * ZC2o + ch];
        }
      }
    }
  }
  return out;
}

// ---- tiling and clip chunking ------------------------------------------

SharedBuffer
MetalMiniMaxH3VideoVae::stitch_(const std::vector<SharedBuffer>& tiles,
                                const minimax_h3::TileSplit& ys,
                                const minimax_h3::TileSplit& xs, int ratio,
                                int C, int T, int out_h, int out_w)
{
  const int rows = (int)ys.start.size();
  const int cols = (int)xs.start.size();
  if (ratio <= 0 || rows == 0 || cols == 0 ||
      (int)tiles.size() != rows * cols) {
    return {};
  }
  SharedBuffer out =
      _mc->make_shared_buffer((std::size_t)C * T * out_h * out_w * 2);
  if (out.empty()) { return {}; }
  auto* dst = static_cast<std::uint16_t*>(out.contents());
  const std::size_t oplane = (std::size_t)out_h * out_w;
  std::vector<float> tmp;

  for (int i = 0; i < rows; ++i) {
    const int th = ys.length[(std::size_t)i] / ratio;
    const int oy = ys.start[(std::size_t)i] / ratio;
    for (int j = 0; j < cols; ++j) {
      const int tw = xs.length[(std::size_t)j] / ratio;
      const int ox = xs.start[(std::size_t)j] / ratio;
      const auto& tb = tiles[(std::size_t)(i * cols + j)];
      if (tb.empty()) { return {}; }
      const auto* cur = static_cast<const std::uint16_t*>(tb.contents());
      const std::size_t plane = (std::size_t)th * tw;
      const std::size_t n = (std::size_t)C * T * plane;
      if (tb.byte_size() < n * 2) { return {}; }
      tmp.resize(n);
      for (std::size_t k = 0; k < n; ++k) { tmp[k] = bf16_to_f32_(cur[k]); }

      // Cross-fade against the tile ABOVE, then against the one to the
      // LEFT. Both read the neighbour's ORIGINAL bytes, not a version
      // already blended this pass -- the reference rebinds a local and
      // leaves the tile grid untouched, so a corner cell mixes three
      // unmodified tiles rather than a chain of two.
      if (i > 0) {
        const int ph = ys.length[(std::size_t)(i - 1)] / ratio;
        int ext = ys.overlap[(std::size_t)(i - 1)] / ratio;
        ext = std::min(std::min(ext, ph), th);
        const auto* up = static_cast<const std::uint16_t*>(
            tiles[(std::size_t)((i - 1) * cols + j)].contents());
        for (int c = 0; c < C; ++c) {
          for (int t = 0; t < T; ++t) {
            const std::size_t cb = ((std::size_t)c * T + t) * plane;
            const std::size_t pb = ((std::size_t)c * T + t) * ph * tw;
            for (int r = 0; r < ext; ++r) {
              const float wb = (float)r / (float)ext;
              for (int x = 0; x < tw; ++x) {
                const std::size_t k = cb + (std::size_t)r * tw + x;
                tmp[k] = (1.0f - wb) *
                             bf16_to_f32_(up[pb + (std::size_t)(ph - ext + r) *
                                                     tw + x]) +
                         wb * tmp[k];
              }
            }
          }
        }
      }
      if (j > 0) {
        const int pw = xs.length[(std::size_t)(j - 1)] / ratio;
        int ext = xs.overlap[(std::size_t)(j - 1)] / ratio;
        ext = std::min(std::min(ext, pw), tw);
        const auto* lf = static_cast<const std::uint16_t*>(
            tiles[(std::size_t)(i * cols + j - 1)].contents());
        for (int c = 0; c < C; ++c) {
          for (int t = 0; t < T; ++t) {
            const std::size_t cb = ((std::size_t)c * T + t) * plane;
            const std::size_t pb = ((std::size_t)c * T + t) * th * pw;
            for (int r = 0; r < th; ++r) {
              for (int x = 0; x < ext; ++x) {
                const float wb = (float)x / (float)ext;
                const std::size_t k = cb + (std::size_t)r * tw + x;
                tmp[k] = (1.0f - wb) *
                             bf16_to_f32_(lf[pb + (std::size_t)r * pw +
                                             (pw - ext + x)]) +
                         wb * tmp[k];
              }
            }
          }
        }
      }

      // A non-last tile contributes everything but the overlap it hands
      // to its successor, so the contributions tile the axis exactly.
      const int wh = (i + 1 < rows)
                         ? th - ys.overlap[(std::size_t)i] / ratio
                         : th;
      const int ww = (j + 1 < cols)
                         ? tw - xs.overlap[(std::size_t)j] / ratio
                         : tw;
      for (int c = 0; c < C; ++c) {
        for (int t = 0; t < T; ++t) {
          const std::size_t cb = ((std::size_t)c * T + t) * plane;
          const std::size_t ob = ((std::size_t)c * T + t) * oplane;
          for (int r = 0; r < wh; ++r) {
            for (int x = 0; x < ww; ++x) {
              dst[ob + (std::size_t)(oy + r) * out_w + (ox + x)] =
                  f32_to_bf16_(tmp[cb + (std::size_t)r * tw + x]);
            }
          }
        }
      }
    }
  }
  return out;
}

SharedBuffer
MetalMiniMaxH3VideoVae::encode_tiled_(const SharedBuffer& x, int T, int H,
                                      int W, std::string* err)
{
  const Config& c = _cfg;
  const auto ys = minimax_h3::split_tiles(H, c.tile_size, c.tile_overlap_min,
                                          c.patch);
  const auto xs = minimax_h3::split_tiles(W, c.tile_size, c.tile_overlap_min,
                                          c.patch);
  if (ys.start.size() <= 1 && xs.start.size() <= 1) {
    return encode(x, T, H, W, err);
  }
  const int IC = c.in_channels;
  const std::size_t plane = (std::size_t)H * W;
  const auto* src = static_cast<const std::uint16_t*>(x.contents());
  std::vector<SharedBuffer> tiles;
  tiles.reserve(ys.start.size() * xs.start.size());
  for (std::size_t i = 0; i < ys.start.size(); ++i) {
    for (std::size_t j = 0; j < xs.start.size(); ++j) {
      const int y0 = ys.start[i], th = ys.length[i];
      const int x0 = xs.start[j], tw = xs.length[j];
      SharedBuffer sub =
          _mc->make_shared_buffer((std::size_t)IC * T * th * tw * 2);
      if (sub.empty()) {
        if (err != nullptr) { *err = "tile allocation failed"; }
        return {};
      }
      auto* d = static_cast<std::uint16_t*>(sub.contents());
      for (int ch = 0; ch < IC; ++ch) {
        for (int t = 0; t < T; ++t) {
          const std::size_t sb = ((std::size_t)ch * T + t) * plane;
          const std::size_t db = ((std::size_t)ch * T + t) * th * tw;
          for (int r = 0; r < th; ++r) {
            std::memcpy(d + db + (std::size_t)r * tw,
                        src + sb + (std::size_t)(y0 + r) * W + x0,
                        (std::size_t)tw * 2);
          }
        }
      }
      SharedBuffer enc = encode(sub, T, th, tw, err);
      if (enc.empty()) { return {}; }
      tiles.push_back(std::move(enc));
    }
  }
  SharedBuffer out =
      stitch_(tiles, ys, xs, c.patch, 2 * c.z_channels, encoded_frames(T),
              H / c.patch, W / c.patch);
  if (out.empty() && err != nullptr) { *err = "tile stitch failed"; }
  return out;
}

SharedBuffer
MetalMiniMaxH3VideoVae::decode_tiled_(const SharedBuffer& z, int LT, int lh,
                                      int lw, std::string* err)
{
  const Config& c = _cfg;
  // The tiles are laid out in PIXEL space and mapped back onto the
  // latent grid, which is what keeps a decode's boundaries on the same
  // pixels the matching encode used.
  const int H = lh * c.patch, W = lw * c.patch;
  const auto ys = minimax_h3::split_tiles(H, c.tile_size, c.tile_overlap_min,
                                          c.patch);
  const auto xs = minimax_h3::split_tiles(W, c.tile_size, c.tile_overlap_min,
                                          c.patch);
  if (ys.start.size() <= 1 && xs.start.size() <= 1) {
    return decode(z, LT, lh, lw, err);
  }
  const int ZC = c.z_channels;
  const std::size_t plane = (std::size_t)lh * lw;
  const auto* src = static_cast<const std::uint16_t*>(z.contents());
  std::vector<SharedBuffer> tiles;
  tiles.reserve(ys.start.size() * xs.start.size());
  for (std::size_t i = 0; i < ys.start.size(); ++i) {
    for (std::size_t j = 0; j < xs.start.size(); ++j) {
      const int y0 = ys.start[i] / c.patch, th = ys.length[i] / c.patch;
      const int x0 = xs.start[j] / c.patch, tw = xs.length[j] / c.patch;
      SharedBuffer sub =
          _mc->make_shared_buffer((std::size_t)ZC * LT * th * tw * 2);
      if (sub.empty()) {
        if (err != nullptr) { *err = "tile allocation failed"; }
        return {};
      }
      auto* d = static_cast<std::uint16_t*>(sub.contents());
      for (int ch = 0; ch < ZC; ++ch) {
        for (int t = 0; t < LT; ++t) {
          const std::size_t sb = ((std::size_t)ch * LT + t) * plane;
          const std::size_t db = ((std::size_t)ch * LT + t) * th * tw;
          for (int r = 0; r < th; ++r) {
            std::memcpy(d + db + (std::size_t)r * tw,
                        src + sb + (std::size_t)(y0 + r) * lw + x0,
                        (std::size_t)tw * 2);
          }
        }
      }
      SharedBuffer dec = decode(sub, LT, th, tw, err);
      if (dec.empty()) { return {}; }
      tiles.push_back(std::move(dec));
    }
  }
  SharedBuffer out = stitch_(tiles, ys, xs, 1, c.out_channels,
                             LT * c.patch_t, H, W);
  if (out.empty() && err != nullptr) { *err = "tile stitch failed"; }
  return out;
}

int
MetalMiniMaxH3VideoVae::video_latent_frames(int T) const
{
  if (T <= 0) { return 0; }
  // A single frame has no temporal extent to chunk, and no tail is
  // dropped from it.
  if (T == 1) { return encoded_frames(1); }
  const int cl = _cfg.clip_length;
  if (cl <= 0) { return 0; }
  const int pad = ((-T % cl) + cl) % cl;
  const int n = (T + pad) / cl;
  const int per = encoded_frames(cl);
  const int total = n * per - _cfg.token_drop;
  return total > 0 ? total : 0;
}

int
MetalMiniMaxH3VideoVae::decoded_frames(int LT) const
{
  const Config& c = _cfg;
  const int tcs = c.tokens_per_chunk();
  if (LT <= 0 || tcs <= 0) { return 0; }
  const int td = c.token_drop;
  const int num_tokens = LT + td;
  const int pad_tokens = ((-num_tokens % tcs) + tcs) % tcs;
  const int chunks = (num_tokens + pad_tokens) / tcs - (td > 0 ? 1 : 0);
  if (chunks <= 0) { return 0; }
  const int cnf  = tcs * c.patch_t;
  const int clip = (tcs + c.token_overlap()) * c.patch_t;
  const int fpp  = c.frame_pre_pad();
  const int head = std::min(cnf, clip) - fpp;
  const int tail = td > 0 ? std::max(std::min(cnf, clip - cnf) - fpp, 0) : 0;
  int frames = chunks * head + tail;
  if (pad_tokens > 0) {
    // The repeated latent frames produced pixel frames nobody asked
    // for. A chunk's LAST latent frame only covers
    // `clip_length % patch_t` of them; every other covers `patch_t`.
    const int intra = c.clip_length % c.patch_t;
    for (int k = 0; k < pad_tokens; ++k) {
      frames -= (intra != 0 && (LT + k) % tcs == 0) ? intra : c.patch_t;
    }
  }
  return frames > 0 ? frames : 0;
}

SharedBuffer
MetalMiniMaxH3VideoVae::encode_video(const SharedBuffer& x, int T, int H,
                                     int W, int* latent_frames,
                                     std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  if (T <= 0) { return fail("empty video"); }
  if (T == 1) {
    SharedBuffer out = encode_tiled_(x, 1, H, W, err);
    if (!out.empty() && latent_frames != nullptr) { *latent_frames = 1; }
    return out;
  }
  const int cl = c.clip_length, IC = c.in_channels;
  if (cl <= 0) { return fail("clip_length is not positive"); }
  const int total = video_latent_frames(T);
  if (total <= 0) {
    return fail("too few frames to encode: " + std::to_string(T) +
                " leaves nothing after token_drop");
  }
  const int pad = ((-T % cl) + cl) % cl;
  const int nclips = (T + pad) / cl;
  const int per = encoded_frames(cl);
  const int lh = H / c.patch, lw = W / c.patch;

  SharedBuffer clip =
      _mc->make_shared_buffer((std::size_t)IC * cl * H * W * 2);
  SharedBuffer out = _mc->make_shared_buffer((std::size_t)(2 * c.z_channels) *
                                             total * lh * lw * 2);
  if (clip.empty() || out.empty()) { return fail("allocation failed"); }
  const std::size_t plane = (std::size_t)H * W;
  const std::size_t lplane = (std::size_t)lh * lw;
  const auto* src = static_cast<const std::uint16_t*>(x.contents());
  auto* dst = static_cast<std::uint16_t*>(out.contents());
  const int ZC2 = 2 * c.z_channels;

  for (int n = 0; n < nclips; ++n) {
    auto* cd = static_cast<std::uint16_t*>(clip.contents());
    for (int ch = 0; ch < IC; ++ch) {
      for (int t = 0; t < cl; ++t) {
        // Past the end the LAST REAL frame repeats. Repeating it is
        // what the reference pads with; zeros would put a hard cut into
        // the causal window of the final latents.
        const int st = std::min(n * cl + t, T - 1);
        std::memcpy(cd + ((std::size_t)ch * cl + t) * plane,
                    src + ((std::size_t)ch * T + st) * plane, plane * 2);
      }
    }
    SharedBuffer enc = encode_tiled_(clip, cl, H, W, err);
    if (enc.empty()) { return {}; }
    const auto* ed = static_cast<const std::uint16_t*>(enc.contents());
    for (int ch = 0; ch < ZC2; ++ch) {
      for (int f = 0; f < per; ++f) {
        const int g = n * per + f;
        if (g >= total) { continue; }        // the token_drop tail
        std::memcpy(dst + ((std::size_t)ch * total + g) * lplane,
                    ed + ((std::size_t)ch * per + f) * lplane, lplane * 2);
      }
    }
  }
  if (latent_frames != nullptr) { *latent_frames = total; }
  return out;
}

SharedBuffer
MetalMiniMaxH3VideoVae::decode_video(const SharedBuffer& z, int LT, int lh,
                                     int lw, int* out_frames,
                                     std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  const int tcs = c.tokens_per_chunk();
  if (LT <= 0 || lh <= 0 || lw <= 0 || tcs <= 0) {
    return fail("empty latent video");
  }
  const int td = c.token_drop, tr = c.patch_t;
  const int tov = c.token_overlap(), fpp = c.frame_pre_pad();
  const int fov = c.frame_overlap(), cnf = tcs * tr;
  const int num_tokens = LT + td;
  const int pad_tokens = ((-num_tokens % tcs) + tcs) % tcs;
  const int chunks = (num_tokens + pad_tokens) / tcs - (td > 0 ? 1 : 0);
  if (chunks <= 0) {
    // Below one chunk there is nothing to decode: `token_drop` already
    // removed more than a chunk's worth of the sequence's tail.
    return fail("only " + std::to_string(LT) + " latent frames: fewer than "
                "the " + std::to_string(tcs + td) + " one chunk needs");
  }
  const int ZC = c.z_channels;
  const int H = lh * c.patch, W = lw * c.patch;
  const std::size_t lplane = (std::size_t)lh * lw;
  const std::size_t plane  = (std::size_t)H * W;
  const int OC = c.out_channels;

  // Repeat the last latent frame up to a whole number of chunks; the
  // pixel frames it invents are cut off again at the end.
  const int zt = LT + pad_tokens;
  SharedBuffer zp = _mc->make_shared_buffer((std::size_t)ZC * zt * lplane * 2);
  if (zp.empty()) { return fail("allocation failed"); }
  {
    const auto* s = static_cast<const std::uint16_t*>(z.contents());
    auto* d = static_cast<std::uint16_t*>(zp.contents());
    for (int ch = 0; ch < ZC; ++ch) {
      for (int t = 0; t < zt; ++t) {
        const int st = std::min(t, LT - 1);
        std::memcpy(d + ((std::size_t)ch * zt + t) * lplane,
                    s + ((std::size_t)ch * LT + st) * lplane, lplane * 2);
      }
    }
  }

  const int span = tcs + tov;
  struct Piece {
    SharedBuffer buf;
    int n = 0;
  };
  std::vector<Piece> pieces;
  Piece overlap;
  SharedBuffer sub =
      _mc->make_shared_buffer((std::size_t)ZC * span * lplane * 2);
  if (sub.empty()) { return fail("allocation failed"); }

  for (int i = 0; i < chunks; ++i) {
    const int start = i * tcs;
    {
      const auto* s = static_cast<const std::uint16_t*>(zp.contents());
      auto* d = static_cast<std::uint16_t*>(sub.contents());
      for (int ch = 0; ch < ZC; ++ch) {
        for (int t = 0; t < span; ++t) {
          const int st = std::min(start + t, zt - 1);
          std::memcpy(d + ((std::size_t)ch * span + t) * lplane,
                      s + ((std::size_t)ch * zt + st) * lplane, lplane * 2);
        }
      }
    }
    SharedBuffer clip = decode_tiled_(sub, span, lh, lw, err);
    if (clip.empty()) { return {}; }
    const int cf = span * tr;
    const auto* cd = static_cast<const std::uint16_t*>(clip.contents());

    // Each decoded clip yields the chunk proper and, when token_drop is
    // on, a trailing piece that becomes the NEXT chunk's cross-fade
    // partner rather than being emitted here.
    for (int j = 0; j < (td > 0 ? 2 : 1); ++j) {
      const int fs = j * cnf;
      const int avail = cf - fs;
      if (avail <= fpp) { continue; }
      const int n = std::min(cnf, avail) - fpp;
      Piece p;
      p.n = n;
      p.buf = _mc->make_shared_buffer((std::size_t)OC * n * plane * 2);
      if (p.buf.empty()) { return fail("allocation failed"); }
      auto* pd = static_cast<std::uint16_t*>(p.buf.contents());
      for (int ch = 0; ch < OC; ++ch) {
        std::memcpy(pd + (std::size_t)ch * n * plane,
                    cd + ((std::size_t)ch * cf + fs + fpp) * plane,
                    (std::size_t)n * plane * 2);
      }
      if (j == 0) {
        if (overlap.n > 0) {
          // Linear cross-fade over the frames the two chunks share.
          const int ext = std::min(std::min(overlap.n, p.n), fov);
          const auto* od = static_cast<const std::uint16_t*>(
              overlap.buf.contents());
          for (int ch = 0; ch < OC; ++ch) {
            for (int f = 0; f < ext; ++f) {
              const float wb = (float)f / (float)ext;
              const std::size_t pb = ((std::size_t)ch * n + f) * plane;
              const std::size_t ob =
                  ((std::size_t)ch * overlap.n + (overlap.n - ext + f)) * plane;
              for (std::size_t q = 0; q < plane; ++q) {
                pd[pb + q] = f32_to_bf16_(
                    (1.0f - wb) * bf16_to_f32_(od[ob + q]) +
                    wb * bf16_to_f32_(pd[pb + q]));
              }
            }
          }
          overlap = Piece{};
        }
        pieces.push_back(std::move(p));
      } else {
        overlap = std::move(p);
      }
    }
  }
  if (overlap.n > 0) { pieces.push_back(std::move(overlap)); }

  int frames = 0;
  for (const Piece& p : pieces) { frames += p.n; }
  int drop = 0;
  if (pad_tokens > 0) {
    const int intra = c.clip_length % tr;
    for (int k = 0; k < pad_tokens; ++k) {
      drop += (intra != 0 && (LT + k) % tcs == 0) ? intra : tr;
    }
  }
  const int keep = frames - drop;
  if (keep <= 0) { return fail("the pad trim removed every decoded frame"); }

  SharedBuffer out =
      _mc->make_shared_buffer((std::size_t)OC * keep * plane * 2);
  if (out.empty()) { return fail("output allocation failed"); }
  auto* dst = static_cast<std::uint16_t*>(out.contents());
  int at = 0;
  for (const Piece& p : pieces) {
    const auto* pd = static_cast<const std::uint16_t*>(p.buf.contents());
    for (int f = 0; f < p.n && at + f < keep; ++f) {
      for (int ch = 0; ch < OC; ++ch) {
        std::memcpy(dst + ((std::size_t)ch * keep + at + f) * plane,
                    pd + ((std::size_t)ch * p.n + f) * plane, plane * 2);
      }
    }
    at += p.n;
  }
  if (out_frames != nullptr) { *out_frames = keep; }
  return out;
}

}  // namespace genai
}  // namespace vpipe
