#include "generative-models/minimax-h3/metal-minimax-h3-latent-upscaler.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
namespace fs = std::filesystem;

namespace {

// The derived-tensor keys this class files its transforms under.
constexpr const char* kKey = "h3-latent-upscaler/";

// MiniMax-H3's latent statistics, as the node hardcodes them (they are the
// H3 VAE's latents_mean / latents_std, to the digit).
constexpr float kMean[24] = {
  0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f,
  -0.5090325474739075f, -0.2727581858634949f, -1.3675414323806763f,
  -0.2553254961967468f, -0.26907554268836975f, -0.5376840829849243f,
  -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
  -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f,
  0.25928452610969543f, -0.30133944749832153f, 0.211341992020607f,
  -1.1206848621368408f, 0.3581933379173279f, -0.04225143790245056f,
  0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f};
constexpr float kStd[24] = {
  1.2223774194717407f, 1.2767263650894165f, 1.6831774711608887f,
  1.7549455165863037f, 1.5636216402053833f, 2.194143533706665f,
  0.9653137922286987f, 1.0569885969161987f, 0.841948926448822f,
  0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
  0.7996809482574463f, 0.44988900423049927f, 0.7197399735450745f,
  0.6936293244361877f, 2.961095094680786f, 2.7694199085235596f,
  3.0496184825897217f, 2.1088054180145264f, 3.276226282119751f,
  3.1627357006073f, 2.2816812992095947f, 2.6127843856811523f};

constexpr int kGatherRows = 2048;   // output rows per gather band
constexpr int kStatBlocks = 512;    // GroupNorm partial-sum blocks

std::vector<float>
read_f32_(const MetalLlamaWeights& wts, MetalCompute* mc,
          const std::string& nm, std::vector<std::int64_t>* shape = nullptr)
{
  const auto* info = wts.info(nm);
  if (info == nullptr || info->shape.empty()) { return {}; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { return {}; }
  std::vector<float> v(n);
  if (info->dtype == "F32") {
    std::memcpy(v.data(), raw.contents(), n * 4);
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = (float)s[i]; }
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t u = (std::uint32_t)s[i] << 16;
      std::memcpy(&v[i], &u, 4);
    }
  } else {
    return {};
  }
  if (shape != nullptr) { *shape = info->shape; }
  return v;
}

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

SharedBuffer
elt_buf_(MetalCompute* mc, const float* src, std::size_t n, bool bf16)
{
  SharedBuffer b = mc->make_shared_buffer(n * 2);
  if (b.empty()) { return {}; }
  if (bf16) {
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = to_bf16_(src[i]); }
  } else {
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)src[i]; }
  }
  return b;
}

SharedBuffer
f32_buf_(MetalCompute* mc, const float* src, std::size_t n)
{
  SharedBuffer b = mc->make_shared_buffer(n * 4);
  if (!b.empty()) { std::memcpy(b.contents(), src, n * 4); }
  return b;
}

// Python's round(): half to even.
double
py_round_(double x)
{
  const double r = std::nearbyint(x);   // default mode: to nearest even
  return r;
}

inline float
silu_(float x)
{
  return x / (1.0f + std::exp(-x));
}

// The entries of one stack, by the index the file gives them: a ResBlock
// has `in_layers`, a TemporalConv `dwconv`.
bool
stack_entries_(const std::vector<std::string>& names, const std::string& stack,
               std::vector<bool>* out, std::string* err)
{
  std::map<int, int> kind;   // index -> 1 res, 2 temporal
  const std::string pre = stack + ".";
  for (const auto& n : names) {
    if (n.compare(0, pre.size(), pre) != 0) { continue; }
    const std::size_t dot = n.find('.', pre.size());
    if (dot == std::string::npos) { continue; }
    const int i = std::atoi(n.substr(pre.size(), dot - pre.size()).c_str());
    const std::string rest = n.substr(dot + 1);
    if (rest.compare(0, 10, "in_layers.") == 0) {
      kind[i] |= 1;
    } else if (rest.compare(0, 7, "dwconv.") == 0) {
      kind[i] |= 2;
    } else if (rest.compare(0, 2, "q.") == 0 ||
               rest.compare(0, 9, "proj_out.") == 0) {
      // The node builds attention blocks only when asked, and forces it
      // off; a file that carries them is a network it never ran.
      if (err != nullptr) {
        *err = stack + " carries attention blocks, which the node never "
                       "runs";
      }
      return false;
    }
  }
  out->clear();
  int expect = 0;
  for (const auto& [i, k] : kind) {
    if (i != expect || (k != 1 && k != 2)) {
      if (err != nullptr) {
        *err = fmt("{} entry {} is neither a ResBlock nor a TemporalConv",
                   stack, i)();
      }
      return false;
    }
    out->push_back(k == 2);
    ++expect;
  }
  if (out->empty()) {
    if (err != nullptr) { *err = "no " + stack; }
    return false;
  }
  return true;
}

}  // namespace

struct MetalMiniMaxH3LatentUpscaler::Pass {
  ComputeEncoder* enc = nullptr;
  SharedBuffer col;          // the gather band
  SharedBuffer part;         // GroupNorm partials [kStatBlocks][2C]
  SharedBuffer stats;        // [2G]
  SharedBuffer ab;           // every norm's [a | b], f32 [n][2C]
  int C = 0;
};

MetalMiniMaxH3LatentUpscaler::~MetalMiniMaxH3LatentUpscaler() = default;

bool
MetalMiniMaxH3LatentUpscaler::config_from_weights(const WeightSet& ws,
                                                  Config* out,
                                                  std::string* err)
{
  auto no = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const MetalLlamaWeights& src = ws.src();
  const auto* ci = src.info("conv_in.weight");
  if (ci == nullptr || ci->shape.size() != 5) {
    return no("no conv_in.weight [C, 24, 3, 3, 3] -- not the MiniMax-H3 "
              "latent upscaler");
  }
  Config c;
  c.channels    = (int)ci->shape[0];
  c.in_channels = (int)ci->shape[1];
  if (ci->shape[2] != 3 || ci->shape[3] != 3 || ci->shape[4] != 3) {
    return no("conv_in is not 3x3x3");
  }
  if (c.channels % c.groups != 0 || c.channels % 4 != 0) {
    return no(fmt("{} channels do not split into {} groups", c.channels,
                  c.groups)());
  }
  if (c.in_channels != 24) {
    return no(fmt("conv_in reads {} channels; MiniMax-H3's latent has 24",
                  c.in_channels)());
  }
  const auto* e0 = src.info("embed.0.weight");
  if (e0 == nullptr || e0->shape.size() != 2 || e0->shape[1] != 1) {
    return no("no embed.0.weight [E, 1]");
  }
  c.embed_dim = (int)e0->shape[0];
  const std::vector<std::string> names = src.tensor_names();
  if (!stack_entries_(names, "in_blocks", &c.in_entries, err) ||
      !stack_entries_(names, "out_blocks", &c.out_entries, err)) {
    return false;
  }
  c.temporal_kernel = 0;
  for (const auto& n : names) {
    if (n.size() > 13 && n.compare(n.size() - 13, 13, "dwconv.weight") == 0) {
      const auto* di = src.info(n);
      if (di != nullptr && di->shape.size() == 5) {
        c.temporal_kernel = (int)di->shape[2];
      }
      break;
    }
  }
  if (c.temporal_kernel != 0 && c.temporal_kernel % 2 == 0) {
    return no(fmt("an even temporal kernel ({})", c.temporal_kernel)());
  }
  *out = std::move(c);
  return true;
}

std::string
MetalMiniMaxH3LatentUpscaler::resolve_weights(const std::string& path,
                                              std::string* err)
{
  std::error_code ec;
  const fs::path p(path);
  auto st_in = [&](const fs::path& d) {
    std::vector<fs::path> v;
    for (const auto& e : fs::directory_iterator(d, ec)) {
      if (e.is_regular_file(ec) && e.path().extension() == ".safetensors") {
        v.push_back(e.path());
      }
    }
    std::sort(v.begin(), v.end());
    return v;
  };
  if (fs::is_regular_file(p, ec)) { return path; }
  if (!fs::is_directory(p, ec)) {
    if (err != nullptr) { *err = "'" + path + "' does not exist"; }
    return {};
  }
  for (const fs::path& d : {p / "latent_upsampler", p}) {
    if (!fs::is_directory(d, ec)) { continue; }
    const auto v = st_in(d);
    if (v.size() == 1) { return v[0].string(); }
    if (v.size() > 1) {
      if (err != nullptr) {
        *err = "'" + d.string() + "' holds " + std::to_string(v.size()) +
               " .safetensors files; name the upscaler's file";
      }
      return {};
    }
  }
  if (err != nullptr) {
    *err = "no latent upscaler in '" + path + "' (expected "
           "latent_upsampler/<file>.safetensors or one .safetensors)";
  }
  return {};
}

MetalMiniMaxH3LatentUpscaler::Target
MetalMiniMaxH3LatentUpscaler::target_for(int h, int w, int width_px,
                                         int height_px, int align)
{
  constexpr int kDown = 16;
  Target t;
  const int a = std::max(1, align);
  const double wa = py_round_((double)width_px / a) * a;
  const double ha = py_round_((double)height_px / a) * a;
  const double wf = py_round_(wa / kDown) * kDown;
  const double hf = py_round_(ha / kDown) * kDown;
  t.w = std::max(1, (int)(wf / kDown));
  t.h = std::max(1, (int)(hf / kDown));
  t.scale = ((double)width_px / ((double)w * kDown) +
             (double)height_px / ((double)h * kDown)) / 2.0;
  return t;
}

MetalMiniMaxH3LatentUpscaler::Target
MetalMiniMaxH3LatentUpscaler::target_for_scale(int h, int w, double scale,
                                               int align)
{
  constexpr int kDown = 16;
  Target t;
  const int a = std::max(1, align);
  const double wa = py_round_((double)w * kDown * scale / a) * a;
  const double ha = py_round_((double)h * kDown * scale / a) * a;
  t.w = std::max(1, (int)(py_round_(wa / kDown)));
  t.h = std::max(1, (int)(py_round_(ha / kDown)));
  t.scale = scale;
  return t;
}

std::size_t
MetalMiniMaxH3LatentUpscaler::scratch_bytes(const Config& cfg, int T, int h,
                                            int w, int H, int W, int i8_tier)
{
  const std::size_t C = (std::size_t)cfg.channels;
  const std::size_t Z = (std::size_t)cfg.in_channels;
  const std::size_t in = (std::size_t)T * h * w, out = (std::size_t)T * H * W;
  // The input stack's x, a, b (and the input) live until the resize has
  // written the output's x; the output stack's three and the result
  // come after they are gone. The larger of the two, plus what every
  // pass keeps.
  const std::size_t stack_in = (3 * in * C + out * C + in * Z) * 2;
  const std::size_t stack_out = (3 * out * C + out * Z) * 2;
  std::size_t b = std::max(stack_in, stack_out);
  b += (std::size_t)kGatherRows * 27 * C * 2;     // the gather band
  b += (std::size_t)kStatBlocks * 2 * C * 4 + 2 * cfg.groups * 4;
  b += (cfg.in_entries.size() + cfg.out_entries.size() + 1) * 2 * 2 * C * 4;
  // The int8 context's scratch at the gather band's GEMM, and -- a held
  // layout widened under its floor -- one conv's weight.
  if (i8_tier >= 0) {
    b += I8GemmContext::forward_bytes_bound(kGatherRows, (int)(27 * C),
                                            (int)C, (int)(27 * C),
                                            i8_tier == 0);
    if (i8_tier >= 1) { b += 27 * C * C * 2; }
  }
  return b;
}

std::size_t
MetalMiniMaxH3LatentUpscaler::scratch_bytes(int T, int h, int w, int H,
                                            int W, int i8_tier)
{
  Config c;
  for (int i = 0; i < 18; ++i) { c.in_entries.push_back(i % 3 == 1); }
  c.out_entries = c.in_entries;
  return scratch_bytes(c, T, h, w, H, W, i8_tier);
}

bool
MetalMiniMaxH3LatentUpscaler::load_conv_(WeightSet& ws,
                                         const std::string& name, Conv* c,
                                         std::string* err)
{
  const auto* info = ws.src().info(name + ".weight");
  if (info == nullptr || info->shape.size() != 5) {
    if (err != nullptr) { *err = "no " + name + ".weight"; }
    return false;
  }
  const auto& sh = info->shape;
  const int Cout = (int)sh[0], Cin = (int)sh[1];
  const int kt = (int)sh[2], kh = (int)sh[3], kw = (int)sh[4];
  const bool k3 = kt == 3 && kh == 3 && kw == 3;
  if (!k3 && !(kt == 1 && kh == 1 && kw == 1)) {
    if (err != nullptr) { *err = name + " is neither 3x3x3 nor 1x1x1"; }
    return false;
  }
  c->cin = Cin;
  c->cout = Cout;
  c->taps = k3 ? 27 : 1;
  const bool bf16 = _bf16;
  const std::string key =
      fmt("{}{}|{}", kKey, bf16 ? "bf16" : "f16", name)();
  // [Cout, (kt, ky, kx), Cin]: the order im2col_hwc_3x3x3_tiled_f16
  // writes a row in.
  auto build = [&]() -> SharedBuffer {
    std::vector<float> w = read_f32_(ws.src(), _mc, name + ".weight");
    if (w.empty()) { return {}; }
    const int taps = c->taps;
    std::vector<float> flat((std::size_t)Cout * taps * Cin);
    for (int o = 0; o < Cout; ++o) {
      for (int i = 0; i < Cin; ++i) {
        for (int k = 0; k < taps; ++k) {
          flat[((std::size_t)o * taps + k) * Cin + i] =
              w[((std::size_t)o * Cin + i) * taps + k];
        }
      }
    }
    return elt_buf_(_mc, flat.data(), flat.size(), bf16);
  };
  // Under a tier that holds the int8 layout, a 3x3x3 conv keeps only the
  // layout: built from the element-type weight, which is then dropped.
  const int K = c->taps * Cin;
  const int gc = (_i8_tier >= 1 && c->taps == 27) ? _i8->fast_group(K) : 0;
  c->gc = 0;
  c->codes = SharedBuffer{};
  c->scales = SharedBuffer{};
  c->w = SharedBuffer{};
  if (gc > 0) {
    const std::size_t nk = (std::size_t)Cout * K;
    const std::size_t sb = I8GemmContext::fast_scale_bytes(Cout, K, gc);
    SharedBuffer both = ws.derived(
        fmt("{}{}|{}|{}", kKey, bf16 ? "bf16" : "f16", _i8->layout_tag(gc),
            name)(),
        [&]() -> SharedBuffer {
          SharedBuffer f = build();
          SharedBuffer out = _mc->make_shared_buffer(nk + sb);
          if (f.empty() || out.empty()) { return {}; }
          CommandStream st = _mc->make_command_stream();
          bool ok = false;
          {
            ComputeEncoder e = st.begin_compute();
            ok = _i8->encode_quant_fast(e, f, out.subview(0, nk),
                                        out.subview(nk, sb), gc, Cout, K);
          }
          return ok && st.commit().wait_ok(nullptr) ? std::move(out)
                                                    : SharedBuffer{};
        });
    if (!both.empty()) {
      c->gc = gc;
      c->codes = both.subview(0, nk);
      c->scales = both.subview(nk, sb);
      _weight_bytes += both.byte_size();
    }
  }
  if (c->gc == 0) {
    c->w = ws.derived(key, build);
    _weight_bytes += c->w.byte_size();
  }
  const std::vector<float> b = read_f32_(ws.src(), _mc, name + ".bias");
  if ((c->gc == 0 && c->w.empty()) || (int)b.size() != Cout) {
    if (err != nullptr) { *err = "cannot read " + name; }
    return false;
  }
  c->b = elt_buf_(_mc, b.data(), b.size(), bf16);
  _weight_bytes += c->b.byte_size();
  return !c->b.empty();
}

bool
MetalMiniMaxH3LatentUpscaler::load_norm_(WeightSet& ws,
                                         const std::string& name, Norm* n,
                                         std::string* err)
{
  n->gamma = read_f32_(ws.src(), _mc, name + ".weight");
  n->beta  = read_f32_(ws.src(), _mc, name + ".bias");
  if ((int)n->gamma.size() != _cfg.channels ||
      (int)n->beta.size() != _cfg.channels) {
    if (err != nullptr) { *err = "cannot read " + name; }
    return false;
  }
  _weight_bytes += 2 * (std::size_t)_cfg.channels * 4;
  return true;
}

bool
MetalMiniMaxH3LatentUpscaler::load_entry_(WeightSet& ws,
                                          const std::string& name,
                                          bool temporal, Entry* e,
                                          std::string* err)
{
  e->temporal = temporal;
  const int C = _cfg.channels;
  if (temporal) {
    Temporal& t = e->tc;
    if (!load_norm_(ws, name + ".norm", &t.norm, err) ||
        !load_conv_(ws, name + ".pwconv", &t.pw, err)) {
      return false;
    }
    std::vector<std::int64_t> sh;
    const std::vector<float> dw =
        read_f32_(ws.src(), _mc, name + ".dwconv.weight", &sh);
    const std::vector<float> db =
        read_f32_(ws.src(), _mc, name + ".dwconv.bias");
    const int K = _cfg.temporal_kernel;
    if (sh.size() != 5 || sh[0] != C || sh[1] != 1 || sh[2] != K ||
        sh[3] != 1 || sh[4] != 1 || (int)db.size() != C ||
        t.pw.cin != C || t.pw.cout != C || t.pw.taps != 1) {
      if (err != nullptr) { *err = name + " is not a TemporalConv"; }
      return false;
    }
    t.dw_w = f32_buf_(_mc, dw.data(), dw.size());
    t.dw_b = f32_buf_(_mc, db.data(), db.size());
    _weight_bytes += (dw.size() + db.size()) * 4;
    return !t.dw_w.empty() && !t.dw_b.empty();
  }
  ResBlock& r = e->res;
  if (!load_norm_(ws, name + ".in_layers.0", &r.in_norm, err) ||
      !load_conv_(ws, name + ".in_layers.2", &r.conv1, err) ||
      !load_norm_(ws, name + ".out_norm", &r.out_norm, err) ||
      !load_conv_(ws, name + ".out_layers.2", &r.conv2, err)) {
    return false;
  }
  std::vector<std::int64_t> sh;
  r.emb_w = read_f32_(ws.src(), _mc, name + ".emb_layers.1.weight", &sh);
  r.emb_b = read_f32_(ws.src(), _mc, name + ".emb_layers.1.bias");
  if (sh.size() != 2 || sh[0] != 2 * C || sh[1] != _cfg.embed_dim ||
      (int)r.emb_b.size() != 2 * C || r.conv1.taps != 27 ||
      r.conv2.taps != 27 || r.conv1.cin != C || r.conv1.cout != C ||
      r.conv2.cin != C || r.conv2.cout != C) {
    // A ResBlock that changes its width would need the 1x1 skip the node
    // builds for it; none of the released ones does.
    if (err != nullptr) { *err = name + " is not a C -> C ResBlock"; }
    return false;
  }
  _weight_bytes += (r.emb_w.size() + r.emb_b.size()) * 4;
  return true;
}

std::unique_ptr<MetalMiniMaxH3LatentUpscaler>
MetalMiniMaxH3LatentUpscaler::load(std::shared_ptr<WeightSet> ws,
                                   MetalCompute* mc, std::string* err,
                                   int i8_tier)
{
  auto no = [&](std::string m) {
    if (err != nullptr && err->empty()) { *err = std::move(m); }
    return nullptr;
  };
  if (!ws || mc == nullptr) { return no("no weights or no metal-compute"); }
  auto m = std::unique_ptr<MetalMiniMaxH3LatentUpscaler>(
      new MetalMiniMaxH3LatentUpscaler());
  m->_ws = std::move(ws);
  m->_mc = mc;
  if (!config_from_weights(*m->_ws, &m->_cfg, err)) { return nullptr; }

  // The element type. f16 by default: the residual stream reaches ~8e3
  // (measured on the golden's latents), well inside f16, which carries
  // four times bf16's precision. VPIPE_H3UP_BF16=1 runs bf16 -- the
  // release's own precision, and the fallback upscale() takes by itself
  // when an f16 pass comes back non-finite.
  m->_bf16 = std::getenv("VPIPE_H3UP_BF16") != nullptr;
  m->_i8_want = i8_tier;
  if (!m->bind_kernels_()) {
    return no("a kernel the upscaler needs is missing from this build");
  }

  if (!m->load_all_(err)) { return nullptr; }
  return m;
}

bool
MetalMiniMaxH3LatentUpscaler::load_all_(std::string* err)
{
  auto no = [&](std::string m) {
    if (err != nullptr && err->empty()) { *err = std::move(m); }
    return false;
  };
  _weight_bytes = 0;
  WeightSet& w = *_ws;
  if (!load_conv_(w, "conv_in", &conv_in_, err) ||
      !load_conv_(w, "conv_out", &conv_out_, err) ||
      !load_norm_(w, "norm_out", &norm_out_, err)) {
    return false;
  }
  if (conv_in_.taps != 27 || conv_out_.taps != 27 ||
      conv_out_.cout != _cfg.in_channels) {
    return no("conv_in / conv_out are not the 3x3x3 head and tail");
  }
  auto load_stack = [&](const char* stack, const std::vector<bool>& kinds,
                        std::vector<Entry>* dst) {
    dst->clear();
    dst->resize(kinds.size());
    for (std::size_t i = 0; i < kinds.size(); ++i) {
      if (!load_entry_(w, fmt("{}.{}", stack, i)(), kinds[i], &(*dst)[i],
                       err)) {
        return false;
      }
    }
    return true;
  };
  if (!load_stack("in_blocks", _cfg.in_entries, &in_) ||
      !load_stack("out_blocks", _cfg.out_entries, &out_)) {
    return false;
  }
  std::vector<std::int64_t> s0, s2;
  embed0_w_ = read_f32_(w.src(), _mc, "embed.0.weight", &s0);
  embed0_b_ = read_f32_(w.src(), _mc, "embed.0.bias");
  embed2_w_ = read_f32_(w.src(), _mc, "embed.2.weight", &s2);
  embed2_b_ = read_f32_(w.src(), _mc, "embed.2.bias");
  const int E = _cfg.embed_dim;
  if ((int)embed0_w_.size() != E || (int)embed0_b_.size() != E ||
      (int)embed2_w_.size() != E * E || (int)embed2_b_.size() != E) {
    return no("the scale embedding is not [E, 1] -> [E, E]");
  }
  _weight_bytes += (std::size_t)(2 * E + E * E + E) * 4;
  return true;
}

bool
MetalMiniMaxH3LatentUpscaler::bind_kernels_()
{
  MetalCompute* mc = _mc;
  const char* sfx = _bf16 ? "_bf16" : "";
  _lib_gemm  = mc->load_library(std::string("dense_gemm") + sfx);
  _lib_elt   = mc->load_library(std::string("llm_elementwise") + sfx);
  _lib_regen = mc->load_library(std::string("minimax_h3_regenerate") + sfx);
  _fn_gemm_64   = _lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  _fn_gemm_32   = _lib_gemm.function("dense_gemm_t_f16");
  _fn_bias_add  = _lib_elt.function("bias_add_rows_f16");
  _fn_im2col3d  = _lib_elt.function("im2col_hwc_3x3x3_tiled_f16");
  _fn_gn_stats  = _lib_elt.function("group_norm_chan_stats_f16");
  _fn_gn_group  = _lib_elt.function("group_norm_group_stats_f32");
  _fn_residual  = _lib_elt.function("residual_add_f16");
  _fn_gn_apply  = _lib_regen.function("h3up_group_norm_apply_f16");
  _fn_dwconv    = _lib_regen.function("h3up_dwconv_t_f16");
  _fn_resize    = _lib_regen.function("h3up_resize_bilinear_f16");
  if (!_fn_gemm_64.valid() || !_fn_gemm_32.valid() ||
      !_fn_bias_add.valid() || !_fn_im2col3d.valid() ||
      !_fn_gn_stats.valid() || !_fn_gn_group.valid() ||
      !_fn_residual.valid() || !_fn_gn_apply.valid() ||
      !_fn_dwconv.valid() || !_fn_resize.valid()) {
    return false;
  }
  _use_mma2 = false;
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_H3UP_NO_MMA2") == nullptr) {
    _lib_mma = mc->load_library(std::string("dense_gemm_mma") + sfx);
    _fn_mma = _lib_mma.function("dense_gemm_mma_t_n128_f16");
    _fn_mma_wide = _lib_mma.function("dense_gemm_mma_t_n128x256_f16");
    _use_mma2 = _fn_mma.valid() && _fn_mma_wide.valid();
  }
  // The int8 context, in the element type (a bf16 fallback makes it
  // again). It declines without matrix cores; VPIPE_I8_GEMM /
  // VPIPE_I8_TIER override the asked tier.
  _i8.reset();
  _i8_tier = -1;
  auto i8 = std::make_unique<I8GemmContext>(mc, _i8_want >= 0, _bf16);
  i8->set_tier(_i8_want >= 2   ? I8GemmContext::Tier::kToken
               : _i8_want == 1 ? I8GemmContext::Tier::kFast
                               : I8GemmContext::Tier::kAccurate);
  i8->set_token_rotation(_i8_want == 2);
  if (i8->enabled()) {
    _i8 = std::move(i8);
    _i8_tier = _i8->tier() == I8GemmContext::Tier::kAccurate ? 0
               : _i8->tier() == I8GemmContext::Tier::kFast   ? 1
               : _i8->token_rotated()                        ? 2
                                                             : 3;
    // A held layout takes the int8 route at a gather band of any size.
    if (_i8_tier >= 1) { _i8->set_min_rows(1); }
  }
  return true;
}

int
MetalMiniMaxH3LatentUpscaler::effective_i8_tier(int i8_tier,
                                                bool matrix_cores)
{
  bool on = i8_tier >= 0;
  if (const char* e = std::getenv("VPIPE_I8_GEMM")) { on = std::atoi(e) != 0; }
  if (!on || !matrix_cores) { return -1; }
  int t = std::clamp(i8_tier, 0, 3);
  bool rot = t == 2;
  if (const char* e = std::getenv("VPIPE_I8_TIER")) {
    if (std::strcmp(e, "token") == 0 || std::strcmp(e, "2") == 0) {
      t = 2;
    } else if (std::strcmp(e, "fast") == 0 || std::strcmp(e, "1") == 0) {
      t = 1;
    } else if (std::strcmp(e, "accurate") == 0 || std::strcmp(e, "0") == 0) {
      t = 0;
    }
  }
  if (const char* e = std::getenv("VPIPE_I8_TOKEN_ROT")) {
    rot = std::atoi(e) != 0;
  }
  if (t >= 2) { return rot ? 2 : 3; }
  return t;
}

std::size_t
MetalMiniMaxH3LatentUpscaler::weight_bytes_for(const std::string& file,
                                               int i8_tier)
{
  // As load_all_ holds them: convs (and their biases) in the element type
  // -- or a 3x3x3's int8 layout -- the depthwise convs, norms and
  // embeddings in f32.
  auto wts = MetalLlamaWeights::open_model(file);
  if (!wts.has_value()) { return 0; }
  std::size_t b = 0;
  for (const std::string& nm : wts->tensor_names()) {
    const auto* ti = wts->info(nm);
    if (ti == nullptr) { continue; }
    std::size_t n = 1;
    for (auto d : ti->shape) { n *= (std::size_t)d; }
    const auto& sh = ti->shape;
    const bool conv = sh.size() == 5 && !(sh[1] == 1 && sh[3] == 1 &&
                                           sh[4] == 1 && sh[2] > 1);
    if (conv) {
      const int taps = (int)(sh[2] * sh[3] * sh[4]);
      const int K = taps * (int)sh[1];
      int gc = 0;
      if (i8_tier >= 2 && taps == 27 && K % 64 == 0 && K <= 16384) {
        gc = I8GemmContext::kTokenGroup;
      } else if (i8_tier == 1 && taps == 27) {
        for (int g : {1024, 512, 256}) {
          if (g <= 512 && K % g == 0) { gc = g; break; }
        }
      }
      b += gc > 0 ? n + I8GemmContext::fast_scale_bytes((int)sh[0], K, gc)
                  : n * 2;
      continue;
    }
    // A conv's bias: the element type; everything else f32.
    const std::string stem =
        nm.size() > 5 && nm.compare(nm.size() - 5, 5, ".bias") == 0
            ? nm.substr(0, nm.size() - 5)
            : std::string();
    const auto* wi = stem.empty() ? nullptr : wts->info(stem + ".weight");
    const bool conv_bias =
        wi != nullptr && wi->shape.size() == 5 &&
        !(wi->shape[1] == 1 && wi->shape[3] == 1 && wi->shape[4] == 1 &&
          wi->shape[2] > 1);
    b += conv_bias ? n * 2 : n * 4;
  }
  return b;
}

// y[M, cout] = x[M, K] W^T + bias, from element offsets xe / ye.
void
MetalMiniMaxH3LatentUpscaler::gemm_(ComputeEncoder& enc,
                                    const SharedBuffer& x, std::size_t xe,
                                    const Conv& c, const SharedBuffer& y,
                                    std::size_t ye, int M, int K)
{
  const int N = c.cout;
  if (c.gc > 0) {
    // A held layout: its tier's GEMM, else the layout widened.
    if (_i8->gemm_fast(enc, x, xe, c.codes, c.scales, c.gc, c.b, y, ye, M,
                       N, K)) {
      ++_i8_gemms;
      return;
    }
    const std::size_t need = (std::size_t)N * K * 2;
    if (_wdeq.byte_size() < need) {
      _wdeq = _mc->make_shared_buffer(need);
      if (_wdeq.empty()) { return; }
    }
    if (!_i8->encode_dequant_fast(enc, c.codes, c.scales, c.gc, _wdeq, N,
                                  K)) {
      return;
    }
    dense_(enc, x, xe, _wdeq, c.b, N, y, ye, M, K);
    return;
  }
  // The accurate tier: the weight quantized per call; a shape it
  // declines falls through with nothing encoded.
  if (_i8 && _i8->gemm(enc, x, xe, c.w, c.b, y, ye, M, N, K)) {
    ++_i8_gemms;
    return;
  }
  dense_(enc, x, xe, c.w, c.b, N, y, ye, M, K);
}

void
MetalMiniMaxH3LatentUpscaler::dense_(ComputeEncoder& enc,
                                     const SharedBuffer& x, std::size_t xe,
                                     const SharedBuffer& w,
                                     const SharedBuffer& bias, int N,
                                     const SharedBuffer& y, std::size_t ye,
                                     int M, int K)
{
  if (_use_mma2 && M >= _mma_min_m && N >= 16 && M <= mma_row_band(N, K)) {
    const bool wide = mma_use_wide_tile(N, K);
    const int BN = wide ? 256 : 128;
    enc.set_function(wide ? _fn_mma_wide : _fn_mma);
    enc.set_buffer(0, x, xe * 2);
    enc.set_buffer(1, w);
    enc.set_buffer(2, w);        // the bias slot, unused
    enc.set_buffer(3, y, ye * 2);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    enc.set_function(_fn_bias_add);
    enc.set_buffer(0, y, ye * 2);
    enc.set_buffer(1, bias);
    enc.set_constant(2, N);
    const unsigned total = (unsigned)((std::size_t)M * N);
    enc.set_constant(3, total);
    enc.dispatch({(unsigned)N, (unsigned)M, 1}, {256, 1, 1});
    return;
  }
  const bool w64 = M >= 128;
  const int bm = w64 ? 64 : 32, bn = w64 ? 64 : 32;
  enc.set_function(w64 ? _fn_gemm_64 : _fn_gemm_32);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, w);
  enc.set_buffer(2, bias);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 1);
  enc.dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});
}

// A 3x3x3 conv (zero padding in time and space) or a 1x1x1 one over a
// whole [T, hh * ww, cin] activation into [T, hh * ww, cout].
void
MetalMiniMaxH3LatentUpscaler::conv3_(Pass& p, const Conv& c,
                                     const SharedBuffer& x,
                                     const SharedBuffer& y, int T, int hh,
                                     int ww)
{
  ComputeEncoder& enc = *p.enc;
  const std::size_t rows = (std::size_t)hh * ww;
  if (c.taps == 1) {
    // A frame at a time: matmul2d's row limits bind long before a clip's
    // rows run out.
    for (int t = 0; t < T; ++t) {
      gemm_(enc, x, (std::size_t)t * rows * c.cin, c, y,
            (std::size_t)t * rows * c.cout, (int)rows, c.cin);
    }
    count_(2ull * rows * T * (std::uint64_t)c.cout * c.cin);
    return;
  }
  const int K = 27 * c.cin;
  for (int t = 0; t < T; ++t) {
    // Tap kt reads frame t + kt - 1; one before the first or past the
    // last reads as zero, and is not counted -- the zeros are padding.
    int mask = 2;
    if (t > 0) { mask |= 1; }
    if (t + 1 < T) { mask |= 4; }
    const int live = 1 + (mask & 1) + ((mask >> 2) & 1);
    count_(2ull * rows * (std::uint64_t)c.cout * 9 * live * c.cin);
    const std::size_t f = (std::size_t)t * rows * c.cin;
    const std::size_t taps[3] = {
      (mask & 1) ? f - rows * c.cin : f, f, (mask & 4) ? f + rows * c.cin : f};
    for (std::size_t r0 = 0; r0 < rows; r0 += kGatherRows) {
      const int mc = (int)std::min<std::size_t>(kGatherRows, rows - r0);
      enc.set_function(_fn_im2col3d);
      for (int k = 0; k < 3; ++k) {
        enc.set_buffer((unsigned)k, x, taps[k] * 2);
      }
      enc.set_buffer(3, p.col);
      enc.set_constant(4, hh);
      enc.set_constant(5, ww);
      enc.set_constant(6, c.cin);
      enc.set_constant(7, (int)r0);
      enc.set_constant(8, mc);
      enc.set_constant(9, mask);
      enc.dispatch({(unsigned)K, (unsigned)mc, 1}, {64, 1, 1});
      gemm_(enc, p.col, 0, c, y,
            ((std::size_t)t * rows + r0) * (std::size_t)c.cout, mc, K);
    }
  }
}

// GroupNorm over the whole [rows, C] activation, with norm `slot`'s
// folded affine, then SiLU.
void
MetalMiniMaxH3LatentUpscaler::norm_(Pass& p, const SharedBuffer& x,
                                    const SharedBuffer& y, std::size_t rows,
                                    int slot, bool silu)
{
  ComputeEncoder& enc = *p.enc;
  const int C = p.C, G = _cfg.groups;
  const int NB = (int)std::min<std::size_t>(kStatBlocks, rows);
  enc.set_function(_fn_gn_stats);
  enc.set_buffer(0, x);
  enc.set_buffer(1, p.part);
  enc.set_constant(2, (int)rows);
  enc.set_constant(3, C);
  enc.set_constant(4, NB);
  enc.dispatch({(unsigned)(256 * NB), 1, 1}, {256, 1, 1});
  enc.set_function(_fn_gn_group);
  enc.set_buffer(0, p.part);
  enc.set_buffer(1, p.stats);
  enc.set_constant(2, (int)rows);
  enc.set_constant(3, C);
  enc.set_constant(4, G);
  enc.set_constant(5, NB);
  enc.set_constant(6, _cfg.eps);
  enc.dispatch({(unsigned)(256 * G), 1, 1}, {256, 1, 1});
  enc.set_function(_fn_gn_apply);
  enc.set_buffer(0, x);
  enc.set_buffer(1, p.stats);
  enc.set_buffer(2, p.ab, (std::size_t)slot * 2 * C * 4);
  enc.set_buffer(3, p.ab, ((std::size_t)slot * 2 + 1) * C * 4);
  enc.set_buffer(4, y);
  enc.set_constant(5, C);
  enc.set_constant(6, G);
  enc.set_constant(7, silu ? 1 : 0);
  enc.dispatch({(unsigned)C, (unsigned)rows, 1}, {256, 1, 1});
}

// One stack entry over the activation x (updated in place), with a and b
// as scratch of x's size. `slot` walks the norms' affines in the order
// upscale() folded them.
void
MetalMiniMaxH3LatentUpscaler::entry_(Pass& p, const Entry& e, int T, int hh,
                                     int ww, const SharedBuffer& x,
                                     const SharedBuffer& a,
                                     const SharedBuffer& b, int* slot)
{
  ComputeEncoder& enc = *p.enc;
  const int C = p.C;
  const std::size_t R = (std::size_t)hh * ww;
  const std::size_t rows = R * T;
  const std::size_t n = rows * C;
  if (e.temporal) {
    norm_(p, x, a, rows, (*slot)++, true);
    const int K = _cfg.temporal_kernel;
    enc.set_function(_fn_dwconv);
    enc.set_buffer(0, a);
    enc.set_buffer(1, e.tc.dw_w);
    enc.set_buffer(2, e.tc.dw_b);
    enc.set_buffer(3, b);
    enc.set_constant(4, T);
    enc.set_constant(5, (int)R);
    enc.set_constant(6, C);
    enc.set_constant(7, K);
    enc.dispatch({(unsigned)C, (unsigned)R, (unsigned)T}, {64, 4, 1});
    count_(2ull * (std::uint64_t)n * (std::uint64_t)K);
    conv3_(p, e.tc.pw, b, a, T, hh, ww);
  } else {
    norm_(p, x, a, rows, (*slot)++, true);
    conv3_(p, e.res.conv1, a, b, T, hh, ww);
    norm_(p, b, a, rows, (*slot)++, true);
    conv3_(p, e.res.conv2, a, b, T, hh, ww);
    // the residual comes from b, not a
    enc.set_function(_fn_residual);
    enc.set_buffer(0, x);
    enc.set_buffer(1, b);
    enc.set_buffer(2, x);
    enc.set_constant(3, (int)n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    return;
  }
  enc.set_function(_fn_residual);
  enc.set_buffer(0, x);
  enc.set_buffer(1, a);
  enc.set_buffer(2, x);
  enc.set_constant(3, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

bool
MetalMiniMaxH3LatentUpscaler::upscale(
    const float* z, int T, int h, int w, int H, int W, double scale,
    Normalize norm, std::vector<float>* out, std::string* err,
    const std::function<void(int, int)>& progress)
{
  auto no = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (z == nullptr || out == nullptr || T <= 0 || h <= 0 || w <= 0 ||
      H <= 0 || W <= 0) {
    return no("upscale: an empty request");
  }
  const int Z = _cfg.in_channels, C = _cfg.channels;
  const std::size_t Rin = (std::size_t)h * w, Rout = (std::size_t)H * W;
  if ((std::size_t)T * Rout * C > 0x7fffffffull) {
    return no(fmt("upscale: {} x {} x {} at {} channels is past 32-bit "
                  "element offsets", T, H, W, C)());
  }

  // The scale embedding and every norm's folded affine, on the host in
  // f32: e = embed(scale - 1), and per ResBlock a modulation (scale,
  // shift) = W_b SiLU(e) + b_b that its second norm folds as gamma (1 + s)
  // and beta (1 + s) + shift.
  const int E = _cfg.embed_dim;
  std::vector<float> e0((std::size_t)E), emb((std::size_t)E);
  for (int i = 0; i < E; ++i) {
    e0[(std::size_t)i] = silu_(embed0_w_[(std::size_t)i] *
                                   (float)(scale - 1.0) +
                               embed0_b_[(std::size_t)i]);
  }
  for (int o = 0; o < E; ++o) {
    float s = embed2_b_[(std::size_t)o];
    for (int i = 0; i < E; ++i) {
      s += embed2_w_[(std::size_t)o * E + i] * e0[(std::size_t)i];
    }
    emb[(std::size_t)o] = silu_(s);   // emb_layers' leading SiLU
  }
  std::vector<float> ab;
  auto push = [&](const Norm& n, const float* mod) {
    const std::size_t at = ab.size();
    ab.resize(at + 2 * (std::size_t)C);
    for (int c = 0; c < C; ++c) {
      float g = n.gamma[(std::size_t)c], b = n.beta[(std::size_t)c];
      if (mod != nullptr) {
        const float s = 1.0f + mod[c], sh = mod[C + c];
        b = b * s + sh;
        g = g * s;
      }
      ab[at + (std::size_t)c] = g;
      ab[at + (std::size_t)C + c] = b;
    }
  };
  std::vector<float> mod((std::size_t)2 * C);
  auto push_stack = [&](const std::vector<Entry>& st) {
    for (const Entry& e : st) {
      if (e.temporal) { push(e.tc.norm, nullptr); continue; }
      push(e.res.in_norm, nullptr);
      for (int o = 0; o < 2 * C; ++o) {
        float s = e.res.emb_b[(std::size_t)o];
        const float* wr = e.res.emb_w.data() + (std::size_t)o * E;
        for (int i = 0; i < E; ++i) { s += wr[i] * emb[(std::size_t)i]; }
        mod[(std::size_t)o] = s;
      }
      push(e.res.out_norm, mod.data());
    }
  };
  push_stack(in_);
  push_stack(out_);
  push(norm_out_, nullptr);

  // The input, channel-last per frame, normalized as asked.
  std::vector<float> xin((std::size_t)T * Rin * Z);
  for (int c = 0; c < Z; ++c) {
    const float m = norm == Normalize::kRelease ? kMean[c] : 0.0f;
    const float s = norm == Normalize::kRelease ? kStd[c] : 1.0f;
    for (int t = 0; t < T; ++t) {
      const float* src = z + ((std::size_t)c * T + t) * Rin;
      float* dst = xin.data() + (std::size_t)t * Rin * Z + c;
      for (std::size_t r = 0; r < Rin; ++r) {
        dst[r * Z] = (src[r] - m) / s;
      }
    }
  }

  MetalCompute* mc = _mc;
  Pass p;
  p.C = C;
  p.col = mc->make_shared_buffer((std::size_t)kGatherRows * 27 * C * 2);
  p.part = mc->make_shared_buffer((std::size_t)kStatBlocks * 2 * C * 4);
  p.stats = mc->make_shared_buffer((std::size_t)_cfg.groups * 2 * 4);
  p.ab = f32_buf_(mc, ab.data(), ab.size());
  SharedBuffer x0 = elt_buf_(mc, xin.data(), xin.size(), _bf16);
  const std::size_t nin = (std::size_t)T * Rin * C;
  const std::size_t nout = (std::size_t)T * Rout * C;
  SharedBuffer xi = mc->make_shared_buffer(nin * 2);
  SharedBuffer ai = mc->make_shared_buffer(nin * 2);
  SharedBuffer bi = mc->make_shared_buffer(nin * 2);
  if (p.col.empty() || p.part.empty() || p.stats.empty() || p.ab.empty() ||
      x0.empty() || xi.empty() || ai.empty() || bi.empty()) {
    return no(fmt("upscale: cannot allocate the input stack's {} MB",
                  (4 * nin * 2) >> 20)());
  }

  const int total = (int)(in_.size() + out_.size()) + 2;
  int done = 0, slot = 0;
  _flops.begin();
  _counted = 0;
  _i8_gemms = 0;
  if (_i8) { _i8->tune_pending(mc); }
  CommandStream st = mc->make_command_stream();
  ComputeEncoder enc = st.begin_compute();
  p.enc = &enc;
  std::string why;
  auto sync = [&]() {
    _flops.flush(st);
    if (!CommandStream::sync_and_reopen(enc, &why)) { return false; }
    if (progress) { progress(++done, total); }
    return true;
  };

  conv3_(p, conv_in_, x0, xi, T, h, w);
  for (const Entry& e : in_) {
    entry_(p, e, T, h, w, xi, ai, bi, &slot);
    if (!sync()) { return no("upscale: the input stack failed: " + why); }
  }
  SharedBuffer xo = mc->make_shared_buffer(nout * 2);
  if (xo.empty()) {
    return no(fmt("upscale: cannot allocate the {} MB output activation",
                  (nout * 2) >> 20)());
  }
  enc.set_function(_fn_resize);
  enc.set_buffer(0, xi);
  enc.set_buffer(1, xo);
  enc.set_constant(2, h);
  enc.set_constant(3, w);
  enc.set_constant(4, H);
  enc.set_constant(5, W);
  enc.set_constant(6, C);
  enc.set_constant(7, (float)h / (float)H);
  enc.set_constant(8, (float)w / (float)W);
  enc.dispatch({(unsigned)C, (unsigned)Rout, (unsigned)T}, {64, 4, 1});
  if (!sync()) { return no("upscale: the resize failed: " + why); }
  // Done with the input stack before the output stack's scratch exists.
  xi = SharedBuffer{};
  ai = SharedBuffer{};
  bi = SharedBuffer{};
  x0 = SharedBuffer{};
  SharedBuffer ao = mc->make_shared_buffer(nout * 2);
  SharedBuffer bo = mc->make_shared_buffer(nout * 2);
  if (ao.empty() || bo.empty()) {
    return no(fmt("upscale: cannot allocate the output stack's {} MB",
                  (3 * nout * 2) >> 20)());
  }
  for (const Entry& e : out_) {
    entry_(p, e, T, H, W, xo, ao, bo, &slot);
    if (!sync()) { return no("upscale: the output stack failed: " + why); }
  }
  norm_(p, xo, ao, (std::size_t)T * Rout, slot++, true);
  SharedBuffer y = mc->make_shared_buffer((std::size_t)T * Rout * Z * 2);
  if (y.empty()) { return no("upscale: cannot allocate the output"); }
  conv3_(p, conv_out_, ao, y, T, H, W);
  enc.end();
  _flops.flush(st);
  if (!st.commit().wait_ok(&why)) {
    return no("upscale: the head failed: " + why);
  }
  if (progress) { progress(++done, total); }
  _last_flops = _counted;

  // Back to [Z, T, H, W] f32, denormalized as it was normalized.
  out->assign((std::size_t)Z * T * Rout, 0.0f);
  bool finite = true;
  if (_bf16) {
    const auto* d = static_cast<const std::uint16_t*>(y.contents());
    for (std::size_t i = 0; i < (std::size_t)T * Rout * Z; ++i) {
      const std::uint32_t u = (std::uint32_t)d[i] << 16;
      float v;
      std::memcpy(&v, &u, 4);
      const std::size_t r = i / Z, c = i % Z;
      (*out)[c * T * Rout + r] = v;
    }
  } else {
    const auto* d = static_cast<const _Float16*>(y.contents());
    for (std::size_t i = 0; i < (std::size_t)T * Rout * Z; ++i) {
      const std::size_t r = i / Z, c = i % Z;
      (*out)[c * T * Rout + r] = (float)d[i];
    }
  }
  for (int c = 0; c < Z; ++c) {
    const float m = norm == Normalize::kRelease ? kMean[c] : 0.0f;
    const float s = norm == Normalize::kRelease ? kStd[c] : 1.0f;
    float* o = out->data() + (std::size_t)c * T * Rout;
    for (std::size_t i = 0; i < (std::size_t)T * Rout; ++i) {
      o[i] = o[i] * s + m;
      finite = finite && std::isfinite(o[i]);
    }
  }
  if (!finite) {
    if (!_bf16) {
      // f16 ran out of range somewhere in the residual stream: the same
      // pass in bf16, whose range is f32's.
      if (mc->session() != nullptr) {
        mc->session()->warn(fmt("MetalMiniMaxH3LatentUpscaler: the f16 pass "
                                "came back non-finite; running it again "
                                "in bf16"));
      }
      _bf16 = true;
      if (!bind_kernels_()) { return no("upscale: no bf16 kernels"); }
      std::string lerr;
      if (!load_all_(&lerr)) { return no("upscale: " + lerr); }
      return upscale(z, T, h, w, H, W, scale, norm, out, err, progress);
    }
    return no("upscale: the result is not finite");
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
