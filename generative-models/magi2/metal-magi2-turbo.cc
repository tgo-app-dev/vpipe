#include "generative-models/magi2/metal-magi2-turbo.h"
#include "generative-models/magi2/magi2-latent.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <map>
#include <fstream>

namespace vpipe {
namespace genai {

using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::ComputeLibrary;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
namespace fs = std::filesystem;

namespace {

using magi2::kLatentMean;
using magi2::kLatentStd;

std::uint16_t
to_bf16(float v)
{
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) { return (std::uint16_t)(u >> 16); }
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

// The GEMM's im2col staging is cut to stay under this (bytes).
constexpr std::size_t kColBudget = 256u << 20;

}  // namespace

struct MetalMagi2Turbo::Kernels {
  ComputeLibrary lib, gemm;
  ComputeFunction im2col, im2col4, rms_act, up2, time_split, unpatch,
      latent_in, add,
      dgemm, dgemm_bn64;
};

MetalMagi2Turbo::~MetalMagi2Turbo() = default;

std::unique_ptr<MetalMagi2Turbo>
MetalMagi2Turbo::load(std::shared_ptr<WeightSet> ckpt, const std::string& dir,
                      MetalCompute* mc, std::string* err)
{
  if (mc == nullptr) {
    if (err != nullptr) { *err = "MAGI-2 turbo decoder: no metal device"; }
    return nullptr;
  }
  std::unique_ptr<MetalMagi2Turbo> m(new MetalMagi2Turbo());
  m->_mc = mc;
  m->_ws = std::move(ckpt);
  if (!m->load_(dir, err)) { return nullptr; }
  return m;
}

std::unique_ptr<MetalMagi2Turbo>
MetalMagi2Turbo::load(const std::string& dir, MetalCompute* mc,
                      std::string* err)
{
  return load(nullptr, dir, mc, err);
}

bool
MetalMagi2Turbo::plan(const std::string& dir, int T, int h, int w,
                      std::size_t* weights, std::size_t* scratch)
{
  // Memoized per directory: the structure is the checkpoint's.
  static std::mutex mu;
  static std::map<std::string, std::shared_ptr<MetalMagi2Turbo>> memo;
  std::shared_ptr<MetalMagi2Turbo> m;
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = memo.find(dir);
    if (it != memo.end()) { m = it->second; }
  }
  if (!m) {
    m.reset(new MetalMagi2Turbo());
    if (!m->load_(dir, nullptr)) { return false; }
    m->_ws.reset();
    std::lock_guard<std::mutex> lk(mu);
    memo[dir] = m;
  }
  if (weights != nullptr) { *weights = (std::size_t)m->_weight_bytes; }
  if (scratch != nullptr) { *scratch = m->scratch_bytes(T, h, w); }
  return true;
}

bool
MetalMagi2Turbo::conv_(const std::string& name, Conv* c, std::string* err)
{
  // The checkpoint is a training state: `state_dict` holds the raw
  // generator (gen_model.) and discriminator, `ema_state_dict` the EMA
  // model the release decodes with, under its DDP prefix (module.) --
  // "ema_state_dict.module.decoder.*" as the archive reader names it.
  const std::string p = "ema_state_dict.module.decoder." + name;
  const auto* wi = _ws->src().info(p + ".weight");
  const auto* bi = _ws->src().info(p + ".bias");
  if (wi == nullptr || bi == nullptr || wi->dtype != "F32" ||
      bi->dtype != "F32") {
    if (err != nullptr) {
      std::string seen;
      int k = 0;
      for (const std::string& nm : _ws->src().tensor_names()) {
        if (nm.find("decoder.conv_in.conv.w") == std::string::npos ||
            k++ >= 6) {
          continue;
        }
        seen += " '" + nm + "'";
      }
      *err = "MAGI-2 turbo decoder: no f32 '" + p + ".weight' / bias" +
             (seen.empty() ? std::string() : " (has" + seen + ")");
    }
    return false;
  }
  const auto& s = wi->shape;
  c->cout = (int)s[0];
  c->cin = (int)s[1];
  if (s.size() == 5) {
    c->kt = (int)s[2]; c->kh = (int)s[3]; c->kw = (int)s[4];
  } else if (s.size() == 4) {
    c->kt = 1; c->kh = (int)s[2]; c->kw = (int)s[3];
  } else {
    if (err != nullptr) { *err = "MAGI-2 turbo decoder: a conv's rank"; }
    return false;
  }
  const int taps = c->kt * c->kh * c->kw;
  if (_mc == nullptr) {               // the plan: shapes, no weights
    _weight_bytes += (std::uint64_t)c->cout * taps * c->cin * 2 +
                     (std::uint64_t)c->cout * 2;
    return true;
  }
  SharedBuffer raw = _ws->read(p + ".weight", _mc,
                               WeightSet::Residency::Copied);
  SharedBuffer rb = _ws->read(p + ".bias", _mc, WeightSet::Residency::Copied);
  if (raw.empty() || rb.empty()) {
    if (err != nullptr) { *err = "MAGI-2 turbo decoder: read " + p; }
    return false;
  }
  // [Cout, Cin, taps] -> [Cout, taps, Cin], bf16.
  c->w = _mc->make_shared_buffer((std::size_t)c->cout * taps * c->cin * 2);
  c->b = _mc->make_shared_buffer((std::size_t)c->cout * 2);
  const float* src = (const float*)raw.contents();
  auto* dst = (std::uint16_t*)c->w.contents();
  for (int o = 0; o < c->cout; ++o) {
    for (int i = 0; i < c->cin; ++i) {
      for (int k = 0; k < taps; ++k) {
        dst[((std::size_t)o * taps + k) * c->cin + i] =
            to_bf16(src[((std::size_t)o * c->cin + i) * taps + k]);
      }
    }
  }
  const float* bs = (const float*)rb.contents();
  auto* bd = (std::uint16_t*)c->b.contents();
  for (int o = 0; o < c->cout; ++o) { bd[o] = to_bf16(bs[o]); }
  _weight_bytes += c->w.byte_size() + c->b.byte_size();
  return true;
}

bool
MetalMagi2Turbo::load_(const std::string& dir, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr && err->empty()) { *err = "MAGI-2 turbo decoder: " + m; }
    return false;
  };
  // The config: the release's file name, or any one JSON beside the ckpt.
  fs::path cfgp;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (e.path().extension() == ".json") { cfgp = e.path(); break; }
  }
  std::ifstream in(cfgp);
  if (!in) { return fail("no config JSON in " + dir); }
  const FlexData cfg = FlexData::from_json(in);
  if (!cfg.is_object()) { return fail("an unreadable config"); }
  auto co = cfg.as_object();
  auto ints = [&](const char* k) {
    std::vector<int> v;
    if (!co.contains(k)) { return v; }
    const FlexData a = co.at(k);
    auto aa = a.as_array();
    for (std::size_t i = 0; i < aa.size(); ++i) {
      const FlexData x = aa.at(i);
      v.push_back(x.is_bool() ? (x.as_bool() ? 1 : 0) : (int)x.as_int(0));
    }
    return v;
  };
  std::vector<int> boc = ints("decoder_block_out_channels");
  std::vector<int> lpb = ints("decoder_layers_per_block");
  std::vector<int> scal = ints("decoder_spatio_temporal_scaling");
  std::vector<int> sonly = ints("decoder_spatio_only");
  std::vector<int> dw = ints("decoder_is_dw_conv");
  if (boc.size() != 4 || lpb.size() != 5 || scal.size() != 4 ||
      sonly.size() != 4) {
    return fail("the decoder's block lists");
  }
  for (int d : dw) {
    if (d != 0) { return fail("depthwise convs are not this port's"); }
  }
  if ((int)co.at("latent_channels").as_int(0) != 48 ||
      (int)co.at("patch_size").as_int(0) != 2 ||
      !co.at("use_unpatchify").as_bool(false) ||
      co.at("decoder_causal").as_bool(true)) {
    return fail("not the TinyShallow layout (48 latents, patch 2, "
                "unpatchify, non-causal)");
  }
  _first = (int)co.at("first_chunk_size").as_int(7);
  _step = (int)co.at("step_size").as_int(7);
  std::reverse(boc.begin(), boc.end());
  std::reverse(lpb.begin(), lpb.end());
  std::reverse(scal.begin(), scal.end());
  std::reverse(sonly.begin(), sonly.end());

  if (!_ws) {
    _ws = WeightSet::open((fs::path(dir) / "checkpoint.ckpt").string(),
                          nullptr);
  }
  if (!_ws) { return fail("checkpoint.ckpt did not open"); }

  if (!conv_("conv_in.conv", &_conv_in, err)) { return fail(""); }
  auto resnet = [&](const std::string& k, bool relu_first, Resnet* r) {
    r->relu_first = relu_first;
    if (!conv_(k + ".conv1.conv", &r->c1, err) ||
        !conv_(k + ".conv2.conv", &r->c2, err)) {
      return false;
    }
    r->shortcut = _ws->src().info("ema_state_dict.module.decoder." + k +
                                  ".conv_shortcut.conv.weight") != nullptr;
    return !r->shortcut || conv_(k + ".conv_shortcut.conv", &r->sc, err);
  };
  for (int i = 0; i < lpb[0]; ++i) {
    Resnet r;
    if (!resnet(fmt("mid_block.resnets.{}", i)(), false, &r)) {
      return fail("");
    }
    _mid.push_back(std::move(r));
  }
  for (int i = 0; i < 4; ++i) {
    Up u;
    const std::string k = fmt("up_blocks.{}", i)();
    u.has_in = _ws->src().info("ema_state_dict.module.decoder." + k +
                               ".conv_in.conv1.conv.weight") != nullptr;
    if (u.has_in && !resnet(k + ".conv_in", false, &u.in)) {
      return fail("");
    }
    if (scal[(std::size_t)i]) {
      u.mode = sonly[(std::size_t)i] ? 1 : 2;
      if (u.mode == 2 &&
          !conv_(k + ".upsamplers.0.time_conv.conv", &u.time, err)) {
        return fail("");
      }
      if (!conv_(k + ".upsamplers.0.resample.1", &u.resample, err)) {
        return fail("");
      }
    }
    for (int j = 0; j < lpb[(std::size_t)i + 1]; ++j) {
      Resnet r;
      if (!resnet(fmt("{}.resnets.{}", k, j)(), scal[(std::size_t)i] != 0,
                  &r)) {
        return fail("");
      }
      u.res.push_back(std::move(r));
    }
    _up.push_back(std::move(u));
  }
  if (!conv_("conv_out.conv", &_conv_out, err)) { return fail(""); }
  if (_mc == nullptr) { return true; }   // the plan stops at the shapes

  _k = std::make_unique<Kernels>();
  Kernels& k = *_k;
  k.lib = _mc->load_library("magi2_bf16");
  k.gemm = _mc->load_library("dense_gemm_bf16");
  struct Need {
    ComputeFunction* fn;
    const ComputeLibrary* lib;
    const char* name;
  };
  const Need need[] = {
      {&k.im2col, &k.lib, "magi2_im2col"},
      {&k.im2col4, &k.lib, "magi2_im2col4"},
      {&k.rms_act, &k.lib, "magi2_rms_act"},
      {&k.up2, &k.lib, "magi2_up2"},
      {&k.time_split, &k.lib, "magi2_time_split"},
      {&k.unpatch, &k.lib, "magi2_unpatch"},
      {&k.latent_in, &k.lib, "magi2_latent_in"},
      {&k.add, &k.lib, "magi2_add"},
      {&k.dgemm, &k.gemm, "dense_gemm_t_bm64_f16"},
      {&k.dgemm_bn64, &k.gemm, "dense_gemm_t_bm64bn64_f16"},
  };
  for (const Need& n : need) {
    *n.fn = n.lib->function(n.name);
    if (!n.fn->valid()) {
      return fail(fmt("kernel '{}' did not load", n.name)());
    }
  }
  _mean = _mc->make_shared_buffer(sizeof kLatentMean);
  _std = _mc->make_shared_buffer(sizeof kLatentStd);
  std::memcpy(_mean.contents(), kLatentMean, sizeof kLatentMean);
  std::memcpy(_std.contents(), kLatentStd, sizeof kLatentStd);
  _stream = _mc->make_command_stream();
  return true;
}

// The largest activation (elements) any op of a window of n latent
// frames writes, walking the decoder as window_ runs it.
std::size_t
MetalMagi2Turbo::act_elems_(int n, int h, int w, bool first) const
{
  std::size_t peak = 0;
  int C = _conv_in.cout, Tn = n, H = h, W = w;
  auto at = [&](int c) {
    peak = std::max(peak, (std::size_t)Tn * H * W * (std::size_t)c);
  };
  at(48);
  at(C);
  auto resnet = [&](const Resnet& r) {
    at(std::max({C, r.c1.cout, r.c2.cout}));
    C = r.c2.cout;
  };
  for (const Resnet& r : _mid) { resnet(r); }
  for (const Up& u : _up) {
    if (u.has_in) { resnet(u.in); }
    if (u.mode == 2) {
      at(2 * C);
      Tn = 2 * Tn - (first ? 1 : 0);
      at(C);
    }
    if (u.mode != 0) {
      H *= 2;
      W *= 2;
      at(std::max(C, u.resample.cout));
      C = u.resample.cout;
    }
    for (const Resnet& r : u.res) { resnet(r); }
  }
  at(std::max(C, _conv_out.cout));
  return peak;
}

namespace {

std::size_t
clip_bytes(int F, int h, int w)
{
  return (std::size_t)3 * F * 16 * h * 16 * w * 4;
}

}  // namespace

// The whole plane's: the work buffers at the widest window -- the first
// (first + 1 latent frames, or the whole padded latent when it is one
// window) or an interior one (step + 2) -- the im2col band, the latent,
// the clip on the GPU and its copy handed back.
std::size_t
MetalMagi2Turbo::whole_scratch_(int T, int h, int w) const
{
  std::size_t act = act_elems_(std::max(T, _first), h, w, true);
  if (T > _first) {
    act = std::max(act_elems_(_first + 1, h, w, true),
                   act_elems_(_step + 2, h, w, false));
  }
  return 3 * act * 2 + kColBudget + (std::size_t)48 * T * h * w * 4 +
         2 * clip_bytes(frames_of(T), h, w);
}

double
MetalMagi2Turbo::receptive_cells() const
{
  double r = 0.0, scale = 1.0;
  auto conv = [&](const Conv& c) {
    r += (double)((c.kh - 1) / 2) / scale;
  };
  auto resnet = [&](const Resnet& rn) {
    conv(rn.c1);
    conv(rn.c2);
    if (rn.shortcut) { conv(rn.sc); }
  };
  conv(_conv_in);
  for (const Resnet& rn : _mid) { resnet(rn); }
  for (const Up& u : _up) {
    if (u.has_in) { resnet(u.in); }
    if (u.mode == 2) { conv(u.time); }
    if (u.mode != 0) {
      scale *= 2.0;
      conv(u.resample);
    }
    for (const Resnet& rn : u.res) { resnet(rn); }
  }
  conv(_conv_out);
  return r;
}

int
MetalMagi2Turbo::halo_cells() const
{
  return (int)std::ceil(receptive_cells() - 1e-9);
}

MetalMagi2Turbo::Tiling
MetalMagi2Turbo::tiling(int T, int h, int w) const
{
  Tiling t;
  t.halo = _tile_halo >= 0 ? _tile_halo : halo_cells();
  if (_tile_h > 0 && _tile_w > 0) {
    t.core_h = std::min(_tile_h, h);
    t.core_w = std::min(_tile_w, w);
  } else {
    t.core_h = h;
    t.core_w = w;
    // The work buffers' bytes at a tile of (core + halo), the widest.
    auto work = [&](int ch, int cw) {
      const int th = std::min(h, ch + 2 * t.halo);
      const int tw = std::min(w, cw + 2 * t.halo);
      const std::size_t whole = whole_scratch_(T, th, tw);
      return whole - kColBudget - (std::size_t)48 * T * th * tw * 4 -
             2 * clip_bytes(frames_of(T), th, tw);
    };
    // Fewest tiles first: grow the grid along the longer side.
    int ny = 1, nx = 1;
    while (work((h + ny - 1) / ny, (w + nx - 1) / nx) > kTileBudget) {
      const int ch = (h + ny - 1) / ny, cw = (w + nx - 1) / nx;
      if (ch <= 1 && cw <= 1) { break; }
      if (cw >= ch) { ++nx; } else { ++ny; }
    }
    t.core_h = (h + ny - 1) / ny;
    t.core_w = (w + nx - 1) / nx;
  }
  t.ny = (h + t.core_h - 1) / t.core_h;
  t.nx = (w + t.core_w - 1) / t.core_w;
  return t;
}

std::size_t
MetalMagi2Turbo::scratch_bytes(int T, int h, int w) const
{
  if (T <= 0 || h <= 0 || w <= 0) { return 0; }
  const Tiling t = tiling(T, h, w);
  if (t.ny == 1 && t.nx == 1) { return whole_scratch_(T, h, w); }
  // The widest tile's own decode, its latent crop on the host, and the
  // whole clip being assembled.
  const int th = std::min(h, t.core_h + 2 * t.halo);
  const int tw = std::min(w, t.core_w + 2 * t.halo);
  return whole_scratch_(T, th, tw) + (std::size_t)48 * T * th * tw * 4 +
         clip_bytes(frames_of(T), h, w);
}

// One window: latent frames [t0, t0 + n) of the (padded) latent decoded,
// its output frames [keep0, keep0 + keep_n) written at output frame f_out.
bool
MetalMagi2Turbo::window_(int t0, int n, bool first, int keep0, int keep_n,
                         int f_out, int T, int h, int w, std::string* err)
{
  const Kernels& k = *_k;
  // The three work buffers hold the window's largest activation.
  const std::size_t need = act_elems_(n, h, w, first) * 2;
  if (_a.byte_size() < need) {
    _a = _mc->make_shared_buffer(need);
    _b = _mc->make_shared_buffer(need);
    _c = _mc->make_shared_buffer(need);
    if (_a.empty() || _b.empty() || _c.empty()) {
      if (err != nullptr) { *err = "MAGI-2 turbo decoder: work buffers"; }
      return false;
    }
  }
  if (_col_cap < kColBudget) {
    _col = _mc->make_shared_buffer(kColBudget);
    _col_cap = kColBudget;
  }
  ComputeEncoder enc = _stream.begin_compute();
  // VPIPE_MAGI2_TURBO_PROFILE: a commit and wait after each op, its time
  // charged to its kind -- the split a decode's time takes, not its speed.
  static const bool prof = [] {
    const char* e = std::getenv("VPIPE_MAGI2_TURBO_PROFILE");
    return e != nullptr && e[0] == '1';
  }();
  static std::map<std::string, std::pair<double, int>> prof_acc;
  auto mark = [&](const std::string& what) {
    if (!prof) { return; }
    const auto t0 = std::chrono::steady_clock::now();
    enc.end();
    _flops.flush(_stream);
    _stream.commit().wait();
    auto& a = prof_acc[what];
    a.first += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    a.second += 1;
    enc = _stream.begin_compute();
  };
  // y[v, Cout] = conv(x) over the window [T', H, W] (channel-last).
  auto conv = [&](const Conv& c, const SharedBuffer& x, const SharedBuffer& y,
                  int Tn, int H, int W) {
    const long vox = (long)Tn * H * W;
    const int taps = c.kt * c.kh * c.kw;
    const int K = taps * c.cin;
    const bool bn64 = c.cout >= 512;
    auto gemm = [&](const SharedBuffer& a, std::size_t a_off, long rows,
                    std::size_t y_off) {
      _flops.gemm(rows, c.cout, K);
      enc.set_function(bn64 ? k.dgemm_bn64 : k.dgemm);
      enc.set_buffer(0, a, a_off);
      enc.set_buffer(1, c.w);
      enc.set_buffer(2, c.b);
      enc.set_buffer(3, y, y_off);
      enc.set_constant(4, K);
      enc.set_constant(5, c.cout);
      enc.set_constant(6, (int)rows);
      enc.set_constant(7, 1);
      const int bn = bn64 ? 64 : 32;
      enc.dispatch({(unsigned)(((c.cout + bn - 1) / bn) * 32),
                    (unsigned)(((rows + 63) / 64) * 2), 2}, {32, 2, 2});
    };
    if (taps == 1) {
      gemm(x, 0, vox, 0);
      mark(fmt("gemm 1x1 {}->{}", c.cin, c.cout)());
      return;
    }
    const long per = std::max<long>(
        64, (long)(kColBudget / ((std::size_t)K * 2)) / 64 * 64);
    for (long v0 = 0; v0 < vox; v0 += per) {
      const int nv = (int)std::min<long>(per, vox - v0);
      const bool four = c.cin % 4 == 0 && k.im2col4.valid();
      enc.set_function(four ? k.im2col4 : k.im2col);
      enc.set_buffer(0, x);
      enc.set_buffer(1, _col);
      enc.set_constant(2, Tn); enc.set_constant(3, H); enc.set_constant(4, W);
      enc.set_constant(5, c.cin);
      enc.set_constant(6, c.kt); enc.set_constant(7, c.kh);
      enc.set_constant(8, c.kw);
      enc.set_constant(9, v0);
      enc.set_constant(10, nv);
      if (four) {
        enc.dispatch({(unsigned)(taps * c.cin / 4), (unsigned)nv, 1},
                     {256, 1, 1});
      } else {
        enc.dispatch({(unsigned)c.cin, (unsigned)taps, (unsigned)nv},
                     {(unsigned)std::min(c.cin, 256), 1, 1});
      }
      mark(fmt("im2col {}x{}x{} {}", c.kt, c.kh, c.kw, c.cin)());
      gemm(_col, 0, nv, (std::size_t)v0 * c.cout * 2);
      mark(fmt("gemm {}x{}x{} {}->{}", c.kt, c.kh, c.kw, c.cin, c.cout)());
    }
  };
  auto rms_act = [&](const SharedBuffer& x, const SharedBuffer& y, long vox,
                     int C, float eps, int act) {
    enc.set_function(k.rms_act);
    enc.set_buffer(0, x); enc.set_buffer(1, y);
    enc.set_constant(2, C); enc.set_constant(3, eps); enc.set_constant(4, act);
    enc.dispatch({32, (unsigned)vox, 1}, {32, 1, 1});
    mark(fmt("rms_act {}", C)());
  };
  auto add = [&](const SharedBuffer& a, const SharedBuffer& b,
                 const SharedBuffer& y, long n) {
    enc.set_function(k.add);
    enc.set_buffer(0, a); enc.set_buffer(1, b); enc.set_buffer(2, y);
    enc.set_constant(3, (unsigned)n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    mark("add");
  };
  // A resnet in place on x (scratch s1, s2): x = conv2(silu(rms(conv1(
  // act(rms(x)))))) + shortcut(x).
  int C = 0, Tn = n, H = h, W = w;
  SharedBuffer* x = &_a;
  SharedBuffer* s1 = &_b;
  SharedBuffer* s2 = &_c;
  auto resnet = [&](const Resnet& r) {
    const long vox = (long)Tn * H * W;
    rms_act(*x, *s1, vox, C, 1e-8f, r.relu_first ? 2 : 1);
    conv(r.c1, *s1, *s2, Tn, H, W);
    rms_act(*s2, *s2, vox, r.c1.cout, 1e-8f, 1);
    conv(r.c2, *s2, *s1, Tn, H, W);           // s1 = h
    if (r.shortcut) {
      rms_act(*x, *s2, vox, C, 1e-6f, 0);
      conv(r.sc, *s2, *x, Tn, H, W);          // x = shortcut(x)
    }
    C = r.c2.cout;
    add(*s1, *x, *x, vox * C);
  };
  // The latent window in, un-whitened.
  enc.set_function(k.latent_in);
  enc.set_buffer(0, _z); enc.set_buffer(1, *s1);
  enc.set_buffer(2, _mean); enc.set_buffer(3, _std);
  enc.set_constant(4, T); enc.set_constant(5, h * w);
  enc.set_constant(6, 48); enc.set_constant(7, t0);
  enc.dispatch({48, (unsigned)(h * w), (unsigned)n}, {48, 1, 1});
  conv(_conv_in, *s1, *x, Tn, H, W);
  C = _conv_in.cout;
  for (const Resnet& r : _mid) { resnet(r); }
  for (const Up& u : _up) {
    if (u.has_in) { resnet(u.in); }
    if (u.mode != 0) {
      if (u.mode == 2) {
        conv(u.time, *x, *s1, Tn, H, W);      // [Tn, HW, 2C]
        const int skip = first ? 1 : 0;
        const int Tn2 = 2 * Tn - skip;
        enc.set_function(k.time_split);
        enc.set_buffer(0, *s1); enc.set_buffer(1, *x);
        enc.set_constant(2, H * W); enc.set_constant(3, C);
        enc.set_constant(4, skip);
        enc.dispatch({(unsigned)C, (unsigned)(H * W), (unsigned)Tn2},
                     {(unsigned)std::min(C, 256), 1, 1});
        Tn = Tn2;
      }
      enc.set_function(k.up2);
      enc.set_buffer(0, *x); enc.set_buffer(1, *s1);
      enc.set_constant(2, H); enc.set_constant(3, W); enc.set_constant(4, C);
      enc.dispatch({(unsigned)C, (unsigned)(4 * H * W), (unsigned)Tn},
                   {(unsigned)std::min(C, 256), 1, 1});
      H *= 2;
      W *= 2;
      conv(u.resample, *s1, *x, Tn, H, W);
      C = u.resample.cout;
    }
    for (const Resnet& r : u.res) { resnet(r); }
  }
  rms_act(*x, *s1, (long)Tn * H * W, C, 1e-8f, 1);
  conv(_conv_out, *s1, *x, Tn, H, W);
  const int Fo = std::min(keep_n, Tn - keep0);
  enc.set_function(k.unpatch);
  enc.set_buffer(0, *x, (std::size_t)keep0 * H * W * _conv_out.cout * 2);
  enc.set_buffer(1, _out);
  enc.set_constant(2, H); enc.set_constant(3, W);
  enc.set_constant(4, _conv_out.cout / 4);
  enc.set_constant(5, Fo);
  enc.set_constant(6, f_out);
  enc.set_constant(7, frames_of(T));
  enc.dispatch({(unsigned)(2 * W), (unsigned)(2 * H), (unsigned)Fo},
               {16, 16, 1});
  enc.end();
  _flops.flush(_stream);
  std::string ge;
  if (!_stream.commit().wait_ok(&ge)) {
    if (err != nullptr) { *err = "MAGI-2 turbo decoder: " + ge; }
    return false;
  }
  if (prof) {
    std::fprintf(stderr, "[turbo profile] window t0 %d n %d at %dx%d:\n", t0,
                 n, h, w);
    for (const auto& kv : prof_acc) {
      std::fprintf(stderr, "  %-28s %4d x  %8.1f ms\n", kv.first.c_str(),
                   kv.second.second, kv.second.first * 1e3);
    }
    prof_acc.clear();
  }
  return true;
}

void
MetalMagi2Turbo::release_scratch()
{
  _a = SharedBuffer();
  _b = SharedBuffer();
  _c = SharedBuffer();
  _col = SharedBuffer();
  _col_cap = 0;
  _z = SharedBuffer();
  _out = SharedBuffer();
}

bool
MetalMagi2Turbo::decode(const float* z, int T, int h, int w,
                        std::vector<float>* out, std::string* err)
{
  if (T <= 0 || h <= 0 || w <= 0) {
    if (err != nullptr) { *err = "MAGI-2 turbo decoder: an empty latent"; }
    return false;
  }
  // THE METER: what an aborted decode counted and never flushed goes.
  _flops.begin();
  const Tiling t = tiling(T, h, w);
  if (t.ny == 1 && t.nx == 1) { return decode_whole_(z, T, h, w, out, err); }
  // Tile by tile: the latent's cells [y0 - halo, y1 + halo) x [x0 - halo,
  // x1 + halo), clipped, decoded whole, and the core's pixels kept.
  const int F = frames_of(T);
  out->assign(clip_bytes(F, h, w) / 4, 0.0f);
  std::vector<float> zc, tile;
  for (int ty = 0; ty < t.ny; ++ty) {
    const int y0 = ty * t.core_h, y1 = std::min(h, y0 + t.core_h);
    const int ya = std::max(0, y0 - t.halo), yb = std::min(h, y1 + t.halo);
    for (int tx = 0; tx < t.nx; ++tx) {
      const int x0 = tx * t.core_w, x1 = std::min(w, x0 + t.core_w);
      const int xa = std::max(0, x0 - t.halo);
      const int xb = std::min(w, x1 + t.halo);
      const int th = yb - ya, tw = xb - xa;
      zc.resize((std::size_t)48 * T * th * tw);
      for (int ct = 0; ct < 48 * T; ++ct) {
        for (int y = 0; y < th; ++y) {
          std::memcpy(&zc[((std::size_t)ct * th + y) * tw],
                      z + ((std::size_t)ct * h + ya + y) * w + xa,
                      (std::size_t)tw * 4);
        }
      }
      if (!decode_whole_(zc.data(), T, th, tw, &tile, err)) { return false; }
      // The core's rows and columns, 16 pixels a cell.
      const int ow = 16 * w, tow = 16 * tw;
      const int ry0 = 16 * (y0 - ya), rx0 = 16 * (x0 - xa);
      const int rows = 16 * (y1 - y0), cols = 16 * (x1 - x0);
      for (int cf = 0; cf < 3 * F; ++cf) {
        for (int y = 0; y < rows; ++y) {
          std::memcpy(
              out->data() + ((std::size_t)cf * 16 * h + 16 * y0 + y) * ow +
                  16 * x0,
              tile.data() + ((std::size_t)cf * 16 * th + ry0 + y) * tow +
                  rx0,
              (std::size_t)cols * 4);
        }
      }
    }
  }
  release_scratch();
  return true;
}

bool
MetalMagi2Turbo::decode_whole_(const float* z, int T, int h, int w,
                               std::vector<float>* out, std::string* err)
{
  const int F = frames_of(T);
  _z = _mc->make_shared_buffer((std::size_t)48 * T * h * w * 4);
  std::memcpy(_z.contents(), z, (std::size_t)48 * T * h * w * 4);
  _out = _mc->make_shared_buffer((std::size_t)3 * F * 16 * h * 16 * w * 4);
  if (_z.empty() || _out.empty()) {
    if (err != nullptr) { *err = "MAGI-2 turbo decoder: buffers"; }
    return false;
  }
  // The release's windows over the padded latent; frames past the real
  // ones (the padding's) are never written.
  int n = T, pad = 0;
  if (n < _first) {
    pad = _first - n;
  } else if ((n - _first) % _step != 0) {
    pad = _step - (n - _first) % _step;
  }
  n += pad;
  const int ov = 4;
  if (n == _first) {
    if (!window_(0, n, true, 0, F, 0, T, h, w, err)) { return false; }
  } else {
    // First: frames [0, first + 1) of the latent, the context's 4 dropped.
    const int f0 = frames_of(_first + 1) - ov;
    if (!window_(0, _first + 1, true, 0, std::min(f0, F), 0, T, h, w, err)) {
      return false;
    }
    int fo = f0;
    for (int i = _first; i < n; i += _step) {
      const bool last = i + _step == n;
      const int left = i - 1;
      const int right = last ? i + _step : i + _step + 1;
      const int frames = 4 * (right - left);   // not the first window
      const int keep = last ? frames - ov : frames - 2 * ov;
      if (fo >= F) { break; }
      if (!window_(left, right - left, false, ov, std::min(keep, F - fo), fo,
                   T, h, w, err)) {
        return false;
      }
      fo += keep;
    }
  }
  const float* o = (const float*)_out.contents();
  out->assign(o, o + (std::size_t)3 * F * 16 * h * 16 * w);
  return true;
}

}  // namespace genai
}  // namespace vpipe
