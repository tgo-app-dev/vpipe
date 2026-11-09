#include "generative-models/rife/metal-rife.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace fs = std::filesystem;

namespace {

// The flow head's output in the conv form: 16 sub-positions (2 x 2
// transposed-conv parities x 2 x 2 pixel-shuffle positions) of 13
// channels, each padded to 16 where the hardware conv runs it (a width of
// 256 takes the op's 64-wide tile where 208 took the 16-wide one, and the
// unpack reads a sub-position as four half4s).
int
last_ks_(bool hw)
{
  return hw ? 16 : 13;
}

int
round32_(int c)
{
  return (c + 31) / 32 * 32;
}

// A tensor as host f32, whatever the file stores. Read UNCACHED: every
// weight is rebuilt into the convs' own layout and the source dropped.
bool
read_host_(WeightSet& ws, MetalCompute* mc, const std::string& nm,
           std::vector<float>* out)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr) { return false; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty() || raw.byte_size() < info->nbytes) { return false; }
  out->resize(n);
  if (info->dtype == "F32") {
    std::memcpy(out->data(), raw.contents(), n * 4);
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = (float)s[i]; }
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t u = (std::uint32_t)s[i] << 16;
      std::memcpy(&(*out)[i], &u, 4);
    }
  } else {
    return false;
  }
  return true;
}

// [cout, 9 * cin] (ky, kx, ci) -> HWIO [3, 3, cin, cout], the hardware
// op's layout.
std::vector<float>
hwio_(const std::vector<float>& g, int cout, int cin)
{
  std::vector<float> h(g.size());
  for (int o = 0; o < cout; ++o) {
    for (int t = 0; t < 9; ++t) {
      for (int c = 0; c < cin; ++c) {
        h[((std::size_t)t * cin + c) * cout + o] =
            g[(std::size_t)o * 9 * cin + (std::size_t)t * cin + c];
      }
    }
  }
  return h;
}

SharedBuffer
f16_buf_(MetalCompute* mc, const std::vector<float>& src)
{
  SharedBuffer b = mc->make_shared_buffer(std::max<std::size_t>(src.size(), 1)
                                          * 2);
  if (b.empty()) { return {}; }
  auto* d = static_cast<_Float16*>(b.contents());
  for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
  return b;
}

std::vector<std::int64_t>
shape_of_(const MetalLlamaWeights& src, const std::string& nm)
{
  const auto* i = src.info(nm);
  return i != nullptr ? i->shape : std::vector<std::int64_t>{};
}

bool
is_shape_(const std::vector<std::int64_t>& s,
          std::initializer_list<std::int64_t> want)
{
  return s.size() == want.size() &&
         std::equal(s.begin(), s.end(), want.begin());
}

// The driver's `scale_list = [16/scale, 8/scale, ...]`, as numerators.
// Data read off a file beside the checkpoint, never run.
bool
driver_scales_(const fs::path& dir, int n, std::vector<double>* out)
{
  std::ifstream f(dir / "RIFE_HDv3.py");
  if (!f) { return false; }
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string txt = ss.str();
  static const std::regex list_re(R"(scale_list\s*=\s*\[([^\]]*)\])");
  std::smatch m;
  if (!std::regex_search(txt, m, list_re)) { return false; }
  const std::string body = m[1].str();
  static const std::regex num_re(R"(([0-9]+(?:\.[0-9]*)?)\s*/\s*scale)");
  std::vector<double> v;
  for (auto it = std::sregex_iterator(body.begin(), body.end(), num_re);
       it != std::sregex_iterator(); ++it) {
    v.push_back(std::stod((*it)[1].str()));
  }
  if ((int)v.size() != n) { return false; }
  for (double x : v) {
    if (!(x > 0.0)) { return false; }
  }
  *out = std::move(v);
  return true;
}

// Each block's working resolution at a frame of `ph` x `pw`: the padded
// size over its effective scale. False when the network's strides do not
// land on whole pixels there.
bool
block_dims_(double s, int ph, int pw, int* hs, int* ws)
{
  const double h = (double)ph / s, w = (double)pw / s;
  const int ih = (int)std::lround(h), iw = (int)std::lround(w);
  if (std::fabs(h - ih) > 1e-9 || std::fabs(w - iw) > 1e-9) { return false; }
  if (ih % 4 != 0 || iw % 4 != 0 || ih < 4 || iw < 4) { return false; }
  *hs = ih;
  *ws = iw;
  return true;
}

}  // namespace

// ---- the checkpoint ------------------------------------------------------

std::string
MetalRife::resolve_weights(const std::string& path, std::string* err)
{
  std::error_code ec;
  const fs::path p(path);
  if (fs::is_regular_file(p, ec)) { return path; }
  if (fs::is_directory(p, ec)) {
    for (const char* sub : {"flownet.pkl", "train_log/flownet.pkl"}) {
      if (fs::is_regular_file(p / sub, ec)) { return (p / sub).string(); }
    }
  }
  if (err != nullptr) {
    *err = fmt("no RIFE checkpoint at '{}' (a directory holding "
               "flownet.pkl -- Practical-RIFE's train_log/ -- or the "
               "file itself)", path)();
  }
  return {};
}

bool
MetalRife::looks_like_rife(const std::string& path)
{
  std::string err;
  const std::string file = resolve_weights(path, &err);
  // A file that is not flownet.pkl is a guess too far for a probe that
  // runs over every directory a browser lists.
  if (file.empty() || fs::path(file).filename() != "flownet.pkl") {
    return false;
  }
  Info info;
  return read_info(file, &info, &err);
}

bool
MetalRife::read_info(const std::string& file, Info* out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  auto src = MetalLlamaWeights::open_model(file);
  if (!src) { return fail(fmt("'{}' is not a readable checkpoint", file)()); }
  Info info;
  if (src->has("module.encode.cnn0.weight")) {
    info.prefix = "module.";
  } else if (!src->has("encode.cnn0.weight")) {
    return fail(fmt(
        "'{}' has no encode.cnn0 -- not a RIFE 4.22-4.26 IFNet (the "
        "releases before 4.7 have no feature Head and are not read)",
        file)());
  }
  const std::string& p = info.prefix;
  auto shp = [&](const std::string& nm) { return shape_of_(*src, p + nm); };
  const auto h0 = shp("encode.cnn0.weight");
  if (h0.size() != 4 || h0[1] != 3 || h0[2] != 3 || h0[3] != 3) {
    return fail("encode.cnn0 is not a 3 -> c 3x3 conv");
  }
  info.head_width = (int)h0[0];
  const int hc = info.head_width;
  if (!is_shape_(shp("encode.cnn1.weight"), {hc, hc, 3, 3}) ||
      !is_shape_(shp("encode.cnn2.weight"), {hc, hc, 3, 3})) {
    return fail("encode.cnn1 / cnn2 are not c -> c 3x3 convs");
  }
  const auto h3 = shp("encode.cnn3.weight");
  if (h3.size() != 4 || h3[0] != hc || h3[2] != 4 || h3[3] != 4) {
    return fail("encode.cnn3 is not a 4x4 transposed conv from c");
  }
  info.features = (int)h3[1];
  const int F = info.features;
  if (F != 4 && F != 8) {
    return fail(fmt("{} feature channels; the kernels read 4 or 8", F)());
  }
  for (int i = 0;; ++i) {
    const std::string b = fmt("block{}.", i)();
    const auto w0 = shp(b + "conv0.0.0.weight");
    if (w0.empty()) { break; }
    const auto w1 = shp(b + "conv0.1.0.weight");
    if (w0.size() != 4 || w1.size() != 4 || w0[2] != 3 || w1[2] != 3 ||
        w1[1] != w0[0]) {
      return fail(fmt("block{}.conv0 is not two 3x3 convs", i)());
    }
    const int c = (int)w1[0];
    const int cin = (int)w0[1];
    const int want_in = i == 0 ? 7 + 2 * F : 20 + 2 * F;
    if (cin != want_in) {
      return fail(fmt(
          "block{} reads {} channels where this network's reads {} -- "
          "not a RIFE 4.22-4.26 IFNet", i, cin, want_in)());
    }
    for (int r = 0; r < 8; ++r) {
      const std::string rb = fmt("{}convblock.{}.", b, r)();
      if (!is_shape_(shp(rb + "conv.weight"), {c, c, 3, 3}) ||
          !is_shape_(shp(rb + "beta"), {1, c, 1, 1})) {
        return fail(fmt("{} is not a width-{} ResConv", rb, c)());
      }
    }
    if (!is_shape_(shp(b + "lastconv.0.weight"), {c, 52, 4, 4})) {
      return fail(fmt(
          "{}lastconv is not a transposed conv to 52 channels (flow, "
          "mask and 8 features, pixel-shuffled) -- not a RIFE 4.22-4.26 "
          "IFNet", b)());
    }
    info.widths.push_back(c);
    info.inputs.push_back(cin);
  }
  if (info.widths.empty()) { return fail("no IFBlocks (block0..)"); }
  const int n = info.blocks();
  info.scales_from_driver =
      driver_scales_(fs::path(file).parent_path(), n, &info.scales);
  if (!info.scales_from_driver) {
    info.scales.clear();
    for (int i = 0; i < n; ++i) {
      info.scales.push_back(std::ldexp(1.0, n - 1 - i));
    }
  }
  std::size_t params = 0;
  for (const std::string& nm : src->tensor_names()) {
    if (nm.rfind(p + "encode.", 0) != 0 && nm.rfind(p + "block", 0) != 0) {
      continue;
    }
    std::size_t k = 1;
    for (auto d : shape_of_(*src, nm)) { k *= (std::size_t)d; }
    params += k;
  }
  info.params = params;
  *out = std::move(info);
  return true;
}

bool
MetalRife::valid_scale(double scale) noexcept
{
  for (double s : {0.25, 0.5, 1.0, 2.0, 4.0}) {
    if (scale == s) { return true; }
  }
  return false;
}

int
MetalRife::pad_multiple(double scale) noexcept
{
  return std::max(128, (int)(128.0 / scale));
}

double
MetalRife::auto_scale(int h, int w) noexcept
{
  return std::max(h, w) > 2560 ? 0.5 : 1.0;
}

// ---- sizes ----------------------------------------------------------------

bool
MetalRife::hw_conv_wanted(const MetalCompute* mc)
{
  return mc != nullptr && mc->supports_matrix_cores() &&
         std::getenv("VPIPE_RIFE_NO_HWCONV") == nullptr;
}

std::size_t
MetalRife::weight_bytes(const Info& info, bool hw)
{
  // Each conv as conv_() runs it: [cout, 9 * cin] + a bias, f16, and its
  // HWIO twin where the hardware op runs.
  auto conv = [hw](std::size_t cin, std::size_t cout) {
    return (cout * 9 * cin * (hw ? 2 : 1) + cout) * 2;
  };
  const int hcp = round32_(info.head_width);
  std::size_t b = conv(4, hcp) + 2 * conv(hcp, hcp) +
                  conv(hcp, 4 * (std::size_t)info.features);
  for (int i = 0; i < info.blocks(); ++i) {
    const int c = info.widths[i];
    const int cp = round32_(c), c2p = round32_(c / 2);
    b += conv(round32_(info.inputs[i]), c2p) + conv(c2p, cp) +
         8 * conv(cp, cp) + conv(cp, 16 * last_ks_(hw));
  }
  return b;
}

std::size_t
MetalRife::Sizes::scratch() const
{
  return imgh + head_a + head_b + head_par + flow + aux + pack + c0a +
         2 * res + last + stage_in + stage_out;
}

MetalRife::Sizes
MetalRife::sizes_for_(const Info& info, int h, int w, double scale, bool f32,
                      bool hw)
{
  Sizes z;
  const int pad = pad_multiple(scale);
  const std::size_t ph = (std::size_t)((h + pad - 1) / pad * pad);
  const std::size_t pw = (std::size_t)((w + pad - 1) / pad * pad);
  const std::size_t P = ph * pw, Q = P / 4;
  const std::size_t F = (std::size_t)info.features;
  const std::size_t hcp = (std::size_t)round32_(info.head_width);
  z.frame_img  = P * 16;
  z.frame_feat = P * F * 2;
  z.imgh       = P * 8;
  z.head_a     = Q * hcp * 2;
  z.head_b     = z.head_a;
  z.head_par   = Q * 4 * F * 2;
  z.flow       = P * 16;
  z.aux        = P * 12 * 2;
  for (int i = 0; i < info.blocks(); ++i) {
    int hs = 0, ws = 0;
    if (!block_dims_(info.scales[i] / scale, (int)ph, (int)pw, &hs, &ws)) {
      continue;
    }
    const std::size_t L = (std::size_t)hs * ws;
    const int c = info.widths[i];
    z.pack = std::max(z.pack, L * round32_(info.inputs[i]) * 2);
    z.c0a  = std::max(z.c0a, L / 4 * round32_(c / 2) * 2);
    z.res  = std::max(z.res, L / 16 * round32_(c) * 2);
    z.last = std::max(z.last, L / 16 * 16 * last_ks_(hw) * 2);
  }
  const std::size_t esz = f32 ? 4 : 1;
  z.stage_in  = (std::size_t)3 * h * w * esz;
  z.stage_out = z.stage_in;
  return z;
}

std::size_t
MetalRife::frame_bytes(const Info& info, int h, int w, double scale)
{
  const Sizes z = sizes_for_(info, h, w, scale, false, false);
  return z.frame_img + z.frame_feat;
}

std::size_t
MetalRife::scratch_bytes(const Info& info, int h, int w, double scale,
                         bool f32, bool hw)
{
  return sizes_for_(info, h, w, scale, f32, hw).scratch();
}

std::size_t
MetalRife::resident_bytes() const noexcept
{
  return _weight_bytes + _scratch_bytes;
}

// ---- load -------------------------------------------------------------------

MetalRife::~MetalRife() = default;

std::unique_ptr<MetalRife>
MetalRife::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                const std::string& file, std::string* err)
{
  auto fail = [&](std::string m) -> std::unique_ptr<MetalRife> {
    if (err != nullptr) { *err = std::move(m); }
    return nullptr;
  };
  if (!ws || mc == nullptr) { return fail("no weight set or GPU"); }
  std::unique_ptr<MetalRife> r(new MetalRife());
  r->_ws = std::move(ws);
  r->_mc = mc;
  r->_use_hw = hw_conv_wanted(mc);
  std::string e;
  if (!read_info(file, &r->_info, &e)) { return fail(e); }
  const Info& info = r->_info;
  const std::string& p = info.prefix;
  WeightSet& W = *r->_ws;

  // A conv's weight [cout, cin, 3, 3] into [cout_p, 9 * cin_p] (ky, kx,
  // ci), zero-padded; `gain` per output channel (a ResConv's beta, which
  // multiplies the conv's output, bias included).
  auto conv3 = [&](const std::string& nm, int cin_p, int cout_p,
                   const std::vector<float>* gain, Conv* out) {
    std::vector<float> w, b;
    if (!read_host_(W, mc, p + nm + ".weight", &w) ||
        !read_host_(W, mc, p + nm + ".bias", &b)) {
      return false;
    }
    const auto s = shape_of_(W.src(), p + nm + ".weight");
    const int co = (int)s[0], ci = (int)s[1];
    std::vector<float> g((std::size_t)cout_p * 9 * cin_p, 0.0f);
    std::vector<float> bb((std::size_t)cout_p, 0.0f);
    for (int o = 0; o < co; ++o) {
      const float k = gain != nullptr ? (*gain)[o] : 1.0f;
      bb[o] = b[o] * k;
      for (int c = 0; c < ci; ++c) {
        for (int t = 0; t < 9; ++t) {
          g[(std::size_t)o * 9 * cin_p + (std::size_t)t * cin_p + c] =
              w[((std::size_t)o * ci + c) * 9 + t] * k;
        }
      }
    }
    out->w = f16_buf_(mc, g);
    out->b = f16_buf_(mc, bb);
    out->cin = cin_p;
    out->cout = cout_p;
    if (r->_use_hw) { out->whwio = f16_buf_(mc, hwio_(g, cout_p, cin_p)); }
    return !out->w.empty() && !out->b.empty() &&
           (!r->_use_hw || !out->whwio.empty());
  };
  // A ConvTranspose2d(k4, s2, p1), weight [cin, cout, 4, 4], as a 3x3
  // conv with 4 * cout outputs: output channel p * cout + o is output
  // parity p = py * 2 + px, and window tap (ky, kx) of it reads kernel
  // tap (3 + py - 2 ky, 3 + px - 2 kx) when that lies in [0, 3] -- out
  // row 2m + py gathers input rows m - 1 + ky. With `ks` (an IFBlock's
  // flow head, PixelShuffle(2) after it: cout 52 = 13 x 2 x 2) the rows
  // are SUB-POSITION major instead -- channel c of shuffle position
  // (sy, sx) at parity (py, px) is row sub * ks + c, sub = (2 py + sy) * 4
  // + 2 px + sx -- 16 * ks rows, the padding zero (see rife_unpack).
  auto deconv = [&](const std::string& nm, int cin_p, int ks, Conv* out) {
    std::vector<float> w, b;
    if (!read_host_(W, mc, p + nm + ".weight", &w) ||
        !read_host_(W, mc, p + nm + ".bias", &b)) {
      return false;
    }
    const auto s = shape_of_(W.src(), p + nm + ".weight");
    const int ci = (int)s[0], co = (int)s[1];
    if (ks > 0 && co != 52) { return false; }
    const int cout_p = ks > 0 ? 16 * ks : 4 * co;
    std::vector<float> g((std::size_t)cout_p * 9 * cin_p, 0.0f);
    std::vector<float> bb((std::size_t)cout_p, 0.0f);
    for (int py = 0; py < 2; ++py) {
      for (int px = 0; px < 2; ++px) {
        const int par = py * 2 + px;
        for (int o = 0; o < co; ++o) {
          // o = 4c + 2 sy + sx under the pixel shuffle.
          const int sub = (2 * py + o % 4 / 2) * 4 + 2 * px + o % 2;
          const std::size_t row =
              ks > 0 ? (std::size_t)sub * ks + o / 4
                     : (std::size_t)par * co + o;
          bb[row] = b[o];
          for (int ky = 0; ky < 3; ++ky) {
            const int ty = 3 + py - 2 * ky;
            if (ty < 0 || ty > 3) { continue; }
            for (int kx = 0; kx < 3; ++kx) {
              const int tx = 3 + px - 2 * kx;
              if (tx < 0 || tx > 3) { continue; }
              for (int c = 0; c < ci; ++c) {
                g[row * 9 * cin_p + (std::size_t)(ky * 3 + kx) * cin_p + c] =
                    w[(((std::size_t)c * co + o) * 4 + ty) * 4 + tx];
              }
            }
          }
        }
      }
    }
    out->w = f16_buf_(mc, g);
    out->b = f16_buf_(mc, bb);
    out->cin = cin_p;
    out->cout = cout_p;
    if (r->_use_hw) { out->whwio = f16_buf_(mc, hwio_(g, cout_p, cin_p)); }
    return !out->w.empty() && !out->b.empty() &&
           (!r->_use_hw || !out->whwio.empty());
  };

  const int hcp = round32_(info.head_width);
  // The Head's first conv reads the frame as half4 (rgb + a zero), so
  // its input is 4 wide, not 32: it takes the gather conv's general path.
  if (!conv3("encode.cnn0", 4, hcp, nullptr, &r->_head[0]) ||
      !conv3("encode.cnn1", hcp, hcp, nullptr, &r->_head[1]) ||
      !conv3("encode.cnn2", hcp, hcp, nullptr, &r->_head[2]) ||
      !deconv("encode.cnn3", hcp, 0, &r->_head[3])) {
    return fail("the feature Head did not read");
  }
  for (int i = 0; i < info.blocks(); ++i) {
    Block blk;
    blk.width = info.widths[i];
    blk.in = info.inputs[i];
    blk.in_pad = round32_(blk.in);
    blk.scale = info.scales[i];
    blk.ks = last_ks_(r->_use_hw);
    const int cp = round32_(blk.width), c2p = round32_(blk.width / 2);
    const std::string b = fmt("block{}.", i)();
    bool ok = conv3(b + "conv0.0.0", blk.in_pad, c2p, nullptr, &blk.c0a) &&
              conv3(b + "conv0.1.0", c2p, cp, nullptr, &blk.c0b);
    for (int k = 0; k < 8 && ok; ++k) {
      std::vector<float> beta;
      ok = read_host_(W, mc, fmt("{}{}convblock.{}.beta", p, b, k)(),
                      &beta) &&
           conv3(fmt("{}convblock.{}.conv", b, k)(), cp, cp, &beta,
                 &blk.res[k]);
    }
    ok = ok && deconv(b + "lastconv.0", cp, blk.ks, &blk.last);
    if (!ok) { return fail(fmt("block{} did not read", i)()); }
    r->_blocks.push_back(std::move(blk));
  }
  if (!r->build_(&e)) { return fail(e); }
  if (!r->_use_hw) {
    // Twins built for an op whose kernels did not load: give them back.
    for (Conv& c : r->_head) { c.whwio = SharedBuffer{}; }
    for (Block& b : r->_blocks) {
      for (Conv* c : {&b.c0a, &b.c0b, &b.last}) { c->whwio = SharedBuffer{}; }
      for (Conv& c : b.res) { c.whwio = SharedBuffer{}; }
    }
  }
  // What the buffers hold, counted -- which the static figure the plan
  // books must equal (rife.allocates_what_it_books).
  std::size_t held = 0;
  auto count = [&held](const Conv& c) {
    held += c.w.byte_size() + c.b.byte_size() + c.whwio.byte_size();
  };
  for (const Conv& c : r->_head) { count(c); }
  for (const Block& b : r->_blocks) {
    for (const Conv* c : {&b.c0a, &b.c0b, &b.last}) { count(*c); }
    for (const Conv& c : b.res) { count(c); }
  }
  r->_weight_bytes = held;
  return r;
}

bool
MetalRife::build_(std::string* err)
{
  _lib_gemm = _mc->load_library("dense_gemm");
  _lib_rife = _mc->load_library("rife");
  static const char* const kConv[4][2] = {
      {"conv3x3_gemm_s1_bn32_f16", "conv3x3_gemm_s1_bn64_f16"},
      {"conv3x3_gemm_s1_bn32_lrelu_f16", "conv3x3_gemm_s1_bn64_lrelu_f16"},
      {"conv3x3_gemm_s1_bn32_addlrelu_f16",
       "conv3x3_gemm_s1_bn64_addlrelu_f16"},
      {"conv3x3_gemm_s2p1_bn32_lrelu_f16",
       "conv3x3_gemm_s2p1_bn64_lrelu_f16"},
  };
  for (int f = 0; f < 4; ++f) {
    for (int b = 0; b < 2; ++b) {
      _fn_conv[f][b] = _lib_gemm.function(kConv[f][b]);
      if (!_fn_conv[f][b].valid()) {
        if (err != nullptr) {
          *err = fmt("kernel {} did not load", kConv[f][b])();
        }
        return false;
      }
    }
  }
  // An unresolved name binds nothing and the op would run whatever kernel
  // was bound before it; refuse at load instead.
  const std::pair<const char*, metal_compute::ComputeFunction*> need[] = {
      {"rife_frame_in_u8", &_fn_in_u8},   {"rife_frame_in_f32", &_fn_in_f32},
      {"rife_head_d2s", &_fn_d2s},        {"rife_pack", &_fn_pack},
      {"rife_unpack", &_fn_unpack},       {"rife_merge_u8", &_fn_merge_u8},
      {"rife_merge_f32", &_fn_merge_f32},
  };
  for (const auto& [name, fn] : need) {
    *fn = _lib_rife.function(name);
    if (!fn->valid()) {
      if (err != nullptr) { *err = fmt("kernel {} did not load", name)(); }
      return false;
    }
  }
  if (_use_hw) {
    // Runtime-compiled: a source error is an INVALID HANDLE, not a build
    // failure. The forms RIFE needs on every network the shape reader
    // accepts must all load, or none of the route is taken.
    _lib_hw = _mc->load_library("conv2d_mma");
    static const char* const kS1[3][3][2] = {
        {{"conv2d_hw_3x3_s1_epi_f16", "conv2d_hw_3x3_s1_epi_tail_f16"},
         {"conv2d_hw_3x3_s1_c32_epi_f16", "conv2d_hw_3x3_s1_c32_epi_tail_f16"},
         {"conv2d_hw_3x3_s1_c16_epi_f16",
          "conv2d_hw_3x3_s1_c16_epi_tail_f16"}},
        {{"conv2d_hw_3x3_s1_epi_lrelu_f16",
          "conv2d_hw_3x3_s1_epi_lrelu_tail_f16"},
         {"conv2d_hw_3x3_s1_c32_epi_lrelu_f16",
          "conv2d_hw_3x3_s1_c32_epi_lrelu_tail_f16"},
         {nullptr, nullptr}},
        {{"conv2d_hw_3x3_s1_epi_addlrelu_f16",
          "conv2d_hw_3x3_s1_epi_addlrelu_tail_f16"},
         {"conv2d_hw_3x3_s1_c32_epi_addlrelu_f16",
          "conv2d_hw_3x3_s1_c32_epi_addlrelu_tail_f16"},
         {nullptr, nullptr}},
    };
    static const char* const kS2[2][2] = {
        {"conv2d_hw_3x3_s2p1_epi_lrelu_f16",
         "conv2d_hw_3x3_s2p1_epi_lrelu_tail_f16"},
        {"conv2d_hw_3x3_s2p1_c32_epi_lrelu_f16",
         "conv2d_hw_3x3_s2p1_c32_epi_lrelu_tail_f16"},
    };
    for (int e = 0; e < 3; ++e) {
      for (int t = 0; t < 3; ++t) {
        for (int k = 0; k < 2; ++k) {
          if (kS1[e][t][k] == nullptr) { continue; }
          _fn_hw_s1[e][t][k] = _lib_hw.function(kS1[e][t][k]);
          if (!_fn_hw_s1[e][t][k].valid()) { _use_hw = false; }
        }
      }
    }
    for (int t = 0; t < 2; ++t) {
      for (int k = 0; k < 2; ++k) {
        _fn_hw_s2[t][k] = _lib_hw.function(kS2[t][k]);
        if (!_fn_hw_s2[t][k].valid()) { _use_hw = false; }
      }
    }
  }
  _cs = _mc->make_command_stream();
  if (!_cs.valid()) {
    if (err != nullptr) { *err = "no command stream"; }
    return false;
  }
  return true;
}

bool
MetalRife::ensure_scratch_(int h, int w, double scale, bool f32)
{
  if (h == _sh && w == _sw && scale == _sscale && f32 == _sf32 &&
      !_flow.empty()) {
    return true;
  }
  const Sizes z = sizes_for_(_info, h, w, scale, f32,
                             !_blocks.empty() && _blocks[0].ks == 16);
  _scratch_bytes = 0;
  auto take = [&](std::size_t bytes, SharedBuffer* b) {
    *b = _mc->make_shared_buffer(std::max<std::size_t>(bytes, 16));
    _scratch_bytes += b->byte_size();
    return !b->empty();
  };
  const bool ok = take(z.imgh, &_imgh) && take(z.head_a, &_head_a) &&
                  take(z.head_b, &_head_b) && take(z.head_par, &_head_par) &&
                  take(z.flow, &_flow) && take(z.aux, &_aux) &&
                  take(z.pack, &_pack) && take(z.c0a, &_c0a) &&
                  take(z.res, &_res_a) && take(z.res, &_res_b) &&
                  take(z.last, &_last) && take(z.stage_in, &_stage_in) &&
                  take(z.stage_out, &_stage_out);
  if (!ok) {
    _sh = _sw = 0;
    return false;
  }
  _sh = h;
  _sw = w;
  _sscale = scale;
  _sf32 = f32;
  return true;
}

// ---- the forward --------------------------------------------------------

void
MetalRife::dispatch_(ComputeEncoder& enc, const char* cat,
                     metal_compute::LaunchDims grid,
                     metal_compute::LaunchDims tg, bool hw)
{
  enc.dispatch(grid, tg);
  if (_prof_cs == nullptr) { return; }
  enc.end();
  _flops.flush(*_prof_cs);
  metal_compute::CommandStream::Fence f = _prof_cs->commit();
  f.wait_ok();
  ProfileRow& r = _profile[hw ? std::string(cat) + "-hw" : std::string(cat)];
  r.ms += f.gpu_times().gpu_s * 1e3;
  ++r.dispatches;
  enc = _prof_cs->begin_compute();
}

void
MetalRife::conv_(ComputeEncoder& enc, const Conv& c, const SharedBuffer& in,
                 const SharedBuffer& out, int H, int W, int stride, int epi,
                 const char* cat, const SharedBuffer* res)
{
  // epi: 0 none, 1 LeakyReLU, 2 + residual then LeakyReLU. A stride-2
  // conv is always followed by LeakyReLU here.
  const int OH = stride == 2 ? H / 2 : H, OW = stride == 2 ? W / 2 : W;
  if (_use_hw && !c.whwio.empty()) {
    // The widest destination tile the width divides by; a grid that is
    // not whole 8 x 8 tiles takes the guarded entry.
    const int tile = c.cout % 64 == 0 ? 0 : c.cout % 32 == 0 ? 1
                   : c.cout % 16 == 0 ? 2 : -1;
    const int tail = OH % 8 != 0 || OW % 8 != 0 ? 1 : 0;
    const metal_compute::ComputeFunction* fn = nullptr;
    if (stride == 2 && (tile == 0 || tile == 1)) {
      fn = &_fn_hw_s2[tile][tail];
    } else if (stride == 1 && tile >= 0) {
      fn = &_fn_hw_s1[epi][tile][tail];
    }
    if (fn != nullptr && fn->valid()) {
      const int tc = tile == 0 ? 64 : tile == 1 ? 32 : 16;
      enc.set_function(*fn);
      enc.set_buffer(0, in);
      enc.set_buffer(1, c.whwio);
      enc.set_buffer(2, out);
      enc.set_constant(3, W);
      enc.set_constant(4, H);
      enc.set_constant(5, c.cin);
      enc.set_constant(6, c.cout);
      enc.set_buffer(7, c.b);
      // Bound whether or not the form reads it.
      if (stride == 1) { enc.set_buffer(8, res != nullptr ? *res : out); }
      dispatch_(enc, cat, {(unsigned)(((OW + 7) / 8) * 128),
                           (unsigned)((OH + 7) / 8),
                           (unsigned)(c.cout / tc)},
                {128, 1, 1}, /*hw=*/true);
      ++_hw_dispatches;
      return;
    }
  }
  const int form = stride == 2 ? 3 : epi;
  const int bn = c.cout >= 64 ? 64 : 32;
  enc.set_function(_fn_conv[form][bn == 64 ? 1 : 0]);
  enc.set_buffer(0, in);
  enc.set_buffer(1, c.w);
  enc.set_buffer(2, c.b);
  enc.set_buffer(3, out);
  enc.set_constant(4, H);       enc.set_constant(5, W);
  enc.set_constant(6, c.cin);   enc.set_constant(7, c.cout);
  enc.set_constant(8, OH);      enc.set_constant(9, OW);
  enc.set_constant(10, 1);
  if (form != 0) { enc.set_buffer(11, res != nullptr ? *res : in); }
  const int M = OH * OW;
  dispatch_(enc, cat, {(unsigned)(((c.cout + bn - 1) / bn) * 32),
                       (unsigned)(((M + 63) / 64) * 2), 2},
            {32, 2, 2});
  ++_gather_dispatches;
}

bool
MetalRife::prepare(const void* planar, bool f32, int h, int w, double scale,
                   Frame* out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (planar == nullptr || out == nullptr || h <= 0 || w <= 0) {
    return fail("empty frame");
  }
  if (!valid_scale(scale)) {
    return fail(fmt("scale {} is not one of 0.25, 0.5, 1, 2, 4", scale)());
  }
  const int pad = pad_multiple(scale);
  const int ph = (h + pad - 1) / pad * pad, pw = (w + pad - 1) / pad * pad;
  for (const Block& b : _blocks) {
    int hs = 0, ws = 0;
    if (!block_dims_(b.scale / scale, ph, pw, &hs, &ws)) {
      return fail(fmt("a {}x{} frame at scale {} puts a block off the "
                      "pixel grid", w, h, scale)());
    }
  }
  if (!ensure_scratch_(h, w, scale, f32)) {
    return fail(fmt("scratch for a {}x{} frame did not allocate", w, h)());
  }
  const std::size_t P = (std::size_t)ph * pw;
  const int F = _info.features;
  if (out->img.byte_size() != P * 16 ||
      out->feat.byte_size() != P * F * 2) {
    out->img = _mc->make_shared_buffer(P * 16);
    out->feat = _mc->make_shared_buffer(P * F * 2);
    if (out->img.empty() || out->feat.empty()) {
      *out = Frame{};
      return fail("frame buffers did not allocate");
    }
  }
  out->h = h; out->w = w; out->ph = ph; out->pw = pw; out->scale = scale;
  std::memcpy(_stage_in.contents(), planar,
              (std::size_t)3 * h * w * (f32 ? 4 : 1));

  _flops.begin();
  _prof_cs = std::getenv("VPIPE_RIFE_PROFILE") != nullptr ? &_cs : nullptr;
  struct ProfGuard {
    metal_compute::CommandStream** cs;
    ~ProfGuard() { *cs = nullptr; }
  } prof_guard{&_prof_cs};
  const long long q = (long long)(ph / 2) * (pw / 2);
  const int hc = _info.head_width;
  {
    ComputeEncoder enc = _cs.begin_compute();
    enc.set_function(f32 ? _fn_in_f32 : _fn_in_u8);
    enc.set_buffer(0, _stage_in);
    enc.set_buffer(1, out->img);
    enc.set_buffer(2, _imgh);
    enc.set_constant(3, h);   enc.set_constant(4, w);
    enc.set_constant(5, ph);  enc.set_constant(6, pw);
    dispatch_(enc, "frame-in", {(unsigned)pw, (unsigned)ph, 1}, {16, 16, 1});
    // The Head: conv s2, conv, conv (LeakyReLU), then its transposed conv
    // as a parity conv and a scatter to full size.
    conv_(enc, _head[0], _imgh, _head_a, ph, pw, 2, 1, "head-conv");
    conv_(enc, _head[1], _head_a, _head_b, ph / 2, pw / 2, 1, 1,
          "head-conv");
    conv_(enc, _head[2], _head_b, _head_a, ph / 2, pw / 2, 1, 1,
          "head-conv");
    conv_(enc, _head[3], _head_a, _head_par, ph / 2, pw / 2, 1, 0,
          "head-deconv");
    _flops.gemm(q, hc, 9LL * 3);
    _flops.gemm(q, hc, 9LL * hc);
    _flops.gemm(q, hc, 9LL * hc);
    _flops.gemm((long long)P, F, 4LL * hc);
    enc.set_function(_fn_d2s);
    enc.set_buffer(0, _head_par);
    enc.set_buffer(1, out->feat);
    enc.set_constant(2, ph);
    enc.set_constant(3, pw);
    enc.set_constant(4, F);
    dispatch_(enc, "head-d2s", {(unsigned)pw, (unsigned)ph, 1}, {16, 16, 1});
    enc.end();
  }
  _flops.flush(_cs);
  std::string ferr;
  if (!_cs.commit().wait_ok(&ferr)) {
    return fail(ferr.empty() ? "GPU failure" : ferr);
  }
  return true;
}

bool
MetalRife::interpolate(const Frame& a, const Frame& b, float t, bool f32,
                       void* out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (!a.ready() || !b.ready() || out == nullptr) {
    return fail("frames not prepared");
  }
  if (a.h != b.h || a.w != b.w || a.scale != b.scale) {
    return fail("the two frames differ in size or scale");
  }
  const int h = a.h, w = a.w, ph = a.ph, pw = a.pw;
  if (!ensure_scratch_(h, w, a.scale, f32)) {
    return fail("scratch did not allocate");
  }
  const int F = _info.features;
  _flops.begin();
  _prof_cs = std::getenv("VPIPE_RIFE_PROFILE") != nullptr ? &_cs : nullptr;
  struct ProfGuard {
    metal_compute::CommandStream** cs;
    ~ProfGuard() { *cs = nullptr; }
  } prof_guard{&_prof_cs};
  {
    ComputeEncoder enc = _cs.begin_compute();
    for (std::size_t i = 0; i < _blocks.size(); ++i) {
      const Block& blk = _blocks[i];
      const float s = (float)(blk.scale / a.scale);
      int hs = 0, ws = 0;
      if (!block_dims_(blk.scale / a.scale, ph, pw, &hs, &ws)) {
        enc.end();
        return fail("a block is off the pixel grid");
      }
      const int first = i == 0 ? 1 : 0;
      enc.set_function(_fn_pack);
      enc.set_buffer(0, a.img);   enc.set_buffer(1, b.img);
      enc.set_buffer(2, a.feat);  enc.set_buffer(3, b.feat);
      enc.set_buffer(4, _flow);   enc.set_buffer(5, _aux);
      enc.set_buffer(6, _pack);
      enc.set_constant(7, ph);    enc.set_constant(8, pw);
      enc.set_constant(9, hs);    enc.set_constant(10, ws);
      enc.set_constant(11, F);    enc.set_constant(12, blk.in_pad);
      enc.set_constant(13, s);    enc.set_constant(14, t);
      enc.set_constant(15, first);
      dispatch_(enc, "pack", {(unsigned)ws, (unsigned)hs, 1}, {16, 16, 1});
      conv_(enc, blk.c0a, _pack, _c0a, hs, ws, 2, 1, "conv-s2");
      conv_(enc, blk.c0b, _c0a, _res_a, hs / 2, ws / 2, 2, 1, "conv-s2");
      const int hq = hs / 4, wq = ws / 4;
      for (int k = 0; k < 8; ++k) {
        const SharedBuffer& x = k % 2 == 0 ? _res_a : _res_b;
        const SharedBuffer& y = k % 2 == 0 ? _res_b : _res_a;
        conv_(enc, blk.res[k], x, y, hq, wq, 1, 2, "resconv", &x);
      }
      conv_(enc, blk.last, _res_a, _last, hq, wq, 1, 0, "lastconv");
      const long long c = blk.width;
      _flops.gemm((long long)(hs / 2) * (ws / 2), c / 2, 9LL * blk.in);
      _flops.gemm((long long)hq * wq, c, 9LL * (c / 2));
      _flops.gemm((long long)hq * wq * 8, c, 9LL * c);
      _flops.gemm((long long)(hs / 2) * (ws / 2), 52, 4LL * c);
      enc.set_function(_fn_unpack);
      enc.set_buffer(0, _last);
      enc.set_buffer(1, _flow);
      enc.set_buffer(2, _aux);
      enc.set_constant(3, ph);   enc.set_constant(4, pw);
      enc.set_constant(5, hs);   enc.set_constant(6, ws);
      enc.set_constant(7, s);    enc.set_constant(8, first);
      enc.set_constant(9, blk.ks);
      dispatch_(enc, "unpack", {(unsigned)pw, (unsigned)ph, 1},
                {16, 16, 1});
    }
    enc.set_function(f32 ? _fn_merge_f32 : _fn_merge_u8);
    enc.set_buffer(0, a.img);
    enc.set_buffer(1, b.img);
    enc.set_buffer(2, _flow);
    enc.set_buffer(3, _aux);
    enc.set_buffer(4, _stage_out);
    enc.set_constant(5, h);   enc.set_constant(6, w);
    enc.set_constant(7, ph);  enc.set_constant(8, pw);
    dispatch_(enc, "merge", {(unsigned)w, (unsigned)h, 1}, {16, 16, 1});
    enc.end();
  }
  _flops.flush(_cs);
  std::string ferr;
  if (!_cs.commit().wait_ok(&ferr)) {
    return fail(ferr.empty() ? "GPU failure" : ferr);
  }
  std::memcpy(out, _stage_out.contents(),
              (std::size_t)3 * h * w * (f32 ? 4 : 1));
  return true;
}

// ---- the scene test ---------------------------------------------------------

double
MetalRife::scene_ssim(const void* pa, const void* pb, bool f32, int h, int w,
                      double scale)
{
  constexpr int kS = 32, kWin = 11, kHalf = 5;
  const int pad = pad_multiple(valid_scale(scale) ? scale : 1.0);
  const int ph = (h + pad - 1) / pad * pad, pw = (w + pad - 1) / pad * pad;
  // F.interpolate(size=(32, 32), bilinear, align_corners=False) of the
  // PADDED frame: zeros past (h, w).
  auto px = [&](const void* p, int c, int y, int x) -> float {
    if (y >= h || x >= w) { return 0.0f; }
    const std::size_t o = ((std::size_t)c * h + y) * w + x;
    return f32 ? static_cast<const float*>(p)[o]
               : static_cast<const std::uint8_t*>(p)[o] / 255.0f;
  };
  auto small = [&](const void* p, std::vector<float>* v) {
    v->assign((std::size_t)3 * kS * kS, 0.0f);
    const float ry = (float)ph / kS, rx = (float)pw / kS;
    for (int oy = 0; oy < kS; ++oy) {
      const float sy = std::max(((float)oy + 0.5f) * ry - 0.5f, 0.0f);
      const int y0 = (int)sy, y1 = std::min(y0 + 1, ph - 1);
      const float ly1 = sy - (float)y0, ly0 = 1.0f - ly1;
      for (int ox = 0; ox < kS; ++ox) {
        const float sx = std::max(((float)ox + 0.5f) * rx - 0.5f, 0.0f);
        const int x0 = (int)sx, x1 = std::min(x0 + 1, pw - 1);
        const float lx1 = sx - (float)x0, lx0 = 1.0f - lx1;
        for (int c = 0; c < 3; ++c) {
          (*v)[((std::size_t)c * kS + oy) * kS + ox] =
              ly0 * (lx0 * px(p, c, y0, x0) + lx1 * px(p, c, y0, x1)) +
              ly1 * (lx0 * px(p, c, y1, x0) + lx1 * px(p, c, y1, x1));
        }
      }
    }
  };
  std::vector<float> a, b;
  small(pa, &a);
  small(pb, &b);
  // ssim_matlab: the frame as a VOLUME [3, 32, 32], an 11-tap Gaussian
  // (sigma 1.5) along all three axes, replicate padding, L = 1.
  double g[kWin], gs = 0.0;
  for (int i = 0; i < kWin; ++i) {
    g[i] = std::exp(-(double)((i - kHalf) * (i - kHalf)) / (2.0 * 1.5 * 1.5));
    gs += g[i];
  }
  for (double& v : g) { v /= gs; }
  const int D = 3;
  auto blur = [&](const std::vector<double>& in) {
    // Separable, axis by axis, each tap's index clamped (replicate).
    std::vector<double> t1(in.size()), t2(in.size()), t3(in.size());
    auto at = [&](int d, int y, int x) {
      return ((std::size_t)d * kS + y) * kS + x;
    };
    for (int d = 0; d < D; ++d) {
      for (int y = 0; y < kS; ++y) {
        for (int x = 0; x < kS; ++x) {
          double s = 0.0;
          for (int i = 0; i < kWin; ++i) {
            s += g[i] * in[at(d, y, std::clamp(x + i - kHalf, 0, kS - 1))];
          }
          t1[at(d, y, x)] = s;
        }
      }
    }
    for (int d = 0; d < D; ++d) {
      for (int y = 0; y < kS; ++y) {
        for (int x = 0; x < kS; ++x) {
          double s = 0.0;
          for (int i = 0; i < kWin; ++i) {
            s += g[i] * t1[at(d, std::clamp(y + i - kHalf, 0, kS - 1), x)];
          }
          t2[at(d, y, x)] = s;
        }
      }
    }
    for (int d = 0; d < D; ++d) {
      for (int y = 0; y < kS; ++y) {
        for (int x = 0; x < kS; ++x) {
          double s = 0.0;
          for (int i = 0; i < kWin; ++i) {
            s += g[i] * t2[at(std::clamp(d + i - kHalf, 0, D - 1), y, x)];
          }
          t3[at(d, y, x)] = s;
        }
      }
    }
    return t3;
  };
  const std::size_t n = a.size();
  std::vector<double> x(n), y(n), xx(n), yy(n), xy(n);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = a[i];  y[i] = b[i];
    xx[i] = (double)a[i] * a[i];
    yy[i] = (double)b[i] * b[i];
    xy[i] = (double)a[i] * b[i];
  }
  const auto mu1 = blur(x), mu2 = blur(y);
  const auto e11 = blur(xx), e22 = blur(yy), e12 = blur(xy);
  constexpr double C1 = 0.01 * 0.01, C2 = 0.03 * 0.03;
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double s1 = e11[i] - mu1[i] * mu1[i];
    const double s2 = e22[i] - mu2[i] * mu2[i];
    const double s12 = e12[i] - mu1[i] * mu2[i];
    sum += ((2.0 * mu1[i] * mu2[i] + C1) * (2.0 * s12 + C2)) /
           ((mu1[i] * mu1[i] + mu2[i] * mu2[i] + C1) * (s1 + s2 + C2));
  }
  return sum / (double)n;
}

}  // namespace genai
}  // namespace vpipe
