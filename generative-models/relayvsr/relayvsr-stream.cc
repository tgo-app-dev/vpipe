#include "generative-models/relayvsr/relayvsr-stream.h"

#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/relayvsr/relayvsr-ops.h"
#include "generative-models/shared/torch-randn.h"

#include <algorithm>
#include <cstring>
#include <filesystem>

namespace vpipe {
namespace genai {

using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace fs = std::filesystem;

namespace {

std::string
format_of(const std::string& file)
{
  std::error_code ec;
  if (!fs::is_regular_file(file, ec) || ec) { return {}; }
  auto w = MetalLlamaWeights::open(file);
  if (!w) { return {}; }
  auto it = w->metadata().find("format");
  return it == w->metadata().end() ? std::string() : it->second;
}

}  // namespace

namespace {

// A checkpoint's role ('w' the keyframe model, 'f' the decoder) and native
// scale, off its format; role 0 for anything else.
struct Kind {
  char role = 0;
  int scale = 0;
};

Kind
kind_of(const std::string& f)
{
  if (f == "refvsr-wan-v1") { return {'w', 4}; }
  if (f == "relayvsr-wan-x2-v1") { return {'w', 2}; }
  if (f == "refvsr-flash-v1") { return {'f', 4}; }
  if (f == "relayvsr-flash-x2-v1") { return {'f', 2}; }
  return {};
}

}  // namespace

bool
resolve_relayvsr_files(const std::string& path, RelayVsrFiles* out,
                       std::string* err, int scale)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (scale != 0 && scale != 2 && scale != 4) {
    return fail(fmt("RelayVSR runs at 2x or 4x, not {}x", scale)());
  }
  std::error_code ec;
  const fs::path p(path);
  // The pairs found, by scale (index 2 and 4).
  RelayVsrFiles by[5];
  auto take = [&](const fs::path& f, const Kind& k) {
    RelayVsrFiles& slot = by[k.scale];
    std::string& dst = k.role == 'w' ? slot.wan : slot.flash;
    if (dst.empty()) { dst = f.string(); }
  };
  std::vector<fs::path> dirs;
  int want = scale;
  if (fs::is_regular_file(p, ec)) {
    // A FILE names its own pair: its partner beside it, or in the sibling
    // folder ComfyUI keeps it in.
    const Kind k = kind_of(format_of(p.string()));
    if (k.role == 0) {
      return fail(fmt("'{}' is not a RelayVSR checkpoint (no refvsr / "
                      "relayvsr format in its metadata)", path)());
    }
    if (want != 0 && k.scale != want) {
      return fail(fmt("'{}' is the {}x checkpoint; {}x was asked", path,
                      k.scale, want)());
    }
    want = k.scale;
    take(p, k);
    const fs::path d = p.parent_path();
    dirs = {d, d.parent_path() / "vae", d.parent_path() / "diffusion_models"};
  } else if (fs::is_directory(p, ec)) {
    dirs = {p, p / "diffusion_models", p / "vae"};
  } else {
    return fail(fmt("'{}' is not a RelayVSR model directory", path)());
  }
  for (const fs::path& d : dirs) {
    if (!fs::is_directory(d, ec)) { continue; }
    // The release's and lightx2v's names first, then the rest by name.
    std::vector<fs::path> files;
    for (const char* n : {"wan.safetensors", "flash.safetensors",
                          "wan_x2.safetensors", "flash_x2.safetensors"}) {
      if (fs::is_regular_file(d / n, ec)) { files.push_back(d / n); }
    }
    std::vector<fs::path> rest;
    for (const auto& e : fs::directory_iterator(d, ec)) {
      if (e.path().extension() == ".safetensors" &&
          std::find(files.begin(), files.end(), e.path()) == files.end()) {
        rest.push_back(e.path());
      }
    }
    std::sort(rest.begin(), rest.end());
    files.insert(files.end(), rest.begin(), rest.end());
    for (const fs::path& f : files) {
      const Kind k = kind_of(format_of(f.string()));
      if (k.role != 0) { take(f, k); }
    }
  }
  auto whole = [&](int s) {
    return !by[s].wan.empty() && !by[s].flash.empty();
  };
  if (want == 0) {
    if (whole(2) && whole(4)) {
      return fail(fmt("'{}' holds RelayVSR at both 2x and 4x -- set "
                      "`scale` (2 or 4), or name one of its checkpoint "
                      "files", path)());
    }
    want = whole(2) ? 2 : whole(4) ? 4 : 0;
  }
  if (want == 0 || !whole(want)) {
    const RelayVsrFiles& f = by[want == 0 ? 4 : want];
    return fail(fmt("'{}' does not hold RelayVSR's two {}checkpoints -- a "
                    "keyframe model (refvsr-wan-v1 / relayvsr-wan-x2-v1) "
                    "and a decoder (refvsr-flash-v1 / relayvsr-flash-x2-v1)"
                    "{}",
                    path, want == 0 ? "" : fmt("{}x ", want)(),
                    f.wan.empty() && f.flash.empty()
                        ? std::string()
                        : f.wan.empty() ? "; the keyframe model is missing"
                                        : "; the decoder is missing")());
  }
  *out = by[want];
  out->scale = want;
  return true;
}

struct RelayVsrStream::Rng {
  explicit Rng(std::uint64_t seed) : gen(seed) {}
  TorchCpuRandn gen;
};

RelayVsrStream::RelayVsrStream(MetalRelayVsrWan* wan,
                               MetalRelayVsrDecoder* dec, MetalCompute* mc,
                               const Options& opt)
  : _wan(wan), _dec(dec), _mc(mc), _opt(opt)
{
  if (_opt.window < 2) { _opt.window = 2; }
  _op16 = std::make_unique<relayvsr::Ops>();
  _op_bf16 = std::make_unique<relayvsr::Ops>();
  std::string e;
  _op16->load(mc, false, &e);
  _op_bf16->load(mc, true, &e);
}

RelayVsrStream::~RelayVsrStream() = default;

void
RelayVsrStream::reset()
{
  _h = _w = _ph = _pw = 0;
  _n_in = _n_out = 0;
  _n_key = 0;
  _first_window = true;
  _ready = _final = false;
  _pos = 0;
  for (Held& f : _window) { _spare.push_back(std::move(f.px)); }
  _window.clear();
  _rng.reset();
  if (_wan != nullptr) { _wan->reset(); }
}

std::uint64_t
RelayVsrStream::run_bytes(const MetalRelayVsrWan::Config& wan,
                          const MetalRelayVsrDecoder::Config& dec, int h,
                          int w, int window)
{
  // Padded to the token stride (8 at 4x, 16 at 2x); the latent grid and
  // the output follow the scale.
  const int ts = wan.token_stride(), ls = wan.latent_stride();
  const std::uint64_t s = (std::uint64_t)wan.upscale;
  const int ph = h + (ts - h % ts) % ts, pw = w + (ts - w % ts) % ts;
  std::uint64_t b = MetalRelayVsrWan::ring_bytes(wan, ph, pw) +
                    MetalRelayVsrWan::scratch_bytes(wan, ph, pw) +
                    2u * MetalRelayVsrDecoder::condition_bytes(dec, ph, pw) +
                    MetalRelayVsrDecoder::scratch_bytes(dec, ph, pw);
  // The window's frames (f32 at worst), the two uploads, the latent and
  // the output frame (f32 at worst).
  b += (std::uint64_t)(window + 1) * 3u * (std::uint64_t)h * w * 4u;
  b += 2u * (std::uint64_t)ph * pw * 3u * 2u;
  b += (std::uint64_t)48 * (ph / ls) * (pw / ls) * 4u;
  b += 3u * s * s * (std::uint64_t)h * w * 4u;
  return b;
}

bool
RelayVsrStream::upload_(const Held& f, bool bf16, SharedBuffer* dst,
                        std::string* err)
{
  const std::size_t bytes = (std::size_t)_ph * _pw * 3 * 2;
  if (dst->empty() || dst->byte_size() < bytes) {
    *dst = _mc->make_shared_buffer(bytes);
    if (dst->empty()) {
      if (err != nullptr) { *err = "frame upload allocation failed"; }
      return false;
    }
  }
  const relayvsr::Ops& op = bf16 ? *_op_bf16 : *_op16;
  if (!_have_cs) {
    _cs = _mc->make_command_stream();
    _have_cs = true;
  }
  ComputeEncoder enc = _cs.begin_compute();
  enc.set_function(op.upload);
  enc.set_buffer(0, f.px); enc.set_buffer(1, *dst);
  enc.set_constant(2, _h); enc.set_constant(3, _w);
  enc.set_constant(4, _ph); enc.set_constant(5, _pw);
  enc.set_constant(6, (int)_fmt);
  enc.dispatch({(unsigned)_pw, (unsigned)_ph, 1}, {32, 8, 1});
  enc.end();
  std::string ge;
  if (!_cs.commit().wait_ok(&ge)) {
    if (err != nullptr) { *err = "frame upload failed: " + ge; }
    return false;
  }
  return true;
}

bool
RelayVsrStream::keyframe_(const Held& f,
                          MetalRelayVsrDecoder::Condition* cond,
                          std::string* err,
                          const std::function<bool()>& stop)
{
  if (!_rng) { _rng = std::make_unique<Rng>(_opt.seed); }
  if (!upload_(f, true, &_lr_bf16, err)) { return false; }
  const int ls = _wan->config().latent_stride();
  const std::vector<float> noise =
      _rng->gen.randn((std::size_t)48 * (_ph / ls) * (_pw / ls));
  if (!_wan->keyframe(_lr_bf16, _ph, _pw, noise, &_latent, err, stop)) {
    return false;
  }
  ++_n_key;
  if (on_keyframe) { on_keyframe(f.index, noise, _latent); }
  return _dec->prepare(_latent, _ph / ls, _pw / ls, cond, err);
}

bool
RelayVsrStream::next(Frame* f, std::string* err,
                     const std::function<bool()>& stop)
{
  if (!_ready) { return false; }
  const bool same = _window.size() == 1;    // a one-frame stream
  const MetalRelayVsrDecoder::Condition& end = same ? _start : _end;
  const int H = scale() * _h, W = scale() * _w;
  const std::size_t bytes = (std::size_t)3 * H * W * (_opt.f32_out ? 4 : 1);
  while (_pos < _window.size()) {
    if (stop && stop()) { return false; }
    if (_pos == 0) { _dec->begin_window(); }
    const Held& in = _window[_pos];
    if (!upload_(in, false, &_lr16, err)) { return false; }
    if (!_dec->decode(_lr16, _ph, _pw, _start, end, H, W, _opt.f32_out,
                      &_frame, err)) {
      return false;
    }
    const std::int64_t index = in.index;
    // The overlap -- the previous window's last frame -- is history only.
    const bool out = _first_window || _pos > 0;
    ++_pos;
    if (out) { ++_n_out; }
    if (_pos == _window.size()) { close_window_(); }
    if (out) {
      f->index = index;
      f->h = H;
      f->w = W;
      f->px = _frame.contents();
      f->bytes = bytes;
      return true;
    }
  }
  return false;
}

void
RelayVsrStream::close_window_()
{
  _ready = false;
  _pos = 0;
  if (_final) {
    reset();
    return;
  }
  _first_window = false;
  while (_window.size() > 1) {
    _spare.push_back(std::move(_window.front().px));
    _window.pop_front();
  }
  std::swap(_start, _end);
}

bool
RelayVsrStream::push(const void* px, In kind, int h, int w, std::string* err,
                     const std::function<bool()>& stop)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (_ready) {
    return fail("RelayVSR: a frame pushed before the window before it was "
                "drained (next())");
  }
  if (h <= 0 || w <= 0) { return fail("an empty frame"); }
  if (_wan->config().upscale != _dec->config().upscale) {
    return fail(fmt("RelayVSR: a {}x keyframe model with a {}x decoder -- "
                    "the two checkpoints must be of one scale",
                    _wan->config().upscale, _dec->config().upscale)());
  }
  if (_n_in > 0 && (h != _h || w != _w || kind != _fmt)) {
    return fail(fmt("RelayVSR: a stream's frames must keep one size and "
                    "type ({}x{} then {}x{})", _w, _h, w, h)());
  }
  if (_n_in == 0) {
    _h = h;
    _w = w;
    const int ts = _wan->config().token_stride();
    _ph = h + (ts - h % ts) % ts;
    _pw = w + (ts - w % ts) % ts;
    _fmt = kind;
    _first_window = true;
    _wan->reset();
    _rng.reset();
  }
  const std::size_t esz = kind == In::kU8 ? 1 : kind == In::kF32 ? 4 : 2;
  const std::size_t bytes = (std::size_t)3 * h * w * esz;
  Held f;
  f.index = _n_in;
  while (!_spare.empty() && f.px.empty()) {
    f.px = std::move(_spare.back());
    _spare.pop_back();
    if (f.px.byte_size() < bytes) { f.px = SharedBuffer{}; }
  }
  if (f.px.empty()) { f.px = _mc->make_shared_buffer(bytes); }
  if (f.px.empty()) { return fail("frame allocation failed"); }
  std::memcpy(f.px.contents(), px, bytes);
  ++_n_in;
  if (f.index == 0) {
    if (!keyframe_(f, &_start, err, stop)) { return false; }
  }
  _window.push_back(std::move(f));
  if ((int)_window.size() < _opt.window) { return true; }
  // The window is whole: its end is the next keyframe, and its frames
  // are next()'s.
  if (!keyframe_(_window.back(), &_end, err, stop)) { return false; }
  _ready = true;
  _pos = 0;
  return true;
}

bool
RelayVsrStream::finish(std::string* err, const std::function<bool()>& stop)
{
  if (_ready) {
    if (err != nullptr) {
      *err = "RelayVSR: finish() before the last window was drained";
    }
    return false;
  }
  if (_window.empty()) { return true; }
  if (_window.size() == 1 && !_first_window) {
    // Its only frame closed the last window and is already out.
    reset();
    return true;
  }
  if (_window.size() > 1 &&
      !keyframe_(_window.back(), &_end, err, stop)) {
    return false;
  }
  _ready = _final = true;
  _pos = 0;
  return true;
}

}  // namespace genai
}  // namespace vpipe
