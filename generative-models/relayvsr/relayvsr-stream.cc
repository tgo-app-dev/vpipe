#include "generative-models/relayvsr/relayvsr-stream.h"

#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/relayvsr/relayvsr-ops.h"
#include "generative-models/shared/torch-randn.h"

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

bool
resolve_relayvsr_files(const std::string& path, RelayVsrFiles* out,
                       std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  std::error_code ec;
  fs::path dir(path);
  if (fs::is_regular_file(dir, ec)) { dir = dir.parent_path(); }
  if (!fs::is_directory(dir, ec)) {
    return fail(fmt("'{}' is not a RelayVSR model directory", path)());
  }
  RelayVsrFiles f;
  // The release's names first; any other safetensors told by its format.
  const fs::path wan = dir / "wan.safetensors";
  const fs::path flash = dir / "flash.safetensors";
  if (format_of(wan.string()) == "refvsr-wan-v1") { f.wan = wan.string(); }
  if (format_of(flash.string()) == "refvsr-flash-v1") {
    f.flash = flash.string();
  }
  if (f.wan.empty() || f.flash.empty()) {
    for (const auto& e : fs::directory_iterator(dir, ec)) {
      if (e.path().extension() != ".safetensors") { continue; }
      const std::string fm = format_of(e.path().string());
      if (fm == "refvsr-wan-v1" && f.wan.empty()) {
        f.wan = e.path().string();
      } else if (fm == "refvsr-flash-v1" && f.flash.empty()) {
        f.flash = e.path().string();
      }
    }
  }
  if (f.wan.empty() || f.flash.empty()) {
    return fail(fmt("'{}' does not hold RelayVSR's two checkpoints -- a "
                    "keyframe model (refvsr-wan-v1, wan.safetensors) and a "
                    "decoder (refvsr-flash-v1, flash.safetensors){}",
                    dir.string(),
                    f.wan.empty() && f.flash.empty()
                        ? std::string()
                        : f.wan.empty() ? "; the keyframe model is missing"
                                        : "; the decoder is missing")());
  }
  *out = f;
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
  const int ph = h + (8 - h % 8) % 8, pw = w + (8 - w % 8) % 8;
  std::uint64_t b = MetalRelayVsrWan::ring_bytes(wan, ph, pw) +
                    MetalRelayVsrWan::scratch_bytes(wan, ph, pw) +
                    2u * MetalRelayVsrDecoder::condition_bytes(dec, ph, pw) +
                    MetalRelayVsrDecoder::scratch_bytes(dec, ph, pw);
  // The window's frames (f32 at worst), the two uploads, the latent and
  // the output frame (f32 at worst).
  b += (std::uint64_t)(window + 1) * 3u * (std::uint64_t)h * w * 4u;
  b += 2u * (std::uint64_t)ph * pw * 3u * 2u;
  b += (std::uint64_t)48 * (ph / 4) * (pw / 4) * 4u;
  b += 3u * 16u * (std::uint64_t)h * w * 4u;
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
  const std::vector<float> noise =
      _rng->gen.randn((std::size_t)48 * (_ph / 4) * (_pw / 4));
  if (!_wan->keyframe(_lr_bf16, _ph, _pw, noise, &_latent, err, stop)) {
    return false;
  }
  ++_n_key;
  if (on_keyframe) { on_keyframe(f.index, noise, _latent); }
  return _dec->prepare(_latent, _ph / 4, _pw / 4, cond, err);
}

bool
RelayVsrStream::next(Frame* f, std::string* err,
                     const std::function<bool()>& stop)
{
  if (!_ready) { return false; }
  const bool same = _window.size() == 1;    // a one-frame stream
  const MetalRelayVsrDecoder::Condition& end = same ? _start : _end;
  const int H = 4 * _h, W = 4 * _w;
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
  if (_n_in > 0 && (h != _h || w != _w || kind != _fmt)) {
    return fail(fmt("RelayVSR: a stream's frames must keep one size and "
                    "type ({}x{} then {}x{})", _w, _h, w, h)());
  }
  if (_n_in == 0) {
    _h = h;
    _w = w;
    _ph = h + (8 - h % 8) % 8;
    _pw = w + (8 - w % 8) % 8;
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
