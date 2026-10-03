#include "generative-models/yue2/yue2-protocol.h"

#include "generative-models/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <unordered_map>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace vpipe::genai::yue2 {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

}  // namespace

std::vector<std::string>
tiktoken_specials()
{
  std::vector<std::string> s = {"<|endoftext|>", "<|im_start|>",
                                "<|im_end|>",    "<R>",
                                "<S>",           "<X>",
                                "<mask>",        "<sep>"};
  for (int i = 0; i < 200; ++i) {
    s.push_back("<extra_" + std::to_string(i) + ">");
  }
  // The reference overwrites two of the extras in place, which is what
  // puts <abc> at 151643 + 204.
  s[204] = "<abc>";
  s[205] = "</abc>";
  return s;
}

bool
parse_cot(std::string_view s, Cot* out)
{
  if (s == "off")    { *out = Cot::kOff;    return true; }
  if (s == "melody") { *out = Cot::kMelody; return true; }
  if (s == "full")   { *out = Cot::kFull;   return true; }
  return false;
}

const char*
cot_name(Cot c)
{
  switch (c) {
  case Cot::kOff:    return "off";
  case Cot::kMelody: return "melody";
  case Cot::kFull:   return "full";
  }
  return "full";
}

std::string_view
instruction(Cot c)
{
  switch (c) {
  case Cot::kOff:
    return "Generate music with codec tokens from the given conditions.";
  case Cot::kMelody:
    return "Generate a melody-only ABC transcription without chord "
           "symbols, then generate music with codec tokens from the given "
           "conditions.";
  case Cot::kFull:
    return "Generate a chord-annotated ABC transcription, then generate "
           "music with codec tokens from the given conditions.";
  }
  return {};
}

Sampling
Sampling::abc_default()
{
  Sampling s;
  s.temperature = 0.7f;
  s.top_p = 0.9f;
  s.top_k = 30;
  s.repetition_penalty = 1.005f;
  s.penalty_window = 100;
  s.min_tokens = 32;
  s.max_tokens = 4096;
  return s;
}

Sampling
Sampling::semantic_default()
{
  return Sampling{};
}

bool
Sampling::valid(std::string* err) const
{
  auto fail = [&](const char* m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (!std::isfinite(temperature) || temperature < 0.0f ||
      temperature > 5.0f) {
    return fail("temperature must be in [0, 5]");
  }
  if (!std::isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f) {
    return fail("top_p must be in (0, 1]");
  }
  if (top_k < 1) { return fail("top_k must be >= 1"); }
  if (!std::isfinite(repetition_penalty) || repetition_penalty <= 0.0f) {
    return fail("repetition_penalty must be > 0");
  }
  if (penalty_window < 1 || penalty_window > 100) {
    return fail("penalty_window must be in [1, 100]");
  }
  if (max_tokens < 1 || min_tokens < 0 || min_tokens > max_tokens) {
    return fail("require 0 <= min_tokens <= max_tokens and max_tokens >= 1");
  }
  return true;
}

float
Request::guidance() const
{
  if (cfg_scale >= 0.0f) { return cfg_scale; }
  return cot == Cot::kOff ? 1.01f : 1.0f;
}

std::string
Request::text() const
{
  std::string t(instruction(cot));
  t += "\n[Tags]\n";
  t += style;
  t += "\n[Lyrics]\n";
  t += lyrics;
  t += "\n";
  return nfc(t);
}

bool
Request::valid(std::string* err) const
{
  auto fail = [&](const char* m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (!abc.empty() && cot == Cot::kOff) {
    return fail("a supplied ABC score needs cot melody or full");
  }
  if (cfg_scale >= 0.0f && (!std::isfinite(cfg_scale) || cfg_scale > 20.0f)) {
    return fail("cfg_scale must be in [0, 20]");
  }
  return true;
}

std::string
nfc(std::string_view s)
{
#ifdef __APPLE__
  bool ascii = true;
  for (unsigned char c : s) {
    if (c >= 0x80) { ascii = false; break; }
  }
  if (ascii) { return std::string(s); }
  CFStringRef in = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8*>(s.data()),
      (CFIndex)s.size(), kCFStringEncodingUTF8, false);
  if (in == nullptr) { return std::string(s); }
  CFMutableStringRef m = CFStringCreateMutableCopy(kCFAllocatorDefault, 0, in);
  CFRelease(in);
  if (m == nullptr) { return std::string(s); }
  CFStringNormalize(m, kCFStringNormalizationFormC);
  const CFIndex len = CFStringGetLength(m);
  CFIndex need = 0;
  CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false,
                   nullptr, 0, &need);
  std::string out((std::size_t)need, '\0');
  CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false,
                   reinterpret_cast<UInt8*>(out.data()), need, nullptr);
  CFRelease(m);
  return out;
#else
  return std::string(s);
#endif
}

std::vector<std::int32_t>
token_prefix(const Request& r, const Tokenizer& tok,
             const std::vector<std::int32_t>* abc_ids)
{
  std::vector<std::int32_t> p{kEod};
  const std::vector<std::int32_t> text = tok.encode(r.text());
  p.insert(p.end(), text.begin(), text.end());
  if (r.cot == Cot::kOff) {
    p.insert(p.end(), {kAbcStart, kAbcEnd, kMusicStart});
    return p;
  }
  std::vector<std::int32_t> own;
  if (abc_ids == nullptr) {
    if (r.abc.empty()) {
      p.push_back(kAbcStart);
      return p;
    }
    own = tok.encode(nfc(r.abc));
    abc_ids = &own;
  }
  p.push_back(kAbcStart);
  p.insert(p.end(), abc_ids->begin(), abc_ids->end());
  p.insert(p.end(), {kAbcEnd, kMusicStart});
  return p;
}

std::vector<std::int32_t>
negative_prefix(const Request& r, const Tokenizer& tok,
                const std::vector<std::int32_t>* abc_ids)
{
  std::vector<std::int32_t> p{kEod};
  const std::vector<std::int32_t> text =
      tok.encode(nfc(instruction(r.cot)));
  p.insert(p.end(), text.begin(), text.end());
  if (r.cot == Cot::kOff) {
    p.push_back(kMusicStart);
    return p;
  }
  std::vector<std::int32_t> own;
  if (abc_ids == nullptr) {
    own = tok.encode(nfc(r.abc));
    abc_ids = &own;
  }
  p.push_back(kAbcStart);
  p.insert(p.end(), abc_ids->begin(), abc_ids->end());
  p.insert(p.end(), {kAbcEnd, kMusicStart});
  return p;
}

std::vector<std::pair<int, int>>
chunk_ranges(int frames, int prefix_tokens, int context)
{
  std::vector<std::pair<int, int>> out;
  const int size = std::min((context - prefix_tokens - 3) / 2, kContext);
  if (frames < 1 || size < 1) { return out; }
  for (int a = 0; a < frames; a += size) {
    out.emplace_back(a, std::min(a + size, frames));
  }
  return out;
}

float
bf16_round(float x)
{
  std::uint32_t u;
  std::memcpy(&u, &x, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) { return x; }   // inf / nan
  u += 0x7fffu + ((u >> 16) & 1u);
  u &= 0xffff0000u;
  float y;
  std::memcpy(&y, &u, 4);
  return y;
}

void
cfg_combine(const float* cond, const float* uncond, float scale, float* out,
            int n)
{
  for (int i = 0; i < n; ++i) {
    const float d = bf16_round(cond[i] - uncond[i]);
    const float m = bf16_round(scale * d);
    out[i] = bf16_round(uncond[i] + m);
  }
}

Window
phase_window(Phase p)
{
  if (p == Phase::kAbc) {
    // Text ids plus </abc>; the specials in between are masked by pick().
    return Window{0, kAbcEnd + 1, kAbcEnd};
  }
  // MUSIC_END sits just below the codec block, so one window holds both.
  return Window{kMusicEnd, kCodecOffset + kCodecSize - kMusicEnd, kMusicEnd};
}

TokenPicker::TokenPicker(Phase p, const Sampling& s, std::uint64_t seed,
                         bool legacy)
    : _phase(p), _s(s), _legacy(legacy), _w(phase_window(p)), _rng(seed)
{
  _scores.resize((std::size_t)_w.rows);
}

std::int32_t
TokenPicker::pick(const float* logits, int step)
{
  const int n = _w.rows;
  float* sc = _scores.data();
  std::memcpy(sc, logits, (std::size_t)n * sizeof(float));
  if (_phase == Phase::kAbc) {
    // Only ordinary text and </abc>: <|endoftext|> .. <abc> are barred.
    for (int i = kEod; i < kAbcEnd; ++i) { sc[i - _w.row0] = kNegInf; }
  }
  if (step < _s.min_tokens) { sc[_w.end - _w.row0] = kNegInf; }

  // The windowed repetition penalty. A logit below zero is MULTIPLIED by
  // penalty^count and one at or above it DIVIDED, so either way a
  // repeated id loses ground.
  if (_s.repetition_penalty != 1.0f && !_history.empty()) {
    const std::size_t win = (std::size_t)_s.penalty_window;
    const std::size_t from =
        _history.size() > win ? _history.size() - win : 0;
    std::unordered_map<std::int32_t, int> freq;
    for (std::size_t i = from; i < _history.size(); ++i) {
      ++freq[_history[i]];
    }
    for (const auto& [id, cnt] : freq) {
      const int j = id - _w.row0;
      if (j < 0 || j >= n) { continue; }
      float a = std::pow(_s.repetition_penalty, (float)cnt);
      float& x = sc[j];
      if (_legacy) {
        a = bf16_round(a);
        x = bf16_round(x < 0.0f ? x * a : x / a);
      } else {
        x = x < 0.0f ? x * a : x / a;
      }
    }
  }

  if (_s.temperature == 0.0f) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
      if (sc[i] > sc[best]) { best = i; }
    }
    return _w.row0 + best;
  }

  // Candidates: every finite score, scaled by the temperature.
  _cand.clear();
  const bool scale = _s.temperature != 1.0f;
  for (int i = 0; i < n; ++i) {
    if (sc[i] == kNegInf) { continue; }
    float v = scale ? sc[i] / _s.temperature : sc[i];
    if (_legacy && scale) { v = bf16_round(v); }
    _cand.emplace_back(v, _w.row0 + i);
  }
  if (_cand.empty()) { return _w.end; }
  auto desc = [](const std::pair<float, std::int32_t>& a,
                 const std::pair<float, std::int32_t>& b) {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
  };
  // top-k keeps everything AT the k-th value, ties included.
  const std::size_t k = std::min<std::size_t>((std::size_t)_s.top_k,
                                              _cand.size());
  std::nth_element(_cand.begin(), _cand.begin() + (long)(k - 1), _cand.end(),
                   desc);
  const float thr = _cand[k - 1].first;
  _cand.erase(std::remove_if(_cand.begin(), _cand.end(),
                             [&](const auto& c) { return c.first < thr; }),
              _cand.end());
  std::sort(_cand.begin(), _cand.end(), desc);

  // Softmax over the kept set (the masked rest contributes exp(-inf)).
  const float mx = _cand.front().first;
  std::vector<double> p(_cand.size());
  double z = 0.0;
  for (std::size_t i = 0; i < _cand.size(); ++i) {
    p[i] = std::exp((double)_cand[i].first - (double)mx);
    z += p[i];
  }
  for (double& v : p) { v /= z; }
  if (_s.top_p < 1.0f) {
    // Drop a candidate once the mass BEFORE it already exceeds top_p;
    // the first one (three under legacy) always stays.
    const std::size_t keep_min = _legacy ? 3 : 1;
    double cum = 0.0;
    std::size_t kept = p.size();
    for (std::size_t i = 0; i < p.size(); ++i) {
      if (i >= keep_min && cum > (double)_s.top_p) { kept = i; break; }
      cum += p[i];
    }
    p.resize(kept);
    double z2 = 0.0;
    for (double v : p) { z2 += v; }
    for (double& v : p) { v /= z2; }
  }
  const double u = std::generate_canonical<double, 53>(_rng);
  double acc = 0.0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    acc += p[i];
    if (u < acc) { return _cand[i].second; }
  }
  return _cand[p.size() - 1].second;
}

std::vector<float>
torch_cpu_randn(std::uint64_t seed, std::size_t n)
{
  std::vector<float> d(n);
  if (n < 16) { return d; }
  // at::mt19937 seeds with the low 32 bits exactly as std::mt19937 does,
  // and CPUGeneratorImpl::random() is one draw of it.
  std::mt19937 gen((std::uint32_t)(seed & 0xffffffffu));
  auto uniform = [&]() -> float {
    // uniform_real_distribution<float>: 24 random bits over 2^24.
    const double x = (double)(gen() & 0xffffffu) * (1.0 / 16777216.0);
    return (float)x;
  };
  auto fill16 = [](float* p) {
    for (int j = 0; j < 8; ++j) {
      const float u1 = 1.0f - p[j];
      const float u2 = p[j + 8];
      const float radius = std::sqrt(-2.0f * std::log(u1));
      // ATen spells theta `2.0f * pi<double> * u2` into a FLOAT: the
      // product is double, the angle and its cos/sin are not. Keeping it
      // double is off by an ulp in about half the draws.
      const float theta = (float)(2.0 * std::numbers::pi * (double)u2);
      p[j]     = radius * std::cos(theta);
      p[j + 8] = radius * std::sin(theta);
    }
  };
  for (std::size_t i = 0; i < n; ++i) { d[i] = uniform(); }
  for (std::size_t i = 0; i + 16 <= n; i += 16) { fill16(d.data() + i); }
  if (n % 16 != 0) {
    float* t = d.data() + n - 16;
    for (int i = 0; i < 16; ++i) { t[i] = uniform(); }
    fill16(t);
  }
  return d;
}

}  // namespace vpipe::genai::yue2
