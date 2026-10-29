#include "generative-models/minimax-music3/metal-music3-dit.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/minimax-music3/music3-protocol.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/shared/torch-randn.h"
#include "generative-models/weight-set.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numbers>

// The host arithmetic -- the time embedding, the blend, the Euler step --
// is the fp32 reference's, operation by operation.
#pragma clang fp contract(off)

namespace vpipe::genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

constexpr const char* kKey = "music3-dit/f16|";
constexpr int kMmaMinRows = 64;

// The steel flash-attention parameter block, as attn_steel.metal reads
// it. Mirrored per family because each one builds its own strides.
struct SteelAttnParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NK;
  int NQ_aligned, NK_aligned;
  int qL_rem, kL_rem, qL_off;
  std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

bool
read_json_(const std::filesystem::path& p, FlexData* out)
{
  std::ifstream in(p);
  if (!in) { return false; }
  try {
    *out = FlexData::from_json(in);
  } catch (const std::exception&) {
    return false;
  }
  return out->is_object();
}

// A tensor widened to f32. The release stores F32; an F16 or BF16 copy of
// the same checkpoint reads too.
bool
read_f32_(WeightSet& ws, MetalCompute* mc, const std::string& nm,
          std::vector<float>* out, std::size_t want = 0)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr) { return false; }
  const std::string& dt = info->dtype;
  const std::size_t es = dt == "F32" ? 4 : (dt == "F16" || dt == "BF16")
                                               ? 2 : 0;
  if (es == 0) { return false; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  if (want != 0 && n != want) { return false; }
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty() || raw.byte_size() < n * es) { return false; }
  out->resize(n);
  if (es == 4) {
    std::memcpy(out->data(), raw.contents(), n * 4);
  } else if (dt == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = (float)s[i]; }
  } else {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t u = (std::uint32_t)s[i] << 16;
      std::memcpy(&(*out)[i], &u, 4);
    }
  }
  return true;
}

// A tensor as the checkpoint stores it -- F32 in the release, F16 or BF16
// in a copy -- read into a buffer of its own. The loader narrows to f16
// STRAIGHT FROM IT: widening 9 GB of F32 through host vectors would leave
// most of them in the process's footprint, since macOS keeps a freed large
// heap block resident and dirty.
struct Raw {
  SharedBuffer buf;
  int es = 0;                 // 4 F32, 2 F16, 1 BF16 (two bytes)
  std::size_t n = 0;
};

bool
read_raw_(WeightSet& ws, MetalCompute* mc, const std::string& nm,
          std::size_t want, Raw* out)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr) { return false; }
  const std::string& dt = info->dtype;
  const int es = dt == "F32" ? 4 : dt == "F16" ? 2 : dt == "BF16" ? 1 : 0;
  if (es == 0) { return false; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  if (want != 0 && n != want) { return false; }
  out->buf = ws.read(nm, mc, WeightSet::Residency::Copied);
  out->es = es;
  out->n = n;
  return !out->buf.empty() &&
         out->buf.byte_size() >= n * (std::size_t)(es == 4 ? 4 : 2);
}

// dst[i] = f16(r[from + i]) for i < n, on this thread.
void
narrow_run_(const Raw& r, std::size_t from, _Float16* dst, std::size_t n)
{
  const void* p = r.buf.contents();
  if (r.es == 4) {
    const float* s = static_cast<const float*>(p) + from;
    for (std::size_t i = 0; i < n; ++i) { dst[i] = (_Float16)s[i]; }
  } else if (r.es == 2) {
    std::memcpy(dst, static_cast<const _Float16*>(p) + from, n * 2);
  } else {
    const std::uint16_t* s = static_cast<const std::uint16_t*>(p) + from;
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t u = (std::uint32_t)s[i] << 16;
      float f;
      std::memcpy(&f, &u, 4);
      dst[i] = (_Float16)f;
    }
  }
}

// The whole of r into dst, in parallel: 2.4 B weights narrow at load.
void
narrow_(const Raw& r, _Float16* dst)
{
  constexpr std::size_t kChunk = (std::size_t)1 << 20;
  const std::size_t n = r.n;
  const std::size_t chunks = (n + kChunk - 1) / kChunk;
  const Raw* rp = &r;
  dispatch_apply(chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                 ^(std::size_t c) {
    const std::size_t a = c * kChunk;
    narrow_run_(*rp, a, dst + a, std::min(n, a + kChunk) - a);
  });
}

// A tensor narrowed to f16 into a buffer of its own; empty on failure.
SharedBuffer
read_f16_(WeightSet& ws, MetalCompute* mc, const std::string& nm,
          std::size_t want)
{
  Raw r;
  if (!read_raw_(ws, mc, nm, want, &r)) { return {}; }
  SharedBuffer o = mc->make_shared_buffer(r.n * 2);
  if (o.empty()) { return {}; }
  narrow_(r, static_cast<_Float16*>(o.contents()));
  return o;
}

// The fast tier's group for K, as I8GemmContext::fast_group picks it: the
// largest of 1024 / 512 / 256 no more than VPIPE_I8_FAST_GROUP (512) that
// divides K.
int
fast_group_for_(int K)
{
  int target = 512;
  if (const char* e = std::getenv("VPIPE_I8_FAST_GROUP")) {
    const int v = std::atoi(e);
    if (v == 256 || v == 512 || v == 1024) { target = v; }
  }
  for (int gc : {1024, 512, 256}) {
    if (gc <= target && K % gc == 0) { return gc; }
  }
  return 0;
}

// Whether a transformer tensor is one of a block's four projections (the
// ones an int8 tier holds): q / k / v, the attention's output, ff_in and
// ff_out.
bool
block_projection_(const std::string& n)
{
  if (n.rfind("transformer_blocks.", 0) != 0) { return false; }
  auto ends = [&](const char* t) {
    const std::size_t l = std::strlen(t);
    return n.size() >= l && n.compare(n.size() - l, l, t) == 0;
  };
  return ends(".attn.to_q.weight") || ends(".attn.to_k.weight") ||
         ends(".attn.to_v.weight") || ends(".attn.to_out.0.weight") ||
         ends(".ff_in.weight") || ends(".ff_out.weight");
}

}  // namespace

// ---- config ---------------------------------------------------------------

bool
MetalMusic3Dit::config_from_dir(const std::string& root, Config* out,
                                std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const std::filesystem::path r(root);
  FlexData t, c;
  if (!read_json_(r / "transformer" / "config.json", &t)) {
    return fail("no transformer/config.json under " + root);
  }
  if (!read_json_(r / "condition_encoder" / "config.json", &c)) {
    return fail("no condition_encoder/config.json under " + root);
  }
  auto to = t.as_object();
  auto co = c.as_object();
  auto cls = [](const auto& o) {
    return o.contains("_class_name") && o.at("_class_name").is_string()
               ? std::string(o.at("_class_name").as_string())
               : std::string();
  };
  if (cls(to) != "MiniMaxMusic3Transformer1DModel") {
    return fail("transformer/ is not MiniMaxMusic3Transformer1DModel");
  }
  if (cls(co) != "MiniMaxMusic3ConditionEncoder") {
    return fail("condition_encoder/ is not MiniMaxMusic3ConditionEncoder");
  }
  auto gi = [](const auto& o, const char* k, int v) {
    return o.contains(k) ? (int)o.at(k).as_int(v) : v;
  };
  Config x;
  x.in_channels = gi(to, "in_channels", x.in_channels);
  x.cond_dim    = gi(to, "condition_dim", x.cond_dim);
  x.layers      = gi(to, "num_layers", x.layers);
  x.heads       = gi(to, "num_attention_heads", x.heads);
  x.head_dim    = gi(to, "attention_head_dim", x.head_dim);
  x.ff_inner    = gi(to, "ff_inner_dim", x.ff_inner);
  x.rotary_dim  = gi(to, "rotary_dim", x.rotary_dim);
  x.fourier_dim = gi(to, "fourier_embedding_dim", x.fourier_dim);
  x.cond_hidden = gi(co, "condition_hidden_dim", x.cond_hidden);
  x.cond_layers = gi(co, "num_condition_layers", x.cond_layers);
  // The rest of the condition encoder's config is the protocol's frame
  // and latent rates; anything else is a different resampling.
  if (gi(co, "out_dim", x.cond_dim) != x.cond_dim ||
      gi(co, "input_sampling_rate", 24000) != 24000 ||
      gi(co, "input_hop_length", 960) != 960 ||
      gi(co, "output_sampling_rate", 44100) != music3::kSampleRate ||
      gi(co, "output_hop_length", 512) != music3::kHop) {
    return fail("condition_encoder/ resamples at rates this port does not");
  }
  if (x.head_dim != 64 || x.rotary_dim > x.head_dim ||
      x.rotary_dim % 2 != 0 || x.in_channels != music3::kLatentChannels ||
      x.cond_hidden * x.cond_layers !=
          music3::kCodebooks * 4096) {
    return fail("a transformer shape this port does not run");
  }
  *out = x;
  return true;
}

int
MetalMusic3Dit::effective_i8_tier(int i8_tier, bool matrix_cores)
{
  // I8GemmContext's own reading of the two, in its order.
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

const char*
MetalMusic3Dit::i8_tier_name(int tier)
{
  switch (tier) {
    case 0: return "accurate";
    case 1: return "fast";
    case 2: return "token";
    case 3: return "token_plain";
    default: return "off";
  }
}

namespace {

// What one tensor holds: x 4 for what `host` names, a block projection's
// int8 layout at a holding tier (codes, and scales at its group), else x 2
// for what narrows to f16.
std::size_t
held_bytes_(const std::string& dir, bool (*host)(const std::string&),
            int tier = -1)
{
  auto wts = MetalLlamaWeights::open_model(dir);
  if (!wts.has_value()) { return 0; }
  std::size_t b = 0;
  for (const std::string& nm : wts->tensor_names()) {
    const auto* ti = wts->info(nm);
    if (ti == nullptr) { continue; }
    std::size_t n = 1;
    for (auto d : ti->shape) { n *= (std::size_t)d; }
    if (host(nm)) {
      b += n * 4;
      continue;
    }
    if (tier >= 1 && ti->shape.size() == 2 && block_projection_(nm)) {
      const int N = (int)ti->shape[0], K = (int)ti->shape[1];
      const int gc = tier >= 2 ? I8GemmContext::kTokenGroup
                               : fast_group_for_(K);
      if (gc > 0) {
        b += n + I8GemmContext::fast_scale_bytes(N, K, gc);
        continue;
      }
    }
    b += n * 2;
  }
  return b;
}

}  // namespace

std::size_t
MetalMusic3Dit::transformer_weight_bytes(const std::string& root,
                                         int effective_tier)
{
  return held_bytes_(
      (std::filesystem::path(root) / "transformer").string(),
      [](const std::string& n) {
        return n.rfind("time_proj.", 0) == 0 || n.rfind("time_embed.", 0) == 0;
      }, effective_tier);
}

std::size_t
MetalMusic3Dit::condition_weight_bytes(const std::string& root)
{
  // The mix's logits become the softmax weights (and the scale), on the
  // host; the conv narrows to f16.
  return held_bytes_(
      (std::filesystem::path(root) / "condition_encoder").string(),
      [](const std::string& n) { return n.rfind("proj.", 0) != 0; });
}

std::size_t
MetalMusic3Dit::resident_bytes() const
{
  std::size_t b = _pre.byte_size() + _proj_in.byte_size() +
                  _proj_out.byte_size() + _post.byte_size() +
                  condition_bytes();
  b += (_fourier.size() + _te1_w.size() + _te1_b.size() + _te2_w.size() +
        _te2_b.size()) * 4;
  for (const Block& k : _blocks) {
    b += k.ln1_w.byte_size() + k.ln1_b.byte_size() + k.ln2_w.byte_size() +
         k.ln2_b.byte_size() + k.qkv.bytes() + k.out.bytes() +
         k.ff_in.bytes() + k.ff_in_b.byte_size() +
         k.ff_out.bytes() + k.ff_out_b.byte_size();
  }
  return b;
}

std::size_t
MetalMusic3Dit::window_scratch_bytes(const Config& c, int effective_tier)
{
  const std::size_t L =
      (std::size_t)music3::latent_length(music3::kChunkFrames);
  const std::size_t M = 2 * (L + 1);
  const std::size_t D = (std::size_t)c.dim();
  const std::size_t Cc = (std::size_t)c.concat();
  std::size_t e = 0;
  e += 2 * L * Cc * 2;                     // _in, _pre_o
  e += M * D * 4;                          // _x, _n, _att, _tmp
  e += M * 3 * D;                          // _qkv
  e += M * 2 * (std::size_t)c.ff_inner;    // _ff
  e += M * (std::size_t)c.ff_inner;        // _glu
  e += 2 * 2 * L * (std::size_t)c.in_channels;   // _po, _pq
  std::size_t b = e * 2;
  b += 2 * M * (std::size_t)(c.rotary_dim / 2) * 4;   // the rope tables
  // The condition encoder at a window: im2col [200, 3 * 4096] and its
  // output [200, 2048], f16.
  b += (std::size_t)music3::kChunkFrames *
       (std::size_t)(3 * c.cond_hidden + c.cond_dim) * 2;
  if (effective_tier >= 0) {
    // The int8 context at a window's rows, by its own plan-time bound:
    // the activation's codes and scales at the widest K (ff_out's) and --
    // accurate, which quantizes the weights per call -- the widest
    // weight's, the split tuner's output and its planes (it weighs a
    // two-way split at every shape: MEASURED 257 MB on the base M5, under
    // the bound's 512 MB plane budget). A held layout tunes nothing.
    const int F2 = 2 * c.ff_inner;
    const bool acc = effective_tier == 0;
    b += I8GemmContext::forward_bytes_bound((int)M, c.ff_inner,
                                            acc ? F2 : 0, acc ? (int)D : 0,
                                            acc);
    // A held layout runs the int8 route at any row count; only a floor
    // set from the environment (VPIPE_I8_GEMM_MIN_M) sends it to the f16
    // stand-in, made once at the widest projection.
    if (!acc && std::getenv("VPIPE_I8_GEMM_MIN_M") != nullptr) {
      b += (std::size_t)F2 * D * 2;
    }
  }
  return b + sizeof(SteelAttnParams);
}

std::size_t
MetalMusic3Dit::scratch_bytes() const
{
  std::size_t b = 0;
  for (const SharedBuffer* s :
       {&_in, &_pre_o, &_x, &_n, &_qkv, &_att, &_tmp, &_ff, &_glu, &_po,
        &_pq, &_rcos, &_rsin, &_ap, &_c_in, &_c_out, &_wdeq}) {
    b += s->byte_size();
  }
  return b;
}

// ---- load -----------------------------------------------------------------

MetalMusic3Dit::~MetalMusic3Dit() = default;

std::unique_ptr<MetalMusic3Dit>
MetalMusic3Dit::load(std::shared_ptr<WeightSet> dit,
                     std::shared_ptr<WeightSet> cond, MetalCompute* mc,
                     const Config& cfg, std::string* err)
{
  if (dit == nullptr || cond == nullptr || mc == nullptr) {
    if (err != nullptr) { *err = "no checkpoint or no metal device"; }
    return nullptr;
  }
  std::unique_ptr<MetalMusic3Dit> m(new MetalMusic3Dit());
  if (!m->init_(dit, cond, mc, cfg, err)) { return nullptr; }
  return m;
}

bool
MetalMusic3Dit::init_(const std::shared_ptr<WeightSet>& dit,
                      const std::shared_ptr<WeightSet>& cond,
                      MetalCompute* mc, const Config& cfg, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  _ws = dit;
  _cond_ws = cond;
  _mc = mc;
  _cfg = cfg;
  WeightSet& w = *dit;
  const int D = cfg.dim();
  const int F = cfg.ff_inner;

  // The int8 context first: a tier that holds the layout converts each
  // projection as it loads. It declines without matrix cores, and
  // VPIPE_I8_GEMM / VPIPE_I8_TIER override the config.
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_tier >= 0,
                                              /*bf16=*/false);
    i8->set_tier(cfg.i8_tier >= 2   ? I8GemmContext::Tier::kToken
                 : cfg.i8_tier == 1 ? I8GemmContext::Tier::kFast
                                    : I8GemmContext::Tier::kAccurate);
    i8->set_token_rotation(cfg.i8_tier == 2);
    if (i8->enabled()) {
      _i8 = std::move(i8);
      _i8_tier = _i8->tier() == I8GemmContext::Tier::kAccurate ? 0
                 : _i8->tier() == I8GemmContext::Tier::kFast   ? 1
                 : _i8->token_rotated()                        ? 2
                                                               : 3;
      // A held layout takes the int8 route at any window: a song's short
      // last one (as few as 696 rows) ran 2.1x faster there than
      // dequantized. Accurate keeps the floor -- it quantizes the weight
      // per call, and under 1024 rows that lost to f16 (353 ms against
      // 318, base M5).
      if (_i8_tier >= 1) { _i8->set_min_rows(1); }
    }
  }

  // A tensor narrowed to f16 as the checkpoint lays it out, kept.
  auto f16 = [&](WeightSet& ws, const std::string& nm,
                 std::size_t want) -> SharedBuffer {
    return ws.derived(std::string(kKey) + nm, [&]() -> SharedBuffer {
      return read_f16_(ws, mc, nm, want);
    });
  };
  // A block projection [N, K] that `build` makes in f16: kept so, or --
  // under a tier that holds the int8 layout -- converted once into it,
  // the f16 built only to be converted and then dropped.
  auto proj = [&](const std::string& key, int N, int K,
                  const std::function<SharedBuffer()>& build) -> Proj {
    Proj p;
    p.n = N;
    p.k = K;
    const int gc = _i8_tier >= 1 ? _i8->fast_group(K) : 0;
    if (gc == 0) {
      p.w = w.derived(std::string(kKey) + key, build);
      return p;
    }
    const std::size_t cb = (std::size_t)N * K;
    const std::size_t sb = I8GemmContext::fast_scale_bytes(N, K, gc);
    SharedBuffer both = w.derived(
        std::string(kKey) + _i8->layout_tag(gc) + "|" + key,
        [&]() -> SharedBuffer {
          SharedBuffer f = build();
          SharedBuffer out = mc->make_shared_buffer(cb + sb);
          if (f.empty() || out.empty()) { return {}; }
          CommandStream& st = stream_();
          bool ok = false;
          {
            ComputeEncoder e = st.begin_compute();
            ok = _i8->encode_quant_fast(e, f, out.subview(0, cb),
                                        out.subview(cb, sb), gc, N, K);
          }
          return ok && st.commit().wait_ok(nullptr) ? std::move(out)
                                                    : SharedBuffer{};
        });
    if (!both.empty()) {
      p.gc = gc;
      p.codes = both.subview(0, cb);
      p.scales = both.subview(cb, sb);
    }
    return p;
  };
  const std::size_t DD = (std::size_t)D * D;
  _blocks.resize((std::size_t)cfg.layers);
  for (int L = 0; L < cfg.layers; ++L) {
    const std::string p = "transformer_blocks." + std::to_string(L) + ".";
    const std::string ls = std::to_string(L);
    Block& b = _blocks[(std::size_t)L];
    b.ln1_w = f16(w, p + "norm1.weight", (std::size_t)D);
    b.ln1_b = f16(w, p + "norm1.bias", (std::size_t)D);
    b.ln2_w = f16(w, p + "norm2.weight", (std::size_t)D);
    b.ln2_b = f16(w, p + "norm2.bias", (std::size_t)D);
    b.out = proj(p + "attn.to_out.0.weight", D, D, [&]() {
      return read_f16_(w, mc, p + "attn.to_out.0.weight", DD);
    });
    b.ff_out = proj(p + "ff_out.weight", D, F, [&]() {
      return read_f16_(w, mc, p + "ff_out.weight", (std::size_t)D * F);
    });
    b.ff_out_b = f16(w, p + "ff_out.bias", (std::size_t)D);
    // q | k | v, one GEMM.
    b.qkv = proj("qkv|" + ls, 3 * D, D, [&]() -> SharedBuffer {
      SharedBuffer o = mc->make_shared_buffer(3 * DD * 2);
      if (o.empty()) { return {}; }
      auto* d = static_cast<_Float16*>(o.contents());
      const char* parts[3] = {"to_q", "to_k", "to_v"};
      for (int i = 0; i < 3; ++i) {
        Raw r;
        if (!read_raw_(w, mc, p + "attn." + parts[i] + ".weight", DD, &r)) {
          return {};
        }
        narrow_(r, d + (std::size_t)i * DD);
      }
      return o;
    });
    // ff_in's rows are [value (F) | gate (F)]; interleaved (gate f, value
    // f) here so one GEMM feeds swiglu_interleaved's silu(gate) * value.
    auto interleave = [&](const std::string& nm,
                          std::size_t row) -> SharedBuffer {
      Raw r;
      if (!read_raw_(w, mc, nm, 2 * (std::size_t)F * row, &r)) { return {}; }
      SharedBuffer o = mc->make_shared_buffer(r.n * 2);
      if (o.empty()) { return {}; }
      _Float16* d = static_cast<_Float16*>(o.contents());
      const Raw* rp = &r;
      const std::size_t FF = (std::size_t)F;
      dispatch_apply(FF, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,
                                                   0), ^(std::size_t f) {
        narrow_run_(*rp, (FF + f) * row, d + (2 * f) * row, row);
        narrow_run_(*rp, f * row, d + (2 * f + 1) * row, row);
      });
      return o;
    };
    b.ff_in = proj("ff_in_il|" + ls, 2 * F, D, [&]() {
      return interleave(p + "ff_in.weight", (std::size_t)D);
    });
    b.ff_in_b = w.derived(std::string(kKey) + "ff_in_b_il|" + ls, [&]() {
      return interleave(p + "ff_in.bias", 1);
    });
    if (b.ln1_w.empty() || b.ln1_b.empty() || b.ln2_w.empty() ||
        b.ln2_b.empty() || b.qkv.empty() || b.out.empty() ||
        b.ff_in.empty() || b.ff_in_b.empty() || b.ff_out.empty() ||
        b.ff_out_b.empty()) {
      return fail("transformer block " + std::to_string(L) +
                  " is incomplete");
    }
  }
  const std::size_t Cc = (std::size_t)cfg.concat();
  _pre = f16(w, "preprocess_conv.weight", Cc * Cc);
  _proj_in = f16(w, "proj_in.weight", (std::size_t)D * Cc);
  _proj_out = f16(w, "proj_out.weight", (std::size_t)cfg.in_channels * D);
  _post = f16(w, "postprocess_conv.weight",
              (std::size_t)cfg.in_channels * cfg.in_channels);
  if (_pre.empty() || _proj_in.empty() || _proj_out.empty() ||
      _post.empty()) {
    return fail("the transformer's input / output projections are "
                "incomplete");
  }
  // The time embedder runs on the host, in f32.
  if (!read_f32_(w, mc, "time_proj.weight", &_fourier,
                 (std::size_t)cfg.fourier_dim / 2) ||
      !read_f32_(w, mc, "time_embed.linear_1.weight", &_te1_w,
                 (std::size_t)D * cfg.fourier_dim) ||
      !read_f32_(w, mc, "time_embed.linear_1.bias", &_te1_b,
                 (std::size_t)D) ||
      !read_f32_(w, mc, "time_embed.linear_2.weight", &_te2_w, DD) ||
      !read_f32_(w, mc, "time_embed.linear_2.bias", &_te2_b,
                 (std::size_t)D)) {
    return fail("the time embedder is incomplete");
  }

  // ---- the condition encoder -------------------------------------------
  WeightSet& c = *cond;
  {
    std::vector<float> logits, scale;
    if (!read_f32_(c, mc, "layer_weight_logits", &logits,
                   (std::size_t)cfg.cond_layers) ||
        !read_f32_(c, mc, "layer_scale", &scale, 1)) {
      return fail("the condition encoder's layer mix is missing");
    }
    // softmax in f32, as torch.softmax(..., dim=0) of a float tensor.
    float mx = logits[0];
    for (float v : logits) { mx = std::max(mx, v); }
    float sum = 0.0f;
    _mix.resize(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
      _mix[i] = std::exp(logits[i] - mx);
      sum += _mix[i];
    }
    for (float& v : _mix) { v = v / sum; }
    _mix.push_back(scale[0]);           // the scale, last
  }
  const int H = cfg.cond_hidden;
  _cond_w = c.derived(std::string(kKey) + "cond.proj", [&]() -> SharedBuffer {
    Raw r;
    if (!read_raw_(c, mc, "proj.weight", (std::size_t)cfg.cond_dim * H * 3,
                   &r)) {
      return {};
    }
    SharedBuffer out = mc->make_shared_buffer(r.n * 2);
    if (out.empty()) { return {}; }
    // [out, in, tap] -> [out, tap * in + in]: the GEMM over an im2col row
    // [x(t-1) | x(t) | x(t+1)].
    _Float16* d = static_cast<_Float16*>(out.contents());
    const Raw* rp = &r;
    const std::size_t HH = (std::size_t)H;
    dispatch_apply((std::size_t)cfg.cond_dim,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(std::size_t o) {
      _Float16 v[3];
      for (std::size_t i = 0; i < HH; ++i) {
        narrow_run_(*rp, (o * HH + i) * 3, v, 3);
        for (std::size_t k = 0; k < 3; ++k) { d[(o * 3 + k) * HH + i] = v[k]; }
      }
    });
    return out;
  });
  _cond_b = f16(c, "proj.bias", (std::size_t)cfg.cond_dim);
  if (_cond_w.empty() || _cond_b.empty()) {
    return fail("the condition encoder's projection is missing");
  }

  // ---- kernels ------------------------------------------------------------
  _lib_gemm = mc->load_library("dense_gemm");
  _lib_elt  = mc->load_library("llm_elementwise");
  _lib_rope = mc->load_library("rope");
  _fn_gemm  = _lib_gemm.function("dense_gemm_t_f16");
  _fn_gemm_bm64 = _lib_gemm.function("dense_gemm_t_bm64_f16");
  _fn_gemm_bm64bn64 = _lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  _fn_bias_rows = _lib_elt.function("bias_add_rows_f16");
  _fn_ln   = _lib_elt.function("layer_norm_affine_f16");
  _fn_add  = _lib_elt.function("residual_add_f16");
  _fn_glu  = _lib_elt.function("swiglu_interleaved_f16");
  _fn_rope = _lib_rope.function("rope_half_part_ftab_inplace_f16");
  if (!_fn_gemm.valid() || !_fn_gemm_bm64.valid() ||
      !_fn_gemm_bm64bn64.valid() || !_fn_bias_rows.valid() ||
      !_fn_ln.valid() || !_fn_add.valid() || !_fn_glu.valid() ||
      !_fn_rope.valid()) {
    return fail("a MiniMax-Music3 DiT kernel did not resolve");
  }
  // Matrix cores: matmul2d GEMMs and the NAX attention (each with an A/B).
  if (mc->supports_matrix_cores()) {
    if (std::getenv("VPIPE_MUSIC3_NO_MMA2") == nullptr) {
      _lib_mma = mc->load_library("dense_gemm_mma");
      _fn_mma = _lib_mma.function("dense_gemm_mma_t_n128_f16");
      _fn_mma_deep = _lib_mma.function("dense_gemm_mma_t_n128x256_f16");
      _use_mma = _fn_mma.valid() && _fn_mma_deep.valid();
    }
    if (std::getenv("VPIPE_MUSIC3_NO_NAX_ATTN") == nullptr) {
      _lib_attn = mc->load_library("attn_steel_nax");
      _use_nax = _lib_attn.valid();
    }
  }
  if (!_use_nax) { _lib_attn = mc->load_library("attn_steel"); }
  if (!_lib_attn.valid()) { return fail("attn_steel did not load"); }
  _ap = mc->make_shared_buffer(sizeof(SteelAttnParams));
  if (_ap.empty()) { return fail("attention params allocation failed"); }
  return true;
}

// ---- scratch --------------------------------------------------------------

bool
MetalMusic3Dit::ensure_scratch_(int L)
{
  if (L <= _scratch_L) { return true; }
  const std::size_t M = 2 * ((std::size_t)L + 1);
  const std::size_t D = (std::size_t)_cfg.dim();
  const std::size_t Cc = (std::size_t)_cfg.concat();
  const std::size_t F = (std::size_t)_cfg.ff_inner;
  const std::size_t C = (std::size_t)_cfg.in_channels;
  auto mk = [&](std::size_t elts) { return _mc->make_shared_buffer(elts * 2); };
  _in = mk(2 * (std::size_t)L * Cc);
  _pre_o = mk(2 * (std::size_t)L * Cc);
  _x = mk(M * D);
  _n = mk(M * D);
  _qkv = mk(M * 3 * D);
  _att = mk(M * D);
  _tmp = mk(M * D);
  _ff = mk(M * 2 * F);
  _glu = mk(M * F);
  _po = mk(2 * (std::size_t)L * C);
  _pq = mk(2 * (std::size_t)L * C);
  const std::size_t half = (std::size_t)_cfg.rotary_dim / 2;
  _rcos = _mc->make_shared_buffer(M * half * 4);
  _rsin = _mc->make_shared_buffer(M * half * 4);
  for (const SharedBuffer* s : {&_in, &_pre_o, &_x, &_n, &_qkv, &_att,
                                &_tmp, &_ff, &_glu, &_po, &_pq, &_rcos,
                                &_rsin}) {
    if (s->empty()) {
      _scratch_L = 0;
      return false;
    }
  }
  _scratch_L = L;
  _attn_L = 0;
  return true;
}

// The rotary tables of a window: positions 0..L (the time token at 0)
// for each of the two guidance rows. f32, as the fp32 reference.
bool
MetalMusic3Dit::rope_table_(int L)
{
  const int half = _cfg.rotary_dim / 2;
  std::vector<float> inv((std::size_t)half);
  for (int i = 0; i < half; ++i) {
    const float e = (float)(2 * i) / (float)_cfg.rotary_dim;
    inv[(std::size_t)i] = 1.0f / std::pow(10000.0f, e);
  }
  auto* c = static_cast<float*>(_rcos.contents());
  auto* s = static_cast<float*>(_rsin.contents());
  const int S = L + 1;
  for (int b = 0; b < 2; ++b) {
    for (int t = 0; t < S; ++t) {
      for (int i = 0; i < half; ++i) {
        const float a = (float)t * inv[(std::size_t)i];
        const std::size_t o = ((std::size_t)b * S + t) * half + i;
        c[o] = std::cos(a);
        s[o] = std::sin(a);
      }
    }
  }
  // The attention's strides follow L too.
  auto* p = static_cast<SteelAttnParams*>(_ap.contents());
  std::memset(p, 0, sizeof(*p));
  const int D = _cfg.dim();
  const int Dh = _cfg.head_dim;
  const int bq = _use_nax ? 64 : 32, bk = _use_nax ? 32 : 16;
  p->B = 2; p->H = _cfg.heads; p->D = Dh; p->qL = S; p->kL = S;
  p->gqa_factor = 1;
  p->scale = 1.0f / std::sqrt((float)Dh);
  p->NQ = (S + bq - 1) / bq; p->NK = (S + bk - 1) / bk;
  p->NQ_aligned = S / bq; p->NK_aligned = S / bk;
  p->qL_rem = S - p->NQ_aligned * bq;
  p->kL_rem = S - p->NK_aligned * bk;
  p->qL_off = 0;
  const std::int64_t row = 3 * (std::int64_t)D;
  for (auto* st : {p->Q_strides, p->K_strides, p->V_strides}) {
    st[0] = (std::int64_t)S * row; st[1] = Dh; st[2] = row;
  }
  p->O_strides[0] = (std::int64_t)S * D;
  p->O_strides[1] = Dh;
  p->O_strides[2] = D;
  metal_compute::FunctionConstants fc;
  fc.set_bool(200, (S % bq) == 0).set_bool(201, (S % bk) == 0)
      .set_bool(300, false).set_bool(301, false).set_bool(302, false);
  _fn_attn = _lib_attn.function(_use_nax ? "attn_steel_nax_h_bd64"
                                         : "attn_steel_h_bd64", fc);
  if (!_fn_attn.valid()) { return false; }
  _attn_L = L;
  return true;
}

// ---- the time token -------------------------------------------------------

void
MetalMusic3Dit::time_embedding_(float t, std::vector<float>* out) const
{
  const int Fh = _cfg.fourier_dim / 2;
  const int D = _cfg.dim();
  // angles = 2 pi t w (the Python scalar 2 pi narrowed to f32 first).
  const float tp = (float)(2.0 * std::numbers::pi) * t;
  std::vector<float> f((std::size_t)2 * Fh);
  for (int i = 0; i < Fh; ++i) {
    const float a = tp * _fourier[(std::size_t)i];
    f[(std::size_t)i] = std::cos(a);
    f[(std::size_t)(Fh + i)] = std::sin(a);
  }
  const int Fd = 2 * Fh;
  std::vector<float> h((std::size_t)D);
  for (int o = 0; o < D; ++o) {
    double acc = 0.0;
    const float* wr = _te1_w.data() + (std::size_t)o * Fd;
    for (int i = 0; i < Fd; ++i) { acc += (double)wr[i] * f[(std::size_t)i]; }
    const float v = (float)acc + _te1_b[(std::size_t)o];
    h[(std::size_t)o] = v / (1.0f + std::exp(-v));
  }
  out->resize((std::size_t)D);
  for (int o = 0; o < D; ++o) {
    double acc = 0.0;
    const float* wr = _te2_w.data() + (std::size_t)o * D;
    for (int i = 0; i < D; ++i) { acc += (double)wr[i] * h[(std::size_t)i]; }
    (*out)[(std::size_t)o] = (float)acc + _te2_b[(std::size_t)o];
  }
}

// ---- GEMM -----------------------------------------------------------------

void
MetalMusic3Dit::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                      std::size_t xe, const SharedBuffer& w,
                      const SharedBuffer* bias, const SharedBuffer& y,
                      std::size_t ye, int M, int N, int K)
{
  _flops.gemm(M, N, K);
  dense_gemm_(enc, x, xe, w, bias, y, ye, M, N, K);
}

bool
MetalMusic3Dit::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                      std::size_t xe, const Proj& w, const SharedBuffer* bias,
                      const SharedBuffer& y, std::size_t ye, int M)
{
  const int N = w.n, K = w.k;
  // Counted once, where it is asked for, whatever route runs it.
  _flops.gemm(M, N, K);
  const SharedBuffer nob;
  if (w.gc != 0) {
    // A held layout: its tier's GEMM, else -- under the int8 floor -- the
    // layout dequantized into the f16 stand-in and the f16 route.
    if (_i8->gemm_fast(enc, x, xe, w.codes, w.scales, w.gc,
                       bias != nullptr ? *bias : nob, y, ye, M, N, K)) {
      ++_i8_gemms;
      return true;
    }
    // Made once, at the widest projection: grown shape by shape it held
    // two buffers while the smaller came back.
    if (_wdeq.empty()) {
      std::size_t widest = 0;
      for (const Block& k : _blocks) {
        for (const Proj* q : {&k.qkv, &k.out, &k.ff_in, &k.ff_out}) {
          widest = std::max(widest, (std::size_t)q->n * q->k);
        }
      }
      _wdeq = _mc->make_shared_buffer(widest * 2);
      if (_wdeq.empty()) { return false; }
    }
    if (!_i8->encode_dequant_fast(enc, w.codes, w.scales, w.gc, _wdeq, N,
                                  K)) {
      return false;
    }
    ++_i8_deq_gemms;
    dense_gemm_(enc, x, xe, _wdeq, bias, y, ye, M, N, K);
    return true;
  }
  // The accurate tier: the f16 weight quantized per call; a shape it
  // declines falls through with nothing encoded.
  if (_i8 && _i8->gemm(enc, x, xe, w.w, bias != nullptr ? *bias : nob, y,
                       ye, M, N, K)) {
    ++_i8_gemms;
    return true;
  }
  dense_gemm_(enc, x, xe, w.w, bias, y, ye, M, N, K);
  return true;
}

void
MetalMusic3Dit::dense_gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                            std::size_t xe, const SharedBuffer& w,
                            const SharedBuffer* bias, const SharedBuffer& y,
                            std::size_t ye, int M, int N, int K)
{
  if (_use_mma && M >= kMmaMinRows && N >= 16 && M <= mma_row_band(N, K)) {
    const bool wide = mma_use_wide_tile(N, K);
    const int RN = wide ? 256 : 128;
    enc.set_function(wide ? _fn_mma_deep : _fn_mma);
    enc.set_buffer(0, x, xe * 2);
    enc.set_buffer(1, w);
    enc.set_buffer(2, w);          // bias slot, unread (has_bias = 0)
    enc.set_buffer(3, y, ye * 2);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    if (bias != nullptr) {
      const std::uint32_t total = (std::uint32_t)((std::size_t)M * N);
      enc.set_function(_fn_bias_rows);
      enc.set_buffer(0, y, ye * 2);
      enc.set_buffer(1, *bias);
      enc.set_constant(2, N);
      enc.set_constant(3, total);
      enc.dispatch({total, 1, 1}, {256, 1, 1});
    }
    return;
  }
  // No matrix cores: the widest steel tile both extents fill.
  int bm = 32, bn = 32;
  const metal_compute::ComputeFunction* fn = &_fn_gemm;
  if (M >= 128) {
    bm = 64;
    if (N >= 64) {
      fn = &_fn_gemm_bm64bn64;
      bn = 64;
    } else {
      fn = &_fn_gemm_bm64;
    }
  }
  enc.set_function(*fn);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, w);
  enc.set_buffer(2, bias != nullptr ? *bias : w);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias != nullptr ? 1 : 0);
  enc.dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});
}

// ---- the condition --------------------------------------------------------

bool
MetalMusic3Dit::condition(const _Float16* hiddens, int frames,
                          std::vector<float>* cond, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (hiddens == nullptr || frames <= 0) { return fail("no hidden states"); }
  const int H = _cfg.cond_hidden;
  const int NL = _cfg.cond_layers;
  const int C = _cfg.cond_dim;
  const std::size_t row = (std::size_t)NL * H;
  if (frames > _cond_frames) {
    _c_in = _mc->make_shared_buffer((std::size_t)frames * 3 * H * 2);
    _c_out = _mc->make_shared_buffer((std::size_t)frames * C * 2);
    if (_c_in.empty() || _c_out.empty()) {
      _cond_frames = 0;
      return fail("condition scratch allocation failed");
    }
    _cond_frames = frames;
  }
  // The layer mix (f32): x[t] = scale * sum_l w_l h[t, l], then the conv's
  // im2col row [x(t-1) | x(t) | x(t+1)], zero past either end.
  std::vector<float> mixed((std::size_t)frames * H);
  const float scale = _mix.back();
  for (int t = 0; t < frames; ++t) {
    const _Float16* hr = hiddens + (std::size_t)t * row;
    float* mr = mixed.data() + (std::size_t)t * H;
    for (int i = 0; i < H; ++i) {
      float acc = 0.0f;
      for (int l = 0; l < NL; ++l) {
        acc += (float)hr[(std::size_t)l * H + i] * _mix[(std::size_t)l];
      }
      mr[i] = scale * acc;
    }
  }
  {
    auto* col = static_cast<_Float16*>(_c_in.contents());
    for (int t = 0; t < frames; ++t) {
      for (int k = 0; k < 3; ++k) {
        const int src = t + k - 1;
        _Float16* dst = col + ((std::size_t)t * 3 + k) * H;
        if (src < 0 || src >= frames) {
          std::memset(dst, 0, (std::size_t)H * 2);
          continue;
        }
        const float* m = mixed.data() + (std::size_t)src * H;
        for (int i = 0; i < H; ++i) { dst[i] = (_Float16)m[i]; }
      }
    }
  }
  _flops.begin();
  CommandStream& s = stream_();
  {
    ComputeEncoder enc = s.begin_compute();
    gemm_(enc, _c_in, 0, _cond_w, &_cond_b, _c_out, 0, frames, C, 3 * H);
  }
  _flops.flush(s);
  std::string gerr;
  if (!s.commit().wait_ok(&gerr)) {
    return fail(gerr.empty() ? "condition encoder failed" : gerr);
  }
  // Nearest-neighbour from the frame rate to the latent rate.
  const int L = music3::latent_length(frames);
  const std::vector<int> idx = music3::nearest_indices(frames, L);
  cond->resize((std::size_t)L * C);
  const auto* o = static_cast<const _Float16*>(_c_out.contents());
  for (int j = 0; j < L; ++j) {
    const _Float16* src = o + (std::size_t)idx[(std::size_t)j] * C;
    float* dst = cond->data() + (std::size_t)j * C;
    for (int i = 0; i < C; ++i) { dst[i] = (float)src[i]; }
  }
  return true;
}

// ---- one evaluation, both guidance rows -----------------------------------

bool
MetalMusic3Dit::forward_(const std::vector<float>& x, float t, int L,
                         std::vector<float>* vc, std::vector<float>* vu,
                         std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int D = _cfg.dim();
  const int Cc = _cfg.concat();
  const int Ci = _cfg.in_channels;
  const int F = _cfg.ff_inner;
  const int S = L + 1;
  const int M = 2 * S;
  // The state into both rows' [x | 0 | condition] inputs (the condition
  // columns were written when the window began).
  {
    auto* in = static_cast<_Float16*>(_in.contents());
    for (int b = 0; b < 2; ++b) {
      for (int j = 0; j < L; ++j) {
        _Float16* r = in + ((std::size_t)b * L + j) * Cc;
        const float* xr = x.data() + (std::size_t)j * Ci;
        for (int c = 0; c < Ci; ++c) { r[c] = (_Float16)xr[c]; }
      }
    }
  }
  // The time token, cached per flow time (a song's steps reuse the grid).
  const std::vector<float>* temb = nullptr;
  for (std::size_t i = 0; i < _temb_t.size(); ++i) {
    if (_temb_t[i] == t) {
      temb = &_temb[i];
      break;
    }
  }
  if (temb == nullptr) {
    _temb_t.push_back(t);
    _temb.emplace_back();
    time_embedding_(t, &_temb.back());
    temb = &_temb.back();
  }
  {
    auto* xs = static_cast<_Float16*>(_x.contents());
    for (int b = 0; b < 2; ++b) {
      _Float16* r = xs + (std::size_t)b * S * D;
      for (int i = 0; i < D; ++i) { r[i] = (_Float16)(*temb)[(std::size_t)i]; }
    }
  }

  const int Dh = _cfg.head_dim;
  const int Hh = _cfg.heads;
  const int bq = _use_nax ? 64 : 32;
  const float eps = _cfg.ln_eps;
  auto add = [&](ComputeEncoder& enc, const SharedBuffer& a,
                 const SharedBuffer& b, int n) {
    enc.set_function(_fn_add);
    enc.set_buffer(0, a);
    enc.set_buffer(1, b);
    enc.set_buffer(2, a);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  auto ln = [&](ComputeEncoder& enc, const SharedBuffer& w,
                const SharedBuffer& b) {
    enc.set_function(_fn_ln);
    enc.set_buffer(0, _x);
    enc.set_buffer(1, w);
    enc.set_buffer(2, b);
    enc.set_buffer(3, _n);
    enc.set_constant(4, D);
    enc.set_constant(5, eps);
    enc.dispatch({256, (unsigned)M, 1}, {256, 1, 1});
  };
  auto rope = [&](ComputeEncoder& enc, int off) {
    enc.set_function(_fn_rope);
    enc.set_buffer(0, _qkv);
    enc.set_buffer(1, _rcos);
    enc.set_buffer(2, _rsin);
    enc.set_constant(3, Hh);
    enc.set_constant(4, M);
    enc.set_constant(5, Dh);
    enc.set_constant(6, _cfg.rotary_dim);
    enc.set_constant(7, 3 * D);
    enc.set_constant(8, off);
    enc.set_constant(9, Dh);
    enc.dispatch({(unsigned)Dh, (unsigned)M, (unsigned)Hh},
                 {(unsigned)Dh, 1, 1});
  };

  // The accurate tier's split tuner runs its own streams: no encoder open.
  if (_i8) { _i8->tune_pending(_mc); }
  _flops.begin();
  CommandStream& s = stream_();
  bool routed = true;
  {
    ComputeEncoder enc = s.begin_compute();
    // [x | 0 | c] + conv1x1([x | 0 | c]), then the projection into the
    // residual stream behind each row's time token.
    gemm_(enc, _in, 0, _pre, nullptr, _pre_o, 0, 2 * L, Cc, Cc);
    add(enc, _pre_o, _in, 2 * L * Cc);
    for (int b = 0; b < 2; ++b) {
      gemm_(enc, _pre_o, (std::size_t)b * L * Cc, _proj_in, nullptr, _x,
            ((std::size_t)b * S + 1) * D, L, D, Cc);
    }
    for (const Block& blk : _blocks) {
      ln(enc, blk.ln1_w, blk.ln1_b);
      routed &= gemm_(enc, _n, 0, blk.qkv, nullptr, _qkv, 0, M);
      rope(enc, 0);
      rope(enc, D);
      enc.set_function(_fn_attn);
      enc.set_buffer(0, _qkv, 0);
      enc.set_buffer(1, _qkv, (std::size_t)D * 2);
      enc.set_buffer(2, _qkv, (std::size_t)2 * D * 2);
      enc.set_buffer(3, _att, 0);
      enc.set_buffer(4, _ap);
      enc.dispatch({32u * (unsigned)((S + bq - 1) / bq), 4u * (unsigned)Hh,
                    2u}, {32, 4, 1});
      _flops.attention(2LL * Hh, (long long)S * S, Dh);
      routed &= gemm_(enc, _att, 0, blk.out, nullptr, _tmp, 0, M);
      add(enc, _x, _tmp, M * D);
      ln(enc, blk.ln2_w, blk.ln2_b);
      routed &= gemm_(enc, _n, 0, blk.ff_in, &blk.ff_in_b, _ff, 0, M);
      enc.set_function(_fn_glu);
      enc.set_buffer(0, _ff);
      enc.set_buffer(1, _glu);
      enc.set_constant(2, M);
      enc.set_constant(3, F);
      enc.dispatch({(unsigned)((std::size_t)M * F), 1, 1}, {256, 1, 1});
      routed &= gemm_(enc, _glu, 0, blk.ff_out, &blk.ff_out_b, _tmp, 0, M);
      add(enc, _x, _tmp, M * D);
    }
    // Drop the time token; project out and through the residual 1x1.
    for (int b = 0; b < 2; ++b) {
      gemm_(enc, _x, ((std::size_t)b * S + 1) * D, _proj_out, nullptr, _po,
            (std::size_t)b * L * Ci, L, Ci, D);
    }
    gemm_(enc, _po, 0, _post, nullptr, _pq, 0, 2 * L, Ci, Ci);
    add(enc, _pq, _po, 2 * L * Ci);
  }
  _flops.flush(s);
  std::string gerr;
  if (!s.commit().wait_ok(&gerr) || !routed) {
    return fail(!gerr.empty() ? gerr
                : routed      ? "DiT forward failed"
                              : "the int8 stand-in could not be had");
  }
  const auto* o = static_cast<const _Float16*>(_pq.contents());
  const std::size_t n = (std::size_t)L * Ci;
  vc->resize(n);
  vu->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    (*vc)[i] = (float)o[i];
    (*vu)[i] = (float)o[n + i];
  }
  return true;
}

bool
MetalMusic3Dit::velocity(const std::vector<float>& x, float t,
                         const std::vector<float>& cond,
                         std::vector<float>* vc, std::vector<float>* vu,
                         std::string* err)
{
  const int Ci = _cfg.in_channels;
  const int L = (int)(x.size() / (std::size_t)Ci);
  if (L <= 0 || x.size() != (std::size_t)L * Ci) {
    if (err != nullptr) { *err = "a state of the wrong size"; }
    return false;
  }
  return begin_window_(cond, L, err) && forward_(x, t, L, vc, vu, err);
}

bool
MetalMusic3Dit::begin_window_(const std::vector<float>& cond, int L,
                              std::string* err)
{
  const int Ci = _cfg.in_channels;
  const int C = _cfg.cond_dim;
  if (L <= 0 || cond.size() != (std::size_t)L * C) {
    if (err != nullptr) { *err = "state and condition disagree"; }
    return false;
  }
  if (!ensure_scratch_(L)) {
    if (err != nullptr) { *err = "DiT scratch allocation failed"; }
    return false;
  }
  if (_attn_L != L && !rope_table_(L)) {
    if (err != nullptr) { *err = "the attention did not specialise"; }
    return false;
  }
  // Condition columns: row 0 the window's, row 1 zeros.
  const int Cc = _cfg.concat();
  auto* in = static_cast<_Float16*>(_in.contents());
  for (int b = 0; b < 2; ++b) {
    for (int j = 0; j < L; ++j) {
      _Float16* r = in + ((std::size_t)b * L + j) * Cc;
      std::memset(r + Ci, 0, (std::size_t)Ci * 2);
      if (b == 0) {
        const float* cr = cond.data() + (std::size_t)j * C;
        for (int i = 0; i < C; ++i) { r[2 * Ci + i] = (_Float16)cr[i]; }
      } else {
        std::memset(r + 2 * Ci, 0, (std::size_t)C * 2);
      }
    }
  }
  return true;
}

CommandStream&
MetalMusic3Dit::stream_()
{
  static const bool fresh =
      std::getenv("VPIPE_MUSIC3_FRESH_STREAMS") != nullptr;
  if (fresh || !_stream.valid()) { _stream = _mc->make_command_stream(); }
  return _stream;
}

// ---- a song ---------------------------------------------------------------

bool
MetalMusic3Dit::synthesize(const _Float16* hiddens, int frames,
                           std::uint64_t seed, int steps, float cfg,
                           std::vector<float>* latents,
                           const ProgressFn& progress, std::string* err,
                           const std::vector<std::vector<float>>* noise)
{
  using clk = std::chrono::steady_clock;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (hiddens == nullptr || frames <= 0) { return fail("no hidden states"); }
  if (steps < 1) { return fail("steps must be >= 1"); }
  const int Ci = _cfg.in_channels;
  const int Cd = _cfg.cond_dim;
  const std::size_t row = (std::size_t)_cfg.cond_layers * _cfg.cond_hidden;
  const std::vector<int> starts = music3::window_starts(frames);
  const int nw = (int)starts.size();
  if (noise != nullptr && (int)noise->size() < nw) {
    return fail("fewer noise tensors than windows");
  }
  const std::vector<float> ts = music3::flow_times(steps);
  _stats = Stats{};
  _stats.windows = nw;
  const int total = nw * steps;
  TorchCpuRandn gen(seed);

  latents->assign((std::size_t)Ci * music3::stitched_latents(frames), 0.0f);
  const int T_all = (int)(latents->size() / (std::size_t)Ci);
  int out_t = 0;
  // The previous window's carry: its latents and condition over
  // [L - 344, L - 172), time-major.
  std::vector<float> prev_lat, prev_cond;
  int prev_len = 0;
  std::vector<float> cond, x, vc, vu, nprompt;
  for (int k = 0; k < nw; ++k) {
    const int s0 = starts[(std::size_t)k];
    const int s1 = std::min(s0 + music3::kChunkFrames, frames);
    if (!condition(hiddens + (std::size_t)s0 * row, s1 - s0, &cond, err)) {
      return false;
    }
    const int L = (int)(cond.size() / (std::size_t)Cd);
    int ov = 0;
    if (prev_len > 0) {
      ov = std::min(prev_len, L);
      std::memcpy(cond.data(), prev_cond.data(),
                  (std::size_t)ov * Cd * sizeof(float));
    }
    // The window's noise, channel-major [128, L] -> time-major.
    std::vector<float> z;
    if (noise != nullptr) {
      z = (*noise)[(std::size_t)k];
    } else {
      z = gen.randn((std::size_t)Ci * L);
    }
    if (z.size() != (std::size_t)Ci * L) {
      return fail("noise of the wrong size");
    }
    x.resize((std::size_t)L * Ci);
    for (int c = 0; c < Ci; ++c) {
      for (int j = 0; j < L; ++j) {
        x[(std::size_t)j * Ci + c] = z[(std::size_t)c * L + j];
      }
    }
    nprompt.assign(x.begin(), x.begin() + (std::ptrdiff_t)ov * Ci);
    if (!begin_window_(cond, L, err)) { return false; }
    for (int i = 0; i < steps; ++i) {
      const float t = ts[(std::size_t)i];
      if (ov > 0) {
        // (1 - (1 - 1e-6) t) * noise + t * previous, in f32.
        const float a = 1.0f - 0.999999f * t;
        for (std::size_t e = 0; e < (std::size_t)ov * Ci; ++e) {
          x[e] = a * nprompt[e] + t * prev_lat[e];
        }
      }
      const auto t0 = clk::now();
      {
        // The denoise on the LLM lane, a span an evaluation.
        PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmDit,
                           kPerfLlmDitBegin, (std::uint64_t)(2 * (L + 1)));
        if (!forward_(x, t, L, &vc, &vu, err)) { return false; }
      }
      _stats.ms += std::chrono::duration<double, std::milli>(clk::now() - t0)
                       .count();
      ++_stats.evals;
      const float dt = ts[(std::size_t)i + 1] - t;
      for (std::size_t e = 0; e < x.size(); ++e) {
        const float v = vu[e] + cfg * (vc[e] - vu[e]);
        x[e] = x[e] + dt * v;
      }
      if (progress && !progress(k * steps + i + 1, total)) {
        if (err != nullptr) { err->clear(); }
        return false;
      }
    }
    if (ov > 0) {
      std::memcpy(x.data(), prev_lat.data(),
                  (std::size_t)ov * Ci * sizeof(float));
    }
    // The carry for the next window.
    const int o0 = std::max(0, L - 2 * music3::kOverlapLatents);
    const int o1 = std::max(o0, L - music3::kOverlapLatents);
    prev_len = o1 - o0;
    prev_lat.assign(x.begin() + (std::ptrdiff_t)o0 * Ci,
                    x.begin() + (std::ptrdiff_t)o1 * Ci);
    prev_cond.assign(cond.begin() + (std::ptrdiff_t)o0 * Cd,
                     cond.begin() + (std::ptrdiff_t)o1 * Cd);
    // This window's share of the stitched song, channel-major.
    const auto kept = music3::kept_latents(k, nw, L);
    for (int j = kept.first; j < kept.second; ++j, ++out_t) {
      if (out_t >= T_all) { return fail("stitched latents overflow"); }
      for (int c = 0; c < Ci; ++c) {
        (*latents)[(std::size_t)c * T_all + out_t] =
            x[(std::size_t)j * Ci + c];
      }
    }
  }
  if (out_t != T_all) { return fail("stitched latents underfilled"); }
  return true;
}

}  // namespace vpipe::genai
