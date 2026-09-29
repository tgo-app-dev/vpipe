#include "generative-models/shared/ane-tier.h"

#include "generative-models/shared/accel-settings.h"
#include "generative-models/shared/ane-ffn.h"

#include <algorithm>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace ane {

// The bag's readers work on any FlexData object, and reading a spec through
// them gives it the bag's semantics: an absent key is the default.
namespace acc = vpipe::genai::accel;

namespace {

enum class Shape { kSwiglu, kGelu, kMatmul };

bool
shape_of(const FlexData& spec, Shape* out, std::string* why)
{
  const std::string kind = acc::text(&spec, kKind);
  if (kind == kKindFfn) {
    const std::string act =
        acc::text(&spec, kActivation, std::string(kActSwiglu));
    if (act == kActSwiglu) { *out = Shape::kSwiglu; return true; }
    if (act == kActGeluTanh) { *out = Shape::kGelu; return true; }
    *why = "unknown ffn activation '" + act + "'";
    return false;
  }
  if (kind == kKindMatmul) { *out = Shape::kMatmul; return true; }
  *why = kind.empty() ? std::string("the spec names no kind")
                      : "unknown kind '" + kind + "'";
  return false;
}

int
chunk_of(const FlexData& spec)
{
  const long long c =
      acc::integer(&spec, kChunk, AneFeedForward::kChunkDefault);
  return (c >= 256 && c <= 16384) ? (int)c : AneFeedForward::kChunkDefault;
}

int
int_of(const FlexData& f, std::string_view k, long long dflt = 0)
{
  return (int)acc::integer(&f, k, dflt);
}

std::vector<std::string_view>
projections(Shape s)
{
  switch (s) {
  case Shape::kSwiglu: return {kProjGate, kProjUp, kProjDown};
  case Shape::kGelu:   return {kProjUp, kProjDown};
  case Shape::kMatmul: return {kProjW};
  }
  return {};
}

// Every binding name a projection may carry.
constexpr std::string_view kFields[] = {
    "w", "codes", "scales", "qbias", "bias",
    "lora0.a", "lora0.b", "lora1.a", "lora1.b",
};

}  // namespace

struct Tier::Impl {
  std::unique_ptr<AneFeedForward> ff;
  Shape       shape = Shape::kSwiglu;
  std::string kind, activation;
  int         seq = 0;           // the current begin()'s rows
  std::string error;
};

std::size_t
runtime_bytes(const FlexData& spec)
{
  Shape s;
  std::string why;
  if (!shape_of(spec, &s, &why)) { return 0; }
  const int c = chunk_of(spec), seq = int_of(spec, kSeq);
  switch (s) {
  case Shape::kSwiglu:
    return AneFeedForward::runtime_bytes(int_of(spec, kHidden),
                                         int_of(spec, kInner), seq, c);
  case Shape::kGelu:
    return AneFeedForward::gelu_runtime_bytes(int_of(spec, kHidden),
                                              int_of(spec, kInner), c);
  case Shape::kMatmul:
    return AneFeedForward::matmul_runtime_bytes(
        int_of(spec, kInWidth), int_of(spec, kOutWidth), seq, c);
  }
  return 0;
}

std::unique_ptr<Tier>
create(const SessionContextIntf* session, metal_compute::MetalCompute* mc,
       const FlexData& spec, std::string* why)
{
  auto refuse = [&](std::string m) {
    if (why != nullptr) { *why = std::move(m); }
    return std::unique_ptr<Tier>();
  };
  Shape s;
  std::string w;
  if (!shape_of(spec, &s, &w)) { return refuse(w); }
  if (session == nullptr || mc == nullptr) {
    return refuse("no session or Metal device to run it on");
  }
  AneFeedForward::Options o;
  o.session = session;
  o.mc      = mc;
  o.tag     = acc::text(&spec, kTag, "ane");
  o.seq     = int_of(spec, kSeq);
  o.rows    = (float)acc::real(&spec, kRows, 0.0);
  o.chunk   = chunk_of(spec);
  o.tile    = int_of(spec, kTile, AneFeedForward::kTile);
  o.profile = acc::flag(&spec, kProfile);
  if (s == Shape::kMatmul) {
    o.matmul = true;
    o.hidden = int_of(spec, kInWidth);
    o.ffn    = int_of(spec, kOutWidth);
  } else {
    o.hidden = int_of(spec, kHidden);
    o.ffn    = int_of(spec, kInner);
    o.biases = acc::flag(&spec, kBiases);
    o.gelu   = s == Shape::kGelu;
  }
  if (o.hidden <= 0 || o.ffn <= 0 || o.seq <= 0 || o.tile <= 0) {
    return refuse("the spec needs positive widths, seq and tile");
  }
  std::unique_ptr<AneFeedForward> ff = AneFeedForward::create(o);
  if (ff == nullptr) {
    return refuse("the ANE module was not armed (the session log says why)");
  }
  auto* p       = new Tier::Impl();
  p->ff         = std::move(ff);
  p->shape      = s;
  p->kind       = acc::text(&spec, kKind);
  p->activation = s == Shape::kMatmul
                      ? std::string()
                      : acc::text(&spec, kActivation,
                                  std::string(kActSwiglu));
  return std::unique_ptr<Tier>(new Tier(p));
}

Tier::Tier(Impl* p) noexcept : _p(p) {}

// Destroying the feed-forward joins its worker, so no job outlives the
// buffers it was handed.
Tier::~Tier() { delete _p; }

Tier::Plan
Tier::plan_block()
{
  switch (_p->ff->plan_block()) {
  case AneFeedForward::Plan::kSplit: return Plan::kSplit;
  case AneFeedForward::Plan::kProbe: return Plan::kProbe;
  case AneFeedForward::Plan::kGpu:   return Plan::kGpu;
  }
  return Plan::kGpu;
}

bool
Tier::needs_barrier() const
{
  return _p->ff->needs_barrier();
}

bool
Tier::stage(int layer, const FlexData& params,
            std::span<const Binding> buffers)
{
  _p->error.clear();
  auto fail = [&](std::string m) {
    _p->error = std::move(m);
    return false;
  };
  const std::vector<std::string_view> projs = projections(_p->shape);
  // Every name must be one this kind reads.
  for (const Binding& b : buffers) {
    bool known = false;
    for (std::string_view pr : projs) {
      for (std::string_view f : kFields) {
        if (b.name == key(pr, f)) { known = true; break; }
      }
      if (known) { break; }
    }
    if (!known) {
      return fail("unknown binding '" + std::string(b.name) + "'");
    }
  }
  auto find = [&](const std::string& n) -> const metal_compute::SharedBuffer* {
    for (const Binding& b : buffers) {
      if (b.name == n) { return b.buf; }
    }
    return nullptr;
  };

  std::vector<AneFfnSource> src;
  src.reserve(projs.size());
  for (std::string_view pr : projs) {
    AneFfnSource s;
    s.w         = find(key(pr, "w"));
    s.codes     = find(key(pr, "codes"));
    s.scales    = find(key(pr, "scales"));
    s.qbias     = find(key(pr, "qbias"));
    s.bias      = find(key(pr, "bias"));
    s.bits      = int_of(params, key(pr, "bits"));
    s.quantized = s.bits > 0;
    s.stride = (std::size_t)std::max(1, int_of(params, key(pr, "stride"), 1));
    s.offset = (std::size_t)std::max(0, int_of(params, key(pr, "offset")));
    for (int i = 0; i < AneFfnSource::kMaxDeltas; ++i) {
      const std::string lp = key(pr, "lora" + std::to_string(i));
      const auto* la = find(lp + ".a");
      const auto* lb = find(lp + ".b");
      if (la == nullptr && lb == nullptr) { continue; }
      const int rank = int_of(params, lp + ".rank");
      if (la == nullptr || lb == nullptr || rank <= 0) {
        return fail("adapter '" + lp + "' needs .a, .b and a positive "
                    ".rank");
      }
      AneFfnSource::Delta& d = s.delta[s.deltas++];
      d.a        = la;
      d.b        = lb;
      d.rank     = rank;
      d.scale    = (float)acc::real(&params, lp + ".scale", 1.0);
      d.b_stride =
          (std::size_t)std::max(1, int_of(params, lp + ".b_stride", 1));
      d.b_offset =
          (std::size_t)std::max(0, int_of(params, lp + ".b_offset"));
    }
    if (!s.stageable()) {
      return fail("projection '" + std::string(pr) +
                  "' binds neither a dense .w nor .codes/.scales/.qbias "
                  "with .bits 4 or 8");
    }
    src.push_back(s);
  }
  _p->ff->stage(layer, src, int_of(params, kGroup));
  return true;
}

bool
Tier::join_stage(int layer)
{
  return _p->ff->join_stage(layer);
}

int
Tier::begin(const FlexData& args, std::span<const Binding> io)
{
  _p->error.clear();
  const metal_compute::SharedBuffer* in = nullptr;
  const metal_compute::SharedBuffer* out = nullptr;
  for (const Binding& b : io) {
    if (b.name == kIoIn) {
      in = b.buf;
    } else if (b.name == kIoOut) {
      out = b.buf;
    } else {
      _p->error = "unknown io binding '" + std::string(b.name) + "'";
      return 0;
    }
  }
  const int seq = int_of(args, kSeq);
  if (in == nullptr || out == nullptr || seq <= 0) {
    _p->error = "begin needs 'in', 'out' and a positive seq";
    return 0;
  }
  _p->seq = seq;
  return _p->ff->begin(*in, *out, seq);
}

bool
Tier::finish(int layer, const FlexData& timing)
{
  return _p->ff->finish(layer, _p->seq, acc::real(&timing, kGpuMs, 0.0),
                        acc::real(&timing, kDrainMs, 0.0));
}

void
Tier::note_probe(const FlexData& timing)
{
  _p->ff->note_probe(acc::real(&timing, kDrainMs, 0.0),
                     acc::real(&timing, kGpuMs, 0.0));
}

void
Tier::join()
{
  _p->ff->join();
}

FlexData
Tier::info() const
{
  FlexData f = FlexData::make_object();
  acc::set_text(&f, kKind, _p->kind);
  if (!_p->activation.empty()) {
    acc::set_text(&f, kActivation, _p->activation);
  }
  acc::set_integer(&f, kChunk, _p->ff->chunk());
  acc::set_integer(&f, kInfoRowsPerBlock, _p->ff->rows_per_block());
  acc::set_flag(&f, kInfoTiming, _p->ff->timing());
  acc::set_integer(&f, kInfoHostBytes, (long long)_p->ff->host_bytes());
  acc::set_text(&f, kInfoError, _p->error);
  acc::set_integer(&f, kInfoLostRow0, _p->ff->lost_row0());
  acc::set_flag(&f, kInfoGpuOnly, _p->ff->gpu_only());
  return f;
}

}  // namespace ane
}  // namespace genai
}  // namespace vpipe
