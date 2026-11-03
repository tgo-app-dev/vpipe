#include "stages/load-tensor-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

constexpr char          kMagic[8]  = {'V','P','T','E','N','S','O','R'};
constexpr std::uint32_t kVersion   = 1;
constexpr std::uint32_t kMaxRank   = 8;

const PortSpec kOports[] = {
  {.name = "tensor", .doc = "the tensor the file carries, sliced when "
                            "`count` asks for a window",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
};

constexpr ConfigKey kAttrs[] = {
  {.key = "path", .type = ConfigType::String, .required = true,
   .doc = "the file to read, as written by save-tensor",
   .is_path = true},
  {.key = "axis", .type = ConfigType::Int, .required = false,
   .doc = "which axis `start`/`count` slice. 0 is the outermost. A "
          "MiniMax-H3 video latent is [z, T, H, W], so its TIME axis is "
          "1; its audio latent is [2, channels, latents] (generate-video's "
          "audio_latent oport), so time is 2 there", .def_int = 0},
  {.key = "start", .type = ConfigType::Int, .required = false,
   .doc = "first index on `axis`. NEGATIVE counts from the end, so -5 is "
          "\"the last five\" without knowing the extent -- the spelling "
          "temporal-slice uses for the same idea", .def_int = 0},
  {.key = "count", .type = ConfigType::Int, .required = false,
   .doc = "how many to take; 0 (the default) means to the end. A window "
          "that runs off the end is REFUSED rather than clamped: a "
          "shorter window is a different request, and quietly honouring "
          "it turns into a quality question no one can trace",
   .def_int = 0},
  {.key = "sideband", .type = ConfigType::Any, .required = false,
   .doc = "a JSON object merged into the emitted beat's sideband, over "
          "whatever the file carried. This is where a graph says what "
          "the tensor IS to its consumer -- MiniMax-H3's reference port "
          "reads `latent: true` to mean \"already encoded, pack these "
          "rows rather than running the VAE over them\""},
};

const StageSpec kSpec = {
  .type_name = "load-tensor",
  .doc       = "Source: reads a tensor written by save-tensor and emits "
               "it as one TensorBeat, optionally a window of it. The "
               "companion of save-tensor, and the way a latent outlives "
               "the run that made it.",
  .display_name = "Load Tensor",
  .category  = StageCategory::Generic,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};

std::size_t
dtype_bytes_(TensorBeat::DType d)
{
  switch (d) {
  case TensorBeat::DType::U8:
  case TensorBeat::DType::I8:   return 1;
  case TensorBeat::DType::Bf16:
  case TensorBeat::DType::F16:  return 2;
  case TensorBeat::DType::F32:  return 4;
  }
  return 0;
}

}  // namespace

bool
LoadTensorStage::resolve_window(std::int64_t extent, std::int64_t start,
                                std::int64_t count, std::int64_t* out_start,
                                std::int64_t* out_count, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (extent <= 0) { return fail("the axis is empty"); }
  std::int64_t s = start < 0 ? extent + start : start;
  if (s < 0 || s >= extent) {
    return fail(fmt("start {} is outside an axis of {}", start, extent)());
  }
  std::int64_t c = count <= 0 ? extent - s : count;
  if (s + c > extent) {
    return fail(fmt("the window [{}, {}) runs past an axis of {}",
                    s, s + c, extent)());
  }
  *out_start = s;
  *out_count = c;
  return true;
}

LoadTensorStage::LoadTensorStage(const SessionContextIntf* session,
                                 string                    id,
                                 vector<InEdge>            iports,
                                 FlexData                  config)
  : TypedStage<LoadTensorStage>(session, std::move(id), std::move(iports),
                                std::move(config))
{
  _path  = attr_path("path", /*for_write=*/false);
  _axis  = attr_int("axis");
  _start = attr_int("start");
  _count = attr_int("count");
  _extra_sideband = attr("sideband");
  if (_path.empty()) {
    fail_config(fmt("LoadTensorStage('{}'): config.path is required",
                    this->id()));
  }
  if (_axis < 0) {
    fail_config(fmt("LoadTensorStage('{}'): axis must be >= 0, got {}",
                    this->id(), _axis));
  }
  allocate_oports(1);
}

const StageSpec&
LoadTensorStage::spec() const noexcept
{
  return kSpec;
}

Job
LoadTensorStage::process(RuntimeContext& ctx)
{
  if (_emitted) { ctx.signal_done(); co_return; }
  _emitted = true;

  std::FILE* f = std::fopen(_path.c_str(), "rb");
  if (f == nullptr) {
    session()->error(fmt("LoadTensorStage('{}'): cannot open '{}'",
                         this->id(), _path));
    ctx.signal_done();
    co_return;
  }
  auto bail = [&](string why) {
    std::fclose(f);
    session()->error(fmt("LoadTensorStage('{}'): '{}': {}", this->id(),
                         _path, why));
    ctx.signal_done();
  };

  char magic[8] = {};
  if (std::fread(magic, 1, 8, f) != 8
      || std::memcmp(magic, kMagic, 8) != 0) {
    bail("not a save-tensor file (bad magic)");
    co_return;
  }
  std::uint32_t ver = 0, dt = 0, rank = 0, reserved = 0;
  std::uint64_t count = 0;
  bool ok = std::fread(&ver, 4, 1, f) == 1
         && std::fread(&dt, 4, 1, f) == 1
         && std::fread(&rank, 4, 1, f) == 1
         && std::fread(&reserved, 4, 1, f) == 1
         && std::fread(&count, 8, 1, f) == 1;
  if (!ok) { bail("truncated header"); co_return; }
  if (ver != kVersion) {
    bail(fmt("version {}, this build reads {}", ver, kVersion)());
    co_return;
  }
  if (rank == 0 || rank > kMaxRank) {
    bail(fmt("rank {} is outside 1..{}", rank, kMaxRank)());
    co_return;
  }
  if (dt > (std::uint32_t)TensorBeat::DType::F16) {
    bail(fmt("dtype {} is not one this build knows", dt)());
    co_return;
  }
  std::vector<std::int64_t> dims((std::size_t)rank, 0);
  for (std::uint32_t i = 0; i < rank; ++i) {
    if (std::fread(&dims[i], 8, 1, f) != 1) {
      bail("truncated shape"); co_return;
    }
    if (dims[i] <= 0) { bail("a non-positive extent"); co_return; }
  }
  const auto dtype = (TensorBeat::DType)dt;
  const std::size_t esz = dtype_bytes_(dtype);
  std::size_t want = esz;
  for (auto d : dims) { want *= (std::size_t)d; }
  if ((std::uint64_t)(want / (esz ? esz : 1)) != count) {
    bail("the shape and the element count disagree"); co_return;
  }

  std::vector<std::uint8_t> raw(want);
  if (want > 0 && std::fread(raw.data(), 1, want, f) != want) {
    bail("truncated payload"); co_return;
  }
  // Whatever follows the payload is the sideband JSON save-tensor wrote.
  std::string js;
  {
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) {
      js.append(buf, n);
    }
  }
  std::fclose(f);

  if ((std::size_t)_axis >= dims.size()) {
    session()->error(fmt(
        "LoadTensorStage('{}'): axis {} but '{}' has rank {}",
        this->id(), _axis, _path, dims.size()));
    ctx.signal_done();
    co_return;
  }

  std::int64_t s0 = 0, n0 = 0;
  std::string werr;
  if (!resolve_window(dims[(std::size_t)_axis], _start, _count, &s0, &n0,
                      &werr)) {
    session()->error(fmt("LoadTensorStage('{}'): '{}' axis {}: {}",
                         this->id(), _path, _axis, werr));
    ctx.signal_done();
    co_return;
  }

  TensorBeat tb;
  tb.dtype = dtype;
  tb.shape = dims;
  tb.shape[(std::size_t)_axis] = n0;

  // Copy the window out. The axis splits the tensor into `outer` blocks
  // of `extent * inner` elements; the window takes `n0 * inner` from
  // each, so this is one memcpy per block whatever the rank.
  std::size_t outer = 1, inner = 1;
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if ((std::int64_t)i < _axis)      { outer *= (std::size_t)dims[i]; }
    else if ((std::int64_t)i > _axis) { inner *= (std::size_t)dims[i]; }
  }
  const std::size_t extent = (std::size_t)dims[(std::size_t)_axis];
  tb.resize_contiguous(outer * (std::size_t)n0 * inner);
  std::uint8_t* dst = tb.bytes_();
  for (std::size_t b = 0; b < outer; ++b) {
    const std::size_t src_off = (b * extent + (std::size_t)s0) * inner * esz;
    const std::size_t dst_off = b * (std::size_t)n0 * inner * esz;
    std::memcpy(dst + dst_off, raw.data() + src_off,
                (std::size_t)n0 * inner * esz);
  }

  if (!js.empty()) {
    FlexData sb = FlexData::from_json(js);
    if (sb.is_object()) { tb.sideband = std::move(sb); }
  }
  if (_extra_sideband.is_object()) {
    if (!tb.sideband.is_object()) { tb.sideband = FlexData::make_object(); }
    for (auto e : _extra_sideband.as_object()) {
      tb.sideband.as_object().insert_or_assign(std::string(e.first),
                                               e.second);
    }
  }

  string dimtxt;
  for (std::size_t i = 0; i < tb.shape.size(); ++i) {
    dimtxt += (i ? "x" : "") + std::to_string(tb.shape[i]);
  }
  session()->info(fmt(
      "LoadTensorStage('{}'): '{}' -> [{}]{}", this->id(), _path, dimtxt,
      (n0 != dims[(std::size_t)_axis] || s0 != 0)
          ? fmt(" (axis {} window [{}, {}) of {})", _axis, s0, s0 + n0,
                extent)()
          : std::string()));

  co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
  ctx.signal_done();
}

VPIPE_REGISTER_STAGE(LoadTensorStage)

}  // namespace vpipe
