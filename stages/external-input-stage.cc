#include "stages/external-input-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/vpipe-format.h"
#include "stages/tensor-buffer.h"

#include <limits>
#include <utility>

using namespace std;

namespace vpipe {

namespace {

const PortSpec kOports[] = {
  {.name = "tensor",
   .doc  = "one TensorBeat per push, and per closed lease",
   .type = &typeid(TensorBeatPayload)},
};

const ConfigKey kPushArgs[] = {
  {.key = "sideband", .type = ConfigType::Object,
   .doc = "attached to the beat as its sideband (timestamps, ids, ...)"},
};
const ConfigKey kLeaseArgs[] = {
  {.key = "type", .type = ConfigType::String, .required = true,
   .doc = "element type of the beat: u8, i8, f16, bf16 or f32"},
  {.key = "shape", .type = ConfigType::Array, .required = true,
   .doc = "the beat's shape, e.g. [3, 1080, 1920]"},
  {.key = "sideband", .type = ConfigType::Object,
   .doc = "attached to the beat as its sideband"},
};
const ConfigKey kSeqResult[] = {
  {.key = "seq", .type = ConfigType::Uint,
   .doc = "the beat's 1-based index on the oport this run"},
};
const ConfigKey kFinishResult[] = {
  {.key = "beats", .type = ConfigType::Uint,
   .doc = "how many beats the stage emitted this run"},
};
const BufferSpec kPushIn[] = {
  {.name = "data", .doc = "the tensor to emit; copied, so the caller's "
                          "memory is free again once the reply arrives",
   .type = kTensorElementTypes},
};
const BufferSpec kLeaseOut[] = {
  {.name = "data", .doc = "the beat's own storage, contiguous; write it, "
                          "then close the command to emit it",
   .type = kTensorElementTypes, .writable = true, .contiguous = true},
};

const CommandSpec kCommands[] = {
  {.name = "push",
   .doc  = "emit a copy of the caller's tensor as one beat; the reply "
           "comes when the oport has taken it",
   .args = kPushArgs, .results = kSeqResult, .in = kPushIn},
  {.name = "lease",
   .doc  = "reply with a writable beat of the given type and shape; the "
           "beat is emitted when the command is closed (dropped if the "
           "pipeline stops first)",
   .args = kLeaseArgs, .results = kSeqResult, .out = kLeaseOut,
   .holds = true},
  {.name = "finish",
   .doc  = "end the stream: consumers see EOS",
   .results = kFinishResult},
};

const StageSpec kSpec = {
  .type_name    = "external-input",
  .doc          = "Source: beats from outside the pipeline. Each `push` "
                  "command emits a copy of the caller's tensor; each "
                  "`lease` hands the caller a writable beat to fill in "
                  "place, emitted when the command closes; `finish` ends "
                  "the stream. Driven from the C++ / Python API.",
  .display_name = "External Input",
  .category     = StageCategory::Control,
  .iports       = {},
  .oports       = kOports,
  .attrs        = {},
  .commands     = kCommands,
};

}  // namespace

ExternalInputStage::ExternalInputStage(const SessionContextIntf* session,
                                       string                    id,
                                       vector<InEdge>            iports,
                                       FlexData                  config)
  : TypedStage<ExternalInputStage>(session, std::move(id), std::move(iports),
                                   std::move(config))
{
  allocate_oports(kSpec.oports.size());
}

const StageSpec&
ExternalInputStage::spec() const noexcept
{
  return kSpec;
}

Job
ExternalInputStage::process(RuntimeContext& ctx)
{
  shared_ptr<StageCommand> cmd = co_await ctx.next_command();
  if (!cmd) {
    // The inbox shut: the pipeline is stopping.
    ctx.signal_done();
    co_return;
  }
  const string_view name = cmd->name();
  if (name == "push") {
    co_await push_(ctx, std::move(cmd));
  } else if (name == "lease") {
    co_await lease_(ctx, std::move(cmd));
  } else if (name == "finish") {
    FlexData r = FlexData::make_object();
    r.as_object().insert("beats", FlexData::make_uint(_emitted));
    cmd->reply(std::move(r));
    ctx.signal_done();
  }
  co_return;
}

Job
ExternalInputStage::push_(RuntimeContext& ctx, shared_ptr<StageCommand> cmd)
{
  const DataBuffer* in = cmd->input("data");
  TensorBeat tb;
  string err;
  if (in == nullptr || !tensor_from_buffer(*in, &tb, &err)) {
    cmd->fail(in == nullptr ? string("no `data`") : err);
    co_return;
  }
  const FlexData sb = cmd->args().as_object().at("sideband");
  if (sb.is_object() && sb.as_object().size() > 0) { tb.sideband = sb; }
  co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
  FlexData r = FlexData::make_object();
  r.as_object().insert("seq", FlexData::make_uint(++_emitted));
  cmd->reply(std::move(r));
  co_return;
}

Job
ExternalInputStage::lease_(RuntimeContext& ctx, shared_ptr<StageCommand> cmd)
{
  const auto args = cmd->args().as_object();
  ElementType et{};
  TensorBeat::DType d{};
  const string tname(args.at("type").as_string(""));
  if (!parse_element_type(tname, &et) || !tensor_dtype_of(et, &d) ||
      et == ElementType::Bytes) {
    cmd->fail(fmt("`type` '{}' is not one of {}", tname,
                  kTensorElementTypes)());
    co_return;
  }
  vector<int64_t> shape;
  {
    const FlexData sh = args.at("shape");
    const auto av = sh.as_array();
    // Bounded well below what an allocation could ever satisfy, so the
    // product below cannot overflow on its way to that refusal.
    constexpr int64_t kMaxElems = int64_t{1} << 40;
    int64_t n = 1;
    for (std::size_t i = 0; i < av.size(); ++i) {
      const FlexData v = av[i];
      const int64_t dim = v.is_uint() ? static_cast<int64_t>(v.as_uint(0))
                                      : v.as_int(-1);
      if (!(v.is_int() || v.is_uint()) || dim < 0 ||
          (dim > 0 && n > kMaxElems / dim)) {
        cmd->fail("`shape` must be non-negative integers of a sane size");
        co_return;
      }
      n *= dim;
      shape.push_back(dim);
    }
  }

  // The beat is built first and SHARED with the reply; after close the
  // stage takes it back whole when the caller let go of every view, and
  // copies it when the caller kept one -- that memory stays the caller's.
  auto hold = make_shared<TensorBeatPayload>();
  hold->dtype = d;
  hold->shape = shape;
  size_t n = 1;
  for (int64_t s : shape) { n *= static_cast<size_t>(s); }
  hold->resize_contiguous(n);
  const FlexData sb = args.at("sideband");
  if (sb.is_object() && sb.as_object().size() > 0) { hold->sideband = sb; }

  const uint64_t seq = _emitted + 1;
  FlexData r = FlexData::make_object();
  r.as_object().insert("seq", FlexData::make_uint(seq));
  DataBuffer view = tensor_view(*hold, "data", hold, true);
  if (!cmd->reply(std::move(r), {std::move(view)})) { co_return; }
  co_await cmd->until_closed();
  if (cmd->cancelled()) { co_return; }

  unique_ptr<BeatPayloadIntf> out;
  if (hold.use_count() == 1) {
    out = make_payload<TensorBeatPayload>(std::move(*hold));
  } else {
    out = hold->clone();
  }
  hold.reset();
  co_await ctx.write(0, std::move(out));
  _emitted = seq;
  co_return;
}

VPIPE_REGISTER_STAGE(ExternalInputStage)
VPIPE_REGISTER_SPEC(ExternalInputStage, kSpec)

}
