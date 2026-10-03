#include "stages/tensor-list-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <memory>
#include <utility>

namespace vpipe {

namespace {

constexpr ConfigKey kAttrs[] = {
  {.key = "sideband", .type = ConfigType::Any, .required = false,
   .doc = "a JSON object set as every emitted list's own sideband -- for "
          "saying something about the list as a whole. What belongs to "
          "one item stays on that item's sideband, which this stage "
          "carries through untouched",
   .def_str = ""},
};

// Untagged: a list of pictures is the common case, but a list of
// latents or of any tensors is the same operation, and a declared tag is
// a connection CONSTRAINT here, not a hint. The TensorBeatPayload type
// is still enforced.
#define VPIPE_TENSOR_LIST_ITEM(n)                                          \
  {.name = "item" #n,                                                      \
   .doc = "OPTIONAL list item " #n ": one TensorBeatPayload per list. "    \
          "Wired items are listed in port order; an unwired port is "      \
          "skipped",                                                       \
   .type = &typeid(TensorBeatPayload), .clock_group = 0}
const PortSpec kIports[] = {
  VPIPE_TENSOR_LIST_ITEM(0), VPIPE_TENSOR_LIST_ITEM(1),
  VPIPE_TENSOR_LIST_ITEM(2), VPIPE_TENSOR_LIST_ITEM(3),
  VPIPE_TENSOR_LIST_ITEM(4), VPIPE_TENSOR_LIST_ITEM(5),
  VPIPE_TENSOR_LIST_ITEM(6), VPIPE_TENSOR_LIST_ITEM(7),
  VPIPE_TENSOR_LIST_ITEM(8), VPIPE_TENSOR_LIST_ITEM(9),
};
#undef VPIPE_TENSOR_LIST_ITEM
static_assert(sizeof(kIports) / sizeof(kIports[0]) ==
              TensorListStage::kMaxItems);

const PortSpec kOports[] = {
  {.name = "list",
   .doc = "ONE TensorListPayload per round -- one item from every wired "
          "iport, in port order. Feeds diffusion-conditioner "
          "`ref_images` and vae-encode `images` (pictures), or "
          "generate-image `ref_latents` (latents)",
   .type = &typeid(TensorListPayload), .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = "tensor-list",
  .doc       = "Collects one beat from each wired input into ONE list, in "
               "port order: the bridge from per-picture beats (a "
               "resample, a host's source) to the list ports of the "
               "multi-reference stages. All or nothing per list.",
  .display_name = "Tensor List",
  .category  = StageCategory::Control,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

TensorListStage::TensorListStage(const SessionContextIntf* s,
                                 std::string               id,
                                 std::vector<InEdge>       iports,
                                 FlexData                  config)
  : TypedStage<TensorListStage>(s, std::move(id), std::move(iports),
                                std::move(config))
{
  // Past the declared ports an edge would be untyped and unchecked;
  // refused rather than quietly accepted.
  const auto& edges = iport_edges();
  for (std::size_t i = kMaxItems; i < edges.size(); ++i) {
    if (edges[i].v != nullptr) {
      fail_config(fmt(
          "TensorListStage('{}'): iport {} is wired, but a list takes at "
          "most {} items", this->id(), i, kMaxItems));
      break;
    }
  }
  {
    auto c = this->config().as_object();
    if (c.contains("sideband")) {
      FlexData v = c.at("sideband");
      if (v.is_object()) {
        _sideband_cfg = v;
      } else if (!v.is_null() &&
                 !(v.is_string() && v.as_string("").empty())) {
        fail_config(fmt(
            "TensorListStage('{}'): sideband must be a JSON object",
            this->id()));
      }
    }
  }
  allocate_oports(kSpec.oports.size());
}

const StageSpec&
TensorListStage::spec() const noexcept
{
  return kSpec;
}

Job
TensorListStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  _emitted = 0;
  co_return;
}

Job
TensorListStage::process(RuntimeContext& ctx)
{
  auto list = std::make_unique<TensorListPayload>();
  if (_sideband_cfg.is_object()) { list->sideband = _sideband_cfg; }
  bool wired = false;
  bool whole = true;
  for (unsigned p = 0; p < ctx.num_iports(); ++p) {
    if (!ctx.iport_connected(p)) { continue; }
    wired = true;
    auto in = co_await ctx.read(p);
    if (!in) {
      // An input ended. At the start of a round that is the end of the
      // run; part-way through, the round is short and is not a list.
      if (!list->items.empty()) {
        session()->warn(fmt(
            "TensorListStage('{}'): iport {} ended after {} item(s) of "
            "this list arrived; dropping the partial list -- every wired "
            "input must deliver one beat per list", this->id(), p,
            list->items.size()));
      }
      ctx.signal_done();
      co_return;
    }
    auto* tb = dynamic_cast<TensorBeatPayload*>(in.get());
    if (tb == nullptr) {
      // Keep reading the other inputs, so the next round starts in step.
      session()->warn(fmt(
          "TensorListStage('{}'): iport {} carried {}, not a tensor; this "
          "list is dropped", this->id(), p, in->describe()));
      whole = false;
      continue;
    }
    // Ours to take: the payload is owned, so the item moves (a shared
    // Metal buffer moves with its handle).
    list->items.push_back(std::move(static_cast<TensorBeat&>(*tb)));
  }
  if (!wired) {
    ctx.signal_done();
    co_return;
  }
  if (!whole) { co_return; }
  ++_emitted;
  co_await ctx.write(0, std::move(list));
  co_return;
}

VPIPE_REGISTER_STAGE(TensorListStage)
VPIPE_REGISTER_SPEC(TensorListStage, kSpec)

}  // namespace vpipe
