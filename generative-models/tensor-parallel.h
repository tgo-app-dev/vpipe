#ifndef VPIPE_GENERATIVE_MODELS_TENSOR_PARALLEL_H
#define VPIPE_GENERATIVE_MODELS_TENSOR_PARALLEL_H

// TENSOR PARALLELISM ACROSS HOSTS. A DiT's attention is split by HEADS
// and its feed-forward by its hidden ROWS between RANKS -- each rank a
// process, on its own Mac, holding only its share of every block's four
// projections -- as Megatron splits a transformer: q / k / v and fc1 by
// their output rows, the out projection and fc2 by their input columns,
// so each of those two yields a PARTIAL SUM of the block's full-width
// output. The ranks exchange the partials and each adds the others', and
// every rank then holds the same residual stream it would have held
// alone. Everything else -- norms, modulation, RoPE, the sampler -- runs
// whole on every rank.
//
// vpipe MOVES NO BYTES BETWEEN MACHINES. A rank hands its partial to a
// CHANNEL and gets the others' back; the channel comes from a TRANSPORT a
// plugin registers by name (register_tp_transport, below), and
// how it carries the bytes -- sockets over Thunderbolt, RDMA, anything --
// is the plugin's. A stage names the transport and passes its parameters
// through unread (generate-video's `tensor_parallel`).
//
// A share need not be even: a rank on a slower Mac takes fewer heads and
// rows, so the ranks reach each exchange together.

#include "common/flex-data.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "common/vpipe-api.h"

// OPTIONAL TO A PLUGIN, as stages/model-source.h is: outside libvpipe the
// functions below are WEAK imports, null on a host built before them (no
// tensor-parallel/1), so a plugin that offers a transport when it can
// still loads there. A strong import would be a missing symbol, refused
// by dyld at dlopen before the plugin's required_features are read.
#if defined(VPIPE_BUILDING_LIBRARY)
#define VPIPE_TP_IMPORT
#else
#define VPIPE_TP_IMPORT __attribute__((weak_import))
#endif

VPIPE_API_BEGIN

namespace vpipe::genai {

// One rank's share: [head0, head1) of the model's `n_heads` attention
// heads and [ffn0, ffn1) of its `n_ffn` feed-forward rows. The ranks'
// shares tile both with no overlap; ranks == 1 is the whole model.
//
// ITS LAYOUT IS THE ABI's (abi/plugin-abi-<N>.txt): a plugin is handed
// one, and from_flex() writes a whole one into the caller's. It grows
// through `extra` (docs/PLUGINS.md, "Versioning"): a new field is a KEY
// there, never a member. None is defined yet.
struct TpShare {
  int rank  = 0;
  int ranks = 1;
  int head0 = 0, head1 = 0;
  int ffn0  = 0, ffn1  = 0;
  int n_heads = 0, n_ffn = 0;
  FlexData extra;

  bool active() const { return ranks > 1; }
  int heads() const { return head1 - head0; }
  int ffn() const { return ffn1 - ffn0; }

  // From a stage's `tensor_parallel` bag: rank, ranks, heads [h0, h1],
  // ffn [f0, f1]. Missing heads / ffn: an even split of `n_heads` /
  // `ffn` by rank. False, with `err`, for a share that does not fit.
  VPIPE_TP_IMPORT static bool from_flex(const FlexData& fd, int n_heads,
                                        int ffn, TpShare* out,
                                        std::string* err);
};

// What a partial's elements are: a transport adding more than one other
// rank's partial adds them as this type.
enum class TpDType : std::uint32_t {
  kBF16 = 1,
  kF16  = 2,
  kF32  = 3,
};

// A rank's link to the others for ONE GENERATION -- a clip: opened as
// its denoise starts and let go when it ends, however it ends (a stop, a
// failed exchange, a refusal), on every rank alike. A rank keeps its
// share's weights between clips or reloads them by its own host's
// memory; the channel, and the exchange count with it, are new each
// clip, so what one rank did with its last one -- stopped part way,
// reloaded -- cannot leave the next out of step.
//
// HOW THE HOST CALLS IT. exchange() and forget() are called one at a
// time -- never two at once, though not always from the same thread (a
// worker carries a long run's exchanges while the GPU works on).
// set_stop() may come at any time, from the thread that runs the stage.
// Nothing the host calls here may throw: an exception is caught and
// taken as the call's failure (open() null, exchange() false), and an
// exchange() that throws fails the run.
//
// THE MEMORY handed to exchange() is the host's few EXCHANGE BUFFERS: a
// buffer is allocated before its first exchange and stays where it is,
// at the same address, until the host calls forget() on it -- before it
// frees or replaces it, never while an exchange of it is in flight -- or
// the channel goes. So a transport may register a buffer with a device
// (RDMA) once, the first time it is handed one, and keep it until then.
class TpChannel {
public:
  virtual ~TpChannel() = default;

  // Hand this rank's partial sum -- `bytes` of it, of `dtype` elements,
  // in memory the CPU may read (a shared Metal buffer's contents, the GPU
  // work that wrote it waited for) -- and return once `others` holds the
  // SUM of every other rank's partial (with two ranks, the peer's own),
  // in the same layout and type. `seq` counts the generation's exchanges
  // from 0 on every rank alike: a transport checks it to catch ranks out
  // of step.
  // False, with `err`, when the link failed or the run was stopped; the
  // run then fails.
  virtual bool exchange(const void* partial, void* others, std::size_t bytes,
                        TpDType dtype, std::uint64_t seq,
                        std::string* err) = 0;

  // [base, base + bytes) -- an exchange buffer, whole -- is about to be
  // freed or replaced: nothing it held may be used again, a device's
  // registration of it included. A transport that keeps nothing of the
  // memory it is handed ignores it.
  virtual void forget(const void* base, std::size_t bytes)
  {
    (void)base;
    (void)bytes;
  }

  // The run's STOP, for a channel to poll while it WAITS -- for the peer
  // to appear, for its partial: a stop asked for then lands there as it
  // does between blocks, not when the peer gives up. Set before a
  // generation, cleared (empty) after it on every way out; a channel
  // calls it under its own lock, so once cleared it is never called
  // again. A channel that polls nothing ignores it. The host also sets a
  // stop that is always true before it lets go of a channel with an
  // exchange still waiting on a peer, so a channel that polls returns.
  virtual void set_stop(std::function<bool()> stop) { (void)stop; }

  // EXTENSION POINT (plugin ABI; docs/PLUGINS.md, "Versioning"). A host
  // newer than the plugin asks for a capability this interface did not
  // have when the plugin was built, by name ("vpipe.<what>/<rev>"); null
  // means "not provided" and the host keeps the older behaviour. A new
  // virtual added here instead would move this vtable under every plugin
  // in the support window.
  virtual void* query_extension(std::string_view id) noexcept
  {
    (void)id;
    return nullptr;
  }
};

// What a plugin registers: channels opened by name.
class TpTransport {
public:
  virtual ~TpTransport() = default;

  virtual std::string_view name() const = 0;

  // A channel for one generation (TpChannel). `params` is the
  // transport's own bag, passed through from the stage unread; `share`
  // the rank it serves. Null, with `err`, when it cannot be opened (the
  // peer is not there). Called again for every clip, once the last
  // clip's channel has been let go.
  virtual std::shared_ptr<TpChannel> open(const FlexData& params,
                                          const TpShare&  share,
                                          std::string*    err) = 0;

  // EXTENSION POINT, as TpChannel's.
  virtual void* query_extension(std::string_view id) noexcept
  {
    (void)id;
    return nullptr;
  }
};

// Register a transport, process-wide (a plugin links the host libvpipe
// shared, so this is the host's one registry). Takes ownership; first-
// wins on `name()`: false, the transport dropped, when the name is taken.
VPIPE_TP_IMPORT bool register_tp_transport(std::unique_ptr<TpTransport> t);

// The transport registered under `name`, or null.
VPIPE_TP_IMPORT TpTransport* find_tp_transport(std::string_view name);

// Whether this host has the functions above: always inside libvpipe; in a
// plugin, whether the weak imports resolved. Ask before calling them.
inline bool
tp_transports_available() noexcept
{
#if defined(VPIPE_BUILDING_LIBRARY)
  return true;
#else
  return &register_tp_transport != nullptr;
#endif
}

}

VPIPE_API_END

#endif
