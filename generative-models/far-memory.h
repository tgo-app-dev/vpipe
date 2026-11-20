#ifndef VPIPE_GENERATIVE_MODELS_FAR_MEMORY_H
#define VPIPE_GENERATIVE_MODELS_FAR_MEMORY_H

// FAR MEMORY: another machine's DRAM lent to this process, read and
// written by offset. A Mixture of Experts too large for this Mac keeps
// its routed experts in a cache refilled from disk on a miss
// (moe-expert-cache.h); with far memory the cache has a SECOND TIER --
// the experts selected next most often after those it holds, kept on
// the other machine and read back over a link (Thunderbolt 5: several
// cables at ~9 GB/s each, microseconds away) instead of from the drive.
// What the far side holds is wholly this side's choice: the far side
// lends a number of bytes and stores what it is handed where it is told.
// It needs no model, no files, nothing of vpipe.
//
// vpipe MOVES NO BYTES BETWEEN MACHINES, as with tensor-parallel.h: a
// plugin registers a FarMemoryTransport by name, and a stage names it
// and passes its parameters through unread (text-chat's `far_memory`).
//
// OPTIONAL TO A PLUGIN: outside libvpipe the functions below are WEAK
// imports, null on a libvpipe built before them (no far-memory/1).

#include "common/flex-data.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "common/vpipe-api.h"

#if defined(VPIPE_BUILDING_LIBRARY)
#define VPIPE_FAR_IMPORT
#else
#define VPIPE_FAR_IMPORT __attribute__((weak_import))
#endif

VPIPE_API_BEGIN

namespace vpipe::genai {

// A piece of a read or write: `bytes` at far offset `offset`, to or from
// `local` (memory the CPU may read and write: a shared Metal buffer's
// contents).
struct FarPiece {
  std::uint64_t offset = 0;
  void*         local  = nullptr;
  std::size_t   bytes  = 0;
};

// The memory lent, open for one model's life.
//
// HOW IT IS CALLED. read() and write() come from several threads at once
// (the expert cache's readers and its writer), each call whole on return:
// a read's bytes in `local`, a write's out of `local` (it may be reused).
// No write is in flight with another call on the same far bytes (reads
// may overlap: a far compute call's experts read back after it was
// dropped as late). Nothing
// called here may throw: an exception is caught and taken as the call's
// failure. A failure is not retried by the caller -- the far tier is let go
// and the model goes on from disk -- so a transport that can recover (a
// cable among several cut) recovers inside the call.
class FarMemory {
public:
  virtual ~FarMemory() = default;

  // How many bytes are lent: offsets [0, bytes()).
  virtual std::uint64_t bytes() const = 0;

  // The pieces, read (or written) together: one call carries an expert's
  // several tensors, so a transport may stripe them across its links.
  virtual bool read(const FarPiece* pieces, std::size_t n,
                    std::string* err) = 0;
  virtual bool write(const FarPiece* pieces, std::size_t n,
                     std::string* err) = 0;

  // A short description for the log ("48 GB on m5-max over 3 links").
  virtual std::string describe() const { return {}; }

  // EXTENSION POINT (plugin ABI; docs/PLUGINS.md, "Versioning"), as
  // TpChannel's.
  virtual void* query_extension(std::string_view id) noexcept
  {
    (void)id;
    return nullptr;
  }
};

// FAR COMPUTE (extension "far-compute/2", asked of a FarMemory through
// query_extension; PROTOTYPE): the far side RUNS experts it holds and
// hands back each one's output, in place of handing back its bytes -- a
// decode row's activation in and the experts' rows out (8 KB each way an
// expert for DeepSeek V4.1 Flash) where a read moves 8-18 MB an expert.
// An expert is named by where its parts lie in the far memory, as the
// cache wrote them; its form says how to read them. (far-compute/1 handed
// back one rounded sum: not the local combine's arithmetic, so it was
// retired, not kept.)
//
// THE CONTRACT. Each call's row is that expert's output as the model's
// own decode kernels make it: act = T(weight * silu(g) * u), g and u the
// row's dot with w1 and w3 each summed in float (silu's argument and u
// clamped to +-swiglu_limit, g from above only, where the limit is above
// 0), then out = T(w2 . act), the dot summed in float -- T the form's
// dtype. A far side running vpipe's own kernels on the same bytes gives
// the local rows bit for bit (the model then combines them as it combines
// its own: the result is the local one). Another's rows are within its
// float sums' rounding, and that costs more than those bits: which pairs
// run far follows the cache's timing, so its tokens are then no longer
// reproducible run to run, nor a speculative decode's verify rows (run
// here) its decode rows (run there) -- DSpark's token-exactness against a
// plain decode. Write the sums in vpipe's order to keep both.
// HOW IT IS CALLED: from a dispatch queue's thread, not the model's (no
// state may be kept per thread), one call at a time for a model (a late
// one holds the next back until it ends), while
// the expert cache's read() and write() of OTHER far bytes go on beside
// it from its own threads; the object lives as long as the FarMemory it
// was asked of. The model waits for an answer only so long -- a few times
// what reading those experts would cost -- then reads them as any miss and
// drops the late answer (its rows go to a buffer of the call's own). A
// call MUST RETURN: the model waits for one still running as it goes (a
// far side hung inside experts() hangs the model's unload). A
// form it does not run (an unknown format or dtype, a bad table) is a
// false return, after which the model reads those experts for the rest of
// its life. An exception is taken as a false return.
inline constexpr std::string_view kFarComputeExtension = "far-compute/2";

// The formats, never renumbered: a far side built before a format sees
// its number as unknown, never as another.
enum : std::uint32_t {
  kFarExpertsDeepseekFp4 = 1,   // DeepSeek V4.1's E2M1 routed experts:
                                // parts w1, its E8M0 scales, w3, its
                                // scales, w2, its scales; arg[] all 0
  kFarExpertsDeepseekGguf = 2,  // DeepSeek V4.1's experts in a GGUF's
                                // own blocks (llama.cpp's IQ2_XXS 16,
                                // IQ2_XS 17, IQ3_XXS 18, IQ2_S 22, IQ1_M
                                // 29): parts 0, 2, 4 w1's, w3's and w2's
                                // slabs ([inter, hidden], [inter, hidden],
                                // [hidden, inter], rows of 256-value
                                // blocks), the odd ones 0; arg[0] |
                                // arg[1] << 32 the far offset of a
                                // FarGgufTables, arg[2] the three types,
                                // w1's in its low byte
};

// A kFarExpertsDeepseekGguf form's tables in the far memory, as the model
// writes them there once: this head, then `words` u64s -- the grids and
// signs of ggml-blocks.h's gpu_iq_tables_native, version 1's layout (u64
// offsets: IQ2_XXS's grid 0, IQ2_XS's 256, IQ1_S / M's 768, the IQ2
// signs' bytes 2816, IQ1_M's grid as nibbles 2832, IQ2_S's grid 3856,
// IQ3_XXS's (u32s) 4880, IQ3_S's (u32s) 5008, IQ4_NL's values 5264). A far
// side that reads another magic, version or size runs none of them.
struct FarGgufTables {
  std::uint32_t magic = 0x54494756u;   // "VGIT"
  std::uint32_t version = 1;
  std::uint64_t words = 0;             // u64s after this head
};
inline constexpr std::uint64_t kFarGgufTablesV1Words = 5266;

struct FarExpertForm {
  std::uint32_t format = 0;     // kFarExperts*
  std::uint32_t dtype = 0;      // the activation's and rows': 0 f16,
                                // 1 bf16
  std::uint32_t hidden = 0, inter = 0;
  float swiglu_limit = 0.0f;
  std::uint32_t arg[3] = {};    // the format's own (0 where it has none)
};

struct FarExpertCall {
  std::uint64_t part[6] = {};   // far offset of each part
  float weight = 0.0f;          // the routing weight, as the model's
  std::uint32_t pad = 0;        // kernels apply it before w2; pad 0
};

class FarCompute {
public:
  virtual ~FarCompute() = default;
  // One row: `x` [hidden] of form.dtype in; `out` [n, hidden] of the same
  // out, call j's row at j (see THE CONTRACT).
  virtual bool experts(const FarExpertForm& form, const void* x,
                       const FarExpertCall* calls, std::size_t n,
                       void* out, std::string* err) = 0;

  virtual void* query_extension(std::string_view id) noexcept
  {
    (void)id;
    return nullptr;
  }
};

// FAR FILES (extension "far-files/1", asked of a FarMemory through
// query_extension; PROTOTYPE): the far side keeps COPIES OF THE MODEL'S
// FILES and reads them off its own drive for this side -- a second drive
// beside this one's, for what neither tier holds. A file is named as it
// lies under a models root (org/repo/<its path in the model's folder>),
// and must be the same size there. Called as FarMemory is (several
// threads; an exception taken as the call's failure); it lives as long as
// its FarMemory.
inline constexpr std::string_view kFarFilesExtension = "far-files/1";

class FarFiles {
public:
  virtual ~FarFiles() = default;
  // The far copy of `name` (`size` bytes): an id, or -1 with `err` (none
  // there, or another size).
  virtual int open(std::string_view name, std::uint64_t size,
                   std::string* err) = 0;
  // `bytes` of file `id` from `offset` into `local`, whole on return.
  // From several threads at once.
  virtual bool read(int id, std::uint64_t offset, void* local,
                    std::size_t bytes, std::string* err) = 0;

  virtual void* query_extension(std::string_view id) noexcept
  {
    (void)id;
    return nullptr;
  }
};

class FarMemoryTransport {
public:
  virtual ~FarMemoryTransport() = default;

  virtual std::string_view name() const = 0;

  // Far memory reached as `params` say (the transport's own bag, passed
  // through from the stage unread). Null, with `err`, when it cannot be.
  virtual std::shared_ptr<FarMemory> open(const FlexData& params,
                                          std::string* err) = 0;

  virtual void* query_extension(std::string_view id) noexcept
  {
    (void)id;
    return nullptr;
  }
};

// Register a transport, process-wide. Takes ownership; first-wins on
// name(): false, the transport dropped, when the name is taken.
VPIPE_FAR_IMPORT bool register_far_memory_transport(
    std::unique_ptr<FarMemoryTransport> t);

// The transport registered under `name`, or null.
VPIPE_FAR_IMPORT FarMemoryTransport* find_far_memory_transport(
    std::string_view name);

// Opens far memory from a stage's `far_memory` bag -- {"transport": name,
// "params": {...}} -- catching what the plugin throws. Null with `err`
// when the bag is not one, no such transport is registered, or the
// transport cannot open it.
VPIPE_FAR_IMPORT std::shared_ptr<FarMemory> open_far_memory(
    const FlexData& bag, std::string* err);

// Whether this libvpipe has the functions above: always inside libvpipe;
// in a plugin, whether the weak imports resolved.
inline bool
far_memory_available() noexcept
{
#if defined(VPIPE_BUILDING_LIBRARY)
  return true;
#else
  return &register_far_memory_transport != nullptr;
#endif
}

}

VPIPE_API_END

#endif
