#ifndef GENERATIVE_MODELS_SHARED_STREAM_PIN_H
#define GENERATIVE_MODELS_SHARED_STREAM_PIN_H

#include "generative-models/llama3/metal-llama-weights.h"

#include <sys/sysctl.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

// Pinned-prefix streaming policy, shared by the diffusion DiTs (Krea-2, FLUX.2,
// Qwen-Image-Edit). In stream_blocks mode a DiT re-reads every block from the
// retained mmap on each forward -- ~one block resident, but the whole weight set
// is re-touched per pass. When there is spare RAM (but not enough to preload the
// whole DiT), pinning a LEADING prefix of blocks resident cuts that re-read
// proportionally: pinned blocks are read once and reused, only the tail streams.
//
// `stream_pin_count` sizes the prefix so that pinned + running stays within a
// fraction (e.g. 0.60) of physical RAM. It is greedy over the ACTUAL per-block
// byte sizes, so heterogeneous stacks (FLUX.2's big double blocks + smaller
// single blocks) pin correctly by memory, not by a uniform count.
//
// WHAT ELSE THE MODEL HOLDS is two terms, and neither used to be counted.
//
// The TRUNK -- everything the stack does not stream -- is measured here: the
// pass below already visits every tensor to bucket it by block, and the ones
// matching no block prefix ARE the trunk. Assuming it was small held for the
// DiTs this was written against and broke on one that is not: LTX-2.5 carries
// 4.6 GB of non-block weights at w8, almost all of it two text connectors, so
// a 16 GB box pinned 20 of 48 blocks believing 1 GB sat beside them.
//
// The ACTIVATION SCRATCH cannot be measured here -- it is a function of the
// BEAT, not of the checkpoint -- so the caller states it. A constant is wrong
// at every geometry but one: MiniMax-H3 allocates ~4 GB at its own sizes and
// LTX-2.5's arena is 1.03 GB at 6630 tokens and grows with them. Both models
// have a static estimator for exactly this (MetalMiniMaxH3Transformer::
// scratch_bytes, BlockScratch::predict_bytes); pass one. The 1 GB default is
// what this always assumed, kept so an un-updated caller is no worse off.
//
// Over-counting is the safe direction: it pins fewer blocks and costs a
// re-read per forward, where under-counting costs the machine.

namespace vpipe {
namespace genai {

// Total physical RAM in bytes, or 0 if unknown.
inline std::size_t stream_physical_ram()
{
  // Same override as stages/model-memory.h's phys_ram(): the pinned-block
  // budget has to agree with the stage-level stream/unload decision, so both
  // read VPIPE_RAM_LIMIT_MB when it is set.
  if (const char* e = std::getenv("VPIPE_RAM_LIMIT_MB")) {
    const long long mb = std::atoll(e);
    if (mb > 0) { return (std::size_t)mb << 20; }
  }
  std::uint64_t mem = 0;
  std::size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) { return 0; }
  return (std::size_t)mem;
}

// Given the streamed blocks' prefixes IN STREAM ORDER (each ending in the
// separator '.', so "transformer_blocks.1." does not also match block 10) and a
// budget FRACTION of physical RAM (e.g. 0.60 => "pinned + running <= 60% of
// RAM"), return how many LEADING blocks to pin resident. Greedy over the actual
// per-block byte sizes. Reserves room within the fraction for the in-flight
// streamed block (+ a double-buffer margin) and ~1 GB of activation scratch +
// always-resident top-level weights, so pinned + running stays within budget.
// Returns 0 when frac <= 0, RAM is unknown, or nothing fits.
inline int stream_pin_count(const MetalLlamaWeights& wts,
                            const std::vector<std::string>& block_prefixes,
                            double frac,
                            std::size_t scratch_bytes = (std::size_t{1} << 30),
                            std::size_t* trunk_out = nullptr)
{
  if (trunk_out != nullptr) { *trunk_out = 0; }
  if (frac <= 0.0 || block_prefixes.empty()) { return 0; }
  const std::size_t ram = stream_physical_ram();
  if (ram == 0) { return 0; }
  // Per-block wired bytes: sum nbytes of every tensor under each prefix. One
  // pass over the (unordered) name set; each name belongs to at most one block.
  // Anything under NONE of them is the trunk.
  std::vector<std::size_t> sizes(block_prefixes.size(), 0);
  std::size_t trunk = 0;
  const std::vector<std::string> names = wts.tensor_names();
  for (const std::string& n : names) {
    const auto* ti = wts.info(n);
    const std::size_t nb = ti != nullptr ? (std::size_t)ti->nbytes : 0;
    bool in_block = false;
    for (std::size_t i = 0; i < block_prefixes.size(); ++i) {
      if (n.rfind(block_prefixes[i], 0) == 0) {
        sizes[i] += nb;
        in_block = true;
        break;
      }
    }
    if (!in_block) { trunk += nb; }
  }
  if (trunk_out != nullptr) { *trunk_out = trunk; }
  std::size_t maxb = 0;
  for (std::size_t s : sizes) { if (s > maxb) { maxb = s; } }
  const std::size_t budget = (std::size_t)(frac * (double)ram);
  const std::size_t reserve = 2 * maxb + trunk + scratch_bytes;
  const std::size_t avail = budget > reserve ? budget - reserve : 0;
  std::size_t cum = 0;
  int k = 0;
  for (std::size_t i = 0; i < block_prefixes.size(); ++i) {
    if (cum + sizes[i] > avail) { break; }
    cum += sizes[i];
    ++k;
  }
  return k;
}

}  // namespace genai
}  // namespace vpipe

#endif  // GENERATIVE_MODELS_SHARED_STREAM_PIN_H
