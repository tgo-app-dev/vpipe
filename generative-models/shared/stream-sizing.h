#ifndef GENERATIVE_MODELS_SHARED_STREAM_SIZING_H
#define GENERATIVE_MODELS_SHARED_STREAM_SIZING_H

#include "generative-models/llama3/metal-llama-weights.h"

#include <sys/sysctl.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

// Two questions a model has to answer from the CHECKPOINT, before it has
// loaded anything: how little can it hold, and -- if it has no residency
// policy -- how much should it pin.
//
// They share a file because they share the pass: both walk the tensor
// table once, bucket every tensor by block prefix, and are then a sum or
// a greedy scan over the result. Nothing else here is common to them.
//
//   stream_floor_bytes()   THE FLOOR: trunk + one block, the least a
//                          streaming model can run on. Every streaming
//                          model wants this, DiTs included -- MiniMax-H3
//                          calls it directly and model_memory::
//                          streaming_floor_bytes() is the wrapper stages
//                          and plug-ins go through.
//
//   stream_pin_count()     THE PINNED PREFIX, and this half is LANGUAGE
//                          MODELS ONLY. No DiT uses it: a prefix sized
//                          before the run from a fraction of total RAM
//                          cannot sense the machine it lands on, so
//                          every block-streaming image and video DiT
//                          replaced it with shared/block-residency.h,
//                          which grows a resident set by MEASURING and
//                          sheds when it finds its own pages outside
//                          RAM. metal-qwen-model and metal-gemma-model
//                          keep it because they have nothing to grow
//                          into. Give them BlockResidency and this half
//                          goes.
//
// ---- the pinned prefix ------------------------------------------------
//
// In stream_blocks mode a model re-reads every block from the checkpoint
// on each forward -- ~one block resident, but the whole weight set is
// re-touched per pass. When there is spare RAM (but not enough to
// preload the whole model), pinning a LEADING prefix cuts that re-read
// proportionally: pinned blocks are read once and reused, only the tail
// streams.
//
// `stream_pin_count` sizes the prefix so that pinned + running stays within a
// fraction (e.g. 0.60) of physical RAM. It is greedy over the ACTUAL per-block
// byte sizes, so heterogeneous stacks (a DiT's big double blocks + smaller
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

// The LEAST a layer-streaming model holds: everything that is not a
// layer, plus the two in-flight slots it refills into.
//
// `stem` is the repeating prefix WITHOUT the index -- "blocks." for a
// DiT, "model.layers." for an LM -- and anything whose name contains
// `exclude` is left out of the layer figure (MiniMax-H3's AdaLN, which
// its bake retires before the first forward).
//
// This is a FLOOR, not a prediction: a resident set grows on top of it
// as free memory allows and is shed when it does not. That is exactly
// what a "will this graph fit" question wants, and it is why the number
// belongs in the plan while the growth does not.
inline std::size_t
stream_floor_bytes(const MetalLlamaWeights& wts, std::string_view stem,
                   std::string_view exclude = {})
{
  std::size_t trunk = 0;
  std::vector<std::size_t> layers;
  for (const std::string& nm : wts.tensor_names()) {
    const auto* ti = wts.info(nm);
    const std::size_t nb = ti != nullptr ? (std::size_t)ti->nbytes : 0;
    if (nm.rfind(std::string(stem), 0) != 0) { trunk += nb; continue; }
    if (!exclude.empty() &&
        nm.find(std::string(exclude)) != std::string::npos) {
      continue;
    }
    // The index runs from the end of the stem to the next '.'.
    const std::size_t i0 = stem.size();
    const std::size_t dot = nm.find('.', i0);
    if (dot == std::string::npos) { trunk += nb; continue; }
    const std::size_t idx =
        (std::size_t)std::atol(nm.substr(i0, dot - i0).c_str());
    if (layers.size() <= idx) { layers.resize(idx + 1, 0); }
    layers[idx] += nb;
  }
  std::size_t widest = 0;
  for (std::size_t b : layers) { if (b > widest) { widest = b; } }
  // No layers under this stem: nothing streams, so the floor is the
  // whole thing. Reporting trunk alone would promise a reduction that
  // cannot happen.
  if (widest == 0) { return 0; }
  return trunk + 2 * widest;
}

// THE MULTI-STACK FORM, for a model whose blocks come in more than one
// kind -- an image DiT's dual-stream and single-stream stacks, which it
// streams BOTH of and keeps a slot pair for EACH.
//
// Measuring such a model against one stem and letting the other stack
// fall into the trunk is safe in the sense that it over-states, and
// useless in practice: MEASURED on a 17316 MB two-stack DiT, one stem
// gives 12324 MB against a true 2848 -- a floor so close to the whole
// checkpoint that declaring it says almost nothing.
//
// Stems are tried IN ORDER and the first prefix match wins, so a list
// holding one stem that is a prefix of another must put the longer
// first. The stems in use do not overlap ("blocks." does not match
// "transformer_blocks.0.x", which starts with 't'), but a new one might.
inline std::size_t
stream_floor_bytes(const MetalLlamaWeights& wts,
                   const std::vector<std::string_view>& stems,
                   std::string_view exclude = {})
{
  if (stems.empty()) { return 0; }
  if (stems.size() == 1) { return stream_floor_bytes(wts, stems[0], exclude); }
  std::size_t trunk = 0;
  std::vector<std::vector<std::size_t>> stacks(stems.size());
  for (const std::string& nm : wts.tensor_names()) {
    const auto* ti = wts.info(nm);
    const std::size_t nb = ti != nullptr ? (std::size_t)ti->nbytes : 0;
    std::size_t si = stems.size();
    for (std::size_t i = 0; i < stems.size(); ++i) {
      if (nm.rfind(std::string(stems[i]), 0) == 0) { si = i; break; }
    }
    if (si == stems.size()) { trunk += nb; continue; }
    if (!exclude.empty() &&
        nm.find(std::string(exclude)) != std::string::npos) {
      continue;
    }
    const std::size_t i0 = stems[si].size();
    const std::size_t dot = nm.find('.', i0);
    if (dot == std::string::npos) { trunk += nb; continue; }
    const std::size_t idx =
        (std::size_t)std::atol(nm.substr(i0, dot - i0).c_str());
    std::vector<std::size_t>& blocks = stacks[si];
    if (blocks.size() <= idx) { blocks.resize(idx + 1, 0); }
    blocks[idx] += nb;
  }
  std::size_t slots = 0;
  for (const std::vector<std::size_t>& blocks : stacks) {
    std::size_t widest = 0;
    for (std::size_t b : blocks) { if (b > widest) { widest = b; } }
    slots += 2 * widest;      // a pair per stack, both alive for the run
  }
  // Nothing matched any stem: nothing streams, and the floor is the
  // whole thing. Reporting the trunk alone would promise a reduction
  // that cannot happen.
  if (slots == 0) { return 0; }
  return trunk + slots;
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

#endif  // GENERATIVE_MODELS_SHARED_STREAM_SIZING_H
