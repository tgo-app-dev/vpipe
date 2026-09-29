// Which IOReport "Energy Model" channels belong to a block's total.
//
// This is a four-line predicate with two expensive failure modes, and
// both have already happened in this tree:
//
//   TOO NARROW -- an exact-name match. macOS 27 renamed every block with
//   a tile index (ANE -> ANE0), so an exact match reports 0.00 W on a
//   busy machine. That is what broke macmon on macOS 27.
//
//   TOO WIDE -- a prefix match. The same group carries "GPU SRAM" and
//   "GPU Energy", the latter being the same energy republished in
//   nanojoules, so a prefix match reported close to DOUBLE the real GPU
//   power -- an idle box read ~48 W.
//
// The channel names below are transcribed from live dumps: an M4 Pro on
// macOS 26, an M5 on macOS 26.6.2, and an M5 Pro on macOS 27.

#include "minitest.h"
#include "common/soc-activity.h"
#include "common/soc-energy-channel.h"

#include <cmath>
#include <vector>

using namespace vpipe;

TEST(soc_energy_channel, bare_stem_matches) {
  // macOS 26 and earlier, both on M4 and on M5 silicon.
  EXPECT_TRUE(soc_block_energy_channel("ANE", "ANE"));
  EXPECT_TRUE(soc_block_energy_channel("GPU", "GPU"));
}

TEST(soc_energy_channel, indexed_stem_matches) {
  // macOS 27 indexes every block.
  EXPECT_TRUE(soc_block_energy_channel("ANE0", "ANE"));
  EXPECT_TRUE(soc_block_energy_channel("GPU0", "GPU"));
  // The index is a TILE NUMBER, so a part with two ANEs publishes both
  // and the block's power is their sum. Nothing in the tree has shipped
  // with two yet; this is the case the suffix exists for.
  EXPECT_TRUE(soc_block_energy_channel("ANE1", "ANE"));
  EXPECT_TRUE(soc_block_energy_channel("ANE10", "ANE"));
}

// THE DOUBLE-COUNT CASES. Each of these is a real channel sitting in
// "Energy Model" beside the block's own, and each was summed into the
// GPU total by the prefix match this replaced.
TEST(soc_energy_channel, companions_are_rejected) {
  // The same energy, republished in nanojoules.
  EXPECT_FALSE(soc_block_energy_channel("GPU Energy", "GPU"));
  // A component already inside the block's figure.
  EXPECT_FALSE(soc_block_energy_channel("GPU SRAM", "GPU"));
  // Real macOS 27 channel names from other groups, which a reader that
  // matched loosely could pick up if it ever stopped filtering by group.
  EXPECT_FALSE(soc_block_energy_channel("ANEXL U", "ANE"));
  EXPECT_FALSE(soc_block_energy_channel("ANE L0 RD", "ANE"));
  EXPECT_FALSE(soc_block_energy_channel("THROT-ANE-SUM", "ANE"));
}

TEST(soc_energy_channel, near_misses_are_rejected) {
  // Shorter than the stem, and a stem that is a prefix of a longer word.
  EXPECT_FALSE(soc_block_energy_channel("AN", "ANE"));
  EXPECT_FALSE(soc_block_energy_channel("", "ANE"));
  EXPECT_FALSE(soc_block_energy_channel("ANEX", "ANE"));
  // Anchored at the front: a block whose name merely ends in the stem is
  // a different block.
  EXPECT_FALSE(soc_block_energy_channel("DISPEXT0", "DISP"));
  // Case-sensitive -- these are kernel names, not user input.
  EXPECT_FALSE(soc_block_energy_channel("ane0", "ANE"));
}

// The other blocks renamed by macOS 27, so the helper is exercised as
// the general rule it is rather than only on ANE and GPU.
TEST(soc_energy_channel, every_renamed_block) {
  EXPECT_TRUE(soc_block_energy_channel("ISP0", "ISP"));
  EXPECT_TRUE(soc_block_energy_channel("DRAM0", "DRAM"));
  EXPECT_TRUE(soc_block_energy_channel("AVE0", "AVE"));
  EXPECT_TRUE(soc_block_energy_channel("AMCC0", "AMCC"));
  EXPECT_TRUE(soc_block_energy_channel("DCS0", "DCS"));
  EXPECT_TRUE(soc_block_energy_channel("MSR0", "MSR"));
  EXPECT_TRUE(soc_block_energy_channel("DISP0", "DISP"));
  EXPECT_TRUE(soc_block_energy_channel("DISPEXT0", "DISPEXT"));
  // New on macOS 27; no unsuffixed spelling ever existed for these.
  EXPECT_TRUE(soc_block_energy_channel("AFR0", "AFR"));
  EXPECT_TRUE(soc_block_energy_channel("FAB0", "FAB"));
}

// ---- the PMP readings the ANE meter falls back to ----------------------
//
// Residencies below are the per-second deltas logged on an M5 Pro
// (macOS 27) and an M4 Pro (macOS 26) under vpipe's ANE-only bench
// (ane_module.ffn_tier_shape_bench at the MiniMax-H3 feed-forward shape)
// and at idle, transcribed as percentages -- only the proportions matter.

namespace {
bool near_(double a, double b) { return std::abs(a - b) < 1e-9; }
}

TEST(soc_activity, floor_share_above_vmin) {
  // M5 Pro, "PMP0 / SOC Floor / ANE-LNK0-AF-BW".
  const std::vector<SocState> idle = {{"VMIN", 100}, {"VOVD", 0},
                                      {"VMAX", 0}};
  const std::vector<SocState> busy = {{"VMIN", 0}, {"VOVD", 81},
                                      {"VMAX", 19}};
  const std::vector<SocState> half = {{"VMIN", 50}, {"VOVD", 50}};
  auto a = soc_floor_active_fraction(idle);
  auto b = soc_floor_active_fraction(busy);
  auto c = soc_floor_active_fraction(half);
  EXPECT_TRUE(a && near_(*a, 0.0));
  EXPECT_TRUE(b && near_(*b, 1.0));
  EXPECT_TRUE(c && near_(*c, 0.5));
}

TEST(soc_activity, floor_silent_or_foreign_is_no_reading) {
  // A channel that reported nothing this interval is SILENT, not idle:
  // the caller must not turn it into 0%.
  const std::vector<SocState> silent = {{"VMIN", 0}, {"VOVD", 0}};
  EXPECT_FALSE(soc_floor_active_fraction(silent).has_value());
  // "DCS Floor / ANE-DCS-BW" counts F1..F8, not VMIN.. -- a different
  // scale, which this helper refuses rather than misreads.
  const std::vector<SocState> dcs = {{"F1", 20}, {"F8", 80}};
  EXPECT_FALSE(soc_floor_active_fraction(dcs).has_value());
}

TEST(soc_activity, bandwidth_bin_labels) {
  auto two = soc_bw_bin_gbps("2GB/s");
  auto ten = soc_bw_bin_gbps("10GB/s");
  auto half = soc_bw_bin_gbps("512MB/s");
  EXPECT_TRUE(two && near_(*two, 2.0));
  EXPECT_TRUE(ten && near_(*ten, 10.0));
  EXPECT_TRUE(half && near_(*half, 0.5));
  // The M4 Pro (macOS 26) right-aligns its labels.
  auto padded = soc_bw_bin_gbps("   2GB/s");
  auto padded16 = soc_bw_bin_gbps("  16GB/s");
  EXPECT_TRUE(padded && near_(*padded, 2.0));
  EXPECT_TRUE(padded16 && near_(*padded16, 16.0));
  EXPECT_FALSE(soc_bw_bin_gbps("VMIN").has_value());
  EXPECT_FALSE(soc_bw_bin_gbps("   ").has_value());
  EXPECT_FALSE(soc_bw_bin_gbps("GB/s").has_value());
  EXPECT_FALSE(soc_bw_bin_gbps("8GB/sX").has_value());
}

TEST(soc_activity, bandwidth_histogram_mean) {
  // M5 Pro, "PMP0 / AF BW / ANE L0 RD+WR" under the bench, deliberately
  // out of order: bins stand for the midpoint below their edge, so
  // (4*1 + 1*5 + 63*7 + 7*9) / 75.
  const std::vector<SocState> busy = {{"8GB/s", 63}, {"2GB/s", 4},
                                      {"12GB/s", 0}, {"6GB/s", 1},
                                      {"10GB/s", 7}, {"4GB/s", 0}};
  auto m = soc_bw_histogram_mean_gbps(busy);
  EXPECT_TRUE(m && near_(*m, 513.0 / 75.0));
  // A channel that does report at idle sits wholly in the lowest bin.
  const std::vector<SocState> low = {{"2GB/s", 100}, {"4GB/s", 0}};
  auto l = soc_bw_histogram_mean_gbps(low);
  EXPECT_TRUE(l && near_(*l, 1.0));
  // An idle ANE's histograms usually report nothing at all.
  const std::vector<SocState> silent = {{"2GB/s", 0}, {"4GB/s", 0}};
  EXPECT_FALSE(soc_bw_histogram_mean_gbps(silent).has_value());
}

TEST(soc_activity, energy_gate_live_counter) {
  // M4 Pro, macOS 26: the channel advances every interval while its block
  // draws power. The first move is not read (no history to tell it from
  // a lump); from the second consecutive one on, every sample is.
  using V = SocEnergyGate::Verdict;
  SocEnergyGate g;
  EXPECT_TRUE(g.feed(true) == V::kUnconfirmed);
  EXPECT_TRUE(g.feed(true) == V::kLive);
  EXPECT_TRUE(g.feed(true) == V::kLive);
  // Once trusted, a zero is an idle block -- 0 W, a real reading.
  EXPECT_TRUE(g.feed(false) == V::kLive);
  EXPECT_TRUE(g.feed(false) == V::kLive);
  EXPECT_TRUE(g.feed(true) == V::kLive);
}

TEST(soc_activity, energy_gate_idle_block_then_busy) {
  // The M4 Pro's ANE channel reads exactly zero while the ANE idles, so
  // from a cold start it is unconfirmed until the ANE works for two
  // intervals running -- then it is live, idle zeros included.
  using V = SocEnergyGate::Verdict;
  SocEnergyGate g;
  for (int i = 0; i < 5; ++i) { EXPECT_TRUE(g.feed(false) == V::kUnconfirmed); }
  EXPECT_TRUE(g.feed(true) == V::kUnconfirmed);
  EXPECT_TRUE(g.feed(true) == V::kLive);
  EXPECT_TRUE(g.feed(false) == V::kLive);
}

TEST(soc_activity, energy_gate_never_trusts_the_macos27_lump) {
  // MEASURED on the M5 Pro, macOS 27: ANE0 zero in 676 of 677 one-second
  // samples and ONE carrying 461 J -- minutes of energy that a per-second
  // reader shows as a 461 W ANE. It must never be read, and neither must
  // the zeros around it, which are silence rather than an idle ANE.
  SocEnergyGate g;
  int live = 0;
  for (int i = 0; i < 677; ++i) {
    if (g.feed(i == 580) == SocEnergyGate::Verdict::kLive) { ++live; }
  }
  EXPECT_TRUE(live == 0);
  EXPECT_FALSE(g.trusted());
}

TEST(soc_activity, energy_gate_short_lived_reader) {
  // `vpipe --gpu-thermal` lives for one 600 ms window, ~7 polls, and
  // starts with no history. On macOS 27 the lump can land in ANY of
  // them, including the first; it must never come out as a reading.
  for (int at = 0; at < 7; ++at) {
    SocEnergyGate g;
    int live = 0;
    for (int i = 0; i < 7; ++i) {
      if (g.feed(i == at) == SocEnergyGate::Verdict::kLive) { ++live; }
    }
    EXPECT_TRUE(live == 0);
  }
}

TEST(soc_activity, energy_gate_is_per_source) {
  // The case the group-wide gate got wrong: on macOS 27 under GPU load
  // the "GPU Energy" twin moves every second while ANE0 stays stale. A
  // gate per source trusts the one and not the other.
  using V = SocEnergyGate::Verdict;
  SocEnergyGate gpu_twin, ane;
  V last_twin = V::kUnconfirmed, lump = V::kLive;
  for (int i = 0; i < 20; ++i) {
    last_twin = gpu_twin.feed(true);
    const V a = ane.feed(i == 12);
    if (i == 12) { lump = a; }
  }
  EXPECT_TRUE(last_twin == V::kLive);
  EXPECT_TRUE(lump == V::kUnconfirmed);
}
