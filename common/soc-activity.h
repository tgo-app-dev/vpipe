// soc-activity.h -- reading how busy a SoC block is from IOReport's
// power-manager (PMP) channels, for when its energy counter says nothing.
//
// The ANE meter was power over a per-chip ceiling: an "Energy Model"
// delta, divided by time. On macOS 27 that counter is not live. MEASURED
// on an M5 Pro: a one-second logger over 11.5 minutes saw the WHOLE group
// -- GPU0 and every CPU core as well as ANE0 -- report zero in 676 of 677
// samples, then publish once, all at once (GPU 24446 J, ANE0 461 J). A
// reader that divides a delta by one second shows 0 W, and then one
// absurd spike.
//
// What does update every second is the power manager's view of the same
// hardware, in state (residency) channels of group "PMP" / "PMP<n>":
//
//   SOC Floor / ANE-LNK0-AF-BW   the fabric floor the ANE's link holds.
//                                Idle it sits at VMIN; MEASURED under a
//                                pure ANE load on the M5 Pro: VMIN 0%,
//                                VOVD 81%, VMAX 19%.
//   AF BW / ANE L0 RD+WR (...)   the ANE's fabric bandwidth, as residency
//                                over "2GB/s" .. bins. Silent at idle,
//                                mostly the 6-10 GB/s bins under load.
//
// Neither is the same measurement as power, and neither holds on every
// machine: on an M4 Pro (macOS 26) the SOC Floor ANE channels stay at
// VMIN under load while the energy counter IS live. So a reader takes
// both and uses whichever moved -- a source that is dead on this machine
// reads zero, not garbage, and needs no per-chip table.

#ifndef VPIPE_SOC_ACTIVITY_H
#define VPIPE_SOC_ACTIVITY_H

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace vpipe {

// One state of an IOReport state channel over a sample delta.
struct SocState {
  std::string_view name;
  std::int64_t     residency = 0;
};

// The share of a floor channel's time spent ABOVE its idle state, 0..1.
// nullopt when the channel has no "VMIN" state (it is some other kind of
// channel) or reported no residency at all this sample (it was silent,
// which is not the same as idle).
inline std::optional<double>
soc_floor_active_fraction(std::span<const SocState> states)
{
  bool         has_idle = false;
  std::int64_t idle = 0, total = 0;
  for (const SocState& s : states) {
    if (s.residency > 0) { total += s.residency; }
    if (s.name == "VMIN") {
      has_idle = true;
      idle     = std::max<std::int64_t>(s.residency, 0);
    }
  }
  if (!has_idle || total <= 0) { return std::nullopt; }
  return 1.0 - (double)idle / (double)total;
}

// A bandwidth bin's label, "8GB/s" or "512MB/s", in GB/s. nullopt for a
// label that is not a bandwidth. Surrounding blanks are ignored: the M4
// Pro (macOS 26) right-aligns its labels as "   2GB/s", the M5 Pro
// (macOS 27) does not, and a parser that expects the digit first reads
// every M4 bin as not-a-bandwidth.
inline std::optional<double>
soc_bw_bin_gbps(std::string_view label)
{
  while (!label.empty() && label.front() == ' ') { label.remove_prefix(1); }
  while (!label.empty() && label.back() == ' ')  { label.remove_suffix(1); }
  std::size_t i = 0;
  double      v = 0.0;
  bool        digits = false;
  for (; i < label.size() && label[i] >= '0' && label[i] <= '9'; ++i) {
    v = v * 10.0 + (double)(label[i] - '0');
    digits = true;
  }
  if (!digits) { return std::nullopt; }
  const std::string_view unit = label.substr(i);
  if (unit == "GB/s") { return v; }
  if (unit == "MB/s") { return v / 1024.0; }
  return std::nullopt;
}

// Mean bandwidth of one histogram channel, in GB/s. The labels name each
// bin's UPPER edge (the lowest is "2GB/s", and an idle channel that does
// report sits entirely in it), so a bin stands for the midpoint between
// its edge and the one below -- the first bin is 0..2 GB/s. nullopt when
// the channel reported nothing, which is what an idle ANE's histograms
// usually do.
inline std::optional<double>
soc_bw_histogram_mean_gbps(std::span<const SocState> states)
{
  struct Bin { double edge; std::int64_t res; };
  std::vector<Bin> bins;
  bins.reserve(states.size());
  for (const SocState& s : states) {
    if (auto e = soc_bw_bin_gbps(s.name)) {
      bins.push_back({*e, std::max<std::int64_t>(s.residency, 0)});
    }
  }
  std::sort(bins.begin(), bins.end(),
            [](const Bin& a, const Bin& b) { return a.edge < b.edge; });
  double       below = 0.0, weighted = 0.0;
  std::int64_t total = 0;
  for (const Bin& b : bins) {
    weighted += 0.5 * (below + b.edge) * (double)b.res;
    total    += b.res;
    below     = b.edge;
  }
  if (total <= 0) { return std::nullopt; }
  return weighted / (double)total;
}

// Whether one energy channel's sample may be read as power.
//
// On macOS 27 the "Energy Model" blocks are not live: the M5 Pro's GPU0,
// ANE0 and CPU channels sat at zero for minutes and then published all
// of it at once -- GPU 24446 J and ANE0 461 J in a single one-second
// delta, which a reader dividing by the interval shows as a 461 W ANE.
// Such a sample is a LUMP. Its twin "GPU Energy" (the same GPU energy in
// nanojoules) IS live there under load, ~27 J every second, so the
// liveness of the GROUP says nothing about the liveness of one channel:
// judging it by "anything moved" let a busy GPU vouch for the ANE's
// lump. The gate therefore watches ONE source -- the channels whose
// energy it is about to read.
//
// A live counter advances every interval while its block draws power:
// MEASURED on an M4 Pro (macOS 26), the group moved in all 200 samples
// polled 20 ms apart. A stale one advances in isolated jumps minutes
// apart. So a source is trusted once it has moved in TWO CONSECUTIVE
// samples, and from then on a zero is a real zero (an idle block) rather
// than silence. Until then nothing it reports is read -- including the
// very first move, which with no history cannot be told from a lump; a
// short-lived reader (`vpipe --gpu-thermal` samples for 600 ms and
// exits) starts every run that way.
class SocEnergyGate {
public:
  // kLive: this sample's value, zero included, is the interval's energy.
  // kUnconfirmed: not known to be live; the value must not be read.
  enum class Verdict { kLive, kUnconfirmed };

  Verdict feed(bool moved) {
    if (moved) {
      if (_prev_moved) { _trusted = true; }
      _prev_moved = true;
    } else {
      _prev_moved = false;
    }
    return _trusted ? Verdict::kLive : Verdict::kUnconfirmed;
  }

  bool trusted() const { return _trusted; }

private:
  bool _prev_moved = false;
  bool _trusted    = false;
};

}

#endif
