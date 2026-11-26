// train-lora as one RANK of a run split across hosts (`tensor_parallel`;
// train-lora-stage.h, "A RANK OF A SPLIT RUN"). What crosses between the
// ranks here, apart from a step's own partial sums (the family's):
//
//   - the DATASET, once, before any model loads: rank 0's encoded
//     captions, latents, soundtracks, keyframes and reference rows, with
//     the run's settings -- a JSON header and the tensors' bytes behind
//     it. A follower trains on exactly what rank 0 encoded, so the ranks'
//     residual streams -- whole on every rank -- are the same stream;
//   - a CONTROL record before each step: whether any rank stops, saves or
//     previews now;
//   - the SHARED factors' gradients after each step's backward, summed;
//   - the gradient's norm, each rank's share counted once;
//   - the ADAPTER, gathered whole for a checkpoint.
//
// Two ranks: a transport hands each the other's bytes, so raw bytes and
// f32 sums cross alike. More ranks would sum what is meant to be copied;
// the stage refuses them.

#include "stages/train-lora-stage.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-registry.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/shared/tp-overlap.h"
#endif

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace vpipe {

#ifdef VPIPE_BUILD_APPLE_SILICON

namespace fs = std::filesystem;

namespace {

// A bf16 cache entry back as f32, as settle_cache_ reads one.
bool
entry_f32_(const genai::train::SampleCache& cache, const std::string& key,
           std::vector<float>* out, std::string* err)
{
  const auto* e = cache.find(key);
  if (e == nullptr) {
    *err = "the cached entry " + key + " vanished";
    return false;
  }
  std::vector<std::uint16_t> b((std::size_t)e->bytes / 2);
  if (!cache.read(key, b.data(), (std::size_t)e->bytes, err)) { return false; }
  out->resize(b.size());
  for (std::size_t i = 0; i < b.size(); ++i) {
    (*out)[i] = genai::train::bf16_to_f32(b[i]);
  }
  return true;
}

FlexData
span_(std::size_t off, std::size_t n)
{
  FlexData a = FlexData::make_array();
  a.as_array().push_back((std::uint64_t)off);
  a.as_array().push_back((std::uint64_t)n);
  return a;
}

FlexData
ints_(const std::vector<int>& v)
{
  FlexData a = FlexData::make_array();
  for (int i : v) { a.as_array().push_back((std::int64_t)i); }
  return a;
}

std::vector<int>
ints_of_(const FlexData& f)
{
  std::vector<int> v;
  if (!f.is_array()) { return v; }
  auto a = f.as_array();
  for (std::size_t i = 0; i < a.size(); ++i) {
    v.push_back((int)a.at(i).as_int(0));
  }
  return v;
}

std::int64_t
geti_(const FlexData& o, const char* k, std::int64_t d = 0)
{
  FlexData c = o;
  auto ob = c.as_object();
  return ob.contains(k) ? ob.at(k).as_int(d) : d;
}

double
getr_(const FlexData& o, const char* k, double d = 0.0)
{
  FlexData c = o;
  auto ob = c.as_object();
  return ob.contains(k) ? ob.at(k).as_real(d) : d;
}

FlexData
get_(const FlexData& o, const char* k)
{
  FlexData c = o;
  auto ob = c.as_object();
  return ob.contains(k) ? ob.at(k) : FlexData();
}

// FNV-1a over bytes; and a word at a time over a long arena.
std::uint64_t
fnv_(const void* p, std::size_t n, std::uint64_t h = 1469598103934665603ull)
{
  const auto* b = static_cast<const std::uint8_t*>(p);
  for (std::size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}

std::uint64_t
words_(const float* p, std::size_t n, std::uint64_t h)
{
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    std::uint64_t w;
    std::memcpy(&w, p + i, 8);
    h = (h ^ w) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
  }
  if (i < n) { h = fnv_(p + i, 4, h); }
  return h;
}

std::uint64_t
str_digest_(std::string_view s)
{
  return fnv_(s.data(), s.size());
}

// A checkpoint's identity -- every safetensors file's header (names,
// dtypes, shapes, offsets), its size and 4 KB of its tensor data at the
// data's start, middle and end, in no order of their names: another
// model, quantization or pruning differs, and so does another PARTITION,
// which the header alone does not tell (MiniMax H3's FL2VA and Ref2VA
// DiTs have byte-identical headers and differ in every weight; so does a
// fine-tune re-saved over the same shapes). The same release under
// another folder name, or a copy, does not. Three reads a file.
// 0 where nothing is there to read.
std::uint64_t
headers_digest_(const std::string& path)
{
  std::error_code ec;
  std::vector<fs::path> files;
  if (fs::is_regular_file(path, ec)) {
    files.push_back(path);
  } else if (fs::is_directory(path, ec)) {
    for (const auto& e : fs::directory_iterator(path, ec)) {
      if (e.path().extension() == ".safetensors") {
        files.push_back(e.path());
      }
    }
  }
  std::vector<std::uint64_t> ds;
  for (const fs::path& f : files) {
    std::ifstream in(f, std::ios::binary);
    std::uint64_t len = 0;
    if (!in.read(reinterpret_cast<char*>(&len), 8) || len > (1u << 28)) {
      continue;
    }
    std::string hdr((std::size_t)len, '\0');
    if (!in.read(hdr.data(), (std::streamsize)len)) { continue; }
    std::uint64_t d = fnv_(hdr.data(), hdr.size());
    const std::uint64_t sz = (std::uint64_t)fs::file_size(f, ec);
    d = fnv_(&sz, 8, d);
    const std::uint64_t base = 8 + len;
    if (sz > base) {
      const std::uint64_t span = sz - base;
      const std::uint64_t n = std::min<std::uint64_t>(span, 4096);
      const std::uint64_t at[3] = {base, base + (span - n) / 2, sz - n};
      std::string buf((std::size_t)n, '\0');
      for (const std::uint64_t a : at) {
        // A read that fails leaves the digest short of it, so the two
        // ranks differ and the check names the DiT.
        if (!in.seekg((std::streamoff)a) ||
            !in.read(buf.data(), (std::streamsize)n)) {
          break;
        }
        d = fnv_(buf.data(), buf.size(), d);
      }
    }
    ds.push_back(d);
  }
  std::sort(ds.begin(), ds.end());
  return ds.empty() ? 0 : fnv_(ds.data(), ds.size() * 8);
}

// THE BYTE PROBE, which a transport that copies returns as it went: -0,
// denormals (a dataset's small sizes read as bf16 are such), NaNs with
// payloads, as bf16 and as f32 words.
constexpr std::uint64_t kProbe[4] = {
    0x7FC1FF8100648000ull,   // bf16: -0, 100 (a denormal), NaN, -NaN
    0x7F810001FFFF0080ull,   // bf16: signalling NaN, denormals, -NaN
    0x7FC0012380000000ull,   // f32: -0, a quiet NaN's payload
    0xFFA0000100000001ull};  // f32: the smallest denormal, -sNaN

// What crosses as a handshake.
enum Hs : int {
  kHsMagic, kHsMode, kHsRank, kHsRanks, kHsHeads, kHsHead0, kHsHead1,
  kHsFfn, kHsFfn0, kHsFfn1, kHsExtra, kHsK0, kHsK1, kHsEvery,
  kHsProbe, kHsWords = kHsProbe + 4};
// 3: the resume exchange carries kResumeKept steps, and a split forward
// meets (set_tp_meet) -- a rank of 2 pairs neither, so it is refused
// here by name rather than by a size at the resume or a forward.
// 4: the layout check digests the kind as each trainer trains it
// (layout_type) and a sample of the DiT's data -- a rank of 3 would read
// every pair as another checkpoint or kind, so it is refused here.
constexpr std::uint64_t kHsVersion = 0x5650544C00000004ull;  // "VPTL" 4

}  // namespace

bool
TrainLoraStage::tp_handshake_(std::string* err)
{
  std::uint64_t mine[kHsWords] = {};
  mine[kHsMagic] = kHsVersion;
  mine[kHsMode] = dp_rank_() ? 2 : 1;
  mine[kHsRank] = (std::uint64_t)rank_index_();
  mine[kHsRanks] = 2;
  if (tp_split_()) {
    mine[kHsHeads] = (std::uint64_t)_tp.n_heads;
    mine[kHsHead0] = (std::uint64_t)_tp.head0;
    mine[kHsHead1] = (std::uint64_t)_tp.head1;
    mine[kHsFfn] = (std::uint64_t)_tp.n_ffn;
    mine[kHsFfn0] = (std::uint64_t)_tp.ffn0;
    mine[kHsFfn1] = (std::uint64_t)_tp.ffn1;
    mine[kHsExtra] = _tp.extra.is_object()
                         ? str_digest_(_tp.extra.to_json())
                         : 0;
  } else {
    mine[kHsK0] = (std::uint64_t)_dp_k[0];
    mine[kHsK1] = (std::uint64_t)_dp_k[1];
  }
  mine[kHsEvery] = (std::uint64_t)_tp_check_every;
  for (int i = 0; i < 4; ++i) { mine[kHsProbe + i] = kProbe[i]; }
  // As bf16 and as f32: the two kinds of record this run sends.
  std::uint64_t got[2][kHsWords] = {};
  auto* mc = session()->services()->metal_compute();
  genai::TpOverlap pipe(mc, _tp_ch, 1, genai::TpDType::kBF16);
  if (!pipe.valid()) {
    *err = "no thread for the handshake";
    return false;
  }
  genai::TpBulk bulk(mc);
  bool ok = bulk.exchange(pipe, mine, got[0], sizeof mine,
                          genai::TpDType::kBF16, &_tp_seq, err) &&
            bulk.exchange(pipe, mine, got[1], sizeof mine,
                          genai::TpDType::kF32, &_tp_seq, err);
  bulk.forget(pipe);
  if (!ok) {
    *err = "the handshake with the other rank: " + *err;
    return false;
  }
  for (int k = 0; k < 2; ++k) {
    for (int i = 0; i < 4; ++i) {
      if (got[k][kHsProbe + i] != kProbe[i]) {
        *err = fmt("the transport altered bytes it carried ({} word {}: "
                   "{:016x} for {:016x}) -- it sums, where a run's records "
                   "need its peer's bytes as they went",
                   k == 0 ? "bf16" : "f32", i, got[k][kHsProbe + i],
                   kProbe[i])();
        return false;
      }
    }
  }
  const std::uint64_t* o = got[1];
  auto refuse = [&](const std::string& what) {
    *err = "the two ranks are not configured as one run: " + what;
    return false;
  };
  if (o[kHsMagic] != kHsVersion) {
    return refuse("the other rank's vpipe speaks another protocol");
  }
  if (o[kHsMode] != mine[kHsMode]) {
    return refuse(fmt("this rank is {}, the other {}", par_key_(),
                      o[kHsMode] == 2 ? "data_parallel"
                                      : "tensor_parallel")());
  }
  if (o[kHsRank] == mine[kHsRank]) {
    return refuse(fmt("both are rank {}", mine[kHsRank])());
  }
  if (o[kHsEvery] != mine[kHsEvery]) {
    return refuse(fmt("check_every {} here, {} there", mine[kHsEvery],
                      o[kHsEvery])());
  }
  if (tp_split_()) {
    const std::uint64_t* r0 = _tp.rank == 0 ? mine : o;
    const std::uint64_t* r1 = _tp.rank == 0 ? o : mine;
    if (o[kHsHeads] != mine[kHsHeads] || o[kHsFfn] != mine[kHsFfn]) {
      return refuse(fmt("their models' widths differ ({} heads and {} rows "
                        "here, {} and {} there)", mine[kHsHeads],
                        mine[kHsFfn], o[kHsHeads], o[kHsFfn])());
    }
    if (r0[kHsHead0] != 0 || r0[kHsHead1] != r1[kHsHead0] ||
        r1[kHsHead1] != mine[kHsHeads] || r0[kHsFfn0] != 0 ||
        r0[kHsFfn1] != r1[kHsFfn0] || r1[kHsFfn1] != mine[kHsFfn]) {
      return refuse(fmt("their shares do not tile the model (heads [{}, {}) "
                        "and [{}, {}) of {}, feed-forward rows [{}, {}) and "
                        "[{}, {}) of {})", r0[kHsHead0], r0[kHsHead1],
                        r1[kHsHead0], r1[kHsHead1], mine[kHsHeads],
                        r0[kHsFfn0], r0[kHsFfn1], r1[kHsFfn0], r1[kHsFfn1],
                        mine[kHsFfn])());
    }
    if (o[kHsExtra] != mine[kHsExtra]) {
      return refuse("their shares' extras differ (min_band_rows and the "
                    "like: the bands each cuts, so the exchanges)");
    }
  } else if (o[kHsK0] != mine[kHsK0] || o[kHsK1] != mine[kHsK1]) {
    return refuse(fmt("samples [{}, {}] here, [{}, {}] there", mine[kHsK0],
                      mine[kHsK1], o[kHsK0], o[kHsK1])());
  }
  return true;
}

bool
TrainLoraStage::tp_check_layout_(std::string* err)
{
  // What each rank trains: its DiT's checkpoint, the frozen adapters
  // under it, the modules -- names, the rank, the shared factors' runs
  // (a split run's own factors differ in width by share; a data-parallel
  // run's arenas are one size).
  enum { kDit, kType, kBase, kMods, kRank, kShared, kA, kB, kN };
  std::uint64_t mine[kN] = {};
  mine[kDit] = headers_digest_(_dit_dir);
  // The kind as each trainer TRAINS it (layout_type) -- never the
  // registry or folder names, nor a partition word that differs from
  // machine to machine (a link, a copy, a record or none) for the same
  // checkpoint where the training does not.
  mine[kType] = str_digest_(_family + "|" + _trainer->layout_type() +
                            "|" + _trainer->adapter_model_type());
  {
    std::uint64_t h = 1469598103934665603ull;
    for (const std::string& b : run_paths_().base_loras) {
      const std::uint64_t d = headers_digest_(b);
      h = fnv_(&d, 8, h);
    }
    mine[kBase] = h;
  }
  {
    std::uint64_t h = 1469598103934665603ull;
    for (const auto& m : _params.modules()) { h = fnv_(m.name.data(),
                                                       m.name.size(), h); }
    mine[kMods] = h;
  }
  mine[kRank] = (std::uint64_t)_params.rank();
  {
    std::uint64_t h = 1469598103934665603ull;
    for (const auto& sp : _params.shared_grad_spans()) {
      const std::uint64_t v = ((std::uint64_t)sp.n << 1) | (sp.b ? 1 : 0);
      h = fnv_(&v, 8, h);
    }
    mine[kShared] = h;
  }
  if (dp_rank_()) {
    mine[kA] = _params.a_elems();
    mine[kB] = _params.b_elems();
  }
  std::uint64_t o[kN] = {};
  if (!bulk_(mine, o, sizeof mine, genai::TpDType::kF32, err)) {
    *err = "comparing the ranks' models: " + *err;
    return false;
  }
  auto refuse = [&](const std::string& what) {
    *err = "the two ranks are not training the same thing: " + what;
    return false;
  };
  // Split, the two halves are one model; data parallel, each trains its
  // samples on its own copy (a host may hold it compressed), said.
  if (o[kDit] != mine[kDit]) {
    if (tp_split_()) {
      return refuse("their DiTs' checkpoints differ (another model, "
                    "partition, quantization or pruning)");
    }
    session()->warn(fmt("TrainLoraStage('{}'): the two hosts' DiT "
                        "checkpoints differ (a compressed copy?); each "
                        "trains its samples on its own", this->id()));
  }
  if (o[kType] != mine[kType]) {
    return refuse("they train their models as different kinds (one "
                  "MiniMax H3 rank Ref2VA, the other not, say)");
  }
  if (o[kBase] != mine[kBase]) {
    return refuse("their frozen adapters (base_lora) differ");
  }
  if (o[kMods] != mine[kMods] || o[kRank] != mine[kRank] ||
      o[kShared] != mine[kShared] || o[kA] != mine[kA] ||
      o[kB] != mine[kB]) {
    return refuse("their adapters are laid out differently (modules, "
                  "rank or shared factors)");
  }
  return true;
}

std::uint64_t
TrainLoraStage::masters_digest_() const
{
  const auto* a = static_cast<const float*>(_params.a_master().contents());
  const auto* b = static_cast<const float*>(_params.b_master().contents());
  std::uint64_t h = 1469598103934665603ull;
  if (tp_split_()) {
    for (const auto& sp : _params.shared_grad_spans()) {
      h = words_((sp.b ? b : a) + sp.off, sp.n, h);
    }
  } else {
    if (_params.a_elems() > 0) { h = words_(a, _params.a_elems(), h); }
    if (_params.b_elems() > 0) { h = words_(b, _params.b_elems(), h); }
  }
  // Never 0: that is "no check this step".
  return h == 0 ? 1 : h;
}

std::size_t
TrainLoraStage::split_reserve_bytes_() const
{
  if (tp_split_()) {
    std::size_t n = 0;
    for (const auto& sp : _params.shared_grad_spans()) { n += sp.n; }
    return 2 * n * 4 + (_trainer ? _trainer->split_exchange_bytes() : 0);
  }
  if (dp_rank_()) {
    // The sync's host copy (the larger arena), the run's own bulk's two
    // chunks.
    return std::max(_params.a_elems(), _params.b_elems()) * 4 +
           2 * ((std::size_t)32 << 20);
  }
  return 0;
}

bool
TrainLoraStage::tp_open_(std::string* err)
{
  _tp_check_every =
      std::max(0, (int)geti_(par_cfg_(), "check_every", 50));
  if (_dp_cfg.is_object()) { return dp_open_(err) && tp_handshake_(err); }
  auto o = _tp_cfg.as_object();
  // The family's widths, which the share tiles.
  const ResolvedModel rm = resolve_model(session(), _hf_dir);
  const std::string root =
      rm.dir.empty() ? resolve_model_dir(session(), _hf_dir) : rm.dir;
  genai::train::RunShape run = run_paths_();
  const genai::train::LoraFamily* fam = nullptr;
  for (const genai::train::LoraFamily* f : genai::train::lora_families()) {
    std::string why;
    if (f->claims(root, rm.model_type, run, &why)) { fam = f; break; }
    if (!why.empty()) {
      *err = "'" + _hf_dir + "': " + why;
      return false;
    }
  }
  if (fam == nullptr) {
    *err = "'" + _hf_dir + "' is not a model this stage can train";
    return false;
  }
  int nh = 0, nf = 0;
  if (!fam->split_dims(root, rm.model_type, run, &nh, &nf, err)) {
    return false;
  }
  genai::TpShare share;
  if (!genai::TpShare::from_flex(_tp_cfg, nh, nf, &share, err)) {
    return false;
  }
  if (share.ranks != 2) {
    *err = fmt("a split run is two ranks ({} given)", share.ranks)();
    return false;
  }
  const std::string tname = o.contains("transport")
                                ? std::string(o.at("transport").as_string(""))
                                : std::string();
  genai::TpTransport* tr = genai::find_tp_transport(tname);
  if (tr == nullptr) {
    *err = fmt("no transport named '{}' -- a plugin registers it", tname)();
    return false;
  }
  const FlexData params = o.contains("params") ? o.at("params") : FlexData();
  std::string terr;
  try {
    _tp_ch = tr->open(params, share, &terr);
  } catch (const std::exception& e) {
    _tp_ch.reset();
    terr = std::string("it threw: ") + e.what();
  } catch (...) {
    _tp_ch.reset();
    terr = "it threw";
  }
  if (!_tp_ch) {
    *err = fmt("transport '{}': {}", tname, terr)();
    return false;
  }
  _tp = share;
  _tp_seq = 0;
  session()->info(fmt(
      "TrainLoraStage('{}'): rank {} of {}: heads [{}, {}) of {}, "
      "feed-forward rows [{}, {}) of {}, through '{}'", this->id(),
      share.rank, share.ranks, share.head0, share.head1, share.n_heads,
      share.ffn0, share.ffn1, share.n_ffn, tname));
  return tp_handshake_(err);
}

void
TrainLoraStage::tp_close_()
{
  if (_trainer && tp_split_()) { _trainer->split_end(); }
  // A data-parallel run's own overlap: its buffers told to the channel
  // before either goes.
  if (_dp_pipe) {
    if (_dp_bulk) { _dp_bulk->forget(*_dp_pipe); }
    _dp_pipe.reset();
  }
  _dp_bulk.reset();
  // Its stop (a reference to the run's context) cleared before it goes.
  if (_tp_ch) {
    try {
      _tp_ch->set_stop({});
    } catch (...) {
    }
  }
  _tp_ch.reset();
  _tp_grace_ms.reset();
  // Commands taken for an agreement that will not come now.
  for (auto& c : _tp_pending) { c->fail("the run ended"); }
  _tp_pending.clear();
}

bool
TrainLoraStage::dp_open_(std::string* err)
{
  auto o = _dp_cfg.as_object();
  const int rank = (int)geti_(_dp_cfg, "rank", -1);
  const int ranks = (int)geti_(_dp_cfg, "ranks", 2);
  if (ranks != 2 || rank < 0 || rank >= ranks) {
    *err = fmt("a data-parallel run is two ranks (rank {} of {} given)",
               rank, ranks)();
    return false;
  }
  if (o.contains("samples")) {
    const std::vector<int> k = ints_of_(o.at("samples"));
    if (k.size() != 2 || k[0] < 1 || k[1] < 1) {
      *err = "samples: two counts of at least 1, rank 0's then rank 1's";
      return false;
    }
    _dp_k[0] = k[0];
    _dp_k[1] = k[1];
  }
  const std::string tname = o.contains("transport")
                                ? std::string(o.at("transport").as_string(""))
                                : std::string();
  genai::TpTransport* tr = genai::find_tp_transport(tname);
  if (tr == nullptr) {
    *err = fmt("no transport named '{}' -- a plugin registers it", tname)();
    return false;
  }
  // The channel's ranks; nothing of the model is shared out.
  genai::TpShare share;
  share.rank = rank;
  share.ranks = ranks;
  const FlexData params = o.contains("params") ? o.at("params") : FlexData();
  std::string terr;
  try {
    _tp_ch = tr->open(params, share, &terr);
  } catch (const std::exception& e) {
    _tp_ch.reset();
    terr = std::string("it threw: ") + e.what();
  } catch (...) {
    _tp_ch.reset();
    terr = "it threw";
  }
  if (!_tp_ch) {
    *err = fmt("transport '{}': {}", tname, terr)();
    return false;
  }
  _dp_rank = rank;
  _tp_seq = 0;
  session()->info(fmt(
      "TrainLoraStage('{}'): data-parallel rank {} of {}: {} and {} "
      "batches of each step, through '{}'", this->id(), rank, ranks,
      _dp_k[0], _dp_k[1], tname));
  return true;
}

bool
TrainLoraStage::bulk_(const void* mine, void* others, std::size_t bytes,
                      genai::TpDType dtype, std::string* err)
{
  if (tp_split_()) {
    return _trainer->split_bulk(mine, others, bytes, dtype, err);
  }
  if (!_tp_ch) {
    *err = "no channel to the other rank";
    return false;
  }
  // The run's own overlap, made at its first exchange after the dataset's
  // (whose overlap let go of the channel's buffers as it went).
  if (!_dp_pipe) {
    auto* mc = session()->services()->metal_compute();
    _dp_pipe = std::make_unique<genai::TpOverlap>(mc, _tp_ch, 1,
                                                  genai::TpDType::kF32);
    if (!_dp_pipe->valid()) {
      _dp_pipe.reset();
      *err = "no thread for the run's exchanges";
      return false;
    }
    _dp_bulk = std::make_unique<genai::TpBulk>(mc);
  }
  return _dp_bulk->exchange(*_dp_pipe, mine, others, bytes, dtype, &_tp_seq,
                            err);
}

bool
TrainLoraStage::dp_sync_gradients_(double* sums, int n, std::string* err)
{
  // Every factor's gradient, A's then B's, summed with the other rank's
  // (its samples'): both then hold the step's whole gradient, in the same
  // order of addition (a + b and b + a are one float).
  const std::pair<const metal_compute::SharedBuffer*, std::size_t> gs[] = {
      {&_params.a_grad(), _params.a_elems()},
      {&_params.b_grad(), _params.b_elems()}};
  // The peer's into one host copy, sized once at the larger arena.
  const std::size_t most = std::max(_params.a_elems(), _params.b_elems());
  if (_sync_others.size() < most) { _sync_others.resize(most); }
  for (const auto& [g, ne] : gs) {
    if (ne == 0) { continue; }
    auto* mine = static_cast<float*>(g->contents());
    float* others = _sync_others.data();
    if (!bulk_(mine, others, ne * 4, genai::TpDType::kF32, err)) {
      return false;
    }
    for (std::size_t i = 0; i < ne; ++i) { mine[i] += others[i]; }
  }
  std::vector<double> o((std::size_t)n);
  if (!bulk_(sums, o.data(), (std::size_t)n * 8, genai::TpDType::kF32,
             err)) {
    return false;
  }
  for (int i = 0; i < n; ++i) { sums[i] += o[(std::size_t)i]; }
  return true;
}

namespace {

// THE DATASET'S BYTES as pieces in the order the header's spans name
// them, sent in CHUNKS of whole pieces -- a run of small ones up to
// kChunk together, a larger one alone -- cut by the pieces' sizes alone,
// so both ranks cut alike: neither side ever holds the dataset twice
// (rank 0 a chunk's staging, and an entry a disk cache read back at a
// time; a follower each tensor in its own buffer as it lands).
constexpr std::size_t kChunk = (std::size_t)32 << 20;

template <typename F>
void
for_chunks_(const std::vector<std::size_t>& sizes, F&& f)
{
  std::size_t i = 0;
  while (i < sizes.size()) {
    if (sizes[i] > kChunk) {
      f(i, i + 1);
      ++i;
      continue;
    }
    std::size_t j = i, sum = 0;
    while (j < sizes.size() && sizes[j] <= kChunk &&
           sum + sizes[j] <= kChunk) {
      sum += sizes[j++];
    }
    f(i, j);
    i = j;
  }
}

}  // namespace

bool
TrainLoraStage::tp_ship_(std::string* err)
{
  auto* mc = session()->services()->metal_compute();
  // Where each piece's bytes are: in memory, or a cache entry (bf16) to
  // read back as f32 when its chunk goes.
  struct Piece {
    const void* p = nullptr;
    std::string key;
    bool raw = false;     // the entry's bytes as stored (a caption's)
  };
  std::vector<Piece> pieces;
  std::vector<std::size_t> sizes;
  std::size_t total = 0;
  auto put = [&](const void* p, std::size_t n) {
    const FlexData sp = span_(total, n);
    if (n > 0) {
      pieces.push_back({p, {}});
      sizes.push_back(n);
      total += n;
    }
    return sp;
  };
  auto put_buf = [&](const metal_compute::SharedBuffer& b) {
    return put(b.empty() ? nullptr : b.contents(),
               b.empty() ? 0 : b.byte_size());
  };
  auto put_key = [&](const std::string& key, FlexData* sp) {
    const auto* e = _cache.find(key);
    if (e == nullptr) {
      *err = "the cached entry " + key + " vanished";
      return false;
    }
    const std::size_t n = (std::size_t)e->bytes / 2 * 4;
    *sp = span_(total, n);
    if (n > 0) {
      pieces.push_back({nullptr, key});
      sizes.push_back(n);
      total += n;
    }
    return true;
  };
  // A cached caption: its bytes as stored, read back as its chunk goes.
  auto put_raw = [&](const std::string& key, FlexData* sp) {
    const auto* e = _cache.find(key);
    if (e == nullptr) {
      *err = "caption " + key + " is in neither RAM nor the cache";
      return false;
    }
    const std::size_t n = (std::size_t)e->bytes;
    *sp = span_(total, n);
    if (n > 0) {
      pieces.push_back({nullptr, key, true});
      sizes.push_back(n);
      total += n;
    }
    return true;
  };
  FlexData h = FlexData::make_object();
  auto ho = h.as_object();
  ho.insert_or_assign("v", FlexData::make_int(2));
  // THE RUN'S SETTINGS, which a follower's become.
  {
    FlexData c = FlexData::make_object();
    auto co = c.as_object();
    co.insert_or_assign("rank", FlexData::make_int(_rank));
    co.insert_or_assign("alpha", FlexData::make_real(_alpha));
    co.insert_or_assign("targets", FlexData::make_string(_targets));
    co.insert_or_assign("steps", FlexData::make_int(_steps_cfg));
    co.insert_or_assign("epochs", FlexData::make_int(_epochs_cfg));
    co.insert_or_assign("batch_size", FlexData::make_int(_batch));
    if (dp_rank_()) {
      co.insert_or_assign("dp_samples", ints_({_dp_k[0], _dp_k[1]}));
    }
    co.insert_or_assign("run_seed", FlexData::make_uint(
        generation_seed(_seed_seq, _seed, _run)));
    co.insert_or_assign("caption_dropout",
                        FlexData::make_real(_caption_dropout));
    co.insert_or_assign("timestep_sampling", FlexData::make_string(_ts_mode));
    co.insert_or_assign("logit_mean", FlexData::make_real(_logit_mean));
    co.insert_or_assign("logit_std", FlexData::make_real(_logit_std));
    co.insert_or_assign("shift", FlexData::make_real(_shift));
    co.insert_or_assign("save_every", FlexData::make_int(_save_every_cfg));
    co.insert_or_assign("preview_every",
                        FlexData::make_int(_preview_every_cfg));
    co.insert_or_assign("preview_steps", FlexData::make_int(_preview_steps));
    co.insert_or_assign("preview_cfg", FlexData::make_real(_preview_cfg));
    co.insert_or_assign("preview_seed",
                        FlexData::make_uint(_preview_seed));
    co.insert_or_assign("i8_gemm", FlexData::make_bool(_i8));
    co.insert_or_assign("audio_loss_weight",
                        FlexData::make_real(_audio_weight));
    co.insert_or_assign("optimizer", _opt_spec.to_flex());
    co.insert_or_assign("trigger", FlexData::make_string(_trigger));
    co.insert_or_assign("base_model", FlexData::make_string(_base_ref));
    ho.insert_or_assign("config", c);
  }
  ho.insert_or_assign("latent_c", FlexData::make_int(_latent_c));
  ho.insert_or_assign("null_text", FlexData::make_int(_null_text));
  ho.insert_or_assign("preview_texts", ints_(_preview_texts));
  {
    FlexData a = FlexData::make_array();
    for (const std::string& p : _preview_prompts) {
      a.as_array().push_back(std::string_view(p));
    }
    ho.insert_or_assign("preview_prompts", a);
  }
  // THE TEXTS: from RAM, or the cache where a disk run left them -- read
  // back an entry at a time as its chunk goes, as a latent is: a caption
  // is [rows, text_dim] bf16 (5,120 wide on H3, ~1-3 MB, more presented
  // with keyframes), so a dataset's texts can outweigh its latents.
  {
    FlexData a = FlexData::make_array();
    for (std::size_t i = 0; i < _texts.size(); ++i) {
      const Text& whole = _texts[i];
      FlexData e = FlexData::make_object();
      auto eo = e.as_object();
      eo.insert_or_assign("rows", FlexData::make_int(whole.rows));
      eo.insert_or_assign("sideband", whole.sideband);
      eo.insert_or_assign("nrv", FlexData::make_int(whole.nrv));
      eo.insert_or_assign("nra", FlexData::make_int(whole.nra));
      if (!whole.buf.empty() || whole.rows == 0) {
        eo.insert_or_assign("buf", put_buf(whole.buf));
      } else {
        FlexData sp;
        if (!put_raw(whole.key, &sp)) { return false; }
        eo.insert_or_assign("buf", sp);
      }
      eo.insert_or_assign("rv", put_buf(whole.rv));
      eo.insert_or_assign("ra", put_buf(whole.ra));
      a.as_array().push_back(e);
    }
    ho.insert_or_assign("texts", a);
  }
  // THE SAMPLES: latents f32 as they are before packing (the family
  // packs them where it loads).
  {
    FlexData a = FlexData::make_array();
    for (const Sample& s : _samples) {
      FlexData e = FlexData::make_object();
      auto eo = e.as_object();
      eo.insert_or_assign("texts", ints_(s.texts));
      eo.insert_or_assign("repeats", FlexData::make_int(s.repeats));
      eo.insert_or_assign("validation", FlexData::make_bool(s.validation));
      FlexData pa = FlexData::make_array();
      for (const Picture& p : s.pics) {
        FlexData q = FlexData::make_object();
        auto qo = q.as_object();
        qo.insert_or_assign("t", FlexData::make_int(p.t));
        qo.insert_or_assign("h", FlexData::make_int(p.h));
        qo.insert_or_assign("w", FlexData::make_int(p.w));
        if (!p.x0.empty()) {
          qo.insert_or_assign("x0", put_buf(p.x0));
        } else {
          FlexData sp;
          if (!put_key(p.key, &sp)) { return false; }
          qo.insert_or_assign("x0", sp);
        }
        qo.insert_or_assign("audio", FlexData::make_bool(p.audio));
        qo.insert_or_assign("ac", FlexData::make_int(p.ac));
        qo.insert_or_assign("an", FlexData::make_int(p.an));
        qo.insert_or_assign("ax0", put_buf(p.ax0));
        qo.insert_or_assign("anchors", FlexData::make_int(p.anchors));
        FlexData ka = FlexData::make_array();
        for (const metal_compute::SharedBuffer& k : p.kx) {
          ka.as_array().push_back(put_buf(k));
        }
        qo.insert_or_assign("kx", ka);
        qo.insert_or_assign("texts", ints_(p.texts));
        pa.as_array().push_back(q);
      }
      eo.insert_or_assign("pics", pa);
      a.as_array().push_back(e);
    }
    ho.insert_or_assign("samples", a);
  }
  {
    FlexData a = FlexData::make_array();
    for (std::size_t n : sizes) { a.as_array().push_back((std::uint64_t)n); }
    ho.insert_or_assign("pieces", a);
  }
  const std::string json = h.to_json();
  // Sizes first, then the header, then the bytes, through one overlap of
  // the run's channel (its buffers forgotten before the model takes it).
  genai::TpOverlap pipe(mc, _tp_ch, 1, genai::TpDType::kBF16);
  if (!pipe.valid()) {
    *err = "tensor parallel: no thread for the dataset's exchange";
    return false;
  }
  genai::TpBulk bulk(mc);
  const std::uint64_t head[2] = {(std::uint64_t)json.size(),
                                 (std::uint64_t)total};
  bool ok = bulk.exchange(pipe, head, nullptr, sizeof head,
                          genai::TpDType::kBF16, &_tp_seq, err) &&
            bulk.exchange(pipe, json.data(), nullptr, json.size(),
                          genai::TpDType::kBF16, &_tp_seq, err);
  std::vector<std::uint8_t> stage, raw;
  std::vector<float> f;
  // A piece's bytes: in memory, or its cache entry read back (a latent
  // as f32, a caption as stored).
  auto bytes_of = [&](std::size_t i, const void** out) {
    if (pieces[i].p != nullptr) {
      *out = pieces[i].p;
      return true;
    }
    if (pieces[i].raw) {
      raw.resize(sizes[i]);
      if (!_cache.read(pieces[i].key, raw.data(), raw.size(), err)) {
        return false;
      }
      *out = raw.data();
      return true;
    }
    if (!entry_f32_(_cache, pieces[i].key, &f, err)) { return false; }
    *out = f.data();
    return true;
  };
  for_chunks_(sizes, [&](std::size_t i, std::size_t j) {
    if (!ok) { return; }
    const void* src = nullptr;
    if (j == i + 1) {
      ok = bytes_of(i, &src) &&
           bulk.exchange(pipe, src, nullptr, sizes[i],
                         genai::TpDType::kBF16, &_tp_seq, err);
      return;
    }
    stage.clear();
    for (std::size_t k = i; k < j && ok; ++k) {
      ok = bytes_of(k, &src);
      if (!ok) { return; }
      const auto* b = static_cast<const std::uint8_t*>(src);
      stage.insert(stage.end(), b, b + sizes[k]);
    }
    ok = bulk.exchange(pipe, stage.data(), nullptr, stage.size(),
                       genai::TpDType::kBF16, &_tp_seq, err);
  });
  bulk.forget(pipe);
  if (!ok) {
    *err = "tensor parallel: handing the dataset over: " + *err;
    return false;
  }
  session()->info(fmt("TrainLoraStage('{}'): handed the dataset to rank 1 "
                      "({} samples, {} captions, {} MB)", this->id(),
                      _samples.size(), _texts.size(),
                      (total + json.size()) >> 20));
  return true;
}

bool
TrainLoraStage::tp_receive_(std::string* err)
{
  auto* mc = session()->services()->metal_compute();
  genai::TpOverlap pipe(mc, _tp_ch, 1, genai::TpDType::kBF16);
  if (!pipe.valid()) {
    *err = "tensor parallel: no thread for the dataset's exchange";
    return false;
  }
  genai::TpBulk bulk(mc);
  std::uint64_t head[2] = {0, 0};
  std::string json;
  bool ok = bulk.exchange(pipe, nullptr, head, sizeof head,
                          genai::TpDType::kBF16, &_tp_seq, err);
  if (ok) {
    json.resize((std::size_t)head[0]);
    ok = bulk.exchange(pipe, nullptr, json.data(), json.size(),
                       genai::TpDType::kBF16, &_tp_seq, err);
  }
  FlexData h;
  if (ok) {
    try {
      h = FlexData::from_json(std::string_view(json));
    } catch (...) {
      h = FlexData();
    }
  }
  // Each piece into a buffer of its own as its chunk lands, by the sizes
  // the header lists (rank 0 cut its chunks by them).
  std::vector<std::size_t> sizes;
  std::map<std::size_t, std::size_t> at;   // a piece's offset -> its index
  std::vector<metal_compute::SharedBuffer> bufs;
  if (ok && (!h.is_object() || geti_(h, "v") != 2)) {
    ok = false;
    *err = "rank 0's dataset is not one this stage reads";
  }
  if (ok) {
    const FlexData pa = get_(h, "pieces");
    std::size_t off = 0;
    if (pa.is_array()) {
      auto a = pa.as_array();
      for (std::size_t i = 0; i < a.size(); ++i) {
        const std::size_t n = (std::size_t)a.at(i).as_int(0);
        at[off] = sizes.size();
        sizes.push_back(n);
        off += n;
      }
    }
    if (off != (std::size_t)head[1]) {
      ok = false;
      *err = "rank 0's dataset lists its pieces wrong";
    }
  }
  for (std::size_t i = 0; ok && i < sizes.size(); ++i) {
    bufs.push_back(mc->make_shared_buffer(sizes[i]));
    if (bufs.back().empty()) {
      ok = false;
      *err = "no room for the dataset";
    }
  }
  std::vector<std::uint8_t> stage;
  if (ok) {
    for_chunks_(sizes, [&](std::size_t i, std::size_t j) {
      if (!ok) { return; }
      if (j == i + 1) {
        ok = bulk.exchange(pipe, nullptr, bufs[i].contents(), sizes[i],
                           genai::TpDType::kBF16, &_tp_seq, err);
        return;
      }
      std::size_t n = 0;
      for (std::size_t k = i; k < j; ++k) { n += sizes[k]; }
      stage.resize(n);
      ok = bulk.exchange(pipe, nullptr, stage.data(), n,
                         genai::TpDType::kBF16, &_tp_seq, err);
      std::size_t o = 0;
      for (std::size_t k = i; ok && k < j; ++k) {
        std::memcpy(bufs[k].contents(), stage.data() + o, sizes[k]);
        o += sizes[k];
      }
    });
  }
  bulk.forget(pipe);
  std::vector<std::uint8_t>().swap(stage);
  if (!ok) {
    *err = "tensor parallel: taking the dataset: " + *err;
    return false;
  }
  auto bytes = [&](const FlexData& sp, metal_compute::SharedBuffer* out,
                   std::string* e) {
    if (!sp.is_array() || sp.as_array().size() != 2) {
      *e = "a malformed span";
      return false;
    }
    const std::size_t off = (std::size_t)sp.as_array().at(0).as_int(0);
    const std::size_t n = (std::size_t)sp.as_array().at(1).as_int(0);
    if (n == 0) { *out = {}; return true; }
    auto it = at.find(off);
    if (it == at.end() || sizes[it->second] != n ||
        bufs[it->second].empty()) {
      *e = "a span that is no piece";
      return false;
    }
    *out = std::move(bufs[it->second]);
    return true;
  };
  // The run's settings.
  {
    const FlexData c = get_(h, "config");
    _rank = (int)geti_(c, "rank", _rank);
    _alpha = getr_(c, "alpha", _alpha);
    _targets = std::string(get_(c, "targets").as_string(_targets));
    _steps_cfg = (int)geti_(c, "steps");
    _epochs_cfg = (int)geti_(c, "epochs");
    _batch = std::max(1, (int)geti_(c, "batch_size", 1));
    if (dp_rank_() && c.as_object().contains("dp_samples")) {
      const std::vector<int> k = ints_of_(c.as_object().at("dp_samples"));
      if (k.size() == 2 && k[0] > 0 && k[1] > 0) {
        _dp_k[0] = k[0];
        _dp_k[1] = k[1];
      }
    }
    _tp_seed = (std::uint64_t)get_(c, "run_seed").as_int(0);
    _tp_seed_given = true;
    _caption_dropout = getr_(c, "caption_dropout");
    _ts_mode = std::string(get_(c, "timestep_sampling").as_string("auto"));
    _logit_mean = getr_(c, "logit_mean");
    _logit_std = getr_(c, "logit_std", 1.0);
    _shift = getr_(c, "shift", 3.0);
    _save_every_cfg = (int)geti_(c, "save_every");
    _preview_every_cfg = (int)geti_(c, "preview_every");
    _preview_steps = (int)geti_(c, "preview_steps", 20);
    _preview_cfg = getr_(c, "preview_cfg", 4.0);
    _preview_seed = (std::uint64_t)get_(c, "preview_seed").as_int(42);
    _i8 = get_(c, "i8_gemm").as_bool(false);
    _audio_weight = getr_(c, "audio_loss_weight", 1.0);
    _trigger = std::string(get_(c, "trigger").as_string(""));
    _base_ref = std::string(get_(c, "base_model").as_string(_base_ref));
    _opt_spec = genai::train::OptimizerSpec{};
    if (!genai::train::OptimizerSpec::from_flex(get_(c, "optimizer"),
                                                &_opt_spec, err)) {
      *err = "rank 0's optimizer: " + *err;
      return false;
    }
  }
  _latent_c = (int)geti_(h, "latent_c");
  _null_text = (int)geti_(h, "null_text", -1);
  _preview_texts = ints_of_(get_(h, "preview_texts"));
  _preview_prompts.clear();
  {
    const FlexData pp = get_(h, "preview_prompts");
    if (pp.is_array()) {
      auto a = pp.as_array();
      for (std::size_t i = 0; i < a.size(); ++i) {
        _preview_prompts.push_back(std::string(a.at(i).as_string("")));
      }
    }
  }
  _texts.clear();
  {
    const FlexData ta = get_(h, "texts");
    auto a = ta.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      const FlexData e = a.at(i);
      Text t;
      t.rows = (int)geti_(e, "rows");
      t.sideband = get_(e, "sideband");
      t.nrv = (int)geti_(e, "nrv");
      t.nra = (int)geti_(e, "nra");
      if (!bytes(get_(e, "buf"), &t.buf, err) ||
          !bytes(get_(e, "rv"), &t.rv, err) ||
          !bytes(get_(e, "ra"), &t.ra, err)) {
        *err = "rank 0's dataset: " + *err;
        return false;
      }
      _texts.push_back(std::move(t));
    }
  }
  _samples.clear();
  {
    const FlexData sa = get_(h, "samples");
    auto a = sa.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      const FlexData e = a.at(i);
      Sample s;
      s.texts = ints_of_(get_(e, "texts"));
      s.repeats = (int)geti_(e, "repeats", 1);
      s.validation = get_(e, "validation").as_bool(false);
      const FlexData pa = get_(e, "pics");
      auto pv = pa.as_array();
      for (std::size_t j = 0; j < pv.size(); ++j) {
        const FlexData q = pv.at(j);
        Picture p;
        p.t = (int)geti_(q, "t", 1);
        p.h = (int)geti_(q, "h");
        p.w = (int)geti_(q, "w");
        p.audio = get_(q, "audio").as_bool(false);
        p.ac = (int)geti_(q, "ac");
        p.an = (int)geti_(q, "an");
        p.anchors = (int)geti_(q, "anchors");
        p.texts = ints_of_(get_(q, "texts"));
        if (!bytes(get_(q, "x0"), &p.x0, err) ||
            !bytes(get_(q, "ax0"), &p.ax0, err)) {
          *err = "rank 0's dataset: " + *err;
          return false;
        }
        const FlexData ka = get_(q, "kx");
        if (ka.is_array()) {
          auto kv = ka.as_array();
          for (std::size_t k = 0; k < kv.size(); ++k) {
            metal_compute::SharedBuffer b;
            if (!bytes(kv.at(k), &b, err)) {
              *err = "rank 0's dataset: " + *err;
              return false;
            }
            p.kx.push_back(std::move(b));
          }
        }
        s.pics.push_back(std::move(p));
      }
      _samples.push_back(std::move(s));
    }
  }
  session()->info(fmt("TrainLoraStage('{}'): took rank 0's dataset ({} "
                      "samples, {} captions, {} MB)", this->id(),
                      _samples.size(), _texts.size(),
                      ((std::size_t)head[1] + json.size()) >> 20));
  return true;
}

bool
TrainLoraStage::tp_sync_gradients_(std::string* err)
{
  const std::vector<genai::train::LoraParams::Span> spans =
      _params.shared_grad_spans();
  std::size_t n = 0;
  for (const auto& s : spans) { n += s.n; }
  // Host copies sized once (booked: split_reserve_bytes_).
  if (_sync_mine.size() < n) {
    _sync_mine.resize(n);
    _sync_others.resize(n);
  }
  float* mine = _sync_mine.data();
  float* others = _sync_others.data();
  auto* ga = static_cast<float*>(_params.a_grad().contents());
  auto* gb = static_cast<float*>(_params.b_grad().contents());
  std::size_t at = 0;
  for (const auto& s : spans) {
    const float* src = (s.b ? gb : ga) + s.off;
    std::memcpy(mine + at, src, s.n * 4);
    at += s.n;
  }
  if (!_trainer->split_bulk(mine, others, n * 4, genai::TpDType::kF32,
                            err)) {
    return false;
  }
  double sq = 0.0;
  at = 0;
  for (const auto& s : spans) {
    float* dst = (s.b ? gb : ga) + s.off;
    for (std::size_t i = 0; i < s.n; ++i) {
      const float v = mine[at + i] + others[at + i];
      dst[i] = v;
      sq += (double)v * (double)v;
    }
    at += s.n;
  }
  _tp_repl_sq = sq;
  return true;
}

bool
TrainLoraStage::tp_control_(const TpControl& mine, TpControl* all,
                            std::string* err)
{
  std::int64_t a[9] = {mine.stop, mine.save, mine.preview, mine.prompt,
                       mine.steps, 0, mine.seed,
                       (mine.due_save ? 1 : 0) | (mine.due_preview ? 2 : 0),
                       0};
  std::memcpy(&a[5], &mine.cfg, 8);
  std::memcpy(&a[8], &mine.digest, 8);
  std::int64_t b[9] = {};
  if (!bulk_(a, b, sizeof a, genai::TpDType::kF32, err)) { return false; }
  // Rank 0's preview arguments where both ask; else the asker's.
  const std::int64_t* r0 = rank_index_() == 0 ? a : b;
  const std::int64_t* r1 = rank_index_() == 0 ? b : a;
  const std::int64_t* pv = r0[2] != 0 ? r0 : r1;
  all->stop = a[0] != 0 || b[0] != 0;
  all->save = a[1] != 0 || b[1] != 0;
  all->preview = a[2] != 0 || b[2] != 0;
  all->prompt = (int)pv[3];
  all->steps = (int)pv[4];
  std::memcpy(&all->cfg, &pv[5], 8);
  all->seed = pv[6];
  all->due_save = ((a[7] | b[7]) & 1) != 0;
  all->due_preview = ((a[7] | b[7]) & 2) != 0;
  // The ranks' digests, where this step checks: alike, or the run ends.
  std::uint64_t other = 0;
  std::memcpy(&other, &b[8], 8);
  if (mine.digest != other) {
    *err = parted_(mine.digest, other);
    return false;
  }
  all->digest = mine.digest;
  return true;
}

std::string
TrainLoraStage::parted_(std::uint64_t mine, std::uint64_t other) const
{
  return fmt("at step {} the ranks no longer hold the same {} (digests "
             "{:016x} here, {:016x} there): their GPUs do not update them "
             "alike (an M4 paired with an M5 is not promised to), or a "
             "rank went wrong. The run ends. No state was written parted, "
             "so its last checkpoint resumes it -- on the same pair of "
             "hosts it may part again", _step,
             tp_split_() ? "shared factors" : "adapter", mine, other)();
}

bool
TrainLoraStage::tp_alike_(std::string* err)
{
  const std::uint64_t mine = masters_digest_();
  std::uint64_t other = 0;
  if (!bulk_(&mine, &other, sizeof mine, genai::TpDType::kF32, err)) {
    return false;
  }
  if (mine != other) {
    *err = parted_(mine, other);
    return false;
  }
  return true;
}

Job
TrainLoraStage::tp_agree_(RuntimeContext& ctx, bool* stopped,
                          std::string* err)
{
  *stopped = false;
  TpControl mine, all;
  mine.stop = ctx.stop_requested();
  mine.due_save = _tp_due_save;
  mine.due_preview = _tp_due_preview;
  // A digest every check_every steps, and at the run's first two
  // agreements: what it starts from (a seed's draw, or the two ranks'
  // resume states) and its first update -- where two GPUs that do not
  // compute alike part already, so a pair that cannot train together is
  // told at once, not check_every steps on. Both ranks count alike.
  if (_tp_check_every > 0 &&
      (_tp_first_checks > 0 ||
       (_step > 0 && _step % _tp_check_every == 0))) {
    mine.digest = masters_digest_();
  }
  if (_tp_first_checks > 0) { --_tp_first_checks; }
  for (const auto& c : _tp_pending) {
    if (c->name() == "save") {
      mine.save = true;
      continue;
    }
    // Several previews asked before one agreement run once, with the
    // last one's arguments; each command is answered by that run.
    const FlexData a = c->args();
    auto ao = a.as_object();
    mine.preview = true;
    mine.prompt = (int)ao.at("prompt").as_int(-1);
    mine.steps = (int)ao.at("steps").as_int(0);
    mine.cfg = ao.at("cfg").as_real(0.0);
    mine.seed = ao.at("seed").as_int(-1);
  }
  if (!tp_control_(mine, &all, err)) {
    *err = std::string(par_key_()) + ": agreeing on the next step: " + *err;
    co_return;
  }
  // What was due on either rank is due on both.
  _tp_due_save = all.due_save;
  _tp_due_preview = all.due_preview;
  // NO STATE IS WRITTEN PARTED: a checkpoint (and its resume state), a
  // stop's resume state and the run's last adapter are checked alike
  // first where this agreement did not check -- else a state saved
  // between a parting and its check would resume both ranks parted, and
  // the run could never pass the check again.
  if (_tp_check_every > 0 && all.digest == 0 &&
      (all.stop || _tp_due_save || all.save || _step >= _total_steps) &&
      !tp_alike_(err)) {
    for (auto& c : _tp_pending) { c->fail(*err); }
    *err = std::string(par_key_()) + ": " + *err;
    co_return;
  }
  // A stop first: its outputs are closed, so no checkpoint beat or
  // preview now -- the run saved for its resume, every rank alike.
  if (all.stop) {
    for (auto& c : _tp_pending) { c->fail("training stopped"); }
    _tp_pending.clear();
    _tp_due_save = _tp_due_preview = false;
    std::string serr;
    (void)save_(false, &serr);
    session()->info(fmt("TrainLoraStage('{}'): stopped at step {} with the "
                        "other rank; the run resumes from here", this->id(),
                        _step));
    tp_close_();
    _bar.finish();
    _phase = Phase::kDone;
    ctx.signal_done();
    *stopped = true;
    co_return;
  }
  // What the last step was due, then what was asked: a command's
  // checkpoint at a step already checkpointed is that one.
  if (_tp_due_save || all.save) {
    FlexData r;
    if (_saved_step == _step && _last_checkpoint.is_object()) {
      r = _last_checkpoint;
    } else {
      co_await checkpoint_(ctx, &r, err);
    }
    for (auto& c : _tp_pending) {
      if (c->name() != "save") { continue; }
      if (!err->empty()) {
        c->fail(*err);
      } else {
        c->reply(FlexData(r));
      }
    }
    _tp_due_save = false;
    if (!err->empty()) { co_return; }
  }
  // A data-parallel run's previews are rank 0's alone (every rank holds
  // the one adapter; the follower has nowhere to write them).
  if (_dp_rank > 0) {
    _tp_due_preview = false;
    all.preview = false;
  }
  if (_tp_due_preview) {
    int n = 0;
    std::string perr;
    co_await previews_(ctx, -1, _preview_steps, _preview_cfg, _preview_seed,
                       &n, &perr);
    if (perr.empty() && fault_("preview")) { perr = "an injected fault"; }
    _tp_due_preview = false;
    // SPLIT, a preview is the ranks' forwards together: one that ended
    // early on this rank left the other's exchanges waiting, or paired
    // with whatever this rank sends next (their counts, which the
    // transport checks, still in step). The run ends; the last
    // checkpoint resumes it.
    if (!perr.empty() && tp_split_()) {
      *err = "a preview failed: " + perr;
      co_return;
    }
    if (!perr.empty()) {
      session()->warn(fmt("TrainLoraStage('{}'): preview failed: {}",
                          this->id(), perr));
    }
  }
  if (all.preview) {
    int n = 0;
    std::string perr;
    co_await previews_(ctx, all.prompt,
                       all.steps > 0 ? all.steps : _preview_steps,
                       all.cfg > 0.0 ? all.cfg : _preview_cfg,
                       all.seed >= 0 ? (std::uint64_t)all.seed
                                     : _preview_seed,
                       &n, &perr);
    if (!perr.empty() && tp_split_()) {
      for (auto& c : _tp_pending) {
        if (c->name() == "preview") { c->fail(perr); }
      }
      *err = "a preview failed: " + perr;
      co_return;
    }
    for (auto& c : _tp_pending) {
      if (c->name() != "preview") { continue; }
      if (!perr.empty()) {
        c->fail(perr);
        continue;
      }
      FlexData r = FlexData::make_object();
      r.as_object().insert_or_assign("step", FlexData::make_int(_step));
      r.as_object().insert_or_assign("previews", FlexData::make_int(n));
      c->reply(std::move(r));
    }
  }
  _tp_pending.clear();
  // A SPLIT run's barrier before the step's first forward, whose GPU work
  // waits on the peer's partials: a checkpoint or a preview one rank made
  // longer (rank 0 writes the adapter) is waited out here, on the CPU,
  // not by a GPU wait the watchdog counts.
  if (tp_split_()) {
    const std::int64_t here = _step;
    std::int64_t there = -1;
    if (!bulk_(&here, &there, sizeof here, genai::TpDType::kF32, err)) {
      *err = std::string(par_key_()) + ": before the step: " + *err;
      co_return;
    }
    // ...and the ranks at one step: one that skipped or repeated a step
    // would train other samples than its peer from here on.
    if (there != here) {
      *err = fmt("{}: before the step: this rank is at step {}, the other "
                 "at {}", par_key_(), here, there)();
      co_return;
    }
  }
}

FlexData
TrainLoraStage::rank_record_() const
{
  FlexData r = FlexData::make_object();
  auto o = r.as_object();
  o.insert_or_assign("mode", FlexData::make_string(
      tp_split_() ? "tensor_parallel" : dp_rank_() ? "data_parallel"
                                                    : "whole"));
  o.insert_or_assign("batch_size", FlexData::make_int(_batch));
  o.insert_or_assign("global_batch", FlexData::make_int(global_batch_()));
  if (tp_rank_()) {
    o.insert_or_assign("rank", FlexData::make_int(rank_index_()));
  }
  if (tp_split_()) {
    o.insert_or_assign("share", ints_({_tp.head0, _tp.head1, _tp.ffn0,
                                       _tp.ffn1}));
  }
  if (dp_rank_()) {
    o.insert_or_assign("dp_samples", ints_({_dp_k[0], _dp_k[1]}));
  }
  return r;
}

bool
TrainLoraStage::resume_matches_(const FlexData& run, std::string* why) const
{
  if (!run.is_object()) {
    *why = "its run.json is unreadable";
    return false;
  }
  if ((std::uint64_t)geti_(run, "seed") != _run_seed) {
    *why = "another seed";
    return false;
  }
  if ((int)geti_(run, "total_steps") != _total_steps) {
    *why = "another length";
    return false;
  }
  if (geti_(run, "samples") != (std::int64_t)_samples.size()) {
    *why = "another dataset";
    return false;
  }
  const FlexData me = rank_record_();
  const FlexData mode = get_(run, "mode");
  // A state from before ranks were recorded is a whole run's.
  const std::string was =
      mode.is_null() ? std::string("whole") : std::string(mode.as_string(""));
  if (was != std::string(get_(me, "mode").as_string(""))) {
    *why = "another kind of run (" + was + ")";
    return false;
  }
  if (!get_(run, "global_batch").is_null() &&
      (geti_(run, "global_batch") != geti_(me, "global_batch") ||
       geti_(run, "batch_size") != geti_(me, "batch_size"))) {
    *why = "another batch";
    return false;
  }
  if (tp_rank_()) {
    if (geti_(run, "rank", -1) != rank_index_()) {
      *why = fmt("rank {}'s state", geti_(run, "rank", -1))();
      return false;
    }
    if (ints_of_(get_(run, "share")) != ints_of_(get_(me, "share")) ||
        ints_of_(get_(run, "dp_samples")) !=
            ints_of_(get_(me, "dp_samples"))) {
      *why = "another share of the run";
      return false;
    }
  }
  return true;
}

namespace {

// A run.json read, null when it does not parse.
FlexData
read_run_json_(const fs::path& p)
{
  try {
    std::ifstream in(p);
    if (!in) { return {}; }
    FlexData r = FlexData::from_json(in);
    return r.is_object() ? r : FlexData{};
  } catch (...) {
    return {};
  }
}

// `step-000500` -> 500; -1 for any other name (a .tmp left by a write
// that never finished among them).
std::int64_t
slot_step_(const std::string& name)
{
  static constexpr std::string_view kPre = "step-";
  if (name.size() <= kPre.size() || name.compare(0, kPre.size(), kPre) != 0) {
    return -1;
  }
  std::int64_t v = 0;
  for (std::size_t i = kPre.size(); i < name.size(); ++i) {
    if (name[i] < '0' || name[i] > '9') { return -1; }
    v = v * 10 + (name[i] - '0');
  }
  return v;
}

}  // namespace

std::vector<TrainLoraStage::ResumeSlot>
TrainLoraStage::resume_slots_(bool* others,
                              std::vector<std::string>* broken) const
{
  if (others != nullptr) { *others = false; }
  if (broken != nullptr) { broken->clear(); }
  std::vector<ResumeSlot> out;
  const fs::path dir = resume_dir_();
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) { return out; }
  auto take = [&](const fs::path& d) {
    if (!fs::is_regular_file(d / "optimizer.bin", ec)) { return; }
    const FlexData run = read_run_json_(d / "run.json");
    if (run.is_null()) { return; }
    // As resume_ checks it.
    std::string why;
    if (!resume_matches_(run, &why)) {
      if (others != nullptr) { *others = true; }
      return;
    }
    const std::int64_t st = geti_(run, "step", -1);
    if (st < 0) { return; }
    // ...and one this optimizer can load: never agreed on otherwise, so
    // a rank whose state was cut short does not pair the run with it at
    // every restart (its load failing each time), but goes back a slot.
    std::string e;
    if (!_opt.check_state((d / "optimizer.bin").string(), &e)) {
      if (broken != nullptr) {
        broken->push_back(fmt("the resume state at step {} does not load "
                              "({})", st, e)());
      }
      return;
    }
    out.push_back(ResumeSlot{st, d});
  };
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (!e.is_directory(ec)) { continue; }
    const std::int64_t st = slot_step_(e.path().filename().string());
    if (st < 0) { continue; }
    take(e.path());
  }
  // An older layout's: the folder's own two files (its run.json alone is
  // the newest slot's record, never a state).
  take(dir);
  // Stable: a step held twice (a slot and the root's older layout, taken
  // last) keeps the slot's.
  std::stable_sort(out.begin(), out.end(),
                   [](const ResumeSlot& a, const ResumeSlot& b) {
                     return a.step > b.step;
                   });
  out.erase(std::unique(out.begin(), out.end(),
                        [](const ResumeSlot& a, const ResumeSlot& b) {
                          return a.step == b.step;
                        }),
            out.end());
  return out;
}

void
TrainLoraStage::resume_tidy_(std::int64_t upto) const
{
  const fs::path dir = resume_dir_();
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) { return; }
  std::vector<std::pair<std::int64_t, fs::path>> slots;
  std::vector<fs::path> gone;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (!e.is_directory(ec)) { continue; }
    const std::string n = e.path().filename().string();
    const std::int64_t st = slot_step_(n);
    if (st < 0) {
      // A write that never finished.
      if (n.size() > 4 && n.compare(n.size() - 4, 4, ".tmp") == 0) {
        gone.push_back(e.path());
      }
      continue;
    }
    if (st > upto) {
      gone.push_back(e.path());
    } else {
      slots.emplace_back(st, e.path());
    }
  }
  std::sort(slots.begin(), slots.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  for (std::size_t i = kResumeKept; i < slots.size(); ++i) {
    gone.push_back(slots[i].second);
  }
  if (slots.size() > (std::size_t)kResumeKept) {
    slots.resize((std::size_t)kResumeKept);
  }
  for (const fs::path& g : gone) { fs::remove_all(g, ec); }
  // The older layout's state, once a slot holds one (or past `upto`).
  if (!slots.empty() || geti_(read_run_json_(dir / "run.json"), "step", -1) >
                            upto) {
    fs::remove(dir / "optimizer.bin", ec);
  }
  // The folder's run.json: the newest kept's record, for a reader that
  // asks what it would resume from.
  if (slots.empty()) {
    if (!fs::is_regular_file(dir / "optimizer.bin", ec)) {
      fs::remove(dir / "run.json", ec);
    }
    return;
  }
  const fs::path tmp = dir / "run.json.tmp";
  fs::copy_file(slots.front().second / "run.json", tmp,
                fs::copy_options::overwrite_existing, ec);
  if (!ec) { fs::rename(tmp, dir / "run.json", ec); }
}

#endif  // VPIPE_BUILD_APPLE_SILICON

}  // namespace vpipe
