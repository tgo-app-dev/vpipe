// weight-set.cc -- the manager-owned view of a checkpoint's weights.
//
// What is under test is the contract models depend on: ask for the same
// directory and get the same set (so its weights are loaded once and
// reference-counted), ask for the same tensor and get aliases of ONE
// buffer (so two models over one checkpoint share bytes rather than
// copy them), and load a part only when something actually needs it.
//
// The fixture writes its own tiny safetensors file, so these run
// everywhere rather than skipping on a missing model.
//
//   vpipe_test --filter '*weight_set*'

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "common/vpipe-format.h"
#include "interfaces/ui-delegate-intf.h"
#include "generative-models/shared/streamed-refill.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace vpipe;
using namespace vpipe::genai;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace {

MetalCompute*
mc_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  return (mc != nullptr && mc->valid()) ? mc : nullptr;
}

// A throwaway model directory holding one `model.safetensors` with two
// f32 tensors, "trunk.w" [4] and "encoder.w" [8]. Deleted on scope exit.
//
// Both tensors sit at 16-aligned file offsets and the header is padded so
// the data blob starts 16-aligned, because load_mapped() only takes the
// zero-copy path for bind-aligned offsets -- an unaligned fixture would
// silently test the copying fallback instead.
class TempCheckpoint {
public:
  // `big` adds one tensor LARGER than MetalCompute's small-allocation
  // heap threshold (64 KB). Below it a Copied load sub-allocates from
  // the shared heap, and a heap sub-allocation declines to park -- its
  // residency is the heap's to manage -- so a parking test built on the
  // 16-byte tensors below measures that rule rather than the one it
  // means to. Off by default: every other test here wants the small
  // fixture.
  explicit TempCheckpoint(bool big = false)
  {
    namespace fs = std::filesystem;
    _dir = (fs::temp_directory_path() /
            ("vpipe-ws-test-" + to_string(::getpid()) + "-" +
             to_string(next_id_()))).string();
    fs::create_directories(_dir);

    for (int i = 0; i < 4; ++i) { _trunk[i] = 1.0f + (float)i; }
    for (int i = 0; i < 8; ++i) { _enc[i] = 100.0f + (float)i; }

    string hdr =
        "{\"trunk.w\":{\"dtype\":\"F32\",\"shape\":[4],"
        "\"data_offsets\":[0,16]},"
        "\"encoder.w\":{\"dtype\":\"F32\",\"shape\":[8],"
        "\"data_offsets\":[16,48]}";
    if (big) {
      hdr += ",\"big.w\":{\"dtype\":\"F32\",\"shape\":[" +
             to_string(kBigElems) + "],\"data_offsets\":[48," +
             to_string(48 + kBigElems * 4) + "]}";
    }
    hdr += "}";
    while (((8 + hdr.size()) % 16) != 0) { hdr.push_back(' '); }

    ofstream out(_dir + "/model.safetensors", ios::binary);
    const uint64_t n = hdr.size();
    out.write(reinterpret_cast<const char*>(&n), 8);
    out.write(hdr.data(), (streamsize)hdr.size());
    out.write(reinterpret_cast<const char*>(_trunk), 16);
    out.write(reinterpret_cast<const char*>(_enc), 32);
    if (big) {
      const vector<float> blob((size_t)kBigElems, 7.5f);
      out.write(reinterpret_cast<const char*>(blob.data()),
                (streamsize)kBigElems * 4);
    }
  }

  // 512 KB: comfortably past the 64 KB heap threshold.
  static constexpr int kBigElems = 131072;

  ~TempCheckpoint()
  {
    std::error_code ec;
    std::filesystem::remove_all(_dir, ec);
  }

  const string& dir() const { return _dir; }
  const float*  trunk() const { return _trunk; }
  const float*  enc() const { return _enc; }

private:
  static int next_id_()
  {
    static int n = 0;
    return ++n;
  }

  string _dir;
  float  _trunk[4]{};
  float  _enc[8]{};
};

// A checkpoint with ONE realistically-sized tensor, for measuring the
// streaming path. The 32-byte tensors above are useless for that: the
// per-read fixed costs swamp the memcpy, so the measurement says nothing
// about how a real DiT block behaves.
class BigCheckpoint {
public:
  explicit BigCheckpoint(std::size_t floats)
  {
    namespace fs = std::filesystem;
    _dir = (fs::temp_directory_path() /
            ("vpipe-ws-big-" + to_string(::getpid()) + "-" +
             to_string(next_id_()))).string();
    fs::create_directories(_dir);
    std::vector<float> v(floats);
    for (std::size_t i = 0; i < floats; ++i) { v[i] = (float)(i & 0xffff); }
    const std::size_t nbytes = floats * 4;

    string hdr = "{\"block.w\":{\"dtype\":\"F32\",\"shape\":[" +
                 to_string(floats) + "],\"data_offsets\":[0," +
                 to_string(nbytes) + "]}}";
    while (((8 + hdr.size()) % 16) != 0) { hdr.push_back(' '); }
    ofstream out(_dir + "/model.safetensors", ios::binary);
    const uint64_t n = hdr.size();
    out.write(reinterpret_cast<const char*>(&n), 8);
    out.write(hdr.data(), (streamsize)hdr.size());
    out.write(reinterpret_cast<const char*>(v.data()), (streamsize)nbytes);
  }

  ~BigCheckpoint()
  {
    std::error_code ec;
    std::filesystem::remove_all(_dir, ec);
  }
  const string& dir() const { return _dir; }

private:
  static int next_id_()
  {
    static int n = 1000;
    return ++n;
  }
  string _dir;
};


// A checkpoint holding one tensor of each dtype a streamed block meets:
// BF16 and U32 (raw), F16 (same width, converted in place) and F32 (half
// the width of its bf16 destination, so unservable). Sizes are tiny --
// what is under test is the DECISION, not the throughput.
class MixedCheckpoint {
public:
  MixedCheckpoint()
  {
    namespace fs = std::filesystem;
    _dir = (fs::temp_directory_path() /
            ("vpipe-ws-mixed-" + to_string(::getpid()) + "-" +
             to_string(next_id_()))).string();
    fs::create_directories(_dir);
    for (int i = 0; i < 8; ++i) {
      _bf[i] = (uint16_t)(0x3f80 + i);            // ~1.0 upward, bf16
      _u32[i] = 0x01020304u + (uint32_t)i;
      _h[i] = (_Float16)(0.5f + 0.125f * (float)i);
      _f32[i] = 1.5f + (float)i;
    }
    string hdr =
        "{\"raw.bf16\":{\"dtype\":\"BF16\",\"shape\":[8],"
        "\"data_offsets\":[0,16]},"
        "\"raw.u32\":{\"dtype\":\"U32\",\"shape\":[8],"
        "\"data_offsets\":[16,48]},"
        "\"conv.f16\":{\"dtype\":\"F16\",\"shape\":[8],"
        "\"data_offsets\":[48,64]},"
        "\"wide.f32\":{\"dtype\":\"F32\",\"shape\":[8],"
        "\"data_offsets\":[64,96]}}";
    while (((8 + hdr.size()) % 16) != 0) { hdr.push_back(' '); }
    ofstream out(_dir + "/model.safetensors", ios::binary);
    const uint64_t n = hdr.size();
    out.write(reinterpret_cast<const char*>(&n), 8);
    out.write(hdr.data(), (streamsize)hdr.size());
    out.write(reinterpret_cast<const char*>(_bf), 16);
    out.write(reinterpret_cast<const char*>(_u32), 32);
    out.write(reinterpret_cast<const char*>(_h), 16);
    out.write(reinterpret_cast<const char*>(_f32), 32);
  }
  ~MixedCheckpoint()
  {
    std::error_code ec;
    std::filesystem::remove_all(_dir, ec);
  }
  const string&   dir() const { return _dir; }
  const uint16_t* bf()  const { return _bf; }
  const uint32_t* u32() const { return _u32; }
  const _Float16* h()   const { return _h; }

private:
  static int next_id_() { static int n = 2000; return ++n; }
  string   _dir;
  uint16_t _bf[8]{};
  uint32_t _u32[8]{};
  _Float16 _h[8]{};
  float    _f32[8]{};
};

}  // namespace

// Two callers naming one directory get the SAME set -- the whole basis
// of "the second stage bumps a reference count instead of loading".
TEST(weight_set, manager_dedups_by_directory) {
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  TempCheckpoint ck;

  auto a = mgr->weight_set(ck.dir());
  ASSERT_TRUE(a != nullptr);
  auto b = mgr->weight_set(ck.dir());
  EXPECT_TRUE(a == b);                     // same object, not just equal
  EXPECT_TRUE(mgr->weight_set_count() == 1u);

  // A different directory is a different model, even in the same family.
  TempCheckpoint other;
  auto c = mgr->weight_set(other.dir());
  ASSERT_TRUE(c != nullptr);
  EXPECT_TRUE(c != a);
  EXPECT_TRUE(mgr->weight_set_count() == 2u);
}

// THE MANAGER OWNS THE CHECKPOINT; a model BORROWS it.
//
// This used to read the other way round -- the cache held weak
// references, so the first stage to load was the owner and the last to
// let go was the undertaker. Sharing then worked only for OVERLAPPING
// lifetimes: a second pipeline over the same model, opened a second
// after the first ended, re-read the whole checkpoint from disk however
// comfortably it fitted in RAM.
//
// Now the set outlives every borrower, and what happens when the last
// one lets go is a POLICY the manager applies (release_unheld_): kept
// and parked, so the next borrower pays neither a reopen nor -- unless
// the box actually took the pages -- a re-read.
TEST(weight_set, outlives_its_last_holder) {
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  TempCheckpoint ck;

  const WeightSet* raw = nullptr;
  {
    auto a = mgr->weight_set(ck.dir());
    ASSERT_TRUE(a != nullptr);
    if (a == nullptr) { return; }
    raw = a.get();
    {
      auto b = mgr->weight_set(ck.dir());
      EXPECT_TRUE(b == a);              // one set, two borrowers
    }
    EXPECT_TRUE(mgr->weight_set_count() == 1u);
  }
  // Every borrower gone, and the checkpoint is still the manager's.
  EXPECT_TRUE(mgr->weight_set_count() == 1u);
  auto again = mgr->weight_set(ck.dir());
  EXPECT_TRUE(again != nullptr && again.get() == raw);   // not a reopen
}

// ...and settling it PARKS it, which is the half that makes owning it
// affordable: the bytes stay reclaimable by the kernel, so a checkpoint
// nobody is using is not a claim on the box.
TEST(weight_set, an_unborrowed_set_is_parked_not_held_hard) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  TempCheckpoint ck(/*big=*/true);

  {
    auto ws = mgr->weight_set(ck.dir());
    ASSERT_TRUE(ws != nullptr);
    if (ws == nullptr) { return; }
    SharedBuffer a = ws->tensor("big.w", mc, WeightSet::Residency::Copied);
    ASSERT_TRUE(!a.empty());
    EXPECT_FALSE(ws->parked());
    // Asking while still borrowing must NOT park: the manager cannot
    // know this borrower is idle, and parking under it would hand it
    // pages the kernel may discard.
    EXPECT_TRUE(mgr->park_weights(ck.dir()) == 0u);
    EXPECT_FALSE(ws->parked());
  }
  // Borrower gone. `pool_weights` is the "settle it now" verb.
  mgr->pool_weights(ck.dir());
  auto back = mgr->weight_set(ck.dir());
  ASSERT_TRUE(back != nullptr);
  if (back == nullptr) { return; }
  EXPECT_TRUE(back->parked());
  // And borrowing it again does not un-park it -- only a READ does,
  // which is what re-reads from disk if the kernel took the pages.
  SharedBuffer b = back->tensor("big.w", mc, WeightSet::Residency::Copied);
  EXPECT_FALSE(back->parked());
  EXPECT_TRUE(!b.empty());
}

// Repeat requests for a tensor return ALIASES of one buffer: same GPU
// allocation, same bytes, loaded once. This is the dedup that made the
// old model-level attempt unnecessary -- and it does not care which
// caller asked first.
TEST(weight_set, tensor_hands_out_aliases_of_one_buffer) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);

  SharedBuffer a = ws->tensor("trunk.w", mc);
  SharedBuffer b = ws->tensor("trunk.w", mc);
  ASSERT_TRUE(!a.empty());
  ASSERT_TRUE(!b.empty());
  EXPECT_TRUE(a.mtl_buffer() == b.mtl_buffer());
  EXPECT_TRUE(a.contents() == b.contents());
  EXPECT_TRUE(a.byte_size() == 16u);
  // One cache entry, not two: the second request cost nothing.
  EXPECT_TRUE(ws->stats().entries == 1u);

  const auto* v = static_cast<const float*>(a.contents());
  EXPECT_TRUE(v[0] == ck.trunk()[0]);
  EXPECT_TRUE(v[3] == ck.trunk()[3]);
}

// An alias must NOT be counted as owning the bytes, or a model that holds
// several would report multiples of its real footprint -- and parking one
// would evict memory it does not own.
TEST(weight_set, aliases_do_not_own_the_allocation) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);
  SharedBuffer a = ws->tensor("trunk.w", mc, WeightSet::Residency::Copied);
  ASSERT_TRUE(!a.empty());
  EXPECT_FALSE(a.is_owned());
  EXPECT_FALSE(a.mark_inactive());       // declines: not ours to park
}

// park_weights() is the BY-DIRECTORY entry point -- what a stage's
// `unload_when_idle: park` reaches for -- and it has to find the set
// that directory opened.
//
// It did not. `_weight_sets` is keyed "<canonical dir>|<variant>" and
// this looked the bare canonical dir up in it, which cannot match even
// in the ordinary empty-variant case, so every call returned 0. Nothing
// caught it because the registry-level park is tested with a synthetic
// owner (weight-registry.cc) and this path had no test at all -- and
// the zero it returned was indistinguishable from the legitimate zero a
// set that reads uncached gives back.
TEST(weight_set, park_weights_finds_the_set_by_directory) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  TempCheckpoint ck(/*big=*/true);

  auto ws = mgr->weight_set(ck.dir());
  ASSERT_TRUE(ws != nullptr);
  // CACHED and Copied: the only combination the registry can park. And
  // the BIG tensor, so the allocation is its own MTL::Buffer rather than
  // a sub-range of the small-allocation heap.
  SharedBuffer a = ws->tensor("big.w", mc, WeightSet::Residency::Copied);
  ASSERT_TRUE(!a.empty());
  const WeightSet* raw = ws.get();
  // Stop BORROWING before asking: the manager owns the checkpoint and
  // will not park one a model still holds. That refusal has its own
  // test (an_unborrowed_set_is_parked_not_held_hard); what this one is
  // about is whether the by-directory lookup finds the set at all.
  ws.reset();

  const std::size_t got = mgr->park_weights(ck.dir());
  EXPECT_TRUE(got > 0u);

  // Idempotent: a set already parked has nothing further to give.
  EXPECT_TRUE(mgr->park_weights(ck.dir()) == 0u);

  // And a directory nobody opened is still 0 -- the fix widens the
  // lookup, it does not make it match anything.
  TempCheckpoint other;
  EXPECT_TRUE(mgr->park_weights(other.dir()) == 0u);

  auto back = mgr->weight_set(ck.dir());
  ASSERT_TRUE(back != nullptr);
  if (back == nullptr) { return; }
  EXPECT_TRUE(back.get() == raw);
  EXPECT_TRUE(back->parked());
  // The bytes are still readable: parking makes pages purgeable, it does
  // not discard them, and nothing here created memory pressure.
  back->reload_weights();
}

// POOLING MUST ARM THE WAY BACK, not just hand the pages over.
//
// pool_weights() marks a released checkpoint purgeable so the box may
// take it and keeps the set so a relaunch pays no re-open. It recorded
// only the KERNEL's half of that: the set's own parked flag stayed
// false, so ensure_active_() -- the one thing that ever takes those
// pages back -- short-circuited on every later read. A checkpoint that
// had been pooled once then served whatever the kernel had left of
// those buffers, and served ZEROS where it had taken them, in silence:
// the "parked weights were reclaimed" warning keys off the very flag
// that was never set.
//
// It reached a picture through the diffusion conditioner, which pools
// `<root>/text_encoder` after each conditioning while keeping its
// `_enc_ws` -- so the next prompt rebuilt the encoder and its embedding
// table out of emptied buffers and the DiT denoised from garbage. The
// second generation in a process was wrong and the first was not.
//
// What a test cannot arrange is the memory pressure that decides
// whether the kernel actually takes the pages, so this pins the
// invariant underneath it instead: parked in the kernel and parked on
// the set are ONE fact, and a read is what undoes both.
TEST(weight_set, pooling_arms_the_reactivate) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  TempCheckpoint ck(/*big=*/true);

  auto ws = mgr->weight_set(ck.dir());
  ASSERT_TRUE(ws != nullptr);
  if (ws == nullptr) { return; }
  // Cached + Copied + past the 64 KB heap threshold: the only shape the
  // registry can actually park -- see the test above, which first passed
  // while proving nothing for want of the last of those.
  SharedBuffer a = ws->tensor("big.w", mc, WeightSet::Residency::Copied);
  ASSERT_TRUE(!a.empty());
  EXPECT_FALSE(ws->parked());
  const WeightSet* raw = ws.get();
  ws.reset();                     // stop borrowing; the manager keeps it

  mgr->pool_weights(ck.dir());
  // Recycling does NOT reactivate, and should not: it hands back the
  // same set without re-opening the checkpoint, and whether the pages
  // survived is not known until something reads them.
  auto again = mgr->weight_set(ck.dir());
  ASSERT_TRUE(again != nullptr);
  if (again == nullptr) { return; }
  EXPECT_TRUE(again.get() == raw);
  // THE REGRESSION. Without it the pages are volatile and nothing is
  // left in the process that will ever ask for them back.
  EXPECT_TRUE(again->parked());

  // The READ is what takes them back -- re-reading from disk if the
  // kernel did take them, which is why the bytes below are the ones the
  // fixture wrote either way.
  SharedBuffer b = again->tensor("big.w", mc, WeightSet::Residency::Copied);
  ASSERT_TRUE(!b.empty());
  EXPECT_FALSE(again->parked());
  const float* p = b.empty() ? nullptr
                             : static_cast<const float*>(b.contents());
  EXPECT_TRUE(p != nullptr);
  if (p != nullptr) {
    EXPECT_TRUE(p[0] == 7.5f);
    EXPECT_TRUE(p[TempCheckpoint::kBigElems - 1] == 7.5f);
  }
}

// A mapped tensor aliases the mmap (no copy); a copied one is an owned
// allocation the residency policy can park. Both must read correctly --
// the distinction is about WHERE the bytes live, not what they are.
TEST(weight_set, mapped_and_copied_both_read_correctly) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto mapped = WeightSet::open(ck.dir(), nullptr);
  auto copied = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(mapped != nullptr);
  ASSERT_TRUE(copied != nullptr);

  SharedBuffer m = mapped->tensor("encoder.w", mc);
  SharedBuffer c = copied->tensor("encoder.w", mc,
                                  WeightSet::Residency::Copied);
  ASSERT_TRUE(!m.empty());
  ASSERT_TRUE(!c.empty());
  const auto* mv = static_cast<const float*>(m.contents());
  const auto* cv = static_cast<const float*>(c.contents());
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(mv[i] == ck.enc()[i]);
    EXPECT_TRUE(cv[i] == ck.enc()[i]);
  }
  // The mapped one is a view into the shard wrap and never owns bytes,
  // so only the copied set reports parkable (owned) bytes.
  EXPECT_TRUE(mapped->stats().mapped_bytes == 32u);
  EXPECT_TRUE(copied->stats().copied_bytes == 32u);
}

// A derived tensor -- one the model TRANSFORMS rather than copies -- is
// built once and shared thereafter. Two models over a checkpoint pay for
// the transform once between them.
TEST(weight_set, derived_builds_once_and_is_shared) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);

  int builds = 0;
  auto build = [&]() -> SharedBuffer {
    ++builds;
    SharedBuffer b = mc->make_shared_buffer(16);
    static_cast<float*>(b.contents())[0] = 42.0f;
    return b;
  };

  SharedBuffer a = ws->derived("t/flat|trunk.w", build);
  SharedBuffer b = ws->derived("t/flat|trunk.w", build);
  ASSERT_TRUE(!a.empty());
  EXPECT_TRUE(builds == 1);
  EXPECT_TRUE(a.mtl_buffer() == b.mtl_buffer());
  EXPECT_TRUE(static_cast<const float*>(b.contents())[0] == 42.0f);

  // A different transform over the same tensor is a DIFFERENT entry: the
  // key names the transform, so two models whose configs disagree cannot
  // pick up each other's bytes.
  SharedBuffer c = ws->derived("t/hwio|trunk.w", build);
  EXPECT_TRUE(builds == 2);
  EXPECT_TRUE(c.mtl_buffer() != a.mtl_buffer());
}

// A derived tensor has no retained transform to rebuild it with, so it
// must NOT be offered for parking -- a reclaimed one would be
// unrecoverable. Only named, re-readable tensors are.
TEST(weight_set, only_reloadable_tensors_are_offered_for_parking) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);
  ws->tensor("trunk.w", mc, WeightSet::Residency::Copied);   // reloadable
  ws->tensor("encoder.w", mc);                               // mapped
  ws->derived("t/x", [&]() { return mc->make_shared_buffer(16); });

  int offered = 0;
  ws->for_each_weight([&offered](SharedBuffer&) { ++offered; });
  EXPECT_TRUE(offered == 1);
}

// Block streaming must NOT cache. A memory-bounded DiT re-reads every
// block per forward precisely so the set does not hold them; caching one
// would put the whole model back in RAM on the box least able to take
// it, and it would do so silently.
TEST(weight_set, streaming_reads_are_never_retained) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);

  SharedBuffer a = ws->stream_tensor("encoder.w", mc);
  SharedBuffer b = ws->stream_tensor("encoder.w", mc);
  ASSERT_TRUE(!a.empty());
  ASSERT_TRUE(!b.empty());
  // Nothing retained, and nothing shared between the two reads.
  EXPECT_TRUE(ws->stats().entries == 0u);
  EXPECT_TRUE(ws->stats().bytes == 0u);
  // The bytes are still correct -- uncached, not unread.
  const auto* v = static_cast<const float*>(a.contents());
  EXPECT_TRUE(v[0] == ck.enc()[0]);

  // Counted as streaming throughput, which is how the manager sees what
  // a bounded model is paying to stay small.
  EXPECT_TRUE(ws->stats().streamed_reads == 2u);
  EXPECT_TRUE(ws->stats().streamed_bytes == 64u);

  int builds = 0;
  auto d = ws->stream_derived([&]() {
    ++builds;
    return mc->make_shared_buffer(16);
  });
  ASSERT_TRUE(!d.empty());
  ws->stream_derived([&]() {
    ++builds;
    return mc->make_shared_buffer(16);
  });
  EXPECT_TRUE(builds == 2);            // rebuilt, never cached
  EXPECT_TRUE(ws->stats().entries == 0u);
  EXPECT_TRUE(ws->stats().streamed_reads == 4u);
}

// The no-allocation streaming read. What matters is that it is the SAME
// bytes as the path it replaces -- it takes a completely different route
// to them (pread off the shard rather than a memcpy out of the mapping),
// so "faster" is only worth anything if "identical" holds.
//
// Both tensors are read, and deliberately: "trunk.w" starts at data
// offset 0 and "encoder.w" at 16, so one of the two is not page-aligned
// in the file. That is the ordinary case for safetensors and the one
// thing that could have made a raw read unusable here.
TEST(weight_set, stream_into_reads_the_same_bytes) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);

  for (const char* nm : {"trunk.w", "encoder.w"}) {
    SharedBuffer want = ws->read(nm, mc, WeightSet::Residency::Copied);
    ASSERT_TRUE(!want.empty());
    SharedBuffer got = mc->make_shared_buffer(want.byte_size());
    ASSERT_TRUE(!got.empty());
    ASSERT_TRUE(ws->stream_into(nm, got.contents(), got.byte_size()));
    EXPECT_TRUE(std::memcmp(got.contents(), want.contents(),
                            want.byte_size()) == 0);
  }
}

// The destination is the caller's, so nothing is retained -- and the
// bytes are still counted, because a model that streams this way is
// paying exactly the same throughput as one that streams the old way and
// the manager has to see it.
TEST(weight_set, stream_into_is_counted_and_retains_nothing) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);

  SharedBuffer slot = mc->make_shared_buffer(32);
  ASSERT_TRUE(!slot.empty());
  // The same destination, twice: this is the whole point of the call.
  ASSERT_TRUE(ws->stream_into("encoder.w", slot.contents(), 32));
  ASSERT_TRUE(ws->stream_into("encoder.w", slot.contents(), 32));
  EXPECT_TRUE(ws->stats().entries == 0u);
  EXPECT_TRUE(ws->stats().bytes == 0u);
  EXPECT_TRUE(ws->stats().streamed_reads == 2u);
  EXPECT_TRUE(ws->stats().streamed_bytes == 64u);
  EXPECT_TRUE(static_cast<const float*>(slot.contents())[0] == ck.enc()[0]);
}

// It refuses rather than half-serves. Each of these is a case where a
// caller must rebuild the buffer by another route, and a `true` that
// left the destination stale or short would be a wrong answer that runs.
TEST(weight_set, stream_into_refuses_what_it_cannot_serve) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);

  SharedBuffer slot = mc->make_shared_buffer(32);
  ASSERT_TRUE(!slot.empty());
  EXPECT_FALSE(ws->stream_into("nope.w", slot.contents(), 32));
  EXPECT_FALSE(ws->stream_into("encoder.w", slot.contents(), 31));
  EXPECT_FALSE(ws->stream_into("encoder.w", nullptr, 32));
  // A refusal is not a read.
  EXPECT_TRUE(ws->stats().streamed_reads == 0u);
}

// The per-tensor refill rule, which is what a block-streamed DiT reaches
// for once it keeps its destinations instead of allocating them.
//
// Raw dtypes land byte-for-byte; f16 is converted where it lies, because
// it is the same WIDTH as the bf16 the forward reads.
TEST(weight_set, refill_places_raw_and_converts_f16) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  MixedCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);

  SharedBuffer bf = mc->make_shared_buffer(16);
  SharedBuffer u32 = mc->make_shared_buffer(32);
  SharedBuffer h = mc->make_shared_buffer(16);
  ASSERT_TRUE(!bf.empty() && !u32.empty() && !h.empty());

  EXPECT_TRUE(refill_streamed_tensor(*ws, "raw.bf16", bf, RefillDst::kBf16)
              == Refill::kFilled);
  EXPECT_TRUE(std::memcmp(bf.contents(), ck.bf(), 16) == 0);

  EXPECT_TRUE(refill_streamed_tensor(*ws, "raw.u32", u32, RefillDst::kBf16)
              == Refill::kFilled);
  EXPECT_TRUE(std::memcmp(u32.contents(), ck.u32(), 32) == 0);

  EXPECT_TRUE(refill_streamed_tensor(*ws, "conv.f16", h, RefillDst::kBf16)
              == Refill::kFilled);
  const auto* got = static_cast<const std::uint16_t*>(h.contents());
  for (int i = 0; i < 8; ++i) {
    // Computed here rather than with the same helper, so this compares
    // the conversion against its definition and not against itself.
    const float f = (float)ck.h()[i];
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const std::uint16_t want =
        (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    EXPECT_TRUE(got[i] == want);
  }

  // REFILLING IS REPEATABLE. The whole point of a reusable destination is
  // that the next block writes over the last one, and an f16 tensor
  // converted twice would compound -- so the second pass has to land on
  // the same bytes as the first.
  EXPECT_TRUE(refill_streamed_tensor(*ws, "conv.f16", h, RefillDst::kBf16)
              == Refill::kFilled);
  const auto* again = static_cast<const std::uint16_t*>(h.contents());
  for (int i = 0; i < 8; ++i) {
    const float f = (float)ck.h()[i];
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    EXPECT_TRUE(again[i] ==
                (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
  }
}

// The two refusals, which mean opposite things to a caller: one says
// "build this tensor the way you always did", the other says "the buffer
// is not usable, rebuild it".
TEST(weight_set, refill_separates_unservable_from_failed) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  MixedCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);

  // f32 into a half-width destination: expected, cheap, and NOT a
  // failure -- this is the case that must not turn a whole block's
  // refill off. LTX-2.5 blocks carry six such tensors among 140.
  SharedBuffer half = mc->make_shared_buffer(16);
  ASSERT_TRUE(!half.empty());
  EXPECT_TRUE(refill_streamed_tensor(*ws, "wide.f32", half,
                                     RefillDst::kBf16)
              == Refill::kUnservable);

  // An empty destination is nothing to fill, not an error.
  SharedBuffer none;
  EXPECT_TRUE(refill_streamed_tensor(*ws, "raw.bf16", none,
                                     RefillDst::kBf16)
              == Refill::kUnservable);

  // A size that disagrees with the checkpoint IS an error: filling it
  // would write the wrong number of bytes and run.
  SharedBuffer wrong = mc->make_shared_buffer(24);
  ASSERT_TRUE(!wrong.empty());
  EXPECT_TRUE(refill_streamed_tensor(*ws, "raw.bf16", wrong,
                                     RefillDst::kBf16)
              == Refill::kFailed);

  // So is a tensor that is not there.
  SharedBuffer ok = mc->make_shared_buffer(16);
  EXPECT_TRUE(refill_streamed_tensor(*ws, "nope", ok, RefillDst::kBf16)
              == Refill::kFailed);

  // f16 under kRaw is unservable too, and THAT is the one worth pinning:
  // the same tensor is servable as bf16 and not as raw bytes, so a
  // caller that converts f16 by another route gets its own bytes back
  // rather than a second conversion.
  SharedBuffer h = mc->make_shared_buffer(16);
  ASSERT_TRUE(!h.empty());
  EXPECT_TRUE(refill_streamed_tensor(*ws, "conv.f16", h, RefillDst::kRaw)
              == Refill::kUnservable);
  EXPECT_TRUE(refill_streamed_tensor(*ws, "conv.f16", h, RefillDst::kBf16)
              == Refill::kFilled);

  // Only the servable one counted as a read.
  EXPECT_TRUE(ws->stats().streamed_reads == 1u);
}

// A tensor read BOTH ways keeps the two paths separate: streaming must
// not be satisfied from the cache, and must not populate it either.
TEST(weight_set, streaming_and_caching_do_not_cross_over) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);

  SharedBuffer cached = ws->tensor("trunk.w", mc);
  SharedBuffer streamed = ws->stream_tensor("trunk.w", mc);
  ASSERT_TRUE(!cached.empty());
  ASSERT_TRUE(!streamed.empty());
  EXPECT_TRUE(ws->stats().entries == 1u);       // only the cached one
  EXPECT_TRUE(ws->stats().streamed_reads == 1u);
  // Same bytes, independent handles.
  EXPECT_TRUE(static_cast<const float*>(streamed.contents())[0]
              == ck.trunk()[0]);
}

// The part machinery behind "no reference image => no VAE encoder": a
// part loads at most once, reports ready, and releases on demand.
TEST(weight_set, part_loads_once_and_releases) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);
  ws->tensor("trunk.w", mc);                        // trunk, no part

  EXPECT_FALSE(ws->part_ready("encoder"));

  int loads = 0;
  auto load = [&]() {
    ++loads;
    return !ws->tensor("encoder.w", mc, WeightSet::Residency::Copied,
                       "encoder").empty();
  };
  EXPECT_TRUE(ws->ensure_part("encoder", load));
  EXPECT_TRUE(ws->ensure_part("encoder", load));   // cached, not re-run
  EXPECT_TRUE(loads == 1);
  EXPECT_TRUE(ws->part_ready("encoder"));
  EXPECT_TRUE(ws->stats().entries == 2u);

  // Releasing drops the part's tensors and lets it load again; the trunk
  // is untouched.
  EXPECT_TRUE(ws->release_part("encoder") == 32u);
  EXPECT_FALSE(ws->part_ready("encoder"));
  EXPECT_TRUE(ws->stats().entries == 1u);
  EXPECT_TRUE(ws->ensure_part("encoder", load));
  EXPECT_TRUE(loads == 2);
}

// A part nobody asks for is never read. This is the requirement stated
// as "no load of the VAE encoder happens if there is no reference image
// input" -- the encoder's bytes stay on disk.
TEST(weight_set, an_unrequested_part_is_never_loaded) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);
  ws->tensor("trunk.w", mc);

  EXPECT_FALSE(ws->part_ready("encoder"));
  EXPECT_TRUE(ws->stats().parts == 0u);
  EXPECT_TRUE(ws->stats().entries == 1u);
  EXPECT_TRUE(ws->stats().bytes == 16u);
}

// Restoring reclaimed weights must write back into the EXISTING
// allocation: models hold aliases of it, and replacing the buffer would
// leave every one of them pointing at bytes nothing refreshes.
TEST(weight_set, reload_restores_in_place) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);
  SharedBuffer alias =
      ws->tensor("trunk.w", mc, WeightSet::Residency::Copied);
  ASSERT_TRUE(!alias.empty());
  auto* held = static_cast<float*>(alias.contents());

  // Stand in for a purge: scribble over the contents, then reload.
  held[0] = -1.0f;
  held[3] = -1.0f;
  EXPECT_TRUE(ws->reload_weights());
  // The ALIAS -- not a fresh handle -- sees the restored bytes, which is
  // the property a live model depends on.
  EXPECT_TRUE(held[0] == ck.trunk()[0]);
  EXPECT_TRUE(held[3] == ck.trunk()[3]);
  EXPECT_TRUE(static_cast<float*>(alias.contents()) == held);
}

// A missing tensor is empty, not a crash or a zero-filled buffer that
// would read as plausible weights.
TEST(weight_set, missing_tensor_is_empty) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;

  auto ws = WeightSet::open(ck.dir(), nullptr);
  ASSERT_TRUE(ws != nullptr);
  EXPECT_FALSE(ws->has("nope.w"));
  EXPECT_TRUE(ws->tensor("nope.w", mc).empty());
  EXPECT_TRUE(ws->stats().entries == 0u);
}

// The manager can say what is resident: which checkpoint, how much of
// it, how much is OS-reclaimable versus owned, and how many models are
// keeping it alive. Routing loads through the manager is what makes
// that question answerable at all.
TEST(weight_set, manager_reports_what_is_resident) {
  Session s;
  MetalCompute* mc = mc_(s);
  auto* mgr = s.generative_model_manager();
  if (mc == nullptr || mgr == nullptr) { return; }
  TempCheckpoint ck;

  EXPECT_TRUE(mgr->weight_report().empty());

  auto a = mgr->weight_set(ck.dir());
  ASSERT_TRUE(a != nullptr);
  a->tensor("trunk.w", mc);                              // mapped
  a->tensor("encoder.w", mc, WeightSet::Residency::Copied, "encoder");
  a->ensure_part("encoder", []() { return true; });

  auto rep = mgr->weight_report();
  ASSERT_TRUE(rep.size() == 1u);
  EXPECT_TRUE(rep[0].dir == a->dir());
  EXPECT_TRUE(rep[0].tensors == 2u);
  EXPECT_TRUE(rep[0].bytes == 48u);
  EXPECT_TRUE(rep[0].mapped_bytes == 16u);
  EXPECT_TRUE(rep[0].copied_bytes == 32u);
  EXPECT_TRUE(rep[0].parts == 1u);
  EXPECT_TRUE(rep[0].holders == 1);          // just `a`

  {
    auto b = mgr->weight_set(ck.dir());
    (void)b;
    EXPECT_TRUE(mgr->weight_report()[0].holders == 2);
  }
  EXPECT_TRUE(mgr->weight_report()[0].holders == 1);
}

// A directory with no checkpoint fails cleanly rather than producing a
// set that yields empty tensors forever.
TEST(weight_set, open_fails_on_a_directory_with_no_checkpoint) {
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  namespace fs = std::filesystem;
  const string dir =
      (fs::temp_directory_path() / "vpipe-ws-empty-test").string();
  fs::create_directories(dir);
  EXPECT_TRUE(WeightSet::open(dir, nullptr) == nullptr);
  EXPECT_TRUE(mgr->weight_set(dir) == nullptr);
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// open_weight_set() is what models actually call. Given a session it must
// hand back the manager's set -- that is the whole dedup mechanism, and a
// model that quietly got a private one instead would still WORK, just
// with a second copy of the weights nobody could see. So assert identity
// with the manager's own lookup, not merely that a set came back.
TEST(weight_set, open_weight_set_returns_the_managers_set) {
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  TempCheckpoint ck;

  auto a = open_weight_set(ck.dir(), &s);
  ASSERT_TRUE(a != nullptr);
  EXPECT_TRUE(a == mgr->weight_set(ck.dir()));   // same object
  EXPECT_TRUE(mgr->weight_set_count() == 1u);

  // A second caller naming the same directory joins the first rather
  // than opening the checkpoint again.
  auto b = open_weight_set(ck.dir(), &s);
  EXPECT_TRUE(b == a);
  EXPECT_TRUE(mgr->weight_set_count() == 1u);
}

// Without a session there is no manager to ask. The fallback must still
// produce a usable set -- the offline tools (quantize, calibration,
// lora-fuse) and the tests run this way -- just an unshared one.
TEST(weight_set, open_weight_set_falls_back_to_a_private_set) {
  TempCheckpoint ck;
  auto a = open_weight_set(ck.dir(), nullptr);
  auto b = open_weight_set(ck.dir(), nullptr);
  ASSERT_TRUE(a != nullptr && b != nullptr);
  EXPECT_TRUE(a != b);                  // private: no sharing, by design
  EXPECT_TRUE(a->dir() == ck.dir());
}

// Two models over one checkpoint share a tensor they BOTH want and
// neither pays for one the other doesn't -- the property that makes the
// per-tensor cache better than the model-level dedup it replaced, which
// had to guess which model would be built first.
TEST(weight_set, two_holders_share_only_what_they_both_ask_for) {
  Session s;
  MetalCompute* mc = mc_(s);
  auto* mgr = s.generative_model_manager();
  if (mc == nullptr || mgr == nullptr) { return; }
  TempCheckpoint ck;

  // First "model": wants only the trunk.
  auto a = open_weight_set(ck.dir(), &s);
  ASSERT_TRUE(a != nullptr);
  SharedBuffer ta = a->tensor("trunk.w", mc, WeightSet::Residency::Copied);
  ASSERT_TRUE(!ta.empty());
  EXPECT_TRUE(a->stats().entries == 1u);

  // Second "model": wants the trunk too, plus one the first never asked
  // for. The shared tensor is the SAME bytes; the extra one is new.
  auto b = open_weight_set(ck.dir(), &s);
  ASSERT_TRUE(b == a);
  SharedBuffer tb = b->tensor("trunk.w", mc, WeightSet::Residency::Copied);
  SharedBuffer eb = b->tensor("encoder.w", mc, WeightSet::Residency::Copied);
  ASSERT_TRUE(!tb.empty() && !eb.empty());
  EXPECT_TRUE(tb.contents() == ta.contents());   // shared, not copied
  EXPECT_TRUE(eb.contents() != ta.contents());
  EXPECT_TRUE(a->stats().entries == 2u);
}

// The window the restore bracket closes. The registry drops ITS lock
// between taking the buffers back and reloading them; in that gap the
// entries are no longer marked parked but their contents were
// discarded. A reader arriving then would hit the cache and be handed
// an alias to garbage. begin_restore() holds _mu across the whole span,
// so the reader WAITS for valid bytes instead.
//
// Asserted by observing that a concurrent tensor() genuinely does not
// return while a restore is open.
TEST(weight_set, a_reader_waits_for_an_open_restore) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;
  auto ws = WeightSet::open(ck.dir(), &s);
  ASSERT_TRUE(ws != nullptr);
  ASSERT_TRUE(!ws->tensor("trunk.w", mc, WeightSet::Residency::Copied)
                   .empty());

  ws->begin_restore();          // the registry's position, mid-restore

  std::atomic<bool> started{false};
  std::atomic<bool> returned{false};
  std::thread reader([&]() {
    started.store(true);
    // A cache HIT: without the bracket this returns instantly, pointing
    // at bytes the kernel may have discarded.
    auto t = ws->tensor("trunk.w", mc, WeightSet::Residency::Copied);
    returned.store(!t.empty());
  });

  // Wait for the reader to actually be in flight first, so "it did not
  // return" cannot pass vacuously because the thread never ran.
  while (!started.load()) { std::this_thread::yield(); }
  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  EXPECT_FALSE(returned.load());     // still blocked -- the point

  ws->end_restore(true);
  reader.join();
  EXPECT_TRUE(returned.load());      // and it completes once bytes are good
}

// end_restore() without a matching begin_restore() must not unlock a
// mutex this thread never took. Cheap, but the failure mode is a
// corrupted mutex rather than a clean assert, so it is worth pinning.
TEST(weight_set, an_unmatched_end_restore_is_inert) {
  Session s;
  TempCheckpoint ck;
  auto ws = WeightSet::open(ck.dir(), &s);
  ASSERT_TRUE(ws != nullptr);
  ws->end_restore(true);
  ws->end_restore(false);
  EXPECT_TRUE(ws->dir() == ck.dir());   // still usable
}

// Cached tensors are shared between models, so a write through one
// holder corrupts every other. Nothing in the type system prevents that
// -- SharedBuffer::contents() is a mutable pointer from a const method
// -- so with VPIPE_WEIGHT_INTEGRITY=1 the set hashes what it caches and
// can say whether the invariant actually held.
TEST(weight_set, integrity_check_catches_a_write_after_load) {
  ::setenv("VPIPE_WEIGHT_INTEGRITY", "1", 1);
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { ::unsetenv("VPIPE_WEIGHT_INTEGRITY"); return; }
  TempCheckpoint ck;
  auto ws = WeightSet::open(ck.dir(), &s);
  ASSERT_TRUE(ws != nullptr);
  ASSERT_TRUE(ws->integrity_enabled());

  // Copied, not Mapped: a mapped tensor is PROT_READ, so the write
  // below would fault rather than corrupt. That asymmetry is exactly
  // why the copied path needs this check and the mapped one does not.
  auto a = ws->tensor("trunk.w", mc, WeightSet::Residency::Copied);
  ASSERT_TRUE(!a.empty());
  EXPECT_TRUE(ws->verify_integrity() == 0u);

  // A second holder, as a real peer model would be.
  auto b = ws->tensor("trunk.w", mc, WeightSet::Residency::Copied);
  ASSERT_TRUE(b.contents() == a.contents());

  // Now break the rule through one alias.
  static_cast<unsigned char*>(a.contents())[0] ^= 0xff;
  EXPECT_TRUE(ws->verify_integrity() == 1u);

  // Put it back and the set is clean again -- the check reports the
  // current state, it does not latch.
  static_cast<unsigned char*>(a.contents())[0] ^= 0xff;
  EXPECT_TRUE(ws->verify_integrity() == 0u);
  ::unsetenv("VPIPE_WEIGHT_INTEGRITY");
}

// Off by default: hashing every tensor at load is a full pass over the
// weights, which is seconds on a 20 GB model and buys nothing in
// production. verify_integrity() then reports 0 because it checked
// nothing, which is why integrity_enabled() exists to tell the two
// apart.
TEST(weight_set, integrity_is_off_unless_asked_for) {
  ::unsetenv("VPIPE_WEIGHT_INTEGRITY");
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  TempCheckpoint ck;
  auto ws = WeightSet::open(ck.dir(), &s);
  ASSERT_TRUE(ws != nullptr);
  EXPECT_FALSE(ws->integrity_enabled());
  auto a = ws->tensor("trunk.w", mc, WeightSet::Residency::Copied);
  ASSERT_TRUE(!a.empty());
  static_cast<unsigned char*>(a.contents())[0] ^= 0xff;
  EXPECT_TRUE(ws->verify_integrity() == 0u);   // not checked, not clean
}

// A reload rewrites the bytes in place, so it must re-arm the hash in
// the same critical section -- otherwise the very next verify would
// report the registry's own legitimate write as corruption.
TEST(weight_set, reload_rearms_the_integrity_hash) {
  ::setenv("VPIPE_WEIGHT_INTEGRITY", "1", 1);
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { ::unsetenv("VPIPE_WEIGHT_INTEGRITY"); return; }
  TempCheckpoint ck;
  auto ws = WeightSet::open(ck.dir(), &s);
  ASSERT_TRUE(ws != nullptr);
  auto a = ws->tensor("trunk.w", mc, WeightSet::Residency::Copied);
  ASSERT_TRUE(!a.empty());

  // Scribble as if the pages had been discarded, then restore.
  static_cast<unsigned char*>(a.contents())[0] ^= 0xff;
  EXPECT_TRUE(ws->verify_integrity() == 1u);
  EXPECT_TRUE(ws->reload_weights());
  EXPECT_TRUE(ws->verify_integrity() == 0u);
  ::unsetenv("VPIPE_WEIGHT_INTEGRITY");
}

// Streaming throughput under concurrency -- the case requirement (4)
// names: two image pipelines sharing one checkpoint. Streaming implies
// Copied (a model that streams turns mmap off), so every block read is a
// full memcpy; if that ran under the set's lock the two pipelines would
// take turns on each other's reads instead of overlapping.
//
// Reports the speedup of 2 threads over 1. Serialised, it sits near
// 1.0x; overlapping, it climbs toward 2x (bounded by memory bandwidth,
// so the bar is deliberately loose -- this is here to catch a
// REGRESSION to lockstep, not to certify a number).
//
//   vpipe_test --filter 'weight_set.streaming_reads_run_concurrently'
TEST(weight_set, streaming_reads_run_concurrently) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  // 4 MB: the order of a single quantised DiT block's codes, which is
  // what the streaming path actually re-reads per layer per step.
  BigCheckpoint ck(1024 * 1024);
  auto ws = WeightSet::open(ck.dir(), &s);
  ASSERT_TRUE(ws != nullptr);

  const int kReads = 300;
  auto burst = [&](int n) {
    for (int i = 0; i < n; ++i) {
      auto b = ws->stream_tensor("block.w", mc,
                                 WeightSet::Residency::Copied);
      if (b.empty()) { return false; }
    }
    return true;
  };

  burst(30);                               // warm allocator + page cache
  using clk = std::chrono::steady_clock;

  const auto t0 = clk::now();
  ASSERT_TRUE(burst(kReads * 2));
  const double serial_ms =
      std::chrono::duration<double, std::milli>(clk::now() - t0).count();

  // Best of three: a single parallel sample swings enough on a busy
  // machine to make a threshold either flaky or useless. The best run is
  // the one least perturbed by whatever else the box was doing.
  double par_ms = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    const auto t1 = clk::now();
    {
      std::thread a([&]() { burst(kReads); });
      std::thread b([&]() { burst(kReads); });
      a.join();
      b.join();
    }
    par_ms = std::min(
        par_ms,
        std::chrono::duration<double, std::milli>(clk::now() - t1).count());
  }

  const double speedup = par_ms > 0.0 ? serial_ms / par_ms : 0.0;
  std::printf("[ws_stream] %d x 4MB reads: 1 thread %.1f ms, 2 threads "
              "%.1f ms -> %.2fx\n", kReads * 2, serial_ms, par_ms, speedup);

  const auto st = ws->stats();
  EXPECT_TRUE(st.streamed_reads >= (std::size_t)(kReads * 4));
  EXPECT_TRUE(st.entries == 0u);           // streaming caches nothing

  // Loose floor. Two threads memcpying 4 MB blocks are bandwidth-bound,
  // so the ceiling is well under 2x. Measured on an M4 Pro: ~1.55-1.7x
  // overlapping, ~1.10x when read() holds the set's lock across the
  // copy. The bar sits between them, nearer the failure side, because
  // what this needs to catch is a regression to LOCKSTEP -- not to
  // certify any particular throughput.
  EXPECT_TRUE(speedup > 1.25);
}

// The cap parks rather than refuses. Parking hands pages to the kernel
// as purgeable: they survive when nothing else needs the RAM, so the
// usual cost is nothing at all, and the next access takes them back.
// That is why going over the cap degrades throughput instead of failing
// a load.
TEST(weight_set, the_memory_cap_parks_least_recently_used_first) {
  Session s;
  MetalCompute* mc = mc_(s);
  auto* mgr = s.generative_model_manager();
  if (mc == nullptr || mgr == nullptr) { return; }
  // 4 MB each. The tiny fixture above is useless here: mark_inactive()
  // declines heap sub-allocations, and a 32-byte buffer is one, so a
  // small checkpoint is simply not parkable and the test would prove
  // nothing.
  BigCheckpoint older(1024 * 1024), newer(1024 * 1024);
  const std::size_t kBytes = 4u * 1024 * 1024;

  auto a = mgr->weight_set(older.dir());
  auto b = mgr->weight_set(newer.dir());
  ASSERT_TRUE(a != nullptr && b != nullptr);
  ASSERT_TRUE(!a->tensor("block.w", mc, WeightSet::Residency::Copied)
                   .empty());
  ASSERT_TRUE(!b->tensor("block.w", mc, WeightSet::Residency::Copied)
                   .empty());
  // `b` is touched last, so `a` is the least-recently-used.
  (void)b->tensor("block.w", mc, WeightSet::Residency::Copied);
  EXPECT_TRUE(a->last_use() < b->last_use());
  EXPECT_TRUE(mgr->active_bytes() == 2 * kBytes);

  // LET GO OF BOTH before the cap runs. Nothing parks a set a model is
  // borrowing -- see an_unreachable_cap_parks_what_it_can_and_stops --
  // so holding these here would test the refusal rather than the order.
  // The manager keeps them either way, which is what makes observing
  // them afterwards possible at all.
  const std::string adir = a->dir(), bdir = b->dir();
  a.reset();
  b.reset();

  // A cap that fits only one of them parks exactly one, the older-touched.
  mgr->set_memory_cap(kBytes + kBytes / 2);
  bool a_parked = false, b_parked = false;
  for (const auto& u : mgr->weight_report()) {
    if (u.dir == adir) { a_parked = u.parked; }
    if (u.dir == bdir) { b_parked = u.parked; }
  }
  EXPECT_TRUE(a_parked);
  EXPECT_FALSE(b_parked);
  // Parked bytes belong to the kernel now, so they stop counting --
  // otherwise the policy would chase a number it can never reach.
  EXPECT_TRUE(mgr->active_bytes() == kBytes);
  // But they are still HELD: resident_weight_bytes() counts them.
  EXPECT_TRUE(mgr->resident_weight_bytes() == 2 * kBytes);

  // Measured BEFORE borrowing again, because borrowing is manager work
  // and manager work settles whatever nobody is holding: `b` is
  // unborrowed and unparked here, so the next call parks it too and
  // active_bytes() goes to 0. That is the policy working, not a leak.
  a = mgr->weight_set(adir);
  ASSERT_TRUE(a != nullptr);
  if (a == nullptr) { mgr->set_memory_cap(0); return; }

  // Touching the parked set takes it back, and the bytes it hands out
  // are still the file's -- a reactivation that silently served
  // discarded pages is the failure this guards.
  auto t = a->tensor("block.w", mc, WeightSet::Residency::Copied);
  ASSERT_TRUE(!t.empty());
  EXPECT_FALSE(a->parked());
  const auto* v = static_cast<const float*>(t.contents());
  EXPECT_TRUE(v[0] == 0.0f && v[5] == 5.0f);

  mgr->set_memory_cap(0);
  EXPECT_TRUE(mgr->memory_cap() == 0u);
}

// An uncapped manager must never park: the feature has to be inert
// unless it was asked for.
TEST(weight_set, no_cap_means_nothing_is_ever_parked) {
  Session s;
  MetalCompute* mc = mc_(s);
  auto* mgr = s.generative_model_manager();
  if (mc == nullptr || mgr == nullptr) { return; }
  TempCheckpoint ck;
  auto ws = mgr->weight_set(ck.dir());
  ASSERT_TRUE(ws != nullptr);
  ASSERT_TRUE(!ws->tensor("encoder.w", mc, WeightSet::Residency::Copied)
                   .empty());
  EXPECT_TRUE(mgr->memory_cap() == 0u);
  EXPECT_TRUE(mgr->enforce_memory_cap() == 0u);
  EXPECT_FALSE(ws->parked());
}

// THE CAP OBEYS THE BORROW RULE TOO, and this is the last place that
// did not.
//
// The cap parks least-recently-used weights to get under its target,
// and the least-recently-used set is very often one a live model is
// still holding -- an idle pipeline's DiT, a VAE between clips. Parking
// under that model makes its cached tensors purgeable, and what reads
// them next is a forward pass over aliases it already holds: it never
// asks the set for anything, so nothing reactivates, and if the kernel
// took the pages it reads zeros. The manager cannot tell a live model
// that is idle from one mid-forward, so it does not try.
//
// The cap is therefore a target twice over: mapped weights and KV are
// unparkable, and borrowed weights are not the manager's to park. What
// returns those is a stage letting go -- an `unload_when_idle` policy,
// not a smaller cap.
TEST(weight_set, an_unreachable_cap_parks_what_it_can_and_stops) {
  Session s;
  MetalCompute* mc = mc_(s);
  auto* mgr = s.generative_model_manager();
  if (mc == nullptr || mgr == nullptr) { return; }
  BigCheckpoint ck(1024 * 1024);
  auto ws = mgr->weight_set(ck.dir());
  ASSERT_TRUE(ws != nullptr);
  if (ws == nullptr) { return; }
  ASSERT_TRUE(!ws->tensor("block.w", mc, WeightSet::Residency::Copied)
                   .empty());

  mgr->set_memory_cap(1);              // unreachable
  // BORROWED, so the cap leaves it alone however far over it is...
  EXPECT_FALSE(ws->parked());
  // ...and asking again is a no-op rather than an endless retry.
  EXPECT_TRUE(mgr->enforce_memory_cap() == 0u);

  // Let go, and the same cap now has something it may take.
  ws.reset();
  EXPECT_TRUE(mgr->enforce_memory_cap() > 0u);
  auto back = mgr->weight_set(ck.dir());
  ASSERT_TRUE(back != nullptr);
  if (back != nullptr) { EXPECT_TRUE(back->parked()); }
  mgr->set_memory_cap(0);
}

// A misaligned pack does not WARN.
//
// It used to, and the finding was real -- F_NOCACHE silently falls back
// to buffered I/O on an unaligned offset, and a streamed checkpoint then
// grew file pages one-for-one with what it read. pread_into() fixed that
// by staging on page boundaries, which left the warning describing a
// solved problem, at OPEN (before anything knows whether this checkpoint
// will ever be streamed), with a remedy -- re-run vpipe's quantizer --
// that means nothing for a published VAE nobody quantized.
//
// So the bar is: opening one says nothing the user has to read. The fact
// is still recorded, but on the LOG delegate, which is where a fact that
// explains a throughput question goes.
namespace {

// Same shape as TempCheckpoint, with the header padded so the data
// section starts OFF a 16-byte boundary -- which is how a real one gets
// this way: the section sits at 8 + header_len, so its alignment is
// whatever the header length happened to be.
class MisalignedCheckpoint {
public:
  MisalignedCheckpoint()
  {
    namespace fs = std::filesystem;
    _dir = (fs::temp_directory_path() /
            ("vpipe-ws-mis-" + to_string(::getpid()))).string();
    fs::create_directories(_dir);
    float v[4] = {1.f, 2.f, 3.f, 4.f};
    string hdr = "{\"trunk.w\":{\"dtype\":\"F32\",\"shape\":[4],"
                 "\"data_offsets\":[0,16]}}";
    while (((8 + hdr.size()) % 16) != 8) { hdr.push_back(' '); }
    ofstream out(_dir + "/model.safetensors", ios::binary);
    const uint64_t n = hdr.size();
    out.write(reinterpret_cast<const char*>(&n), 8);
    out.write(hdr.data(), (streamsize)hdr.size());
    out.write(reinterpret_cast<const char*>(v), 16);
  }
  ~MisalignedCheckpoint()
  {
    std::error_code ec;
    std::filesystem::remove_all(_dir, ec);
  }
  const string& dir() const { return _dir; }
private:
  string _dir;
};

struct WarnCounter : public UiDelegateIntf {
  std::vector<std::string> lines;
  void error(const VpipeFormat& f) override { lines.push_back(f()); }
  void warn (const VpipeFormat& f) override { lines.push_back(f()); }
  void info (const VpipeFormat&) override {}
  UiInputStatus getline(const VpipeFormat&, std::string&,
                        const std::function<bool()>&) override
  {
    return UiInputStatus::Eof;
  }
  std::unique_ptr<UiTextStream> open_text_stream() override
  {
    return std::make_unique<NullUiTextStream>();
  }
};

}  // namespace

TEST(weight_set, a_misaligned_pack_does_not_warn)
{
  MisalignedCheckpoint ck;
  Session sess;
  auto  ui_owned = std::make_unique<WarnCounter>();
  auto* ui = ui_owned.get();
  sess.set_ui_delegate(std::move(ui_owned));

  // The fixture is actually misaligned, or this test passes by having
  // nothing to report.
  auto raw = genai::MetalLlamaWeights::open_model(ck.dir());
  ASSERT_TRUE(raw.has_value());
  if (!raw.has_value()) { return; }
  const auto a = raw->alignment();
  EXPECT_TRUE(a.misaligned > 0 || a.bad_shards > 0);

  ui->lines.clear();
  auto ws = genai::WeightSet::open(ck.dir(), &sess);
  EXPECT_TRUE(ws != nullptr);
  for (const std::string& l : ui->lines) {
    std::printf("[weight_set] unexpected: %s\n", l.c_str());
  }
  EXPECT_TRUE(ui->lines.empty());
}
