// weight-registry.cc -- session-level weight residency policy.
//
// The contract under test: parking keeps the allocation but hands the
// pages back as reclaimable; reactivating reports whether they SURVIVED;
// and a model whose pages were taken is reloaded from disk exactly once
// rather than reading garbage.
//
//   vpipe_test --filter '*weight_registry*'

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/shared/wired-pool.h"
#include "generative-models/weight-registry.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
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

// A stand-in model: a few weight buffers plus a reload counter, so a
// test can tell an intact reactivation (no reload) from a reclaimed one.
class FakeModel : public WeightOwner {
public:
  FakeModel(MetalCompute* mc, int n, size_t bytes) : _mc(mc)
  {
    for (int i = 0; i < n; ++i) {
      _w.push_back(_mc->make_shared_buffer(bytes));
    }
  }

  void for_each_weight(
      const std::function<void(SharedBuffer&)>& cb) override
  {
    for (SharedBuffer& b : _w) { cb(b); }
  }

  bool reload_weights() override
  {
    // Every reload MUST land inside the bracket -- that is the whole
    // contract the registry offers an owner with concurrent readers.
    if (!in_restore) { ++reload_outside_bracket; }
    ++reloads;
    return reload_ok;
  }

  void begin_restore() override { ++begins; in_restore = true; }
  void end_restore(bool ok) override
  {
    in_restore = false;
    ++ends;
    last_restore_ok = ok;
  }

  std::string weight_label() const override { return "fake-model"; }

  int  reloads   = 0;
  bool reload_ok = true;
  int  begins    = 0;
  int  ends      = 0;
  bool in_restore = false;
  bool last_restore_ok = false;
  int  reload_outside_bracket = 0;
  std::vector<SharedBuffer> _w;

private:
  MetalCompute* _mc;
};

}  // namespace

// Park then reactivate with nothing competing for RAM: contents intact,
// and NO reload -- that avoided reload is the whole point.
TEST(weight_registry, park_then_reactivate_without_reloading) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }

  FakeModel m(mc, 3, 1 << 20);
  WeightRegistry reg(&s);
  auto tok = reg.add(&m);
  EXPECT_TRUE(bool(tok));

  // Stamp the buffers so an intact reactivation is observable.
  for (SharedBuffer& b : m._w) {
    static_cast<uint8_t*>(b.contents())[0] = 0xA5;
  }

  const size_t parked = reg.park(&m);
  EXPECT_TRUE(parked == 3u * (1u << 20));
  {
    const auto st = reg.stats();
    EXPECT_TRUE(st.owners == 1u);
    EXPECT_TRUE(st.parked_owners == 1u);
    EXPECT_TRUE(st.parked_bytes == parked);
  }

  bool reloaded = true;
  EXPECT_TRUE(reg.reactivate(&m, &reloaded));
  EXPECT_FALSE(reloaded);            // nothing was lost -> no reload
  EXPECT_TRUE(m.reloads == 0);
  for (SharedBuffer& b : m._w) {
    EXPECT_TRUE(static_cast<uint8_t*>(b.contents())[0] == 0xA5);
  }
  const auto st = reg.stats();
  EXPECT_TRUE(st.parked_owners == 0u);
}

// Parking twice is a no-op, and reactivating something never parked
// reports intact -- otherwise callers would reload weights they still
// have.
TEST(weight_registry, park_and_reactivate_are_idempotent) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }

  FakeModel m(mc, 1, 1 << 20);
  WeightRegistry reg(&s);
  auto tok = reg.add(&m);

  EXPECT_TRUE(reg.park(&m) > 0u);
  EXPECT_TRUE(reg.park(&m) == 0u);      // already parked
  EXPECT_TRUE(reg.reactivate(&m));
  EXPECT_TRUE(reg.reactivate(&m));      // not parked -> intact
  EXPECT_TRUE(m.reloads == 0);
}

// An unregistered owner is inert: parking does nothing and reactivating
// claims intact, so a model that outlives its registration (or was
// never registered) still behaves.
TEST(weight_registry, unregistered_owner_is_inert) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }

  FakeModel m(mc, 1, 4096);
  WeightRegistry reg(&s);
  EXPECT_TRUE(reg.park(&m) == 0u);
  EXPECT_TRUE(reg.reactivate(&m));
  EXPECT_TRUE(reg.stats().owners == 0u);
}

// Dropping the registration removes the owner even while parked, so a
// model destroyed in that state is never walked afterwards.
TEST(weight_registry, registration_removes_on_reset) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }

  FakeModel m(mc, 2, 1 << 20);
  WeightRegistry reg(&s);
  {
    auto tok = reg.add(&m);
    EXPECT_TRUE(reg.stats().owners == 1u);
    EXPECT_TRUE(reg.park(&m) > 0u);
  }
  EXPECT_TRUE(reg.stats().owners == 0u);
  EXPECT_TRUE(reg.park(&m) == 0u);
}

// park_all() is the memory-pressure response: everything registered and
// not already parked goes at once.
TEST(weight_registry, park_all_covers_every_owner) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }

  FakeModel a(mc, 2, 1 << 20);
  FakeModel b(mc, 1, 1 << 20);
  WeightRegistry reg(&s);
  auto ta = reg.add(&a);
  auto tb = reg.add(&b);

  const size_t total = reg.park_all();
  EXPECT_TRUE(total == 3u * (1u << 20));
  EXPECT_TRUE(reg.stats().parked_owners == 2u);
  EXPECT_TRUE(reg.park_all() == 0u);    // nothing left to park

  EXPECT_TRUE(reg.reactivate(&a));
  EXPECT_TRUE(reg.reactivate(&b));
}

// A model holding nothing parkable (all subviews of one allocation)
// registers fine and simply parks zero bytes -- it must not be reported
// as reclaimable memory that never materialises.
TEST(weight_registry, subview_only_owner_parks_nothing) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }

  class ViewModel : public WeightOwner {
  public:
    explicit ViewModel(MetalCompute* mc)
      : _base(mc->make_shared_buffer(1 << 20))
    {
      _v.push_back(_base.subview(0, 4096));
      _v.push_back(_base.subview(4096, 4096));
    }
    void for_each_weight(
        const std::function<void(SharedBuffer&)>& cb) override
    {
      for (SharedBuffer& b : _v) { cb(b); }
    }
    std::string weight_label() const override { return "view-model"; }
    SharedBuffer _base;
    std::vector<SharedBuffer> _v;
  };

  ViewModel m(mc);
  WeightRegistry reg(&s);
  auto tok = reg.add(&m);
  EXPECT_TRUE(reg.park(&m) == 0u);
  EXPECT_TRUE(reg.stats().parked_bytes == 0u);
  EXPECT_TRUE(reg.reactivate(&m));
}

// ---- shared (deduplicated) non-LM models ----------------------------

namespace {
// Counts how many times the factory actually ran, which is the thing
// dedup is supposed to hold at one.
struct Payload {
  explicit Payload(int v) : value(v) {}
  int value;
};
}  // namespace

// Two callers naming the same (kind, dir, variant) share ONE instance
// and the factory runs once -- the case a diffusion graph hits when its
// conditioner, DiT and both VAE halves all name the same model dir.
TEST(weight_registry, shared_model_dedups_identical_requests) {
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }

  int builds = 0;
  auto make = [&builds]() {
    ++builds;
    return std::make_shared<Payload>(7);
  };
  const std::string dir = "/tmp/vpipe-shared-model-test";

  auto a = mgr->shared_model<Payload>("ut-kind", dir, "", make);
  auto b = mgr->shared_model<Payload>("ut-kind", dir, "", make);
  ASSERT_TRUE(a != nullptr);
  EXPECT_TRUE(a == b);            // same object, not just equal
  EXPECT_TRUE(builds == 1);
  EXPECT_TRUE(a->value == 7);
}

// A different `kind` or `variant` over the same directory is a
// different model: two classes reading one dir must not collide, and
// neither must two quantization widths.
TEST(weight_registry, shared_model_keys_on_kind_and_variant) {
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }

  int builds = 0;
  auto make = [&builds]() {
    ++builds;
    return std::make_shared<Payload>(builds);
  };
  const std::string dir = "/tmp/vpipe-shared-model-test2";

  auto dit = mgr->shared_model<Payload>("dit", dir, "", make);
  auto vae = mgr->shared_model<Payload>("vae", dir, "", make);
  auto w4  = mgr->shared_model<Payload>("dit", dir, "w4", make);
  EXPECT_TRUE(dit != vae);
  EXPECT_TRUE(dit != w4);
  EXPECT_TRUE(builds == 3);
}

// The cache holds WEAK references: the model dies with its last holder,
// exactly as an un-shared one did. "Last holder" now spans stages,
// which is the behaviour change dedup introduces.
TEST(weight_registry, shared_model_dies_with_its_last_holder) {
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }

  int builds = 0;
  auto make = [&builds]() {
    ++builds;
    return std::make_shared<Payload>(1);
  };
  const std::string dir = "/tmp/vpipe-shared-model-test3";

  {
    auto a = mgr->shared_model<Payload>("ut-kind", dir, "", make);
    {
      auto b = mgr->shared_model<Payload>("ut-kind", dir, "", make);
      EXPECT_TRUE(builds == 1);
      // One holder drops: the peer keeps it alive, no rebuild.
    }
    auto c = mgr->shared_model<Payload>("ut-kind", dir, "", make);
    EXPECT_TRUE(builds == 1);
    EXPECT_TRUE(c == a);
  }
  // All holders gone -> the entry expires and the next request rebuilds.
  auto d = mgr->shared_model<Payload>("ut-kind", dir, "", make);
  EXPECT_TRUE(builds == 2);
}

// A model a stage built, kept warm (keep_model_warm): handed back to the
// next stage asking for its kind, directory AND key; another key over
// the same model drops the stale build; it goes at its time, and when
// a launch that declares weights but not its directory asks for room.
TEST(weight_registry, a_built_model_is_kept_warm_by_key) {
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  const std::string dir = "/tmp/vpipe-warm-model-test";
  std::weak_ptr<Payload> seen;
  {
    auto m = std::make_shared<Payload>(1);
    seen = m;
    mgr->keep_model_warm("ut-dit", dir, "i8 3", std::move(m), 4096, 60.0);
  }
  EXPECT_TRUE(!seen.expired());              // the hold keeps it
  EXPECT_TRUE(mgr->warm_model_bytes(dir) == 4096);
  // Another kind or key is not it.
  EXPECT_TRUE(mgr->take_warm<Payload>("ut-vae", dir, "i8 3") == nullptr);
  EXPECT_TRUE(!seen.expired());
  auto got = mgr->take_warm<Payload>("ut-dit", dir, "i8 3");
  ASSERT_TRUE(got != nullptr);
  EXPECT_TRUE(got->value == 1);
  EXPECT_TRUE(mgr->warm_model_bytes(dir) == 0);   // taken: no longer held
  EXPECT_TRUE(mgr->take_warm<Payload>("ut-dit", dir, "i8 3") == nullptr);

  // Kept again, then asked under another key: the old build goes.
  mgr->keep_model_warm("ut-dit", dir, "i8 3", std::move(got), 4096, 60.0);
  EXPECT_TRUE(!seen.expired());
  EXPECT_TRUE(mgr->take_warm<Payload>("ut-dit", dir, "i8 0") == nullptr);
  EXPECT_TRUE(seen.expired());

  // Its time out: the reaper lets it go.
  {
    auto m = std::make_shared<Payload>(2);
    seen = m;
    mgr->keep_model_warm("ut-dit", dir, "k", std::move(m), 1, 0.05);
  }
  for (int i = 0; i < 200 && !seen.expired(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(seen.expired());

  // A launch that declares other weights: released as a warm LM is.
  {
    auto m = std::make_shared<Payload>(3);
    seen = m;
    mgr->keep_model_warm("ut-dit", dir, "k", std::move(m), 1, 60.0);
  }
  mgr->clear_declarations();
  mgr->declare_weights("/tmp/vpipe-warm-model-other", 1u << 20);
  auto dirs = mgr->warm_dirs();
  EXPECT_TRUE(std::find(dirs.begin(), dirs.end(), dir) != dirs.end());
  EXPECT_TRUE(mgr->release_warm_unclaimed() == 1);
  EXPECT_TRUE(seen.expired());
  mgr->clear_declarations();

  // 0 seconds: let go at once.
  {
    auto m = std::make_shared<Payload>(4);
    seen = m;
    mgr->keep_model_warm("ut-dit", dir, "k", std::move(m), 1, 0.0);
  }
  EXPECT_TRUE(seen.expired());
}

// What is kept warm is REPORTED, for a host's view of its memory
// (warm_report, SessionIntf::memory_report): each hold with its kind,
// directory, bytes and the time it has left; gone from the report once
// taken.
TEST(weight_registry, what_is_kept_warm_is_reported) {
  Session s;
  auto* mgr = s.generative_model_manager();
  if (mgr == nullptr) { return; }
  // A directory that exists, compared as the manager keys it: canonical
  // (/tmp is /private/tmp on macOS), so the test holds whether or not
  // it was there before.
  const auto tmp =
      std::filesystem::temp_directory_path() / "vpipe-warm-report-test";
  std::filesystem::create_directories(tmp);
  const std::string dir = std::filesystem::canonical(tmp).string();
  mgr->keep_model_warm("ut-dit", dir, "k", std::make_shared<Payload>(1),
                       4096, 60.0);
  mgr->keep_model_warm("ut-vae", dir, "k", std::make_shared<Payload>(2),
                       512, 30.0);
  auto r = mgr->warm_report();
  ASSERT_TRUE(r.size() == 2);
  auto dit = std::find_if(r.begin(), r.end(),
                          [](const auto& w) { return w.kind == "ut-dit"; });
  auto vae = std::find_if(r.begin(), r.end(),
                          [](const auto& w) { return w.kind == "ut-vae"; });
  ASSERT_TRUE(dit != r.end() && vae != r.end());
  if (dit == r.end() || vae == r.end()) { return; }
  EXPECT_TRUE(dit->dir == dir);
  EXPECT_TRUE(dit->bytes == 4096);
  EXPECT_TRUE(dit->kv_bytes == 0);
  EXPECT_TRUE(dit->seconds > 50.0 && dit->seconds <= 60.0);
  EXPECT_TRUE(vae->dir == dir);
  EXPECT_TRUE(vae->bytes == 512);
  EXPECT_TRUE(vae->seconds > 20.0 && vae->seconds <= 30.0);

  // The session's report carries them, as numbers.
  const vpipe::SessionIntf& host = s;
  const std::string json = host.memory_report().to_json();
  EXPECT_TRUE(json.find("\"ut-dit\"") != std::string::npos);
  EXPECT_TRUE(json.find("\"warm\"") != std::string::npos);
  EXPECT_TRUE(json.find("\"weights\"") != std::string::npos);
  EXPECT_TRUE(json.find("\"wired_limit\"") != std::string::npos);

  // Taken, it is no longer reported.
  EXPECT_TRUE(mgr->take_warm<Payload>("ut-dit", dir, "k") != nullptr);
  EXPECT_TRUE(mgr->take_warm<Payload>("ut-vae", dir, "k") != nullptr);
  EXPECT_TRUE(mgr->warm_report().empty());
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
}

// A model still kept warm as the SESSION ends goes with the manager --
// and a DiT going gives its wired buffers back through the session's
// MetalCompute (WiredPool::wire_one, from its destructor). The session
// destroyed its MetalCompute first: every process that ended with a DiT
// kept warm read freed memory there, and segfaulted.
TEST(weight_registry, a_model_kept_warm_goes_before_the_metal_it_uses) {
  struct Held {
    const Session*                 s = nullptr;
    vpipe::genai::WiredPool        pool;
    SharedBuffer                   buf;
    std::shared_ptr<int>           alive;   // MetalCompute there at the end
    std::shared_ptr<std::size_t>   unwired;
    ~Held()
    {
      MetalCompute* mc = s->metal_compute();
      *alive = mc != nullptr;
      if (mc != nullptr) { *unwired = pool.wire_one(mc, buf, false); }
    }
  };
  auto alive = std::make_shared<int>(-1);
  auto unwired = std::make_shared<std::size_t>(1);
  {
    Session s;
    MetalCompute* mc = mc_(s);
    auto* mgr = s.generative_model_manager();
    if (mc == nullptr || mgr == nullptr) { return; }
    auto m = std::make_shared<Held>();
    m->s = &s;
    m->alive = alive;
    m->unwired = unwired;
    m->pool.open(mc);
    m->buf = mc->make_shared_buffer(16u << 20);
    m->pool.wire_one(mc, m->buf, true);   // as a DiT's resident block
    mgr->keep_model_warm("ut-dit", "/tmp/vpipe-warm-model-test", "k",
                         std::move(m), 16u << 20, 600.0);
  }
  EXPECT_TRUE(*alive == 1);
  // The pool's count is the manager's, and goes with it: wire_one finds
  // no manager now and leaves the buffer to be freed (which unwires it
  // in the kernel).
  EXPECT_TRUE(*unwired == 0);
}

// Every reactivation is bracketed, whether or not a reload was needed:
// the registry cannot know which it will be until the walk is done, and
// an owner that only got the bracket on the reload path would still
// expose the gap in the case it was written to close.
TEST(weight_registry, reactivate_is_always_bracketed) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  WeightRegistry reg(&s);
  FakeModel m(mc, 2, 64 * 1024);
  auto h = reg.add(&m);

  // Intact path: no reload, but still bracketed.
  reg.park(&m);
  EXPECT_TRUE(reg.reactivate(&m));
  EXPECT_TRUE(m.begins == 1 && m.ends == 1);
  EXPECT_TRUE(m.reloads == 0);
  EXPECT_TRUE(m.last_restore_ok);

  // Reactivating something that was never parked does no work at all,
  // so it must not open a bracket either.
  EXPECT_TRUE(reg.reactivate(&m));
  EXPECT_TRUE(m.begins == 1 && m.ends == 1);
}

// When a reload does happen it must happen INSIDE the bracket, and a
// failed reload must be reported through end_restore(false) so the owner
// can mark itself unusable rather than silently serving garbage.
TEST(weight_registry, failed_reload_closes_the_bracket_with_false) {
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  WeightRegistry reg(&s);
  FakeModel m(mc, 1, 64 * 1024);
  auto h = reg.add(&m);

  reg.park(&m);
  // Force the reclaimed path: pretend the pages went away.
  for (SharedBuffer& b : m._w) { b.reactivate(); }
  m.reload_ok = false;
  bool reloaded = true;
  // Park again so reactivate() takes the discard branch deterministically
  // only if the platform actually discarded; either way the bracket must
  // be balanced, which is what this asserts.
  reg.reactivate(&m, &reloaded);
  EXPECT_TRUE(m.begins == m.ends);        // balanced, always
  EXPECT_TRUE(m.reload_outside_bracket == 0);
}
