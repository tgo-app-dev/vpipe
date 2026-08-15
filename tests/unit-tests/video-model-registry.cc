// VideoModelRegistry: the VIDEO-model plugin extension point, and the
// catalogue-injection seam next to it. Both are what lets a family that
// does not ship with vpipe be resolvable by generate-video and fetchable
// by model-fetch.
//
// The generate-video CONSULT itself (a claim_for() before the built-in
// wan / minimax-h3 probes) needs a checkpoint on disk to exercise, so it
// is not tested here; this isolates the registry contract the consult
// depends on -- first-wins, claim order, a throwing probe not being
// fatal -- plus the catalogue lifetime rule, which is the part a plugin
// can get silently wrong.

#include "minitest.h"

#include "generative-models/video-model-registry.h"
#include "stages/model-catalog.h"

#include <memory>
#include <stdexcept>
#include <string>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

// A family that claims exactly the roots it was told to.
class StubFamily : public VideoModelFamily {
public:
  StubFamily(std::string tag, std::string want, int align = 0)
    : _tag(std::move(tag)), _want(std::move(want)), _align(align)
  {
  }

  std::string_view tag() const noexcept override { return _tag; }

  bool claims(const std::string& root, const std::string&) const override
  {
    return root == _want;
  }

  int align_frames(const std::string&, int frames) const override
  {
    if (_align <= 1) { return frames; }
    // Round UP to the next multiple-of-_align plus one, the shape every
    // video VAE rule in this tree has.
    const int k = (frames - 1 + _align - 1) / _align;
    return k * _align + 1;
  }

  std::unique_ptr<VideoGenerator> load(const VideoModelCreateArgs&) override
  {
    return nullptr;   // not exercised here
  }

private:
  std::string _tag;
  std::string _want;
  int         _align;
};

// A family whose probe throws. The registry must skip it and keep
// asking, because one broken plugin cannot be allowed to make every
// other family unreachable.
class ThrowingFamily : public VideoModelFamily {
public:
  std::string_view tag() const noexcept override { return "throws-on-claim"; }
  bool claims(const std::string&, const std::string&) const override
  {
    throw std::runtime_error("probe blew up");
  }
  std::unique_ptr<VideoGenerator> load(const VideoModelCreateArgs&) override
  {
    return nullptr;
  }
};

}  // namespace

TEST(video_model_registry, add_find_first_wins)
{
  VideoModelRegistry& reg = VideoModelRegistry::get();

  EXPECT_TRUE(reg.find("stub-fam-a") == nullptr);
  EXPECT_TRUE(reg.add(std::make_unique<StubFamily>("stub-fam-a",
                                                   "/tmp/stub-a")));
  ASSERT_TRUE(reg.find("stub-fam-a") != nullptr);

  // First-wins: a second family for the same tag is refused, and the
  // family already present keeps the tag.
  VideoModelFamily* first = reg.find("stub-fam-a");
  EXPECT_FALSE(reg.add(std::make_unique<StubFamily>("stub-fam-a",
                                                    "/tmp/somewhere-else")));
  EXPECT_TRUE(reg.find("stub-fam-a") == first);

  // A null family, and one that cannot name itself, are both refused.
  EXPECT_FALSE(reg.add(nullptr));
  EXPECT_FALSE(reg.add(std::make_unique<StubFamily>("", "/tmp/x")));
}

TEST(video_model_registry, claim_for_matches_only_its_own)
{
  VideoModelRegistry& reg = VideoModelRegistry::get();
  reg.add(std::make_unique<StubFamily>("stub-fam-b", "/tmp/stub-b"));

  VideoModelFamily* f = reg.claim_for(nullptr, "/tmp/stub-b", "");
  ASSERT_TRUE(f != nullptr);
  EXPECT_TRUE(f->tag() == "stub-fam-b");

  // A root nobody claims falls through to null, which is what leaves
  // generate-video on its built-in wan / minimax-h3 path.
  EXPECT_TRUE(reg.claim_for(nullptr, "/tmp/nothing-claims-this", "")
              == nullptr);
}

TEST(video_model_registry, throwing_probe_is_skipped_not_fatal)
{
  VideoModelRegistry& reg = VideoModelRegistry::get();
  reg.add(std::make_unique<ThrowingFamily>());
  reg.add(std::make_unique<StubFamily>("stub-fam-c", "/tmp/stub-c"));

  // The throwing family is asked (it was registered first among these)
  // and must not stop the walk.
  VideoModelFamily* f = reg.claim_for(nullptr, "/tmp/stub-c", "");
  ASSERT_TRUE(f != nullptr);
  EXPECT_TRUE(f->tag() == "stub-fam-c");
}

TEST(video_model_registry, align_frames_is_asked_of_the_family)
{
  StubFamily fam("stub-align", "/tmp/x", /*align=*/8);
  // LTX-2.5's rule: frames % 8 == 1.
  EXPECT_TRUE(fam.align_frames("/tmp/x", 121) == 121);
  EXPECT_TRUE(fam.align_frames("/tmp/x", 120) == 121);
  EXPECT_TRUE(fam.align_frames("/tmp/x", 122) == 129);
}

TEST(model_catalog, plugin_entries_are_appended_and_deduped)
{
  const std::size_t before = model_catalog().size();

  ModelCatalogEntry e;
  e.family      = "TestVendor";
  e.version     = "9.9";
  e.param_class = "1B";
  e.variant     = "unit test";
  e.hf_path     = "test-vendor/unit-test-model-xyz";
  e.model_type  = "test-xyz";

  EXPECT_TRUE(register_catalog_entries({e}) == 1);
  EXPECT_TRUE(model_catalog().size() == before + 1);

  // Findable exactly as a built-in entry is -- that is the point of
  // injecting rather than keeping a parallel list.
  const ModelCatalogEntry* got = catalog_by_path(e.hf_path);
  ASSERT_TRUE(got != nullptr);
  EXPECT_TRUE(got->model_type == "test-xyz");

  // The drill-down sees it too.
  bool in_families = false;
  for (const auto& f : catalog_families()) {
    if (f == "TestVendor") { in_families = true; }
  }
  EXPECT_TRUE(in_families);

  // Re-registering the same entry is a no-op: two plugins shipping one
  // repo must not produce two menu rows for one model.
  EXPECT_TRUE(register_catalog_entries({e}) == 0);
  EXPECT_TRUE(model_catalog().size() == before + 1);
}

TEST(model_catalog, pointers_survive_a_later_registration)
{
  // The lifetime rule register_catalog_entries promises: a pointer
  // handed out before a registration stays valid after it. A plugin
  // loading second must not invalidate what the first one's models
  // resolved to.
  ModelCatalogEntry a;
  a.family = "TestVendor2";  a.version = "1";  a.param_class = "1B";
  a.variant = "first";       a.hf_path = "test-vendor/lifetime-a";
  a.model_type = "test-life";
  ASSERT_TRUE(register_catalog_entries({a}) == 1);

  const ModelCatalogEntry* held = catalog_by_path("test-vendor/lifetime-a");
  ASSERT_TRUE(held != nullptr);
  const std::string family_before = held->family;

  ModelCatalogEntry b;
  b.family = "TestVendor3";  b.version = "1";  b.param_class = "1B";
  b.variant = "second";      b.hf_path = "test-vendor/lifetime-b";
  b.model_type = "test-life";
  ASSERT_TRUE(register_catalog_entries({b}) == 1);

  // `held` still points at a live, unchanged entry.
  EXPECT_TRUE(held->family == family_before);
  EXPECT_TRUE(held->hf_path == "test-vendor/lifetime-a");
  EXPECT_TRUE(catalog_by_path("test-vendor/lifetime-b") != nullptr);
}
