#include "minitest.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "stages/model-catalog.h"
#include "stages/model-source.h"
#include "stages/resumable-fetch.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace std;
using namespace vpipe;
namespace fs = std::filesystem;

// LIVE checks against modelscope.cn -- gated on VPIPE_MODELSCOPE_LIVE=1
// because they need the network and a route to a Chinese host, neither of
// which a default test run should assume.
//
// What they are FOR: model-source.cc encodes claims about a server this
// tree does not control -- that the tree API answers in a particular
// shape, that the default branch is `master`, that every file carries a
// content sha256, and that each catalogue entry's pinned files exist
// under the mirrored path. Those cannot be proven by a captured page,
// which only pins what the parser does with an answer we already have.
//
// They deliberately do NOT download a model. The five families here are
// 35-500 GB apiece; what is checked is the LISTING (free) plus, once, a
// small real file end to end through fetch_file -- enough to prove the
// URL shape, the resume/verify path and the published digest, without
// moving a checkpoint.

namespace {

bool
live_()
{
  const char* e = std::getenv("VPIPE_MODELSCOPE_LIVE");
  return e != nullptr && *e == '1';
}

// The listing for `hf_path` as this source serves it, or an empty vector
// with `err` set.
vector<HfFile>
list_(const ModelSource& src, const string& hf_path, string& resolved,
      string& err)
{
  const MirrorStatus st = mirror_repo(src.name(), hf_path, resolved);
  if (st == MirrorStatus::Absent) {
    err = "absent on " + string(src.label());
    return {};
  }
  string body;
  long   status = 0;
  if (!http_get_text(src.tree_url(resolved, src.default_revision()), "",
                     true, 120, body, status, err)) {
    return {};
  }
  FlexData tree;
  try {
    tree = FlexData::from_json(body);
  } catch (const std::exception& e) {
    err = string("bad JSON: ") + e.what();
    return {};
  }
  return src.parse_tree(tree);
}

// What one entry's pinned files looked like on the live mirror. The
// minitest macros are members of the test class, so a helper cannot
// assert -- it counts, and the TEST body does the asserting.
struct PinReport {
  bool   listed = false;   // the tree API answered with files
  size_t pinned = 0;
  size_t missing = 0;      // pinned, but not in the listing
  size_t no_size = 0;      // listed with no published size
  size_t no_digest = 0;    // listed with no published checksum
};

PinReport
check_entry_(const ModelSource& src, const ModelCatalogEntry& e)
{
  PinReport r;
  r.pinned = e.files.size();
  string resolved, err;
  const vector<HfFile> files = list_(src, e.hf_path, resolved, err);
  if (files.empty()) {
    std::printf("  [%s] LISTING FAILED (%s)\n", e.hf_path.c_str(),
                err.c_str());
    return r;
  }
  r.listed = true;
  if (e.files.empty()) {
    // A whole-repo entry (SenseNova-U1.5) pins nothing, so there is no
    // per-file check to run and the LISTING is the whole of it. Say so:
    // "0 pinned, 0 missing" would otherwise read as a check that passed
    // rather than one that had nothing to do.
    std::printf("  [%s -> %s] whole repo (no pinned subset), %zu files "
                "listed\n", e.hf_path.c_str(), resolved.c_str(),
                files.size());
    return r;
  }
  for (const string& want : e.files) {
    const HfFile* f = nullptr;
    for (const HfFile& c : files) {
      if (c.path == want) { f = &c; break; }
    }
    if (f == nullptr) {
      std::printf("  [%s] MISSING pinned file '%s'\n", resolved.c_str(),
                  want.c_str());
      ++r.missing;
      continue;
    }
    // A pin that resolves but publishes no size or no checksum would
    // download unverifiable -- worth failing on, not just noting.
    if (f->size == 0)      { ++r.no_size; }
    if (f->sha256.empty()) { ++r.no_digest; }
  }
  std::printf("  [%s -> %s] %zu pinned, %zu files listed, missing=%zu "
              "no-size=%zu no-digest=%zu\n",
              e.hf_path.c_str(), resolved.c_str(), r.pinned, files.size(),
              r.missing, r.no_size, r.no_digest);
  return r;
}

// Every catalogue entry published from `hf_path` whose `name` contains
// `name_hint` (empty -> all of them).
vector<const ModelCatalogEntry*>
entries_(const string& hf_path, const string& name_hint)
{
  vector<const ModelCatalogEntry*> out;
  for (const ModelCatalogEntry* e : catalog_all_by_path(hf_path)) {
    if (name_hint.empty() || e->name.find(name_hint) != string::npos) {
      out.push_back(e);
    }
  }
  return out;
}

}

// Every pinned file of the five first-wave families resolves on
// ModelScope, at a published size, with a published digest.
TEST(model_source_live, pinned_files_resolve_on_modelscope) {
  if (!live_()) { return; }
  const ModelSource* ms = model_source("modelscope");
  ASSERT_TRUE(ms != nullptr);
  if (ms == nullptr) { return; }
  // Constructing a Session loads $VPIPE_PLUGINS, and a plugin's
  // catalogue entries only exist once it has. Without this the two
  // out-of-tree families below are skipped -- which is the honest
  // outcome when no plugin path was given, but a poor default when one
  // was. Set VPIPE_PLUGINS to the LTX-2.5 / SenseNova-U1.5 .so paths to
  // cover them here.
  Session sess;

  const char* kRepos[] = {
      "MiniMaxAI/MiniMax-H3",              // FL2VA + Ref2VA partitions
      "Comfy-Org/MiniMax-H3",              // the repack, both partitions
      "black-forest-labs/FLUX.2-klein-9B",
      "krea/Krea-2-Turbo",
      "krea/Krea-2-Raw",
      "krea/Krea-2-LoRA-softwatercolor",
      // The two MiniMax-H3 Turbo adapter repos. Both are mirrored under
      // the IDENTICAL owner/repo, so neither needs a row in the mirror
      // table -- which is exactly why they are listed here: a mirror
      // that works by DEFAULT has nothing declaring it, so the only
      // thing that can catch it going away is asking.
      "lightx2v/Minimax-h3-Turbo",
      "larryvrh/MiniMax-H3-Turbo-Lora",
      "Lightricks/LTX-2.5",                // plugin, when loaded
      "sensenova/SenseNova-U1.5-8B-MoT",   // plugin, when loaded
  };
  size_t checked = 0;
  for (const char* repo : kRepos) {
    const auto es = entries_(repo, "");
    if (es.empty()) {
      // The two plugin families are only in the catalogue when their
      // plugin is loaded; say so rather than silently pass.
      std::printf("  [%s] not in this build's catalogue -- skipped\n",
                  repo);
      continue;
    }
    for (const ModelCatalogEntry* e : es) {
      const PinReport r = check_entry_(*ms, *e);
      EXPECT_TRUE(r.listed);
      EXPECT_TRUE(r.missing == 0);
      EXPECT_TRUE(r.no_size == 0);
      EXPECT_TRUE(r.no_digest == 0);
      ++checked;
    }
  }
  std::printf("  checked %zu catalogue entries\n", checked);
  EXPECT_TRUE(checked > 0);
}

// The rename is real: MiniMaxAI/MiniMax-H3 is NOT there under its
// upstream name, and MiniMax/MiniMax-H3 is. Without this the mirror row
// could be wrong in either direction and the test above would still pass
// (it only ever asks for the resolved path).
TEST(model_source_live, minimax_rename_is_necessary) {
  if (!live_()) { return; }
  const ModelSource* ms = model_source("modelscope");
  ASSERT_TRUE(ms != nullptr);
  if (ms == nullptr) { return; }
  string body, err;
  long   status = 0;
  const bool upstream_ok = http_get_text(
      ms->tree_url("MiniMaxAI/MiniMax-H3", ms->default_revision()), "",
      true, 120, body, status, err);
  EXPECT_FALSE(upstream_ok);          // 404 under the HuggingFace name
  const bool renamed_ok = http_get_text(
      ms->tree_url("MiniMax/MiniMax-H3", ms->default_revision()), "",
      true, 120, body, status, err);
  EXPECT_TRUE(renamed_ok);
}

// ModelScope answers an UNKNOWN revision with HTTP 200 and an empty file
// list rather than an error, which is why default_revision() is
// "master" and why a wrong one cannot be diagnosed from the status code.
TEST(model_source_live, unknown_revision_lists_empty_not_error) {
  if (!live_()) { return; }
  const ModelSource* ms = model_source("modelscope");
  ASSERT_TRUE(ms != nullptr);
  if (ms == nullptr) { return; }
  const string repo = "sensenova/SenseNova-U1.5-8B-MoT";
  for (const char* rev : {"main", "no-such-revision"}) {
    string body, err;
    long   status = 0;
    const bool ok = http_get_text(ms->tree_url(repo, rev), "", true, 120,
                                  body, status, err);
    EXPECT_TRUE(ok);                  // the REQUEST succeeds...
    EXPECT_TRUE(status == 200);
    FlexData tree = FlexData::from_json(body);
    EXPECT_TRUE(ms->parse_tree(tree).empty());   // ...and lists nothing
  }
  string body, err;
  long   status = 0;
  ASSERT_TRUE(http_get_text(ms->tree_url(repo, "master"), "", true, 120,
                            body, status, err));
  FlexData tree = FlexData::from_json(body);
  EXPECT_FALSE(ms->parse_tree(tree).empty());
}

// One small real file, all the way through fetch_file: the URL shape
// works, the bytes arrive, and the digest ModelScope publishes matches
// what check_file_digest computes. This is the claim that would silently
// reject every download if it were wrong -- ModelScope's Sha256 being a
// content hash rather than a git blob id.
TEST(model_source_live, small_file_downloads_and_verifies) {
  if (!live_()) { return; }
  const ModelSource* ms = model_source("modelscope");
  ASSERT_TRUE(ms != nullptr);
  if (ms == nullptr) { return; }
  Session sess;

  // Mostly non-LFS files, so the check covers the small-file case where
  // HuggingFace would have published a git blob id instead -- plus one
  // LFS object, because those are the ones that redirect to
  // cdn-lfs-cn-1.modelscope.cn with a signed auth_key, a hop the plain
  // files never take and the one every model WEIGHT does.
  const std::pair<const char*, const char*> kCases[] = {
      {"black-forest-labs/FLUX.2-klein-9B", "model_index.json"},
      {"krea/Krea-2-Turbo", "scheduler/scheduler_config.json"},
      {"MiniMax/MiniMax-H3", "FL2VA/audio_vae/config.yaml"},
      {"sensenova/SenseNova-U1.5-8B-MoT", "config.json"},
      {"Lightricks/LTX-2.5",
       "model_patches/ltx-2.5-duration-head-bf16.safetensors"},
  };
  const fs::path tmp = fs::temp_directory_path() / "vpipe-ms-live";
  std::error_code ec;
  fs::remove_all(tmp, ec);

  for (const auto& [repo, file] : kCases) {
    string resolved, err;
    const vector<HfFile> files = list_(*ms, repo, resolved, err);
    ASSERT_TRUE(!files.empty());
    if (files.empty()) { continue; }
    const HfFile* f = nullptr;
    for (const HfFile& c : files) {
      if (c.path == file) { f = &c; break; }
    }
    ASSERT_TRUE(f != nullptr);
    if (f == nullptr) { continue; }
    EXPECT_TRUE(!f->sha256.empty());

    FetchRequest req;
    req.url          = ms->file_url(resolved, ms->default_revision(),
                                    f->path);
    req.want.size    = f->size;
    req.want.sha256  = f->sha256;
    FetchOpts o;
    o.stall_s     = 60;
    o.retries     = 2;
    o.verify      = true;
    o.xet_streams = 0;                 // no content store on this side

    const fs::path dest = tmp / repo / f->path;
    long      status = 0;
    string    derr;
    FileCheck fc = FileCheck::NotPublished;
    const bool ok = fetch_file(&sess, req, o, dest, status, derr, nullptr,
                               nullptr, &fc);
    std::printf("  [%s] %s -> ok=%d check=%d (%s)\n", resolved.c_str(),
                f->path.c_str(), (int)ok, (int)fc, derr.c_str());
    EXPECT_TRUE(ok);
    // Ok, not NotPublished: the digest was compared and it matched.
    EXPECT_TRUE(fc == FileCheck::Ok);
    EXPECT_TRUE(fs::exists(dest, ec));
  }
  fs::remove_all(tmp, ec);
}

// A LARGE LFS object, end to end: the CDN redirect, a transfer long
// enough for the stall window and the progress callback to matter, and
// the published sha256 over a quarter-gigabyte of content.
//
// Behind its OWN gate (VPIPE_MODELSCOPE_LIVE_BIG=1) because it moves
// 262 MB: the tests above answer "is the wiring right", this one answers
// "does a real weight file actually arrive", and only the second is
// worth minutes of link time on every run.
TEST(model_source_live, large_lfs_file_downloads_and_verifies) {
  const char* big = std::getenv("VPIPE_MODELSCOPE_LIVE_BIG");
  if (!live_() || big == nullptr || *big != '1') { return; }
  const ModelSource* ms = model_source("modelscope");
  ASSERT_TRUE(ms != nullptr);
  if (ms == nullptr) { return; }
  Session sess;

  const string repo = "Lightricks/LTX-2.5";
  const string file =
      "latent_upscale_models/"
      "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors";
  string resolved, err;
  const vector<HfFile> files = list_(*ms, repo, resolved, err);
  ASSERT_TRUE(!files.empty());
  if (files.empty()) { return; }
  const HfFile* f = nullptr;
  for (const HfFile& c : files) {
    if (c.path == file) { f = &c; break; }
  }
  ASSERT_TRUE(f != nullptr);
  if (f == nullptr) { return; }
  EXPECT_TRUE(f->size > (std::uint64_t{200} << 20));
  EXPECT_TRUE(!f->sha256.empty());

  FetchRequest req;
  req.url         = ms->file_url(resolved, ms->default_revision(), f->path);
  req.want.size   = f->size;
  req.want.sha256 = f->sha256;
  FetchOpts o;
  o.stall_s     = 120;
  o.retries     = 3;
  o.verify      = true;
  o.xet_streams = 0;

  const fs::path tmp = fs::temp_directory_path() / "vpipe-ms-live-big";
  std::error_code ec;
  fs::remove_all(tmp, ec);
  const fs::path dest = tmp / f->path;
  long      status = 0;
  string    derr;
  FileCheck fc = FileCheck::NotPublished;
  const bool ok = fetch_file(&sess, req, o, dest, status, derr, nullptr,
                             nullptr, &fc);
  std::printf("  [%s] %s (%s) -> ok=%d check=%d (%s)\n", resolved.c_str(),
              f->path.c_str(), human_bytes(f->size).c_str(), (int)ok,
              (int)fc, derr.c_str());
  EXPECT_TRUE(ok);
  EXPECT_TRUE(fc == FileCheck::Ok);
  EXPECT_TRUE(fs::exists(dest, ec));
  EXPECT_TRUE(fs::file_size(dest, ec) == f->size);
  fs::remove_all(tmp, ec);
}
