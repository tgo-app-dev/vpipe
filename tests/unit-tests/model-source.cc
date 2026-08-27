#include "minitest.h"
#include "stages/model-catalog.h"
#include "stages/model-source.h"
#include "common/flex-data.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

bool
has_(const vector<string>& v, const string& x)
{
  return std::find(v.begin(), v.end(), x) != v.end();
}

const HfFile*
find_(const vector<HfFile>& v, const string& path)
{
  for (const HfFile& f : v) {
    if (f.path == path) { return &f; }
  }
  return nullptr;
}

// A trimmed but SHAPE-EXACT capture of
// modelscope.cn/api/v1/models/<repo>/repo/files -- the wrapper object,
// the capitalised keys, a "tree" entry among the "blob"s, and one LFS
// and one non-LFS file, both carrying Sha256. Kept verbatim from a real
// answer so the parser is tested against the server's spelling rather
// than against what this file thinks it is.
constexpr const char* kModelScopePage = R"({
  "Code": 200,
  "Data": {
    "Files": [
      {"CommitMessage": "Upload folder using ModelScope SDK",
       "IsLFS": false, "Mode": "33188", "Name": "config.json",
       "Path": "config.json", "Revision": "baf901bb",
 "Sha256": "6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b",
       "Size": 2631, "Type": "blob"},
      {"IsLFS": false, "Mode": "16384", "Name": "vae",
       "Path": "vae", "Sha256": "", "Size": 0, "Type": "tree"},
      {"IsLFS": true, "Mode": "33188", "Name": "vocab.json",
       "Path": "vocab.json", "Revision": "1867eea2",
 "Sha256": "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d",
       "Size": 3383407, "Type": "blob"}
    ]
  },
  "Message": "", "RequestId": "", "Success": true
})";

}

TEST(model_source, sources_registered) {
  const auto names = model_source_names();
  EXPECT_TRUE(has_(names, "huggingface"));
  EXPECT_TRUE(has_(names, "modelscope"));
  EXPECT_TRUE(model_source("huggingface") != nullptr);
  EXPECT_TRUE(model_source("modelscope") != nullptr);
  EXPECT_TRUE(model_source("no-such-source") == nullptr);
}

// The two sources disagree about the default branch, and getting it
// wrong on ModelScope yields an EMPTY listing rather than an error --
// so this is pinned rather than left to a comment.
TEST(model_source, default_revisions_differ) {
  const ModelSource* hf = model_source("huggingface");
  const ModelSource* ms = model_source("modelscope");
  ASSERT_TRUE(hf != nullptr && ms != nullptr);
  if (hf == nullptr || ms == nullptr) { return; }
  EXPECT_TRUE(string(hf->default_revision()) == "main");
  EXPECT_TRUE(string(ms->default_revision()) == "master");
}

TEST(model_source, url_shapes) {
  const ModelSource* hf = model_source("huggingface");
  const ModelSource* ms = model_source("modelscope");
  ASSERT_TRUE(hf != nullptr && ms != nullptr);
  if (hf == nullptr || ms == nullptr) { return; }

  EXPECT_TRUE(hf->tree_url("krea/Krea-2-Turbo", "main")
      == "https://huggingface.co/api/models/krea/Krea-2-Turbo"
         "/tree/main?recursive=true");
  EXPECT_TRUE(hf->file_url("krea/Krea-2-Turbo", "main", "vae/config.json")
      == "https://huggingface.co/krea/Krea-2-Turbo/resolve/main"
         "/vae/config.json");

  EXPECT_TRUE(ms->tree_url("krea/Krea-2-Turbo", "master")
      == "https://modelscope.cn/api/v1/models/krea/Krea-2-Turbo"
         "/repo/files?Revision=master&Recursive=True");
  EXPECT_TRUE(ms->file_url("krea/Krea-2-Turbo", "master",
                           "vae/config.json")
      == "https://modelscope.cn/models/krea/Krea-2-Turbo/resolve/master"
         "/vae/config.json");
  EXPECT_TRUE(ms->web_url("krea/Krea-2-Turbo")
      == "https://modelscope.cn/models/krea/Krea-2-Turbo");
}

// Only HuggingFace has a content-addressed store, so a mirror must not
// be handed a Xet hash to reconstruct from.
TEST(model_source, only_huggingface_has_xet) {
  EXPECT_TRUE(model_source("huggingface")->supports_xet());
  EXPECT_FALSE(model_source("modelscope")->supports_xet());
}

// The wrapper, the capitalised keys, and "blob" vs "tree".
TEST(model_source, modelscope_tree_parse) {
  FlexData page = FlexData::from_json(kModelScopePage);
  const vector<HfFile> files = modelscope_tree_files(page);
  // The "tree" row is a directory and must not appear.
  EXPECT_TRUE(files.size() == 2);
  if (files.size() != 2) { return; }

  const HfFile* cfg = find_(files, "config.json");
  const HfFile* voc = find_(files, "vocab.json");
  ASSERT_TRUE(cfg != nullptr && voc != nullptr);
  if (cfg == nullptr || voc == nullptr) { return; }

  EXPECT_TRUE(cfg->size == 2631);
  EXPECT_TRUE(voc->size == 3383407);
  // ModelScope publishes a CONTENT sha256 for every file, LFS or not --
  // verified by download-and-hash across five repos. So both rows land
  // in `sha256` and NEITHER in `git_oid`: a git blob id in the sha256
  // slot would make check_file_digest reject a perfectly good download.
  EXPECT_TRUE(cfg->sha256
      == "6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b");
  EXPECT_TRUE(voc->sha256
      == "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d");
  EXPECT_TRUE(cfg->git_oid.empty());
  EXPECT_TRUE(voc->git_oid.empty());
  // No content store on this side.
  EXPECT_TRUE(cfg->xet_hash.empty());
  EXPECT_TRUE(voc->xet_hash.empty());
}

// A HuggingFace page must NOT parse as a ModelScope one (and vice
// versa): both are JSON, so a mixed-up source would otherwise fail as an
// empty repo rather than as the wrong parser.
TEST(model_source, tree_parsers_do_not_cross) {
  constexpr const char* kHfPage =
      R"([{"type":"file","path":"config.json","size":2631,
           "oid":"abc123"}])";
  FlexData hf_page = FlexData::from_json(kHfPage);
  EXPECT_TRUE(modelscope_tree_files(hf_page).empty());

  FlexData ms_page = FlexData::from_json(kModelScopePage);
  EXPECT_TRUE(hf_tree_files(ms_page).empty());
}

// The mirror table records DIFFERENCES only; everything else is
// mirrored under the identical owner/repo.
TEST(model_source, mirror_paths) {
  string out;
  // Renamed: MiniMax publishes as "MiniMaxAI" upstream, "MiniMax" here.
  EXPECT_TRUE(mirror_repo("modelscope", "MiniMaxAI/MiniMax-H3", out)
              == MirrorStatus::Renamed);
  EXPECT_TRUE(out == "MiniMax/MiniMax-H3");

  // Same: the common case, and what an unlisted repo gets.
  EXPECT_TRUE(mirror_repo("modelscope", "Lightricks/LTX-2.5", out)
              == MirrorStatus::Same);
  EXPECT_TRUE(out == "Lightricks/LTX-2.5");
  EXPECT_TRUE(mirror_repo("modelscope", "krea/Krea-2-Turbo", out)
              == MirrorStatus::Same);
  EXPECT_TRUE(out == "krea/Krea-2-Turbo");

  // Absent: known to have no counterpart, so the fetch can say so
  // instead of relaying a 404.
  EXPECT_TRUE(mirror_repo("modelscope", "mgwr/M87", out)
              == MirrorStatus::Absent);
  EXPECT_TRUE(out.empty());

  // The rename is per-SOURCE: on HuggingFace the upstream path stands.
  EXPECT_TRUE(mirror_repo("huggingface", "MiniMaxAI/MiniMax-H3", out)
              == MirrorStatus::Same);
  EXPECT_TRUE(out == "MiniMaxAI/MiniMax-H3");
  EXPECT_TRUE(mirror_repo("huggingface", "mgwr/M87", out)
              == MirrorStatus::Same);
  EXPECT_TRUE(out == "mgwr/M87");
}

// A plugin may add rows, but may not redirect a repo the tree already
// knows where to find.
TEST(model_source, mirror_registration_is_first_wins) {
  string out;
  const size_t took = register_mirror_repos({
      {.source = "modelscope", .hf_path = "example/only-in-plugin",
       .path = "mirror-owner/only-in-plugin"},
      {.source = "modelscope", .hf_path = "MiniMaxAI/MiniMax-H3",
       .path = "hijacked/elsewhere"},
  });
  EXPECT_TRUE(took == 1);
  EXPECT_TRUE(mirror_repo("modelscope", "example/only-in-plugin", out)
              == MirrorStatus::Renamed);
  EXPECT_TRUE(out == "mirror-owner/only-in-plugin");
  // The built-in row still stands.
  EXPECT_TRUE(mirror_repo("modelscope", "MiniMaxAI/MiniMax-H3", out)
              == MirrorStatus::Renamed);
  EXPECT_TRUE(out == "MiniMax/MiniMax-H3");
}

// ModelScope's web URL carries a "models/" segment that HuggingFace's
// does not, so one blind "first two segments" rule cannot serve both --
// a pasted ModelScope URL would otherwise normalise to "models/<owner>".
TEST(model_source, normalize_model_ref_both_hosts) {
  EXPECT_TRUE(normalize_model_ref("https://huggingface.co/krea/Krea-2-Turbo")
              == "krea/Krea-2-Turbo");
  EXPECT_TRUE(normalize_model_ref("huggingface.co/krea/Krea-2-Turbo/tree/main")
              == "krea/Krea-2-Turbo");
  EXPECT_TRUE(normalize_model_ref(
      "https://modelscope.cn/models/krea/Krea-2-Turbo")
              == "krea/Krea-2-Turbo");
  EXPECT_TRUE(normalize_model_ref(
      "https://www.modelscope.cn/models/MiniMax/MiniMax-H3/files")
              == "MiniMax/MiniMax-H3");
  EXPECT_TRUE(normalize_model_ref("  krea/Krea-2-Turbo/ ")
              == "krea/Krea-2-Turbo");
  EXPECT_TRUE(normalize_model_ref("").empty());
  EXPECT_TRUE(normalize_model_ref("no-slash").empty());
  // "models" as a HuggingFace OWNER must survive: the segment is only
  // dropped when the host said modelscope.cn.
  EXPECT_TRUE(normalize_model_ref("https://huggingface.co/models/thing")
              == "models/thing");
}

// Every catalogue entry the mirror table calls Renamed or Absent must
// name a repo that is actually in the catalogue -- otherwise the row is
// a typo that silently never fires.
TEST(model_source, mirror_rows_match_catalogue) {
  const char* kRenamed[] = {"MiniMaxAI/MiniMax-H3"};
  const char* kAbsent[]  = {"mgwr/M87", "RudySen/Krea2-realism-V2",
                            "conradlocke/krea2-identity-edit"};
  for (const char* p : kRenamed) {
    EXPECT_TRUE(!catalog_all_by_path(p).empty());
  }
  for (const char* p : kAbsent) {
    EXPECT_TRUE(!catalog_all_by_path(p).empty());
  }
}
