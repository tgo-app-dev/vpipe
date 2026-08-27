#include "stages/model-source.h"

#include "common/flex-data.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

void
strip_prefix_(string& s, const char* p)
{
  const size_t n = std::char_traits<char>::length(p);
  if (s.size() >= n && s.compare(0, n, p) == 0) {
    s.erase(0, n);
  }
}

// ---- HuggingFace -------------------------------------------------------

class HuggingFaceSource final : public ModelSource
{
public:
  const char* name() const override { return "huggingface"; }
  const char* label() const override { return "HuggingFace"; }
  const char* default_revision() const override { return "main"; }

  vector<string> token_env() const override
  {
    return {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"};
  }

  string web_url(const string& repo) const override
  {
    return "https://huggingface.co/" + repo;
  }

  string tree_url(const string& repo, const string& rev) const override
  {
    return "https://huggingface.co/api/models/" + repo + "/tree/" + rev
         + "?recursive=true";
  }

  vector<HfFile> parse_tree(const FlexData& body) const override
  {
    return hf_tree_files(body);
  }

  string file_url(const string& repo, const string& rev,
                  const string& path) const override
  {
    return "https://huggingface.co/" + repo + "/resolve/" + rev + "/"
         + path;
  }

  // The content store this catalogue's big shards mostly come from.
  bool supports_xet() const override { return true; }

  string empty_tree_hint(const string& repo, const string&) const override
  {
    return "'" + repo + "' lists no files -- private, gated without a "
           "usable token, or misspelled";
  }
};

// ---- ModelScope --------------------------------------------------------

// modelscope.cn (魔搭), the mirror that is reachable from mainland China.
//
// Three things differ from HuggingFace beyond the host, and each of them
// has bitten a naive port of this code:
//
//  - The listing is WRAPPED: `{"Code":200,"Data":{"Files":[...]},...}`
//    rather than a bare array, and the field names are capitalised
//    (Path / Size / Sha256 / Type / IsLFS). A directory is Type "tree", a
//    file is "blob" -- not HuggingFace's "directory" / "file".
//  - The default branch is `master`, and an unknown revision is answered
//    with HTTP 200 and an EMPTY Files array, never an error. So a fetch
//    pointed at "main" here reports a repo holding nothing.
//  - EVERY file carries a content SHA-256, LFS or not. HuggingFace
//    publishes the content hash only for LFS objects and leaves the git
//    blob id (a SHA-1 over "blob <size>\0" + content) for everything
//    else, which is why FileDigest carries both -- here only the first is
//    ever populated, and small files get a STRONGER check than they do
//    upstream. VERIFIED by download-and-hash across five repos
//    (SenseNova-U1.5, FLUX.2-klein-9B, Krea-2-Turbo, LTX-2.5,
//    MiniMax-H3), LFS and non-LFS alike: the published Sha256 is the
//    hash of the CONTENT in every case, never the git blob id.
//
// There is no Xet-equivalent content store, so every file streams.
class ModelScopeSource final : public ModelSource
{
public:
  const char* name() const override { return "modelscope"; }
  const char* label() const override { return "ModelScope"; }
  const char* default_revision() const override { return "master"; }

  vector<string> token_env() const override
  {
    return {"MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN"};
  }

  string web_url(const string& repo) const override
  {
    return "https://modelscope.cn/models/" + repo;
  }

  string tree_url(const string& repo, const string& rev) const override
  {
    return "https://modelscope.cn/api/v1/models/" + repo
         + "/repo/files?Revision=" + rev + "&Recursive=True";
  }

  vector<HfFile> parse_tree(const FlexData& body) const override
  {
    return modelscope_tree_files(body);
  }

  // Both the `/api/v1/models/<repo>/repo?FilePath=` form and this one
  // serve the same bytes; this one is chosen because it needs no query
  // escaping of the path and reads like the HuggingFace URL beside it.
  // MEASURED byte-identical against the API form on a small config.
  string file_url(const string& repo, const string& rev,
                  const string& path) const override
  {
    return "https://modelscope.cn/models/" + repo + "/resolve/" + rev + "/"
         + path;
  }

  string empty_tree_hint(const string& repo,
                         const string& rev) const override
  {
    return "'" + repo + "' lists no files at revision '" + rev
         + "' -- ModelScope answers an unknown revision with an empty "
           "list rather than an error, so check the revision (its default "
           "branch is 'master', not 'main') before concluding the repo is "
           "missing";
  }
};

const HuggingFaceSource kHuggingFace;
const ModelScopeSource  kModelScope;

const ModelSource* const kSources[] = {&kHuggingFace, &kModelScope};

// ---- the mirror table --------------------------------------------------

// ONLY the differences; see mirror_repo's contract. A repo mirrored under
// the identical owner/repo -- which is most of them -- says nothing here.
const MirrorEntry kMirrors[] = {
    // MiniMax publishes as "MiniMaxAI" on HuggingFace and "MiniMax" on
    // ModelScope. The repo is otherwise the same tree, FL2VA/ and
    // Ref2VA/ partitions included: all 57 files each entry pins are
    // present at identical sizes.
    //
    // NB the two file lists are not identical everywhere, which is why
    // a whole-repo entry can end up with a slightly different directory
    // depending on the source: ModelScope adds its own
    // `configuration.json` and carries a different `.gitattributes`.
    // MEASURED on SenseNova-U1.5 (the one entry here that pins nothing):
    // 19 files on ModelScope against 18 on HuggingFace, every weight
    // shard byte-for-byte the same size. Nothing this tree reads is
    // affected -- but an entry that PINS files is held to its list on
    // both sides, so a genuine divergence surfaces as a missing pin
    // rather than as a quietly different model.
    {.source = "modelscope",
     .hf_path = "MiniMaxAI/MiniMax-H3",
     .path = "MiniMax/MiniMax-H3"},
    // Three Krea-2 adapters are individual-author uploads with no
    // ModelScope counterpart. Marked absent so the fetch says so instead
    // of handing back a 404 the user has to interpret.
    {.source = "modelscope", .hf_path = "mgwr/M87", .path = ""},
    {.source = "modelscope",
     .hf_path = "RudySen/Krea2-realism-V2", .path = ""},
    {.source = "modelscope",
     .hf_path = "conradlocke/krea2-identity-edit", .path = ""},
};

// Plugin-contributed rows, published as immutable snapshots for the same
// lifetime reason model_catalog() does it: callers hold references.
std::mutex&
mirror_mutex_()
{
  static std::mutex m;
  return m;
}

vector<shared_ptr<const vector<MirrorEntry>>>&
mirror_snapshots_()
{
  static vector<shared_ptr<const vector<MirrorEntry>>> v;
  return v;
}

shared_ptr<const vector<MirrorEntry>>
current_mirrors_()
{
  std::lock_guard<std::mutex> lk(mirror_mutex_());
  if (mirror_snapshots_().empty()) {
    return nullptr;
  }
  return mirror_snapshots_().back();
}

}

vector<HfFile>
modelscope_tree_files(const FlexData& body)
{
  vector<HfFile> out;
  if (!body.is_object()) {
    return out;
  }
  // Own each level before viewing it: as_object()/as_array() hand back
  // VIEWS, and a view of a temporary dangles.
  auto root = body.as_object();
  if (!root.contains("Data")) {
    return out;
  }
  FlexData data = root.at("Data");
  if (!data.is_object()) {
    return out;
  }
  auto dobj = data.as_object();
  if (!dobj.contains("Files")) {
    return out;
  }
  FlexData files = dobj.at("Files");
  if (!files.is_array()) {
    return out;
  }
  auto arr = files.as_array();
  for (size_t i = 0; i < arr.size(); ++i) {
    FlexData entry = arr.at(i);
    if (!entry.is_object()) {
      continue;
    }
    auto obj = entry.as_object();
    // "blob" is a file, "tree" a directory. Anything else is a shape
    // this parser does not know and is skipped rather than guessed at.
    const string type = obj.contains("Type")
        ? string(obj.at("Type").as_string("")) : "";
    if (type != "blob") {
      continue;
    }
    if (!obj.contains("Path")) {
      continue;
    }
    string path(obj.at("Path").as_string(""));
    if (path.empty()) {
      continue;
    }
    HfFile f;
    f.path = std::move(path);
    f.size = obj.contains("Size") ? obj.at("Size").as_uint(0) : 0;
    // Content hash for every file, LFS or not -- so `git_oid` stays
    // empty and check_file_digest takes the sha256 branch throughout.
    f.sha256 = obj.contains("Sha256")
        ? string(obj.at("Sha256").as_string("")) : "";
    out.push_back(std::move(f));
  }
  return out;
}

const ModelSource*
model_source(const string& name)
{
  for (const ModelSource* s : kSources) {
    if (name == s->name()) {
      return s;
    }
  }
  return nullptr;
}

const ModelSource&
default_model_source()
{
  if (const char* e = std::getenv("VPIPE_MODEL_SOURCE")) {
    if (const ModelSource* s = model_source(e)) {
      return *s;
    }
  }
  return kHuggingFace;
}

vector<string>
model_source_names()
{
  vector<string> v;
  v.reserve(std::size(kSources));
  for (const ModelSource* s : kSources) {
    v.emplace_back(s->name());
  }
  return v;
}

MirrorStatus
mirror_repo(const string& source_name, const string& hf_path, string& out)
{
  auto scan = [&](const MirrorEntry& m, MirrorStatus& st) {
    if (m.source != source_name || m.hf_path != hf_path) {
      return false;
    }
    if (m.path.empty()) {
      out.clear();
      st = MirrorStatus::Absent;
    } else {
      out = m.path;
      st = MirrorStatus::Renamed;
    }
    return true;
  };
  MirrorStatus st = MirrorStatus::Same;
  for (const MirrorEntry& m : kMirrors) {
    if (scan(m, st)) {
      return st;
    }
  }
  if (auto snap = current_mirrors_()) {
    for (const MirrorEntry& m : *snap) {
      if (scan(m, st)) {
        return st;
      }
    }
  }
  out = hf_path;
  return MirrorStatus::Same;
}

size_t
register_mirror_repos(vector<MirrorEntry> entries)
{
  std::lock_guard<std::mutex> lk(mirror_mutex_());
  auto next = make_shared<vector<MirrorEntry>>();
  if (!mirror_snapshots_().empty()) {
    *next = *mirror_snapshots_().back();
  }
  size_t taken = 0;
  for (MirrorEntry& e : entries) {
    if (e.source.empty() || e.hf_path.empty()) {
      continue;
    }
    // First-wins, built-ins included: a plugin cannot redirect a repo the
    // tree already knows where to find.
    bool dup = false;
    for (const MirrorEntry& m : kMirrors) {
      if (m.source == e.source && m.hf_path == e.hf_path) { dup = true; }
    }
    for (const MirrorEntry& m : *next) {
      if (m.source == e.source && m.hf_path == e.hf_path) { dup = true; }
    }
    if (dup) {
      continue;
    }
    next->push_back(std::move(e));
    ++taken;
  }
  if (taken > 0) {
    mirror_snapshots_().push_back(next);
  }
  return taken;
}

string
normalize_model_ref(const string& input)
{
  string s = input;
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  if (s.empty()) {
    return {};
  }
  strip_prefix_(s, "https://");
  strip_prefix_(s, "http://");
  strip_prefix_(s, "www.");
  // A ModelScope web URL carries a "models/" segment ahead of the pair
  // that HuggingFace's does not, so the prefix has to come off WITH the
  // host -- stripping it unconditionally would eat the owner of a
  // HuggingFace repo that happens to be called "models".
  const bool modelscope = s.compare(0, 14, "modelscope.cn/") == 0;
  strip_prefix_(s, "modelscope.cn/");
  strip_prefix_(s, "huggingface.co/");
  if (modelscope) {
    strip_prefix_(s, "models/");
  }
  s = s.substr(0, s.find_first_of("?#"));

  string owner, repo;
  size_t i = 0;
  auto next_segment = [&](string& dst) {
    while (i < s.size() && s[i] == '/') { ++i; }
    const size_t start = i;
    while (i < s.size() && s[i] != '/') { ++i; }
    dst = s.substr(start, i - start);
  };
  next_segment(owner);
  next_segment(repo);
  if (owner.empty() || repo.empty()) {
    return {};
  }
  return owner + "/" + repo;
}

}
