// ModelRegisterStage + detect_model_dir(): config validation, the
// directory-shape detection that fills the record (runtime type + I/O
// modalities), and real-LMDB registration runs driven through the
// register_once() test seam (a Session constructed with a temp db path
// yields a live env, as in the model-remove-stage test).

#include "minitest.h"

#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/session.h"
#include "pipeline/stage-registry.h"
#include "stages/model-catalog.h"
#include "stages/model-detect.h"
#include "stages/model-register-stage.h"
#include "stages/model-registry.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <streambuf>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Silence the deliberate warns (missing dir / clobber refusal).
class CerrSilencer {
public:
  CerrSilencer() : _saved(std::cerr.rdbuf()) { std::cerr.rdbuf(&_null); }
  ~CerrSilencer() { std::cerr.rdbuf(_saved); }
private:
  struct NullBuf : std::streambuf {
    int overflow(int c) override { return c; }
  };
  std::streambuf* _saved;
  NullBuf         _null;
};

string
make_tempdir_()
{
  auto base = filesystem::temp_directory_path() / "vpipe_model_reg_XXXXXX";
  string tmpl = base.string();
  if (!mkdtemp(tmpl.data())) {
    throw runtime_error("mkdtemp failed");
  }
  return tmpl;
}

struct TempDir {
  string path;
  TempDir() : path(make_tempdir_()) {}
  ~TempDir()
  {
    error_code ec;
    filesystem::remove_all(path, ec);
  }
};

string
db_cfg_(const string& path)
{
  return R"({"db":{"path":")" + path + R"("}})";
}

FlexData
cfg_(string_view dir, string_view key = "", string_view model_type = "",
     bool overwrite = false)
{
  FlexData c = FlexData::make_object();
  auto o = c.as_object();
  o.insert_or_assign("model_dir", FlexData::make_string(dir));
  if (!key.empty()) {
    o.insert_or_assign("key", FlexData::make_string(key));
  }
  if (!model_type.empty()) {
    o.insert_or_assign("model_type", FlexData::make_string(model_type));
  }
  o.insert_or_assign("overwrite_existing", FlexData::make_bool(overwrite));
  return c;
}

// Write `text` to <dir>/<rel>, creating parent directories.
void
write_file_(const filesystem::path& dir, const string& rel,
            const string& text)
{
  const auto p = dir / rel;
  error_code ec;
  filesystem::create_directories(p.parent_path(), ec);
  ofstream(p) << text;
}

// Build an <owner>/<repo> model dir under `root` carrying `config_json`
// as its config.json, and return the repo dir. The owner/repo nesting is
// what model-fetch writes, and what the default registry key is derived
// from.
filesystem::path
make_model_dir_(const string& root, const string& owner, const string& repo,
                const string& config_json)
{
  const auto dir = filesystem::path(root) / owner / repo;
  error_code ec;
  filesystem::create_directories(dir, ec);
  if (!config_json.empty()) {
    write_file_(dir, "config.json", config_json);
  }
  return dir;
}

bool
has_modality_(const vector<string>& v, const string& m)
{
  return find(v.begin(), v.end(), m) != v.end();
}

// Read a registry record back as FlexData (Null when absent).
FlexData
read_record_(LmdbEnv& env, string_view key)
{
  LmdbDb  db(env, kModelRegistryDb);
  LmdbTxn txn(env, LmdbTxn::Mode::ReadOnly);
  auto    view = db.get(txn, key);
  if (!view) { return {}; }
  const string bytes(*view);
  txn.abort();
  try {
    return FlexData::from_binary(bytes);
  } catch (...) {
    return {};
  }
}

string
rec_str_(const FlexData& rec, const char* key)
{
  if (!rec.is_object()) { return {}; }
  auto o = rec.as_object();
  return o.contains(key) ? string(o.at(key).as_string("")) : string();
}

}  // namespace

TEST(model_register_stage, type_is_registered)
{
  EXPECT_TRUE(std::string_view(ModelRegisterStage::kTypeName) ==
              "model-register");
  EXPECT_TRUE(StageRegistry::get().find_id("model-register") !=
              StageTypeId::unknown);
}

TEST(model_register_stage, model_dir_required_deferred)
{
  Session sess;
  ModelRegisterStage s(&sess, "reg", vector<InEdge>{},
                       FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());   // deferred, ctor never throws
  EXPECT_TRUE(s.num_oports() == 1);
}

// A directory that does not exist yet is NOT a config error: in a recipe
// an upstream quantize/fuse may not have produced it at config time.
// It fails at run time instead, without emitting a summary.
TEST(model_register_stage, missing_dir_fails_at_run_not_config)
{
  TempDir tdir;
  CerrSilencer hush;
  Session sess(db_cfg_(tdir.path));
  ModelRegisterStage s(&sess, "reg", vector<InEdge>{},
                       cfg_(tdir.path + "/not-there"));
  EXPECT_TRUE(s.config_error().empty());
  const auto r = s.register_once();
  EXPECT_FALSE(r.ok);
}

// ---- detection ------------------------------------------------------

// A catalogued repo laid out as <owner>/<repo> is recognized from the
// path alone and takes the curated metadata (this is the common case:
// a model-fetch'd tree copied from another machine).
TEST(model_register_stage, detects_catalogued_repo_by_path)
{
  TempDir tdir;
  const auto dir = make_model_dir_(tdir.path, "mlx-community",
                                   "gemma-4-e4b-it-4bit", "{}");
  const DetectedModel d = detect_model_dir(dir.string());
  EXPECT_TRUE(d.detected_by == "catalog");
  EXPECT_TRUE(d.model_type == "gemma4");
  EXPECT_TRUE(d.family == "Gemma");
  EXPECT_TRUE(d.param_class == "E4B");
  EXPECT_TRUE(has_modality_(d.inputs, "image"));
  EXPECT_TRUE(has_modality_(d.outputs, "text"));
}

// An UNcatalogued Qwen3.5 checkpoint is recognized from its config.json,
// and its I/O comes out multimodal because the config carries a vision
// tower.
TEST(model_register_stage, detects_qwen35_from_config)
{
  TempDir tdir;
  const auto dir = make_model_dir_(
      tdir.path, "local", "Qwen3.5-4B-tuned",
      R"({"model_type":"qwen3_5","vision_config":{"depth":24},)"
      R"("quantization":{"bits":4,"group_size":64}})");
  const DetectedModel d = detect_model_dir(dir.string());
  EXPECT_TRUE(d.detected_by == "config");
  EXPECT_TRUE(d.model_type == "qwen3.5");
  EXPECT_TRUE(d.param_class == "4B");
  EXPECT_TRUE(d.variant == "4-bit");
  EXPECT_TRUE(has_modality_(d.inputs, "text"));
  EXPECT_TRUE(has_modality_(d.inputs, "image"));
  EXPECT_TRUE(has_modality_(d.outputs, "text"));
}

// The SAME family without a vision_config is text-only: the per-type
// defaults describe the family at its fullest, so they are trimmed to
// what this checkpoint actually carries. Claiming image input here would
// offer a text-only dump to a visual-qa field that cannot use it.
TEST(model_register_stage, trims_io_to_what_the_checkpoint_carries)
{
  TempDir tdir;
  const auto dir = make_model_dir_(tdir.path, "local", "Qwen3.5-4B-textonly",
                                   R"({"model_type":"qwen3_5"})");
  const DetectedModel d = detect_model_dir(dir.string());
  EXPECT_TRUE(d.model_type == "qwen3.5");
  EXPECT_TRUE(has_modality_(d.inputs, "text"));
  EXPECT_FALSE(has_modality_(d.inputs, "image"));
  EXPECT_FALSE(has_modality_(d.inputs, "video"));
}

// The trim must not fire for a family that signals its modalities some
// other way. Qwen3-ASR wraps everything under thinker_config and carries
// no audio_config, so reading that absence as "no audio" would turn a
// transcription model into a text-to-text one -- which is exactly what a
// quantized ASR checkpoint registered as, before this was scoped.
TEST(model_register_stage, keeps_audio_for_asr_without_audio_config)
{
  TempDir tdir;
  const auto dir = make_model_dir_(
      tdir.path, "local", "asr-1.7b-w4",
      R"({"model_type":"qwen3_asr",)"
      R"("thinker_config":{"num_hidden_layers":28}})");
  const DetectedModel d = detect_model_dir(dir.string());
  EXPECT_TRUE(d.model_type == "qwen3-asr");
  EXPECT_TRUE(has_modality_(d.inputs, "audio"));
  EXPECT_FALSE(has_modality_(d.inputs, "text"));   // audio in, text out
  EXPECT_TRUE(has_modality_(d.outputs, "text"));
}

// Gemma-4 reports one config model_type for every size; the E-sizes run
// the per-layer-embedding path and the rest the unified one, which is the
// split the runtime cares about.
TEST(model_register_stage, gemma_size_picks_the_runtime_variant)
{
  TempDir tdir;
  const auto e = make_model_dir_(tdir.path, "local", "gemma-4-e4b-it-local",
                                 R"({"model_type":"gemma4"})");
  const auto u = make_model_dir_(tdir.path, "local", "gemma-4-31B-it-local",
                                 R"({"model_type":"gemma4"})");
  EXPECT_TRUE(detect_model_dir(e.string()).model_type == "gemma4");
  EXPECT_TRUE(detect_model_dir(u.string()).model_type == "gemma4_unified");
}

// A diffusers pipeline is recognized from transformer/config.json, and an
// edit checkpoint is told from a t2i one by name (their transformer
// configs are identical).
TEST(model_register_stage, detects_diffusers_pipeline)
{
  TempDir tdir;
  const auto dir = filesystem::path(tdir.path) / "local" / "FLUX.2-klein-4B";
  error_code ec;
  filesystem::create_directories(dir, ec);
  write_file_(dir, "transformer/config.json",
              R"({"_class_name":"Flux2Transformer2DModel"})");
  const DetectedModel d = detect_model_dir(dir.string());
  EXPECT_TRUE(d.detected_by == "diffusers");
  EXPECT_TRUE(d.model_type == "flux2");
  EXPECT_TRUE(has_modality_(d.inputs, "image"));
  EXPECT_TRUE(has_modality_(d.outputs, "image"));

  const auto edit = filesystem::path(tdir.path) / "local" / "Boogu-Edit-Turbo";
  filesystem::create_directories(edit, ec);
  write_file_(edit, "transformer/config.json",
              R"({"_class_name":"BooguImageTransformer2DModel"})");
  EXPECT_TRUE(detect_model_dir(edit.string()).model_type ==
              "boogu-image-edit");
}

// A BARE DiT (diffusers weights + config, no pipeline around them --
// model-quantize's DiT-only output) gets the "<family>-dit" tag that
// generate-image's `dit_dir` picker filters on, NOT the pipeline tag: it
// cannot be loaded as a pipeline, so offering it to hf_dir would fail.
TEST(model_register_stage, detects_bare_dit_component)
{
  TempDir tdir;
  const auto dir =
      filesystem::path(tdir.path) / "local" / "FLUX.2-klein-9B-dit-w4g64";
  error_code ec;
  filesystem::create_directories(dir, ec);
  write_file_(dir, "config.json",
              R"({"_class_name":"Flux2Transformer2DModel"})");
  const DetectedModel d = detect_model_dir(dir.string());
  EXPECT_TRUE(d.detected_by == "diffusers-component");
  EXPECT_TRUE(d.model_type == "flux2-dit");
  EXPECT_TRUE(d.category == "component");
  EXPECT_TRUE(d.inputs.empty() && d.outputs.empty());
}

// A GGUF repo takes its family + size from the REPO name (the .gguf file
// names carry the quant, and an mmproj companion is not the checkpoint).
TEST(model_register_stage, detects_gguf_repo_from_name)
{
  TempDir tdir;
  const auto dir = filesystem::path(tdir.path) / "someone" / "Qwen3.5-4B-GGUF";
  error_code ec;
  filesystem::create_directories(dir, ec);
  write_file_(dir, "mmproj-BF16.gguf", "x");
  write_file_(dir, "Qwen3.5-4B-Q4_K_M.gguf", "x");
  const DetectedModel d = detect_model_dir(dir.string());
  EXPECT_TRUE(d.detected_by == "gguf");
  EXPECT_TRUE(d.model_type == "qwen3.5");
  EXPECT_TRUE(d.param_class == "4B");
}

// An unrecognizable directory registers, but claims NO type -- better a
// model the pickers don't offer than one offered to a stage that cannot
// load it.
TEST(model_register_stage, unknown_dir_claims_no_type)
{
  TempDir tdir;
  const auto dir = make_model_dir_(tdir.path, "local", "mystery-thing",
                                   R"({"model_type":"something_else"})");
  const DetectedModel d = detect_model_dir(dir.string());
  EXPECT_TRUE(d.model_type.empty());
  EXPECT_TRUE(d.inputs.empty());
  EXPECT_TRUE(d.outputs.empty());
}

// ---- registration ---------------------------------------------------

// The default key is <owner>/<repo>, and the record carries the detected
// type + modalities, so resolve_model_dir() finds the model by key.
TEST(model_register_stage, registers_with_derived_key_and_metadata)
{
  TempDir tdir;
  Session sess(db_cfg_(tdir.path));
  LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);

  const auto dir = make_model_dir_(tdir.path, "local", "Qwen3.5-4B-tuned",
                                   R"({"model_type":"qwen3_5"})");
  write_file_(dir, "model.safetensors", "weights");

  ModelRegisterStage s(&sess, "reg", vector<InEdge>{}, cfg_(dir.string()));
  const auto r = s.register_once();
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(r.key == "local/Qwen3.5-4B-tuned");
  EXPECT_FALSE(r.replaced);

  const FlexData rec = read_record_(*env, r.key);
  ASSERT_TRUE(rec.is_object());
  EXPECT_TRUE(rec_str_(rec, "model_type") == "qwen3.5");
  EXPECT_TRUE(rec_str_(rec, "local_path") == r.local_path);
  EXPECT_TRUE(rec_str_(rec, "hf_path") == "local/Qwen3.5-4B-tuned");
  auto ro = rec.as_object();
  EXPECT_TRUE(ro.contains("inputs") && ro.contains("outputs"));
  EXPECT_TRUE(ro.at("registered_from_disk").as_bool(false));
  EXPECT_TRUE(ro.at("file_count").as_uint(0) == 2u);   // config + weights

  // The whole point: the key now resolves to the directory.
  EXPECT_TRUE(resolve_model_dir(&sess, r.key) == r.local_path);
  EXPECT_TRUE(model_dir_available(&sess, r.key));
}

// An explicit model_type overrides detection AND re-derives the I/O, so
// the record cannot end up with one family's type and another's
// modalities.
TEST(model_register_stage, explicit_model_type_overrides_detection)
{
  TempDir tdir;
  Session sess(db_cfg_(tdir.path));
  const auto dir = make_model_dir_(tdir.path, "local", "mystery-thing",
                                   R"({"model_type":"something_else"})");

  ModelRegisterStage s(&sess, "reg", vector<InEdge>{},
                       cfg_(dir.string(), "", "flux2"));
  const auto r = s.register_once();
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(r.detected.model_type == "flux2");
  EXPECT_TRUE(has_modality_(r.detected.outputs, "image"));
}

// Re-registering over an existing key is refused unless asked, and the
// refusal leaves the original record intact.
TEST(model_register_stage, refuses_to_clobber_unless_overwrite)
{
  TempDir tdir;
  CerrSilencer hush;
  Session sess(db_cfg_(tdir.path));
  LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);

  const auto first = make_model_dir_(tdir.path, "local", "same-key",
                                     R"({"model_type":"qwen3_5"})");
  ModelRegisterStage s1(&sess, "reg", vector<InEdge>{}, cfg_(first.string()));
  ASSERT_TRUE(s1.register_once().ok);
  const string first_path = rec_str_(read_record_(*env, "local/same-key"),
                                     "local_path");

  // A DIFFERENT directory claiming the same derived key.
  const auto second = make_model_dir_(tdir.path + "/elsewhere", "local",
                                      "same-key", R"({"model_type":"gemma4"})");
  ModelRegisterStage s2(&sess, "reg2", vector<InEdge>{},
                        cfg_(second.string()));
  const auto r2 = s2.register_once();
  EXPECT_FALSE(r2.ok);
  EXPECT_TRUE(rec_str_(read_record_(*env, "local/same-key"), "local_path")
              == first_path);   // untouched

  // With the opt-in it replaces.
  ModelRegisterStage s3(&sess, "reg3", vector<InEdge>{},
                        cfg_(second.string(), "", "", /*overwrite*/ true));
  const auto r3 = s3.register_once();
  ASSERT_TRUE(r3.ok);
  EXPECT_TRUE(r3.replaced);
  EXPECT_TRUE(rec_str_(read_record_(*env, "local/same-key"), "local_path")
              == r3.local_path);
}

// Re-registering the SAME directory is idempotent: the desired end-state
// already holds, so it succeeds without the overwrite opt-in. Without
// this a relaunched recipe would halt on its second run -- the failure
// the relaunch sweep exists to catch, with the state in LMDB rather than
// in a stage member.
TEST(model_register_stage, re_registering_same_dir_is_idempotent)
{
  TempDir tdir;
  Session sess(db_cfg_(tdir.path));
  LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);

  const auto dir = make_model_dir_(tdir.path, "local", "same-dir",
                                   R"({"model_type":"qwen3_5"})");
  ModelRegisterStage s(&sess, "reg", vector<InEdge>{}, cfg_(dir.string()));

  const auto r1 = s.register_once();
  ASSERT_TRUE(r1.ok);
  EXPECT_FALSE(r1.replaced);

  // Same stage object, same directory, no overwrite_existing.
  const auto r2 = s.register_once();
  EXPECT_TRUE(r2.ok);            // succeeds -> a cascade keeps flowing
  EXPECT_TRUE(r2.replaced);      // and the record was refreshed
  EXPECT_TRUE(rec_str_(read_record_(*env, r2.key), "local_path")
              == r1.local_path);
}

// An explicit key wins over the derived one, and a relative model_dir is
// stored ABSOLUTE (the record outlives this process's cwd).
TEST(model_register_stage, explicit_key_and_absolute_path)
{
  TempDir tdir;
  Session sess(db_cfg_(tdir.path));
  const auto dir = make_model_dir_(tdir.path, "local", "some-model",
                                   R"({"model_type":"qwen3_5"})");

  ModelRegisterStage s(&sess, "reg", vector<InEdge>{},
                       cfg_(dir.string(), "my-favourite-model"));
  const auto r = s.register_once();
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(r.key == "my-favourite-model");
  EXPECT_TRUE(filesystem::path(r.local_path).is_absolute());
  EXPECT_TRUE(resolve_model_dir(&sess, "my-favourite-model") == r.local_path);
}

// ---- CoreML supplements: the record must name the BUNDLE -------------
//
// CoreML loads a *.mlpackage / *.mlmodelc bundle, never the directory
// that contains one. model-fetch has always resolved that when it
// unpacks an archive; model-register stored whatever directory it was
// handed, so a hand-registered supplement produced a record that every
// CoreML stage then failed to load ("modelWithContentsOfURL failed").

// coreml_artifact() is the one place the container -> bundle layout is
// spelled out, so pin its four answers directly.
TEST(model_register_stage, coreml_artifact_resolves_the_bundle)
{
  TempDir tdir;
  const auto root = filesystem::path(tdir.path);

  // A container holding one package resolves to the package.
  const auto box = root / "vpipe-supplement" / "tower_768x480";
  write_file_(box, "tower_768x480_w8.mlpackage/Manifest.json", "{}");
  const auto pkg = (box / "tower_768x480_w8.mlpackage").string();
  EXPECT_TRUE(coreml_artifact(box.string()) == pkg);

  // A bundle passes through unchanged, so a caller can apply it to a
  // path of either shape.
  EXPECT_TRUE(coreml_artifact(pkg) == pkg);

  // A compiled-only container resolves too -- this is the shape that
  // used to fall through model-fetch's *.mlpackage-only search and
  // register the directory with a warning.
  const auto cbox = root / "vpipe-supplement" / "compiled_only";
  write_file_(cbox, "tower.mlmodelc/coremldata.bin", "x");
  EXPECT_TRUE(coreml_artifact(cbox.string())
              == (cbox / "tower.mlmodelc").string());

  // A plain checkpoint directory has nothing to resolve, and empty is
  // how callers know to keep the directory they already had.
  const auto ckpt = make_model_dir_(tdir.path, "local", "a-checkpoint",
                                    R"({"model_type":"qwen3_5"})");
  EXPECT_TRUE(coreml_artifact(ckpt.string()).empty());
  EXPECT_TRUE(coreml_artifact("").empty());
  EXPECT_TRUE(coreml_artifact((root / "nope").string()).empty());
}

// Registering the CONTAINER directory of a supplement -- which is what a
// user has on disk and what the recipe names -- must register the bundle
// inside it, while the KEY still comes from the container. Deriving the
// key from the bundle would key the model under the archive's stem
// ("tower_768x480/tower_768x480_w8.mlpackage") instead of the name it
// was published under.
TEST(model_register_stage, coreml_container_registers_the_bundle)
{
  TempDir tdir;
  Session sess(db_cfg_(tdir.path));
  LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);

  // The vpipe-supplement layout: <owner>/vpipe-supplement/<name>/<pkg>.
  const auto box = make_model_dir_(tdir.path, "vpipe-supplement",
                                   "qwen3_5_vision_vid_768x480", "");
  write_file_(box, "qwen3_5_vision_vid_768x480_w8.mlpackage/Manifest.json",
              "{}");
  const auto pkg =
      (box / "qwen3_5_vision_vid_768x480_w8.mlpackage").string();

  ModelRegisterStage s(&sess, "reg", vector<InEdge>{}, cfg_(box.string()));
  const auto r = s.register_once();
  ASSERT_TRUE(r.ok);

  // The record names the bundle...
  EXPECT_TRUE(r.local_path == pkg);
  EXPECT_TRUE(rec_str_(read_record_(*env, r.key), "local_path") == pkg);
  // ...and resolve_model_dir hands a CoreML stage something loadable,
  // which is the whole point.
  EXPECT_TRUE(resolve_model_dir(&sess, r.key) == pkg);

  // The key and hf_path still describe the container.
  EXPECT_TRUE(r.key == "vpipe-supplement/qwen3_5_vision_vid_768x480");
  EXPECT_TRUE(rec_str_(read_record_(*env, r.key), "hf_path")
              == "vpipe-supplement/qwen3_5_vision_vid_768x480");
  // Detection ran on the bundle, so the shape is recognised.
  EXPECT_TRUE(r.detected.category == "supplement");
}

TEST(model_registry, a_pinned_subtree_resolves_to_the_subtree)
{
  // WHERE A RECORD'S FILES ACTUALLY ARE, when the repo holds more than
  // one model.
  //
  // `local_path` is where the repo was downloaded. A catalogue entry may
  // pin a SUBTREE of it: VDN-H3 publishes two stages from one repo as
  // stage-b-step-2000/... and stage-dmd-step-250/..., so both keys
  // resolve to the same root and the root itself holds no config at all.
  // A consumer handed the root reports the repo as malformed -- which is
  // the loud half. The quiet half is a stage that scans the root for
  // .safetensors, finds none and declares NOTHING for a 4.28 GB
  // checkpoint.
  auto rm = [](std::vector<std::string> files) {
    ResolvedModel r;
    r.dir = "/models/OpenVDN/vdn-minimax-h3";
    r.files = std::move(files);
    r.from_registry = true;
    return r;
  };
  namespace fs = std::filesystem;
  const std::string root = "/models/OpenVDN/vdn-minimax-h3";

  // The reported case, with the record model-fetch actually wrote.
  const std::string dmd = resolved_subtree_dir(rm({
      "stage-dmd-step-250/linear_branch/config.json",
      "stage-dmd-step-250/linear_branch/model.safetensors",
      "stage-dmd-step-250/adapters/default/adapter_config.json",
      "stage-dmd-step-250/adapters/default/adapter_model.safetensors",
      "stage-dmd-step-250/adapters/turbo/adapter_config.json",
      "stage-dmd-step-250/adapters/turbo/adapter_model.safetensors",
      "stage-dmd-step-250/model_spec.json",
      "stage-dmd-step-250/metadata.json"}));
  EXPECT_TRUE(dmd == (fs::path(root) / "stage-dmd-step-250").string());

  // The OTHER stage of the same repo must land somewhere else, or the
  // two keys are still indistinguishable and this bought nothing.
  const std::string b = resolved_subtree_dir(rm({
      "stage-b-step-2000/linear_branch/config.json",
      "stage-b-step-2000/model_spec.json"}));
  EXPECT_TRUE(b == (fs::path(root) / "stage-b-step-2000").string());
  EXPECT_TRUE(b != dmd);

  // FILES AT THE ROOT ANSWER THE ROOT -- the H3 DiT partitions, which
  // share a directory and are told apart by which file they pin, not by
  // where it sits. Anything else here would move every existing
  // consumer.
  EXPECT_TRUE(resolved_subtree_dir(rm({"a.safetensors",
                                       "b.safetensors"})) == root);
  // Mixed depths share no prefix, so: the root.
  EXPECT_TRUE(resolved_subtree_dir(rm({"transformer/a.safetensors",
                                       "config.json"})) == root);
  // A whole-repo fetch and a plain filesystem path pin nothing.
  EXPECT_TRUE(resolved_subtree_dir(rm({})) == root);
  // One file in a subtree is still that subtree.
  EXPECT_TRUE(resolved_subtree_dir(rm({"vae/diffusion_pytorch_model.bin"}))
              == (fs::path(root) / "vae").string());
  // And a deeper common prefix is followed all the way down.
  EXPECT_TRUE(resolved_subtree_dir(rm({"a/b/c/x.json", "a/b/c/y.json"}))
              == (fs::path(root) / "a" / "b" / "c").string());
}

TEST(model_registry, the_vdn_catalogue_entries_pin_distinct_subtrees)
{
  // The catalogue is where the shape above comes from, so it is what has
  // to keep it. Both VDN entries share one hf_path on purpose; if a
  // later edit gave them files at the repo root -- or the same subtree --
  // the two keys would resolve to one directory again and the attach
  // would pick whichever stage the tree happened to hold.
  const std::vector<ModelCatalogEntry>& all = model_catalog();
  const ModelCatalogEntry* b = nullptr;
  const ModelCatalogEntry* d = nullptr;
  for (const ModelCatalogEntry& e : all) {
    if (e.name == "OpenVDN/vdn-minimax-h3-stage-b") { b = &e; }
    if (e.name == "OpenVDN/vdn-minimax-h3-stage-dmd") { d = &e; }
  }
  ASSERT_TRUE(b != nullptr && d != nullptr);
  if (b == nullptr || d == nullptr) { return; }
  EXPECT_TRUE(b->hf_path == d->hf_path);      // one repo, two records
  auto sub = [](const ModelCatalogEntry& e) {
    ResolvedModel r;
    r.dir = "/root";
    r.files = e.files;
    r.from_registry = true;
    return resolved_subtree_dir(r);
  };
  const std::string sb = sub(*b), sd = sub(*d);
  std::printf("[registry] stage-b -> '%s', stage-dmd -> '%s'\n", sb.c_str(),
              sd.c_str());
  EXPECT_TRUE(sb != "/root");
  EXPECT_TRUE(sd != "/root");
  EXPECT_TRUE(sb != sd);
  // And each must be the directory that actually holds a config, which
  // is what vdn::load_config looks for.
  EXPECT_TRUE(sb.find("stage-b") != std::string::npos);
  EXPECT_TRUE(sd.find("stage-dmd") != std::string::npos);

  // EVERY OTHER ENTRY THAT PINS A SUBTREE, named rather than asserted.
  // The same shape is latent wherever one exists: a consumer that takes
  // local_path and looks for a config beside it finds the repo root. Not
  // a failure -- most consumers want the root, which is exactly why the
  // descent is opt-in -- but this is the list to check when adding one.
  int n = 0;
  for (const ModelCatalogEntry& e : all) {
    ResolvedModel r;
    r.dir = "/root";
    r.files = e.files;
    r.from_registry = true;
    const std::string sub = resolved_subtree_dir(r);
    if (sub != "/root") {
      std::printf("[registry]   pins a subtree: %-46s -> %s\n",
                  e.name.c_str(), sub.c_str() + 6);
      ++n;
    }
  }
  std::printf("[registry] %d catalogue entries pin a subtree\n", n);
}
