// ModelRemoveStage: config-validation + port-shape tests plus real-LMDB
// removal runs driven through the remove_once() test seam (a Session
// constructed with a temp db path yields a live env, as in the
// videos-db-cleanup-stage test).

#include "minitest.h"

#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/session.h"
#include "pipeline/stage-registry.h"
#include "stages/model-remove-stage.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Redirect std::cerr for the duration of a test that intentionally logs a
// warn (missing model / unsafe path), so the suite output stays clean.
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
  auto base = filesystem::temp_directory_path() / "vpipe_model_rm_XXXXXX";
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

FlexData
cfg_(string_view model, bool delete_files = false, bool missing_ok = true,
     string_view models_db = "")
{
  FlexData c = FlexData::make_object();
  auto o = c.as_object();
  o.insert_or_assign("model", FlexData::make_string(model));
  o.insert_or_assign("delete_files", FlexData::make_bool(delete_files));
  o.insert_or_assign("missing_ok", FlexData::make_bool(missing_ok));
  if (!models_db.empty()) {
    o.insert_or_assign("models_db", FlexData::make_string(models_db));
  }
  return c;
}

// Seed a models-DB record (optionally with a local_path). Scoped so no
// LmdbDb handle stays open across the stage's own db construction.
void
seed_(LmdbEnv& env, string_view db, string_view key, string_view local_path)
{
  FlexData rec = FlexData::make_object();
  if (!local_path.empty()) {
    rec.as_object().insert_or_assign(
        "local_path", FlexData::make_string(local_path));
  }
  LmdbDb  d(env, db);
  LmdbTxn txn(env, LmdbTxn::Mode::ReadWrite);
  const string bytes = rec.to_binary();
  d.put(txn, key, bytes);
  txn.commit();
}

bool
has_record_(LmdbEnv& env, string_view db, string_view key)
{
  LmdbDb  d(env, db);
  LmdbTxn txn(env, LmdbTxn::Mode::ReadOnly);
  const bool present = d.get(txn, key).has_value();
  txn.abort();
  return present;
}

string
db_cfg_(const string& path)
{
  return R"({"db":{"path":")" + path + R"("}})";
}

}  // namespace

TEST(model_remove_stage, type_is_registered)
{
  EXPECT_TRUE(std::string_view(ModelRemoveStage::kTypeName) ==
              "model-remove");
  EXPECT_TRUE(StageRegistry::get().find_id("model-remove") !=
              StageTypeId::unknown);
}

// model is required -> a missing one is recorded in config_error() and
// deferred to launch (ctor never throws).
TEST(model_remove_stage, missing_model_deferred)
{
  Session sess;
  CerrSilencer hush;
  ModelRemoveStage s(&sess, "rm", std::vector<InEdge>{},
                     FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());
}

TEST(model_remove_stage, config_defaults)
{
  Session sess;
  ModelRemoveStage s(&sess, "rm", std::vector<InEdge>{}, cfg_("some-model"));
  EXPECT_TRUE(s.model() == "some-model");
  EXPECT_TRUE(s.models_db() == "models");
  EXPECT_FALSE(s.delete_files());
  EXPECT_TRUE(s.missing_ok());
  EXPECT_TRUE(s.config_error().empty());
}

TEST(model_remove_stage, config_overrides)
{
  Session sess;
  ModelRemoveStage s(&sess, "rm", std::vector<InEdge>{},
                     cfg_("m", /*delete_files*/ true, /*missing_ok*/ false,
                          /*models_db*/ "registry"));
  EXPECT_TRUE(s.models_db() == "registry");
  EXPECT_TRUE(s.delete_files());
  EXPECT_FALSE(s.missing_ok());
}

// One trigger iport (any beat type) + one FlexData summary oport so the
// stage cascades into a preparation recipe / save-text.
TEST(model_remove_stage, trigger_and_summary_ports)
{
  Session sess;
  ModelRemoveStage s(&sess, "rm", std::vector<InEdge>{}, cfg_("m"));
  const StageSpec& sp = s.spec();
  ASSERT_TRUE(sp.iports.size() == 1u);
  ASSERT_TRUE(sp.oports.size() == 1u);
  EXPECT_TRUE(std::string_view(sp.iports[0].name) == "trigger");
  EXPECT_TRUE(sp.iports[0].type == nullptr);          // any beat type
  EXPECT_TRUE(std::string_view(sp.oports[0].name) == "summary");
  // By mangled name, not typeid pointer (stage in libvpipe vs test image).
  ASSERT_TRUE(sp.oports[0].type != nullptr);
  EXPECT_TRUE(std::string_view(sp.oports[0].type->name())
              == typeid(FlexDataPayload).name());
}

TEST(model_remove_stage, removes_registered_record)
{
  TempDir tdir;
  Session sess(db_cfg_(tdir.path));
  LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);

  seed_(*env, "models", "org/repo", "");
  EXPECT_TRUE(has_record_(*env, "models", "org/repo"));

  ModelRemoveStage s(&sess, "rm", std::vector<InEdge>{}, cfg_("org/repo"));
  const auto r = s.remove_once();
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.present);
  EXPECT_TRUE(r.removed);
  EXPECT_FALSE(r.files_deleted);
  EXPECT_FALSE(has_record_(*env, "models", "org/repo"));
}

// missing_ok (default) -> a not-registered model is a success (already
// removed), so a cascade keeps flowing; nothing is removed.
TEST(model_remove_stage, missing_ok_is_success)
{
  TempDir tdir;
  Session sess(db_cfg_(tdir.path));
  ASSERT_TRUE(sess.lmdb_env() != nullptr);

  ModelRemoveStage s(&sess, "rm", std::vector<InEdge>{}, cfg_("no/such"));
  const auto r = s.remove_once();
  EXPECT_TRUE(r.ok);
  EXPECT_FALSE(r.present);
  EXPECT_FALSE(r.removed);
}

// missing_ok=false -> a not-registered model is a failure (halts a cascade:
// ok=false so process() emits no summary).
TEST(model_remove_stage, missing_not_ok_is_failure)
{
  TempDir tdir;
  Session sess(db_cfg_(tdir.path));
  CerrSilencer hush;
  ASSERT_TRUE(sess.lmdb_env() != nullptr);

  ModelRemoveStage s(&sess, "rm", std::vector<InEdge>{},
                     cfg_("no/such", /*delete_files*/ false,
                          /*missing_ok*/ false));
  const auto r = s.remove_once();
  EXPECT_FALSE(r.ok);
  EXPECT_FALSE(r.present);
  EXPECT_FALSE(r.removed);
}

// delete_files=true also removes the recorded local_path directory.
TEST(model_remove_stage, delete_files_removes_local_path)
{
  TempDir tdir;
  Session sess(db_cfg_(tdir.path));
  LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);

  // A model dir with a file inside, under the temp root.
  const auto model_dir = filesystem::path(tdir.path) / "the-model";
  error_code ec;
  filesystem::create_directories(model_dir, ec);
  { std::ofstream(model_dir / "weights.bin") << "xx"; }
  ASSERT_TRUE(filesystem::exists(model_dir, ec));

  seed_(*env, "models", "org/m", model_dir.string());

  ModelRemoveStage s(&sess, "rm", std::vector<InEdge>{},
                     cfg_("org/m", /*delete_files*/ true));
  const auto r = s.remove_once();
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.removed);
  EXPECT_TRUE(r.files_deleted);
  EXPECT_TRUE(r.local_path == model_dir.string());
  EXPECT_FALSE(filesystem::exists(model_dir, ec));
  EXPECT_FALSE(has_record_(*env, "models", "org/m"));
}

// Default (delete_files=false) removes only the record; the on-disk model
// directory is left untouched.
TEST(model_remove_stage, keeps_files_when_delete_files_false)
{
  TempDir tdir;
  Session sess(db_cfg_(tdir.path));
  LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);

  const auto model_dir = filesystem::path(tdir.path) / "keep-me";
  error_code ec;
  filesystem::create_directories(model_dir, ec);
  { std::ofstream(model_dir / "weights.bin") << "xx"; }

  seed_(*env, "models", "org/keep", model_dir.string());

  ModelRemoveStage s(&sess, "rm", std::vector<InEdge>{}, cfg_("org/keep"));
  const auto r = s.remove_once();
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.removed);
  EXPECT_FALSE(r.files_deleted);
  EXPECT_TRUE(filesystem::exists(model_dir, ec));   // files survive
  EXPECT_FALSE(has_record_(*env, "models", "org/keep"));
}

// The removal is scoped to the configured sub-db; a same-keyed record in
// another sub-db is not touched.
TEST(model_remove_stage, only_removes_from_named_sub_db)
{
  TempDir tdir;
  Session sess(db_cfg_(tdir.path));
  LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);

  seed_(*env, "models", "shared/key", "");
  seed_(*env, "other",  "shared/key", "");

  ModelRemoveStage s(&sess, "rm", std::vector<InEdge>{},
                     cfg_("shared/key", /*delete_files*/ false,
                          /*missing_ok*/ true, /*models_db*/ "models"));
  const auto r = s.remove_once();
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.removed);
  EXPECT_FALSE(has_record_(*env, "models", "shared/key"));
  EXPECT_TRUE(has_record_(*env, "other", "shared/key"));   // untouched
}
