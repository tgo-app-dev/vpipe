#include "minitest.h"
#include "common/image-metadata.h"
#include "pipeline/stage-registry.h"
#include "vpipe/vpipe.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/load-image-stage.h"
#include "stages/save-image-stage.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <streambuf>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

class CerrSilencer {
public:
  CerrSilencer() : _saved(cerr.rdbuf()), _null() { cerr.rdbuf(&_null); }
  ~CerrSilencer() { cerr.rdbuf(_saved); }
private:
  struct NullBuf : public streambuf {
    int overflow(int c) override { return c; }
  };
  streambuf* _saved;
  NullBuf    _null;
};

// Test-only source: 0 iports, 1 oport. Emits `n` solid-color planar U8
// RGB TensorBeats [3, H, W] (channel order R,G,B) then signals done.
class SolidSource : public TypedStage<SolidSource> {
public:
  static constexpr const char* kTypeName = "ut-solid-source";
  using TypedStage::TypedStage;

  int     w = 4, h = 3, n = 1;
  uint8_t r = 0xFF, g = 0x80, b = 0x40;
  bool    noisy = false;   // high-frequency fill (JPEG quality has teeth)
  // When set, stamped onto each beat's sideband exactly as vae-decode does,
  // so save-image sees a GENERATED image.
  string  model_name;

  SolidSource(const SessionContextIntf* s, string id,
              vector<InEdge> in, FlexData cfg)
    : TypedStage<SolidSource>(s, std::move(id), std::move(in),
                              std::move(cfg))
  {
    allocate_oports(1);
  }

  // The very thing Stage::reset_run_state exists for: without this the
  // source is exhausted after run 1 and a relaunch emits nothing.
  void reset_run_state() override { _emitted = 0; }

  Job
  process(RuntimeContext& ctx) override
  {
    if (_emitted >= n) { ctx.signal_done(); co_return; }
    auto out = make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::U8;
    out->shape = {3, h, w};
    out->resize_contiguous(static_cast<size_t>(3) * h * w);
    uint8_t* d = out->as_u8();
    const size_t plane = static_cast<size_t>(h) * w;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const size_t i = static_cast<size_t>(y) * w + x;
        if (noisy) {
          // Pseudo-random high-frequency content per channel so the JPEG
          // quantizer has AC energy to discard at low quality.
          d[i]             = (uint8_t)((x * 37 + y * 101) * 53 + 17);
          d[plane + i]     = (uint8_t)((x * 91 + y * 13) * 29 + 61);
          d[2 * plane + i] = (uint8_t)((x * 7  + y * 200) * 71 + 5);
        } else {
          d[i]             = r;
          d[plane + i]     = g;
          d[2 * plane + i] = b;
        }
      }
    }
    if (!model_name.empty()) {
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign("model_name",
                                      FlexData::make_string(model_name));
      out->sideband = std::move(sb);
    }
    ++_emitted;
    co_await ctx.write(0, std::move(out));
    if (_emitted >= n) { ctx.signal_done(); }
  }

private:
  int _emitted = 0;
};

// Test-only sink: 1 iport, 0 oports. Captures received payloads.
class CaptureSink : public TypedStage<CaptureSink> {
public:
  static constexpr const char* kTypeName = "ut-store-capture";
  using TypedStage::TypedStage;
  vector<unique_ptr<BeatPayloadIntf>> got;
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    got.push_back(std::move(p));
  }
};

// Read the first `k` bytes of a file (empty on failure).
string
head_bytes_(const string& path, size_t k)
{
  ifstream in(path, ios::binary);
  string s(k, '\0');
  in.read(&s[0], static_cast<streamsize>(k));
  s.resize(static_cast<size_t>(in.gcount()));
  return s;
}

size_t
file_size_(const string& path)
{
  ifstream in(path, ios::binary | ios::ate);
  return in ? static_cast<size_t>(in.tellg()) : 0;
}

string
tmp_path_(const string& suffix)
{
  return string("/tmp/vpipe-save-image-") + to_string(getpid()) + suffix;
}

// Drive one SolidSource -> SaveImageStage pipeline to completion.
void
run_store_(Session& sess, SolidSource*& src_out, FlexData store_cfg,
           int n, uint8_t r, uint8_t g, uint8_t b,
           bool noisy = false, int w = 4, int h = 3,
           const string& model_name = {})
{
  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<SolidSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->n = n; src_u->r = r; src_u->g = g; src_u->b = b;
  src_u->noisy = noisy; src_u->w = w; src_u->h = h;
  src_u->model_name = model_name;
  auto* src = static_cast<SolidSource*>(pl->insert_stage(std::move(src_u)));
  src_out = src;

  auto st_u = make_unique<SaveImageStage>(
      &sess, "store", vector<InEdge>{{src, 0}}, std::move(store_cfg));
  pl->insert_stage(std::move(st_u));

  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { rt.stop(); return; }
  rt.wait_idle();
  rt.stop();
}

}   // namespace

// path is required; a missing one is deferred to launch (config_error set).
TEST(save_image_stage, config_path_required_deferred) {
  Session sess;
  SaveImageStage s(&sess, "st", vector<InEdge>{}, FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());
  EXPECT_TRUE(s.num_oports() == 0);   // sink
}

// An unsupported format is rejected at construction (deferred error).
TEST(save_image_stage, config_bad_format_deferred) {
  Session sess;
  FlexData cfg = FlexData::from_json(
      R"({"path":"/tmp/x.gif","format":"gif"})");
  SaveImageStage s(&sess, "st", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

// A good config constructs cleanly (format inferred from the extension).
TEST(save_image_stage, config_ok_png) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("path", FlexData::make_string("/tmp/x.png"));
  SaveImageStage s(&sess, "st", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.path() == "/tmp/x.png");
}

// End-to-end: emit a solid RGB beat, write a PNG, verify the magic bytes,
// then decode it back with load-image and check the pixels round-trip
// exactly (PNG is lossless RGB -- no YUV loss).
TEST(save_image_stage, writes_png_roundtrip) {
  Session sess;
  CerrSilencer hush;

  const string path = tmp_path_(".png");
  remove(path.c_str());
  const uint8_t R = 0x12, G = 0xAB, B = 0xF0;

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("path", FlexData::make_string(path));
  SolidSource* src = nullptr;
  run_store_(sess, src, std::move(cfg), 1, R, G, B);

  EXPECT_TRUE(src != nullptr);
  // File exists with the PNG signature.
  const string sig = head_bytes_(path, 8);
  EXPECT_TRUE(sig.size() == 8);
  EXPECT_TRUE(static_cast<uint8_t>(sig[0]) == 0x89 && sig[1] == 'P' &&
              sig[2] == 'N' && sig[3] == 'G');

  // Round-trip decode via load-image.
  auto pl = make_unique<Pipeline>("p2", &sess);
  FlexData licfg = FlexData::make_object();
  licfg.as_object().insert("url", FlexData::make_string(path));
  auto li_u = make_unique<LoadImageStage>(
      &sess, "li", vector<InEdge>{}, std::move(licfg));
  auto* li = static_cast<LoadImageStage*>(pl->insert_stage(std::move(li_u)));

  auto cap_u = make_unique<CaptureSink>(
      &sess, "cap", vector<InEdge>{{li, 0}}, FlexData::make_object());
  auto* cap = static_cast<CaptureSink*>(pl->insert_stage(std::move(cap_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  remove(path.c_str());

  EXPECT_TRUE(cap->got.size() == 1);
  if (cap->got.empty()) { return; }
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(cap->got[0].get());
  EXPECT_TRUE(tb != nullptr);
  if (!tb) { return; }
  EXPECT_TRUE(tb->shape.size() == 3 && tb->shape[0] == 3);
  auto bytes = tb->materialize_contiguous();
  const size_t plane = static_cast<size_t>(tb->shape[1]) * tb->shape[2];
  EXPECT_TRUE(bytes.size() == 3 * plane);
  if (bytes.size() != 3 * plane) { return; }
  const uint8_t* d = bytes.data();
  EXPECT_TRUE(d[0] == R);
  EXPECT_TRUE(d[plane] == G);
  EXPECT_TRUE(d[2 * plane] == B);
}

// The quality knob has teeth: a low-quality JPEG is smaller than a
// high-quality one for the same image.
// The filename index is PER-RUN state: it restarts every launch, so
// each run REWRITES the same file. The stage survives a stop/relaunch,
// so before this was reset the second run wrote `name-000001.png` and
// the third `name-000002.png` -- the file being watched stopped
// updating, silently.
TEST(save_image_stage, relaunch_rewrites_the_same_file) {
  Session sess;
  CerrSilencer hush;
  const string path = tmp_path_(".png");
  const string sfx1 = path.substr(0, path.size() - 4) + "-000001.png";
  remove(path.c_str());
  remove(sfx1.c_str());

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<SolidSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->n = 1; src_u->r = 1; src_u->g = 2; src_u->b = 3;
  auto* src = static_cast<SolidSource*>(pl->insert_stage(std::move(src_u)));
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("path", FlexData::make_string(path));
  auto st_u = make_unique<SaveImageStage>(
      &sess, "store", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* st = static_cast<SaveImageStage*>(pl->insert_stage(std::move(st_u)));

  for (int run = 0; run < 3; ++run) {
    PipelineRuntime rt(pl.get(), &sess);
    EXPECT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
  }
  EXPECT_TRUE(st->images_written() == 1u);   // per-run counter, not 3
  EXPECT_TRUE(!head_bytes_(path, 8).empty());        // the file exists
  EXPECT_TRUE(head_bytes_(sfx1, 8).empty());         // and no -000001
  remove(path.c_str());
  remove(sfx1.c_str());
}

TEST(save_image_stage, jpeg_quality_affects_size) {
  Session sess;
  CerrSilencer hush;

  const string lo = tmp_path_("-lo.jpg");
  const string hi = tmp_path_("-hi.jpg");
  remove(lo.c_str());
  remove(hi.c_str());

  {
    FlexData cfg = FlexData::from_json(
        string("{\"path\":\"") + lo + "\",\"quality\":15}");
    SolidSource* s = nullptr;
    // High-frequency content (noisy=true) so the quantization tables have
    // AC energy to discard -- a solid color compresses the same at any
    // quality. 64x64 makes the size gap unambiguous.
    run_store_(sess, s, std::move(cfg), 1, 0, 0, 0, /*noisy=*/true, 64, 64);
  }
  {
    FlexData cfg = FlexData::from_json(
        string("{\"path\":\"") + hi + "\",\"quality\":98}");
    SolidSource* s = nullptr;
    run_store_(sess, s, std::move(cfg), 1, 0, 0, 0, /*noisy=*/true, 64, 64);
  }

  const size_t lo_sz = file_size_(lo);
  const size_t hi_sz = file_size_(hi);
  remove(lo.c_str());
  remove(hi.c_str());
  EXPECT_TRUE(lo_sz > 0);
  EXPECT_TRUE(hi_sz > 0);
  EXPECT_TRUE(hi_sz > lo_sz);
}

// A stream of images with a printf token lands in distinct files.
TEST(save_image_stage, indexed_stream_paths) {
  Session sess;
  CerrSilencer hush;

  const string p0 = tmp_path_("-seq0.png");
  const string p1 = tmp_path_("-seq1.png");
  remove(p0.c_str());
  remove(p1.c_str());
  const string tmpl = tmp_path_("-seq%d.png");

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("path", FlexData::make_string(tmpl));
  SolidSource* s = nullptr;
  run_store_(sess, s, std::move(cfg), 2, 0x10, 0x20, 0x30);

  const bool has0 = file_size_(p0) > 0;
  const bool has1 = file_size_(p1) > 0;
  remove(p0.c_str());
  remove(p1.c_str());
  EXPECT_TRUE(has0);
  EXPECT_TRUE(has1);
}

// The metadata iport is OPTIONAL: the stage still constructs (and stays a
// sink) with only the image edge wired, which is how every pre-existing
// graph is shaped.
TEST(save_image_stage, metadata_iport_is_declared_and_optional) {
  const StageSpec* si = StageRegistry::get().spec("save-image");
  const StageSpec* li = StageRegistry::get().spec("load-image");
  ASSERT_TRUE(si != nullptr && li != nullptr);
  ASSERT_TRUE(si->iports.size() == 2 && li->oports.size() == 2);
  // save-image's metadata iport must accept exactly what load-image's
  // metadata oport emits, or the two cannot be wired together.
  EXPECT_TRUE(si->iports[1].type == li->oports[1].type);
  EXPECT_TRUE(si->oports.empty());

  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("path", FlexData::make_string("/tmp/x.png"));
  SaveImageStage s(&sess, "st", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
}

// Provenance: generate-image stamps `model_name` on the latent sideband and
// vae-decode carries it onto the image, so a GENERATED image reaches
// save-image knowing what produced it. That must land in EXIF Software as
// "Vpipe <version> <hash> with <model>".
TEST(save_image_stage, generated_image_records_the_model_in_exif_software) {
  Session sess;
  CerrSilencer hush;
  const string path = tmp_path_(".png");
  remove(path.c_str());

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("path", FlexData::make_string(path));
  SolidSource* src = nullptr;
  run_store_(sess, src, std::move(cfg), 1, 0x12, 0xAB, 0xF0,
             false, 4, 3, "local/Some-Model-4bit");
  ASSERT_TRUE(src != nullptr);

  const auto blob = imgmeta::read_exif_blob(path);
  ASSERT_TRUE(!blob.empty());
  FlexData d = imgmeta::parse_exif(blob);   // as_object() is a view
  ASSERT_TRUE(d.is_object());
  auto o = d.as_object();
  ASSERT_TRUE(o.contains("Software"));
  const string sw(o.at("Software").as_string(""));
  std::printf("[save_image_stage] Software = %s\n", sw.c_str());
  EXPECT_TRUE(sw.rfind("Vpipe ", 0) == 0);
  EXPECT_TRUE(sw.find(vpipe_version_number()) != string::npos);
  EXPECT_TRUE(sw.find(vpipe_build_hash()) != string::npos);
  EXPECT_TRUE(sw.find(" with local/Some-Model-4bit") != string::npos);
  remove(path.c_str());
}

// ...and an image with NO model on its sideband (a plain load -> save copy)
// must get no Software tag at all: vpipe did not author it.
TEST(save_image_stage, unmarked_image_gets_no_software_tag) {
  Session sess;
  CerrSilencer hush;
  const string path = tmp_path_(".png");
  remove(path.c_str());

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("path", FlexData::make_string(path));
  SolidSource* src = nullptr;
  run_store_(sess, src, std::move(cfg), 1, 0x12, 0xAB, 0xF0);
  ASSERT_TRUE(src != nullptr);
  EXPECT_TRUE(imgmeta::read_exif_blob(path).empty());
  remove(path.c_str());
}
