// stage-relaunch-sweep.cc -- one test for a whole BUG CLASS.
//
// Stopping a pipeline destroys the runtime, not the stages: only unload
// or re-materialize destroys a Stage. So every stage object sees many
// launches, and any member meaning "something that happened during THIS
// run" is wrong on the second one unless Stage::reset_run_state() clears
// it. That has bitten repeatedly, always silently -- a one-shot source
// that never re-emits, a latched selection that ignores the new beat, a
// filename counter that stops overwriting the file being watched.
//
// This sweeps every SOURCE stage: launch it twice over the same object
// and assert run 2 emits what run 1 did. It would have caught
// model-select, text-prompt, chrono, load-image and load-text before any
// of them reached a user.
//
// Adding a source type? Put it in kSweep. The completeness test below
// FAILS on a registered source that isn't listed, so the decision is
// forced rather than forgotten. If it genuinely can't run headless, list
// it with skip=true and a reason -- that is still a decision on record.
//
//   vpipe_test --filter '*relaunch_sweep*'

#include "minitest.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-registry.h"
#include "pipeline/stage-spec.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <string_view>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

class CerrHush {
public:
  CerrHush() : _saved(cerr.rdbuf()), _null() { cerr.rdbuf(&_null); }
  ~CerrHush() { cerr.rdbuf(_saved); }
private:
  struct NullBuf : public streambuf {
    int overflow(int c) override { return c; }
  };
  streambuf* _saved;
  NullBuf    _null;
};

// Counts beats on iport 0 and never blocks the producer.
class CountSink : public TypedStage<CountSink> {
public:
  static constexpr const char* kTypeName = "ut-relaunch-count-sink";
  using TypedStage::TypedStage;
  int n = 0;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    ++n;
  }
};

struct SweepEntry {
  const char* type;
  const char* config;   // minimal config that makes it actually EMIT
  bool        skip;
  const char* why;      // required when skip
};

// A 1x1 opaque-black PNG, so load-image has something real to decode
// without shipping a fixture file.
const unsigned char kPng1x1[] = {
  0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0x00,0x00,0x00,0x0d,
  0x49,0x48,0x44,0x52,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,
  0x08,0x02,0x00,0x00,0x00,0x90,0x77,0x53,0xde,0x00,0x00,0x00,
  0x0c,0x49,0x44,0x41,0x54,0x08,0xd7,0x63,0x60,0x60,0x60,0x00,
  0x00,0x00,0x04,0x00,0x01,0x27,0x34,0x27,0x0a,0x00,0x00,0x00,
  0x00,0x49,0x45,0x4e,0x44,0xae,0x42,0x60,0x82
};

string
tmpdir_()
{
  const char* t = std::getenv("TMPDIR");
  return string(t && *t ? t : "/tmp");
}

// Written once and reused by the load-* entries.
string
png_path_()
{
  const string p = tmpdir_() + "/vpipe-sweep-1x1.png";
  std::ofstream o(p, std::ios::binary);
  o.write(reinterpret_cast<const char*>(kPng1x1), sizeof kPng1x1);
  return p;
}

string
txt_path_()
{
  const string p = tmpdir_() + "/vpipe-sweep.txt";
  std::ofstream o(p);
  o << "one line\n";
  return p;
}

}  // namespace

// Launch `type` twice over ONE stage object and report the beats each
// run produced. -1 means the stage could not be constructed/configured.
struct SweepResult { int run1 = -1; int run2 = -1; };

static SweepResult
sweep_one_(const SweepEntry& e)
{
  SweepResult r;
  Session sess;
  CerrHush hush;
  auto pl = make_unique<Pipeline>("sweep", &sess);

  FlexData cfg = FlexData::from_json(e.config);
  StagePtr sp  = StageRegistry::get().create(
      e.type, &sess, "src", vector<InEdge>{}, std::move(cfg));
  if (!sp) { return r; }
  Stage* src = pl->insert_stage(std::move(sp));
  if (src == nullptr || !src->config_error().empty()) { return r; }

  auto sk_u = make_unique<CountSink>(
      &sess, "sink", vector<InEdge>{{src, 0}}, FlexData::make_object());
  auto* sk = static_cast<CountSink*>(pl->insert_stage(std::move(sk_u)));

  int per_run[2] = {0, 0};
  for (int run = 0; run < 2; ++run) {
    const int before = sk->n;
    PipelineRuntime rt(pl.get(), &sess);
    if (!rt.launch()) { return r; }
    rt.wait_idle(30000);
    rt.stop();
    per_run[run] = sk->n - before;
  }
  r.run1 = per_run[0];
  r.run2 = per_run[1];
  return r;
}

static const SweepEntry*
sweep_table_(size_t* n);

TEST(relaunch_sweep, sources_emit_again_on_every_launch)
{
  size_t n = 0;
  const SweepEntry* tbl = sweep_table_(&n);
  for (size_t i = 0; i < n; ++i) {
    if (tbl[i].skip) {
      std::printf("[relaunch_sweep] %-26s SKIPPED (%s)\n",
                  tbl[i].type, tbl[i].why);
      continue;
    }
    const SweepResult r = sweep_one_(tbl[i]);
    std::printf("[relaunch_sweep] %-26s run1=%d run2=%d\n",
                tbl[i].type, r.run1, r.run2);
    // run1 > 0 keeps this from passing vacuously on a stage left inert
    // by its config; run2 == run1 is the property under test.
    EXPECT_TRUE(r.run1 > 0);
    EXPECT_TRUE(r.run2 == r.run1);
  }
}

// Every registered source must appear in kSweep. "Source" = declares no
// iports at all, or declares only ones its own spec documents as
// optional (load-image's pacing trigger, text-prompt's gate) -- those
// run standalone and are exactly as exposed to this bug.
//
// The optional-iport half is a doc-text heuristic because PortSpec has
// no `required` flag. It errs toward INCLUDING a stage, so the failure
// mode is "you must list it", never silent omission.
TEST(relaunch_sweep, every_registered_source_is_covered)
{
  size_t n = 0;
  const SweepEntry* tbl = sweep_table_(&n);

  for (const auto& [id, name] : StageRegistry::get().all()) {
    (void)id;
    if (name.rfind("ut-", 0) == 0) { continue; }   // test-only stages
    const StageSpec* sp = StageRegistry::get().spec(name);
    if (sp == nullptr || sp->oports.empty()) { continue; }

    bool source = sp->iports.empty();
    if (!source) {
      source = true;
      for (const auto& ip : sp->iports) {
        string d(ip.doc);
        for (char& c : d) { c = (char)tolower((unsigned char)c); }
        if (d.find("optional") == string::npos) { source = false; break; }
      }
    }
    if (!source) { continue; }

    bool listed = false;
    for (size_t i = 0; i < n && !listed; ++i) {
      listed = (name == tbl[i].type);
    }
    if (!listed) {
      std::printf("[relaunch_sweep] UNCOVERED SOURCE: %s -- add it to "
                  "kSweep in stage-relaunch-sweep.cc\n", name.c_str());
    }
    EXPECT_TRUE(listed);
  }
}

static const SweepEntry*
sweep_table_(size_t* n)
{
  static const string png = png_path_();
  static const string txt = txt_path_();
  static const string load_image_cfg =
      "{\"url\":[\"" + png + "\"]}";
  static const string load_text_cfg =
      "{\"path\":[\"" + txt + "\"]}";

  static const SweepEntry kSweep[] = {
    {"chrono", "{\"frequency_hz\":200,\"count\":3}", false, nullptr},
    {"model-select", "{\"hf_dir\":\"/models/x\"}", false, nullptr},
    {"sampler-select", "{}", false, nullptr},
    {"diffusion-sampler-select", "{}", false, nullptr},
    {"scheduler-select", "{}", false, nullptr},
    {"text-prompt", "{\"text\":\"hello\"}", false, nullptr},
    {"load-image", load_image_cfg.c_str(), false, nullptr},
    {"load-text", load_text_cfg.c_str(), false, nullptr},

    // Decisions on record rather than silent gaps. Every one of these
    // was read for the latch pattern this test exists to catch (a member
    // gating emission that reset_run_state() does not clear) and none
    // has one: the model-* / lora-fuse action stages redo their whole
    // job on each launch, and the capture stages are driven by a device
    // rather than a counter.
    {"text-input", "{}", true, "reads stdin; no input in the runner"},
    {"load-video", "{}", true, "needs a decodable video fixture"},
    {"audio-capture", "{}", true, "needs a capture device"},
    {"video-capture", "{}", true, "needs a camera"},
    {"rtsp-capture", "{}", true, "needs a reachable RTSP camera"},
    {"feedback-tx", "{}", true, "needs a feedback-rx peer in the graph"},
    {"create-mask", "{}", true, "needs an interactive mask editor client"},
    {"model-fetch", "{}", true, "downloads from the network"},
    {"model-remove", "{}", true, "destructive: deletes a local model dir"},
    {"model-register", "{}", true,
     "writes a registry record; the sweep Session has no db.path, so it "
     "would land in the runner's CWD models DB"},
    {"lora-fuse", "{}", true, "needs a base model + LoRA on disk"},
    {"model-quantize", "{}", true, "needs a model on disk; minutes per run"},
    {"model-eval", "{}", true, "needs a model + eval dataset on disk"},
    {"model-benchmark", "{}", true, "needs a model on disk; minutes per run"},
  };
  *n = sizeof kSweep / sizeof kSweep[0];
  return kSweep;
}
