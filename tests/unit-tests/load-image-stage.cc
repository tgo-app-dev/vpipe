#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/chrono-stage.h"
#include "stages/load-image-stage.h"

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

// Test-only sink: 1 iport, 0 oports. Captures every received payload
// into `captured` until EOS, then signals done. Used to inspect the
// TensorBeats emitted by LoadImageStage end-to-end.
class SinkCapture : public TypedStage<SinkCapture> {
public:
  static constexpr const char* kTypeName = "ut-sink-capture";
  using TypedStage::TypedStage;

  vector<unique_ptr<BeatPayloadIntf>> captured;

  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) {
      ctx.signal_done();
      co_return;
    }
    captured.push_back(std::move(p));
  }
};

string
write_test_ppm_(int W, int H, uint8_t r, uint8_t g, uint8_t b)
{
  string path = string("/tmp/vpipe-load-image-")
              + to_string(getpid()) + ".ppm";
  ofstream out(path, ios::binary);
  out << "P6\n" << W << " " << H << "\n255\n";
  for (int i = 0; i < W * H; ++i) {
    out.put(static_cast<char>(r));
    out.put(static_cast<char>(g));
    out.put(static_cast<char>(b));
  }
  out.close();
  return path;
}

}

TEST(load_image_stage, config_url_string) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("url", FlexData::make_string("/tmp/x.png"));
  LoadImageStage s(&sess, "li", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.urls().size() == 1);
  EXPECT_TRUE(s.urls()[0] == "/tmp/x.png");
  EXPECT_TRUE(s.num_oports() == 1);
}

TEST(load_image_stage, config_url_array) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"url":["/a.png","/b.jpg","/c.webp"]})");
  LoadImageStage s(&sess, "li", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.urls().size() == 3);
  EXPECT_TRUE(s.urls()[0] == "/a.png");
  EXPECT_TRUE(s.urls()[1] == "/b.jpg");
  EXPECT_TRUE(s.urls()[2] == "/c.webp");
}

// Construction succeeds for any config; a missing/empty url is
// recorded in config_error() and deferred to launch.
TEST(load_image_stage, config_url_required_deferred) {
  Session sess;
  LoadImageStage s(&sess, "li", vector<InEdge>{},
                   FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());
}

TEST(load_image_stage, config_empty_url_deferred) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("url", FlexData::make_string(""));
  LoadImageStage s(&sess, "li", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(load_image_stage, config_empty_array_deferred) {
  Session sess;
  FlexData cfg = FlexData::from_json(R"({"url":[]})");
  LoadImageStage s(&sess, "li", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

// End-to-end: write a tiny PPM, run the pipeline, expect one
// TensorBeat with the painted color in R,G,B order.
TEST(load_image_stage, decodes_solid_ppm) {
  Session sess;
  CerrSilencer hush;

  const int W = 4;
  const int H = 2;
  const uint8_t R = 0xFF, G = 0x80, B = 0x40;
  string path = write_test_ppm_(W, H, R, G, B);

  auto pl = make_unique<Pipeline>("p", &sess);

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("url", FlexData::make_string(path));
  auto li_u = make_unique<LoadImageStage>(
      &sess, "li", vector<InEdge>{}, std::move(cfg));
  auto* li = static_cast<LoadImageStage*>(
      pl->insert_stage(std::move(li_u)));

  auto sink_u = make_unique<SinkCapture>(
      &sess, "sink", vector<InEdge>{{li, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  remove(path.c_str());

  EXPECT_TRUE(sink->captured.size() == 1);
  if (sink->captured.empty()) {
    return;
  }
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(
      sink->captured[0].get());
  EXPECT_TRUE(tb != nullptr);
  if (!tb) {
    return;
  }
  EXPECT_TRUE(tb->shape.size() == 3);
  EXPECT_TRUE(tb->shape[0] == 3);
  EXPECT_TRUE(tb->shape[1] == H);
  EXPECT_TRUE(tb->shape[2] == W);
  EXPECT_TRUE(tb->dtype == TensorBeat::DType::U8);

  // Materialise into a contiguous [3, H, W] U8 buffer so the plane
  // stride is exactly H*W and indexing is straightforward whether
  // FFmpeg used pitch padding or not.
  auto bytes = tb->materialize_contiguous();
  EXPECT_TRUE(bytes.size() == static_cast<size_t>(3 * H * W));
  const uint8_t* d = bytes.data();
  // Sample the first pixel of each plane. The R/G/B->Y'CbCr round
  // trip through gbrp swscale loses a handful of LSBs, so accept a
  // small absolute tolerance instead of an exact byte match.
  auto near = [](uint8_t a, uint8_t b) {
    int da = static_cast<int>(a) - static_cast<int>(b);
    return (da > -4 && da < 4);
  };
  EXPECT_TRUE(near(d[0], R));
  EXPECT_TRUE(near(d[W * H + 0], G));
  EXPECT_TRUE(near(d[2 * W * H + 0], B));
}

// Multi-URL emits exactly one TensorBeat per URL.
TEST(load_image_stage, multi_url_emits_each) {
  Session sess;
  CerrSilencer hush;

  string p1 = write_test_ppm_(2, 2, 0xFF, 0x00, 0x00);
  // Distinct second path so we exercise multi-url, but identical
  // dimensions keep the test simple.
  string p2 = string("/tmp/vpipe-load-image-")
            + to_string(getpid()) + "-b.ppm";
  {
    ofstream out(p2, ios::binary);
    out << "P6\n2 2\n255\n";
    for (int i = 0; i < 4; ++i) {
      out.put(0x00); out.put(0xFF); out.put(0x00);
    }
  }

  auto pl = make_unique<Pipeline>("p", &sess);

  FlexData cfg = FlexData::make_object();
  {
    FlexData urls = FlexData::make_array();
    urls.as_array().push_back(string_view(p1));
    urls.as_array().push_back(string_view(p2));
    cfg.as_object().insert("url", std::move(urls));
  }
  auto li_u = make_unique<LoadImageStage>(
      &sess, "li", vector<InEdge>{}, std::move(cfg));
  auto* li = static_cast<LoadImageStage*>(
      pl->insert_stage(std::move(li_u)));

  auto sink_u = make_unique<SinkCapture>(
      &sess, "sink", vector<InEdge>{{li, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  remove(p1.c_str());
  remove(p2.c_str());

  EXPECT_TRUE(sink->captured.size() == 2);
}

// Paced mode: when an iport is wired to a chrono source, each tick
// triggers one decode. With 3 URLs we expect exactly 3 emissions,
// regardless of how many ticks the chrono stage fires (chrono.count
// is set to 3 here to bound the test; the load-image stage signals
// done after the third URL anyway).
TEST(load_image_stage, chrono_paced) {
  Session sess;
  CerrSilencer hush;

  string p1 = write_test_ppm_(2, 2, 0xFF, 0, 0);
  string p2 = string("/tmp/vpipe-load-image-") + to_string(getpid())
            + "-c2.ppm";
  {
    ofstream out(p2, ios::binary);
    out << "P6\n2 2\n255\n";
    for (int i = 0; i < 4; ++i) {
      out.put(0); out.put(0xFF); out.put(0);
    }
  }
  string p3 = string("/tmp/vpipe-load-image-") + to_string(getpid())
            + "-c3.ppm";
  {
    ofstream out(p3, ios::binary);
    out << "P6\n2 2\n255\n";
    for (int i = 0; i < 4; ++i) {
      out.put(0); out.put(0); out.put(0xFF);
    }
  }

  auto pl = make_unique<Pipeline>("p", &sess);

  // 1 kHz chrono with count=3 so the test is bounded even if the
  // load-image stage misbehaves and never signals done.
  FlexData chrono_cfg = FlexData::from_json(
      R"({"frequency_hz":1000.0,"count":3})");
  auto chrono_u = make_unique<ChronoStage>(
      &sess, "ck", vector<InEdge>{}, std::move(chrono_cfg));
  auto* chrono = static_cast<ChronoStage*>(
      pl->insert_stage(std::move(chrono_u)));

  FlexData cfg = FlexData::make_object();
  {
    FlexData urls = FlexData::make_array();
    urls.as_array().push_back(string_view(p1));
    urls.as_array().push_back(string_view(p2));
    urls.as_array().push_back(string_view(p3));
    cfg.as_object().insert("url", std::move(urls));
  }
  auto li_u = make_unique<LoadImageStage>(
      &sess, "li", vector<InEdge>{{chrono, 0}}, std::move(cfg));
  auto* li = static_cast<LoadImageStage*>(
      pl->insert_stage(std::move(li_u)));

  auto sink_u = make_unique<SinkCapture>(
      &sess, "sink", vector<InEdge>{{li, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  remove(p1.c_str());
  remove(p2.c_str());
  remove(p3.c_str());

  EXPECT_TRUE(sink->captured.size() == 3);
}
