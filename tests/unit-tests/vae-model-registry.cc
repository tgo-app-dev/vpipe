// VaeModelRegistry: the VAE-decoder plugin extension point, the
// counterpart to VideoModelRegistry. It is what lets an out-of-tree
// family's latents become frames through the STOCK `vae-decode` stage
// rather than a stage of its own.
//
// The vae-decode CONSULT itself (a claim_for() before the built-in
// `_class_name` chain) needs a checkpoint on disk to exercise, so it is
// not tested here -- the same boundary video-model-registry.cc draws.
// This isolates the registry contract that consult depends on:
// first-wins, claim order, a throwing probe not being fatal, the refusal
// to take a built-in family's NAME, and the frame-chunk contract
// including its abort path.

#include "minitest.h"

#include "generative-models/vae-model-registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

// A decoder that hands its frames over in a configurable number of
// chunks, so the sink contract can be checked both ways round.
class StubDecoder : public VaeDecoder {
public:
  StubDecoder(int frames, int chunks) : _frames(frames), _chunks(chunks) {}

  int latent_channels() const override { return 128; }
  int spatial_compression() const override { return 32; }
  int temporal_compression() const override { return 8; }

  bool decode(const VaeDecodeRequest& req, const VaeFrameSink& sink,
              std::string* err) override
  {
    (void)req;
    (void)err;
    const int H = 2, W = 2;
    _pixels.assign((std::size_t)3 * _frames * H * W, 0.25f);
    int done = 0;
    for (int c = 0; c < _chunks && done < _frames; ++c) {
      const int n = (c == _chunks - 1) ? (_frames - done)
                                       : (_frames / _chunks);
      VaeFrameChunk k;
      k.rgb          = _pixels.data();
      k.channels     = 3;
      k.frame0       = done;
      k.n            = n;
      k.height       = H;
      k.width        = W;
      k.frames_total = _frames;
      if (!sink(k)) { return false; }     // the abort path
      done += n;
    }
    return true;
  }

private:
  int _frames, _chunks;
  std::vector<float> _pixels;
};

class StubFamily : public VaeModelFamily {
public:
  StubFamily(std::string tag, std::string want, bool throws = false)
    : _tag(std::move(tag)), _want(std::move(want)), _throws(throws)
  {
  }

  std::string_view tag() const noexcept override { return _tag; }

  bool claims(const std::string& root, const std::string&,
              const std::string&) const override
  {
    if (_throws) { throw std::runtime_error("probe blew up"); }
    return root == _want;
  }

  std::unique_ptr<VaeDecoder>
  load_decoder(const VaeModelCreateArgs&) override
  {
    return std::make_unique<StubDecoder>(9, 1);
  }

private:
  std::string _tag, _want;
  bool _throws;
};

}  // namespace

TEST(vae_model_registry, add_is_first_wins_on_tag)
{
  auto& r = VaeModelRegistry::get();
  EXPECT_TRUE(r.add(std::make_unique<StubFamily>("ut-vae-a", "/roots/a")));
  // A second family claiming a tag already present is refused, so two
  // plugins shipping one model cannot silently shadow each other.
  EXPECT_FALSE(r.add(std::make_unique<StubFamily>("ut-vae-a", "/roots/z")));
  EXPECT_TRUE(r.find("ut-vae-a") != nullptr);
  // The refused one did not replace it: the survivor still claims /roots/a.
  EXPECT_TRUE(r.claim_for(nullptr, "/roots/a", "/roots/a", "") != nullptr);
  EXPECT_TRUE(r.claim_for(nullptr, "/roots/z", "/roots/z", "") == nullptr);
}

TEST(vae_model_registry, refuses_a_built_in_family_name)
{
  auto& r = VaeModelRegistry::get();
  // Dispatch in vae-decode is pointer-guarded, so a collision would
  // still run the right code -- but every log line would read as a
  // built-in, which is why the name is refused outright.
  for (const char* built_in : {"wan", "minimax-h3", "flux2", "mage",
                               "krea2"}) {
    EXPECT_FALSE(r.add(std::make_unique<StubFamily>(built_in, "/roots/x")));
  }
  EXPECT_TRUE(r.claim_for(nullptr, "/roots/x", "/roots/x", "") == nullptr);
}

TEST(vae_model_registry, a_throwing_probe_is_skipped_not_fatal)
{
  auto& r = VaeModelRegistry::get();
  EXPECT_TRUE(r.add(std::make_unique<StubFamily>("ut-vae-throws", "/nope",
                                                 /*throws=*/true)));
  EXPECT_TRUE(r.add(std::make_unique<StubFamily>("ut-vae-b", "/roots/b")));
  // The thrower is asked first (registration order) and must not stop
  // the walk, nor take the host down.
  VaeModelFamily* f = r.claim_for(nullptr, "/roots/b", "/roots/b", "");
  EXPECT_TRUE(f != nullptr);
  if (f != nullptr) { EXPECT_TRUE(f->tag() == "ut-vae-b"); }
}

TEST(vae_model_registry, decoded_frames_is_the_causal_rule)
{
  StubDecoder d(9, 1);
  // 1 + r*(T-1) with r == 8: the rule every video VAE in this tree has.
  EXPECT_TRUE(d.decoded_frames(2) == 9);
  EXPECT_TRUE(d.decoded_frames(1) == 1);
  EXPECT_TRUE(d.decoded_frames(0) == 0);
  // An image VAE inherits 1 from temporal_compression() == 1.
  class ImageDec : public StubDecoder {
  public:
    ImageDec() : StubDecoder(1, 1) {}
    int temporal_compression() const override { return 1; }
  } img;
  EXPECT_TRUE(img.decoded_frames(1) == 1);
}

TEST(vae_model_registry, the_sink_sees_every_frame_once_in_order)
{
  for (int chunks : {1, 3}) {
    StubDecoder d(9, chunks);
    VaeDecodeRequest req;
    req.shape = {128, 2, 2, 2};
    std::vector<int> seen;
    int total_claimed = 0;
    std::string err;
    const bool ok = d.decode(req, [&](const VaeFrameChunk& c) {
      total_claimed = c.frames_total;
      for (int k = 0; k < c.n; ++k) { seen.push_back(c.frame0 + k); }
      return true;
    }, &err);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(total_claimed == 9);
    EXPECT_TRUE(seen.size() == 9);
    // Disjoint and in frame order, whether it arrived in one chunk or
    // several -- chunking is permitted, never required.
    bool ordered = true;
    for (std::size_t i = 0; i < seen.size(); ++i) {
      if (seen[i] != (int)i) { ordered = false; }
    }
    EXPECT_TRUE(ordered);
  }
}

// The ENCODER half of the same family. Both extra loaders are ADDITIVE
// virtuals defaulting to "decline", so a family that only decodes video
// must keep working without mentioning either -- which is the thing that
// makes them safe to add to a shipped interface.
TEST(vae_model_registry, the_extra_loaders_default_to_declining)
{
  StubFamily f("ut-vae-defaults", "/roots/x");
  VaeModelCreateArgs args;
  args.root = "/roots/x";
  EXPECT_TRUE(f.load_encoder(args) == nullptr);
  EXPECT_TRUE(f.load_audio_decoder(args) == nullptr);
  // ...while the one it DOES override still answers.
  EXPECT_TRUE(f.load_decoder(args) != nullptr);
}

TEST(vae_model_registry, an_encoder_reports_its_own_latent_shape)
{
  // The stage publishes the shape the encoder RETURNS, never one it
  // predicted from spatial_compression() -- a family whose latent has a
  // time axis and one whose latent does not are both legal, and only the
  // encoder knows which it is.
  class StubEncoder : public VaeEncoder {
  public:
    int latent_channels() const override { return 128; }
    int spatial_compression() const override { return 32; }
    int temporal_compression() const override { return 8; }
    bool encode(const VaeEncodeRequest& req, std::vector<float>* out,
                std::vector<int>* shape, std::string* err) override
    {
      if (req.height % 32 != 0 || req.width % 32 != 0) {
        if (err != nullptr) { *err = "not a multiple of 32"; }
        return false;
      }
      const std::vector<int> s = {128, 1, req.height / 32, req.width / 32};
      std::size_t n = 1;
      for (int d : s) { n *= (std::size_t)d; }
      out->assign(n, 0.5f);
      *shape = s;
      return true;
    }
  } e;

  VaeEncodeRequest req;
  req.frames = 1;
  req.height = 320;
  req.width  = 512;
  std::vector<float> out;
  std::vector<int> shape;
  std::string err;
  EXPECT_TRUE(e.encode(req, &out, &shape, &err));
  EXPECT_TRUE(shape.size() == 4);
  EXPECT_TRUE(shape[0] == 128 && shape[1] == 1 && shape[2] == 10 &&
              shape[3] == 16);
  EXPECT_TRUE(out.size() == 128u * 10u * 16u);

  // A size the family cannot take is REFUSED with a reason, not rounded:
  // the stage emits no beat, which is what every refusal in this graph
  // does.
  req.height = 300;
  EXPECT_FALSE(e.encode(req, &out, &shape, &err));
  EXPECT_TRUE(!err.empty());
}

TEST(vae_model_registry, a_false_sink_aborts_the_decode)
{
  StubDecoder d(9, 3);
  VaeDecodeRequest req;
  req.shape = {128, 2, 2, 2};
  int calls = 0;
  std::string err;
  // Returning false is how a Stop mid-decode gets out; decode() must
  // then report failure rather than claim a partial clip succeeded.
  const bool ok = d.decode(req, [&](const VaeFrameChunk&) {
    ++calls;
    return false;
  }, &err);
  EXPECT_FALSE(ok);
  EXPECT_TRUE(calls == 1);
}
