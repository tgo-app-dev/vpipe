#ifndef VPIPE_GENERATIVE_MODELS_VAE_MODEL_REGISTRY_H
#define VPIPE_GENERATIVE_MODELS_VAE_MODEL_REGISTRY_H

#include "common/flex-data.h"
#include "pipeline/memory-plan.h"
#include "pipeline/resource-plan.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
class SessionContextIntf;
namespace metal_compute { class MetalCompute; }
}

namespace vpipe::genai {

// An out-of-tree VAE family, for `vae-decode`.
//
// WHY THIS IS NOT A VIRTUAL BASE OVER THE BUILT-IN DECODERS. The stage
// holds five concrete types side by side, and they genuinely do not
// share a shape: MetalKrea2Vae::decode(zw, h8, w8) returns one f16
// buffer; MetalWanVae::decode(zw, T, h8, w8, sink) STREAMS chunks;
// MetalMiniMaxH3VideoVae::decode_video returns the whole clip and does
// so in IMAGENET-NORMALIZED space rather than [-1, 1]. An interface wide
// enough for all three describes none of them -- the same argument
// video-model-registry.h makes about the per-step DiT forward.
//
// So the seam is drawn one level up, at ONE DECODE: given a latent and
// its sideband, produce RGB frames. Un-whitening, tiling, colour-space
// un-normalisation and residency all live INSIDE the family, because
// each built-in already does them differently and each knows its own
// rule. The stage keeps the ports, the U8 quantisation, the per-frame
// beat, the sideband stamping, the model-select latch and the
// idle-unload policy.
//
// The registry is consulted BEFORE the built-in `_class_name` chain,
// mirroring how generate-video consults VideoModelRegistry before its
// built-in wan / minimax-h3 dispatch. The built-in branches are
// unchanged: this adds a path, it does not reroute the existing ones.

// One decode's input. Every pointer is BORROWED from the beat that
// outlives the call; nothing here is owned.
struct VaeDecodeRequest {
  // The latent EXACTLY as the beat carried it: f32, row-major over
  // `shape`, and still in NORMALISED latent space. Un-whitening is the
  // decoder's business, as it is for every built-in here -- the
  // statistics live in the checkpoint the family read, not in the stage.
  //
  // `shape` is 3-D [z, h, w] for an image VAE or 4-D [z, T, h, w] for a
  // video one. The stage refuses any other rank before it gets here, so
  // a family with a batch axis learns that from this comment rather than
  // from a runtime skip.
  const float*     latent = nullptr;
  std::vector<int> shape;

  // The clip's frame rate: the producer's when the latent carried one,
  // else the stage's `fps` config. Passed down because a family may need
  // it (a temporal-tiling rule, a stamped pts); the STAGE stamps the
  // sideband, so a family that ignores this loses nothing.
  double fps = 16.0;

  // The latent beat's sideband, UNREAD -- the same discipline
  // VideoGenRequest::model_config uses, so a key a family starts
  // honouring later needs no change in vae-decode. Null / non-object
  // when the producer sent none.
  const FlexData* sideband = nullptr;

  // Returns false to ABORT (the pipeline is stopping). Called at
  // whatever granularity the family has. A family that never calls it
  // cannot be interrupted, which on a long decode is a Stop that looks
  // like a hang -- but a family that STREAMS gets cancellation for free,
  // because the sink's return value is also an abort point.
  std::function<bool(int done, int total)> progress;
};

// One chunk of decoded frames, in frame order.
//
// A STRUCT rather than positional arguments so it can grow additively --
// an alpha plane, a per-chunk pts, a wider dtype -- without breaking
// every family that compiled against it.
struct VaeFrameChunk {
  // f32, CHANNEL-FIRST OVER THE CHUNK: [channels][n][height][width], in
  // [-1, 1].
  //
  // f32 because a chunk is small by construction and the host's single
  // (x+1)/2*255 is the only place pixels get quantised. A plain pointer
  // rather than a SharedBuffer because a decoder that computed into HOST
  // memory must not have to copy into a GPU buffer to hand it over.
  //
  // [-1, 1] IS THE CONTRACT, not a suggestion. MiniMax-H3 decodes in
  // imagenet-normalized space and the STAGE un-normalizes it; a plugin
  // family in another space un-normalizes INSIDE itself, because the
  // alternative is the host learning each family's colour convention.
  const float* rgb = nullptr;
  // 3 normally. The host maps output channel c to min(c, channels-1), so
  // a 1-channel decoder comes out greyscale rather than read out of
  // bounds.
  int channels = 3;
  int frame0   = 0;      // this chunk's first frame within the clip
  int n        = 0;      // frames in this chunk
  int height   = 0;      // constant across one decode
  int width    = 0;

  // Frames in the WHOLE clip -- what the beat's `frames` sideband key
  // becomes. AUTHORITATIVE; VaeDecoder::decoded_frames() below is for
  // the log line and the preflight only. A family that mispredicts its
  // own geometry should produce a confusing log, not a mislabelled beat.
  int frames_total = 0;
};

// Returning false ABORTS the decode; decode() then returns false too.
using VaeFrameSink = std::function<bool(const VaeFrameChunk&)>;

// ONE resident VAE decoder of a family. Built by
// VaeModelFamily::load_decoder() and held by the stage for as long as
// that checkpoint is the model.
class VaeDecoder {
public:
  virtual ~VaeDecoder() = default;

  // Geometry, for the log line the stage writes and for its output
  // preflight. NEVER for the beat's shape -- that comes from the chunk.
  virtual int latent_channels() const = 0;
  virtual int spatial_compression() const = 0;   // 8 wan, 16 h3, 32 LTX
  virtual int temporal_compression() const { return 1; }  // 1 = image VAE

  // Frames a latent with `latent_frames` on its time axis decodes to.
  // The default is the causal rule every video VAE in this tree has
  // (1 + r*(T-1)); an image VAE inherits 1 from r == 1. Must be CHEAP:
  // the stage asks before it allocates the clip.
  virtual int decoded_frames(int latent_frames) const
  {
    const int r = temporal_compression();
    return latent_frames <= 0 ? 0 : 1 + r * (latent_frames - 1);
  }

  // Decode one latent. `sink` is called one or more times, in frame
  // order, with disjoint chunks covering [0, frames_total).
  //
  // A decoder that can only produce the whole clip calls the sink ONCE
  // with frame0 = 0 and n = frames_total; chunking is permitted, never
  // required. False means it failed and produced nothing usable -- the
  // stage then writes no beat, which is what every refusal in this graph
  // does -- and `err` carries the reason for its one warn line. NEVER
  // throws across this boundary.
  virtual bool decode(const VaeDecodeRequest& req, const VaeFrameSink& sink,
                      std::string* err) = 0;

  // Drop what can be dropped between beats, for the softer of the two
  // residency levers: the stage destroys the decoder outright when its
  // verdict is "unload", and would call this when the verdict is "keep
  // resident" but a peer wants room. NOTHING CALLS IT YET -- it ships
  // now so that wiring GenerativeModelManager::reclaim_at_least to it
  // later is additive rather than an interface change.
  virtual void release_idle() {}

  // Bytes held, for the model manager's session-wide view. Should be
  // answered: a decoder whose weights are invisible to a WeightSet makes
  // every peer size itself against a checkpoint it cannot see. See
  // docs/MODEL-MEMORY.md, "Declarations".
  virtual std::uint64_t resident_bytes() const { return 0; }
};

// ---- the AUDIO half -------------------------------------------------
//
// `audio-vae-decode` is a separate stage from `vae-decode` because the
// two produce different things -- one emits RGB frames, the other emits
// samples, and a stage's output type is what a graph wires against. It
// is NOT a separate registry: a family that generates both modalities
// answers for both, so `load_audio_decoder` is an additive virtual on
// VaeModelFamily with a default that declines.

struct AudioVaeDecodeRequest {
  // The latent soundtrack EXACTLY as the beat carried it, f32 and still
  // in whatever space the family's own generator emits. Shapes differ by
  // family -- MiniMax-H3 sends [stereo, channels, frames], LTX-2.5 sends
  // [channels, frames, mel_bins] -- so the shape is passed through and
  // the FAMILY interprets it, exactly as the video side does.
  const float*     latent = nullptr;
  std::vector<int> shape;

  // Latent frames per second, off the producer's sideband
  // (`latents_per_second`). 0 when it sent none.
  double latents_per_second = 0.0;

  // The beat's sideband, UNREAD.
  const FlexData* sideband = nullptr;

  // False ABORTS. A decode short enough never to call it is fine; a long
  // one that does not is a Stop that looks like a hang.
  std::function<bool(int done, int total)> progress;
};

// ONE resident audio VAE decoder.
//
// No sink here, unlike the video side: a soundtrack is small by
// construction (5 s of 48 kHz stereo is 2 MB of f32), so the chunking
// that exists to bound a 121-frame clip would be machinery for nothing.
class AudioVaeDecoder {
public:
  virtual ~AudioVaeDecoder() = default;

  // The rate of the PCM this produces. The STAGE stamps it on the beat,
  // so a family that reports one rate and emits another is a clip that
  // plays at the wrong speed -- which no amount of listening to the
  // samples alone would reveal.
  virtual int sample_rate() const = 0;

  // Decode to PLANAR f32 PCM. `pcm` is [channels][n_samples] laid out
  // channel-major and clamped to [-1, 1]; `shape` is {channels,
  // n_samples}. False means it failed and produced nothing usable --
  // the stage then writes no beat. NEVER throws across this boundary.
  virtual bool decode(const AudioVaeDecodeRequest& req,
                      std::vector<float>* pcm, std::vector<int>* shape,
                      std::string* err) = 0;

  virtual void release_idle() {}
  virtual std::uint64_t resident_bytes() const { return 0; }
};

// Everything a family's load receives. Mirrors VideoModelCreateArgs.
// ---------------------------------------------------------------------
// The ENCODER half.
//
// `vae-encode` has the same hardcoded `_class_name` chain `vae-decode`
// had, and the same reason to open it: an out-of-tree family's latent
// geometry is its own (LTX-2.5 is 32x at 128 channels, where every
// built-in here is 8x or 16x at 16 or 32), so a graph that can decode a
// family's latents but not produce them can generate but never
// reference an image.
//
// SINGLE IMAGE IN, deliberately. The stage reads one RGB beat and the
// video families synthesise a clip from it, so `frames` is what the
// FAMILY was asked to span rather than a count of pictures supplied. A
// family that wants a real multi-frame reference needs the stage to
// gather frames, which it does not do for anyone yet.
struct VaeEncodeRequest {
  // f32 [3][frames][height][width], channel-first, already in [-1, 1]
  // and already letterboxed to the target size. `frames` is 1 unless the
  // stage built a clip.
  const float* pixels = nullptr;
  int frames = 1;
  int height = 0;
  int width  = 0;
};

// ONE resident VAE encoder.
class VaeEncoder {
public:
  virtual ~VaeEncoder() = default;

  virtual int latent_channels() const = 0;
  virtual int spatial_compression() const = 0;
  virtual int temporal_compression() const { return 1; }

  // Encode. Writes a channel-first f32 latent and its shape --
  // [z, h, w] for an image VAE, [z, T, h, w] for a video one -- and the
  // stage publishes THAT shape rather than one it predicted. False
  // means it warned and produced nothing, which makes the stage emit no
  // beat; it must never throw across this boundary.
  virtual bool encode(const VaeEncodeRequest& req, std::vector<float>* out,
                      std::vector<int>* shape, std::string* err) = 0;

  virtual void release_idle() {}
  virtual std::uint64_t resident_bytes() const { return 0; }
};

// One AUDIO encode: a whole reference soundtrack, in one call.
//
// NOT A STREAM. An audio VAE is causal and compresses time, so chunks
// encoded separately and concatenated are a different tensor from one
// encode of the same samples — the same reason the video encoder takes a
// clip rather than frames. `audio-vae-encode` therefore accumulates
// across beats and calls this ONCE, at drain.
struct AudioVaeEncodeRequest {
  // f32 PLANAR [channels][n_samples], in [-1, 1].
  const float* pcm = nullptr;
  int channels    = 0;
  int n_samples   = 0;
  // What the samples ARE, which need not be what the encoder wants. A
  // family that cannot resample says so and refuses rather than
  // encoding at the wrong rate — a silent rate mismatch is a reference
  // that is the right length and the wrong pitch.
  int sample_rate = 0;
};

// ONE resident audio VAE encoder.
class AudioVaeEncoder {
public:
  virtual ~AudioVaeEncoder() = default;

  // The rate this encoder wants. The stage reports it so a graph can set
  // `audio-to-pcm`'s `output_sample_rate` to match.
  virtual int sample_rate() const = 0;
  // How many channels it takes: 1 or 2. A family that wants stereo and
  // is handed mono should duplicate rather than refuse.
  virtual int channels() const { return 1; }

  // Encode. Writes the latent and ITS shape; the stage publishes that
  // shape rather than one it predicted. A family whose latent feeds
  // `generate-video`'s `ref_audio_rows` should emit the [rows, dim]
  // form that port takes, not its own unpatchified layout.
  virtual bool encode(const AudioVaeEncodeRequest& req,
                      std::vector<float>* out, std::vector<int>* shape,
                      std::string* err) = 0;

  virtual void release_idle() {}
  virtual std::uint64_t resident_bytes() const { return 0; }
};

struct VaeModelCreateArgs {
  std::string root;      // the resolved checkpoint dir
  // resolve_vae_dir(root) -- what the built-ins open. A CONVENIENCE and
  // not the answer: it knows two layouts (`vae/config.json` and
  // `video_vae/`) and returns `root` UNCHANGED for anything else, which
  // is every Comfy-Org pack -- exactly the case this extension point
  // exists for. A family whose VAE is a bare .safetensors resolves it
  // from `root` itself and should expect vae_dir == root.
  std::string vae_dir;
  std::string model_type;                    // models-DB hint, may be ""
  metal_compute::MetalCompute* metal   = nullptr;
  const SessionContextIntf*    session = nullptr;
};

// A VAE FAMILY: process-wide, stateless, one per architecture.
class VaeModelFamily {
public:
  virtual ~VaeModelFamily() = default;

  // The family tag, as it appears in this stage's logs ("ltx-2.5").
  virtual std::string_view tag() const noexcept = 0;

  // Does this checkpoint's VAE belong to this family? Given BOTH the
  // model root and resolve_vae_dir(root), because the two DISAGREE for
  // any pack that keeps its config in a safetensors `__metadata__`
  // instead of a config.json -- which is the case this extension point
  // exists for.
  //
  // Must be CHEAP (one header read) and SURE. It is asked BEFORE the
  // built-in `_class_name` chain, so a loose claim shadows a working
  // built-in path -- claiming a checkpoint that is not yours loads at
  // full cost and decodes nonsense, which is the worst failure here.
  virtual bool claims(const std::string& root, const std::string& vae_dir,
                      const std::string& model_type) const = 0;

  // What loading this decoder will take, for the planning phase that
  // runs before any stage initialises. Empty means the family declines,
  // which makes it invisible to peers sizing themselves. Declare the VAE
  // ONLY: `root` also holds the DiT, which generate-video claims.
  virtual std::vector<ResourceClaim>
  declare_resources(const std::string& /*root*/,
                    const std::string& /*vae_dir*/) const { return {}; }

  // WHICH of a family's VAEs is being asked about. A family that
  // generates a soundtrack ships two, in two files, with two lifetimes
  // -- they are loaded and dropped by different stages, in different
  // phases -- so every question below has to name one.
  enum class Role { kVideo, kAudio };

  // WHERE this family's weights actually live under `root`, when that
  // is not a directory the host can find on its own.
  //
  // The host resolves a VAE by looking for `vae/config.json` and falls
  // back to `root` when there is none -- which is every Comfy-style
  // single-file pack. `root` is then the name it releases, pools and
  // reports a phase release for, and on a repack that is the whole
  // repository: a 5 GB VAE crediting itself with the 39 GB DiT beside
  // it. Naming the file here keeps every one of those operations on the
  // bytes this decoder owns.
  //
  // Empty -- the default -- means "the host's answer is right", which it
  // is for any diffusers layout.
  virtual std::string vae_path(const std::string& /*root*/, Role /*role*/) const
  {
    return {};
  }

  // The same checkpoint in the TOPOLOGICAL plan's terms. See
  // VideoModelFamily::declare_holdings and docs/MODEL-MEMORY.md; a VAE
  // normally has no streaming form, so `floor` stays 0 and is read as
  // `preload`. `releases` and `reclaimable` are the STAGE's policy and
  // are stamped on afterwards.
  virtual std::vector<StageHolding>
  declare_holdings(const std::string& /*root*/, Role /*role*/) const
  {
    return {};
  }

  // Directories whose weights are resident BESIDE this decode, for the
  // stage's idle-unload verdict.
  //
  // The stage's own guess is the DIFFUSERS spelling (root/transformer,
  // root/text_encoder, root/mllm), which sums to ZERO on a Comfy-style
  // pack -- and a zero peer footprint reads as "the box is roomy, keep
  // the VAE resident", which beside a 39 GB DiT is the exact wrong
  // answer. The default here is the whole root: it over-counts by the
  // VAE's own bytes, which is wrong in the SAFE direction.
  virtual std::vector<std::string> idle_peers(const std::string& root) const
  {
    return {root};
  }

  // Build the decoder. Null (after warning through `args.session`)
  // leaves the stage inert rather than taking the pipeline down.
  //
  // Named load_DECODER so that a later load_encoder() for `vae-encode`
  // -- the symmetric stage, which has the same hardcoded `_class_name`
  // chain -- is an additive virtual with a default rather than a rename
  // of the whole interface.
  virtual std::unique_ptr<VaeDecoder>
  load_decoder(const VaeModelCreateArgs& args) = 0;

  // Build the ENCODER, for `vae-encode`. Defaults to declining, so a
  // family that only decodes need not mention it -- and so a family
  // whose encoder is not ported yet leaves the stage on its built-in
  // chain rather than half-claiming it.
  virtual std::unique_ptr<VaeEncoder>
  load_encoder(const VaeModelCreateArgs& /*args*/) { return nullptr; }

  // Build the AUDIO encoder, for `audio-vae-encode` -- a reference
  // soundtrack for a family that conditions on one. Defaults to
  // declining, like the other two.
  virtual std::unique_ptr<AudioVaeEncoder>
  load_audio_encoder(const VaeModelCreateArgs& /*args*/) { return nullptr; }

  // Build the AUDIO decoder, for `audio-vae-decode`. Defaults to
  // declining, so a video-only family need not mention it -- and so a
  // family that generates a soundtrack but has not ported its audio VAE
  // yet leaves the stage inert rather than emitting silence.
  virtual std::unique_ptr<AudioVaeDecoder>
  load_audio_decoder(const VaeModelCreateArgs& /*args*/) { return nullptr; }
};

// Process-wide family set. Same singleton discipline as StageRegistry
// and VideoModelRegistry: a plugin MUST link the host libvpipe shared so
// it registers into THIS instance rather than forking a second one.
class VaeModelRegistry {
public:
  static VaeModelRegistry& get() noexcept;

  // Takes ownership. First-wins on `tag()`: a second family claiming a
  // tag already present is ignored and returns false.
  bool add(std::unique_ptr<VaeModelFamily> f);

  // The first registered family that claims this checkpoint, or null.
  // Order is registration order, which for plugins is load order -- so a
  // deliberately narrow `claims` is a family's own responsibility. Never
  // throws: a family whose `claims` throws is skipped and warned.
  VaeModelFamily* claim_for(const SessionContextIntf* session,
                            const std::string&        root,
                            const std::string&        vae_dir,
                            const std::string&        model_type) const;

  VaeModelFamily* find(std::string_view tag) const noexcept;

  std::vector<VaeModelFamily*> all() const;

private:
  VaeModelRegistry() = default;

  mutable std::mutex                           _mu;
  std::vector<std::unique_ptr<VaeModelFamily>> _families;
};

}

#endif
