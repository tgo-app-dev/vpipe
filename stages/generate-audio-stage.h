#ifndef VPIPE_STAGES_GENERATE_AUDIO_STAGE_H
#define VPIPE_STAGES_GENERATE_AUDIO_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::genai {
class MetalYue2Model;
class Tokenizer;
}

namespace vpipe {

// Generate audio from text: a song from a style prompt and lyrics.
//
// The audio counterpart of generate-image / generate-video, for models
// whose whole job is a soundtrack. It emits a LATENT, like they do, and
// `audio-vae-decode` turns it into PCM -- so the VAE is chosen there,
// independently of the generator (YuE2 ships two decoders for one
// latent space).
//
// The model is picked from the checkpoint's config.json. Today that is
// YuE2 (m-a-p/YuE2-3B, model_type "yue2"), which writes a song in three
// passes over one checkpoint:
//
//   1. PLAN    an ABC score (melody, and chords for cot=full), unless the
//              request supplies one or cot=off. Emitted on `score`.
//   2. SONG    one semantic codec token per 25 Hz frame until the model
//              ends the song (or max_seconds). The model decides the
//              length; nothing here asks for one.
//   3. LATENTS 32 midpoint flow-matching steps over a second weight set,
//              to [1, 64, frames] VAE latents at 25 per second.
//
//   iport0  `prompt` FlexData: a string (the LYRICS), or an object with
//           any of "style" (alias "tags"), "lyrics", "abc", "cot",
//           "seed", "cfg_scale"; absent keys fall back to this stage's
//           config. One song per beat. UNWIRED, the stage writes ONE song
//           from its config and ends.
//
//   oport0  `audio_latent` TensorBeat f32 [1, 64, frames], sideband
//           latents_per_second = 25. Wire to audio-vae-decode with a
//           YuE2-Vae checkpoint.
//   oport1  `score` FlexData string: the ABC score the song followed --
//           planned, or the one supplied. Not written for cot=off.
//
// Sampling is the protocol's own per phase (ABC and SONG differ), with
// the common knobs exposed. A sampled song is reproducible for a seed,
// but it is not the reference implementation's song for that seed: the
// flow matching's noise is torch-identical, the token draws are not.
class GenerateAudioStage final : public TypedStage<GenerateAudioStage> {
public:
  static constexpr const char* kTypeName = "generate-audio";

  GenerateAudioStage(const SessionContextIntf* session, std::string id,
                     std::vector<InEdge> iports, FlexData config);
  ~GenerateAudioStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process(RuntimeContext& ctx) override;

  std::vector<ResourceClaim> declare_resources() const override;
  StageMemory declare_memory() const override;

  const StageSpec& spec() const noexcept override;

  // What one song costs beside the weights, as the claims book it:
  //   held       the AR pool and the NAR row scratch -- never given back,
  //              so booked for the whole run
  //   transient  the flow matching's chunk K/V (denoise)
  //   decode     the downstream VAE decode's arena (decode-audio)
  //   pcm        the PCM that decode hands on (decode-audio)
  // The generator books the last two, as generate-video does for its
  // soundtrack: it is the stage that knows how long the song can be.
  struct SongBytes {
    std::size_t held = 0, transient = 0, decode = 0, pcm = 0;
  };

  // Test-only.
  std::uint64_t songs_emitted() const noexcept { return _songs; }
  const std::string& last_score() const noexcept { return _last_score; }
  // The plan's figures, and what the model holds beside its weights now.
  SongBytes planned_song_bytes() const { return song_bytes_(-1, -1, false); }
  std::size_t held_scratch_bytes() const;
  // The claims' labels: one per stage instance, since two claims sharing
  // a label are counted once.
  std::string scratch_label_() const;

private:
  // Config. Defaults live in kSpec.attrs.
  std::string _hf_dir;
  std::string _style, _lyrics, _abc, _cot;
  std::int64_t _seed{};
  double _cfg_scale{};
  int _ode_steps{};
  double _max_seconds{};
  double _abc_temperature{}, _abc_top_p{};
  int _abc_top_k{};
  double _abc_penalty{};
  double _song_temperature{}, _song_top_p{};
  int _song_top_k{};
  double _song_penalty{};
  int _lm_quant_bits = 8;   // lm_quant: 8 = the AR stack w8 in memory
  // The flow matching's acceleration tiers, settled in the ctor.
  bool _i8_gemm{};
  bool _ane_ffn{};
  double _ane_rows{};
  int _ane_layers{};
  bool _sage_attn{};
  int _sage_dense_layers{};
  bool _sol_attn{};
  double _sol_tau{};
  int _sol_dense_layers{};
  int _sol_key_block{};
  int _sol_local_radius{};
  // The plan books the ANE module under this label and grants it here.
  std::string ane_claim_label_() const;

  std::unique_ptr<genai::MetalYue2Model> _model;
  std::unique_ptr<genai::Tokenizer> _tok;
  bool _one_shot_done = false;
  std::uint64_t _songs = 0;
  std::string _last_score;
  // A wired prompt can ask for guidance per song (cfg_scale, cot) where
  // the config did not; the claims are revised up when one does.
  bool _prompt_wired = false;
  SongBytes _booked;   // what the claims say now

  // One song's costs. prefix / frames < 0 take the plan's bounds (the
  // score budget plus style and lyrics; max_seconds or the protocol's
  // 9000 tokens); `guided` adds to what the config already asks for.
  SongBytes song_bytes_(int prefix, int frames, bool guided) const;
  // The longest song the plan books: max_seconds or the protocol's cap.
  int plan_frames_() const;
  // Raise the claims to cover `b` -- never lower them: the pool keeps
  // what it grew to, and a later song may be longer.
  void revise_song_bytes_(const SongBytes& b);
};

}  // namespace vpipe

#endif
