#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_LORA_TRAINER_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_LORA_TRAINER_H

// MiniMax-H3 behind train::LoraTrainer: the model, its trainer
// (minimax-h3-trainer.h), and what the family knows that the stage does
// not -- how a sample becomes the packed sequence generate-video would
// build for the same request, how a latent and a soundtrack are packed,
// how the audio's sigma follows the video's, and how a preview runs.
//
// A SAMPLE IS A T2VA REQUEST. A picture is one latent frame (the video
// VAE's still encode, the first latent of a clip); a clip is its 5n + 2
// latent frames. The audio rows are always there, as inference lays them
// out: audio_latent_num_frames of the sample's pixel frames, stereo
// channel-major -- the soundtrack's when the sample has one, else zeros
// noised at the audio's sigma and kept out of the loss (no soundtrack is
// not silence: training silence would teach the adapter to emit it).
//
// ONE u, TWO SHIFTS. The stage draws one sigma; that is the video's, and
// the audio's is the same u through the audio's shift: u = s / (12 -
// 11 s), s_a = 3u / (1 + 2u) at the release's (12, 3) -- how inference
// ties the two schedules, and what the community trainers do.
// schedule_mu() is ln 12: the stage's logit-normal draw pushed through
// the exponential shift at mu = ln 12 IS H3's linear shift 12.
//
// A preview is generate-video's loop (denoise()) with the live adapter:
// the base model's own steps at shift (12, 3), guidance-distilled (no
// CFG), from the preview seed's noise.
//
// KEYFRAMES (FL2VA's i2va / l2va / fl2va): a sample's anchors are the
// conditioning video rows of generate-video's keyframe layout, at the
// keyframe level, and a caption presented with them carries the vision
// block's row tags.
//
// REFERENCES (Ref2VA): a caption video-ref-encoder encoded with a
// sample's references carries the request's plan on its sideband (each
// reference's kind and latent geometry, the presentation's row tags),
// and its reference rows beside it -- the layout is generate-video's
// reference layout for the same request, the rows its conditioning rows
// as they arrive (at the encoder's level; clean on the release).

#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/minimax-h3/minimax-h3-trainer.h"
#include "generative-models/train/lora-trainer.h"

#include <cstdint>
#include <memory>
#include <string>

namespace vpipe {
namespace genai {

class WeightSet;

class MiniMaxH3LoraTrainer final : public train::LoraTrainer {
 public:
  // Which H3 model: the video DiT (T2VA / FL2VA / Ref2VA), P-H3-Image
  // (FL2VA under PrunaAI's adapter, one frame, its shift-1 grid) or
  // MiniMax-H3-Image (audio-less, one output head a step).
  enum class Variant { kVideo, kPh3, kH3i };

  struct LoadArgs {
    std::shared_ptr<WeightSet> ws;         // the DiT's weights
    metal_compute::MetalCompute* mc = nullptr;
    MetalMiniMaxH3Transformer::Config cfg; // dims, partition, prune_adaln
    bool stream_blocks = false;
    std::string partition;                 // "fl2va" | "ref2va" | ""
    Variant variant = Variant::kVideo;
    // Frozen adapters in the slots below the trained one (P-H3-Image's,
    // a style the new one stacks on), bound as the model's own LoRAs;
    // their dX is in every step's backward.
    std::vector<MetalMiniMaxH3Transformer::LoraSpec> base_loras;
    // P-H3-Image: its grid as run. MiniMax-H3-Image: the release's grid
    // as written and its per-step heads' weights.
    std::vector<float> ph3_sigmas;
    std::vector<double> h3i_sigmas;
    std::shared_ptr<WeightSet> heads_ws;
    // The run's widest sample (sequence_rows) and longest caption: the
    // forward's scratch is built once at them (set_scratch_floor), not
    // again at every change of aspect or caption length.
    int floor_seq = 0, floor_text = 0;
  };
  static std::unique_ptr<MiniMaxH3LoraTrainer> load(const LoadArgs& a,
                                                    std::string* err);

  // The audio's sigma for the video's: one u through both shifts.
  static double audio_sigma(double sigma_video, double video_shift = 12.0,
                            double audio_shift = 3.0);
  // A clip's pixel frames for its latent frames (1 for a picture, 17n + 5
  // for 5n + 2); 0 for a count the video VAE cannot produce.
  static int pixel_frames(int latent_frames);
  // The sequence a sample of this geometry packs: text, video and audio
  // rows (no conditioning rows in v1).
  static int sequence_rows(const MetalMiniMaxH3Transformer::Config& c,
                           int frames, int grid_h, int grid_w,
                           int text_rows, int anchors = 0,
                           int cond_rows = 0);
  // What a step holds at that geometry: the trainer's arena and the
  // tap forward's own scratch, `widen` the checkpoint's widest narrow
  // projection (MetalMiniMaxH3Transformer::fp8_widen_bytes).
  static std::size_t plan_arena_bytes(
      const MetalMiniMaxH3Transformer::Config& c, int frames, int grid_h,
      int grid_w, int text_rows, int rank, std::size_t widen = 0,
      int anchors = 0, int cond_rows = 0, bool keep_all = false);

  const char* family() const noexcept override { return "minimax-h3"; }
  std::vector<train::ModuleShape>
  modules(const std::string& targets) const override;
  int bind(train::LoraParams* params, std::string* err) override;
  metal_compute::SharedBuffer pack_latent(const float* cthw, int c, int t,
                                          int h, int w,
                                          std::string* err) override;
  int latent_channels() const noexcept override;
  metal_compute::SharedBuffer pack_audio(const float* audio, int stereo,
                                         int channels, int stride,
                                         int latents,
                                         std::string* err) override;
  std::vector<std::int64_t> latent_shape(int t, int h, int w) const override;
  int audio_latents(int frames) const override;
  bool step(const train::TrainSample& s, train::StepStats* out,
            std::string* err) override;
  bool eval_loss(const train::TrainSample& s, double* loss,
                 std::string* err) override;
  double schedule_mu(int grid_h, int grid_w) const noexcept override;
  bool preview(const train::PreviewRequest& r, std::vector<float>* chw,
               std::string* err) override;
  void set_stop(std::function<bool()> stop) override;
  void set_options(const FlexData& o) override;
  std::size_t arena_bytes(int frames, int grid_h, int grid_w, int text_rows,
                          int rank, bool keep_all) const override;
  std::uint64_t resident_bytes() const override;
  // Guidance-distilled: no unconditional pass to keep.
  bool trains_unconditional() const override { return false; }
  std::map<std::string, std::string> metadata() const override;
  std::string adapter_model_type() const override
  {
    return "minimax-h3-lora";
  }
  std::string parent_model_type() const override;
  // Only Ref2VA trains otherwise (its packed sequence, its references):
  // FL2VA and a checkpoint that names no partition train alike, so one
  // machine's "" and another's "fl2va" for the same DiT agree.
  std::string layout_type() const override
  {
    return layout_type_of(_variant, _partition);
  }
  static std::string layout_type_of(Variant v, const std::string& partition);
  // A rank (LoadArgs::cfg.tp): its share's cuts, its channel, bulk bytes.
  bool split() const override { return _m->config().tp.active(); }
  bool split_cuts(const std::vector<train::ModuleShape>& mods,
                  std::map<std::string, train::ModuleCut>* out,
                  std::string* err) const override
  {
    return _t->split_cuts(mods, out, err);
  }
  bool split_begin(std::shared_ptr<TpChannel> ch, std::uint64_t first_seq,
                   std::string* err) override
  {
    if (!_m->begin_tp_generation(std::move(ch), err, first_seq)) {
      return false;
    }
    _m->set_tp_meet(true);   // a rank's tuning waited out on the CPU
    return true;
  }
  void split_end() override { _m->end_tp_generation(); }
  std::size_t split_exchange_bytes() const override { return _split_bytes; }
  bool split_bulk(const void* mine, void* others, std::size_t bytes,
                  TpDType dtype, std::string* err) override
  {
    return _m->tp_bulk(mine, others, bytes, dtype, err);
  }
  Variant variant() const noexcept { return _variant; }

  MetalMiniMaxH3Transformer* model() const noexcept { return _m.get(); }
  MiniMaxH3Trainer* trainer() const noexcept { return _t.get(); }
  // The packed sequence a sample trains on (tests: it is generate-
  // video's for the same request).
  bool layout_of(const train::TrainSample& s, minimax_h3::PackedLayout* L,
                 std::string* err) const
  {
    int alat = -1;
    if (s.audio_x0 != nullptr) {
      alat = (int)(s.audio_x0->byte_size() /
                   ((std::size_t)minimax_h3::kAudioChannels *
                    _m->config().audio_channels * 2));
    }
    return layout_(s.sideband, s.text_rows, s.frames, s.grid_h, s.grid_w,
                   alat, s.anchors, L, err, nullptr);
  }
  // The last step's two losses, unweighted (the metrics' split).
  double last_loss_video() const noexcept { return _lv; }
  double last_loss_audio() const noexcept { return _la; }

 private:
  MiniMaxH3LoraTrainer() = default;
  // The packed sequence of a T2VA sample (generate-video's builder for
  // the partition), the soundtrack's `audio_latents` (-1: the frames').
  // `refs` (when given) says whether the layout packs references.
  bool layout_(const FlexData* sideband, int text_rows, int frames, int gh,
               int gw, int audio_latents, int anchors,
               minimax_h3::PackedLayout* L, std::string* err,
               bool* refs) const;
  // The trainer's sample, the missing soundtrack's rows filled here.
  bool sample_(const train::TrainSample& s, minimax_h3::PackedLayout* L,
               MiniMaxH3Trainer::Sample* q, std::string* err);

  // MiniMax-H3-Image: the release step a sample trains (its sigma as run
  // and its head), drawn by the sample's seed.
  int h3i_step_(std::uint64_t seed) const;

  metal_compute::MetalCompute* _mc = nullptr;
  std::string _partition;
  Variant _variant = Variant::kVideo;
  std::vector<std::string> _base_loras;
  std::vector<float> _ph3_sigmas;
  std::vector<float> _h3i_run;            // grid_as_run
  double _vshift = 12.0, _ashift = 3.0;
  // The model before its trainer: the trainer unbinds as it goes.
  std::size_t _split_bytes = 0;         // split_exchange_bytes
  std::unique_ptr<MetalMiniMaxH3Transformer> _m;
  std::unique_ptr<MiniMaxH3Trainer> _t;
  std::function<bool()> _stop;
  // A missing soundtrack's rows: zeros, and noise from the sample's seed.
  metal_compute::SharedBuffer _azero, _anoise;
  // The keyframes' rows at their level, where it is below clean.
  metal_compute::SharedBuffer _cond;
  double _lv = 0.0, _la = 0.0;
};

}  // namespace genai
}  // namespace vpipe

#endif
