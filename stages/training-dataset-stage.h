#ifndef VPIPE_STAGES_TRAINING_DATASET_STAGE_H
#define VPIPE_STAGES_TRAINING_DATASET_STAGE_H

// training-dataset -- a folder of pictures and their captions, as the
// three streams a LoRA training graph encodes and trains on.
//
//   oport 2  manifest   ONE FlexData beat, first: every sample, its
//                       caption variants and its (bucket, flip) pictures,
//                       in the order the other two ports carry them.
//   oport 1  captions   prompt text for the diffusion-conditioner, one
//                       beat per caption variant, then the EMPTY prompt
//                       (caption dropout trains against its encoding),
//                       then the preview prompts.
//   oport 0  images     planar U8 [C, H, W] for vae-encode, one beat per
//                       (sample, resolution, flip), each already fitted
//                       to its bucket.
////   iport 0  model      OPTIONAL model-select source. The encoded dataset
//                       is cached per model, so it names which.
//
// THE CACHE. With a model wired, every caption and picture has a key --
// the caption's final text; the picture's file, size, mtime, bucket, flip
// and alpha mode -- and the encodes a previous run left in cache_dir are
// not emitted again: the manifest marks them cached and train-lora reads
// them back. A re-run over an unchanged folder encodes nothing (the
// conditioner and the VAE never load), and an edited picture re-encodes
// alone.
//
// Every beat is marked as part of a series (beat::kBatch), so the
// conditioner and the VAE hold their models across it and release them at
// end of stream, instead of after every beat as a generation graph wants.
//
// The trainer pairs the streams by the manifest's counts and refuses a
// mismatch naming both stages: a caption paired with the wrong picture is
// a wrong answer nothing downstream can detect.

#include "common/ffmpeg-libraries.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace vpipe {

class TrainingDatasetStage final : public TypedStage<TrainingDatasetStage> {
public:
  static constexpr const char* kTypeName = "training-dataset";

  TrainingDatasetStage(const SessionContextIntf* session, std::string id,
                       std::vector<InEdge> iports, FlexData config);
  ~TrainingDatasetStage() override;

  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;
  const StageSpec& spec() const noexcept override;

  struct Picture {
    int w = 0, h = 0;      // the bucket
    bool flip = false;
    std::string key;       // the cache key (empty: not cached)
    bool cached = false;
  };
  struct Sample {
    std::string path;      // absolute
    std::string rel;       // relative to the folder, for the manifest
    std::uint64_t size = 0;
    std::int64_t mtime = 0;
    int repeats = 1;
    std::vector<std::string> captions;   // variants, trigger applied
    std::vector<std::string> caption_keys;
    std::vector<bool> caption_cached;
    std::vector<Picture> pictures;
    bool validation = false;             // held out, scored, not trained
  };

  // Scan the folder into samples (public for tests). False, with the
  // reason, when there is nothing to train on.
  bool scan(std::vector<Sample>* out, std::string* err) const;
  FlexData manifest(const std::vector<Sample>& samples) const;

  // The bucket for a w x h picture at `resolution` (the side of a square
  // of the same area), with sides multiples of `step` and the aspect
  // clamped to max_aspect. Public for tests.
  static void bucket(int w, int h, int resolution, int step,
                     double max_aspect, int* bw, int* bh);
  // `caption` with the trigger applied by `tmpl` ({trigger}, {caption}):
  // unchanged when it already names the trigger as a word, and a
  // `[trigger]` placeholder replaced in place. Public for tests.
  static std::string apply_trigger(const std::string& caption,
                                   const std::string& trigger,
                                   const std::string& tmpl);
  // Keys every caption and picture and marks what `cached` holds. Public
  // for tests.
  static void key_samples(std::vector<Sample>* samples, bool keep_alpha,
                          const std::set<std::string>& cached);
  // Marks `n` samples held out, chosen by `seed`: the same ones every run.
  static void hold_out(std::vector<Sample>* samples, int n,
                       std::uint64_t seed);
  // The held-out count `validation` asks for over `samples` pictures:
  // auto (-1) is 1% of a set of 500 or more, at most 64, and none below.
  static int validation_count(int validation, int samples);
  // ~/Library/Caches/vpipe/train.
  static std::string default_cache_root();

  void apply_constant(unsigned iport, const FlexData& beat) override;

private:
  std::string _dir;
  std::string _captions = "sidecar";      // sidecar | jsonl | none
  std::string _caption_ext = ".txt";
  std::string _trigger;
  std::string _template = "{trigger}, {caption}";
  std::string _missing = "trigger_only";  // trigger_only | skip | error
  bool _shuffle_tags = false;
  int _keep_tags = 1;
  int _variants = 0;                      // 0: auto
  std::vector<int> _resolutions;
  int _bucket_step = 32;
  double _max_aspect = 2.0;
  bool _flip = false;
  bool _keep_alpha = false;
  std::vector<std::string> _preview;
  std::uint64_t _seed = 0;
  std::string _cache_cfg = "auto";        // auto | none | a directory
  int _validation = -1;                   // -1: auto
  std::string _hf_dir;                    // from the model port

  // The run's cache: where, and the keys of what it does not re-emit.
  std::string _cache_dir;
  std::string _null_key;
  bool _null_cached = false;
  std::vector<std::string> _preview_keys;
  std::vector<bool> _preview_cached;
  bool _model_latched = false;

  const FFmpegLibraries* _libs = nullptr;
  // Emission state.
  bool _started = false;
  std::vector<Sample> _samples;
  std::size_t _next = 0;
  int _tail = 0;           // 0: samples; 1: the empty prompt + previews
};

}  // namespace vpipe

#endif
