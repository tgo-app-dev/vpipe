#ifndef VPIPE_STAGES_TRAINING_DATASET_STAGE_H
#define VPIPE_STAGES_TRAINING_DATASET_STAGE_H

// training-dataset -- a folder of pictures (and clips) and their
// captions, as the streams a LoRA training graph encodes and trains on.
//
//   oport 2  manifest   ONE FlexData beat, first: every sample, its
//                       caption variants and its (bucket, flip) pictures,
//                       in the order the other ports carry them.
//   oport 1  captions   prompt text for the diffusion-conditioner, one
//                       beat per caption variant, then the EMPTY prompt
//                       (caption dropout trains against its encoding),
//                       then the preview prompts.
//   oport 0  images     planar U8 [C, H, W] for vae-encode, one beat per
//                       (sample, resolution, flip), each already fitted
//                       to its bucket -- or, for a CLIP's window, its
//                       frames [F, 3, H, W] at `fps`, which vae-encode
//                       encodes as one clip.
//   oport 3  audio      a clip window's soundtrack, f32 planar [2, N] at
//                       `audio_sample_rate`, for audio-vae-encode: one
//                       beat per window that has one, in the manifest's
//                       order.
//   oport 4  keyframes  for the diffusion-conditioner's ref_images: ONE
//                       list per caption beat, in step with oport 1 --
//                       empty for a plain caption, a keyframe window's
//                       first / last frames for its presented captions.
//
// KEYFRAME TASKS (`task_mix`, e.g. {t2va: 0.5, i2va: 0.3, fl2va: 0.2}):
// each clip window draws one task by the seed. i2va anchors its first
// frame, l2va its last, fl2va both: the anchors follow the window on
// oport 0 as stills (what vae-encode makes of a keyframe at inference),
// and with oport 4 wired the window's captions are encoded again with
// its keyframes shown -- FL2VA's "<Picture i>: " presentation.
//
// CLIPS (`media`: auto takes .mp4 .mov .mkv .webm .avi .m4v as clips) are
// put on `fps` the way ffmpeg's fps filter does it (frames held and
// dropped whole), cut into windows of a `frames` bucket -- the longest a
// clip fills -- each fitted to its aspect bucket like a picture, with the
// window's own stretch of the soundtrack.
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
  // A clip's decode holds its windows' source frames (and its
  // soundtrack) on the host while they are fitted: booked per clip.
  std::vector<ResourceClaim> declare_resources() const override;

  struct Picture {
    int w = 0, h = 0;      // the bucket
    bool flip = false;
    std::string key;       // the cache key (empty: not cached)
    bool cached = false;
    // A clip window: its frames at `fps` (0: a still) from `start`, and
    // whether it carries the window's soundtrack, keyed apart.
    int frames = 0;
    int start = 0;
    bool audio = false;
    std::string audio_key;
    bool audio_cached = false;
    // KEYFRAME TASKS (MiniMax-H3 FL2VA's i2va / l2va / fl2va): the
    // window's first and / or last frame as stills -- the anchors, each
    // encoded as inference encodes a keyframe -- (bit 0 first, bit 1
    // last), keyed apart; and, when the keyframes are SHOWN to the
    // conditioner, the window's own encodes of its captions presented
    // with them.
    int anchors = 0;
    std::vector<std::string> anchor_keys;
    std::vector<bool> anchor_cached;
    std::vector<std::string> captions;
    std::vector<std::string> caption_keys;
    std::vector<bool> caption_cached;
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
    bool clip = false;                   // a video file
    int clip_frames = 0;                 // its frames at `fps`
    int src_w = 0, src_h = 0;            // as probed: what a decode holds
    double src_fps = 0.0, src_seconds = 0.0;
    // REFERENCES (MiniMax-H3 Ref2VA): files the sample is conditioned on,
    // in the model's reading order -- a `<stem>.refs` sidecar (one path a
    // line) or a metadata.jsonl row's `references`. Absolute; each
    // caption beat names them, and `ref_frames` (the target's frames:
    // the window's, 1 for a picture) as the request's duration.
    std::vector<std::string> references;
    std::string ref_sig;                 // path|size|mtime each, keyed
    int ref_frames = 1;
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
                          const std::set<std::string>& cached,
                          double fps = 24.0, int audio_rate = 32000);
  // Marks `n` samples held out, chosen by `seed`: the same ones every run.
  static void hold_out(std::vector<Sample>* samples, int n,
                       std::uint64_t seed);
  // The held-out count `validation` asks for over `samples` pictures:
  // auto (-1) is 1% of a set of 500 or more, at most 64, and none below.
  static int validation_count(int validation, int samples);
  // ~/Library/Caches/vpipe/train.
  static std::string default_cache_root();
  // The longest of `buckets` a clip of `clip_frames` fills; 0 for none.
  static int clip_bucket(int clip_frames, const std::vector<int>& buckets);
  // Where `count` windows of `window` frames start in a clip of
  // `clip_frames`: spread evenly from the first frame to the last
  // window's, each once.
  static std::vector<int> window_starts(int clip_frames, int window,
                                        int count);
  // The clip's frames at `fps` after the fps filter, from what a probe
  // says of it (`src_frames` at `src_fps`, else `seconds` long).
  static int frames_at_fps(long long src_frames, double src_fps,
                           double seconds, double fps);
  // A task name's anchor bits (t2va 0, i2va 1, l2va 2, fl2va 3); -1 for
  // a name that is not one.
  static int task_anchors(const std::string& task);
  // The task a window draws from `tasks` ((anchor bits, weight)), by its
  // own key: the same every run.
  static int draw_task(const std::vector<std::pair<int, double>>& tasks,
                       std::uint64_t seed, const std::string& window);

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
  std::string _media = "auto";            // auto | pictures | clips
  // task_mix: the modes a clip window draws from, with their weights.
  std::vector<std::pair<int, double>> _tasks;   // (anchor bits, weight)
  bool _show_keyframes = false;           // oport 4 wired this run
  // A clip decode's booked bytes (the largest so far), under
  // decode_label_().
  std::size_t _decode_booked = 0;
  std::string decode_label_() const
  {
    return "training-dataset-decode:" + this->id();
  }
  double _fps = 24.0;
  std::vector<int> _frames;               // clip window buckets
  int _windows = 1;
  bool _audio = true;                     // audio: track | none
  int _audio_rate = 32000;
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
