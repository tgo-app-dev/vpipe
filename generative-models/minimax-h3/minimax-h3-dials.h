#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_DIALS_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_DIALS_H

// DIALS: training-free steering for MiniMax H3, read as ComfyUI-Dials /
// ComfyUI-MAINodes write and apply them (MATLOWAI/H3-Dials publishes
// them). A dial is ONE vector the width of the DiT's hidden size, made
// from prompt pairs: the mean refined prompt of each pair's B side less
// its A side. Its strength is a signed slider -- negative toward A,
// positive toward B -- and it is added to EVERY prompt row, at one of
// two seams:
//
//   post  after the token refiner's final norm, where the DiT reads the
//         prompt: `diffusion_model.token_refiner.dials_shift` [hidden].
//         The text rows become rows + strength * vec, the sum rounded
//         to the stream's bf16 as ComfyUI's `context + shift` does.
//   pre   into condition_proj's BIAS, before the refiner:
//         `diffusion_model.condition_proj.diff_b` [hidden] -- ComfyUI's
//         own weight-patch spelling, so its stock LoRA loader applies
//         it. The bias becomes bf16(b + strength * vec). The refiner
//         sits between this seam and the DiT, so a pre dial only
//         approximates a post one (about 0.6x the effect a unit).
//
// A ROW is one dial at one strength, in the Dial Rack's JSON (MAINodes'
// `rack`): {"dial": file, "strength": s, "on": true, "solo": false,
// "time": {...}}. Rows without `time` are CONSTANT: applied on every
// forward of every step, the rack's post rows summed in double into one
// shift. A row WITH `time` follows a curve over CLIP time inside one
// generation (MAINodes' timecurve.py): for the first `steps` sampling
// steps the DiT runs twice on the same latent, without and with the
// dial, and the two velocities are blended per latent frame,
//
//     v = v0 + M(t) (v_dial - v0)
//
// M on the video's latent frames (the curve's mean over the pixel
// frames each covers: 1, 4, 4, 4, 4 a group of five) and on the audio's
// (the curve at each audio latent's start, 40 a second) -- the same
// curve ("follow", the default) or none ("protect": the soundtrack is
// the undialled branch's). From step `steps` on the row is off. Rows
// sharing one curve share one branch, their shifts summed.
//
// What MAINodes' rack also does and this does not: hold rows, step
// envelopes, per-block strengths, wired dials, music-driven curves
// (`"type": "audio"`), the de-rope hold map. Each is refused by name.

#include "common/flex-data.h"

#include <string>
#include <utility>
#include <vector>

namespace vpipe {
namespace genai {
namespace minimax_h3 {

// The family a dial file says it was made for, when it says nothing.
inline constexpr const char* kDialFamily = "MiniMax H3";

enum class DialSeam { kPre, kPost };

struct Dial {
  std::string path;      // the file read
  std::string name;      // its metadata's "name", else the file's stem
  std::string family;    // its "model_family", else kDialFamily
  DialSeam    seam = DialSeam::kPost;
  std::vector<float> vec;   // [hidden], the file's values widened
};

// Read a dial file: a safetensors holding exactly ONE tensor,
// `diffusion_model.token_refiner.dials_shift` (post) or
// `diffusion_model.condition_proj.diff_b` (pre) -- `model.diffusion_model.`
// and bare names read too -- BF16, F16 or F32, of `hidden` elements,
// every one finite. A dial made for another family (MOSS-Image's, Krea
// 2's) is refused by name. False, with `err`, otherwise.
bool read_dial(const std::string& file, int hidden, Dial* out,
               std::string* err);

// A row's time curve: MAINodes' `"type": "keys"`.
struct DialCurve {
  // [time, value] keys, times strictly increasing. Held flat before the
  // first and after the last.
  std::vector<std::pair<double, double>> points;
  // Times in seconds, or (false, the default) normalized clip time: 0
  // the first frame, 1 the clip's end.
  bool   seconds = false;
  // Step interpolation (each key's value held until the next), or
  // linear (the default).
  bool   step = false;
  // Linear keys' rate smoothing, 0..1 (0 straight lines): the slope
  // ramps across each key instead of jumping, through every key and
  // without overshoot. Step keys ignore it.
  double smooth = 0.0;
  // The blend steps k: the curve blends over steps [0, k), and the row
  // is off after them.
  int    steps = 2;
  // The soundtrack takes the undialled branch ("protect"), or (false,
  // "follow") the same curve as the video.
  bool   protect_audio = false;

  // The curve's value at clip time `t` seconds, for a clip of
  // `duration` seconds (normalized keys scale by it).
  double at(double t, double duration) const;
  // One string per distinct curve, so rows sharing one share a branch.
  std::string key() const;
};

struct DialRow {
  std::string dial;        // the file, as the row names it
  double      strength = 0.0;
  bool        on = true;
  bool        solo = false;
  bool        timed = false;
  DialCurve   time;
};

// Parse a `dials` value: an array of rows, one row object, or a string
// holding either as JSON (a Dial Rack's `rack`, pasted). The rack's UI
// fields it does not read are ignored; what it cannot do is refused by
// name (hold, envelope, blocks, an `input` row, an audio-driven curve).
// An empty or null value is no rows. False, with `err`, otherwise.
bool parse_dial_rows(const FlexData& v, std::vector<DialRow>* out,
                     std::string* err);

// The rows that apply, by the rack's rule: on, strength not 0, and the
// soloed ones alone when any is soloed.
std::vector<DialRow> active_dial_rows(const std::vector<DialRow>& rows);

// The rows back as the JSON array parse_dial_rows reads.
FlexData dial_rows_to_flex(const std::vector<DialRow>& rows);

// A curve over the clip: M per video latent frame (`latent_frames` of
// them, each the curve's MEAN over the pixel frames it covers --
// 1, 4, 4, 4, 4 a group of five -- at `fps`) and per audio latent
// (`audio_latents`, the curve at j / 40 s; all 0 when the curve protects
// the soundtrack). The clip's duration, for normalized keys, is its
// pixel frames / fps.
void dial_frame_weights(const DialCurve& c, int latent_frames,
                        int audio_latents, double fps,
                        std::vector<float>* video,
                        std::vector<float>* audio);

// What a request's rows come to, with each row's dial read.
struct DialPlan {
  // The constant rows: the pre seam's bias delta and the post seam's
  // shift, [hidden] each, empty where no row uses the seam. More than
  // one row on a seam is summed in double, as the rack sums them.
  std::vector<float> pre, post;
  // One branch per distinct curve.
  struct Group {
    std::string        label;   // its rows' dial names, " + "-joined
    std::vector<float> shift;   // [hidden]
    DialCurve          curve;
  };
  std::vector<Group> groups;
  // One line a row, for the log.
  std::string report;
  bool empty() const { return pre.empty() && post.empty() && groups.empty(); }
};

// `rows` (active ones) with `dials` read for them, one each in order.
// A pre row with a time curve is refused (only the post seam varies
// over time, as in the rack). False, with `err`.
bool compile_dial_plan(const std::vector<DialRow>& rows,
                       const std::vector<Dial>& dials, DialPlan* out,
                       std::string* err);

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
