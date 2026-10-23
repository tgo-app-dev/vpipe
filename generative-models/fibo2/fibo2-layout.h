#ifndef GENERATIVE_MODELS_FIBO2_FIBO2_LAYOUT_H
#define GENERATIVE_MODELS_FIBO2_FIBO2_LAYOUT_H

// Fibo 2.0 (Bria; briaai/Fibo-2.0-*): the sequence its DiT runs over, the
// patch packing, the sizes it was trained at, and the edit prompt its
// Qwen3-VL encoder reads. Everything here is the reference's
// (diffusers' BriaFibo2Pipeline / BriaFibo2Transformer2DModel) restated;
// tools/dump_fibo2_golden.py holds the two to each other.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {
namespace genai {
namespace fibo2 {

// The text encoder layers behind the transformer's six text inputs: the
// Perceiver's bundle, then one per injection block. A layer index is
// transformers' `hidden_states[i]` (0 the embeddings, i the output of
// decoder layer i - 1); each layer is scaled to unit RMS per token (no
// weight, f32) and a bundle concatenates its three.
inline constexpr int kBundles = 6;
inline constexpr int kBundleLayers[kBundles][3] = {
    {9, 20, 31}, {5, 16, 27}, {7, 18, 29},
    {9, 20, 31}, {11, 22, 33}, {13, 24, 35}};

// The chat turn a prompt is wrapped in for Qwen3-VL, and where an image
// stands in it.
inline constexpr const char* kVisionMarker =
    "<|vision_start|><|image_pad|><|vision_end|>";
std::string prompt_template(const std::string& text);

// Qwen3-VL sees an image to edit at most this many pixels a side; the
// reference edits up to this many images at once.
inline constexpr int kEncoderImageSide = 768;
inline constexpr int kMaxImages = 5;

// One forward's joint sequence, in the ORDER THIS PORT RUNS IT:
//
//   [ image rows | context image rows ... | gist rows ]
//
// The reference runs [gist | image | context]. Attention is bidirectional
// and every row carries its position in its rotary angles, so the order
// changes nothing but summation order (the reference says the same of its
// own reordering under padding). This one makes the NOISY rows -- the
// image -- one prefix, and the noise refiner's rows (image and context) a
// prefix of the main stack's.
//
// Rotary coordinates (t, h, w): gist k at (1 + k, 0, 0); the image on
// plane max_gist + 1; context image k on plane max_gist + 2 + k; each
// image's rows and columns from 0.
struct Layout {
  int grid_h = 0, grid_w = 0;          // the image's patch grid
  int img_tokens = 0;
  std::vector<std::pair<int, int>> ctx_grids;   // (rows, cols) per context
  int ctx_tokens = 0;
  int gist = 0;
  int refiner_len() const { return img_tokens + ctx_tokens; }
  int joint_len() const { return img_tokens + ctx_tokens + gist; }
  int gist_off() const { return img_tokens + ctx_tokens; }
  std::vector<std::int32_t> pos_t, pos_h, pos_w;
};

// `max_gist`: the transformer's max_num_gist_tokens, which places the
// image plane. False (with *err) on a degenerate geometry.
bool build_layout(int grid_h, int grid_w,
                  const std::vector<std::pair<int, int>>& ctx_grids,
                  int gist, int max_gist, Layout* out, std::string* err);

// The gist count for a text of `text_len` tokens: lengths between
// min_len and max_len map linearly onto [min_gist, max_gist], rounded UP
// to a multiple of `step` (compute_gist_count).
int gist_count(int text_len, int min_gist, int max_gist, int step,
               int min_len, int max_len);

// A latent [C, H, W] (f32) as patch rows [(H/p)(W/p), C p p], features
// in (channel, dy, dx) order -- this model's training order, NOT Z-Image's
// channel-fastest one -- and back.
void pack_latents(const float* z, int channels, int lat_h, int lat_w,
                  int patch, float* rows);
void unpack_latents(const float* rows, int channels, int lat_h, int lat_w,
                    int patch, float* z);

// The (width, height) the model was trained at for an aspect ratio: of
// the sizes divisible by `multiple`, at least 4 of it a side and within
// 10% of `area`, the one cropping least from a picture of that aspect,
// then the one closest to `area`, then the wider (get_image_size).
std::pair<int, int> image_size(double aspect, long long area = 1024 * 1024,
                               int multiple = 16);

// The edit prompt Qwen3-VL reads, from a user prompt and the number of
// images to edit (encode_prompt's _get_edit_prompt): one compact JSON
// object -- a vision marker per image ("image", or "image_1".. for more
// than one), the caption of the result when the prompt carries one
// ("structured_caption"), then "edit_instruction". A prompt that is not
// a JSON object is the instruction alone. A JSON prompt's caption is
// cleaned as training cleaned captions (empty values and the score
// fields dropped). `text_images`: mark each text_render entry that has
// text with an "image_tokens" vision marker (its rendered text image
// follows the images to edit). False (with *err) on a JSON prompt that
// is not a valid edit prompt.
bool edit_prompt(const std::string& prompt, int n_images, bool text_images,
                 std::string* out, std::string* err);

// The prompt Qwen3-VL reads for a GENERATION (no image to edit): a JSON
// caption in the form training saw it -- cleaned, serialized compactly
// -- and any other prompt as given.
std::string generation_prompt(const std::string& prompt);

// The NEGATIVE a guided run (Fibo 2.0 Dev) uses when none is given: the
// null caption the model was trained with for classifier-free guidance,
// every key of the structured caption with every value empty
// (DEFAULT_NEGATIVE_PROMPT). A negative is encoded AS GIVEN -- neither
// cleaned like a generation's caption nor shown the pictures an edit
// edits -- which is how the reference's encode_prompt takes it.
inline constexpr const char* kNullCaption =
    "{\"short_description\":\"\",\"objects\":[],\"background_setting\":"
    "\"\",\"lighting\":{},\"aesthetics\":{},\"photographic_"
    "characteristics\":{},\"style_medium\":\"\",\"context\":\"\","
    "\"artistic_style\":\"\"}";

// The texts of the caption's text_render entries that have text, in
// order -- the images a text-image renderer draws. Empty for a prompt
// without a JSON caption.
std::vector<std::string> caption_texts(const std::string& prompt);

}  // namespace fibo2
}  // namespace genai
}  // namespace vpipe

#endif
