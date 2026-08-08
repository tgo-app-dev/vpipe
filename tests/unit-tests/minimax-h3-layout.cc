// The MiniMax-H3 packed layout and its two sigma schedules, against the
// diffusers reference.
//
// This model's structure IS the layout: the transformer is handed one
// packed sequence plus a per-row modality tag, timestep index and
// (t, h, w) rotary coordinate, and does full self-attention over
// whatever it is given. So a wrong grid or a wrong tag is not a
// numerical wobble -- it is a different model, and it produces plausible
// video rather than an error.
//
// The goldens are EMBEDDED rather than read from a golden dir, and this
// file gates on no env var: it is pure integer/double arithmetic with no
// weights and no GPU, so there is nothing to skip for, and an
// env-gated version of it would report success while testing nothing.
// They were produced by running the reference's own helpers
// (`_spatial_position_grid`, `_temporal_position_grid`,
// `MiniMaxH3PrepareLayoutStep.build_packed_sequence`,
// `MiniMaxH3SetTimestepsStep.build_row_timesteps` and
// `MiniMaxH3Scheduler`) -- not by transcribing them a second time.

#include "minitest.h"

#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/minimax-h3/minimax-h3-scheduler.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vpipe::genai;
namespace h3 = vpipe::genai::minimax_h3;

namespace {

const double kSpatial0[] = {
    3.9051368637047297, 4.9130421250626686, 5.9209473864206075,
    6.9288526477785473, 7.9367579091364862, 8.9446631704944259,
    9.9525684318523648, 10.960473693210304, 11.968378954568243,
    12.976284215926182, 13.98418947728412, 14.992094738642059, 16,
    17.007905261357941, 18.015810522715878, 19.023715784073815,
    20.031621045431756, 21.039526306789696, 22.047431568147633,
    23.055336829505574, 24.063242090863511, 25.071147352221452,
    26.079052613579389, 27.08695787493733
};
const double kSpatial1[] = {
    -5.1660104885167222, -4.1581052271587833, -3.1501999658008444,
    -2.1422947044429055, -1.1343894430849666, -0.12648418172702769,
    0.88142107963091121, 1.8893263409888501, 2.897231602346789,
    3.9051368637047279, 4.9130421250626668, 5.9209473864206057,
    6.9288526477785446, 7.9367579091364835, 8.9446631704944224,
    9.9525684318523613, 10.9604736932103, 11.968378954568241,
    12.976284215926178, 13.984189477284115, 14.992094738642056,
    15.999999999999996, 17.007905261357934, 18.015810522715871,
    19.023715784073811, 20.031621045431752, 21.039526306789689,
    22.047431568147626, 23.055336829505567, 24.063242090863508,
    25.071147352221445, 26.079052613579382, 27.086957874937323,
    28.094863136295263, 29.102768397653204, 30.110673659011137,
    31.118578920369078, 32.126484181727022, 33.134389443084956,
    34.142294704442889, 35.150199965800837, 36.158105227158771
};
const double kSpatial2[] = {
    4.0743041200011216, 6.459443296000897, 8.8445824720006723,
    11.229721648000448, 13.614860824000223, 15.999999999999998,
    18.385139175999775, 20.770278351999551, 23.155417527999326,
    25.540556703999101
};
const int kSpatialDim[]  = {48, 84, 20};
const int kSpatialPatch[]= {2, 2, 2};
const int kSpatialLh[]   = {48, 48, 20};
const int kSpatialLw[]   = {84, 84, 36};

const double kTemporal0[] = {
    0, 1.6666666666666667, 8.3333333333333339, 15, 21.666666666666668,
    28.333333333333336, 30.000000000000004
};
const double kTemporal1[] = {
    37, 38.666666666666664, 45.333333333333336, 52, 58.666666666666671,
    65.333333333333343, 67, 73.666666666666671, 80.333333333333343, 87,
    93.666666666666657, 95.333333333333329
};
const double kTemporal2[] = {
    5, 6.666666666666667, 13.333333333333334, 20, 26.666666666666668,
    33.333333333333336, 35, 41.666666666666671, 48.333333333333336, 55,
    61.666666666666664, 63.333333333333329, 70, 76.666666666666671,
    83.333333333333343, 90.000000000000014, 91.666666666666686
};
const int kTemporalN[]      = {7, 12, 17};
const double kTemporalOrig[]= {0.0, 37.0, 5.0};

const double kPos0[] = {
    0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 4, 0, 16,
    5, 0, 16, 6, 0, 16, 4, 0, 0, 4, 0, 16, 4, 16, 0, 4, 16, 16,
    5.666666666666667, 0, 0, 5.666666666666667, 0, 16, 5.666666666666667, 16,
    0, 5.666666666666667, 16, 16, 12.333333333333334, 0, 0,
    12.333333333333334, 0, 16, 12.333333333333334, 16, 0, 12.333333333333334,
    16, 16, 19, 0, 0, 19, 0, 16, 19, 16, 0, 19, 16, 16, 25.666666666666668,
    0, 0, 25.666666666666668, 0, 16, 25.666666666666668, 16, 0,
    25.666666666666668, 16, 16, 32.333333333333336, 0, 0, 32.333333333333336,
    0, 16, 32.333333333333336, 16, 0, 32.333333333333336, 16, 16, 34, 0, 0,
    34, 0, 16, 34, 16, 0, 34, 16, 16
};
const int kTags0[] = {
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
const int kAdaln0[] = {
    1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
const int kVidIdx0[] = {
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37
};
const float kUniq0[] = {
    0.3125f, 0.5f
};
const double kPos1[] = {
    0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 5, 0, 16, 5, 16, 0,
    5, 16, 16, 5, 0, 0, 6, 0, 0, 7, 0, 0, 5, 0, 16, 6, 0, 16, 7, 0, 16, 5, 0,
    0, 5, 0, 16, 5, 16, 0, 5, 16, 16, 6.666666666666667, 0, 0,
    6.666666666666667, 0, 16, 6.666666666666667, 16, 0, 6.666666666666667,
    16, 16, 13.333333333333334, 0, 0, 13.333333333333334, 0, 16,
    13.333333333333334, 16, 0, 13.333333333333334, 16, 16, 20, 0, 0, 20, 0,
    16, 20, 16, 0, 20, 16, 16, 26.666666666666668, 0, 0, 26.666666666666668,
    0, 16, 26.666666666666668, 16, 0, 26.666666666666668, 16, 16,
    33.333333333333336, 0, 0, 33.333333333333336, 0, 16, 33.333333333333336,
    16, 0, 33.333333333333336, 16, 16, 35, 0, 0, 35, 0, 16, 35, 16, 0, 35,
    16, 16
};
const int kTags1[] = {
    1, 1, 0, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
const int kAdaln1[] = {
    1, 1, 0, 0, 1, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
const int kVidIdx1[] = {
    5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42
};
const float kUniq1[] = {
    0.3125f, 0.5f, 0.75f
};
const double kPos2[] = {
    0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 2.9360547051563817, -3.5959179422654266, 3,
    2.9360547051563817, 9.4680273525781935, 3, 2.9360547051563817,
    22.531972647421814, 3, 16, -3.5959179422654266, 3, 16,
    9.4680273525781935, 3, 16, 22.531972647421814, 31.333333333333332,
    2.9360547051563817, -3.5959179422654266, 31.333333333333332,
    2.9360547051563817, 9.4680273525781935, 31.333333333333332,
    2.9360547051563817, 22.531972647421814, 31.333333333333332, 16,
    -3.5959179422654266, 31.333333333333332, 16, 9.4680273525781935,
    31.333333333333332, 16, 22.531972647421814, 3, 0, -3.5959179422654266, 4,
    0, -3.5959179422654266, 5, 0, -3.5959179422654266, 6, 0,
    -3.5959179422654266, 3, 0, 22.531972647421814, 4, 0, 22.531972647421814,
    5, 0, 22.531972647421814, 6, 0, 22.531972647421814, 3,
    2.9360547051563817, -3.5959179422654266, 3, 2.9360547051563817,
    9.4680273525781935, 3, 2.9360547051563817, 22.531972647421814, 3, 16,
    -3.5959179422654266, 3, 16, 9.4680273525781935, 3, 16,
    22.531972647421814, 4.666666666666667, 2.9360547051563817,
    -3.5959179422654266, 4.666666666666667, 2.9360547051563817,
    9.4680273525781935, 4.666666666666667, 2.9360547051563817,
    22.531972647421814, 4.666666666666667, 16, -3.5959179422654266,
    4.666666666666667, 16, 9.4680273525781935, 4.666666666666667, 16,
    22.531972647421814, 11.333333333333334, 2.9360547051563817,
    -3.5959179422654266, 11.333333333333334, 2.9360547051563817,
    9.4680273525781935, 11.333333333333334, 2.9360547051563817,
    22.531972647421814, 11.333333333333334, 16, -3.5959179422654266,
    11.333333333333334, 16, 9.4680273525781935, 11.333333333333334, 16,
    22.531972647421814, 18, 2.9360547051563817, -3.5959179422654266, 18,
    2.9360547051563817, 9.4680273525781935, 18, 2.9360547051563817,
    22.531972647421814, 18, 16, -3.5959179422654266, 18, 16,
    9.4680273525781935, 18, 16, 22.531972647421814, 24.666666666666668,
    2.9360547051563817, -3.5959179422654266, 24.666666666666668,
    2.9360547051563817, 9.4680273525781935, 24.666666666666668,
    2.9360547051563817, 22.531972647421814, 24.666666666666668, 16,
    -3.5959179422654266, 24.666666666666668, 16, 9.4680273525781935,
    24.666666666666668, 16, 22.531972647421814, 31.333333333333336,
    2.9360547051563817, -3.5959179422654266, 31.333333333333336,
    2.9360547051563817, 9.4680273525781935, 31.333333333333336,
    2.9360547051563817, 22.531972647421814, 31.333333333333336, 16,
    -3.5959179422654266, 31.333333333333336, 16, 9.4680273525781935,
    31.333333333333336, 16, 22.531972647421814
};
const int kTags2[] = {
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
const int kAdaln2[] = {
    1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
const int kVidIdx2[] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58
};
const float kUniq2[] = {
    0.3125f, 0.5f, 0.75f
};

const float kSig0[] = {
    1.0f, 0.986301303f, 0.967741847f, 0.941176414f, 0.900000036f, 0.827586174f,
    0.666666687f, 0.0f
};
const float kSig1[] = {
    1.0f, 0.995391726f, 0.990291297f, 0.984615326f, 0.978260934f, 0.971098244f,
    0.962962985f, 0.953642368f, 0.942857146f, 0.930232525f, 0.915254235f,
    0.897196233f, 0.875f, 0.847058773f, 0.810810804f, 0.761904716f,
    0.692307651f, 0.585365891f, 0.400000006f, 0.0f
};
const float kSig2[] = {
    1.0f, 0.997867823f, 0.995633185f, 0.993288636f, 0.990825653f, 0.988235295f,
    0.985507309f, 0.982630253f, 0.979591846f, 0.976377904f, 0.972972989f,
    0.969359398f, 0.965517223f, 0.961424351f, 0.957055211f, 0.952380955f,
    0.947368383f, 0.941979468f, 0.93617022f, 0.929889321f, 0.923076928f,
    0.915662646f, 0.907562971f, 0.898678422f, 0.888888896f, 0.878048778f,
    0.865979373f, 0.852459013f, 0.837209284f, 0.819875777f, 0.800000012f,
    0.776978374f, 0.75f, 0.717948675f, 0.679245293f, 0.631578922f,
    0.571428597f, 0.493150681f, 0.387096792f, 0.235294133f, 0.0f
};
const float kSig3[] = {
    1.0f, 0.947368383f, 0.882352889f, 0.799999952f, 0.692307711f, 0.545454621f,
    0.333333373f, 0.0f
};
const float kSig4[] = {
    1.0f, 0.981818259f, 0.96226418f, 0.941176414f, 0.918367386f, 0.893616974f,
    0.866666675f, 0.837209284f, 0.804878116f, 0.769230783f, 0.729729772f,
    0.685714245f, 0.636363566f, 0.580645144f, 0.517241418f, 0.444444418f,
    0.359999985f, 0.260869563f, 0.142857149f, 0.0f
};
const float kSig5[] = {
    1.0f, 0.991525471f, 0.982758582f, 0.973684251f, 0.964285672f, 0.954545438f,
    0.944444478f, 0.933962166f, 0.923076987f, 0.911764622f, 0.899999976f,
    0.887755156f, 0.87499994f, 0.861702204f, 0.847826064f, 0.833333313f,
    0.818181813f, 0.802325487f, 0.785714388f, 0.768292665f, 0.75f,
    0.730769217f, 0.710526288f, 0.689189255f, 0.666666687f, 0.642857134f,
    0.617646992f, 0.590909064f, 0.5625f, 0.532258093f, 0.5f, 0.465517223f,
    0.428571463f, 0.388888866f, 0.346153885f, 0.300000012f, 0.25f,
    0.195652187f, 0.13636364f, 0.0714285746f, 0.0f
};
const double kSchedShift[]= {12.0, 12.0, 12.0, 3.0, 3.0, 3.0};
const int kSchedSteps[]   = {8, 20, 41, 8, 20, 41};

const float kTrajX0[] = {
    1.54099607f, -0.293428898f, -2.17878938f, 0.568431258f, -1.08452237f,
    -1.39859545f, 0.403346837f, 0.838026345f, -0.719257593f
};
const float kTrajV[] = {
    -0.403343529f, -0.596635342f, 0.182036489f, -0.856674612f, 1.10060418f,
    -1.07118738f, 0.122701243f, -0.566317499f, 0.373114645f
};
const float kTraj[] = {
    1.53547084f, -0.301602036f, -2.17629552f, 0.556695938f, -1.06944549f,
    -1.41326928f, 0.405027688f, 0.830268562f, -0.714146435f, 1.52798498f,
    -0.312675267f, -2.17291689f, 0.540796518f, -1.04901874f, -1.43314993f,
    0.407304972f, 0.819757998f, -0.707221627f, 1.51726997f, -0.328525156f,
    -2.16808105f, 0.518038571f, -1.01978076f, -1.4616065f, 0.410564601f,
    0.804713547f, -0.697309613f, 1.50066173f, -0.353092462f, -2.1605854f,
    0.482763797f, -0.974461854f, -1.50571418f, 0.415617019f, 0.78139466f,
    -0.681946099f, 1.47145402f, -0.396297127f, -2.14740348f, 0.420728683f,
    -0.894762874f, -1.58328295f, 0.424502283f, 0.740385473f, -0.654927433f,
    1.40654826f, -0.492307365f, -2.11811042f, 0.282873005f, -0.717654169f,
    -1.75565791f, 0.444247305f, 0.649253964f, -0.594886005f, 1.13765264f,
    -0.89006424f, -1.99675274f, -0.288243443f, 0.0160819888f, -2.46978283f,
    0.526048124f, 0.271708965f, -0.346142888f
};
const int kTrajSteps = 7;

// The three packed cases, in the order of the golden arrays above.
struct PackedCase {
  const char*       name;
  std::vector<int>  text_tags;
  int               nlat, lh, lw, naud;
  std::vector<h3::Anchor> anchors;
  const double*     pos;
  const int*        tags;
  const int*        adaln;
  const int*        vidx;
  const float*      uniq;
  int               n_uniq;
};

std::vector<PackedCase>
packed_cases()
{
  using A = h3::Anchor;
  return {
      {"t2va", {1, 1, 1, 1}, 7, 4, 4, 3, {}, kPos0, kTags0, kAdaln0,
       kVidIdx0, kUniq0, (int)(sizeof(kUniq0) / sizeof(float))},
      {"fl2va_first", {1, 1, 0, 0, 1}, 7, 4, 4, 3, {A::kFirst}, kPos1,
       kTags1, kAdaln1, kVidIdx1, kUniq1,
       (int)(sizeof(kUniq1) / sizeof(float))},
      {"fl2va_both", {1, 1, 1}, 6, 4, 6, 4, {A::kFirst, A::kLast}, kPos2,
       kTags2, kAdaln2, kVidIdx2, kUniq2,
       (int)(sizeof(kUniq2) / sizeof(float))},
  };
}

}  // namespace

// ---- geometry ---------------------------------------------------------

TEST(minimax_h3_layout, canvas_resolution)
{
  // {aspect_w, aspect_h, height, width} from the reference. 16:9 at short
  // edge 768 lands exactly on the 768*1344 area cap; 21:9 is the case the
  // cap actually bites on, and it rounds DOWN the short edge to 672.
  const int kCases[][4] = {{16, 9, 768, 1344},   {9, 16, 1344, 768},
                           {4, 3, 768, 1024},    {1, 1, 768, 768},
                           {21, 9, 672, 1536},   {1360, 768, 768, 1344}};
  for (const auto& c : kCases) {
    int h = 0, w = 0;
    ASSERT_TRUE(h3::resolve_canvas_size(c[0], c[1], 32, 768, 768 * 1344, &h,
                                        &w));
    EXPECT_TRUE(h == c[2]);
    EXPECT_TRUE(w == c[3]);
    // Both axes must be multiples of 32 -- the VAE's 16x spatial times the
    // transformer's 2x spatial patch. Anything else does not patchify.
    EXPECT_TRUE(h % 32 == 0 && w % 32 == 0);
  }
  // Outside the trained 1:4 .. 4:1 band, and degenerate input.
  int h = 0, w = 0;
  EXPECT_FALSE(h3::resolve_canvas_size(5, 1, 32, 768, 768 * 1344, &h, &w));
  EXPECT_FALSE(h3::resolve_canvas_size(1, 5, 32, 768, 768 * 1344, &h, &w));
  EXPECT_FALSE(h3::resolve_canvas_size(0, 1, 32, 768, 768 * 1344, &h, &w));
}

TEST(minimax_h3_layout, frame_counts)
{
  // {requested, aligned, latent frames, audio latents}. The video VAE can
  // only encode 17n + 5 frames, so a request is rounded UP -- 96 becomes
  // 107, not 101, because 101 % 17 = 16 rather than 5.
  const int kCases[][4] = {{1, 5, 2, 8},       {96, 107, 32, 178},
                           {100, 107, 32, 178}, {101, 107, 32, 178},
                           {120, 124, 37, 207}, {200, 209, 62, 348},
                           {360, 362, 107, 603}};
  for (const auto& c : kCases) {
    const int aligned = h3::align_num_frames(c[0], 17, 5);
    EXPECT_TRUE(aligned == c[1]);
    EXPECT_TRUE(h3::video_latent_num_frames(aligned, 17, 5) == c[2]);
    EXPECT_TRUE(h3::audio_latent_num_frames(aligned) == c[3]);
  }
  // A frame count off the 17n + 5 grid has no latent count, and saying so
  // is the point: silently rounding here would desynchronise the layout
  // from what the VAE will actually decode.
  EXPECT_TRUE(h3::video_latent_num_frames(100, 17, 5) == 0);
  EXPECT_TRUE(h3::align_num_frames(0, 17, 5) == 0);
}

// ---- the packed sequence at PRODUCTION geometry -----------------------

// The cases above are toy-sized (4x4 latent, 3 audio latents) because
// they embed whole arrays. Production is 16x16 to 34x60 with thousands
// of rows, so this checks ORDER-SENSITIVE INVARIANTS instead: per-axis
// sum/min/max plus a dot product against an index ramp. Sums alone are
// permutation-blind, which is exactly the bug class that would put the
// right values on the wrong rows; the ramp term moves the moment any row
// changes place.
//
// `prod_wide` is the one that earns its keep: aspect normalization only
// does anything when lh != lw, every packed case above is square or
// nearly so, and it is the only case whose width grid goes NEGATIVE
// (left = (1 - ratio)/2 with ratio > 1).
TEST(minimax_h3_layout, packed_sequence_production)
{
  struct ProdCase {
    const char* name;
    int ntext, nlat, lh, lw, naud;
    int seq, video_rows, audio_rows, tags_sum;
    double sum[3], mn[3], mx[3], ramp[3];
  };
  // From the reference `build_packed_sequence`, via
  // h3ref/gen_layout_prod.py.
  const ProdCase kCases[] = {
      {"prod256", 16, 7, 16, 16, 37, 538, 448, 74, 164,
       {16524.0, 6272.0, 7308.0},
       {0.0, 0.0, 0.0},
       {52.0, 28.0, 28.0},
       {5137360.666667, 2041536.0, 2049236.0}},
      {"prod768", 16, 7, 48, 48, 37, 4122, 4032, 74, 164,
       {127628.0, 61824.0, 62958.6666666667},
       {0.0, 0.0, 0.0},
       {52.0, 30.666667, 30.666667},
       {315290662.0, 136352832.0, 130508593.333333}},
      {"prod_wide", 16, 17, 34, 60, 93, 8872, 8670, 186, 388,
       {526072.0, 132577.3751538939, 135421.5956589255},
       {0.0, 0.0, -5.254757},
       {108.0, 26.627379, 35.837773},
       {2935883836.0, 610282642.164032, 602843934.915509}},
  };

  for (const ProdCase& c : kCases) {
    const std::vector<int> text_tags((std::size_t)c.ntext, 1);
    h3::PackedLayout L;
    ASSERT_TRUE(h3::build_packed_sequence(text_tags, c.nlat, c.lh, c.lw,
                                          c.naud, 2, 2, h3::kAudioChannels,
                                          {}, &L));
    EXPECT_TRUE(L.seq_len == c.seq);
    EXPECT_TRUE((int)L.video_indices.size() == c.video_rows);
    EXPECT_TRUE(L.num_audio_rows == c.audio_rows);

    int tags_sum = 0;
    for (int r = 0; r < L.seq_len; ++r) {
      tags_sum += L.token_tags[(std::size_t)r];
    }
    EXPECT_TRUE(tags_sum == c.tags_sum);

    for (int a = 0; a < 3; ++a) {
      double sum = 0.0, ramp = 0.0;
      double mn = 0.0, mx = 0.0;
      for (int r = 0; r < L.seq_len; ++r) {
        const double v = L.position_ids[(std::size_t)r * 3 + (std::size_t)a];
        sum += v;
        ramp += v * (double)r;
        if (r == 0 || v < mn) { mn = v; }
        if (r == 0 || v > mx) { mx = v; }
      }
      // The ramp term is O(1e9) at production sizes, so it is compared
      // RELATIVELY -- an absolute epsilon there would be either vacuous
      // for the big cases or impossible for the small ones.
      const double rel =
          std::fabs(c.ramp[a]) > 1.0
              ? std::fabs(ramp - c.ramp[a]) / std::fabs(c.ramp[a])
              : std::fabs(ramp - c.ramp[a]);
      const bool ok = std::fabs(sum - c.sum[a]) < 1e-6 * (1.0 + std::fabs(sum)) &&
                      std::fabs(mn - c.mn[a]) < 1e-5 &&
                      std::fabs(mx - c.mx[a]) < 1e-5 && rel < 1e-9;
      if (!ok) {
        std::printf("[minimax_h3_layout] %s axis %d: sum %.6f vs %.6f  "
                    "min %.6f vs %.6f  max %.6f vs %.6f  ramp %.6f vs %.6f\n",
                    c.name, a, sum, c.sum[a], mn, c.mn[a], mx, c.mx[a], ramp,
                    c.ramp[a]);
      }
      EXPECT_TRUE(ok);
    }
    std::printf("[minimax_h3_layout] %-9s seq=%5d video=%5d audio=%3d "
                "positions+order match the reference\n",
                c.name, L.seq_len, (int)L.video_indices.size(),
                L.num_audio_rows);
  }
}

// ---- rotary grids -----------------------------------------------------

TEST(minimax_h3_layout, rotary_grids)
{
  const double* kSpatial[] = {kSpatial0, kSpatial1, kSpatial2};
  const std::size_t kSpatialN[] = {sizeof(kSpatial0) / sizeof(double),
                                   sizeof(kSpatial1) / sizeof(double),
                                   sizeof(kSpatial2) / sizeof(double)};
  double worst = 0.0;
  for (int c = 0; c < 3; ++c) {
    const double area =
        std::sqrt((double)kSpatialLh[c] * (double)kSpatialLw[c]);
    const std::vector<double> g =
        h3::spatial_position_grid(kSpatialDim[c], kSpatialPatch[c], area);
    ASSERT_TRUE(g.size() == kSpatialN[c]);
    for (std::size_t i = 0; i < g.size(); ++i) {
      worst = std::max(worst, std::fabs(g[i] - kSpatial[c][i]));
    }
  }
  std::printf("[minimax_h3_layout] spatial grid max abs diff = %.3e\n", worst);
  // Both sides are double arithmetic over the same expression, so this is
  // an equality check with room for the last bit only.
  EXPECT_TRUE(worst < 1e-12);

  const double* kTemporal[] = {kTemporal0, kTemporal1, kTemporal2};
  const std::size_t kTemporalN2[] = {sizeof(kTemporal0) / sizeof(double),
                                     sizeof(kTemporal1) / sizeof(double),
                                     sizeof(kTemporal2) / sizeof(double)};
  double tworst = 0.0;
  for (int c = 0; c < 3; ++c) {
    const std::vector<double> g =
        h3::temporal_position_grid(kTemporalN[c], kTemporalOrig[c]);
    ASSERT_TRUE(g.size() == kTemporalN2[c]);
    for (std::size_t i = 0; i < g.size(); ++i) {
      tworst = std::max(tworst, std::fabs(g[i] - kTemporal[c][i]));
    }
  }
  std::printf("[minimax_h3_layout] temporal grid max abs diff = %.3e\n",
              tworst);
  EXPECT_TRUE(tworst < 1e-12);
  // The spacing is 5/3 * (1, 4, 4, 4, 4) and NOT uniform: the first latent
  // of each group of five is a quarter of the step of the other four. A
  // uniform grid would pass a "monotone increasing" check and be wrong.
  const std::vector<double> g = h3::temporal_position_grid(7, 0.0);
  EXPECT_TRUE(std::fabs((g[1] - g[0]) - 5.0 / 3.0) < 1e-12);
  EXPECT_TRUE(std::fabs((g[2] - g[1]) - 20.0 / 3.0) < 1e-12);
  EXPECT_TRUE(std::fabs((g[6] - g[5]) - 5.0 / 3.0) < 1e-12);
}

// ---- the packed sequence ----------------------------------------------

TEST(minimax_h3_layout, packed_sequence)
{
  for (const PackedCase& c : packed_cases()) {
    h3::PackedLayout L;
    ASSERT_TRUE(h3::build_packed_sequence(c.text_tags, c.nlat, c.lh, c.lw,
                                          c.naud, 2, 2, h3::kAudioChannels,
                                          c.anchors, &L));
    const int rows_per_frame = (c.lh / 2) * (c.lw / 2);
    EXPECT_TRUE(L.num_condition_rows ==
                (int)c.anchors.size() * rows_per_frame);
    EXPECT_TRUE(L.seq_len ==
                (int)c.text_tags.size() + L.num_condition_rows +
                    c.naud * h3::kAudioChannels + c.nlat * rows_per_frame);

    double worst = 0.0;
    for (std::size_t i = 0; i < L.position_ids.size(); ++i) {
      worst = std::max(worst, std::fabs(L.position_ids[i] - c.pos[i]));
    }
    int tag_bad = 0, idx_bad = 0;
    for (int r = 0; r < L.seq_len; ++r) {
      if (L.token_tags[(std::size_t)r] != c.tags[r]) { ++tag_bad; }
    }
    for (std::size_t i = 0; i < L.video_indices.size(); ++i) {
      if (L.video_indices[i] != c.vidx[i]) { ++idx_bad; }
    }

    std::vector<float> uniq;
    std::vector<int>   row_idx;
    h3::build_row_timesteps(L, 0.3125f, 0.5f, 0.75f, &uniq, &row_idx);
    const std::vector<int> adaln = h3::build_adaln_indices(L, row_idx);
    int ada_bad = 0;
    for (int r = 0; r < L.seq_len; ++r) {
      if (adaln[(std::size_t)r] != c.adaln[r]) { ++ada_bad; }
    }
    std::printf("[minimax_h3_layout] %-12s seq=%3d pos diff=%.3e tags=%d "
                "vidx=%d adaln=%d\n",
                c.name, L.seq_len, worst, tag_bad, idx_bad, ada_bad);
    EXPECT_TRUE(worst < 1e-12);
    EXPECT_TRUE(tag_bad == 0);
    EXPECT_TRUE(idx_bad == 0);
    EXPECT_TRUE(ada_bad == 0);
    ASSERT_TRUE((int)uniq.size() == c.n_uniq);
    for (int i = 0; i < c.n_uniq; ++i) { EXPECT_TRUE(uniq[i] == c.uniq[i]); }
  }
}

TEST(minimax_h3_layout, packed_sequence_rejects_bad_geometry)
{
  h3::PackedLayout L;
  const std::vector<int> t = {1, 1};
  // A latent size that is not a multiple of the patch cannot be packed.
  EXPECT_FALSE(h3::build_packed_sequence(t, 4, 5, 4, 2, 2, 2, 2, {}, &L));
  EXPECT_FALSE(h3::build_packed_sequence(t, 4, 4, 5, 2, 2, 2, 2, {}, &L));
  EXPECT_FALSE(h3::build_packed_sequence(t, 0, 4, 4, 2, 2, 2, 2, {}, &L));
}

// ---- the scheduler ----------------------------------------------------

TEST(minimax_h3_sched, sigma_schedules)
{
  const float* kSig[] = {kSig0, kSig1, kSig2, kSig3, kSig4, kSig5};
  const std::size_t kSigN[] = {
      sizeof(kSig0) / sizeof(float), sizeof(kSig1) / sizeof(float),
      sizeof(kSig2) / sizeof(float), sizeof(kSig3) / sizeof(float),
      sizeof(kSig4) / sizeof(float), sizeof(kSig5) / sizeof(float)};
  double worst = 0.0;
  for (int c = 0; c < 6; ++c) {
    MiniMaxH3Scheduler s(kSchedShift[c]);
    ASSERT_TRUE(s.set_timesteps(kSchedSteps[c]));
    ASSERT_TRUE(s.sigmas().size() == kSigN[c]);
    for (std::size_t i = 0; i < kSigN[c]; ++i) {
      worst = std::max(worst, (double)std::fabs(s.sigmas()[i] - kSig[c][i]));
    }
    // t = 1 - sigma, and there is one FEWER timestep than sigma: the
    // terminal sigma gets no model evaluation.
    EXPECT_TRUE(s.timesteps().size() + 1 == s.sigmas().size());
    for (std::size_t i = 0; i < s.timesteps().size(); ++i) {
      EXPECT_TRUE(s.timesteps()[i] == 1.0f - s.sigmas()[i]);
    }
    // Strictly decreasing, starting at 1 and ending at exactly 0.
    EXPECT_TRUE(s.sigmas().front() == 1.0f);
    EXPECT_TRUE(s.sigmas().back() == 0.0f);
    for (std::size_t i = 1; i < s.sigmas().size(); ++i) {
      EXPECT_TRUE(s.sigmas()[i] < s.sigmas()[i - 1]);
    }
  }
  std::printf("[minimax_h3_sched] sigma max abs diff = %.3e over 6 "
              "schedules\n", worst);
  // Five of the six schedules come back BIT-IDENTICAL; the residual is a
  // single float32 ulp (1.19e-7 at sigma = 1) on the sixth, left by
  // torch.linspace's rounding, which has no closed form that reproduces
  // it for every step count. It cannot reach the model: the transformer
  // consumes the timestep as bf16, whose mantissa is 8 bits.
  EXPECT_TRUE(worst < 2e-7);
  // The video and audio shifts must NOT produce the same grid -- the whole
  // point of two schedulers is that the soundtrack denoises on its own
  // curve.
  MiniMaxH3Scheduler v(12.0), a(3.0);
  ASSERT_TRUE(v.set_timesteps(20) && a.set_timesteps(20));
  EXPECT_TRUE(std::fabs(v.sigmas()[10] - a.sigmas()[10]) > 0.05f);
  EXPECT_FALSE(v.set_timesteps(1));
}

TEST(minimax_h3_sched, euler_trajectory)
{
  const std::size_t n = sizeof(kTrajX0) / sizeof(float);
  MiniMaxH3Scheduler s(12.0);
  ASSERT_TRUE(s.set_timesteps(8));
  ASSERT_TRUE(s.num_inference_steps() == kTrajSteps);
  std::vector<float> x(kTrajX0, kTrajX0 + n);
  double worst = 0.0;
  int worst_step = -1;
  for (int i = 0; i < kTrajSteps; ++i) {
    ASSERT_TRUE(s.step(kTrajV, i, x.data(), n));
    // Compared per STEP, not only at the end: a sign error in the x0
    // estimate is partly absorbed by the blend and can look smaller at the
    // last step than it was in the middle.
    double num = 0.0, den = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
      const double d = (double)x[k] - (double)kTraj[(std::size_t)i * n + k];
      num += d * d;
      den += (double)kTraj[(std::size_t)i * n + k] *
             (double)kTraj[(std::size_t)i * n + k];
    }
    const double r = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    if (r > worst) { worst = r; worst_step = i; }
  }
  std::printf("[minimax_h3_sched] trajectory worst per-step rel-L2 = %.3e "
              "(step %d of %d)\n", worst, worst_step, kTrajSteps);
  EXPECT_TRUE(worst < 1e-6);
  EXPECT_FALSE(s.step(kTrajV, kTrajSteps, x.data(), n));
  EXPECT_FALSE(s.step(kTrajV, -1, x.data(), n));
}

// The velocity is DATA-ward, so x0 = x + sigma*v. Pinned on its own
// because it is the one place a port of this model is most likely to
// follow the usual flow-match convention and subtract -- which still
// denoises, just towards the wrong thing.
TEST(minimax_h3_sched, velocity_is_data_ward)
{
  MiniMaxH3Scheduler s(12.0);
  ASSERT_TRUE(s.set_timesteps(8));
  const float sigma = s.sigmas()[0];
  const float ratio = s.sigmas()[1] / s.sigmas()[0];
  float x = 2.0f;
  const float v = 0.5f;
  ASSERT_TRUE(s.step(&v, 0, &x, 1));
  const float x0_plus  = 2.0f + sigma * v;
  const float expect   = ratio * 2.0f + (1.0f - ratio) * x0_plus;
  EXPECT_TRUE(std::fabs(x - expect) < 1e-6f);
  // The minus convention would land somewhere else entirely.
  const float x0_minus = 2.0f - sigma * v;
  EXPECT_TRUE(std::fabs(x - (ratio * 2.0f + (1.0f - ratio) * x0_minus)) >
              1e-3f);
}

// The VAE's spatial tile layout, against the reference's `_split_tiles`.
//
// This is geometry the encoder and decoder must agree on exactly, and
// it is not something a stage can approximate: a tile decoded on its own
// builds its rope from its OWN extent, so shifting a boundary by one
// latent cell changes what the decoder computes rather than just where
// the seam falls. The cases below are the reference's own output for a
// 256-pixel tile with 64 minimum overlap.
TEST(minimax_h3_layout, vae_tile_split)
{
  struct Case {
    int length;
    std::vector<int> start;
    std::vector<int> overlap;
  };
  const std::vector<Case> cases = {
      // Shorter than a tile, and exactly one tile: no overlaps at all.
      {128, {0}, {}},
      {256, {0}, {}},
      {384, {0, 128}, {128}},
      {512, {0, 128, 256}, {128, 128}},
      // 720 needs a fourth tile: three would leave the union short once
      // the minimum overlaps are paid for, which is what the reference's
      // grow loop is for and what a plain ceiling gets wrong.
      {720, {0, 144, 304, 464}, {112, 96, 96}},
      {1024, {0, 192, 384, 576, 768}, {64, 64, 64, 64}},
      {1280, {0, 160, 320, 496, 672, 848, 1024}, {96, 96, 80, 80, 80, 80}},
      {1920,
       {0, 176, 352, 528, 704, 896, 1088, 1280, 1472, 1664},
       {80, 80, 80, 80, 64, 64, 64, 64, 64}},
  };
  for (const Case& c : cases) {
    const auto s = minimax_h3::split_tiles(c.length, 256, 64, 16);
    EXPECT_TRUE(s.start == c.start);
    EXPECT_TRUE(s.overlap == c.overlap);
    if (s.start != c.start || s.overlap != c.overlap) {
      std::printf("[minimax_h3_layout] tile split %d: got %zu tiles\n",
                  c.length, s.start.size());
      continue;
    }
    EXPECT_TRUE(s.length.size() == s.start.size());
    EXPECT_TRUE(s.overlap.size() + 1 == s.start.size());
    // The union must cover the axis exactly -- a layout that overshoots
    // reads past the frame and one that falls short leaves a band the
    // encode never sees.
    EXPECT_TRUE(s.start.back() + s.length.back() == c.length);
    for (std::size_t i = 0; i + 1 < s.start.size(); ++i) {
      // Every overlap is at least the minimum and a whole number of
      // latent cells, so no tile boundary splits one.
      EXPECT_TRUE(s.overlap[i] >= 64);
      EXPECT_TRUE(s.overlap[i] % 16 == 0);
      EXPECT_TRUE(s.start[i] + s.length[i] - s.overlap[i] == s.start[i + 1]);
    }
  }
}

// `audio_channels` here is the STEREO count (2), not the 32-wide audio
// LATENT each of those rows carries. The two are both plausible-looking
// integers on a transformer config, they are both called "channels", and
// swapping them produces a sequence that packs and denoises without
// complaint -- 16x the audio rows, at 16x the cost, decoding as a
// soundtrack whose two "channels" are slices of one 32-way split. So pin
// what the parameter means, since nothing downstream can tell.
TEST(minimax_h3_layout, audio_channels_is_the_stereo_count)
{
  const int naud = 7;
  const std::vector<int> tags(4, h3::kTextTag);
  h3::PackedLayout L;
  ASSERT_TRUE(h3::build_packed_sequence(tags, 2, 4, 4, naud, 2, 2,
                                        h3::kAudioChannels, {}, &L));
  EXPECT_TRUE(h3::kAudioChannels == 2);
  EXPECT_TRUE(L.num_audio_rows == naud * h3::kAudioChannels);

  // Channel-major, and the two channels are told apart POSITIONALLY: they
  // sit at opposite ends of the width grid and share a rotary clock. If
  // the rows were packed frame-major, or both channels landed on the same
  // width, the model would have no way to separate them.
  const double w_left  = L.position_ids[(std::size_t)L.audio_start * 3 + 2];
  const double w_right =
      L.position_ids[(std::size_t)(L.audio_start + naud) * 3 + 2];
  EXPECT_TRUE(w_left != w_right);
  for (int i = 0; i < naud; ++i) {
    const std::size_t l = (std::size_t)(L.audio_start + i) * 3;
    const std::size_t r = (std::size_t)(L.audio_start + naud + i) * 3;
    EXPECT_TRUE(L.position_ids[l + 2] == w_left);
    EXPECT_TRUE(L.position_ids[r + 2] == w_right);
    // Same time coordinate: the two channels are simultaneous.
    EXPECT_TRUE(L.position_ids[l] == L.position_ids[r]);
  }

  // And the latent width really would pack a different sequence.
  h3::PackedLayout L32;
  ASSERT_TRUE(h3::build_packed_sequence(tags, 2, 4, 4, naud, 2, 2, 32, {},
                                        &L32));
  EXPECT_TRUE(L32.num_audio_rows == naud * 32);
  EXPECT_TRUE(L32.seq_len > L.seq_len);
}
