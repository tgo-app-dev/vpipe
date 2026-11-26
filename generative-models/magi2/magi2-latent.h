#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_LATENT_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_LATENT_H

// The Wan2.2 latent's per-channel statistics, as the release whitens it
// (Wan2_2_VAE: (mu - mean) / std) and its turbo decoder un-whitens it
// (z * std + mean). One table: the two are the same constants.

namespace vpipe {
namespace genai {
namespace magi2 {

inline constexpr float kLatentMean[48] = {
    -0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f,  0.1838f,
    0.1557f,  -0.1382f, 0.0542f,  0.2813f,  0.0891f,  0.1570f,  -0.0098f,
    0.0375f,  -0.1825f, -0.2246f, -0.1207f, -0.0698f, 0.5109f,  0.2665f,
    -0.2108f, -0.2158f, 0.2502f,  -0.2055f, -0.0322f, 0.1109f,  0.1567f,
    -0.0729f, 0.0899f,  -0.2799f, -0.1230f, -0.0313f, -0.1649f, 0.0117f,
    0.0723f,  -0.2839f, -0.2083f, -0.0520f, 0.3748f,  0.0152f,  0.1957f,
    0.1433f,  -0.2944f, 0.3573f,  -0.0548f, -0.1681f, -0.0667f};
inline constexpr float kLatentStd[48] = {
    0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.4990f, 0.4818f, 0.5013f,
    0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f, 0.4978f,
    0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f,
    0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f, 0.6866f, 0.4093f,
    0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f,
    0.3971f, 1.0600f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f};

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_LATENT_H
