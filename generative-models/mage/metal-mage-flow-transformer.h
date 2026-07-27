#ifndef GENERATIVE_MODELS_MAGE_METAL_MAGE_FLOW_TRANSFORMER_H
#define GENERATIVE_MODELS_MAGE_METAL_MAGE_FLOW_TRANSFORMER_H

#include "generative-models/qwen-image/metal-qwen-image-transformer.h"

namespace vpipe {
namespace genai {

// Mage-Flow's NR-MMDiT (microsoft/Mage-Flow*, `MageFlow` in the checkpoint)
// is the Qwen-Image dual-stream MMDiT topology, so it is wired onto
// MetalQwenImageTransformer rather than a copy of it.
//
// Verified against the reference in tests/unit-tests/mage-dit.cc: temb
// 0.004, embedders 2e-6, block-0 attention 0.009, block 0 0.009 rel-L2.
//
// Read against the reference (mage_flow/models/modules/mage_layers.py) and
// confirmed numerically:
//   * Block: img_mod/txt_mod -> Linear(dim, 6*dim), chunk(2) into norm1/norm2
//     halves, each chunk(3) into (shift, scale, gate); LayerNorm without
//     affine; joint attention over concat[text, image] with per-head q/k
//     RMSNorm on both streams; GELU (tanh-approximate) FeedForward. Identical
//     to QwenImageTransformerBlock, and the checkpoint tensor NAMES match
//     exactly (transformer_blocks.N.attn.{to,add_}*, img_mod.1,
//     img_mlp.net.{0.proj,2}, img_in/txt_in/txt_norm, time_text_embed.
//     timestep_embedder.linear_{1,2}, norm_out.linear, proj_out).
//   * RoPE: MageFlowEmbedRope == QwenEmbedRope -- 3 axes [16,56,56], theta
//     10000, scale_rope centering (height positions run
//     -(h - h/2) .. h/2 - 1, which is exactly build_rope_'s `r - hoff`), and
//     each packed segment takes its own frame index.
//   * Timestep: Timesteps(256, flip_sin_to_cos=true, downscale_freq_shift=0,
//     scale=1000) -> the same time_proj_ as Qwen-Image. There is no guidance
//     embedder (guidance_embed=false) and no pooled-text vector
//     (vec_in_dim=0; the reference adds a zero txt_vec to temb).
//
// TWO behavioural differences, both carried by Config flags:
//   * Mage-Flow leaves the text stream UNROTATED
//     (apply_text_rotary_emb=false) -> Config::rotate_txt.
//   * Mage-Flow vendors a get_timestep_embedding that rounds the sinusoidal
//     FREQUENCY table to bf16 before forming the angle, and was trained with
//     that rounding -> Config::bf16_time_freqs. Do not "fix" this to fp32:
//     the angle reaches ~750 rad, so the rounding is worth 7.6% of temb and
//     cascades into every block's modulation.
//
// Shape differences from Qwen-Image-Edit are pure config: 12 dual-stream
// blocks (no single-stream tail), context_in_dim 2560 (a single last-hidden
// tap of Qwen3-VL, not a multi-layer concat), and in_channels 128 at
// patch_size 1 -- one token per latent pixel, so the caller feeds the
// MageVAE latent straight in with no 2x2 pack/unpack.
using MetalMageFlowTransformer = MetalQwenImageTransformer;

// The Mage-Flow 4B DiT config (transformer/config.json). Edit and t2i
// checkpoints share it; only the weights differ.
inline MetalQwenImageTransformer::Config
mage_flow_dit_config()
{
  MetalQwenImageTransformer::Config c;
  c.hidden      = 3072;
  c.n_heads     = 24;
  c.head_dim    = 128;
  c.n_layers    = 12;
  c.in_channels = 128;   // latent channels, patch_size 1
  c.txt_dim     = 2560;  // Qwen3-VL hidden
  c.ffn         = 12288; // mlp_ratio 4.0
  c.time_proj   = 256;
  c.norm_eps    = 1e-6f;
  c.rope_theta  = 10000;
  c.axes[0] = 16; c.axes[1] = 56; c.axes[2] = 56;
  c.rotate_txt  = false;
  // Mage-Flow's vendored get_timestep_embedding rounds the frequency table
  // to bf16 and the model was trained that way (see Config).
  c.bf16_time_freqs = true;
  // ...and its forward casts the TIMESTEP to bf16 too (`timesteps.to(
  // img.dtype)`, the model being bf16). Both matter for the same reason: the
  // angle is sigma*1000*freq ~ 950 rad, so a 2e-3 relative error anywhere in
  // it is radians of phase.
  c.bf16_timestep = true;
  return c;
}

}  // namespace genai
}  // namespace vpipe

#endif
