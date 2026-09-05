#ifndef VPIPE_STAGES_MINIMAX_H3_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_MINIMAX_H3_MODEL_CONFIG_STAGE_H

#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source stage: the MINIMAX-H3-specific half of a `generate-video`
// request.
//
// H3 generates video AND audio from one packed sequence, which is where
// all of its knobs come from and why none of them means anything to a
// video-only family. The two modalities run their own sigma schedules
// (video shift 12, audio shift 3) stepped in lockstep; the pinned
// CONDITION rows -- keyframe anchors, and on Ref2VA the reference
// soundtracks -- sit at a timestep of their own rather than being
// denoised; and the soundtrack has a duration, normally the video's but
// not necessarily. H3 is guidance-DISTILLED, so unlike Wan it has no
// guidance scale at all: there is no unconditional pass to blend with.
//
// The beat is a FlexData object tagged `generate-video-config`; the
// consuming stage passes it to
// MetalMiniMaxH3Transformer::GenerationParams unread, so a knob added
// there needs no change in `generate-video`. See
// stages/model-config-source.h for the contract.
//
// With no trigger iport wired it emits once and signals done. With one
// wired, each inbound beat re-emits (any payload -- receipt is the
// signal); EOS upstream ends the stage.
//
// Configuration (FlexData object):
//   video_shift  (real) -- sigma shift for the VIDEO schedule (12.0)
//   audio_shift  (real) -- sigma shift for the AUDIO schedule (3.0)
//   condition_timestep       (real) -- the level the pinned keyframe
//                              rows are conditioned at; 1.0 is CLEAN in
//                              this model's t = 1 - sigma convention
//   condition_audio_timestep (real) -- the same for a Ref2VA reference
//                              SOUNDTRACK's rows
//   audio_seconds (real) -- soundtrack duration; 0 derives it from the
//                              video's frames / fps, which is what keeps
//                              the two modalities the same length
//   linear_branch (string) -- VDN-H3 release root; turns every main
//                              block's attention into the hybrid. A
//                              SECOND checkpoint beside the DiT, not a
//                              replacement for it
//   lora / lora2 (+ _scale, _qkv_layout) -- TWO runtime LoRA slots,
//                              applied together: the usual pairing is a
//                              few-step Turbo distillation and a style
//                              or identity adapter, with independent,
//                              live strengths
class MiniMaxH3ModelConfigStage final
  : public ModelConfigSourceStage<MiniMaxH3ModelConfigStage> {
public:
  static constexpr const char* kTypeName = "minimax-h3-model-config";

  MiniMaxH3ModelConfigStage(const SessionContextIntf* session,
                            std::string               id,
                            std::vector<InEdge>       iports,
                            FlexData                  config);

  const StageSpec& spec() const noexcept override;

  // The beat this stage emits. Public so a test can read it without
  // running a pipeline, and so the shape stays one expression.
  FlexData resolved_config() const;
  // The default log line would print the object; this one reads the
  // audio duration out loud, which "0" does not.
  void report_config(const FlexData& fd) const;

private:
  double _video_shift   = 12.0;
  double _audio_shift   = 3.0;
  double _cond_timestep = 1.0;
  double _cond_audio_timestep = 1.0;
  double _audio_seconds = 0.0;
  // The VDN-H3 release root, or empty for the stock attention.
  std::string _linear_branch;
  std::string _lora;
  double _lora_scale = 1.0;
  std::string _lora2;
  double _lora2_scale = 1.0;
  std::string _lora2_qkv;
  // "auto" | "flat" | "per_head"; empty keeps the beat silent. See the
  // key's doc -- it decides whether a FUSED qkv adapter's rows are
  // permuted into a per-head DiT's order.
  std::string _lora_qkv;
};

}  // namespace vpipe

#endif
