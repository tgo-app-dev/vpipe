#ifndef VPIPE_STAGES_SAMPLER_SPEC_H
#define VPIPE_STAGES_SAMPLER_SPEC_H

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/sampler.h"
#endif

#include <string>

namespace vpipe {

#ifdef VPIPE_BUILD_APPLE_SILICON

// Decode one `sampler-select` spec beat into SamplerParams, logging what was
// resolved. Shared by every generative LLM stage that exposes a sampler iport
// (text-chat / visual-qa / realtime-vqa / audio-transcribe / text-to-speech),
// so they all report and mis-wire identically.
//
// Anything unusable -- a non-FlexData payload, or a DIFFUSION spec from
// `diffusion-sampler-select` -- warns, sets *accepted to false and returns the
// default SamplerParams. Callers MUST keep their existing params when
// !accepted rather than storing the return value: the fallback is not always
// greedy (text-to-speech's audio channel falls back to MOSS's own sampling,
// because greedy audio degenerates into silence), and a rejected beat must
// leave that alone. Returning greedy on rejection is only the convenience for
// the stages whose fallback IS greedy.
//
// Note this cannot be checked by the port's payload type: both select stages
// emit FlexDataPayload, so a swap type-checks and would otherwise parse to
// "every knob default" -- indistinguishable from a working no-op.
//
// `who` is the caller's own log prefix, e.g. "TextChatStage('chat')" or
// "TextToSpeechStage('tts') audio", so one message format serves all of them.
inline genai::SamplerParams
sampler_params_from_beat(const BeatPayloadIntf*    beat,
                         const SessionContextIntf* session,
                         const std::string&        who,
                         bool*                     accepted = nullptr)
{
  if (accepted != nullptr) { *accepted = false; }
  genai::SamplerParams p;             // defaults == greedy (argmax)
  const auto* fd = dynamic_cast<const FlexDataPayload*>(beat);
  if (fd == nullptr) {
    session->warn(fmt("{}: sampler port expects a FlexData spec from a "
                      "sampler-select stage; keeping the current sampler",
                      who));
    return p;
  }
  if (genai::is_diffusion_sampler_spec(fd->data)) {
    session->warn(fmt("{}: sampler port is wired to diffusion-sampler-select, "
                      "whose spec programs the DIFFUSION integrator and "
                      "carries no token knobs; wire sampler-select instead. "
                      "Keeping the current sampler", who));
    return p;
  }
  p = genai::parse_sampler_config(fd->data);
  if (accepted != nullptr) { *accepted = true; }
  const genai::Sampler probe(p);
  session->info(fmt("{}: sampler = {} (temperature {}, top_k {}, top_p {}, "
                    "min_p {}, rep {}, presence {}, seed {})", who,
                    probe.is_argmax() ? "greedy" : "sampled", p.temperature,
                    p.top_k, p.top_p, p.min_p, p.repetition_penalty,
                    p.presence_penalty, p.seed));
  return p;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

}

#endif
