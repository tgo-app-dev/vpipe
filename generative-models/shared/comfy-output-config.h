#ifndef GENERATIVE_MODELS_SHARED_COMFY_OUTPUT_CONFIG_H
#define GENERATIVE_MODELS_SHARED_COMFY_OUTPUT_CONFIG_H

#include <string>

namespace vpipe {

class FlexData;

namespace genai {

// The config.json to write beside a pass whose INPUT was a Comfy-Org
// single-file component and whose OUTPUT is a directory.
//
// A repack has no config.json -- each component's config lives in its
// own safetensors `__metadata__`, and for the text encoder not even that
// (only the tap; the geometry is in the tensor shapes). Any pass that
// reads one file and writes a directory of shards has to synthesize one,
// because from there on it is an ordinary directory checkpoint and every
// loader in this tree reads config.json.
//
// Shared between the quantizer and the LoRA fuse rather than copied,
// because what it carries across is not boilerplate: `qkv_per_head`
// records that Comfy-Org reordered the fused qkv projection, and
// `_minimax_h3_partition` records which of the two TASKS the weights
// are. Both are invisible in the tensors -- same names, same shapes --
// and both lived only in the source FILENAME, which a directory output
// destroys. A pass that forgets either produces a checkpoint that loads
// and computes nonsense, so there should be exactly one implementation
// to forget it in.
//
// `file` is the source component. False + *err when it is not a
// Comfy-Org component this can describe.
bool comfy_output_config(const std::string& file, FlexData& out,
                         std::string* err);

}  // namespace genai
}  // namespace vpipe

#endif
