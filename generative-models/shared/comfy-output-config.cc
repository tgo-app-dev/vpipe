#include "generative-models/shared/comfy-output-config.h"

#include "common/flex-data.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"
#include "generative-models/shared/comfy-checkpoint.h"

#include <filesystem>
#include <string>

namespace vpipe {
namespace genai {

// The config.json to write beside a quantized Comfy-Org component.
//
// A repack has none -- each component's config lives in its own
// safetensors `__metadata__`, and for the text encoder not even that
// (only the tap; the geometry is in the tensor shapes). The OUTPUT has
// to have one, because it is an ordinary directory checkpoint from here
// on and every loader in this tree reads config.json.
//
// Which model this is comes from two places that have to agree:
//   * the file's own `__metadata__` key, which names the component; and
//   * the REPO around it -- `<file>/../..` -- whose diffusion_models/
//     entry names the architecture. A bare text-encoder file carries no
//     architecture at all, so without the repo there is nothing saying
//     it is H3's Qwen3-VL rather than any other Comfy-Org encoder.
// The repo check is skipped when the file was named directly out of its
// tree (no sibling components found), since that is a deliberate act;
// what it must never do is silently accept a MISMATCH.
bool
comfy_output_config(const std::string& file, FlexData& out, std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "comfy-config: " + std::move(m); }
    return false;
  };
  const fs::path repo = fs::path(file).parent_path().parent_path();
  bool repo_is_h3 = false, repo_seen = false;
  for (const comfy::Component& c : comfy::scan_repo(repo.string())) {
    if (c.role != "diffusion_models") { continue; }
    repo_seen = true;
    FlexData md;
    if (!comfy::metadata_json(c.file, "config", md, nullptr) ||
        !md.is_object() || !md.as_object().contains("transformer")) {
      continue;
    }
    const FlexData t = md.as_object().at("transformer");
    auto to = t.as_object();
    const FlexData im = to.contains("image_model") ? to.at("image_model")
                                                   : FlexData();
    if (std::string(im.as_string("")) == "minimax_h3") { repo_is_h3 = true; }
  }
  if (repo_seen && !repo_is_h3) {
    return fail("'" + file + "' sits in a Comfy-Org repo whose DiT is not "
                "minimax_h3; this pass does not know that architecture");
  }

  // ---- the DiT ---------------------------------------------------
  if (comfy::is_component(file, "config")) {
    FlexData md;
    std::string cerr;
    if (!comfy::metadata_json(file, "config", md, &cerr) ||
        !md.is_object() || !md.as_object().contains("transformer")) {
      return fail("no embedded config in " + file +
                  (cerr.empty() ? "" : " (" + cerr + ")"));
    }
    out = md.as_object().at("transformer");
    if (!out.is_object()) {
      return fail("embedded 'transformer' is not an object");
    }
    auto o = out.as_object();
    const FlexData im = o.contains("image_model") ? o.at("image_model")
                                                  : FlexData();
    if (std::string(im.as_string("")) != "minimax_h3") {
      return fail("unsupported Comfy-Org image_model '" +
                  std::string(im.as_string("")) + "'");
    }
    o.insert_or_assign("_class_name",
                       FlexData::make_string("MiniMaxH3DiTModel"));
    // MUST survive: Comfy-Org's conversion reorders the fused qkv
    // projection, this pass copies that order through verbatim, and the
    // names and shapes are identical either way -- so an output that
    // does not SAY which order it is in gets read as the released one
    // and computes nonsense.
    o.insert_or_assign("qkv_per_head", FlexData::make_bool(false));
    // MUST survive for the same reason, and it is the reason the
    // filename existed: `fl2va` and `ref2va` are two TASKS over one
    // architecture with byte-identical configs, and a repack names the
    // partition in the filename and nowhere else. This output is a
    // DIRECTORY of shards, so that name is gone -- and a Ref2VA
    // checkpoint read as `fl2va` loads, runs at full 33B cost and
    // generates video conditioned on nothing.
    {
      const std::string part =
          MetalMiniMaxH3Transformer::partition_of(file);
      if (!part.empty()) {
        o.insert_or_assign(MetalMiniMaxH3Transformer::kPartitionKey,
                           FlexData::make_string(part));
      }
    }
    return true;
  }

  // ---- the text encoder ------------------------------------------
  // Round-tripped through the encoder's OWN config reader rather than
  // re-measured here: it already derives the geometry from the tensor
  // shapes (there is no config to read), and going back out through the
  // same fields is what guarantees the written config reloads to the
  // Config the source produced.
  if (comfy::is_component(file, "minimax_h3_te")) {
    MiniMaxH3TextEncoder::Config c;
    std::string cerr;
    if (!MiniMaxH3TextEncoder::config_from_json(file, c, &cerr)) {
      return fail(cerr);
    }
    out = FlexData::make_object();
    FlexData tc = FlexData::make_object();
    {
      auto t = tc.as_object();
      auto put = [&](const char* k, long long v) {
        t.insert_or_assign(k, FlexData::make_int(v));
      };
      put("num_hidden_layers", c.total_layers);
      put("hidden_size", c.lm.hidden);
      put("num_attention_heads", c.lm.n_heads);
      put("num_key_value_heads", c.lm.n_kv_heads);
      put("head_dim", c.lm.head_dim);
      put("intermediate_size", c.lm.ffn_inner);
      put("vocab_size", c.lm.vocab);
      t.insert_or_assign("rope_theta",
                         FlexData::make_real((double)c.lm.rope_theta));
      t.insert_or_assign("rms_norm_eps",
                         FlexData::make_real((double)c.lm.rms_eps));
    }
    auto o = out.as_object();
    o.insert_or_assign("text_config", std::move(tc));
    // The encoder's reader keys on this to accept the file as a
    // Qwen3-VL, and on tie_word_embeddings for the head it never loads.
    o.insert_or_assign("model_type", FlexData::make_string("qwen3_vl"));
    o.insert_or_assign("tie_word_embeddings", FlexData::make_bool(false));
    return true;
  }
  return fail("'" + file + "' is not a Comfy-Org component this pass can "
              "write a config for");
}
}  // namespace genai
}  // namespace vpipe
