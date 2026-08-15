// The Gemma-4 side of generative-models/hidden-state-encoder.h.
//
// Deliberately thin: everything model-specific already exists on
// MetalGemmaModel (forward_embeddings_taps), so this is the adapter that
// opens a checkpoint and states which architecture strings it answers to.
// A second family (Qwen3) is the same forty lines against its own model.

#include "generative-models/hidden-state-encoder.h"

#include "generative-models/gemma4/metal-gemma-model.h"
#include "generative-models/model-loader.h"
#include "generative-models/weight-set.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace vpipe::genai {

namespace {

class GemmaHiddenStateEncoder : public HiddenStateEncoder {
public:
  GemmaHiddenStateEncoder(std::unique_ptr<MetalGemmaModel> m,
                          std::shared_ptr<WeightSet>       ws)
      : _m(std::move(m)), _ws(std::move(ws)) {}

  int n_layers() const noexcept override { return _m->config().n_layers; }
  int hidden_dim() const noexcept override { return _m->config().hidden; }

  int max_tap_index() const noexcept override
  {
    const auto& c = _m->config();
    return (c.num_kv_shared > 0) ? c.first_shared() : c.n_layers;
  }

  bool encode(const std::vector<std::int32_t>& ids,
              const HiddenTapRequest& req, HiddenTapResult* out,
              std::string* err) override
  {
    if (out == nullptr) {
      if (err != nullptr) { *err = "no result to write into"; }
      return false;
    }
    // A FRESH context per encode. The tap path refuses a prompt that
    // would wrap the sliding ring from a non-zero offset, so carrying KV
    // over from the previous caption would make the second encode of a
    // run fail where the first succeeded.
    _m->release_kv(_cid);
    _cid.v = _cid.v + 1;

    std::string e;
    auto buf = _m->forward_embeddings_taps(_cid, ids, req.indices,
                                           req.key_valid_len, &e);
    if (buf.empty()) {
      if (err != nullptr) { *err = std::move(e); }
      return false;
    }
    out->data   = std::move(buf);
    out->slots  = (int)req.indices.size();
    out->tokens = (int)ids.size();
    out->hidden = _m->config().hidden;
    out->dtype  = _m->config().use_bf16 ? "bf16" : "f16";
    return true;
  }

private:
  std::unique_ptr<MetalGemmaModel> _m;
  // Held for this encoder's lifetime, per the WeightSet contract.
  std::shared_ptr<WeightSet>       _ws;
  ContextId                        _cid{0};
};

// The config, from wherever the caller left it.
bool
config_for_(const HiddenStateEncoderArgs& args, ModelConfig* out,
            std::string* err)
{
  FlexData cfg = args.config;
  if (!cfg.is_object()) {
    const std::filesystem::path p =
        std::filesystem::path(args.dir) / "config.json";
    std::ifstream in(p);
    if (!in) {
      *err = "no config was supplied and '" + p.string() + "' is not "
             "readable";
      return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    cfg = FlexData::from_json(ss.str());
  }
  auto parsed = ModelLoader::parse_config(cfg);
  if (!parsed) {
    *err = "the config is not a Gemma-4 model config this build can parse";
    return false;
  }
  *out = std::move(*parsed);
  return true;
}

std::unique_ptr<HiddenStateEncoder>
make_(const HiddenStateEncoderArgs& args)
{
  auto warn = [&](const std::string& m) {
    if (args.err != nullptr) { *args.err = m; }
    if (args.session != nullptr) {
      args.session->warn(fmt("gemma hidden-state encoder: {}", m));
    }
  };
  if (args.metal == nullptr || !args.metal->valid()) {
    warn("no usable metal-compute backend");
    return nullptr;
  }

  ModelConfig mc;
  std::string err;
  if (!config_for_(args, &mc, &err)) {
    warn(err);
    return nullptr;
  }

  auto ws = open_weight_set(args.dir, args.session);
  if (!ws) {
    warn("could not open '" + args.dir + "'");
    return nullptr;
  }
  auto model = MetalGemmaModel::load(ws, args.metal,
                                     MetalGemmaModel::config_from(mc));
  if (!model) {
    warn("could not build the model from '" + args.dir + "'");
    return nullptr;
  }
  return std::make_unique<GemmaHiddenStateEncoder>(std::move(model),
                                                   std::move(ws));
}

}  // namespace

void
register_builtin_hidden_state_encoders(HiddenStateEncoderRegistry& r)
{
  // Both spellings HuggingFace uses: `architectures[0]` and
  // `model_type`. A checkpoint carries one or the other and the registry
  // does not know which it read.
  r.register_arch("gemma4", make_);
  r.register_arch("Gemma4ForConditionalGeneration", make_);
  r.register_arch("gemma4_unified", make_);
  r.register_arch("Gemma4UnifiedForConditionalGeneration", make_);
  r.register_arch("gemma3n", make_);
  r.register_arch("Gemma3nForConditionalGeneration", make_);
}

}  // namespace vpipe::genai
