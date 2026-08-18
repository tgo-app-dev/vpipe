#ifndef VPIPE_GENERATIVE_MODELS_QUANTIZE_FAMILY_REGISTRY_H
#define VPIPE_GENERATIVE_MODELS_QUANTIZE_FAMILY_REGISTRY_H

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe::genai {

// An out-of-tree multi-component checkpoint that `model-quantize` can
// take down its SELF-CONTAINED path: copy every component, quantize the
// one named by `target`, and register the result as a whole model.
//
// WHY THIS EXISTS. That path already works for the in-tree families, but
// reaching it needs the stage to recognise the repo, and recognition ran
// through a closed switch over in-tree DiT class names plus one
// hardcoded probe. An out-of-tree family therefore fell through to the
// SINGLE-COMPONENT path, which can only be handed one file at a time and
// writes its output beside the file it read. That is a working way to
// quantize -- it is what the LTX-2.5 plugin did first -- but it cannot
// produce one directory holding a quantized DiT, a quantized encoder and
// the untouched VAEs, so an out-of-tree model could not be packaged the
// way every in-tree one is.
//
// WHAT A FAMILY SUPPLIES IS DATA, NOT BEHAVIOUR. Everything that differs
// between repacks is where a component lives and which tensors of it are
// matrices; the assembly, the quantization and the registration are the
// same work. So a family describes its components and the stage does the
// rest -- which also means a family cannot get the parts wrong that are
// not its business.
//
// The registry is consulted BEFORE the built-in detection, mirroring how
// VideoModelRegistry sits before the built-in wan / minimax-h3 dispatch.
// The in-tree families are unchanged and unregistered: this adds a path,
// it does not reroute the existing ones.

// One quantizable component of a repack.
struct QuantizableComponent {
  // The `target` config value that selects this component. Matched
  // case-insensitively and exactly -- a family wanting aliases lists the
  // component more than once. "dit" and "text_encoder" are conventional
  // but nothing here requires them.
  std::string target;

  // The repack subdirectory it lives in ("diffusion_models"). This is
  // BOTH where the source is looked for and where the quantized output
  // is written inside the assembled model, so the output keeps the
  // input's shape and a second pass over a different component finds
  // the first pass's result exactly where it expects it.
  std::string role;

  // Its safetensors `__metadata__` key, and filename hints (best first)
  // used to pick between several files in `role`. A repack keeps its
  // config in the metadata rather than a config.json, so this is how a
  // component is identified at all.
  std::string              meta_key;
  std::vector<std::string> prefer;

  // Which tensors are quantized. `scope` is a tensor-name PREFIX; empty
  // means the whole component. `all_in_scope` takes every 2-D float
  // tensor in scope -- right for a tower whose linear leaves are named
  // unusually; false keeps the standard leaf rule, which skips norms and
  // embeddings.
  //
  // A family that gets this wrong produces a checkpoint that loads, runs
  // and generates the wrong thing, so scope is the field to think about:
  // prefer naming the block stack over quantizing everything and
  // excluding what breaks.
  std::string scope;
  bool        all_in_scope = true;

  // Comma-separated substrings kept DENSE within scope. Appended to the
  // stage's own `quant_exclude`, never replacing it.
  std::string exclude;

  // Recorded in the output's config so a component that leaves the
  // pipeline it came from is still recognisable. Optional: its own
  // config says what architecture it is, not what job it did.
  std::string component_tag;
};

class QuantizableFamily {
public:
  virtual ~QuantizableFamily() = default;

  // The family tag, as it appears in this stage's logs ("ltx-2.5").
  virtual std::string_view tag() const noexcept = 0;

  // Does this checkpoint root belong to this family?
  //
  // Must be CHEAP (a header read, not a load) and SURE. It is asked
  // BEFORE the built-in detection, so a loose claim shadows a working
  // built-in path -- and claiming a repo that is not yours means
  // quantizing someone else's weights with your scope, which produces a
  // plausible checkpoint that is wrong.
  virtual bool claims(const std::string& root) const = 0;

  // The components this family can quantize. Order is the order they are
  // reported in when `target` names none of them, so put the one a
  // caller most likely wants first.
  virtual std::vector<QuantizableComponent> components() const = 0;
};

class QuantizeFamilyRegistry {
public:
  static QuantizeFamilyRegistry& get() noexcept;

  // Takes ownership. First-wins on `tag()`: a second family claiming a
  // tag already present is ignored and returns false.
  bool add(std::unique_ptr<QuantizableFamily> f);

  // The first registered family that claims this root, or null.
  // Registration order, which for plugins is load order -- so a
  // deliberately narrow `claims` is a family's own responsibility. Never
  // throws: a family whose `claims` throws is skipped.
  QuantizableFamily* claim_for(const std::string& root) const;

  QuantizableFamily* find(std::string_view tag) const noexcept;

  std::vector<QuantizableFamily*> all() const;

private:
  QuantizeFamilyRegistry() = default;

  mutable std::mutex                             _mu;
  std::vector<std::unique_ptr<QuantizableFamily>> _families;
};

}  // namespace vpipe::genai

#endif
