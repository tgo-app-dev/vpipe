#ifndef TYPED_STAGE_H
#define TYPED_STAGE_H

#include "pipeline/stage.h"
#include "pipeline/stage-registry.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

// CRTP base for concrete stages. Derived must declare:
//   static constexpr const char* kTypeName = "...";
//
// type() and type_name() are final; a Derived class cannot
// accidentally desync the id from the name.
template <class Derived>
class TypedStage : public Stage {
public:
  using Stage::Stage;
  ~TypedStage() override = default;

  StageTypeId      type()      const final { return _type_id; }
  std::string_view type_name() const final { return Derived::kTypeName; }

  // Public so VPIPE_REGISTER_STAGE can odr-use it from a .cc.
  static StageTypeId type_id() { return _type_id; }

private:
  static StagePtr factory(const SessionContextIntf* s,
                          std::string               id,
                          std::vector<InEdge>       iports,
                          FlexData                  config)
  {
    return std::make_unique<Derived>(s, std::move(id),
                                     std::move(iports),
                                     std::move(config));
  }

  // Inline static const: one definition per specialization, vague
  // linkage (COMDAT-folded) so all TUs see the same id. Initialized
  // on first odr-use of _type_id (or via VPIPE_REGISTER_STAGE).
  static inline const StageTypeId _type_id =
    StageRegistry::get().register_type(Derived::kTypeName,
                                       &TypedStage::factory);
};

// Force registration at static-init time even when no instance of T
// is ever constructed and `_type_id` is never otherwise odr-used. A
// class-template static data member is NOT implicitly instantiated
// on its enclosing class's instantiation -- it requires odr-use. Put
// this macro at namespace scope in the Derived stage's .cc file:
//
//   namespace vpipe {
//   class FooStage : public TypedStage<FooStage> { ... };
//   VPIPE_REGISTER_STAGE(FooStage)
//   }
//
// Do NOT remove this macro under "looks unused" -- without it a
// stage referenced only by string lookup never registers.
#define VPIPE_REGISTER_STAGE(T)                                        \
  namespace { const auto _vpipe_stage_reg_##T =                        \
    ::vpipe::TypedStage<T>::type_id(); }

// Register a stage's file-static StageSpec with the registry so tooling
// (the web-ui composer) can read its category / ports / docs without
// constructing an instance. Place at namespace scope in the stage's
// .cc, right after VPIPE_REGISTER_STAGE, passing the in-scope kSpec:
//
//   VPIPE_REGISTER_STAGE(FooStage)
//   VPIPE_REGISTER_SPEC(FooStage, kSpec)
//
// `SPEC` must have static storage duration (the registry keeps a
// pointer). Independent of VPIPE_REGISTER_STAGE; order doesn't matter.
#define VPIPE_REGISTER_SPEC(T, SPEC)                                    \
  namespace { const auto _vpipe_spec_reg_##T = [] {                    \
    ::vpipe::StageRegistry::get().set_spec((SPEC).type_name, &(SPEC)); \
    return 0; }(); }

}

#endif
