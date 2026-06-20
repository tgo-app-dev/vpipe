#ifndef STAGE_REGISTRY_H
#define STAGE_REGISTRY_H

#include "pipeline/stage.h"
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vpipe {

// Process-wide map of stage type-name <-> StageTypeId plus a factory
// per type. Plugin libraries MUST link the host libvpipe shared so
// they observe the same singleton; static linking would fork the
// registry and split type-id space.
//
// StageTypeId values are NOT stable across runs / builds -- they
// reflect the dynamic order in which TypedStage<T>::_type_id is
// initialized. Persist stages by type_name(); never by type().
class StageRegistry {
public:
  using Factory = StagePtr (*)(const SessionContextIntf*,
                               std::string,
                               std::vector<InEdge>,
                               FlexData);

  static StageRegistry& get() noexcept;

  // Registers `type_name` with the given factory. Returns the
  // assigned StageTypeId (>= 1). If the same name is registered
  // twice, returns the existing id and ignores the new factory --
  // this keeps duplicate-include footguns from crashing init.
  StageTypeId register_type(std::string_view type_name, Factory f);

  StageTypeId      find_id  (std::string_view type_name) const noexcept;
  std::string_view find_name(StageTypeId)                const noexcept;

  // Optional per-type StageSpec (the formal description used by
  // tooling). Registered via VPIPE_REGISTER_SPEC alongside the
  // factory; `spec` is a pointer to a stage's file-static kSpec, which
  // has static storage duration. Stages that don't register one return
  // nullptr from spec(). Registering is idempotent (last write wins
  // is avoided -- first registration sticks, like register_type).
  void             set_spec(std::string_view type_name,
                            const StageSpec* spec) noexcept;
  const StageSpec* spec(std::string_view type_name) const noexcept;

  // Construct a Stage of `type_name`. Returns nullptr if the type
  // isn't registered. The trailing `config` is forwarded to the
  // stage's constructor; defaulted to an empty object so callers
  // that don't have or need a configuration tree can omit it.
  StagePtr create(std::string_view          type_name,
                  const SessionContextIntf* session,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config
                    = FlexData::make_object()) const;

  std::vector<std::pair<StageTypeId, std::string>> all() const;

private:
  StageRegistry() = default;

  struct StringHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view sv) const noexcept
    { return std::hash<std::string_view>{}(sv); }
    std::size_t operator()(const std::string& s) const noexcept
    { return std::hash<std::string_view>{}(s); }
  };
  struct StringEq {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b)
      const noexcept { return a == b; }
    bool operator()(const std::string& a, std::string_view b)
      const noexcept { return a == b; }
    bool operator()(std::string_view a, const std::string& b)
      const noexcept { return a == b; }
    bool operator()(const std::string& a, const std::string& b)
      const noexcept { return a == b; }
  };

  struct Entry {
    std::string name;
    Factory     factory;
  };

  // Index is (id - 1). Entry::name owns the canonical string and
  // backs the string_views handed back from find_name.
  std::vector<Entry> _entries;
  std::unordered_map<std::string, StageTypeId,
                     StringHash, StringEq> _by_name;
  // Type-name -> StageSpec*, populated by set_spec (VPIPE_REGISTER_SPEC).
  // Sparse: only stages that declared a spec appear.
  std::unordered_map<std::string, const StageSpec*,
                     StringHash, StringEq> _specs;
  mutable std::mutex _mu;
};

}

#endif
