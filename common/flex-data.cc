#include "flex-data.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace vpipe {
using namespace std;

// =============================================================================
// internal types
// =============================================================================

namespace {
struct StringHash {
  using is_transparent = void;
  size_t operator()(string_view sv)  const noexcept
    { return hash<string_view>{}(sv); }
  size_t operator()(const string& s) const noexcept
    { return hash<string_view>{}(s);  }
};
struct StringEq {
  using is_transparent = void;
  bool operator()(string_view a, string_view b)     const noexcept
    { return a == b; }
  bool operator()(const string& a, string_view b)   const noexcept
    { return a == b; }
  bool operator()(string_view a, const string& b)   const noexcept
    { return a == b; }
  bool operator()(const string& a, const string& b) const noexcept
    { return a == b; }
};
}

struct ArrayStorage {
  enum class Mode : unsigned char {
    Hetero, Bool, Int, Uint, Real, String, Null
  };
  Mode mode = Mode::Hetero;
  vector<unique_ptr<FlexData::Impl>> hetero;
  vector<int64_t>       ints;
  vector<uint64_t>      uints;
  vector<double>        reals;
  vector<string>        strings;
  vector<unsigned char> bools;
  size_t                null_count = 0;
};

struct ObjectStorage {
  vector<pair<string, unique_ptr<FlexData::Impl>>>    entries;
  unordered_map<string, size_t, StringHash, StringEq> index;
};

class FlexData::Impl {
public:
  FlexKind kind = FlexKind::Null;
  variant<monostate, bool, int64_t, uint64_t, double, string,
          ArrayStorage, ObjectStorage> val;

  unique_ptr<Impl> clone() const;
};

// =============================================================================
// helpers
// =============================================================================

namespace {

unique_ptr<FlexData::Impl> make_kind(FlexKind k) {
  auto p  = make_unique<FlexData::Impl>();
  p->kind = k;
  return p;
}

unique_ptr<FlexData::Impl> clone_impl(const FlexData::Impl& src) {
  auto out  = make_unique<FlexData::Impl>();
  out->kind = src.kind;
  switch (src.kind) {
    case FlexKind::Null:   out->val = monostate{};            break;
    case FlexKind::Bool:   out->val = get<bool>(src.val);     break;
    case FlexKind::Int:    out->val = get<int64_t>(src.val);  break;
    case FlexKind::Uint:   out->val = get<uint64_t>(src.val); break;
    case FlexKind::Real:   out->val = get<double>(src.val);   break;
    case FlexKind::String: out->val = get<string>(src.val);   break;
    case FlexKind::Array: {
      const auto& sa = get<ArrayStorage>(src.val);
      ArrayStorage da;
      da.mode       = sa.mode;
      da.ints       = sa.ints;
      da.uints      = sa.uints;
      da.reals      = sa.reals;
      da.strings    = sa.strings;
      da.bools      = sa.bools;
      da.null_count = sa.null_count;
      da.hetero.reserve(sa.hetero.size());
      for (auto& p : sa.hetero) da.hetero.push_back(clone_impl(*p));
      out->val = std::move(da);
      break;
    }
    case FlexKind::Object: {
      const auto& so = get<ObjectStorage>(src.val);
      ObjectStorage doi;
      doi.entries.reserve(so.entries.size());
      for (auto& [k, v] : so.entries) {
        doi.entries.emplace_back(k, clone_impl(*v));
      }
      doi.index.reserve(doi.entries.size());
      for (size_t i = 0; i < doi.entries.size(); ++i) {
        doi.index.emplace(doi.entries[i].first, i);
      }
      out->val = std::move(doi);
      break;
    }
  }
  return out;
}

void demote_array_to_hetero(ArrayStorage& a) {
  if (a.mode == ArrayStorage::Mode::Hetero) return;
  vector<unique_ptr<FlexData::Impl>> h;
  switch (a.mode) {
    case ArrayStorage::Mode::Bool:
      h.reserve(a.bools.size());
      for (unsigned char b : a.bools) {
        auto p  = make_unique<FlexData::Impl>();
        p->kind = FlexKind::Bool;
        p->val  = static_cast<bool>(b != 0);
        h.push_back(std::move(p));
      }
      a.bools.clear();
      a.bools.shrink_to_fit();
      break;
    case ArrayStorage::Mode::Int:
      h.reserve(a.ints.size());
      for (int64_t i : a.ints) {
        auto p  = make_unique<FlexData::Impl>();
        p->kind = FlexKind::Int;
        p->val  = i;
        h.push_back(std::move(p));
      }
      a.ints.clear();
      a.ints.shrink_to_fit();
      break;
    case ArrayStorage::Mode::Uint:
      h.reserve(a.uints.size());
      for (uint64_t i : a.uints) {
        auto p  = make_unique<FlexData::Impl>();
        p->kind = FlexKind::Uint;
        p->val  = i;
        h.push_back(std::move(p));
      }
      a.uints.clear();
      a.uints.shrink_to_fit();
      break;
    case ArrayStorage::Mode::Real:
      h.reserve(a.reals.size());
      for (double d : a.reals) {
        auto p  = make_unique<FlexData::Impl>();
        p->kind = FlexKind::Real;
        p->val  = d;
        h.push_back(std::move(p));
      }
      a.reals.clear();
      a.reals.shrink_to_fit();
      break;
    case ArrayStorage::Mode::String:
      h.reserve(a.strings.size());
      for (auto& s : a.strings) {
        auto p  = make_unique<FlexData::Impl>();
        p->kind = FlexKind::String;
        p->val  = std::move(s);
        h.push_back(std::move(p));
      }
      a.strings.clear();
      a.strings.shrink_to_fit();
      break;
    case ArrayStorage::Mode::Null:
      h.reserve(a.null_count);
      for (size_t i = 0; i < a.null_count; ++i) {
        h.push_back(make_kind(FlexKind::Null));
      }
      a.null_count = 0;
      break;
    default:
      break;
  }
  a.hetero = std::move(h);
  a.mode   = ArrayStorage::Mode::Hetero;
}

ArrayStorage::Mode kind_to_mode(FlexKind k) {
  switch (k) {
    case FlexKind::Null:   return ArrayStorage::Mode::Null;
    case FlexKind::Bool:   return ArrayStorage::Mode::Bool;
    case FlexKind::Int:    return ArrayStorage::Mode::Int;
    case FlexKind::Uint:   return ArrayStorage::Mode::Uint;
    case FlexKind::Real:   return ArrayStorage::Mode::Real;
    case FlexKind::String: return ArrayStorage::Mode::String;
    default:               return ArrayStorage::Mode::Hetero;
  }
}

const ArrayStorage&  as_array_store(const FlexData::Impl* p)
  { return get<ArrayStorage>(p->val); }
ArrayStorage&        as_array_store_mut(FlexData::Impl* p)
  { return get<ArrayStorage>(p->val); }
const ObjectStorage& as_object_store(const FlexData::Impl* p)
  { return get<ObjectStorage>(p->val); }
ObjectStorage&       as_object_store_mut(FlexData::Impl* p)
  { return get<ObjectStorage>(p->val); }

size_t array_size(const ArrayStorage& a) {
  switch (a.mode) {
    case ArrayStorage::Mode::Hetero: return a.hetero.size();
    case ArrayStorage::Mode::Bool:   return a.bools.size();
    case ArrayStorage::Mode::Int:    return a.ints.size();
    case ArrayStorage::Mode::Uint:   return a.uints.size();
    case ArrayStorage::Mode::Real:   return a.reals.size();
    case ArrayStorage::Mode::String: return a.strings.size();
    case ArrayStorage::Mode::Null:   return a.null_count;
  }
  return 0;
}

FlexData materialize_at(const ArrayStorage& a, size_t idx) {
  switch (a.mode) {
    case ArrayStorage::Mode::Hetero:
      return FlexData{a.hetero[idx]->clone()};
    case ArrayStorage::Mode::Bool:
      return FlexData::make_bool(a.bools[idx] != 0);
    case ArrayStorage::Mode::Int:
      return FlexData::make_int(a.ints[idx]);
    case ArrayStorage::Mode::Uint:
      return FlexData::make_uint(a.uints[idx]);
    case ArrayStorage::Mode::Real:
      return FlexData::make_real(a.reals[idx]);
    case ArrayStorage::Mode::String:
      return FlexData::make_string(a.strings[idx]);
    case ArrayStorage::Mode::Null:
      return FlexData::make_null();
  }
  return FlexData::make_null();
}

}  // anonymous namespace

unique_ptr<FlexData::Impl> FlexData::Impl::clone() const
  { return clone_impl(*this); }

// =============================================================================
// ParseError
// =============================================================================

FlexData::ParseError::ParseError(size_t line, size_t column, const string& msg)
  : Error("line " + to_string(line) + ", column " + to_string(column) + ": "
            + msg),
    _line(line),
    _column(column) {}

size_t FlexData::ParseError::line()   const noexcept { return _line; }
size_t FlexData::ParseError::column() const noexcept { return _column; }

// =============================================================================
// BinaryError
// =============================================================================

FlexData::BinaryError::BinaryError(size_t byte_offset, const string& msg)
  : Error("byte " + to_string(byte_offset) + ": " + msg),
    _byte_offset(byte_offset) {}

size_t FlexData::BinaryError::byte_offset() const noexcept
  { return _byte_offset; }

// =============================================================================
// FlexData lifecycle and factories
// =============================================================================

FlexData::FlexData() : _impl(nullptr) {}
FlexData::FlexData(unique_ptr<Impl> p) noexcept : _impl(std::move(p)) {}

FlexData::FlexData(const FlexData& other)
  : _impl(other._impl ? other._impl->clone() : nullptr) {}

FlexData::FlexData(FlexData&& other) noexcept = default;

FlexData& FlexData::operator=(const FlexData& other) {
  if (this != &other) _impl = other._impl ? other._impl->clone() : nullptr;
  return *this;
}

FlexData& FlexData::operator=(FlexData&& other) noexcept = default;

FlexData::~FlexData() = default;

FlexData FlexData::make_null() { return FlexData{}; }
FlexData FlexData::make_bool(bool v) {
  auto p = make_unique<Impl>();
  p->kind = FlexKind::Bool; p->val = v;
  return FlexData{std::move(p)};
}
FlexData FlexData::make_int(int64_t v) {
  auto p = make_unique<Impl>();
  p->kind = FlexKind::Int; p->val = v;
  return FlexData{std::move(p)};
}
FlexData FlexData::make_uint(uint64_t v) {
  auto p = make_unique<Impl>();
  p->kind = FlexKind::Uint; p->val = v;
  return FlexData{std::move(p)};
}
FlexData FlexData::make_real(double v) {
  auto p = make_unique<Impl>();
  p->kind = FlexKind::Real; p->val = v;
  return FlexData{std::move(p)};
}
FlexData FlexData::make_string(string_view v) {
  auto p = make_unique<Impl>();
  p->kind = FlexKind::String; p->val = string{v};
  return FlexData{std::move(p)};
}
FlexData FlexData::make_array() {
  auto p = make_unique<Impl>();
  p->kind = FlexKind::Array; p->val = ArrayStorage{};
  return FlexData{std::move(p)};
}
FlexData FlexData::make_object() {
  auto p = make_unique<Impl>();
  p->kind = FlexKind::Object; p->val = ObjectStorage{};
  return FlexData{std::move(p)};
}

// =============================================================================
// introspection
// =============================================================================

FlexKind FlexData::kind() const noexcept
  { return _impl ? _impl->kind : FlexKind::Null; }
bool FlexData::is_null()   const noexcept { return kind() == FlexKind::Null;   }
bool FlexData::is_bool()   const noexcept { return kind() == FlexKind::Bool;   }
bool FlexData::is_int()    const noexcept { return kind() == FlexKind::Int;    }
bool FlexData::is_uint()   const noexcept { return kind() == FlexKind::Uint;   }
bool FlexData::is_real()   const noexcept { return kind() == FlexKind::Real;   }
bool FlexData::is_string() const noexcept { return kind() == FlexKind::String; }
bool FlexData::is_array()  const noexcept { return kind() == FlexKind::Array;  }
bool FlexData::is_object() const noexcept { return kind() == FlexKind::Object; }
bool FlexData::is_scalar() const noexcept {
  switch (kind()) {
    case FlexKind::Null: case FlexKind::Bool:  case FlexKind::Int:
    case FlexKind::Uint: case FlexKind::Real:  case FlexKind::String:
      return true;
    default:
      return false;
  }
}

// =============================================================================
// scalar accessors
// =============================================================================

bool        FlexData::get_bool()   const {
  if (!is_bool()) throw TypeError("FlexData::get_bool: kind is not Bool");
  return get<bool>(_impl->val);
}
int64_t     FlexData::get_int()    const {
  if (!is_int()) throw TypeError("FlexData::get_int: kind is not Int");
  return get<int64_t>(_impl->val);
}
uint64_t    FlexData::get_uint()   const {
  if (!is_uint()) throw TypeError("FlexData::get_uint: kind is not Uint");
  return get<uint64_t>(_impl->val);
}
double      FlexData::get_real()   const {
  if (!is_real()) throw TypeError("FlexData::get_real: kind is not Real");
  return get<double>(_impl->val);
}
string_view FlexData::get_string() const {
  if (!is_string()) throw TypeError("FlexData::get_string: kind is not String");
  return get<string>(_impl->val);
}

bool        FlexData::as_bool(bool def)          const noexcept
  { return is_bool()   ? get<bool>(_impl->val)                : def; }
int64_t     FlexData::as_int(int64_t def)        const noexcept {
  if (is_int())  { return                       get<int64_t> (_impl->val);  }
  if (is_uint()) { return static_cast<int64_t>(get<uint64_t>(_impl->val)); }
  if (is_real()) { return static_cast<int64_t>(get<double>  (_impl->val)); }
  return def;
}
uint64_t    FlexData::as_uint(uint64_t def)      const noexcept {
  if (is_uint()) { return                        get<uint64_t>(_impl->val);  }
  if (is_int())  { return static_cast<uint64_t>(get<int64_t> (_impl->val)); }
  if (is_real()) { return static_cast<uint64_t>(get<double>  (_impl->val)); }
  return def;
}
double      FlexData::as_real(double def)        const noexcept {
  if (is_real()) { return                      get<double>  (_impl->val);  }
  if (is_int())  { return static_cast<double>(get<int64_t> (_impl->val)); }
  if (is_uint()) { return static_cast<double>(get<uint64_t>(_impl->val)); }
  return def;
}
string_view FlexData::as_string(string_view def) const noexcept
  { return is_string() ? string_view{get<string>(_impl->val)} : def; }

// =============================================================================
// view dispatch + span fast paths
// =============================================================================

FlexData::ArrayView      FlexData::as_array()         {
  if (!is_array()) throw TypeError("FlexData::as_array: kind is not Array");
  return ArrayView{_impl.get()};
}
FlexData::ConstArrayView FlexData::as_array() const   {
  if (!is_array()) throw TypeError("FlexData::as_array: kind is not Array");
  return ConstArrayView{_impl.get()};
}
FlexData::ObjectView     FlexData::as_object()        {
  if (!is_object()) throw TypeError("FlexData::as_object: kind is not Object");
  return ObjectView{_impl.get()};
}
FlexData::ConstObjectView FlexData::as_object() const {
  if (!is_object()) throw TypeError("FlexData::as_object: kind is not Object");
  return ConstObjectView{_impl.get()};
}

span<const int64_t>  FlexData::as_int_span()  const noexcept {
  if (!is_array()) return {};
  const auto& a = get<ArrayStorage>(_impl->val);
  return a.mode == ArrayStorage::Mode::Int ? span<const int64_t>{a.ints} :
                                             span<const int64_t>{};
}
span<const uint64_t> FlexData::as_uint_span() const noexcept {
  if (!is_array()) return {};
  const auto& a = get<ArrayStorage>(_impl->val);
  return a.mode == ArrayStorage::Mode::Uint ? span<const uint64_t>{a.uints} :
                                              span<const uint64_t>{};
}
span<const double>   FlexData::as_real_span() const noexcept {
  if (!is_array()) return {};
  const auto& a = get<ArrayStorage>(_impl->val);
  return a.mode == ArrayStorage::Mode::Real ? span<const double>{a.reals} :
                                              span<const double>{};
}
span<const unsigned char> FlexData::as_bool_span() const noexcept {
  if (!is_array()) return {};
  const auto& a = get<ArrayStorage>(_impl->val);
  return a.mode == ArrayStorage::Mode::Bool ?
           span<const unsigned char>{a.bools} : span<const unsigned char>{};
}

// =============================================================================
// ConstArrayView / ArrayView
// =============================================================================

FlexData::ConstArrayView::ConstArrayView(const Impl* arr) noexcept : _arr(arr)
  {}

FlexData::ConstArrayView::iterator::iterator() noexcept : _arr(nullptr), _idx(0)
  {}
FlexData::ConstArrayView::iterator::iterator
  (const Impl* arr, size_t idx) noexcept
  : _arr(arr), _idx(idx)
  {}

FlexData FlexData::ConstArrayView::iterator::operator*() const
  { return materialize_at(as_array_store(_arr), _idx); }
FlexData::ConstArrayView::iterator&
FlexData::ConstArrayView::iterator::operator++() noexcept
  { ++_idx; return *this; }
FlexData::ConstArrayView::iterator
FlexData::ConstArrayView::iterator::operator++(int) noexcept
  { iterator t = *this; ++_idx; return t; }
FlexData::ConstArrayView::iterator&
FlexData::ConstArrayView::iterator::operator--() noexcept
  { --_idx; return *this; }
FlexData::ConstArrayView::iterator
FlexData::ConstArrayView::iterator::operator--(int) noexcept
  { iterator t = *this; --_idx; return t; }
FlexData::ConstArrayView::iterator&
FlexData::ConstArrayView::iterator::operator+=(difference_type n) noexcept {
  _idx = static_cast<size_t>(static_cast<ptrdiff_t>(_idx) + n);
  return *this;
}
FlexData::ConstArrayView::iterator&
FlexData::ConstArrayView::iterator::operator-=(difference_type n) noexcept
  { return (*this += -n); }
FlexData::ConstArrayView::iterator
FlexData::ConstArrayView::iterator::operator+(difference_type n) const noexcept
  { iterator t = *this; t += n; return t; }
FlexData::ConstArrayView::iterator
FlexData::ConstArrayView::iterator::operator-(difference_type n) const noexcept
  { iterator t = *this; t -= n; return t; }
FlexData::ConstArrayView::iterator::difference_type
FlexData::ConstArrayView::iterator::operator-
  (const iterator& other) const noexcept
{
  return static_cast<difference_type>(_idx)
           - static_cast<difference_type>(other._idx);
}
FlexData
FlexData::ConstArrayView::iterator::operator[](difference_type n) const {
  return materialize_at(as_array_store(_arr),
                        static_cast<size_t>(static_cast<ptrdiff_t>(_idx) + n));
}
bool FlexData::ConstArrayView::iterator::operator==
  (const iterator& o) const noexcept
  { return _arr == o._arr && _idx == o._idx; }
bool FlexData::ConstArrayView::iterator::operator!=
  (const iterator& o) const noexcept
  { return !(*this == o); }
bool FlexData::ConstArrayView::iterator::operator<
  (const iterator& o) const noexcept
  { return _idx <  o._idx; }
bool FlexData::ConstArrayView::iterator::operator<=
  (const iterator& o) const noexcept
  { return _idx <= o._idx; }
bool FlexData::ConstArrayView::iterator::operator>
  (const iterator& o) const noexcept
  { return _idx >  o._idx; }
bool FlexData::ConstArrayView::iterator::operator>=
  (const iterator& o) const noexcept
  { return _idx >= o._idx; }

FlexData::ConstArrayView::iterator
FlexData::ConstArrayView::begin() const noexcept
  { return iterator{_arr, 0}; }
FlexData::ConstArrayView::iterator
FlexData::ConstArrayView::end()   const noexcept
  { return iterator{_arr, array_size(as_array_store(_arr))}; }
size_t   FlexData::ConstArrayView::size()  const noexcept
  { return array_size(as_array_store(_arr)); }
bool     FlexData::ConstArrayView::empty() const noexcept
  { return size() == 0; }
FlexData FlexData::ConstArrayView::operator[](size_t i) const
  { return materialize_at(as_array_store(_arr), i); }
FlexData FlexData::ConstArrayView::at(size_t i) const {
  if (i >= size()) {
    throw out_of_range("FlexData::ArrayView::at: index out of range");
  }
  return materialize_at(as_array_store(_arr), i);
}

FlexData::ArrayView::ArrayView(Impl* arr) noexcept
  : ConstArrayView(arr), _mut(arr) {}

void FlexData::ArrayView::push_back(FlexData v) {
  auto& a = as_array_store_mut(_mut);
  FlexKind vk = v._impl ? v._impl->kind : FlexKind::Null;

  // If empty and still in default Hetero mode, switch to homogeneous mode
  // matching the value's kind so subsequent typed pushes stay typed.
  if (array_size(a) == 0 && a.mode == ArrayStorage::Mode::Hetero) {
    auto m = kind_to_mode(vk);
    if (m != ArrayStorage::Mode::Hetero) a.mode = m;
  }

  if (a.mode == ArrayStorage::Mode::Hetero) {
    a.hetero.push_back(v._impl ? std::move(v._impl) :
                                 make_kind(FlexKind::Null));
    return;
  }

  bool matches = false;
  switch (a.mode) {
    case ArrayStorage::Mode::Bool:   matches = (vk == FlexKind::Bool);   break;
    case ArrayStorage::Mode::Int:    matches = (vk == FlexKind::Int);    break;
    case ArrayStorage::Mode::Uint:   matches = (vk == FlexKind::Uint);   break;
    case ArrayStorage::Mode::Real:   matches = (vk == FlexKind::Real);   break;
    case ArrayStorage::Mode::String: matches = (vk == FlexKind::String); break;
    case ArrayStorage::Mode::Null:   matches = (vk == FlexKind::Null);   break;
    default: break;
  }
  if (!matches) {
    demote_array_to_hetero(a);
    a.hetero.push_back(v._impl ? std::move(v._impl) :
                                 make_kind(FlexKind::Null));
    return;
  }
  switch (a.mode) {
    case ArrayStorage::Mode::Bool:
      a.bools.push_back(static_cast<unsigned char>(
                          get<bool>(v._impl->val) ? 1 : 0));
      break;
    case ArrayStorage::Mode::Int:
      a.ints.push_back(get<int64_t>(v._impl->val));
      break;
    case ArrayStorage::Mode::Uint:
      a.uints.push_back(get<uint64_t>(v._impl->val));
      break;
    case ArrayStorage::Mode::Real:
      a.reals.push_back(get<double>(v._impl->val));
      break;
    case ArrayStorage::Mode::String:
      a.strings.push_back(get<string>(std::move(v._impl->val)));
      break;
    case ArrayStorage::Mode::Null:
      ++a.null_count;
      break;
    default:
      break;
  }
}

void FlexData::ArrayView::push_back(bool v) {
  auto& a = as_array_store_mut(_mut);
  if (array_size(a) == 0 && a.mode == ArrayStorage::Mode::Hetero) {
    a.mode = ArrayStorage::Mode::Bool;
  }
  if (a.mode != ArrayStorage::Mode::Bool) {
    demote_array_to_hetero(a);
    auto p  = make_unique<Impl>();
    p->kind = FlexKind::Bool;
    p->val  = v;
    a.hetero.push_back(std::move(p));
  } else {
    a.bools.push_back(static_cast<unsigned char>(v ? 1 : 0));
  }
}

#define VPIPE_TYPED_PUSH(TYPE_, MODE_, VEC_)                                  \
void FlexData::ArrayView::push_back(TYPE_ v) {                                \
  auto& a = as_array_store_mut(_mut);                                         \
  if (array_size(a) == 0 && a.mode == ArrayStorage::Mode::Hetero) {           \
    a.mode = ArrayStorage::Mode::MODE_;                                       \
  }                                                                           \
  if (a.mode != ArrayStorage::Mode::MODE_) {                                  \
    demote_array_to_hetero(a);                                                \
    auto p  = make_unique<Impl>();                                            \
    p->kind = FlexKind::MODE_;                                                \
    p->val  = v;                                                              \
    a.hetero.push_back(std::move(p));                                         \
  } else {                                                                    \
    a.VEC_.push_back(v);                                                      \
  }                                                                           \
}

VPIPE_TYPED_PUSH(int64_t, Int,  ints)
VPIPE_TYPED_PUSH(uint64_t,Uint, uints)
VPIPE_TYPED_PUSH(double,  Real, reals)

#undef VPIPE_TYPED_PUSH

// String version is special because the typed vector and the hetero impl
// expect different types (`string` vs assigned via `string{v}`).
void FlexData::ArrayView::push_back(string_view v) {
  auto& a = as_array_store_mut(_mut);
  if (array_size(a) == 0 && a.mode == ArrayStorage::Mode::Hetero) {
    a.mode = ArrayStorage::Mode::String;
  }
  if (a.mode != ArrayStorage::Mode::String) {
    demote_array_to_hetero(a);
    auto p  = make_unique<Impl>();
    p->kind = FlexKind::String;
    p->val  = string{v};
    a.hetero.push_back(std::move(p));
  } else {
    a.strings.emplace_back(v);
  }
}

bool FlexData::ArrayView::pop_back() {
  auto& a = as_array_store_mut(_mut);
  if (array_size(a) == 0) return false;
  switch (a.mode) {
    case ArrayStorage::Mode::Hetero: a.hetero.pop_back();  break;
    case ArrayStorage::Mode::Bool:   a.bools.pop_back();   break;
    case ArrayStorage::Mode::Int:    a.ints.pop_back();    break;
    case ArrayStorage::Mode::Uint:   a.uints.pop_back();   break;
    case ArrayStorage::Mode::Real:   a.reals.pop_back();   break;
    case ArrayStorage::Mode::String: a.strings.pop_back(); break;
    case ArrayStorage::Mode::Null:   --a.null_count;       break;
  }
  return true;
}

void FlexData::ArrayView::clear() noexcept {
  auto& a = as_array_store_mut(_mut);
  a.hetero.clear();
  a.ints.clear();
  a.uints.clear();
  a.reals.clear();
  a.strings.clear();
  a.bools.clear();
  a.null_count = 0;
  a.mode       = ArrayStorage::Mode::Hetero;
}

void FlexData::ArrayView::reserve(size_t n) {
  auto& a = as_array_store_mut(_mut);
  switch (a.mode) {
    case ArrayStorage::Mode::Hetero: a.hetero.reserve(n);  break;
    case ArrayStorage::Mode::Bool:   a.bools.reserve(n);   break;
    case ArrayStorage::Mode::Int:    a.ints.reserve(n);    break;
    case ArrayStorage::Mode::Uint:   a.uints.reserve(n);   break;
    case ArrayStorage::Mode::Real:   a.reals.reserve(n);   break;
    case ArrayStorage::Mode::String: a.strings.reserve(n); break;
    case ArrayStorage::Mode::Null:   break;
  }
}

bool FlexData::ArrayView::set(size_t i, FlexData v) {
  auto& a = as_array_store_mut(_mut);
  if (i >= array_size(a)) return false;
  FlexKind vk = v._impl ? v._impl->kind : FlexKind::Null;

  if (a.mode == ArrayStorage::Mode::Hetero) {
    a.hetero[i] = v._impl ? std::move(v._impl) : make_kind(FlexKind::Null);
    return true;
  }

  bool matches = false;
  switch (a.mode) {
    case ArrayStorage::Mode::Bool:   matches = (vk == FlexKind::Bool);   break;
    case ArrayStorage::Mode::Int:    matches = (vk == FlexKind::Int);    break;
    case ArrayStorage::Mode::Uint:   matches = (vk == FlexKind::Uint);   break;
    case ArrayStorage::Mode::Real:   matches = (vk == FlexKind::Real);   break;
    case ArrayStorage::Mode::String: matches = (vk == FlexKind::String); break;
    case ArrayStorage::Mode::Null:   matches = (vk == FlexKind::Null);   break;
    default: break;
  }
  if (!matches) {
    demote_array_to_hetero(a);
    a.hetero[i] = v._impl ? std::move(v._impl) : make_kind(FlexKind::Null);
    return true;
  }
  switch (a.mode) {
    case ArrayStorage::Mode::Bool:
      a.bools[i]   = static_cast<unsigned char>(
                       get<bool>(v._impl->val) ? 1 : 0);
      break;
    case ArrayStorage::Mode::Int:
      a.ints[i]    = get<int64_t>(v._impl->val);
      break;
    case ArrayStorage::Mode::Uint:
      a.uints[i]   = get<uint64_t>(v._impl->val);
      break;
    case ArrayStorage::Mode::Real:
      a.reals[i]   = get<double>(v._impl->val);
      break;
    case ArrayStorage::Mode::String:
      a.strings[i] = get<string>(std::move(v._impl->val));
      break;
    case ArrayStorage::Mode::Null:
      /* nothing to store; size unchanged */
      break;
    default: break;
  }
  return true;
}

// =============================================================================
// ConstObjectView / ObjectView
// =============================================================================

FlexData::ConstObjectView::ConstObjectView(const Impl* obj) noexcept
  : _obj(obj) {}

FlexData::ConstObjectView::iterator::iterator() noexcept
  : _obj(nullptr), _idx(0) {}
FlexData::ConstObjectView::iterator::iterator
  (const Impl* obj, size_t idx) noexcept
  : _obj(obj), _idx(idx) {}

FlexData::ConstObjectView::Entry
FlexData::ConstObjectView::iterator::operator*() const {
  const auto& o = as_object_store(_obj);
  return Entry(string_view{o.entries[_idx].first},
               FlexData{o.entries[_idx].second->clone()});
}
FlexData::ConstObjectView::iterator&
FlexData::ConstObjectView::iterator::operator++() noexcept
  { ++_idx; return *this; }
FlexData::ConstObjectView::iterator
FlexData::ConstObjectView::iterator::operator++(int) noexcept
  { iterator t = *this; ++_idx; return t; }
bool FlexData::ConstObjectView::iterator::operator==
  (const iterator& o) const noexcept
  { return _obj == o._obj && _idx == o._idx; }
bool FlexData::ConstObjectView::iterator::operator!=
  (const iterator& o) const noexcept
  { return !(*this == o); }

FlexData::ConstObjectView::iterator
FlexData::ConstObjectView::begin() const noexcept
  { return iterator{_obj, 0}; }
FlexData::ConstObjectView::iterator
FlexData::ConstObjectView::end()   const noexcept
  { return iterator{_obj, as_object_store(_obj).entries.size()}; }
size_t FlexData::ConstObjectView::size()  const noexcept
  { return as_object_store(_obj).entries.size(); }
bool   FlexData::ConstObjectView::empty() const noexcept
  { return size() == 0; }
bool   FlexData::ConstObjectView::contains(string_view key) const noexcept {
  const auto& o = as_object_store(_obj);
  return o.index.find(key) != o.index.end();
}
FlexData::ConstObjectView::iterator
FlexData::ConstObjectView::find(string_view key) const noexcept {
  const auto& o = as_object_store(_obj);
  auto it = o.index.find(key);
  return it == o.index.end() ? end() : iterator{_obj, it->second};
}
FlexData FlexData::ConstObjectView::at(string_view key) const {
  const auto& o = as_object_store(_obj);
  auto it = o.index.find(key);
  if (it == o.index.end()) {
    throw out_of_range("FlexData::ObjectView::at: missing key");
  }
  return FlexData{o.entries[it->second].second->clone()};
}

FlexData::ObjectView::ObjectView(Impl* obj) noexcept
  : ConstObjectView(obj), _mut(obj) {}

bool FlexData::ObjectView::insert(string_view key, FlexData val) {
  auto& o = as_object_store_mut(_mut);
  if (o.index.find(key) != o.index.end()) return false;
  size_t idx = o.entries.size();
  o.entries.emplace_back(string{key}, val._impl ? std::move(val._impl) :
                                                  make_kind(FlexKind::Null));
  o.index.emplace(o.entries.back().first, idx);
  return true;
}

void FlexData::ObjectView::insert_or_assign(string_view key, FlexData val) {
  auto& o = as_object_store_mut(_mut);
  auto it = o.index.find(key);
  if (it == o.index.end()) {
    size_t idx = o.entries.size();
    o.entries.emplace_back(string{key}, val._impl ? std::move(val._impl) :
                                                    make_kind(FlexKind::Null));
    o.index.emplace(o.entries.back().first, idx);
  } else {
    o.entries[it->second].second = val._impl ? std::move(val._impl) :
                                               make_kind(FlexKind::Null);
  }
}

bool FlexData::ObjectView::erase(string_view key) {
  auto& o = as_object_store_mut(_mut);
  auto it = o.index.find(key);
  if (it == o.index.end()) return false;
  size_t pos = it->second;
  o.index.erase(it);
  o.entries.erase(o.entries.begin() + static_cast<ptrdiff_t>(pos));
  for (auto& p : o.index) {
    if (p.second > pos) --p.second;
  }
  return true;
}

void FlexData::ObjectView::clear() noexcept {
  auto& o = as_object_store_mut(_mut);
  o.entries.clear();
  o.index.clear();
}

// =============================================================================
// JSON serializer (standard JSON; Int/Uint indistinguishable on the wire)
// =============================================================================

namespace {
void escape_json_string(string_view s, ostream& out) {
  out.put('"');
  for (char ch : s) {
    auto uc = static_cast<unsigned char>(ch);
    switch (ch) {
      case '"':  out.write("\\\"", 2); break;
      case '\\': out.write("\\\\", 2); break;
      case '\b': out.write("\\b",  2); break;
      case '\f': out.write("\\f",  2); break;
      case '\n': out.write("\\n",  2); break;
      case '\r': out.write("\\r",  2); break;
      case '\t': out.write("\\t",  2); break;
      default:
        if (uc < 0x20) {
          char buf[8];
          int n = snprintf(buf, sizeof(buf), "\\u%04x",
                           static_cast<unsigned>(uc));
          out.write(buf, n);
        } else {
          out.put(ch);
        }
    }
  }
  out.put('"');
}

void emit_indent(ostream& out, int depth) {
  for (int i = 0; i < depth; ++i) out.write("  ", 2);
}

void serialize_real(double v, ostream& out) {
  if (!std::isfinite(v)) { out.write("null", 4); return; }
  char buf[64];
  auto res = std::to_chars(buf, buf + sizeof(buf), v,
                           std::chars_format::general);
  if (res.ec != errc{}) { out.write("null", 4); return; }
  out.write(buf, res.ptr - buf);
}
void serialize_int(int64_t v, ostream& out) {
  char buf[32];
  auto res = std::to_chars(buf, buf + sizeof(buf), v);
  out.write(buf, res.ptr - buf);
}
void serialize_uint(uint64_t v, ostream& out) {
  char buf[32];
  auto res = std::to_chars(buf, buf + sizeof(buf), v);
  out.write(buf, res.ptr - buf);
}

void
serialize(const FlexData::Impl& impl, ostream& out, int depth, bool pretty);

void
serialize_array(const ArrayStorage& a, ostream& out, int depth, bool pretty) {
  size_t n = array_size(a);
  if (n == 0) { out.write("[]", 2); return; }
  out.put('[');
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) out.put(',');
    if (pretty) { out.put('\n'); emit_indent(out, depth + 1); }
    switch (a.mode) {
      case ArrayStorage::Mode::Hetero:
        serialize(*a.hetero[i], out, depth + 1, pretty);
        break;
      case ArrayStorage::Mode::Bool:
        out.write((a.bools[i] != 0) ? "true" : "false",
                  (a.bools[i] != 0) ? 4 : 5);
        break;
      case ArrayStorage::Mode::Int:
        serialize_int(a.ints[i], out);
        break;
      case ArrayStorage::Mode::Uint:
        serialize_uint(a.uints[i], out);
        break;
      case ArrayStorage::Mode::Real:
        serialize_real(a.reals[i], out);
        break;
      case ArrayStorage::Mode::String:
        escape_json_string(a.strings[i], out);
        break;
      case ArrayStorage::Mode::Null:
        out.write("null", 4);
        break;
    }
  }
  if (pretty) { out.put('\n'); emit_indent(out, depth); }
  out.put(']');
}

void
serialize_object(const ObjectStorage& o, ostream& out, int depth, bool pretty) {
  if (o.entries.empty()) { out.write("{}", 2); return; }
  out.put('{');
  for (size_t i = 0; i < o.entries.size(); ++i) {
    if (i > 0) out.put(',');
    if (pretty) { out.put('\n'); emit_indent(out, depth + 1); }
    escape_json_string(o.entries[i].first, out);
    out.put(':');
    if (pretty) out.put(' ');
    serialize(*o.entries[i].second, out, depth + 1, pretty);
  }
  if (pretty) { out.put('\n'); emit_indent(out, depth); }
  out.put('}');
}

void
serialize(const FlexData::Impl& impl, ostream& out, int depth, bool pretty) {
  switch (impl.kind) {
    case FlexKind::Null:
      out.write("null", 4);
      break;
    case FlexKind::Bool:
      {
        bool b = get<bool>(impl.val);
        out.write(b ? "true" : "false", b ? 4 : 5);
      }
      break;
    case FlexKind::Int:
      serialize_int(get<int64_t>(impl.val), out);
      break;
    case FlexKind::Uint:
      serialize_uint(get<uint64_t>(impl.val), out);
      break;
    case FlexKind::Real:
      serialize_real(get<double>(impl.val), out);
      break;
    case FlexKind::String:
      escape_json_string(get<string>(impl.val), out);
      break;
    case FlexKind::Array:
      serialize_array(get<ArrayStorage>(impl.val), out, depth, pretty);
      break;
    case FlexKind::Object:
      serialize_object(get<ObjectStorage>(impl.val), out, depth, pretty);
      break;
  }
}
}  // anonymous namespace

void FlexData::to_json(ostream& out, bool pretty) const {
  if (!_impl) { out.write("null", 4); return; }
  serialize(*_impl, out, 0, pretty);
}

string FlexData::to_json(bool pretty) const {
  ostringstream oss;
  to_json(oss, pretty);
  return std::move(oss).str();
}

// =============================================================================
// JSON5 parser
// =============================================================================

namespace {

// Char-by-char source abstraction so the parser can read either a string_view
// (zero-copy) or an istream (small bounded-buffer streaming, never the whole
// input in memory).
class CharSource {
public:
  virtual ~CharSource() = default;
  virtual bool eof() const = 0;               // true when no more chars
  virtual char peek() const = 0;              // UB if eof()
  virtual char peek_at(size_t off) const = 0; // returns '\0' past end
  virtual bool has_at(size_t off) const = 0;  // true if a char is available at offset
  virtual void advance() = 0;                 // consumes one
};

class StringViewSource : public CharSource {
public:
  explicit StringViewSource(string_view s) : _s(s) {}
  bool eof()                  const override { return _pos >= _s.size(); }
  char peek()                 const override { return _s[_pos]; }
  char peek_at(size_t off)    const override
    { return (_pos + off) < _s.size() ? _s[_pos + off] : '\0'; }
  bool has_at(size_t off)     const override { return _pos + off <  _s.size(); }
  void advance()                    override { if (_pos < _s.size()) ++_pos; }
private:
  string_view _s;
  size_t      _pos = 0;
};

class IstreamSource : public CharSource {
public:
  explicit IstreamSource(istream& is) : _is(is) {}
  bool eof()               const override { return !ensure(1); }
  char peek()              const override
    { ensure(1); return _len > 0 ? _buf[_head] : '\0'; }
  char peek_at(size_t off) const override {
    ensure(off + 1);
    return _len > off ? _buf[(_head + off) % kBuf] : '\0';
  }
  bool has_at(size_t off)  const override { return ensure(off + 1); }
  void advance()                 override {
    if (!ensure(1)) return;
    _head = (_head + 1) % kBuf;
    --_len;
  }
private:
  // Need at most ~8 chars of lookahead (e.g. "Infinity"); 64 leaves headroom.
  static constexpr size_t kBuf = 64;
  bool ensure(size_t n) const {
    if (n > kBuf) return false;
    while (_len < n) {
      char c;
      if (!_is.get(c)) break;
      _buf[(_head + _len) % kBuf] = c;
      ++_len;
    }
    return _len >= n;
  }
  istream&       _is;
  mutable char   _buf[kBuf]{};
  mutable size_t _head = 0;
  mutable size_t _len  = 0;
};

class Parser {
public:
  explicit Parser(CharSource& src) : _source(src), _line(1), _col(1) {}

  unique_ptr<FlexData::Impl> parse_top() {
    skip_ws();
    if (eof()) throw FlexData::ParseError(_line, _col, "empty input");
    auto v = parse_value();
    skip_ws();
    if (!eof()) throw FlexData::ParseError(_line, _col,
                                           "trailing characters after value");
    return v;
  }

private:
  CharSource& _source;
  size_t      _line;
  size_t      _col;

  bool eof() const { return _source.eof(); }
  char peek() const { return _source.peek(); }
  char peek_at(size_t off) const { return _source.peek_at(off); }
  void advance() {
    if (eof()) return;
    char c = _source.peek();
    if (c == '\n') { ++_line; _col = 1; }
    else           { ++_col; }
    _source.advance();
  }

  [[noreturn]] void fail(const string& msg) const {
    throw FlexData::ParseError(_line, _col, msg);
  }

  static bool is_hex_digit(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
           (c >= 'A' && c <= 'F');
  }
  static bool is_ident_start(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           c == '_' || c == '$';
  }
  static bool is_ident_cont(char c) {
    return is_ident_start(c) || (c >= '0' && c <= '9');
  }

  void skip_ws() {
    while (!eof()) {
      char c = peek();
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        { advance(); continue; }
      if (c == '/' && _source.has_at(1)) {
        char d = peek_at(1);
        if (d == '/') {
          advance(); advance();
          while (!eof() && peek() != '\n') advance();
          continue;
        }
        if (d == '*') {
          advance(); advance();
          while (!eof()) {
            if (peek() == '*' && peek_at(1) == '/')
              { advance(); advance(); break; }
            advance();
          }
          continue;
        }
      }
      return;
    }
  }

  unique_ptr<FlexData::Impl> parse_value() {
    skip_ws();
    if (eof()) fail("expected value, got end of input");
    char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"' || c == '\'') {
      auto p  = make_unique<FlexData::Impl>();
      p->kind = FlexKind::String;
      p->val  = parse_string_lit();
      return p;
    }
    if (c == 't' || c == 'f') return parse_bool_keyword();
    if (c == 'n')             return parse_null_keyword();
    if (c == 'I' || c == 'N') return parse_special_number(false);
    if (c == '-' || c == '+' || (c >= '0' && c <= '9') || c == '.') {
      return parse_number();
    }
    fail(string{"unexpected character '"} + c + "'");
  }

  unique_ptr<FlexData::Impl> parse_object() {
    advance();  // '{'
    auto p  = make_unique<FlexData::Impl>();
    p->kind = FlexKind::Object;
    p->val  = ObjectStorage{};
    auto& o = get<ObjectStorage>(p->val);
    skip_ws();
    if (!eof() && peek() == '}') { advance(); return p; }
    while (true) {
      skip_ws();
      if (eof()) fail("unterminated object");
      string key = parse_key();
      skip_ws();
      if (eof() || peek() != ':') fail("expected ':' after object key");
      advance();
      auto val = parse_value();
      // last-write-wins semantics for duplicate keys
      auto idxIt = o.index.find(key);
      if (idxIt == o.index.end()) {
        size_t idx = o.entries.size();
        o.entries.emplace_back(key, std::move(val));
        o.index.emplace(o.entries.back().first, idx);
      } else {
        o.entries[idxIt->second].second = std::move(val);
      }
      skip_ws();
      if (eof()) fail("unterminated object");
      if (peek() == ',') {
        advance();
        skip_ws();
        if (!eof() && peek() == '}') { advance(); return p; }
        continue;
      }
      if (peek() == '}') { advance(); return p; }
      fail("expected ',' or '}' in object");
    }
  }

  unique_ptr<FlexData::Impl> parse_array() {
    advance();  // '['
    auto p  = make_unique<FlexData::Impl>();
    p->kind = FlexKind::Array;
    p->val  = ArrayStorage{};
    auto& a = get<ArrayStorage>(p->val);
    skip_ws();
    if (!eof() && peek() == ']') { advance(); return p; }
    while (true) {
      a.hetero.push_back(parse_value());
      skip_ws();
      if (eof()) fail("unterminated array");
      if (peek() == ',') {
        advance();
        skip_ws();
        if (!eof() && peek() == ']') { advance(); break; }
        continue;
      }
      if (peek() == ']') { advance(); break; }
      fail("expected ',' or ']' in array");
    }
    homogenize_array(a);
    return p;
  }

  static void homogenize_array(ArrayStorage& a) {
    if (a.hetero.empty()) return;
    FlexKind k = a.hetero.front()->kind;
    for (auto& p : a.hetero) {
      if (p->kind != k) return;
    }
    switch (k) {
      case FlexKind::Bool:
        a.bools.reserve(a.hetero.size());
        for (auto& p : a.hetero) {
          a.bools.push_back(static_cast<unsigned char>(
                              get<bool>(p->val) ? 1 : 0));
        }
        a.hetero.clear(); a.hetero.shrink_to_fit();
        a.mode = ArrayStorage::Mode::Bool;
        break;
      case FlexKind::Int:
        a.ints.reserve(a.hetero.size());
        for (auto& p : a.hetero) a.ints.push_back(get<int64_t>(p->val));
        a.hetero.clear(); a.hetero.shrink_to_fit();
        a.mode = ArrayStorage::Mode::Int;
        break;
      case FlexKind::Uint:
        a.uints.reserve(a.hetero.size());
        for (auto& p : a.hetero) a.uints.push_back(get<uint64_t>(p->val));
        a.hetero.clear(); a.hetero.shrink_to_fit();
        a.mode = ArrayStorage::Mode::Uint;
        break;
      case FlexKind::Real:
        a.reals.reserve(a.hetero.size());
        for (auto& p : a.hetero) a.reals.push_back(get<double>(p->val));
        a.hetero.clear(); a.hetero.shrink_to_fit();
        a.mode = ArrayStorage::Mode::Real;
        break;
      case FlexKind::String:
        a.strings.reserve(a.hetero.size());
        for (auto& p : a.hetero) {
          a.strings.push_back(get<string>(std::move(p->val)));
        }
        a.hetero.clear(); a.hetero.shrink_to_fit();
        a.mode = ArrayStorage::Mode::String;
        break;
      case FlexKind::Null:
        a.null_count = a.hetero.size();
        a.hetero.clear(); a.hetero.shrink_to_fit();
        a.mode = ArrayStorage::Mode::Null;
        break;
      default:
        break;
    }
  }

  string parse_key() {
    if (eof()) fail("expected object key");
    char c = peek();
    if (c == '"' || c == '\'') return parse_string_lit();
    if (!is_ident_start(c)) fail("expected object key (string or identifier)");
    string key;
    key.push_back(peek());
    advance();
    while (!eof() && is_ident_cont(peek())) {
      key.push_back(peek());
      advance();
    }
    return key;
  }

  string parse_string_lit() {
    char quote = peek();
    advance();
    string out;
    while (!eof()) {
      char c = peek();
      if (c == quote) { advance(); return out; }
      if (c == '\n')  fail("unterminated string (newline in string literal)");
      if (c == '\\') {
        advance();
        if (eof()) fail("unterminated string escape");
        char e = peek();
        advance();
        switch (e) {
          case '"':  out.push_back('"');  break;
          case '\'': out.push_back('\''); break;
          case '\\': out.push_back('\\'); break;
          case '/':  out.push_back('/');  break;
          case 'b':  out.push_back('\b'); break;
          case 'f':  out.push_back('\f'); break;
          case 'n':  out.push_back('\n'); break;
          case 'r':  out.push_back('\r'); break;
          case 't':  out.push_back('\t'); break;
          case 'v':  out.push_back('\v'); break;
          case '0':  out.push_back('\0'); break;
          case '\n': /* line continuation */ break;
          case 'x': {
            char buf[3] = {0,0,0};
            for (int k = 0; k < 2; ++k) {
              if (eof() || !is_hex_digit(peek())) fail("invalid \\x escape");
              buf[k] = peek(); advance();
            }
            unsigned v = 0;
            std::from_chars(buf, buf + 2, v, 16);
            out.push_back(static_cast<char>(v));
            break;
          }
          case 'u': {
            char buf[5] = {0,0,0,0,0};
            for (int k = 0; k < 4; ++k) {
              if (eof() || !is_hex_digit(peek())) fail("invalid \\u escape");
              buf[k] = peek(); advance();
            }
            unsigned cp = 0;
            std::from_chars(buf, buf + 4, cp, 16);
            if (cp < 0x80) {
              out.push_back(static_cast<char>(cp));
            } else if (cp < 0x800) {
              out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
              out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
              out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
              out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
              out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            break;
          }
          default:
            out.push_back(e);
            break;
        }
      } else {
        out.push_back(c);
        advance();
      }
    }
    fail("unterminated string");
  }

  bool consume_word(string_view w) {
    for (size_t i = 0; i < w.size(); ++i) {
      if (!_source.has_at(i)) return false;
      if (_source.peek_at(i) != w[i]) return false;
    }
    if (_source.has_at(w.size()) && is_ident_cont(_source.peek_at(w.size()))) {
      return false;
    }
    for (size_t i = 0; i < w.size(); ++i) advance();
    return true;
  }

  unique_ptr<FlexData::Impl> parse_bool_keyword() {
    if (consume_word("true")) {
      auto p = make_unique<FlexData::Impl>();
      p->kind = FlexKind::Bool; p->val = true;
      return p;
    }
    if (consume_word("false")) {
      auto p = make_unique<FlexData::Impl>();
      p->kind = FlexKind::Bool; p->val = false;
      return p;
    }
    fail("invalid token");
  }
  unique_ptr<FlexData::Impl> parse_null_keyword() {
    if (consume_word("null")) {
      auto p = make_unique<FlexData::Impl>();
      p->kind = FlexKind::Null;
      return p;
    }
    fail("invalid token");
  }

  unique_ptr<FlexData::Impl> parse_special_number(bool negative) {
    if (consume_word("Infinity")) {
      auto p  = make_unique<FlexData::Impl>();
      p->kind = FlexKind::Real;
      p->val  = negative ? -std::numeric_limits<double>::infinity()
                         :  std::numeric_limits<double>::infinity();
      return p;
    }
    if (consume_word("NaN")) {
      auto p  = make_unique<FlexData::Impl>();
      p->kind = FlexKind::Real;
      p->val  = std::numeric_limits<double>::quiet_NaN();
      return p;
    }
    fail("invalid token");
  }

  unique_ptr<FlexData::Impl> parse_number() {
    bool neg = false;
    if (peek() == '+') advance();
    else if (peek() == '-') { neg = true; advance(); }

    if (!eof() && (peek() == 'I' || peek() == 'N')) {
      return parse_special_number(neg);
    }

    // Hex
    if (!eof() && peek() == '0' && (peek_at(1) == 'x' || peek_at(1) == 'X')) {
      advance(); advance();
      string digits;
      while (!eof() && is_hex_digit(peek()))
        { digits.push_back(peek()); advance(); }
      if (digits.empty()) fail("invalid hex literal");
      uint64_t mag = 0;
      auto res = std::from_chars(digits.data(), digits.data() + digits.size(),
                                 mag, 16);
      if (res.ec == errc::result_out_of_range) {
        fail("hex literal exceeds 64-bit range");
      }
      if (res.ec != errc{} || res.ptr != digits.data() + digits.size()) {
        fail("invalid hex literal");
      }
      auto p = make_unique<FlexData::Impl>();
      if (neg) {
        constexpr uint64_t kMaxNeg
          = static_cast<uint64_t>(numeric_limits<int64_t>::max()) + 1ULL;
        if (mag > kMaxNeg) fail("negative hex literal exceeds int64 range");
        int64_t v = (mag == kMaxNeg) ? numeric_limits<int64_t>::min() :
                                       -static_cast<int64_t>(mag);
        p->kind = FlexKind::Int;
        p->val  = v;
      } else if (mag <= static_cast<uint64_t>(numeric_limits<int64_t>::max())) {
        p->kind = FlexKind::Int;
        p->val  = static_cast<int64_t>(mag);
      } else {
        p->kind = FlexKind::Uint;
        p->val  = mag;
      }
      return p;
    }

    // Decimal
    string num;
    bool   has_int  = false;
    bool   has_dot  = false;
    bool   has_frac = false;
    bool   has_exp  = false;
    while (!eof() && peek() >= '0' && peek() <= '9')
      { num.push_back(peek()); advance(); has_int = true; }
    if (!eof() && peek() == '.') {
      num.push_back('.');
      advance();
      has_dot = true;
      while (!eof() && peek() >= '0' && peek() <= '9')
        { num.push_back(peek()); advance(); has_frac = true; }
    }
    if (!eof() && (peek() == 'e' || peek() == 'E')) {
      num.push_back(peek());
      advance();
      has_exp = true;
      if (!eof() && (peek() == '+' || peek() == '-'))
        { num.push_back(peek()); advance(); }
      bool has_exp_digit = false;
      while (!eof() && peek() >= '0' && peek() <= '9')
        { num.push_back(peek()); advance(); has_exp_digit = true; }
      if (!has_exp_digit) fail("expected exponent digits");
    }
    if (!has_int && !has_frac) fail("invalid number");

    if (!has_dot && !has_exp) {
      // Integer
      uint64_t mag = 0;
      auto res = std::from_chars(num.data(), num.data() + num.size(), mag);
      if (res.ec == errc::result_out_of_range) {
        fail("integer literal exceeds 64-bit range");
      }
      if (res.ec != errc{} || res.ptr != num.data() + num.size()) {
        fail("invalid integer literal");
      }
      auto p = make_unique<FlexData::Impl>();
      if (neg) {
        constexpr uint64_t kMaxNeg
          = static_cast<uint64_t>(numeric_limits<int64_t>::max()) + 1ULL;
        if (mag > kMaxNeg) fail("negative integer exceeds int64 range");
        int64_t v = (mag == kMaxNeg) ? numeric_limits<int64_t>::min() :
                                       -static_cast<int64_t>(mag);
        p->kind = FlexKind::Int;
        p->val  = v;
      } else if (mag <= static_cast<uint64_t>(numeric_limits<int64_t>::max())) {
        p->kind = FlexKind::Int;
        p->val  = static_cast<int64_t>(mag);
      } else {
        p->kind = FlexKind::Uint;
        p->val  = mag;
      }
      return p;
    }

    // Real -- normalize ".5" / "5." / "5.e10" if from_chars rejects them
    double v   = 0.0;
    auto   res = std::from_chars(num.data(), num.data() + num.size(), v,
                                 std::chars_format::general);
    if (res.ec != errc{} || res.ptr != num.data() + num.size()) {
      string norm{num};
      if (!norm.empty() && norm.front() == '.') norm.insert(norm.begin(), '0');
      size_t dot = norm.find('.');
      if (dot != string::npos) {
        size_t after = dot + 1;
        if (after >= norm.size() || norm[after] == 'e' || norm[after] == 'E') {
          norm.insert(after, "0");
        }
      }
      res = std::from_chars(norm.data(), norm.data() + norm.size(), v,
                            std::chars_format::general);
      if (res.ec != errc{} || res.ptr != norm.data() + norm.size())
        fail("invalid real literal");
    }
    if (neg) v = -v;
    auto p  = make_unique<FlexData::Impl>();
    p->kind = FlexKind::Real;
    p->val  = v;
    return p;
  }
};

}  // anonymous namespace

FlexData FlexData::from_json(string_view src) {
  StringViewSource source{src};
  Parser           parser{source};
  return FlexData{parser.parse_top()};
}

FlexData FlexData::from_json(istream& in) {
  IstreamSource source{in};
  Parser        parser{source};
  return FlexData{parser.parse_top()};
}

// =============================================================================
// native binary format
// =============================================================================
//
// Stream header (8 B): magic 'V','P','F','D' (u32 LE), version (u16), flags
// (u16). Followed by exactly one root record at offset 8.
//
// Record header (standard, 8 B):
//   [tag:u8][sub_mode:u8][reserved:u16][count:u32]
// Record header (extended, 16 B; tag bit 7 set):
//   [tag:u8][sub_mode:u8][reserved:u16][reserved:u32][count:u64]
//
// Tag low 7 bits: 0=Null, 1=BoolFalse, 2=BoolTrue, 3=Int, 4=Uint, 5=Real,
//                 6=String, 7=Array, 8=Object.
// Array sub_mode: 0=Hetero, 1=Int, 2=Uint, 3=Real, 4=Bool, 5=String, 6=Null.
//
// Every record header starts at an 8-byte-aligned offset relative to the
// stream start; every record self-pads its payload tail with zero bytes to
// the next 8-byte boundary.

namespace {

static_assert(std::endian::native == std::endian::little,
              "FlexData binary format requires a little-endian host.");

constexpr uint32_t kMagic   = 0x44465056u;  // 'V','P','F','D' in little-endian
constexpr uint16_t kVersion = 1;

constexpr uint8_t kTagExtCount  = 0x80;
constexpr uint8_t kTagMask      = 0x7f;
constexpr uint8_t kTagNull      = 0x00;
constexpr uint8_t kTagBoolFalse = 0x01;
constexpr uint8_t kTagBoolTrue  = 0x02;
constexpr uint8_t kTagInt       = 0x03;
constexpr uint8_t kTagUint      = 0x04;
constexpr uint8_t kTagReal      = 0x05;
constexpr uint8_t kTagString    = 0x06;
constexpr uint8_t kTagArray     = 0x07;
constexpr uint8_t kTagObject    = 0x08;

constexpr uint8_t kSubHetero = 0x00;
constexpr uint8_t kSubInt    = 0x01;
constexpr uint8_t kSubUint   = 0x02;
constexpr uint8_t kSubReal   = 0x03;
constexpr uint8_t kSubBool   = 0x04;
constexpr uint8_t kSubString = 0x05;
constexpr uint8_t kSubNull   = 0x06;

inline size_t pad_to_8(size_t n) { return (8 - (n & 7)) & 7; }

// ----- writer ---------------------------------------------------------------

class BinarySink {
public:
  explicit BinarySink(ostream& out) : _out(out) {}

  void write_bytes(const void* p, size_t n) {
    _out.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
    _pos += n;
  }
  void pad_to_align() {
    static constexpr char zeros[8] = {};
    size_t pad = pad_to_8(_pos);
    if (pad) write_bytes(zeros, pad);
  }
  void write_header(uint8_t tag, uint8_t sub, uint64_t count) {
    if (count <= numeric_limits<uint32_t>::max()) {
      uint8_t hdr[8] = {};
      hdr[0] = tag;
      hdr[1] = sub;
      uint32_t c = static_cast<uint32_t>(count);
      memcpy(hdr + 4, &c, 4);
      write_bytes(hdr, 8);
    } else {
      uint8_t hdr[16] = {};
      hdr[0] = tag | kTagExtCount;
      hdr[1] = sub;
      memcpy(hdr + 8, &count, 8);
      write_bytes(hdr, 16);
    }
  }
  template<class T> void write_scalar(T v) {
    static_assert(sizeof(T) == 8);
    write_bytes(&v, 8);
  }
  size_t pos() const { return _pos; }

private:
  ostream& _out;
  size_t   _pos = 0;
};

// ----- reader ---------------------------------------------------------------

class BinarySource {
public:
  virtual ~BinarySource() = default;
  virtual bool read_bytes(void* dst, size_t n) = 0;
  size_t pos() const { return _pos; }

protected:
  size_t _pos = 0;
};

class StringViewBinarySource : public BinarySource {
public:
  explicit StringViewBinarySource(string_view s) : _s(s) {}
  bool read_bytes(void* dst, size_t n) override {
    if (_pos + n > _s.size()) {
      size_t avail = _s.size() - _pos;
      if (avail) memcpy(dst, _s.data() + _pos, avail);
      _pos += avail;
      return false;
    }
    if (n) memcpy(dst, _s.data() + _pos, n);
    _pos += n;
    return true;
  }

private:
  string_view _s;
};

class IstreamBinarySource : public BinarySource {
public:
  explicit IstreamBinarySource(istream& in) : _in(in) {}
  bool read_bytes(void* dst, size_t n) override {
    if (n == 0) return true;
    _in.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    auto got = static_cast<size_t>(_in.gcount());
    _pos += got;
    return got == n;
  }

private:
  istream& _in;
};

[[noreturn]] void bin_throw(const BinarySource& src, const char* msg) {
  throw FlexData::BinaryError(src.pos(), msg);
}

void read_or_throw(BinarySource& src, void* dst, size_t n, const char* what) {
  if (!src.read_bytes(dst, n)) bin_throw(src, what);
}

void skip_or_throw(BinarySource& src, size_t n, const char* what) {
  uint8_t buf[64];
  while (n > 0) {
    size_t k = n < sizeof(buf) ? n : sizeof(buf);
    if (!src.read_bytes(buf, k)) bin_throw(src, what);
    n -= k;
  }
}

void pad_skip(BinarySource& src) {
  size_t pad = pad_to_8(src.pos());
  if (pad) skip_or_throw(src, pad, "truncated record padding");
}

struct RecordHeader {
  uint8_t  tag;       // low 7 bits, ext flag already stripped
  uint8_t  sub_mode;
  uint64_t count;
};

RecordHeader read_record_header(BinarySource& src) {
  uint8_t hdr[8];
  read_or_throw(src, hdr, 8, "truncated record header");
  if (hdr[2] != 0 || hdr[3] != 0)
    bin_throw(src, "non-zero reserved bytes in record header");
  RecordHeader rh;
  rh.tag      = hdr[0] & kTagMask;
  rh.sub_mode = hdr[1];
  bool ext    = (hdr[0] & kTagExtCount) != 0;
  if (ext) {
    if (hdr[4] != 0 || hdr[5] != 0 || hdr[6] != 0 || hdr[7] != 0)
      bin_throw(src, "non-zero pad in extended record header");
    read_or_throw(src, &rh.count, 8, "truncated extended record header");
  } else {
    uint32_t c32;
    memcpy(&c32, hdr + 4, 4);
    rh.count = c32;
  }
  return rh;
}

// ----- encoder --------------------------------------------------------------

void serialize_bin(const FlexData::Impl& impl, BinarySink& sink);

void serialize_bin_string(string_view s, BinarySink& sink) {
  sink.write_header(kTagString, 0, s.size());
  if (!s.empty()) sink.write_bytes(s.data(), s.size());
  sink.pad_to_align();
}

void serialize_bin_array(const ArrayStorage& a, BinarySink& sink) {
  size_t n = array_size(a);
  switch (a.mode) {
    case ArrayStorage::Mode::Hetero:
      sink.write_header(kTagArray, kSubHetero, n);
      for (auto& p : a.hetero) serialize_bin(*p, sink);
      break;
    case ArrayStorage::Mode::Int:
      sink.write_header(kTagArray, kSubInt, n);
      if (n) sink.write_bytes(a.ints.data(), n * 8);
      break;
    case ArrayStorage::Mode::Uint:
      sink.write_header(kTagArray, kSubUint, n);
      if (n) sink.write_bytes(a.uints.data(), n * 8);
      break;
    case ArrayStorage::Mode::Real:
      sink.write_header(kTagArray, kSubReal, n);
      if (n) sink.write_bytes(a.reals.data(), n * 8);
      break;
    case ArrayStorage::Mode::Bool:
      sink.write_header(kTagArray, kSubBool, n);
      if (n) sink.write_bytes(a.bools.data(), n);
      sink.pad_to_align();
      break;
    case ArrayStorage::Mode::String:
      sink.write_header(kTagArray, kSubString, n);
      for (auto& s : a.strings) serialize_bin_string(s, sink);
      break;
    case ArrayStorage::Mode::Null:
      sink.write_header(kTagArray, kSubNull, n);
      break;
  }
}

void serialize_bin_object(const ObjectStorage& o, BinarySink& sink) {
  sink.write_header(kTagObject, 0, o.entries.size());
  for (auto& [k, v] : o.entries) {
    serialize_bin_string(k, sink);
    serialize_bin(*v, sink);
  }
}

void serialize_bin(const FlexData::Impl& impl, BinarySink& sink) {
  switch (impl.kind) {
    case FlexKind::Null:
      sink.write_header(kTagNull, 0, 0);
      break;
    case FlexKind::Bool:
      sink.write_header(get<bool>(impl.val) ? kTagBoolTrue : kTagBoolFalse,
                        0, 0);
      break;
    case FlexKind::Int:
      sink.write_header(kTagInt, 0, 0);
      sink.write_scalar(get<int64_t>(impl.val));
      break;
    case FlexKind::Uint:
      sink.write_header(kTagUint, 0, 0);
      sink.write_scalar(get<uint64_t>(impl.val));
      break;
    case FlexKind::Real:
      sink.write_header(kTagReal, 0, 0);
      sink.write_scalar(get<double>(impl.val));
      break;
    case FlexKind::String:
      serialize_bin_string(get<string>(impl.val), sink);
      break;
    case FlexKind::Array:
      serialize_bin_array(get<ArrayStorage>(impl.val), sink);
      break;
    case FlexKind::Object:
      serialize_bin_object(get<ObjectStorage>(impl.val), sink);
      break;
  }
}

// ----- decoder --------------------------------------------------------------

unique_ptr<FlexData::Impl> parse_bin(BinarySource& src);

string parse_bin_string_inner(BinarySource& src, uint64_t byte_len) {
  string s;
  s.resize(byte_len);
  if (byte_len) read_or_throw(src, s.data(), byte_len, "truncated string");
  pad_skip(src);
  return s;
}

unique_ptr<FlexData::Impl> parse_bin_array(BinarySource& src,
                                           uint8_t sub_mode,
                                           uint64_t count) {
  auto out  = make_kind(FlexKind::Array);
  out->val  = ArrayStorage{};
  auto& a   = get<ArrayStorage>(out->val);
  switch (sub_mode) {
    case kSubHetero:
      a.mode = ArrayStorage::Mode::Hetero;
      a.hetero.reserve(count);
      for (uint64_t i = 0; i < count; ++i) a.hetero.push_back(parse_bin(src));
      break;
    case kSubInt:
      a.mode = ArrayStorage::Mode::Int;
      a.ints.resize(count);
      if (count) read_or_throw(src, a.ints.data(), count * 8,
                               "truncated int array");
      break;
    case kSubUint:
      a.mode = ArrayStorage::Mode::Uint;
      a.uints.resize(count);
      if (count) read_or_throw(src, a.uints.data(), count * 8,
                               "truncated uint array");
      break;
    case kSubReal:
      a.mode = ArrayStorage::Mode::Real;
      a.reals.resize(count);
      if (count) read_or_throw(src, a.reals.data(), count * 8,
                               "truncated real array");
      break;
    case kSubBool:
      a.mode = ArrayStorage::Mode::Bool;
      a.bools.resize(count);
      if (count) read_or_throw(src, a.bools.data(), count,
                               "truncated bool array");
      pad_skip(src);
      break;
    case kSubString:
      a.mode = ArrayStorage::Mode::String;
      a.strings.reserve(count);
      for (uint64_t i = 0; i < count; ++i) {
        auto rh = read_record_header(src);
        if (rh.tag != kTagString)
          bin_throw(src, "expected string record in string array");
        if (rh.sub_mode != 0)
          bin_throw(src, "non-zero sub_mode in string record");
        a.strings.push_back(parse_bin_string_inner(src, rh.count));
      }
      break;
    case kSubNull:
      a.mode       = ArrayStorage::Mode::Null;
      a.null_count = count;
      break;
    default:
      bin_throw(src, "unknown array sub-mode");
  }
  return out;
}

unique_ptr<FlexData::Impl>
parse_bin_object(BinarySource& src, uint64_t count) {
  auto out = make_kind(FlexKind::Object);
  out->val = ObjectStorage{};
  auto& o  = get<ObjectStorage>(out->val);
  o.entries.reserve(count);
  o.index.reserve(count);
  for (uint64_t i = 0; i < count; ++i) {
    auto rh = read_record_header(src);
    if (rh.tag != kTagString)
      bin_throw(src, "expected string key in object");
    if (rh.sub_mode != 0)
      bin_throw(src, "non-zero sub_mode in object key");
    string key = parse_bin_string_inner(src, rh.count);
    auto val = parse_bin(src);
    o.index.emplace(key, o.entries.size());
    o.entries.emplace_back(std::move(key), std::move(val));
  }
  return out;
}

unique_ptr<FlexData::Impl> parse_bin(BinarySource& src) {
  auto rh = read_record_header(src);
  switch (rh.tag) {
    case kTagNull:
      if (rh.sub_mode != 0) bin_throw(src, "non-zero sub_mode in null record");
      return make_kind(FlexKind::Null);
    case kTagBoolFalse: {
      if (rh.sub_mode != 0) bin_throw(src, "non-zero sub_mode in bool record");
      auto p = make_kind(FlexKind::Bool); p->val = false; return p;
    }
    case kTagBoolTrue: {
      if (rh.sub_mode != 0) bin_throw(src, "non-zero sub_mode in bool record");
      auto p = make_kind(FlexKind::Bool); p->val = true;  return p;
    }
    case kTagInt: {
      if (rh.sub_mode != 0) bin_throw(src, "non-zero sub_mode in int record");
      int64_t v;
      read_or_throw(src, &v, 8, "truncated int payload");
      auto p = make_kind(FlexKind::Int); p->val = v; return p;
    }
    case kTagUint: {
      if (rh.sub_mode != 0) bin_throw(src, "non-zero sub_mode in uint record");
      uint64_t v;
      read_or_throw(src, &v, 8, "truncated uint payload");
      auto p = make_kind(FlexKind::Uint); p->val = v; return p;
    }
    case kTagReal: {
      if (rh.sub_mode != 0) bin_throw(src, "non-zero sub_mode in real record");
      double v;
      read_or_throw(src, &v, 8, "truncated real payload");
      auto p = make_kind(FlexKind::Real); p->val = v; return p;
    }
    case kTagString: {
      if (rh.sub_mode != 0)
        bin_throw(src, "non-zero sub_mode in string record");
      auto p = make_kind(FlexKind::String);
      p->val = parse_bin_string_inner(src, rh.count);
      return p;
    }
    case kTagArray:
      return parse_bin_array(src, rh.sub_mode, rh.count);
    case kTagObject:
      if (rh.sub_mode != 0)
        bin_throw(src, "non-zero sub_mode in object record");
      return parse_bin_object(src, rh.count);
    default:
      bin_throw(src, "unknown record tag");
  }
}

// ----- entry-point helpers --------------------------------------------------

void write_stream_header(BinarySink& sink) {
  uint8_t hdr[8] = {};
  uint32_t magic   = kMagic;
  uint16_t version = kVersion;
  uint16_t flags   = 0;
  memcpy(hdr,     &magic,   4);
  memcpy(hdr + 4, &version, 2);
  memcpy(hdr + 6, &flags,   2);
  sink.write_bytes(hdr, 8);
}

void read_stream_header(BinarySource& src) {
  uint8_t hdr[8];
  read_or_throw(src, hdr, 8, "truncated stream header");
  uint32_t magic;
  uint16_t version;
  uint16_t flags;
  memcpy(&magic,   hdr,     4);
  memcpy(&version, hdr + 4, 2);
  memcpy(&flags,   hdr + 6, 2);
  if (magic   != kMagic)   bin_throw(src, "bad magic");
  if (version != kVersion) bin_throw(src, "unsupported version");
  if (flags   != 0)        bin_throw(src, "non-zero flags");
}

}  // anonymous namespace

void FlexData::to_binary(ostream& out) const {
  BinarySink sink{out};
  write_stream_header(sink);
  if (!_impl) {
    auto null = make_kind(FlexKind::Null);
    serialize_bin(*null, sink);
  } else {
    serialize_bin(*_impl, sink);
  }
}

string FlexData::to_binary() const {
  ostringstream oss;
  to_binary(oss);
  return std::move(oss).str();
}

FlexData FlexData::from_binary(string_view src) {
  StringViewBinarySource source{src};
  read_stream_header(source);
  return FlexData{parse_bin(source)};
}

FlexData FlexData::from_binary(istream& in) {
  IstreamBinarySource source{in};
  read_stream_header(source);
  return FlexData{parse_bin(source)};
}

}
