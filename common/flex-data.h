#ifndef FLEX_DATA_H
#define FLEX_DATA_H

#include "common/beat-payload-intf.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace vpipe {

enum class FlexKind : unsigned char {
  Null, Bool, Int, Uint, Real, String, Array, Object
};

class FlexData {
public:
  // Internal storage type. Forward-declared here so the nested view types
  // can reference it; the full definition lives in flex-data.cc.
  class Impl;

  class Error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
  };
  class TypeError : public Error {
  public:
    using Error::Error;
  };
  class ParseError : public Error {
  public:
    ParseError(size_t line, size_t column, const std::string& msg);
    size_t line() const noexcept;
    size_t column() const noexcept;
  private:
    size_t _line;
    size_t _column;
  };
  class BinaryError : public Error {
  public:
    BinaryError(size_t byte_offset, const std::string& msg);
    size_t byte_offset() const noexcept;
  private:
    size_t _byte_offset;
  };

  // factories
  FlexData();                                  // Null
  static FlexData make_null();
  static FlexData make_bool(bool v);
  static FlexData make_int(int64_t v);
  static FlexData make_uint(uint64_t v);
  static FlexData make_real(double v);
  static FlexData make_string(std::string_view v);
  static FlexData make_array();
  static FlexData make_object();

  // value semantics (deep clone on copy)
  FlexData(const FlexData& other);
  FlexData(FlexData&& other) noexcept;
  FlexData& operator=(const FlexData& other);
  FlexData& operator=(FlexData&& other) noexcept;
  ~FlexData();

  // introspection
  FlexKind kind() const noexcept;
  bool is_null() const noexcept;
  bool is_bool() const noexcept;
  bool is_int() const noexcept;
  bool is_uint() const noexcept;
  bool is_real() const noexcept;
  bool is_string() const noexcept;
  bool is_array() const noexcept;
  bool is_object() const noexcept;
  bool is_scalar() const noexcept;

  // strict scalar accessors -- throw TypeError on wrong kind
  bool             get_bool() const;
  int64_t          get_int() const;
  uint64_t         get_uint() const;
  double           get_real() const;
  std::string_view get_string() const;

  // lenient scalar accessors -- never throw. Numeric accessors
  // (as_int / as_uint / as_real) cross-cast among the numeric kinds
  // via static_cast: e.g. as_int() on a Real truncates toward zero;
  // as_uint() on a negative Int wraps. Default is returned only when
  // the kind is non-numeric (or non-matching for as_bool/as_string).
  bool             as_bool(bool def = false) const noexcept;
  int64_t          as_int(int64_t def = 0) const noexcept;
  uint64_t         as_uint(uint64_t def = 0) const noexcept;
  double           as_real(double def = 0.0) const noexcept;
  std::string_view as_string(std::string_view def = {}) const noexcept;

  // forward-declared view types
  class ConstArrayView;
  class ArrayView;
  class ConstObjectView;
  class ObjectView;

  // throw TypeError on wrong kind
  ArrayView       as_array();
  ConstArrayView  as_array() const;
  ObjectView      as_object();
  ConstObjectView as_object() const;

  // homogeneous fast path -- empty span when storage isn't in matching mode
  std::span<const int64_t>       as_int_span()  const noexcept;
  std::span<const uint64_t>      as_uint_span() const noexcept;
  std::span<const double>        as_real_span() const noexcept;
  std::span<const unsigned char> as_bool_span() const noexcept;

  // JSON
  std::string to_json(bool pretty = false) const;
  void        to_json(std::ostream& out, bool pretty = false) const;
  static FlexData from_json(std::string_view src);
  static FlexData from_json(std::istream& in);

  // Native binary format -- compact, 8B-aligned, little-endian on the wire.
  // Not interoperable with any external standard.
  std::string to_binary() const;
  void        to_binary(std::ostream& out) const;
  static FlexData from_binary(std::string_view src);
  static FlexData from_binary(std::istream& in);

  // ----- View types ------------------------------------------------------

  class ConstArrayView {
  public:
    class iterator {
    public:
      using value_type        = FlexData;
      using reference         = FlexData;
      using pointer           = void;
      using difference_type   = std::ptrdiff_t;
      using iterator_category = std::random_access_iterator_tag;

      iterator() noexcept;
      FlexData operator*() const;
      iterator& operator++() noexcept;
      iterator  operator++(int) noexcept;
      iterator& operator--() noexcept;
      iterator  operator--(int) noexcept;
      iterator& operator+=(difference_type n) noexcept;
      iterator& operator-=(difference_type n) noexcept;
      iterator  operator+ (difference_type n) const noexcept;
      iterator  operator- (difference_type n) const noexcept;
      difference_type operator-(const iterator& other) const noexcept;
      FlexData operator[](difference_type n) const;
      bool operator==(const iterator& other) const noexcept;
      bool operator!=(const iterator& other) const noexcept;
      bool operator< (const iterator& other) const noexcept;
      bool operator<=(const iterator& other) const noexcept;
      bool operator> (const iterator& other) const noexcept;
      bool operator>=(const iterator& other) const noexcept;
    private:
      friend class ConstArrayView;
      iterator(const FlexData::Impl* arr, size_t idx) noexcept;
      const FlexData::Impl* _arr;
      size_t _idx;
    };

    iterator begin() const noexcept;
    iterator end()   const noexcept;
    size_t   size()  const noexcept;
    bool     empty() const noexcept;
    FlexData operator[](size_t i) const;     // UB if i >= size()
    FlexData at(size_t i) const;             // throws std::out_of_range

  protected:
    friend class FlexData;
    explicit ConstArrayView(const FlexData::Impl* arr) noexcept;
    const FlexData::Impl* _arr;
  };

  class ArrayView : public ConstArrayView {
  public:
    void push_back(FlexData v);
    void push_back(bool v);
    void push_back(int64_t v);
    void push_back(uint64_t v);
    void push_back(double v);
    void push_back(std::string_view v);
    bool pop_back();                          // false on empty
    void clear() noexcept;
    void reserve(size_t n);
    bool set(size_t i, FlexData v);           // false if i >= size()
  private:
    friend class FlexData;
    explicit ArrayView(FlexData::Impl* arr) noexcept;
    FlexData::Impl* _mut;
  };

  class ConstObjectView {
  public:
    using Entry = std::pair<std::string_view, FlexData>;

    class iterator {
    public:
      using value_type        = Entry;
      using reference         = Entry;
      using pointer           = void;
      using difference_type   = std::ptrdiff_t;
      using iterator_category = std::forward_iterator_tag;

      iterator() noexcept;
      Entry operator*() const;
      iterator& operator++() noexcept;
      iterator  operator++(int) noexcept;
      bool operator==(const iterator& other) const noexcept;
      bool operator!=(const iterator& other) const noexcept;
    private:
      friend class ConstObjectView;
      iterator(const FlexData::Impl* obj, size_t idx) noexcept;
      const FlexData::Impl* _obj;
      size_t _idx;
    };

    iterator begin() const noexcept;
    iterator end()   const noexcept;
    size_t   size()  const noexcept;
    bool     empty() const noexcept;
    bool     contains(std::string_view key) const noexcept;
    iterator find(std::string_view key) const noexcept;
    FlexData at(std::string_view key) const;  // throws std::out_of_range

  protected:
    friend class FlexData;
    explicit ConstObjectView(const FlexData::Impl* obj) noexcept;
    const FlexData::Impl* _obj;
  };

  class ObjectView : public ConstObjectView {
  public:
    bool insert(std::string_view key, FlexData val);  // false if key existed
    void insert_or_assign(std::string_view key, FlexData val);
    bool erase(std::string_view key);  // true if removed
    void clear() noexcept;
  private:
    friend class FlexData;
    explicit ObjectView(FlexData::Impl* obj) noexcept;
    FlexData::Impl* _mut;
  };

  // Public-but-internal: takes ownership of an already-built Impl. Public
  // so anonymous-namespace helpers in the .cc can build FlexData values.
  // Users can't call it usefully because Impl is opaque from the header.
  explicit FlexData(std::unique_ptr<Impl> impl) noexcept;

private:
  std::unique_ptr<Impl> _impl;
};

// Pipeline-edge payload wrapping a FlexData value. FlexData has
// deep-clone copy semantics today, so clone() is a deep copy. For
// fanout-hot uses we may later switch FlexData::Impl to
// shared_ptr<const Impl> to make this O(1); for now production
// edges are single-cursor so this is never invoked on the hot path.
class FlexDataPayload : public BeatPayloadIntf {
public:
  FlexData data;

  FlexDataPayload() = default;
  explicit
  FlexDataPayload(const FlexData& d)
    : data(d)
  {}
  explicit
  FlexDataPayload(FlexData&& d) noexcept
    : data(std::move(d))
  {}

  std::unique_ptr<BeatPayloadIntf>
  clone() const override
  {
    return std::make_unique<FlexDataPayload>(data);
  }

  std::string
  describe() const override
  {
    return "FlexData";
  }
};

}

#endif
