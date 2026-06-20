#include "minitest.h"
#include "common/flex-data.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace std;

// =============================================================================
// construction & value semantics
// =============================================================================

TEST(flexdata, default_is_null) {
  FlexData v;
  EXPECT_TRUE(v.is_null());
  EXPECT_TRUE(v.kind() == FlexKind::Null);
}

TEST(flexdata, make_each_kind) {
  EXPECT_TRUE(FlexData::make_null().is_null());
  EXPECT_TRUE(FlexData::make_bool(true).is_bool());
  EXPECT_TRUE(FlexData::make_int(int64_t{42}).is_int());
  EXPECT_TRUE(FlexData::make_uint(uint64_t{42}).is_uint());
  EXPECT_TRUE(FlexData::make_real(3.14).is_real());
  EXPECT_TRUE(FlexData::make_string("hi").is_string());
  EXPECT_TRUE(FlexData::make_array().is_array());
  EXPECT_TRUE(FlexData::make_object().is_object());
}

TEST(flexdata, copy_constructs_deep) {
  auto orig = FlexData::make_array();
  orig.as_array().push_back(int64_t{1});
  orig.as_array().push_back(int64_t{2});
  FlexData copy = orig;
  copy.as_array().push_back(int64_t{3});
  EXPECT_TRUE(orig.as_array().size() == 2);
  EXPECT_TRUE(copy.as_array().size() == 3);
}

TEST(flexdata, move_leaves_source_null) {
  auto a = FlexData::make_int(int64_t{7});
  FlexData b = std::move(a);
  EXPECT_TRUE(b.is_int());
  EXPECT_TRUE(b.get_int() == 7);
  EXPECT_TRUE(a.is_null());
}

TEST(flexdata, copy_assign_self_safe) {
  auto a = FlexData::make_string("hello");
  auto* pa = &a;
  a = *pa;
  EXPECT_TRUE(a.is_string());
  EXPECT_TRUE(a.get_string() == "hello");
}

TEST(flexdata, deep_nested_copy) {
  auto root = FlexData::make_object();
  auto inner = FlexData::make_array();
  inner.as_array().push_back(int64_t{1});
  inner.as_array().push_back(int64_t{2});
  root.as_object().insert("nums", std::move(inner));
  root.as_object().insert("name", FlexData::make_string("hello"));

  FlexData copy = root;
  copy.as_object().at("nums").as_array().push_back(int64_t{99});  // touches the COPY's clone
  // wait — at() returns a *clone* by value, so the push_back above mutates a temporary
  // and is discarded. The cloned tree under `copy` is unaffected. We instead replace.
  auto nums2 = copy.as_object().at("nums");
  nums2.as_array().push_back(int64_t{99});
  copy.as_object().insert_or_assign("nums", std::move(nums2));

  EXPECT_TRUE(root.as_object().at("nums").as_array().size() == 2);
  EXPECT_TRUE(copy.as_object().at("nums").as_array().size() == 3);
}

// =============================================================================
// scalar accessors
// =============================================================================

TEST(flexdata, get_returns_value_on_correct_kind) {
  auto i = FlexData::make_int(int64_t{-100});
  EXPECT_TRUE(i.get_int() == -100);
  auto u = FlexData::make_uint(uint64_t{200});
  EXPECT_TRUE(u.get_uint() == 200);
  auto r = FlexData::make_real(2.5);
  EXPECT_TRUE(r.get_real() == 2.5);
  auto b = FlexData::make_bool(true);
  EXPECT_TRUE(b.get_bool() == true);
  auto s = FlexData::make_string("hello");
  EXPECT_TRUE(s.get_string() == "hello");
}

TEST(flexdata, get_throws_typeerror_on_wrong_kind) {
  auto i = FlexData::make_int(int64_t{1});
  bool threw = false;
  try { (void)i.get_string(); }
  catch (const FlexData::TypeError&) { threw = true; }
  EXPECT_TRUE(threw);
}

TEST(flexdata, as_X_returns_default_on_wrong_kind) {
  auto s = FlexData::make_string("hello");
  EXPECT_TRUE(s.as_int(int64_t{42}) == 42);
  EXPECT_TRUE(s.as_real(3.14) == 3.14);
  EXPECT_TRUE(s.as_bool(true) == true);
}

TEST(flexdata, as_int_cross_casts_from_uint_and_real) {
  // From Uint
  EXPECT_TRUE(FlexData::make_uint(uint64_t{7}).as_int() == 7);
  // From Real (truncation toward zero)
  EXPECT_TRUE(FlexData::make_real(3.9).as_int() == 3);
  EXPECT_TRUE(FlexData::make_real(-2.7).as_int() == -2);
  // From Int (identity)
  EXPECT_TRUE(FlexData::make_int(int64_t{-9}).as_int() == -9);
}

TEST(flexdata, as_uint_cross_casts_from_int_and_real) {
  EXPECT_TRUE(FlexData::make_int(int64_t{42}).as_uint() == 42u);
  EXPECT_TRUE(FlexData::make_real(5.7).as_uint() == 5u);
  EXPECT_TRUE(FlexData::make_uint(uint64_t{11}).as_uint() == 11u);
}

TEST(flexdata, as_real_cross_casts_from_int_and_uint) {
  EXPECT_TRUE(FlexData::make_int(int64_t{-3}).as_real() == -3.0);
  EXPECT_TRUE(FlexData::make_uint(uint64_t{17}).as_real() == 17.0);
  EXPECT_TRUE(FlexData::make_real(2.5).as_real() == 2.5);
}

TEST(flexdata, as_numeric_still_default_on_non_numeric) {
  auto s = FlexData::make_string("hello");
  EXPECT_TRUE(s.as_int(int64_t{99}) == 99);
  EXPECT_TRUE(s.as_uint(uint64_t{99}) == 99u);
  EXPECT_TRUE(s.as_real(9.5) == 9.5);
  auto b = FlexData::make_bool(true);
  EXPECT_TRUE(b.as_int(int64_t{42}) == 42);
  EXPECT_TRUE(b.as_real(2.5) == 2.5);
}

// =============================================================================
// array ops + iteration
// =============================================================================

TEST(flexdata, array_push_pop_size) {
  auto a = FlexData::make_array();
  auto v = a.as_array();
  EXPECT_TRUE(v.empty());
  v.push_back(int64_t{1});
  v.push_back(int64_t{2});
  v.push_back(int64_t{3});
  EXPECT_TRUE(v.size() == 3);
  EXPECT_TRUE(v.pop_back());
  EXPECT_TRUE(v.size() == 2);
  v.clear();
  EXPECT_TRUE(v.empty());
  EXPECT_FALSE(v.pop_back());
}

TEST(flexdata, array_at_throws_on_oob) {
  auto a = FlexData::make_array();
  a.as_array().push_back(int64_t{1});
  bool threw = false;
  try { (void)a.as_array().at(5); }
  catch (const out_of_range&) { threw = true; }
  EXPECT_TRUE(threw);
}

TEST(flexdata, array_iteration_yields_in_order) {
  auto a = FlexData::make_array();
  for (int64_t i = 0; i < 5; ++i) a.as_array().push_back(i);
  size_t n = 0;
  int64_t expected = 0;
  for (auto it = a.as_array().begin(); it != a.as_array().end(); ++it) {
    EXPECT_TRUE((*it).get_int() == expected);
    ++expected;
    ++n;
  }
  EXPECT_TRUE(n == 5);
}

TEST(flexdata, array_range_for) {
  auto a = FlexData::make_array();
  a.as_array().push_back(int64_t{10});
  a.as_array().push_back(int64_t{20});
  int64_t sum = 0;
  for (auto v : a.as_array()) sum += v.get_int();
  EXPECT_TRUE(sum == 30);
}

TEST(flexdata, array_set_replaces) {
  auto a = FlexData::make_array();
  auto v = a.as_array();
  v.push_back(int64_t{1});
  v.push_back(int64_t{2});
  v.push_back(int64_t{3});
  EXPECT_TRUE(v.set(1, FlexData::make_int(int64_t{99})));
  EXPECT_TRUE(v.at(1).get_int() == 99);
  EXPECT_FALSE(v.set(10, FlexData::make_int(int64_t{0})));
}

// =============================================================================
// homogeneous storage
// =============================================================================

TEST(flexdata, array_int_homogeneous_span) {
  auto a = FlexData::make_array();
  for (int64_t i = 0; i < 1000; ++i) a.as_array().push_back(i);
  auto sp = a.as_int_span();
  EXPECT_TRUE(sp.size() == 1000);
  EXPECT_TRUE(sp[500] == 500);
}

TEST(flexdata, array_uint_homogeneous_span) {
  auto a = FlexData::make_array();
  for (uint64_t i = 0; i < 100; ++i) a.as_array().push_back(i);
  auto sp = a.as_uint_span();
  EXPECT_TRUE(sp.size() == 100);
  EXPECT_TRUE(sp[10] == 10ULL);
}

TEST(flexdata, array_real_homogeneous_span) {
  auto a = FlexData::make_array();
  for (int i = 0; i < 50; ++i) a.as_array().push_back(double(i) + 0.5);
  auto sp = a.as_real_span();
  EXPECT_TRUE(sp.size() == 50);
  EXPECT_TRUE(sp[2] == 2.5);
}

TEST(flexdata, array_bool_homogeneous_span) {
  auto a = FlexData::make_array();
  a.as_array().push_back(true);
  a.as_array().push_back(false);
  a.as_array().push_back(true);
  auto sp = a.as_bool_span();
  EXPECT_TRUE(sp.size() == 3);
  EXPECT_TRUE(sp[0] == 1);
  EXPECT_TRUE(sp[1] == 0);
  EXPECT_TRUE(sp[2] == 1);
}

TEST(flexdata, array_demotes_on_mixed_type) {
  auto a = FlexData::make_array();
  a.as_array().push_back(int64_t{1});
  a.as_array().push_back(int64_t{2});
  EXPECT_TRUE(a.as_int_span().size() == 2);
  // Push a Uint -- demotes the Int storage to hetero.
  a.as_array().push_back(uint64_t{3});
  EXPECT_TRUE(a.as_int_span().size() == 0);
  EXPECT_TRUE(a.as_array().size() == 3);
  EXPECT_TRUE(a.as_array().at(0).get_int() == 1);
  EXPECT_TRUE(a.as_array().at(2).get_uint() == 3);
}

TEST(flexdata, array_homogeneous_iter_materializes_flexdata) {
  auto a = FlexData::make_array();
  for (int64_t i = 0; i < 4; ++i) a.as_array().push_back(i);
  vector<int64_t> got;
  for (auto v : a.as_array()) got.push_back(v.get_int());
  EXPECT_TRUE(got == (vector<int64_t>{0, 1, 2, 3}));
}

// =============================================================================
// object ops + insertion order
// =============================================================================

TEST(flexdata, object_insert_lookup) {
  auto o = FlexData::make_object();
  EXPECT_TRUE(o.as_object().insert("a", FlexData::make_int(int64_t{1})));
  EXPECT_TRUE(o.as_object().contains("a"));
  EXPECT_FALSE(o.as_object().contains("missing"));
  EXPECT_TRUE(o.as_object().at("a").get_int() == 1);
}

TEST(flexdata, object_duplicate_returns_false) {
  auto o = FlexData::make_object();
  EXPECT_TRUE(o.as_object().insert("a", FlexData::make_int(int64_t{1})));
  EXPECT_FALSE(o.as_object().insert("a", FlexData::make_int(int64_t{2})));
  EXPECT_TRUE(o.as_object().at("a").get_int() == 1);
}

TEST(flexdata, object_insert_or_assign_overwrites) {
  auto o = FlexData::make_object();
  o.as_object().insert("a", FlexData::make_int(int64_t{1}));
  o.as_object().insert_or_assign("a", FlexData::make_int(int64_t{99}));
  EXPECT_TRUE(o.as_object().at("a").get_int() == 99);
}

TEST(flexdata, object_erase) {
  auto o = FlexData::make_object();
  o.as_object().insert("a", FlexData::make_int(int64_t{1}));
  o.as_object().insert("b", FlexData::make_int(int64_t{2}));
  EXPECT_TRUE(o.as_object().erase("a"));
  EXPECT_FALSE(o.as_object().contains("a"));
  EXPECT_TRUE(o.as_object().contains("b"));
  EXPECT_FALSE(o.as_object().erase("a"));
}

TEST(flexdata, object_iteration_preserves_insertion_order) {
  auto o = FlexData::make_object();
  o.as_object().insert("z", FlexData::make_int(int64_t{1}));
  o.as_object().insert("a", FlexData::make_int(int64_t{2}));
  o.as_object().insert("m", FlexData::make_int(int64_t{3}));
  vector<string> keys;
  for (auto entry : o.as_object()) {
    keys.emplace_back(entry.first);
  }
  EXPECT_TRUE(keys == (vector<string>{"z", "a", "m"}));
}

TEST(flexdata, object_at_throws_on_missing) {
  auto o = FlexData::make_object();
  bool threw = false;
  try { (void)o.as_object().at("nope"); }
  catch (const out_of_range&) { threw = true; }
  EXPECT_TRUE(threw);
}

TEST(flexdata, object_find_returns_end_on_missing) {
  auto o = FlexData::make_object();
  o.as_object().insert("a", FlexData::make_int(int64_t{1}));
  EXPECT_TRUE(o.as_object().find("missing") == o.as_object().end());
  EXPECT_TRUE(o.as_object().find("a") != o.as_object().end());
}

// =============================================================================
// JSON round-trip
// =============================================================================

TEST(flexdata, json_roundtrip_scalars) {
  EXPECT_TRUE(FlexData::from_json("null").is_null());
  EXPECT_TRUE(FlexData::from_json("true").get_bool() == true);
  EXPECT_TRUE(FlexData::from_json("false").get_bool() == false);
  EXPECT_TRUE(FlexData::from_json("42").get_int() == 42);
  EXPECT_TRUE(FlexData::from_json("-7").get_int() == -7);
  EXPECT_TRUE(FlexData::from_json("3.14").get_real() == 3.14);
  EXPECT_TRUE(FlexData::from_json("\"hi\"").get_string() == "hi");
}

TEST(flexdata, json_roundtrip_nested) {
  string src = "{\"a\":1,\"b\":[1,2,3],\"c\":{\"d\":\"x\"}}";
  auto v = FlexData::from_json(src);
  EXPECT_TRUE(v.to_json() == src);
}

TEST(flexdata, json_roundtrip_homogeneous_int_array) {
  auto v = FlexData::from_json("[1,2,3,4,5]");
  EXPECT_TRUE(v.as_int_span().size() == 5);
  EXPECT_TRUE(v.to_json() == "[1,2,3,4,5]");
}

TEST(flexdata, json_pretty_format) {
  auto v = FlexData::make_object();
  v.as_object().insert("a", FlexData::make_int(int64_t{1}));
  v.as_object().insert("b", FlexData::make_int(int64_t{2}));
  string out = v.to_json(true);
  EXPECT_TRUE(out == "{\n  \"a\": 1,\n  \"b\": 2\n}");
}

TEST(flexdata, json_string_escapes) {
  auto v = FlexData::make_string("a\"b\\c\nd");
  string out = v.to_json();
  EXPECT_TRUE(out == "\"a\\\"b\\\\c\\nd\"");
  // round-trip
  auto v2 = FlexData::from_json(out);
  EXPECT_TRUE(v2.get_string() == "a\"b\\c\nd");
}

TEST(flexdata, json_float_no_locale_dependence) {
  auto v = FlexData::make_real(1.5);
  string out = v.to_json();
  EXPECT_TRUE(out.find('.') != string::npos);
  EXPECT_TRUE(out.find(',') == string::npos);
}

TEST(flexdata, json_object_output_preserves_insertion_order) {
  auto v = FlexData::make_object();
  v.as_object().insert("z", FlexData::make_int(int64_t{1}));
  v.as_object().insert("a", FlexData::make_int(int64_t{2}));
  v.as_object().insert("m", FlexData::make_int(int64_t{3}));
  EXPECT_TRUE(v.to_json() == "{\"z\":1,\"a\":2,\"m\":3}");
}

TEST(flexdata, json_nan_inf_emitted_as_null) {
  EXPECT_TRUE(FlexData::make_real(numeric_limits<double>::quiet_NaN()).to_json()
                == "null");
  EXPECT_TRUE(FlexData::make_real(numeric_limits<double>::infinity()).to_json()
                == "null");
  EXPECT_TRUE(FlexData::make_real(-numeric_limits<double>::infinity()).to_json()
                == "null");
}

TEST(flexdata, json_uint_above_int64_max_roundtrips_as_uint) {
  auto v = FlexData::make_uint(uint64_t{0xFFFFFFFFFFFFFFFEULL});
  string out = v.to_json();
  auto v2 = FlexData::from_json(out);
  EXPECT_TRUE(v2.is_uint());
  EXPECT_TRUE(v2.get_uint() == 0xFFFFFFFFFFFFFFFEULL);
}

TEST(flexdata, json_uint_in_int64_range_roundtrips_as_int) {
  auto v = FlexData::make_uint(uint64_t{42});
  auto v2 = FlexData::from_json(v.to_json());
  // Documents the lost distinction: 42 fits in int64, comes back as Int.
  EXPECT_TRUE(v2.is_int());
  EXPECT_TRUE(v2.get_int() == 42);
}

// =============================================================================
// JSON5 parse extensions
// =============================================================================

TEST(flexdata, json5_unquoted_keys) {
  auto v = FlexData::from_json("{foo:1, bar_baz:2}");
  EXPECT_TRUE(v.as_object().at("foo").get_int() == 1);
  EXPECT_TRUE(v.as_object().at("bar_baz").get_int() == 2);
}

TEST(flexdata, json5_single_quoted_string) {
  auto v = FlexData::from_json("'hello'");
  EXPECT_TRUE(v.get_string() == "hello");
}

TEST(flexdata, json5_single_quote_inline_double_quote) {
  auto v = FlexData::from_json("'he said \"hi\"'");
  EXPECT_TRUE(v.get_string() == "he said \"hi\"");
}

TEST(flexdata, json5_hex_number_int) {
  auto v = FlexData::from_json("0xff");
  EXPECT_TRUE(v.is_int());
  EXPECT_TRUE(v.get_int() == 255);
}

TEST(flexdata, json5_hex_number_uint_above_int64_max) {
  auto v = FlexData::from_json("0xFFFFFFFFFFFFFFFE");
  EXPECT_TRUE(v.is_uint());
  EXPECT_TRUE(v.get_uint() == 0xFFFFFFFFFFFFFFFEULL);
}

TEST(flexdata, json5_hex_overflow_above_uint64_max_throws_parseerror) {
  bool threw = false;
  try { (void)FlexData::from_json("0xFFFFFFFFFFFFFFFFFF"); }
  catch (const FlexData::ParseError&) { threw = true; }
  EXPECT_TRUE(threw);
}

TEST(flexdata, json5_decimal_uint_above_int64_max) {
  // 2^63 = 9223372036854775808 (one above int64 max)
  auto v = FlexData::from_json("9223372036854775808");
  EXPECT_TRUE(v.is_uint());
  EXPECT_TRUE(v.get_uint() == 9223372036854775808ULL);
}

TEST(flexdata, json5_leading_decimal_point) {
  auto v = FlexData::from_json(".5");
  EXPECT_TRUE(v.is_real());
  EXPECT_TRUE(v.get_real() == 0.5);
}

TEST(flexdata, json5_trailing_decimal_point) {
  auto v = FlexData::from_json("5.");
  EXPECT_TRUE(v.is_real());
  EXPECT_TRUE(v.get_real() == 5.0);
}

TEST(flexdata, json5_positive_sign) {
  auto v = FlexData::from_json("+42");
  EXPECT_TRUE(v.is_int());
  EXPECT_TRUE(v.get_int() == 42);
}

TEST(flexdata, json5_line_comment) {
  auto v = FlexData::from_json("// a comment\n42");
  EXPECT_TRUE(v.get_int() == 42);
}

TEST(flexdata, json5_block_comment) {
  auto v = FlexData::from_json("/* ignored */ 42 /* trailing */");
  EXPECT_TRUE(v.get_int() == 42);
}

TEST(flexdata, json5_trailing_comma_array) {
  auto v = FlexData::from_json("[1, 2, 3,]");
  EXPECT_TRUE(v.as_array().size() == 3);
  EXPECT_TRUE(v.as_array().at(2).get_int() == 3);
}

TEST(flexdata, json5_trailing_comma_object) {
  auto v = FlexData::from_json("{a:1, b:2,}");
  EXPECT_TRUE(v.as_object().size() == 2);
  EXPECT_TRUE(v.as_object().at("b").get_int() == 2);
}

TEST(flexdata, json5_infinity_nan) {
  auto a = FlexData::from_json("Infinity");
  EXPECT_TRUE(a.is_real());
  EXPECT_TRUE(std::isinf(a.get_real()) && a.get_real() > 0);
  auto b = FlexData::from_json("-Infinity");
  EXPECT_TRUE(b.is_real());
  EXPECT_TRUE(std::isinf(b.get_real()) && b.get_real() < 0);
  auto c = FlexData::from_json("NaN");
  EXPECT_TRUE(c.is_real());
  EXPECT_TRUE(std::isnan(c.get_real()));
}

// =============================================================================
// parser errors
// =============================================================================

TEST(flexdata, parse_throws_on_unterminated_string) {
  bool threw = false;
  try { (void)FlexData::from_json("\"open"); }
  catch (const FlexData::ParseError&) { threw = true; }
  EXPECT_TRUE(threw);
}

TEST(flexdata, parse_throws_on_invalid_token) {
  bool threw = false;
  try { (void)FlexData::from_json("@@@"); }
  catch (const FlexData::ParseError&) { threw = true; }
  EXPECT_TRUE(threw);
}

TEST(flexdata, parse_throws_on_unbalanced_bracket) {
  bool threw = false;
  try { (void)FlexData::from_json("[1, 2, 3"); }
  catch (const FlexData::ParseError&) { threw = true; }
  EXPECT_TRUE(threw);
}

TEST(flexdata, parse_error_line_column_correct) {
  bool threw = false;
  try {
    // Offending '@' is at line 2, column 1.
    (void)FlexData::from_json("[\n@");
  } catch (const FlexData::ParseError& e) {
    threw = true;
    EXPECT_TRUE(e.line() == 2);
    EXPECT_TRUE(e.column() == 1);
  }
  EXPECT_TRUE(threw);
}

TEST(flexdata, parse_throws_on_empty_input) {
  bool threw = false;
  try { (void)FlexData::from_json(""); }
  catch (const FlexData::ParseError&) { threw = true; }
  EXPECT_TRUE(threw);
}

// =============================================================================
// stream JSON APIs
// =============================================================================

TEST(flexdata, to_json_ostream_matches_string_form) {
  auto v = FlexData::make_object();
  v.as_object().insert("a", FlexData::make_int(int64_t{1}));
  v.as_object().insert("b", FlexData::make_array());
  v.as_object().at("b").to_json();  // unrelated; ensure no side effects on v

  ostringstream oss;
  v.to_json(oss);
  EXPECT_TRUE(oss.str() == v.to_json());
}

TEST(flexdata, to_json_ostream_pretty) {
  auto v = FlexData::make_object();
  v.as_object().insert("a", FlexData::make_int(int64_t{1}));
  v.as_object().insert("b", FlexData::make_int(int64_t{2}));
  ostringstream oss;
  v.to_json(oss, true);
  EXPECT_TRUE(oss.str() == "{\n  \"a\": 1,\n  \"b\": 2\n}");
}

TEST(flexdata, from_json_istream_basic) {
  istringstream iss("{\"a\":1,\"b\":[1,2,3]}");
  auto v = FlexData::from_json(iss);
  EXPECT_TRUE(v.as_object().at("a").get_int() == 1);
  EXPECT_TRUE(v.as_object().at("b").as_int_span().size() == 3);
}

TEST(flexdata, from_json_istream_json5_features) {
  // Mix of unquoted keys, single-quoted string, hex, comment, trailing comma.
  istringstream iss("{a:0xff, b:'hi', /* note */ c:[1,2,3,]}");
  auto v = FlexData::from_json(iss);
  EXPECT_TRUE(v.as_object().at("a").get_int() == 255);
  EXPECT_TRUE(v.as_object().at("b").get_string() == "hi");
  EXPECT_TRUE(v.as_object().at("c").as_array().size() == 3);
}

TEST(flexdata, from_json_istream_long_input_stays_streaming) {
  // Build a moderately large input (~16 KB) and parse it from a stream.
  // The point is correctness with input larger than the parser's internal
  // lookahead buffer (64 bytes).
  ostringstream gen;
  gen << '[';
  for (int i = 0; i < 1000; ++i) {
    if (i > 0) gen << ',';
    gen << i;
  }
  gen << ']';
  istringstream iss(std::move(gen).str());
  auto v = FlexData::from_json(iss);
  EXPECT_TRUE(v.as_int_span().size() == 1000);
  EXPECT_TRUE(v.as_int_span()[999] == 999);
}

TEST(flexdata, from_json_istream_throws_parseerror) {
  istringstream iss("[1, 2, ");  // unterminated
  bool threw = false;
  try { (void)FlexData::from_json(iss); }
  catch (const FlexData::ParseError&) { threw = true; }
  EXPECT_TRUE(threw);
}

TEST(flexdata, stream_roundtrip) {
  // Write to ostream, read back from istream, get the same value.
  auto tags = FlexData::make_array();
  tags.as_array().push_back(int64_t{1});
  tags.as_array().push_back(int64_t{2});
  auto v = FlexData::make_object();
  v.as_object().insert("name", FlexData::make_string("alice"));
  v.as_object().insert("age",  FlexData::make_int(int64_t{30}));
  v.as_object().insert("tags", std::move(tags));

  stringstream ss;
  v.to_json(ss);
  auto v2 = FlexData::from_json(ss);
  EXPECT_TRUE(v2.as_object().at("name").get_string() == "alice");
  EXPECT_TRUE(v2.as_object().at("age").get_int() == 30);
  EXPECT_TRUE(v2.as_object().at("tags").as_int_span().size() == 2);
}

// =============================================================================
// native binary format
// =============================================================================

TEST(flexdata, bin_roundtrip_null) {
  auto v  = FlexData::make_null();
  auto v2 = FlexData::from_binary(v.to_binary());
  EXPECT_TRUE(v2.is_null());
}

TEST(flexdata, bin_roundtrip_bool) {
  auto t = FlexData::from_binary(FlexData::make_bool(true).to_binary());
  EXPECT_TRUE(t.get_bool() == true);
  auto f = FlexData::from_binary(FlexData::make_bool(false).to_binary());
  EXPECT_TRUE(f.get_bool() == false);
}

TEST(flexdata, bin_roundtrip_int) {
  auto v = FlexData::from_binary(
    FlexData::make_int(int64_t{-1234567890123}).to_binary());
  EXPECT_TRUE(v.is_int());
  EXPECT_TRUE(v.get_int() == -1234567890123);
}

TEST(flexdata, bin_roundtrip_uint_above_int_max) {
  auto v = FlexData::from_binary(
    FlexData::make_uint(uint64_t{0xFFFFFFFFFFFFFFFEULL}).to_binary());
  EXPECT_TRUE(v.is_uint());
  EXPECT_TRUE(v.get_uint() == 0xFFFFFFFFFFFFFFFEULL);
}

TEST(flexdata, bin_roundtrip_real) {
  auto v = FlexData::from_binary(FlexData::make_real(3.14).to_binary());
  EXPECT_TRUE(v.get_real() == 3.14);
}

TEST(flexdata, bin_roundtrip_real_special_values) {
  auto pinf = FlexData::from_binary(
    FlexData::make_real(numeric_limits<double>::infinity()).to_binary());
  EXPECT_TRUE(std::isinf(pinf.get_real()) && pinf.get_real() > 0);
  auto ninf = FlexData::from_binary(
    FlexData::make_real(-numeric_limits<double>::infinity()).to_binary());
  EXPECT_TRUE(std::isinf(ninf.get_real()) && ninf.get_real() < 0);
  auto nan = FlexData::from_binary(
    FlexData::make_real(numeric_limits<double>::quiet_NaN()).to_binary());
  EXPECT_TRUE(std::isnan(nan.get_real()));
}

TEST(flexdata, bin_roundtrip_string_with_nul_and_high_bytes) {
  string raw{'h', '\0', 'i', '\xff', '\x7f'};
  auto v = FlexData::from_binary(FlexData::make_string(raw).to_binary());
  EXPECT_TRUE(v.is_string());
  EXPECT_TRUE(v.get_string() == raw);
}

TEST(flexdata, bin_roundtrip_int_array_homogeneous_storage) {
  auto a = FlexData::make_array();
  for (int64_t i = 0; i < 1000; ++i) a.as_array().push_back(i);
  auto v = FlexData::from_binary(a.to_binary());
  auto sp = v.as_int_span();
  EXPECT_TRUE(sp.size() == 1000);
  EXPECT_TRUE(sp[0] == 0);
  EXPECT_TRUE(sp[999] == 999);
}

TEST(flexdata, bin_roundtrip_uint_array_homogeneous_storage) {
  auto a = FlexData::make_array();
  for (uint64_t i = 0; i < 100; ++i) a.as_array().push_back(i);
  auto v = FlexData::from_binary(a.to_binary());
  auto sp = v.as_uint_span();
  EXPECT_TRUE(sp.size() == 100);
  EXPECT_TRUE(sp[42] == 42);
}

TEST(flexdata, bin_roundtrip_real_array_homogeneous_storage) {
  auto a = FlexData::make_array();
  for (int i = 0; i < 50; ++i) a.as_array().push_back(double(i) + 0.25);
  auto v = FlexData::from_binary(a.to_binary());
  auto sp = v.as_real_span();
  EXPECT_TRUE(sp.size() == 50);
  EXPECT_TRUE(sp[5] == 5.25);
}

TEST(flexdata, bin_roundtrip_bool_array_homogeneous_storage) {
  auto a = FlexData::make_array();
  for (int i = 0; i < 13; ++i) a.as_array().push_back(bool(i & 1));
  auto v = FlexData::from_binary(a.to_binary());
  auto sp = v.as_bool_span();
  EXPECT_TRUE(sp.size() == 13);
  EXPECT_TRUE(sp[0] == 0);
  EXPECT_TRUE(sp[1] == 1);
  EXPECT_TRUE(sp[12] == 0);
}

TEST(flexdata, bin_roundtrip_string_array) {
  auto a = FlexData::make_array();
  for (int i = 0; i < 100; ++i)
    a.as_array().push_back(string_view("item " + to_string(i)));
  auto v = FlexData::from_binary(a.to_binary());
  EXPECT_TRUE(v.as_array().size() == 100);
  EXPECT_TRUE(v.as_array().at(0).get_string() == "item 0");
  EXPECT_TRUE(v.as_array().at(99).get_string() == "item 99");
}

TEST(flexdata, bin_roundtrip_null_array) {
  auto a = FlexData::make_array();
  for (int i = 0; i < 7; ++i) a.as_array().push_back(FlexData::make_null());
  auto v = FlexData::from_binary(a.to_binary());
  EXPECT_TRUE(v.as_array().size() == 7);
  for (auto e : v.as_array()) EXPECT_TRUE(e.is_null());
}

TEST(flexdata, bin_roundtrip_hetero_array) {
  auto a = FlexData::make_array();
  a.as_array().push_back(int64_t{1});
  a.as_array().push_back(int64_t{2});
  a.as_array().push_back(uint64_t{3});  // demotes to hetero
  auto v = FlexData::from_binary(a.to_binary());
  EXPECT_TRUE(v.as_array().size() == 3);
  EXPECT_TRUE(v.as_array().at(0).get_int() == 1);
  EXPECT_TRUE(v.as_array().at(1).get_int() == 2);
  EXPECT_TRUE(v.as_array().at(2).get_uint() == 3);
}

TEST(flexdata, bin_roundtrip_object_preserves_order) {
  auto o = FlexData::make_object();
  o.as_object().insert("z", FlexData::make_int(int64_t{1}));
  o.as_object().insert("a", FlexData::make_int(int64_t{2}));
  o.as_object().insert("m", FlexData::make_int(int64_t{3}));
  auto v = FlexData::from_binary(o.to_binary());
  vector<string> keys;
  for (auto entry : v.as_object()) keys.emplace_back(entry.first);
  EXPECT_TRUE(keys == (vector<string>{"z", "a", "m"}));
  EXPECT_TRUE(v.as_object().at("a").get_int() == 2);
}

TEST(flexdata, bin_roundtrip_deep_nesting) {
  auto inner = FlexData::make_array();
  for (int64_t i = 0; i < 4; ++i) inner.as_array().push_back(i);
  auto sub = FlexData::make_object();
  sub.as_object().insert("nums", std::move(inner));
  sub.as_object().insert("note", FlexData::make_string("hello"));
  auto root = FlexData::make_object();
  root.as_object().insert("sub", std::move(sub));
  root.as_object().insert("flag", FlexData::make_bool(true));

  auto v = FlexData::from_binary(root.to_binary());
  EXPECT_TRUE(v.as_object().at("flag").get_bool() == true);
  EXPECT_TRUE(v.as_object().at("sub").as_object().at("note").get_string()
                == "hello");
  EXPECT_TRUE(v.as_object().at("sub").as_object().at("nums")
                .as_int_span().size() == 4);
}

TEST(flexdata, bin_byte_identical_re_encode) {
  auto root = FlexData::make_object();
  root.as_object().insert("arr", FlexData::make_array());
  for (int64_t i = 0; i < 5; ++i)
    root.as_object().at("arr").as_array().push_back(i);
  // Mutate in place after at(): at() returns a clone. Build the array
  // separately then assign so the value is actually populated.
  auto arr = FlexData::make_array();
  for (int64_t i = 0; i < 5; ++i) arr.as_array().push_back(i);
  root.as_object().insert_or_assign("arr", std::move(arr));
  root.as_object().insert("name", FlexData::make_string("xyz"));

  string b1 = root.to_binary();
  auto   v  = FlexData::from_binary(b1);
  string b2 = v.to_binary();
  EXPECT_TRUE(b1 == b2);
}

TEST(flexdata, bin_alignment_invariant_for_top_level_int) {
  auto v = FlexData::make_int(int64_t{0x1122334455667788LL});
  string b = v.to_binary();
  EXPECT_TRUE(b.size() == 24);                 // 8 stream + 8 hdr + 8 payload
  EXPECT_TRUE((unsigned char)b[0] == 'V');
  EXPECT_TRUE((unsigned char)b[1] == 'P');
  EXPECT_TRUE((unsigned char)b[2] == 'F');
  EXPECT_TRUE((unsigned char)b[3] == 'D');
  EXPECT_TRUE((unsigned char)b[4] == 1);       // version low
  EXPECT_TRUE((unsigned char)b[5] == 0);       // version high
  EXPECT_TRUE((unsigned char)b[6] == 0);       // flags low
  EXPECT_TRUE((unsigned char)b[7] == 0);       // flags high
  EXPECT_TRUE((unsigned char)b[8] == 0x03);    // tag = Int
  EXPECT_TRUE((unsigned char)b[9] == 0);       // sub_mode
  EXPECT_TRUE((unsigned char)b[10] == 0);      // reserved
  EXPECT_TRUE((unsigned char)b[11] == 0);
  EXPECT_TRUE((unsigned char)b[12] == 0);      // count low
  EXPECT_TRUE((unsigned char)b[16] == 0x88);   // payload first byte (LE)
  EXPECT_TRUE((unsigned char)b[23] == 0x11);   // payload last byte
}

TEST(flexdata, bin_stream_roundtrip) {
  auto root = FlexData::make_object();
  root.as_object().insert("name", FlexData::make_string("alice"));
  root.as_object().insert("age",  FlexData::make_int(int64_t{30}));
  stringstream ss;
  root.to_binary(ss);
  auto v = FlexData::from_binary(ss);
  EXPECT_TRUE(v.as_object().at("name").get_string() == "alice");
  EXPECT_TRUE(v.as_object().at("age").get_int() == 30);
}

TEST(flexdata, bin_stream_long_input_roundtrip) {
  // 1000-element int array exercises bulk-read into vector storage.
  auto a = FlexData::make_array();
  for (int64_t i = 0; i < 1000; ++i) a.as_array().push_back(i);
  stringstream ss;
  a.to_binary(ss);
  auto v = FlexData::from_binary(ss);
  EXPECT_TRUE(v.as_int_span().size() == 1000);
  EXPECT_TRUE(v.as_int_span()[500] == 500);
}

TEST(flexdata, bin_throws_on_bad_magic) {
  string b(8, '\0');  // wrong magic, valid length
  bool threw = false;
  try { (void)FlexData::from_binary(b); }
  catch (const FlexData::BinaryError&) { threw = true; }
  EXPECT_TRUE(threw);
}

TEST(flexdata, bin_throws_on_bad_version) {
  string b = FlexData::make_null().to_binary();
  b[4] = 99;  // bump version
  bool threw = false;
  try { (void)FlexData::from_binary(b); }
  catch (const FlexData::BinaryError&) { threw = true; }
  EXPECT_TRUE(threw);
}

TEST(flexdata, bin_throws_on_truncated_payload) {
  string b = FlexData::make_int(int64_t{42}).to_binary();
  b.resize(b.size() - 1);  // chop off the last byte of the i64 payload
  bool threw = false;
  try { (void)FlexData::from_binary(b); }
  catch (const FlexData::BinaryError& e) {
    threw = true;
    EXPECT_TRUE(e.byte_offset() > 0);
  }
  EXPECT_TRUE(threw);
}

TEST(flexdata, bin_throws_on_unknown_tag) {
  string b = FlexData::make_null().to_binary();
  b[8] = 0x7e;  // bogus tag, ext-flag clear
  bool threw = false;
  try { (void)FlexData::from_binary(b); }
  catch (const FlexData::BinaryError&) { threw = true; }
  EXPECT_TRUE(threw);
}

TEST(flexdata, bin_throws_on_nonzero_reserved_bytes) {
  string b = FlexData::make_int(int64_t{42}).to_binary();
  b[10] = 0x01;  // reserved byte in record header must be zero
  bool threw = false;
  try { (void)FlexData::from_binary(b); }
  catch (const FlexData::BinaryError&) { threw = true; }
  EXPECT_TRUE(threw);
}

TEST(flexdata, bin_extended_count_header_path) {
  // Hand-craft a stream whose root is a Null record using the extended-header
  // form (16 B header, count=0 in u64). This exercises the bit-7 dispatch
  // without needing to allocate a >4G-element array.
  uint8_t buf[8 + 16] = {
    'V','P','F','D',         // magic
    0x01, 0x00,              // version
    0x00, 0x00,              // flags
    0x80, 0x00,              // tag=Null with ext-count flag, sub=0
    0x00, 0x00,              // reserved (must be 0)
    0x00, 0x00, 0x00, 0x00,  // pad (must be 0)
    0x00, 0x00, 0x00, 0x00,  // count low half
    0x00, 0x00, 0x00, 0x00,  // count high half
  };
  string_view sv(reinterpret_cast<const char*>(buf), sizeof(buf));
  auto v = FlexData::from_binary(sv);
  EXPECT_TRUE(v.is_null());
}
