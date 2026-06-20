#include "minitest.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <list>
#include <string>
#include <tuple>

#ifdef _WIN32
#else
#include <time.h>
typedef struct timespec TIME;
#endif

using namespace std;

namespace minitest {

inline void
cout_color_red() { cout << "\033[31m"; }

inline void
cout_color_green() { cout << "\033[32m"; }

inline void
cout_color_reset() { cout << "\033[0m"; }

inline void
monotonic_time_stamp(TIME* t) { clock_gettime(CLOCK_MONOTONIC, t); }

double
elapsed_time(const TIME* begin, const TIME* end)
{
  int64_t i = 0, f = 0;
  if (end->tv_nsec - begin->tv_nsec < 0) {
    i = end->tv_sec - begin->tv_sec - 1;
    f = 1e9 + end->tv_nsec - begin->tv_nsec;
  } else {
    i = end->tv_sec - begin->tv_sec;
    f = end->tv_nsec - begin->tv_nsec;
  }
  return static_cast<double>(i) + static_cast<double>(f) / 1.e9;
}

////////////////////////////////////////////////////////////////////////////////

TestPlanConfig::TestPlanConfig()
  : also_run_disabled(false),
    colored_output(true),
    list_only(false),
    print_time(true)
{}

class TestResult {
public:
  TestResult() {};
  void register_failure(string_view file, size_t line)
  { _failures.push_back(make_tuple(file, line)); };
  const list<tuple<string_view, size_t> >& get_failures() const
  { return _failures; };
private:
  list<tuple<string_view, size_t> > _failures;
};

TestCaseBase::TestCaseBase()
  : _result(make_unique<TestResult>())
{}

TestCaseBase::~TestCaseBase()
{}

const TestResult*
TestCaseBase::result() const
{ return _result.get(); }

void
TestCaseBase::report_failure(string_view file, size_t line)
{
  cout << file << ":" << line << ": Failure" << endl;
  _result->register_failure(file, line);
}

void
TestCaseBase::report_expect_true_failure(string_view cond,
                                         string_view file, size_t line)
{
  cout << file << ":" << line << ": Failure" << endl;
  cout << "Value of: " << cond << endl;
  cout << "  Actual: false" << endl;
  cout << "Expected: true" << endl;
  _result->register_failure(file, line);
}

void
TestCaseBase::report_expect_false_failure(string_view cond,
                                          string_view file, size_t line)
{
  cout << file << ":" << line << ": Failure" << endl;
  cout << "Value of: " << cond << endl;
  cout << "  Actual: true" << endl;
  cout << "Expected: false" << endl;
  _result->register_failure(file, line);
}

class TestManagerImpl : public TestManager {
public:
  TestManagerImpl() {};
  virtual ~TestManagerImpl() {};

private:
  const TestRegistryStatus* register_test(string_view test_suite,
                                          string_view test_case,
                                          TestCasePtr test_class);
  int run(const TestPlanConfig& cfg);

private:
  list<tuple<string, TestCasePtr> > _registry;
  list<TestRegistryStatus>          _status;
};

TestManager*
TestManager::get_instance()
{
  static TestManagerImpl x;
  return &x;
}

const TestRegistryStatus*
TestManagerImpl::register_test(string_view test_suite,
                               string_view test_case,
                               TestCasePtr test_class)
{
  string full_test_id{test_suite};
  full_test_id += ".";
  full_test_id += test_case;
  _registry.push_back(make_tuple(full_test_id, std::move(test_class)));
  _status.push_back({ true, test_case.length() < 9 ||
                            test_case.substr(0, 9) != "DISABLED_" });
  return &(_status.back());
}

static string_view
suite_from_test_id(string_view id)
{
  return id.substr(0, id.find('.'));
}

class TestFilter
{
public:
  TestFilter(string_view filter);
  bool operator()(string_view input) const;
private:
  static size_t match(string_view in, string_view target,
                      bool match_start);
  bool matches_alt_(const list<string_view>& alt,
                    string_view             input) const;
  void parse_alt_(string_view alt_str);
private:
  // Each entry is one alternative; the filter matches if ANY
  // alternative matches the input. Top-level ':' in the filter
  // string separates alternatives. (Test names contain only [a-z_.0-9]
  // so ':' is unambiguous.)
  list<list<string_view>> _alternatives;
};

size_t
TestFilter::match(string_view in, string_view target, bool match_start)
{
  if (in.length() >= target.length()) {
    size_t search_range = match_start ? 1 : (in.length() - target.length() + 1);
    for (size_t ret=0; ret<search_range; ++ret) {
      size_t tp = 0;
      for (; tp<target.length(); ++tp) {
        if (target[tp] != in[ret+tp] && target[tp] != '?') { break; }
      }
      if (tp == target.length()) { return ret; }
    }
  }
  return string_view::npos;
}

void
TestFilter::parse_alt_(string_view alt_str)
{
  list<string_view> segs;
  size_t seg_begin = 0;
  for (size_t pos=0; pos<alt_str.length(); ++pos) {
    if (alt_str[pos] != '*') { continue; }
    if (seg_begin != pos) {
      segs.push_back(alt_str.substr(seg_begin, pos - seg_begin));
    }
    if (segs.size() == 0 || segs.back() != "*") {
      segs.push_back(alt_str.substr(pos, 1));
    }
    seg_begin = pos + 1;
  }
  size_t remain = alt_str.length() - seg_begin;
  if (remain) { segs.push_back(alt_str.substr(seg_begin, remain)); }
  _alternatives.push_back(std::move(segs));
}

TestFilter::TestFilter(string_view filter)
{
  // Split on top-level ':' into independent alternatives. Empty
  // alternatives (consecutive ':' or leading/trailing ':') are
  // dropped silently.
  size_t start = 0;
  while (start <= filter.length()) {
    size_t end = filter.find(':', start);
    if (end == string_view::npos) { end = filter.length(); }
    auto piece = filter.substr(start, end - start);
    if (!piece.empty()) {
      parse_alt_(piece);
    }
    if (end == filter.length()) { break; }
    start = end + 1;
  }
}

bool
TestFilter::matches_alt_(const list<string_view>& _filter,
                         string_view             input) const
{
  if (_filter.size() == 0) { return true; }
  else if (_filter.size() == 1) {
    const string_view& f = _filter.front();
    if (f == "*") { return true; }
    else { return match(input, f, true) == 0 && input.length() == f.length(); }
  }
  auto it = _filter.begin();
  if (*it != "*") {
    const string_view& f = *it;
    if (match(input, f, true) == 0) { input = input.substr(f.length()); }
    else { return false; }
    ++ it;
  }
  auto last = _filter.end();
  if (*(prev(last)) != "*") {
    const string_view& f = *(prev(last));
    if (f.length() > input.length()) { return false; }
    size_t l_start = input.length() - f.length();
    string_view l_input = input.substr(l_start);
    if (match(l_input, f, true) == 0) { input = input.substr(0, l_start); }
    else { return false; }
    -- last;
  }
  for (; it!=last; ++it) {
    if (*it == "*") continue;
    size_t next_start = match(input, *it, false);
    if (next_start == string_view::npos) { return false; }
    input = input.substr(next_start + it->length());
  }
  return input.length() == 0 || (*(prev(it))) == "*";
}

bool
TestFilter::operator()(string_view input) const
{
  // No alternatives means "no filter" -> accept everything.
  if (_alternatives.empty()) { return true; }
  for (const auto& alt : _alternatives) {
    if (matches_alt_(alt, input)) { return true; }
  }
  return false;
}

int
TestManagerImpl::run(const TestPlanConfig& cfg)
{
  _registry.sort([](auto& a, auto& b) { return get<0>(a) < get<0>(b); });

  TestFilter tfilter(cfg.filter);

  /* filter tests */
  list<tuple<string_view, size_t> > suite_info;
  list<tuple<const tuple<string, TestCasePtr>*,
             tuple<string_view, size_t>*> > filtered_tests;
  string_view last_suite;
  for (auto& entry : _registry) {
    auto is_test_disabled = [&cfg](string_view tid, string_view sid)
                            { return tid.length() > sid.length() + 10 &&
                                     tid.substr(sid.length() + 1, 9)
                                       == "DISABLED_" &&
                                     !cfg.also_run_disabled; };
    string_view test_id    = get<0>(entry);
    string_view curr_suite = suite_from_test_id(test_id);
    if (curr_suite != last_suite) {
      last_suite = curr_suite;
      suite_info.push_back(make_tuple(curr_suite, 0));
    }
    if (!is_test_disabled(test_id, curr_suite) && tfilter(test_id)) {
      filtered_tests.push_back(make_tuple(&entry, &(suite_info.back())));
      ++ get<1>(suite_info.back());
    }
  }
  // Filter out suites that ended up with no matching tests. Use
  // list::remove_if (the member, not std::remove_if): the std
  // version moves VALUES toward the front of the list and leaves
  // moved-from values in the tail nodes before erase() destroys
  // them. We have already stashed addresses of these list nodes
  // inside `filtered_tests`; the member function only unlinks the
  // empty-suite nodes and leaves every surviving node at its
  // original address, so those pointers stay valid.
  suite_info.remove_if([](auto& x) { return get<1>(x) == 0; });

  if (cfg.list_only) {
    string_view last_suite;
    for (auto& test : filtered_tests) {
      auto&& [ tinfo, sinfo ] = test;
      string_view curr_suite = get<0>(*sinfo);
      string_view test_id    = get<0>(*tinfo);
      if (curr_suite != last_suite) {
        last_suite = curr_suite;
        cout << curr_suite << "." << endl;
      }
      cout << "  " << test_id.substr(curr_suite.length() + 1) << endl;
    }
    return 0;
  }

#define GREEN_COUT(x)                                                          \
  if (cfg.colored_output)                                                      \
    { cout_color_green(); cout << x; cout_color_reset(); }                     \
  else { cout << x; }

#define RED_COUT(x)                                                            \
  if (cfg.colored_output)                                                      \
    { cout_color_red(); cout << x; cout_color_reset(); }                       \
  else { cout << x; }

  /* run tests */
  GREEN_COUT("[==========] ");
  cout << "Running " << filtered_tests.size() << " tests from "
       << suite_info.size() << " test suite." << endl;
  list<string_view> failed_tests;
  size_t tests_in_suite_ran = 0, tests_ran = 0, suites_ran = 0;
  tuple<string_view, size_t>* last_suite_info = nullptr;
  TIME all_begin, all_end, suite_begin, suite_end, begin, end;
  monotonic_time_stamp(&all_begin);
  for (auto& test : filtered_tests) {
    auto&& [ test_info, curr_suite_info ] = test;
    auto&& [ test_id, test_class ] = *test_info;
    auto&& [ suite_id, suite_test_count ] = *curr_suite_info;
    if (curr_suite_info != last_suite_info) {
      ++ suites_ran;
      tests_in_suite_ran = 0;
      last_suite_info = curr_suite_info;
      GREEN_COUT("[----------] ");
      cout << suite_test_count
           << (suite_test_count > 1 ? " tests from " : " test from ")
           << suite_id << endl;
      monotonic_time_stamp(&suite_begin);
    }
    GREEN_COUT("[ RUN      ] ");
    cout << test_id << endl;
    monotonic_time_stamp(&begin);
    test_class->run();
    monotonic_time_stamp(&end);
    ++ tests_in_suite_ran;
    ++ tests_ran;
    auto& result = test_class->result()->get_failures();
    if (result.size()) {
      RED_COUT("[  FAILED  ] ");
      failed_tests.push_back(test_id);
    } else {
      GREEN_COUT("[       OK ] ");
    }
    cout << test_id;
    if (cfg.print_time) {
      cout << " (" << setprecision(3) << fixed
           << elapsed_time(&begin, &end) * 1000.f << " ms)";
    }
    cout << endl;
    if (tests_in_suite_ran == suite_test_count) {
      monotonic_time_stamp(&suite_end);
      GREEN_COUT("[----------] ");
      cout << tests_in_suite_ran
           << (tests_in_suite_ran > 1 ? " tests" : " test") << " from "
           << suite_id;
      if (cfg.print_time) {
        cout << " (" << setprecision(3) << fixed
             << elapsed_time(&suite_begin, &suite_end) * 1000.f
             << " ms total)";
      }
      cout << endl;
      cout << endl;
    }
  }
  monotonic_time_stamp(&all_end);
  cout << endl;

  /* print summary information */
  GREEN_COUT("[==========] ");
  cout << tests_ran << " tests from " << suites_ran << " test suite ran.";
  if (cfg.print_time) {
    cout <<" (" << setprecision(3) << fixed
         << elapsed_time(&all_begin, &all_end) * 1000.f << " ms total)";
  }
  cout << endl;
  GREEN_COUT("[  PASSED  ] ");
  cout << tests_ran - failed_tests.size() << " tests." << endl;
  if (failed_tests.size()) {
    cout << tests_ran - failed_tests.size() << " tests." << endl;
    RED_COUT("[  FAILED  ] ");
    cout << failed_tests.size()
         << (failed_tests.size() > 1 ? " test" : " tests")
         << ", listed below:" << endl;
    for (auto& entry : failed_tests) {
      RED_COUT("[  FAILED  ] ");
      cout << entry << endl;
    }
    cout << endl << failed_tests.size()
         << (failed_tests.size() == 1 ? " FAILED TEST" : " FAILED TESTS")
         << endl;
    return 1;
  }
  return 0;
}

}
