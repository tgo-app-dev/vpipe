/*******************************************************************************
 * @file minitest.h
 *
 * @brief Defines macros and interface classes for the minitest library.
 *******************************************************************************
 */

#ifndef MINITEST_H
#define MINITEST_H

#include <memory>
#include <string_view>

////////////////////////////////////////////////////////////////////////////////

/*******************************************************************************
 * @def TEST(suite, case)
 *
 * @brief Defines a test case.
 *
 * Users don't need to worry about the internal content of this definition.
 *******************************************************************************
 */

#define TEST(suite, case)                                                      \
  class minitest_ ## suite ## _ ## case : public minitest::TestCaseBase {      \
  public:                                                                      \
    minitest_ ## suite ## _ ## case() {};                                      \
    virtual ~minitest_ ## suite ## _ ## case() {};                             \
  private:                                                                     \
    void run();                                                                \
    static const minitest::TestRegistryStatus*                                 \
      minitest_reg_stat_ ## suite ## _ ## case;                                \
  };                                                                           \
  const minitest::TestRegistryStatus* minitest_reg_stat_ ## suite ## _ ## case \
    = static_cast<minitest::TestCollector*>                                    \
        (minitest::TestManager::get_instance())                                \
        ->register_test(# suite, # case,                                       \
                        std::make_unique<minitest_ ## suite ## _ ## case>());  \
  void minitest_ ## suite ## _ ## case::run()

#define EXPECT_TRUE(cond)                                                      \
  if (!(cond)) { report_expect_true_failure(# cond, __FILE__, __LINE__); }

#define ASSERT_TRUE(cond) EXPECT_TRUE(cond)

#define EXPECT_FALSE(cond)                                                     \
  if ((cond)) { report_expect_false_failure(# cond, __FILE__, __LINE__); }

#define ASSERT_FALSE(cond) EXPECT_FALSE(cond)

////////////////////////////////////////////////////////////////////////////////

namespace minitest {

class TestResult;
typedef std::unique_ptr<TestResult> TestResultPtr;

class TestCaseBase {
public:
  TestCaseBase();
  virtual ~TestCaseBase();

  virtual void run() = 0;
  const TestResult* result() const;

protected:
  void report_failure(std::string_view file, size_t line);
  void report_expect_true_failure(std::string_view cond,
                                  std::string_view file, size_t line);
  void report_expect_false_failure(std::string_view cond,
                                   std::string_view file, size_t line);
private:
  TestResultPtr _result;
};
typedef std::unique_ptr<TestCaseBase> TestCasePtr;

struct TestPlanConfig {
  bool also_run_disabled;
  bool colored_output;
  bool list_only;
  bool print_time;
  std::string_view filter;
  TestPlanConfig();
};

struct TestRegistryStatus {
  bool registered;
  bool enabled;
};

class TestCollector {
public:
  TestCollector() {};
  virtual ~TestCollector() {};

  virtual const TestRegistryStatus* register_test(std::string_view test_suite,
                                                  std::string_view test_case,
                                                  TestCasePtr test_class) = 0;
};

class TestManager : public TestCollector {
public:
  TestManager(const TestManager&) = delete;
  virtual ~TestManager() {};

  static TestManager* get_instance();

  virtual int run(const TestPlanConfig& cfg) = 0;

protected:
  TestManager() {};
};

}

#endif
