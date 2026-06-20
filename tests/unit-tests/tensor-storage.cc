#include "minitest.h"
#include "apple-silicon/tensor-storage.h"

#include <cstdint>

using namespace vpipe;

namespace {

// File-scope counter so a plain-function-pointer deleter (the only
// kind ExternalStorageHandle accepts) can record that it ran.
int g_deleter_run_count = 0;
void counting_deleter_(void*) { ++g_deleter_run_count; }
void no_op_deleter_(void*)    {}

}  // namespace

TEST(tensor_storage, external_storage_handle_default_is_empty)
{
  ExternalStorageHandle h;
  EXPECT_TRUE(h.mtl_buffer == nullptr);
  EXPECT_TRUE(h.contents   == nullptr);
  EXPECT_TRUE(h.byte_size  == 0u);
  EXPECT_TRUE(h.deleter    == nullptr);
}

TEST(tensor_storage, deleter_fires_on_destruction)
{
  g_deleter_run_count = 0;
  uint8_t dummy = 0;
  {
    ExternalStorageHandle h;
    h.mtl_buffer = &dummy;
    h.contents   = &dummy;
    h.byte_size  = 1;
    h.deleter    = &counting_deleter_;
    // h dropped at end of scope
  }
  EXPECT_TRUE(g_deleter_run_count == 1);
}

TEST(tensor_storage, deleter_not_fired_when_mtl_buffer_null)
{
  g_deleter_run_count = 0;
  {
    ExternalStorageHandle h;
    h.deleter = &counting_deleter_;
    // mtl_buffer == nullptr, so deleter must NOT run
  }
  EXPECT_TRUE(g_deleter_run_count == 0);
}

TEST(tensor_storage, deleter_not_fired_when_deleter_null)
{
  uint8_t dummy = 0;
  ExternalStorageHandle h;
  h.mtl_buffer = &dummy;
  h.deleter    = nullptr;
  // Must not crash even though mtl_buffer is set; the deleter
  // pointer being null is the gate. (No assertion needed beyond
  // "doesn't crash".)
  (void)no_op_deleter_;  // suppress unused-warning when no_op
                        // helper isn't otherwise used.
}

TEST(tensor_storage, shared_handle_is_non_copyable)
{
  // Compile-time check: ExternalStorageHandle deleted its copy ops,
  // so we can't copy or copy-assign one. The static_asserts below
  // would fail to compile if anyone reintroduced copyability.
  static_assert(!std::is_copy_constructible_v<ExternalStorageHandle>);
  static_assert(!std::is_copy_assignable_v<ExternalStorageHandle>);
  EXPECT_TRUE(true);
}
