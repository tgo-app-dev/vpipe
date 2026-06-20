#include "minitest.h"
#include "common/library-handle.h"
#include "common/session.h"
#include "common/vpipe-format.h"
#include <exception>

using namespace std;
using namespace vpipe;

#if defined(__APPLE__)
static constexpr const char* kSystemLib = "/usr/lib/libSystem.B.dylib";
#elif defined(__linux__)
static constexpr const char* kSystemLib = "libc.so.6";
#else
static constexpr const char* kSystemLib = nullptr;
#endif

TEST(libraryhandle, open_system_lib_succeeds) {
  if (!kSystemLib) {
    return;
  }
  Session s;
  LibraryHandle h(&s, kSystemLib);
  EXPECT_TRUE(h.valid());
  EXPECT_TRUE(h.path() == kSystemLib);
}

TEST(libraryhandle, open_missing_required_throws) {
  Session s;
  bool threw = false;
  try {
    LibraryHandle h(&s, "/nonexistent_xyz_xyz.dylib");
    (void)h;
  } catch (exception&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(libraryhandle, open_missing_optional_warns_and_invalid) {
  Session s;
  LibraryHandle h(&s, "/nonexistent_xyz_xyz.dylib",
                  LibraryHandle::LoadMode::Optional);
  EXPECT_FALSE(h.valid());
  EXPECT_FALSE(static_cast<bool>(h));
}

TEST(libraryhandle, resolve_known_symbol) {
  if (!kSystemLib) {
    return;
  }
  Session s;
  LibraryHandle h(&s, kSystemLib);
  EXPECT_TRUE(h.get_symbol("malloc") != nullptr);
  EXPECT_TRUE(h.require_symbol("malloc") != nullptr);

  using malloc_t = void* (size_t);
  using free_t   = void  (void*);
  auto* my_malloc = h.symbol<malloc_t>("malloc");
  auto* my_free   = h.symbol<free_t>("free");
  EXPECT_TRUE(my_malloc != nullptr);
  EXPECT_TRUE(my_free != nullptr);
  void* p = my_malloc(8);
  EXPECT_TRUE(p != nullptr);
  my_free(p);
}

TEST(libraryhandle, resolve_missing_symbol) {
  if (!kSystemLib) {
    return;
  }
  Session s;
  LibraryHandle h(&s, kSystemLib);
  EXPECT_TRUE(h.get_symbol("definitely_not_a_real_symbol_xyz") == nullptr);

  bool threw = false;
  try {
    (void)h.require_symbol("definitely_not_a_real_symbol_xyz");
  } catch (exception&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(libraryhandle, move_transfers_ownership) {
  if (!kSystemLib) {
    return;
  }
  Session s;
  LibraryHandle a(&s, kSystemLib);
  EXPECT_TRUE(a.valid());

  LibraryHandle b(std::move(a));
  EXPECT_TRUE(b.valid());
  EXPECT_FALSE(a.valid());

  LibraryHandle c(&s, kSystemLib);
  EXPECT_TRUE(c.valid());
  c = std::move(b);
  EXPECT_TRUE(c.valid());
  EXPECT_FALSE(b.valid());
}

TEST(libraryhandle, path_round_trips) {
  if (!kSystemLib) {
    return;
  }
  Session s;
  LibraryHandle h(&s, kSystemLib);
  EXPECT_TRUE(h.path() == kSystemLib);
}
