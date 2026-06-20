#include "minitest.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"

#include <Metal/Metal.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

MetalCompute*
get_mc_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    return nullptr;
  }
  return mc;
}

std::string
unique_tmp_path_(const char* suffix)
{
  auto dir = std::filesystem::temp_directory_path();
  auto p = dir / (std::string("vpipe-mc-archive-")
                  + std::to_string(::getpid())
                  + "-" + suffix + ".metallib-cache");
  std::filesystem::remove(p);  // start clean
  return p.string();
}

}  // namespace

TEST(metal_compute_binary_archive, default_no_archive) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // Without explicit set_binary_archive_path, save returns false
  // and the archive counters stay at zero.
  EXPECT_FALSE(mc->save_binary_archive());
  ComputeLibrary lib = mc->load_library("noop");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("noop");
  if (!fn.valid()) {
    return;
  }
  const auto s = mc->pso_archive_stats();
  EXPECT_TRUE(s.compiled_with_archive_set == 0);
  EXPECT_TRUE(s.added_to_archive == 0);
}

TEST(metal_compute_binary_archive,
     set_path_enables_archive_counting) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  const std::string path = unique_tmp_path_("set");
  EXPECT_TRUE(mc->set_binary_archive_path(path));

  ComputeLibrary lib = mc->load_library("noop");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("noop");
  if (!fn.valid()) {
    return;
  }
  const auto s = mc->pso_archive_stats();
  EXPECT_TRUE(s.compiled_with_archive_set >= 1);
  // added_to_archive may or may not increment depending on whether
  // Metal accepted the descriptor (some drivers reject; others
  // accept). We don't assert on it here; the load test below
  // proves the end-to-end persistence path.
}

TEST(metal_compute_binary_archive, save_writes_non_empty_file) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  const std::string path = unique_tmp_path_("save");
  if (!mc->set_binary_archive_path(path)) {
    return;
  }
  ComputeLibrary lib = mc->load_library("noop");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("noop");
  if (!fn.valid()) {
    return;
  }

  EXPECT_TRUE(mc->save_binary_archive());

  std::ifstream in(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(in.good());
  const std::streamsize size = in.tellg();
  EXPECT_TRUE(size > 0);
  in.close();

  // Cleanup.
  std::filesystem::remove(path);
}

TEST(metal_compute_binary_archive, empty_path_disables) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // Set then unset.
  const std::string path = unique_tmp_path_("disable");
  EXPECT_TRUE(mc->set_binary_archive_path(path));
  EXPECT_TRUE(mc->set_binary_archive_path(""));
  // After disabling, save returns false.
  EXPECT_FALSE(mc->save_binary_archive());
}

TEST(metal_compute_binary_archive,
     reload_existing_archive_succeeds) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // Create + save a non-trivial archive.
  const std::string path = unique_tmp_path_("reload");
  if (!mc->set_binary_archive_path(path)) {
    return;
  }
  ComputeLibrary lib = mc->load_library("noop");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("noop");
  if (!fn.valid()) {
    return;
  }
  if (!mc->save_binary_archive()) {
    std::filesystem::remove(path);
    return;
  }

  // Fresh Session/MetalCompute; point it at the same archive
  // file. set_binary_archive_path should succeed (load path).
  {
    Session sess2;
    MetalCompute* mc2 = get_mc_(sess2);
    if (mc2 != nullptr) {
      EXPECT_TRUE(mc2->set_binary_archive_path(path));
    }
  }
  std::filesystem::remove(path);
}
