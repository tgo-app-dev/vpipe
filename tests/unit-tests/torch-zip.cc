// A torch.save() checkpoint, read in place without torch.
//
// THE ARCHIVES HERE ARE BUILT BY HAND, byte for byte in the shape torch
// writes them: STORED entries with a trailing data descriptor (so every
// local header says zero and only the central directory has the sizes),
// payloads padded to 64 bytes through an extra field, ZIP64 end records,
// and a protocol-2 pickle using the opcodes torch's own pickler emits for
// a state_dict. That keeps the default suite free of a Python or torch
// dependency while exercising the same paths a real file takes.
//
// The real files are the last test, gated on the published FlashVSR
// directory, and they are what the synthetic ones were checked against.
//
// Env:
//   VPIPE_FLASHVSR_TEST_MODEL_PATH  a FlashVSR-v1.1 directory AS PUBLISHED
//                                   (the torch files beside the denoiser)
//   VPIPE_TORCH_ZIP_DUMP            also print every tensor, sorted, for a
//                                   diff against an independent reader

#include "minitest.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/torch-zip.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace vpipe::genai;

namespace {

namespace fs = std::filesystem;

// ---- a pickle, in torch's own opcodes --------------------------------

struct Pickle {
  std::vector<std::uint8_t> b;

  Pickle() { b = {0x80, 0x02}; }                       // PROTO 2
  void op(std::uint8_t o) { b.push_back(o); }
  void
  global(const char* mod, const char* name)
  {
    op('c');
    for (const char* p = mod; *p; ++p) { b.push_back((std::uint8_t)*p); }
    b.push_back('\n');
    for (const char* p = name; *p; ++p) { b.push_back((std::uint8_t)*p); }
    b.push_back('\n');
  }
  void
  str(const std::string& s)
  {
    op('X');
    const std::uint32_t n = (std::uint32_t)s.size();
    for (int k = 0; k < 4; ++k) { b.push_back((std::uint8_t)(n >> (8 * k))); }
    b.insert(b.end(), s.begin(), s.end());
  }
  void
  i32(std::int32_t v)
  {
    op('J');
    for (int k = 0; k < 4; ++k) {
      b.push_back((std::uint8_t)((std::uint32_t)v >> (8 * k)));
    }
  }
  void
  ints(const std::vector<std::int64_t>& v)
  {
    op('(');
    for (const auto x : v) { i32((std::int32_t)x); }
    op('t');
  }
  // torch._utils._rebuild_tensor_v2(storage, offset, size, stride, False,
  //                                 OrderedDict())
  void
  tensor(const char* storage_cls, const std::string& key, std::int64_t numel,
         std::int64_t offset, const std::vector<std::int64_t>& shape,
         const std::vector<std::int64_t>& stride)
  {
    global("torch._utils", "_rebuild_tensor_v2");
    op('(');
    op('(');
    str("storage");
    global("torch", storage_cls);
    str(key);
    str("cpu");
    i32((std::int32_t)numel);
    op('t');
    op('Q');                                           // BINPERSID
    i32((std::int32_t)offset);
    ints(shape);
    ints(stride);
    op(0x89);                                          // NEWFALSE
    global("collections", "OrderedDict");
    op(')');
    op('R');
    op('t');
    op('R');
  }
  void
  begin_dict()
  {
    global("collections", "OrderedDict");
    op(')');
    op('R');
    op('(');
  }
  void end_dict() { op('u'); }                         // SETITEMS
  std::vector<std::uint8_t> done() { op('.'); return b; }
};

// ---- a zip, in torch's own layout ------------------------------------

struct Zip {
  std::vector<std::uint8_t> b;
  struct E {
    std::string   name;
    std::uint64_t lho, size;
    std::uint16_t method;
  };
  std::vector<E> es;

  void
  le(std::uint64_t v, int n)
  {
    for (int k = 0; k < n; ++k) { b.push_back((std::uint8_t)(v >> (8 * k))); }
  }
  void u16(std::uint16_t v) { le(v, 2); }
  void u32(std::uint32_t v) { le(v, 4); }
  void u64(std::uint64_t v) { le(v, 8); }

  void
  add(const std::string& name, const std::vector<std::uint8_t>& data,
      std::uint16_t method = 0)
  {
    const std::uint64_t lho = b.size();
    // Pad the payload onto a 64-byte boundary through an extra field,
    // the way torch does.
    const std::uint64_t bare = lho + 30 + name.size() + 4;
    const std::uint16_t pad = (std::uint16_t)((64 - bare % 64) % 64);
    u32(0x04034b50); u16(45); u16(0x08); u16(method); u16(0); u16(0);
    u32(0); u32(0); u32(0);                            // descriptor has them
    u16((std::uint16_t)name.size()); u16((std::uint16_t)(4 + pad));
    b.insert(b.end(), name.begin(), name.end());
    u16(0x4b46); u16(pad);
    b.insert(b.end(), pad, 0);
    b.insert(b.end(), data.begin(), data.end());
    u32(0x08074b50); u32(0); u32((std::uint32_t)data.size());
    u32((std::uint32_t)data.size());
    es.push_back(E{name, lho, data.size(), method});
  }

  std::vector<std::uint8_t>
  finish()
  {
    const std::uint64_t cd_off = b.size();
    for (const E& e : es) {
      u32(0x02014b50); u16(45); u16(45); u16(0x08); u16(e.method);
      u16(0); u16(0); u32(0);
      u32((std::uint32_t)e.size); u32((std::uint32_t)e.size);
      u16((std::uint16_t)e.name.size()); u16(0); u16(0); u16(0); u16(0);
      u32(0); u32((std::uint32_t)e.lho);
      b.insert(b.end(), e.name.begin(), e.name.end());
    }
    const std::uint64_t cd_size = b.size() - cd_off;
    const std::uint64_t z64 = b.size();
    u32(0x06064b50); u64(44); u16(45); u16(45); u32(0); u32(0);
    u64(es.size()); u64(es.size()); u64(cd_size); u64(cd_off);
    u32(0x07064b50); u32(0); u64(z64); u32(1);
    u32(0x06054b50); u16(0); u16(0); u16(0xffff); u16(0xffff);
    u32(0xffffffff); u32(0xffffffff); u16(0);
    return b;
  }
};

std::vector<std::uint8_t>
f32_bytes(const std::vector<float>& v)
{
  std::vector<std::uint8_t> out(v.size() * 4);
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

fs::path
write_tmp_(const std::string& stem, const std::vector<std::uint8_t>& bytes)
{
  const fs::path p = fs::temp_directory_path() /
      (stem + "-" + std::to_string((long)::getpid()) + ".pth");
  std::ofstream o(p, std::ios::binary);
  o.write(reinterpret_cast<const char*>(bytes.data()), (long)bytes.size());
  return p;
}

// read_index over a file, with its message.
bool
index_of_(const fs::path& p, std::vector<torch_zip::Tensor>* out,
          std::string* err)
{
  const int fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0) { *err = "open"; return false; }
  struct stat st {};
  ::fstat(fd, &st);
  const bool ok = torch_zip::read_index(fd, (std::uint64_t)st.st_size, out,
                                        err);
  ::close(fd);
  return ok;
}

}  // namespace

// The whole path a real state_dict takes: two storages of two dtypes, and
// one tensor that is a VIEW into another's storage at a non-zero offset --
// which is how the Wan VAE's 194 tensors are laid out.
TEST(torch_zip, a_state_dict_maps_as_a_shard)
{
  Pickle pk;
  pk.begin_dict();
  pk.str("a.weight");
  pk.tensor("FloatStorage", "0", 9, 0, {2, 3}, {3, 1});
  pk.str("b");
  pk.tensor("BFloat16Storage", "1", 4, 0, {4}, {1});
  pk.str("c");
  pk.tensor("FloatStorage", "0", 9, 6, {3}, {1});
  pk.end_dict();

  Zip z;
  z.add("ckpt/data.pkl", pk.done());
  z.add("ckpt/byteorder", {'l', 'i', 't', 't', 'l', 'e'});
  z.add("ckpt/data/0", f32_bytes({0, 1, 2, 3, 4, 5, 6, 7, 8}));
  z.add("ckpt/data/1", {1, 2, 3, 4, 5, 6, 7, 8});
  z.add("ckpt/version", {'3'});
  const fs::path p = write_tmp_("vpipe-torchzip-sd", z.finish());

  auto w = MetalLlamaWeights::open(p.string());
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }
  const auto* a = w->info("a.weight");
  const auto* b = w->info("b");
  const auto* c = w->info("c");
  ASSERT_TRUE(a != nullptr && b != nullptr && c != nullptr);
  if (a == nullptr || b == nullptr || c == nullptr) { return; }
  EXPECT_TRUE(a->dtype == "F32" && a->shape == std::vector<int64_t>({2, 3}));
  EXPECT_TRUE(a->nbytes == 24);
  EXPECT_TRUE(b->dtype == "BF16" && b->nbytes == 8);
  EXPECT_TRUE(c->dtype == "F32" && c->nbytes == 12);
  // A tensor that STARTS its storage sits on the 64-byte grid torch pads
  // to, which is what makes a torch file zero-copy mappable at all. A
  // view does not: `c` starts 24 bytes in, so it is exactly one of the
  // misaligned tensors the loader copies instead -- as the Wan VAE's
  // shared-storage views are.
  EXPECT_TRUE(a->offset % 64 == 0 && b->offset % 64 == 0);
  EXPECT_TRUE((c->offset - a->offset) == 24);
  EXPECT_TRUE(w->alignment().misaligned == 1);

  std::vector<float> fa(6), fc(3);
  EXPECT_TRUE(w->read_into("a.weight", fa.data(), 24));
  EXPECT_TRUE(w->read_into("c", fc.data(), 12));
  EXPECT_TRUE(fa == std::vector<float>({0, 1, 2, 3, 4, 5}));
  // The view starts SIX elements into its storage, not at its start.
  EXPECT_TRUE(fc == std::vector<float>({6, 7, 8}));
  std::uint8_t bb[8] = {};
  EXPECT_TRUE(w->read_into("b", bb, 8));
  EXPECT_TRUE(bb[0] == 1 && bb[7] == 8);
  std::error_code ec;
  fs::remove(p, ec);
}

// A training checkpoint's wrapper is unwrapped, nested dicts are joined
// with '.', and a bare tensor is named for its archive.
TEST(torch_zip, names_follow_the_object_graph)
{
  {
    Pickle pk;
    pk.op('}');                                        // EMPTY_DICT
    pk.str("state_dict");
    pk.op('}');
    pk.str("enc");
    pk.begin_dict();
    pk.str("w");
    pk.tensor("HalfStorage", "0", 2, 0, {2}, {1});
    pk.end_dict();
    pk.op('s');                                        // SETITEM
    pk.op('s');
    Zip z;
    z.add("archive/data.pkl", pk.done());
    z.add("archive/data/0", {0, 0, 0, 0});
    const fs::path p = write_tmp_("vpipe-torchzip-nested", z.finish());
    std::vector<torch_zip::Tensor> t;
    std::string err;
    const bool ok = index_of_(p, &t, &err);
    if (!ok) { std::printf("  nested: %s\n", err.c_str()); }
    EXPECT_TRUE(ok && t.size() == 1 && t[0].name == "enc.w" &&
                t[0].dtype == "F16");
    std::error_code ec;
    fs::remove(p, ec);
  }
  {
    Pickle pk;
    pk.tensor("BFloat16Storage", "0", 6, 0, {1, 2, 3}, {6, 3, 1});
    Zip z;
    z.add("posi_prompt/data.pkl", pk.done());
    z.add("posi_prompt/data/0", std::vector<std::uint8_t>(12, 7));
    const fs::path p = write_tmp_("vpipe-torchzip-bare", z.finish());
    std::vector<torch_zip::Tensor> t;
    std::string err;
    const bool ok = index_of_(p, &t, &err);
    if (!ok) { std::printf("  bare: %s\n", err.c_str()); }
    EXPECT_TRUE(ok && t.size() == 1 && t[0].name == "posi_prompt" &&
                t[0].shape == std::vector<int64_t>({1, 2, 3}));
    std::error_code ec;
    fs::remove(p, ec);
  }
}

// A TRAINING STATE: `state_dict` unwrapped as before, and its sibling
// dicts kept under their own keys -- the EMA model a release decodes with
// sits in `ema_state_dict` (MAGI-2's turbo decoder). An epoch count beside
// them is no tensor and is left out.
TEST(torch_zip, a_training_state_keeps_its_ema_beside_the_model)
{
  Pickle pk;
  pk.op('}');                                        // EMPTY_DICT
  pk.str("epoch");
  pk.i32(3);
  pk.op('s');
  pk.str("state_dict");
  pk.begin_dict();
  pk.str("w");
  pk.tensor("HalfStorage", "0", 2, 0, {2}, {1});
  pk.end_dict();
  pk.op('s');
  pk.str("ema_state_dict");
  pk.begin_dict();
  pk.str("w");
  pk.tensor("HalfStorage", "1", 2, 0, {2}, {1});
  pk.end_dict();
  pk.op('s');
  Zip z;
  z.add("archive/data.pkl", pk.done());
  z.add("archive/data/0", {0, 0, 0, 0});
  z.add("archive/data/1", {1, 1, 1, 1});
  const fs::path p = write_tmp_("vpipe-torchzip-ema", z.finish());
  std::vector<torch_zip::Tensor> t;
  std::string err;
  const bool ok = index_of_(p, &t, &err);
  if (!ok) { std::printf("  ema: %s\n", err.c_str()); }
  bool raw = false, ema = false;
  for (const auto& x : t) {
    raw = raw || x.name == "w";
    ema = ema || x.name == "ema_state_dict.w";
  }
  std::printf("[torch_zip] training state: %zu tensors, raw %d ema %d\n",
              t.size(), raw ? 1 : 0, ema ? 1 : 0);
  EXPECT_TRUE(ok && t.size() == 2 && raw && ema);
  std::error_code ec;
  fs::remove(p, ec);
}

// What it will not do, each refused with a message naming why. The first
// is the one that matters: a pickle naming any global off the allowlist
// is not executed, not skipped -- the read fails.
TEST(torch_zip, refuses_what_it_cannot_map)
{
  auto expect_refusal = [&](const char* what,
                            const std::vector<std::uint8_t>& archive,
                            const char* needle) {
    const fs::path p = write_tmp_("vpipe-torchzip-refuse", archive);
    std::vector<torch_zip::Tensor> t;
    std::string err;
    const bool ok = index_of_(p, &t, &err);
    std::printf("  %-14s -> %s\n", what, ok ? "ACCEPTED" : err.c_str());
    EXPECT_TRUE(!ok && t.empty());
    EXPECT_TRUE(err.find(needle) != std::string::npos);
    EXPECT_TRUE(!MetalLlamaWeights::open(p.string()).has_value());
    std::error_code ec;
    fs::remove(p, ec);
  };
  {
    Pickle pk;
    pk.global("os", "system");
    pk.str("echo hi");
    pk.op(0x85);                                       // TUPLE1
    pk.op('R');
    Zip z;
    z.add("a/data.pkl", pk.done());
    expect_refusal("foreign global", z.finish(), "refusing global 'os.system'");
  }
  {
    Pickle pk;
    pk.begin_dict();
    pk.str("t");
    pk.tensor("FloatStorage", "0", 6, 0, {2, 3}, {1, 2});   // transposed
    pk.end_dict();
    Zip z;
    z.add("a/data.pkl", pk.done());
    z.add("a/data/0", std::vector<std::uint8_t>(24, 0));
    expect_refusal("non-contiguous", z.finish(), "not contiguous");
  }
  {
    Pickle pk;
    pk.begin_dict();
    pk.str("t");
    pk.tensor("FloatStorage", "0", 2, 0, {2}, {1});
    pk.end_dict();
    Zip z;
    z.add("a/data.pkl", pk.done());
    z.add("a/data/0", std::vector<std::uint8_t>(8, 0), /*method=*/8);
    expect_refusal("deflated", z.finish(), "compressed");
  }
  {
    Pickle pk;
    pk.begin_dict();
    pk.str("t");
    pk.tensor("FloatStorage", "0", 2, 1, {2}, {1});     // one past the end
    pk.end_dict();
    Zip z;
    z.add("a/data.pkl", pk.done());
    z.add("a/data/0", std::vector<std::uint8_t>(8, 0));
    expect_refusal("overrun", z.finish(), "past the end");
  }
}

// FlashVSR-v1.1's three torch files as the model page publishes them. The
// offsets asserted are the ones an independent reader (Python's zipfile
// and pickle, no torch) computed for the same files; VPIPE_TORCH_ZIP_DUMP
// prints the whole table for the full diff.
TEST(torch_zip, the_published_flashvsr_files)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  const fs::path r(root);
  if (!fs::exists(r / "LQ_proj_in.ckpt")) {
    std::printf("  not the published layout (no LQ_proj_in.ckpt); skipped\n");
    return;
  }
  struct Want {
    const char* file;
    std::size_t tensors;
    const char* name;
    const char* dtype;
    std::vector<int64_t> shape;
    std::uint64_t offset, nbytes;
  };
  const std::vector<Want> wants = {
    {"LQ_proj_in.ckpt", 8, "conv1.weight", "BF16", {2048, 768, 4, 3, 3},
     1088, 113246208},
    {"Wan2.1_VAE.pth", 194, "conv1.weight", "F32", {32, 32, 1, 1, 1},
     253752704, 4096},
    {"posi_prompt.pth", 1, "posi_prompt", "BF16", {1, 512, 4096}, 448,
     4194304},
  };
  for (const Want& wt : wants) {
    std::vector<torch_zip::Tensor> t;
    std::string err;
    const bool ok = index_of_(r / wt.file, &t, &err);
    std::printf("  %-16s %s, %zu tensors\n", wt.file,
                ok ? "read" : err.c_str(), t.size());
    ASSERT_TRUE(ok);
    if (!ok) { continue; }
    EXPECT_TRUE(t.size() == wt.tensors);
    auto it = std::find_if(t.begin(), t.end(), [&](const torch_zip::Tensor& x) {
      return x.name == wt.name;
    });
    ASSERT_TRUE(it != t.end());
    if (it == t.end()) { continue; }
    EXPECT_TRUE(it->dtype == wt.dtype && it->shape == wt.shape);
    EXPECT_TRUE(it->offset == wt.offset && it->nbytes == wt.nbytes);
    if (std::getenv("VPIPE_TORCH_ZIP_DUMP") != nullptr) {
      std::sort(t.begin(), t.end(), [](const auto& x, const auto& y) {
        return x.name < y.name;
      });
      for (const auto& x : t) {
        std::string shape;
        for (std::size_t k = 0; k < x.shape.size(); ++k) {
          shape += (k ? "x" : "") + std::to_string(x.shape[k]);
        }
        std::printf("DUMP %s %s %s %s %llu %llu\n", wt.file, x.name.c_str(),
                    x.dtype.c_str(), shape.c_str(),
                    (unsigned long long)x.offset,
                    (unsigned long long)x.nbytes);
      }
    }
  }
}
