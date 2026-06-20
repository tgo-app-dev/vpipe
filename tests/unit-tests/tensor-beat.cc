#include "minitest.h"
#include "apple-silicon/tensor-beat.h"

#include <cstdint>
#include <cstring>
#include <vector>

using namespace vpipe;

namespace {

// Build a contiguous f32 TensorBeat with the given shape and storage
// pre-zeroed to `n_elems * 4` bytes (default dtype is F32).
TensorBeat
make_f32_contig_(std::vector<int64_t> shape)
{
  TensorBeat tb;
  tb.shape = std::move(shape);
  tb.resize_contiguous(tb.element_count());
  return tb;
}

// Assign one f32 element by element index (0-based, flat). Caller is
// responsible for index validity.
void
set_f32_(TensorBeat& tb, size_t flat_idx, float v)
{
  tb.as_f32()[flat_idx] = v;
}

}  // namespace

TEST(tensor_beat, contiguous_default)
{
  TensorBeat tb = make_f32_contig_({3, 4, 5});
  EXPECT_TRUE(tb.is_contiguous());
  EXPECT_TRUE(tb.element_count() == 60u);
  EXPECT_TRUE(tb.dtype == TensorBeat::DType::F32);
  EXPECT_TRUE(tb.element_byte_size() == 4u);
  EXPECT_TRUE(tb.data.size() == 60u * 4u);
  auto s = tb.contiguous_strides();
  EXPECT_TRUE(s.size() == 3u);
  EXPECT_TRUE(s[0] == 20);
  EXPECT_TRUE(s[1] == 5);
  EXPECT_TRUE(s[2] == 1);
}

TEST(tensor_beat, pitch_padded_strided_layout_is_not_contiguous)
{
  TensorBeat tb;
  tb.dtype   = TensorBeat::DType::F32;
  tb.shape   = {3, 4, 5};
  tb.strides = {4 * 8, 8, 1};       // pitch 8 over inner width 5
  tb.data.assign(3 * 4 * 8 * sizeof(float), 0); // 96 floats of storage
  EXPECT_TRUE(!tb.is_contiguous());
  EXPECT_TRUE(tb.element_count() == 60u);   // logical count unchanged
  EXPECT_TRUE(tb.data.size() == 96u * sizeof(float));
}

TEST(tensor_beat, strides_equal_to_contiguous_is_contiguous)
{
  // Explicit, matching strides count as contiguous.
  TensorBeat tb;
  tb.shape   = {2, 3, 4};
  tb.strides = {12, 4, 1};
  tb.resize_contiguous(24);
  // resize_contiguous clears strides; restore them so we test the
  // "explicit strides matching contiguous" branch.
  tb.strides = {12, 4, 1};
  EXPECT_TRUE(tb.is_contiguous());
}

TEST(tensor_beat, materialize_contiguous_already_contiguous_is_a_copy)
{
  TensorBeat tb = make_f32_contig_({2, 3});
  const float values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  for (size_t i = 0; i < 6; ++i) {
    set_f32_(tb, i, values[i]);
  }
  auto out_f32 = tb.materialize_contiguous_as<float>();
  EXPECT_TRUE(out_f32.size() == 6u);
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_TRUE(out_f32[i] == values[i]);
  }
}

TEST(tensor_beat, materialize_contiguous_pitch_padded)
{
  // shape [3, 4, 5], pitch 8. Inner stride is 1, outer two carry
  // the pitch. Fill storage so that data[c*32 + y*8 + x] = encodes
  // (c, y, x) -- specifically c*100 + y*10 + x.
  TensorBeat tb;
  tb.dtype          = TensorBeat::DType::F32;
  tb.shape          = {3, 4, 5};
  tb.strides        = {4 * 8, 8, 1};
  tb.storage_offset = 0;
  tb.data.assign(3 * 4 * 8 * sizeof(float), 0);
  // Initialise padding with -1.0f sentinel; only logical cells then
  // overwritten with c*100 + y*10 + x.
  {
    float* p = reinterpret_cast<float*>(tb.data.data());
    for (size_t i = 0; i < 3u * 4u * 8u; ++i) {
      p[i] = -1.0f;
    }
  }
  float* f = reinterpret_cast<float*>(tb.data.data());
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 5; ++x) {
        f[c * 32 + y * 8 + x] =
          static_cast<float>(c * 100 + y * 10 + x);
      }
    }
  }
  auto out = tb.materialize_contiguous_as<float>();
  EXPECT_TRUE(out.size() == 60u);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 5; ++x) {
        const float expected = static_cast<float>(c * 100 + y * 10 + x);
        EXPECT_TRUE(out[c * 20 + y * 5 + x] == expected);
      }
    }
  }
}

TEST(tensor_beat, zero_dim_scalar_round_trip)
{
  // PyTorch convention: shape={} is a 0-d scalar with 1 element.
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = {};
  tb.resize_contiguous(1);
  set_f32_(tb, 0, 42.5f);
  EXPECT_TRUE(tb.element_count() == 1u);
  EXPECT_TRUE(tb.is_contiguous());
  auto out = tb.materialize_contiguous_as<float>();
  EXPECT_TRUE(out.size() == 1u);
  EXPECT_TRUE(out[0] == 42.5f);
}

TEST(tensor_beat, materialize_contiguous_one_d_strided)
{
  // Degenerate but valid: shape size 1, strides set.
  // Innermost stride must be 1 for v1; pitch only applies to outer
  // dims. So a 1-D strided layout is essentially contiguous unless
  // storage_offset is non-zero -- which v1 doesn't use. Treat as
  // contiguous.
  TensorBeat tb;
  tb.dtype   = TensorBeat::DType::F32;
  tb.shape   = {5};
  tb.strides = {1};
  tb.resize_contiguous(5);
  tb.strides = {1};  // restore after resize_contiguous() cleared
  const float values[5] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  for (size_t i = 0; i < 5; ++i) {
    set_f32_(tb, i, values[i]);
  }
  EXPECT_TRUE(tb.is_contiguous());
  auto out = tb.materialize_contiguous_as<float>();
  EXPECT_TRUE(out.size() == 5u);
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_TRUE(out[i] == values[i]);
  }
}

TEST(tensor_beat, u8_dtype_storage_and_accessors)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::U8;
  tb.shape = {3, 4};
  tb.resize_contiguous(12);
  EXPECT_TRUE(tb.element_byte_size() == 1u);
  EXPECT_TRUE(tb.data.size() == 12u);
  for (size_t i = 0; i < 12; ++i) {
    tb.as_u8()[i] = static_cast<uint8_t>(i * 7);
  }
  auto out = tb.materialize_contiguous_as<uint8_t>();
  EXPECT_TRUE(out.size() == 12u);
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_TRUE(out[i] == static_cast<uint8_t>(i * 7));
  }
}

TEST(tensor_beat, dtype_names_and_byte_sizes)
{
  EXPECT_TRUE(TensorBeat::byte_size_of(TensorBeat::DType::U8) == 1u);
  EXPECT_TRUE(TensorBeat::byte_size_of(TensorBeat::DType::I8) == 1u);
  EXPECT_TRUE(TensorBeat::byte_size_of(TensorBeat::DType::Bf16) == 2u);
  EXPECT_TRUE(TensorBeat::byte_size_of(TensorBeat::DType::F32) == 4u);
  using namespace std::string_literals;
  EXPECT_TRUE(std::string(
      TensorBeat::name_of(TensorBeat::DType::U8))  == "u8"s);
  EXPECT_TRUE(std::string(
      TensorBeat::name_of(TensorBeat::DType::I8))  == "i8"s);
  EXPECT_TRUE(std::string(
      TensorBeat::name_of(TensorBeat::DType::Bf16)) == "bf16"s);
  EXPECT_TRUE(std::string(
      TensorBeat::name_of(TensorBeat::DType::F32)) == "f32"s);
}

TEST(tensor_beat, data_buffer_is_simd_aligned)
{
  // AlignedAllocator guarantees 64-byte alignment. Build a few
  // tensors of different dtypes/sizes and confirm.
  for (size_t n : {1u, 7u, 64u, 256u, 1024u}) {
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = {static_cast<int64_t>(n)};
    tb.resize_contiguous(n);
    uintptr_t addr = reinterpret_cast<uintptr_t>(tb.data.data());
    EXPECT_TRUE((addr % 64) == 0);
  }
}

TEST(tensor_beat, copy_cpu_cached_is_deep)
{
  // A copy of a CpuCached TensorBeat must duplicate `data` so the
  // two are independent.
  TensorBeat src = make_f32_contig_({2, 3});
  for (size_t i = 0; i < 6; ++i) {
    set_f32_(src, i, static_cast<float>(i + 1));
  }
  TensorBeat dst = src;
  EXPECT_TRUE(dst.element_count() == 6u);
  EXPECT_TRUE(dst.data.data() != src.data.data());
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_TRUE(dst.as_f32()[i] == src.as_f32()[i]);
  }
  // Mutating the copy must not perturb the source.
  set_f32_(dst, 0, -42.0f);
  EXPECT_TRUE(src.as_f32()[0] == 1.0f);
  EXPECT_TRUE(dst.as_f32()[0] == -42.0f);
}

TEST(tensor_beat, copy_shared_storage_materialises_to_cpu_cached)
{
  // Synthesise a "Shared" TensorBeat without a real MTLBuffer: the
  // copy ctor only reads byte_size + contents, so a stack-backed
  // ExternalStorageHandle (with no deleter) is enough to exercise
  // the materialise-on-copy path.
  uint8_t backing[12];
  for (size_t i = 0; i < 12; ++i) {
    backing[i] = static_cast<uint8_t>(i * 3 + 1);
  }
  TensorBeat src;
  src.dtype = TensorBeat::DType::U8;
  src.shape = {3, 4};
  src.external = std::make_unique<ExternalStorageHandle>();
  src.external->mtl_buffer = nullptr;  // no real buffer; no deleter
  src.external->contents   = backing;
  src.external->byte_size  = 12;
  src.external->deleter    = nullptr;
  EXPECT_TRUE(src.storage_class() == TensorStorageClass::Shared);
  EXPECT_TRUE(src.byte_size() == 12u);
  // Copy must materialise into `data` (CpuCached) and leave the
  // copy independent of `backing`.
  TensorBeat dst = src;
  EXPECT_TRUE(dst.storage_class() == TensorStorageClass::CpuCached);
  EXPECT_TRUE(dst.external == nullptr);
  EXPECT_TRUE(dst.data.size() == 12u);
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_TRUE(dst.as_u8()[i] == backing[i]);
  }
  // Mutating the underlying Shared buffer must not perturb the
  // copy (it took an independent snapshot).
  backing[0] = 0xff;
  EXPECT_TRUE(dst.as_u8()[0] == 1u);
}

TEST(tensor_beat, move_preserves_shared_storage)
{
  // Moving must transfer the unique_ptr without materialising.
  uint8_t backing[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  TensorBeat src;
  src.dtype = TensorBeat::DType::U8;
  src.shape = {8};
  src.external = std::make_unique<ExternalStorageHandle>();
  src.external->contents  = backing;
  src.external->byte_size = 8;
  ExternalStorageHandle* raw = src.external.get();
  TensorBeat dst = std::move(src);
  EXPECT_TRUE(dst.external.get() == raw);
  EXPECT_TRUE(src.external == nullptr);
  EXPECT_TRUE(dst.storage_class() == TensorStorageClass::Shared);
}

TEST(tensor_beat, sideband_default_is_null_and_copies_deep)
{
  TensorBeat a = make_f32_contig_({2, 2});
  EXPECT_TRUE(a.sideband.is_null());

  FlexData obj = FlexData::make_object();
  obj.as_object().insert_or_assign("timestamp_us",
                                   FlexData::make_uint(1234567ull));
  a.sideband = std::move(obj);
  EXPECT_TRUE(a.sideband.is_object());
  EXPECT_TRUE(a.sideband.as_object().at("timestamp_us").get_uint()
              == 1234567ull);

  // Copy is deep: mutating the source's sideband after the copy must
  // not perturb the destination.
  TensorBeat b = a;
  EXPECT_TRUE(b.sideband.is_object());
  EXPECT_TRUE(b.sideband.as_object().at("timestamp_us").get_uint()
              == 1234567ull);
  a.sideband.as_object().insert_or_assign(
      "timestamp_us", FlexData::make_uint(9999ull));
  EXPECT_TRUE(b.sideband.as_object().at("timestamp_us").get_uint()
              == 1234567ull);

  // Clone via TensorBeatPayload must also carry the sideband.
  TensorBeatPayload pld(b);
  auto cloned = pld.clone();
  const auto* cloned_tbp =
      dynamic_cast<const TensorBeatPayload*>(cloned.get());
  EXPECT_TRUE(cloned_tbp != nullptr);
  EXPECT_TRUE(cloned_tbp->sideband.is_object());
  EXPECT_TRUE(cloned_tbp->sideband.as_object()
                .at("timestamp_us").get_uint() == 1234567ull);
}
