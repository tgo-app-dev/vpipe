#ifndef VPIPE_GENERATIVE_MODELS_GGUF_FILE_H
#define VPIPE_GENERATIVE_MODELS_GGUF_FILE_H

// GgufFile -- a small, MLX-free reader for GGUF v2/v3 model files
// (llama.cpp's container format). It mmaps the file, parses the
// metadata key/value table and the tensor index, and hands out:
//
//   * typed metadata accessors (scalars, strings, and lazily-parsed
//     arrays -- the big token/merge arrays are re-read from the mmap on
//     demand so a weights-only load never materialises them), and
//   * raw tensor views (ggml type + dims + a pointer into the mmap),
//     plus a row dequantiser for the float/Q6_K types the converter
//     needs (Q4_0 is repacked block-wise by the converter itself, never
//     round-tripped through f32).
//
// Deliberately dependency-free (no MLX, no metal): both the MLX loader
// and the metal weight reader build their backend tensors from the same
// GgufFile so a GGUF checkpoint converts identically on both paths.
//
// ggml stores tensor dims in `ne` order (ne[0] is the fastest-moving /
// contiguous axis). For a Linear weight that means dims = {in, out};
// each of the `out` rows is `in` contiguous (possibly quantised)
// values -- exactly the per-row layout MLX/PyTorch use as [out, in].

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe::genai {

class GgufFile {
public:
  // The subset of ggml tensor types this checkpoint family uses. Values
  // match ggml_type; unknown types are still parsed (size math via the
  // block table) but cannot be dequantised.
  enum GgmlType : std::uint32_t {
    kF32  = 0,
    kF16  = 1,
    kQ4_0 = 2,
    kQ8_0 = 8,
    kQ4_K = 12,
    kQ5_K = 13,
    kQ6_K = 14,
    kBF16 = 30,
  };

  struct Tensor {
    std::string          name;
    std::vector<std::int64_t> dims;   // ggml ne order (ne[0] contiguous)
    std::uint32_t        type = kF32; // ggml_type
    std::uint64_t        offset = 0;  // bytes from the data section start
    const std::uint8_t*  data = nullptr;  // mmap + data_start + offset
    std::uint64_t        nbytes = 0;
    std::int64_t numel() const;
  };

  static std::optional<GgufFile> open(const std::string& path);

  GgufFile(GgufFile&&) noexcept;
  GgufFile& operator=(GgufFile&&) noexcept;
  GgufFile(const GgufFile&)            = delete;
  GgufFile& operator=(const GgufFile&) = delete;
  ~GgufFile();

  // ---- metadata scalars -------------------------------------------
  bool                        has(const std::string& key) const;
  std::optional<std::uint64_t> get_uint(const std::string& key) const;
  std::optional<std::int64_t>  get_int(const std::string& key) const;
  std::optional<double>        get_float(const std::string& key) const;
  std::optional<bool>          get_bool(const std::string& key) const;
  std::optional<std::string>   get_string(const std::string& key) const;

  // ---- metadata arrays (lazily decoded from the mmap) -------------
  // Integer-typed arrays (UINT8..INT64) coerced to int64.
  std::vector<std::int64_t>    get_int_array(const std::string& key) const;
  std::vector<float>           get_float_array(const std::string& key) const;
  std::vector<std::string>     get_string_array(const std::string& key) const;
  // Number of elements in an array-valued key (0 if absent / not array).
  std::size_t                  array_len(const std::string& key) const;

  // ---- tensors -----------------------------------------------------
  const Tensor* tensor(const std::string& name) const;
  const std::vector<Tensor>& tensors() const { return _tensors; }

  // Dequantise one row (the `in` contiguous values for output index
  // `row`) of a 2-D tensor to f32. Supports F32/F16/BF16/Q8_0/Q4_0/
  // Q4_K/Q5_K/Q6_K. `out` must hold dims[0] floats. False on
  // unsupported type or out-of-range row.
  bool dequant_row_f32(const Tensor& t, std::int64_t row,
                       float* out) const;

  // Whole-tensor f32 dequant for small 1-D tensors (norms etc.).
  // `out` must hold numel() floats.
  bool dequant_all_f32(const Tensor& t, float* out) const;

private:
  GgufFile() = default;

  // One metadata value. Scalars keep their bits inline; arrays keep the
  // element type + count + a file offset so the (possibly huge) payload
  // is decoded on demand straight from the mmap.
  struct Value {
    std::uint32_t type = 0;        // gguf_metadata_value_type
    std::uint64_t scalar = 0;      // scalar bit pattern (int/float/bool)
    std::string   str;             // when type == STRING
    std::uint32_t arr_elem = 0;    // when type == ARRAY: element type
    std::uint64_t arr_count = 0;
    std::uint64_t arr_off = 0;     // byte offset (from file base) of elems
  };

  const Value* find_(const std::string& key) const;

  int                 _fd = -1;
  const std::uint8_t* _base = nullptr;   // mmap base
  std::size_t         _map_size = 0;
  std::uint64_t       _data_start = 0;   // tensor data section offset
  std::unordered_map<std::string, Value> _meta;
  std::vector<Tensor> _tensors;
  std::unordered_map<std::string, std::size_t> _tensor_index;
};

}  // namespace vpipe::genai

#endif
