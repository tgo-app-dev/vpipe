#include "generative-models/llama3/metal-llama-weights.h"

#include "generative-models/shared/gguf-convert.h"
#include "generative-models/shared/gguf-file.h"
#include "generative-models/model-loader.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace vpipe::genai {

// GGUF backing state: the mmapped GgufFile, the on-demand converter, and
// the HF-name -> conversion-spec map that load() uses to materialise each
// tensor straight into a SharedBuffer (no staging arena). Held by
// unique_ptr so the GgufFile's address (captured by the converter) stays
// stable across MetalLlamaWeights moves.
struct GgufBacking {
  std::optional<GgufFile>                           gguf;
  std::unique_ptr<GgufConverterBase>                conv;
  std::unordered_map<std::string, ConvertedTensorSpec> specs;
};

bool
MetalLlamaWeights::map_shard_(const std::string& safetensors_path)
{
  const int fd = ::open(safetensors_path.c_str(), O_RDONLY);
  if (fd < 0) {
    return false;
  }
  struct stat st {};
  if (::fstat(fd, &st) != 0 || st.st_size < 8) {
    ::close(fd);
    return false;
  }
  const std::size_t file_size = static_cast<std::size_t>(st.st_size);

  // safetensors: u64 LE header length, then that many JSON bytes,
  // then the tensor data blob (data_offsets are relative to it).
  std::uint64_t header_len = 0;
  if (::pread(fd, &header_len, 8, 0) != 8) {
    ::close(fd);
    return false;
  }
  if (header_len == 0 || 8 + header_len > file_size) {
    ::close(fd);
    return false;
  }
  std::string header(static_cast<std::size_t>(header_len), '\0');
  if (::pread(fd, header.data(), header_len, 8) !=
      static_cast<ssize_t>(header_len)) {
    ::close(fd);
    return false;
  }

  FlexData j;
  try {
    j = FlexData::from_json(header);
  } catch (...) {
    ::close(fd);
    return false;
  }
  if (!j.is_object()) {
    ::close(fd);
    return false;
  }

  void* base = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED) {
    ::close(fd);
    return false;
  }

  const int shard_idx = static_cast<int>(_shards.size());
  Shard sh;
  sh.fd = fd;
  sh.base = base;
  sh.map_size = file_size;
  sh.data_start = 8 + header_len;
  _shards.push_back(sh);

  for (auto entry : j.as_object()) {
    const std::string_view key = entry.first;
    if (key == "__metadata__") {
      continue;
    }
    const FlexData& e = entry.second;
    if (!e.is_object()) {
      continue;
    }
    auto eo = e.as_object();
    if (!eo.contains("dtype") || !eo.contains("shape") ||
        !eo.contains("data_offsets")) {
      continue;
    }
    TensorInfo ti;
    // NOTE: at() returns a FlexData by value; as_array()/as_string()
    // return views INTO it, so the FlexData must outlive the view --
    // bind to named locals, never `at(...).as_array()` inline.
    // Use the cross-casting as_* accessors (Int/Uint/Real are distinct
    // variants; the get_* forms throw on a kind mismatch).
    const FlexData dtype_fd = eo.at("dtype");
    const FlexData shape_fd = eo.at("shape");
    const FlexData off_fd = eo.at("data_offsets");
    ti.dtype = std::string(dtype_fd.as_string());
    for (auto d : shape_fd.as_array()) {
      ti.shape.push_back(d.as_int());
    }
    auto off = off_fd.as_array();
    if (off.size() < 2) {
      continue;
    }
    ti.offset = off[0].as_uint();
    ti.nbytes = off[1].as_uint() - ti.offset;
    ti.shard = shard_idx;
    _tensors.emplace(std::string(key), std::move(ti));
  }
  return true;
}

std::optional<MetalLlamaWeights>
MetalLlamaWeights::open(const std::string& safetensors_path)
{
  MetalLlamaWeights w;
  if (!w.map_shard_(safetensors_path)) {
    return std::nullopt;
  }
  return w;
}

std::optional<MetalLlamaWeights>
MetalLlamaWeights::open_model(const std::string& model_dir)
{
  namespace fs = std::filesystem;
  const fs::path dir(model_dir);
  std::error_code ec;

  // mlx-optiq stores the (unquantized BF16) vision tower in a SIDECAR
  // (`optiq_vision.safetensors`, HF `vision_tower.*` names) that is NOT listed
  // in model.safetensors.index.json -- the index only maps the quantized LM
  // shards. Map the sidecar too so the metal vision encoder finds its weights;
  // it's harmless for the LM load (which never references vision_tower.*), and
  // the mmap is lazy (no pages faulted until a tensor is load()ed). No-op when
  // the sidecar is absent (every non-optiq checkpoint).
  auto map_vision_sidecar_ = [&](MetalLlamaWeights& w) {
    const fs::path side = dir / "optiq_vision.safetensors";
    if (fs::exists(side, ec)) { w.map_shard_(side.string()); }
  };

  // GGUF layout: a `.gguf` checkpoint. Parse it, derive the config, and
  // register every output tensor (HF name + dtype/shape) against the
  // converter; load() materialises each one on demand.
  const std::string gguf_path = find_gguf_in_dir(model_dir);
  if (!gguf_path.empty()) {
    auto gfile = GgufFile::open(gguf_path);
    if (!gfile) { return std::nullopt; }
    ModelConfig cfg;
    if (!gguf_to_model_config(*gfile, &cfg)) { return std::nullopt; }
    auto backing = std::make_unique<GgufBacking>();
    backing->gguf = std::move(gfile);
    // Pick the arch-specific converter: qwen35 keeps weights native
    // k-quant (no requant); gemma4 repacks to the affine layout.
    auto arch = backing->gguf->get_string("general.architecture");
    if (arch && *arch == "qwen35") {
      backing->conv =
          std::make_unique<GgufQwen35Converter>(&*backing->gguf, cfg);
    } else {
      backing->conv =
          std::make_unique<GgufGemma4Converter>(&*backing->gguf, cfg);
    }
    MetalLlamaWeights w;
    for (const auto& spec : backing->conv->specs()) {
      TensorInfo ti;
      ti.dtype  = spec.dtype;
      ti.shape  = spec.shape;
      ti.offset = 0;
      ti.nbytes = spec.nbytes;
      ti.shard  = -2;   // GGUF-backed sentinel
      w._tensors.emplace(spec.hf_name, ti);
      backing->specs.emplace(spec.hf_name, spec);
    }
    w._gguf = std::move(backing);
    return w;
  }

  // Sharded layout: model.safetensors.index.json maps tensor name ->
  // shard filename. We only need the distinct shard filenames; each
  // shard's own header carries its tensors and their (shard-relative)
  // offsets. Mirror the MLX loader: collect the unique set and mmap
  // each once, in sorted order for determinism.
  // Diffusers checkpoints (Krea-2 transformer, Qwen-Image VAE) name their
  // shards/index `diffusion_pytorch_model.*` instead of `model.*`; accept
  // either. The index's weight_map still carries the real shard filenames,
  // so only the index (and single-file) NAME needs the fallback.
  fs::path index = dir / "model.safetensors.index.json";
  if (!fs::exists(index, ec)) {
    const fs::path dindex =
        dir / "diffusion_pytorch_model.safetensors.index.json";
    if (fs::exists(dindex, ec)) { index = dindex; }
  }
  if (fs::exists(index, ec)) {
    std::ifstream in(index);
    if (!in) {
      return std::nullopt;
    }
    FlexData idx;
    try {
      idx = FlexData::from_json(in);
    } catch (...) {
      return std::nullopt;
    }
    if (!idx.is_object()) {
      return std::nullopt;
    }
    auto root = idx.as_object();
    if (!root.contains("weight_map")) {
      return std::nullopt;
    }
    const FlexData wm_fd = root.at("weight_map");
    if (!wm_fd.is_object()) {
      return std::nullopt;
    }
    std::set<std::string> shard_names;
    for (auto entry : wm_fd.as_object()) {
      const FlexData& v = entry.second;
      if (!v.is_string()) {
        return std::nullopt;
      }
      shard_names.emplace(v.as_string());
    }
    if (shard_names.empty()) {
      return std::nullopt;
    }
    MetalLlamaWeights w;
    for (const auto& name : shard_names) {
      if (!w.map_shard_((dir / name).string())) {
        return std::nullopt;
      }
    }
    map_vision_sidecar_(w);
    return w;
  }

  // Index-less sharded layout: some checkpoints (e.g. MOSS-TTS-8B) ship
  // model-00001-of-00002.safetensors + model-00002-of-... with NO
  // index.json. Glob the shards and map each in sorted order; each shard's
  // own header is self-describing.
  if (!fs::exists(dir / "model.safetensors", ec)
      && !fs::exists(dir / "diffusion_pytorch_model.safetensors", ec)) {
    std::set<std::string> shard_names;
    std::error_code lec;
    for (const auto& de : fs::directory_iterator(dir, lec)) {
      const fs::path& p = de.path();
      const std::string fn = p.filename().string();
      if (p.extension() == ".safetensors"
          && (fn.rfind("model-", 0) == 0
              || fn.rfind("diffusion_pytorch_model-", 0) == 0)) {
        shard_names.emplace(fn);
      }
    }
    if (!shard_names.empty()) {
      MetalLlamaWeights w;
      for (const auto& name : shard_names) {
        if (!w.map_shard_((dir / name).string())) {
          return std::nullopt;
        }
      }
      map_vision_sidecar_(w);
      return w;
    }
  }

  // Single-file layout (model.safetensors or the diffusers name).
  fs::path sf = dir / "model.safetensors";
  if (!fs::exists(sf, ec)) {
    const fs::path dsf = dir / "diffusion_pytorch_model.safetensors";
    if (fs::exists(dsf, ec)) { sf = dsf; }
  }
  auto single = open(sf.string());
  if (single) { map_vision_sidecar_(*single); }
  return single;
}

MetalLlamaWeights::MetalLlamaWeights(MetalLlamaWeights&& o) noexcept
{
  *this = std::move(o);
}

MetalLlamaWeights&
MetalLlamaWeights::operator=(MetalLlamaWeights&& o) noexcept
{
  if (this != &o) {
    // Release our no-copy wrappers BEFORE unmapping the pages they reference.
    _shard_maps.clear();
    for (auto& sh : _shards) {
      if (sh.base != nullptr) { ::munmap(sh.base, sh.map_size); }
      if (sh.fd >= 0) { ::close(sh.fd); }
    }
    _shards = std::move(o._shards);
    _shard_maps = std::move(o._shard_maps);
    _tensors = std::move(o._tensors);
    _gguf = std::move(o._gguf);
    o._shards.clear();      // moved-from vectors are empty; make it explicit
    o._shard_maps.clear();
  }
  return *this;
}

MetalLlamaWeights::~MetalLlamaWeights()
{
  // Release the no-copy MTL buffers before unmapping their backing pages.
  _shard_maps.clear();
  for (auto& sh : _shards) {
    if (sh.base != nullptr) { ::munmap(sh.base, sh.map_size); }
    if (sh.fd >= 0) { ::close(sh.fd); }
  }
}

bool
MetalLlamaWeights::has(const std::string& name) const
{
  return _tensors.find(name) != _tensors.end();
}

const MetalLlamaWeights::TensorInfo*
MetalLlamaWeights::info(const std::string& name) const
{
  auto it = _tensors.find(name);
  return it == _tensors.end() ? nullptr : &it->second;
}

std::vector<std::string>
MetalLlamaWeights::tensor_names() const
{
  std::vector<std::string> names;
  names.reserve(_tensors.size());
  for (const auto& kv : _tensors) { names.push_back(kv.first); }
  return names;
}

metal_compute::SharedBuffer
MetalLlamaWeights::load(const std::string& name,
                        metal_compute::MetalCompute* mc) const
{
  const TensorInfo* ti = info(name);
  if (ti == nullptr || mc == nullptr) {
    return {};
  }
  metal_compute::SharedBuffer buf = mc->make_shared_buffer(ti->nbytes);
  if (buf.empty()) {
    return buf;
  }
  // GGUF-backed: convert this tensor straight into the SharedBuffer.
  if (ti->shard == -2 && _gguf) {
    auto it = _gguf->specs.find(name);
    if (it == _gguf->specs.end() ||
        !_gguf->conv->convert(it->second,
                              static_cast<std::uint8_t*>(buf.contents()))) {
      return {};
    }
    return buf;
  }
  const Shard& sh = _shards[static_cast<std::size_t>(ti->shard)];
  const auto* src =
      static_cast<const std::uint8_t*>(sh.base) + sh.data_start + ti->offset;
  std::memcpy(buf.contents(), src, ti->nbytes);
  return buf;
}

metal_compute::SharedBuffer
MetalLlamaWeights::load_mapped(const std::string& name,
                              metal_compute::MetalCompute* mc) const
{
  const TensorInfo* ti = info(name);
  if (ti == nullptr || mc == nullptr) {
    return {};
  }
  // GGUF tensors are converted on load (not a straight byte view), and only
  // real shards can be wrapped. Anything else copies.
  if (ti->shard < 0) {
    return load(name, mc);
  }
  const std::size_t si = static_cast<std::size_t>(ti->shard);
  if (si >= _shards.size()) {
    return load(name, mc);
  }
  const Shard& sh = _shards[si];

  // The GPU byte offset of this tensor within the whole-shard buffer, and a
  // conservative bind-alignment guard (16 bytes comfortably covers Metal's
  // device-buffer offset requirement and every weight element size). A
  // misaligned or out-of-range tensor falls back to a copy so correctness is
  // never at the mercy of the on-disk packing.
  const std::size_t goff = sh.data_start + ti->offset;
  const std::size_t end  = goff + ti->nbytes;
  if ((goff & 0xF) != 0 || sh.base == nullptr || end > sh.map_size) {
    return load(name, mc);
  }

  // Lazily wrap the whole shard once (newBufferWithBytesNoCopy over the mmap).
  if (_shard_maps.size() < _shards.size()) {
    _shard_maps.resize(_shards.size());
  }
  if (_shard_maps[si].empty()) {
    _shard_maps[si] = mc->make_no_copy_buffer(sh.base, sh.map_size);
    if (_shard_maps[si].empty()) {
      return load(name, mc);        // wrap failed -> copy
    }
  }
  return _shard_maps[si].subview(goff, ti->nbytes);
}

}  // namespace vpipe::genai
