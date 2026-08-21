#include "generative-models/llama3/metal-llama-weights.h"

#include "generative-models/shared/gguf-convert.h"
#include "generative-models/shared/gguf-file.h"
#include "generative-models/model-loader.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>

#include <vector>
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
  // The streaming descriptor. Its absence is not an error -- pread_into()
  // simply reports it cannot serve, and every caller has a mapped
  // fallback.
  sh.fd_stream = ::open(safetensors_path.c_str(), O_RDONLY);
#ifdef F_NOCACHE
  if (sh.fd_stream >= 0) { (void)::fcntl(sh.fd_stream, F_NOCACHE, 1); }
#endif
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

  // A checkpoint named by FILE rather than by directory: the Comfy-Org
  // repacks (Comfy-Org/MiniMax-H3, Comfy-Org/Wan-Animate-2) ship one
  // freely-named .safetensors per component under diffusion_models/ or
  // vae/, so there is no directory whose contents this could glob and
  // no config.json beside it. Its own header is self-describing, which
  // is all the single-file path needs.
  //
  // A `.gguf` named by file is NOT that case -- it falls through to the
  // GGUF branch below, whose find_gguf_in_dir() takes a file path as
  // well as a directory. Sending it here instead mmapped a GGUF as
  // safetensors and failed, which read as "the model does not load"
  // even though the tokenizer (which resolves the path itself) had
  // already come up. Naming the file is how a caller picks ONE quant out
  // of a repo directory that holds several, so it has to work.
  if (fs::is_regular_file(dir, ec) && !ec && dir.extension() != ".gguf") {
    return open(model_dir);
  }

  // mlx-optiq stores the (unquantized BF16) vision tower in a SIDECAR
  // (`optiq_vision.safetensors`, HF `vision_tower.*` names) that is NOT listed
  // in model.safetensors.index.json -- the index only maps the quantized LM
  // shards. Map the sidecar too so the metal vision encoder finds its weights;
  // it's harmless for the LM load (which never references vision_tower.*), and
  // the mmap is lazy (no pages faulted until a tensor is load()ed). No-op when
  // the sidecar is absent (every non-optiq checkpoint).
  // Two sidecar layouts are in the wild: the older packs (Qwen3.5-4B/9B
  // OptiQ) keep it at the model-dir ROOT, the newer ones (Qwen3.6-27B /
  // 35B-A3B, gemma-4 12B/26B/31B OptiQ) moved it into an `optiq/` subdir
  // alongside mtp.safetensors. Probe root first, then the subdir.
  auto map_vision_sidecar_ = [&](MetalLlamaWeights& w) {
    const fs::path cands[] = {dir / "optiq_vision.safetensors",
                              dir / "optiq" / "optiq_vision.safetensors"};
    for (const fs::path& side : cands) {
      if (fs::exists(side, ec)) { w.map_shard_(side.string()); break; }
    }
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
    bool mapped_any = false;
    for (const auto& name : shard_names) {
      const fs::path shard = dir / name;
      // A shard listed in the index but ABSENT on disk is an optional
      // sidecar the download skipped (e.g. an OptiQ text-only checkpoint
      // whose vision/audio adaptor shard under optiq/ was not fetched). Skip
      // it: its tensors stay unregistered (has() == false), so a text-only
      // load proceeds and the multimodal probe reports "no adaptor". A shard
      // that EXISTS but fails to map is still fatal (corrupt/unreadable).
      if (!fs::exists(shard, ec)) { continue; }
      if (!w.map_shard_(shard.string())) {
        return std::nullopt;
      }
      mapped_any = true;
    }
    if (!mapped_any) {
      return std::nullopt;   // every listed shard absent -> nothing to load
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
      if (sh.fd_stream >= 0) { ::close(sh.fd_stream); }
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
    if (sh.fd_stream >= 0) { ::close(sh.fd_stream); }
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

MetalLlamaWeights::Alignment
MetalLlamaWeights::alignment() const
{
  Alignment a;
  a.shards = (int)_shards.size();
  for (const Shard& sh : _shards) {
    if ((sh.data_start & 0xF) != 0) { ++a.bad_shards; }
  }
  for (const auto& kv : _tensors) {
    const TensorInfo& ti = kv.second;
    if (ti.shard < 0 || (std::size_t)ti.shard >= _shards.size()) { continue; }
    ++a.tensors;
    const std::uint64_t off =
        _shards[(std::size_t)ti.shard].data_start + ti.offset;
    if ((off & 0xF) != 0) { ++a.misaligned; }
  }
  return a;
}

bool
MetalLlamaWeights::is_gguf() const noexcept
{
  return _gguf != nullptr;
}

bool
MetalLlamaWeights::pread_into(const std::string& name, void* dst,
                              std::size_t cap, bool uncached) const
{
  const TensorInfo* ti = info(name);
  if (ti == nullptr || dst == nullptr || cap < ti->nbytes) {
    return false;
  }
  // GGUF tensors are CONVERTED on the way out, so the file holds no bytes
  // this could place. shard == -2 marks them; anything else out of range
  // is a malformed entry.
  if (ti->shard < 0 ||
      static_cast<std::size_t>(ti->shard) >= _shards.size()) {
    return false;
  }
  const Shard& sh = _shards[static_cast<std::size_t>(ti->shard)];
  const int fd = uncached && sh.fd_stream >= 0 ? sh.fd_stream : sh.fd;
  if (fd < 0) { return false; }
  const std::uint64_t base = sh.data_start + ti->offset;
  auto* out = static_cast<std::uint8_t*>(dst);

  // F_NOCACHE ONLY BYPASSES THE BUFFER CACHE ON A PAGE-ALIGNED OFFSET.
  //
  // Unbuffered I/O has alignment requirements, and a read that does not
  // meet them does not fail -- it silently falls back to the buffered
  // path and fills the cache with exactly the bytes the caller asked not
  // to keep. MEASURED, 4 GB read from one file with F_NOCACHE set: a
  // page-aligned offset grows file-backed pages by 0 MB, the SAME read
  // at offset+8 grows them by 1813 MB.
  //
  // That matters because a safetensors data section is 8 + header_len
  // into the file, so its alignment is whatever the header length
  // happened to be -- and every tensor inherits it. On a checkpoint
  // whose section is off-boundary (Comfy-Org repacks, and anything an
  // older vpipe quantizer wrote) EVERY streamed read was cached, which
  // is where the tens of gigabytes of file cache came from that the
  // wired pool then had to compete with.
  //
  // So read on page boundaries and copy the wanted bytes out. The extra
  // memcpy runs at memory bandwidth; the cache it avoids was costing the
  // model its resident set.
  if (uncached && sh.fd_stream >= 0) {
    const std::size_t page =
        static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    if (page != 0 && (base % page) != 0) {
      // One page-aligned staging buffer per thread: the streamer calls
      // this from a prefetch thread while the forward calls it from its
      // own, and a per-call allocation would be a megabyte-scale malloc
      // in the middle of every block read.
      constexpr std::size_t kChunk = 8ull << 20;      // page multiple
      static thread_local std::vector<std::uint8_t> stage(kChunk + 16384);
      std::size_t done = 0;
      std::uint64_t cur = base;
      while (done < ti->nbytes) {
        const std::uint64_t off = cur & ~(std::uint64_t)(page - 1);
        const std::size_t slack = (std::size_t)(cur - off);
        std::size_t want = kChunk - slack;
        if (want > ti->nbytes - done) { want = ti->nbytes - done; }
        const ssize_t n = ::pread(fd, stage.data(), slack + want,
                                  static_cast<off_t>(off));
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0 || (std::size_t)n <= slack) { return false; }
        const std::size_t got = (std::size_t)n - slack;
        const std::size_t take = got < want ? got : want;
        std::memcpy(out + done, stage.data() + slack, take);
        done += take;
        cur += take;
      }
      return true;
    }
  }

  std::size_t done = 0;
  while (done < ti->nbytes) {
    const ssize_t n = ::pread(fd, out + done, ti->nbytes - done,
                              static_cast<off_t>(base + done));
    if (n < 0 && errno == EINTR) { continue; }
    // A short read that is not EINTR is a truncated or racing file. The
    // destination is left partly written, which is why this returns
    // false rather than trying to paper over it -- the caller must fall
    // back to a path that rewrites the whole buffer, not top this one up.
    if (n <= 0) { return false; }
    done += static_cast<std::size_t>(n);
  }
  return true;
}

metal_compute::SharedBuffer
MetalLlamaWeights::load(const std::string& name,
                        metal_compute::MetalCompute* mc,
                        LoadCost* cost) const
{
  const TensorInfo* ti = info(name);
  if (ti == nullptr || mc == nullptr) {
    return {};
  }
  const auto t_a0 = cost != nullptr ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
  metal_compute::SharedBuffer buf = mc->make_shared_buffer(ti->nbytes);
  if (cost != nullptr) {
    cost->alloc_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_a0).count();
  }
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
  const auto t_f0 = cost != nullptr ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
  // An UNCACHED pread wherever the tensor is big enough to be worth the
  // syscall, in preference to faulting it through the mapping.
  //
  // It is faster -- the numbers below are why the hint underneath it
  // exists -- but the reason it is the default is what it does NOT do:
  // a mapped read leaves the bytes in the unified buffer cache, so a
  // model that re-reads its stack every forward grows file-backed pages
  // without bound. On a big box that is invisible. On a 24 GB one it is
  // the whole failure: the cache competes with the model's own
  // activations, the compressor takes what it can, and the run swaps
  // without making progress. Nothing here needs the pages to persist --
  // the bytes are being copied into a buffer this process owns.
  //
  // The threshold is the same one the hint uses, and for the same
  // reason: a checkpoint holds thousands of small tensors, a syscall
  // each would cost more than the fault, and they are not what fills a
  // cache.
  if (ti->nbytes >= (1u << 20) &&
      pread_into(name, buf.contents(), buf.byte_size())) {
    if (cost != nullptr) {
      cost->fetch_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_f0).count();
    }
    return buf;
  }
  // Tell the kernel what this memcpy is about to touch. Faulting a
  // mapping one cluster at a time is what makes the mapped read slow --
  // MEASURED on an M5 at 0.69-1.09 GB/s against 6.8 GB/s for a pread of
  // the same bytes -- and a single WILLNEED over the range recovers most
  // of the gap (to 3.7-5.3 GB/s) without changing who owns the memory.
  // Advisory: if the kernel declines, this is exactly the read it was.
  //
  // pread_into() is the rest of that gap, but it needs a destination the
  // caller already owns; this helps every streaming model that still
  // allocates per block.
  //
  // Only for reads big enough to be worth a syscall. A checkpoint has
  // thousands of small tensors and the hint buys nothing on a range the
  // kernel's own clustering already covers in one fault.
  if (ti->nbytes >= (1u << 20)) {
    const auto page = static_cast<std::uintptr_t>(::getpagesize());
    const auto lo = reinterpret_cast<std::uintptr_t>(src) & ~(page - 1);
    const std::size_t span =
        ti->nbytes + (reinterpret_cast<std::uintptr_t>(src) - lo);
    (void)::madvise(reinterpret_cast<void*>(lo), span, MADV_WILLNEED);
  }
  std::memcpy(buf.contents(), src, ti->nbytes);
  if (cost != nullptr) {
    cost->fetch_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_f0).count();
  }
  return buf;
}

bool
MetalLlamaWeights::read_into(const std::string& name, void* dst,
                             std::size_t cap) const
{
  const TensorInfo* ti = info(name);
  if (ti == nullptr || dst == nullptr || cap < ti->nbytes) {
    return false;
  }
  // GGUF tensors are CONVERTED, not copied, and the converter writes
  // through its own destination -- which is `dst` here, so no staging
  // buffer is needed after all.
  if (ti->shard == -2) {
    if (!_gguf) { return false; }
    auto it = _gguf->specs.find(name);
    return it != _gguf->specs.end()
        && _gguf->conv->convert(it->second, static_cast<std::uint8_t*>(dst));
  }
  if (ti->shard < 0 ||
      static_cast<std::size_t>(ti->shard) >= _shards.size()) {
    return false;
  }
  const Shard& sh = _shards[static_cast<std::size_t>(ti->shard)];
  // Uncached first, for the reason load() gives: these bytes are being
  // copied into the caller's memory, so leaving a cached copy behind
  // buys nothing and costs a 24 GB box its working set.
  if (ti->nbytes >= (1u << 20) && pread_into(name, dst, cap)) {
    return true;
  }
  if (sh.base == nullptr) { return false; }
  const auto* src =
      static_cast<const std::uint8_t*>(sh.base) + sh.data_start + ti->offset;
  std::memcpy(dst, src, ti->nbytes);
  return true;
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
    // SAY SO, once per shard.
    //
    // This fallback is correct and it used to be silent, which made it
    // the most expensive quiet decision in the loader: a checkpoint
    // whose data section is misaligned copies EVERY tensor, so a model
    // asking for `Mapped` gets gigabytes of anonymous memory and no
    // indication that it did. MEASURED on a 22B DiT: 4349 of 4349
    // tensors, 39.1 GB, all copied because the file's data begins at
    // 677624 -- 8 mod 16. The symptom reached the operator as a box
    // thrashing with 36 GB in the compressor, which is a long way from
    // the cause.
    //
    // Once per shard, and it has to name WHICH alignment failed.
    //
    // There are two, and only one of them is about the data section. A
    // shard whose section start is misaligned copies everything; a shard
    // whose section start is FINE can still copy most of itself, because
    // tensors are packed contiguously and one whose size is not a
    // multiple of 16 shifts every tensor after it. Reporting the section
    // start alone then produces a message that contradicts itself --
    // "not mappable ... starts at 69888 (0 mod 16)" -- and sends the
    // reader looking at the wrong number. So say which it is, and for
    // the per-tensor case count them.
    if ((goff & 0xF) != 0 && mc->session() != nullptr) {
      if (_shard_unmappable_said.size() < _shards.size()) {
        _shard_unmappable_said.resize(_shards.size(), false);
      }
      if (!_shard_unmappable_said[si]) {
        _shard_unmappable_said[si] = true;
        if ((sh.data_start & 0xF) != 0) {
          mc->session()->warn(fmt(
              "weights: shard {} is NOT zero-copy mappable -- its data "
              "section starts at {} ({} mod 16, and a Metal buffer offset "
              "must be 16-byte aligned), so EVERY tensor in it is COPIED "
              "into anonymous memory instead. Residency::Mapped is a "
              "no-op for this shard; size the box for the copies. If "
              "this model was quantized by an earlier version of vpipe, "
              "re-running the quantization lays it down aligned",
              si, sh.data_start, sh.data_start & 0xF));
        } else {
          std::size_t bad = 0, all = 0;
          for (const auto& kv : _tensors) {
            if (kv.second.shard != static_cast<int>(si)) { continue; }
            ++all;
            if (((sh.data_start + kv.second.offset) & 0xF) != 0) { ++bad; }
          }
          mc->session()->warn(fmt(
              "weights: shard {} is only PARTLY zero-copy mappable -- its "
              "data section is 16-byte aligned but {} of its {} tensors "
              "are not, so those are COPIED into anonymous memory. A "
              "tensor whose SIZE is not a multiple of 16 shifts every "
              "tensor after it; writing those last is what avoids it. If "
              "this model was quantized by an earlier version of vpipe, "
              "re-running the quantization does exactly that",
              si, bad, all));
        }
      }
    }
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
