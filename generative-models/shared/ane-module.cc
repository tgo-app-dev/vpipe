#include "generative-models/shared/ane-module.h"

#include <CoreVideo/CoreVideo.h>

#include "apple-silicon/coreml/ane-emitter.h"
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace vpipe {
namespace genai {

namespace {

namespace fs = std::filesystem;

// The weight blob's on-disk shape, confirmed by stamping sparse
// markers at known coordinates and locating them: a 64-byte file
// header, then per weight a 64-byte entry header followed by the raw
// data. Every marker landed at exactly `data_offset + 2*(k*N + n)`,
// the file size was exactly the sum of headers and data, and the
// background value appeared exactly (K*N - markers) times -- so the
// data really is plain row-major with nothing done to it.
constexpr std::uint32_t kEntryMagic  = 0xdeadbeefu;
constexpr std::size_t   kFileHeader  = 64;
constexpr std::size_t   kEntryHeader = 64;

std::uint32_t
rd_u32_(const unsigned char* p)
{
  std::uint32_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

std::uint64_t
rd_u64_(const unsigned char* p)
{
  std::uint64_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

std::string
weight_bin_of_(const std::string& mlmodelc)
{
  return (fs::path(mlmodelc) / "weights" / "weight.bin").string();
}

}  // namespace

bool
parse_weight_blob(const std::string&          weight_bin_path,
                  std::vector<AneWeightSlot>* out,
                  std::string*                err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (out == nullptr) { return fail("null out"); }
  out->clear();

  std::error_code ec;
  const auto size = (std::size_t)fs::file_size(weight_bin_path, ec);
  if (ec) { return fail("cannot stat " + weight_bin_path); }
  if (size < kFileHeader) { return fail("weight blob too small"); }

  std::ifstream f(weight_bin_path, std::ios::binary);
  if (!f) { return fail("cannot open " + weight_bin_path); }
  std::vector<unsigned char> buf(size);
  f.read((char*)buf.data(), (std::streamsize)size);
  if (!f) { return fail("short read on " + weight_bin_path); }

  // File header: entry count at +0. Anything absurd means the format
  // moved, and stamping bytes blind into a moved format is exactly the
  // failure this check exists to prevent.
  const std::uint32_t count = rd_u32_(buf.data());
  if (count == 0 || count > 4096u) {
    return fail("implausible weight count " + std::to_string(count));
  }

  std::size_t off = kFileHeader;
  for (std::uint32_t i = 0; i < count; ++i) {
    if (off + kEntryHeader > size) {
      return fail("entry " + std::to_string(i) + " header past EOF");
    }
    const unsigned char* h = buf.data() + off;
    if (rd_u32_(h) != kEntryMagic) {
      return fail("entry " + std::to_string(i) + " bad magic");
    }
    const std::size_t nbytes = (std::size_t)rd_u64_(h + 8);
    const std::size_t doff   = (std::size_t)rd_u64_(h + 16);
    if (doff != off + kEntryHeader) {
      return fail("entry " + std::to_string(i) + " unexpected data offset");
    }
    if (doff + nbytes > size) {
      return fail("entry " + std::to_string(i) + " data past EOF");
    }
    out->push_back(AneWeightSlot{doff, nbytes});
    off = doff + nbytes;
  }
  if (off != size) {
    return fail("entries do not tile the blob (" + std::to_string(off) +
                " != " + std::to_string(size) + ")");
  }
  return true;
}


AneModule::~AneModule() = default;

bool
AneModule::valid() const noexcept
{
  return _model != nullptr;
}

std::unique_ptr<AneModule>
AneModule::build(const SessionContextIntf* session,
                 const std::string& template_dir,
                 const std::string& work_dir,
                 std::span<const AneWeight> weights)
{
  if (session == nullptr) { return nullptr; }
  auto* svc = session->services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager()
                               : nullptr;
  if (mgr == nullptr) {
    session->warn(fmt("ane-module: no CoreML manager (non-Apple build?)"));
    return nullptr;
  }

  std::error_code ec;
  if (!fs::exists(fs::path(template_dir) / "weights" / "weight.bin", ec)) {
    session->warn(fmt("ane-module: template has no weight blob: {}",
                      template_dir));
    return nullptr;
  }

  // Fresh copy per build: the stamp is destructive and two modules
  // over one template must not share bytes.
  fs::remove_all(work_dir, ec);
  fs::copy(template_dir, work_dir,
           fs::copy_options::recursive | fs::copy_options::overwrite_existing,
           ec);
  if (ec) {
    session->warn(fmt("ane-module: cannot stage template into {}: {}",
                      work_dir, ec.message()));
    return nullptr;
  }

  const std::string blob = weight_bin_of_(work_dir);
  std::vector<AneWeightSlot> slots;
  std::string err;
  if (!parse_weight_blob(blob, &slots, &err)) {
    session->warn(fmt("ane-module: weight blob not in the expected "
                      "format ({}); the caller should keep its GPU path",
                      err));
    fs::remove_all(work_dir, ec);
    return nullptr;
  }
  if (slots.size() != weights.size()) {
    session->warn(fmt("ane-module: template wants {} weights, caller "
                      "gave {}", slots.size(), weights.size()));
    fs::remove_all(work_dir, ec);
    return nullptr;
  }
  for (std::size_t i = 0; i < slots.size(); ++i) {
    if (slots[i].bytes != weights[i].bytes || weights[i].data == nullptr) {
      session->warn(fmt("ane-module: weight {} is {} bytes, slot is {}",
                        i, weights[i].bytes, slots[i].bytes));
      fs::remove_all(work_dir, ec);
      return nullptr;
    }
  }

  {
    std::fstream f(blob, std::ios::binary | std::ios::in | std::ios::out);
    if (!f) {
      session->warn(fmt("ane-module: cannot open blob for write: {}", blob));
      fs::remove_all(work_dir, ec);
      return nullptr;
    }
    for (std::size_t i = 0; i < slots.size(); ++i) {
      f.seekp((std::streamoff)slots[i].offset);
      f.write((const char*)weights[i].data,
              (std::streamsize)weights[i].bytes);
      if (!f) {
        session->warn(fmt("ane-module: short write stamping weight {}", i));
        fs::remove_all(work_dir, ec);
        return nullptr;
      }
    }
  }

  // 3 == CPU+NeuralEngine. Deliberately NOT "All": it forces the
  // ANE-or-CPU choice, so a graph that cannot run on the ANE shows up
  // as a large time rather than quietly succeeding on the GPU.
  auto model = mgr->load(work_dir, 3);
  if (model == nullptr || !model->valid()) {
    session->warn(fmt("ane-module: load failed for {}", work_dir));
    fs::remove_all(work_dir, ec);
    return nullptr;
  }
  if (model->input_names().size() != 1u ||
      model->output_names().size() != 1u) {
    session->warn(fmt("ane-module: expected exactly one input and one "
                      "output"));
    return nullptr;
  }

  // The staged copy has done its job: CoreML has read the weights into
  // its own compiled bundle by the time load() returns, and the model
  // keeps working -- byte-identically -- once the directory is gone.
  // Removing it here matters because the copy is the size of the
  // WEIGHTS: a 28-block Krea-2 would otherwise strand ~17 GB per run,
  // permanently, in whatever directory the graph named.
  fs::remove_all(work_dir, ec);

  std::unique_ptr<AneModule> m(new AneModule());
  m->_model    = std::move(model);
  m->_path     = work_dir;
  m->_in_name  = m->_model->input_names()[0];
  m->_out_name = m->_model->output_names()[0];

  const auto in_it  = m->_model->input_descs().find(m->_in_name);
  const auto out_it = m->_model->output_descs().find(m->_out_name);
  if (in_it == m->_model->input_descs().end() ||
      out_it == m->_model->output_descs().end()) {
    session->warn(fmt("ane-module: introspection failed"));
    return nullptr;
  }
  // Zero-copy on the output needs a fixed shape whose native dtype is
  // what we ask for; otherwise predict() quietly converts into its own
  // buffer and the whole point of the UMA path is lost.
  if (!out_it->second.fixed ||
      in_it->second.dtype != CoreMLDType::F16 ||
      out_it->second.dtype != CoreMLDType::F16) {
    session->warn(fmt("ane-module: template is not fixed-shape fp16 on "
                      "both ends; the zero-copy path would not be taken"));
    return nullptr;
  }
  m->_in_shape  = in_it->second.shape;
  m->_out_shape = out_it->second.shape;
  auto elems = [](const std::vector<std::int64_t>& s) {
    std::size_t n = 1;
    for (std::int64_t d : s) { n *= (std::size_t)(d > 0 ? d : 1); }
    return n;
  };
  m->_in_elems  = elems(m->_in_shape);
  m->_out_elems = elems(m->_out_shape);

  // Residency is a property of the GRAPH, not of the stamped bytes, so
  // it is decided offline by tools/make_ane_module.py (which can read
  // the compute plan) and recorded beside the template. Absent sidecar
  // == unverified, reported as not-resident rather than assumed.
  {
    std::ifstream s(template_dir + ".census");
    std::string   line;
    if (s && std::getline(s, line)) {
      m->_ane_resident = line.find("ANE") != std::string::npos &&
                         line.find("ANE=0") == std::string::npos;
    }
  }
  return m;
}

namespace {

// Gather logical matrices into per-tile slabs, n-tile major and k-tile
// minor. ONE implementation, because the stamped and the emitted path
// must agree about the order exactly -- every ordering produces
// correctly SHAPED slabs, so a disagreement is silent.
bool
gather_tiles_(std::span<const AneTiledWeight> weights,
              std::vector<std::vector<std::uint16_t>>* staging)
{
  for (const AneTiledWeight& w : weights) {
    if (w.data == nullptr || w.K == 0 || w.N == 0 || w.Bk == 0 ||
        w.Bn == 0) {
      return false;
    }
    const auto* src = static_cast<const std::uint16_t*>(w.data);
    const std::size_t n_tiles = (w.N + w.Bn - 1) / w.Bn;
    const std::size_t k_tiles = (w.K + w.Bk - 1) / w.Bk;
    for (std::size_t j = 0; j < n_tiles; ++j) {
      const std::size_t n0 = j * w.Bn;
      const std::size_t n1 = std::min(n0 + w.Bn, w.N);
      for (std::size_t t = 0; t < k_tiles; ++t) {
        const std::size_t k0 = t * w.Bk;
        const std::size_t k1 = std::min(k0 + w.Bk, w.K);
        std::vector<std::uint16_t> tile((k1 - k0) * (n1 - n0));
        for (std::size_t k = k0; k < k1; ++k) {
          std::memcpy(&tile[(k - k0) * (n1 - n0)], &src[k * w.N + n0],
                      (n1 - n0) * sizeof(std::uint16_t));
        }
        staging->push_back(std::move(tile));
      }
    }
  }
  return true;
}

}  // namespace

std::unique_ptr<AneModule>
AneModule::build_tiled(const SessionContextIntf*       session,
                       const std::string&              template_dir,
                       const std::string&              work_dir,
                       std::span<const AneTiledWeight> weights)
{
  if (session == nullptr) { return nullptr; }
  std::vector<std::vector<std::uint16_t>> staging;
  if (!gather_tiles_(weights, &staging)) {
    session->warn(fmt("ane-module: malformed tiled weight"));
    return nullptr;
  }
  std::vector<AneWeight> slabs;
  slabs.reserve(staging.size());
  for (const auto& t : staging) {
    slabs.push_back(AneWeight{t.data(), t.size() * sizeof(std::uint16_t)});
  }
  return build(session, template_dir, work_dir, slabs);
}

bool
AneModule::emitter_usable(const SessionContextIntf* session)
{
  if (session == nullptr) { return false; }
  auto* svc = session->services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return false; }
  // Once per process: the answer depends on the OS and the hardware,
  // neither of which changes under a running process.
  static bool checked = false;
  static bool ok      = false;
  if (checked) { return ok; }
  checked = true;
  std::string err;
  std::error_code ec;
  const std::string scratch =
      std::filesystem::temp_directory_path(ec).string();
  ok = AneEmitter::self_test(*mgr, scratch, &err);
  if (!ok) {
    session->warn(fmt("ane-module: the emitted CoreML format did not "
                      "verify on this machine ({}); every consumer keeps "
                      "its GPU path", err));
  }
  return ok;
}

std::unique_ptr<AneModule>
AneModule::build_emitted(const SessionContextIntf* session,
                         const AneGraphSpec&       spec,
                         std::span<const AneTiledWeight> weights,
                         const std::string&        work_dir)
{
  if (session == nullptr) { return nullptr; }
  auto* svc = session->services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return nullptr; }

  std::vector<std::vector<std::uint16_t>> staging;
  if (!gather_tiles_(weights, &staging)) {
    session->warn(fmt("ane-module: malformed tiled weight"));
    return nullptr;
  }
  std::vector<AneBlobSlab> slabs;
  slabs.reserve(staging.size());
  for (const auto& t : staging) {
    slabs.push_back(AneBlobSlab{t.data(), t.size() * sizeof(std::uint16_t)});
  }

  std::string err;
  std::error_code ec;
  if (!AneEmitter::emit(spec, slabs, work_dir, &err)) {
    session->warn(fmt("ane-module: emit failed ({}); the caller should "
                      "keep its GPU path", err));
    fs::remove_all(work_dir, ec);
    return nullptr;
  }

  auto model = mgr->load(work_dir, 3);
  fs::remove_all(work_dir, ec);     // CoreML has the weights by now
  if (model == nullptr || !model->valid()) {
    session->warn(fmt("ane-module: emitted model did not load"));
    return nullptr;
  }

  std::unique_ptr<AneModule> m(new AneModule());
  m->_model     = std::move(model);
  m->_path      = work_dir;
  m->_in_name   = AneEmitter::input_name();
  m->_out_name  = AneEmitter::output_name(spec);
  m->_in_shape  = {spec.M, spec.K};
  m->_out_shape = {spec.M,
                   spec.kind == AneGraphSpec::Kind::Matmul ? spec.N : spec.K};
  m->_in_elems  = AneEmitter::input_elems(spec);
  m->_out_elems = AneEmitter::output_elems(spec);
  // Emission is not a template, so there is no census sidecar to read.
  // Residency is a property of the SHAPE and is reported by the tool
  // that explores shapes; a consumer that needs certainty should check
  // its own timings.
  m->_ane_resident = true;
  return m;
}

bool
AneModule::run(const metal_compute::SharedBuffer& x, std::size_t xe,
               const metal_compute::SharedBuffer& y, std::size_t ye)
{
  if (_model == nullptr) { return false; }
  if (x.contents() == nullptr || y.contents() == nullptr) { return false; }
  if (x.byte_size() < (xe + _in_elems) * 2 ||
      y.byte_size() < (ye + _out_elems) * 2) {
    return false;
  }

  CoreMLPredictInput in;
  in.name  = _in_name;
  in.data  = static_cast<const std::uint16_t*>(x.contents()) + xe;
  in.dtype = CoreMLDType::F16;
  in.shape = _in_shape;

  CoreMLPredictOutput out;
  out.name          = _out_name;
  out.want          = CoreMLDType::F16;
  out.backing       = static_cast<std::uint16_t*>(y.contents()) + ye;
  out.backing_elems = _out_elems;

  const CoreMLPredictInput ins[1]  = {in};
  CoreMLPredictOutput      outs[1] = {out};
  return _model->predict(ins, outs);
}

std::unique_ptr<AneWeightSlots>
AneWeightSlots::create(const AneGraphSpec& spec)
{
  const std::vector<AneEmitter::Feature> f = AneEmitter::input_features(spec);
  if (!spec.runtime_weights || f.size() < 2u) { return nullptr; }
  std::unique_ptr<AneWeightSlots> s(new AneWeightSlots());
  CFDictionaryRef empty = CFDictionaryCreate(
      kCFAllocatorDefault, nullptr, nullptr, 0,
      &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  const void* keys[] = {kCVPixelBufferIOSurfacePropertiesKey};
  const void* vals[] = {empty};
  CFDictionaryRef attrs = CFDictionaryCreate(
      kCFAllocatorDefault, keys, vals, 1, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  bool ok = true;
  for (std::size_t i = 1; i < f.size() && ok; ++i) {
    const int rows = (int)f[i].shape[0], cols = (int)f[i].shape[1];
    CVPixelBufferRef pb = nullptr;
    // An int8 input (a rotated int8 graph's matrices) is an 8-bit pixel
    // buffer, which CoreML reads as MLMultiArrayDataTypeInt8.
    ok = CVPixelBufferCreate(kCFAllocatorDefault, (size_t)cols, (size_t)rows,
                             f[i].int8 ? kCVPixelFormatType_OneComponent8
                                       : kCVPixelFormatType_OneComponent16Half,
                             attrs, &pb) == kCVReturnSuccess &&
         pb != nullptr && CVPixelBufferGetIOSurface(pb) != nullptr;
    if (pb != nullptr) {
      s->_pb.push_back(pb);
      s->_i8.push_back(f[i].int8);
      s->_rows.push_back(rows);
      s->_cols.push_back(cols);
    }
  }
  CFRelease(attrs);
  CFRelease(empty);
  return ok ? std::move(s) : nullptr;
}

AneWeightSlots::~AneWeightSlots()
{
  for (void* pb : _pb) {
    CVPixelBufferRelease(static_cast<CVPixelBufferRef>(pb));
  }
}

std::size_t
AneWeightSlots::bytes() const noexcept
{
  std::size_t n = 0;
  for (std::size_t i = 0; i < _pb.size(); ++i) {
    n += (std::size_t)_rows[i] * (std::size_t)_cols[i] * (_i8[i] ? 1 : 2);
  }
  return n;
}

bool
AneWeightSlots::fill(
    std::size_t i,
    const std::function<void(void* base, std::size_t bytes_per_row)>& fill)
{
  if (i >= _pb.size()) { return false; }
  auto pb = static_cast<CVPixelBufferRef>(_pb[i]);
  if (CVPixelBufferLockBaseAddress(pb, 0) != kCVReturnSuccess) {
    return false;
  }
  fill(CVPixelBufferGetBaseAddress(pb), CVPixelBufferGetBytesPerRow(pb));
  CVPixelBufferUnlockBaseAddress(pb, 0);
  return true;
}

std::unique_ptr<AneModule>
AneModule::build_runtime(const SessionContextIntf* session,
                         const AneGraphSpec&       spec,
                         const std::string&        work_dir)
{
  if (session == nullptr || !spec.runtime_weights) { return nullptr; }
  auto* svc = session->services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return nullptr; }

  std::string err;
  std::error_code ec;
  if (!AneEmitter::emit(spec, {}, work_dir, &err)) {
    session->warn(fmt("ane-module: runtime-weight emit failed ({}); the "
                      "caller should keep its GPU path", err));
    fs::remove_all(work_dir, ec);
    return nullptr;
  }
  auto model = mgr->load(work_dir, 3);
  // No weight file to map, and the compiled program is cached on the
  // graph's content, so nothing here needs to outlive the load.
  fs::remove_all(work_dir, ec);
  if (model == nullptr || !model->valid()) {
    session->warn(fmt("ane-module: runtime-weight model did not load"));
    return nullptr;
  }

  std::unique_ptr<AneModule> m(new AneModule());
  m->_model     = std::move(model);
  m->_path      = work_dir;
  m->_in_name   = AneEmitter::input_name();
  m->_out_name  = AneEmitter::output_name(spec);
  m->_in_shape  = {spec.M, spec.K};
  m->_out_shape = {spec.M,
                   spec.kind == AneGraphSpec::Kind::Matmul ? spec.N : spec.K};
  m->_in_elems  = AneEmitter::input_elems(spec);
  m->_out_elems = AneEmitter::output_elems(spec);
  m->_ane_resident = true;
  const std::vector<AneEmitter::Feature> f = AneEmitter::input_features(spec);
  for (std::size_t i = 1; i < f.size(); ++i) {
    m->_w_names.push_back(f[i].name);
    m->_w_shapes.push_back(f[i].shape);
  }
  return m;
}

bool
AneModule::run(const metal_compute::SharedBuffer& x, std::size_t xe,
               const metal_compute::SharedBuffer& y, std::size_t ye,
               const AneWeightSlots& weights)
{
  if (_model == nullptr || _w_names.empty()) { return false; }
  if (x.contents() == nullptr || y.contents() == nullptr) { return false; }
  if (x.byte_size() < (xe + _in_elems) * 2 ||
      y.byte_size() < (ye + _out_elems) * 2) {
    return false;
  }
  if (weights.count() != _w_names.size()) { return false; }
  std::vector<CoreMLPredictInput> ins(1 + _w_names.size());
  ins[0].name  = _in_name;
  ins[0].data  = static_cast<const std::uint16_t*>(x.contents()) + xe;
  ins[0].dtype = CoreMLDType::F16;
  ins[0].shape = _in_shape;
  for (std::size_t i = 0; i < _w_names.size(); ++i) {
    // A slot of the wrong shape would bind and compute garbage, so it is
    // refused here rather than trusted.
    if (weights.rows(i) != _w_shapes[i][0] ||
        weights.cols(i) != _w_shapes[i][1]) {
      return false;
    }
    ins[i + 1].name         = _w_names[i];
    ins[i + 1].pixel_buffer = weights.pixel_buffer(i);
    ins[i + 1].dtype        = CoreMLDType::F16;
    ins[i + 1].shape        = _w_shapes[i];
  }
  CoreMLPredictOutput out;
  out.name          = _out_name;
  out.want          = CoreMLDType::F16;
  out.backing       = static_cast<std::uint16_t*>(y.contents()) + ye;
  out.backing_elems = _out_elems;
  CoreMLPredictOutput outs[1] = {out};
  return _model->predict(ins, outs);
}

}  // namespace genai
}  // namespace vpipe
