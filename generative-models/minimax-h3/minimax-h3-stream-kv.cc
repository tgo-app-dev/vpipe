#include "generative-models/minimax-h3/minimax-h3-stream-kv.h"

#include "common/temp-root.h"
#include "common/vpipe-format.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <filesystem>

namespace vpipe {
namespace genai {
namespace minimax_h3 {

using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::SharedBuffer;

// A committed chunk's records, one a layer. Unlinked as soon as it is
// made: closing the descriptor -- the retention letting the chunk go, or
// the process ending however it ends -- is what frees the space.
struct StreamKv::Segment {
  int fd = -1;
  int rows = 0;               // as stored: audio + video
  std::size_t record = 0;     // bytes a layer, a page multiple
  ~Segment()
  {
    if (fd >= 0) { ::close(fd); }
  }
};

namespace {

bool
pwrite_all_(int fd, const void* p, std::size_t n, std::uint64_t off)
{
  const auto* c = static_cast<const char*>(p);
  while (n > 0) {
    const ssize_t w = ::pwrite(fd, c, n, (off_t)off);
    if (w < 0 && errno == EINTR) { continue; }
    if (w <= 0) { return false; }
    c += w;
    n -= (std::size_t)w;
    off += (std::uint64_t)w;
  }
  return true;
}

bool
pread_all_(int fd, void* p, std::size_t n, std::uint64_t off)
{
  auto* c = static_cast<char*>(p);
  while (n > 0) {
    const ssize_t r = ::pread(fd, c, n, (off_t)off);
    if (r < 0 && errno == EINTR) { continue; }
    if (r <= 0) { return false; }
    c += r;
    n -= (std::size_t)r;
    off += (std::uint64_t)r;
  }
  return true;
}

std::size_t
round_up_(std::size_t n, std::size_t to)
{
  return (n + to - 1) / to * to;
}

std::shared_ptr<StreamKv::Segment>
make_segment_(std::string* err)
{
  // The application's temp root (VPIPE_TMPDIR: a host puts it where it
  // keeps its own scratch), never the OS one.
  const std::filesystem::path& dir = temp_root();
  std::string tmpl = (dir / "vpipe-h3-kv-XXXXXX").string();
  std::vector<char> name(tmpl.begin(), tmpl.end());
  name.push_back('\0');
  const int fd = ::mkstemp(name.data());
  if (fd < 0) {
    if (err != nullptr) {
      *err = fmt("stream kv: cannot create a file in {}", dir.string())();
    }
    return nullptr;
  }
  ::unlink(name.data());
  // Uncached both ways: the file is read back once a forward, in full, so
  // a page cache over it only competes with the model for the RAM this
  // exists to give back. Every offset and length is a page multiple, the
  // condition under which F_NOCACHE is honoured at all.
  (void)::fcntl(fd, F_NOCACHE, 1);
  auto seg = std::make_shared<StreamKv::Segment>();
  seg->fd = fd;
  return seg;
}

}  // namespace

StreamKv::~StreamKv()
{
  // A read or write still in flight writes into (or reads from) a slot.
  for (int s = 0; s < 2; ++s) { (void)join_(s); }
}

bool
StreamKv::init(metal_compute::MetalCompute* mc, int layers, int heads,
               int head_dim, int capacity, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    _k.clear();
    _v.clear();
    return false;
  };
  if (mc == nullptr || layers <= 0 || heads <= 0 || head_dim <= 0 ||
      capacity <= 0) {
    return fail("stream kv: bad geometry");
  }
  for (int s = 0; s < 2; ++s) { (void)join_(s); }
  _mc = mc;
  _layers = 0;
  _heads = heads;
  _hd = head_dim;
  _cap = capacity;
  _commits.clear();
  _k.clear();
  _v.clear();
  _sk = SharedBuffer{};
  _sv = SharedBuffer{};
  _staging_rows = 0;
  _disk = false;
  _bits = 16;
  _slot[0] = SharedBuffer{};
  _slot[1] = SharedBuffer{};
  _pending.reset();
  const std::size_t bytes =
      (std::size_t)heads * (std::size_t)capacity * (std::size_t)head_dim * 2;
  _k.reserve((std::size_t)layers);
  _v.reserve((std::size_t)layers);
  for (int l = 0; l < layers; ++l) {
    _k.push_back(mc->make_shared_buffer(bytes));
    _v.push_back(mc->make_shared_buffer(bytes));
    if (_k.back().empty() || _v.back().empty()) {
      return fail(fmt("stream kv: cannot allocate layer {} of {} ({} MB a "
                      "tensor, {} rows)", l, layers, bytes >> 20,
                      capacity)());
    }
  }
  _layers = layers;
  return true;
}

bool
StreamKv::init(metal_compute::MetalCompute* mc, int layers, int heads,
               int head_dim, int capacity, const Storage& st,
               std::string* err)
{
  if (!st.disk) {
    if (st.bits != 16) {
      if (err != nullptr) {
        *err = "stream kv: an 8-bit cache is kept on disk only";
      }
      return false;
    }
    return init(mc, layers, heads, head_dim, capacity, err);
  }
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    _slot[0] = SharedBuffer{};
    _slot[1] = SharedBuffer{};
    _layers = 0;
    return false;
  };
  for (int s = 0; s < 2; ++s) { (void)join_(s); }
  _commits.clear();
  _k.clear();
  _v.clear();
  _sk = SharedBuffer{};
  _sv = SharedBuffer{};
  _staging_rows = 0;
  _pending.reset();
  _layers = 0;
  if (mc == nullptr || layers <= 0 || heads <= 0 || head_dim <= 0 ||
      capacity <= 0 || st.slot_rows < capacity || st.new_rows <= 0) {
    return fail("stream kv: bad geometry");
  }
  if (st.bits != 16 && (st.bits != 8 || head_dim % 64 != 0)) {
    return fail(fmt("stream kv: {} bits over a head of {} (16, or 8 over "
                    "a multiple of 64)", st.bits, head_dim)());
  }
  _mc = mc;
  _heads = heads;
  _hd = head_dim;
  _cap = capacity;
  _disk = true;
  _bits = st.bits;
  const long pg = ::sysconf(_SC_PAGESIZE);
  _page = pg > 0 ? (std::size_t)pg : 16384;
  if (_bits == 8) {
    const metal_compute::ComputeLibrary lib =
        mc->load_library("llm_elementwise_bf16");
    _fn_quant = lib.function("kv_quant_u8g64");
    _fn_dequant = lib.function("kv_dequant_u8g64");
    if (!_fn_quant.valid() || !_fn_dequant.valid()) {
      return fail("stream kv: the 8-bit kernels are missing");
    }
  }
  // Each committed chunk's record is rounded to a page on its own, and at
  // most three are held beside the new one.
  const std::size_t slot = record_bytes_(st.slot_rows) + 3 * _page +
                           record_bytes_(st.new_rows);
  for (int s = 0; s < 2; ++s) {
    _slot[s] = mc->make_shared_buffer(slot);
    if (_slot[s].empty()) {
      return fail(fmt("stream kv: cannot allocate a {} MB layer slot",
                      slot >> 20)());
    }
  }
  _slot_layer[0] = _slot_layer[1] = -1;
  _layers = layers;
  return true;
}

void
StreamKv::clear()
{
  for (int s = 0; s < 2; ++s) { (void)join_(s); }
  _commits.clear();
  _pending.reset();
}

std::size_t
StreamKv::q_bytes_(int n) const
{
  return (std::size_t)_heads * (std::size_t)n * (std::size_t)_hd;
}

std::size_t
StreamKv::g_bytes_(int n) const
{
  return (std::size_t)_heads * (std::size_t)n * (std::size_t)(_hd / 64) * 2;
}

std::size_t
StreamKv::part_bytes_(int n) const
{
  return _bits == 8 ? q_bytes_(n) + 2 * g_bytes_(n) : 2 * q_bytes_(n);
}

std::size_t
StreamKv::record_bytes_(int n) const
{
  return round_up_(2 * part_bytes_(n), _page);
}

int
StreamKv::rows() const
{
  int n = 0;
  for (const Commit& c : _commits) { n += c.effective(); }
  return n;
}

int
StreamKv::audio_rows() const
{
  int n = 0;
  for (const Commit& c : _commits) { n += c.audio_dropped ? 0 : c.audio; }
  return n;
}

int
StreamKv::video_rows() const
{
  int n = 0;
  for (const Commit& c : _commits) { n += c.video; }
  return n;
}

std::size_t
StreamKv::bytes_for(int layers, int heads, int head_dim, int capacity,
                    int max_seq)
{
  const std::size_t row = (std::size_t)heads * (std::size_t)head_dim * 2;
  return (std::size_t)layers * 2 * row * (std::size_t)capacity +
         2 * row * (std::size_t)(capacity + max_seq);
}

std::size_t
StreamKv::bytes_for(int layers, int heads, int head_dim, int capacity,
                    int max_seq, const Storage& st)
{
  if (!st.disk) {
    return bytes_for(layers, heads, head_dim, capacity, max_seq);
  }
  // The slots as init() makes them, page slack included.
  const std::size_t page = 16384;
  const std::size_t el = st.bits == 8 ? 64 + 4 : 128;   // per 64 values
  auto rec = [&](int n) {
    return round_up_(2 * (std::size_t)heads * (std::size_t)n *
                         (std::size_t)(head_dim / 64) * el,
                     page);
  };
  const std::size_t row = (std::size_t)heads * (std::size_t)head_dim * 2;
  return 2 * (rec(st.slot_rows) + 3 * page + rec(st.new_rows)) +
         2 * row * (std::size_t)(capacity + max_seq);
}

std::size_t
StreamKv::bytes() const
{
  std::size_t n = _sk.byte_size() + _sv.byte_size();
  for (const SharedBuffer& b : _k) { n += b.byte_size(); }
  for (const SharedBuffer& b : _v) { n += b.byte_size(); }
  n += _slot[0].byte_size() + _slot[1].byte_size();
  return n;
}

std::size_t
StreamKv::disk_bytes() const
{
  std::size_t n = 0;
  for (const Commit& c : _commits) {
    if (c.seg) { n += c.seg->record * (std::size_t)_layers; }
  }
  return n;
}

std::size_t
StreamKv::staging_bytes(int seq) const
{
  return 2 * (std::size_t)_heads * (std::size_t)(_cap + seq) *
         (std::size_t)_hd * 2;
}

bool
StreamKv::adopt_staging(const SharedBuffer& buf, int rows)
{
  const std::size_t half =
      (std::size_t)_heads * (std::size_t)rows * (std::size_t)_hd * 2;
  if (rows <= 0 || buf.byte_size() < 2 * half) { return false; }
  _sk = buf.subview(0, half);
  _sv = buf.subview(half, half);
  if (_sk.empty() || _sv.empty()) {
    _sk = SharedBuffer{};
    _sv = SharedBuffer{};
    _staging_rows = 0;
    return false;
  }
  _staging_rows = rows;
  return true;
}

bool
StreamKv::ensure_staging(int seq)
{
  if (_mc == nullptr || seq <= 0) { return false; }
  const int need = _cap + seq;
  if (need <= _staging_rows && !_sk.empty() && !_sv.empty()) { return true; }
  const std::size_t bytes =
      (std::size_t)_heads * (std::size_t)need * (std::size_t)_hd * 2;
  _sk = SharedBuffer{};
  _sv = SharedBuffer{};
  _sk = _mc->make_shared_buffer(bytes);
  _sv = _mc->make_shared_buffer(bytes);
  if (_sk.empty() || _sv.empty()) {
    _staging_rows = 0;
    return false;
  }
  _staging_rows = need;
  return true;
}

int
StreamKv::drop_audio()
{
  int dropped = 0;
  for (Commit& c : _commits) {
    if (!c.audio_dropped && c.audio > 0) {
      dropped += c.audio;
      c.audio_dropped = true;
    }
  }
  return dropped;
}

bool
StreamKv::plan_commit(int n_text, int audio, int video, CommitPlan* out,
                      std::string* err) const
{
  if (out == nullptr || audio < 0 || video <= 0 || n_text < 0) {
    if (err != nullptr) { *err = "stream kv: bad commit"; }
    return false;
  }
  const int R = rows();
  // Each commit's place in STAGING, which holds the effective rows only.
  std::vector<int> at(_commits.size(), 0);
  {
    int e = 0;
    for (std::size_t i = 0; i < _commits.size(); ++i) {
      at[i] = e;
      e += _commits[i].effective();
    }
  }
  // The list after appending, then upstream's retain_sink_and_recent_
  // commits(): from three commits on, the first is aged to its video and
  // only the two latest are kept whole.
  struct Pick { int index; bool video_only; };   // index -1: the new one
  const int total = (int)_commits.size() + 1;
  std::vector<Pick> keep;
  if (total <= 2) {
    for (int i = 0; i < (int)_commits.size(); ++i) {
      keep.push_back({i, false});
    }
    keep.push_back({-1, false});
  } else {
    keep.push_back({0, true});
    for (int i = std::max(1, total - 2); i < total; ++i) {
      keep.push_back({i == total - 1 ? -1 : i, false});
    }
  }
  CommitPlan p;
  auto add = [&](int src, int n) {
    if (n <= 0) { return; }
    if (!p.ranges.empty() &&
        p.ranges.back().src + p.ranges.back().n == src) {
      p.ranges.back().n += n;
    } else {
      p.ranges.push_back({src, n});
    }
    p.rows_after += n;
  };
  p.new_src = R + n_text;
  p.new_audio = audio;
  p.new_video = video;
  for (const Pick& k : keep) {
    p.picks.push_back({k.index, k.video_only});
    if (k.index < 0) {
      add(R + n_text, audio + video);
      p.after.push_back({audio, video, false});
      continue;
    }
    const Commit& c = _commits[(std::size_t)k.index];
    const int a = c.audio_dropped ? 0 : c.audio;
    if (!k.video_only) { add(at[(std::size_t)k.index], a); }
    add(at[(std::size_t)k.index] + a, c.video);
    p.after.push_back({k.video_only ? 0 : a, c.video, false});
  }
  if (p.rows_after > _cap) {
    if (err != nullptr) {
      *err = fmt("stream kv: the retained set after this commit is {} rows, "
                 "past the {} the cache was sized for", p.rows_after,
                 _cap)();
    }
    return false;
  }
  *out = std::move(p);
  return true;
}

void
StreamKv::copy_rows_(ComputeEncoder& enc, const ComputeFunction& copy_rect,
                     const SharedBuffer& src, std::size_t src_off,
                     int src_rows, int src_row0, const SharedBuffer& dst,
                     std::size_t dst_off, int dst_rows, int dst_row0,
                     int n) const
{
  if (n <= 0) { return; }
  // Each HEAD is one "row" of the rect copy: n * head_dim contiguous
  // elements, a head's whole stride apart in each buffer.
  const int row_elems = n * _hd;
  enc.set_function(copy_rect);
  enc.set_buffer(0, src, src_off);
  enc.set_buffer(1, dst, dst_off);
  enc.set_constant(2, src_row0 * _hd);
  enc.set_constant(3, dst_row0 * _hd);
  enc.set_constant(4, _heads);
  enc.set_constant(5, row_elems);
  enc.set_constant(6, src_rows * _hd);
  enc.set_constant(7, dst_rows * _hd);
  enc.dispatch({(unsigned)(_heads * row_elems), 1, 1}, {256, 1, 1});
}

void
StreamKv::load_part_(ComputeEncoder& enc, const ComputeFunction& copy_rect,
                     const SharedBuffer& buf, std::size_t off, int rows,
                     int row0, int n, const SharedBuffer& dst, int dst_rows,
                     int dst_row0) const
{
  if (n <= 0) { return; }
  if (_bits != 8) {
    copy_rows_(enc, copy_rect, buf, off, rows, row0, dst, 0, dst_rows,
               dst_row0, n);
    return;
  }
  enc.set_function(_fn_dequant);
  enc.set_buffer(0, buf, off);
  enc.set_buffer(1, buf, off + q_bytes_(rows));
  enc.set_buffer(2, buf, off + q_bytes_(rows) + g_bytes_(rows));
  enc.set_buffer(3, dst);
  enc.set_constant(4, _heads);
  enc.set_constant(5, n);
  enc.set_constant(6, _hd);
  enc.set_constant(7, rows);
  enc.set_constant(8, row0);
  enc.set_constant(9, dst_rows);
  enc.set_constant(10, dst_row0);
  enc.dispatch({(unsigned)(_heads * n * (_hd / 4)), 1, 1}, {256, 1, 1});
}

void
StreamKv::store_part_(ComputeEncoder& enc, const ComputeFunction& copy_rect,
                      const SharedBuffer& src, int src_rows, int src_row0,
                      int n, const SharedBuffer& buf, std::size_t off) const
{
  if (n <= 0) { return; }
  if (_bits != 8) {
    copy_rows_(enc, copy_rect, src, 0, src_rows, src_row0, buf, off, n, 0,
               n);
    return;
  }
  enc.set_function(_fn_quant);
  enc.set_buffer(0, src);
  enc.set_buffer(1, buf, off);
  enc.set_buffer(2, buf, off + q_bytes_(n));
  enc.set_buffer(3, buf, off + q_bytes_(n) + g_bytes_(n));
  enc.set_constant(4, _heads);
  enc.set_constant(5, n);
  enc.set_constant(6, _hd);
  enc.set_constant(7, src_rows);
  enc.set_constant(8, src_row0);
  enc.dispatch({(unsigned)(_heads * n * (_hd / 64) * 32), 1, 1},
               {256, 1, 1});
}

void
StreamKv::encode_gather(ComputeEncoder& enc, const ComputeFunction& copy_rect,
                        int layer, const SharedBuffer& kh,
                        const SharedBuffer& vh, int seq) const
{
  const int R = rows();
  const int SR = R + seq;
  if (_disk) {
    // Each commit's records sit apart in the slot, [audio | video] a head;
    // an aged chunk's audio is skipped where it lies.
    const SharedBuffer& slot = _slot[layer & 1];
    int dst = 0;
    for (std::size_t i = 0; i < _commits.size(); ++i) {
      const Commit& c = _commits[i];
      const int n = c.audio + c.video;
      const int row0 = c.audio_dropped ? c.audio : 0;
      const int e = c.effective();
      load_part_(enc, copy_rect, slot, _slot_off[i], n, row0, e, _sk, SR,
                 dst);
      load_part_(enc, copy_rect, slot, _slot_off[i] + part_bytes_(n), n,
                 row0, e, _sv, SR, dst);
      dst += e;
    }
  } else {
    // The effective cache rows, skipping any whose audio is dropped.
    int phys = 0, dst = 0;
    auto both = [&](int src_row0, int dst_row0, int n) {
      copy_rows_(enc, copy_rect, _k[(std::size_t)layer], 0, _cap, src_row0,
                 _sk, 0, SR, dst_row0, n);
      copy_rows_(enc, copy_rect, _v[(std::size_t)layer], 0, _cap, src_row0,
                 _sv, 0, SR, dst_row0, n);
    };
    int run_src = 0, run_dst = 0, run_n = 0;
    auto flush = [&]() {
      both(run_src, run_dst, run_n);
      run_n = 0;
    };
    auto take = [&](int src, int n) {
      if (n <= 0) { return; }
      if (run_n > 0 && run_src + run_n == src) {
        run_n += n;
      } else {
        if (run_n > 0) { flush(); }
        run_src = src;
        run_dst = dst;
        run_n = n;
      }
      dst += n;
    };
    for (const Commit& c : _commits) {
      if (!c.audio_dropped) { take(phys, c.audio); }
      phys += c.audio;
      take(phys, c.video);
      phys += c.video;
    }
    if (run_n > 0) { flush(); }
  }
  // This forward's own rows, after them.
  copy_rows_(enc, copy_rect, kh, 0, seq, 0, _sk, 0, SR, R, seq);
  copy_rows_(enc, copy_rect, vh, 0, seq, 0, _sv, 0, SR, R, seq);
}

void
StreamKv::encode_commit(ComputeEncoder& enc, const ComputeFunction& copy_rect,
                        int layer, const CommitPlan& plan, int seq) const
{
  const int SR = rows() + seq;
  if (_disk) {
    // Only the new chunk is written: the chunks kept are in their files
    // already, and the ones let go are simply not read again.
    const SharedBuffer& slot = _slot[layer & 1];
    const int n = plan.new_audio + plan.new_video;
    store_part_(enc, copy_rect, _sk, SR, plan.new_src, n, slot, _new_off);
    store_part_(enc, copy_rect, _sv, SR, plan.new_src, n, slot,
                _new_off + part_bytes_(n));
    return;
  }
  int dst = 0;
  for (const CommitPlan::Range& r : plan.ranges) {
    copy_rows_(enc, copy_rect, _sk, 0, SR, r.src, _k[(std::size_t)layer], 0,
               _cap, dst, r.n);
    copy_rows_(enc, copy_rect, _sv, 0, SR, r.src, _v[(std::size_t)layer], 0,
               _cap, dst, r.n);
    dst += r.n;
  }
}

void
StreamKv::finish_commit(const CommitPlan& plan)
{
  if (!_disk) {
    _commits = plan.after;
    return;
  }
  // The files follow the picks; an aged chunk keeps its audio bytes,
  // skipped from now on.
  std::vector<Commit> next;
  next.reserve(plan.picks.size());
  for (const CommitPlan::Pick& k : plan.picks) {
    if (k.index < 0) {
      Commit c;
      c.audio = plan.new_audio;
      c.video = plan.new_video;
      c.seg = std::move(_pending);
      next.push_back(std::move(c));
      continue;
    }
    Commit c = _commits[(std::size_t)k.index];
    if (k.video_only) { c.audio_dropped = true; }
    next.push_back(std::move(c));
  }
  _commits = std::move(next);
  _pending.reset();
}

// ---- a forward's I/O, on disk -----------------------------------------

bool
StreamKv::join_(int s)
{
  if (!_io[s].valid()) { return true; }
  return _io[s].get();
}

bool
StreamKv::begin_forward(const CommitPlan* plan, std::string* err)
{
  _stats = IoStats{};
  if (!_disk) { return true; }
  for (int s = 0; s < 2; ++s) { (void)join_(s); }
  _slot_layer[0] = _slot_layer[1] = -1;
  _pending.reset();
  _pending_rows = 0;
  _slot_off.assign(_commits.size(), 0);
  std::size_t off = 0;
  for (std::size_t i = 0; i < _commits.size(); ++i) {
    _slot_off[i] = off;
    off += _commits[i].seg ? _commits[i].seg->record : 0;
  }
  _new_off = off;
  if (plan != nullptr) {
    _pending_rows = plan->new_audio + plan->new_video;
    off += record_bytes_(_pending_rows);
  }
  if (off > _slot[0].byte_size()) {
    if (err != nullptr) {
      *err = fmt("stream kv: a layer's {} MB of records exceed its {} MB "
                 "slot", off >> 20, _slot[0].byte_size() >> 20)();
    }
    return false;
  }
  if (plan != nullptr) {
    _pending = make_segment_(err);
    if (!_pending) { return false; }
    _pending->rows = _pending_rows;
    _pending->record = record_bytes_(_pending_rows);
  }
  prefetch(0);
  return true;
}

void
StreamKv::prefetch(int layer)
{
  if (!_disk || layer < 0 || layer >= _layers ||
      _slot_off.size() != _commits.size()) {
    return;
  }
  const int s = layer & 1;
  if (_slot_layer[s] == layer) { return; }
  _slot_layer[s] = layer;
  struct Read { int fd; std::uint64_t at; std::size_t n; char* dst; };
  std::vector<Read> reads;
  auto* base = static_cast<char*>(_slot[s].contents());
  for (std::size_t i = 0; i < _commits.size(); ++i) {
    const Segment* g = _commits[i].seg.get();
    if (g == nullptr || g->record == 0) { continue; }
    reads.push_back({g->fd, (std::uint64_t)layer * g->record, g->record,
                     base + _slot_off[i]});
    _stats.read += g->record;
  }
  if (reads.empty()) { return; }
  // After whatever the slot still owes its file (layer - 2's new rows).
  std::future<bool> prev = std::move(_io[s]);
  _io[s] = std::async(std::launch::async,
                      [prev = std::move(prev), reads]() mutable {
                        bool ok = !prev.valid() || prev.get();
                        for (const Read& r : reads) {
                          ok = ok && pread_all_(r.fd, r.dst, r.n, r.at);
                        }
                        return ok;
                      });
}

bool
StreamKv::acquire(int layer, std::string* err)
{
  if (!_disk) { return true; }
  const int s = layer & 1;
  if (_slot_layer[s] != layer) { prefetch(layer); }
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = join_(s);
  _stats.wait_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  if (!ok && err != nullptr) {
    *err = fmt("stream kv: layer {}'s cached keys and values could not be "
               "read back (or the layer before could not be written)",
               layer)();
  }
  return ok;
}

void
StreamKv::layer_done(int layer)
{
  if (!_disk || !_pending || layer < 0 || layer >= _layers) { return; }
  const int s = layer & 1;
  const int fd = _pending->fd;
  const std::size_t n = _pending->record;
  const std::uint64_t at = (std::uint64_t)layer * n;
  const char* src = static_cast<const char*>(_slot[s].contents()) + _new_off;
  _stats.written += n;
  std::future<bool> prev = std::move(_io[s]);
  _io[s] = std::async(std::launch::async,
                      [prev = std::move(prev), fd, src, n, at]() mutable {
                        const bool ok = !prev.valid() || prev.get();
                        return ok && pwrite_all_(fd, src, n, at);
                      });
}

bool
StreamKv::end_forward(std::string* err)
{
  if (!_disk) { return true; }
  bool ok = true;
  for (int s = 0; s < 2; ++s) { ok = join_(s) && ok; }
  _slot_layer[0] = _slot_layer[1] = -1;
  if (!ok && err != nullptr) {
    *err = "stream kv: the cache's file could not be read or written";
  }
  return ok;
}

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
