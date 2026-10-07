#include "generative-models/minimax-h3/minimax-h3-stream-kv.h"

#include "common/vpipe-format.h"

namespace vpipe {
namespace genai {
namespace minimax_h3 {

using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::SharedBuffer;

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
  _mc = mc;
  _heads = heads;
  _hd = head_dim;
  _cap = capacity;
  _commits.clear();
  _k.clear();
  _v.clear();
  _sk = SharedBuffer{};
  _sv = SharedBuffer{};
  _staging_rows = 0;
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
  return true;
}

void
StreamKv::clear()
{
  _commits.clear();
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
StreamKv::bytes() const
{
  std::size_t n = _sk.byte_size() + _sv.byte_size();
  for (const SharedBuffer& b : _k) { n += b.byte_size(); }
  for (const SharedBuffer& b : _v) { n += b.byte_size(); }
  return n;
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
  for (const Pick& k : keep) {
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
                     const SharedBuffer& src, int src_rows, int src_row0,
                     const SharedBuffer& dst, int dst_rows, int dst_row0,
                     int n) const
{
  if (n <= 0) { return; }
  // Each HEAD is one "row" of the rect copy: n * head_dim contiguous
  // elements, a head's whole stride apart in each buffer.
  const int row_elems = n * _hd;
  enc.set_function(copy_rect);
  enc.set_buffer(0, src);
  enc.set_buffer(1, dst);
  enc.set_constant(2, src_row0 * _hd);
  enc.set_constant(3, dst_row0 * _hd);
  enc.set_constant(4, _heads);
  enc.set_constant(5, row_elems);
  enc.set_constant(6, src_rows * _hd);
  enc.set_constant(7, dst_rows * _hd);
  enc.dispatch({(unsigned)(_heads * row_elems), 1, 1}, {256, 1, 1});
}

void
StreamKv::encode_gather(ComputeEncoder& enc, const ComputeFunction& copy_rect,
                        int layer, const SharedBuffer& kh,
                        const SharedBuffer& vh, int seq) const
{
  const int R = rows();
  const int SR = R + seq;
  // The effective cache rows, skipping any whose audio is dropped.
  int phys = 0, dst = 0;
  auto both = [&](int src_row0, int dst_row0, int n) {
    copy_rows_(enc, copy_rect, _k[(std::size_t)layer], _cap, src_row0, _sk,
               SR, dst_row0, n);
    copy_rows_(enc, copy_rect, _v[(std::size_t)layer], _cap, src_row0, _sv,
               SR, dst_row0, n);
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
  // This forward's own rows, after them.
  copy_rows_(enc, copy_rect, kh, seq, 0, _sk, SR, R, seq);
  copy_rows_(enc, copy_rect, vh, seq, 0, _sv, SR, R, seq);
}

void
StreamKv::encode_commit(ComputeEncoder& enc, const ComputeFunction& copy_rect,
                        int layer, const CommitPlan& plan, int seq) const
{
  const int SR = rows() + seq;
  int dst = 0;
  for (const CommitPlan::Range& r : plan.ranges) {
    copy_rows_(enc, copy_rect, _sk, SR, r.src, _k[(std::size_t)layer], _cap,
               dst, r.n);
    copy_rows_(enc, copy_rect, _sv, SR, r.src, _v[(std::size_t)layer], _cap,
               dst, r.n);
    dst += r.n;
  }
}

void
StreamKv::finish_commit(const CommitPlan& plan)
{
  _commits = plan.after;
}

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
