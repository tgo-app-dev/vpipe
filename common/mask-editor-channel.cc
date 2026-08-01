#include "common/mask-editor-channel.h"

#include <chrono>
#include <utility>

namespace vpipe {

void
MaskEditorChannel::publish(Frame f)
{
  {
    std::lock_guard<std::mutex> lk(_mu);
    if (_closed) { return; }
    const std::uint64_t next = _frame.version + 1;
    _frame         = std::move(f);
    _frame.version = next;
    _frame.closed  = false;
  }
  _cv.notify_all();
}

std::uint64_t
MaskEditorChannel::commit(std::vector<std::uint8_t> png)
{
  std::uint64_t seq = 0;
  {
    std::lock_guard<std::mutex> lk(_mu);
    if (_closed) { return _commit.seq; }
    _commit.png =
        std::make_shared<const std::vector<std::uint8_t>>(std::move(png));
    seq = ++_commit.seq;
  }
  _cv.notify_all();
  return seq;
}

MaskEditorChannel::Commit
MaskEditorChannel::wait_commit(std::uint64_t since, int timeout_ms) const
{
  std::unique_lock<std::mutex> lk(_mu);
  _cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
               [&] { return _closed || _commit.seq != since; });
  return _commit;
}

std::uint64_t
MaskEditorChannel::commit_seq() const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  return _commit.seq;
}

void
MaskEditorChannel::close()
{
  {
    std::lock_guard<std::mutex> lk(_mu);
    if (_closed) { return; }
    _closed       = true;
    _frame.closed = true;
  }
  _cv.notify_all();
}

MaskEditorChannel::Frame
MaskEditorChannel::snapshot() const
{
  std::lock_guard<std::mutex> lk(_mu);
  return _frame;
}

MaskEditorChannel::Frame
MaskEditorChannel::wait_change(std::uint64_t since, int timeout_ms) const
{
  std::unique_lock<std::mutex> lk(_mu);
  _cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
               [&] { return _closed || _frame.version != since; });
  return _frame;
}

bool
MaskEditorChannel::closed() const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  return _closed;
}

}
