#include "pipeline/runtime-context.h"

#include <utility>

namespace vpipe {

RuntimeContext::RuntimeContext(std::vector<EdgeReader*>  in_readers,
                               std::vector<OportBuffer*> out_bufs,
                               std::atomic<bool>*        stop)
  : _in_readers(std::move(in_readers))
  , _out_bufs(std::move(out_bufs))
  , _stop(stop)
{}

RuntimeContext::~RuntimeContext() = default;

unsigned
RuntimeContext::num_iports() const noexcept
{
  return static_cast<unsigned>(_in_readers.size());
}

EdgeReader*
RuntimeContext::reader_(unsigned p) const noexcept
{
  if (p < _in_readers.size() && _in_readers[p] != nullptr) {
    return _in_readers[p];
  }
  return &_eos_reader;
}

bool
RuntimeContext::iport_connected(unsigned p) const noexcept
{
  return p < _in_readers.size()
      && _in_readers[p] != nullptr
      && _in_readers[p]->parent() != nullptr;
}

std::uint32_t
RuntimeContext::backlog(unsigned p) const noexcept
{
  return reader_(p)->backlog();
}

bool
RuntimeContext::eos(unsigned p) const
{
  return reader_(p)->at_eos();
}

unsigned
RuntimeContext::num_oports() const noexcept
{
  return static_cast<unsigned>(_out_bufs.size());
}

bool
RuntimeContext::has_consumers(unsigned out_port) const noexcept
{
  return out_port < _out_bufs.size()
      && _out_bufs[out_port]
      && _out_bufs[out_port]->num_cursors() > 0;
}

EdgeReader::ReadAwaiter
RuntimeContext::read(unsigned in_port)
{
  return reader_(in_port)->read();
}

EdgeReader::PeekAwaiter
RuntimeContext::peek(unsigned in_port, std::uint32_t offset)
{
  return reader_(in_port)->peek(offset);
}

// ---- read_any -------------------------------------------------------------

bool
RuntimeContext::ReadAnyAwaiter::await_ready()
{
  for (auto* r : _readers) {
    if (r->readable_now()) { return true; }
  }
  return _inbox != nullptr && _inbox->ready();
}

bool
RuntimeContext::ReadAnyAwaiter::await_suspend(std::coroutine_handle<> h)
{
  _state = std::make_shared<MultiReadWaiter>();
  _state->h = h;
  // Register on every not-yet-ready port. A port that is already
  // readable returns Ready (no registration) -> resume inline.
  bool any_ready = false;
  for (auto* r : _readers) {
    if (r->register_multi(_state) == EdgeReader::MultiReg::Ready) {
      any_ready = true;
    }
  }
  CommandInbox* inbox = _inbox;
  if (inbox != nullptr && !inbox->register_waiter(_state)) {
    any_ready = true;
  }
  // Arm last, under the state lock: if a registered port already
  // fired (or one was Ready), resume inline instead of suspending,
  // and claim `fired` so a late notify() becomes a no-op.
  bool inline_resume = any_ready;
  {
    std::lock_guard<std::mutex> lk(_state->mu);
    if (_state->fired) { inline_resume = true; }
    _state->armed = true;
    if (inline_resume) { _state->fired = true; }
  }
  if (inline_resume) {
    for (auto* r : _readers) { r->deregister_multi(_state); }
    if (inbox != nullptr) { inbox->deregister_waiter(_state); }
    return false;
  }
  return true;
}

void
RuntimeContext::ReadAnyAwaiter::await_resume()
{
  // Clear our registrations on the ports that did not win the wake
  // (the winner already cleared its own slot).
  if (_state) {
    for (auto* r : _readers) { r->deregister_multi(_state); }
    if (_inbox != nullptr) { _inbox->deregister_waiter(_state); }
  }
}

RuntimeContext::ReadAnyAwaiter
RuntimeContext::read_any(std::vector<unsigned> ports, bool commands)
{
  std::vector<EdgeReader*> rs;
  rs.reserve(ports.size());
  for (unsigned p : ports) { rs.push_back(reader_(p)); }
  return ReadAnyAwaiter{ std::move(rs), {}, commands ? _inbox : nullptr };
}

// ---- commands -------------------------------------------------------------

bool
RuntimeContext::has_commands() const noexcept
{
  return _inbox != nullptr;
}

std::shared_ptr<StageCommand>
RuntimeContext::try_command()
{
  return _inbox != nullptr ? _inbox->take() : nullptr;
}

NextCommandAwaiter
RuntimeContext::next_command()
{
  return NextCommandAwaiter{ _inbox, {}, {} };
}

void
RuntimeContext::cancel_open_commands(std::string_view why)
{
  if (_inbox != nullptr) { _inbox->cancel_open(why); }
}

void
RuntimeContext::attach_commands(CommandInbox* inbox) noexcept
{
  _inbox = inbox;
}

void
RuntimeContext::shut_commands(std::string_view why)
{
  if (_inbox != nullptr) { _inbox->shutdown(why); }
}

// ---- the rest -------------------------------------------------------------

EdgeReader::AcquireAwaiter
RuntimeContext::acquire(unsigned in_port)
{
  return reader_(in_port)->acquire();
}

void
RuntimeContext::release_read(unsigned in_port, std::uint32_t n)
{
  reader_(in_port)->release_read(n);
}

OportBuffer::WriteAwaiter
RuntimeContext::write(unsigned out_port, std::unique_ptr<BeatPayloadIntf> t)
{
  return _out_bufs[out_port]->write(std::move(t));
}

bool
RuntimeContext::write_sync(unsigned out_port,
                           std::unique_ptr<BeatPayloadIntf> t)
{
  return _out_bufs[out_port]->push_sync(std::move(t));
}

void
RuntimeContext::signal_done() noexcept
{
  _done = true;
}

bool
RuntimeContext::done() const noexcept
{
  return _done;
}

bool
RuntimeContext::stop_requested() const noexcept
{
  return _stop && _stop->load(std::memory_order_acquire);
}

void
RuntimeContext::close_outputs()
{
  for (auto* buf : _out_bufs) {
    if (buf) { buf->close(); }
  }
}

}
