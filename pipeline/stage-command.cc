#include "pipeline/stage-command.h"

#include "common/thread-pool.h"
#include "pipeline/oport-buffer.h"
#include "pipeline/stage-config.h"
#include "pipeline/stage.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <format>
#include <utility>

using namespace std;

namespace vpipe {

// ---- element types ------------------------------------------------------

namespace {

struct TypeRow {
  ElementType      t;
  std::string_view name;
  std::size_t      size;
};

constexpr TypeRow kTypes[] = {
  {ElementType::Bytes, "bytes", 1},
  {ElementType::U8,    "u8",    1},
  {ElementType::I8,    "i8",    1},
  {ElementType::U16,   "u16",   2},
  {ElementType::I16,   "i16",   2},
  {ElementType::U32,   "u32",   4},
  {ElementType::I32,   "i32",   4},
  {ElementType::U64,   "u64",   8},
  {ElementType::I64,   "i64",   8},
  {ElementType::F16,   "f16",   2},
  {ElementType::BF16,  "bf16",  2},
  {ElementType::F32,   "f32",   4},
  {ElementType::F64,   "f64",   8},
};

std::string_view
trim_(std::string_view s) noexcept
{
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
    s.remove_prefix(1);
  }
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
    s.remove_suffix(1);
  }
  return s;
}

// Split on `sep`, trimming each piece. An empty input gives ONE empty
// piece, so "" as a shape alternative means rank 0.
vector<std::string_view>
split_(std::string_view s, char sep)
{
  vector<std::string_view> out;
  for (;;) {
    const std::size_t at = s.find(sep);
    out.push_back(trim_(s.substr(0, at)));
    if (at == std::string_view::npos) { break; }
    s.remove_prefix(at + 1);
  }
  return out;
}

string
shape_str_(const vector<std::int64_t>& shape)
{
  string s = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) { s += ","; }
    s += to_string(shape[i]);
  }
  return s + "]";
}

}  // namespace

std::string_view
element_type_name(ElementType t) noexcept
{
  for (const TypeRow& r : kTypes) {
    if (r.t == t) { return r.name; }
  }
  return "bytes";
}

bool
parse_element_type(std::string_view name, ElementType* out) noexcept
{
  name = trim_(name);
  for (const TypeRow& r : kTypes) {
    if (r.name == name) {
      if (out != nullptr) { *out = r.t; }
      return true;
    }
  }
  return false;
}

std::size_t
element_size(ElementType t) noexcept
{
  for (const TypeRow& r : kTypes) {
    if (r.t == t) { return r.size; }
  }
  return 1;
}

// ---- BufferLayout / DataBuffer ------------------------------------------

std::int64_t
BufferLayout::element_count() const noexcept
{
  std::int64_t n = 1;
  for (std::int64_t d : shape) { n *= d < 0 ? 0 : d; }
  return n;
}

std::vector<std::int64_t>
BufferLayout::contiguous_strides() const
{
  std::vector<std::int64_t> st(shape.size(), 0);
  std::int64_t step = static_cast<std::int64_t>(element_size(type));
  for (std::size_t i = shape.size(); i-- > 0;) {
    st[i] = step;
    step *= shape[i] < 1 ? 1 : shape[i];
  }
  return st;
}

bool
BufferLayout::is_contiguous() const noexcept
{
  if (strides.empty()) { return true; }
  if (strides.size() != shape.size()) { return false; }
  const auto want = contiguous_strides();
  for (std::size_t i = 0; i < shape.size(); ++i) {
    // A unit dimension is never stepped over, so its stride is free --
    // numpy reports whatever its allocator left there.
    if (shape[i] != 1 && strides[i] != want[i]) { return false; }
  }
  return true;
}

std::size_t
BufferLayout::extent_bytes() const noexcept
{
  if (element_count() == 0) { return 0; }
  const auto st = strides.empty() || strides.size() != shape.size()
                      ? contiguous_strides() : strides;
  std::int64_t last = 0;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    last += (shape[i] - 1) * st[i];
  }
  return static_cast<std::size_t>(last) + element_size(type);
}

BufferLayout
BufferLayout::contiguous(ElementType t, std::vector<std::int64_t> shape)
{
  BufferLayout l;
  l.type  = t;
  l.shape = std::move(shape);
  return l;
}

bool
DataBuffer::valid() const noexcept
{
  const std::size_t need = layout.extent_bytes();
  if (need == 0) { return true; }
  return data != nullptr && need <= size;
}

DataBuffer
DataBuffer::view(std::string name, const void* data, BufferLayout layout,
                 std::shared_ptr<void> owner)
{
  DataBuffer b;
  b.name   = std::move(name);
  b.size   = layout.extent_bytes();
  b.layout = std::move(layout);
  b.data   = const_cast<void*>(data);
  b.owner  = std::move(owner);
  return b;
}

std::string_view
command_state_name(CommandState s) noexcept
{
  switch (s) {
    case CommandState::Invalid:   return "invalid";
    case CommandState::Pending:   return "pending";
    case CommandState::Active:    return "active";
    case CommandState::Replied:   return "replied";
    case CommandState::Failed:    return "failed";
    case CommandState::Cancelled: return "cancelled";
    case CommandState::Closed:    return "closed";
  }
  return "invalid";
}

// ---- CommandHandle ------------------------------------------------------

CommandHandle::CommandHandle(std::shared_ptr<StageCommand> cmd) noexcept
  : _cmd(std::move(cmd))
{}

CommandHandle::~CommandHandle()
{
  close();
}

CommandHandle::CommandHandle(CommandHandle&& o) noexcept
  : _cmd(std::move(o._cmd))
{}

CommandHandle&
CommandHandle::operator=(CommandHandle&& o) noexcept
{
  if (this != &o) {
    close();
    _cmd = std::move(o._cmd);
  }
  return *this;
}

std::uint64_t
CommandHandle::id() const noexcept
{
  return _cmd ? _cmd->id() : 0;
}

CommandState
CommandHandle::state() const
{
  return _cmd ? _cmd->state() : CommandState::Invalid;
}

CommandState
CommandHandle::wait(int timeout_ms) const
{
  return _cmd ? _cmd->wait(timeout_ms) : CommandState::Invalid;
}

bool
CommandHandle::ok() const
{
  return state() == CommandState::Replied;
}

std::string
CommandHandle::result_json() const
{
  return _cmd ? _cmd->result().to_json() : std::string("null");
}

std::string
CommandHandle::error() const
{
  return _cmd ? _cmd->error() : std::string("no command");
}

std::vector<DataBuffer>
CommandHandle::buffers() const
{
  return _cmd ? _cmd->outputs() : std::vector<DataBuffer>{};
}

bool
CommandHandle::buffer(std::string_view name, DataBuffer* out) const
{
  for (DataBuffer& b : buffers()) {
    if (b.name == name) {
      if (out != nullptr) { *out = std::move(b); }
      return true;
    }
  }
  return false;
}

void
CommandHandle::close()
{
  if (_cmd) { _cmd->close(); }
}

// ---- StageCommand -------------------------------------------------------

StageCommand::StageCommand(const CommandSpec* spec, FlexData args,
                           std::vector<DataBuffer> in)
  : _spec(spec)
  , _args(std::move(args))
  , _in(std::move(in))
  , _result(FlexData::make_null())
{}

std::shared_ptr<StageCommand>
StageCommand::refused(std::string error)
{
  auto c = make_shared<StageCommand>(nullptr, FlexData::make_object(),
                                     std::vector<DataBuffer>{});
  c->_state = CommandState::Failed;
  c->_error = std::move(error);
  return c;
}

std::string_view
StageCommand::name() const noexcept
{
  return _spec != nullptr ? _spec->name : std::string_view{};
}

const CommandSpec*
StageCommand::spec() const noexcept
{
  return _spec;
}

std::uint64_t
StageCommand::id() const noexcept
{
  return _id;
}

const FlexData&
StageCommand::args() const noexcept
{
  return _args;
}

const std::vector<DataBuffer>&
StageCommand::inputs() const noexcept
{
  return _in;
}

const DataBuffer*
StageCommand::input(std::string_view name) const noexcept
{
  for (const DataBuffer& b : _in) {
    if (b.name == name) { return &b; }
  }
  return nullptr;
}

bool
StageCommand::reply(FlexData result, std::vector<DataBuffer> out)
{
  if (_spec == nullptr) { return false; }
  std::string err;
  if (!check_buffers(_spec->out, out, true, &err)) {
    fail("the stage replied with a buffer its spec does not describe: " +
         err);
    return false;
  }
  if (result.is_null()) { result = FlexData::make_object(); }
  {
    lock_guard<mutex> lk(_mu);
    if (_closed || (_state != CommandState::Pending &&
                    _state != CommandState::Active)) {
      return false;
    }
    _result = std::move(result);
    _out    = std::move(out);
    _state  = CommandState::Replied;
  }
  _cv.notify_all();
  return true;
}

void
StageCommand::fail(std::string message)
{
  {
    lock_guard<mutex> lk(_mu);
    if (_closed || (_state != CommandState::Pending &&
                    _state != CommandState::Active)) {
      return;
    }
    _state = CommandState::Failed;
    _error = std::move(message);
  }
  _cv.notify_all();
}

bool
StageCommand::closed() const noexcept
{
  lock_guard<mutex> lk(_mu);
  return _closed;
}

bool
StageCommand::cancelled() const noexcept
{
  lock_guard<mutex> lk(_mu);
  return _cancelled;
}

StageCommand::CloseAwaiter
StageCommand::until_closed()
{
  return CloseAwaiter{shared_from_this()};
}

bool
StageCommand::CloseAwaiter::await_ready() const noexcept
{
  return !_cmd || _cmd->closed();
}

bool
StageCommand::CloseAwaiter::await_suspend(std::coroutine_handle<> h)
{
  // A LOCAL copy: once _closer is published and the lock is released,
  // close() may resume the coroutine on another thread and destroy the
  // frame this awaiter lives in. Nothing below the unlock may touch it.
  std::shared_ptr<StageCommand> cmd = _cmd;
  lock_guard<mutex> lk(cmd->_mu);
  if (cmd->_closed || cmd->_pool == nullptr) { return false; }
  cmd->_closer = h;
  return true;
}

CommandState
StageCommand::state() const noexcept
{
  lock_guard<mutex> lk(_mu);
  return state_locked_();
}

CommandState
StageCommand::state_locked_() const noexcept
{
  if (_caller_closed) { return CommandState::Closed; }
  const bool open = _state == CommandState::Pending ||
                    _state == CommandState::Active;
  if (_cancelled && open) { return CommandState::Cancelled; }
  return _state;
}

CommandState
StageCommand::wait(int timeout_ms) const
{
  unique_lock<mutex> lk(_mu);
  auto settled = [&] {
    return _closed || (_state != CommandState::Pending &&
                       _state != CommandState::Active);
  };
  if (timeout_ms < 0) {
    _cv.wait(lk, settled);
  } else {
    _cv.wait_for(lk, chrono::milliseconds(timeout_ms), settled);
  }
  return state_locked_();
}

FlexData
StageCommand::result() const
{
  lock_guard<mutex> lk(_mu);
  return _state == CommandState::Replied ? _result : FlexData::make_null();
}

std::string
StageCommand::error() const
{
  lock_guard<mutex> lk(_mu);
  return _error;
}

std::vector<DataBuffer>
StageCommand::outputs() const
{
  lock_guard<mutex> lk(_mu);
  return _out;
}

void
StageCommand::close()
{
  std::vector<DataBuffer> dropped;
  std::coroutine_handle<> closer{};
  {
    lock_guard<mutex> lk(_mu);
    if (_caller_closed) { return; }
    _caller_closed = true;
    const bool already = _closed;
    _closed = true;
    closer  = already ? std::coroutine_handle<>{} : _closer;
    _closer = {};
    // The command's own references to the stage's memory go now, so a
    // stage that finds itself the last holder can reuse it without a
    // copy. A caller that kept a DataBuffer keeps its owner.
    dropped.swap(_out);
  }
  _cv.notify_all();
  wake_closer_(closer);
  // `dropped` releases here, outside the lock: an owner's destructor may
  // be arbitrary (a Python buffer defers to its interpreter).
}

void
StageCommand::set_queued_(std::uint64_t id, ThreadPool* pool) noexcept
{
  lock_guard<mutex> lk(_mu);
  _id   = id;
  _pool = pool;
}

void
StageCommand::activate_() noexcept
{
  lock_guard<mutex> lk(_mu);
  if (_state == CommandState::Pending && !_closed) {
    _state = CommandState::Active;
  }
}

void
StageCommand::cancel_(std::string_view why)
{
  std::coroutine_handle<> closer{};
  {
    lock_guard<mutex> lk(_mu);
    if (_closed) { return; }
    _closed    = true;
    _cancelled = true;
    // An unanswered command becomes Cancelled. An ANSWERED one keeps its
    // outcome and its buffers -- the caller still holds views of them,
    // and their owners keep that memory alive -- but the stage is released
    // from holding it: from here it promises nothing about the contents.
    if (_state == CommandState::Pending || _state == CommandState::Active) {
      _error = std::string(why);
    }
    closer  = _closer;
    _closer = {};
  }
  _cv.notify_all();
  wake_closer_(closer);
}

void
StageCommand::wake_closer_(std::coroutine_handle<> h) noexcept
{
  ThreadPool* pool = nullptr;
  {
    lock_guard<mutex> lk(_mu);
    pool = _pool;
  }
  if (h && pool != nullptr) { pool->schedule(h); }
}

// ---- CommandInbox -------------------------------------------------------

CommandInbox::CommandInbox(std::string stage_id, ThreadPool* pool,
                           std::size_t capacity)
  : _stage_id(std::move(stage_id))
  , _pool(pool)
  , _capacity(capacity == 0 ? 1 : capacity)
{}

CommandInbox::~CommandInbox()
{
  shutdown("the stage is gone");
}

bool
CommandInbox::post(const std::shared_ptr<StageCommand>& cmd,
                   std::string* err)
{
  vector<shared_ptr<MultiReadWaiter>> wake;
  {
    lock_guard<mutex> lk(_mu);
    if (_shut) {
      if (err != nullptr) {
        *err = std::format("stage '{}' is not taking commands: {}",
                           _stage_id, _shut_why);
      }
      return false;
    }
    std::size_t open = 0;
    for (const auto& q : _queue) {
      if (!q->closed()) { ++open; }
    }
    if (open >= _capacity) {
      if (err != nullptr) {
        *err = std::format("stage '{}' already has {} commands waiting "
                           "and is not taking them", _stage_id, open);
      }
      return false;
    }
    cmd->set_queued_(_next_id++, _pool);
    _queue.push_back(cmd);
    wake.swap(_waiters);
  }
  for (auto& w : wake) { w->notify(); }
  return true;
}

std::shared_ptr<StageCommand>
CommandInbox::take()
{
  lock_guard<mutex> lk(_mu);
  while (!_queue.empty()) {
    shared_ptr<StageCommand> c = std::move(_queue.front());
    _queue.pop_front();
    // A command its caller already gave up on is not work.
    if (c->closed()) { continue; }
    c->activate_();
    _taken.erase(remove_if(_taken.begin(), _taken.end(),
                           [](const weak_ptr<StageCommand>& w) {
                             return w.expired();
                           }),
                 _taken.end());
    _taken.push_back(c);
    return c;
  }
  return nullptr;
}

bool
CommandInbox::ready() const
{
  lock_guard<mutex> lk(_mu);
  if (_shut) { return true; }
  for (const auto& q : _queue) {
    if (!q->closed()) { return true; }
  }
  return false;
}

bool
CommandInbox::shut() const
{
  lock_guard<mutex> lk(_mu);
  return _shut;
}

bool
CommandInbox::register_waiter(const std::shared_ptr<MultiReadWaiter>& w)
{
  lock_guard<mutex> lk(_mu);
  if (_shut) { return false; }
  for (const auto& q : _queue) {
    if (!q->closed()) { return false; }
  }
  if (w->pool == nullptr) { w->pool = _pool; }
  _waiters.push_back(w);
  return true;
}

void
CommandInbox::deregister_waiter(const std::shared_ptr<MultiReadWaiter>& w)
{
  lock_guard<mutex> lk(_mu);
  _waiters.erase(std::remove(_waiters.begin(), _waiters.end(), w),
                 _waiters.end());
}

void
CommandInbox::shutdown(std::string_view why)
{
  deque<shared_ptr<StageCommand>>     queued;
  vector<weak_ptr<StageCommand>>      taken;
  vector<shared_ptr<MultiReadWaiter>> wake;
  {
    lock_guard<mutex> lk(_mu);
    if (_shut) { return; }
    _shut     = true;
    _shut_why = std::string(why);
    queued.swap(_queue);
    taken.swap(_taken);
    wake.swap(_waiters);
  }
  for (auto& c : queued) { c->cancel_(why); }
  for (auto& w : taken) {
    if (auto c = w.lock()) { c->cancel_(why); }
  }
  for (auto& w : wake) { w->notify(); }
}

void
CommandInbox::cancel_open(std::string_view why)
{
  vector<weak_ptr<StageCommand>> taken;
  {
    lock_guard<mutex> lk(_mu);
    taken.swap(_taken);
  }
  for (auto& w : taken) {
    if (auto c = w.lock()) { c->cancel_(why); }
  }
}

// ---- NextCommandAwaiter -------------------------------------------------

bool
NextCommandAwaiter::await_ready()
{
  if (_inbox == nullptr) { return true; }
  _got = _inbox->take();
  return _got != nullptr || _inbox->shut();
}

bool
NextCommandAwaiter::await_suspend(std::coroutine_handle<> h)
{
  // Locals only past the point the waiter is armed: a post() on another
  // thread may resume the coroutine and free this awaiter's frame.
  CommandInbox* inbox = _inbox;
  auto st = make_shared<MultiReadWaiter>();
  st->h = h;
  _state = st;
  if (!inbox->register_waiter(st)) { return false; }
  bool inline_resume = false;
  {
    lock_guard<mutex> lk(st->mu);
    inline_resume = st->fired;
    st->armed     = true;
    if (inline_resume) { st->fired = true; }
  }
  if (inline_resume) {
    inbox->deregister_waiter(st);
    return false;
  }
  return true;
}

std::shared_ptr<StageCommand>
NextCommandAwaiter::await_resume()
{
  if (_got) { return std::move(_got); }
  if (_inbox == nullptr) { return nullptr; }
  if (_state) { _inbox->deregister_waiter(_state); }
  return _inbox->take();
}

// ---- validation ---------------------------------------------------------

namespace {

bool
is_identifier_(std::string_view t) noexcept
{
  if (t.empty()) { return false; }
  if (!(std::isalpha(static_cast<unsigned char>(t[0])) || t[0] == '_')) {
    return false;
  }
  for (char c : t) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
      return false;
    }
  }
  return true;
}

bool
parse_dim_(std::string_view t, std::int64_t* out) noexcept
{
  if (t.empty()) { return false; }
  std::int64_t v = 0;
  for (char c : t) {
    if (!std::isdigit(static_cast<unsigned char>(c))) { return false; }
    v = v * 10 + (c - '0');
  }
  *out = v;
  return true;
}

// One token against one dimension, binding a NAME into `syms`.
bool
match_dim_(std::string_view tok, std::int64_t d, ShapeSymbols& syms)
{
  if (tok == "?") { return true; }
  std::int64_t want = 0;
  if (parse_dim_(tok, &want)) { return d == want; }
  if (!is_identifier_(tok)) { return false; }
  for (const auto& [name, v] : syms.bound) {
    if (name == tok) { return v == d; }
  }
  syms.bound.emplace_back(std::string(tok), d);
  return true;
}

bool
match_alternative_(std::string_view alt, const vector<std::int64_t>& shape,
                   ShapeSymbols& syms)
{
  const vector<std::string_view> toks =
      alt.empty() ? vector<std::string_view>{} : split_(alt, ',');
  std::size_t ell = toks.size();
  for (std::size_t i = 0; i < toks.size(); ++i) {
    if (toks[i] == "...") {
      if (ell != toks.size()) { return false; }   // two ellipses
      ell = i;
    }
  }
  if (ell == toks.size()) {
    if (shape.size() != toks.size()) { return false; }
    for (std::size_t i = 0; i < toks.size(); ++i) {
      if (!match_dim_(toks[i], shape[i], syms)) { return false; }
    }
    return true;
  }
  const std::size_t head = ell;
  const std::size_t tail = toks.size() - ell - 1;
  if (shape.size() < head + tail) { return false; }
  for (std::size_t i = 0; i < head; ++i) {
    if (!match_dim_(toks[i], shape[i], syms)) { return false; }
  }
  for (std::size_t i = 0; i < tail; ++i) {
    if (!match_dim_(toks[ell + 1 + i], shape[shape.size() - tail + i],
                    syms)) {
      return false;
    }
  }
  return true;
}

string
names_of_(std::span<const BufferSpec> spec)
{
  string s;
  for (const BufferSpec& b : spec) {
    if (!s.empty()) { s += ", "; }
    s += string(b.name);
  }
  return s.empty() ? string("none") : s;
}

bool
check_one_(const BufferSpec& sp, DataBuffer& b, bool outbound,
           ShapeSymbols& syms, string* err)
{
  auto bad = [&](const string& why) {
    if (err != nullptr) { *err = std::format("buffer '{}': {}", b.name, why); }
    return false;
  };
  BufferLayout& l = b.layout;
  if (!l.strides.empty() && l.strides.size() != l.shape.size()) {
    return bad(std::format("{} strides for {} dimensions",
                           l.strides.size(), l.shape.size()));
  }
  for (std::int64_t d : l.shape) {
    if (d < 0) { return bad("a negative dimension"); }
  }
  for (std::int64_t st : l.strides) {
    if (st < 0) { return bad("a negative stride"); }
  }

  // Element type. "bytes" takes any one-byte type as a flat, contiguous
  // string and hands the stage Bytes plus the declared format.
  const std::string_view types = trim_(sp.type);
  bool type_ok = types.empty();
  for (std::string_view t : split_(types, ',')) {
    if (t.empty()) { continue; }
    if (t == "bytes") {
      const bool one_byte = l.type == ElementType::Bytes ||
                            l.type == ElementType::U8 ||
                            l.type == ElementType::I8;
      if (one_byte && l.shape.size() == 1 && l.is_contiguous()) {
        if (!l.format.empty() && !sp.format.empty() &&
            l.format != sp.format) {
          return bad(std::format("format '{}' where '{}' is expected",
                                 l.format, sp.format));
        }
        l.type = ElementType::Bytes;
        if (l.format.empty()) { l.format = string(sp.format); }
        type_ok = true;
        break;
      }
      continue;
    }
    ElementType want{};
    if (parse_element_type(t, &want) && want == l.type) {
      type_ok = true;
      break;
    }
  }
  if (!type_ok) {
    return bad(std::format(
        "type {}{} where '{}' is expected", element_type_name(l.type),
        l.type == ElementType::Bytes || l.type == ElementType::U8
            ? std::format(" rank {}", l.shape.size()) : string(),
        sp.type));
  }
  if (sp.contiguous && !l.is_contiguous()) {
    return bad("must be contiguous");
  }
  if (!match_shape(sp.shape, l.shape, &syms)) {
    return bad(std::format("shape {} does not match '{}'",
                           shape_str_(l.shape), sp.shape));
  }
  if (!b.valid()) {
    return bad(std::format("the layout reaches {} bytes but the buffer "
                           "has {}{}", l.extent_bytes(), b.size,
                           b.data == nullptr ? " (no data)" : ""));
  }
  // Who may write is the SPEC's statement, not the buffer's: an inbound
  // buffer is the caller's memory and never the stage's to write, and an
  // outbound one is writable exactly when its channel says so.
  b.writable = outbound && sp.writable;
  return true;
}

}  // namespace

bool
match_shape(std::string_view pattern, const std::vector<std::int64_t>& shape,
            ShapeSymbols* symbols)
{
  pattern = trim_(pattern);
  if (pattern.empty()) { return true; }
  ShapeSymbols scratch;
  ShapeSymbols& base = symbols != nullptr ? *symbols : scratch;
  for (std::string_view alt : split_(pattern, '|')) {
    ShapeSymbols trial = base;
    if (match_alternative_(alt, shape, trial)) {
      base = std::move(trial);
      return true;
    }
  }
  return false;
}

bool
check_buffers(std::span<const BufferSpec> spec, std::vector<DataBuffer>& bufs,
              bool outbound, std::string* err)
{
  const char* dir = outbound ? "reply" : "command";
  for (std::size_t i = 0; i < bufs.size(); ++i) {
    bool known = false;
    for (const BufferSpec& sp : spec) {
      if (sp.name == bufs[i].name) { known = true; break; }
    }
    if (!known) {
      if (err != nullptr) {
        *err = std::format("the {} declares no buffer '{}' (it declares: "
                           "{})", dir, bufs[i].name, names_of_(spec));
      }
      return false;
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (bufs[j].name == bufs[i].name) {
        if (err != nullptr) {
          *err = std::format("buffer '{}' is given twice", bufs[i].name);
        }
        return false;
      }
    }
  }
  ShapeSymbols syms;
  for (const BufferSpec& sp : spec) {
    DataBuffer* b = nullptr;
    for (DataBuffer& c : bufs) {
      if (c.name == sp.name) { b = &c; break; }
    }
    if (b == nullptr) {
      if (sp.optional) { continue; }
      if (err != nullptr) {
        *err = std::format("the {} requires buffer '{}'", dir, sp.name);
      }
      return false;
    }
    if (!check_one_(sp, *b, outbound, syms, err)) { return false; }
  }
  return true;
}

bool
check_args(std::span<const ConfigKey> spec, FlexData& args, std::string* err)
{
  if (args.is_null()) { args = FlexData::make_object(); }
  if (!args.is_object()) {
    if (err != nullptr) { *err = "the arguments must be a JSON object"; }
    return false;
  }
  auto declared = [&](std::string_view k) -> const ConfigKey* {
    for (const ConfigKey& c : spec) {
      if (c.key == k) { return &c; }
    }
    return nullptr;
  };
  auto fits = [](const FlexData& v, ConfigType t) {
    switch (t) {
      case ConfigType::Any:    return true;
      case ConfigType::Bool:   return v.is_bool();
      case ConfigType::Int:    return v.is_int() || v.is_uint();
      case ConfigType::Uint:   return v.is_uint() ||
                                      (v.is_int() && v.as_int(-1) >= 0);
      case ConfigType::Real:   return v.is_real() || v.is_int() ||
                                      v.is_uint();
      case ConfigType::String:
      case ConfigType::Text:   return v.is_string();
      case ConfigType::Array:  return v.is_array();
      case ConfigType::Object: return v.is_object();
    }
    return false;
  };
  auto o = args.as_object();
  for (const auto& [k, v] : o) {
    const ConfigKey* c = declared(k);
    if (c == nullptr) {
      if (err != nullptr) {
        string names;
        for (const ConfigKey& d : spec) {
          if (!names.empty()) { names += ", "; }
          names += string(d.key);
        }
        *err = std::format("no argument '{}' (the command takes: {})", k,
                           names.empty() ? string("none") : names);
      }
      return false;
    }
    if (!fits(v, c->type)) {
      if (err != nullptr) {
        *err = std::format("argument '{}' must be {}", k,
                           config_type_name(c->type));
      }
      return false;
    }
  }
  for (const ConfigKey& c : spec) {
    if (o.contains(c.key)) { continue; }
    if (c.required) {
      if (err != nullptr) {
        *err = std::format("argument '{}' is required", c.key);
      }
      return false;
    }
    o.insert(c.key, config_default_value(c));
  }
  return true;
}

// ---- the host's entry points --------------------------------------------

std::shared_ptr<StageCommand>
open_stage_command(Stage& s, std::string_view name, FlexData args,
                   std::vector<DataBuffer> in)
{
  const StageSpec&   sp = s.spec();
  const CommandSpec* cs = find_command(sp, name);
  if (cs == nullptr) {
    string names;
    for (const CommandSpec& c : spec_commands(sp)) {
      if (!names.empty()) { names += ", "; }
      names += string(c.name);
    }
    return StageCommand::refused(std::format(
        "stage '{}' ({}) has no command '{}'; {}", s.id(), s.type_name(),
        name, names.empty() ? string("it declares none")
                            : "it declares: " + names));
  }
  std::string err;
  if (!check_args(cs->args, args, &err) ||
      !check_buffers(cs->in, in, false, &err)) {
    return StageCommand::refused(std::format(
        "stage '{}' command '{}': {}", s.id(), name, err));
  }
  std::shared_ptr<CommandInbox> inbox = s.command_inbox();
  if (!inbox) {
    return StageCommand::refused(std::format(
        "stage '{}' is not running: a command reaches a stage only while "
        "its pipeline is launched", s.id()));
  }
  auto cmd = make_shared<StageCommand>(cs, std::move(args), std::move(in));
  if (!inbox->post(cmd, &err)) { return StageCommand::refused(err); }
  return cmd;
}

FlexData
describe_commands(const StageSpec& spec)
{
  auto keys = [](std::span<const ConfigKey> ks) {
    FlexData a = FlexData::make_array();
    auto av = a.as_array();
    for (const ConfigKey& k : ks) {
      FlexData e = FlexData::make_object();
      auto eo = e.as_object();
      eo.insert("key", FlexData::make_string(k.key));
      eo.insert("type", FlexData::make_string(config_type_name(k.type)));
      eo.insert("required", FlexData::make_bool(k.required));
      if (!k.doc.empty()) { eo.insert("doc", FlexData::make_string(k.doc)); }
      eo.insert("default", config_default_value(k));
      av.push_back(std::move(e));
    }
    return a;
  };
  auto bufs = [](std::span<const BufferSpec> bs) {
    FlexData a = FlexData::make_array();
    auto av = a.as_array();
    for (const BufferSpec& b : bs) {
      FlexData e = FlexData::make_object();
      auto eo = e.as_object();
      eo.insert("name", FlexData::make_string(b.name));
      eo.insert("doc", FlexData::make_string(b.doc));
      eo.insert("type", FlexData::make_string(b.type));
      eo.insert("shape", FlexData::make_string(b.shape));
      eo.insert("format", FlexData::make_string(b.format));
      eo.insert("optional", FlexData::make_bool(b.optional));
      eo.insert("writable", FlexData::make_bool(b.writable));
      eo.insert("contiguous", FlexData::make_bool(b.contiguous));
      av.push_back(std::move(e));
    }
    return a;
  };
  FlexData arr = FlexData::make_array();
  auto av = arr.as_array();
  for (const CommandSpec& c : spec_commands(spec)) {
    FlexData e = FlexData::make_object();
    auto eo = e.as_object();
    eo.insert("name", FlexData::make_string(c.name));
    eo.insert("doc", FlexData::make_string(c.doc));
    eo.insert("holds", FlexData::make_bool(c.holds));
    eo.insert("args", keys(c.args));
    eo.insert("results", keys(c.results));
    eo.insert("in", bufs(c.in));
    eo.insert("out", bufs(c.out));
    av.push_back(std::move(e));
  }
  return arr;
}

}
