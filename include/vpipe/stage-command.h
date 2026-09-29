// vpipe::CommandHandle / DataBuffer -- talking to a RUNNING stage.
//
// A stage may declare COMMAND CHANNELS in its spec: named requests a
// caller outside the pipeline sends it while the pipeline runs, each
// carrying JSON arguments and, optionally, DATA. Data moves by
// reference in both directions, so a command costs no copy however
// large its buffers are:
//
//   in   the caller hands the stage buffers IT owns. The stage reads
//        them -- copies them into a beat, decodes them, whatever the
//        command means -- and may keep a reference past its reply; the
//        buffer's `owner` keeps the memory alive until the last
//        reference is dropped, on whichever side that is.
//   out  the stage replies with views of ITS OWN memory. The caller
//        reads them in place -- or, for a writable buffer, writes them
//        -- until it CLOSES the command, and only then may the stage
//        release or reuse that memory. A stage whose exposed bytes
//        would otherwise travel on holds its output while the command
//        is open, so everything downstream of it waits exactly that
//        long.
//
// One command is one request and one reply, in the order the stage's
// spec declares, and commands to one stage are served first-in
// first-out: N commands are N requests, never merged.
//
// The protocol, the layout grammar and the stage side are specified in
// docs/STAGE-COMMANDS.md.
//
// Threading: every method here may be called from any thread. A handle
// is move-only; destroying it CLOSES the command.

#ifndef VPIPE_STAGE_COMMAND_H
#define VPIPE_STAGE_COMMAND_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "vpipe/export.h"

VPIPE_API_BEGIN

namespace vpipe {

// The element type of a buffer.
//
// `Bytes` is an opaque byte string -- an encoded image, a file -- whose
// meaning is named by BufferLayout::format. The numeric types are the
// usual fixed-width integers and IEEE floats; BF16 is bfloat16.
enum class ElementType : std::uint8_t {
  Bytes = 0,
  U8, I8, U16, I16, U32, I32, U64, I64,
  F16, BF16, F32, F64,
};

// "bytes", "u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "f16",
// "bf16", "f32", "f64". Never null.
std::string_view element_type_name(ElementType) noexcept;

// The inverse; false (leaving *out untouched) for an unknown name.
bool parse_element_type(std::string_view name, ElementType* out) noexcept;

// Bytes per element (1 for Bytes).
std::size_t element_size(ElementType) noexcept;

// How the bytes at DataBuffer::data are laid out. Row-major by
// default; `strides` generalises that to any layout a numpy array or a
// DLPack tensor can describe, including padded rows. Strides are in
// BYTES, one per dimension, and are never negative.
struct BufferLayout {
  ElementType               type = ElementType::U8;
  std::vector<std::int64_t> shape;     // elements per dimension
  std::vector<std::int64_t> strides;   // bytes; empty = C-contiguous
  // For Bytes: what the bytes encode, as a media type ("image/png").
  // Empty when the channel's spec says, or when it does not matter.
  std::string               format;

  // Product of `shape` (1 for a scalar, 0 when any dim is 0).
  std::int64_t element_count() const noexcept;

  // True when the layout is C-contiguous: empty strides, or strides
  // equal to the row-major ones.
  bool is_contiguous() const noexcept;

  // The row-major strides for `shape`, whatever `strides` says.
  std::vector<std::int64_t> contiguous_strides() const;

  // Bytes from the first element to one past the last element the
  // layout reaches -- the size a buffer must at least have.
  std::size_t extent_bytes() const noexcept;

  // A C-contiguous layout.
  static BufferLayout contiguous(ElementType t,
                                 std::vector<std::int64_t> shape);
};

// One named buffer in a command, in either direction. A VIEW: it does
// not own `data` unless `owner` does, and copying a DataBuffer copies
// the view and shares the owner.
struct DataBuffer {
  std::string           name;
  BufferLayout          layout;
  void*                 data     = nullptr;
  std::size_t           size     = 0;      // bytes addressable at data
  bool                  writable = false;
  // Keeps `data` alive for as long as any copy of this buffer does.
  // For an INBOUND buffer it is the caller's: set it to something that
  // owns the memory (the vector, the array, a custom deleter), or leave
  // it empty and keep the memory alive yourself until the command has
  // been closed AND the stage has let go -- which, since a stage may
  // keep a reference past its reply, is only knowable through `owner`.
  // For an OUTBOUND buffer it is the stage's, and holding it keeps the
  // memory ALLOCATED past close() -- never its contents unchanged.
  std::shared_ptr<void> owner;

  // True when `data` is set and the layout's extent fits in `size`.
  bool valid() const noexcept;

  // A view of `data` laid out as `layout`, sized to the layout's
  // extent, read-only, kept alive by `owner`.
  static DataBuffer view(std::string name, const void* data,
                         BufferLayout layout,
                         std::shared_ptr<void> owner = {});
};

// Where a command is. Pending -> Active -> Replied -> Closed is the
// ordinary life; Failed and Cancelled are its other ends.
//
//   Pending    queued for the stage, not yet taken.
//   Active     the stage has taken it and not answered.
//   Replied    answered: result_json() and buffers() are readable, and
//              the stage is waiting for close() if it holds anything.
//   Failed     refused -- by the host before it reached the stage (an
//              unknown command, bad arguments, a buffer that does not
//              match the spec, a stage that is not running) or by the
//              stage itself. error() says which.
//   Cancelled  the pipeline stopped, or the stage reached the end of
//              its stream, before the command was ANSWERED. An answer,
//              once given, is final: a stop while the caller still
//              reads a reply leaves it Replied -- its buffers keep
//              their memory -- and only releases the stage from
//              holding it, so their contents are no longer promised.
//   Closed     the caller closed it. Terminal from the caller's side.
enum class CommandState : std::uint8_t {
  Invalid = 0,
  Pending,
  Active,
  Replied,
  Failed,
  Cancelled,
  Closed,
};

// "invalid", "pending", "active", "replied", "failed", "cancelled",
// "closed". Never null.
std::string_view command_state_name(CommandState) noexcept;

class StageCommand;   // library-internal

// The caller's side of one command. Returned by StageHandle::command();
// see vpipe/pipeline-handle.h.
class CommandHandle final {
public:
  CommandHandle() noexcept = default;
  ~CommandHandle();
  CommandHandle(CommandHandle&&) noexcept;
  CommandHandle& operator=(CommandHandle&&) noexcept;
  CommandHandle(const CommandHandle&)            = delete;
  CommandHandle& operator=(const CommandHandle&) = delete;

  // True when this handle refers to a command -- including one that
  // was refused, whose state() is Failed.
  bool valid() const noexcept { return static_cast<bool>(_cmd); }
  explicit operator bool() const noexcept { return valid(); }

  // The stage-local sequence number the command was queued under
  // (1-based); 0 for a command refused before it was queued.
  std::uint64_t id() const noexcept;

  CommandState state() const;

  // Block until the command is answered, refused, cancelled or closed,
  // or `timeout_ms` passes (negative = forever, 0 = probe). Returns the
  // state at that moment, so Pending / Active means the wait timed out.
  CommandState wait(int timeout_ms = -1) const;

  // True once the stage has replied, and nothing has cancelled it.
  bool ok() const;

  // The stage's JSON result ("{}" when it sent none, "null" before a
  // reply), and the reason for a Failed / Cancelled state ("" else).
  std::string result_json() const;
  std::string error() const;

  // The stage's buffers, readable in place from the reply until
  // close(). Each copy shares the stage's owner. Empty before a reply
  // and after close().
  std::vector<DataBuffer> buffers() const;
  bool buffer(std::string_view name, DataBuffer* out) const;

  // Tell the stage the caller is done: it may release or reuse what it
  // exposed, and a stage holding its output resumes. Idempotent.
  void close();

private:
  friend class HandleAccess;
  explicit CommandHandle(std::shared_ptr<StageCommand> cmd) noexcept;

  std::shared_ptr<StageCommand> _cmd;
};

}

VPIPE_API_END

#endif
