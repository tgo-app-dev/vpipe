// Python bindings for the vpipe public API. Kept deliberately thin:
// this file mirrors the C++ surface and lets the Python-side
// __init__.py decide what high-level conveniences to layer on top
// (default-session creation, config resolution, etc.).

#include "vpipe/pipeline-handle.h"
#include "vpipe/session-intf.h"
#include "vpipe/session-manager.h"
#include "vpipe/stage-command.h"
#include "vpipe/status.h"
#include "vpipe/vpipe.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace vpipe;

namespace {

// ---- stage commands: buffers across the language boundary -------------
//
// Data crosses by REFERENCE both ways. A caller's array is handed to the
// stage as a view that keeps the Python object alive; a stage's reply
// comes back as an ndarray over the stage's own bytes, whose owner keeps
// that memory alive. Neither direction copies.

using InArray = nb::ndarray<nb::ro, nb::device::cpu>;

// A Python array must be released under the GIL, but the LAST reference
// to it may be dropped on one of vpipe's worker threads -- a stage that
// kept a caller's buffer past its reply. Taking the GIL there could
// deadlock against a Python thread blocked in a call that holds it, so a
// release off the GIL is QUEUED and run by the interpreter itself
// (Py_AddPendingCall, which needs no GIL), or by the next binding call.
std::mutex            g_defer_mu;
std::vector<InArray*> g_defer;
bool                  g_defer_armed = false;

int
drain_deferred_(void*)
{
  std::vector<InArray*> v;
  {
    std::lock_guard<std::mutex> lk(g_defer_mu);
    v.swap(g_defer);
    g_defer_armed = false;
  }
  for (InArray* p : v) { delete p; }
  return 0;
}

bool
finalizing_()
{
#if PY_VERSION_HEX >= 0x030D0000
  return Py_IsFinalizing() != 0;
#else
  return _Py_IsFinalizing() != 0;
#endif
}

void
release_array_(InArray* p)
{
  // At interpreter shutdown there is no one left to release it to.
  if (!Py_IsInitialized() || finalizing_()) { return; }
  if (PyGILState_Check()) {
    delete p;
    return;
  }
  bool arm = false;
  {
    std::lock_guard<std::mutex> lk(g_defer_mu);
    g_defer.push_back(p);
    if (!g_defer_armed) { g_defer_armed = arm = true; }
  }
  if (arm && Py_AddPendingCall(&drain_deferred_, nullptr) != 0) {
    // The interpreter's queue is full; the next binding call drains.
    std::lock_guard<std::mutex> lk(g_defer_mu);
    g_defer_armed = false;
  }
}

bool
element_of_(nb::dlpack::dtype d, ElementType* out)
{
  using nb::dlpack::dtype_code;
  if (d.lanes != 1) { return false; }
  switch (static_cast<dtype_code>(d.code)) {
    case dtype_code::UInt:
      switch (d.bits) {
        case 8:  *out = ElementType::U8;  return true;
        case 16: *out = ElementType::U16; return true;
        case 32: *out = ElementType::U32; return true;
        case 64: *out = ElementType::U64; return true;
        default: return false;
      }
    case dtype_code::Int:
      switch (d.bits) {
        case 8:  *out = ElementType::I8;  return true;
        case 16: *out = ElementType::I16; return true;
        case 32: *out = ElementType::I32; return true;
        case 64: *out = ElementType::I64; return true;
        default: return false;
      }
    case dtype_code::Float:
      switch (d.bits) {
        case 16: *out = ElementType::F16; return true;
        case 32: *out = ElementType::F32; return true;
        case 64: *out = ElementType::F64; return true;
        default: return false;
      }
    case dtype_code::Bfloat:
      if (d.bits == 16) { *out = ElementType::BF16; return true; }
      return false;
    default:
      return false;
  }
}

nb::dlpack::dtype
dlpack_of_(ElementType t)
{
  using nb::dlpack::dtype_code;
  auto dt = [](dtype_code c, int bits) {
    nb::dlpack::dtype d;
    d.code  = static_cast<std::uint8_t>(c);
    d.bits  = static_cast<std::uint8_t>(bits);
    d.lanes = 1;
    return d;
  };
  switch (t) {
    case ElementType::Bytes:
    case ElementType::U8:   return dt(dtype_code::UInt, 8);
    case ElementType::I8:   return dt(dtype_code::Int, 8);
    case ElementType::U16:  return dt(dtype_code::UInt, 16);
    case ElementType::I16:  return dt(dtype_code::Int, 16);
    case ElementType::U32:  return dt(dtype_code::UInt, 32);
    case ElementType::I32:  return dt(dtype_code::Int, 32);
    case ElementType::U64:  return dt(dtype_code::UInt, 64);
    case ElementType::I64:  return dt(dtype_code::Int, 64);
    case ElementType::F16:  return dt(dtype_code::Float, 16);
    case ElementType::BF16: return dt(dtype_code::Bfloat, 16);
    case ElementType::F32:  return dt(dtype_code::Float, 32);
    case ElementType::F64:  return dt(dtype_code::Float, 64);
  }
  return dt(dtype_code::UInt, 8);
}

// Anything nanobind imports as a CPU array -- a numpy / torch / jax
// array, bytes, bytearray, memoryview, array.array, any DLPack or
// buffer-protocol object -- as a view the stage may keep.
DataBuffer
buffer_from_py_(const std::string& name, nb::handle obj)
{
  InArray a;
  if (!nb::try_cast<InArray>(obj, a)) {
    throw nb::type_error(("buffer '" + name + "': expected an array, "
                          "bytes, or a buffer-protocol / DLPack object")
                             .c_str());
  }
  DataBuffer b;
  b.name = name;
  if (!element_of_(a.dtype(), &b.layout.type)) {
    throw nb::type_error(("buffer '" + name + "': unsupported element type")
                             .c_str());
  }
  const auto es = static_cast<std::int64_t>(element_size(b.layout.type));
  for (std::size_t i = 0; i < a.ndim(); ++i) {
    b.layout.shape.push_back(static_cast<std::int64_t>(a.shape(i)));
    b.layout.strides.push_back(a.stride(i) * es);   // elements -> bytes
  }
  if (b.layout.is_contiguous()) { b.layout.strides.clear(); }
  b.data = const_cast<void*>(a.data());
  b.size = b.layout.extent_bytes();
  auto* keep = new InArray(std::move(a));
  b.owner = std::shared_ptr<void>(keep, [](void* p) {
    release_array_(static_cast<InArray*>(p));
  });
  return b;
}

std::vector<DataBuffer>
buffers_from_py_(nb::object buffers)
{
  std::vector<DataBuffer> in;
  if (buffers.is_none()) { return in; }
  if (!nb::isinstance<nb::dict>(buffers)) {
    throw nb::type_error("buffers must be a dict of name -> array");
  }
  for (auto [k, v] : nb::borrow<nb::dict>(buffers)) {
    in.push_back(buffer_from_py_(nb::cast<std::string>(k), v));
  }
  return in;
}

// A stage's buffer as an ndarray over the stage's own bytes. Read-only
// unless the channel declares it writable. The array holds the buffer's
// owner, so the MEMORY outlives close(); the contents are the stage's
// again from then on.
nb::object
array_of_(const DataBuffer& b)
{
  const std::int64_t es =
      static_cast<std::int64_t>(element_size(b.layout.type));
  std::vector<std::size_t> shape(b.layout.shape.begin(),
                                 b.layout.shape.end());
  std::vector<std::int64_t> strides;
  for (std::int64_t s : b.layout.strides.empty()
                            ? b.layout.contiguous_strides()
                            : b.layout.strides) {
    if (s % es != 0) {
      throw nb::value_error(("buffer '" + b.name + "': a stride is not a "
                             "whole number of elements").c_str());
    }
    strides.push_back(s / es);
  }
  auto* keep = new std::shared_ptr<void>(b.owner);
  nb::capsule owner(keep, [](void* p) noexcept {
    delete static_cast<std::shared_ptr<void>*>(p);
  });
  // array_api: an object with the buffer protocol (memoryview(),
  // np.asarray()) AND __dlpack__ (np / torch / jax from_dlpack()).
  if (b.writable) {
    nb::ndarray<nb::array_api, nb::device::cpu> arr(
        b.data, shape.size(), shape.data(), owner, strides.data(),
        dlpack_of_(b.layout.type));
    return nb::cast(arr, nb::rv_policy::reference);
  }
  nb::ndarray<nb::array_api, nb::ro, nb::device::cpu> arr(
      b.data, shape.size(), shape.data(), owner, strides.data(),
      dlpack_of_(b.layout.type));
  return nb::cast(arr, nb::rv_policy::reference);
}

std::string
json_of_(nb::object v)
{
  if (v.is_none()) { return {}; }
  if (nb::isinstance<nb::str>(v)) { return nb::cast<std::string>(v); }
  nb::object dumps = nb::module_::import_("json").attr("dumps");
  return nb::cast<std::string>(dumps(v));
}

nb::object
from_json_(const std::string& s)
{
  nb::object loads = nb::module_::import_("json").attr("loads");
  return loads(nb::cast(s));
}

// Wait in short GIL-free slices so a Ctrl-C still lands, the same
// shape as SessionIntf.wait_pipelines below.
CommandState
wait_command_(const CommandHandle& c, int timeout_ms, int poll_ms)
{
  if (poll_ms <= 0) { poll_ms = 100; }
  int remaining = timeout_ms;
  for (;;) {
    const int slice =
        remaining < 0 ? poll_ms : (remaining < poll_ms ? remaining : poll_ms);
    CommandState st;
    {
      nb::gil_scoped_release rel;
      st = c.wait(slice);
    }
    drain_deferred_(nullptr);
    if (st != CommandState::Pending && st != CommandState::Active) {
      return st;
    }
    if (PyErr_CheckSignals() != 0) { throw nb::python_error(); }
    if (timeout_ms >= 0) {
      remaining -= slice;
      if (remaining <= 0) { return st; }
    }
  }
}

}  // namespace

NB_MODULE(_vpipe, m)
{
  m.doc() = "vpipe core bindings (internal). See the vpipe package "
            "for the user-facing API.";

  m.def("vpipe_version", &vpipe_version);

  nb::class_<Status>(m, "Status")
    .def(nb::init<>())
    .def_rw("code", &Status::code)
    .def("__bool__",
         [](const Status& s) { return s.code == 0; })
    .def("__repr__",
         [](const Status& s) {
           return std::string("Status(code=") +
                  std::to_string(s.code) + ")";
         });

  // StageHandle is a value type. Like PipelineHandle, the impl is
  // owned by the session-side PipelineHandleImpl; the handle is
  // valid for the pipeline's lifetime.
  nb::class_<StageHandle>(m, "StageHandle")
    .def("__bool__",
         [](const StageHandle& h) { return h.valid(); })
    .def("__repr__",
         [](const StageHandle& h) {
           return h.valid()
             ? std::string("StageHandle(<live>)")
             : std::string("StageHandle(<null>)");
         })
    .def("num_oports", &StageHandle::num_oports)
    // config_schema() returns the stage's configuration descriptor as
    // a Python list of dicts (key/type/required/doc/default/current),
    // parsed from the C++ JSON via json.loads so callers get native
    // Python objects -- the mirror image of insert_stage's dict config.
    .def("config_schema",
         [](const StageHandle& h) {
           std::string js = h.config_schema_json();
           nb::object loads =
               nb::module_::import_("json").attr("loads");
           return loads(nb::cast(js));
         })
    .def_prop_ro("id", &StageHandle::id)
    // The command channels the stage declares, as a list of dicts; see
    // StageHandle::commands_json and docs/STAGE-COMMANDS.md.
    .def("commands",
         [](const StageHandle& h) { return from_json_(h.commands_json()); })
    // Send a command. `args` is a dict (or JSON string); `buffers` maps
    // a buffer name to anything array-like, passed by reference -- the
    // stage reads the caller's memory in place. Returns at once; a
    // refusal comes back as a CommandHandle in state "failed".
    .def("command",
         [](const StageHandle& h, std::string_view name, nb::object args,
            nb::object buffers) {
           drain_deferred_(nullptr);
           std::string aj = json_of_(args);
           std::vector<DataBuffer> in = buffers_from_py_(buffers);
           return h.command(name, aj, std::move(in));
         },
         nb::arg("name"), nb::arg("args") = nb::none(),
         nb::arg("buffers") = nb::none())
    // command + wait + close, for a command whose answer is its result:
    // returns the result dict, raises RuntimeError with the stage's
    // reason on anything but a reply (TimeoutError when it times out).
    .def("call",
         [](const StageHandle& h, std::string_view name, nb::object args,
            nb::object buffers, int timeout_ms) {
           drain_deferred_(nullptr);
           std::string aj = json_of_(args);
           std::vector<DataBuffer> in = buffers_from_py_(buffers);
           CommandHandle c = h.command(name, aj, std::move(in));
           const CommandState st = wait_command_(c, timeout_ms, 100);
           if (st == CommandState::Replied) {
             nb::object r = from_json_(c.result_json());
             c.close();
             return r;
           }
           const std::string why = c.error();
           c.close();
           if (st == CommandState::Pending || st == CommandState::Active) {
             PyErr_SetString(PyExc_TimeoutError,
                             ("command '" + std::string(name) +
                              "' timed out").c_str());
             throw nb::python_error();
           }
           throw std::runtime_error("command '" + std::string(name) +
                                    "' " +
                                    std::string(command_state_name(st)) +
                                    (why.empty() ? "" : ": " + why));
         },
         nb::arg("name"), nb::arg("args") = nb::none(),
         nb::arg("buffers") = nb::none(), nb::arg("timeout_ms") = -1);

  // One command in flight. Close it (or leave the `with` block, or drop
  // it) when done with its buffers: a stage that HOLDS keeps the
  // pipeline downstream of it paused until then.
  nb::class_<CommandHandle>(m, "CommandHandle")
    .def("__bool__", [](const CommandHandle& c) { return c.valid(); })
    .def("__repr__",
         [](const CommandHandle& c) {
           return "CommandHandle(" + std::to_string(c.id()) + ", " +
                  std::string(command_state_name(c.state())) + ")";
         })
    .def_prop_ro("id", &CommandHandle::id)
    .def_prop_ro("state",
                 [](const CommandHandle& c) {
                   return std::string(command_state_name(c.state()));
                 })
    .def_prop_ro("ok", &CommandHandle::ok)
    .def_prop_ro("error", &CommandHandle::error)
    .def_prop_ro("result",
                 [](const CommandHandle& c) {
                   return from_json_(c.result_json());
                 })
    // name -> ndarray over the stage's bytes (zero-copy). Numpy takes
    // them with np.asarray(); anything DLPack-aware with from_dlpack().
    .def_prop_ro("buffers",
                 [](const CommandHandle& c) {
                   nb::dict d;
                   for (const DataBuffer& b : c.buffers()) {
                     d[nb::str(b.name.c_str())] = array_of_(b);
                   }
                   return d;
                 })
    // Block until answered / refused / cancelled, or `timeout_ms`
    // (negative = forever); returns the state name. Ctrl-C interrupts.
    .def("wait",
         [](const CommandHandle& c, int timeout_ms, int poll_ms) {
           return std::string(command_state_name(
               wait_command_(c, timeout_ms, poll_ms)));
         },
         nb::arg("timeout_ms") = -1, nb::arg("poll_ms") = 100)
    .def("close",
         [](CommandHandle& c) {
           {
             nb::gil_scoped_release rel;
             c.close();
           }
           drain_deferred_(nullptr);
         })
    .def("__enter__", [](CommandHandle& c) -> CommandHandle& { return c; },
         nb::rv_policy::reference)
    .def("__exit__",
         [](CommandHandle& c, nb::handle, nb::handle, nb::handle) {
           {
             nb::gil_scoped_release rel;
             c.close();
           }
           drain_deferred_(nullptr);
         },
         nb::arg("exc_type").none(), nb::arg("exc").none(),
         nb::arg("tb").none());

  // PipelineHandle is a value type wrapping an opaque impl pointer.
  // We bind it without a constructor -- handles only ever come from
  // SessionIntf::{load,create}_pipeline.
  //
  // insert_stage accepts iports as a list of (StageHandle, oport)
  // pairs, and config as either a JSON string or a Python dict (or
  // None). When given a dict we serialize it through json.dumps and
  // pass the resulting JSON to the C++ side. This lets Python users
  // write `cfg={"period_seconds": 1, "count": 5}` directly without
  // ever touching the JSON layer.
  nb::class_<PipelineHandle>(m, "PipelineHandle")
    .def("__bool__",
         [](const PipelineHandle& h) { return h.valid(); })
    .def("__repr__",
         [](const PipelineHandle& h) {
           return h.valid()
             ? std::string("PipelineHandle(<live>)")
             : std::string("PipelineHandle(<null>)");
         })
    .def("insert_stage",
         [](PipelineHandle& self,
            std::string type,
            std::string id,
            std::vector<std::tuple<StageHandle, unsigned>> iports_in,
            nb::object config) {
           std::string config_json;
           if (!config.is_none()) {
             if (nb::isinstance<nb::str>(config)) {
               config_json = nb::cast<std::string>(config);
             } else {
               // json.dumps(dict_or_list) -> str
               nb::object dumps =
                   nb::module_::import_("json").attr("dumps");
               config_json =
                   nb::cast<std::string>(dumps(config));
             }
           }
           std::vector<StagePortHandle> iports;
           iports.reserve(iports_in.size());
           for (auto& [stage, port] : iports_in) {
             iports.push_back(StagePortHandle{stage, port});
           }
           return self.insert_stage(std::move(type),
                                    std::move(id),
                                    std::move(iports),
                                    std::move(config_json));
         },
         nb::arg("type"),
         nb::arg("id"),
         nb::arg("iports") =
             std::vector<std::tuple<StageHandle, unsigned>>{},
         nb::arg("config") = nb::none())
    .def("insert_pipeline",
         [](PipelineHandle& self, std::string id) {
           return self.insert_pipeline(std::move(id));
         },
         nb::arg("id"))
    // A stage by id -- how a LOADED pipeline's stages are reached.
    // "sub/stage" reaches into a nested pipeline. Null handle if none.
    .def("stage",
         [](PipelineHandle& self, std::string_view id) {
           return self.stage(id);
         },
         nb::arg("id"));

  // Abstract base; never constructed from Python. Sessions are owned
  // by the SessionManager and surfaced as raw references.
  nb::class_<SessionIntf>(m, "SessionIntf")
    .def("load_pipeline",
         [](SessionIntf& self, std::string_view path) {
           return self.load_pipeline(path);
         },
         nb::arg("path"))
    .def("create_pipeline",
         [](SessionIntf& self, std::string id) {
           return self.create_pipeline(std::move(id));
         },
         nb::arg("id"))
    .def("launch_pipeline", &SessionIntf::launch_pipeline,
         nb::arg("handle"))
    .def("pause_pipeline",  &SessionIntf::pause_pipeline,
         nb::arg("handle"))
    // stop_pipeline / unload_pipeline both block until every stage's
    // driver coroutine has reached final_suspend. During that wait we
    // hold no Python state, so release the GIL: it lets the C++
    // logging delegates run (some sinks need to acquire the GIL on
    // dispatch from worker threads), and -- more importantly -- it
    // keeps the interpreter responsive to a follow-up Ctrl-C from a
    // user who is impatient with a stuck shutdown. Without the
    // release, holding the GIL while wait_idle blocks would cause
    // any Python sink dispatch from a worker thread to deadlock the
    // shutdown.
    .def("stop_pipeline",
         [](SessionIntf& self, PipelineHandle h) {
           nb::gil_scoped_release rel;
           return self.stop_pipeline(h);
         },
         nb::arg("handle"))
    .def("unload_pipeline",
         [](SessionIntf& self, PipelineHandle h) {
           nb::gil_scoped_release rel;
           return self.unload_pipeline(h);
         },
         nb::arg("handle"))
    // wait_pipelines: block until every launched pipeline finishes
    // (one-shot stages get to complete naturally). The default
    // call blocks forever, but Python signals must still be
    // serviced -- a Ctrl-C from the shell should raise
    // KeyboardInterrupt promptly even though the C++ side is
    // sitting on a condvar. We therefore implement the Python
    // wrapper as a polling loop: release the GIL, take a short
    // bounded wait (default 100 ms), re-acquire the GIL, run
    // PyErr_CheckSignals() to deliver any pending signal handler
    // (which may raise KeyboardInterrupt), and loop until the C++
    // wait returns Status{0} or the caller's total timeout
    // elapses. `poll_ms` is the per-iteration C++ wait granularity
    // and is purely a tuning knob (smaller -> snappier Ctrl-C
    // response, more wakeups).
    .def("wait_pipelines",
         [](SessionIntf& self, int timeout_ms, int poll_ms) {
           if (poll_ms <= 0) {
             poll_ms = 100;
           }
           auto cap = [&](int budget) {
             if (budget < 0) { return poll_ms; }
             return budget < poll_ms ? budget : poll_ms;
           };
           int remaining = timeout_ms;
           while (true) {
             Status s;
             {
               nb::gil_scoped_release rel;
               s = self.wait_pipelines(cap(remaining));
             }
             if (s.code == 0) {
               return s;
             }
             if (PyErr_CheckSignals() != 0) {
               throw nb::python_error();
             }
             if (timeout_ms >= 0) {
               remaining -= poll_ms;
               if (remaining <= 0) {
                 return s;       // Status{4} -- timeout
               }
             }
           }
         },
         nb::arg("timeout_ms") = -1,
         nb::arg("poll_ms")    = 100)
    // store_pipeline has two overloads: one with just the handle
    // (uses the path remembered from a prior load_pipeline / a
    // path-taking store) and one that takes a path.
    .def("store_pipeline",
         [](SessionIntf& self, PipelineHandle h) {
           return self.store_pipeline(h);
         },
         nb::arg("handle"))
    .def("store_pipeline",
         [](SessionIntf& self, PipelineHandle h,
            std::string_view path) {
           return self.store_pipeline(h, path);
         },
         nb::arg("handle"), nb::arg("path"))
    // debug_level is overloaded on (unsigned) and (string_view).
    // Bind both explicitly so Python sees one method that accepts
    // either an int or a str. We dispatch on Python type since
    // nanobind picks overloads in registration order.
    .def("debug_level",
         [](SessionIntf& self, unsigned u) {
           return self.debug_level(u);
         },
         nb::arg("level"))
    .def("debug_level",
         [](SessionIntf& self, std::string_view name) {
           return self.debug_level(name);
         },
         nb::arg("level"))
    .def("log_to_stdout", &SessionIntf::log_to_stdout)
    .def("log_to_db",     &SessionIntf::log_to_db)
    .def("enable_profiling",
         [](SessionIntf& self, unsigned n) {
           return self.enable_profiling(n);
         },
         nb::arg("max_events_per_stage"))
    .def("disable_profiling", &SessionIntf::disable_profiling)
    .def("dump_profiling",
         [](SessionIntf& self, std::string_view path) {
           return self.dump_profiling(path);
         },
         nb::arg("path"));

  // Singleton with a protected dtor; bind without any constructor and
  // only ever return references.
  nb::class_<SessionManager>(m, "SessionManager")
    .def_static("get",
                &SessionManager::get,
                nb::rv_policy::reference)
    .def("create_session",
         [](SessionManager& self, std::string_view cfg) {
           // Cast away const: the Python-facing SessionIntf binding
           // exposes mutating methods, and the underlying object is
           // not actually const -- create_session returns
           // 'const SessionIntf*' purely as a pointer-stability hint.
           return const_cast<SessionIntf*>(self.create_session(cfg));
         },
         nb::arg("config") = std::string_view(""),
         nb::rv_policy::reference)
    .def("destroy_session",
         [](SessionManager& self, SessionIntf* s) {
           self.destroy_session(s);
         },
         nb::arg("session"))
    .def("num_sessions", &SessionManager::num_sessions);
}
