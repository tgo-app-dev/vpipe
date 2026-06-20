// Python bindings for the vpipe public API. Kept deliberately thin:
// this file mirrors the C++ surface and lets the Python-side
// __init__.py decide what high-level conveniences to layer on top
// (default-session creation, config resolution, etc.).

#include "vpipe/pipeline-handle.h"
#include "vpipe/session-intf.h"
#include "vpipe/session-manager.h"
#include "vpipe/status.h"
#include "vpipe/vpipe.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace vpipe;

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
         });

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
