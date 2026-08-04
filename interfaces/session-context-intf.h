#ifndef SESSION_CONTEXT_INTF_H
#define SESSION_CONTEXT_INTF_H

#include "interfaces/fs-sandbox-intf.h"
#include "interfaces/log-sink-intf.h"
#include "interfaces/perf-sink-intf.h"
#include "interfaces/runtime-services-intf.h"
#include "interfaces/ui-port-intf.h"

namespace vpipe {

class SessionServicesIntf;

// Resources every SessionMember (Vertex, Stage, EdgeBuffer, ...)
// reaches for during normal operation. Held by pointer through
// SessionMember::session().
//
// This is a COMPOSITION of roles, not a list of methods. Each base is a
// separate, independently usable interface:
//
//   LogSinkIntf         error / warn / info / log_*
//   UiPortIntf          getline, text streams, interrupts, i18n
//   RuntimeServicesIntf thread pool, default edge capacity
//   PerfSinkIntf        profiling toggles + the record_perf_event path
//   FsSandboxIntf       confine_path and the sandbox's shape
//
// Take the NARROWEST one that does the job. A helper that only reports
// should accept a LogSinkIntf&, not a whole session context -- that is
// most of them, since roughly 90% of all calls made through this
// interface are logging. Everything that already takes a
// SessionContextIntf* keeps working; inheriting the roles changes no
// call site.
//
// The heavyweight subsystems (FFmpeg, LMDB, CoreML, the generative-model
// manager, metal-compute) are deliberately NOT here. They used to be,
// which put five subsystem names in the lowest interface layer in the
// tree and made every logging-only translation unit carry them. They
// now live behind services(); see interfaces/session-services-intf.h.
class SessionContextIntf : public LogSinkIntf,
                           public UiPortIntf,
                           public RuntimeServicesIntf,
                           public PerfSinkIntf,
                           public FsSandboxIntf {
public:
  SessionContextIntf() {};
  ~SessionContextIntf() override = default;

  // The session's heavyweight subsystems. NEVER null: a context that
  // owns none inherits an inert instance whose accessors all answer
  // nullptr, which is the same answer they gave when they hung off this
  // interface directly. So `services()->metal_compute()` needs no null
  // check on the services hop -- only on the subsystem, exactly as
  // before.
  virtual SessionServicesIntf* services() const;
};

}

#endif
