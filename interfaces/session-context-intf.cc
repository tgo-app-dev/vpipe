#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

namespace vpipe {

SessionServicesIntf*
SessionContextIntf::services() const
{
  // A context that owns no subsystems still answers, so callers never
  // null-check the services hop itself.
  return null_session_services();
}

}
