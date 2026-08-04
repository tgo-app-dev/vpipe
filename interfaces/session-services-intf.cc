#include "interfaces/session-services-intf.h"

namespace vpipe {

SessionServicesIntf*
null_session_services() noexcept
{
  // Function-local static: no ordering hazard against a caller running
  // from another translation unit's static init.
  static SessionServicesIntf inert;
  return &inert;
}

}
