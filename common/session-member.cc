#include "common/session-member.h"
#include "interfaces/session-context-intf.h"

namespace vpipe {

SessionMember::SessionMember(const SessionContextIntf* s)
  : _session(s)
{
}

const SessionContextIntf*
SessionMember::session() const
{
  return _session;
}

}

