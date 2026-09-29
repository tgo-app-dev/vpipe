#ifndef SESSION_MEMBER_H
#define SESSION_MEMBER_H

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {

class SessionContextIntf;

class SessionMember {
public:
  SessionMember(const SessionContextIntf*);
  virtual ~SessionMember() = default;

  const SessionContextIntf* session() const;

private:
  const SessionContextIntf* _session;
};

}

VPIPE_API_END

#endif

