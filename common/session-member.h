#ifndef SESSION_MEMBER_H
#define SESSION_MEMBER_H

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

#endif

