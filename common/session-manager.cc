#include "vpipe/session-manager.h"
#include "common/session.h"
#include <memory>
#include <unordered_map>

using namespace std;

namespace vpipe {

class SessionManagerImpl final : public SessionManager
{
public:
  SessionManagerImpl() {};
  ~SessionManagerImpl() = default;
  const SessionIntf* create_session(string_view cfg)
  {
    return adopt_(make_unique<Session>(cfg));
  };
  const SessionIntf* create_session_from(const FlexData& cfg)
  {
    return adopt_(make_unique<Session>(cfg));
  }
  void destroy_session(const SessionIntf* cptr)
  {
    auto it = _s.find(cptr);
    if (it != _s.end()) {
      _s.erase(it);
    }
  };
  unsigned num_sessions() const
  {
    return _s.size();
  };
private:
  const SessionIntf* adopt_(unique_ptr<Session> uptr)
  {
    const SessionIntf* cptr = uptr.get();
    _s.insert({ cptr, std::move(uptr) });
    return cptr;
  }

  unordered_map<const SessionIntf*, unique_ptr<Session>> _s;
};

// Non-virtual (session-manager.h): every SessionManager is the one below.
const SessionIntf*
SessionManager::create_session(const FlexData& config)
{
  return static_cast<SessionManagerImpl&>(*this).create_session_from(config);
}

SessionManager& SessionManager::get()
{
  static SessionManagerImpl m;
  return m;
}


}

