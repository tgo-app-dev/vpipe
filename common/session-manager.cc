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
    auto uptr = make_unique<Session>(cfg);
    const SessionIntf* cptr = uptr.get();
    _s.insert({ cptr, std::move(uptr) });
    return cptr;
  };
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
  unordered_map<const SessionIntf*, unique_ptr<Session>> _s;
};

SessionManager& SessionManager::get()
{
  static SessionManagerImpl m;
  return m;
}


}

