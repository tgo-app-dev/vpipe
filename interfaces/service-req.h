// service-req.h -- one stage's stated dependency on a session service.
//
// Deliberately knows nothing about any particular service: just an id
// and whether the stage can run without it. That is what lets
// pipeline/stage.h declare the hook without pulling the service types
// back into the 74 translation units that stopped seeing them when the
// subsystems moved behind SessionContextIntf::services().
//
// The typed constructors -- require_service<T>() / optional_service<T>()
// -- live in session-services-intf.h, next to ServiceId. A stage that
// names a service already includes that header, because it is about to
// consume the thing.

#ifndef SERVICE_REQ_H
#define SERVICE_REQ_H

#include <string_view>

namespace vpipe {

struct ServiceReq {
  // The service's stable id (ServiceId<T>::value).
  std::string_view id;
  // Required: the pipeline refuses to launch without it, naming the
  // stage. Optional: the stage is expected to cope with absence, and
  // says so here rather than leaving the reader to infer it from a
  // null check buried in initialize().
  bool             required = true;
};

}

#endif
