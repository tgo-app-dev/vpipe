// runtime-services-intf.h -- the pipeline runtime's own knobs.
//
// The two things a SessionMember needs from the session in order to
// RUN, as opposed to in order to report or to reach a subsystem: the
// worker pool that drives stage drivers, and the default edge-buffer
// capacity. Both are pure -- unlike the other roles there is no
// sensible inert answer for a thread pool.

#ifndef RUNTIME_SERVICES_INTF_H
#define RUNTIME_SERVICES_INTF_H

namespace vpipe {

class ThreadPool;

class RuntimeServicesIntf {
public:
  virtual ~RuntimeServicesIntf() = default;

  // The shared worker pool that drives stage drivers and wakes
  // suspended awaiters. Always non-null on a fully-constructed
  // Session. Members schedule resumed coroutines through it.
  virtual ThreadPool* thread_pool() const noexcept = 0;
};

}

#endif
