#ifndef VPIPE_GENERATIVE_MODELS_SHARED_TP_OVERLAP_H
#define VPIPE_GENERATIVE_MODELS_SHARED_TP_OVERLAP_H

// A RANK'S EXCHANGES, OFF THE ENCODING THREAD (tensor-parallel.h). A DiT
// split across hosts hands a partial sum to its channel twice a block.
// Done in line, the encoder commits, waits for the GPU, waits for the
// link and starts again: the GPU idles for every exchange. Here the
// encoder never waits. It POSTS an exchange and encodes on; the GPU
// signals `ready` once the partial is written; a worker thread hands it
// to the channel and sets `done` when the others' sum is back; the GPU
// work that adds the sum waits on `done`. What is encoded between the
// signal and the wait -- the next band's attention, the band before's
// feed-forward -- runs while the bytes cross.
//
// READY IS SIGNALED PER LANE -- a partial buffer -- each lane counting
// its own uses. One counter for every partial would let a later signal
// stand for an earlier partial: GPU work with no buffer in common runs
// in any order, so a band's fc2, signaled after a band's out projection
// was posted, can finish first, and the worker, told "ready past n",
// would send an out projection not yet written (seen: two ranks of one
// clip parting ways, then a GPU page fault). A lane's uses are ordered
// by its buffer: each write follows the last read of the one before.
// DONE counts exchanges, the n-th posted done at n: the worker takes
// them in the order posted -- every rank posts alike, so each channel
// sees one sequence -- and answers each before the next. A wait the GPU
// is given must follow the signal of its own exchange and of every one
// posted before it, or the GPU waits on a worker that waits on the
// GPU.
//
// A failed exchange releases every wait (done set past anything ever
// posted) and keeps its reason: the forward runs out its GPU work on
// wrong sums and fails when it reads failed(). So does one whose channel
// THREW: the channel is a plugin's, and an exception off the worker
// would end the process.
//
// Every call into the channel -- exchange() and forget() -- goes through
// here, one at a time (tensor-parallel.h's contract).

#include "apple-silicon/metal-compute/event.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/tensor-parallel.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class TpOverlap {
public:
  // `lanes`: the partial buffers the caller posts from, each with a
  // ready() event of its own; `dtype` their elements'.
  TpOverlap(metal_compute::MetalCompute* mc, std::shared_ptr<TpChannel> ch,
            int lanes, TpDType dtype = TpDType::kBF16);
  // Abandons what is pending and joins the worker.
  ~TpOverlap();
  TpOverlap(const TpOverlap&)            = delete;
  TpOverlap& operator=(const TpOverlap&) = delete;

  // Both events made and the worker running.
  bool valid() const { return _ok && _th.joinable(); }

  // Queue an exchange of `bytes` from `partial` into `others` -- lane
  // `lane`'s buffers -- `seq` the channel's. The buffers are the
  // caller's to keep until the GPU has waited for this exchange. The
  // GPU signals ticket.ready to ready_at once `partial` is written, in
  // the command buffer that wrote it, and waits done() for done_at
  // before it reads `others`.
  struct Ticket {
    const metal_compute::Event* ready = nullptr;
    std::uint64_t ready_at = 0;
    std::uint64_t done_at  = 0;
  };
  Ticket post(int lane, const metal_compute::SharedBuffer& partial,
              const metal_compute::SharedBuffer& others, std::size_t bytes,
              std::uint64_t seq);

  // An exchange in line, the GPU drained by the caller: waits for what
  // was posted to finish first.
  bool exchange_now(const void* partial, void* others, std::size_t bytes,
                    std::uint64_t seq, std::string* err);

  // An exchange buffer about to be freed or replaced (TpChannel::forget):
  // told to the channel once what was posted has been answered, never
  // beside an exchange.
  void forget(const metal_compute::SharedBuffer& b);

  const metal_compute::Event& done() const { return _done; }

  bool failed() const { return _failed.load(std::memory_order_acquire); }
  std::string error() const;

  // Exchanges posted and not yet answered.
  bool pending() const
  {
    return _finished.load(std::memory_order_acquire) <
           _posted.load(std::memory_order_acquire);
  }

  // Give up on what is pending: the worker stops waiting for the GPU and
  // every GPU wait is released, now and from then on. For a forward that
  // ends early, whose later partials were never committed. Fails the
  // overlap: an exchange skipped leaves the ranks out of step.
  void abandon(const std::string& why);

  // What the link did since the last call: exchanges, bytes, and the
  // worker's time in the channel against its time waiting for the GPU
  // (the link idle). Reset by the call.
  struct Stats {
    std::uint64_t exchanges = 0;
    std::uint64_t bytes = 0;
    double link_ms = 0.0;
    double idle_ms = 0.0;
  };
  Stats take_stats();

private:
  struct Job {
    const void*   partial = nullptr;
    void*         others  = nullptr;
    std::size_t   bytes   = 0;
    std::uint64_t seq     = 0;
    int           lane    = 0;
    std::uint64_t ready_at = 0;
    std::uint64_t v       = 0;       // its done value
  };
  struct Lane {
    metal_compute::Event ready;
    std::uint64_t        uses = 0;
  };

  void run_();
  bool wait_ready_(metal_compute::Event& ev, std::uint64_t v);
  void fail_(const std::string& why);
  // The channel's exchange, under _ch_mu; an exception is its failure.
  bool call_exchange_(const void* partial, void* others, std::size_t bytes,
                      std::uint64_t seq, std::string* err);

  std::shared_ptr<TpChannel> _ch;
  TpDType                    _dtype = TpDType::kBF16;
  std::deque<Lane>           _lanes;
  metal_compute::Event       _done;
  bool                       _ok = false;

  mutable std::mutex         _mu;
  std::condition_variable    _cv;        // a job posted, or stop
  std::condition_variable    _idle;      // the queue emptied
  std::deque<Job>            _q;
  bool                       _stop = false;
  std::atomic<bool>          _abandoned{false};
  std::atomic<bool>          _failed{false};
  std::string                _err;
  std::atomic<std::uint64_t> _posted{0};
  std::atomic<std::uint64_t> _finished{0};
  Stats                      _stats;

  std::mutex                 _ch_mu;     // the channel, one caller at a time
  std::thread                _th;
};

}

#endif
