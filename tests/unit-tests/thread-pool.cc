#include "minitest.h"
#include "common/job.h"
#include "common/session.h"
#include "common/thread-pool.h"

#include <atomic>
#include <chrono>
#include <coroutine>
#include <future>
#include <memory>
#include <thread>

using namespace std;

namespace {

// Trivial coroutine: sets a promise to the resuming thread id, then
// completes. Used to assert the pool actually resumes work on a
// worker thread (not on the caller).
vpipe::Job
record_thread_and_finish(promise<thread::id> p)
{
  p.set_value(this_thread::get_id());
  co_return;
}

// Coroutine that just increments an atomic and completes.
vpipe::Job
inc_counter(atomic<unsigned>* counter)
{
  counter->fetch_add(1, memory_order_relaxed);
  co_return;
}

}

TEST(thread_pool, runs_scheduled_handle) {
  vpipe::Session sess;
  vpipe::ThreadPool pool(2, &sess);

  promise<thread::id> p;
  auto fut = p.get_future();

  vpipe::Job j = record_thread_and_finish(std::move(p));
  pool.schedule(j.handle());

  // Wait for the coroutine to set the promise. If this hangs the
  // pool didn't pick the work up.
  auto status = fut.wait_for(chrono::seconds(2));
  EXPECT_TRUE(status == future_status::ready);

  thread::id ran_on = fut.get();
  EXPECT_TRUE(ran_on != this_thread::get_id());
}

TEST(thread_pool, drains_many_jobs) {
  vpipe::Session sess;
  vpipe::ThreadPool pool(4, &sess);

  constexpr unsigned N = 200;
  atomic<unsigned> counter{0};
  vector<vpipe::Job> jobs;
  jobs.reserve(N);
  for (unsigned i = 0; i < N; ++i) {
    auto j = inc_counter(&counter);
    pool.schedule(j.handle());
    jobs.push_back(std::move(j));
  }

  // Wait for completion. The pool destructor would also drain, but
  // doing it here lets us assert the count cleanly.
  for (int spin = 0; spin < 200; ++spin) {
    if (counter.load(memory_order_relaxed) == N) break;
    this_thread::sleep_for(chrono::milliseconds(10));
  }
  EXPECT_TRUE(counter.load(memory_order_relaxed) == N);
}

TEST(thread_pool, num_workers_floor_one) {
  vpipe::Session sess;
  // Asking for 0 workers must still spin up at least one so the pool
  // can make progress.
  vpipe::ThreadPool pool(0, &sess);
  EXPECT_TRUE(pool.num_workers() == 1);
}

TEST(thread_pool, num_workers_from_session_config) {
  vpipe::Session sess(R"({"pool":{"num_workers":3}})");
  EXPECT_TRUE(sess.pool() != nullptr);
  EXPECT_TRUE(sess.pool()->num_workers() == 3);
}

namespace {

// Coroutine that records pool->worker_id_of_current_thread() into a
// promise so the parent test can assert.
vpipe::Job
record_worker_id(const vpipe::ThreadPool* pool, promise<unsigned> p)
{
  p.set_value(pool->worker_id_of_current_thread());
  co_return;
}

}

TEST(thread_pool, worker_id_inside_a_job_is_in_range) {
  vpipe::Session    sess;
  vpipe::ThreadPool pool(4, &sess);

  // Outside any worker thread the pool returns the not_a_worker
  // sentinel.
  EXPECT_TRUE(pool.worker_id_of_current_thread() ==
              vpipe::ThreadPool::not_a_worker);

  // Schedule a Job and observe the id from inside it.
  promise<unsigned> p;
  auto fut = p.get_future();
  vpipe::Job j = record_worker_id(&pool, std::move(p));
  pool.schedule(j.handle());

  auto status = fut.wait_for(chrono::seconds(2));
  EXPECT_TRUE(status == future_status::ready);
  unsigned wid = fut.get();
  EXPECT_TRUE(wid < pool.num_workers());
}

TEST(thread_pool, worker_id_for_unrelated_thread_is_sentinel) {
  vpipe::Session    sess;
  vpipe::ThreadPool pool(2, &sess);

  // A non-worker std::thread sees the sentinel.
  promise<unsigned> p;
  auto fut = p.get_future();
  thread t([&]() {
    p.set_value(pool.worker_id_of_current_thread());
  });
  t.join();
  EXPECT_TRUE(fut.get() == vpipe::ThreadPool::not_a_worker);
}

TEST(thread_pool, worker_id_does_not_leak_across_pools) {
  vpipe::Session    sess;
  vpipe::ThreadPool pool_a(2, &sess);
  vpipe::ThreadPool pool_b(2, &sess);

  // A coroutine running on pool_a sees a valid id from pool_a but
  // not_a_worker from pool_b -- the lookup is per-instance.
  promise<pair<unsigned, unsigned>> p;
  auto fut = p.get_future();
  auto job = [](const vpipe::ThreadPool* a,
                const vpipe::ThreadPool* b,
                promise<pair<unsigned, unsigned>> pp) -> vpipe::Job {
    pp.set_value({a->worker_id_of_current_thread(),
                  b->worker_id_of_current_thread()});
    co_return;
  }(&pool_a, &pool_b, std::move(p));
  pool_a.schedule(job.handle());

  auto status = fut.wait_for(chrono::seconds(2));
  EXPECT_TRUE(status == future_status::ready);
  auto [from_a, from_b] = fut.get();
  EXPECT_TRUE(from_a < pool_a.num_workers());
  EXPECT_TRUE(from_b == vpipe::ThreadPool::not_a_worker);
}

// ---- Job::quiescent() -------------------------------------------------
//
// The signal a peer thread uses to decide it may destroy a coroutine
// frame. `done()` cannot serve: the frame flips into the done state
// BEFORE FinalAwaiter::await_suspend runs, and await_suspend reads the
// promise -- so a destroyer released by done() frees the frame one
// access too early. That race is what ThreadSanitizer caught between
// PipelineRuntime's driver teardown and a pool worker.
//
// The ordering itself is a race and cannot be observed reliably from a
// test, so what is pinned here is the CONTRACT the fix rests on:
// quiescent() is false until the coroutine has actually reached
// final_suspend, and true afterwards.

namespace {

// Suspends once, so the test can observe a started-but-unfinished
// coroutine before letting it run to completion.
vpipe::Job
suspend_then_finish(atomic<bool>* ran)
{
  ran->store(true, memory_order_release);
  co_await std::suspend_always{};
  co_return;
}

}

// An UNSTARTED Job has not reached final_suspend, so it is not
// quiescent -- and neither is it done. Destroying one is safe only
// because its owner is the only thread that has ever touched it.
TEST(thread_pool, job_unstarted_is_not_quiescent) {
  atomic<bool> ran{false};
  vpipe::Job j = suspend_then_finish(&ran);
  EXPECT_TRUE(j.valid());
  EXPECT_FALSE(ran.load(memory_order_acquire));   // initial_suspend
  EXPECT_FALSE(j.done());
  EXPECT_FALSE(j.quiescent());
}

// Mid-flight (started, suspended, not finished) is likewise not
// quiescent: the coroutine still owns its frame.
TEST(thread_pool, job_midflight_is_not_quiescent) {
  atomic<bool> ran{false};
  vpipe::Job j = suspend_then_finish(&ran);
  j.handle().resume();                            // runs to the co_await
  EXPECT_TRUE(ran.load(memory_order_acquire));
  EXPECT_FALSE(j.done());
  EXPECT_FALSE(j.quiescent());
  j.handle().resume();                            // now runs to completion
  EXPECT_TRUE(j.done());
  EXPECT_TRUE(j.quiescent());
}

// A completed coroutine is quiescent, and quiescence implies done --
// the whole point being that the converse does NOT hold.
TEST(thread_pool, job_completed_is_quiescent) {
  atomic<unsigned> n{0};
  vpipe::Job j = inc_counter(&n);
  EXPECT_FALSE(j.quiescent());
  j.handle().resume();
  EXPECT_TRUE(n.load(memory_order_relaxed) == 1);
  EXPECT_TRUE(j.done());
  EXPECT_TRUE(j.quiescent());
}

// An empty Job owns no frame, so there is nothing to wait for: both
// predicates say "go ahead". A waiter that skipped this would spin
// forever on a default-constructed Job.
TEST(thread_pool, job_empty_is_quiescent) {
  vpipe::Job j;
  EXPECT_FALSE(j.valid());
  EXPECT_TRUE(j.done());
  EXPECT_TRUE(j.quiescent());
}

// Quiescence survives a move: the flag lives in the coroutine FRAME,
// not in the Job handle, so moving the handle cannot lose it. A
// PipelineRuntime that moved its drivers into place after they ran
// would otherwise wait on a flag that reset itself.
TEST(thread_pool, job_quiescence_follows_the_frame) {
  atomic<unsigned> n{0};
  vpipe::Job j = inc_counter(&n);
  j.handle().resume();
  EXPECT_TRUE(j.quiescent());
  vpipe::Job moved = std::move(j);
  EXPECT_TRUE(moved.quiescent());
  EXPECT_TRUE(j.quiescent());        // empty now -- nothing to wait for
  EXPECT_FALSE(j.valid());
}
