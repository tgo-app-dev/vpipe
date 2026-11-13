#include "generative-models/tensor-parallel.h"

#include "common/vpipe-format.h"

#include <exception>
#include <mutex>
#include <utility>

namespace vpipe::genai {

namespace {

// [a, b] from an array of two integers; false when it is not one.
bool
range_of(const FlexData& fd, int* a, int* b)
{
  if (!fd.is_array()) { return false; }
  auto arr = fd.as_array();
  if (arr.size() != 2) { return false; }
  *a = (int)arr.at(0).as_int(-1);
  *b = (int)arr.at(1).as_int(-1);
  return true;
}

// The even split of `n` by rank: the first `n % ranks` ranks take one
// more.
void
even(int n, int rank, int ranks, int* a, int* b)
{
  const int base = n / ranks, extra = n % ranks;
  *a = rank * base + (rank < extra ? rank : extra);
  *b = *a + base + (rank < extra ? 1 : 0);
}

}

bool
TpShare::from_flex(const FlexData& fd, int n_heads, int ffn, TpShare* out,
                   std::string* err)
{
  auto fail = [&](const std::string& why) {
    if (err != nullptr) { *err = why; }
    return false;
  };
  if (!fd.is_object()) { return fail("tensor_parallel is not an object"); }
  auto o = fd.as_object();
  TpShare s;
  s.n_heads = n_heads;
  s.n_ffn   = ffn;
  s.rank  = o.contains("rank") ? (int)o.at("rank").as_int(-1) : 0;
  s.ranks = o.contains("ranks") ? (int)o.at("ranks").as_int(-1) : 1;
  if (s.ranks < 1 || s.rank < 0 || s.rank >= s.ranks) {
    return fail(fmt("rank {} of {} ranks", s.rank, s.ranks)());
  }
  even(n_heads, s.rank, s.ranks, &s.head0, &s.head1);
  even(ffn, s.rank, s.ranks, &s.ffn0, &s.ffn1);
  if (o.contains("heads") && !range_of(o.at("heads"), &s.head0, &s.head1)) {
    return fail("heads is not [first, end]");
  }
  if (o.contains("ffn") && !range_of(o.at("ffn"), &s.ffn0, &s.ffn1)) {
    return fail("ffn is not [first, end]");
  }
  if (s.head0 < 0 || s.head1 > n_heads || s.head0 >= s.head1) {
    return fail(fmt("heads [{}, {}) of {}", s.head0, s.head1, n_heads)());
  }
  if (s.ffn0 < 0 || s.ffn1 > ffn || s.ffn0 >= s.ffn1) {
    return fail(fmt("ffn rows [{}, {}) of {}", s.ffn0, s.ffn1, ffn)());
  }
  // One rank holding everything is the unsplit model whatever `ranks`
  // says; the shares of the others are theirs to check.
  *out = s;
  return true;
}

namespace {

// The registry: first-wins by name, as the model families' are.
struct Registry {
  std::mutex                                mu;
  std::vector<std::unique_ptr<TpTransport>> transports;
};

Registry&
registry() noexcept
{
  static Registry r;
  return r;
}

// A transport's name, "" when asking throws: the plugin's code, called
// from the host's (tensor-parallel.h: nothing called may throw).
std::string_view
name_of(const TpTransport& t) noexcept
{
  try {
    return t.name();
  } catch (...) {
    return {};
  }
}

}

bool
register_tp_transport(std::unique_ptr<TpTransport> t)
{
  if (!t) { return false; }
  const std::string_view nm = name_of(*t);
  if (nm.empty()) { return false; }           // nameless, or it threw
  Registry& r = registry();
  std::lock_guard<std::mutex> lk(r.mu);
  for (const auto& e : r.transports) {
    if (name_of(*e) == nm) { return false; }  // first-wins
  }
  r.transports.push_back(std::move(t));
  return true;
}

TpTransport*
find_tp_transport(std::string_view name)
{
  Registry& r = registry();
  std::lock_guard<std::mutex> lk(r.mu);
  for (const auto& e : r.transports) {
    if (name_of(*e) == name) { return e.get(); }
  }
  return nullptr;
}

}
