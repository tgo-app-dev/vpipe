#include "generative-models/shared/torch-zip.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace vpipe::genai::torch_zip {

namespace {

bool
fail_(std::string* err, std::string m)
{
  if (err != nullptr) { *err = std::move(m); }
  return false;
}

bool
pread_all_(int fd, void* dst, std::size_t n, std::uint64_t off)
{
  auto* p = static_cast<std::uint8_t*>(dst);
  std::size_t done = 0;
  while (done < n) {
    const ssize_t r = ::pread(fd, p + done, n - done, (off_t)(off + done));
    if (r < 0 && errno == EINTR) { continue; }
    if (r <= 0) { return false; }
    done += (std::size_t)r;
  }
  return true;
}

inline std::uint16_t
u16_(const std::uint8_t* p)
{
  return (std::uint16_t)(p[0] | (p[1] << 8));
}

inline std::uint32_t
u32_(const std::uint8_t* p)
{
  return (std::uint32_t)p[0] | ((std::uint32_t)p[1] << 8) |
         ((std::uint32_t)p[2] << 16) | ((std::uint32_t)p[3] << 24);
}

inline std::uint64_t
u64_(const std::uint8_t* p)
{
  return (std::uint64_t)u32_(p) | ((std::uint64_t)u32_(p + 4) << 32);
}

// ---- the archive -----------------------------------------------------

struct Entry {
  std::uint16_t method = 0;
  std::uint64_t usize  = 0;
  std::uint64_t lho    = 0;   // local header offset
};

constexpr std::uint32_t kEocdSig    = 0x06054b50;
constexpr std::uint32_t kZ64LocSig  = 0x07064b50;
constexpr std::uint32_t kZ64EocdSig = 0x06064b50;
constexpr std::uint32_t kCentralSig = 0x02014b50;
constexpr std::uint32_t kLocalSig   = 0x04034b50;

// The central directory. SIZES COME FROM HERE, never from the local
// headers: torch writes its entries with a trailing data descriptor, so
// every local header says zero.
bool
read_central_(int fd, std::uint64_t size,
              std::unordered_map<std::string, Entry>* out, std::string* err)
{
  constexpr std::uint64_t kTail = 22 + 65535 + 20;
  const std::uint64_t tail_n = size < kTail ? size : kTail;
  if (tail_n < 22) { return fail_(err, "too short to be a zip archive"); }
  std::vector<std::uint8_t> tail((std::size_t)tail_n);
  if (!pread_all_(fd, tail.data(), tail.size(), size - tail_n)) {
    return fail_(err, "cannot read the archive's end record");
  }
  std::int64_t at = -1;
  for (std::int64_t i = (std::int64_t)tail_n - 22; i >= 0; --i) {
    if (u32_(tail.data() + i) == kEocdSig) { at = i; break; }
  }
  if (at < 0) { return fail_(err, "no zip end-of-directory record"); }
  const std::uint8_t* e = tail.data() + at;
  std::uint64_t n_total = u16_(e + 10);
  std::uint64_t cd_size = u32_(e + 12);
  std::uint64_t cd_off  = u32_(e + 16);
  // ZIP64, which torch writes whatever the archive's size: the 32-bit
  // record is then a placeholder and the real numbers are one hop away.
  if (at >= 20 && u32_(e - 20) == kZ64LocSig) {
    const std::uint64_t z64 = u64_(e - 20 + 8);
    std::uint8_t r[56];
    if (z64 + sizeof(r) > size || !pread_all_(fd, r, sizeof(r), z64) ||
        u32_(r) != kZ64EocdSig) {
      return fail_(err, "a zip64 locator points at no zip64 record");
    }
    n_total = u64_(r + 32);
    cd_size = u64_(r + 40);
    cd_off  = u64_(r + 48);
  }
  if (cd_off + cd_size > size || cd_size > (64u << 20)) {
    return fail_(err, "the zip central directory is out of range");
  }
  std::vector<std::uint8_t> cd((std::size_t)cd_size);
  if (!pread_all_(fd, cd.data(), cd.size(), cd_off)) {
    return fail_(err, "cannot read the zip central directory");
  }
  std::size_t p = 0;
  for (std::uint64_t k = 0; k < n_total; ++k) {
    if (p + 46 > cd.size() || u32_(cd.data() + p) != kCentralSig) {
      return fail_(err, "a zip central directory entry is malformed");
    }
    const std::uint8_t* h = cd.data() + p;
    Entry en;
    en.method = u16_(h + 10);
    std::uint64_t csize = u32_(h + 20);
    en.usize = u32_(h + 24);
    const std::size_t nl = u16_(h + 28), el = u16_(h + 30),
                      cl = u16_(h + 32);
    en.lho = u32_(h + 42);
    if (p + 46 + nl + el + cl > cd.size()) {
      return fail_(err, "a zip central directory entry overruns it");
    }
    std::string name((const char*)h + 46, nl);
    // The zip64 extra carries, IN THIS ORDER, only the fields whose
    // 32-bit slot holds the all-ones placeholder.
    const std::uint8_t* x = h + 46 + nl;
    for (std::size_t q = 0; q + 4 <= el;) {
      const std::uint16_t id = u16_(x + q), len = u16_(x + q + 2);
      if (q + 4 + len > el) { break; }
      if (id == 0x0001) {
        std::size_t f = q + 4;
        auto take = [&](std::uint64_t* v) {
          if (*v != 0xffffffffu || f + 8 > q + 4 + len) { return; }
          *v = u64_(x + f);
          f += 8;
        };
        take(&en.usize);
        take(&csize);
        take(&en.lho);
      }
      q += 4 + len;
    }
    (void)csize;
    out->emplace(std::move(name), en);
    p += 46 + nl + el + cl;
  }
  return true;
}

bool
data_offset_(int fd, std::uint64_t size, const Entry& en, std::uint64_t* off,
             std::string* err)
{
  std::uint8_t h[30];
  if (en.lho + sizeof(h) > size || !pread_all_(fd, h, sizeof(h), en.lho) ||
      u32_(h) != kLocalSig) {
    return fail_(err, "a zip entry's local header is missing");
  }
  *off = en.lho + 30 + u16_(h + 26) + u16_(h + 28);
  if (*off + en.usize > size) {
    return fail_(err, "a zip entry runs past the end of the file");
  }
  return true;
}

// ---- the pickle ------------------------------------------------------

struct Value;
using VP = std::shared_ptr<Value>;

struct Value {
  enum class K {
    None, Bool, Int, Float, Str, Bytes, Tuple, List, Dict, Global, Storage,
    Tensor
  };
  K k = K::None;
  std::int64_t i = 0;
  // Str / Bytes contents; a Global's "module name"; a Storage's key.
  std::string s;
  std::vector<VP> items;                   // Tuple, List
  std::vector<std::pair<VP, VP>> dict;     // Dict, insertion order
  // Storage and Tensor.
  std::string dtype;
  std::uint64_t numel = 0;                 // Storage, in elements
  std::uint64_t storage_offset = 0;        // Tensor, in elements
  std::vector<std::int64_t> shape, stride; // Tensor
  VP storage;                              // Tensor
};

VP
mk_(Value::K k)
{
  auto v = std::make_shared<Value>();
  v->k = k;
  return v;
}

struct DtypeName {
  const char* torch;
  const char* st;
};

// The storage classes torch.save() writes into a persistent id...
constexpr DtypeName kStorages[] = {
  {"FloatStorage", "F32"},  {"DoubleStorage", "F64"},
  {"HalfStorage", "F16"},   {"BFloat16Storage", "BF16"},
  {"LongStorage", "I64"},   {"IntStorage", "I32"},
  {"ShortStorage", "I16"},  {"CharStorage", "I8"},
  {"ByteStorage", "U8"},    {"BoolStorage", "BOOL"},
  {"UntypedStorage", "U8"},
};
// ...and the dtype objects a newer writer names instead.
constexpr DtypeName kDtypes[] = {
  {"float32", "F32"}, {"float", "F32"},   {"float64", "F64"},
  {"double", "F64"},  {"float16", "F16"}, {"half", "F16"},
  {"bfloat16", "BF16"}, {"int64", "I64"}, {"long", "I64"},
  {"int32", "I32"},   {"int", "I32"},     {"int16", "I16"},
  {"short", "I16"},   {"int8", "I8"},     {"uint8", "U8"},
  {"bool", "BOOL"},
};

const char*
torch_dtype_(const std::string& global)
{
  if (global.rfind("torch ", 0) != 0) { return nullptr; }
  const std::string n = global.substr(6);
  for (const auto& d : kStorages) {
    if (n == d.torch) { return d.st; }
  }
  for (const auto& d : kDtypes) {
    if (n == d.torch) { return d.st; }
  }
  return nullptr;
}

std::uint64_t
itemsize_(const std::string& st)
{
  if (st == "F64" || st == "I64") { return 8; }
  if (st == "F32" || st == "I32") { return 4; }
  if (st == "F16" || st == "BF16" || st == "I16") { return 2; }
  return 1;
}

// THE ALLOWLIST. Recognised by name; nothing here is ever called.
bool
allowed_global_(const std::string& g)
{
  if (g == "collections OrderedDict" || g == "builtins dict" ||
      g == "__builtin__ dict") {
    return true;
  }
  if (g == "torch._utils _rebuild_tensor_v2" ||
      g == "torch._utils _rebuild_tensor" ||
      g == "torch._utils _rebuild_parameter" || g == "torch Size") {
    return true;
  }
  return torch_dtype_(g) != nullptr;
}

class Unpickler {
 public:
  Unpickler(const std::uint8_t* p, std::size_t n) : _p(p), _n(n) {}

  bool
  run(VP* out, std::string* err)
  {
    while (_pos < _n) {
      const std::uint8_t op = _p[_pos++];
      if (op == '.') {                                   // STOP
        if (_stack.empty()) { return fail_(err, "pickle: empty at STOP"); }
        *out = _stack.back();
        return true;
      }
      if (!step_(op, err)) { return false; }
    }
    return fail_(err, "pickle: ended without STOP");
  }

 private:
  bool
  need_(std::size_t k, std::string* err)
  {
    if (_pos + k <= _n) { return true; }
    return fail_(err, "pickle: truncated");
  }
  bool
  pop_(VP* v, std::string* err)
  {
    if (_stack.empty() ||
        (!_marks.empty() && _marks.back() >= _stack.size())) {
      return fail_(err, "pickle: stack underflow");
    }
    *v = std::move(_stack.back());
    _stack.pop_back();
    return true;
  }
  bool
  pop_mark_(std::vector<VP>* items, std::string* err)
  {
    if (_marks.empty()) { return fail_(err, "pickle: no MARK"); }
    const std::size_t m = _marks.back();
    _marks.pop_back();
    items->assign(std::make_move_iterator(_stack.begin() + (long)m),
                  std::make_move_iterator(_stack.end()));
    _stack.resize(m);
    return true;
  }
  void push_(VP v) { _stack.push_back(std::move(v)); }

  bool
  str_(std::size_t len, Value::K k, std::string* err)
  {
    if (!need_(len, err)) { return false; }
    VP v = mk_(k);
    v->s.assign((const char*)_p + _pos, len);
    _pos += len;
    push_(std::move(v));
    return true;
  }
  bool
  int_(std::int64_t x)
  {
    VP v = mk_(Value::K::Int);
    v->i = x;
    push_(std::move(v));
    return true;
  }
  bool
  tuple_(std::size_t n, std::string* err)
  {
    VP t = mk_(Value::K::Tuple);
    t->items.resize(n);
    for (std::size_t k = n; k-- > 0;) {
      if (!pop_(&t->items[k], err)) { return false; }
    }
    push_(std::move(t));
    return true;
  }

  bool reduce_(std::string* err);
  bool persid_(std::string* err);
  bool step_(std::uint8_t op, std::string* err);

  const std::uint8_t* _p;
  std::size_t _n;
  std::size_t _pos = 0;
  std::vector<VP> _stack;
  std::vector<std::size_t> _marks;
  std::unordered_map<std::uint64_t, VP> _memo;
};

bool
Unpickler::reduce_(std::string* err)
{
  VP args, fn;
  if (!pop_(&args, err) || !pop_(&fn, err)) { return false; }
  if (fn->k != Value::K::Global || args->k != Value::K::Tuple) {
    return fail_(err, "pickle: REDUCE of something that is not a global");
  }
  const std::string& g = fn->s;
  if (g == "collections OrderedDict" || g == "builtins dict" ||
      g == "__builtin__ dict") {
    push_(mk_(Value::K::Dict));
    return true;
  }
  if (g == "torch Size" || g == "torch._utils _rebuild_parameter") {
    if (args->items.empty()) { return fail_(err, "pickle: " + g + "()"); }
    push_(args->items[0]);
    return true;
  }
  // _rebuild_tensor_v2(storage, storage_offset, size, stride, ...)
  auto ints = [](const VP& t, std::vector<std::int64_t>* v) {
    if (t->k != Value::K::Tuple) { return false; }
    for (const VP& e : t->items) {
      if (e->k != Value::K::Int) { return false; }
      v->push_back(e->i);
    }
    return true;
  };
  const auto& a = args->items;
  VP t = mk_(Value::K::Tensor);
  if (a.size() < 4 || a[0]->k != Value::K::Storage ||
      a[1]->k != Value::K::Int || a[1]->i < 0 || !ints(a[2], &t->shape) ||
      !ints(a[3], &t->stride)) {
    return fail_(err, "pickle: " + g + " with arguments this reader does "
                      "not understand");
  }
  t->storage = a[0];
  t->dtype = a[0]->dtype;
  t->storage_offset = (std::uint64_t)a[1]->i;
  push_(std::move(t));
  return true;
}

bool
Unpickler::persid_(std::string* err)
{
  // ('storage', <storage class or dtype>, key, location, numel)
  VP pid;
  if (!pop_(&pid, err)) { return false; }
  const auto& it = pid->items;
  if (pid->k != Value::K::Tuple || it.size() < 5 ||
      it[0]->k != Value::K::Str || it[0]->s != "storage" ||
      it[1]->k != Value::K::Global || it[4]->k != Value::K::Int) {
    return fail_(err, "pickle: a persistent id that is not a storage");
  }
  const char* st = torch_dtype_(it[1]->s);
  if (st == nullptr) {
    return fail_(err, "pickle: unknown storage type " + it[1]->s);
  }
  VP s = mk_(Value::K::Storage);
  s->dtype = st;
  s->numel = (std::uint64_t)(it[4]->i < 0 ? 0 : it[4]->i);
  if (it[2]->k == Value::K::Str) {
    s->s = it[2]->s;
  } else if (it[2]->k == Value::K::Int) {
    s->s = std::to_string(it[2]->i);
  } else {
    return fail_(err, "pickle: a storage key that is not a string");
  }
  push_(std::move(s));
  return true;
}

bool
Unpickler::step_(std::uint8_t op, std::string* err)
{
  switch (op) {
    case 0x80:                                           // PROTO
      if (!need_(1, err)) { return false; }
      ++_pos;
      return true;
    case 0x95:                                           // FRAME
      if (!need_(8, err)) { return false; }
      _pos += 8;
      return true;
    case 'c': {                                          // GLOBAL
      std::string parts[2];
      for (std::string& part : parts) {
        const void* nl = std::memchr(_p + _pos, '\n', _n - _pos);
        if (nl == nullptr) { return fail_(err, "pickle: bad GLOBAL"); }
        const std::size_t len =
            (std::size_t)((const std::uint8_t*)nl - (_p + _pos));
        part.assign((const char*)_p + _pos, len);
        _pos += len + 1;
      }
      VP g = mk_(Value::K::Global);
      g->s = parts[0] + " " + parts[1];
      if (!allowed_global_(g->s)) {
        return fail_(err, "refusing global '" + parts[0] + "." + parts[1] +
                          "': this reader builds data and calls nothing, "
                          "and a plain state_dict names no such thing");
      }
      push_(std::move(g));
      return true;
    }
    case 0x93: {                                         // STACK_GLOBAL
      VP name, mod;
      if (!pop_(&name, err) || !pop_(&mod, err)) { return false; }
      VP g = mk_(Value::K::Global);
      g->s = mod->s + " " + name->s;
      if (!allowed_global_(g->s)) {
        return fail_(err, "refusing global '" + mod->s + "." + name->s + "'");
      }
      push_(std::move(g));
      return true;
    }
    case 'q':                                            // BINPUT
      if (!need_(1, err) || _stack.empty()) {
        return fail_(err, "pickle: bad BINPUT");
      }
      _memo[_p[_pos++]] = _stack.back();
      return true;
    case 'r':                                            // LONG_BINPUT
      if (!need_(4, err) || _stack.empty()) {
        return fail_(err, "pickle: bad LONG_BINPUT");
      }
      _memo[u32_(_p + _pos)] = _stack.back();
      _pos += 4;
      return true;
    case 0x94:                                           // MEMOIZE
      if (_stack.empty()) { return fail_(err, "pickle: bad MEMOIZE"); }
      _memo[_memo.size()] = _stack.back();
      return true;
    case 'h':                                            // BINGET
    case 'j': {                                          // LONG_BINGET
      const std::size_t w = op == 'h' ? 1 : 4;
      if (!need_(w, err)) { return false; }
      const std::uint64_t key = w == 1 ? _p[_pos] : u32_(_p + _pos);
      _pos += w;
      auto f = _memo.find(key);
      if (f == _memo.end()) { return fail_(err, "pickle: unset memo"); }
      push_(f->second);
      return true;
    }
    case '(':                                            // MARK
      _marks.push_back(_stack.size());
      return true;
    case 't': {                                          // TUPLE
      VP t = mk_(Value::K::Tuple);
      if (!pop_mark_(&t->items, err)) { return false; }
      push_(std::move(t));
      return true;
    }
    case ')': push_(mk_(Value::K::Tuple)); return true; // EMPTY_TUPLE
    case 0x85: return tuple_(1, err);                    // TUPLE1
    case 0x86: return tuple_(2, err);                    // TUPLE2
    case 0x87: return tuple_(3, err);                    // TUPLE3
    case ']': push_(mk_(Value::K::List)); return true;  // EMPTY_LIST
    case 'l': {                                          // LIST
      VP l = mk_(Value::K::List);
      if (!pop_mark_(&l->items, err)) { return false; }
      push_(std::move(l));
      return true;
    }
    case 'a': {                                          // APPEND
      VP v;
      if (!pop_(&v, err)) { return false; }
      if (_stack.empty() || _stack.back()->k != Value::K::List) {
        return fail_(err, "pickle: APPEND to a non-list");
      }
      _stack.back()->items.push_back(std::move(v));
      return true;
    }
    case 'e': {                                          // APPENDS
      std::vector<VP> items;
      if (!pop_mark_(&items, err)) { return false; }
      if (_stack.empty() || _stack.back()->k != Value::K::List) {
        return fail_(err, "pickle: APPENDS to a non-list");
      }
      for (VP& v : items) { _stack.back()->items.push_back(std::move(v)); }
      return true;
    }
    case '}': push_(mk_(Value::K::Dict)); return true;  // EMPTY_DICT
    case 'd':                                            // DICT
    case 'u': {                                          // SETITEMS
      std::vector<VP> items;
      if (!pop_mark_(&items, err)) { return false; }
      if (items.size() % 2 != 0) { return fail_(err, "pickle: odd dict"); }
      if (op == 'd') { push_(mk_(Value::K::Dict)); }
      if (_stack.empty() || _stack.back()->k != Value::K::Dict) {
        return fail_(err, "pickle: SETITEMS on a non-dict");
      }
      for (std::size_t k = 0; k < items.size(); k += 2) {
        _stack.back()->dict.emplace_back(items[k], items[k + 1]);
      }
      return true;
    }
    case 's': {                                          // SETITEM
      VP v, key;
      if (!pop_(&v, err) || !pop_(&key, err)) { return false; }
      if (_stack.empty() || _stack.back()->k != Value::K::Dict) {
        return fail_(err, "pickle: SETITEM on a non-dict");
      }
      _stack.back()->dict.emplace_back(std::move(key), std::move(v));
      return true;
    }
    case 'X':                                            // BINUNICODE
      if (!need_(4, err)) { return false; }
      _pos += 4;
      return str_(u32_(_p + _pos - 4), Value::K::Str, err);
    case 0x8c:                                           // SHORT_BINUNICODE
      if (!need_(1, err)) { return false; }
      ++_pos;
      return str_(_p[_pos - 1], Value::K::Str, err);
    case 0x8d:                                           // BINUNICODE8
      if (!need_(8, err)) { return false; }
      _pos += 8;
      return str_((std::size_t)u64_(_p + _pos - 8), Value::K::Str, err);
    case 'B':                                            // BINBYTES
      if (!need_(4, err)) { return false; }
      _pos += 4;
      return str_(u32_(_p + _pos - 4), Value::K::Bytes, err);
    case 'C':                                            // SHORT_BINBYTES
      if (!need_(1, err)) { return false; }
      ++_pos;
      return str_(_p[_pos - 1], Value::K::Bytes, err);
    case 'J':                                            // BININT
      if (!need_(4, err)) { return false; }
      _pos += 4;
      return int_((std::int32_t)u32_(_p + _pos - 4));
    case 'K':                                            // BININT1
      if (!need_(1, err)) { return false; }
      return int_(_p[_pos++]);
    case 'M':                                            // BININT2
      if (!need_(2, err)) { return false; }
      _pos += 2;
      return int_(u16_(_p + _pos - 2));
    case 0x8a: {                                         // LONG1
      if (!need_(1, err)) { return false; }
      const std::size_t len = _p[_pos++];
      if (len > 8 || !need_(len, err)) {
        return fail_(err, "pickle: a LONG1 wider than 64 bits");
      }
      std::uint64_t u = 0;
      for (std::size_t k = 0; k < len; ++k) {
        u |= (std::uint64_t)_p[_pos + k] << (8 * k);
      }
      if (len > 0 && len < 8 && (_p[_pos + len - 1] & 0x80) != 0) {
        u |= ~0ull << (8 * len);                         // sign-extend
      }
      _pos += len;
      return int_((std::int64_t)u);
    }
    case 'N': push_(mk_(Value::K::None)); return true;  // NONE
    case 0x88:                                           // NEWTRUE
    case 0x89: {                                         // NEWFALSE
      VP b = mk_(Value::K::Bool);
      b->i = op == 0x88 ? 1 : 0;
      push_(std::move(b));
      return true;
    }
    case 'G':                                            // BINFLOAT
      if (!need_(8, err)) { return false; }
      _pos += 8;
      push_(mk_(Value::K::Float));
      return true;
    case 'Q': return persid_(err);                       // BINPERSID
    case 'R':                                            // REDUCE
    case 0x81: return reduce_(err);                      // NEWOBJ
    case 'b': {                                          // BUILD
      // The state an OrderedDict carries (a state_dict's `_metadata`) or
      // a tensor's. Neither changes a byte of any tensor.
      VP state;
      if (!pop_(&state, err)) { return false; }
      if (_stack.empty()) { return fail_(err, "pickle: BUILD on nothing"); }
      return true;
    }
    case '0': {                                          // POP
      VP v;
      return pop_(&v, err);
    }
    case '1': {                                          // POP_MARK
      std::vector<VP> items;
      return pop_mark_(&items, err);
    }
    case '2':                                            // DUP
      if (_stack.empty()) { return fail_(err, "pickle: DUP on nothing"); }
      push_(_stack.back());
      return true;
    default: {
      char hex[8];
      std::snprintf(hex, sizeof(hex), "0x%02x", (unsigned)op);
      return fail_(err, std::string("pickle: opcode ") + hex +
                        " is not one a state_dict uses");
    }
  }
}

// ---- the table -------------------------------------------------------

struct Blob {
  std::uint64_t offset = 0;
  std::uint64_t size   = 0;
};

bool
emit_(const std::string& name, const Value& t,
      const std::unordered_map<std::string, Blob>& blobs,
      std::vector<Tensor>* out, std::string* err)
{
  auto bad = [&](const std::string& why) {
    return fail_(err, "tensor '" + name + "': " + why);
  };
  if (t.shape.size() != t.stride.size()) {
    return bad("shape and stride disagree in rank");
  }
  std::uint64_t numel = 1;
  for (const auto d : t.shape) {
    if (d < 0) { return bad("negative dimension"); }
    numel *= (std::uint64_t)d;
  }
  // C ORDER OR NOTHING. A size-1 axis has no meaningful stride, so it is
  // not checked; everything else must step exactly like a contiguous
  // tensor, or the bytes at the offset are not the tensor.
  std::int64_t expect = 1;
  for (std::size_t k = t.shape.size(); k-- > 0;) {
    if (numel > 0 && t.shape[k] != 1 && t.stride[k] != expect) {
      return bad("not contiguous; this reader maps and does not copy");
    }
    expect *= t.shape[k];
  }
  auto b = blobs.find(t.storage->s);
  if (b == blobs.end()) {
    return bad("its storage '" + t.storage->s + "' is not in the archive");
  }
  const std::uint64_t item = itemsize_(t.dtype);
  const std::uint64_t start = t.storage_offset * item;
  const std::uint64_t n = numel * item;
  if (start + n > b->second.size) {
    return bad("it runs past the end of its storage");
  }
  Tensor o;
  o.name = name;
  o.dtype = t.dtype;
  o.shape = t.shape;
  o.offset = b->second.offset + start;
  o.nbytes = n;
  out->push_back(std::move(o));
  return true;
}

bool
walk_(const std::string& prefix, const Value& d,
      const std::unordered_map<std::string, Blob>& blobs,
      std::vector<Tensor>* out, std::string* err)
{
  for (const auto& kv : d.dict) {
    if (kv.first->k != Value::K::Str) { continue; }
    const std::string name = prefix + kv.first->s;
    if (kv.second->k == Value::K::Tensor) {
      if (!emit_(name, *kv.second, blobs, out, err)) { return false; }
    } else if (kv.second->k == Value::K::Dict) {
      if (!walk_(name + ".", *kv.second, blobs, out, err)) { return false; }
    }
    // Anything else -- an epoch count, a config string -- is not a tensor
    // and has no place in a weight table.
  }
  return true;
}

}  // namespace

bool
looks_like_zip(int fd)
{
  std::uint8_t m[4];
  return pread_all_(fd, m, sizeof(m), 0) && u32_(m) == kLocalSig;
}

bool
read_index(int fd, std::uint64_t size, std::vector<Tensor>* out,
           std::string* err)
{
  if (out == nullptr) { return fail_(err, "no output"); }
  out->clear();
  std::unordered_map<std::string, Entry> entries;
  if (!read_central_(fd, size, &entries, err)) { return false; }

  // THE ARCHIVE'S ROOT is whatever directory holds data.pkl -- torch
  // names it after the file it was saved to, so it is found, not assumed.
  std::string root;
  for (const auto& kv : entries) {
    const std::string& n = kv.first;
    static constexpr const char kPkl[] = "data.pkl";
    if (n.size() >= 8 && n.compare(n.size() - 8, 8, kPkl) == 0 &&
        (n.size() == 8 || n[n.size() - 9] == '/')) {
      const std::string r = n.substr(0, n.size() - 8);
      if (root.empty() || r.size() < root.size()) { root = r; }
    }
  }
  auto pkl = entries.find(root + "data.pkl");
  if (pkl == entries.end()) {
    return fail_(err, "a zip archive, but not a torch checkpoint (no "
                      "data.pkl)");
  }
  if (auto bo = entries.find(root + "byteorder"); bo != entries.end()) {
    std::uint64_t off = 0;
    char buf[16] = {};
    if (bo->second.usize >= sizeof(buf) ||
        !data_offset_(fd, size, bo->second, &off, err) ||
        !pread_all_(fd, buf, (std::size_t)bo->second.usize, off)) {
      return fail_(err, "unreadable byteorder record");
    }
    if (std::string(buf) != "little") {
      return fail_(err, std::string("a '") + buf + "'-endian archive");
    }
  }

  // Every storage blob, by key, where its bytes start.
  std::unordered_map<std::string, Blob> blobs;
  const std::string data_dir = root + "data/";
  for (const auto& kv : entries) {
    if (kv.first.rfind(data_dir, 0) != 0) { continue; }
    if (kv.second.method != 0) {
      return fail_(err, "storage '" + kv.first + "' is compressed; torch "
                        "stores them, and a compressed one cannot be "
                        "mapped");
    }
    Blob b;
    if (!data_offset_(fd, size, kv.second, &b.offset, err)) { return false; }
    b.size = kv.second.usize;
    blobs.emplace(kv.first.substr(data_dir.size()), b);
  }

  if (pkl->second.method != 0 || pkl->second.usize > (256u << 20)) {
    return fail_(err, "data.pkl is compressed or implausibly large");
  }
  std::uint64_t pkl_off = 0;
  if (!data_offset_(fd, size, pkl->second, &pkl_off, err)) { return false; }
  std::vector<std::uint8_t> pk((std::size_t)pkl->second.usize);
  if (!pread_all_(fd, pk.data(), pk.size(), pkl_off)) {
    return fail_(err, "cannot read data.pkl");
  }
  VP top;
  if (!Unpickler(pk.data(), pk.size()).run(&top, err)) { return false; }

  bool ok = false;
  if (top->k == Value::K::Tensor) {
    // A bare tensor, named for its archive: the file's own stem.
    std::string name =
        root.empty() ? "tensor" : root.substr(0, root.size() - 1);
    const std::size_t slash = name.rfind('/');
    if (slash != std::string::npos) { name = name.substr(slash + 1); }
    ok = emit_(name, *top, blobs, out, err);
  } else if (top->k == Value::K::Dict) {
    // A training checkpoint wraps its weights one level down. Unwrapped
    // only when that is ALL it is -- a dict whose tensors sit beside it
    // would lose them.
    const Value* d = top.get();
    for (const auto& kv : top->dict) {
      if (kv.first->k == Value::K::Str && kv.first->s == "state_dict" &&
          kv.second->k == Value::K::Dict) {
        d = kv.second.get();
      }
    }
    ok = walk_("", *d, blobs, out, err);
    // ...and its SIBLINGS too, under their own keys: a training state
    // keeps the EMA model the release decodes with beside the raw one
    // (`ema_state_dict`). Additive -- the unwrapped names are as they
    // were -- and best effort: a sibling this reader cannot index (or
    // whose names collide) is left out rather than failing a load that
    // reads what it needs from `state_dict`.
    if (ok && d != top.get()) {
      std::unordered_set<std::string> have;
      for (const Tensor& t : *out) { have.insert(t.name); }
      for (const auto& kv : top->dict) {
        if (kv.first->k != Value::K::Str || kv.first->s == "state_dict" ||
            kv.second->k != Value::K::Dict) {
          continue;
        }
        std::vector<Tensor> more;
        std::string serr;
        if (!walk_(kv.first->s + ".", *kv.second, blobs, &more, &serr)) {
          continue;
        }
        for (Tensor& t : more) {
          if (have.insert(t.name).second) { out->push_back(std::move(t)); }
        }
      }
    }
  } else {
    return fail_(err, "the checkpoint holds neither a tensor nor a dict of "
                      "them");
  }
  if (!ok) { out->clear(); }
  return ok;
}

}  // namespace vpipe::genai::torch_zip
