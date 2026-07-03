#include "stages/model-benchmark-stage.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/ui-delegate-intf.h"
#include "stages/gpu-telemetry.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/generative-model-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/quantize/calibration.h"
#include "generative-models/sampler.h"
#include <chrono>
#include <functional>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#endif

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

namespace {

// Parse a comma-separated list of context lengths into a sorted, unique
// vector of positive ints.
std::vector<int>
parse_contexts_(const std::string& csv)
{
  std::vector<int> out;
  std::size_t i = 0;
  while (i < csv.size()) {
    std::size_t j = csv.find(',', i);
    if (j == std::string::npos) { j = csv.size(); }
    const std::string tok = csv.substr(i, j - i);
    int v = 0;
    bool any = false;
    for (char c : tok) {
      if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); any = true; }
    }
    if (any && v > 0) { out.push_back(v); }
    i = j + 1;
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

}  // namespace

ModelBenchmarkStage::ModelBenchmarkStage(const SessionContextIntf* s,
                                         std::string               id,
                                         std::vector<InEdge>       in,
                                         FlexData                  config)
  : TypedStage<ModelBenchmarkStage>(s, std::move(id), std::move(in),
                                    std::move(config))
{
  _model         = attr_str("model");
  _models_db     = attr_str("models_db");
  _contexts      = parse_contexts_(attr_str("contexts"));
  _decode_tokens = static_cast<int>(attr_uint("decode_tokens"));
  _prefill_chunk = static_cast<int>(attr_uint("prefill_chunk"));
  _warmup        = static_cast<int>(attr_uint("warmup"));
  _seed          = static_cast<std::uint64_t>(attr_uint("seed"));
  _cooldown_s         = attr_real("cooldown_s");
  _temperature        = attr_real("temperature");
  _top_k              = attr_uint("top_k");
  _top_p              = attr_real("top_p");
  _min_p              = attr_real("min_p");
  _repetition_penalty = attr_real("repetition_penalty");
  _presence_penalty   = attr_real("presence_penalty");
  _mtp                = attr_bool("mtp");
  _mtp_require_exact  = attr_bool("mtp_require_exact");
  if (_models_db.empty()) { _models_db = "models"; }
  if (_contexts.empty())  { _contexts = {1024, 2048, 4096}; }

  if (_model.empty()) {
    fail_config(fmt("ModelBenchmarkStage('{}'): config.model is required",
                    this->id()));
  }
  allocate_oports(spec().oports.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "model", .type = ConfigType::String, .required = true,
   .doc = "text LLM to benchmark: a models-DB key (model-fetch / "
          "model-quantize) or a model directory path",
   .suggest_db = "models",
   .suggest_db_type = "qwen3.5,qwen3.6,gemma4,gemma4_unified"},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db for model-key resolution", .def_str = "models"},
  {.key = "contexts", .type = ConfigType::String,
   .doc = "comma-separated context lengths to benchmark (e.g. add "
          "\"8192,16384\" for longer probes)",
   .def_str = "1024,2048,4096"},
  {.key = "decode_tokens", .type = ConfigType::Uint,
   .doc = "tokens to decode per decode test", .def_uint = 128},
  {.key = "prefill_chunk", .type = ConfigType::Uint,
   .doc = "chunk size for the incremental-prefill-at-depth probe",
   .def_uint = 1024},
  {.key = "warmup", .type = ConfigType::Uint,
   .doc = "warmup iterations before timing (discarded)", .def_uint = 1},
  {.key = "seed", .type = ConfigType::Uint,
   .doc = "RNG seed for the synthetic prompt (also the sampler seed)",
   .def_uint = 1234},
  {.key = "cooldown_s", .type = ConfigType::Real,
   .doc = "seconds to sleep after each timed test so the GPU thermals "
          "settle; skipped after the last test", .def_real = 0.0},
  {.key = "temperature", .type = ConfigType::Real,
   .doc = "decode sampler temperature (0 = greedy/argmax)",
   .def_real = 0.0},
  {.key = "top_k", .type = ConfigType::Uint,
   .doc = "decode sampler top-k (0 = disabled)", .def_uint = 0},
  {.key = "top_p", .type = ConfigType::Real,
   .doc = "decode sampler nucleus top-p (1.0 = disabled)",
   .def_real = 1.0},
  {.key = "min_p", .type = ConfigType::Real,
   .doc = "decode sampler min-p (0.0 = disabled)", .def_real = 0.0},
  {.key = "repetition_penalty", .type = ConfigType::Real,
   .doc = "decode repetition penalty (1.0 = disabled)",
   .def_real = 1.0},
  {.key = "presence_penalty", .type = ConfigType::Real,
   .doc = "decode presence penalty (0.0 = disabled)", .def_real = 0.0},
  {.key = "mtp", .type = ConfigType::Bool,
   .doc = "use MTP speculative decode for the decode test when the "
          "model carries an MTP head (real-text prompt + acceptance "
          "rate); pdecode run-ahead otherwise", .def_bool = false},
  {.key = "mtp_require_exact", .type = ConfigType::Bool,
   .doc = "require MTP greedy to be token-exact vs serial greedy before "
          "using it; on divergence fall back to pdecode. Default false: "
          "MTP is a valid greedy decode even when it diverges (the batched "
          "verify isn't bit-identical to the single-row decode in bf16), so "
          "it is benchmarked anyway with a note", .def_bool = false},
};
const StageSpec kSpec = {
  .type_name = "model-benchmark",
  .doc       = "Source: one-shot LM throughput benchmark. Loads a "
               "language model (a models-DB key or a path) and times "
               "prefill + incremental-prefill + decode at several "
               "context lengths, recording GPU temperature / frequency "
               "/ utilization / power during each test, and logs a "
               "Markdown report. 0 in / 0 out.",
  .display_name = "Model Benchmark",
  .category  = StageCategory::Preparation,
  .iports    = {},
  .oports    = {},
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
ModelBenchmarkStage::spec() const noexcept
{
  return kSpec;
}

#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

// One benchmark result row: a tok/s figure + the GPU telemetry recorded
// while it ran.
struct Row {
  int          ctx    = 0;
  double       tok_s  = 0.0;
  bool         ok     = false;
  // MTP prediction acceptance fraction for the decode test, or < 0 when
  // the decode used the pdecode path (no MTP).
  double       accept = -1.0;
  // Set when tok_s exceeds the memory-bandwidth ceiling for the model's
  // resident size -- a physically-impossible rate signalling a broken
  // decode path.
  bool         implausible = false;
  GpuTelemetry tel;
};

// Deterministic synthetic prompt: `n` token ids kept clear of the
// special-token edges of the vocab (an LCG seeded by `seed`).
std::vector<std::int32_t>
make_prompt_(int n, int vocab, std::uint64_t seed)
{
  std::vector<std::int32_t> v;
  if (n <= 0) { return v; }
  v.reserve(static_cast<std::size_t>(n));
  const int lo = 10;
  const int hi = std::max(lo + 1, vocab - 10);
  const std::uint64_t span = static_cast<std::uint64_t>(hi - lo);
  std::uint64_t s = seed ? seed : 1;
  for (int i = 0; i < n; ++i) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const int id = lo + static_cast<int>((s >> 33) % span);
    v.push_back(static_cast<std::int32_t>(id));
  }
  return v;
}

std::string toks_cell_(const Row& r)
{
  if (!r.ok) { return "n/a"; }
  return fmt("{:.1f}{}", r.tok_s, r.implausible ? " (!)" : "")();
}
std::string mhz_cell_(const MetricAgg& m)
{
  return m.ok ? fmt("{:.0f}", m.avg)() : std::string("n/a");
}
std::string temp_cell_(const MetricAgg& m)
{
  return m.ok ? fmt("{:.0f}/{:.0f}", m.avg, m.max)()
              : std::string("n/a");
}
std::string util_cell_(const MetricAgg& m)
{
  return m.ok ? fmt("{:.0f}", m.avg)() : std::string("n/a");
}
std::string mem_cell_(const MetricAgg& m)
{
  return m.ok ? fmt("{:.0f}", m.max)() : std::string("n/a");
}
std::string accept_cell_(const Row& r)
{
  return r.accept >= 0.0 ? fmt("{:.0f}%", r.accept * 100.0)()
                         : std::string("—");
}

// A decode tok/s that exceeds what memory bandwidth could sustain for a
// model of this resident size flags a broken decode path (a verify not
// reading the full model per token). kPeakBw sits above any current
// Apple-Silicon UMA; kSlack covers MTP speculation (<=~3x single-stream)
// + measurement noise. Needs footprint telemetry; absent -> no verdict.
bool decode_implausible_(const Row& r)
{
  if (!r.ok || !r.tel.footprint_mb.ok) { return false; }
  const double bytes = r.tel.footprint_mb.max * 1024.0 * 1024.0;
  if (bytes <= 1.0) { return false; }
  constexpr double kPeakBw = 1200.0e9;   // bytes/s, > any Apple Si UMA
  constexpr double kSlack  = 4.0;        // MTP spec + noise headroom
  return r.tok_s > kPeakBw / bytes * kSlack;
}

// MTP greedy decode is token-exact vs serial greedy by construction. Run a
// short decode both ways from the same prompt and compare; substantial
// divergence means the MTP path is broken/unvalidated for this model. The
// MTP stream may or may not include the prefill's first token, so the
// better of the two offset alignments is taken. Returns true when MTP looks
// correct (or the run is too short to judge).
bool mtp_matches_serial_(genai::LoadedLanguageModel& lm,
                         const std::vector<std::int32_t>& prompt, int k)
{
  if (prompt.empty() || k < 4) { return true; }
  std::vector<std::int32_t> ser;
  {
    auto ctx = lm.make_context();
    std::int32_t t = lm.prefill(ctx, prompt);
    while (static_cast<int>(ser.size()) < k && t >= 0) {
      ser.push_back(t);
      t = lm.next_token_greedy(ctx);
    }
  }
  std::vector<std::int32_t> got;
  {
    auto ctx = lm.make_context();
    const std::int32_t f = lm.prefill(ctx, prompt);
    if (f < 0) { return true; }
    genai::SamplerParams greedy;
    greedy.temperature = 0.0f;
    auto on = [&](std::span<const std::int32_t> s) {
      for (std::int32_t tk : s) { got.push_back(tk); }
      return static_cast<int>(got.size()) < k;
    };
    lm.mtp_generate(ctx, f, k, greedy,
                    [](std::int32_t) { return false; }, on, nullptr,
                    nullptr);
  }
  if (ser.size() < 4 || got.size() < 4) { return true; }
  auto mism = [&](std::size_t off) -> std::size_t {
    if (ser.size() <= off) { return got.size(); }
    const std::size_t n = std::min(ser.size() - off, got.size());
    std::size_t m = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (ser[i + off] != got[i]) { ++m; }
    }
    return m;
  };
  const std::size_t best = std::min(mism(0), mism(1));
  const std::size_t n = std::min(ser.size(), got.size());
  return best * 2 < n;   // a correct MTP gives best == 0
}

}  // namespace
#endif  // VPIPE_BUILD_APPLE_SILICON

#ifdef VPIPE_BUILD_APPLE_SILICON
genai::SamplerParams
ModelBenchmarkStage::sampler_params_() const
{
  genai::SamplerParams sp;
  sp.temperature        = static_cast<float>(_temperature);
  sp.top_k              = static_cast<int>(_top_k);
  sp.top_p              = static_cast<float>(_top_p);
  sp.min_p              = static_cast<float>(_min_p);
  sp.repetition_penalty = static_cast<float>(_repetition_penalty);
  sp.presence_penalty   = static_cast<float>(_presence_penalty);
  sp.seed               = _seed;
  return sp;
}
#endif  // VPIPE_BUILD_APPLE_SILICON

std::vector<std::int32_t>
ModelBenchmarkStage::real_prompt_(int n) const
{
  std::vector<std::int32_t> out;
  if (n <= 0 || _real_corpus.empty()) { return out; }
  out.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    out.push_back(
        _real_corpus[static_cast<std::size_t>(i) % _real_corpus.size()]);
  }
  return out;
}

bool
ModelBenchmarkStage::benchmark_once(const std::function<bool()>& stop)
{
  (void)stop;
#ifdef VPIPE_BUILD_APPLE_SILICON
  auto* mgr = session() ? session()->generative_model_manager() : nullptr;
  if (mgr == nullptr) {
    session()->warn(fmt(
        "ModelBenchmarkStage('{}'): no GenerativeModelManager (is this "
        "an apple-silicon build?)", this->id()));
    return false;
  }

  const std::string dir =
      resolve_model_dir(session(), _models_db, _model);
  session()->info(fmt(
      "ModelBenchmarkStage('{}'): loading '{}'", this->id(), _model));
  genai::LoadSpec spec;
  spec.hf_dir = dir;
  std::shared_ptr<genai::LoadedLanguageModel> lm = mgr->load(spec);
  if (!lm || !lm->valid()) {
    session()->warn(fmt(
        "ModelBenchmarkStage('{}'): failed to load '{}'", this->id(),
        _model));
    return false;
  }

  // Pin the working set wired-resident for the whole benchmark.
  auto wired = lm->wired_scope();
  const int vocab = lm->config().vocab_size;

  if (stop()) {
    session()->info(fmt(
        "ModelBenchmarkStage('{}'): stop requested; skipping benchmark",
        this->id()));
    return true;
  }

  using clock = std::chrono::steady_clock;

  // Warmup (discarded): a short prefill + decode primes kernels + the
  // residency set so the first timed test isn't penalised.
  for (int w = 0; w < _warmup; ++w) {
    auto ctx = lm->make_context();
    const auto p = make_prompt_(64, vocab, _seed);
    lm->prefill(ctx, p);
    lm->next_token_greedy(ctx);
  }

  // Progress bar (-> info()): one tick per timed test. Per-step detail
  // goes to log_normal() so info() carries just the bar + final report.
  const int total = static_cast<int>(_contexts.size()) * 3;
  int step = 0;
  // Redraw the bar in place (carriage-return) on a live text stream rather
  // than emitting a new info() line per tick. Padded to a stable width so a
  // shorter redraw fully overwrites a longer prior frame.
  std::unique_ptr<UiTextStream> bar_stream = session()->open_text_stream();
  auto progress = [&](const std::string& label) {
    constexpr int width = 24;
    const double frac =
        total > 0 ? static_cast<double>(step) / total : 1.0;
    int fill = static_cast<int>(frac * width + 0.5);
    fill = std::min(width, std::max(0, fill));
    std::string bar(static_cast<std::size_t>(fill), '#');
    bar += std::string(static_cast<std::size_t>(width - fill), '-');
    std::string line = fmt("\r[{}] {:.0f}% ({}/{}) {}", bar, frac * 100.0,
                           step, total, label)();
    while (line.size() < 72) { line += ' '; }   // wipe stale tail
    bar_stream->write(line);
  };

  // Set once a stop is observed; gates the remaining tests + the report note.
  bool stopped = false;

  // Sleep `cooldown_s` between tests so the GPU thermals settle, but poll the
  // stop predicate in short slices so a stop during the pause is responsive
  // (the very last test, step == total, never sleeps).
  auto cooldown = [&]() {
    if (_cooldown_s <= 0.0 || step >= total) { return; }
    const auto end = clock::now() +
        std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(_cooldown_s));
    while (clock::now() < end) {
      if (stop()) { stopped = true; return; }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  };

  std::vector<Row> prefill_rows;
  std::vector<Row> chunk_rows;
  std::vector<Row> decode_rows;

  // 1. Full prefill -- time a whole N-token prompt.
  for (int n : _contexts) {
    if (stop()) { stopped = true; break; }
    progress(fmt("prefill {} tok", n)());
    Row r; r.ctx = n;
    auto ctx = lm->make_context();
    const auto prompt = make_prompt_(n, vocab, _seed);
    GpuTelemetrySampler tel;
    tel.start();
    const auto t0 = clock::now();
    const std::int32_t next = lm->prefill(ctx, prompt);
    const auto t1 = clock::now();
    r.tel = tel.stop();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    if (next >= 0 && sec > 0.0) {
      r.tok_s = static_cast<double>(n) / sec;
      r.ok    = true;
    }
    session()->log_normal(fmt(
        "[bench] prefill {} tok: {} tok/s, {} MHz, {}C, {} MB", n,
        toks_cell_(r), mhz_cell_(r.tel.freq_mhz),
        temp_cell_(r.tel.temp_c), mem_cell_(r.tel.footprint_mb)));
    ++step;
    cooldown();
    prefill_rows.push_back(std::move(r));
  }

  // 2. Incremental prefill -- a `prefill_chunk`-token prefill once the KV
  //    already holds N tokens.
  for (int depth : _contexts) {
    if (stopped || stop()) { stopped = true; break; }
    progress(fmt("chunk @{}", depth)());
    Row r; r.ctx = depth;
    auto ctx = lm->make_context();
    const auto fill = make_prompt_(depth, vocab, _seed);
    if (lm->prefill(ctx, fill) >= 0) {
      const auto chunk = make_prompt_(
          _prefill_chunk, vocab, _seed + static_cast<std::uint64_t>(depth));
      GpuTelemetrySampler tel;
      tel.start();
      const auto t0 = clock::now();
      const std::int32_t next = lm->prefill(ctx, chunk);
      const auto t1 = clock::now();
      r.tel = tel.stop();
      const double sec = std::chrono::duration<double>(t1 - t0).count();
      if (next >= 0 && sec > 0.0) {
        r.tok_s = static_cast<double>(_prefill_chunk) / sec;
        r.ok    = true;
      }
    }
    session()->log_normal(fmt(
        "[bench] chunk @{}: {} tok/s, {} MHz, {}C, {} MB", depth,
        toks_cell_(r), mhz_cell_(r.tel.freq_mhz),
        temp_cell_(r.tel.temp_c), mem_cell_(r.tel.footprint_mb)));
    ++step;
    cooldown();
    chunk_rows.push_back(std::move(r));
  }

  // 3. Decode -- time decoding `decode_tokens` tokens after an N-token
  //    prefill. The production pdecode run-ahead (depth-2) path drives
  //    decode by default; when `mtp` is set and the model carries an MTP
  //    head the decode test runs MTP speculative decode primed with real
  //    text and records the prediction acceptance rate.
  const int                  n_decode = _decode_tokens;
  const genai::SamplerParams sp       = sampler_params_();
  bool use_mtp = _mtp && lm->mtp_available();
  bool mtp_unverified = false;
  genai::SamplerParams sp_verify = sp;
  if (use_mtp && !stopped) {
    // Production setting: prefix-seed the drafter for higher acceptance.
    lm->set_mtp_prefix_seed(true);
    // The MTP verify ignores repetition / presence penalties, so strip
    // them (and warn if the user set any).
    if (sp.repetition_penalty != 1.0f || sp.presence_penalty != 0.0f) {
      session()->log_normal(fmt(
          "[bench] MTP verify ignores repetition/presence penalties; "
          "they are dropped for the decode test"));
    }
    sp_verify.repetition_penalty = 1.0f;
    sp_verify.presence_penalty   = 0.0f;
    // Build the real-text decode corpus once: flatten the built-in
    // calibration corpus into one long token stream (tiled per context).
    const int seq_len = _contexts.empty() ? 1 : _contexts.back();
    auto seqs = genai::build_builtin_calibration_corpus(
        lm->tokenizer(), 8, std::max(1, seq_len), false);
    _real_corpus.clear();
    for (const auto& s : seqs) {
      for (std::int32_t t : s) { _real_corpus.push_back(t); }
    }
    // Self-check: MTP greedy vs serial greedy. A divergence does NOT mean MTP
    // is broken -- the speculative verify still only emits the argmax of a
    // valid forward pass, so MTP output is a valid greedy decode. It just is
    // not bit-identical to the single-row serial decode (expected in bf16: the
    // batched verify GEMV isn't bit-identical to the single-row qmv; f16 masks
    // the ~1 ULP, bf16 occasionally flips an argmax and cascades). So by
    // default we note it and benchmark MTP anyway; mtp_require_exact restores
    // the strict fallback for callers that need a bit-exact greedy drop-in.
    if (!stop() && !mtp_matches_serial_(*lm, real_prompt_(128), 16)) {
      mtp_unverified = true;
      if (_mtp_require_exact) {
        session()->warn(fmt(
            "ModelBenchmarkStage('{}'): MTP greedy diverged from serial "
            "greedy and mtp_require_exact is set -- falling back to pdecode",
            this->id()));
        use_mtp = false;
      } else {
        session()->log_normal(fmt(
            "ModelBenchmarkStage('{}'): NOTE -- MTP greedy diverges from "
            "serial greedy for this model/dtype (expected in bf16: the "
            "batched speculative verify is not bit-identical to the "
            "single-row decode; both are valid greedy decodes, so MTP output "
            "will not match non-MTP greedy token-for-token). Benchmarking MTP "
            "anyway; set mtp_require_exact=true to fall back to pdecode.",
            this->id()));
      }
    }
  }
  session()->log_normal(fmt(
      "[bench] decode path: {}",
      use_mtp
          ? (mtp_unverified
                 ? "MTP speculative (real text, prefix-seed); NOT bit-exact "
                   "vs serial greedy in this dtype"
                 : "MTP speculative (real text, prefix-seed)")
          : (mtp_unverified
                 ? "pdecode run-ahead (depth 2); MTP diverged + "
                   "mtp_require_exact"
                 : "pdecode run-ahead (depth 2)")));

  for (int n : _contexts) {
    if (stopped || stop()) { stopped = true; break; }
    progress(fmt("decode @{}", n)());
    Row r; r.ctx = n;
    auto ctx = lm->make_context();
    std::vector<std::int32_t> prompt;
    if (use_mtp) {
      prompt = real_prompt_(n);
      if (prompt.empty()) { prompt = make_prompt_(n, vocab, _seed); }
    } else {
      prompt = make_prompt_(n, vocab, _seed);
    }
    const std::int32_t first = lm->prefill(ctx, prompt);
    if (first >= 0) {
      GpuTelemetrySampler tel;
      int decoded = 0;
      if (use_mtp) {
        // One on_tokens callback per speculation round; sum its spans for
        // total accepted tokens. acceptance = produced/rounds - 1 (the
        // fraction of drafted tokens accepted) for depth-1 MTP.
        int  rounds = 0;
        auto on_tok = [&](std::span<const std::int32_t> s) {
          ++rounds;
          (void)s;
          return !stop();   // returning false aborts the MTP decode early
        };
        auto never_stop = [](std::int32_t) { return false; };
        int  produced = 0;
        tel.start();
        const auto t0 = clock::now();
        lm->mtp_generate(ctx, first, n_decode + 1, sp_verify, never_stop,
                         on_tok, &produced, nullptr);
        const auto t1 = clock::now();
        r.tel = tel.stop();
        const double sec =
            std::chrono::duration<double>(t1 - t0).count();
        decoded = produced;
        if (rounds > 0) {
          r.accept =
              static_cast<double>(produced) / rounds - 1.0;
        }
        if (decoded > 0 && sec > 0.0) {
          r.tok_s = static_cast<double>(decoded) / sec;
          r.ok    = true;
        }
      } else {
        tel.start();
        const auto t0 = clock::now();
        const bool pipe =
            lm->pdecode_begin(ctx, first, {}, sp, n_decode + 1);
        if (pipe) {
          const bool ra = lm->pdecode_supports_runahead();
          int inflight = 0;
          if (lm->pdecode_commit(ctx)) { ++inflight; }
          if (ra && inflight > 0 && n_decode > 1) {
            if (lm->pdecode_commit(ctx)) { ++inflight; }
          }
          while (decoded < n_decode && inflight > 0 && !stop()) {
            const std::int32_t nx = lm->pdecode_next(ctx);
            --inflight;
            if (nx < 0) { break; }
            ++decoded;
            if (decoded + inflight < n_decode) {
              if (lm->pdecode_commit(ctx)) { ++inflight; }
            }
          }
          lm->pdecode_end(ctx);
        } else {
          for (int k = 0; k < n_decode && !stop(); ++k) {
            if (lm->next_token_greedy(ctx) < 0) { break; }
            ++decoded;
          }
        }
        const auto t1 = clock::now();
        r.tel = tel.stop();
        const double sec =
            std::chrono::duration<double>(t1 - t0).count();
        if (decoded > 0 && sec > 0.0) {
          r.tok_s = static_cast<double>(decoded) / sec;
          r.ok    = true;
        }
      }
    }
    if (decode_implausible_(r)) {
      r.implausible = true;
      session()->warn(fmt(
          "ModelBenchmarkStage('{}'): decode @{} = {:.1f} tok/s exceeds "
          "the bandwidth ceiling for this model's resident size (~{:.0f} "
          "GB); the decode likely did not run the full model per token",
          this->id(), n, r.tok_s,
          r.tel.footprint_mb.max / 1024.0));
    }
    std::string accept_note;
    if (use_mtp && r.accept >= 0.0) {
      accept_note = fmt(", MTP accept {:.0f}%", r.accept * 100.0)();
    }
    session()->log_normal(fmt(
        "[bench] decode @{}: {} tok/s, {} MHz, {}C, {} MB{}", n,
        toks_cell_(r), mhz_cell_(r.tel.freq_mhz),
        temp_cell_(r.tel.temp_c), mem_cell_(r.tel.footprint_mb),
        accept_note));
    ++step;
    cooldown();
    decode_rows.push_back(std::move(r));
  }
  progress(stopped ? "stopped" : "complete");
  bar_stream->end();   // finalize the bar line before the Markdown report

  // ---- Markdown report ----
  const std::string gpu = GpuTelemetrySampler::gpu_model();
  const std::uint64_t cores = GpuTelemetrySampler::gpu_core_count();
  std::string thermal = "n/a";
  if (!prefill_rows.empty() && !prefill_rows.back().tel.thermal_state
                                    .empty()) {
    thermal = prefill_rows.back().tel.thermal_state;
  }

  std::string md;
  md += fmt("## Model benchmark: {}\n\n", _model)();
  md += fmt("- GPU: {} ({} cores)\n",
            gpu.empty() ? std::string("unknown") : gpu, cores)();
  md += fmt("- decode tokens/test: {}, prefill chunk: {}\n",
            _decode_tokens, _prefill_chunk)();
  std::string dpath = use_mtp
      ? std::string("MTP speculative (real text, prefix-seed)")
      : std::string("pdecode run-ahead (depth 2)");
  if (mtp_unverified) {
    dpath += use_mtp
        ? std::string(" [not bit-exact vs serial greedy in this dtype -- a "
                      "valid decode, not a greedy drop-in]")
        : std::string(" [MTP diverged from serial greedy + mtp_require_exact "
                      "-> fell back]");
  }
  md += fmt("- decode path: {}\n", dpath)();
  std::string samp = _temperature <= 0.0
      ? std::string("greedy (argmax)")
      : fmt("temp {:.2f}, top_k {}, top_p {:.2f}, min_p {:.2f}, "
            "rep_penalty {:.2f}, presence_penalty {:.2f}, seed {}",
            _temperature, _top_k, _top_p, _min_p, _repetition_penalty,
            _presence_penalty, _seed)();
  md += fmt("- sampling: {}\n", samp)();
  md += fmt("- cooldown: {}s\n", _cooldown_s)();
  md += fmt("- thermal state during run: {}\n", thermal)();
  if (stopped) {
    const std::size_t done = prefill_rows.size() + chunk_rows.size() +
                             decode_rows.size();
    md += fmt("- **stopped early**: {} of {} tests completed; partial "
              "results below\n", done, total)();
  }

  md += "\n### Full prefill (whole prompt)\n";
  md += "| Context | Prefill tok/s | GPU MHz (avg) | GPU °C "
        "(avg/max) | Util% (avg) | RAM MB (peak) |\n";
  md += "|---|---|---|---|---|---|\n";
  for (const Row& r : prefill_rows) {
    md += fmt("| {} | {} | {} | {} | {} | {} |\n", r.ctx, toks_cell_(r),
              mhz_cell_(r.tel.freq_mhz), temp_cell_(r.tel.temp_c),
              util_cell_(r.tel.util_pct),
              mem_cell_(r.tel.footprint_mb))();
  }

  md += fmt("\n### Incremental prefill -- {}-token chunk at depth\n",
            _prefill_chunk)();
  md += "| Depth | Chunk tok/s | GPU MHz | GPU °C (avg/max) | "
        "Util% | RAM MB (peak) |\n";
  md += "|---|---|---|---|---|---|\n";
  for (const Row& r : chunk_rows) {
    md += fmt("| {} | {} | {} | {} | {} | {} |\n", r.ctx, toks_cell_(r),
              mhz_cell_(r.tel.freq_mhz), temp_cell_(r.tel.temp_c),
              util_cell_(r.tel.util_pct),
              mem_cell_(r.tel.footprint_mb))();
  }

  md += "\n### Decode at context\n";
  md += "| Context | Decode tok/s | GPU MHz | GPU °C (avg/max) | "
        "Util% | RAM MB (peak) | MTP accept% |\n";
  md += "|---|---|---|---|---|---|---|\n";
  for (const Row& r : decode_rows) {
    md += fmt("| {} | {} | {} | {} | {} | {} | {} |\n", r.ctx,
              toks_cell_(r), mhz_cell_(r.tel.freq_mhz),
              temp_cell_(r.tel.temp_c), util_cell_(r.tel.util_pct),
              mem_cell_(r.tel.footprint_mb), accept_cell_(r))();
  }
  bool any_impl = false;
  for (const Row& r : decode_rows) {
    if (r.implausible) { any_impl = true; }
  }
  if (any_impl) {
    md += "\n> **(!)** A flagged decode rate exceeds the memory-bandwidth "
          "ceiling for this model's resident size -- the decode almost "
          "certainly did not read the full model per token (a broken or "
          "unvalidated decode path). Treat it as invalid, not a real "
          "generation rate.\n";
  }

  session()->info(fmt("\n{}", md));
  return true;
#else
  session()->warn(fmt(
      "ModelBenchmarkStage('{}'): built without VPIPE_BUILD_APPLE_"
      "SILICON; the LM subsystem is unavailable", this->id()));
  return false;
#endif
}

Job
ModelBenchmarkStage::process(RuntimeContext& ctx)
{
  if (ctx.stop_requested()) {
    ctx.signal_done();
    co_return;
  }
  session()->info(fmt(
      "ModelBenchmarkStage('{}'): benchmarking '{}'", this->id(),
      _model));
  benchmark_once([&ctx] { return ctx.stop_requested(); });
  ctx.signal_done();
  co_return;
}

VPIPE_REGISTER_STAGE(ModelBenchmarkStage)
VPIPE_REGISTER_SPEC(ModelBenchmarkStage, kSpec)

}  // namespace vpipe
