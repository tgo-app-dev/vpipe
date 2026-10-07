// speculative-dflash.cc -- DFlash / DFlash 2 block-drafter speculative
// decode (generative-models/shared/dflash-drafter.h): the drafter against
// the reference's own model.py, and the decode token-exact against the
// plain serial one, greedy and sampled.
//
// Env (paths absolute):
//   VPIPE_DFLASH_TEST_TARGET  the target model (mlx-community/Qwen3.8-27B-4bit
//                             for incoai/Qwen3.8-27B-DFlash2)
//   VPIPE_DFLASH_TEST_DRAFT   the drafter directory
//   VPIPE_DFLASH_GOLDEN       a dir written by tools/dump_dflash_golden.py
//                             for that drafter + target (drafter test only)
//   VPIPE_DFLASH_GEN_TOKENS   decode length (default 96)
//   VPIPE_DFLASH_PROMPT       the user turn
//   VPIPE_DFLASH_BLOCKS       block lengths to sweep, "2,4,8"; 0 = adaptive
//   VPIPE_DFLASH_DRAFT_BITS   the drafter's precision (default 8)
//   VPIPE_DFLASH_DTYPE        f16 | bf16, the target's dtype (default bf16;
//                             f16 for the sampled test)
//   VPIPE_DFLASH_DEBUG_LAYERS compare every drafter layer to the golden's
//   VPIPE_DFLASH_DUMP         ... and write vpipe's layers to this dir

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

#include "common/flex-bag.h"
#include "common/flex-data.h"
#include "generative-models/shared/dflash-drafter.h"
#include "generative-models/speculative-decode.h"

#include <filesystem>

namespace {

namespace fs = std::filesystem;

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

float
from_bf16_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

template <class T>
std::vector<T>
read_raw_(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  std::vector<T> v;
  if (!f) { return v; }
  f.seekg(0, std::ios::end);
  const std::size_t n = (std::size_t)f.tellg() / sizeof(T);
  f.seekg(0);
  v.resize(n);
  f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * sizeof(T)));
  return v;
}

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

void
write_text_(const fs::path& p, const std::string& s)
{
  std::ofstream f(p);
  f << s;
}

// The two configs this tree is exercised on, verbatim in shape.
constexpr const char* kDFlash2Config = R"({
  "architectures": ["DFlash2DraftModel"], "is_causal": false,
  "dflash_config": {"block_size": 8, "conv_group_size": 16,
    "conv_kernel_size": 2, "mask_token_id": 248070, "selector_rank": 256,
    "selector_top_k": 16, "target_layer_ids": [5, 19, 33, 47, 61]},
  "head_dim": 128, "hidden_act": "silu", "hidden_size": 5120,
  "intermediate_size": 17408,
  "layer_types": ["sliding_attention", "sliding_attention",
    "sliding_attention", "sliding_attention", "sliding_attention"],
  "model_type": "qwen3", "num_attention_heads": 32,
  "num_hidden_layers": 5, "num_key_value_heads": 8,
  "num_target_layers": 64, "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
  "sliding_window": 2048, "vocab_size": 248320})";

constexpr const char* kDFlash1Config = R"({
  "architectures": ["DFlashDraftModel"],
  "dflash_config": {"block_size": 16, "mask_token_id": 248077,
    "target_layer_ids": [1, 5, 9, 13, 17, 21, 25, 29]},
  "head_dim": 128, "hidden_act": "silu", "hidden_size": 2560,
  "intermediate_size": 9216,
  "layer_types": ["sliding_attention", "sliding_attention",
    "sliding_attention", "sliding_attention", "sliding_attention",
    "full_attention"],
  "num_attention_heads": 32, "num_hidden_layers": 6,
  "num_key_value_heads": 8, "num_target_layers": 32,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
  "sliding_window": 4096, "vocab_size": 248320})";

}  // namespace

// The config reader: what each field turns into, for both versions --
// in particular the attention masks, which the reference derives from
// layer_types and is_causal rather than stating.
TEST(dflash, config_parse) {
  const fs::path dir = fs::temp_directory_path() / "vpipe-dflash-cfg";
  fs::create_directories(dir / "v2");
  fs::create_directories(dir / "v1");
  fs::create_directories(dir / "no");
  write_text_(dir / "v2" / "config.json", kDFlash2Config);
  write_text_(dir / "v1" / "config.json", kDFlash1Config);
  write_text_(dir / "no" / "config.json",
              R"({"architectures": ["Qwen3ForCausalLM"]})");

  std::string err;
  auto c2 = genai::DFlashConfig::from_dir((dir / "v2").string(), &err);
  if (!c2) { std::printf("[dflash] v2 config: %s\n", err.c_str()); }
  ASSERT_TRUE(c2.has_value());
  if (c2) {
    EXPECT_TRUE(c2->v2());
    EXPECT_TRUE(std::string(c2->kind()) == "dflash2");
    EXPECT_TRUE(c2->block_size == 8);
    EXPECT_TRUE(c2->mask_token_id == 248070);
    EXPECT_TRUE((c2->target_layer_ids == std::vector<int>{5, 19, 33, 47, 61}));
    EXPECT_TRUE(c2->conv_kernel == 2 && c2->conv_group == 16);
    EXPECT_TRUE(c2->selector_rank == 256 && c2->selector_top_k == 16);
    EXPECT_TRUE(c2->qd() == 4096 && c2->kvd() == 1024);
    EXPECT_TRUE(c2->rope_theta == 1.0e7f);
    // is_causal false overrides "sliding => causal".
    for (int i = 0; i < 5; ++i) {
      EXPECT_TRUE(c2->layer_window[(std::size_t)i] == 2048);
      EXPECT_TRUE(!c2->layer_causal[(std::size_t)i]);
    }
  }
  auto c1 = genai::DFlashConfig::from_dir((dir / "v1").string(), &err);
  ASSERT_TRUE(c1.has_value());
  if (c1) {
    EXPECT_TRUE(!c1->v2());
    EXPECT_TRUE(c1->block_size == 16 && c1->target_layer_ids.size() == 8);
    // No is_causal: sliding layers are causal, the full one is not.
    for (int i = 0; i < 5; ++i) {
      EXPECT_TRUE(c1->layer_window[(std::size_t)i] == 4096);
      EXPECT_TRUE(c1->layer_causal[(std::size_t)i]);
    }
    EXPECT_TRUE(c1->layer_window[5] == 0 && !c1->layer_causal[5]);
  }
  EXPECT_TRUE(!genai::DFlashConfig::from_dir((dir / "no").string()));
  EXPECT_TRUE(genai::is_dflash_drafter_dir((dir / "v2").string()));
  EXPECT_TRUE(!genai::is_dflash_drafter_dir((dir / "no").string()));
  fs::remove_all(dir);
}

namespace {

struct DFlashFixture {
  std::unique_ptr<genai::MetalQwenModel> model;
  std::unique_ptr<genai::Tokenizer> tok;
};

// Load the target once and attach the drafter at `bits`: bf16 like
// text-chat unless `bf16` says otherwise; VPIPE_DFLASH_DTYPE=f16|bf16
// overrides either way. Null model when the env is unset.
DFlashFixture
load_fixture_(Session& sess, int bits, int gen, bool bf16_dflt = true)
{
  DFlashFixture fx;
  const char* target = std::getenv("VPIPE_DFLASH_TEST_TARGET");
  const char* draft = std::getenv("VPIPE_DFLASH_TEST_DRAFT");
  if (!target || !*target || !draft || !*draft) { return fx; }
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return fx; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(target);
  if (!cfg) { return fx; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  // f16 has 3 more mantissa bits, which make rounding-order splits rarer.
  const char* dte = std::getenv("VPIPE_DFLASH_DTYPE");
  bool bf16 = bf16_dflt;
  if (dte != nullptr && std::string(dte) == "f16") { bf16 = false; }
  if (dte != nullptr && std::string(dte) == "bf16") { bf16 = true; }
  mcfg.use_bf16 = bf16;
  mcfg.page_tokens = 512;
  mcfg.max_pages = std::max(16, (4 * (gen + 512)) / 512 + 8);
  fx.model = genai::MetalQwenModel::load(target, mc, mcfg);
  if (!fx.model) { return fx; }
  genai::MetalDFlashDrafter::Options opt;
  opt.quant_bits = bits;
  opt.target_bf16 = bf16;
  std::string err;
  auto d = genai::MetalDFlashDrafter::load(
      genai::WeightSet::open(draft, &sess), mc, opt, &err);
  if (!d) {
    std::printf("[dflash] drafter load failed: %s\n", err.c_str());
    fx.model.reset();
    return fx;
  }
  if (!fx.model->attach_dflash(std::move(d), &err)) {
    std::printf("[dflash] attach failed: %s\n", err.c_str());
    fx.model.reset();
    return fx;
  }
  fx.tok = genai::Tokenizer::from_huggingface_json(
      std::string(target) + "/tokenizer.json", &sess);
  return fx;
}

}  // namespace

// The drafter against z-lab/dflash's model.py on the golden's synthetic
// context: final hidden, the head's top-16 candidates, and the drafts
// (DFlash 2's selector path / DFlash's argmax), for two blocks -- the
// second after an incremental ingest. Dense first (the bar: bf16 against
// the reference's bf16), then the in-memory w8 / w4 holdings.
TEST(metal_lm_smoke, dflash_drafter_matches_reference) {
  const char* gdir = std::getenv("VPIPE_DFLASH_GOLDEN");
  if (!gdir || !*gdir) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  std::ifstream gj(std::string(gdir) + "/golden.json");
  if (!gj) { return; }
  const FlexData g = FlexData::from_json(gj);
  const FlexData::ConstObjectView gv = g.as_object();
  const int H = (int)gv.at("hidden").as_int(0);
  const int nT = (int)gv.at("n_taps").as_int(0);
  const int L = (int)gv.at("block").as_int(0);
  const int n1 = (int)gv.at("n1").as_int(0);
  const int n2 = (int)gv.at("n2").as_int(0);
  const FlexData anchors_fd = gv.at("anchors");
  std::vector<std::int32_t> anchors;
  {
    const FlexData::ConstArrayView a = anchors_fd.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      anchors.push_back((std::int32_t)a[i].as_int(0));
    }
  }
  const bool v2 = gv.at("kind").as_string() == "dflash2";

  Session sess;
  DFlashFixture fx = load_fixture_(sess, 0, 64);
  if (!fx.model) { return; }
  auto* mc = sess.metal_compute();
  for (const int bits : {0, 8, 4}) {
    if (bits != 0) {
      // Same target, the drafter re-held at `bits`.
      genai::MetalDFlashDrafter::Options opt;
      opt.quant_bits = bits;
      opt.target_bf16 = true;
      std::string err;
      auto nd = genai::MetalDFlashDrafter::load(
          genai::WeightSet::open(std::getenv("VPIPE_DFLASH_TEST_DRAFT"),
                                 &sess),
          mc, opt, &err);
      ASSERT_TRUE(nd != nullptr);
      if (!nd) { return; }
      ASSERT_TRUE(fx.model->attach_dflash(std::move(nd), &err));
    }
    genai::MetalDFlashDrafter* d = fx.model->dflash_mut();
    genai::DFlashTarget* tgt = fx.model->dflash_target();
    ASSERT_TRUE(d != nullptr && tgt != nullptr);
    if (!d || !tgt) { return; }
    EXPECT_TRUE((int)d->config().target_layer_ids.size() == nT);
    d->reset();
    d->set_debug_layers(std::getenv("VPIPE_DFLASH_DEBUG_LAYERS") != nullptr);
    auto ids = mc->make_shared_buffer((std::size_t)L * 4);
    auto out = mc->make_shared_buffer((std::size_t)L * 4);
    int start = 0;
    for (int b = 0; b < 2; ++b) {
      const int n = b == 0 ? n1 : n2;
      const std::vector<float> ctx =
          read_raw_<float>(std::string(gdir) + "/ctx" +
                           std::to_string(b + 1) + ".f32");
      ASSERT_TRUE(ctx.size() == (std::size_t)n * nT * H);
      if (ctx.size() != (std::size_t)n * nT * H) { return; }
      // Concat [n, nT*H] -> the target's layer-major taps [nT][n][H].
      auto taps = mc->make_shared_buffer((std::size_t)nT * n * H * 2);
      auto* tp = static_cast<std::uint16_t*>(taps.contents());
      for (int j = 0; j < nT; ++j) {
        for (int r = 0; r < n; ++r) {
          for (int h = 0; h < H; ++h) {
            tp[((std::size_t)j * n + r) * H + h] =
                to_bf16_(ctx[(std::size_t)r * nT * H + (std::size_t)j * H + h]);
          }
        }
      }
      d->fill_block_ids(static_cast<std::int32_t*>(ids.contents()),
                        anchors[(std::size_t)b], L);
      metal_compute::CommandStream st = mc->make_command_stream();
      {
        metal_compute::ComputeEncoder enc = st.begin_compute();
        d->encode_ingest(enc, taps, n, 0, n, start);
        d->encode_draft(enc, *tgt, ids, L, start + n, out, 0);
      }
      st.commit().wait();
      start += n;
      if (b == 0 && std::getenv("VPIPE_DFLASH_DEBUG_LAYERS") != nullptr) {
        const auto* dl =
            static_cast<const std::uint16_t*>(d->debug_layers().contents());
        for (int li = 0; li < d->config().n_layers && dl != nullptr; ++li) {
          const std::vector<float> rl = read_raw_<float>(
              std::string(gdir) + "/layer" + std::to_string(li) + ".f32");
          std::vector<float> gl((std::size_t)L * H);
          for (std::size_t i = 0; i < gl.size(); ++i) {
            gl[i] = from_bf16_(dl[(std::size_t)li * L * H + i]);
          }
          std::printf("[dflash_ref] w%d layer %d: rel-L2 %.5f (row0 %.5f)\n",
                      bits, li, rel_l2_(gl.data(), rl.data(), gl.size()),
                      rel_l2_(gl.data(), rl.data(), (std::size_t)H));
          if (const char* dp = std::getenv("VPIPE_DFLASH_DUMP")) {
            std::ofstream f(std::string(dp) + "/w" + std::to_string(bits) +
                                "_layer" + std::to_string(li) + ".f32",
                            std::ios::binary);
            f.write(reinterpret_cast<const char*>(gl.data()),
                    (std::streamsize)(gl.size() * 4));
          }
        }
      }

      const std::string sfx = std::to_string(b + 1);
      const std::vector<float> ref_h =
          read_raw_<float>(std::string(gdir) + "/hidden" + sfx + ".f32");
      ASSERT_TRUE(ref_h.size() == (std::size_t)(L - 1) * H);
      std::vector<float> got_h((std::size_t)(L - 1) * H);
      const auto* hp =
          static_cast<const std::uint16_t*>(d->last_hidden().contents());
      for (std::size_t i = 0; i < got_h.size(); ++i) {
        got_h[i] = from_bf16_(hp[(std::size_t)H + i]);   // rows 1 ..
      }
      const double rel = rel_l2_(got_h.data(), ref_h.data(), got_h.size());
      // The yardstick: fp32. The residual stream runs to tens of thousands,
      // so bf16 absorption alone moves a row by percents; vpipe's bf16 is
      // held to its distance from fp32 against the reference's own bf16.
      const std::vector<float> f32_h = read_raw_<float>(
          std::string(gdir) + "/hidden" + sfx + "_fp32.f32");
      double rel32 = -1.0, ref32 = -1.0;
      if (f32_h.size() == got_h.size()) {
        rel32 = rel_l2_(got_h.data(), f32_h.data(), got_h.size());
        ref32 = rel_l2_(ref_h.data(), f32_h.data(), got_h.size());
      }
      const std::vector<std::int32_t> ref_path =
          read_raw_<std::int32_t>(std::string(gdir) + "/path" + sfx + ".i32");
      const auto* op = static_cast<const std::int32_t*>(out.contents());
      int path_eq = 0;
      for (int i = 0; i < L - 1; ++i) {
        if (op[i] == ref_path[(std::size_t)i]) { ++path_eq; }
      }
      int cand_hits = 0, cand_tot = 0;
      if (v2) {
        const std::vector<std::int32_t> rc = read_raw_<std::int32_t>(
            std::string(gdir) + "/cand" + sfx + "_idx.i32");
        const auto* gc =
            static_cast<const std::int32_t*>(d->last_cand_idx().contents());
        for (int r = 0; r < L - 1; ++r) {
          for (int k = 0; k < 16; ++k) {
            const std::int32_t want = rc[(std::size_t)r * 16 + k];
            ++cand_tot;
            for (int q = 0; q < 16; ++q) {
              if (gc[r * 16 + q] == want) { ++cand_hits; break; }
            }
          }
        }
      }
      std::printf("[dflash_ref] w%d block %d: hidden vs ref-bf16 %.4f, vs "
                  "fp32 %.4f (ref-bf16 vs fp32 %.4f) | top-16 %d/%d | path "
                  "%d/%d\n", bits, b + 1, rel, rel32, ref32, cand_hits,
                  cand_tot, path_eq, L - 1);
      if (bits == 0) {
        // Dense bf16: as far from fp32 as the reference's bf16 is, give or
        // take; and the same drafts but for near-ties.
        if (ref32 >= 0.0) { EXPECT_TRUE(rel32 < 2.0 * ref32 + 0.01); }
        EXPECT_TRUE(path_eq >= L - 3);
        if (v2) { EXPECT_TRUE(cand_hits >= cand_tot - 12); }
      } else if (bits == 8) {
        if (ref32 >= 0.0) { EXPECT_TRUE(rel32 < 2.5 * ref32 + 0.02); }
      }
    }
  }
}

namespace {

std::vector<std::int32_t>
prompt_ids_(genai::Tokenizer& tok)
{
  const char* p = std::getenv("VPIPE_DFLASH_PROMPT");
  const std::string user = (p && *p) ? p
      : "Write a short Python function that returns the n-th Fibonacci "
        "number iteratively, then explain in two sentences why it is O(n).";
  // The Qwen3.x chat frame, thinking closed, so the run is the answer.
  const std::string text = "<|im_start|>user\n" + user +
                           "<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
                           "</think>\n\n";
  return tok.encode(text);
}

std::int32_t
argmax_(const std::vector<float>& v)
{
  std::int32_t best = 0;
  float bv = v.empty() ? 0.0f : v[0];
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
  }
  return best;
}

// The split-K verify GEMV is the default wherever the target runs the
// affine GEMVs (not k-quant, not dense) unless VPIPE_QMV_KSPLIT=0.
bool
ksplit_expected_(const genai::MetalQwenModel& m)
{
  const char* e = std::getenv("VPIPE_QMV_KSPLIT");
  return (e == nullptr || std::atoi(e) >= 1) && !m.uses_kquant()
         && !m.uses_dense();
}

int
gen_tokens_()
{
  if (const char* e = std::getenv("VPIPE_DFLASH_GEN_TOKENS")) {
    const int v = std::atoi(e);
    if (v > 0) { return v; }
  }
  return 96;
}

}  // namespace

// GREEDY DFlash decode is token-exact with the serial greedy loop from the
// identical prefilled state, at the trained block and a smaller one; the
// drafter must also draft well (more than two tokens a round on code).
TEST(metal_lm_smoke, dflash_speculative_token_exact) {
  ::unsetenv("VPIPE_LLM_BACKEND");
  int bits = 8;
  if (const char* e = std::getenv("VPIPE_DFLASH_DRAFT_BITS")) {
    bits = std::atoi(e);
  }
  const int kGen = gen_tokens_();
  Session sess;
  DFlashFixture fx = load_fixture_(sess, bits, kGen);
  if (!fx.model || !fx.tok) { return; }
  genai::MetalQwenModel& m = *fx.model;
  const std::vector<std::int32_t> ids = prompt_ids_(*fx.tok);
  ASSERT_TRUE(!ids.empty());

  const std::vector<float> lg = m.prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax_(lg);
  const int Lt = m.dflash()->config().block_size;
  std::vector<int> blocks = {Lt, 2, 0};
  if (const char* e = std::getenv("VPIPE_DFLASH_BLOCKS")) {
    // A sweep, e.g. "2,3,4,5,6,8".
    blocks.clear();
    std::stringstream ss(e);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      // 0 = adaptive (the tuner picks a length per round).
      const int v = std::atoi(tok.c_str());
      if (v == 0 || v >= 2) { blocks.push_back(v); }
    }
  }
  std::vector<genai::ContextId> kids;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    kids.push_back(m.context_manager()->branch(m.root_context()));
    ASSERT_TRUE(kids.back().valid());
  }

  // The serial reference, with each step's top-2 logits: bf16 logits
  // carry 8 bits, so two decode paths that round in a different order can
  // split a 1-ulp near-tie -- vpipe's own forward() and forward_argmax()
  // do, on the 4B. A divergence is accepted ONLY at such a tie; up to it
  // the tokens must be identical.
  std::vector<std::int32_t> ref{first};
  std::vector<float> margin, top;
  for (int i = 0; i < kGen; ++i) {
    const std::vector<float> lv = m.forward(ref.back());
    if (lv.empty()) { break; }
    const std::int32_t t = argmax_(lv);
    float second = -1e30f;
    for (std::size_t v = 0; v < lv.size(); ++v) {
      if ((std::int32_t)v != t && lv[v] > second) { second = lv[v]; }
    }
    margin.push_back(lv[(std::size_t)t] - second);
    top.push_back(lv[(std::size_t)t]);
    ref.push_back(t);
  }
  // The speed baseline: the production greedy loop, on its own branch.
  double serial_tps = 0.0;
  {
    const genai::ContextId sb =
        m.context_manager()->branch(m.root_context());
    std::vector<std::int32_t> pids;
    const auto s0 = std::chrono::steady_clock::now();
    EXPECT_TRUE(m.decode_pipelined(sb, first, std::min(kGen, 64), pids));
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - s0).count();
    serial_tps = (double)pids.size() / (ms / 1e3);
  }
  std::printf("[dflash] %s w%d: prompt %zu tok, serial %.1f tok/s\n",
              m.dflash()->config().kind(), bits, ids.size(), serial_tps);
  const std::uint64_t ks0 = m.qmv_ksplit_dispatches();

  for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
    std::vector<std::int32_t> got;
    genai::MetalQwenModel::SpecStats st;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = m.dflash_decode(kids[bi], first, (int)ref.size(), got,
                                    {}, blocks[bi], &st);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    EXPECT_TRUE(ok);
    int mism = 0, first_bad = -1;
    for (std::size_t i = 0; i < std::min(ref.size(), got.size()); ++i) {
      if (ref[i] != got[i]) {
        ++mism;
        if (first_bad < 0) { first_bad = (int)i; }
      }
    }
    bool tie_ok = true;
    if (first_bad > 0) {
      const std::size_t k = (std::size_t)first_bad - 1;
      const float mg = k < margin.size() ? margin[k] : -1.0f;
      const float tv = k < top.size() ? std::fabs(top[k]) : 0.0f;
      // Two bf16 ulps at the top logit's magnitude.
      const float ulp = std::ldexp(1.0f, (int)std::floor(std::log2(
                            std::max(tv, 1e-3f))) - 7);
      tie_ok = mg >= 0.0f && mg <= 2.0f * ulp;
      std::printf("[dflash] first divergence at %d: serial %d vs spec %d, "
                  "serial top-2 margin %.4f (2 ulp %.4f) -> %s\n",
                  first_bad, ref[(std::size_t)first_bad],
                  got[(std::size_t)first_bad], mg, 2.0f * ulp,
                  tie_ok ? "a near-tie" : "NOT a tie");
    }
    const double tpr = st.rounds > 0 ? (double)got.size() / st.rounds : 0.0;
    const double tps = (double)got.size() / (ms / 1e3);
    std::printf("[dflash] block %s: %zu tok in %.0f ms (%.1f tok/s, %.2fx) "
                "| rounds %ld, %.2f tok/round, accepted %ld/%ld | mism %d\n",
                blocks[bi] == 0 ? "auto" : std::to_string(blocks[bi]).c_str(),
                got.size(), ms, tps, tps / serial_tps, st.rounds,
                tpr, st.accepted, st.drafted, mism);
    EXPECT_TRUE(first_bad < 0 || (first_bad > 0 && tie_ok));
    EXPECT_TRUE(got.size() == ref.size());
    // The drafts must land, or "exact" means only that every round kept
    // its anchor: past 2 a round from 4 up, past 1.2 for 2 and adaptive.
    EXPECT_TRUE(tpr > (blocks[bi] >= 4 ? 2.0 : 1.2));
  }
  // The verify ran on the split-K tiles where they are the default.
  if (ksplit_expected_(m)) {
    EXPECT_TRUE(m.qmv_ksplit_dispatches() > ks0);
  }
}

// SAMPLED DFlash decode is token-exact with the GPU-resident pipelined
// sampler (pdecode) under the same seed: every committed token is the
// verifier's per-slot sample; the drafts only decide how many land a
// round. With and without penalties (the penalised path builds per-row
// seen sets from the drafts, a second command buffer a round).
//
// In F16, as the MTP sampled tests run: a sample flips when the draw
// lands within rounding of a CDF boundary, and with bf16's 8 bits two
// paths that sum in a different order cross one within a few dozen
// tokens -- run to run, as the plain pipelined decode does. The drafts
// must still land (more than one token a round), or "exact" means only
// that every round kept its anchor.
TEST(metal_lm_smoke, dflash_speculative_sampled_token_exact) {
  ::unsetenv("VPIPE_LLM_BACKEND");
  const int kGen = std::min(gen_tokens_(), 64);
  Session sess;
  DFlashFixture fx = load_fixture_(sess, 8, kGen, /*bf16_dflt=*/false);
  if (!fx.model || !fx.tok) { return; }
  genai::MetalQwenModel& m = *fx.model;
  const std::vector<std::int32_t> ids = prompt_ids_(*fx.tok);
  const std::vector<float> lg = m.prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax_(lg);

  // Exactness against the pipelined sampler needs the verify to round as
  // the single-row decode does, so the first pass pins the row-identical
  // GEMV tiles. The split-K default (another reduction order) then makes
  // the same draws, held only to "the drafts land, the length is right":
  // a 1-ulp logit difference moves a CDF boundary rarely, and when it does
  // the run takes another, equally valid path from that token on.
  for (const bool ks : {false, true}) {
  if (ks && !ksplit_expected_(m)) { continue; }
  m.set_qmv_ksplit(ks, false);
  const std::uint64_t ks0 = m.qmv_ksplit_dispatches();
  for (const bool penal : {false, true}) {
    genai::GpuSamplerParams sp;
    sp.greedy = false;
    sp.temperature = 1.0f;
    sp.top_k = 20;
    sp.top_p = 0.95f;
    sp.seed = 20261005ull;
    if (const char* e = std::getenv("VPIPE_DFLASH_SEED")) {
      sp.seed = std::strtoull(e, nullptr, 10);
    }
    if (penal) { sp.repetition_penalty = 1.1f; sp.presence_penalty = 0.2f; }
    const genai::ContextId a =
        m.context_manager()->branch(m.root_context());
    const genai::ContextId b =
        m.context_manager()->branch(m.root_context());
    ASSERT_TRUE(a.valid() && b.valid());
    // Reference: the pipelined sampler, one token at a time.
    std::vector<std::int32_t> ref{first};
    ASSERT_TRUE(m.pdecode_begin(a, first, ids, sp, kGen + 1));
    for (int i = 0; i < kGen; ++i) {
      if (!m.pdecode_commit(a)) { break; }
      const std::int32_t t = m.pdecode_next(a);
      if (t < 0) { break; }
      ref.push_back(t);
    }
    m.pdecode_end(a);
    genai::MtpDecodeCtl ctl;
    ctl.sampler = sp;
    ctl.prime = ids;
    std::vector<std::int32_t> got;
    genai::MetalQwenModel::SpecStats st;
    EXPECT_TRUE(m.dflash_decode(b, first, (int)ref.size(), got, ctl, 0, &st));
    int mism = 0, first_bad = -1;
    for (std::size_t i = 0; i < std::min(ref.size(), got.size()); ++i) {
      if (ref[i] != got[i]) {
        ++mism;
        if (first_bad < 0) { first_bad = (int)i; }
      }
    }
    if (std::getenv("VPIPE_DFLASH_DUMP_TOKENS")) {
      std::printf("[dflash] ref:");
      for (std::size_t i = 0; i < std::min<std::size_t>(12, ref.size()); ++i) {
        std::printf(" %d", ref[i]);
      }
      std::printf("\n[dflash] got:");
      for (std::size_t i = 0; i < std::min<std::size_t>(12, got.size()); ++i) {
        std::printf(" %d", got[i]);
      }
      std::printf("\n");
    }
    std::printf("[dflash] sampled%s%s: %zu tok, %ld rounds (%.2f "
                "tok/round) | mism %d (first at %d)\n",
                ks ? " split-K" : "", penal ? " +penalties" : "",
                got.size(), st.rounds,
                st.rounds ? (double)got.size() / st.rounds : 0.0, mism,
                first_bad);
    if (!ks) { EXPECT_TRUE(mism == 0); }
    EXPECT_TRUE(got.size() == ref.size());
    EXPECT_TRUE(st.rounds > 0 && (double)got.size() / st.rounds > 1.2);
  }
  if (ks) { EXPECT_TRUE(m.qmv_ksplit_dispatches() > ks0); }
  }
}

// The FRAMEWORK path, as text-chat runs it: the drafter named in a
// LoadSpec (load_spec::kDraftDir), attached by the model manager, and
// two chat turns on one context decoded through LoadedLanguageModel::
// spec_generate against the serial greedy loop on another. The second
// turn is what exercises the drafter's context across a turn boundary:
// its cache holds the first turn's prompt and answer, then the new
// prefill's rows.
TEST(metal_lm_smoke, dflash_chat_through_manager) {
  const char* target = std::getenv("VPIPE_DFLASH_TEST_TARGET");
  const char* draft = std::getenv("VPIPE_DFLASH_TEST_DRAFT");
  if (!target || !*target || !draft || !*draft) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec spec;
  spec.hf_dir = target;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  bag::set_text(spec.extra, genai::load_spec::kDraftDir, draft);
  bag::set_integer(spec.extra, genai::load_spec::kDraftBits, 8);
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (!lm || !lm->valid()) { return; }
  ASSERT_TRUE(lm->spec_decode_available());
  const std::string kind = lm->spec_drafter();
  EXPECT_TRUE(kind == "dflash2" || kind == "dflash");
  // A second load of the same spec shares the model; one without the
  // drafter is a different cache entry.
  EXPECT_TRUE(mgr->load(spec).get() == lm.get());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);
  if (!tpl) { return; }
  auto is_stop = [tpl](std::int32_t id) { return tpl->is_stop_token(id); };
  const std::int32_t close_id = tpl->assistant_close_token_id();
  const int kBudget = std::min(gen_tokens_(), 128);
  const char* turns[2] = {
      "Write a Python function that checks whether a string is a "
      "palindrome, ignoring case and punctuation.",
      "Now write three unit tests for it with pytest."};

  // The reference for each turn runs on a BRANCH of the speculative
  // context taken right after the turn's prefill, so both decode from the
  // identical history -- including the previous turn's speculative answer
  // -- and a near-tie split in turn 1 cannot desynchronise turn 2.
  auto spec_ctx = lm->make_context();
  for (int t = 0; t < 2; ++t) {
    std::vector<std::int32_t> ids;
    tpl->render_user_turn(turns[t], /*is_first_turn=*/t == 0, &ids);
    ASSERT_TRUE(!ids.empty());
    const std::int32_t first = lm->prefill(spec_ctx, ids);
    ASSERT_TRUE(first >= 0);
    // Serial greedy on the branch, with each step's top-2 margin (the
    // near-tie rule of dflash_speculative_token_exact).
    std::vector<std::int32_t> ref;
    std::vector<float> margin, top;
    {
      auto ref_ctx = lm->branch(spec_ctx);
      std::int32_t cur = first;
      while ((int)ref.size() < kBudget && !is_stop(cur)) {
        ref.push_back(cur);
        cur = lm->next_token(ref_ctx, cur);
        if (cur < 0) { break; }
        const std::vector<float>& lv = lm->last_logits_host();
        float second = -1e30f;
        for (std::size_t v = 0; v < lv.size(); ++v) {
          if ((std::int32_t)v != cur && lv[v] > second) { second = lv[v]; }
        }
        margin.push_back(lv.empty() ? -1.0f : lv[(std::size_t)cur] - second);
        top.push_back(lv.empty() ? 0.0f : lv[(std::size_t)cur]);
      }
    }
    std::vector<std::int32_t> got;
    int produced = 0;
    bool hit = false;
    genai::SpecDecodeResult sr;
    auto on_toks = [&](std::span<const std::int32_t> toks) -> bool {
      for (std::int32_t id : toks) { got.push_back(id); }
      return true;
    };
    EXPECT_TRUE(lm->spec_generate(spec_ctx, first, kBudget,
                                  genai::SamplerParams{}, is_stop, on_toks,
                                  &produced, &hit, ids, 0, &sr));
    if (close_id >= 0) { lm->next_token_greedy(spec_ctx, close_id); }
    int mism = 0, first_mism = -1;
    for (std::size_t i = 0; i < std::min(ref.size(), got.size()); ++i) {
      if (ref[i] != got[i]) {
        if (first_mism < 0) { first_mism = (int)i; }
        ++mism;
      }
    }
    bool tie_ok = first_mism < 0;
    if (first_mism > 0) {
      const std::size_t k = (std::size_t)first_mism - 1;
      const float mg = k < margin.size() ? margin[k] : -1.0f;
      const float tv = k < top.size() ? std::fabs(top[k]) : 0.0f;
      const float ulp = std::ldexp(1.0f, (int)std::floor(std::log2(
                            std::max(tv, 1e-3f))) - 7);
      tie_ok = mg >= 0.0f && mg <= 2.0f * ulp;
      std::printf("[dflash_chat] divergence at %d: %d vs %d, margin %.4f "
                  "(2 ulp %.4f) -> %s\n", first_mism,
                  ref[(std::size_t)first_mism], got[(std::size_t)first_mism],
                  mg, 2.0f * ulp, tie_ok ? "a near-tie" : "NOT a tie");
    }
    std::printf("[dflash_chat] turn %d: prompt %zu, ref %zu got %zu, %ld "
                "rounds (%.2f a round), mism %d (first %d)\n", t + 1,
                ids.size(), ref.size(), got.size(), sr.rounds,
                sr.rounds ? (double)got.size() / sr.rounds : 0.0, mism,
                first_mism);
    EXPECT_TRUE(tie_ok);
  }
}
