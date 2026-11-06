// The session's kernel-tuning answers (KernelTuningCache,
// generative-models/shared/tuned-answers.h): kept for the session, read
// by the next tuner to ask the same question -- a model loaded again --
// and by nobody else: another session, another question, or a session
// told not to keep them measures as before.

#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/kernel-sets/decode-gqa-attn-set.h"
#include "generative-models/shared/kernel-sets/prefill-gqa-attn-set.h"
#include "generative-models/shared/kernel-sets/vae-conv3x3-tune.h"
#include "generative-models/shared/kernel-sets/vae-mid-attn-tune.h"
#include "generative-models/shared/mma-splitk.h"
#include "generative-models/shared/tuned-answers.h"
#include "generative-models/flux2/metal-flux2-vae.h"
#include "generative-models/krea2/metal-krea2-vae.h"
#include "generative-models/model-loader.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/shared/kernel-autotune.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/generate-audio-stage.h"
#include "stages/generate-image-stage.h"
#include "stages/generate-video-stage.h"
#include "stages/latent-upscale-stage.h"
#include "stages/text-to-speech-stage.h"
#include "stages/vae-decode-stage.h"
#include "stages/vae-encode-stage.h"
#include "stages/video-ref-encoder-stage.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;
using vpipe::genai::TuneKey;
using vpipe::genai::find_tuned;
using vpipe::genai::store_tuned;
using vpipe::genai::tuning_cache;

namespace {

MetalCompute*
matrix_cores_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid() || !mc->supports_matrix_cores()) {
    std::printf("[kernel_tuning] no matrix cores here; not exercised\n");
    return nullptr;
  }
  return mc;
}

}  // namespace

// The service itself: on by default, off by the config or the env (the
// env winning either way), its answers integers under whole keys, and
// what MetalCompute's session reaches is the session's own.
TEST(kernel_tuning, the_session_keeps_answers_unless_told_not_to)
{
  ::unsetenv("VPIPE_TUNING_CACHE");
  Session sess;
  KernelTuningCache* c = sess.services()->kernel_tuning_cache();
  ASSERT_TRUE(c != nullptr);
  if (c == nullptr) { return; }
  EXPECT_TRUE(sess.kernel_tuning_cache() == c);
  if (MetalCompute* mc = sess.metal_compute(); mc != nullptr) {
    EXPECT_TRUE(tuning_cache(mc) == c);
  }

  const TuneKey k = TuneKey("test/1")("N", 3072)("route", "requant")
      ("bf16", true)("zero", 0);
  EXPECT_TRUE(k.str() == "test/1 N=3072 route=requant bf16=1 zero=0");
  std::vector<int> v;
  EXPECT_TRUE(!find_tuned(c, k, v));
  store_tuned(c, k, std::vector<int>{2, 0, -1});
  EXPECT_TRUE(find_tuned(c, k, v));
  EXPECT_TRUE(v == (std::vector<int>{2, 0, -1}));
  // One integer asked of three is not an answer to that question.
  int one = 7;
  EXPECT_TRUE(!find_tuned(c, k, one) && one == 7);
  store_tuned(c, TuneKey("test/1 single"), 5);
  EXPECT_TRUE(find_tuned(c, TuneKey("test/1 single"), one) && one == 5);
  // Text that does not read as integers is no answer: measured again,
  // it is filed over.
  c->store("test/1 bad", "2,x");
  EXPECT_TRUE(!find_tuned(c, TuneKey("test/1 bad"), v));
  // A key is compared whole.
  EXPECT_TRUE(!find_tuned(c, TuneKey("test/1"), v));
  const KernelTuningCache::Stats st = c->stats();
  EXPECT_TRUE(st.entries == 3);
  EXPECT_TRUE(st.hits >= 3 && st.misses >= 2);

  // Another session keeps its own.
  Session other;
  EXPECT_TRUE(other.kernel_tuning_cache() != nullptr);
  EXPECT_TRUE(other.kernel_tuning_cache() != c);
  EXPECT_TRUE(!find_tuned(other.kernel_tuning_cache(), k, v));

  Session off(R"({"kernel_tuning": {"cache": false}})");
  EXPECT_TRUE(off.kernel_tuning_cache() == nullptr);
  EXPECT_TRUE(off.services()->kernel_tuning_cache() == nullptr);
  if (MetalCompute* mc = off.metal_compute(); mc != nullptr) {
    EXPECT_TRUE(tuning_cache(mc) == nullptr);
  }
  ::setenv("VPIPE_TUNING_CACHE", "0", 1);
  Session env_off;
  EXPECT_TRUE(env_off.kernel_tuning_cache() == nullptr);
  ::setenv("VPIPE_TUNING_CACHE", "1", 1);
  Session env_on(R"({"kernel_tuning": {"cache": false}})");
  EXPECT_TRUE(env_on.kernel_tuning_cache() != nullptr);
  ::unsetenv("VPIPE_TUNING_CACHE");
}

// THE i8 SPLIT, which every picture model tunes at its second step: a
// second context in the session -- the model loaded again -- runs the
// first one's answer from its very first call and records nothing to
// measure, at a similar size too (the same row bucket). Another element
// type, another session, or a session keeping none measures again.
TEST(kernel_tuning, i8_split_answers_are_kept_for_the_session)
{
  ::unsetenv("VPIPE_TUNING_CACHE");
  Session sess;
  MetalCompute* mc = matrix_cores_(sess);
  if (mc == nullptr) { return; }
  const int N = 512, K = 2048;
  auto gemm = [&](MetalCompute* m, genai::I8GemmContext& ctx, int M) {
    SharedBuffer x = m->make_shared_buffer((std::size_t)M * K * 2);
    SharedBuffer w = m->make_shared_buffer((std::size_t)N * K * 2);
    SharedBuffer y = m->make_shared_buffer((std::size_t)M * N * 2);
    bool ok = false;
    {
      CommandStream st = m->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        ok = ctx.gemm(e, x, 0, w, y, 0, M, N, K);
      }
      st.commit().wait();
    }
    return ok;
  };

  genai::I8GemmContext a(mc, /*want=*/true, /*bf16=*/false);
  if (!a.enabled() || !a.split_available()) {
    std::printf("[kernel_tuning] no i8 split kernels; not exercised\n");
    return;
  }
  ASSERT_TRUE(gemm(mc, a, 1536));
  EXPECT_TRUE(a.pending_for_test() == 1);
  a.tune_pending(mc);
  EXPECT_TRUE(a.measured_for_test() == 1);
  EXPECT_TRUE(a.tuned_for_test(1536, N, K));
  const int pick = a.plan_for_test(1536, N, K);

  // The model loaded again, at a similar size: 2048 rows share 1536's
  // bucket, so its answer.
  genai::I8GemmContext b(mc, /*want=*/true, /*bf16=*/false);
  EXPECT_TRUE(gemm(mc, b, 2048));
  EXPECT_TRUE(b.pending_for_test() == 0);
  EXPECT_TRUE(b.tuned_for_test(2048, N, K));
  EXPECT_TRUE(b.plan_for_test(2048, N, K) == pick);
  b.tune_pending(mc);
  EXPECT_TRUE(b.measured_for_test() == 0);
  // ...but not past it: 4096 rows is another bucket, another question.
  EXPECT_TRUE(gemm(mc, b, 4096));
  EXPECT_TRUE(b.pending_for_test() == 1);

  // Another element type is another question.
  genai::I8GemmContext c(mc, /*want=*/true, /*bf16=*/true);
  EXPECT_TRUE(gemm(mc, c, 1536));
  EXPECT_TRUE(c.pending_for_test() == 1);

  // Another session has none of it; one told to keep none never will.
  Session other;
  genai::I8GemmContext d(other.metal_compute(), true, false);
  EXPECT_TRUE(gemm(other.metal_compute(), d, 1536));
  EXPECT_TRUE(d.pending_for_test() == 1);
  Session off(R"({"kernel_tuning": {"cache": false}})");
  for (int i = 0; i < 2; ++i) {
    genai::I8GemmContext e(off.metal_compute(), true, false);
    EXPECT_TRUE(gemm(off.metal_compute(), e, 1536));
    EXPECT_TRUE(e.pending_for_test() == 1);
    e.tune_pending(off.metal_compute());
    EXPECT_TRUE(e.measured_for_test() == 1);
  }
  std::printf("[kernel_tuning] i8 split S=%d at %dx%dx%d, read by the next "
              "context\n", pick, 1536, N, K);
}

// THE DENSE SPLIT-K, which a DiT tunes through its own closure: a caller
// that names what the closure encodes has its answer kept, and the next
// instance asking the same never runs it. One that names nothing keeps
// its answer to itself, as before.
TEST(kernel_tuning, a_named_split_is_read_by_the_next_instance)
{
  ::unsetenv("VPIPE_TUNING_CACHE");
  Session sess;
  MetalCompute* mc = matrix_cores_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib_dense = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise");
  const int K = 12288, N = 512, M = 1024;
  int runs = 0;
  auto run = [&](ComputeEncoder&) { ++runs; };

  genai::MmaSplitK a;
  a.load(mc, lib_dense, lib_elt);
  if (!a.enabled) {
    std::printf("[kernel_tuning] no split-K kernels; not exercised\n");
    return;
  }
  const int pick = a.tune(mc, K, N, M, run, "kernel_tuning test");
  if (runs == 0) {
    std::printf("[kernel_tuning] one split candidate at %dx%dx%d; not "
                "exercised\n", M, N, K);
    return;
  }

  genai::MmaSplitK b;
  b.load(mc, lib_dense, lib_elt);
  runs = 0;
  EXPECT_TRUE(b.tune(mc, K, N, M, run, "kernel_tuning test") == pick);
  EXPECT_TRUE(runs == 0);
  // Another caller's closure is another question.
  genai::MmaSplitK c;
  c.load(mc, lib_dense, lib_elt);
  runs = 0;
  (void)c.tune(mc, K, N, M, run, "kernel_tuning other");
  EXPECT_TRUE(runs > 0);
  // Unnamed: this instance's alone, measured.
  genai::MmaSplitK d;
  d.load(mc, lib_dense, lib_elt);
  runs = 0;
  (void)d.tune(mc, K, N, M, run);
  EXPECT_TRUE(runs > 0);
  // ...and filed nowhere, so the next unnamed one measures too.
  genai::MmaSplitK e;
  e.load(mc, lib_dense, lib_elt);
  runs = 0;
  (void)e.tune(mc, K, N, M, run);
  EXPECT_TRUE(runs > 0);
}

// THE LM ATTENTION SETS, which a chat model tunes at load: loaded again in
// the session -- a helper unloaded for another model's memory -- they read
// the first load's members and splits and measure nothing.
TEST(kernel_tuning, the_attention_sets_read_the_session_s_picks)
{
  ::unsetenv("VPIPE_TUNING_CACHE");
  ::unsetenv("VPIPE_QWEN_GQA_AUTOTUNE");
  ::unsetenv("VPIPE_QWEN_PREFILL_AUTOTUNE");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  ComputeLibrary lib_sdpa = mc->load_library("sdpa");
  ComputeLibrary lib_attn = mc->load_library("attn_steel");
  ComputeLibrary lib_mma = mc->load_library("sdpa_mma");
  const bool m5 = mc->supports_matrix_cores();
  const genai::DecodeGqaAttnSet::Dims dd{256, 16, 4};

  genai::DecodeGqaAttnSet d1, d2;
  if (!d1.load(lib_sdpa, false) || !d2.load(lib_sdpa, false)) {
    std::printf("[kernel_tuning] no decode set; not exercised\n");
    return;
  }
  genai::TuningReport r1, r2;
  d1.prepare(mc, dd, r1);
  d2.prepare(mc, dd, r2);
  std::printf("[kernel_tuning] decode: %s | again: %s\n",
              r1.summary().c_str(), r2.summary().c_str());
  EXPECT_TRUE(r1.summary().find("the session's") == std::string::npos);
  EXPECT_TRUE(r2.summary().find("the session's") != std::string::npos);
  for (int pos : {100, 8000, 30000}) {
    EXPECT_TRUE(d1.uses_vec2(pos) == d2.uses_vec2(pos));
    EXPECT_TRUE(d1.split_for(pos) == d2.split_for(pos));
  }

  genai::PrefillGqaAttnSet p1, p2;
  if (!p1.load(lib_sdpa, &lib_attn, &lib_mma, m5) ||
      !p2.load(lib_sdpa, &lib_attn, &lib_mma, m5)) {
    std::printf("[kernel_tuning] no prefill set; not exercised\n");
    return;
  }
  genai::TuningReport q1, q2;
  p1.prepare(mc, {256, 16, 4}, q1);
  p2.prepare(mc, {256, 16, 4}, q2);
  std::printf("[kernel_tuning] prefill: %s | again: %s\n",
              q1.summary().c_str(), q2.summary().c_str());
  EXPECT_TRUE(q2.summary().find("the session's") != std::string::npos);
  for (int n : {600, 2000, 3000, 7000}) {
    EXPECT_TRUE(std::string(p1.kernel_name(n)) ==
                std::string(p2.kernel_name(n)));
  }
}

// UNTUNED (the accel key `autotune` false): nothing is measured and no
// answer kept in the session is read -- one measured by a tuned load
// first, so an untuned one that read it would show -- and each tuner
// takes its believed choice: the i8 split's single op, the decode set's
// vec2, the prefill set's NAX attention where the GPU has it, a VAE's
// conv route and mid-attention by the GPU alone. Two untuned loads agree,
// whatever a timing would have said.
TEST(kernel_tuning, an_untuned_load_measures_nothing_and_reads_no_answer)
{
  ::unsetenv("VPIPE_TUNING_CACHE");
  ::unsetenv("VPIPE_QWEN_GQA_AUTOTUNE");
  ::unsetenv("VPIPE_QWEN_PREFILL_AUTOTUNE");
  ::unsetenv("VPIPE_QWEN_PREFILL_ATTN");
  namespace conv = genai::vae_conv3x3;
  namespace mid = genai::vae_mid_attn;
  // The GPU's own answers, no timing anywhere.
  EXPECT_TRUE(conv::believed(true) == conv::Kind::kIm2col);
  EXPECT_TRUE(conv::believed(false) == conv::Kind::kOnChip);
  EXPECT_TRUE(mid::believed_order(true).front() == mid::Kind::kWide32);
  EXPECT_TRUE(mid::believed_order(false).front() == mid::Kind::kMat);
  EXPECT_TRUE(mid::believed_order(true).back() == mid::Kind::kScalar);
  EXPECT_TRUE(mid::believed_order(false).back() == mid::Kind::kScalar);

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const bool m5 = mc->supports_matrix_cores();

  // ---- the i8 split ------------------------------------------------
  if (m5) {
    const int N = 512, K = 2048, M = 1536;
    auto gemm = [&](genai::I8GemmContext& ctx) {
      SharedBuffer x = mc->make_shared_buffer((std::size_t)M * K * 2);
      SharedBuffer w = mc->make_shared_buffer((std::size_t)N * K * 2);
      SharedBuffer y = mc->make_shared_buffer((std::size_t)M * N * 2);
      bool ok = false;
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        ok = ctx.gemm(e, x, 0, w, y, 0, M, N, K);
      }
      st.commit().wait();
      return ok;
    };
    genai::I8GemmContext tuned(mc, true, false);
    if (tuned.enabled() && tuned.split_available()) {
      ASSERT_TRUE(gemm(tuned));
      tuned.tune_pending(mc);                 // files an answer
      EXPECT_TRUE(tuned.measured_for_test() == 1);
      genai::I8GemmContext u(mc, true, false);
      u.set_autotune(false);
      ASSERT_TRUE(gemm(u));
      EXPECT_TRUE(u.pending_for_test() == 0);  // nothing recorded
      EXPECT_TRUE(!u.tuned_for_test(M, N, K)); // the answer not read
      u.tune_pending(mc);
      EXPECT_TRUE(u.measured_for_test() == 0);
      EXPECT_TRUE(u.plan_for_test(M, N, K) == 0);
      EXPECT_TRUE(u.tune(mc, K, N, M, [](ComputeEncoder&) {}, "test") == 0);
      std::printf("[kernel_tuning] untuned i8: unsplit, the tuned load's "
                  "S=%d unread\n", tuned.plan_for_test(M, N, K));
    }
  }

  // ---- the decode attention set -------------------------------------
  ComputeLibrary lib_sdpa = mc->load_library("sdpa");
  ComputeLibrary lib_attn = mc->load_library("attn_steel");
  ComputeLibrary lib_mma = mc->load_library("sdpa_mma");
  ComputeLibrary lib_nax;
  if (m5) { lib_nax = mc->load_library("attn_steel_nax"); }
  const genai::DecodeGqaAttnSet::Dims dd{256, 16, 4};
  genai::DecodeGqaAttnSet dt, du;
  if (dt.load(lib_sdpa, false) && du.load(lib_sdpa, false)) {
    genai::TuningReport rt, ru;
    dt.prepare(mc, dd, rt);                   // files an answer
    du.prepare(mc, dd, ru, /*measure=*/false);
    std::printf("[kernel_tuning] decode tuned: %s | untuned: %s\n",
                rt.summary().c_str(), ru.summary().c_str());
    EXPECT_TRUE(ru.summary().find("untuned vec2") != std::string::npos);
    EXPECT_TRUE(ru.summary().find("the session's") == std::string::npos);
    for (int pos : {100, 8000, 30000}) { EXPECT_TRUE(du.uses_vec2(pos)); }
  }

  // ---- the prefill attention set ------------------------------------
  genai::PrefillGqaAttnSet pt, pu, pv;
  if (pt.load(lib_sdpa, &lib_attn, &lib_mma, m5, &lib_nax) &&
      pu.load(lib_sdpa, &lib_attn, &lib_mma, m5, &lib_nax) &&
      pv.load(lib_sdpa, &lib_attn, &lib_mma, m5, &lib_nax)) {
    genai::TuningReport qt, qu, qv;
    pt.prepare(mc, {256, 16, 4}, qt);         // files an answer
    pu.prepare(mc, {256, 16, 4}, qu, /*measure=*/false);
    pv.prepare(mc, {256, 16, 4}, qv, /*measure=*/false);
    std::printf("[kernel_tuning] prefill tuned: %s | untuned: %s\n",
                qt.summary().c_str(), qu.summary().c_str());
    EXPECT_TRUE(qu.summary().find("untuned") != std::string::npos);
    const bool nax = pu.has("nax-split") || pu.has("nax");
    for (int n : {600, 2000, 3000, 7000}) {
      const std::string k = pu.kernel_name(n);
      EXPECT_TRUE(k == pv.kernel_name(n));    // two loads agree
      if (nax) {
        EXPECT_TRUE(k == (pu.has("nax-split") ? "nax-split" : "nax"));
      } else if (n >= 2000) {
        EXPECT_TRUE(k == "steel" || k == "mma");
      }
    }
  }
}

// THE KEY REACHES WHAT THE TUNERS READ. Every stage that builds a tuned
// model takes `autotune` (bool, default true); the four that hand a
// registered family an acceleration bag put it there, which is the word a
// plug-in family reads. The built-in models take it by direct assignment
// to their Configs, so a family without the field does not compile.
TEST(kernel_tuning, the_stages_carry_the_autotune_key)
{
  namespace accel = genai::accel;
  Session sess;
  FlexData off = FlexData::make_object();
  {
    auto o = off.as_object();
    o.insert_or_assign("autotune", FlexData::make_bool(false));
  }
  const FlexData dflt = FlexData::make_object();
  auto declares = [](const StageSpec& sp) {
    for (const ConfigKey& k : sp.attrs) {
      if (k.key == accel::kAutotune) {
        return k.type == ConfigType::Bool && k.def_bool;
      }
    }
    return false;
  };
  const std::vector<InEdge> none;
  GenerateImageStage gi(&sess, "gi", none, off), gi0(&sess, "gi", none, dflt);
  GenerateVideoStage gv(&sess, "gv", none, off), gv0(&sess, "gv", none, dflt);
  VaeDecodeStage vd(&sess, "vd", none, off), vd0(&sess, "vd", none, dflt);
  VaeEncodeStage ve(&sess, "ve", none, off), ve0(&sess, "ve", none, dflt);
  DiffusionConditionerStage dc(&sess, "dc", none, off);
  GenerateAudioStage ga(&sess, "ga", none, off);
  LatentUpscaleStage lu(&sess, "lu", none, off);
  TextToSpeechStage tts(&sess, "tts", none, off);
  VideoRefEncoderStage vr(&sess, "vr", none, off);
  for (const StageSpec* sp : {&gi.spec(), &gv.spec(), &vd.spec(),
                              &ve.spec(), &dc.spec(), &ga.spec(),
                              &lu.spec(), &tts.spec(), &vr.spec()}) {
    if (!declares(*sp)) {
      std::printf("[kernel_tuning] %s: no `autotune` bool defaulting "
                  "true\n", std::string(sp->type_name).c_str());
    }
    EXPECT_TRUE(declares(*sp));
  }
  for (const FlexData* bag : {&gi.accel_settings(), &gv.accel_settings(),
                              &vd.accel_settings(), &ve.accel_settings()}) {
    EXPECT_FALSE(accel::flag(bag, accel::kAutotune, true));
  }
  for (const FlexData* bag : {&gi0.accel_settings(), &gv0.accel_settings(),
                              &vd0.accel_settings(), &ve0.accel_settings()}) {
    EXPECT_TRUE(accel::flag(bag, accel::kAutotune, false));
  }
}

// AN UNTUNED BACKBONE TIMES NOTHING. A Qwen3.5 model (head dim 256: its
// decode and prefill attention sets tune at the first decode step) through
// load, a prefill and decode steps: untuned, the shared helpers take no
// sample (genai::autotune_samples()); the same in a fresh session, tuned,
// takes some -- the control that says the zero is not a path that never
// ran. VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(kernel_tuning, an_untuned_backbone_times_nothing)
{
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (path == nullptr || *path == '\0') { return; }
  ::unsetenv("VPIPE_TUNING_CACHE");
  ::unsetenv("VPIPE_QWEN_GQA_AUTOTUNE");
  ::unsetenv("VPIPE_QWEN_PREFILL_AUTOTUNE");
  ::unsetenv("VPIPE_QWEN_PREFILL_ATTN");
  // Samples taken by one load + prefill + 4 steps, -1 when it did not run.
  auto run = [&](bool tune) -> long long {
    Session sess;
    MetalCompute* mc = sess.metal_compute();
    if (mc == nullptr || !mc->valid()) { return -1; }
    genai::ModelLoader loader(&sess);
    auto cfg = loader.load_config(path);
    if (!cfg) { return -1; }
    auto mcfg = genai::MetalQwenModel::config_from(*cfg);
    mcfg.autotune = tune;
    const std::uint64_t s0 = genai::autotune_samples();
    auto m = genai::MetalQwenModel::load(path, mc, mcfg);
    if (!m) { return -1; }
    auto root = m->context_manager()->acquire_root();
    std::vector<std::int32_t> ids;
    for (int i = 0; i < 48; ++i) { ids.push_back(1000 + 7 * i); }
    if (m->prefill(root, ids).empty()) { return -1; }
    std::int32_t next = 1234;
    for (int i = 0; i < 4 && next >= 0; ++i) {
      next = m->decode_step_fast(root, next);
    }
    if (next < 0) { return -1; }
    return (long long)(genai::autotune_samples() - s0);
  };
  const long long untuned = run(false);
  const long long tuned = run(true);
  std::printf("[kernel_tuning] backbone samples: untuned %lld, tuned %lld\n",
              untuned, tuned);
  ASSERT_TRUE(untuned >= 0 && tuned >= 0);
  EXPECT_TRUE(untuned == 0);
  EXPECT_TRUE(tuned > 0);
}

// AN UNTUNED VAE TIMES NOTHING, and decodes as the last untuned load did.
// The Z-Image VAE (the 2D AutoencoderKL: its mid-block attention is tuned
// at load on every GPU) and Qwen-Image-2.1's (a 1152-wide mid; its conv
// route is tuned on matrix cores), each loaded untuned TWICE -- no sample,
// the same bytes -- and then tuned. These tuners keep a PROCESS memo, so
// a tuned load after an earlier one of the same VAE in this process may
// have nothing left to time; the untuned loads read no memo either way,
// but their zero says the most when the tuned control measured, and the
// arm says which. VPIPE_Z_IMAGE_TEST_MODEL_PATH,
// VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH (the roots).
TEST(kernel_tuning, an_untuned_vae_times_nothing)
{
  ::unsetenv("VPIPE_VAE_ATTN_NO_AUTOTUNE");
  ::unsetenv("VPIPE_VAE_CONV_NO_AUTOTUNE");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto latent = [&](std::size_t n) {
    SharedBuffer z = mc->make_shared_buffer(n * 2);
    if (!z.empty()) {
      std::mt19937 rng(11);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      auto* d = static_cast<_Float16*>(z.contents());
      for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)nd(rng); }
    }
    return z;
  };
  auto same = [](const SharedBuffer& a, const SharedBuffer& b) {
    return !a.empty() && a.byte_size() == b.byte_size() &&
           std::memcmp(a.contents(), b.contents(), a.byte_size()) == 0;
  };
  // Load and decode once: the samples it took, the picture in *out.
  struct Arm {
    const char* name;
    std::function<long long(bool, SharedBuffer*)> run;
  };
  std::vector<Arm> arms;
  if (const char* r = std::getenv("VPIPE_Z_IMAGE_TEST_MODEL_PATH")) {
    const std::string dir = std::string(r) + "/vae";
    arms.push_back({"z-image vae", [&, dir](bool tune, SharedBuffer* out) {
      genai::MetalFlux2Vae::Config c;
      c.autotune = tune;
      const std::uint64_t s0 = genai::autotune_samples();
      auto m = genai::MetalFlux2Vae::load(dir, mc, c);
      if (!m) { return -1LL; }
      const int h = 32, w = 32;
      const SharedBuffer z =
          latent((std::size_t)m->config().dit_channels() * h * w);
      *out = m->decode(z, h, w);
      if (out->empty()) { return -1LL; }
      return (long long)(genai::autotune_samples() - s0);
    }});
  }
  if (const char* r = std::getenv("VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH")) {
    const std::string dir = std::string(r) + "/vae";
    arms.push_back({"qwen-image-2.1 vae", [&, dir](bool tune,
                                                   SharedBuffer* out) {
      std::ifstream in(dir + "/config.json");
      if (!in) { return -1LL; }
      const FlexData fd = FlexData::from_json(in);
      if (!fd.is_object()) { return -1LL; }
      genai::MetalKrea2Vae::Config c;
      genai::MetalKrea2Vae::config_from_json(fd, &c);
      c.autotune = tune;
      const std::uint64_t s0 = genai::autotune_samples();
      auto m = genai::MetalKrea2Vae::load(dir, mc, c);
      if (!m) { return -1LL; }
      const int h = 16, w = 16;
      *out = m->decode(latent((std::size_t)c.z_dim * h * w), h, w);
      if (out->empty()) { return -1LL; }
      return (long long)(genai::autotune_samples() - s0);
    }});
  }
  for (const Arm& a : arms) {
    SharedBuffer p1, p2, p3;
    const long long u1 = a.run(false, &p1);
    const long long u2 = a.run(false, &p2);
    const long long t = a.run(true, &p3);
    std::printf("[kernel_tuning] %s samples: untuned %lld / %lld, tuned "
                "%lld (%s); the untuned pictures %s\n", a.name, u1, u2, t,
                t > 0 ? "the control measured" : "a process memo answered",
                same(p1, p2) ? "IDENTICAL" : "DIFFERENT");
    ASSERT_TRUE(u1 >= 0 && u2 >= 0 && t >= 0);
    EXPECT_TRUE(u1 == 0 && u2 == 0);
    EXPECT_TRUE(same(p1, p2));
  }
}
