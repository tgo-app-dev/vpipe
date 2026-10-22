// The conditioning encoders' layer streaming, family by family.
//
// Every Qwen3 / Qwen3-VL text encoder the diffusion conditioner drives,
// and MOSS-Image's Qwen3.5 hybrid, streams its layers through the
// raw-source slot pair and grows a resident set by measurement
// (MetalQwenModel::Config::stream_decode),
// where it used to build each layer inside the prefill, free it, and
// keep a leading prefix sized from a fraction of RAM (pin_frac). The
// conditioner asks for both; the loader takes the slots wherever
// streams_raw() says it can.
//
// THE BAR IS BYTE-IDENTICAL, at each family's own taps: a streamed
// layer's products come from the same builder as a resident one's, so
// anything else means the two paths read the checkpoint differently.
// Against the resident model where the box holds it beside the run
// (else the prefill path's stream, itself held byte-identical to
// resident by flux2_smoke and minimax_h3_text_enc), over:
//
//   * a prompt with growth OFF (no reserve declared), which is also where
//     the streamable claim's floor is checked against what the model
//     holds: the floor's slot pair is the live one, byte for byte;
//   * a second prompt with growth on, then the first again -- the
//     second and third forwards run part-resident, part-streamed;
//   * the grounded (mROPE) taps the image-aware families take (not the
//     hybrid: MOSS-Image conditions on text alone);
//   * a layer-major batch, which streams by windows and keeps nothing.
//
// One env var per family, the repo root its tests already use. Unset,
// the family is skipped -- the line printed says which ran.

#include "tests/minitest.h"
#include "tests/unit-tests/residency-reserve.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/shared/stream-sizing.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/model-memory.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using vpipe::FlexData;
using vpipe::Session;
using vpipe::genai::ContextId;
using vpipe::genai::ContextManager;
using vpipe::genai::MetalLlamaWeights;
using vpipe::genai::MetalQwenModel;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace {

// `cond` evaluated once; returns from the enclosing void function.
#define ENCR_REQUIRE(cond)                                                 \
  do {                                                                     \
    if (!(cond)) {                                                         \
      report_expect_true_failure(#cond, __FILE__, __LINE__);               \
      return;                                                              \
    }                                                                      \
  } while (0)

// In a family's run (a free function, which minitest's own macros are
// not): record a failure, and stop the run at a failed precondition.
#define FAM_CHECK(cond)                                                    \
  do {                                                                     \
    if (!(cond)) { fails.push_back(#cond); }                               \
  } while (0)
#define FAM_REQUIRE(cond)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      fails.push_back(#cond);                                              \
      return fails;                                                        \
    }                                                                      \
  } while (0)

struct Family {
  const char*      name;
  const char*      env;    // the repo root the family's tests use
  const char*      sub;    // the encoder under it
  std::vector<int> taps;   // the conditioner's, 0-based
  int              depth;  // layers loaded; 0 = the checkpoint's
  // A gated-DeltaNet hybrid (Qwen3.5): configured through the LM's own
  // mapping, as the conditioner does, and text only (no grounded taps).
  bool             hybrid = false;
};

// The conditioner's taps (diffusion-conditioner-stage.cc).
const std::vector<Family>&
families_()
{
  static const std::vector<Family> f = {
      {"flux2", "VPIPE_FLUX2_TEST_MODEL_PATH", "text_encoder",
       {8, 17, 26}, 0},
      {"z-image", "VPIPE_Z_IMAGE_TEST_MODEL_PATH", "text_encoder", {34}, 0},
      {"qwen-image-21", "VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH",
       "text_encoder", {35}, 0},
      {"flux3", "VPIPE_FLUX3_TEST_MODEL_PATH", "text_encoder",
       {3, 7, 11, 15, 19, 23, 27, 31}, 32},
      {"krea2", "VPIPE_KREA2_TEST_MODEL_PATH", "text_encoder",
       {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34}, 0},
      {"boogu", "VPIPE_BOOGU_TEST_MODEL_PATH", "mllm", {35}, 0},
      {"moss-image", "VPIPE_MOSS_IMAGE_TEST_MODEL_PATH", "text_encoder",
       {31}, 0, true},
  };
  return f;
}

// The encoder as the conditioner configures it: a backbone-only bf16
// Qwen3 / Qwen3-VL stack, its geometry off config.json (a nested
// text_config for the VL ones), its wrapper off the tensor names.
bool
config_(const std::string& dir, const MetalLlamaWeights& wts, int depth,
        MetalQwenModel::Config* out, std::string* embed_name)
{
  MetalQwenModel::Config c;
  c.full_attn_interval = 1;
  c.use_bf16 = true;
  c.dense = true;
  c.zero_centered_norm = false;
  c.attn_output_gate = false;
  c.qk_norm = true;
  c.attention_bias = false;
  c.backbone_only = true;
  c.max_seq = 1024;
  c.page_tokens = 256;
  c.rms_eps = 1e-6f;
  c.rope_theta = 1.0e6f;
  std::ifstream in(std::filesystem::path(dir) / "config.json");
  if (!in) { return false; }
  FlexData fd = FlexData::from_json(in);
  if (!fd.is_object()) { return false; }
  FlexData tc = fd;
  if (fd.as_object().contains("text_config")) {
    tc = fd.as_object().at("text_config");
  }
  if (!tc.is_object()) { return false; }
  const auto o = tc.as_object();
  auto gi = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  c.n_layers   = gi("num_hidden_layers", 36);
  c.hidden     = gi("hidden_size", 2560);
  c.n_heads    = gi("num_attention_heads", 32);
  c.n_kv_heads = gi("num_key_value_heads", 8);
  c.head_dim   = gi("head_dim", c.hidden / c.n_heads);
  c.rotary_dim = c.head_dim;
  c.ffn_inner  = gi("intermediate_size", 9728);
  c.vocab      = gi("vocab_size", 151936);
  if (o.contains("rope_theta")) {
    c.rope_theta = (float)o.at("rope_theta").as_real(c.rope_theta);
  }
  if (depth > 0) { c.n_layers = std::min(c.n_layers, depth); }
  for (const char* p : {"model.language_model.", "language_model.",
                        "model."}) {
    if (wts.info(std::string(p) + "layers.0.self_attn.q_proj.weight") !=
        nullptr) {
      c.weight_prefix = p;
      c.model_seg = "";
      *embed_name = std::string(p) + "embed_tokens.weight";
      *out = c;
      return true;
    }
  }
  return false;
}

// MOSS-Image's encoder as the conditioner configures it
// (encoder_config_moss_image_): a Qwen3.5-9B hybrid through config_from,
// bf16, no head, zero-centred norms, under `language_model.`.
bool
hybrid_config_(const Session& sess, const std::string& dir,
               MetalQwenModel::Config* out, std::string* embed_name)
{
  vpipe::genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(dir);
  if (!mcfg.has_value()) { return false; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  c.use_bf16 = true;
  c.backbone_only = true;
  c.weight_prefix = "language_model.";
  c.model_seg = "";
  c.zero_centered_norm = true;
  c.max_seq = 1024;
  c.page_tokens = 256;
  *embed_name = "language_model.embed_tokens.weight";
  *out = c;
  return !c.dense;
}

// One forward's taps, as host bytes. `x` is consumed.
std::vector<std::uint8_t>
taps_(MetalQwenModel& m, SharedBuffer x, int n, const std::vector<int>& taps)
{
  ContextManager* cm = m.context_manager();
  const ContextId cid = cm->acquire_root();
  SharedBuffer t = m.forward_embeddings_taps(cid, x, n, taps);
  cm->release(cid);
  if (t.empty()) { return {}; }
  const auto* p = static_cast<const std::uint8_t*>(t.contents());
  return {p, p + t.byte_size()};
}

// The grounded form: three position axes, a 4x4 grid in the middle.
std::vector<std::uint8_t>
mrope_taps_(MetalQwenModel& m, SharedBuffer x, int n,
            const std::vector<int>& taps)
{
  std::vector<std::int32_t> pos((std::size_t)3 * n);
  for (int t = 0; t < n; ++t) {
    int tt = t, hh = t, ww = t;
    if (t >= 8 && t < 24) {
      const int k = t - 8;
      tt = 8;
      hh = 8 + k / 4;
      ww = 8 + k % 4;
    } else if (t >= 24) {
      tt = hh = ww = t - 12;
    }
    pos[(std::size_t)t] = tt;
    pos[(std::size_t)n + t] = hh;
    pos[(std::size_t)2 * n + t] = ww;
  }
  ContextManager* cm = m.context_manager();
  const ContextId cid = cm->acquire_root();
  SharedBuffer t = m.forward_embeddings_taps_mrope(cid, x, n, pos, taps);
  cm->release(cid);
  if (t.empty()) { return {}; }
  const auto* p = static_cast<const std::uint8_t*>(t.contents());
  return {p, p + t.byte_size()};
}

std::vector<std::string>
run_family_(const Family& f)
{
  std::vector<std::string> fails;
  const char* root = std::getenv(f.env);
  if (root == nullptr || *root == '\0') {
    std::printf("[encoder_residency] %s: %s unset, skipped\n", f.name,
                f.env);
    return fails;
  }
  const std::string dir = (std::filesystem::path(root) / f.sub).string();
  if (!std::filesystem::exists(std::filesystem::path(dir) /
                               "config.json")) {
    std::printf("[encoder_residency] %s: no %s, skipped\n", f.name,
                dir.c_str());
    return fails;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  FAM_REQUIRE(mc != nullptr);
  auto wts = MetalLlamaWeights::open_model(dir);
  FAM_REQUIRE(wts.has_value());
  MetalQwenModel::Config base;
  std::string embed_name;
  FAM_REQUIRE(f.hybrid ? hybrid_config_(sess, dir, &base, &embed_name)
                       : config_(dir, *wts, f.depth, &base, &embed_name));

  // Two prompts of embedding rows, gathered on the host as the
  // conditioner gathers them.
  const int H = base.hidden;
  const int n1 = 40, n2 = 29;
  std::vector<std::uint8_t> rows1((std::size_t)n1 * H * 2);
  std::vector<std::uint8_t> rows2((std::size_t)n2 * H * 2);
  {
    SharedBuffer emb = wts->load(embed_name, mc);
    FAM_REQUIRE(!emb.empty());
    const auto* tbl = static_cast<const std::uint8_t*>(emb.contents());
    for (int i = 0; i < n1; ++i) {
      std::memcpy(rows1.data() + (std::size_t)i * H * 2,
                  tbl + (std::size_t)(700 + 41 * i) * H * 2,
                  (std::size_t)H * 2);
    }
    for (int i = 0; i < n2; ++i) {
      std::memcpy(rows2.data() + (std::size_t)i * H * 2,
                  tbl + (std::size_t)(1300 + 23 * i) * H * 2,
                  (std::size_t)H * 2);
    }
  }
  auto x_of = [&](const std::vector<std::uint8_t>& r) {
    SharedBuffer x = mc->make_shared_buffer(r.size());
    if (!x.empty()) { std::memcpy(x.contents(), r.data(), r.size()); }
    return x;
  };
  const std::vector<int>& taps = f.taps;

  // ---- the reference ---------------------------------------------------
  // Resident where the box holds it with room to spare; else the prefill
  // path's stream (one layer at a time, nothing pinned).
  std::size_t disk = 0;
  for (const std::string& nm : wts->tensor_names()) {
    if (const auto* ti = wts->info(nm)) { disk += (std::size_t)ti->nbytes; }
  }
  const bool resident_fits = disk < vpipe::model_memory::phys_ram() / 3;
  std::vector<std::uint8_t> want1, want2, want_m;
  {
    MetalQwenModel::Config c = base;
    c.stream_layers = !resident_fits;
    c.pin_frac = 0.0;
    auto m = MetalQwenModel::load(dir, mc, c);
    FAM_REQUIRE(m != nullptr);
    FAM_REQUIRE(!m->streams_decode());
    want1 = taps_(*m, x_of(rows1), n1, taps);
    want2 = taps_(*m, x_of(rows2), n2, taps);
    if (!f.hybrid) { want_m = mrope_taps_(*m, x_of(rows1), n1, taps); }
  }
  FAM_REQUIRE(!want1.empty() && !want2.empty() &&
              (f.hybrid || !want_m.empty()));

  // ---- through the slots ---------------------------------------------
  MetalQwenModel::Config c = base;
  c.stream_layers = true;
  c.stream_decode = true;
  c.pin_frac = 0.5;   // asked for, and ignored: residency replaces it
  // The rule the stages read before they build, against what load() did.
  const bool predicted = MetalQwenModel::streams_raw(*wts, c);
  FAM_CHECK(MetalQwenModel::pinned_layer_count(*wts, c) == 0);
  auto m = MetalQwenModel::load(dir, mc, c);
  FAM_REQUIRE(m != nullptr);
  FAM_CHECK(predicted);
  FAM_REQUIRE(m->streams_decode() == predicted);
  FAM_REQUIRE(m->streams_decode());
  FAM_CHECK(m->pinned_layers() == 0);
  auto same = [](const std::vector<std::uint8_t>& a,
                 const std::vector<std::uint8_t>& b) {
    return !a.empty() && a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size()) == 0;
  };

  // 1. No reserve declared: growth stays off, so this forward holds the
  // floor -- and the streamable claim must have promised exactly it.
  const std::vector<std::uint8_t> g1 = taps_(*m, x_of(rows1), n1, taps);
  FAM_CHECK(same(g1, want1));
  FAM_CHECK(m->resident_layer_count() == 0);
  std::string stem = base.weight_prefix + base.model_seg + "layers.";
  std::size_t trunk = 0;
  for (const std::string& nm : wts->tensor_names()) {
    if (vpipe::genai::stream_sizing_alias(nm)) { continue; }
    if (nm.rfind(stem, 0) == 0) { continue; }
    if (const auto* ti = wts->info(nm)) { trunk += (std::size_t)ti->nbytes; }
  }
  const std::size_t floor = MetalQwenModel::stream_floor_bytes(*wts, stem);
  const std::size_t held = (std::size_t)m->resident_bytes();
  const std::size_t final_norm = (std::size_t)H * 2;
  std::printf("[encoder_residency] %s: floor %zu MB (trunk %zu MB + slot "
              "pair %zu MB); held at the floor %zu MB\n", f.name,
              floor >> 20, trunk >> 20, (floor - trunk) >> 20, held >> 20);
  // The slot pair booked is the slot pair held, to the byte (with the
  // slots switched off for an A/B, nothing is held at the floor).
  const bool slots_off = std::getenv("VPIPE_QWEN_NO_SLOTS") != nullptr;
  FAM_CHECK(floor > trunk &&
            (slots_off || held == floor - trunk + final_norm));

  // 2. Growth on: the box has room for at least the probe.
  m->set_residency_reserve(test_residency_reserve(1ull << 30));
  m->set_residency_schedule(4);
  const std::vector<std::uint8_t> g2 = taps_(*m, x_of(rows2), n2, taps);
  FAM_CHECK(same(g2, want2));
  const int kept2 = m->resident_layer_count();
  // 3. The first prompt again, against what was kept.
  const std::vector<std::uint8_t> g3 = taps_(*m, x_of(rows1), n1, taps);
  FAM_CHECK(same(g3, want1));
  const int kept3 = m->resident_layer_count();
  // 4. Grounded.
  const std::vector<std::uint8_t> gm =
      f.hybrid ? std::vector<std::uint8_t>{}
               : mrope_taps_(*m, x_of(rows1), n1, taps);
  FAM_CHECK(f.hybrid || same(gm, want_m));
  // What each layer keeps, off the checkpoint: a kept bf16 layer is its
  // checkpoint bytes -- q|k|v fused, the rest owned as read; a GDN
  // layer's A_log and dt_bias in f32.
  std::vector<std::size_t> per((std::size_t)base.n_layers, 0);
  {
    auto ends = [](const std::string& s, const char* t) {
      const std::size_t n = std::strlen(t);
      return s.size() >= n && s.compare(s.size() - n, n, t) == 0;
    };
    for (const std::string& nm : wts->tensor_names()) {
      if (nm.rfind(stem, 0) != 0) { continue; }
      const int L = std::atoi(nm.c_str() + stem.size());
      const auto* ti = wts->info(nm);
      if (L < 0 || L >= base.n_layers || ti == nullptr) { continue; }
      std::size_t numel = 1;
      for (auto d : ti->shape) { numel *= (std::size_t)d; }
      const bool f32 = ends(nm, ".A_log") || ends(nm, ".dt_bias");
      per[(std::size_t)L] += numel * (f32 ? 4 : 2);
    }
  }
  const int ran = std::min(base.n_layers,
                           *std::max_element(taps.begin(), taps.end()) + 1);
  // 5. Part-resident, whatever the box: the slot pair given back if every
  // layer the taps run is kept, then the deeper half of what is kept by
  // its exact bytes -- from the tail, as eviction takes them.
  const std::size_t freed = m->release_idle_slots();
  const int keep_r = kept3 / 2;
  std::size_t tail = 0;
  for (int L = keep_r; L < kept3; ++L) { tail += per[(std::size_t)L]; }
  const std::size_t gave = tail > 0 ? m->release_resident_layers(tail) : 0;
  const int kept_r = m->resident_layer_count();
  FAM_CHECK(gave == tail && kept_r == keep_r);
  // 6. A layer-major batch: windows, nothing kept or freed under them --
  // the layers given back built for it by the per-layer builder, and
  // dropped again after.
  std::vector<SharedBuffer> xs;
  xs.push_back(x_of(rows1));
  xs.push_back(x_of(rows2));
  std::vector<SharedBuffer*> xp = {&xs[0], &xs[1]};
  std::vector<SharedBuffer> bt =
      m->forward_embeddings_taps_batch(xp, {n1, n2}, taps);
  FAM_REQUIRE(bt.size() == 2);
  auto bytes = [](const SharedBuffer& b) {
    const auto* p = static_cast<const std::uint8_t*>(b.contents());
    return std::vector<std::uint8_t>(p, p + b.byte_size());
  };
  FAM_CHECK(same(bytes(bt[0]), want1));
  FAM_CHECK(same(bytes(bt[1]), want2));
  FAM_CHECK(m->resident_layer_count() == kept_r);
  // 7. What it holds, to the byte: the layers kept (residency admits in
  // stack order, and eviction takes the tail, so they are the first), the
  // final norm, and the slot pair unless every layer the taps ran was
  // kept when the pair was asked for back.
  std::size_t kept_bytes = 0;
  for (int L = 0; L < kept_r; ++L) { kept_bytes += per[(std::size_t)L]; }
  const std::size_t pair = floor - trunk;
  const std::size_t want_held = kept_bytes + final_norm +
                                (kept3 >= ran || slots_off ? 0 : pair);
  const std::size_t held_end = (std::size_t)m->resident_bytes();
  std::printf("[encoder_residency] %s vs %s: %d of %d layers kept after "
              "the second forward, %d after the third, %d once half were "
              "given back; held %zu MB (want %zu), %zu MB of slots released "
              "here; prompts, grounded and batch byte-identical: %d %d %d "
              "%d %d %d\n", f.name,
              resident_fits ? "resident" : "the prefill stream", kept2, ran,
              kept3, kept_r, held_end >> 20, want_held >> 20, freed >> 20,
              (int)same(g1, want1), (int)same(g2, want2),
              (int)same(g3, want1),
              (int)(f.hybrid || same(gm, want_m)),
              (int)same(bytes(bt[0]), want1),
              (int)same(bytes(bt[1]), want2));
  FAM_CHECK(kept2 > 0);
  FAM_CHECK(kept3 >= kept2);
  FAM_CHECK(held_end == want_held);
  return fails;
}

// A safetensors file of zeros: name, dtype, shape.
struct StTensor {
  std::string               name;
  std::string               dtype;
  std::vector<std::int64_t> shape;
};

bool
write_st_(const std::filesystem::path& p, const std::vector<StTensor>& ts)
{
  std::string hdr = "{";
  std::uint64_t off = 0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    std::uint64_t n = ts[i].dtype == "U32" || ts[i].dtype == "F32" ? 4 : 2;
    std::string shape;
    for (std::size_t d = 0; d < ts[i].shape.size(); ++d) {
      n *= (std::uint64_t)ts[i].shape[d];
      shape += (d ? "," : "") + std::to_string(ts[i].shape[d]);
    }
    hdr += (i ? "," : "");
    hdr += "\"" + ts[i].name + "\":{\"dtype\":\"" + ts[i].dtype +
           "\",\"shape\":[" + shape + "],\"data_offsets\":[" +
           std::to_string(off) + "," + std::to_string(off + n) + "]}";
    off += n;
  }
  hdr += "}";
  std::ofstream f(p, std::ios::binary);
  if (!f) { return false; }
  const std::uint64_t hl = hdr.size();
  f.write(reinterpret_cast<const char*>(&hl), 8);
  f.write(hdr.data(), (std::streamsize)hdr.size());
  const std::vector<char> zero((std::size_t)off, 0);
  f.write(zero.data(), (std::streamsize)zero.size());
  return (bool)f;
}

// A prompt source and a beat sink, for the conditioner on its own.
class SourcePrompt : public vpipe::TypedStage<SourcePrompt> {
public:
  static constexpr const char* kTypeName = "ut-source-prompt-enc-residency";
  SourcePrompt(const vpipe::SessionContextIntf* s, std::string id,
               std::vector<vpipe::InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::vector<std::string> prompts;
  std::size_t next = 0;
  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    if (next < prompts.size()) {
      co_await ctx.write(0, std::make_unique<vpipe::FlexDataPayload>(
                                FlexData::make_string(prompts[next++])));
      co_return;
    }
    ctx.signal_done();
    co_return;
  }
};

class SinkBeats : public vpipe::TypedStage<SinkBeats> {
public:
  static constexpr const char* kTypeName = "ut-sink-beats-enc-residency";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<vpipe::BeatPayloadIntf>> captured;
  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

// A beat's tensor bytes, whatever its element type.
std::vector<std::uint8_t>
beat_bytes_(const vpipe::BeatPayloadIntf* b)
{
  const auto* tb = dynamic_cast<const vpipe::TensorBeatPayload*>(b);
  if (tb == nullptr) { return {}; }
  const auto* p = static_cast<const std::uint8_t*>((const void*)tb->as_u8());
  return {p, p + tb->byte_size()};
}

}  // namespace

// THE FLOOR IS THE PATH THE LOADER TAKES. A streamable claim promises
// what the encoder holds at its least, and that depends on how it
// streams: through the raw-source slot pair, each slot a layer's sources
// plus the fused q|k|v built beside them; or, where that path cannot
// build the stack, the prefill path's layer at a time (counted as two).
// No model and no GPU: the query reads a safetensors header.
TEST(encoder_residency, the_floor_is_the_path_the_loader_takes)
{
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "vpipe-enc-floor";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  const fs::path file = dir / "model.safetensors";
  // hidden 64, 4 q heads / 2 kv heads of 16, ffn 128; two layers.
  const std::int64_t H = 64, Q = 64, K = 32, F = 128, D = 16;
  enum class Kind { kRaw, kNoQkNorm, kPack4 };
  auto write = [&](Kind kind) {
    std::vector<StTensor> ts = {
        {"model.embed_tokens.weight", "BF16", {256, H}},
        {"model.norm.weight", "BF16", {H}},
    };
    for (int l = 0; l < 2; ++l) {
      const std::string p = "model.layers." + std::to_string(l) + ".";
      ts.push_back({p + "input_layernorm.weight", "BF16", {H}});
      ts.push_back({p + "post_attention_layernorm.weight", "BF16", {H}});
      if (kind != Kind::kNoQkNorm) {
        ts.push_back({p + "self_attn.q_norm.weight", "BF16", {D}});
        ts.push_back({p + "self_attn.k_norm.weight", "BF16", {D}});
      } else {
        ts.push_back({p + "self_attn.q_proj.bias", "BF16", {Q}});
      }
      const struct { const char* nm; std::int64_t n, k; } pr[] = {
          {"self_attn.q_proj", Q, H}, {"self_attn.k_proj", K, H},
          {"self_attn.v_proj", K, H}, {"self_attn.o_proj", H, Q},
          {"mlp.gate_proj", F, H},    {"mlp.up_proj", F, H},
          {"mlp.down_proj", H, F},
      };
      for (const auto& x : pr) {
        const std::string nm = p + x.nm;
        if (kind == Kind::kPack4) {
          ts.push_back({nm + ".weight", "U32", {x.n, x.k / 8}});
          ts.push_back({nm + ".scales", "BF16", {x.n, x.k / 64}});
          ts.push_back({nm + ".biases", "BF16", {x.n, x.k / 64}});
        } else {
          ts.push_back({nm + ".weight", "BF16", {x.n, x.k}});
        }
      }
    }
    fs::remove(file, ec);
    return write_st_(file, ts);
  };
  const std::size_t trunk = (std::size_t)(256 * H + H) * 2;
  const std::size_t projs =
      (std::size_t)(Q * H + 2 * K * H + H * Q + 2 * F * H + H * F) * 2;
  const std::size_t norms = (std::size_t)(2 * H + 2 * D) * 2;
  const std::size_t qkv = (std::size_t)(Q + 2 * K) * H * 2;

  ENCR_REQUIRE(write(Kind::kRaw));
  {
    auto wts = MetalLlamaWeights::open_model(dir.string());
    ENCR_REQUIRE(wts.has_value());
    const std::size_t f =
        MetalQwenModel::stream_floor_bytes(*wts, "model.layers.");
    std::printf("[encoder_residency] raw: floor %zu, want %zu\n", f,
                trunk + 2 * (projs + norms + qkv));
    // The slot pair: every source, plus the fused q|k|v.
    EXPECT_TRUE(f == trunk + 2 * (projs + norms + qkv));
  }
  ENCR_REQUIRE(write(Kind::kNoQkNorm));
  {
    auto wts = MetalLlamaWeights::open_model(dir.string());
    ENCR_REQUIRE(wts.has_value());
    const std::size_t f =
        MetalQwenModel::stream_floor_bytes(*wts, "model.layers.");
    const std::size_t layer = projs + (std::size_t)(2 * H + Q) * 2;
    std::printf("[encoder_residency] no q/k norms: floor %zu, want %zu\n",
                f, trunk + 2 * layer);
    EXPECT_TRUE(f == trunk + 2 * layer);
  }
  ENCR_REQUIRE(write(Kind::kPack4));
  {
    auto wts = MetalLlamaWeights::open_model(dir.string());
    ENCR_REQUIRE(wts.has_value());
    const std::size_t f =
        MetalQwenModel::stream_floor_bytes(*wts, "model.layers.");
    const std::size_t g = vpipe::genai::stream_floor_bytes(*wts,
                                                           "model.layers.");
    std::printf("[encoder_residency] 4-bit pack: floor %zu, the prefill "
                "path's %zu\n", f, g);
    EXPECT_TRUE(f == g && f > trunk);
  }
  fs::remove_all(dir, ec);
}

TEST(encoder_residency, flux2)
{
  for (const std::string& w : run_family_(families_()[0])) {
    std::printf("[encoder_residency] FAILED: %s\n", w.c_str());
    EXPECT_TRUE(w.empty());
  }
}
TEST(encoder_residency, z_image)
{
  for (const std::string& w : run_family_(families_()[1])) {
    std::printf("[encoder_residency] FAILED: %s\n", w.c_str());
    EXPECT_TRUE(w.empty());
  }
}
TEST(encoder_residency, qwen_image_21)
{
  for (const std::string& w : run_family_(families_()[2])) {
    std::printf("[encoder_residency] FAILED: %s\n", w.c_str());
    EXPECT_TRUE(w.empty());
  }
}
TEST(encoder_residency, flux3)
{
  for (const std::string& w : run_family_(families_()[3])) {
    std::printf("[encoder_residency] FAILED: %s\n", w.c_str());
    EXPECT_TRUE(w.empty());
  }
}
TEST(encoder_residency, krea2)
{
  for (const std::string& w : run_family_(families_()[4])) {
    std::printf("[encoder_residency] FAILED: %s\n", w.c_str());
    EXPECT_TRUE(w.empty());
  }
}
TEST(encoder_residency, boogu)
{
  for (const std::string& w : run_family_(families_()[5])) {
    std::printf("[encoder_residency] FAILED: %s\n", w.c_str());
    EXPECT_TRUE(w.empty());
  }
}
// The hybrid: 24 gated-DeltaNet and 8 full-attention layers through one
// slot pair laid out as a pool both kinds refill.
TEST(encoder_residency, moss_image)
{
  for (const std::string& w : run_family_(families_()[6])) {
    std::printf("[encoder_residency] FAILED: %s\n", w.c_str());
    EXPECT_TRUE(w.empty());
  }
}

// WHAT A PROMPT COSTS, as the conditioner spends it: load the encoder,
// encode the prompt and its negative (and, apart, a third pass: the
// steady state), let it go -- three prompts in one
// process, one arm per process (the page cache one arm leaves is the
// other's head start). VPIPE_ENC_RESIDENCY_BENCH=<family> (flux2,
// z-image, qwen-image-21, moss-image, ...; its env var set as for the
// tests above),
// VPIPE_ENC_RESIDENCY_BENCH_ARM=pin (the prefill path and the plan's
// prefix) | res (the slot pair, residency reserving 2 GB as a 1024^2
// graph's arenas would). Prints; asserts only that it ran.
TEST(encoder_residency, a_prompt_s_cost)
{
  const char* fam = std::getenv("VPIPE_ENC_RESIDENCY_BENCH");
  const char* arm = std::getenv("VPIPE_ENC_RESIDENCY_BENCH_ARM");
  if (fam == nullptr || arm == nullptr) { return; }
  const Family* f = nullptr;
  for (const Family& x : families_()) {
    if (std::string(x.name) == fam) { f = &x; }
  }
  ENCR_REQUIRE(f != nullptr);
  const char* root = std::getenv(f->env);
  ENCR_REQUIRE(root != nullptr);
  const std::string dir = (std::filesystem::path(root) / f->sub).string();
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ENCR_REQUIRE(mc != nullptr);
  auto wts = MetalLlamaWeights::open_model(dir);
  ENCR_REQUIRE(wts.has_value());
  MetalQwenModel::Config base;
  std::string embed_name;
  ENCR_REQUIRE(f->hybrid
                   ? hybrid_config_(sess, dir, &base, &embed_name)
                   : config_(dir, *wts, f->depth, &base, &embed_name));
  const bool res = std::string(arm) == "res";
  const auto plan = vpipe::model_memory::plan_streaming(
      &sess, dir, std::string(), vpipe::model_memory::kStreamHeadroom, 0,
      vpipe::model_memory::kPhaseCondition);
  base.stream_layers = true;
  base.stream_decode = res;
  base.pin_frac = res ? 0.0 : plan.pin_frac;
  const int H = base.hidden, n = 96;
  std::vector<std::uint8_t> rows((std::size_t)n * H * 2);
  {
    SharedBuffer emb = wts->load(embed_name, mc);
    ENCR_REQUIRE(!emb.empty());
    const auto* tbl = static_cast<const std::uint8_t*>(emb.contents());
    for (int i = 0; i < n; ++i) {
      std::memcpy(rows.data() + (std::size_t)i * H * 2,
                  tbl + (std::size_t)(900 + 17 * i) * H * 2,
                  (std::size_t)H * 2);
    }
  }
  using clk = std::chrono::steady_clock;
  auto ms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  for (int beat = 0; beat < 3; ++beat) {
    // What the stage's beat does before it loads: wait out the last
    // encoder's memory coming back.
    (void)mc->memory_budget_settled();
    const auto t0 = clk::now();
    auto m = MetalQwenModel::load(dir, mc, base);
    ENCR_REQUIRE(m != nullptr);
    const auto t1 = clk::now();
    if (res) {
      m->set_residency_reserve(test_residency_reserve(2ull << 30));
      m->set_residency_schedule(2);
    }
    double pass[3] = {0, 0, 0};
    for (int k = 0; k < 3; ++k) {
      SharedBuffer x = mc->make_shared_buffer(rows.size());
      std::memcpy(x.contents(), rows.data(), rows.size());
      const auto a = clk::now();
      ENCR_REQUIRE(!taps_(*m, std::move(x), n, f->taps).empty());
      pass[k] = ms(a, clk::now());
    }
    const int kept = m->resident_layer_count();
    const int pinned = m->pinned_layers();
    const auto t2 = clk::now();
    m.reset();
    const auto t3 = clk::now();
    std::printf("[encoder_residency] %s %s beat %d: load %.0f ms, prompt "
                "%.0f ms, negative %.0f ms (a third pass %.0f), let go %.0f "
                "ms -- %.0f ms in all but the third (%d pinned, %d kept)\n",
                f->name, arm, beat + 1, ms(t0, t1), pass[0], pass[1],
                pass[2], ms(t2, t3), ms(t0, t3) - pass[2], pinned, kept);
  }
}

// A TEXT ENCODER KEPT WARM CARRIES NOTHING A PLAN SIZED. With a pinned
// prefix it did -- the prefix was sized from this launch's box, so a
// warm encoder was taken back only by a launch that would pin as many
// (moss_image_cond.an_encoder_kept_warm_is_taken_only_as_this_plan_pins
// still holds that for the encoders the slots cannot build). A resident
// set grown by measurement is sized again before every prompt, so a
// launch under a smaller box takes it back too. Three graphs through the
// conditioner (keep_loaded, `keep`), the last under a 32 GB box: built,
// taken back, taken back -- the conditioning the same bytes each time,
// and the layers grown in the first resident when the next prompts
// begin. Z-Image's encoder (VPIPE_Z_IMAGE_TEST_MODEL_PATH, the repo
// root): the smallest of the families here.
TEST(encoder_residency, a_warm_encoder_is_taken_back_whatever_the_box)
{
  const char* root = std::getenv("VPIPE_Z_IMAGE_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  using vpipe::DiffusionConditionerStage;
  auto run = [&](bool* taken, int* resident) {
    std::vector<std::uint8_t> out;
    auto pl = std::make_unique<vpipe::Pipeline>("enc-residency-warm", &sess);
    auto srcu = std::make_unique<SourcePrompt>(
        &sess, "src", std::vector<vpipe::InEdge>{}, FlexData::make_object());
    srcu->prompts = {"a lighthouse on a cliff at dusk, oil on canvas"};
    auto* src = static_cast<SourcePrompt*>(pl->insert_stage(std::move(srcu)));
    FlexData c = FlexData::make_object();
    c.as_object().insert("hf_dir", FlexData::make_string(root));
    c.as_object().insert("keep_loaded", FlexData::make_real(60.0));
    c.as_object().insert("unload_when_idle", FlexData::make_string("keep"));
    auto* cond = static_cast<DiffusionConditionerStage*>(pl->insert_stage(
        std::make_unique<DiffusionConditionerStage>(
            &sess, "cond", std::vector<vpipe::InEdge>{{src, 0}},
            std::move(c))));
    auto* pos = static_cast<SinkBeats*>(pl->insert_stage(
        std::make_unique<SinkBeats>(&sess, "pos",
                                    std::vector<vpipe::InEdge>{{cond, 0}},
                                    FlexData::make_object())));
    pl->insert_stage(std::make_unique<SinkBeats>(
        &sess, "neg", std::vector<vpipe::InEdge>{{cond, 1}},
        FlexData::make_object()));
    vpipe::PipelineRuntime rt(pl.get(), &sess);
    const bool launched = rt.launch();
    if (launched) { rt.wait_idle(); }
    rt.stop();
    *taken = cond->encoder_taken_warm();
    *resident = cond->encoder_layers_resident_at_prompt();
    if (launched && pos->captured.size() == 1) {
      out = beat_bytes_(pos->captured[0].get());
    }
    return out;   // the graph goes here, the encoder kept warm
  };
  bool t[3] = {true, false, false};
  int r[3] = {0, 0, 0};
  const std::vector<std::uint8_t> first = run(&t[0], &r[0]);
  const std::vector<std::uint8_t> again = run(&t[1], &r[1]);
  setenv("VPIPE_RAM_LIMIT_MB", "32768", 1);
  const std::vector<std::uint8_t> small = run(&t[2], &r[2]);
  unsetenv("VPIPE_RAM_LIMIT_MB");
  ENCR_REQUIRE(!first.empty() && again.size() == first.size() &&
               small.size() == first.size());
  const bool same = again == first;
  const bool same_small = small == first;
  std::printf("[encoder_residency] taken back %d / %d / %d (the last under "
              "a 32 GB box); layers resident as each prompt began %d / %d "
              "/ %d; conditioning %s, %s under it\n", (int)t[0], (int)t[1],
              (int)t[2], r[0], r[1], r[2],
              same ? "byte-identical" : "DIFFERS",
              same_small ? "byte-identical" : "DIFFERS");
  EXPECT_TRUE(!t[0] && t[1] && t[2]);
  // Residency engaged (not -1), grew in the first graph, and the set it
  // grew came back with the encoder.
  EXPECT_TRUE(r[0] == 0 && r[1] > 0 && r[2] > 0);
  EXPECT_TRUE(same);
  EXPECT_TRUE(same_small);
}
