// FP8 checkpoints and Krea-2's native single-file layout, through
// model-quantize.
//
// Always-on: the FP8 decoder over every code of both formats, the native ->
// diffusers name map, and the whole quantizer path on a SYNTHETIC native
// FP8 file small enough to write in the test (hidden 128, two blocks):
// bits=16 must reproduce every FP8 value exactly under the diffusers name
// and shape, bits=8 must quantize the block linears, and the stage must
// route a bare file to the krea2 DiT path. Plus the refusals: scaled FP8,
// a DiT tensor the layout cannot place, and a dense pass over an already
// quantized source.
//
// And the RUNTIME path on the same tiny model: MetalKrea2Transformer
// reading the FP8 file itself must be bit-identical to it reading the
// dense bf16 conversion -- preloaded, streamed through the slots, and
// streamed with the slots off -- in both FP8 formats.
//
// Env-gated: VPIPE_KREA2_NATIVE_TEST_FILE (a real native checkpoint, e.g.
// lustifyNSFWCheckpoint_v10Krea2_fp8.safetensors) together with
// VPIPE_KREA2_TEST_MODEL_PATH (the Krea-2-Turbo root) checks the map
// against the published transformer: same tensor set, same shapes, same
// config. Header reads only, so it costs nothing.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/krea2/krea2-native-checkpoint.h"
#include "generative-models/krea2/metal-krea2-transformer.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/lora-fusion.h"
#include "generative-models/quantize/model-quantizer.h"
#include "generative-models/quantize/quant-source.h"
#include "generative-models/shared/fp8-layout.h"
#include "generative-models/shared/fp8.h"
#include "generative-models/weight-set.h"
#include "stages/model-detect.h"
#include "stages/model-quantize-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <streambuf>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

namespace fs = std::filesystem;

class CerrSilencer {
public:
  CerrSilencer() : _saved(std::cerr.rdbuf()) { std::cerr.rdbuf(&_null); }
  ~CerrSilencer() { std::cerr.rdbuf(_saved); }
private:
  struct NullBuf : std::streambuf {
    int overflow(int c) override { return c; }
  };
  std::streambuf* _saved;
  NullBuf         _null;
};

float
bf16_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// One tensor of a synthetic checkpoint.
struct T {
  std::string               name;
  std::string               dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
};

// Write `ts` as one safetensors file, with `meta` as its `__metadata__`
// (string -> string; values must need no JSON escaping).
bool
write_safetensors_(const fs::path& p, const std::vector<T>& ts,
                   const std::map<std::string, std::string>& meta = {})
{
  std::string hdr = "{";
  if (!meta.empty()) {
    hdr += "\"__metadata__\":{";
    bool first = true;
    for (const auto& [k, v] : meta) {
      hdr += (first ? "" : ",") + std::string("\"") + k + "\":\"" + v +
             "\"";
      first = false;
    }
    hdr += "}";
    if (!ts.empty()) { hdr += ","; }
  }
  std::uint64_t off = 0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    std::string shp;
    for (std::size_t d = 0; d < ts[i].shape.size(); ++d) {
      shp += (d ? "," : "") + std::to_string(ts[i].shape[d]);
    }
    hdr += (i ? "," : "") + std::string("\"") + ts[i].name +
           "\":{\"dtype\":\"" + ts[i].dtype + "\",\"shape\":[" + shp +
           "],\"data_offsets\":[" + std::to_string(off) + "," +
           std::to_string(off + ts[i].bytes.size()) + "]}";
    off += ts[i].bytes.size();
  }
  hdr += "}";
  while (hdr.size() % 8 != 0) { hdr += ' '; }
  std::ofstream f(p, std::ios::binary);
  if (!f) { return false; }
  const std::uint64_t n = hdr.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(hdr.data(), (std::streamsize)hdr.size());
  for (const T& t : ts) {
    f.write(reinterpret_cast<const char*>(t.bytes.data()),
            (std::streamsize)t.bytes.size());
  }
  return f.good();
}

// A tiny native Krea-2 DiT, every tensor F8_E4M3 as the community file
// ships it. Codes are random but never the NaN code, and small enough
// (|v| < 2) to look like weights.
std::vector<T>
tiny_native_(std::mt19937& rng, const std::string& prefix = "",
             int max_code = 0x3F, const std::string& dtype = "F8_E4M3")
{
  const std::int64_t H = 128, FF = 256, TH = 128, TFF = 256, IC = 64;
  const std::int64_t TD = 64, HD = 128;
  // 0x3F is < 2.0; 0x27 is < 0.24, small enough for a forward to stay in
  // range through every block.
  std::uniform_int_distribution<int> mag(0, max_code);
  std::uniform_int_distribution<int> sgn(0, 1);
  std::vector<T> ts;
  auto add = [&](const std::string& n, std::vector<std::int64_t> shape) {
    std::size_t numel = 1;
    for (auto d : shape) { numel *= (std::size_t)d; }
    T t{prefix + n, dtype, std::move(shape), {}};
    t.bytes.resize(numel);
    for (auto& b : t.bytes) {
      b = (std::uint8_t)(mag(rng) | (sgn(rng) ? 0x80 : 0));
    }
    ts.push_back(std::move(t));
  };
  auto block = [&](const std::string& p, std::int64_t h, std::int64_t ff,
                   bool main) {
    add(p + "attn.wq.weight", {h, h});
    add(p + "attn.wk.weight", {h, h});
    add(p + "attn.wv.weight", {h, h});
    add(p + "attn.wo.weight", {h, h});
    add(p + "attn.gate.weight", {h, h});
    add(p + "attn.qknorm.qnorm.scale", {HD});
    add(p + "attn.qknorm.knorm.scale", {HD});
    add(p + "mlp.gate.weight", {ff, h});
    add(p + "mlp.up.weight", {ff, h});
    add(p + "mlp.down.weight", {h, ff});
    add(p + "prenorm.scale", {h});
    add(p + "postnorm.scale", {h});
    if (main) { add(p + "mod.lin", {6 * h}); }
  };
  for (int i = 0; i < 2; ++i) {
    block("blocks." + std::to_string(i) + ".", H, FF, true);
    block("txtfusion.layerwise_blocks." + std::to_string(i) + ".", TH, TFF,
          false);
    block("txtfusion.refiner_blocks." + std::to_string(i) + ".", TH, TFF,
          false);
  }
  add("txtfusion.projector.weight", {1, 12});
  add("first.weight", {H, IC});
  add("first.bias", {H});
  add("tmlp.0.weight", {H, TD});
  add("tmlp.0.bias", {H});
  add("tmlp.2.weight", {H, H});
  add("tmlp.2.bias", {H});
  add("tproj.1.weight", {6 * H, H});
  add("tproj.1.bias", {6 * H});
  add("txtmlp.0.scale", {TH});
  add("txtmlp.1.weight", {H, TH});
  add("txtmlp.1.bias", {H});
  add("txtmlp.3.weight", {H, H});
  add("txtmlp.3.bias", {H});
  add("last.linear.weight", {IC, H});
  add("last.linear.bias", {IC});
  add("last.norm.scale", {H});
  add("last.modulation.lin", {2, H});
  return ts;
}

std::vector<std::uint8_t>
f32_bytes_(const std::vector<float>& v)
{
  std::vector<std::uint8_t> b(v.size() * 4);
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

std::vector<std::uint8_t>
str_bytes_(const std::string& s)
{
  return std::vector<std::uint8_t>(s.begin(), s.end());
}

// f32 -> bf16, nearest-even: the reference rounding for a scaled value.
std::uint16_t
bf16_rne_(float v)
{
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// Make a tiny native checkpoint SCALED, the way ComfyUI's tools write one.
// Every Linear weight (`.weight`) gets a scale that is not a power of two,
// so the product really rounds.
//   kLegacy     `scaled_fp8` marker + one `<layer>.scale_weight` a layer.
//   kComfyU8Row weights stored as U8, a `<layer>.comfy_quant` JSON record
//               each, and a `<layer>.weight_scale` of one value PER ROW.
enum class ScaleStyle { kLegacy, kComfyU8Row };

void
scale_native_(std::vector<T>& ts, std::mt19937& rng, ScaleStyle st)
{
  std::uniform_real_distribution<float> sd(0.37f, 1.9f);
  std::vector<T> extra;
  for (T& t : ts) {
    if (t.name.size() < 7 ||
        t.name.compare(t.name.size() - 7, 7, ".weight") != 0) {
      continue;
    }
    const std::string layer = t.name.substr(0, t.name.size() - 7);
    if (st == ScaleStyle::kLegacy) {
      extra.push_back({layer + ".scale_weight", "F32", {},
                       f32_bytes_({sd(rng)})});
      continue;
    }
    t.dtype = "U8";
    extra.push_back({layer + ".comfy_quant", "U8", {},
                     str_bytes_("{\"format\": \"float8_e4m3fn\"}")});
    extra.back().shape = {(std::int64_t)extra.back().bytes.size()};
    std::vector<float> rows((std::size_t)t.shape[0]);
    for (float& r : rows) { r = sd(rng); }
    extra.push_back({layer + ".weight_scale", "F32",
                     {t.shape[0], 1}, f32_bytes_(rows)});
  }
  if (st == ScaleStyle::kLegacy) {
    extra.push_back({"scaled_fp8", "F8_E4M3", {}, {0}});
  }
  ts.insert(ts.end(), extra.begin(), extra.end());
}

MetalKrea2Transformer::Config
tiny_cfg_()
{
  MetalKrea2Transformer::Config cfg;
  cfg.hidden = 128;       cfg.n_heads = 1;      cfg.n_kv_heads = 1;
  cfg.head_dim = 128;     cfg.ffn = 256;        cfg.n_layers = 2;
  cfg.in_channels = 64;   cfg.timestep_dim = 64;
  cfg.text_hidden = 128;  cfg.text_heads = 1;   cfg.text_kv_heads = 1;
  cfg.text_ffn = 256;     cfg.n_text_layers = 12;
  cfg.n_layerwise = 2;    cfg.n_refiner = 2;
  return cfg;
}

fs::path
scratch_(const std::string& tag)
{
  const fs::path p = fs::temp_directory_path() /
                     ("vpipe-k2n-" + tag + "-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(p, ec);
  return p;
}

FlexData
read_json_(const fs::path& p)
{
  std::ifstream in(p);
  if (!in) { return FlexData(); }
  return FlexData::from_json(in);
}

}  // namespace

// Every E4M3 code against the format's definition, and its bf16 table
// entry against the float it stands for -- exactly, which is the claim
// the dense pass rests on.
TEST(fp8, e4m3_decodes_every_code_exactly)
{
  using namespace fp8;
  const auto& lut = bf16_table(Format::kE4M3);
  int nan = 0;
  for (int c = 0; c < 256; ++c) {
    const int e = (c >> 3) & 0xF, m = c & 7;
    const double sign = (c & 0x80) ? -1.0 : 1.0;
    const float got = to_f32((std::uint8_t)c, Format::kE4M3);
    if (e == 15 && m == 7) {
      EXPECT_TRUE(std::isnan(got));
      EXPECT_TRUE(std::isnan(bf16_f32_(lut[(std::size_t)c])));
      ++nan;
      continue;
    }
    const double want = e == 0 ? sign * std::ldexp((double)m, -9)
                               : sign * std::ldexp(1.0 + m / 8.0, e - 7);
    EXPECT_TRUE((double)got == want);
    EXPECT_TRUE(bf16_f32_(lut[(std::size_t)c]) == got);
  }
  EXPECT_TRUE(nan == 2);   // S.1111.111 only: no infinities in E4M3FN
  EXPECT_TRUE(to_f32(0x7E, Format::kE4M3) == 448.0f);
  EXPECT_TRUE(to_f32(0x01, Format::kE4M3) == std::ldexp(1.0f, -9));
  EXPECT_TRUE(to_f32(0x08, Format::kE4M3) == std::ldexp(1.0f, -6));
  EXPECT_TRUE(std::signbit(to_f32(0x80, Format::kE4M3)));
}

// E5M2 is the top byte of an IEEE half, which gives an independent
// reference for all 256 codes, infinities and NaNs included.
TEST(fp8, e5m2_is_the_top_byte_of_a_half)
{
  using namespace fp8;
  const auto& lut = bf16_table(Format::kE5M2);
  for (int c = 0; c < 256; ++c) {
    const std::uint16_t hb = (std::uint16_t)(c << 8);
    _Float16 h;
    std::memcpy(&h, &hb, 2);
    const float want = (float)h;
    const float got = to_f32((std::uint8_t)c, Format::kE5M2);
    const float via = bf16_f32_(lut[(std::size_t)c]);
    if (std::isnan(want)) {
      EXPECT_TRUE(std::isnan(got) && std::isnan(via));
    } else {
      EXPECT_TRUE(got == want && via == want);
    }
  }
  EXPECT_TRUE(format_of("F8_E4M3") == Format::kE4M3);
  EXPECT_TRUE(format_of("F8_E5M2") == Format::kE5M2);
  EXPECT_TRUE(format_of("BF16") == Format::kNone);
  EXPECT_TRUE(is_aux_name("blocks.0.attn.wq.scale_weight"));
  EXPECT_TRUE(is_aux_name("model.diffusion_model.scaled_fp8"));
  EXPECT_TRUE(is_aux_name("x.lora_A.weight_scale"));
  EXPECT_FALSE(is_aux_name("blocks.0.prenorm.scale"));
}

TEST(krea2_native, names_map_to_diffusers)
{
  const std::map<std::string, std::string> want = {
    {"blocks.7.attn.wq.weight", "transformer_blocks.7.attn.to_q.weight"},
    {"blocks.7.attn.wo.weight", "transformer_blocks.7.attn.to_out.0.weight"},
    {"blocks.7.attn.gate.weight", "transformer_blocks.7.attn.to_gate.weight"},
    {"blocks.7.attn.qknorm.knorm.scale",
     "transformer_blocks.7.attn.norm_k.weight"},
    {"blocks.27.mlp.down.weight", "transformer_blocks.27.ff.down.weight"},
    {"blocks.0.postnorm.scale", "transformer_blocks.0.norm2.weight"},
    {"txtfusion.refiner_blocks.1.attn.wv.weight",
     "text_fusion.refiner_blocks.1.attn.to_v.weight"},
    {"txtfusion.layerwise_blocks.0.prenorm.scale",
     "text_fusion.layerwise_blocks.0.norm1.weight"},
    {"txtfusion.projector.weight", "text_fusion.projector.weight"},
    {"first.bias", "img_in.bias"},
    {"tmlp.2.weight", "time_embed.linear_2.weight"},
    {"tproj.1.bias", "time_mod_proj.bias"},
    {"txtmlp.0.scale", "txt_in.norm.weight"},
    {"txtmlp.3.weight", "txt_in.linear_2.weight"},
    {"last.norm.scale", "final_layer.norm.weight"},
    {"last.modulation.lin", "final_layer.scale_shift_table"},
  };
  for (const auto& [from, to] : want) {
    std::string got;
    std::vector<std::int64_t> shp;
    ASSERT_TRUE(krea2::to_diffusers(from, {8, 8}, &got, &shp));
    EXPECT_TRUE(got == to);
    EXPECT_TRUE((shp == std::vector<std::int64_t>{8, 8}));
  }
  // The one reshape: the flat per-block modulation becomes [6, H].
  std::string got;
  std::vector<std::int64_t> shp;
  ASSERT_TRUE(krea2::to_diffusers("blocks.3.mod.lin", {36864}, &got, &shp));
  EXPECT_TRUE(got == "transformer_blocks.3.scale_shift_table");
  EXPECT_TRUE((shp == std::vector<std::int64_t>{6, 6144}));
  // Names the layout does not define are refused, never guessed.
  EXPECT_FALSE(krea2::to_diffusers("blocks.0.attn.wq.bias", {8}, &got, &shp));
  EXPECT_FALSE(krea2::to_diffusers("blocks.0.mod.lin", {7}, &got, &shp));
  EXPECT_FALSE(krea2::to_diffusers("txtfusion.layerwise_blocks.0.mod.lin",
                                   {48}, &got, &shp));
  EXPECT_FALSE(krea2::to_diffusers("blocks.x.prenorm.scale", {8}, &got,
                                   &shp));
  EXPECT_FALSE(krea2::to_diffusers("pos_embed", {8}, &got, &shp));
}

// The whole path on a synthetic FP8 file: dense, then 8-bit, then the
// stage's routing of a bare file.
TEST(krea2_native, fp8_file_dequantizes_and_quantizes)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  CerrSilencer hush;
  std::mt19937 rng(44);
  const std::vector<T> src_t = tiny_native_(rng);
  const fs::path dir = scratch_("src");
  fs::create_directories(dir);
  const fs::path file = dir / "tiny_krea2_fp8.safetensors";
  ASSERT_TRUE(write_safetensors_(file, src_t));
  ASSERT_TRUE(krea2::is_native_dit_file(file.string()));
  // Registered as the DiT it is, so generate-image's dit_dir picker offers
  // it -- the file carries no config for any other rung to read.
  {
    const DetectedModel d = detect_model_dir(file.string());
    EXPECT_TRUE(d.model_type == "krea2-dit");
    EXPECT_TRUE(d.detected_by == "krea2-native");
  }

  // ---- bits = 16: every value exact, under its diffusers name. ----------
  const fs::path dense = scratch_("dense");
  {
    QuantizeOptions opt;
    opt.bits = 16;
    std::string err;
    ASSERT_TRUE(ModelQuantizer(mc).run(file.string(), dense.string(), opt,
                                       &err));
  }
  auto w = MetalLlamaWeights::open_model(dense.string());
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }
  EXPECT_TRUE(w->tensor_names().size() == src_t.size());
  const auto& lut = fp8::bf16_table(fp8::Format::kE4M3);
  int checked = 0;
  for (const T& t : src_t) {
    std::string name;
    std::vector<std::int64_t> shape;
    ASSERT_TRUE(krea2::to_diffusers(t.name, t.shape, &name, &shape));
    const auto* ti = w->info(name);
    ASSERT_TRUE(ti != nullptr);
    if (ti == nullptr) { continue; }
    EXPECT_TRUE(ti->dtype == "BF16");
    EXPECT_TRUE(ti->shape == shape);
    auto buf = w->load(name, mc);
    ASSERT_TRUE(buf.byte_size() == t.bytes.size() * 2);
    const auto* got = static_cast<const std::uint16_t*>(buf.contents());
    bool same = true;
    for (std::size_t i = 0; i < t.bytes.size(); ++i) {
      same = same && got[i] == lut[t.bytes[i]];
    }
    EXPECT_TRUE(same);
    ++checked;
  }
  EXPECT_TRUE(checked == (int)src_t.size());
  EXPECT_TRUE(w->has("transformer_blocks.1.scale_shift_table"));
  {
    FlexData cfg = read_json_(dense / "config.json");
    ASSERT_TRUE(cfg.is_object());
    auto o = cfg.as_object();
    EXPECT_TRUE(std::string(o.at("_class_name").as_string("")) ==
                "Krea2Transformer2DModel");
    EXPECT_TRUE(o.at("num_layers").as_int(0) == 2);
    EXPECT_TRUE(o.at("num_attention_heads").as_int(0) == 1);
    EXPECT_TRUE(o.at("intermediate_size").as_int(0) == 256);
    EXPECT_TRUE(o.at("num_text_layers").as_int(0) == 12);
    EXPECT_TRUE(o.at("num_refiner_text_blocks").as_int(0) == 2);
    EXPECT_TRUE(o.at("timestep_embed_dim").as_int(0) == 64);
    EXPECT_FALSE(o.contains("quantization"));   // dense means dense
  }

  // ---- bits = 8: the block linears quantize, the rest passes. -----------
  const fs::path q8 = scratch_("w8");
  {
    QuantizeOptions opt;
    opt.bits = 8;
    opt.group = 64;
    opt.quant_linears = {"to_q", "to_k", "to_v", "to_gate", "0", "gate",
                         "up", "down", "linear_1", "linear_2", "img_in",
                         "time_mod_proj", "linear"};
    std::string err;
    ASSERT_TRUE(ModelQuantizer(mc).run(file.string(), q8.string(), opt,
                                       &err));
  }
  auto wq = MetalLlamaWeights::open_model(q8.string());
  ASSERT_TRUE(wq.has_value());
  if (!wq.has_value()) { return; }
  const auto* qc = wq->info("transformer_blocks.0.ff.down.weight");
  ASSERT_TRUE(qc != nullptr);
  if (qc != nullptr) {
    EXPECT_TRUE(qc->dtype == "U32");
    EXPECT_TRUE(wq->has("transformer_blocks.0.ff.down.scales"));
  }
  const auto* nt = wq->info("transformer_blocks.0.norm1.weight");
  ASSERT_TRUE(nt != nullptr && nt->dtype == "BF16");
  {
    FlexData cfg = read_json_(q8 / "config.json");
    ASSERT_TRUE(cfg.is_object());
    auto o = cfg.as_object();
    ASSERT_TRUE(o.contains("quantization"));
    FlexData qb = o.at("quantization");
    EXPECT_TRUE(qb.as_object().at("bits").as_int(0) == 8);
  }

  // ---- A dense pass over an already-quantized source is refused. --------
  {
    QuantizeOptions opt;
    opt.bits = 16;
    std::string err;
    const fs::path again = scratch_("again");
    EXPECT_FALSE(ModelQuantizer(mc).run(q8.string(), again.string(), opt,
                                        &err));
    EXPECT_TRUE(err.find("affine-quantized") != std::string::npos);
    std::error_code ec;
    fs::remove_all(again, ec);
  }

  // ---- The stage routes a bare native file to the krea2 DiT path. -------
  {
    std::string fam;
    EXPECT_TRUE(ModelQuantizeStage::resolve_source_dit_dir(file.string(),
                                                           &fam) ==
                file.string());
    EXPECT_TRUE(fam == "krea2");
    const fs::path staged = scratch_("stage");
    FlexData cfg = FlexData::make_object();
    {
      auto o = cfg.as_object();
      o.insert("src_model", FlexData::make_string(file.string()));
      o.insert("output_name", FlexData::make_string(staged.string()));
      o.insert("bits", FlexData::make_int(16));
      o.insert("skip_existing", FlexData::make_bool(false));
    }
    ModelQuantizeStage s(&sess, "mq-k2n", std::vector<InEdge>{},
                         std::move(cfg));
    ASSERT_TRUE(s.config_error().empty());
    EXPECT_TRUE(s.quantize_once());
    EXPECT_TRUE(fs::exists(staged / "config.json"));
    std::error_code ec;
    fs::remove_all(staged, ec);
  }

  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::remove_all(dense, ec);
  fs::remove_all(q8, ec);
}

// Prefixed keys (a full ComfyUI checkpoint) are found; bundled non-DiT
// tensors are ignored; an unplaceable DiT tensor refuses.
TEST(krea2_native, prefixes_and_refusals)
{
  CerrSilencer hush;
  std::mt19937 rng(7);
  const fs::path dir = scratch_("ref");
  fs::create_directories(dir);

  {
    std::vector<T> ts = tiny_native_(rng, "model.diffusion_model.");
    ts.push_back({"text_encoders.qwen3.embed.weight", "BF16", {4},
                  std::vector<std::uint8_t>(8, 0)});
    const fs::path f = dir / "prefixed.safetensors";
    ASSERT_TRUE(write_safetensors_(f, ts));
    std::string err;
    auto s = QuantSource::open(f.string(), &err);
    ASSERT_TRUE(s.has_value());
    if (s.has_value()) {
      EXPECT_TRUE(s->layout() == "krea2-native");
      EXPECT_TRUE(s->n_ignored() == 1);
      EXPECT_TRUE(s->n_fp8() == (int)ts.size() - 1);
      EXPECT_TRUE(s->info("transformer_blocks.0.attn.to_q.weight") !=
                  nullptr);
      EXPECT_TRUE(s->config() != nullptr);
    }
  }
  {
    std::vector<T> ts = tiny_native_(rng);
    ts.push_back({"blocks.0.attn.wq.bias", "F8_E4M3", {128},
                  std::vector<std::uint8_t>(128, 0)});
    const fs::path f = dir / "stray.safetensors";
    ASSERT_TRUE(write_safetensors_(f, ts));
    std::string err;
    EXPECT_FALSE(QuantSource::open(f.string(), &err).has_value());
    EXPECT_TRUE(err.find("blocks.0.attn.wq.bias") != std::string::npos);
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// The map against the published transformer. Header reads only.
TEST(krea2_native, real_file_matches_the_published_transformer)
{
  const char* file = std::getenv("VPIPE_KREA2_NATIVE_TEST_FILE");
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (file == nullptr || *file == '\0' || root == nullptr || *root == '\0') {
    return;
  }
  std::string err;
  auto s = QuantSource::open(file, &err);
  ASSERT_TRUE(s.has_value());
  auto ref = MetalLlamaWeights::open_model(std::string(root) +
                                           "/transformer");
  ASSERT_TRUE(ref.has_value());
  if (!s.has_value() || !ref.has_value()) { return; }
  const auto names = s->tensor_names();
  EXPECT_TRUE(names.size() == ref->tensor_names().size());
  int missing = 0, shape_diff = 0;
  for (const auto& n : names) {
    const auto* a = s->info(n);
    const auto* b = ref->info(n);
    if (b == nullptr) {
      if (++missing <= 5) { std::printf("  missing in ref: %s\n", n.c_str()); }
      continue;
    }
    if (a->shape != b->shape) { ++shape_diff; }
  }
  std::printf("[krea2_native] %zu tensors, %d FP8, %d not in the published "
              "transformer, %d shape mismatches\n", names.size(), s->n_fp8(),
              missing, shape_diff);
  EXPECT_TRUE(missing == 0);
  EXPECT_TRUE(shape_diff == 0);
  // Every key the published config.json states, the synthesized one
  // states identically.
  FlexData want = read_json_(fs::path(root) / "transformer" / "config.json");
  ASSERT_TRUE(want.is_object() && s->config() != nullptr);
  FlexData got = *s->config();
  auto wo = want.as_object();
  auto go = got.as_object();
  int diff = 0;
  for (const char* k : {"_class_name", "attention_head_dim", "in_channels",
                        "intermediate_size", "num_attention_heads",
                        "num_key_value_heads", "num_layers",
                        "num_layerwise_text_blocks", "num_refiner_text_blocks",
                        "num_text_layers", "text_hidden_dim",
                        "text_intermediate_size", "text_num_attention_heads",
                        "text_num_key_value_heads", "timestep_embed_dim"}) {
    if (!go.contains(k) || go.at(k).to_json(false) != wo.at(k).to_json(false)) {
      std::printf("  config differs at %s\n", k);
      ++diff;
    }
  }
  if (go.at("axes_dims_rope").to_json(false) !=
      wo.at("axes_dims_rope").to_json(false)) {
    ++diff;
  }
  EXPECT_TRUE(go.at("rope_theta").as_real(0) == wo.at("rope_theta").as_real(1));
  EXPECT_TRUE(diff == 0);
}

// RUNTIME FP8: the transformer reads the FP8 file itself -- names
// translated by the reader, projections kept as codes and expanded on the
// GPU per GEMM -- and must compute EXACTLY what the dense bf16 conversion
// of the same file computes, since every FP8 value is a bf16 value and
// the expanded weight then takes the same kernels. Bit-identical, not
// close, and across all three ways the blocks can arrive: preloaded,
// streamed through the slot pair (FP8 codes refilled raw, the modulation
// tables widened in place), and streamed with the slots switched off.
//
// A tiny config (hidden 128, one head of 128, two blocks) so it runs
// everywhere; the geometry is small, so on a matrix-core box most of it
// takes the steel path rather than matmul2d -- the real-size run is what
// exercises that one.
//
// One GPU decoder per format, so each is run: E4M3 through its shift-and-
// scale, E5M2 through the half reinterpretation. (A lambda, not a helper:
// minitest's EXPECT/ASSERT only exist inside a TEST body.)
TEST(krea2_native, fp8_runtime_matches_dense)
{
  auto check = [&](const std::string& label, const std::vector<T>& content) {
    Session sess;
    metal_compute::MetalCompute* mc = sess.metal_compute();
    if (mc == nullptr) { return; }
    CerrSilencer hush;
    std::mt19937 rng(2026);
    const fs::path dir = scratch_("rt");
    fs::create_directories(dir);
    const fs::path file = dir / "tiny_krea2_fp8.safetensors";
    ASSERT_TRUE(write_safetensors_(file, content));
    const fs::path dense = scratch_("rt-dense");
    {
      QuantizeOptions opt;
      opt.bits = 16;
      std::string err;
      ASSERT_TRUE(ModelQuantizer(mc).run(file.string(), dense.string(), opt,
                                         &err));
    }

    const MetalKrea2Transformer::Config cfg = tiny_cfg_();

    const int TS = 8, GH = 4, GW = 4, IS = GH * GW;
    std::normal_distribution<float> nd(0.0f, 0.5f);
    auto f16buf = [&](std::size_t n) {
      metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
      auto* d = static_cast<_Float16*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)nd(rng); }
      return b;
    };
    const metal_compute::SharedBuffer ehs =
        f16buf((std::size_t)TS * cfg.n_text_layers * cfg.text_hidden);
    const metal_compute::SharedBuffer lat = f16buf((std::size_t)IS * 64);

    // One text pass and one denoise step, as f16 bits.
    struct Out { std::vector<std::uint16_t> text, vel; std::size_t held = 0; };
    auto run = [&](const std::string& src, bool stream) {
      Out o;
      auto ws = WeightSet::open(src, nullptr);
      if (!ws) { return o; }
      auto m = MetalKrea2Transformer::load(ws, mc, cfg, stream);
      if (m == nullptr) { return o; }
      auto t = m->forward_text(ehs, TS);
      if (t.empty()) { return o; }
      auto v = m->forward_dit(t, TS, lat, IS, GH, GW, 0.5f);
      if (v.empty()) { return o; }
      const auto* tp = static_cast<const std::uint16_t*>(t.contents());
      const auto* vp = static_cast<const std::uint16_t*>(v.contents());
      o.text.assign(tp, tp + (std::size_t)TS * cfg.hidden);
      o.vel.assign(vp, vp + (std::size_t)IS * 64);
      o.held = ws->stats().bytes;
      return o;
    };
    const Out ref = run(dense.string(), false);
    ASSERT_TRUE(!ref.vel.empty());
    if (ref.vel.empty()) { return; }
    // Not vacuous: a finite, non-trivial velocity.
    int finite = 0, nonzero = 0;
    for (std::uint16_t b : ref.vel) {
      _Float16 h;
      std::memcpy(&h, &b, 2);
      if (std::isfinite((float)h)) { ++finite; }
      if ((float)h != 0.0f) { ++nonzero; }
    }
    EXPECT_TRUE(finite == (int)ref.vel.size());
    EXPECT_TRUE(nonzero > (int)ref.vel.size() / 2);

    const Out pre = run(file.string(), false);
    const Out str = run(file.string(), true);
    ::setenv("VPIPE_KREA2_NO_SLOTS", "1", 1);
    const Out nos = run(file.string(), true);
    ::unsetenv("VPIPE_KREA2_NO_SLOTS");
    for (const Out* o : {&pre, &str, &nos}) {
      ASSERT_TRUE(!o->vel.empty());
      EXPECT_TRUE(o->text == ref.text);
      EXPECT_TRUE(o->vel == ref.vel);
    }
    // Engaged, not merely equal: the FP8 model holds its projections as
    // codes, so the set it cached is well under the dense one's.
    std::printf("[krea2_native] %s held: dense %zu B, fp8 %zu B\n",
                label.c_str(), ref.held, pre.held);
    EXPECT_TRUE(pre.held > 0 && pre.held * 10 < ref.held * 7);

    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::remove_all(dense, ec);
  };
  // 0x27 is small in either format: < 0.24 as E4M3, < 0.03 as E5M2.
  std::mt19937 rng(2026);
  check("F8_E4M3", tiny_native_(rng, "", 0x27, "F8_E4M3"));
  check("F8_E5M2", tiny_native_(rng, "", 0x27, "F8_E5M2"));
  // SCALED, both spellings: per-tensor through the legacy marker, and
  // per-ROW through comfy_quant records over U8-stored weights -- the
  // scaled expansion kernel, the scale refill in the slots and the U8
  // dtype all have to agree with the dense conversion to the bit.
  {
    std::vector<T> ts = tiny_native_(rng, "", 0x27, "F8_E4M3");
    scale_native_(ts, rng, ScaleStyle::kLegacy);
    check("scaled legacy", ts);
  }
  {
    std::vector<T> ts = tiny_native_(rng, "", 0x27, "F8_E4M3");
    scale_native_(ts, rng, ScaleStyle::kComfyU8Row);
    check("scaled comfy_quant per-row", ts);
  }
}

// The transformer's own refusals: a layer in an encoding it cannot decode
// (NVFP4) and a native file with a DiT tensor the translation cannot place
// both load as NOTHING, with the reason in the log -- not as a model
// missing a term.
TEST(krea2_native, loader_refuses_what_it_cannot_read)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  CerrSilencer hush;
  std::mt19937 rng(9);
  const fs::path dir = scratch_("refuse");
  fs::create_directories(dir);
  const MetalKrea2Transformer::Config cfg = tiny_cfg_();
  const std::pair<const char*, T> extra[] = {
      {"nvfp4.safetensors",
       T{"blocks.0.attn.wq.comfy_quant", "U8", {18},
         str_bytes_("{\"format\":\"nvfp4\"}")}},
      {"stray.safetensors",
       T{"blocks.0.attn.wq.bias", "F8_E4M3", {128},
         std::vector<std::uint8_t>(128, 0)}},
  };
  for (const auto& [name, t] : extra) {
    std::vector<T> ts = tiny_native_(rng, "", 0x27);
    ts.push_back(t);
    const fs::path f = dir / name;
    ASSERT_TRUE(write_safetensors_(f, ts));
    auto ws = WeightSet::open(f.string(), nullptr);
    ASSERT_TRUE(ws != nullptr);
    if (!ws) { continue; }
    EXPECT_TRUE(MetalKrea2Transformer::load(ws, mc, cfg) == nullptr);
  }
  // And the same file without the extra tensor loads -- so the two
  // refusals above are about the tensor, not the harness.
  {
    const fs::path f = dir / "clean.safetensors";
    ASSERT_TRUE(write_safetensors_(f, tiny_native_(rng, "", 0x27)));
    auto ws = WeightSet::open(f.string(), nullptr);
    ASSERT_TRUE(ws != nullptr);
    if (ws) {
      EXPECT_TRUE(MetalKrea2Transformer::load(ws, mc, cfg) != nullptr);
    }
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// Every spelling of scaled FP8 in circulation decodes to the same values,
// and every one this cannot decode is refused BY NAME -- never read as
// codes. One [4, 8] weight per file, so the arithmetic is checked
// directly: value = bf16(f32(code) * scale), nearest-even.
TEST(fp8_layout, every_scaled_spelling_decodes_alike)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  CerrSilencer hush;
  const fs::path dir = scratch_("f8l");
  fs::create_directories(dir);
  std::mt19937 rng(11);
  std::uniform_int_distribution<int> cd(0, 0x7E);   // never the NaN code
  std::vector<std::uint8_t> codes(32);
  for (auto& c : codes) { c = (std::uint8_t)cd(rng); }
  const std::vector<float> row_s = {0.37f, 1.3f, 0.71f, 1.9f};
  const float one_s = 0.4375f * 1.1f;
  auto want = [&](fp8::Format f, bool per_row) {
    std::vector<std::uint16_t> w(32);
    for (int i = 0; i < 32; ++i) {
      const float s = per_row ? row_s[(std::size_t)(i / 8)] : one_s;
      w[(std::size_t)i] = bf16_rne_(fp8::to_f32(codes[(std::size_t)i], f) *
                                    s);
    }
    return w;
  };
  const std::string e4 = "{\"format\": \"float8_e4m3fn\"}";
  const std::string e5 = "{\"format\": \"float8_e5m2\"}";
  struct Case {
    const char* label;
    std::vector<T> ts;
    std::map<std::string, std::string> meta;
    fp8::Format f;
    bool per_row;
    std::size_t aux;
  };
  const std::vector<Case> cases = {
      {"legacy marker + scale_weight",
       {{"a.weight", "F8_E4M3", {4, 8}, codes},
        {"a.scale_weight", "F32", {}, f32_bytes_({one_s})},
        {"scaled_fp8", "F8_E4M3", {}, {0}}},
       {}, fp8::Format::kE4M3, false, 2},
      {"comfy_quant over U8",
       {{"a.weight", "U8", {4, 8}, codes},
        {"a.comfy_quant", "U8", {(std::int64_t)e4.size()}, str_bytes_(e4)},
        {"a.weight_scale", "F32", {}, f32_bytes_({one_s})}},
       {}, fp8::Format::kE4M3, false, 2},
      {"header metadata over U8",
       {{"a.weight", "U8", {4, 8}, codes},
        {"a.weight_scale", "F32", {1}, f32_bytes_({one_s})}},
       {{"_quantization_metadata",
         "{\\\"layers\\\": {\\\"a\\\": {\\\"format\\\": "
         "\\\"float8_e4m3fn\\\"}}}"}},
       fp8::Format::kE4M3, false, 1},
      {"per-row scale",
       {{"a.weight", "F8_E4M3", {4, 8}, codes},
        {"a.weight_scale", "F32", {4, 1}, f32_bytes_(row_s)}},
       {}, fp8::Format::kE4M3, true, 1},
      {"E5M2 via comfy_quant",
       {{"a.weight", "U8", {4, 8}, codes},
        {"a.comfy_quant", "U8", {(std::int64_t)e5.size()}, str_bytes_(e5)},
        {"a.weight_scale", "F32", {}, f32_bytes_({one_s})}},
       {}, fp8::Format::kE5M2, false, 2},
  };
  for (const Case& c : cases) {
    const fs::path f = dir / "one.safetensors";
    ASSERT_TRUE(write_safetensors_(f, c.ts, c.meta));
    auto w = MetalLlamaWeights::open(f.string());
    ASSERT_TRUE(w.has_value());
    if (!w.has_value()) { continue; }
    fp8::Layout l;
    std::string err;
    const bool ok = fp8::scan(*w, &l, &err);
    if (!ok) { std::printf("[fp8_layout] %s: %s\n", c.label, err.c_str()); }
    ASSERT_TRUE(ok);
    const fp8::Weight* fw = l.find("a.weight");
    ASSERT_TRUE(fw != nullptr);
    if (fw == nullptr) { continue; }
    EXPECT_TRUE(fw->format == c.f);
    EXPECT_TRUE(!fw->scale.empty());
    EXPECT_TRUE(fw->per_row == c.per_row);
    EXPECT_TRUE(l.aux.size() == c.aux);
    std::vector<float> sc;
    ASSERT_TRUE(fp8::read_scale(*w, *fw, 4, &sc, nullptr));
    std::vector<std::uint16_t> got(32);
    fp8::decode_bf16(codes.data(), got.data(), 4, 8, fw->format, sc);
    const bool same = got == want(c.f, c.per_row);
    if (!same) { std::printf("[fp8_layout] %s: values differ\n", c.label); }
    EXPECT_TRUE(same);
    // And through the quantizer's view: the scale folded in, the
    // encoding's own tensors hidden.
    if (mc != nullptr) {
      auto qs = QuantSource::open(f.string(), &err);
      ASSERT_TRUE(qs.has_value());
      if (!qs.has_value()) { continue; }
      EXPECT_TRUE(qs->tensor_names().size() == 1);
      auto b = qs->load("a.weight", mc);
      ASSERT_TRUE(b.byte_size() == 64);
      EXPECT_TRUE(std::memcmp(b.contents(), want(c.f, c.per_row).data(),
                              64) == 0);
      EXPECT_TRUE(qs->n_scaled() == 1);
    }
  }
  // Refused, each with the reason in the message.
  const std::string nv = "{\"format\": \"nvfp4\"}";
  const std::vector<std::pair<std::vector<T>, const char*>> bad = {
      {{{"a.weight", "U8", {4, 8}, codes},
        {"a.comfy_quant", "U8", {(std::int64_t)nv.size()}, str_bytes_(nv)},
        {"a.weight_scale", "F32", {}, f32_bytes_({one_s})}},
       "nvfp4"},
      {{{"a.weight", "F8_E4M3", {4, 8}, codes},
        {"a.weight_scale", "F32", {4, 2},
         f32_bytes_({1, 1, 1, 1, 1, 1, 1, 1})}},
       "BLOCK"},
      {{{"a.weight", "F8_E4M3", {4, 8}, codes},
        {"a.weight_scale", "F32", {}, f32_bytes_({one_s})},
        {"a.weight_scale_2", "F32", {}, f32_bytes_({one_s})}},
       "second-stage"},
      {{{"a.weight", "BF16", {4, 4}, std::vector<std::uint8_t>(32, 0)},
        {"a.weight_scale", "F32", {}, f32_bytes_({one_s})}},
       "a.weight"},
  };
  for (const auto& [ts, why] : bad) {
    const fs::path f = dir / "bad.safetensors";
    ASSERT_TRUE(write_safetensors_(f, ts));
    auto w = MetalLlamaWeights::open(f.string());
    ASSERT_TRUE(w.has_value());
    if (!w.has_value()) { continue; }
    fp8::Layout l;
    std::string err;
    EXPECT_FALSE(fp8::scan(*w, &l, &err));
    const bool named = err.find(why) != std::string::npos;
    if (!named) { std::printf("[fp8_layout] '%s': %s\n", why, err.c_str()); }
    EXPECT_TRUE(named);
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// FP8 LoRA files, plain and scaled, apply EXACTLY as their bf16 twins --
// files holding the values the FP8 ones decode to -- both at runtime (a
// dense tiny Krea-2 with the adapter bound) and through lora-fuse.
TEST(krea2_native, fp8_lora_matches_its_bf16_twin)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  CerrSilencer hush;
  std::mt19937 rng(77);
  const fs::path dir = scratch_("lora");
  fs::create_directories(dir);
  // The base: a tiny model, converted dense so only the LoRA is FP8.
  const fs::path basef = dir / "base_fp8.safetensors";
  ASSERT_TRUE(write_safetensors_(basef, tiny_native_(rng, "", 0x27)));
  const fs::path base = dir / "base";
  {
    QuantizeOptions opt;
    opt.bits = 16;
    std::string err;
    ASSERT_TRUE(ModelQuantizer(mc).run(basef.string(), base.string(), opt,
                                       &err));
  }
  // Rank-4 factors on four projections, as FP8 codes.
  const int R = 4, H = 128, FF = 256;
  struct Mod { std::string name; int n, k; };
  const std::vector<Mod> mods = {
      {"transformer_blocks.0.attn.to_q", H, H},
      {"transformer_blocks.0.ff.down", H, FF},
      {"transformer_blocks.1.attn.to_v", H, H},
      {"transformer_blocks.1.ff.up", FF, H}};
  std::uniform_int_distribution<int> cd(0, 0x2F);
  std::uniform_int_distribution<int> sg(0, 1);
  std::uniform_real_distribution<float> sd(0.37f, 1.9f);
  std::vector<T> plain, scaled, twin_plain, twin_scaled;
  auto factor = [&](const std::string& name, int rows, int cols) {
    std::vector<std::uint8_t> c((std::size_t)rows * cols);
    for (auto& b : c) {
      b = (std::uint8_t)(cd(rng) | (sg(rng) ? 0x80 : 0));
    }
    const float s = sd(rng);
    std::vector<std::uint8_t> tp(c.size() * 2), ts(c.size() * 2);
    for (std::size_t i = 0; i < c.size(); ++i) {
      const float v = fp8::to_f32(c[i], fp8::Format::kE4M3);
      const std::uint16_t a = bf16_rne_(v), b = bf16_rne_(v * s);
      std::memcpy(&tp[2 * i], &a, 2);
      std::memcpy(&ts[2 * i], &b, 2);
    }
    plain.push_back({name, "F8_E4M3", {rows, cols}, c});
    scaled.push_back({name, "F8_E4M3", {rows, cols}, c});
    scaled.push_back({name.substr(0, name.size() - 7) + ".weight_scale",
                      "F32", {}, f32_bytes_({s})});
    twin_plain.push_back({name, "BF16", {rows, cols}, tp});
    twin_scaled.push_back({name, "BF16", {rows, cols}, ts});
  };
  for (const Mod& m : mods) {
    factor(m.name + ".lora_A.weight", R, m.k);
    factor(m.name + ".lora_B.weight", m.n, R);
  }
  const fs::path lp = dir / "lora_fp8.safetensors";
  const fs::path ls = dir / "lora_fp8_scaled.safetensors";
  const fs::path tp = dir / "lora_twin.safetensors";
  const fs::path tsc = dir / "lora_twin_scaled.safetensors";
  ASSERT_TRUE(write_safetensors_(lp, plain));
  ASSERT_TRUE(write_safetensors_(ls, scaled));
  ASSERT_TRUE(write_safetensors_(tp, twin_plain));
  ASSERT_TRUE(write_safetensors_(tsc, twin_scaled));

  const MetalKrea2Transformer::Config cfg = tiny_cfg_();
  const int TS = 8, GH = 4, GW = 4, IS = GH * GW;
  std::normal_distribution<float> nd(0.0f, 0.5f);
  auto f16buf = [&](std::size_t n) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)nd(rng); }
    return b;
  };
  const metal_compute::SharedBuffer ehs =
      f16buf((std::size_t)TS * cfg.n_text_layers * cfg.text_hidden);
  const metal_compute::SharedBuffer lat = f16buf((std::size_t)IS * 64);
  int bound = -1;
  auto run = [&](const std::string& lora) {
    std::vector<std::uint16_t> out;
    std::vector<MetalKrea2Transformer::LoraSpec> specs;
    if (!lora.empty()) { specs.push_back({lora, 1.0f}); }
    auto m = MetalKrea2Transformer::load(base.string(), mc, cfg, false,
                                         specs);
    if (m == nullptr) { return out; }
    bound = m->lora_modules();
    auto t = m->forward_text(ehs, TS);
    if (t.empty()) { return out; }
    auto v = m->forward_dit(t, TS, lat, IS, GH, GW, 0.5f);
    if (v.empty()) { return out; }
    const auto* vp = static_cast<const std::uint16_t*>(v.contents());
    out.assign(vp, vp + (std::size_t)IS * 64);
    return out;
  };
  const auto none = run("");
  const auto a = run(lp.string());
  EXPECT_TRUE(bound == (int)mods.size());   // every module bound
  const auto a_twin = run(tp.string());
  const auto b = run(ls.string());
  EXPECT_TRUE(bound == (int)mods.size());
  const auto b_twin = run(tsc.string());
  ASSERT_TRUE(!none.empty() && !a.empty() && !b.empty());
  EXPECT_TRUE(a == a_twin);
  EXPECT_TRUE(b == b_twin);
  EXPECT_TRUE(a != none);   // the adapter moved the output
  EXPECT_TRUE(a != b);      // and the scale is not ignored

  // lora-fuse: the scaled FP8 adapter fuses to the same bytes as its twin.
  const fs::path fa = dir / "fused_fp8", fb = dir / "fused_twin";
  std::string err;
  ASSERT_TRUE(fuse_lora(mc, base.string(), ls.string(), fa.string(), 1.0f,
                        &err));
  ASSERT_TRUE(fuse_lora(mc, base.string(), tsc.string(), fb.string(), 1.0f,
                        &err));
  auto wa = MetalLlamaWeights::open_model(fa.string());
  auto wb = MetalLlamaWeights::open_model(fb.string());
  ASSERT_TRUE(wa.has_value() && wb.has_value());
  if (wa.has_value() && wb.has_value()) {
    int same = 0;
    for (const Mod& m : mods) {
      auto x = wa->load(m.name + ".weight", mc);
      auto y = wb->load(m.name + ".weight", mc);
      if (!x.empty() && x.byte_size() == y.byte_size() &&
          std::memcmp(x.contents(), y.contents(), x.byte_size()) == 0) {
        ++same;
      }
    }
    EXPECT_TRUE(same == (int)mods.size());
  }
  // An FP8 BASE is refused with the reason, not as an unknown dtype.
  EXPECT_FALSE(fuse_lora(mc, basef.string(), tp.string(),
                         (dir / "fused_bad").string(), 1.0f, &err));
  std::error_code ec;
  fs::remove_all(dir, ec);
}
