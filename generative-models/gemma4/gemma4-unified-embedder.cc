#include "generative-models/gemma4/gemma4-unified-embedder.h"

#include "generative-models/shared/gguf-file.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>

namespace vpipe::genai {

namespace {

namespace fs = std::filesystem;

// Dequantise a 2-D GGUF weight [in, out] (ne order) into a row-major
// [out, in] f32 buffer. Returns false on missing tensor.
bool
load_weight_(const GgufFile& g, const std::string& name,
             std::vector<float>* out, std::int64_t* in_out,
             std::int64_t* out_out)
{
  const GgufFile::Tensor* t = g.tensor(name);
  if (t == nullptr || t->dims.size() < 2) { return false; }
  const std::int64_t in = t->dims[0];
  const std::int64_t outn = t->dims[1];
  out->assign(static_cast<std::size_t>(in * outn), 0.0f);
  for (std::int64_t j = 0; j < outn; ++j) {
    if (!g.dequant_row_f32(*t, j,
                           out->data() + static_cast<std::size_t>(j * in))) {
      return false;
    }
  }
  if (in_out) { *in_out = in; }
  if (out_out) { *out_out = outn; }
  return true;
}

bool
load_vec_(const GgufFile& g, const std::string& name, std::vector<float>* out)
{
  const GgufFile::Tensor* t = g.tensor(name);
  if (t == nullptr) { return false; }
  out->assign(static_cast<std::size_t>(t->numel()), 0.0f);
  return g.dequant_all_f32(*t, out->data());
}

// Convert one bf16 lane (top 16 bits of an f32) to f32.
inline float
bf16_to_f32_(std::uint16_t h)
{
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// Convert one IEEE f16 lane to f32.
inline float
f16_to_f32_(std::uint16_t h)
{
  const std::uint32_t sign = (std::uint32_t(h) & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1fu;
  const std::uint32_t man = h & 0x3ffu;
  std::uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      int e = -1;
      std::uint32_t m = man;
      do { m <<= 1; ++e; } while ((m & 0x400u) == 0);
      m &= 0x3ffu;
      bits = sign | ((127 - 15 - e) << 23) | (m << 13);
    }
  } else if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// Read a named safetensors tensor (bf16 / f16 / f32) into a row-major f32
// buffer, keeping the on-disk element order. Tries `name` then a
// `model.`-prefixed spelling. Returns false on a missing tensor. `numel`
// (if non-null) receives the element count.
bool
st_load_f32_(const MetalLlamaWeights& w, metal_compute::MetalCompute* mc,
             const std::string& name, std::vector<float>* out,
             std::int64_t* numel)
{
  const MetalLlamaWeights::TensorInfo* ti = w.info(name);
  std::string key = name;
  if (ti == nullptr) {
    key = "model." + name;
    ti = w.info(key);
  }
  if (ti == nullptr) { return false; }
  metal_compute::SharedBuffer buf = w.load(key, mc);
  if (buf.empty()) { return false; }
  std::int64_t n = 1;
  for (std::int64_t d : ti->shape) { n *= d; }
  out->assign(static_cast<std::size_t>(n), 0.0f);
  const void* src = buf.contents();
  if (ti->dtype == "F32") {
    std::memcpy(out->data(), src,
                static_cast<std::size_t>(n) * sizeof(float));
  } else if (ti->dtype == "BF16") {
    const auto* h = static_cast<const std::uint16_t*>(src);
    for (std::int64_t i = 0; i < n; ++i) {
      (*out)[static_cast<std::size_t>(i)] = bf16_to_f32_(h[i]);
    }
  } else if (ti->dtype == "F16") {
    const auto* h = static_cast<const std::uint16_t*>(src);
    for (std::int64_t i = 0; i < n; ++i) {
      (*out)[static_cast<std::size_t>(i)] = f16_to_f32_(h[i]);
    }
  } else {
    return false;
  }
  if (numel) { *numel = n; }
  return true;
}

// Reorder the length-(C*P*P) fastest axis of each of `rows` rows from HF's
// [KH,KW,C] (channels innermost) patch flatten into the forward's
// [C,KH,KW] (channels outermost). llama.cpp's mmproj converter bakes this
// permutation in; the raw safetensors keep HF order.
void
reorder_patch_axis_(std::vector<float>* v, int rows, int C, int P)
{
  const int inn = C * P * P;
  std::vector<float> tmp(static_cast<std::size_t>(rows) * inn);
  for (int r = 0; r < rows; ++r) {
    const float* src = v->data() + static_cast<std::size_t>(r) * inn;
    float* dst = tmp.data() + static_cast<std::size_t>(r) * inn;
    for (int c = 0; c < C; ++c) {
      for (int kh = 0; kh < P; ++kh) {
        for (int kw = 0; kw < P; ++kw) {
          dst[(c * P + kh) * P + kw] = src[(kh * P + kw) * C + c];
        }
      }
    }
  }
  *v = std::move(tmp);
}

// LayerNorm over a length-D vector in place: (x-mean)/sqrt(var+eps)*w + b.
void
layernorm_(float* x, int D, const float* w, const float* b, float eps)
{
  double mean = 0.0;
  for (int i = 0; i < D; ++i) { mean += x[i]; }
  mean /= D;
  double var = 0.0;
  for (int i = 0; i < D; ++i) {
    const double d = x[i] - mean;
    var += d * d;
  }
  var /= D;
  const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
  for (int i = 0; i < D; ++i) {
    x[i] = (static_cast<float>(x[i] - mean) * inv) * w[i] + b[i];
  }
}

// Weightless RMSNorm: x / sqrt(mean(x^2) + eps).
void
rmsnorm_(float* x, int D, float eps)
{
  double ms = 0.0;
  for (int i = 0; i < D; ++i) { ms += static_cast<double>(x[i]) * x[i]; }
  ms /= D;
  const float inv = 1.0f / std::sqrt(static_cast<float>(ms) + eps);
  for (int i = 0; i < D; ++i) { x[i] *= inv; }
}

// out[j] = dot(x[0:in], W[j*in : j*in+in]) (+ bias[j]).  W is [out, in].
void
matvec_(const float* x, const float* W, const float* bias, int in, int out,
        float* dst)
{
  for (int j = 0; j < out; ++j) {
    const float* w = W + static_cast<std::size_t>(j) * in;
    float acc = 0.0f;
    for (int i = 0; i < in; ++i) { acc += x[i] * w[i]; }
    dst[j] = bias ? acc + bias[j] : acc;
  }
}

int
round_by_(int x, int f)
{
  return static_cast<int>(std::lround(static_cast<double>(x) / f)) * f;
}
int
ceil_by_(double x, int f)
{
  return static_cast<int>(std::ceil(x / f)) * f;
}
int
floor_by_(double x, int f)
{
  return static_cast<int>(std::floor(x / f)) * f;
}

}  // namespace

bool
Gemma4UnifiedEmbedder::has_unified_safetensors(const std::string& model_dir)
{
  auto w = MetalLlamaWeights::open_model(model_dir);
  if (!w) { return false; }
  return w->has("model.embed_vision.embedding_projection.weight") ||
         w->has("embed_vision.embedding_projection.weight") ||
         w->has("model.embed_audio.embedding_projection.weight") ||
         w->has("embed_audio.embedding_projection.weight");
}

std::unique_ptr<Gemma4UnifiedEmbedder>
Gemma4UnifiedEmbedder::load_safetensors(const std::string& model_dir,
                                        metal_compute::MetalCompute* mc)
{
  if (mc == nullptr) { return nullptr; }
  auto w = MetalLlamaWeights::open_model(model_dir);
  if (!w) { return nullptr; }

  auto m = std::unique_ptr<Gemma4UnifiedEmbedder>(new Gemma4UnifiedEmbedder());

  // ---- Vision adaptor (model.vision_embedder.* + model.embed_vision.*) ----
  // patch_dense.weight is [out=embed, in=patch_in] row-major -- the SAME
  // layout load_weight_ produces from the GGUF v.patch_embd.weight, so a
  // straight bf16->f32 copy suffices (no transpose).
  const bool have_vis =
      st_load_f32_(*w, mc, "vision_embedder.patch_dense.weight", &m->_w_patch,
                   nullptr);
  if (have_vis) {
    const MetalLlamaWeights::TensorInfo* pd =
        w->info("model.vision_embedder.patch_dense.weight");
    if (pd == nullptr) {
      pd = w->info("vision_embedder.patch_dense.weight");
    }
    const bool ok =
        pd != nullptr && pd->shape.size() == 2 &&
        st_load_f32_(*w, mc, "vision_embedder.patch_dense.bias",
                     &m->_b_patch, nullptr) &&
        st_load_f32_(*w, mc, "vision_embedder.patch_ln1.weight", &m->_ln1_w,
                     nullptr) &&
        st_load_f32_(*w, mc, "vision_embedder.patch_ln1.bias", &m->_ln1_b,
                     nullptr) &&
        st_load_f32_(*w, mc, "vision_embedder.patch_ln2.weight", &m->_ln2_w,
                     nullptr) &&
        st_load_f32_(*w, mc, "vision_embedder.patch_ln2.bias", &m->_ln2_b,
                     nullptr) &&
        st_load_f32_(*w, mc, "vision_embedder.pos_norm.weight", &m->_ln3_w,
                     nullptr) &&
        st_load_f32_(*w, mc, "vision_embedder.pos_norm.bias", &m->_ln3_b,
                     nullptr) &&
        st_load_f32_(*w, mc, "embed_vision.embedding_projection.weight",
                     &m->_w_proj, nullptr);
    // pos_embedding is [pos_max, 2, embed] row-major; the forward expects the
    // GGUF layout [2, pos_max, embed] (block0 = column table, block1 = row
    // table). Transpose the two leading axes (== what llama.cpp's converter
    // did), so _pos is element-identical to the GGUF path.
    std::vector<float> pos_st;
    std::int64_t pos_n = 0;
    const bool have_pos =
        st_load_f32_(*w, mc, "vision_embedder.pos_embedding", &pos_st,
                     &pos_n);
    if (ok && have_pos) {
      m->_embed = static_cast<int>(pd->shape[0]);     // 3840
      m->_patch_in = static_cast<int>(pd->shape[1]);  // 6912
      const std::size_t D = static_cast<std::size_t>(m->_embed);
      const std::size_t pm =
          static_cast<std::size_t>(pos_n) / (D * 2);
      m->_pos_max = static_cast<int>(pm);
      m->_pos.assign(static_cast<std::size_t>(pos_n), 0.0f);
      for (std::size_t p = 0; p < pm; ++p) {
        for (std::size_t s = 0; s < 2; ++s) {
          const float* srow = pos_st.data() + (p * 2 + s) * D;
          float* drow = m->_pos.data() + (s * pm + p) * D;
          std::memcpy(drow, srow, D * sizeof(float));
        }
      }
      // Patch-space (6912) tensors flatten as [KH,KW,C] in HF; permute the
      // ln1 gamma/beta + patch_dense columns to the [C,KH,KW] order the
      // forward's im2col uses.
      const int C = 3;
      const int P = static_cast<int>(
          std::lround(std::sqrt(static_cast<double>(m->_patch_in) / C)));
      reorder_patch_axis_(&m->_ln1_w, 1, C, P);
      reorder_patch_axis_(&m->_ln1_b, 1, C, P);
      reorder_patch_axis_(&m->_w_patch, m->_embed, C, P);
      m->_has_vision = true;
    }
  }

  // ---- Audio adaptor (model.embed_audio.embedding_projection.weight) ------
  // [out=embed, in=audio_frame] row-major -- direct copy, as the GGUF path.
  if (st_load_f32_(*w, mc, "embed_audio.embedding_projection.weight",
                   &m->_w_aproj, nullptr)) {
    const MetalLlamaWeights::TensorInfo* ap =
        w->info("model.embed_audio.embedding_projection.weight");
    if (ap == nullptr) {
      ap = w->info("embed_audio.embedding_projection.weight");
    }
    if (ap != nullptr && ap->shape.size() == 2) {
      m->_audio_frame = static_cast<int>(ap->shape[1]);   // 640
      if (m->_embed == 0) { m->_embed = static_cast<int>(ap->shape[0]); }
      m->_has_audio = true;
    }
  }

  if (!m->_has_vision && !m->_has_audio) { return nullptr; }
  return m;
}

std::string
Gemma4UnifiedEmbedder::find_mmproj(const std::string& model_dir)
{
  std::error_code ec;
  if (!fs::is_directory(model_dir, ec)) { return std::string(); }
  for (const auto& e : fs::directory_iterator(model_dir, ec)) {
    const fs::path p = e.path();
    if (p.extension() != ".gguf") { continue; }
    if (p.filename().string().rfind("mmproj", 0) == 0) { return p.string(); }
  }
  return std::string();
}

std::unique_ptr<Gemma4UnifiedEmbedder>
Gemma4UnifiedEmbedder::load(const std::string& mmproj_path)
{
  auto g = GgufFile::open(mmproj_path);
  if (!g) { return nullptr; }
  const auto arch = g->get_string("general.architecture");
  if (!arch || *arch != "clip") { return nullptr; }

  auto m = std::unique_ptr<Gemma4UnifiedEmbedder>(new Gemma4UnifiedEmbedder());

  const auto vproj = g->get_string("clip.vision.projector_type");
  const auto aproj = g->get_string("clip.audio.projector_type");
  std::int64_t in = 0, out = 0;

  if (vproj && *vproj == "gemma4uv") {
    const bool ok =
        load_weight_(*g, "v.patch_embd.weight", &m->_w_patch, &in, &out) &&
        load_vec_(*g, "v.patch_embd.bias", &m->_b_patch) &&
        load_vec_(*g, "v.patch_norm.1.weight", &m->_ln1_w) &&
        load_vec_(*g, "v.patch_norm.1.bias", &m->_ln1_b) &&
        load_vec_(*g, "v.patch_norm.2.weight", &m->_ln2_w) &&
        load_vec_(*g, "v.patch_norm.2.bias", &m->_ln2_b) &&
        load_vec_(*g, "v.patch_norm.3.weight", &m->_ln3_w) &&
        load_vec_(*g, "v.patch_norm.3.bias", &m->_ln3_b) &&
        load_vec_(*g, "v.position_embd.weight", &m->_pos) &&
        load_weight_(*g, "mm.input_projection.weight", &m->_w_proj,
                     nullptr, nullptr);
    if (ok) {
      m->_embed = static_cast<int>(out);          // 3840
      m->_patch_in = static_cast<int>(in);        // 6912
      // position_embd is [embed, pos_max, 2]; pos_max = numel/(embed*2).
      m->_pos_max =
          static_cast<int>(m->_pos.size() / (std::size_t)(m->_embed * 2));
      m->_has_vision = true;
    }
  }

  if (aproj && *aproj == "gemma4ua") {
    std::int64_t ain = 0, aout = 0;
    if (load_weight_(*g, "mm.a.input_projection.weight", &m->_w_aproj, &ain,
                     &aout)) {
      m->_audio_frame = static_cast<int>(ain);    // 640
      if (m->_embed == 0) { m->_embed = static_cast<int>(aout); }
      m->_has_audio = true;
    }
  }

  if (!m->_has_vision && !m->_has_audio) { return nullptr; }
  return m;
}

void
Gemma4UnifiedEmbedder::smart_resize(int H, int W, int* th, int* tw) const
{
  const int f = _patch_px;                       // 48
  const double min_pixels = 40.0 * f * f;        // 92160
  const double max_pixels = 280.0 * f * f;       // 645120
  int h_bar = std::max(f, round_by_(H, f));
  int w_bar = std::max(f, round_by_(W, f));
  const double area = static_cast<double>(H) * W;
  if (static_cast<double>(h_bar) * w_bar > max_pixels) {
    const double beta = std::sqrt(area / max_pixels);
    h_bar = std::max(f, floor_by_(H / beta, f));
    w_bar = std::max(f, floor_by_(W / beta, f));
  } else if (static_cast<double>(h_bar) * w_bar < min_pixels) {
    const double beta = std::sqrt(min_pixels / area);
    h_bar = ceil_by_(H * beta, f);
    w_bar = ceil_by_(W * beta, f);
  }
  *th = h_bar;
  *tw = w_bar;
}

std::optional<Gemma4UnifiedEmbedder::EncodedImage>
Gemma4UnifiedEmbedder::encode_image(const std::uint8_t* rgb_chw, int H,
                                    int W) const
{
  if (!_has_vision || rgb_chw == nullptr || H <= 0 || W <= 0) {
    return std::nullopt;
  }
  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmVision,
                     kPerfLlmVisionBegin, 1);
  int th = 0, tw = 0;
  smart_resize(H, W, &th, &tw);

  // Corner-aligned (align_corners) bilinear resize, planar [3,H,W] u8 ->
  // [3,th,tw] f32 / 255 (mean 0, std 1). Identity when th==H && tw==W.
  // (TODO: llama.cpp uses min-scale + PAD_CEIL letterbox; aspect-preserving
  // smart-resize keeps content ~filling the target so the difference is
  // sub-pixel -- refine if a real-image token-exact check needs it.)
  std::vector<float> img(static_cast<std::size_t>(3) * th * tw);
  const double ry = (th > 1) ? static_cast<double>(H - 1) / (th - 1) : 0.0;
  const double rx = (tw > 1) ? static_cast<double>(W - 1) / (tw - 1) : 0.0;
  for (int c = 0; c < 3; ++c) {
    const std::uint8_t* src = rgb_chw + static_cast<std::size_t>(c) * H * W;
    float* dst = img.data() + static_cast<std::size_t>(c) * th * tw;
    for (int yy = 0; yy < th; ++yy) {
      const double sy = yy * ry;
      const int y0 = static_cast<int>(std::floor(sy));
      const int y1 = std::min(y0 + 1, H - 1);
      const float dy = static_cast<float>(sy - y0);
      for (int xx = 0; xx < tw; ++xx) {
        const double sx = xx * rx;
        const int x0 = static_cast<int>(std::floor(sx));
        const int x1 = std::min(x0 + 1, W - 1);
        const float dx = static_cast<float>(sx - x0);
        const float v00 = src[y0 * W + x0], v01 = src[y0 * W + x1];
        const float v10 = src[y1 * W + x0], v11 = src[y1 * W + x1];
        const float v0 = v00 * (1 - dx) + v01 * dx;
        const float v1 = v10 * (1 - dx) + v11 * dx;
        dst[yy * tw + xx] = (v0 * (1 - dy) + v1 * dy) / 255.0f;
      }
    }
  }

  const int P = _patch_px;
  const int ncols = tw / P, nrows = th / P;
  const int n = ncols * nrows;
  const int D = _embed, IN = _patch_in;

  EncodedImage r;
  r.n_tokens = n;
  r.grid_h = nrows;
  r.grid_w = ncols;
  r.rows.assign(static_cast<std::size_t>(n) * D, 0.0f);

  std::vector<float> patch(static_cast<std::size_t>(IN));
  std::vector<float> emb(static_cast<std::size_t>(D));
  for (int pr = 0; pr < nrows; ++pr) {
    for (int pc = 0; pc < ncols; ++pc) {
      // im2col: 6912 = [C, KH, KW] with KW fastest.
      for (int c = 0; c < 3; ++c) {
        const float* plane = img.data() + static_cast<std::size_t>(c) * th * tw;
        for (int kh = 0; kh < P; ++kh) {
          const float* row = plane + (std::size_t)(pr * P + kh) * tw + pc * P;
          float* pd = patch.data() + (std::size_t)(c * P + kh) * P;
          for (int kw = 0; kw < P; ++kw) { pd[kw] = row[kw]; }
        }
      }
      layernorm_(patch.data(), IN, _ln1_w.data(), _ln1_b.data(), _eps_ln);
      matvec_(patch.data(), _w_patch.data(), _b_patch.data(), IN, D,
              emb.data());
      layernorm_(emb.data(), D, _ln2_w.data(), _ln2_b.data(), _eps_ln);
      // Separable additive position embedding: tbl_x[col] + tbl_y[row].
      const float* tx = _pos.data() + (std::size_t)pc * D;
      const float* ty = _pos.data() +
          ((std::size_t)_pos_max + pr) * D;
      for (int d = 0; d < D; ++d) { emb[d] += tx[d] + ty[d]; }
      layernorm_(emb.data(), D, _ln3_w.data(), _ln3_b.data(), _eps_ln);
      rmsnorm_(emb.data(), D, _eps_rms);
      matvec_(emb.data(), _w_proj.data(), nullptr, D, D,
              r.rows.data() + (std::size_t)(pr * ncols + pc) * D);
    }
  }
  return r;
}

std::optional<Gemma4UnifiedEmbedder::EncodedAudio>
Gemma4UnifiedEmbedder::encode_audio(const float* pcm, std::size_t n) const
{
  if (!_has_audio || pcm == nullptr || n == 0) { return std::nullopt; }
  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmAudio,
                     kPerfLlmAudioBegin, static_cast<std::uint64_t>(n));
  const int F = _audio_frame, D = _embed;
  const int n_tok = static_cast<int>((n + F - 1) / F);

  EncodedAudio r;
  r.n_tokens = n_tok;
  r.rows.assign(static_cast<std::size_t>(n_tok) * D, 0.0f);

  std::vector<float> frame(static_cast<std::size_t>(F));
  for (int t = 0; t < n_tok; ++t) {
    for (int f = 0; f < F; ++f) {
      const std::size_t idx = static_cast<std::size_t>(t) * F + f;
      frame[f] = (idx < n) ? pcm[idx] : 0.0f;
    }
    rmsnorm_(frame.data(), F, _eps_rms);
    matvec_(frame.data(), _w_aproj.data(), nullptr, F, D,
            r.rows.data() + static_cast<std::size_t>(t) * D);
  }
  return r;
}

}  // namespace vpipe::genai
