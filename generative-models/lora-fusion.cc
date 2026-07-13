#include "generative-models/lora-fusion.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/quantize/safetensors-writer.h"
#include "interfaces/session-context-intf.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

float
bf16_to_f32(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t
f32_to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  // round-to-nearest-even.
  const std::uint32_t r = (u >> 16) & 1;
  u += 0x7fff + r;
  return (std::uint16_t)(u >> 16);
}

// Read a raw tensor buffer (dtype BF16/F16/F32) into f32.
std::vector<float>
to_f32(const SharedBuffer& buf, const std::string& dtype, std::size_t n)
{
  std::vector<float> v(n);
  if (dtype == "F32") {
    std::memcpy(v.data(), buf.contents(), n * 4);
  } else if (dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(buf.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = (float)s[i]; }
  } else if (dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(buf.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = bf16_to_f32(s[i]); }
  } else {
    v.clear();
  }
  return v;
}

// Encode an f32 weight back to `dtype`, returning the raw bytes.
std::vector<std::uint8_t>
from_f32(const std::vector<float>& w, const std::string& dtype)
{
  std::vector<std::uint8_t> out;
  if (dtype == "F32") {
    out.resize(w.size() * 4);
    std::memcpy(out.data(), w.data(), out.size());
  } else if (dtype == "F16") {
    out.resize(w.size() * 2);
    auto* d = reinterpret_cast<_Float16*>(out.data());
    for (std::size_t i = 0; i < w.size(); ++i) { d[i] = (_Float16)w[i]; }
  } else {   // BF16
    out.resize(w.size() * 2);
    auto* d = reinterpret_cast<std::uint16_t*>(out.data());
    for (std::size_t i = 0; i < w.size(); ++i) { d[i] = f32_to_bf16(w[i]); }
  }
  return out;
}

const std::string kSufA = ".lora_A.weight";
const std::string kSufB = ".lora_B.weight";

void
replace_all(std::string& s, const std::string& from, const std::string& to)
{
  if (from.empty()) { return; }
  std::size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) {
    s.replace(p, from.size(), to);
    p += to.size();
  }
}

// Map an ai-toolkit / ComfyUI adapter module name ("diffusion_model.*") to the
// diffusers base-weight name used by the Krea-2 DiT. Handles the three block
// containers (blocks -> transformer_blocks; txtfusion.{layerwise,refiner}_blocks
// -> text_fusion.{...}) and the per-block submodule renames (mlp -> ff; attn
// wq/wk/wv/wo/gate -> to_q/to_k/to_v/to_out.0/to_gate). Returns "" when the
// module isn't in that convention.
std::string
remap_ai_toolkit(std::string m)
{
  const std::string kPre = "diffusion_model.";
  if (m.size() < kPre.size() || m.compare(0, kPre.size(), kPre) != 0) {
    return {};
  }
  m = m.substr(kPre.size());
  if (m.compare(0, 7, "blocks.") == 0) {
    m = "transformer_" + m;              // blocks.N -> transformer_blocks.N
  } else {
    replace_all(m, "txtfusion.", "text_fusion.");   // layerwise/refiner_blocks
  }
  replace_all(m, ".mlp.", ".ff.");        // mlp.{gate,up,down} -> ff.{...}
  replace_all(m, ".attn.wq", ".attn.to_q");
  replace_all(m, ".attn.wk", ".attn.to_k");
  replace_all(m, ".attn.wv", ".attn.to_v");
  replace_all(m, ".attn.wo", ".attn.to_out.0");
  replace_all(m, ".attn.gate", ".attn.to_gate");
  return m;
}

// A resolved LoRA/LoKr adapter targeting one base weight.
struct Adapter {
  bool        lokr = false;
  // Low-rank LoRA: base += scale * (B @ A).
  std::string a, b;
  // LoKr: base += scale * kron(w1, w2); each factor is either a full matrix
  // (`w1`/`w2`) or a low-rank product (`w1a @ w1b` / `w2a @ w2b`).
  std::string w1, w1a, w1b, w2, w2a, w2b;
  std::string alpha;   // optional scalar; "" when absent
};

bool
ends_with(const std::string& s, const std::string& suf)
{
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

}  // namespace

bool
fuse_lora(MetalCompute* mc, const std::string& base_dir,
          const std::string& lora_path, const std::string& out_dir,
          float scale, std::string* err, const std::function<bool()>& stop)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  namespace fs = std::filesystem;
  if (mc == nullptr) { return fail("lora-fuse: null metal-compute"); }

  auto baseopt = MetalLlamaWeights::open_model(base_dir);
  if (!baseopt.has_value()) {
    return fail("lora-fuse: cannot open base model: " + base_dir);
  }
  auto loraopt = MetalLlamaWeights::open(lora_path);
  if (!loraopt.has_value()) {
    return fail("lora-fuse: cannot open LoRA: " + lora_path);
  }
  const MetalLlamaWeights& base = *baseopt;
  const MetalLlamaWeights& lora = *loraopt;

  // Match an adapter `<module>` to a base weight: try <module>.weight, strip
  // the leading component segment (diffusers' "transformer." etc.), then the
  // ai-toolkit / ComfyUI name remap (diffusion_model.* -> diffusers).
  auto find_base = [&](const std::string& module) -> std::string {
    if (base.info(module + ".weight") != nullptr) { return module + ".weight"; }
    const auto dot = module.find('.');
    if (dot != std::string::npos) {
      const std::string s = module.substr(dot + 1) + ".weight";
      if (base.info(s) != nullptr) { return s; }
    }
    const std::string rm = remap_ai_toolkit(module);
    if (!rm.empty() && base.info(rm + ".weight") != nullptr) {
      return rm + ".weight";
    }
    return {};
  };

  // Load a LoRA-side tensor into f32 (empty on miss / unsupported dtype).
  auto load_f32 = [&](const std::string& tn) -> std::vector<float> {
    const auto* ti = lora.info(tn);
    if (ti == nullptr) { return {}; }
    std::size_t n = 1;
    for (auto d : ti->shape) { n *= (std::size_t)d; }
    SharedBuffer b = lora.load(tn, mc);
    if (b.empty()) { return {}; }
    return to_f32(b, ti->dtype, n);
  };

  // Build base_weight_name -> Adapter (low-rank LoRA or LoKr).
  std::unordered_map<std::string, Adapter> fuse;
  // Low-rank LoRA: <module>.lora_A.weight + <module>.lora_B.weight.
  for (const std::string& name : lora.tensor_names()) {
    if (!ends_with(name, kSufA)) { continue; }
    const std::string module = name.substr(0, name.size() - kSufA.size());
    const std::string bname = module + kSufB;
    if (lora.info(bname) == nullptr) { continue; }
    const std::string base_name = find_base(module);
    if (base_name.empty()) { continue; }
    Adapter ad;
    ad.a = name;
    ad.b = bname;
    if (lora.info(module + ".alpha") != nullptr) { ad.alpha = module + ".alpha"; }
    fuse[base_name] = ad;
  }
  // LoKr: <module>.lokr_w{1,2}[ _a/_b ] (+ optional <module>.alpha).
  {
    std::unordered_set<std::string> lokr_modules;
    for (const std::string& name : lora.tensor_names()) {
      for (const char* suf : {".lokr_w1", ".lokr_w2", ".lokr_w1_a",
                              ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b"}) {
        if (ends_with(name, suf)) {
          lokr_modules.insert(name.substr(0, name.size() - std::strlen(suf)));
          break;
        }
      }
    }
    for (const std::string& module : lokr_modules) {
      const std::string base_name = find_base(module);
      if (base_name.empty() || fuse.count(base_name) != 0) { continue; }
      Adapter ad;
      ad.lokr = true;
      ad.w1  = lora.info(module + ".lokr_w1") ? module + ".lokr_w1" : "";
      ad.w1a = lora.info(module + ".lokr_w1_a") ? module + ".lokr_w1_a" : "";
      ad.w1b = lora.info(module + ".lokr_w1_b") ? module + ".lokr_w1_b" : "";
      ad.w2  = lora.info(module + ".lokr_w2") ? module + ".lokr_w2" : "";
      ad.w2a = lora.info(module + ".lokr_w2_a") ? module + ".lokr_w2_a" : "";
      ad.w2b = lora.info(module + ".lokr_w2_b") ? module + ".lokr_w2_b" : "";
      if (lora.info(module + ".alpha") != nullptr) { ad.alpha = module + ".alpha"; }
      fuse[base_name] = ad;
    }
  }
  if (fuse.empty()) {
    return fail("lora-fuse: no LoRA tensors matched the base model");
  }

  std::error_code ec;
  fs::create_directories(out_dir, ec);
  SafetensorsWriter wr(out_dir, 5ull << 30);

  // Resolve a LoKr factor to a dense [rows,cols] matrix: a full `w` tensor, or
  // the low-rank product `wa[rows,r] @ wb[r,cols]`.
  auto lokr_factor = [&](const std::string& w, const std::string& wa,
                         const std::string& wb, std::vector<float>& out,
                         int& rows, int& cols) -> bool {
    if (!w.empty()) {
      const auto* ti = lora.info(w);
      if (ti == nullptr || ti->shape.size() != 2) { return false; }
      rows = (int)ti->shape[0];
      cols = (int)ti->shape[1];
      out = load_f32(w);
      return !out.empty();
    }
    const auto* ai = lora.info(wa);
    const auto* bi = lora.info(wb);
    if (ai == nullptr || bi == nullptr || ai->shape.size() != 2 ||
        bi->shape.size() != 2 || ai->shape[1] != bi->shape[0]) {
      return false;
    }
    rows = (int)ai->shape[0];
    cols = (int)bi->shape[1];
    const int r = (int)ai->shape[1];
    const std::vector<float> A = load_f32(wa), B = load_f32(wb);
    if (A.empty() || B.empty()) { return false; }
    out.assign((std::size_t)rows * cols, 0.0f);
    for (int i = 0; i < rows; ++i) {
      for (int t = 0; t < r; ++t) {
        const float av = A[(std::size_t)i * r + t];
        if (av == 0.0f) { continue; }
        const float* bp = B.data() + (std::size_t)t * cols;
        float* op = out.data() + (std::size_t)i * cols;
        for (int j = 0; j < cols; ++j) { op[j] += av * bp[j]; }
      }
    }
    return true;
  };

  std::vector<std::string> names = base.tensor_names();
  int n_fused = 0, n_pass = 0, n_lokr = 0;
  for (const std::string& name : names) {
    if (stop()) { return fail("lora-fuse: stopped"); }
    const auto* ti = base.info(name);
    if (ti == nullptr) { continue; }
    SharedBuffer wb = base.load(name, mc);
    if (wb.empty()) { return fail("lora-fuse: base load failed: " + name); }

    auto it = fuse.find(name);
    if (it == fuse.end() || ti->shape.size() != 2) {
      // pass through byte-for-byte.
      if (!wr.add(name, ti->dtype, ti->shape, wb.contents(), ti->nbytes)) {
        return fail("lora-fuse: passthrough write failed: " + name);
      }
      ++n_pass;
      continue;
    }

    const int N = (int)ti->shape[0], K = (int)ti->shape[1];
    std::vector<float> W = to_f32(wb, ti->dtype, (std::size_t)N * K);
    if (W.empty()) { return fail("lora-fuse: unsupported dtype for " + name); }
    const Adapter& ad = it->second;

    if (!ad.lokr) {
      const auto* ai = lora.info(ad.a);
      const auto* bi = lora.info(ad.b);
      if (ai == nullptr || bi == nullptr || ai->shape.size() != 2 ||
          bi->shape.size() != 2) {
        return fail("lora-fuse: bad LoRA shapes for " + name);
      }
      const int rank = (int)ai->shape[0];
      if ((int)ai->shape[1] != K || (int)bi->shape[0] != N ||
          (int)bi->shape[1] != rank) {
        return fail("lora-fuse: LoRA/base shape mismatch for " + name);
      }
      const std::vector<float> A = load_f32(ad.a), B = load_f32(ad.b);
      if (A.empty() || B.empty()) {
        return fail("lora-fuse: LoRA load failed for " + name);
      }
      // Effective scale: user scale, times alpha/rank if the adapter carries an
      // alpha (kohya / ai-toolkit); diffusers LoRAs omit alpha (scale=1).
      float s = scale;
      if (!ad.alpha.empty()) {
        const std::vector<float> av = load_f32(ad.alpha);
        if (!av.empty() && rank > 0) { s *= av[0] / (float)rank; }
      }
      // W += s * (B @ A): rank-1 updates (cache-friendly over A/W rows).
      for (int n = 0; n < N; ++n) {
        float* wr_ = W.data() + (std::size_t)n * K;
        const float* br = B.data() + (std::size_t)n * rank;
        for (int r = 0; r < rank; ++r) {
          const float b = s * br[r];
          if (b == 0.0f) { continue; }
          const float* ar = A.data() + (std::size_t)r * K;
          for (int k = 0; k < K; ++k) { wr_[k] += b * ar[k]; }
        }
      }
    } else {
      // LoKr: W += s * kron(w1[a,b], w2[c,d]); N=a*c, K=b*d.
      std::vector<float> w1, w2;
      int a = 0, b = 0, c = 0, d = 0;
      if (!lokr_factor(ad.w1, ad.w1a, ad.w1b, w1, a, b) ||
          !lokr_factor(ad.w2, ad.w2a, ad.w2b, w2, c, d)) {
        return fail("lora-fuse: bad LoKr factors for " + name);
      }
      if ((std::int64_t)a * c != N || (std::int64_t)b * d != K) {
        return fail("lora-fuse: LoKr/base shape mismatch for " + name);
      }
      // Full-matrix LoKr uses no alpha rescaling (LyCORIS sets scale=1); a
      // low-rank factor contributes alpha/rank when an alpha is present.
      float s = scale;
      const bool decomposed = ad.w1.empty() || ad.w2.empty();
      if (decomposed && !ad.alpha.empty()) {
        const std::vector<float> av = load_f32(ad.alpha);
        const int r = ad.w2.empty() ? (int)lora.info(ad.w2a)->shape[1]
                                    : (int)lora.info(ad.w1a)->shape[1];
        if (!av.empty() && r > 0) { s *= av[0] / (float)r; }
      }
      for (int I = 0; I < N; ++I) {
        const int i1 = I / c, i2 = I % c;
        float* wr_ = W.data() + (std::size_t)I * K;
        const float* w1r = w1.data() + (std::size_t)i1 * b;
        const float* w2r = w2.data() + (std::size_t)i2 * d;
        for (int j1 = 0; j1 < b; ++j1) {
          const float w1v = s * w1r[j1];
          if (w1v == 0.0f) { continue; }
          float* wp = wr_ + (std::size_t)j1 * d;
          for (int j2 = 0; j2 < d; ++j2) { wp[j2] += w1v * w2r[j2]; }
        }
      }
      ++n_lokr;
    }

    const std::vector<std::uint8_t> bytes = from_f32(W, ti->dtype);
    if (!wr.add(name, ti->dtype, ti->shape, bytes.data(), bytes.size())) {
      return fail("lora-fuse: fused write failed: " + name);
    }
    ++n_fused;
  }
  if (!wr.close()) { return fail("lora-fuse: finalize shards failed"); }

  // Copy config.json (+ any other small json sidecars) verbatim.
  for (const char* sc : {"config.json"}) {
    const fs::path src = fs::path(base_dir) / sc;
    if (fs::exists(src)) {
      fs::copy_file(src, fs::path(out_dir) / sc,
                    fs::copy_options::overwrite_existing, ec);
    }
  }

  if (mc->session() != nullptr) {
    mc->session()->log_normal(fmt(
        "lora-fuse: {} weights fused ({} LoKr, scale {}), {} passthrough -> {}",
        n_fused, n_lokr, scale, n_pass, out_dir));
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
