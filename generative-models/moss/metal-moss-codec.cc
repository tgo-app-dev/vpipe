#include "generative-models/moss/metal-moss-codec.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {

// FNV-1a fingerprint of a model dir's *.safetensors (name+size+mtime), so a
// converted-weight cache is invalidated when the source checkpoint changes.
std::uint64_t codec_source_fingerprint_(const std::string& dir) {
  namespace fs = std::filesystem;
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&](std::uint64_t v) { h = (h ^ v) * 1099511628211ull; };
  std::error_code ec;
  std::vector<std::string> names;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (ec) { break; }
    if (e.path().extension() == ".safetensors") {
      names.push_back(e.path().filename().string());
    }
  }
  std::sort(names.begin(), names.end());   // order-independent
  for (const auto& n : names) {
    const fs::path p = fs::path(dir) / n;
    const auto sz = fs::file_size(p, ec);
    if (ec) { continue; }
    const auto mt = fs::last_write_time(p, ec);
    if (ec) { continue; }
    for (char c : n) { mix((std::uint64_t)(unsigned char)c); }
    mix((std::uint64_t)sz);
    mix((std::uint64_t)mt.time_since_epoch().count());
  }
  return h;
}

std::size_t numel_(const std::vector<std::int64_t>& s) {
  std::size_t n = 1;
  for (auto d : s) { n *= (std::size_t)d; }
  return n;
}
// Convert an F32 checkpoint tensor to an f16 SharedBuffer (same row-major
// layout). Empty on a missing tensor.
SharedBuffer to_f16_(const MetalLlamaWeights& wts, metal_compute::MetalCompute* mc,
                     const std::string& name) {
  const auto* info = wts.info(name);
  if (info == nullptr) { return {}; }
  SharedBuffer src = wts.load(name, mc);
  if (src.empty()) { return {}; }
  const std::size_t n = numel_(info->shape);
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  const auto* s = static_cast<const float*>(src.contents());
  auto* d = static_cast<_Float16*>(out.contents());
  for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)s[i]; }
  return out;
}
// to_f16 with each output ROW scaled by scale[row] (folds a per-channel
// LayerScale into the projection weight: weight is [out, in], scale is [out]).
SharedBuffer to_f16_rowscale_(const MetalLlamaWeights& wts,
                              metal_compute::MetalCompute* mc,
                              const std::string& wname,
                              const std::string& sname) {
  const auto* wi = wts.info(wname);
  const auto* si = wts.info(sname);
  if (wi == nullptr || si == nullptr || wi->shape.size() != 2) { return {}; }
  const int out = (int)wi->shape[0], in = (int)wi->shape[1];
  SharedBuffer ws = wts.load(wname, mc), ss = wts.load(sname, mc);
  if (ws.empty() || ss.empty()) { return {}; }
  SharedBuffer o = mc->make_shared_buffer((std::size_t)out * in * 2);
  const auto* w = static_cast<const float*>(ws.contents());
  const auto* sc = static_cast<const float*>(ss.contents());
  auto* d = static_cast<_Float16*>(o.contents());
  for (int r = 0; r < out; ++r) {
    for (int c = 0; c < in; ++c) {
      d[(std::size_t)r * in + c] = (_Float16)(sc[r] * w[(std::size_t)r * in + c]);
    }
  }
  return o;
}
// Fold a weight-normalized 1x1 conv (kernel_size 1) into a plain [out,in]
// f16 matrix: w_eff[o,i] = g[o] * v[o,i] / ||v[o,:]||_2  (parametrizations
// original0=g [out,1,1], original1=v [out,in,1]).
SharedBuffer fold_wnconv_(const MetalLlamaWeights& wts,
                          metal_compute::MetalCompute* mc,
                          const std::string& prefix) {
  const std::string gn = prefix + ".parametrizations.weight.original0";
  const std::string vn = prefix + ".parametrizations.weight.original1";
  const auto* vi = wts.info(vn);
  const auto* gi = wts.info(gn);
  if (vi == nullptr || gi == nullptr || vi->shape.size() < 2) { return {}; }
  const int out = (int)vi->shape[0], in = (int)vi->shape[1];
  SharedBuffer vs = wts.load(vn, mc), gs = wts.load(gn, mc);
  if (vs.empty() || gs.empty()) { return {}; }
  const auto* v = static_cast<const float*>(vs.contents());
  const auto* g = static_cast<const float*>(gs.contents());
  SharedBuffer o = mc->make_shared_buffer((std::size_t)out * in * 2);
  auto* d = static_cast<_Float16*>(o.contents());
  for (int r = 0; r < out; ++r) {
    double ss = 0.0;
    for (int c = 0; c < in; ++c) {
      const double x = v[(std::size_t)r * in + c];
      ss += x * x;
    }
    const float inv = (float)(1.0 / std::sqrt(ss));
    for (int c = 0; c < in; ++c) {
      d[(std::size_t)r * in + c] =
          (_Float16)(g[r] * v[(std::size_t)r * in + c] * inv);
    }
  }
  return o;
}
}  // namespace

std::unique_ptr<MetalMossCodec>
MetalMossCodec::load(const std::string& model_dir,
                     metal_compute::MetalCompute* mc, bool int8,
                     bool with_encoder) {
  if (mc == nullptr || !mc->valid()) { return nullptr; }
  auto self = std::unique_ptr<MetalMossCodec>(new MetalMossCodec());
  self->_mc = mc;
  self->_int8 = int8;
  self->_with_encoder = with_encoder;
  // int8 g32: quant kernel (build-time) + the fused affine GEMM (decode). The
  // steel GEMM dequantizes the weight inside its MMA loader, so it reads the
  // int8 weight directly (half the f16 bandwidth) -- no separate dequant pass
  // or f16 temp. _lib_elt is reused for the forward-pass kernels below.
  self->_lib_elt = mc->load_library("llm_elementwise");
  self->_lib_qmm = mc->load_library("affine_qmm_steel");
  self->_fn_quant = self->_lib_elt.function("quant_f16_to_u8g32");
  self->_fn_qmm8g32 = self->_lib_qmm.function("affine_qmm_steel_w8g32");
  if (int8 && (!self->_fn_quant.valid() || !self->_fn_qmm8g32.valid())) {
    return nullptr;
  }

  {
    std::ifstream in(model_dir + "/config.json");
    if (in) {
      try {
        FlexData root = FlexData::from_json(in);
        if (root.is_object()) {
          const auto ro = root.as_object();
          if (ro.contains("sample_rate")) {
            self->_sample_rate = (int)ro.at("sample_rate").as_int(24000);
          }
        }
      } catch (...) {}
    }
  }

  // The 8B MOSS-Audio-Tokenizer decoder structure (stable; see config.json
  // decoder_kwargs). 4 ProjectedTransformers, patch upsamplers x2,x2,x2,x240.
  // context = round(frame_rate * 10s); frame_rate = 12.5,25,50,100 Hz.
  const StageCfg cfgs[4] = {
      {768, 1280, 32, 20, 5120, 1280,   2,  125},  // decoder.0
      {640,  768, 12, 12, 3072,  768,   2,  250},  // decoder.2
      {384,  768, 12, 12, 3072,  768,   2,  500},  // decoder.4
      {384,  768, 12, 12, 3072,  240, 240, 1000},  // decoder.6
  };
  const int module_idx[4] = {0, 2, 4, 6};

  // Set up shapes first so the cache reader can populate the same slots the
  // builder would (the weight buffers are then filled from the cache, else by
  // converting the safetensors).
  self->_stages.resize(4);
  for (int s = 0; s < 4; ++s) {
    Stage& st = self->_stages[(std::size_t)s];
    st.cfg = cfgs[s];
    st.layers.resize((std::size_t)cfgs[s].n_layers);
    // GEMM weight [N,K] shapes (K = contraction; must be % 32 for g32).
    const StageCfg& c = st.cfg;
    st.in_proj.N  = c.d_model; st.in_proj.K  = c.in_dim;
    st.out_proj.N = c.out_dim; st.out_proj.K = c.d_model;
    for (auto& L : st.layers) {
      L.qkvw.N = 3 * c.d_model; L.qkvw.K = c.d_model;
      L.ow.N   = c.d_model;     L.ow.K   = c.d_model;
      L.fc1.N  = c.ff;          L.fc1.K  = c.d_model;
      L.fc2.N  = c.d_model;     L.fc2.K  = c.ff;
    }
  }
  self->_codebook.resize((std::size_t)self->_n_vq);
  self->_q_outw.resize((std::size_t)self->_n_vq);
  self->_q_outb.resize((std::size_t)self->_n_vq);

  // Encoder stages (mirror of the decoder; see config.json encoder_kwargs).
  // StageCfg::patch is the DOWN patch applied BEFORE the stage's transformer
  // (240,2,2,2 -> a 1920x total downsample). enc.3/5/7 have in_dim==d_model
  // (Identity input_proj). context = round(frame_rate*10s); the frame rate
  // RISES from 12.5Hz to 100Hz going from waveform inward, so the first
  // encoder stage runs at 100Hz (context 1000) down to 12.5Hz (125).
  const StageCfg enc_cfgs[4] = {
      { 240,  768, 12, 12, 3072, 384, 240, 1000},  // encoder.1
      { 768,  768, 12, 12, 3072, 384,   2,  500},  // encoder.3
      { 768,  768, 12, 12, 3072, 640,   2,  250},  // encoder.5
      {1280, 1280, 32, 20, 5120, 768,   2,  125},  // encoder.7
  };
  const int enc_module_idx[4] = {1, 3, 5, 7};
  if (with_encoder) {
    self->_enc_stages.resize(4);
    self->_q_inw.resize((std::size_t)self->_n_vq);
    self->_q_inb.resize((std::size_t)self->_n_vq);
    self->_codebook_norm.resize((std::size_t)self->_n_vq);
    for (int s = 0; s < 4; ++s) {
      Stage& st = self->_enc_stages[(std::size_t)s];
      st.cfg = enc_cfgs[s];
      st.layers.resize((std::size_t)enc_cfgs[s].n_layers);
      const StageCfg& c = st.cfg;
      // input_proj exists only when in_dim != d_model (else Identity, N=K=0).
      st.in_proj.N = (c.in_dim != c.d_model) ? c.d_model : 0;
      st.in_proj.K = (c.in_dim != c.d_model) ? c.in_dim  : 0;
      st.out_proj.N = c.out_dim; st.out_proj.K = c.d_model;
      for (auto& L : st.layers) {
        L.qkvw.N = 3 * c.d_model; L.qkvw.K = c.d_model;
        L.ow.N   = c.d_model;     L.ow.K   = c.d_model;
        L.fc1.N  = c.ff;          L.fc1.K  = c.d_model;
        L.fc2.N  = c.d_model;     L.fc2.K  = c.ff;
      }
    }
  }

  // Visit every persistent weight buffer in a FIXED order -- used to read or
  // write the converted-weight cache (must match the builder's order below).
  // A QuantWeight contributes 3 buffers (w + scale + bias; scale/bias empty
  // => 0 bytes in the cache for the f16 variant).
  auto each_weight = [&](auto&& f) {
    auto qw = [&](QuantWeight& q) { f(q.w); f(q.scale); f(q.bias); };
    for (auto& b : self->_codebook) { f(b); }
    for (auto& b : self->_q_outw)   { f(b); }
    for (auto& b : self->_q_outb)   { f(b); }
    f(self->_rvq_outw); f(self->_rvq_outb);
    for (auto& st : self->_stages) {
      qw(st.in_proj);
      if (st.cfg.out_dim != st.cfg.d_model) { qw(st.out_proj); }
      for (auto& L : st.layers) {
        f(L.n1w); f(L.n1b); f(L.n2w); f(L.n2b);
        qw(L.qkvw); qw(L.ow); qw(L.fc1); qw(L.fc2);
      }
    }
    if (self->_with_encoder) {     // encode path (only in the "-enc" cache)
      f(self->_rvq_inw); f(self->_rvq_inb);
      for (auto& b : self->_q_inw) { f(b); }
      for (auto& b : self->_q_inb) { f(b); }
      for (auto& st : self->_enc_stages) {
        qw(st.in_proj);    // Identity (in_dim==d_model) => empty entries
        qw(st.out_proj);   // always present for encoder stages
        for (auto& L : st.layers) {
          f(L.n1w); f(L.n1b); f(L.n2w); f(L.n2b);
          qw(L.qkvw); qw(L.ow); qw(L.fc1); qw(L.fc2);
        }
      }
    }
  };
  std::uint32_t n_weights = 0;
  each_weight([&](SharedBuffer&) { ++n_weights; });

  // ---- converted-weight cache --------------------------------------------
  // First load converts the ~6.6 GB F32 checkpoint to the ~3.3 GB f16 weights
  // and writes them to a sidecar; later loads memcpy the cache back (half the
  // disk read, no conversion). Keyed by the source fingerprint; atomic via
  // .tmp+rename; best-effort (read-only model dir => just no caching).
  // Disable with VPIPE_MOSS_NO_CACHE.
  static const std::uint32_t kCacheMagic = 0x4d50564du;   // "MVPM"
  static const std::uint32_t kCacheVer   = 2u;   // bump on any weight-layout
                                                 // change (e.g. int8 fused-GEMM)
  const bool cache_off = std::getenv("VPIPE_MOSS_NO_CACHE") != nullptr;
  const std::uint64_t fp = codec_source_fingerprint_(model_dir);
  // Variant in the filename so f16 and int8 caches coexist; kCacheVer guards
  // format changes, the fingerprint guards source-checkpoint changes.
  const std::string cache_path = model_dir + "/.vpipe-codec-cache-" +
                                 (int8 ? "i8g32" : "f16") +
                                 (with_encoder ? "-enc" : "") + ".bin";

  bool from_cache = false;
  if (!cache_off) {
    std::ifstream cf(cache_path, std::ios::binary);
    if (cf) {
      std::uint32_t magic = 0, ver = 0, nt = 0;
      std::uint64_t cfp = 0;
      cf.read(reinterpret_cast<char*>(&magic), 4);
      cf.read(reinterpret_cast<char*>(&ver), 4);
      cf.read(reinterpret_cast<char*>(&cfp), 8);
      cf.read(reinterpret_cast<char*>(&nt), 4);
      if (cf && magic == kCacheMagic && ver == kCacheVer && cfp == fp &&
          nt == n_weights) {
        bool good = true;
        each_weight([&](SharedBuffer& b) {
          if (!good) { return; }
          std::uint64_t nb = 0;
          if (!cf.read(reinterpret_cast<char*>(&nb), 8)) { good = false; return; }
          b = mc->make_shared_buffer((std::size_t)nb);
          if (b.empty() ||
              !cf.read(static_cast<char*>(b.contents()), (std::streamsize)nb)) {
            good = false;
          }
        });
        from_cache = good && static_cast<bool>(cf);
      }
    }
  }

  if (!from_cache) {
    auto wts_opt = MetalLlamaWeights::open_model(model_dir);
    if (!wts_opt) { return nullptr; }
    const MetalLlamaWeights& wts = *wts_opt;

    // Store an f16 GEMM weight into a QuantWeight slot (N/K preset): f16 mode
    // keeps it as-is; int8 mode quantizes to uint8 g32 on the GPU and frees
    // the f16 source.
    auto store_qw = [&](QuantWeight& q, SharedBuffer f16) {
      // int8 needs K % 32 (group) and N % 32 (the steel GEMM's aligned_N);
      // weights that don't qualify (e.g. out_proj N=240) stay f16.
      if (!self->_int8 || f16.empty() || q.K <= 0 || (q.K % 32) != 0 ||
          q.N <= 0 || (q.N % 32) != 0) {
        q.w = std::move(f16);
        return;
      }
      const int N = q.N, K = q.K, G = K / 32;
      q.w     = mc->make_shared_buffer((std::size_t)N * K);          // uint8
      q.scale = mc->make_shared_buffer((std::size_t)N * G * 2);       // f16
      q.bias  = mc->make_shared_buffer((std::size_t)N * G * 2);       // f16
      if (q.w.empty() || q.scale.empty() || q.bias.empty()) { return; }
      metal_compute::CommandStream stream = mc->make_command_stream();
      {
        ComputeEncoder enc = stream.begin_compute();
        enc.set_function(self->_fn_quant);
        enc.set_buffer(0, f16); enc.set_buffer(1, q.w);
        enc.set_buffer(2, q.scale); enc.set_buffer(3, q.bias);
        enc.set_constant(4, N); enc.set_constant(5, K);
        enc.dispatch({(unsigned)(N * G), 1, 1}, {256, 1, 1});
      }
      stream.commit().wait();   // f16 freed on return
    };

    bool ok = true;
    // ---- RVQ (quantizer) decode weights ----------------------------------
    for (int i = 0; i < self->_n_vq; ++i) {
      const std::string q = "quantizer.quantizers." + std::to_string(i);
      self->_codebook[(std::size_t)i] = to_f16_(wts, mc, q + ".codebook.weight");
      self->_q_outw[(std::size_t)i] = fold_wnconv_(wts, mc, q + ".out_proj");
      self->_q_outb[(std::size_t)i] = to_f16_(wts, mc, q + ".out_proj.bias");
      ok = ok && !self->_codebook[(std::size_t)i].empty() &&
           !self->_q_outw[(std::size_t)i].empty() &&
           !self->_q_outb[(std::size_t)i].empty();
    }
    self->_rvq_outw = fold_wnconv_(wts, mc, "quantizer.output_proj");
    self->_rvq_outb = to_f16_(wts, mc, "quantizer.output_proj.bias");
    ok = ok && !self->_rvq_outw.empty() && !self->_rvq_outb.empty();

    // ---- decoder transformer stages --------------------------------------
    for (int s = 0; s < 4; ++s) {
      Stage& st = self->_stages[(std::size_t)s];
      const std::string base = "decoder." + std::to_string(module_idx[s]) + ".";
      store_qw(st.in_proj, to_f16_(wts, mc, base + "input_proj.weight"));
      ok = ok && !st.in_proj.empty();
      if (st.cfg.out_dim != st.cfg.d_model) {
        store_qw(st.out_proj, to_f16_(wts, mc, base + "output_proj.weight"));
        ok = ok && !st.out_proj.empty();
      }
      for (int l = 0; l < st.cfg.n_layers; ++l) {
        Layer& L = st.layers[(std::size_t)l];
        const std::string p =
            base + "transformer.layers." + std::to_string(l) + ".";
        L.n1w = to_f16_(wts, mc, p + "norm1.weight");
        L.n1b = to_f16_(wts, mc, p + "norm1.bias");
        L.n2w = to_f16_(wts, mc, p + "norm2.weight");
        L.n2b = to_f16_(wts, mc, p + "norm2.bias");
        store_qw(L.qkvw, to_f16_(wts, mc, p + "self_attn.in_projs.0.weight"));
        // Fold the two per-channel LayerScales into out_proj / linear2.
        store_qw(L.ow, to_f16_rowscale_(wts, mc,
                          p + "self_attn.out_projs.0.weight",
                          p + "layer_scale_1.scale"));
        store_qw(L.fc1, to_f16_(wts, mc, p + "linear1.weight"));
        store_qw(L.fc2, to_f16_rowscale_(wts, mc, p + "linear2.weight",
                          p + "layer_scale_2.scale"));
        ok = ok && !L.n1w.empty() && !L.n1b.empty() && !L.n2w.empty() &&
             !L.n2b.empty() && !L.qkvw.empty() && !L.ow.empty() &&
             !L.fc1.empty() && !L.fc2.empty();
      }
    }

    // ---- encode path (optional) ------------------------------------------
    if (self->_with_encoder) {
      // Quantizer encode weights: input_proj (768->512) folded wnconv + bias,
      // per-codebook in_proj (512->8) folded wnconv + bias.
      self->_rvq_inw = fold_wnconv_(wts, mc, "quantizer.input_proj");
      self->_rvq_inb = to_f16_(wts, mc, "quantizer.input_proj.bias");
      ok = ok && !self->_rvq_inw.empty() && !self->_rvq_inb.empty();
      for (int i = 0; i < self->_n_vq; ++i) {
        const std::string q = "quantizer.quantizers." + std::to_string(i);
        self->_q_inw[(std::size_t)i] = fold_wnconv_(wts, mc, q + ".in_proj");
        self->_q_inb[(std::size_t)i] = to_f16_(wts, mc, q + ".in_proj.bias");
        ok = ok && !self->_q_inw[(std::size_t)i].empty() &&
             !self->_q_inb[(std::size_t)i].empty();
      }
      // Encoder transformer stages (mirror the decoder load; enc.3/5/7 have
      // an Identity input_proj -> no input_proj.weight tensor).
      for (int s = 0; s < 4; ++s) {
        Stage& st = self->_enc_stages[(std::size_t)s];
        const std::string base =
            "encoder." + std::to_string(enc_module_idx[s]) + ".";
        if (st.cfg.in_dim != st.cfg.d_model) {
          store_qw(st.in_proj, to_f16_(wts, mc, base + "input_proj.weight"));
          ok = ok && !st.in_proj.empty();
        }
        store_qw(st.out_proj, to_f16_(wts, mc, base + "output_proj.weight"));
        ok = ok && !st.out_proj.empty();
        for (int l = 0; l < st.cfg.n_layers; ++l) {
          Layer& L = st.layers[(std::size_t)l];
          const std::string p =
              base + "transformer.layers." + std::to_string(l) + ".";
          L.n1w = to_f16_(wts, mc, p + "norm1.weight");
          L.n1b = to_f16_(wts, mc, p + "norm1.bias");
          L.n2w = to_f16_(wts, mc, p + "norm2.weight");
          L.n2b = to_f16_(wts, mc, p + "norm2.bias");
          store_qw(L.qkvw, to_f16_(wts, mc, p + "self_attn.in_projs.0.weight"));
          store_qw(L.ow, to_f16_rowscale_(wts, mc,
                            p + "self_attn.out_projs.0.weight",
                            p + "layer_scale_1.scale"));
          store_qw(L.fc1, to_f16_(wts, mc, p + "linear1.weight"));
          store_qw(L.fc2, to_f16_rowscale_(wts, mc, p + "linear2.weight",
                            p + "layer_scale_2.scale"));
          ok = ok && !L.n1w.empty() && !L.n1b.empty() && !L.n2w.empty() &&
               !L.n2b.empty() && !L.qkvw.empty() && !L.ow.empty() &&
               !L.fc1.empty() && !L.fc2.empty();
        }
      }
    }
    if (!ok) { return nullptr; }

    if (!cache_off) {
      const std::string tmp = cache_path + ".tmp";
      std::ofstream cf(tmp, std::ios::binary | std::ios::trunc);
      if (cf) {
        cf.write(reinterpret_cast<const char*>(&kCacheMagic), 4);
        cf.write(reinterpret_cast<const char*>(&kCacheVer), 4);
        cf.write(reinterpret_cast<const char*>(&fp), 8);
        cf.write(reinterpret_cast<const char*>(&n_weights), 4);
        bool good = true;
        each_weight([&](SharedBuffer& b) {
          if (!good) { return; }
          const std::uint64_t nb = b.byte_size();
          cf.write(reinterpret_cast<const char*>(&nb), 8);
          cf.write(static_cast<const char*>(b.contents()), (std::streamsize)nb);
          if (!cf) { good = false; }
        });
        cf.close();
        if (good && cf) { std::rename(tmp.c_str(), cache_path.c_str()); }
        else { std::remove(tmp.c_str()); }
      }
    }
  }

  // Derived (not cached): the L2-normalized codebook for the encode-side
  // cosine-nearest search. F.normalize semantics: v / max(||v||_2, eps).
  if (self->_with_encoder) {
    const int CD = self->_codebook_dim, Csz = self->_codebook_size;
    for (int i = 0; i < self->_n_vq; ++i) {
      SharedBuffer nb = mc->make_shared_buffer((std::size_t)Csz * CD * 2);
      const auto* s =
          static_cast<const _Float16*>(self->_codebook[(std::size_t)i].contents());
      auto* d = static_cast<_Float16*>(nb.contents());
      for (int r = 0; r < Csz; ++r) {
        double ss = 0.0;
        for (int e = 0; e < CD; ++e) {
          const double x = (double)s[(std::size_t)r * CD + e];
          ss += x * x;
        }
        const double inv = 1.0 / std::max(std::sqrt(ss), 1e-12);
        for (int e = 0; e < CD; ++e) {
          d[(std::size_t)r * CD + e] =
              (_Float16)((double)s[(std::size_t)r * CD + e] * inv);
        }
      }
      self->_codebook_norm[(std::size_t)i] = std::move(nb);
    }
  }

  // inv_freq for the interleaved RoPE (head_dim 64, max_period 10000).
  const int hd = 64, half = hd / 2;
  self->_inv_freq = mc->make_shared_buffer((std::size_t)half * sizeof(float));
  auto* invf = static_cast<float*>(self->_inv_freq.contents());
  for (int i = 0; i < half; ++i) {
    invf[i] = 1.0f / std::pow(10000.0f, (2.0f * (float)i) / (float)hd);
  }

  // ---- kernels (f16) ------------------------------------------------
  self->_lib_gemm = mc->load_library("dense_gemm");
  self->_lib_vis = mc->load_library("qwen3_5_vision");
  // _lib_elt already loaded up front (int8 quant/dequant kernels).
  self->_lib_sdpa = mc->load_library("sdpa");
  self->_lib_rope = mc->load_library("rope");
  self->_fn_gemm = self->_lib_gemm.function("dense_gemm_t_f16");
  self->_fn_ln = self->_lib_vis.function("layer_norm_bias_f16");
  self->_fn_gelu = self->_lib_vis.function("gelu_erf_f16");
  self->_fn_hslice = self->_lib_elt.function("head_slice_f16");
  self->_fn_transpose = self->_lib_elt.function("transpose_abd_f16");
  self->_fn_residual = self->_lib_elt.function("residual_add_f16");
  self->_fn_sdpa = self->_lib_sdpa.function("sdpa_causal_window_f16");
  self->_fn_rope = self->_lib_rope.function("rope_interleaved_f16");
  self->_fn_ring_append = self->_lib_elt.function("ring_append_f16");
  if (!self->_fn_gemm.valid() || !self->_fn_ln.valid() ||
      !self->_fn_gelu.valid() || !self->_fn_hslice.valid() ||
      !self->_fn_transpose.valid() || !self->_fn_residual.valid() ||
      !self->_fn_sdpa.valid() || !self->_fn_rope.valid()) {
    return nullptr;
  }

  // M5 matrix-core decode acceleration (mirrors MetalMossCodecV2): the f16
  // dense GEMM -> matmul2d, and the scalar windowed-causal attention -> the
  // head_dim-64 flash kernel. Gated on GPU matrix-core support so M4/older
  // keep the byte-identical steel+scalar path. VPIPE_MOSS_CODEC_NO_MMA2 /
  // NO_ATTN_MMA force the steel/scalar path (A/B + safety). The int8 GEMM
  // (fused affine steel) is left unchanged.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_MOSS_CODEC_NO_MMA2") == nullptr) {
    self->_lib_dense_mma = mc->load_library("dense_gemm_mma");
    self->_fn_gemm_mma =
        self->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    self->_fn_gemm_mma_deep =
        self->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    self->_mma_available =
        self->_fn_gemm_mma.valid() && self->_fn_gemm_mma_deep.valid();
    self->_use_mma2 = self->_mma_available;
    // Int8 (w8 g32) matrix-core GEMM: dequant-once (-> f16 dense_gemm_mma).
    // Only meaningful in int8 mode; the kernel load is harmless otherwise.
    self->_lib_dequant = mc->load_library("affine_dequant");
    self->_fn_dequant_w8g32 =
        self->_lib_dequant.function("affine_dequant_w8g32");
  }
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_MOSS_CODEC_NO_ATTN_MMA") == nullptr) {
    self->_lib_sdpa_mma = mc->load_library("sdpa_mma");
    self->_fn_sdpa_mma =
        self->_lib_sdpa_mma.function("sdpa_causal_mma2_d64_f16");
    self->_attn_mma_available = self->_fn_sdpa_mma.valid();
    self->_use_attn_mma = self->_attn_mma_available;
  }
  self->_ok = true;
  return self;
}

SharedBuffer
MetalMossCodec::rvq_decode_(const std::vector<std::vector<std::int32_t>>& codes,
                            int T, int n_active) {
  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  const int CD = _codebook_dim, RV = _rvq_dim, OD = _code_dim;
  // Residual truncation: sum only the first `na` codebooks (0 => all).
  const int na = (n_active > 0 && n_active < _n_vq) ? n_active : _n_vq;

  // Host-gather each codebook's rows for the T frames.
  std::vector<SharedBuffer> gathered((std::size_t)na);
  for (int cb = 0; cb < na; ++cb) {
    gathered[(std::size_t)cb] = buf((std::size_t)T * CD);
    auto* g = static_cast<_Float16*>(gathered[(std::size_t)cb].contents());
    const auto* tbl =
        static_cast<const _Float16*>(_codebook[(std::size_t)cb].contents());
    for (int t = 0; t < T; ++t) {
      int code = codes[(std::size_t)t][(std::size_t)cb];
      if (code < 0) { code = 0; }
      if (code >= _codebook_size) { code = _codebook_size - 1; }
      for (int e = 0; e < CD; ++e) {
        g[(std::size_t)t * CD + e] = tbl[(std::size_t)code * CD + e];
      }
    }
  }
  SharedBuffer emb = buf((std::size_t)T * RV);
  std::memset(emb.contents(), 0, (std::size_t)T * RV * 2);
  SharedBuffer tmp = buf((std::size_t)T * RV);
  SharedBuffer hidden = buf((std::size_t)T * OD);

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // scalar bias-GEMM (handles tiny K=8 + always applies the conv bias).
    auto lib_gemm_bias = _lib_gemm.function("dense_gemm_bias_f16");
    auto gemm_bias = [&](const SharedBuffer& xin, const SharedBuffer& w,
                         const SharedBuffer& b, const SharedBuffer& y, int M,
                         int N, int K) {
      enc.set_function(lib_gemm_bias);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, b);
      enc.set_buffer(3, y);
      enc.set_constant(4, M);
      enc.set_constant(5, N);
      enc.set_constant(6, K);
      enc.set_constant(7, 1);
      enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                    (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& b,
                        const SharedBuffer& out, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a);
      enc.set_buffer(1, b);
      enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    for (int cb = 0; cb < na; ++cb) {
      gemm_bias(gathered[(std::size_t)cb], _q_outw[(std::size_t)cb],
                _q_outb[(std::size_t)cb], tmp, T, RV, CD);
      residual(emb, tmp, emb, T * RV);
    }
    gemm_bias(emb, _rvq_outw, _rvq_outb, hidden, T, OD, RV);
  }
  stream.commit().wait();
  return hidden;
}

SharedBuffer
MetalMossCodec::run_stage_(const Stage& st, int T, const SharedBuffer& in,
                           std::vector<SharedBuffer>* kc,
                           std::vector<SharedBuffer>* vc, int pos,
                           int ring_cap) {
  const bool streaming = (kc != nullptr && vc != nullptr);
  const StageCfg& c = st.cfg;
  const int d = c.d_model, heads = c.n_heads, hd = d / heads, ff = c.ff;
  const int in_dim = c.in_dim, out_dim = c.out_dim;
  const float scale = 1.0f / std::sqrt((float)hd);
  const float eps = 1e-5f;
  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };

  SharedBuffer x = buf((std::size_t)T * d);
  // Identity input_proj (in_dim == d_model, e.g. encoder.3/5/7): x := in.
  // Host copy before the stream opens (in is host-ready; UMA coherent).
  const bool id_in = st.in_proj.empty();
  if (id_in) {
    std::memcpy(x.contents(), in.contents(), (std::size_t)T * d * 2);
  }
  SharedBuffer n1 = buf((std::size_t)T * d), qkv = buf((std::size_t)T * 3 * d);
  SharedBuffer q3 = buf((std::size_t)T * d), k3 = buf((std::size_t)T * d),
               v3 = buf((std::size_t)T * d);
  SharedBuffer qt = buf((std::size_t)T * d), kt = buf((std::size_t)T * d),
               vt = buf((std::size_t)T * d);
  SharedBuffer atb = buf((std::size_t)T * d), att = buf((std::size_t)T * d);
  SharedBuffer o = buf((std::size_t)T * d), o2 = buf((std::size_t)T * d);
  SharedBuffer h = buf((std::size_t)T * ff);
  SharedBuffer out =
      (out_dim != d) ? buf((std::size_t)T * out_dim) : SharedBuffer{};

  // Int8 + matrix-core: one reusable f16 scratch to dequant each weight into
  // before the f16 dense_gemm_mma (dequant-once). matmul2d streams f16 from
  // DRAM (it can't consume int8), so materializing once here + the fast direct
  // matmul2d beats every fused int8 path at the codec's M. Sized to the stage's
  // largest weight [N,K]; allocated only when this stage has int8 weights.
  // (Double-buffering to overlap the dequant with the prior matmul was measured
  // a no-op -- the ~15% over f16 is the materialization bandwidth, not a stall.)
  const bool stage_int8 =
      !st.layers.empty() && st.layers[0].qkvw.is_int8();
  SharedBuffer deqw;
  if (_use_mma2 && _fn_dequant_w8g32.valid() && stage_int8) {
    const int mx = std::max(std::max(3 * d, ff), std::max(in_dim, out_dim));
    deqw = buf((std::size_t)d * (std::size_t)mx);
  }

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto gemm = [&](const SharedBuffer& xin, const SharedBuffer& w,
                    const SharedBuffer& y, int M, int N, int K) {
      if (_use_mma2) {
        // M5 matrix-core matmul2d GEMM y = x @ w^T (no bias). n128 tile fits
        // every codec GEMM (max K = ff = 5120 < the 6144 deep threshold).
        const bool deep = (K >= 6144);
        const int BN = deep ? 256 : 128;
        enc.set_function(deep ? _fn_gemm_mma_deep : _fn_gemm_mma);
        enc.set_buffer(0, xin);
        enc.set_buffer(1, w);
        enc.set_buffer(2, w);
        enc.set_buffer(3, y);
        enc.set_constant(4, K);
        enc.set_constant(5, N);
        enc.set_constant(6, M);
        enc.set_constant(7, 0);
        enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                      (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
        return;
      }
      enc.set_function(_fn_gemm);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, w);    // bias slot unused (has_bias=0)
      enc.set_buffer(3, y);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.set_constant(6, M);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
    };
    // GEMM with a (possibly int8) weight: f16 weights use the dense GEMM; int8
    // ones use the fused affine steel GEMM, which dequantizes inside its MMA
    // loader (reads the int8 weight directly -- no dequant pass, no f16 temp,
    // half the weight bandwidth). y[M,N] = x[M,K] @ dequant(w)[N,K]^T.
    auto gemm_q = [&](const SharedBuffer& xin, const QuantWeight& qw,
                      const SharedBuffer& y, int M, int N, int K) {
      if (!qw.is_int8()) { gemm(xin, qw.w, y, M, N, K); return; }
      if (_use_mma2 && !deqw.empty()) {
        // M5 matrix-core int8: dequant the w8g32 weight to f16 once (grid
        // {K/4, N}), then the f16 matmul2d GEMM reads it (GPU-ordered in this
        // command buffer). Beats both the steel w8 GEMM and a fused dequant-
        // in-matmul2d at the codec's M (the fused kernel re-dequants per tile).
        enc.set_function(_fn_dequant_w8g32);
        enc.set_buffer(0, qw.w);
        enc.set_buffer(1, qw.scale);
        enc.set_buffer(2, qw.bias);
        enc.set_buffer(3, deqw);
        enc.set_constant(4, K);
        enc.set_constant(5, N);
        enc.dispatch({(unsigned)(K / 4), (unsigned)N, 1}, {64, 1, 1});
        gemm(xin, deqw, y, M, N, K);
        return;
      }
      enc.set_function(_fn_qmm8g32);
      enc.set_buffer(0, qw.w);
      enc.set_buffer(1, qw.scale);
      enc.set_buffer(2, qw.bias);
      enc.set_buffer(3, xin);
      enc.set_buffer(4, y);
      enc.set_constant(5, K);
      enc.set_constant(6, N);
      enc.set_constant(7, M);
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
    };
    auto ln = [&](const SharedBuffer& xin, const SharedBuffer& w,
                  const SharedBuffer& b, const SharedBuffer& y, int R, int Hd) {
      enc.set_function(_fn_ln);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, b);
      enc.set_buffer(3, y);
      enc.set_constant(4, Hd);
      enc.set_constant(5, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto gelu = [&](const SharedBuffer& xin, const SharedBuffer& y, int nn) {
      enc.set_function(_fn_gelu);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, y);
      enc.set_constant(2, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto hslice = [&](const SharedBuffer& inb, const SharedBuffer& outb, int Hh,
                      int Sd, int Wd, int off) {
      enc.set_function(_fn_hslice);
      enc.set_buffer(0, inb);
      enc.set_buffer(1, outb);
      enc.set_constant(2, Hh);
      enc.set_constant(3, Sd);
      enc.set_constant(4, Wd);
      enc.set_constant(5, off);
      const int zero = 0;
      enc.set_constant(6, zero);
      enc.set_constant(7, zero);
      enc.dispatch({(unsigned)(Hh * Wd), 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& inb, const SharedBuffer& outb,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, inb);
      enc.set_buffer(1, outb);
      enc.set_constant(2, A);
      enc.set_constant(3, Bd);
      enc.set_constant(4, hd);
      enc.dispatch({(unsigned)hd, (unsigned)Bd, (unsigned)A}, {(unsigned)hd, 1, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& b,
                        const SharedBuffer& outb, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a);
      enc.set_buffer(1, b);
      enc.set_buffer(2, outb);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto rope = [&](const SharedBuffer& xb) {
      enc.set_function(_fn_rope);
      enc.set_buffer(0, xb);
      enc.set_buffer(1, _inv_freq);
      enc.set_constant(2, heads);
      enc.set_constant(3, T);
      enc.set_constant(4, hd);
      const int off = pos;   // absolute position of the first (new) frame
      enc.set_constant(5, off);
      enc.dispatch({(unsigned)(hd / 2), (unsigned)T, (unsigned)heads}, {1, 1, 1});
    };
    // See codec-v2: one-shot decodes all T from local K/V (linear); streaming
    // decodes the T new frames (queries) at offset `pos` over the windowed ring.
    const int Tkv      = streaming ? (pos + T) : T;
    const int qoff     = streaming ? pos : 0;
    const int kvstride = streaming ? ring_cap : T;
    const int ringcap  = streaming ? ring_cap : T;
    auto attn = [&](const SharedBuffer& q, const SharedBuffer& k,
                    const SharedBuffer& v, const SharedBuffer& outb) {
      if (_use_attn_mma && hd == 64) {
        // M5 matrix-core windowed-causal flash attention (BQ=16 query tile,
        // 128 threads). Same [heads,*,hd] layout + causal/window/ring semantics.
        enc.set_function(_fn_sdpa_mma);
        enc.set_buffer(0, q);
        enc.set_buffer(1, k);
        enc.set_buffer(2, v);
        enc.set_buffer(3, outb);
        enc.set_constant(4, scale);
        enc.set_constant(5, Tkv);
        enc.set_constant(6, hd);
        enc.set_constant(7, heads);
        enc.set_constant(8, heads);
        enc.set_constant(9, T);
        enc.set_constant(10, qoff);
        enc.set_constant(11, kvstride);
        enc.set_constant(12, c.context);
        enc.set_constant(13, streaming ? ringcap : 0);
        enc.dispatch({128, (unsigned)heads, (unsigned)((T + 15) / 16)},
                     {128, 1, 1});
        return;
      }
      enc.set_function(_fn_sdpa);
      enc.set_buffer(0, q);
      enc.set_buffer(1, k);
      enc.set_buffer(2, v);
      enc.set_buffer(3, outb);
      enc.set_constant(4, scale);
      enc.set_constant(5, Tkv);        // T_kv
      enc.set_constant(6, hd);         // D
      enc.set_constant(7, heads);      // Hq
      enc.set_constant(8, heads);      // Hkv
      enc.set_constant(9, T);          // n_q
      enc.set_constant(10, qoff);      // q_offset
      enc.set_constant(11, kvstride);  // kv_stride
      enc.set_constant(12, c.context); // window (>= T => plain causal)
      enc.set_constant(13, ringcap);   // ring_cap
      enc.dispatch({32, (unsigned)heads, (unsigned)T}, {32, 1, 1});
    };
    // Scatter the T new frames' per-head K (or V) rows into a windowed ring.
    auto ring_append = [&](const SharedBuffer& src, const SharedBuffer& ring) {
      enc.set_function(_fn_ring_append);
      enc.set_buffer(0, src); enc.set_buffer(1, ring);
      enc.set_constant(2, heads); enc.set_constant(3, T); enc.set_constant(4, hd);
      enc.set_constant(5, ring_cap); enc.set_constant(6, pos);
      enc.dispatch({(unsigned)hd, (unsigned)T, (unsigned)heads}, {1, 1, 1});
    };

    if (!id_in) { gemm_q(in, st.in_proj, x, T, d, in_dim); }   // input_proj
    for (int l = 0; l < c.n_layers; ++l) {
      const Layer& L = st.layers[(std::size_t)l];
      ln(x, L.n1w, L.n1b, n1, T, d);
      gemm_q(n1, L.qkvw, qkv, T, 3 * d, d);
      hslice(qkv, q3, T, 3 * d, d, 0);
      hslice(qkv, k3, T, 3 * d, d, d);
      hslice(qkv, v3, T, 3 * d, d, 2 * d);
      transpose(q3, qt, T, heads);     // [T,heads,hd] -> [heads,T,hd]
      transpose(k3, kt, T, heads);
      transpose(v3, vt, T, heads);
      rope(qt);
      rope(kt);
      if (streaming) {
        ring_append(kt, (*kc)[(std::size_t)l]);
        ring_append(vt, (*vc)[(std::size_t)l]);
        attn(qt, (*kc)[(std::size_t)l], (*vc)[(std::size_t)l], atb);
      } else {
        attn(qt, kt, vt, atb);
      }
      transpose(atb, att, heads, T);   // [heads,T,hd] -> [T,heads,hd]
      gemm_q(att, L.ow, o, T, d, d);
      residual(x, o, x, T * d);
      ln(x, L.n2w, L.n2b, n1, T, d);
      gemm_q(n1, L.fc1, h, T, ff, d);
      gelu(h, h, T * ff);
      gemm_q(h, L.fc2, o2, T, d, ff);
      residual(x, o2, x, T * d);
    }
    if (out_dim != d) { gemm_q(x, st.out_proj, out, T, out_dim, d); }
  }
  stream.commit().wait();
  if (out_dim != d) { return out; }
  return x;
}

std::vector<float>
MetalMossCodec::decode(const std::vector<std::vector<std::int32_t>>& codes,
                       std::vector<std::vector<float>>* stages, int n_active) {
  std::vector<float> wave;
  const int T = (int)codes.size();
  if (T <= 0 || !_ok) { return wave; }

  // audio-codec perf block (LLM lane): RVQ decode + 4 transformer/upsample
  // stages -> waveform. value = input frame count.
  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmAudioCodec,
                     kPerfLlmAudioCodecBegin, (std::uint64_t)T);

  // Channel-major [C][T] copy of a time-major [T,C] f16 buffer (for golden
  // comparison: the reference dumps [C, T]).
  auto chan_major = [](const SharedBuffer& b, int t, int cc) {
    std::vector<float> out((std::size_t)cc * t);
    const auto* s = static_cast<const _Float16*>(b.contents());
    for (int i = 0; i < t; ++i) {
      for (int c = 0; c < cc; ++c) {
        out[(std::size_t)c * t + i] = (float)s[(std::size_t)i * cc + c];
      }
    }
    return out;
  };

  SharedBuffer cur = rvq_decode_(codes, T, n_active);   // [T, 768]
  if (stages != nullptr) { stages->push_back(chan_major(cur, T, _code_dim)); }

  int curT = T;
  for (int si = 0; si < 4; ++si) {
    const Stage& st = _stages[(std::size_t)si];
    SharedBuffer out = run_stage_(st, curT, cur);   // [curT, out_dim]
    if (stages != nullptr) {
      stages->push_back(chan_major(out, curT, st.cfg.out_dim));
    }
    // Patch upsample (host): out[t][c*P+p] -> up[t*P+p][c].
    const int P = st.cfg.patch, Cout = st.cfg.out_dim / P, Tout = curT * P;
    SharedBuffer up = _mc->make_shared_buffer((std::size_t)Tout * Cout * 2);
    const auto* s = static_cast<const _Float16*>(out.contents());
    auto* d = static_cast<_Float16*>(up.contents());
    for (int t = 0; t < curT; ++t) {
      for (int c = 0; c < Cout; ++c) {
        for (int p = 0; p < P; ++p) {
          d[(std::size_t)(t * P + p) * Cout + c] =
              s[(std::size_t)t * st.cfg.out_dim + c * P + p];
        }
      }
    }
    if (stages != nullptr) { stages->push_back(chan_major(up, Tout, Cout)); }
    cur = std::move(up);
    curT = Tout;
  }

  // cur is [curT, 1] = [samples, 1].
  wave.resize((std::size_t)curT);
  const auto* w = static_cast<const _Float16*>(cur.contents());
  for (int i = 0; i < curT; ++i) { wave[(std::size_t)i] = (float)w[i]; }
  return wave;
}

std::unique_ptr<MetalMossCodec::StreamState>
MetalMossCodec::decode_stream_begin(int max_chunk_frames, int n_active) const {
  if (!_ok || max_chunk_frames <= 0) { return nullptr; }
  auto st = std::make_unique<StreamState>();
  const std::size_t ns = _stages.size();
  st->kc.resize(ns);
  st->vc.resize(ns);
  st->pos.assign(ns, 0);
  st->cap.assign(ns, 0);
  st->max_chunk = max_chunk_frames;
  st->n_active  = n_active;
  int patchprod = 1;
  for (std::size_t si = 0; si < ns; ++si) {
    const StageCfg& c = _stages[si].cfg;
    const int hd = c.d_model / c.n_heads;
    const int cap = c.context + max_chunk_frames * patchprod;
    st->cap[si] = cap;
    const std::size_t ring =
        (std::size_t)c.n_heads * (std::size_t)cap * (std::size_t)hd;
    st->kc[si].reserve((std::size_t)c.n_layers);
    st->vc[si].reserve((std::size_t)c.n_layers);
    for (int l = 0; l < c.n_layers; ++l) {
      st->kc[si].push_back(_mc->make_shared_buffer(ring * 2));
      st->vc[si].push_back(_mc->make_shared_buffer(ring * 2));
    }
    patchprod *= c.patch;
  }
  return st;
}

std::vector<float>
MetalMossCodec::decode_stream_chunk(
    StreamState& st, const std::vector<std::vector<std::int32_t>>& codes) {
  std::vector<float> wave;
  const int Cnew = (int)codes.size();
  if (Cnew <= 0 || !_ok || st.pos.size() != _stages.size()
      || Cnew > st.max_chunk) {
    return wave;
  }
  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmAudioCodec,
                     kPerfLlmAudioCodecBegin, (std::uint64_t)Cnew);

  SharedBuffer cur = rvq_decode_(codes, Cnew, st.n_active);   // [Cnew, 768]
  int curT = Cnew;
  for (std::size_t si = 0; si < _stages.size(); ++si) {
    const Stage& stg = _stages[si];
    const int pos = st.pos[si];
    SharedBuffer out =
        run_stage_(stg, curT, cur, &st.kc[si], &st.vc[si], pos, st.cap[si]);
    st.pos[si] = pos + curT;
    const int P = stg.cfg.patch, Cout = stg.cfg.out_dim / P, Tout = curT * P;
    SharedBuffer up = _mc->make_shared_buffer((std::size_t)Tout * Cout * 2);
    const auto* s = static_cast<const _Float16*>(out.contents());
    auto* d = static_cast<_Float16*>(up.contents());
    for (int t = 0; t < curT; ++t) {
      for (int co = 0; co < Cout; ++co) {
        for (int p = 0; p < P; ++p) {
          d[(std::size_t)(t * P + p) * Cout + co] =
              s[(std::size_t)t * stg.cfg.out_dim + co * P + p];
        }
      }
    }
    cur = std::move(up);
    curT = Tout;
  }
  wave.resize((std::size_t)curT);
  const auto* w = static_cast<const _Float16*>(cur.contents());
  for (int i = 0; i < curT; ++i) { wave[(std::size_t)i] = (float)w[i]; }
  return wave;
}

std::vector<std::vector<std::int32_t>>
MetalMossCodec::encode(const std::vector<float>& wave_in) {
  std::vector<std::vector<std::int32_t>> codes;
  if (!_ok || !_with_encoder || wave_in.empty()) { return codes; }

  // audio-codec perf block (LLM lane): the encode mirror of decode().
  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmAudioCodec,
                     kPerfLlmAudioCodecBegin, (std::uint64_t)wave_in.size());

  const int ds = 1920;
  const int N = (int)wave_in.size();
  const int Npad = ((N + ds - 1) / ds) * ds;   // pad to a whole frame

  // Waveform as a 1-channel f16 buffer [Npad, 1] (zero-padded tail).
  SharedBuffer cur = _mc->make_shared_buffer((std::size_t)Npad * 2);
  {
    auto* w = static_cast<_Float16*>(cur.contents());
    for (int i = 0; i < Npad; ++i) {
      w[i] = (_Float16)(i < N ? wave_in[(std::size_t)i] : 0.0f);
    }
  }
  int curT = Npad, curC = 1;

  // DOWN patch-reshape: [Tin, Cin] -> [Tin/P, Cin*P],
  // out[t][c*P+p] = in[t*P+p][c] (the inverse of decode's UP patch).
  auto down_patch = [&](const SharedBuffer& inb, int Tin, int Cin, int P) {
    const int Tout = Tin / P, Cout = Cin * P;
    SharedBuffer outb = _mc->make_shared_buffer((std::size_t)Tout * Cout * 2);
    const auto* s = static_cast<const _Float16*>(inb.contents());
    auto* dptr = static_cast<_Float16*>(outb.contents());
    for (int t = 0; t < Tout; ++t) {
      for (int cc = 0; cc < Cin; ++cc) {
        for (int p = 0; p < P; ++p) {
          dptr[(std::size_t)t * Cout + cc * P + p] =
              s[(std::size_t)(t * P + p) * Cin + cc];
        }
      }
    }
    return outb;
  };

  for (int si = 0; si < 4; ++si) {
    const Stage& st = _enc_stages[(std::size_t)si];
    const int P = st.cfg.patch;
    SharedBuffer patched = down_patch(cur, curT, curC, P);
    const int Tst = curT / P;               // frames into this stage
    cur = run_stage_(st, Tst, patched);     // [Tst, out_dim]
    curT = Tst;
    curC = st.cfg.out_dim;
  }
  // cur = [curT, code_dim], curT = Npad/1920.
  return encode_rvq_(cur, curT);
}

std::vector<std::vector<std::int32_t>>
MetalMossCodec::encode_rvq_(const SharedBuffer& hidden, int T) {
  const int CD = _codebook_dim, RV = _rvq_dim, OD = _code_dim;
  const int Csz = _codebook_size, NV = _n_vq;
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)NV, 0));

  const auto* hid = static_cast<const _Float16*>(hidden.contents());   // [T,OD]
  const auto* Win = static_cast<const _Float16*>(_rvq_inw.contents()); // [RV,OD]
  const auto* bin = static_cast<const _Float16*>(_rvq_inb.contents()); // [RV]

  std::vector<float> resid((std::size_t)RV);   // running residual (rvq_dim)
  std::vector<float> z((std::size_t)CD);        // per-codebook in_proj out
  for (int t = 0; t < T; ++t) {
    // input_proj: resid = Win @ hidden[t] + bin.
    const _Float16* hr = hid + (std::size_t)t * OD;
    for (int o = 0; o < RV; ++o) {
      const _Float16* wr = Win + (std::size_t)o * OD;
      float acc = (float)bin[o];
      for (int k = 0; k < OD; ++k) { acc += (float)wr[k] * (float)hr[k]; }
      resid[(std::size_t)o] = acc;
    }
    // LFQ residual loop over the n_vq codebooks.
    for (int cb = 0; cb < NV; ++cb) {
      const auto* Wq =
          static_cast<const _Float16*>(_q_inw[(std::size_t)cb].contents());  // [CD,RV]
      const auto* bq =
          static_cast<const _Float16*>(_q_inb[(std::size_t)cb].contents());  // [CD]
      // z = in_proj_cb(resid).
      for (int e = 0; e < CD; ++e) {
        const _Float16* wr = Wq + (std::size_t)e * RV;
        float acc = (float)bq[e];
        for (int k = 0; k < RV; ++k) { acc += (float)wr[k] * resid[(std::size_t)k]; }
        z[(std::size_t)e] = acc;
      }
      // Cosine-nearest over the L2-normalized codebook. argmax(z . cb_norm)
      // == argmax(z_norm . cb_norm) (z's norm is a positive per-step
      // constant), reproducing the reference argmin squared distance.
      const auto* cbn =
          static_cast<const _Float16*>(_codebook_norm[(std::size_t)cb].contents());  // [Csz,CD]
      int best = 0;
      float bestdot = -std::numeric_limits<float>::infinity();
      for (int code = 0; code < Csz; ++code) {
        const _Float16* cr = cbn + (std::size_t)code * CD;
        float dot = 0.0f;
        for (int e = 0; e < CD; ++e) { dot += (float)cr[e] * z[(std::size_t)e]; }
        if (dot > bestdot) { bestdot = dot; best = code; }
      }
      codes[(std::size_t)t][(std::size_t)cb] = best;
      // residual -= out_proj(raw codebook[cb][best]) (== the decode path).
      const auto* Wout =
          static_cast<const _Float16*>(_q_outw[(std::size_t)cb].contents());  // [RV,CD]
      const auto* bout =
          static_cast<const _Float16*>(_q_outb[(std::size_t)cb].contents());  // [RV]
      const auto* cbr =
          static_cast<const _Float16*>(_codebook[(std::size_t)cb].contents()); // [Csz,CD]
      const _Float16* crow = cbr + (std::size_t)best * CD;
      for (int o = 0; o < RV; ++o) {
        const _Float16* wr = Wout + (std::size_t)o * CD;
        float acc = (float)bout[o];
        for (int e = 0; e < CD; ++e) { acc += (float)wr[e] * (float)crow[e]; }
        resid[(std::size_t)o] -= acc;
      }
    }
  }
  return codes;
}

}  // namespace vpipe::genai
