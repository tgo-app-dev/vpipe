#include "generative-models/quantize/calibration.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/ui-delegate-intf.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/tokenizer.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/task.h>

namespace vpipe::genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

std::uint64_t
host_free_ram_bytes()
{
  vm_size_t page = 0;
  if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS || page == 0) {
    return 0;
  }
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&vm),
                        &cnt) != KERN_SUCCESS) {
    return 0;
  }
  // "Available now" = everything the OS can hand back without swapping out
  // anonymous memory: truly free + speculative + reclaimable clean
  // (purgeable) + inactive. Inactive is INCLUDED: streaming a 67 GB
  // checkpoint via mmap fills inactive with CLEAN file-backed pages that the
  // OS evicts on demand, so excluding it under-reports headroom by tens of GB
  // and trips the guard on cache, not on real (wired) pressure. The actual
  // panic risk is OUR resident/wired growth, guarded separately by
  // process_memory_bytes -- this floor is the system-wide backstop.
  const std::uint64_t pages =
      (std::uint64_t)vm.free_count + vm.speculative_count +
      vm.purgeable_count + vm.inactive_count;
  return pages * (std::uint64_t)page;
}

void
process_memory_bytes(std::uint64_t* resident, std::uint64_t* wired)
{
  if (resident) { *resident = 0; }
  if (wired) { *wired = 0; }
  task_vm_info_data_t ti{};
  mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&ti), &cnt) == KERN_SUCCESS) {
    if (resident) { *resident = (std::uint64_t)ti.phys_footprint; }
  }
  // Per-process wired is not exposed; report SYSTEM-WIDE wired (the "Pages
  // wired down" an external vm_stat monitor watches) so the two cross-check.
  vm_size_t page = 0;
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t vcnt = HOST_VM_INFO64_COUNT;
  if (wired && host_page_size(mach_host_self(), &page) == KERN_SUCCESS &&
      host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&vm),
                        &vcnt) == KERN_SUCCESS) {
    *wired = (std::uint64_t)vm.wire_count * (std::uint64_t)page;
  }
}

namespace {

// Wrap already-encoded `body` ids as a single chat USER turn so the model's
// special control tokens land in the calibration distribution. ChatML is
// preferred (the Qwen3/MOSS backbones use <|im_start|>/<|im_end|>); the
// Llama-3 header tokens are the fallback. Fragments around the markers are
// BPE-encoded separately and the special ids injected, mirroring how the
// LM/TTS stages build their prompts. The body is placed UNCLIPPED between the
// wrapper tokens (the caller reserves room), so the closing <|im_end|> /
// <|eot_id|> survive. Returns false when the tokenizer carries neither
// family's control tokens (caller then uses the raw body).
bool
wrap_chat_turn_(const Tokenizer& tok, const std::vector<std::int32_t>& body,
                std::vector<std::int32_t>* dst)
{
  auto enc = [&](std::string_view s) {
    auto ids = tok.encode(s);
    dst->insert(dst->end(), ids.begin(), ids.end());
  };
  auto put_body = [&]() { dst->insert(dst->end(), body.begin(), body.end()); };
  const std::int32_t im_start = tok.special_token_id("<|im_start|>");
  const std::int32_t im_end   = tok.special_token_id("<|im_end|>");
  if (im_start >= 0 && im_end >= 0) {
    dst->push_back(im_start); enc("user\n"); put_body();
    dst->push_back(im_end); enc("\n");
    dst->push_back(im_start); enc("assistant\n");
    return true;
  }
  const std::int32_t bos = tok.special_token_id("<|begin_of_text|>");
  const std::int32_t sh  = tok.special_token_id("<|start_header_id|>");
  const std::int32_t eh  = tok.special_token_id("<|end_header_id|>");
  const std::int32_t eot = tok.special_token_id("<|eot_id|>");
  if (sh >= 0 && eh >= 0 && eot >= 0) {
    if (bos >= 0) { dst->push_back(bos); }
    dst->push_back(sh); enc("user"); dst->push_back(eh); enc("\n\n");
    put_body(); dst->push_back(eot);
    dst->push_back(sh); enc("assistant"); dst->push_back(eh); enc("\n\n");
    return true;
  }
  return false;
}

// Throttled in-place progress bar -- redraws on a carriage-return only
// when the integer percentage changes (the frame is space-padded so a
// shorter redraw fully overwrites a longer prior one).
void quant_progress_(vpipe::UiTextStream* bar, const char* tag, int done,
                     int total, int& last_pct)
{
  if (bar == nullptr || total <= 0) { return; }
  int pct = static_cast<int>(static_cast<long>(done) * 100 / total);
  if (pct < 0) { pct = 0; } else if (pct > 100) { pct = 100; }
  if (pct == last_pct) { return; }
  last_pct = pct;
  constexpr int W = 24;
  const int fill = pct * W / 100;
  std::string b(static_cast<std::size_t>(fill), '#');
  b += std::string(static_cast<std::size_t>(W - fill), '-');
  std::string line = fmt("\r[{}] {}% {} ({}/{})", b, pct, tag, done,
                         total)();
  while (line.size() < 64) { line += ' '; }   // wipe stale tail
  bar->write(line);
}

}  // namespace

std::vector<std::vector<std::int32_t>>
build_builtin_calibration_corpus(const Tokenizer& tok, int max_seqs,
                                 int seq_len, bool apply_chat_template)
{
  std::vector<std::vector<std::int32_t>> corpus;
  if (max_seqs <= 0 || seq_len <= 0) { return corpus; }
  const std::string_view all = builtin_calibration_text();

  // Group the newline-separated corpus into ~seq_len-token documents. A char
  // budget (~4 chars/token) bounds each document; the final per-document
  // tokenization is truncated to seq_len so denser text stays in range. A
  // document boundary is preferred at a line break so we don't split mid-line.
  const std::size_t char_budget = (std::size_t)seq_len * 4;
  std::string doc;
  doc.reserve(char_budget + 256);

  // Reserve a few slots for the chat wrapper tokens so the closing marker
  // survives the seq_len cap (the body is truncated, not the wrapper).
  const int wrap_reserve = apply_chat_template ? 12 : 0;
  auto flush = [&]() {
    if (doc.empty()) { return; }
    auto e = tok.encode(doc);
    std::vector<std::int32_t> body(e.begin(), e.end());
    const int cap = seq_len - wrap_reserve;
    if (cap >= 1 && (int)body.size() > cap) {
      body.resize((std::size_t)cap);
    }
    std::vector<std::int32_t> ids;
    if (!apply_chat_template || !wrap_chat_turn_(tok, body, &ids)) {
      ids = std::move(body);
    }
    if ((int)ids.size() > seq_len) { ids.resize((std::size_t)seq_len); }
    if (!ids.empty()) { corpus.push_back(std::move(ids)); }
    doc.clear();
  };

  std::size_t pos = 0;
  while (pos < all.size() && (int)corpus.size() < max_seqs) {
    std::size_t nl = all.find('\n', pos);
    const std::size_t end = (nl == std::string_view::npos) ? all.size() : nl;
    const std::string_view line = all.substr(pos, end - pos);
    pos = (nl == std::string_view::npos) ? all.size() : nl + 1;
    if (!doc.empty()) { doc.push_back('\n'); }
    doc.append(line);
    if (doc.size() >= char_budget) { flush(); }
  }
  if ((int)corpus.size() < max_seqs) { flush(); }
  return corpus;
}

bool
collect_backbone_calibration(
    MetalCompute* mc, const std::string& model_dir,
    const MetalQwenModel::Config& cfg,
    const std::vector<std::vector<std::int32_t>>& corpus,
    const std::string& out_calib_dir, std::string* err,
    const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  auto fail = [&](const std::string& m) { if (err) { *err = m; }
                                          return false; };
  if (mc == nullptr) { return fail("calib: null MetalCompute"); }

  auto bb = MetalQwenModel::load(model_dir, mc, cfg);
  if (!bb) { return fail("calib: backbone load failed: " + model_dir); }

  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts.has_value()) { return fail("calib: cannot open " + model_dir); }
  const std::string en =
      cfg.weight_prefix + cfg.model_seg + "embed_tokens.weight";
  SharedBuffer emb = wts->load(en, mc);
  if (emb.empty()) { return fail("calib: embed table missing: " + en); }
  const int H = cfg.hidden;
  const auto* etab = static_cast<const std::uint16_t*>(emb.contents());

  const SessionContextIntf* S = mc->session();
  if (S) {
    S->log_normal(fmt(
        "calibration: {} sequences over the loaded backbone",
        corpus.size()));
  }

  std::unique_ptr<vpipe::UiTextStream> bar;
  if (S) { bar = S->open_text_stream(); }
  bb->calib_begin();
  ContextManager* cm = bb->context_manager();
  int cal_pct = -1;
  int si = 0;
  for (const auto& seq : corpus) {
    if (stop()) { bb->calib_end(); return fail("calib: stopped by request"); }
    ++si;
    if (S) { S->log_verbose(fmt("  calib seq {}/{}", si, corpus.size())); }
    quant_progress_(bar.get(), "calib", si, (int)corpus.size(), cal_pct);
    const int n = (int)seq.size();
    if (n <= 0) { continue; }
    // Gather the token embeddings (bf16 compute dtype) host-side.
    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * H * 2);
    if (x.empty()) { bb->calib_end(); return fail("calib: embed alloc"); }
    auto* xd = static_cast<std::uint16_t*>(x.contents());
    for (int t = 0; t < n; ++t) {
      int id = seq[(std::size_t)t];
      if (id < 0 || id >= cfg.vocab) { id = 0; }
      std::memcpy(xd + (std::size_t)t * H,
                  etab + (std::size_t)id * H, (std::size_t)H * 2);
    }
    const ContextId cid = cm->acquire_root();
    SharedBuffer hn = bb->forward_embeddings_hidden(cid, x, n);
    cm->release(cid);
    if (hn.empty()) { bb->calib_end(); return fail("calib: forward failed"); }
  }
  bb->calib_end();
  if (bar) { bar->end(); }   // finalize the bar line before any summary

  std::error_code ec;
  fs::create_directories(out_calib_dir, ec);
  auto write = [&](const char* nm, const std::vector<std::vector<float>>& v,
                   int dim) -> bool {
    std::ofstream o((fs::path(out_calib_dir) / nm).string(),
                    std::ios::binary | std::ios::trunc);
    if (!o) { return false; }
    for (const auto& row : v) {
      if ((int)row.size() != dim) { return false; }
      o.write(reinterpret_cast<const char*>(row.data()),
              (std::streamsize)dim * 4);
    }
    return (bool)o;
  };
  if (!write("calib_qkv.f32", bb->calib_qkv(), H) ||
      !write("calib_gateup.f32", bb->calib_gateup(), H) ||
      !write("calib_down.f32", bb->calib_down(), cfg.ffn_inner)) {
    return fail("calib: write failed in " + out_calib_dir);
  }
  return true;
}

bool
collect_backbone_calibration_streaming(
    MetalCompute* mc, const std::string& model_dir,
    const MetalQwenModel::Config& cfg,
    const std::vector<std::vector<std::int32_t>>& corpus,
    const std::string& out_calib_dir, std::string* err,
    std::uint64_t min_free_bytes, const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  auto fail = [&](const std::string& m) { if (err) { *err = m; }
                                          return false; };
  if (mc == nullptr) { return fail("calib-stream: null MetalCompute"); }
  if (!cfg.is_moe()) {
    return fail("calib-stream: only the MoE path uses streaming calibration "
                "(dense backbones use collect_backbone_calibration)");
  }
  const SessionContextIntf* S = mc->session();

  auto mem_line = [&](const char* tag, int L) {
    std::uint64_t res = 0, wired = 0;
    process_memory_bytes(&res, &wired);
    const std::uint64_t freeb = host_free_ram_bytes();
    std::fprintf(stderr,
        "[calib-stream] %-10s L=%-3d | resident %6.2f GB | sys-wired %6.2f GB "
        "| free %6.2f GB\n", tag, L, (double)res / (1ull << 30),
        (double)wired / (1ull << 30), (double)freeb / (1ull << 30));
  };

  // mmap the checkpoint (pageable; tensors are pulled one layer at a time).
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts.has_value()) { return fail("calib-stream: cannot open " + model_dir); }

  const int H = cfg.hidden;
  const int nL = cfg.n_layers;

  // ---- residual stream: embed the corpus, held for the whole pass --------
  // (tens of MB: sum(n_i) * H * 2 bytes). The embed table is loaded, used,
  // and FREED before the layer stream so its ~1 GB doesn't co-reside.
  std::size_t total_tok = 0;
  for (const auto& s : corpus) { total_tok += s.size(); }
  if (total_tok == 0) { return fail("calib-stream: empty corpus"); }
  std::vector<SharedBuffer> resid(corpus.size());
  {
    const std::string en =
        cfg.weight_prefix + cfg.model_seg + "embed_tokens.weight";
    SharedBuffer emb = wts->load(en, mc);
    if (emb.empty()) { return fail("calib-stream: embed table missing: " + en); }
    const auto* etab = static_cast<const std::uint16_t*>(emb.contents());
    for (std::size_t si = 0; si < corpus.size(); ++si) {
      const auto& seq = corpus[si];
      const int n = (int)seq.size();
      if (n <= 0) { continue; }
      SharedBuffer x = mc->make_shared_buffer((std::size_t)n * H * 2);
      if (x.empty()) { return fail("calib-stream: residual alloc"); }
      auto* xd = static_cast<std::uint16_t*>(x.contents());
      for (int t = 0; t < n; ++t) {
        int id = seq[(std::size_t)t];
        if (id < 0 || id >= cfg.vocab) { id = 0; }
        std::memcpy(xd + (std::size_t)t * H,
                    etab + (std::size_t)id * H, (std::size_t)H * 2);
      }
      resid[si] = std::move(x);
    }
  }  // emb freed here
  const double resid_mb = (double)(total_tok * H * 2) / (1024.0 * 1024.0);
  std::fprintf(stderr,
      "[calib-stream] residual stream held: %.1f MB (%zu tokens, %zu seqs)\n",
      resid_mb, total_tok, corpus.size());
  mem_line("post-embed", -1);

  // PROBE mode (VPIPE_CALIB_STREAM_PROBE): build/free each layer to PROVE the
  // loader is memory-bounded, WITHOUT running the forward (so it needs no full
  // GDN/MoE config beyond the layer shapes). The real run (below) runs the
  // per-layer forward + per-expert taps.
  const bool probe = [] {
    const char* p = std::getenv("VPIPE_CALIB_STREAM_PROBE");
    return p != nullptr && std::atoi(p) != 0;
  }();

  // Streaming-calibration model: kernels + _ctx + dense/MoE detection, but
  // NO per-layer weights (calib_stream skips the layer-weight load), so the
  // 35B never resides whole. calib_build_layer streams one layer at a time.
  MetalQwenModel::Config scfg = cfg;
  scfg.calib_stream = true;
  auto bb = MetalQwenModel::load(model_dir, mc, scfg);
  if (!bb) { return fail("calib-stream: backbone (calib_stream) load failed"); }
  if (!probe) { bb->calib_begin_streaming(); }

  std::vector<int> seq_lens(corpus.size());
  for (std::size_t si = 0; si < corpus.size(); ++si) {
    if (stop()) { return fail("calib-stream: stopped by request"); }
    seq_lens[si] = (int)corpus[si].size();
  }

  std::uint64_t peak_wired = 0, peak_resident = 0;
  // The real panic guard: OUR resident (mlock-wired UMA) must never approach
  // RAM. One streamed layer is ~1.6 GB transient, so a generous ceiling
  // (overridable) catches a full-load regression long before the box hangs.
  std::uint64_t resident_ceiling = (std::uint64_t)16 << 30;
  if (const char* rc = std::getenv("VPIPE_CALIB_STREAM_RES_CEIL_GB")) {
    const long g = std::atol(rc);
    if (g > 0) { resident_ceiling = (std::uint64_t)g << 30; }
  }
  if (S) {
    S->log_normal(fmt(
        "calibration (streaming): {} layers over {} sequences", nL,
        corpus.size()));
  }
  std::unique_ptr<vpipe::UiTextStream> bar;
  if (S) { bar = S->open_text_stream(); }
  int cal_pct = -1;
  for (int L = 0; L < nL; ++L) {
    if (stop()) { return fail("calib-stream: stopped by request"); }
    if (S) { S->log_verbose(fmt("  calib layer {}/{}", L + 1, nL)); }
    quant_progress_(bar.get(), "calib", L, nL, cal_pct);
    const std::uint64_t freeb = host_free_ram_bytes();
    if (freeb != 0 && freeb < min_free_bytes) {
      char m[256];
      std::snprintf(m, sizeof m,
          "calib-stream: ABORT before layer %d -- available RAM %.2f GB < "
          "guard %.2f GB (refusing to risk a panic)", L,
          (double)freeb / (1ull << 30),
          (double)min_free_bytes / (1ull << 30));
      return fail(m);
    }
    std::uint64_t res0 = 0, wired0 = 0;
    process_memory_bytes(&res0, &wired0);
    if (res0 > resident_ceiling) {
      char m[256];
      std::snprintf(m, sizeof m,
          "calib-stream: ABORT before layer %d -- process resident %.2f GB > "
          "ceiling %.2f GB (a layer was NOT freed -- full-load regression)", L,
          (double)res0 / (1ull << 30),
          (double)resident_ceiling / (1ull << 30));
      return fail(m);
    }
    // Build ONLY layer L's weights into the model's transient _layers[L].
    std::uint64_t layer_bytes = 0;
    std::string le;
    if (!bb->calib_build_layer(*wts, L, &layer_bytes, &le)) {
      return fail(le);
    }
    {
      std::uint64_t res = 0, wired = 0;
      process_memory_bytes(&res, &wired);
      if (wired > peak_wired) { peak_wired = wired; }
      if (res > peak_resident) { peak_resident = res; }
    }
    if (L < 3 || L == nL - 1 || (L % 8) == 0) {
      std::fprintf(stderr,
          "[calib-stream] layer %2d loaded: %.2f GB\n",
          L, (double)layer_bytes / (1ull << 30));
      mem_line("loaded", L);
    }
    if (!probe) {
      std::string re;
      if (!bb->calib_run_layer(L, resid, seq_lens, &re)) {
        bb->calib_free_layer(L);
        return fail(re);
      }
    }
    bb->calib_free_layer(L);   // FREE layer L before L+1 (the invariant)
  }
  if (bar) { bar->end(); }   // finalize the bar line before any summary
  std::fprintf(stderr,
      "[calib-stream] streamed %d layers; PEAK process-resident %.2f GB, "
      "PEAK system-wired %.2f GB\n", nL,
      (double)peak_resident / (1ull << 30),
      (double)peak_wired / (1ull << 30));

  if (probe) { (void)out_calib_dir; return true; }   // bound check only

  // ---- write the calibration stats ------------------------------------
  std::error_code ec;
  fs::create_directories(out_calib_dir, ec);
  auto write = [&](const char* nm, const std::vector<std::vector<float>>& v,
                   std::size_t dim) -> bool {
    std::ofstream o((fs::path(out_calib_dir) / nm).string(),
                    std::ios::binary | std::ios::trunc);
    if (!o) { return false; }
    for (const auto& row : v) {
      if (row.size() != dim) { return false; }
      o.write(reinterpret_cast<const char*>(row.data()),
              (std::streamsize)dim * 4);
    }
    return (bool)o;
  };
  const std::size_t Hh = (std::size_t)cfg.hidden;
  const std::size_t EH = (std::size_t)cfg.n_experts * cfg.hidden;
  const std::size_t EI = (std::size_t)cfg.n_experts * cfg.moe_inner;
  if (!write("calib_qkv.f32", bb->calib_qkv(), Hh) ||
      !write("calib_gateup.f32", bb->calib_gateup(), Hh) ||
      !write("calib_expert_gateup.f32", bb->calib_expert_gateup(), EH) ||
      !write("calib_expert_down.f32", bb->calib_expert_down(), EI)) {
    return fail("calib-stream: write failed in " + out_calib_dir);
  }
  // Sanity summary: stats must be finite + non-zero (a dead forward gives all
  // zeros; a broken one gives NaN/Inf).
  auto summarize = [&](const char* nm,
                       const std::vector<std::vector<float>>& v) {
    double mx = 0.0, mn = 1e30; long nz = 0, tot = 0, bad = 0;
    for (const auto& row : v) {
      for (float f : row) {
        ++tot;
        if (!std::isfinite(f)) { ++bad; continue; }
        if (f != 0.0f) { ++nz; }
        if (f > mx) { mx = f; }
        if (f < mn) { mn = f; }
      }
    }
    std::fprintf(stderr,
        "[calib-stream] %-20s rows=%zu n=%ld nonzero=%ld nonfinite=%ld "
        "min=%.4g max=%.4g\n", nm, v.size(), tot, nz, bad, mn, mx);
  };
  summarize("calib_qkv", bb->calib_qkv());
  summarize("calib_gateup", bb->calib_gateup());
  summarize("calib_expert_gateup", bb->calib_expert_gateup());
  summarize("calib_expert_down", bb->calib_expert_down());
  return true;
}

}  // namespace vpipe::genai
