#include "generative-models/mage/mage-screen.h"

#include "common/flex-data.h"

#include <cctype>
#include <string_view>

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/context-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/token-muxer.h"
#include "generative-models/tokenizer.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <cstring>
#endif

namespace vpipe {
namespace genai {

namespace {

std::string
trim_(const std::string& s)
{
  std::size_t b = 0, e = s.size();
  auto sp = [](unsigned char c) { return std::isspace(c) != 0; };
  while (b < e && sp((unsigned char)s[b])) { ++b; }
  while (e > b && sp((unsigned char)s[e - 1])) { --e; }
  return s.substr(b, e - b);
}

}  // namespace

bool
mage_parse_verdict(const std::string& text, MageScreenVerdict* out)
{
  if (out == nullptr) { return false; }
  std::string s = trim_(text);
  if (s.empty()) { return false; }

  // Strip a ``` / ```json fence if the model wrapped its JSON in one (the
  // system prompt forbids it, but a classifier that disobeys the format must
  // still be UNDERSTOOD -- silently failing closed on a well-formed verdict
  // just because it arrived fenced would block benign prompts).
  if (s.rfind("```", 0) == 0) {
    s.erase(0, 3);
    if (s.rfind("json", 0) == 0) { s.erase(0, 4); }
    const std::size_t close = s.rfind("```");
    if (close != std::string::npos) { s.erase(close); }
    s = trim_(s);
  }

  // First BALANCED top-level object. Brace counting is string- and
  // escape-aware, so a brace inside "reason" cannot end the object early.
  const std::size_t start = s.find('{');
  if (start == std::string::npos) { return false; }
  int depth = 0;
  bool in_str = false, esc = false;
  std::size_t end = std::string::npos;
  for (std::size_t i = start; i < s.size(); ++i) {
    const char c = s[i];
    if (esc) { esc = false; continue; }
    if (in_str) {
      if (c == '\\') { esc = true; }
      else if (c == '"') { in_str = false; }
      continue;
    }
    if (c == '"') { in_str = true; }
    else if (c == '{') { ++depth; }
    else if (c == '}') {
      if (--depth == 0) { end = i; break; }
    }
  }
  if (end == std::string::npos) { return false; }

  FlexData fd = FlexData::from_json(
      std::string_view(s).substr(start, end - start + 1));
  if (!fd.is_object()) { return false; }
  auto o = fd.as_object();
  // "violates" is the verdict; a JSON object WITHOUT it is not a verdict at
  // all (fail closed) rather than an implicit allow.
  if (!o.contains("violates")) { return false; }

  MageScreenVerdict v;
  v.violates = o.at("violates").as_bool(true);
  if (o.contains("categories")) {
    FlexData cats = o.at("categories");     // as_array() is a view: keep it
    if (cats.is_array()) {
      for (const auto& c : cats.as_array()) {
        const std::string_view sv = c.as_string("");
        if (!sv.empty()) { v.categories.emplace_back(sv); }
      }
    }
  }
  if (o.contains("reason")) {
    v.reason = trim_(std::string(o.at("reason").as_string("")));
  }
  v.raw = text;
  *out = std::move(v);
  return true;
}

#ifdef VPIPE_BUILD_APPLE_SILICON

namespace {

std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

std::int32_t
argmax_(const std::vector<float>& v)
{
  std::int32_t best = 0;
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[(std::size_t)best]) { best = (std::int32_t)i; }
  }
  return best;
}

}  // namespace

MageScreenVerdict
mage_screen(MetalQwenModel& lm, const Tokenizer& tok,
            const MageScreenRequest& req, const SessionContextIntf* session)
{
  MageScreenVerdict v;     // FAIL-CLOSED: blocks unless the model clears it
  const bool grounded = req.n_img > 0 && req.vision != nullptr;

  // An empty text-to-image prompt is not a violation (the reference says so
  // explicitly). With a source image there IS something to judge, so an
  // empty EDIT instruction still gets screened.
  const std::string instruction = trim_(req.prompt);
  if (instruction.empty() && !grounded) {
    v.violates = false;
    v.reason = "empty prompt";
    return v;
  }

  const std::int32_t im_start = tok.special_token_id("<|im_start|>");
  const std::int32_t im_end   = tok.special_token_id("<|im_end|>");
  const std::int32_t vis_beg  = tok.special_token_id("<|vision_start|>");
  const std::int32_t vis_end  = tok.special_token_id("<|vision_end|>");
  const std::int32_t img_pad  = tok.special_token_id("<|image_pad|>");
  if (im_start < 0 || im_end < 0
      || (grounded && (vis_beg < 0 || vis_end < 0 || img_pad < 0))) {
    if (session != nullptr) {
      session->error(fmt("mage_screen: the tokenizer is missing the Qwen "
                         "chat specials; BLOCKING (fail-closed)"));
    }
    return v;
  }

  // Render the classifier turn EXACTLY as the reference's
  // apply_chat_template(..., add_generation_prompt=True) does:
  //   <|im_start|>system\n{policy}<|im_end|>\n
  //   <|im_start|>user\n[<|vision_start|>{pads}<|vision_end|>]{ask}<|im_end|>\n
  //   <|im_start|>assistant\n
  // Assembled from PARTS (specials pushed by id) rather than scanned out of
  // one big string -- the policy text is full of punctuation and quoting, and
  // a marker scanner over it is a way to get silently wrong tokens.
  std::vector<std::int32_t> ids;
  std::vector<TokenRef>     refs;
  auto put_id = [&](std::int32_t id) {
    ids.push_back(id);
    TokenRef r; r.kind = TokenRef::Kind::Text; r.text_id = id;
    refs.push_back(r);
  };
  auto put_text = [&](std::string_view t) {
    for (const std::int32_t id : tok.encode(t)) { put_id(id); }
  };

  put_id(im_start);
  put_text("system\n");
  put_text(grounded ? kMageFilterEditSystem : kMageFilterSystem);
  put_id(im_end);
  put_text("\n");
  put_id(im_start);
  put_text("user\n");

  int first_pad = -1;
  if (grounded) {
    put_id(vis_beg);
    first_pad = (int)ids.size();
    for (int i = 0; i < req.n_img; ++i) {
      ids.push_back(img_pad);
      TokenRef r;
      r.kind = TokenRef::Kind::ImageTokens;
      r.image_token_offset = i;
      refs.push_back(r);
    }
    put_id(vis_end);
    // The reference builds one image content item per source PIL and then
    // this sentence; vpipe's conditioner grounds on a single reference image.
    put_text("There is 1 source image(s) above. Edit instruction: "
             + (instruction.empty() ? std::string("(no textual instruction)")
                                    : instruction)
             + "\nClassify this edit request.");
  } else {
    put_text("Prompt to classify:\n" + instruction);
  }
  put_id(im_end);
  put_text("\n");
  put_id(im_start);
  put_text("assistant\n");

  const int n = (int)ids.size();
  const int H = lm.config().hidden;
  if (n <= 0 || H <= 0) { return v; }

  // Embeddings via the model's own muxer -- the encoder is loaded WITH its
  // token embeddings (not backbone-only) precisely so it can generate.
  metal_compute::SharedBuffer x = lm.embed_text_buf(ids);
  if (x.empty() || x.byte_size() < (std::size_t)n * H * 2) {
    if (session != nullptr) {
      session->error(fmt("mage_screen: token embedding failed; BLOCKING "
                         "(fail-closed)"));
    }
    return v;
  }

  // Splice the tower's f16 rows over the image_pad embeddings.
  if (grounded && first_pad >= 0) {
    const auto* vt = static_cast<const _Float16*>(req.vision->contents());
    const std::size_t need = (std::size_t)req.n_img * H * 2;
    if (req.vision->byte_size() < need) { return v; }
    const bool bf16 = lm.config().use_bf16;
    auto* xh = static_cast<std::uint16_t*>(x.contents());
    for (int j = 0; j < req.n_img; ++j) {
      const std::size_t dst = (std::size_t)(first_pad + j) * H;
      for (int h = 0; h < H; ++h) {
        const float e = (float)vt[(std::size_t)j * H + h];
        if (bf16) {
          xh[dst + h] = f32_to_bf16_(e);
        } else {
          const _Float16 hf = (_Float16)e;
          std::memcpy(&xh[dst + h], &hf, 2);
        }
      }
    }
  }

  // mROPE position ids (grounded only). NOTE this is the STOCK Qwen3-VL
  // position rule, not the sequential-arange one the Mage conditioning path
  // uses: the reference screens via hf.generate() on the raw Qwen3-VL module,
  // which never goes through TextEncoder.forward's position_ids override.
  std::vector<std::int32_t> pos;
  int rope_next = -1;
  if (grounded) {
    std::vector<std::int32_t> pt, ph, pw;
    const std::pair<int, int> grid{req.img_mh, req.img_mw};
    if (!build_mrope_position_ids(refs, std::span<const std::pair<int, int>>(
                                            &grid, 1), &pt, &ph, &pw,
                                  &rope_next)) {
      if (session != nullptr) {
        session->error(fmt("mage_screen: mROPE position build failed; "
                           "BLOCKING (fail-closed)"));
      }
      return v;
    }
    pos.reserve((std::size_t)3 * n);
    pos.insert(pos.end(), pt.begin(), pt.end());
    pos.insert(pos.end(), ph.begin(), ph.end());
    pos.insert(pos.end(), pw.begin(), pw.end());
  }

  MetalQwenModel::DeepstackInject ds;
  const bool use_ds = grounded && first_pad >= 0 && !req.deepstack.empty();
  if (use_ds) {
    for (int i = 0; i < (int)req.deepstack.size(); ++i) {
      ds.feats.push_back(req.deepstack[(std::size_t)i]);
      ds.layers.push_back(i);
    }
    ds.row0 = first_pad;
    ds.rows = req.n_img;
  }

  ContextManager* cm = lm.context_manager();
  if (cm == nullptr) { return v; }
  const ContextId cid = cm->acquire_root();
  if (!cid.valid()) {
    if (session != nullptr) {
      session->error(fmt("mage_screen: no KV context; BLOCKING (fail-closed)"));
    }
    return v;
  }

  std::vector<float> logits =
      grounded ? lm.prefill_multimodal_buf(cid, std::move(x), pos, n,
                                           use_ds ? &ds : nullptr)
               : lm.prefill_embeddings_buf(cid, std::move(x), n);
  if (logits.empty()) {
    cm->release(cid);
    if (session != nullptr) {
      session->error(fmt("mage_screen: classifier prefill failed; BLOCKING "
                         "(fail-closed)"));
    }
    return v;
  }

  // Greedy decode (the reference's do_sample=False) to the JSON verdict.
  const int cap = req.max_new_tokens > 0 ? req.max_new_tokens
                                         : (grounded ? 192 : 160);
  std::vector<std::int32_t> gen;
  gen.reserve((std::size_t)cap);
  std::int32_t id = argmax_(logits);
  int rope = rope_next;
  for (int step = 0; step < cap && id != im_end; ++step) {
    gen.push_back(id);
    logits = lm.forward(cid, id, grounded ? rope++ : -1);
    if (logits.empty()) {
      cm->release(cid);
      if (session != nullptr) {
        session->error(fmt("mage_screen: classifier decode failed; BLOCKING "
                           "(fail-closed)"));
      }
      return v;
    }
    id = argmax_(logits);
  }
  cm->release(cid);

  const std::string out = trim_(tok.decode(gen));
  if (!mage_parse_verdict(out, &v)) {
    v = MageScreenVerdict{};              // back to the blocking default
    v.categories.push_back("policy");
    v.reason = "unparsable classifier output (blocked)";
    v.raw = out;
  }
  return v;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

}  // namespace genai
}  // namespace vpipe
