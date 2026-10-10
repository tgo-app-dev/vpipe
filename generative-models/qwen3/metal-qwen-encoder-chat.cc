#include "generative-models/qwen3/metal-qwen-encoder-chat.h"

#include "generative-models/context-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/token-muxer.h"
#include "generative-models/tokenizer.h"

#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

namespace vpipe::genai {

namespace {

// The host's cap when a request names none: comfortably past a JSON
// verdict, and short enough that a classifier which never stops costs
// seconds rather than minutes.
constexpr int kDefaultCap = 256;

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

bool
QwenEncoderChat::generate(const screen::ChatRequest& req, std::string* text,
                          std::string* why)
{
  _last_error.clear();
  auto fail = [&](const char* msg) {
    _last_error = msg;
    if (why != nullptr) { *why = msg; }
    return false;
  };
  if (text == nullptr) { return fail("no output string"); }
  text->clear();

  // Render the turn from its PARTS: specials pushed by id, text through
  // the tokenizer, the picture as one image-pad position per vision row.
  std::vector<std::int32_t> ids;
  std::vector<TokenRef>     refs;
  auto put_id = [&](std::int32_t id) {
    ids.push_back(id);
    TokenRef r;
    r.kind = TokenRef::Kind::Text;
    r.text_id = id;
    refs.push_back(r);
  };
  int first_pad = -1;
  int images = 0;
  for (const screen::ChatPart& p : req.parts) {
    switch (p.kind) {
    case screen::ChatPart::Kind::Special: {
      const std::int32_t id = _tok.special_token_id(p.text);
      if (id < 0) { return fail("a special token the tokenizer lacks"); }
      put_id(id);
      break;
    }
    case screen::ChatPart::Kind::Text:
      for (const std::int32_t id : _tok.encode(p.text)) { put_id(id); }
      break;
    case screen::ChatPart::Kind::Image: {
      if (_picture == nullptr || _picture->vision == nullptr ||
          _picture->rows <= 0) {
        return fail("an image part with no picture bound to this pass");
      }
      if (images++ > 0) { return fail("more than one image part"); }
      const std::int32_t pad = _tok.special_token_id("<|image_pad|>");
      if (pad < 0) { return fail("the tokenizer has no <|image_pad|>"); }
      first_pad = (int)ids.size();
      for (int i = 0; i < _picture->rows; ++i) {
        ids.push_back(pad);
        TokenRef r;
        r.kind = TokenRef::Kind::ImageTokens;
        r.image_token_offset = i;
        refs.push_back(r);
      }
      break;
    }
    default:
      return fail("an unknown part kind");
    }
  }
  std::int32_t stop = -1;
  if (!req.stop.empty()) {
    stop = _tok.special_token_id(req.stop);
    if (stop < 0) { return fail("a stop token the tokenizer lacks"); }
  }

  const int n = (int)ids.size();
  const int H = _lm.config().hidden;
  if (n <= 0 || H <= 0) { return fail("an empty turn"); }

  // Embeddings through the model's own muxer -- which is why the encoder
  // must be loaded with its embedding table and head.
  metal_compute::SharedBuffer x = _lm.embed_text_buf(ids);
  if (x.empty() || x.byte_size() < (std::size_t)n * H * 2) {
    return fail("token embedding failed");
  }

  // The tower's f16 rows over the image-pad embeddings, in the model's
  // element type.
  const bool grounded = first_pad >= 0;
  if (grounded) {
    const std::size_t need = (std::size_t)_picture->rows * H * 2;
    if (_picture->vision->byte_size() < need) {
      return fail("the picture's vision rows are short");
    }
    const auto* vt =
        static_cast<const _Float16*>(_picture->vision->contents());
    const bool bf16 = _lm.config().use_bf16;
    auto* xh = static_cast<std::uint16_t*>(x.contents());
    for (int j = 0; j < _picture->rows; ++j) {
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

  // The STOCK Qwen3-VL position rule when a picture rides the turn: 3-axis
  // mROPE over its merged grid. A text-only turn takes plain positions.
  std::vector<std::int32_t> pos;
  int rope_next = -1;
  if (grounded) {
    std::vector<std::int32_t> pt, ph, pw;
    const std::pair<int, int> grid{_picture->grid_h, _picture->grid_w};
    if (!build_mrope_position_ids(
            refs, std::span<const std::pair<int, int>>(&grid, 1), &pt, &ph,
            &pw, &rope_next)) {
      return fail("the multimodal position build failed");
    }
    pos.reserve((std::size_t)3 * n);
    pos.insert(pos.end(), pt.begin(), pt.end());
    pos.insert(pos.end(), ph.begin(), ph.end());
    pos.insert(pos.end(), pw.begin(), pw.end());
  }

  MetalQwenModel::DeepstackInject ds;
  const bool use_ds = grounded && !_picture->deepstack.empty();
  if (use_ds) {
    for (int i = 0; i < (int)_picture->deepstack.size(); ++i) {
      ds.feats.push_back(_picture->deepstack[(std::size_t)i]);
      ds.layers.push_back(i);
    }
    ds.row0 = first_pad;
    ds.rows = _picture->rows;
  }

  ContextManager* cm = _lm.context_manager();
  if (cm == nullptr) { return fail("the encoder has no KV context"); }
  const ContextId cid = cm->acquire_root();
  if (!cid.valid()) { return fail("no KV context available"); }

  std::vector<float> logits =
      grounded ? _lm.prefill_multimodal_buf(cid, std::move(x), pos, n,
                                            use_ds ? &ds : nullptr)
               : _lm.prefill_embeddings_buf(cid, std::move(x), n);
  if (logits.empty()) {
    cm->release(cid);
    return fail("prefill failed");
  }

  const int cap = req.max_new_tokens > 0 ? req.max_new_tokens : kDefaultCap;
  std::vector<std::int32_t> gen;
  gen.reserve((std::size_t)cap);
  std::int32_t id = argmax_(logits);
  int rope = rope_next;
  for (int step = 0; step < cap && id != stop; ++step) {
    gen.push_back(id);
    logits = _lm.forward(cid, id, grounded ? rope++ : -1);
    if (logits.empty()) {
      cm->release(cid);
      return fail("decode failed");
    }
    id = argmax_(logits);
  }
  cm->release(cid);
  *text = _tok.decode(gen);
  return true;
}

}  // namespace vpipe::genai
