// moss-tts-delay.cc -- the MOSS-TTS delay-pattern family (MossTTSDelay:
// MOSS-TTS 1.0 8B and MOSS-TTS-v1.5): the v1.5 text normalizer, the
// prompt grid, the rolling de-delay, the teacher-forced forward against
// the PyTorch reference, streaming against one-shot decode, and an ASR
// round trip through the text-to-speech stage.
//
// Env (each optional; a test whose inputs are unset returns early):
//   VPIPE_MOSS_TTS_V15_MODEL   OpenMOSS-Team/MOSS-TTS-v1.5 (bf16 as
//                              published, or a model-quantize'd dir)
//   VPIPE_MOSS_TTS_V15_GOLDEN  a dir written by tools/dump_moss_tts_golden.py
//   VPIPE_MOSS_CODEC_MODEL     OpenMOSS-Team/MOSS-Audio-Tokenizer
//   VPIPE_QWEN3_ASR_TEST_MODEL_PATH  Qwen3-ASR, for the round trip

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/moss/metal-moss-codec.h"
#include "generative-models/moss/metal-moss-tts-model.h"
#include "generative-models/moss/moss-delay-processor.h"
#include "generative-models/moss/moss-text-normalizer.h"
#include "generative-models/tokenizer.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "generative-models/shared/mma-splitk.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/shared/kernel-autotune.h"
#include "generative-models/shared/i8-gemm.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "pipeline/memory-plan.h"
#include "stages/audio-transcribe-stage.h"
#include "stages/audio-video/audio-temporal-resample-stage.h"
#include "stages/model-memory.h"
#include "stages/text-to-speech-stage.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using vpipe::genai::Tokenizer;
using vpipe::metal_compute::ComputeEncoder;

namespace {

const char*
env_(const char* k)
{
  const char* v = std::getenv(k);
  return (v != nullptr && *v != '\0') ? v : nullptr;
}

template <class T>
std::vector<T>
read_raw_(const std::string& path, std::size_t count)
{
  std::vector<T> v(count);
  std::ifstream f(path, std::ios::binary);
  if (!f) { return {}; }
  f.read(reinterpret_cast<char*>(v.data()),
         static_cast<std::streamsize>(count * sizeof(T)));
  if (static_cast<std::size_t>(f.gcount()) != count * sizeof(T)) { return {}; }
  return v;
}

std::vector<std::vector<std::int32_t>>
rows_of_(const std::vector<std::int32_t>& flat, int n, int ch)
{
  std::vector<std::vector<std::int32_t>> g(
      static_cast<std::size_t>(n),
      std::vector<std::int32_t>(static_cast<std::size_t>(ch)));
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < ch; ++c) {
      g[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
          flat[static_cast<std::size_t>(r * ch + c)];
    }
  }
  return g;
}

// A box too small to hold the 8B resident beside the rest of the process:
// what the stage would decide (it streams there), and where a test must
// not load bf16 resident or it pushes the machine into swap.
bool
small_box_()
{
  const std::size_t ram = model_memory::phys_ram();
  return ram > 0 && ram < ((std::size_t)32 << 30);
}

FlexData
read_json_(const std::string& path)
{
  std::ifstream in(path);
  if (!in) { return {}; }
  return FlexData::from_json(in);
}

}  // namespace

// ---- the v1.5 normalizer ----------------------------------------------

// The cases the reference script ships with (its own TEST_CASES), and its
// idempotence check. Needs no model.
TEST(moss_tts_delay, normalizer_script_cases)
{
  struct Case { const char* name; const char* in; const char* out; };
  static const Case kCases[] = {
  {"dot_map_sentence",
   "2026 年 3 月 31 日，安全研究员 Chaofan Shou (@Fried_rice) 发现 Anthropic "
   "的 npm 包中暴露了 .map 文件，",
   "2026年3月31日，安全研究员 Chaofan Shou (@Fried_rice) 发现 Anthropic 的 np"
   "m 包中暴露了 .map 文件，"},
  {"dot_tokens",
   "别把 .env、.npmrc、.gitignore 提交上去。",
   "别把 .env、.npmrc、.gitignore 提交上去。"},
  {"file_names",
   "请检查 bundle.min.js、package.json 和 processing_moss_tts.py。",
   "请检查 bundle.min.js、package.json 和 processing_moss_tts.py。"},
  {"index_d_ts",
   "index.d.ts 里也有同样的问题。",
   "index.d.ts 里也有同样的问题。"},
  {"version_build",
   "Bug 的讨论可以精确到 v2.3.1 (Build 15)。",
   "Bug 的讨论可以精确到 v2.3.1 (Build 15)。"},
  {"version_rc",
   "3.0.0-rc.1 还不能上生产。",
   "3.0.0-rc.1 还不能上生产。"},
  {"jar_name",
   "fabric-api-0.91.3+1.20.2.jar 需要单独下载。",
   "fabric-api-0.91.3+1.20.2.jar 需要单独下载。"},
  {"url",
   "仓库地址是 https://github.com/instructkr/claude-code",
   "仓库地址是 https://github.com/instructkr/claude-code"},
  {"email",
   "联系邮箱：ops+tts@example.ai",
   "联系邮箱：ops+tts@example.ai"},
  {"mention",
   "@Fried_rice 说这是 source map 暴露。",
   "@Fried_rice 说这是 source map 暴露。"},
  {"reddit",
   "去 r/singularity 看讨论。",
   "去 r/singularity 看讨论。"},
  {"hashtag_chain",
   "#张雪峰#张雪峰[话题]#张雪峰事件",
   "#张雪峰#张雪峰[话题]#张雪峰事件"},
  {"mention_hashtag_boundary",
   "关注@biscuit0228_并转发#thetime_tbs",
   "关注 @biscuit0228_ 并转发 #thetime_tbs"},
  {"speaker_bracket",
   "[S1]你好。[S2]收到。",
   "[S1]你好。[S2]收到。"},
  {"event_bracket",
   "请模仿 {whisper} 的语气说“别出声”。",
   "请模仿 {whisper} 的语气说“别出声”。"},
  {"order_bracket",
   "订单号：[AB-1234-XYZ]",
   "订单号：[AB-1234-XYZ]"},
  {"struct_headline",
   "〖重磅〗《新品发布》——现在开始！",
   "重磅。新品发布。现在开始！"},
  {"struct_notice",
   "【公告】今天 20:00 维护——预计 30 分钟。",
   "公告。今天20:00维护。预计30分钟。"},
  {"struct_quote_chain",
   "『特别提醒』「不要外传」",
   "特别提醒。不要外传。"},
  {"flow_arrow_chain",
   "请求接入 -> 身份与策略判定 -> 域服务处理",
   "请求接入，身份与策略判定，域服务处理"},
  {"flow_arrow_no_space",
   "A->B",
   "A，B"},
  {"flow_arrow_unicode",
   "配置中心→推理编排→运行时执行",
   "配置中心，推理编排，运行时执行"},
  {"flow_arrow_maas_example",
   "MaaS 主链遵循请求接入 -> 身份与策略判定 -> 域服务处理 -> 推理编排 -> 运行"
   "时执行的在线数据面结构。Dashboard 不直接承载单次在线推理请求，而是负责统"
   "一配置、统一治理、统一运营和统一展示。",
   "MaaS 主链遵循请求接入，身份与策略判定，域服务处理，推理编排，运行时执行的"
   "在线数据面结构。Dashboard 不直接承载单次在线推理请求，而是负责统一配置、"
   "统一治理、统一运营和统一展示。"},
  {"embedded_title",
   "我喜欢《哈姆雷特》这本书。",
   "我喜欢《哈姆雷特》这本书。"},
  {"noise_qe",
   "真的假的？？？！！！",
   "真的假的？！"},
  {"noise_ellipsis",
   "这个包把 app.js.map 也发上去了......太离谱了！！！",
   "这个包把 app.js.map 也发上去了。太离谱了！"},
  {"noise_ellipsis_cn",
   "【系统提示】请模仿{sad}低沉语气，说“今天下雨了……”",
   "系统提示。请模仿{sad}低沉语气，说“今天下雨了。”"},
  {"english_spaces",
   "This   is   a   test.",
   "This is a test."},
  {"chinese_spaces",
   "这 是\u3000一 段  含有多种空白的文本。",
   "这是一段含有多种空白的文本。"},
  {"mixed_spaces_1",
   "这是Anthropic的npm包",
   "这是 Anthropic 的 npm 包"},
  {"mixed_spaces_2",
   "今天update到v2.3.1了",
   "今天 update 到 v2.3.1 了"},
  {"mixed_spaces_3",
   "处理app.js.map文件",
   "处理 app.js.map 文件"},
  {"markdown_link",
   "详情见 [release note](https://github.com/example/release)",
   "详情见 release note https://github.com/example/release"},
  {"markdown_heading",
   "# I made a free open source app to help with markdown files",
   "I made a free open source app to help with markdown files"},
  {"list_lines",
   "- 修复 .map 泄露\n- 发布 v2.3.1",
   "修复 .map 泄露。发布 v2.3.1"},
  {"numbered_lines",
   "1. 安装依赖\n2. 运行测试\n3. 发布 v2.3.1",
   "安装依赖。运行测试。发布 v2.3.1"},
  {"newlines",
   "第一行\n第二行\n第三行",
   "第一行。第二行。第三行"},
  {"zero_width_url",
   "详见 https://x.com/\u200bSafety",
   "详见 https://x.com/Safety"},
  };
  int bad = 0;
  for (const auto& c : kCases) {
    const std::string got = moss_tts_normalize_text(c.in);
    if (got != c.out) {
      ++bad;
      std::fprintf(stderr, "[moss-norm] %s\n  want '%s'\n  got  '%s'\n",
                   c.name, c.out, got.c_str());
    }
    if (moss_tts_normalize_text(got) != got) {
      ++bad;
      std::fprintf(stderr, "[moss-norm] %s not idempotent\n", c.name);
    }
  }
  std::fprintf(stderr, "[moss-norm] %zu script cases, %d failures\n",
               sizeof(kCases) / sizeof(kCases[0]), bad);
  EXPECT_TRUE(bad == 0);
}

// The reference normalizer's output for the script cases plus the corners
// the dumper adds (README texts, NBSP, fullwidth digits, e-mail after CJK,
// arrows, markdown, zero-width / soft hyphen / private use, U+2028, ...).
TEST(moss_tts_delay, normalizer_matches_reference)
{
  const char* gold = env_("VPIPE_MOSS_TTS_V15_GOLDEN");
  if (gold == nullptr) { return; }
  FlexData pairs = read_json_(std::string(gold) + "/normalize.json");
  ASSERT_TRUE(pairs.is_array());
  if (!pairs.is_array()) { return; }
  const auto arr = pairs.as_array();
  int bad = 0;
  for (std::size_t i = 0; i < arr.size(); ++i) {
    const FlexData p = arr[i];
    const std::string in(p.as_array()[0].as_string(""));
    const std::string want(p.as_array()[1].as_string(""));
    const std::string got = moss_tts_normalize_text(in);
    if (got != want) {
      ++bad;
      std::fprintf(stderr, "[moss-norm-ref] case %zu\n  in   '%s'\n"
                   "  want '%s'\n  got  '%s'\n", i, in.c_str(), want.c_str(),
                   got.c_str());
    }
  }
  std::fprintf(stderr, "[moss-norm-ref] %zu cases, %d mismatches\n",
               arr.size(), bad);
  EXPECT_TRUE(arr.size() >= 60);
  EXPECT_TRUE(bad == 0);
}

// ---- the rolling de-delay ---------------------------------------------

// Feeding rows one at a time must give exactly the batch de-delay
// (apply_de_delay_pattern, all-pad frames dropped, pad clamped).
TEST(moss_tts_delay, dedelay_matches_batch)
{
  const int nv = 8, pad = 1024, G = 40;
  std::vector<std::vector<std::int32_t>> rows(
      G, std::vector<std::int32_t>(1 + nv, pad));
  // Speech in rows [3, 30): codebook cb active in rows [3 + cb, 30 + cb).
  for (int r = 0; r < G; ++r) {
    for (int cb = 0; cb < nv; ++cb) {
      if (r >= 3 + cb && r < 30 + cb) {
        rows[r][1 + cb] = (r * 37 + cb * 11) % pad;
      }
    }
  }
  rows[12][1 + 2] = pad;            // a pad inside speech: kept, clamped
  std::vector<std::vector<std::int32_t>> batch;
  for (int f = 0; f + nv - 1 < G; ++f) {
    std::vector<std::int32_t> fr(nv);
    bool all_pad = true;
    for (int cb = 0; cb < nv; ++cb) {
      int v = rows[f + cb][1 + cb];
      if (v != pad) { all_pad = false; }
      if (v < 0 || v >= pad) { v = pad - 1; }
      fr[cb] = v;
    }
    if (!all_pad) { batch.push_back(fr); }
  }
  MossDelayDedelay dd(nv, pad);
  std::vector<std::vector<std::int32_t>> rolled;
  std::vector<std::int32_t> fr;
  for (const auto& r : rows) {
    if (dd.push(r, fr)) { rolled.push_back(fr); }
  }
  EXPECT_TRUE(batch.size() == 27);
  EXPECT_TRUE(rolled == batch);
  // reset() starts a new utterance from scratch.
  dd.reset();
  std::vector<std::vector<std::int32_t>> again;
  for (const auto& r : rows) {
    if (dd.push(r, fr)) { again.push_back(fr); }
  }
  EXPECT_TRUE(again == batch);
}

// ---- the prompt grid --------------------------------------------------

// The reference processor's [seq, 1 + n_vq] grid for each case (language,
// instruction, tokens, a [pause] marker, multi-line text, a voice-clone
// reference), rebuilt here from the raw text: normalizer, then builder.
TEST(moss_tts_delay, grid_matches_reference)
{
  const char* model = env_("VPIPE_MOSS_TTS_V15_MODEL");
  const char* gold = env_("VPIPE_MOSS_TTS_V15_GOLDEN");
  if (model == nullptr || gold == nullptr) { return; }
  Session sess;
  auto tok = Tokenizer::from_huggingface_json(
      std::string(model) + "/tokenizer.json", &sess);
  ASSERT_TRUE(tok != nullptr);
  if (!tok) { return; }
  const std::string gd = std::string(gold) + "/grids/";
  FlexData cases = read_json_(gd + "cases.json");
  ASSERT_TRUE(cases.is_array());
  if (!cases.is_array()) { return; }
  const auto arr = cases.as_array();
  int checked = 0;
  for (std::size_t i = 0; i < arr.size(); ++i) {
    const FlexData c = arr[i];
    const auto o = c.as_object();
    const std::string name(o.at("name").as_string(""));
    const int seq = static_cast<int>(o.at("seq").as_int(0));
    const int ch = static_cast<int>(o.at("channels").as_int(0));
    const int nref = static_cast<int>(o.at("ref").as_int(0));
    auto str_or_none = [&](const char* k) {
      const FlexData v = o.at(k);
      if (v.is_string()) { return std::string(v.as_string("")); }
      if (v.is_int()) { return std::to_string(v.as_int(0)); }
      return std::string("None");
    };
    MossDelayUserFields f;
    f.language    = str_or_none("language");
    f.instruction = str_or_none("instruction");
    f.tokens      = str_or_none("tokens");
    std::vector<std::vector<std::int32_t>> ref;
    if (nref > 0) {
      ref = rows_of_(read_raw_<std::int32_t>(gd + name + ".ref.i32",
                                             (std::size_t)nref * (ch - 1)),
                     nref, ch - 1);
    }
    const std::string text =
        moss_tts_normalize_text(std::string(o.at("text").as_string("")));
    const auto want = rows_of_(
        read_raw_<std::int32_t>(gd + name + ".i32", (std::size_t)seq * ch),
        seq, ch);
    const auto got = moss_delay_build_grid(*tok, text, MossDelayPromptIds{},
                                           f, nref > 0 ? &ref : nullptr);
    const bool same = got == want;
    std::fprintf(stderr, "[moss-grid] %-13s seq ref=%d mine=%zu %s\n",
                 name.c_str(), seq, got.size(), same ? "EXACT" : "DIFFERS");
    if (!same) {
      for (std::size_t r = 0; r < std::min(got.size(), want.size()); ++r) {
        if (got[r] != want[r]) {
          std::fprintf(stderr, "   first diff at row %zu: mine %d ref %d\n",
                       r, got[r][0], want[r][0]);
          break;
        }
      }
    }
    EXPECT_TRUE(same);
    ++checked;
  }
  EXPECT_TRUE(checked >= 8);
}

// ---- the forward against the reference -----------------------------

namespace {

// The teacher-forced golden: the reference's sampled run and its fp32 /
// bf16 logits at every position that predicts a generated row.
struct ForwardGolden {
  int seq = 0, G = 0, ch = 0, nv = 0, ac = 0;
  std::vector<int> ctrl;
  std::vector<std::vector<std::int32_t>> prompt, gen;
  std::vector<float> a32, a16, t32, t16;   // [G][nv][ac+1], [G][ctrl]
  std::vector<std::int32_t> targmax;      // [G]
  bool ok = false;
};

ForwardGolden
load_forward_golden_(const std::string& gold)
{
  ForwardGolden g;
  const std::string d = gold + "/forward/";
  FlexData meta = read_json_(d + "meta.json");
  if (!meta.is_object()) { return g; }
  const auto o = meta.as_object();
  g.seq = (int)o.at("seq").as_int(0);
  g.G   = (int)o.at("G").as_int(0);
  g.ch  = (int)o.at("channels").as_int(0);
  g.nv  = (int)o.at("n_vq").as_int(0);
  g.ac  = (int)o.at("audio_vocab").as_int(0);
  const FlexData ctrl = o.at("ctrl_ids");
  for (std::size_t i = 0; i < ctrl.as_array().size(); ++i) {
    g.ctrl.push_back((int)ctrl.as_array()[i].as_int(0));
  }
  const std::size_t na = (std::size_t)g.G * g.nv * (g.ac + 1);
  const std::size_t nt = (std::size_t)g.G * g.ctrl.size();
  g.prompt = rows_of_(read_raw_<std::int32_t>(d + "input_ids.i32",
                                              (std::size_t)g.seq * g.ch),
                      g.seq, g.ch);
  g.gen = rows_of_(read_raw_<std::int32_t>(d + "gen.i32",
                                           (std::size_t)g.G * g.ch),
                   g.G, g.ch);
  g.a32 = read_raw_<float>(d + "audio_logits_fp32.f32", na);
  g.a16 = read_raw_<float>(d + "audio_logits_bf16.f32", na);
  g.t32 = read_raw_<float>(d + "text_ctrl_fp32.f32", nt);
  g.t16 = read_raw_<float>(d + "text_ctrl_bf16.f32", nt);
  g.targmax = read_raw_<std::int32_t>(d + "text_argmax_fp32.i32", g.G);
  g.ok = g.seq > 0 && g.G > 0 && !g.a32.empty() && !g.a16.empty() &&
         !g.t32.empty() && !g.targmax.empty();
  return g;
}

// Distances from the fp32 reference over the codebooks that were ACTIVE
// (sampled) at each row, plus agreement on the fp32-DECISIVE picks: a
// top-1 that beats the runner-up by more than `margin`, where two bf16
// implementations have no business disagreeing.
struct TfStats {
  double rel = 0.0;          // audio logits, active codebooks
  double text_rel = 0.0;     // the control-token text logits
  int decisive = 0, agree = 0;
  bool row0 = false;         // the free text pick at row 0 (audio_start)
};

TfStats
tf_stats_(const ForwardGolden& g, const std::vector<float>& mine_a,
          const std::vector<float>& mine_t, const float* ref_a,
          const float* ref_t, std::int32_t mine_row0,
          float margin = 1.0f)
{
  TfStats s;
  const int pad = g.ac;
  double num = 0, den = 0;
  for (int r = 0; r < g.G; ++r) {
    for (int cb = 0; cb < g.nv; ++cb) {
      if (g.gen[(std::size_t)r][(std::size_t)(1 + cb)] == pad) { continue; }
      const std::size_t base = ((std::size_t)r * g.nv + cb) * (g.ac + 1);
      int b1 = 0;
      float v1 = -1e30f, v2 = -1e30f;
      for (int c = 0; c < g.ac; ++c) {
        const double a = mine_a[base + c], b = g.a32[base + c];
        num += (a - b) * (a - b);
        den += b * b;
        const float v = g.a32[base + c];
        if (v > v1) { v2 = v1; v1 = v; b1 = c; } else if (v > v2) { v2 = v; }
      }
      if (v1 - v2 > margin) {
        ++s.decisive;
        int mb = 0;
        float mv = -1e30f;
        for (int c = 0; c < g.ac; ++c) {
          if (mine_a[base + c] > mv) { mv = mine_a[base + c]; mb = c; }
        }
        if (mb == b1) { ++s.agree; }
      }
    }
  }
  (void)ref_a;
  s.rel = den > 0 ? std::sqrt(num / den) : 0.0;
  double tn = 0, td = 0;
  for (std::size_t i = 0; i < g.t32.size(); ++i) {
    const double a = mine_t[i], b = ref_t[i];
    tn += (a - b) * (a - b);
    td += b * b;
  }
  s.text_rel = td > 0 ? std::sqrt(tn / td) : 0.0;
  s.row0 = mine_row0 == g.targmax[0];
  return s;
}

// Teacher-force the golden's rows through `m`, collecting the audio logits
// (flattened like the golden), the control-token text logits and the
// masked text argmax at row 0.
void
teacher_force_(MetalMossTtsModel& m, const ForwardGolden& g,
               std::vector<float>* audio, std::vector<float>* text,
               std::int32_t* row0, int rows = -1)
{
  const int G = rows > 0 ? std::min(rows, g.G) : g.G;
  audio->assign((std::size_t)G * g.nv * (g.ac + 1), 0.0f);
  text->assign((std::size_t)G * g.ctrl.size(), 0.0f);
  const auto& c = m.config();
  std::vector<std::vector<std::int32_t>> rws(g.gen.begin(),
                                             g.gen.begin() + G);
  m.teacher_force_logits(
      g.prompt, rws, /*full_text=*/true,
      [&](int r, const std::vector<std::vector<float>>& a,
          const std::vector<float>& t) {
        for (int cb = 0; cb < g.nv; ++cb) {
          std::copy(a[(std::size_t)cb].begin(), a[(std::size_t)cb].end(),
                    audio->begin() +
                        ((std::size_t)r * g.nv + cb) * (g.ac + 1));
        }
        for (std::size_t k = 0; k < g.ctrl.size(); ++k) {
          (*text)[(std::size_t)r * g.ctrl.size() + k] =
              t[(std::size_t)g.ctrl[k]];
        }
        if (r == 0) {
          float bv = -1e30f;
          int best = -1;
          for (int i = 0; i < (int)t.size(); ++i) {
            if (i == c.pad_token || i == c.audio_gen_slot ||
                i == c.audio_delay_slot || i == c.audio_end) {
              continue;
            }
            if (t[(std::size_t)i] > bv) { bv = t[(std::size_t)i]; best = i; }
          }
          *row0 = best;
        }
      });
}

}  // namespace

// The backbone + heads against the PyTorch reference, teacher-forced over
// the reference's own sampled run (G rows). The holdings of the same
// published bf16 checkpoint:
//
//   bf16       held as it sits; held to the reference's OWN bf16 error
//   w8 trunk16 backbone quantized in memory, embedding + heads bf16 --
//              the control that isolates what trunk_w8 adds
//   w8         backbone AND text embedding + text head at w8 (the default)
//   pack       VPIPE_MOSS_TTS_V15_W8_PACK, a model-quantize'd w8 pack of
//              the same checkpoint, trunk at w8; streamed on a small box.
//              model-quantize writes a MOSS pack with the in-memory
//              build's own kernel (QuantizeOptions::bf16_scales), so its
//              logits must EQUAL the w8 arm's, bit for bit
//
// All must miss no more fp32-decisive picks than the reference's own bf16
// run does, and pick audio_start first. MEASURED on the M4 Pro (131 rows):
// reference bf16 rel 0.0070, 236/237 decisive; vpipe bf16 0.0048, 237/237;
// vpipe w8 (bf16 trunk) 0.0075, 236/237, holding 9983 MB against 16193.
TEST(moss_tts_delay, v15_forward_matches_reference)
{
  const char* model = env_("VPIPE_MOSS_TTS_V15_MODEL");
  const char* gold = env_("VPIPE_MOSS_TTS_V15_GOLDEN");
  if (model == nullptr || gold == nullptr) { return; }
  const ForwardGolden g = load_forward_golden_(gold);
  ASSERT_TRUE(g.ok);
  if (!g.ok) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // The reference's own bf16 run, by the same metric -- the bar.
  const TfStats ref = tf_stats_(g, g.a16, g.t16, nullptr, g.t32.data(),
                                g.targmax[0]);
  std::fprintf(stderr, "[moss-fwd] reference bf16: rel %.5f text %.5f "
               "decisive %d/%d\n", ref.rel, ref.text_rel, ref.agree,
               ref.decisive);

  const char* pack = env_("VPIPE_MOSS_TTS_V15_W8_PACK");
  struct Arm { const char* name; int qb; bool trunk_w8; const char* dir; };
  std::vector<Arm> arms = {{"bf16", 0, false, model},
                           {"w8 trunk16", 8, false, model},
                           {"w8", 8, true, model}};
  if (pack != nullptr) { arms.push_back({"pack", 8, true, pack}); }
  std::vector<float> w8_a, w8_t;   // the in-memory w8 arm, for the pack's
  for (const Arm& arm : arms) {
    const int qb = arm.qb;
    // A small box streams, as the stage would, and skips the arms that
    // exist only to compare (bf16, and the bf16-trunk control).
    if ((qb == 0 || !arm.trunk_w8) && small_box_()) { continue; }
    MossTtsLoadOptions opts;
    opts.quant_bits = qb;
    opts.trunk_w8 = arm.trunk_w8;
    opts.stream = small_box_();
    const auto t0 = std::chrono::steady_clock::now();
    auto m = MetalMossTtsModel::load(arm.dir, mc, opts);
    ASSERT_TRUE(m != nullptr && m->valid());
    if (!m) { continue; }
    if (opts.stream) {
      m->set_residency_reserve((std::size_t)512 << 20);
      m->set_residency_schedule(g.G);
    }
    const double load_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const bool is_pack = arm.dir == pack;
    EXPECT_TRUE(m->quantized_on_load() == (qb == 8 && !is_pack));
    EXPECT_TRUE(m->trunk_quantized() == arm.trunk_w8);
    if (is_pack) { EXPECT_TRUE(m->streams_pack() == opts.stream); }
    std::vector<float> a, t;
    std::int32_t row0 = -1;
    const auto t1 = std::chrono::steady_clock::now();
    teacher_force_(*m, g, &a, &t, &row0);
    const double tf_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t1).count();
    const TfStats s = tf_stats_(g, a, t, nullptr, g.t32.data(), row0);
    std::fprintf(stderr, "[moss-fwd] %-10s load %.1fs (%zu MB held%s) tf "
                 "%d rows %.1fs | rel %.5f text %.5f decisive %d/%d row0 "
                 "%s\n", arm.name, load_s, m->resident_bytes() >> 20,
                 m->streams() ? ", streaming" : "",
                 g.G, tf_s, s.rel, s.text_rel, s.agree, s.decisive,
                 s.row0 ? "audio_start" : "WRONG");
    EXPECT_TRUE(s.row0);
    if (std::string(arm.name) == "w8") { w8_a = a; w8_t = t; }
    if (is_pack && !w8_a.empty()) {
      const bool same =
          a.size() == w8_a.size() && t.size() == w8_t.size() &&
          std::memcmp(a.data(), w8_a.data(), a.size() * sizeof(float)) == 0 &&
          std::memcmp(t.data(), w8_t.data(), t.size() * sizeof(float)) == 0;
      std::fprintf(stderr, "[moss-fwd] pack against in-memory w8: %s\n",
                   same ? "IDENTICAL" : "DIFFER");
      EXPECT_TRUE(same);
    }
    // No more decisive misses than the reference's own bf16 run makes
    // (it misses one at this margin: two bf16 implementations can land
    // either side of a 1-logit gap after 36 layers).
    EXPECT_TRUE(s.decisive > 100 &&
                s.decisive - s.agree <= ref.decisive - ref.agree);
    if (qb == 0) {
      // bf16 against bf16: within 1.5x the reference's own bf16 distance.
      EXPECT_TRUE(s.rel <= 1.5 * ref.rel + 1e-3);
    } else {
      // w8 g64: quantization noise on top. Held to 4x the bf16 floor,
      // which keeps it well under the scale where picks start to move.
      EXPECT_TRUE(s.rel <= 4.0 * ref.rel + 2e-3);
    }
  }
}

// STREAMED == RESIDENT, byte for byte: the same w8 build either way, so a
// streamed layer must reproduce a resident one exactly. Three arms over
// the golden's prompt + its first rows:
//
//   resident  every layer built at load
//   streamed  nothing admitted: every layer read into a slot per forward
//   mixed     residency grows on the prefill, then the tail half is
//             released, so the decode runs resident and streamed layers
//             side by side
//
// The checkpoint is read once per streamed layer per forward, so this is
// slow off an external drive; the row counts are kept small for that.
TEST(moss_tts_delay, v15_streamed_matches_resident)
{
  const char* model = env_("VPIPE_MOSS_TTS_V15_MODEL");
  const char* gold = env_("VPIPE_MOSS_TTS_V15_GOLDEN");
  if (model == nullptr || gold == nullptr) { return; }
  const ForwardGolden g = load_forward_golden_(gold);
  ASSERT_TRUE(g.ok);
  if (!g.ok) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (small_box_()) {
    std::fprintf(stderr, "[moss-stream] box too small to hold the resident "
                 "arm; the identity is box-independent (run it on 32 GB+)\n");
    return;
  }
  const int rows = 3;

  std::vector<float> ra, rt;
  std::int32_t r0 = -1;
  {
    auto m = MetalMossTtsModel::load(model, mc, MossTtsLoadOptions{});
    ASSERT_TRUE(m != nullptr);
    if (!m) { return; }
    EXPECT_TRUE(!m->streams());
    teacher_force_(*m, g, &ra, &rt, &r0, rows);
  }
  auto same = [&](const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) { return false; }
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
  };

  MossTtsLoadOptions so;
  so.stream = true;
  {
    auto m = MetalMossTtsModel::load(model, mc, so);
    ASSERT_TRUE(m != nullptr);
    if (!m) { return; }
    EXPECT_TRUE(m->streams());
    // A reserve no box has: nothing is admitted, every layer streams.
    m->set_residency_reserve((std::size_t)1 << 50);
    std::vector<float> a, t;
    std::int32_t z = -1;
    const auto t0 = std::chrono::steady_clock::now();
    teacher_force_(*m, g, &a, &t, &z, rows);
    std::fprintf(stderr, "[moss-stream] streamed: %d resident of %d, "
                 "%.1fs for prefill + %d rows | audio %s text %s\n",
                 m->resident_layer_count(), m->n_layers(),
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count(),
                 rows, same(a, ra) ? "IDENTICAL" : "DIFFER",
                 same(t, rt) ? "IDENTICAL" : "DIFFER");
    EXPECT_TRUE(m->resident_layer_count() == 0);
    EXPECT_TRUE(same(a, ra) && same(t, rt));
  }
  {
    auto m = MetalMossTtsModel::load(model, mc, so);
    ASSERT_TRUE(m != nullptr);
    if (!m) { return; }
    m->set_residency_reserve(0);
    m->set_residency_schedule(64);
    // Grow on a first pass, then give the tail half back and stop growth.
    std::vector<float> a, t;
    std::int32_t z = -1;
    teacher_force_(*m, g, &a, &t, &z, 1);
    const int grown = m->resident_layer_count();
    // One built layer, from the checkpoint's own sizes: release half.
    const std::size_t per = MetalMossTtsModel::plan_memory(model, 8).layer;
    m->release_resident_layers(per * (std::size_t)(grown / 2));
    m->set_residency_reserve((std::size_t)1 << 50);
    const int kept = m->resident_layer_count();
    teacher_force_(*m, g, &a, &t, &z, rows);
    std::fprintf(stderr, "[moss-stream] mixed: grew to %d, kept %d of %d "
                 "| audio %s text %s\n", grown, kept, m->n_layers(),
                 same(a, ra) ? "IDENTICAL" : "DIFFER",
                 same(t, rt) ? "IDENTICAL" : "DIFFER");
    EXPECT_TRUE(grown > 0 && kept > 0 && kept < m->n_layers());
    EXPECT_TRUE(same(a, ra) && same(t, rt));
  }
}

// A STREAMED PACK == THE SAME PACK RESIDENT, byte for byte. Resident, a
// pack loads through the backbone's ordinary pack loader; streamed, its
// layers are read into slots as they sit and the build only fuses q|k|v
// and interleaves gate|up (copy_rows_u32). Two different routes to one
// layout, so this is what proves the row copies right -- a wrong row map
// would still run. Arms as v15_streamed_matches_resident: every layer
// streamed, then mixed. VPIPE_MOSS_TTS_V15_W8_PACK = a model-quantize'd
// (bits 8, group 64) copy of the v1.5 checkpoint.
TEST(moss_tts_delay, v15_pack_streamed_matches_resident)
{
  const char* pack = env_("VPIPE_MOSS_TTS_V15_W8_PACK");
  const char* gold = env_("VPIPE_MOSS_TTS_V15_GOLDEN");
  if (pack == nullptr || gold == nullptr) { return; }
  const ForwardGolden g = load_forward_golden_(gold);
  ASSERT_TRUE(g.ok);
  if (!g.ok) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (small_box_()) {
    std::fprintf(stderr, "[moss-pack] box too small to hold the resident "
                 "arm; the identity is box-independent (run it on 32 GB+)\n");
    return;
  }
  const int rows = 3;
  auto same = [&](const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) { return false; }
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
  };

  std::vector<float> ra, rt;
  std::int32_t r0 = -1;
  {
    auto m = MetalMossTtsModel::load(pack, mc, MossTtsLoadOptions{});
    ASSERT_TRUE(m != nullptr);
    if (!m) { return; }
    EXPECT_TRUE(!m->streams() && !m->quantized_on_load());
    teacher_force_(*m, g, &ra, &rt, &r0, rows);
  }
  MossTtsLoadOptions so;
  so.stream = true;
  {
    auto m = MetalMossTtsModel::load(pack, mc, so);
    ASSERT_TRUE(m != nullptr);
    if (!m) { return; }
    EXPECT_TRUE(m->streams() && m->streams_pack());
    m->set_residency_reserve((std::size_t)1 << 50);   // admit nothing
    std::vector<float> a, t;
    std::int32_t z = -1;
    const auto t0 = std::chrono::steady_clock::now();
    teacher_force_(*m, g, &a, &t, &z, rows);
    std::fprintf(stderr, "[moss-pack] streamed: %d resident of %d, %zu MB "
                 "held, %.1fs for prefill + %d rows | audio %s text %s\n",
                 m->resident_layer_count(), m->n_layers(),
                 m->resident_bytes() >> 20,
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count(),
                 rows, same(a, ra) ? "IDENTICAL" : "DIFFER",
                 same(t, rt) ? "IDENTICAL" : "DIFFER");
    EXPECT_TRUE(m->resident_layer_count() == 0);
    EXPECT_TRUE(same(a, ra) && same(t, rt));
  }
  {
    auto m = MetalMossTtsModel::load(pack, mc, so);
    ASSERT_TRUE(m != nullptr);
    if (!m) { return; }
    m->set_residency_reserve(0);
    m->set_residency_schedule(64);
    std::vector<float> a, t;
    std::int32_t z = -1;
    teacher_force_(*m, g, &a, &t, &z, 1);
    const int grown = m->resident_layer_count();
    const std::size_t per = MetalMossTtsModel::plan_memory(pack, 8).layer;
    m->release_resident_layers(per * (std::size_t)(grown / 2));
    m->set_residency_reserve((std::size_t)1 << 50);
    const int kept = m->resident_layer_count();
    teacher_force_(*m, g, &a, &t, &z, rows);
    std::fprintf(stderr, "[moss-pack] mixed: grew to %d, kept %d of %d "
                 "| audio %s text %s\n", grown, kept, m->n_layers(),
                 same(a, ra) ? "IDENTICAL" : "DIFFER",
                 same(t, rt) ? "IDENTICAL" : "DIFFER");
    EXPECT_TRUE(grown > 0 && kept > 0 && kept < m->n_layers());
    EXPECT_TRUE(same(a, ra) && same(t, rt));
  }
}

// ---- streaming PCM ----------------------------------------------------

// The stage's streamed path, end to end below the stage: rows de-delayed
// as they arrive and decoded by the codec's windowed streaming decode in
// stream_chunk_frames chunks must give the one-shot PCM exactly -- and the
// per-row callback must not perturb sampling (same seed, same rows).
TEST(moss_tts_delay, v15_stream_pcm_matches_oneshot)
{
  const char* model = env_("VPIPE_MOSS_TTS_V15_MODEL");
  const char* codec_dir = env_("VPIPE_MOSS_CODEC_MODEL");
  if (model == nullptr || codec_dir == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MossTtsLoadOptions lo;
  lo.stream = small_box_();
  auto m = MetalMossTtsModel::load(model, mc, lo);
  if (m && lo.stream) {
    m->set_residency_reserve((std::size_t)512 << 20);
    m->set_residency_schedule(256);
  }
  auto codec = MetalMossCodec::load(codec_dir, mc);
  ASSERT_TRUE(m != nullptr && codec != nullptr && codec->valid());
  if (!m || !codec) { return; }
  auto tok = Tokenizer::from_huggingface_json(
      std::string(model) + "/tokenizer.json", &sess);
  ASSERT_TRUE(tok != nullptr);
  if (!tok) { return; }
  const auto& c = m->config();
  MossDelayPromptIds ids;
  ids.n_vq = c.n_vq;
  ids.audio_pad = c.audio_pad_code;
  MossDelayUserFields f;
  f.language = "English";
  const auto prompt = moss_delay_build_grid(
      *tok, moss_tts_normalize_text("Streaming audio should match exactly."),
      ids, f);
  MossSampling sp;
  sp.temperature = 1.7f; sp.top_k = 25; sp.top_p = 0.8f;
  const std::uint64_t seed = 4242;

  const auto rows1 = m->generate_delay(prompt, 220, sp, {}, seed);
  MossDelayDedelay dd1(c.n_vq, c.audio_pad_code);
  std::vector<std::vector<std::int32_t>> frames;
  std::vector<std::int32_t> fr;
  for (const auto& r : rows1) {
    if (dd1.push(r, fr)) { frames.push_back(fr); }
  }
  ASSERT_TRUE(frames.size() > 16);
  if (frames.size() <= 16) { return; }
  const std::vector<float> full = codec->decode(frames, nullptr);

  const int chunk = 16;
  auto st = codec->decode_stream_begin(chunk);
  ASSERT_TRUE(st != nullptr);
  MossDelayDedelay dd2(c.n_vq, c.audio_pad_code);
  std::vector<std::vector<std::int32_t>> pend;
  std::vector<float> streamed;
  auto flush = [&]() {
    if (pend.empty()) { return; }
    const std::vector<float> pcm = codec->decode_stream_chunk(*st, pend);
    streamed.insert(streamed.end(), pcm.begin(), pcm.end());
    pend.clear();
  };
  const auto rows2 = m->generate_delay(
      prompt, 220, sp, {}, seed, [&](const std::vector<std::int32_t>& r) {
        if (dd2.push(r, fr)) {
          pend.push_back(fr);
          if ((int)pend.size() >= chunk) { flush(); }
        }
        return true;
      });
  flush();
  EXPECT_TRUE(rows2 == rows1);
  double num = 0, den = 0, maxd = 0;
  const std::size_t n = std::min(full.size(), streamed.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)full[i] - streamed[i];
    num += d * d;
    den += (double)full[i] * full[i];
    maxd = std::max(maxd, std::fabs(d));
  }
  std::fprintf(stderr, "[moss-stream-pcm] %zu rows, %zu frames, %zu vs %zu "
               "samples, rel %.3g max|d| %.3g\n", rows1.size(), frames.size(),
               full.size(), streamed.size(), den > 0 ? std::sqrt(num / den)
                                                     : 0.0, maxd);
  EXPECT_TRUE(full.size() == streamed.size());
  EXPECT_TRUE(maxd == 0.0);
}

// ---- the stage: memory declarations ---------------------------------

// What text-to-speech declares for MOSS-TTS-v1.5 before anything loads,
// in both ledgers: the LM at what it will HOLD (w8 in memory, not the
// bf16 on disk) with a streaming floor it can reach, the codec at f16
// without the encoder, a streamable weight claim, and a scratch claim.
// No model is loaded -- the plan comes from the checkpoints' tables.
TEST(moss_tts_delay, stage_declares_held_bytes)
{
  const char* model = env_("VPIPE_MOSS_TTS_V15_MODEL");
  const char* codec_dir = env_("VPIPE_MOSS_CODEC_MODEL");
  if (model == nullptr || codec_dir == nullptr) { return; }
  Session sess;
  auto make = [&](const char* quant, const char* dir) {
    FlexData cfg = FlexData::make_object();
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(dir));
    o.insert("codec_dir", FlexData::make_string(codec_dir));
    o.insert("lm_quant", FlexData::make_string(quant));
    return std::make_unique<TextToSpeechStage>(
        &sess, "tts", std::vector<InEdge>{}, std::move(cfg));
  };
  const std::size_t lm_disk = 16979683328ull;   // the index's total_size
  const char* pack = env_("VPIPE_MOSS_TTS_V15_W8_PACK");
  for (const char* q : {"w8", "bf16", "pack"}) {
    const bool is_pack = std::string(q) == "pack";
    if (is_pack && pack == nullptr) { continue; }
    auto st = make(is_pack ? "w8" : q, is_pack ? pack : model);
    EXPECT_TRUE(st->config_error().empty());
    const StageMemory m = st->declare_memory();
    ASSERT_TRUE(m.holdings.size() == 2);
    if (m.holdings.size() != 2) { continue; }
    const StageHolding& lm = m.holdings[0];
    const StageHolding& cc = m.holdings[1];
    std::fprintf(stderr, "[moss-plan] lm_quant=%s: LM preload %zu MB floor "
                 "%zu MB | codec %zu MB | scratch %zu MB\n", q,
                 lm.preload >> 20, lm.floor >> 20, cc.preload >> 20,
                 m.scratch >> 20);
    if (std::string(q) != "bf16") {
      // ~53% of the backbone and of the text embedding + head, plus the
      // bf16 audio heads and embeddings: 8.8 GB of 16.2 -- and the same
      // whether the w8 is built in memory or read from a pack.
      EXPECT_TRUE(lm.preload < lm_disk * 3 / 5 && lm.preload > lm_disk / 2);
    } else {
      EXPECT_TRUE(lm.preload > lm_disk * 9 / 10);
    }
    // Streaming holds the trunk and two slots. A pack's slot holds its
    // layer's w8 bytes rather than a bf16 copy, so its floor is lower.
    EXPECT_TRUE(lm.floor > 0 && lm.floor < lm.preload / 2);
    if (is_pack) { EXPECT_TRUE(lm.floor < lm.preload / 3); }
    EXPECT_TRUE(cc.preload > 0 &&
                cc.preload < (std::size_t)2 << 30);   // f16 decoder only
    EXPECT_TRUE(m.scratch > 0);
    const auto claims = st->declare_resources();
    int streamable = 0, scratch = 0;
    for (const auto& c : claims) {
      if (c.floor_bytes > 0) { ++streamable; }
      if (c.kind == "activation-scratch") { ++scratch; }
    }
    EXPECT_TRUE(streamable == 1 && scratch == 1);
  }
}

// ---- the stage: what it says, heard back ----------------------------

#if defined(VPIPE_BUILD_APPLE_SILICON)
namespace {

class TextBeats : public TypedStage<TextBeats> {
public:
  static constexpr const char* kTypeName = "ut-moss-text-beats";
  using TypedStage::TypedStage;
  std::vector<std::string> texts;
  Job process(RuntimeContext& ctx) override
  {
    if (_i >= texts.size()) { ctx.signal_done(); co_return; }
    co_await ctx.write(0, make_payload<FlexDataPayload>(
        FlexData::make_string(texts[_i++])));
  }
private:
  std::size_t _i = 0;
};

class TranscriptSink : public TypedStage<TranscriptSink> {
public:
  static constexpr const char* kTypeName = "ut-moss-transcripts";
  using TypedStage::TypedStage;
  std::vector<std::string> got;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    if (const auto* f = dynamic_cast<const FlexDataPayload*>(p.get())) {
      if (f->data.is_object() && f->data.as_object().contains("text")) {
        got.emplace_back(f->data.as_object().at("text").as_string(""));
      }
    }
  }
};

// Lower-case words with punctuation dropped; Qwen3-ASR's leading
// "language X<asr_text>" tag removed.
std::vector<std::string>
words_(std::string s)
{
  const std::size_t tag = s.find("<asr_text>");
  if (tag != std::string::npos) { s = s.substr(tag + 10); }
  std::vector<std::string> out;
  std::string w;
  for (unsigned char ch : s) {
    if (std::isalnum(ch) || ch >= 0x80 || ch == '\'') {
      w.push_back((char)std::tolower(ch));
    } else if (!w.empty()) {
      out.push_back(w);
      w.clear();
    }
  }
  if (!w.empty()) { out.push_back(w); }
  return out;
}

double
wer_(const std::vector<std::string>& ref, const std::vector<std::string>& hyp)
{
  std::vector<std::size_t> prev(hyp.size() + 1), cur(hyp.size() + 1);
  for (std::size_t j = 0; j <= hyp.size(); ++j) { prev[j] = j; }
  for (std::size_t i = 1; i <= ref.size(); ++i) {
    cur[0] = i;
    for (std::size_t j = 1; j <= hyp.size(); ++j) {
      cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ref[i - 1] == hyp[j - 1] ? 0 : 1)});
    }
    std::swap(prev, cur);
  }
  return ref.empty() ? 0.0 : (double)prev[hyp.size()] / (double)ref.size();
}

}  // namespace

// MOSS-TTS-v1.5 through the text-to-speech stage (in-memory w8, the
// default), straight into Qwen3-ASR: English, then French with the
// language named, as the model card asks. Word error against the text.
TEST(moss_tts_delay, v15_stage_speech_transcribes)
{
  const char* model = env_("VPIPE_MOSS_TTS_V15_MODEL");
  const char* codec_dir = env_("VPIPE_MOSS_CODEC_MODEL");
  const char* asr = env_("VPIPE_QWEN3_ASR_TEST_MODEL_PATH");
  if (model == nullptr || codec_dir == nullptr || asr == nullptr) { return; }
  const std::vector<std::pair<std::string, std::string>> cases = {
    {"English", "The quick brown fox jumps over the lazy dog. Speech "
                "synthesis on Apple silicon is fast."},
    {"French", "Bonjour, je voudrais essayer une voix française naturelle "
               "et stable."},
  };
  for (const auto& [lang, text] : cases) {
    Session sess;
    Pipeline pl("moss-asr", &sess);
    auto src = std::make_unique<TextBeats>(&sess, "text",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
    src->texts = {text};
    src->allocate_oports(1);
    auto* s0 = pl.insert_stage(std::move(src));
    FlexData tcfg = FlexData::make_object();
    {
      auto o = tcfg.as_object();
      o.insert("hf_dir", FlexData::make_string(model));
      o.insert("codec_dir", FlexData::make_string(codec_dir));
      o.insert("language", FlexData::make_string(lang));
      // One PCM beat per utterance: the ASR stage takes a beat as a clip.
      o.insert("stream_chunk_frames", FlexData::make_int(0));
      o.insert("max_new_tokens", FlexData::make_int(400));
    }
    auto* tts = pl.insert_stage(std::make_unique<TextToSpeechStage>(
        &sess, "tts", std::vector<InEdge>{{s0, 0}}, std::move(tcfg)));
    // Qwen3-ASR reads 16 kHz and does not resample a block-mode clip.
    FlexData rcfg = FlexData::make_object();
    {
      auto o = rcfg.as_object();
      o.insert("output_sample_rate", FlexData::make_int(16000));
      o.insert("stacked", FlexData::make_bool(true));
    }
    auto* rs = pl.insert_stage(std::make_unique<AudioTemporalResampleStage>(
        &sess, "resample", std::vector<InEdge>{{tts, 0}}, std::move(rcfg)));
    FlexData acfg = FlexData::make_object();
    {
      auto o = acfg.as_object();
      o.insert("hf_dir", FlexData::make_string(asr));
      o.insert("max_new_tokens", FlexData::make_int(256));
    }
    auto* a = pl.insert_stage(std::make_unique<AudioTranscribeStage>(
        &sess, "asr", std::vector<InEdge>{{rs, 0}}, std::move(acfg)));
    auto sink = std::make_unique<TranscriptSink>(
        &sess, "sink", std::vector<InEdge>{{a, 0}}, FlexData::make_object());
    auto* sk = static_cast<TranscriptSink*>(pl.insert_stage(std::move(sink)));
    PipelineRuntime rt(&pl, &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    ASSERT_TRUE(sk->got.size() == 1);
    if (sk->got.size() != 1) { continue; }
    const double w = wer_(words_(text), words_(sk->got[0]));
    std::fprintf(stderr, "[moss-asr] %s: WER %.3f\n  said  '%s'\n  heard "
                 "'%s'\n", lang.c_str(), w, text.c_str(), sk->got[0].c_str());
    EXPECT_TRUE(w <= 0.15);
  }
}
#endif  // VPIPE_BUILD_APPLE_SILICON

// ---- the int8 prefill tier (matrix cores) ------------------------------

// MOSS-TTS-v1.5's backbone prefill through the dynamic-int8 tier, on a
// matrix-core GPU. The tier's floors are M >= 1024 rows and K >= 1024, so
// the prompt here is long-form text built past 1024 rows (the golden's
// 75-row prompt could never reach it; a decode step is one row). Asserts
// ENGAGEMENT before accuracy -- an unwired tier passes every numeric bar --
// then holds the int8 run to the dense run on the logits that predict the
// next rows. Times both prefills. Skips without matrix cores. MEASURED on
// the M5 16 GB (streaming, 10/36 resident): 1140 rows, 142 GEMMs taken,
// rel-L2 0.0074 against the dense run.
TEST(moss_tts_delay, v15_i8_prefill_engages)
{
  const char* model = env_("VPIPE_MOSS_TTS_V15_MODEL");
  if (model == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) {
    std::fprintf(stderr, "[moss-i8] no matrix cores: the int8 tier is an "
                 "M5 path, nothing to check here\n");
    return;
  }
  auto tok = Tokenizer::from_huggingface_json(
      std::string(model) + "/tokenizer.json", &sess);
  ASSERT_TRUE(tok != nullptr);
  if (!tok) { return; }
  const std::string para =
      "Long-form speech synthesis reads a whole chapter at once, so the "
      "prompt the backbone prefills can run to well over a thousand rows. "
      "That is the one place a text-to-speech model meets matrices tall "
      "enough for the int8 tier, because generation itself is one row per "
      "frame. ";
  std::string text;
  MossDelayUserFields f;
  f.language = "English";
  std::vector<std::vector<std::int32_t>> prompt;
  while ((int)prompt.size() < 1100) {
    text += para;
    prompt = moss_delay_build_grid(*tok, moss_tts_normalize_text(text),
                                   MossDelayPromptIds{}, f);
  }
  // A 16 GB box streams the backbone, as the stage would decide; the
  // tier sits in the same GEMM helper either way.
  MossTtsLoadOptions opts;
  opts.stream = model_memory::phys_ram() < ((std::size_t)32 << 30);
  auto m = MetalMossTtsModel::load(model, mc, opts);
  ASSERT_TRUE(m != nullptr);
  if (!m) { return; }
  if (opts.stream) {
    m->set_residency_reserve((std::size_t)512 << 20);
    m->set_residency_schedule(64);
  }
  const auto& c = m->config();
  // Two teacher-forced rows after the prompt: audio_start, then a
  // gen-slot row with codes, so the second set of logits reads K/V the
  // int8 prefill wrote.
  std::vector<std::int32_t> r0(1 + c.n_vq, c.audio_pad_code);
  r0[0] = c.audio_start;
  std::vector<std::int32_t> r1(1 + c.n_vq, c.audio_pad_code);
  r1[0] = c.audio_gen_slot;
  r1[1] = 17;
  const std::vector<std::vector<std::int32_t>> rows = {r0, r1};
  auto run = [&](bool i8, std::vector<float>* out, double* sec) {
    m->set_i8_gemm(i8);
    out->clear();
    const auto t0 = std::chrono::steady_clock::now();
    m->teacher_force_logits(
        prompt, rows, false,
        [&](int, const std::vector<std::vector<float>>& a,
            const std::vector<float>&) {
          for (const auto& cb : a) {
            out->insert(out->end(), cb.begin(), cb.begin() + c.audio_vocab);
          }
        });
    *sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
  };
  // Warm once (pipeline compilation, residency), then measure.
  std::vector<float> warm, dense, int8;
  double tw = 0, td = 0, ti = 0;
  run(false, &warm, &tw);
  run(false, &dense, &td);
  const std::uint64_t n0 = m->i8_gemm_count();
  run(true, &int8, &ti);
  const std::uint64_t took = m->i8_gemm_count() - n0;
  const bool on = m->i8_gemm_enabled();
  double num = 0, den = 0;
  for (std::size_t i = 0; i < dense.size() && i < int8.size(); ++i) {
    const double d = (double)dense[i] - int8[i];
    num += d * d;
    den += (double)dense[i] * dense[i];
  }
  const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
  // Top-1 agreement on the codebooks the dense run is decisive about.
  int decisive = 0, agree = 0;
  for (std::size_t base = 0; base + c.audio_vocab <= dense.size();
       base += c.audio_vocab) {
    int b1 = 0, bi = 0;
    float v1 = -1e30f, v2 = -1e30f, vi = -1e30f;
    for (int k = 0; k < c.audio_vocab; ++k) {
      const float v = dense[base + k];
      if (v > v1) { v2 = v1; v1 = v; b1 = k; } else if (v > v2) { v2 = v; }
      if (int8[base + k] > vi) { vi = int8[base + k]; bi = k; }
    }
    if (v1 - v2 > 1.0f) { ++decisive; agree += bi == b1 ? 1 : 0; }
  }
  std::fprintf(stderr, "[moss-i8] prompt %zu rows | stream %d (%d/%d "
               "resident) | i8 %s, took %llu GEMMs | prefill+2 dense "
               "%.2fs int8 %.2fs | logits rel %.4f, decisive %d/%d\n",
               prompt.size(), (int)opts.stream, m->resident_layer_count(),
               m->n_layers(), on ? "on" : "OFF", (unsigned long long)took,
               td, ti, rel, agree, decisive);
  EXPECT_TRUE(on);
  // Every layer's q|k|v, o, gate|up and down at the prompt's rows (the
  // last layer prunes its MLP to the one row the head reads).
  EXPECT_TRUE(took >= 4u * (std::uint64_t)m->n_layers() - 2u);
  EXPECT_TRUE(rel < 0.05);
  // Few codebooks are decisive at a 1-logit margin over two rows (the
  // inactive ones are flat), so this is a sanity bar; rel-L2 is the measure.
  EXPECT_TRUE(decisive >= 4 && agree >= decisive * 9 / 10);
}

// The three routes one MOSS-TTS-v1.5 prefill GEMM can take on a matrix-core
// GPU, at the backbone's own shapes and a long-form prompt's rows: the
// steel w8 GEMM (what a w8 prefill ran before the matrix-core tier took
// w8), w8 dequant + matmul2d (the dense default now, split-K where the
// model's own MmaSplitK plans one), and the native int8 GEMM (i8_gemm).
// Synthetic operands -- the rates are data-independent.
// Opt-in: VPIPE_MOSS_I8_BENCH=1 (rows from VPIPE_MOSS_I8_BENCH_M, default
// 1140). Interleaved arms, best of three rounds, so a throttling box
// shifts every arm alike.
TEST(moss_tts_delay, i8_shape_rates)
{
  if (env_("VPIPE_MOSS_I8_BENCH") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  const int M = env_("VPIPE_MOSS_I8_BENCH_M")
                    ? std::atoi(env_("VPIPE_MOSS_I8_BENCH_M")) : 1140;
  auto lq = mc->load_library("affine_qmm_steel_bf16");
  auto ld = mc->load_library("affine_dequant_bf16");
  auto lm = mc->load_library("dense_gemm_mma_bf16");
  auto fq = lq.function("affine_qmm_steel_w8g64");
  auto fd = ld.function("affine_dequant_w8g64");
  auto fm = lm.function("dense_gemm_mma_t_n128_f16");
  auto fw = lm.function("dense_gemm_mma_t_n128x256_f16");
  ASSERT_TRUE(fq.valid() && fd.valid() && fm.valid() && fw.valid());
  // The dense route is the model's own: split-K where it plans one (the
  // down projection, K 12288), else one matmul2d at the tile rule's pick.
  auto le = mc->load_library("llm_elementwise_bf16");
  MmaSplitK splitk;
  splitk.load(mc, lm, le);
  I8GemmContext i8(mc, true, /*bf16=*/true);
  ASSERT_TRUE(i8.enabled() && i8.native_available(8));
  struct Shape { const char* what; int N, K; };
  const Shape shapes[] = {{"q|k|v", 6144, 4096}, {"o", 4096, 4096},
                          {"gate|up", 24576, 4096}, {"down", 4096, 12288}};
  double tot[3] = {0, 0, 0};
  for (const Shape& sh : shapes) {
    const int N = sh.N, K = sh.K;
    auto x = mc->make_shared_buffer((std::size_t)M * K * 2);
    auto w = mc->make_shared_buffer((std::size_t)N * K);
    auto sc = mc->make_shared_buffer((std::size_t)N * (K / 64) * 2);
    auto bi = mc->make_shared_buffer((std::size_t)N * (K / 64) * 2);
    auto deq = mc->make_shared_buffer((std::size_t)N * K * 2);
    auto y = mc->make_shared_buffer((std::size_t)M * N * 2);
    ASSERT_TRUE(!x.empty() && !w.empty() && !deq.empty() && !y.empty());
    std::memset(x.contents(), 0x3c, x.byte_size());   // bf16 ~0.0117
    std::memset(w.contents(), 0x5a, w.byte_size());
    std::memset(sc.contents(), 0x3a, sc.byte_size());
    std::memset(bi.contents(), 0xba, bi.byte_size());
    auto steel = [&](ComputeEncoder& enc) {
      enc.set_function(fq);
      enc.set_buffer(0, w); enc.set_buffer(1, sc); enc.set_buffer(2, bi);
      enc.set_buffer(3, x); enc.set_buffer(4, y);
      enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
    };
    auto mma = [&](ComputeEncoder& enc) {
      enc.set_function(fd);
      enc.set_buffer(0, w); enc.set_buffer(1, sc); enc.set_buffer(2, bi);
      enc.set_buffer(3, deq);
      enc.set_constant(4, K); enc.set_constant(5, N);
      enc.dispatch({(unsigned)(K / 4), (unsigned)N, 1}, {64, 1, 1});
      if (splitk.encode(mc, enc, x, deq, y, K, N, M)) { return; }
      const bool deep = mma_use_wide_tile(N, K);
      const int BN = deep ? 256 : 128;
      enc.set_function(deep ? fw : fm);
      enc.set_buffer(0, x); enc.set_buffer(1, deq); enc.set_buffer(2, deq);
      enc.set_buffer(3, y);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    };
    auto int8 = [&](ComputeEncoder& enc) {
      (void)i8.gemm_affine(enc, x, 0, w, sc, bi, 8, y, 0, M, N, K);
    };
    const int reps = 8;
    double best[3] = {1e30, 1e30, 1e30};
    for (int round = 0; round < 3; ++round) {
      for (int a = 0; a < 3; ++a) {
        const int arm = (a + round) % 3;
        const double t = autotune_time(mc, reps, [&](ComputeEncoder& enc) {
          if (arm == 0) { steel(enc); }
          else if (arm == 1) { mma(enc); }
          else { int8(enc); }
        }) / reps;
        best[arm] = std::min(best[arm], t);
      }
    }
    const double flop = 2.0 * M * N * K;
    std::fprintf(stderr, "[moss-i8-rate] %-8s M=%d N=%-6d K=%-6d | steel w8 "
                 "%6.2f ms (%5.1f TF/s) | dense route %6.2f ms (%5.1f) | "
                 "int8 %6.2f ms (%5.1f) | int8 vs dense %.2fx\n", sh.what, M, N,
                 K, best[0] * 1e3, flop / best[0] / 1e12, best[1] * 1e3,
                 flop / best[1] / 1e12, best[2] * 1e3, flop / best[2] / 1e12,
                 best[1] / best[2]);
    for (int a = 0; a < 3; ++a) { tot[a] += best[a]; }
    if (env_("VPIPE_MOSS_I8_BENCH_SPLIT") == nullptr) { continue; }
    // Where the dense route's time goes: the dequant, and each tile.
    double part[3] = {1e30, 1e30, 1e30};
    for (int round = 0; round < 3; ++round) {
      for (int a = 0; a < 3; ++a) {
        const double t = autotune_time(mc, reps, [&](ComputeEncoder& enc) {
          if (a == 0) {
            enc.set_function(fd);
            enc.set_buffer(0, w); enc.set_buffer(1, sc);
            enc.set_buffer(2, bi); enc.set_buffer(3, deq);
            enc.set_constant(4, K); enc.set_constant(5, N);
            enc.dispatch({(unsigned)(K / 4), (unsigned)N, 1}, {64, 1, 1});
            return;
          }
          const int BN = a == 2 ? 256 : 128;
          enc.set_function(a == 2 ? fw : fm);
          enc.set_buffer(0, x); enc.set_buffer(1, deq);
          enc.set_buffer(2, deq); enc.set_buffer(3, y);
          enc.set_constant(4, K); enc.set_constant(5, N);
          enc.set_constant(6, M); enc.set_constant(7, 0);
          enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                        (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
        }) / reps;
        part[a] = std::min(part[a], t);
      }
    }
    std::fprintf(stderr, "[moss-i8-rate]   %-8s dequant %6.2f ms (%5.1f "
                 "GB/s) | mma 128x128 %6.2f ms (%5.1f TF/s) | mma 128x256 "
                 "%6.2f ms (%5.1f TF/s) | single op picks %s%s\n", sh.what,
                 part[0] * 1e3, (double)N * K * 3 / part[0] / 1e9,
                 part[1] * 1e3, flop / part[1] / 1e12, part[2] * 1e3,
                 flop / part[2] / 1e12,
                 mma_use_wide_tile(N, K) ? "128x256" : "128x128",
                 splitk.plan(K, N, M).fn != nullptr ? ", route splits K"
                                                     : "");
  }
  // One layer = one of each; the backbone has 36.
  std::fprintf(stderr, "[moss-i8-rate] per layer: steel %.2f ms, "
               "dense route %.2f ms, int8 %.2f ms -> x36: %.2f / %.2f / "
               "%.2f s\n", tot[0] * 1e3, tot[1] * 1e3, tot[2] * 1e3,
               tot[0] * 36, tot[1] * 36, tot[2] * 36);
}
