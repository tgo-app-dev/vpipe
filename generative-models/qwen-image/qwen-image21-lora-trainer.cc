#include "generative-models/qwen-image/qwen-image21-lora-trainer.h"

#include "common/flex-data.h"
#include "generative-models/gen-input.h"
#include "generative-models/krea2/flow-sampler.h"
#include "generative-models/weight-set.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>

namespace vpipe {
namespace genai {

using metal_compute::SharedBuffer;

namespace {

std::uint16_t
bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

float
f32_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// The inference schedule's constants, which generate-image hardcodes for
// this family: FlowMatchEulerDiscreteScheduler with an exponential
// dynamic shift over 256..8192 image tokens and a terminal stretch.
constexpr double kBaseShift = 0.5;
constexpr double kMaxShift = 0.9;
constexpr int kBaseSeq = 256;
constexpr int kMaxSeq = 8192;
constexpr double kShiftTerminal = 0.02;

}  // namespace

bool
QwenImage21LoraTrainer::claims(const std::string& transformer_dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(transformer_dir) / "config.json");
  if (!in) { return false; }
  FlexData fd;
  try {
    fd = FlexData::from_json(in);
  } catch (...) {
    return false;
  }
  if (!fd.is_object()) { return false; }
  auto o = fd.as_object();
  return o.contains("_class_name") &&
         std::string(o.at("_class_name").as_string("")) ==
             "QwenImage21Transformer2DModel";
}

std::unique_ptr<QwenImage21LoraTrainer>
QwenImage21LoraTrainer::load(const LoadArgs& a, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<QwenImage21LoraTrainer>();
  };
  if (a.ws == nullptr || a.mc == nullptr) {
    return fail("qwen-image-2.1 trainer: no weights or metal context");
  }
  std::unique_ptr<QwenImage21LoraTrainer> t(new QwenImage21LoraTrainer());
  t->_mc = a.mc;
  t->_m = MetalQwenImage21Transformer::load(a.ws, a.mc, a.cfg,
                                            a.stream_blocks);
  if (t->_m == nullptr) {
    return fail("qwen-image-2.1 trainer: the DiT did not load");
  }
  std::string e;
  t->_t = QwenImage21Trainer::create(t->_m.get(), &e);
  if (t->_t == nullptr) { return fail(e); }
  return t;
}

std::vector<train::ModuleShape>
QwenImage21LoraTrainer::modules(const std::string& targets) const
{
  return _t->modules(targets);
}

int
QwenImage21LoraTrainer::bind(train::LoraParams* params, std::string* err)
{
  return _t->bind(params, err);
}

int
QwenImage21LoraTrainer::latent_channels() const noexcept
{
  return _m->config().in_channels;
}

SharedBuffer
QwenImage21LoraTrainer::pack_latent(const float* chw, int c, int h, int w,
                                    std::string* err)
{
  const int IC = _m->config().in_channels;
  if (c != IC || (h % 2) != 0 || (w % 2) != 0) {
    if (err != nullptr) {
      *err = "qwen-image-2.1 trainer: a latent must be [" +
             std::to_string(IC) + ", h, w] with an EVEN grid (pictures in "
             "multiples of 32); got [" + std::to_string(c) + ", " +
             std::to_string(h) + ", " + std::to_string(w) + "]";
    }
    return {};
  }
  SharedBuffer b = _mc->make_shared_buffer((std::size_t)h * w * IC * 2);
  if (b.empty()) { return b; }
  // Channel-first [c, h, w] -> packed [h * w, c], the layout the DiT reads.
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int k = 0; k < IC; ++k) {
        d[((std::size_t)y * w + x) * IC + k] =
            bf16_(chw[((std::size_t)k * h + y) * w + x]);
      }
    }
  }
  return b;
}

bool
QwenImage21LoraTrainer::layout_(const FlexData* sideband, int text_rows,
                                int gh, int gw, qi21::Layout* lay,
                                std::string* err) const
{
  std::vector<std::uint8_t> slot;
  if (sideband != nullptr && sideband->is_object()) {
    FlexData sb = *sideband;
    auto o = sb.as_object();
    if (o.contains(cond_sideband::kImgSlots)) {
      const FlexData f = o.at(cond_sideband::kImgSlots);
      if (f.is_array()) {
        auto a = f.as_array();
        for (std::size_t i = 0; i < a.size(); ++i) {
          slot.push_back(a.at(i).as_int(0) != 0 ? 1 : 0);
        }
      }
    }
  }
  if (slot.empty()) { slot.assign((std::size_t)text_rows, 0); }
  if ((int)slot.size() != text_rows) {
    if (err != nullptr) {
      *err = "qwen-image-2.1 trainer: the conditioning's slot mask covers " +
             std::to_string(slot.size()) + " rows of " +
             std::to_string(text_rows) +
             " -- wire a qwen-image-21 diffusion-conditioner";
    }
    return false;
  }
  for (std::uint8_t v : slot) {
    if (v != 0) {
      if (err != nullptr) {
        *err = "qwen-image-2.1 trainer: a caption grounded on reference "
               "pictures is edit training, which is not supported yet";
      }
      return false;
    }
  }
  slot.insert(slot.end(), (std::size_t)(gh * gw / 4), 1);
  std::string lerr;
  if (!qi21::build_layout(slot, {}, {qi21::ImgBlock{1, gh, gw}}, lay,
                          &lerr)) {
    if (err != nullptr) { *err = "qwen-image-2.1 trainer: " + lerr; }
    return false;
  }
  return true;
}

bool
QwenImage21LoraTrainer::step(const train::TrainSample& s,
                             train::StepStats* out, std::string* err)
{
  qi21::Layout lay;
  if (!layout_(s.sideband, s.text_rows, s.grid_h, s.grid_w, &lay, err)) {
    return false;
  }
  QwenImage21Trainer::Sample q;
  q.x0 = s.x0;
  q.noise = s.noise;
  q.txt = s.text;
  q.layout = &lay;
  q.sigma = s.sigma;
  q.weight = s.weight;
  QwenImage21Trainer::Result r;
  if (!_t->step(q, &r, err)) { return false; }
  if (out != nullptr) {
    out->loss = r.loss;
    out->forward_ms = r.forward_ms;
    out->backward_ms = r.backward_ms;
  }
  return true;
}

bool
QwenImage21LoraTrainer::eval_loss(const train::TrainSample& s, double* loss,
                                  std::string* err)
{
  qi21::Layout lay;
  if (!layout_(s.sideband, s.text_rows, s.grid_h, s.grid_w, &lay, err)) {
    return false;
  }
  QwenImage21Trainer::Sample q;
  q.x0 = s.x0;
  q.noise = s.noise;
  q.txt = s.text;
  q.layout = &lay;
  q.sigma = s.sigma;
  q.weight = s.weight;
  return _t->loss(q, loss, err);
}

double
QwenImage21LoraTrainer::schedule_mu(int grid_h, int grid_w) const noexcept
{
  // diffusers' calculate_shift: a line through (base_seq, base_shift) and
  // (max_seq, max_shift), evaluated at the image's token count.
  const double m = (kMaxShift - kBaseShift) / (double)(kMaxSeq - kBaseSeq);
  const double b = kBaseShift - m * kBaseSeq;
  return (double)grid_h * grid_w * m + b;
}

bool
QwenImage21LoraTrainer::preview(const train::PreviewRequest& r,
                                std::vector<float>* chw, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (r.text == nullptr) { return fail("preview: no conditioning"); }
  const int IC = _m->config().in_channels;
  const int lh = r.grid_h, lw = r.grid_w, img_seq = lh * lw;
  qi21::Layout lay_p, lay_n;
  if (!layout_(r.sideband, r.text_rows, lh, lw, &lay_p, err)) { return false; }
  const bool cfg = r.neg_text != nullptr && r.neg_rows > 0 && r.cfg != 1.0f;
  if (cfg &&
      !layout_(r.neg_sideband, r.neg_rows, lh, lw, &lay_n, err)) {
    return false;
  }
  FlowSchedulerSpec sched;
  sched.dynamic_shift = true;
  sched.shift_type = "exponential";
  sched.base_shift = kBaseShift; sched.max_shift = kMaxShift;
  sched.base_seq = kBaseSeq; sched.max_seq = kMaxSeq;
  sched.shift_terminal = kShiftTerminal;
  sched.steps = r.steps;
  sched.img_seq_len = img_seq;
  FlowSampler sampler(FlowSamplerSpec{}, sched);
  std::vector<float> packed((std::size_t)img_seq * IC);
  {
    std::mt19937_64 rng(r.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : packed) { v = nd(rng); }
  }
  metal_compute::MetalCompute* mc = _mc;
  SharedBuffer lat = mc->make_shared_buffer(packed.size() * 2);
  if (lat.empty()) { return fail("preview: allocation failed"); }
  MetalQwenImage21Transformer::KvCache kv_p, kv_n;
  bool ok = true;
  int step_i = 0;
  auto run = [&](const SharedBuffer& txt, const qi21::Layout& lay,
                 MetalQwenImage21Transformer::KvCache* kv, double sigma) {
    MetalQwenImage21Transformer::Request q;
    q.latents = &lat;
    q.txt = &txt;
    q.layout = &lay;
    q.timestep = (float)sigma;
    if (lay.prefix_len > 0) {
      q.kv = kv;
      q.kv_mode = step_i == 0 ? MetalQwenImage21Transformer::KvMode::kFill
                              : MetalQwenImage21Transformer::KvMode::kCached;
    }
    std::string e;
    SharedBuffer v = _m->forward(q, &e);
    if (v.empty()) {
      ok = false;
      if (err != nullptr) { *err = e; }
    }
    return v;
  };
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    auto* lb = static_cast<std::uint16_t*>(lat.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) { lb[k] = bf16_(cand[k]); }
    SharedBuffer v = run(*r.text, lay_p, &kv_p, sigma);
    if (!ok) { return {}; }
    std::vector<float> o(cand.size());
    const auto* vp = static_cast<const std::uint16_t*>(v.contents());
    for (std::size_t k = 0; k < o.size(); ++k) { o[k] = f32_(vp[k]); }
    if (cfg) {
      SharedBuffer nv = run(*r.neg_text, lay_n, &kv_n, sigma);
      if (!ok) { return {}; }
      const auto* np = static_cast<const std::uint16_t*>(nv.contents());
      for (std::size_t k = 0; k < o.size(); ++k) {
        const float n = f32_(np[k]);
        o[k] = n + r.cfg * (o[k] - n);
      }
    }
    return o;
  };
  for (int i = 0; i < sampler.steps(); ++i) {
    if (_stop && _stop()) {
      if (err != nullptr) { err->clear(); }
      return false;
    }
    step_i = i;
    sampler.step(i, packed, denoise);
    if (!ok) { return false; }
  }
  chw->assign((std::size_t)IC * img_seq, 0.0f);
  for (int t = 0; t < img_seq; ++t) {
    const int y = t / lw, x = t % lw;
    for (int c = 0; c < IC; ++c) {
      (*chw)[((std::size_t)c * lh + y) * lw + x] =
          packed[(std::size_t)t * IC + c];
    }
  }
  return true;
}

void
QwenImage21LoraTrainer::set_stop(std::function<bool()> stop)
{
  _stop = stop;
  _t->set_stop(stop);
  _m->set_stream_stop(std::move(stop));
}

void
QwenImage21LoraTrainer::set_options(const FlexData& o)
{
  if (!o.is_object()) { return; }
  FlexData c = o;
  auto ob = c.as_object();
  if (ob.contains(train::opt::kKeepAll)) {
    _t->set_keep_all(ob.at(train::opt::kKeepAll).as_bool(false));
  }
  if (ob.contains(train::opt::kAneBackward)) {
    _t->set_ane_backward(ob.at(train::opt::kAneBackward).as_bool(false));
  }
  if (_m->streaming() && ob.contains(train::opt::kResidencyReserve)) {
    _m->set_residency_reserve(
        (std::size_t)ob.at(train::opt::kResidencyReserve).as_int(0));
  }
  if (_m->streaming() && ob.contains(train::opt::kResidencySteps)) {
    _m->set_residency_schedule(
        (int)ob.at(train::opt::kResidencySteps).as_int(1));
  }
}

std::size_t
QwenImage21LoraTrainer::arena_bytes(int grid_h, int grid_w, int text_rows,
                                    int rank, bool keep_all) const
{
  return QwenImage21Trainer::arena_bytes(_m->config(),
                                         text_rows + grid_h * grid_w, rank,
                                         keep_all);
}

std::uint64_t
QwenImage21LoraTrainer::resident_bytes() const
{
  return _m->resident_bytes();
}

}  // namespace genai
}  // namespace vpipe
