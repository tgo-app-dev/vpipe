// VOSR 2.0: the checkpoint reader, the packaging, and the two stage
// keys the graph shape depends on.
//
// WHAT IS WORTH PINNING HERE is the REFUSALS. Every VOSR variant --
// multi-step, 0.5B, the SD2 autoencoder, an auxiliary time condition --
// ships under the same tensor names as the one this tree implements, so
// a reader that waves them through loads cleanly, spends a minute and
// emits a plausible picture of nothing. There is no error to catch
// downstream, which is what makes the accept/refuse matrix the test.
//
// The forward itself is not tested here: it needs 5.6 GB of weights and
// a GPU. What that leaves testable without either is the config, the
// packaging and the stage surface, and those are what this covers.

#include "generative-models/vosr/metal-dinov2-encoder.h"
#include "generative-models/vosr/metal-vosr-transformer.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/generate-image-stage.h"
#include "stages/model-catalog.h"
#include "pipeline/stage-registry.h"
#include "common/session.h"
#include "minitest.h"

#include <unistd.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
namespace fs = std::filesystem;

namespace {

// The shipped VOSR2/args.json, verbatim but for whatever a test
// overrides. Written to a temp dir because read_config walks UP from the
// directory it is given, which is itself a property worth exercising:
// the weights live in `checkpoints/` and the config does not.
struct Ckpt {
  fs::path root;
  explicit Ckpt(const std::string& tag)
  {
    root = fs::temp_directory_path() /
           ("vpipe-vosr-" + tag + "-" + std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root / "checkpoints");
  }
  ~Ckpt() { std::error_code ec; fs::remove_all(root, ec); }

  void args(const std::string& body) const
  {
    std::ofstream o(root / "args.json");
    o << body;
  }
  // A file that makes `checkpoints/` look like it holds weights.
  void weights() const
  {
    std::ofstream o(root / "checkpoints" / "ema_model.safetensors");
    o << "not really";
  }
};

const char* kShipped = R"({
  "resolution": 512, "patch_size": 2, "mlp_ratio": 4,
  "use_qknorm": true, "use_swiglu": true, "use_rope": true,
  "use_rmsnorm": true, "wo_shift": false,
  "dim": 1536, "depth": 36, "num_heads": 24,
  "ae_type": "qwen", "dinov2_size": 448, "enc_type": "dinov2l",
  "enc_dim": 1024, "layer_dinov2b_list": [17], "encdim_ratio": 3,
  "auxiliary_time_cond": false, "distill_type": "onestep"
})";

// The same repo's VOSR 1 checkpoints at 1.4B, their args.json VERBATIM
// (CSWRY/VOSR, VOSR_1.4B_ms/ and VOSR_1.4B_os/). Their tensors are
// VOSR 2.0's shapes -- the multi-step file is even the same size -- so
// the config is all that tells them apart.
const char* kVosr1Ms = R"({
    "resolution": 512,
    "patch_size": 2,
    "mlp_ratio": 4,
    "use_qknorm": true,
    "use_swiglu": true,
    "use_rope": true,
    "use_rmsnorm": true,
    "wo_shift": false,
    "dim": 1536,
    "depth": 36,
    "num_heads": 24,
    "ae_type": "qwen",
    "ae_path": "preset/ckpts/Qwen-Image-vae-2d",
    "time_dist": ["uniform", 0, 1.0],
    "cfg_ratio": 0.1,
    "cfg_scale": 2.0,
    "dinov2_size": 448,
    "enc_type": "dinov2l",
    "enc_dim": 1024,
    "layer_dinov2b_list": [17],
    "interp_type": "lin",
    "encdim_ratio": 3,
    "weak_cond_strength_aelq_list": [0.05, 0.25],
    "cond_strength_aelq_list": [5, 1]
})";

const char* kVosr1Os = R"({
    "resolution": 512,
    "patch_size": 2,
    "mlp_ratio": 4,
    "use_qknorm": true,
    "use_swiglu": true,
    "use_rope": true,
    "use_rmsnorm": true,
    "wo_shift": false,
    "dim": 1536,
    "depth": 36,
    "num_heads": 24,
    "ae_type": "qwen",
    "ae_path": "preset/ckpts/Qwen-Image-vae-2d",
    "time_dist": ["uniform", 0, 1.0],
    "cfg_ratio": 0.1,
    "cfg_scale": 0.5,
    "dinov2_size": 448,
    "enc_type": "dinov2l",
    "enc_dim": 1024,
    "layer_dinov2b_list": [17],
    "interp_type": "lin",
    "encdim_ratio": 3,
    "weak_cond_strength_aelq_list": [0.05, 0.25],
    "weak_cond_strength_aelq": 0.1,
    "cond_strength_aelq_list": [5, 1],
    "t_start": 0,
    "t_end": 1,
    "distill_type": "rcgm"
})";

bool
has_key_(const StageSpec& sp, const char* key)
{
  for (const ConfigKey& k : sp.attrs) {
    if (k.key == key) { return true; }
  }
  return false;
}

const ConfigKey*
key_(const StageSpec& sp, const char* key)
{
  for (const ConfigKey& k : sp.attrs) {
    if (k.key == key) { return &k; }
  }
  return nullptr;
}

}  // namespace

// ---- the reader ------------------------------------------------------

TEST(vosr, the_shipped_config_reads)
{
  Ckpt c("shipped");
  c.args(kShipped);
  genai::MetalVosrTransformer::Config cfg;
  std::string why;
  ASSERT_TRUE(genai::MetalVosrTransformer::read_config(c.root.string(), &cfg,
                                                       &why));
  EXPECT_TRUE(why.empty());
  if (!why.empty()) { return; }
  EXPECT_TRUE(cfg.dim == 1536);
  EXPECT_TRUE(cfg.depth == 36);
  EXPECT_TRUE(cfg.n_heads == 24);
  // Derived, not stated: head_dim, the SwiGLU width and the cross-
  // attention projection width are all computed from the config, and a
  // wrong derivation is a wrong tensor shape at load.
  EXPECT_TRUE(cfg.head_dim == 64);
  EXPECT_TRUE(cfg.ffn == 4096);      // int(2/3 * 4 * 1536)
  EXPECT_TRUE(cfg.enc_mlp == 4608);  // dim * encdim_ratio
  // resolution 512 / VAE 8 / patch 2 -- the grid the RoPE trained on, and
  // the denominator every other grid is rescaled into.
  EXPECT_TRUE(cfg.train_grid == 32);
  EXPECT_TRUE(cfg.enc_dim == 1024);
  EXPECT_TRUE(cfg.enc_layer == 17);
  EXPECT_TRUE(cfg.dinov2_size == 448);
}

TEST(vosr, a_picture_past_the_trained_grid_tiles_by_default)
{
  // The rule that decides it, pinned here because the decision is made
  // in a stage that needs a 1.4B checkpoint to reach.
  //
  // WHY THE TRAINED GRID and not "as large as fits": the reference's
  // one-step sampler applies a rotary table built once for that grid,
  // so it cannot run any other token count at all, and its tiling
  // exists to put every tile back on it. Running a bigger grid through
  // a rescaled table puts every other token on a position the weights
  // never saw -- a periodic weave, reported from the field on faces at
  // a 1024 output and absent both at 512 and tiled.
  genai::MetalVosrTransformer::Config cfg;   // 32 x patch 2 = 64 cells
  EXPECT_TRUE(cfg.train_grid == 32);
  EXPECT_TRUE(cfg.patch == 2);

  // 512 output pixels is the trained grid exactly: one pass, no seams.
  EXPECT_TRUE(genai::MetalVosrTransformer::default_tile(64, 64, cfg) == 0);
  EXPECT_TRUE(genai::MetalVosrTransformer::default_tile(32, 48, cfg) == 0);
  // Past it on either axis, tile AT the trained grid.
  EXPECT_TRUE(genai::MetalVosrTransformer::default_tile(128, 128, cfg) == 64);
  EXPECT_TRUE(genai::MetalVosrTransformer::default_tile(65, 64, cfg) == 64);
  EXPECT_TRUE(genai::MetalVosrTransformer::default_tile(64, 96, cfg) == 64);
  EXPECT_TRUE(genai::MetalVosrTransformer::default_tile(512, 384, cfg) == 64);

  // A checkpoint whose grid did not read stays out of the way rather
  // than dividing by it.
  genai::MetalVosrTransformer::Config bad;
  bad.train_grid = 0;
  EXPECT_TRUE(genai::MetalVosrTransformer::default_tile(128, 128, bad) == 0);
}

TEST(vosr, the_config_is_found_from_the_weights_directory)
{
  // args.json sits at the checkpoint ROOT and the weights a level below,
  // so a reader that only looks where it was pointed finds nothing.
  Ckpt c("walkup");
  c.args(kShipped);
  genai::MetalVosrTransformer::Config cfg;
  EXPECT_TRUE(genai::MetalVosrTransformer::read_config(
      (c.root / "checkpoints").string(), &cfg, nullptr));
}

// Each of these is a real published VOSR checkpoint that would load
// through the same tensor names and compute the wrong answer.
TEST(vosr, the_variants_this_tree_does_not_implement_are_refused)
{
  struct Case { const char* tag; const char* key; const char* val; };
  const Case cases[] = {
      {"multistep", "\"distill_type\"", "\"multistep\""},
      {"sd2",       "\"ae_type\"",      "\"sd2\""},
      {"auxtime",   "\"auxiliary_time_cond\"", "true"},
      {"woshift",   "\"wo_shift\"",     "true"},
      {"norope",    "\"use_rope\"",     "false"},
      {"normsnorm", "\"use_rmsnorm\"",  "false"},
      {"noswiglu",  "\"use_swiglu\"",   "false"},
      {"noqknorm",  "\"use_qknorm\"",   "false"},
  };
  for (const Case& cs : cases) {
    Ckpt c(cs.tag);
    std::string body(kShipped);
    // Overwrite the shipped value by appending a later duplicate is not
    // safe in JSON, so patch the key in place.
    const std::string k(cs.key);
    const std::size_t at = body.find(k);
    ASSERT_TRUE(at != std::string::npos);
    if (at == std::string::npos) { continue; }
    const std::size_t colon = body.find(':', at);
    std::size_t end = body.find_first_of(",}", colon);
    body = body.substr(0, colon + 1) + " " + cs.val + body.substr(end);
    c.args(body);
    genai::MetalVosrTransformer::Config cfg;
    std::string why;
    const bool ok =
        genai::MetalVosrTransformer::read_config(c.root.string(), &cfg, &why);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(why.empty());
  }
}

// The published VOSR 1 checkpoints that share VOSR 2.0's shapes, as
// published: the multi-step one says nothing about distillation, and
// reading that silence as "one step" ran it as VOSR 2.0.
TEST(vosr, the_release_s_vosr_1_checkpoints_are_refused)
{
  struct Case { const char* tag; const char* body; const char* names; };
  const Case cases[] = {
      {"v1ms", kVosr1Ms, "multi-step"},
      {"v1os", kVosr1Os, "rcgm"},
  };
  for (const Case& cs : cases) {
    Ckpt c(cs.tag);
    c.args(cs.body);
    genai::MetalVosrTransformer::Config cfg;
    std::string why;
    const bool ok =
        genai::MetalVosrTransformer::read_config(c.root.string(), &cfg, &why);
    EXPECT_FALSE(ok);
    // Refused for the RIGHT reason, said in words a user can act on.
    EXPECT_TRUE(why.find(cs.names) != std::string::npos);
    if (why.find(cs.names) == std::string::npos) {
      std::fprintf(stderr, "[vosr] %s refused as: %s\n", cs.tag,
                   why.c_str());
    }
  }
}

TEST(vosr, a_directory_with_no_args_json_is_not_a_vosr_checkpoint)
{
  Ckpt c("bare");
  c.weights();
  genai::MetalVosrTransformer::Config cfg;
  std::string why;
  EXPECT_FALSE(genai::MetalVosrTransformer::read_config(c.root.string(), &cfg,
                                                        &why));
  EXPECT_FALSE(why.empty());
}

TEST(vosr, the_weights_directory_resolves)
{
  Ckpt c("weights");
  c.args(kShipped);
  EXPECT_TRUE(genai::MetalVosrTransformer::weights_dir(c.root.string())
                  .empty());   // nothing to find yet
  c.weights();
  const std::string w =
      genai::MetalVosrTransformer::weights_dir(c.root.string());
  EXPECT_TRUE(w == (c.root / "checkpoints").string());
}

// The layout model-fetch actually produces: the release publishes
// several models from one repo, so the fetched root holds `VOSR2/`
// beside `Qwen-Image-vae-2d/` and neither the config nor the weights are
// where a checkout puts them. This is the shape every shipped pipeline
// points at, so it is the one that has to resolve.
TEST(vosr, a_fetched_repo_root_resolves)
{
  const fs::path root = fs::temp_directory_path() /
                        ("vpipe-vosr-fetched-" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "VOSR2" / "checkpoints");
  fs::create_directories(root / "Qwen-Image-vae-2d");
  { std::ofstream o(root / "VOSR2" / "args.json"); o << kShipped; }
  { std::ofstream o(root / "VOSR2" / "checkpoints" / "ema_model.safetensors");
    o << "x"; }

  genai::MetalVosrTransformer::Config cfg;
  std::string why;
  EXPECT_TRUE(genai::MetalVosrTransformer::read_config(root.string(), &cfg,
                                                       &why));
  EXPECT_TRUE(genai::MetalVosrTransformer::weights_dir(root.string()) ==
              (root / "VOSR2" / "checkpoints").string());
  // The FILE, not the directory: `ema_model.safetensors` is not a name
  // the checkpoint opener globs for, so a directory here opens nothing
  // and the stage goes inert with no tensor to name.
  EXPECT_TRUE(genai::MetalVosrTransformer::weights_path(root.string()) ==
              (root / "VOSR2" / "checkpoints" / "ema_model.safetensors")
                  .string());
  std::error_code ec;
  fs::remove_all(root, ec);
}

// ---- the vision tower ------------------------------------------------

TEST(vosr, the_dinov2_config_reads)
{
  const fs::path d = fs::temp_directory_path() /
                     ("vpipe-dinov2-" + std::to_string(::getpid()));
  fs::remove_all(d);
  fs::create_directories(d);
  {
    std::ofstream o(d / "config.json");
    o << R"({"hidden_size": 1024, "num_hidden_layers": 24,
             "num_attention_heads": 16, "patch_size": 14, "mlp_ratio": 4,
             "image_size": 518, "layer_norm_eps": 1e-06,
             "use_swiglu_ffn": false})";
  }
  genai::MetalDinov2Encoder::Config cfg;
  ASSERT_TRUE(genai::MetalDinov2Encoder::read_config(d.string(), &cfg));
  EXPECT_TRUE(cfg.hidden == 1024);
  
  EXPECT_TRUE(cfg.depth == 24);
  EXPECT_TRUE(cfg.head_dim == 64);
  EXPECT_TRUE(cfg.ffn == 4096);
  // 518 / 14 -- the position grid, which is resampled to whatever grid
  // the request implies and is nothing like it (37 vs 32 at 448px).
  EXPECT_TRUE(cfg.pos_grid == 37);
  // 448 / 14 = 32 tokens a side.
  {
    genai::MetalDinov2Encoder::Config c2 = cfg;
    (void)c2;
  }

  // The giant's SwiGLU MLP is a different block and this tower does not
  // build one, so it is refused rather than mis-loaded.
  {
    std::ofstream o(d / "config.json");
    o << R"({"hidden_size": 1536, "num_hidden_layers": 40,
             "num_attention_heads": 24, "patch_size": 14, "mlp_ratio": 4,
             "use_swiglu_ffn": true})";
  }
  genai::MetalDinov2Encoder::Config g;
  EXPECT_FALSE(genai::MetalDinov2Encoder::read_config(d.string(), &g));
  std::error_code ec;
  fs::remove_all(d, ec);
}

// ---- packaging -------------------------------------------------------

TEST(vosr, the_catalogue_lists_the_restorer_and_its_tower)
{
  bool restorer = false, tower = false;
  for (const ModelCatalogEntry& e : model_catalog()) {
    if (e.model_type == "vosr") {
      restorer = true;
      // The VAE travels WITH the restorer: VOSR ships an image-only
      // extraction of the Qwen-Image autoencoder, and a fetch that
      // brought only the DiT would leave the graph unable to encode or
      // decode anything.
      bool has_vae = false, has_dit = false;
      for (const std::string& f : e.files) {
        if (f.find("Qwen-Image-vae-2d") != std::string::npos) {
          has_vae = true;
        }
        if (f.find("ema_model.safetensors") != std::string::npos) {
          has_dit = true;
        }
      }
      EXPECT_TRUE(has_vae);
      EXPECT_TRUE(has_dit);
    }
    if (e.model_type == "dinov2") {
      tower = true;
      // A SUPPLEMENT, so the model browser offers it under the restorer
      // rather than as something to generate with.
      EXPECT_TRUE(e.parent_model_type == "vosr");
      EXPECT_TRUE(catalog_category(e) == "supplement");
    }
  }
  EXPECT_TRUE(restorer);
  EXPECT_TRUE(tower);
}

// ---- the stage surface -----------------------------------------------

TEST(vosr, the_restorer_keys_are_on_generate_image)
{
  const StageSpec* spp = StageRegistry::get().spec("generate-image");
  ASSERT_TRUE(spp != nullptr);
  if (spp == nullptr) { return; }
  const StageSpec& sp = *spp;
  EXPECT_TRUE(has_key_(sp, "reference_mode"));
  const ConfigKey* rm = key_(sp, "reference_mode");
  ASSERT_TRUE(rm != nullptr);
  if (rm == nullptr) { return; }
  EXPECT_TRUE(rm->type == ConfigType::String);
  // `auto` and not `latch`: an edit graph gets the old behaviour from
  // the family it names, and a restoration graph does not have to know
  // this key exists.
  EXPECT_TRUE(rm->def_str == "auto");
  // ...and the TILING is not here any more. It is one family's knob on a
  // stage that serves seven, which is what vosr-model-config is for, and
  // an unknown key is reported nowhere -- so a key left behind here
  // would read as configured and do nothing.
  EXPECT_FALSE(has_key_(sp, "tile_size"));
  EXPECT_FALSE(has_key_(sp, "tile_overlap"));
}

TEST(vosr, the_tiling_keys_are_on_the_vosr_config_source)
{
  const StageSpec* spp = StageRegistry::get().spec("vosr-model-config");
  ASSERT_TRUE(spp != nullptr);
  if (spp == nullptr) { return; }
  const StageSpec& sp = *spp;
  EXPECT_TRUE(sp.category == StageCategory::ModelSpecificConfig);
  const ConfigKey* ts = key_(sp, "tile_size");
  const ConfigKey* to = key_(sp, "tile_overlap");
  ASSERT_TRUE(ts != nullptr);
  ASSERT_TRUE(to != nullptr);
  if (ts == nullptr || to == nullptr) { return; }
  EXPECT_TRUE(ts->type == ConfigType::Int);
  EXPECT_TRUE(to->type == ConfigType::Int);
  EXPECT_TRUE(to->def_int == 32);
  // Neither is required: UNSET is the answer a graph should get without
  // asking, which is to tile at the grid the weights were distilled at.
  EXPECT_FALSE(ts->required);
  EXPECT_FALSE(to->required);
  // The trigger iport is the contract every config source shares, and
  // the oport carries the tag a model_config iport accepts.
  ASSERT_TRUE(sp.iports.size() == 1);
  ASSERT_TRUE(sp.oports.size() == 1);
  if (sp.oports.empty()) { return; }
  EXPECT_TRUE(std::string(sp.oports[0].name) == "model_config");
  EXPECT_TRUE(std::string(sp.oports[0].tags) == "model-config");
}

TEST(vosr, the_tower_keys_are_on_the_conditioner)
{
  const StageSpec* spp =
      StageRegistry::get().spec("diffusion-conditioner");
  ASSERT_TRUE(spp != nullptr);
  if (spp == nullptr) { return; }
  const StageSpec& sp = *spp;
  EXPECT_TRUE(has_key_(sp, "encoder_dir"));
  EXPECT_TRUE(has_key_(sp, "reference_mode"));
  const ConfigKey* ed = key_(sp, "encoder_dir");
  ASSERT_TRUE(ed != nullptr);
  if (ed == nullptr) { return; }
  // Filtered to the tower, and NOT on the diffusion-model channel: a
  // model-select source feeding the graph its restorer must not also
  // overwrite the encoder field.
  EXPECT_TRUE(ed->suggest_db_type == "dinov2");
  EXPECT_TRUE(ed->model_channel.empty());
}

TEST(vosr, every_stage_that_touches_a_vosr_checkpoint_offers_it)
{
  // The four stages a restoration graph wires. A picker that filters one
  // of them out leaves a field the composer cannot fill.
  for (const char* st : {"generate-image", "diffusion-conditioner",
                         "vae-encode", "vae-decode"}) {
    const StageSpec* sp = StageRegistry::get().spec(st);
    ASSERT_TRUE(sp != nullptr);
    if (sp == nullptr) { continue; }
    const ConfigKey* hf = key_(*sp, "hf_dir");
    ASSERT_TRUE(hf != nullptr);
    if (hf == nullptr) { continue; }
    ASSERT_TRUE(!hf->suggest_db_type.empty());
    EXPECT_TRUE(hf->suggest_db_type.find("vosr") != std::string_view::npos);
  }
}

// ---------------------------------------------------------------------
// THE ACCELERATIONS, AGAINST THIS TREE'S OWN ALU BASELINE.
//
// vosr_reference next door checks the PORT against the published
// implementation and needs a dumped golden. This checks something else
// and needs only the weights: that turning on the matrix-core kernels,
// the int8 GEMMs, the int8 QK and the routed attention does not change
// what the model computes beyond what each of them is entitled to.
//
// The baseline is the SAME BUILD with VPIPE_VOSR_NO_MMA2 and
// VPIPE_VOSR_NO_ATTN_NAX set, so what is isolated is the acceleration
// and not the port. Noise is supplied rather than seeded -- two arms
// cannot agree on a pseudo-random field, and the whole comparison rests
// on holding it fixed.
//
// Env: VPIPE_VOSR_TEST_MODEL_PATH (the fetched CSWRY/VOSR root).
namespace {

using vpipe::genai::MetalVosrTransformer;

std::vector<float> ramp_f32_(std::size_t n, float k, std::uint32_t seed)
{
  std::vector<float> v(n);
  std::uint32_t s = seed;
  for (std::size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = k * ((float)((s >> 9) & 0x7fff) / 16384.0f - 1.0f);
  }
  return v;
}

}  // namespace

TEST(vosr_accel, the_accelerations_agree_with_the_alu_baseline)
{
  const char* root = std::getenv("VPIPE_VOSR_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  vpipe::Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  MetalVosrTransformer::Config base;
  std::string why;
  if (!MetalVosrTransformer::read_config(root, &base, &why)) {
    std::printf("[vosr_accel] config: %s\n", why.c_str());
    return;
  }
  const std::string wdir = MetalVosrTransformer::weights_path(root);
  if (wdir.empty()) { return; }

  // A 64x64 latent -> 32x32 patch grid -> 1024 self-attention tokens,
  // which is long enough that Sol has blocks to choose between.
  // TWO GRIDS. Attention is O(T^2) where the block GEMMs are O(T), so a
  // small grid under-represents exactly the two accelerations that only
  // touch attention -- at 32x32 patches the whole self-attention is a
  // few percent of the forward and sage/sol measure as noise. 64x64 is
  // the 4x-upscale regime the restorer is actually for.
  const int grid = std::getenv("VPIPE_VOSR_ACCEL_BIG") != nullptr ? 128 : 64;
  const int lh = grid, lw = grid;
  const int C = base.latent_ch, E = base.enc_dim;
  const int cond_rows = 1024;
  std::printf("[vosr_accel] latent %dx%d -> %d patch tokens\n", lh, lw,
              (lh / base.patch) * (lw / base.patch));
  const std::vector<float> lq =
      ramp_f32_((std::size_t)C * lh * lw, 1.0f, 0x51a6eu);
  const std::vector<float> noise =
      ramp_f32_((std::size_t)C * lh * lw, 1.0f, 0xc0ffeeu);
  const std::vector<float> condf =
      ramp_f32_((std::size_t)cond_rows * E, 0.5f, 0x9a5eu);
  std::vector<std::uint16_t> cond((std::size_t)cond_rows * E);
  for (std::size_t i = 0; i < cond.size(); ++i) {
    std::uint32_t u;
    std::memcpy(&u, &condf[i], 4);
    cond[i] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }

  double last_ms = 0.0;
  auto run = [&](const MetalVosrTransformer::Config& cfg,
                 std::vector<float>* out) {
    std::string err;
    auto m = MetalVosrTransformer::load(wdir, mc, cfg, &err);
    if (m == nullptr) {
      std::printf("[vosr_accel] load: %s\n", err.c_str());
      return false;
    }
    MetalVosrTransformer::RestoreRequest req;
    req.lq = lq.data();
    req.lh = lh; req.lw = lw;
    req.cond = cond.data();
    req.cond_rows = cond_rows;
    req.steps = 1;
    req.noise = noise.data();
    req.tile = 0;
    // WARM TWICE, not once. The first forward builds pipelines and
    // scratch; the SECOND is where the int8 context tunes its split
    // width, because tune_pending() replays shapes the first one
    // recorded. Timing the second would have charged that arm a
    // load-time cost as if it were per-forward.
    for (int w = 0; w < 2; ++w) {
      if (!m->restore(req, out, &err)) {
        std::printf("[vosr_accel] restore: %s\n", err.c_str());
        return false;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = m->restore(req, out, &err);
    last_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
    if (!ok) { std::printf("[vosr_accel] restore: %s\n", err.c_str()); }
    return ok;
  };
  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0.0, d = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
      n += (a[i] - b[i]) * (a[i] - b[i]);
      d += (double)a[i] * a[i];
    }
    return d > 0.0 ? std::sqrt(n / d) : 0.0;
  };

  // 1. the ALU baseline.
  ::setenv("VPIPE_VOSR_NO_MMA2", "1", 1);
  ::setenv("VPIPE_VOSR_NO_ATTN_NAX", "1", 1);
  std::vector<float> alu;
  const bool have_alu = run(base, &alu);
  ::unsetenv("VPIPE_VOSR_NO_MMA2");
  ::unsetenv("VPIPE_VOSR_NO_ATTN_NAX");
  ASSERT_TRUE(have_alu);
  if (!have_alu || alu.empty()) { return; }
  const double alu_ms = last_ms;
  std::printf("[vosr_accel] %-10s %8.1f ms (baseline)\n", "alu", alu_ms);

  if (!mc->supports_matrix_cores()) {
    std::printf("[vosr_accel] no matrix cores -- only the ALU arm exists, "
                "SKIPPED\n");
    return;
  }

  struct Arm { const char* name; MetalVosrTransformer::Config cfg;
               double bound; };
  std::vector<Arm> arms;
  { Arm a{"nax", base, 0.05}; arms.push_back(a); }
  { Arm a{"nax+i8", base, 0.15}; a.cfg.i8_gemm = true; arms.push_back(a); }
  { Arm a{"nax+sage", base, 0.10}; a.cfg.sage.enabled = true;
    arms.push_back(a); }
  { Arm a{"nax+sol", base, 0.50}; a.cfg.sol.enabled = true;
    a.cfg.sol.tau = 1.0f; a.cfg.sol.dense_layers = 1;
    arms.push_back(a); }

  for (const Arm& a : arms) {
    std::vector<float> got;
    if (!run(a.cfg, &got) || got.size() != alu.size()) {
      std::printf("[vosr_accel] %-10s DID NOT RUN\n", a.name);
      EXPECT_TRUE(false);
      continue;
    }
    const double r = rel(got, alu);
    std::size_t differ = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (got[i] != alu[i]) { ++differ; }
    }
    std::printf("[vosr_accel] %-10s %8.1f ms (%.2fx)  vs ALU: rel-L2 "
                "%.4f, %.1f%% differ\n", a.name, last_ms,
                last_ms > 0.0 ? alu_ms / last_ms : 0.0, r,
                100.0 * (double)differ / (double)got.size());
    // Each bound is what that arm is entitled to, not one number for
    // all four: matmul2d is a different accumulation order, int8 is an
    // int8 quantization, and Sol drops key blocks.
    EXPECT_TRUE(std::isfinite(r) && r < a.bound);
    // ...and it RAN. An arm that silently fell through to the baseline
    // would score a perfect zero, which is the failure a bound alone
    // reads as the best possible result.
    EXPECT_TRUE(differ > got.size() / 100);
  }
}
