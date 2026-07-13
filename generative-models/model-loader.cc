#include "generative-models/model-loader.h"

#include "generative-models/gemma4/gemma4-unified-embedder.h"
#include "generative-models/shared/gguf-convert.h"
#include "generative-models/shared/gguf-file.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"


#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe::genai {

namespace {

// Read a JSON file from disk. Returns nullopt on I/O error or parse
// error; the caller logs context-aware diagnostics.
optional<FlexData>
read_json_file_(const filesystem::path& p)
{
  ifstream in(p);
  if (!in) {
    return nullopt;
  }
  try {
    return FlexData::from_json(in);
  } catch (const exception&) {
    return nullopt;
  }
}

// Pull the four shape-affecting head counts off of a HF config
// object. Some configs omit `head_dim` and rely on
// hidden_size / num_attention_heads; some omit `num_key_value_heads`
// for non-GQA models. Fill in the implied defaults.
void
parse_attention_(const FlexData::ConstObjectView& root, ModelConfig* out)
{
  if (root.contains("num_hidden_layers")) {
    out->n_layers = static_cast<int>(
        root.at("num_hidden_layers").as_int(0));
  }
  if (root.contains("num_attention_heads")) {
    out->n_heads = static_cast<int>(
        root.at("num_attention_heads").as_int(0));
  }
  if (root.contains("num_key_value_heads")) {
    out->n_kv_heads = static_cast<int>(
        root.at("num_key_value_heads").as_int(0));
  } else {
    // Pre-GQA Llama: kv_heads == n_heads.
    out->n_kv_heads = out->n_heads;
  }
  if (root.contains("hidden_size")) {
    out->hidden = static_cast<int>(
        root.at("hidden_size").as_int(0));
  }
  if (root.contains("head_dim")) {
    out->head_dim = static_cast<int>(
        root.at("head_dim").as_int(0));
  } else if (out->n_heads > 0) {
    out->head_dim = out->hidden / out->n_heads;
  }
  if (root.contains("intermediate_size")) {
    out->ffn_inner = static_cast<int>(
        root.at("intermediate_size").as_int(0));
  }
}

// Walk an object view and fill the dense-model fields from it. Both the
// top-level config and (for multimodal configs like Qwen3.5) the nested
// `text_config` object are valid sources -- HF wraps the LLM-side
// parameters in `text_config` whenever the architecture also has a
// vision tower. We call this twice: first against the root (so legacy
// configs that put hidden_size etc. at the top level still work), then
// against `text_config` if present (which takes precedence because the
// root duplicates for non-LLM fields like image_token_id are absent
// from the nested view).
void
parse_dense_fields_(const FlexData::ConstObjectView& root, ModelConfig* out)
{
  parse_attention_(root, out);

  if (root.contains("vocab_size")) {
    out->vocab_size = static_cast<int>(
        root.at("vocab_size").as_int(0));
  }
  if (root.contains("rope_theta")) {
    out->rope_theta = static_cast<float>(
        root.at("rope_theta").as_real(0.0));
  }
  if (root.contains("tie_word_embeddings")) {
    out->tie_word_embeddings =
        root.at("tie_word_embeddings").as_bool(false);
  }
  if (root.contains("rms_norm_eps")) {
    out->rms_eps = static_cast<float>(
        root.at("rms_norm_eps").as_real(1e-5));
  }
  if (root.contains("max_position_embeddings")) {
    out->max_position_embeddings = static_cast<int>(
        root.at("max_position_embeddings").as_int(0));
  }
}

// Pull Qwen3.5-specific hybrid-attention / mrope / partial-rotary
// fields from a config object view. Safe to call against either the
// root or the nested text_config; it only writes fields that are
// present in the view.
void
parse_hybrid_fields_(const FlexData::ConstObjectView& root, ModelConfig* out)
{
  // ---- per-layer kind (linear vs full attention) -------------------
  if (root.contains("layer_types")) {
    auto lt_fd = root.at("layer_types");
    if (lt_fd.is_array()) {
      auto arr = lt_fd.as_array();
      out->is_linear_layer.clear();
      out->is_linear_layer.reserve(arr.size());
      bool any_linear = false;
      for (size_t i = 0; i < arr.size(); ++i) {
        // HF Qwen3.5 uses the strings "linear_attention" and
        // "full_attention". Anything else is treated as "full" so a
        // typo in the config does not silently route a layer through
        // the SSM branch.
        string s(arr.at(i).as_string(""));
        bool lin = (s == "linear_attention");
        any_linear = any_linear || lin;
        out->is_linear_layer.push_back(lin);
      }
      // `is_linear_layer` non-empty is the "hybrid SSM model" signal
      // (drives ctx-spec, is_hybrid eval fences, etc.). Models that
      // reuse `layer_types` for a non-SSM split (Gemma-4: sliding vs
      // full attention) have NO linear layers -- leave the vector
      // empty so they are not mistaken for hybrids.
      if (!any_linear) {
        out->is_linear_layer.clear();
      }
    }
  }
  if (root.contains("full_attention_interval")) {
    out->full_attention_interval = static_cast<int>(
        root.at("full_attention_interval").as_int(0));
  }

  // ---- linear-attention sizing (Gated DeltaNet) --------------------
  if (root.contains("linear_num_key_heads")) {
    out->linear_num_k_heads = static_cast<int>(
        root.at("linear_num_key_heads").as_int(0));
  }
  if (root.contains("linear_num_value_heads")) {
    out->linear_num_v_heads = static_cast<int>(
        root.at("linear_num_value_heads").as_int(0));
  }
  if (root.contains("linear_key_head_dim")) {
    out->linear_k_head_dim = static_cast<int>(
        root.at("linear_key_head_dim").as_int(0));
  }
  if (root.contains("linear_value_head_dim")) {
    out->linear_v_head_dim = static_cast<int>(
        root.at("linear_value_head_dim").as_int(0));
  }
  if (root.contains("linear_conv_kernel_dim")) {
    out->linear_conv_kernel = static_cast<int>(
        root.at("linear_conv_kernel_dim").as_int(0));
  }

  // ---- mrope / partial rotary -------------------------------------
  // Qwen3.5 nests these under `rope_parameters` rather than at the
  // text_config root. Honor either location so legacy configs (or
  // future ones that hoist the fields back out) keep working.
  auto absorb_rope = [out](const FlexData::ConstObjectView& src) {
    if (src.contains("rope_theta")) {
      // text_config may override the older top-level rope_theta.
      out->rope_theta = static_cast<float>(
          src.at("rope_theta").as_real(out->rope_theta));
    }
    if (src.contains("partial_rotary_factor")) {
      out->partial_rotary_factor = static_cast<float>(
          src.at("partial_rotary_factor").as_real(1.0));
    }
    if (src.contains("mrope_section")) {
      auto ms = src.at("mrope_section");
      if (ms.is_array()) {
        auto arr = ms.as_array();
        out->mrope_section.clear();
        out->mrope_section.reserve(arr.size());
        for (size_t i = 0; i < arr.size(); ++i) {
          out->mrope_section.push_back(
              static_cast<int>(arr.at(i).as_int(0)));
        }
      }
    }
  };
  if (root.contains("rope_parameters")) {
    auto rp = root.at("rope_parameters");
    if (rp.is_object()) {
      absorb_rope(rp.as_object());
    }
  }
  // Also accept the older flat layout (mrope_section / partial_rotary_
  // factor directly on the view) -- this is what some Qwen2.5-VL
  // configs use.
  absorb_rope(root);

  // ---- Q-projection output gate -----------------------------------
  if (root.contains("attn_output_gate")) {
    out->attn_output_gate = root.at("attn_output_gate").as_bool(false);
  }

  // ---- Mixture-of-Experts (Qwen3.5-MoE) ---------------------------
  // num_experts > 0 switches every layer's MLP to a SparseMoeBlock.
  if (root.contains("num_experts")) {
    out->num_experts = static_cast<int>(
        root.at("num_experts").as_int(0));
  }
  if (root.contains("num_experts_per_tok")) {
    out->num_experts_per_tok = static_cast<int>(
        root.at("num_experts_per_tok").as_int(0));
  }
  if (root.contains("moe_intermediate_size")) {
    out->moe_intermediate_size = static_cast<int>(
        root.at("moe_intermediate_size").as_int(0));
  }
  if (root.contains("shared_expert_intermediate_size")) {
    out->shared_expert_inter = static_cast<int>(
        root.at("shared_expert_intermediate_size").as_int(0));
  }
  if (root.contains("norm_topk_prob")) {
    out->norm_topk_prob = root.at("norm_topk_prob").as_bool(true);
  }
  // Gemma-4 MoE spelling: top-k is `top_k_experts` and the block is gated by
  // `enable_moe_block` (Qwen uses num_experts_per_tok / always-on). Map top-k
  // into the shared num_experts_per_tok field.
  if (root.contains("top_k_experts")) {
    out->num_experts_per_tok = static_cast<int>(
        root.at("top_k_experts").as_int(0));
  }
  if (root.contains("enable_moe_block")) {
    out->enable_moe_block = root.at("enable_moe_block").as_bool(false);
  }
}

// Pull Gemma-4 text-family sizing off a config object view. Safe to
// call against either the root or the nested `text_config`; it only
// writes fields present in the view. `present` and the derived
// defaults (head_dim_sliding, is_full_layer sizing) are finalized by
// the caller once the architecture is known.
void
parse_gemma4_fields_(const FlexData::ConstObjectView& root, ModelConfig* out)
{
  auto& g = out->gemma4;
  if (root.contains("head_dim")) {
    g.head_dim_sliding = static_cast<int>(root.at("head_dim").as_int(0));
  }
  if (root.contains("global_head_dim")) {
    g.head_dim_full = static_cast<int>(
        root.at("global_head_dim").as_int(0));
  }
  if (root.contains("num_kv_shared_layers")) {
    g.num_kv_shared_layers = static_cast<int>(
        root.at("num_kv_shared_layers").as_int(0));
  }
  if (root.contains("hidden_size_per_layer_input")) {
    g.hidden_per_layer_input = static_cast<int>(
        root.at("hidden_size_per_layer_input").as_int(0));
  }
  if (root.contains("vocab_size_per_layer_input")) {
    g.vocab_per_layer_input = static_cast<int>(
        root.at("vocab_size_per_layer_input").as_int(0));
  }
  if (root.contains("sliding_window")) {
    g.sliding_window = static_cast<int>(
        root.at("sliding_window").as_int(0));
  }
  if (root.contains("final_logit_softcapping")) {
    g.final_logit_softcapping = static_cast<float>(
        root.at("final_logit_softcapping").as_real(0.0));
  }
  if (root.contains("attention_k_eq_v")) {
    g.attention_k_eq_v = root.at("attention_k_eq_v").as_bool(false);
  }
  if (root.contains("num_global_key_value_heads")) {
    g.num_global_kv_heads = static_cast<int>(
        root.at("num_global_key_value_heads").as_int(0));
  }

  // rope_parameters nests one sub-object per attention kind.
  if (root.contains("rope_parameters")) {
    auto rp_fd = root.at("rope_parameters");
    if (rp_fd.is_object()) {
      auto rp = rp_fd.as_object();
      if (rp.contains("sliding_attention")) {
        auto s_fd = rp.at("sliding_attention");
        if (s_fd.is_object()) {
          auto s = s_fd.as_object();
          if (s.contains("rope_theta")) {
            g.rope_theta_sliding = static_cast<float>(
                s.at("rope_theta").as_real(g.rope_theta_sliding));
          }
        }
      }
      if (rp.contains("full_attention")) {
        auto f_fd = rp.at("full_attention");
        if (f_fd.is_object()) {
          auto f = f_fd.as_object();
          if (f.contains("rope_theta")) {
            g.rope_theta_full = static_cast<float>(
                f.at("rope_theta").as_real(g.rope_theta_full));
          }
          if (f.contains("partial_rotary_factor")) {
            g.full_partial_rotary_factor = static_cast<float>(
                f.at("partial_rotary_factor").as_real(
                    g.full_partial_rotary_factor));
          }
        }
      }
    }
  }

  // layer_types -> per-layer attention kind (true == full_attention).
  if (root.contains("layer_types")) {
    auto lt_fd = root.at("layer_types");
    if (lt_fd.is_array()) {
      auto arr = lt_fd.as_array();
      g.is_full_layer.clear();
      g.is_full_layer.reserve(arr.size());
      for (size_t i = 0; i < arr.size(); ++i) {
        string s(arr.at(i).as_string(""));
        g.is_full_layer.push_back(s == "full_attention");
      }
    }
  }
}

bool
parse_config_(const FlexData& cfg, ModelConfig* out)
{
  if (!cfg.is_object()) {
    return false;
  }
  auto outer = cfg.as_object();

  if (outer.contains("architectures")) {
    auto arr_fd = outer.at("architectures");
    if (arr_fd.is_array()) {
      auto arr = arr_fd.as_array();
      if (arr.size() > 0) {
        out->architecture = string(arr.at(0).as_string(""));
      }
    }
  }

  // Some omni-modal architectures (Qwen3-ASR, Qwen2.5-Omni) wrap every
  // text_config / vision_config / audio_config / token-id field one
  // level deeper under `thinker_config`. Promote that view to the
  // current root so the same field-parsing paths below pick up the
  // values transparently. Fields that always live at the outermost
  // level (support_languages, quantization) keep reading from `outer`.
  FlexData thinker_holder;  // holds storage when we descend
  bool     used_thinker = false;
  if (outer.contains("thinker_config")) {
    auto tc_fd = outer.at("thinker_config");
    if (tc_fd.is_object()) {
      thinker_holder = tc_fd;
      used_thinker = true;
    }
  }
  auto root = used_thinker ? thinker_holder.as_object() : outer;

  // architecture might also be set inside thinker_config; prefer the
  // outer value when both are present.
  if (out->architecture.empty() && root.contains("architectures")) {
    auto arr_fd = root.at("architectures");
    if (arr_fd.is_array()) {
      auto arr = arr_fd.as_array();
      if (arr.size() > 0) {
        out->architecture = string(arr.at(0).as_string(""));
      }
    }
  }

  // Multimodal configs (Qwen3.5, Qwen2-VL, etc.) put every LLM-side
  // field under `text_config`. Read root first so legacy single-section
  // Llama configs still parse, then re-read from text_config so its
  // values override any defaults the root left behind.
  parse_dense_fields_(root, out);
  parse_hybrid_fields_(root, out);
  parse_gemma4_fields_(root, out);
  if (root.contains("text_config")) {
    auto tc = root.at("text_config");
    if (tc.is_object()) {
      auto tcobj = tc.as_object();
      parse_dense_fields_(tcobj, out);
      parse_hybrid_fields_(tcobj, out);
      parse_gemma4_fields_(tcobj, out);
    }
  }

  // MOSS-TTS-Local (model_type "moss_tts_local", arch "MossTTSLocalModel"):
  // a TTS model whose text backbone is a dense Qwen3 LM. The Qwen3 dims live
  // under `qwen3_config` (not the usual `text_config`), so parse them into the
  // standard dense fields. This lets the metal Qwen text-backbone exec be built
  // for text evaluation (e.g. WikiText-2 perplexity) -- it does NOT affect the
  // TTS forward path, which loads the backbone directly via MetalMossV15Model.
  if (out->architecture == "MossTTSLocalModel" &&
      root.contains("qwen3_config")) {
    auto qc = root.at("qwen3_config");
    if (qc.is_object()) {
      auto qcobj = qc.as_object();
      parse_dense_fields_(qcobj, out);
    }
  }

  // MOSS-TTS-Realtime (arch "MossTTSRealtime"): a streaming TTS whose text
  // backbone is a dense Qwen3-1.7B under `language_config` (not `text_config`).
  // Parse those dims so the metal Qwen text-backbone exec can be built for
  // evaluation. NB the backbone has NO real text head (the model predicts audio
  // via a depth transformer, not text) -- absolute perplexity is meaningless;
  // the meaningful eval is the A/B divergence mode (e.g. 8-bit vs bf16
  // quantization error). Does not affect the TTS forward (text-to-speech loads
  // the backbone directly via MetalMossRtModel).
  if (out->architecture == "MossTTSRealtime" &&
      root.contains("language_config")) {
    auto lc = root.at("language_config");
    if (lc.is_object()) {
      auto lcobj = lc.as_object();
      parse_dense_fields_(lcobj, out);
    }
  }

  // Finalize the Gemma-4 view: mark present (which selects
  // Gemma4ModelExec) and fill derived defaults. Done here, after both
  // parse passes, so head_dim / n_layers reflect the text_config view.
  // Both the e4b `gemma4` arch and the 12B `gemma4_unified` arch route here.
  if (out->architecture == "Gemma4ForConditionalGeneration"
      || out->architecture == "Gemma4UnifiedForConditionalGeneration") {
    out->gemma4.present = true;
    if (out->gemma4.head_dim_sliding == 0) {
      out->gemma4.head_dim_sliding = out->head_dim;
    }
    if (out->gemma4.head_dim_full == 0) {
      out->gemma4.head_dim_full = out->gemma4.head_dim_sliding;
    }
    // Generic consumers (ctx bookkeeping) see the sliding head_dim.
    out->head_dim = out->gemma4.head_dim_sliding;

    // Per-layer K/V head count. gemma4_unified full-attention layers use
    // `num_global_kv_heads` (typically 1) and share K as V; sliding layers
    // and every e4b layer use the generic n_kv_heads. Sized to n_layers when
    // layer_types was parsed; otherwise left empty (consumers fall back to
    // the uniform n_kv_heads).
    auto& g = out->gemma4;
    const int n = static_cast<int>(g.is_full_layer.size());
    if (n > 0 && out->n_kv_heads > 0) {
      g.layer_n_kv_heads.assign(static_cast<std::size_t>(n), out->n_kv_heads);
      if (g.attention_k_eq_v && g.num_global_kv_heads > 0) {
        for (int L = 0; L < n; ++L) {
          if (g.is_full_layer[static_cast<std::size_t>(L)]) {
            g.layer_n_kv_heads[static_cast<std::size_t>(L)] =
                g.num_global_kv_heads;
          }
        }
      }
    }
  }

  // ---- rope_scaling (Llama 3.1 long-context) -----------------------
  if (root.contains("rope_scaling")) {
    auto rs_fd = root.at("rope_scaling");
    if (rs_fd.is_object()) {
      auto rs = rs_fd.as_object();
      // HF uses "rope_type" on Llama 3.1 (older configs say "type").
      string kind;
      if (rs.contains("rope_type")) {
        kind = string(rs.at("rope_type").as_string(""));
      } else if (rs.contains("type")) {
        kind = string(rs.at("type").as_string(""));
      }
      if (kind == "llama3") {
        out->rope_scaling.kind = "llama3";
        if (rs.contains("factor")) {
          out->rope_scaling.factor =
              static_cast<float>(rs.at("factor").as_real(1.0));
        }
        if (rs.contains("low_freq_factor")) {
          out->rope_scaling.low_freq_factor =
              static_cast<float>(rs.at("low_freq_factor").as_real(1.0));
        }
        if (rs.contains("high_freq_factor")) {
          out->rope_scaling.high_freq_factor =
              static_cast<float>(rs.at("high_freq_factor").as_real(1.0));
        }
        if (rs.contains("original_max_position_embeddings")) {
          out->rope_scaling.original_max_position_embeddings =
              static_cast<int>(
                  rs.at("original_max_position_embeddings").as_int(0));
        }
      }
    }
  }

  // ---- vision_config (Qwen3-VL family) ----------------------------
  // HF nests every vision-tower field under `vision_config`. We
  // parse what we need and leave VisionConfig.present == false for
  // text-only models so consumers can branch cheaply.
  if (root.contains("vision_config")) {
    auto vc_fd = root.at("vision_config");
    if (vc_fd.is_object()) {
      auto vc = vc_fd.as_object();
      out->vision.present = true;
      auto get_int_ = [&](const char* key, int& dst) {
        if (vc.contains(key)) {
          dst = static_cast<int>(vc.at(key).as_int(0));
        }
      };
      get_int_("depth",                   out->vision.depth);
      get_int_("hidden_size",             out->vision.hidden_size);
      get_int_("intermediate_size",       out->vision.intermediate_size);
      get_int_("num_heads",               out->vision.num_heads);
      get_int_("in_channels",             out->vision.in_channels);
      get_int_("patch_size",              out->vision.patch_size);
      get_int_("spatial_merge_size",      out->vision.spatial_merge_size);
      get_int_("temporal_patch_size",     out->vision.temporal_patch_size);
      get_int_("out_hidden_size",         out->vision.out_hidden_size);
      get_int_("num_position_embeddings", out->vision.num_position_embeddings);
      if (vc.contains("deepstack_visual_indexes")) {
        auto ds_fd = vc.at("deepstack_visual_indexes");
        if (ds_fd.is_array()) {
          for (FlexData v : ds_fd.as_array()) {
            out->vision.deepstack_visual_indexes.push_back(
                static_cast<int>(v.as_int(0)));
          }
        }
      }

      // ---- Gemma-4 vision tower (different key names + extras) -------
      // Gemma's vision_config uses num_hidden_layers / num_attention_
      // heads (not depth / num_heads), carries head_dim + pooling +
      // position-embedding sizing, and projects to the text hidden
      // (no out_hidden_size in vision_config). Its image preproc is
      // rescale-only.
      if (out->architecture == "Gemma4ForConditionalGeneration") {
        get_int_("num_hidden_layers",       out->vision.depth);
        get_int_("num_attention_heads",     out->vision.num_heads);
        get_int_("head_dim",                out->vision.vit_head_dim);
        get_int_("pooling_kernel_size",     out->vision.pooling_kernel_size);
        get_int_("default_output_length",
                 out->vision.default_output_length);
        // Video frames use the Gemma4VideoProcessor's smaller per-frame
        // budget (default 70 vs the 280 still budget above). It is a
        // processor constant, not in vision_config; allow an explicit
        // override but default to 70.
        out->vision.video_default_output_length = 70;
        get_int_("video_default_output_length",
                 out->vision.video_default_output_length);
        get_int_("position_embedding_size",
                 out->vision.position_embedding_size);
        if (vc.contains("rope_parameters")) {
          auto rp_fd = vc.at("rope_parameters");
          if (rp_fd.is_object()) {
            auto rp = rp_fd.as_object();
            if (rp.contains("rope_theta")) {
              out->vision.vit_rope_theta = static_cast<float>(
                  rp.at("rope_theta").as_real(100.0));
            }
          }
        }
        out->vision.out_hidden_size = out->hidden;   // text hidden
        for (int i = 0; i < 3; ++i) {
          out->vision.image_mean[i] = 0.0f;
          out->vision.image_std[i]  = 1.0f;
        }
      }
    }
  }

  // ---- audio_config (Qwen3-ASR family) ----------------------------
  // HF nests every audio-tower field under `audio_config`. We parse
  // what we need and leave AudioConfig.present == false for non-audio
  // models so consumers can branch cheaply.
  if (root.contains("audio_config")) {
    auto ac_fd = root.at("audio_config");
    if (ac_fd.is_object()) {
      auto ac = ac_fd.as_object();
      out->audio.present = true;
      auto get_int_ = [&](const char* key, int& dst) {
        if (ac.contains(key)) {
          dst = static_cast<int>(ac.at(key).as_int(0));
        }
      };
      get_int_("d_model",                 out->audio.d_model);
      get_int_("encoder_layers",          out->audio.encoder_layers);
      get_int_("encoder_attention_heads", out->audio.encoder_attention_heads);
      get_int_("encoder_ffn_dim",         out->audio.encoder_ffn_dim);
      get_int_("downsample_hidden_size",  out->audio.downsample_hidden_size);
      get_int_("num_mel_bins",            out->audio.num_mel_bins);
      get_int_("max_source_positions",    out->audio.max_source_positions);
      get_int_("output_dim",              out->audio.output_dim);
      get_int_("n_window",                out->audio.n_window);
      get_int_("n_window_infer",          out->audio.n_window_infer);
      get_int_("conv_chunksize",          out->audio.conv_chunksize);
      if (ac.contains("scale_embedding")) {
        out->audio.scale_embedding =
            ac.at("scale_embedding").as_bool(false);
      }

      // ---- Gemma-4 USM Conformer (different key names + extras) -----
      if (out->architecture == "Gemma4ForConditionalGeneration") {
        get_int_("hidden_size",          out->audio.d_model);
        get_int_("num_hidden_layers",    out->audio.encoder_layers);
        get_int_("num_attention_heads",  out->audio.encoder_attention_heads);
        get_int_("output_proj_dims",     out->audio.output_dim);
        get_int_("conv_kernel_size",     out->audio.conv_kernel_size);
        get_int_("attention_chunk_size", out->audio.attention_chunk_size);
        get_int_("attention_context_left",
                 out->audio.attention_context_left);
        get_int_("attention_context_right",
                 out->audio.attention_context_right);
        out->audio.encoder_ffn_dim = out->audio.d_model * 4;
        auto get_real_ = [&](const char* key, float& dst) {
          if (ac.contains(key)) {
            dst = static_cast<float>(ac.at(key).as_real(dst));
          }
        };
        get_real_("attention_logit_cap", out->audio.attention_logit_cap);
        get_real_("gradient_clipping",   out->audio.gradient_clipping);
        get_real_("residual_weight",     out->audio.residual_weight);
        get_real_("rms_norm_eps",        out->audio.audio_rms_eps);
        if (ac.contains("subsampling_conv_channels")) {
          auto cc = ac.at("subsampling_conv_channels");
          if (cc.is_array()) {
            for (FlexData v : cc.as_array()) {
              out->audio.subsampling_conv_channels.push_back(
                  static_cast<int>(v.as_int(0)));
            }
          }
        }
      }
    }
  }

  // ---- audio token ids (Qwen3-ASR thinker_config) ------------------
  // These sit at thinker_config's root (NOT inside audio_config). Read
  // them off whichever view they're available in.
  auto absorb_audio_tokens = [out](const FlexData::ConstObjectView& src) {
    if (src.contains("audio_start_token_id")) {
      out->audio_start_token_id = static_cast<std::int32_t>(
          src.at("audio_start_token_id").as_int(-1));
    }
    if (src.contains("audio_end_token_id")) {
      out->audio_end_token_id = static_cast<std::int32_t>(
          src.at("audio_end_token_id").as_int(-1));
    }
    if (src.contains("audio_token_id")) {
      out->audio_pad_token_id = static_cast<std::int32_t>(
          src.at("audio_token_id").as_int(-1));
    }
  };
  absorb_audio_tokens(root);

  // ---- support_languages (Qwen3-ASR multilingual list) -------------
  // Lives at the OUTERMOST config level (not inside thinker_config),
  // so read from `outer` even when thinker_config was promoted to
  // `root` above.
  if (outer.contains("support_languages")) {
    auto sl_fd = outer.at("support_languages");
    if (sl_fd.is_array()) {
      for (FlexData v : sl_fd.as_array()) {
        out->support_languages.emplace_back(v.as_string(""));
      }
    }
  }

  // ---- quantization (MLX affine, present on mlx-converted checkpoints) -
  // Quantization applies to the whole bundle, so it lives at the
  // outermost level for omni-modal checkpoints.
  if (outer.contains("quantization")) {
    auto q_fd = outer.at("quantization");
    if (q_fd.is_object()) {
      auto q = q_fd.as_object();
      if (q.contains("bits")) {
        out->quantization.bits =
            static_cast<int>(q.at("bits").as_int(0));
      }
      if (q.contains("group_size")) {
        out->quantization.group_size =
            static_cast<int>(q.at("group_size").as_int(0));
      }
    }
  }
  return true;
}


// preprocessor_config.json (optional). HF stores the image
// normalisation mean/std and pixel-count clamps here, not in
// config.json. For VLMs we MUST honour these -- Qwen3-VL ships
// SigLIP-style [0.5,0.5,0.5]/[0.5,0.5,0.5] which is materially
// different from the OpenAI/CLIP defaults baked into VisionConfig
// (off by ~2x in scale + per-channel mean shift). Missing file or
// missing fields fall back to the existing defaults silently. No-op
// for text-only checkpoints (vision.present == false).
void
apply_preprocessor_(const filesystem::path&    dir,
                    const string&               dir_str,
                    ModelConfig*                config,
                    const SessionContextIntf*   session)
{
  if (!config->vision.present) {
    return;
  }
  filesystem::path pp_path = dir / "preprocessor_config.json";
  auto pp_fd = read_json_file_(pp_path);
  if (!pp_fd || !pp_fd->is_object()) {
    return;
  }
  auto pp = pp_fd->as_object();
  auto read_triple_ = [&](const char* key, float (&dst)[3]) {
    if (!pp.contains(key)) { return; }
    FlexData v = pp.at(key);
    if (!v.is_array()) { return; }
    auto arr = v.as_array();
    if (arr.size() != 3) { return; }
    for (int i = 0; i < 3; ++i) {
      dst[i] = static_cast<float>(arr[i].as_real(dst[i]));
    }
  };
  read_triple_("image_mean", config->vision.image_mean);
  read_triple_("image_std",  config->vision.image_std);
  if (session) {
    session->info(fmt(
        "ModelLoader::load('{}'): preprocessor_config.json: "
        "image_mean=[{:.4f}, {:.4f}, {:.4f}] "
        "image_std=[{:.4f}, {:.4f}, {:.4f}]",
        dir_str,
        config->vision.image_mean[0],
        config->vision.image_mean[1],
        config->vision.image_mean[2],
        config->vision.image_std[0],
        config->vision.image_std[1],
        config->vision.image_std[2]));
  }
}

}

namespace {

// Build a LoadedWeights from a GGUF checkpoint. The config is always
// populated from the GGUF metadata; tensors are materialised into MLX
// arrays only on the MLX build (the metal backend reads the GGUF itself
// through MetalLlamaWeights, driving the same converter). q4_0 linears
// become affine 4-bit g32 (lossless), the q6_K token table affine 8-bit
// g32; both backends see byte-identical converted weights.
bool
load_gguf_(const std::string& gguf_path, LoadedWeights* out,
           const SessionContextIntf* session)
{
  auto g = GgufFile::open(gguf_path);
  if (!g) {
    if (session) {
      session->warn(fmt("ModelLoader: GgufFile::open('{}') failed",
                        gguf_path));
    }
    return false;
  }
  if (!gguf_to_model_config(*g, &out->config)) {
    if (session) {
      session->warn(fmt(
          "ModelLoader: '{}' is not a supported gemma4 GGUF", gguf_path));
    }
    return false;
  }
  // gemma4_unified multimodal: the shallow vision/audio embedder weights
  // live in a SEPARATE mmproj GGUF (projector_type gemma4uv / gemma4ua).
  // Detect it and flag the config so the loader builds a
  // Gemma4UnifiedEmbedder (encoder-less), NOT the e4b ViT/Conformer.
  {
    namespace fs = std::filesystem;
    const std::string mmp = Gemma4UnifiedEmbedder::find_mmproj(
        fs::path(gguf_path).parent_path().string());
    if (!mmp.empty()) {
      if (auto mg = GgufFile::open(mmp)) {
        const auto a = mg->get_string("general.architecture");
        if (a && *a == "clip") {
          const auto vp = mg->get_string("clip.vision.projector_type");
          const auto ap = mg->get_string("clip.audio.projector_type");
          // Qwen3-VL's GGUF converter writes the projector type at the
          // top-level `clip.projector_type`; gemma uses the `.vision.`/
          // `.audio.`-prefixed keys above. Accept either spelling.
          const auto cp = mg->get_string("clip.projector_type");
          if (vp && *vp == "gemma4uv") {
            out->config.vision.present = true;
            out->config.vision.unified = true;
            out->config.vision.mmproj_path = mmp;
            out->config.vision.out_hidden_size = out->config.hidden;
          }
          if (ap && *ap == "gemma4ua") {
            out->config.audio.present = true;
            out->config.audio.unified = true;
            out->config.audio.mmproj_path = mmp;
            out->config.audio.output_dim = out->config.hidden;
          }
          // Qwen3.5-VL vision tower (a full ViT, not gemma's shallow embedder).
          // Same arch as the safetensors tower; the mmproj just renames the
          // tensors (CLIP layout) + keeps BF16/F32. Populate VisionConfig from
          // the clip.vision.* metadata; the metal encoder loads via mmproj_path.
          if ((cp && *cp == "qwen3vl_merger") ||
              (vp && *vp == "qwen3vl_merger")) {
            auto gi = [&](const char* k) -> int {
              const auto v = mg->get_int(k);
              return v ? static_cast<int>(*v) : 0;
            };
            auto& vis = out->config.vision;
            vis.present              = true;
            vis.mmproj_path          = mmp;
            vis.depth                = gi("clip.vision.block_count");
            vis.hidden_size          = gi("clip.vision.embedding_length");
            vis.intermediate_size    = gi("clip.vision.feed_forward_length");
            vis.num_heads            = gi("clip.vision.attention.head_count");
            vis.patch_size           = gi("clip.vision.patch_size");
            vis.spatial_merge_size   = gi("clip.vision.spatial_merge_size");
            vis.temporal_patch_size  = 2;   // Qwen-VL: 2-frame patch conv
            vis.in_channels          = 3;
            vis.out_hidden_size      = out->config.hidden;   // == LM hidden
            // num_position_embeddings = the learned pos table's row count
            // (ggml ne1 of v.position_embd.weight [hidden, num_pos]).
            if (const auto* pe = mg->tensor("v.position_embd.weight");
                pe != nullptr && pe->dims.size() >= 2) {
              vis.num_position_embeddings = static_cast<int>(pe->dims[1]);
            }
            const auto im = mg->get_float_array("clip.vision.image_mean");
            const auto is = mg->get_float_array("clip.vision.image_std");
            for (int i = 0; i < 3; ++i) {
              if (i < static_cast<int>(im.size())) { vis.image_mean[i] = im[i]; }
              if (i < static_cast<int>(is.size())) { vis.image_std[i] = is[i]; }
            }
          }
        }
      }
    }
  }
  (void)session;
  return true;
}

}  // namespace

ModelLoader::ModelLoader(const SessionContextIntf* s)
  : SessionMember(s)
{
}

optional<LoadedWeights>
ModelLoader::load(string_view hf_dir) const
{
  // Copy the path into a stable string up front: the log delegate's
  // async consumer renders fmt() lambdas on a worker thread, so any
  // string_view captured into a format closure must reference
  // storage that outlives this call. Local strings copied into the
  // [=]-capture handle that.
  const string dir_str(hf_dir);
  filesystem::path dir(dir_str);

  // 0. GGUF checkpoint: no config.json -- config + (MLX) tensors come
  // from the .gguf itself.
  if (const string gguf_path = find_gguf_in_dir(dir_str);
      !gguf_path.empty()) {
    LoadedWeights out;
    if (!load_gguf_(gguf_path, &out, session())) {
      return nullopt;
    }
    return out;
  }

  // 1. config.json
  filesystem::path config_path = dir / "config.json";
  auto cfg_fd = read_json_file_(config_path);
  if (!cfg_fd) {
    if (session()) {
      session()->warn(fmt(
          "ModelLoader::load('{}'): config.json missing or unreadable",
          dir_str));
    }
    return nullopt;
  }
  LoadedWeights out;
  if (!parse_config_(*cfg_fd, &out.config)) {
    if (session()) {
      session()->warn(fmt(
          "ModelLoader::load('{}'): config.json is not a JSON object",
          dir_str));
    }
    return nullopt;
  }


  // 3. preprocessor_config.json (optional, VLM normalisation mean/std).
  apply_preprocessor_(dir, dir_str, &out.config, session());

  // 4. gemma4_unified from RAW safetensors: unlike the GGUF variant (whose
  // adaptor lives in a sibling mmproj-*.gguf), the raw 12B carries the
  // shallow vision/audio adaptor weights INSIDE model.safetensors. Probe
  // for them and flag the config so the loader builds a
  // Gemma4UnifiedEmbedder via load_safetensors(model_dir).
  if (out.config.architecture == "Gemma4UnifiedForConditionalGeneration"
      && Gemma4UnifiedEmbedder::has_unified_safetensors(dir_str)) {
    out.config.vision.present    = true;
    out.config.vision.unified    = true;
    out.config.vision.unified_st = true;
    out.config.vision.out_hidden_size = out.config.hidden;
    out.config.audio.present     = true;
    out.config.audio.unified     = true;
    out.config.audio.unified_st  = true;
    out.config.audio.output_dim  = out.config.hidden;
  }

  return out;
}

optional<ModelConfig>
ModelLoader::load_config(string_view hf_dir) const
{
  const string dir_str(hf_dir);
  filesystem::path dir(dir_str);

  // GGUF checkpoint: config comes from the .gguf metadata.
  if (const string gguf_path = find_gguf_in_dir(dir_str);
      !gguf_path.empty()) {
    auto g = GgufFile::open(gguf_path);
    if (!g) {
      if (session()) {
        session()->warn(fmt(
            "ModelLoader::load_config: GgufFile::open('{}') failed",
            gguf_path));
      }
      return nullopt;
    }
    ModelConfig config;
    if (!gguf_to_model_config(*g, &config)) {
      if (session()) {
        session()->warn(fmt(
            "ModelLoader::load_config('{}'): unsupported GGUF", dir_str));
      }
      return nullopt;
    }
    return config;
  }

  filesystem::path config_path = dir / "config.json";
  auto cfg_fd = read_json_file_(config_path);
  if (!cfg_fd) {
    if (session()) {
      session()->warn(fmt(
          "ModelLoader::load_config('{}'): config.json missing or "
          "unreadable", dir_str));
    }
    return nullopt;
  }
  ModelConfig config;
  if (!parse_config_(*cfg_fd, &config)) {
    if (session()) {
      session()->warn(fmt(
          "ModelLoader::load_config('{}'): config.json is not a JSON "
          "object", dir_str));
    }
    return nullopt;
  }
  apply_preprocessor_(dir, dir_str, &config, session());
  return config;
}

}
