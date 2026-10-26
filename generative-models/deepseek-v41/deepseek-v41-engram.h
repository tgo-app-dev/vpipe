#ifndef VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_DEEPSEEK_V41_ENGRAM_H
#define VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_DEEPSEEK_V41_ENGRAM_H

// DeepSeek-V4.1's ENGRAM hashing (the release's inference/engram.py): which
// rows of the n-gram tables a position reads.
//
// Token ids go through a COMPRESSED map first -- tokens that normalize
// alike (NFKC, NFD, accents stripped, lowercase, whitespace runs to one
// space, stripped) share one id, so " The", "the" and "THE" hash the same
// -- then each position is hashed with the max_ngram - 1 tokens before it:
// the running XOR of token * multiplier[layer][lookback] after lookback i
// is the hash of the (i+1)-gram, taken modulo a prime per (n-gram, head)
// and offset into that prime's range of the layer's table. Look-back stops
// at the sequence start and at an image span (DEAD): those slots read the
// pad token's id. The multipliers are numpy's default_rng(10007 * layer)
// .integers(0, bound, 4) * 2 + 1 -- PCG64 restated here bit for bit -- and
// the primes are drawn in order from engram_vocab_size, never reused.
//
// Everything derives from the tokenizer and the config, so the table the
// checkpoint was trained with is checked twice: the compressed vocabulary
// must have the config's size, and each layer's primes must sum to its
// table's rows.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vpipe::genai {

struct DeepseekV41Config;

class DeepseekV41Engram {
public:
  static constexpr std::int32_t kDead = -1;

  // From the checkpoint's tokenizer.json. False with `err` set when the
  // tokenizer or the layout disagrees with the config.
  bool init(const DeepseekV41Config& cfg, const std::string& tokenizer_json,
            std::string* err);

  // The compressed id of each token id ([vocab]).
  const std::vector<std::int32_t>& token_map() const { return _map; }
  int compressed_vocab() const { return _cvocab; }
  // multipliers[layer slot][lookback], primes[layer slot][ngram-2][head].
  const std::vector<std::vector<std::int64_t>>& multipliers() const
  {
    return _mult;
  }
  const std::vector<std::vector<std::int64_t>>& primes() const
  {
    return _primes;
  }
  int layers() const { return (int)_mult.size(); }
  int cols() const { return _cols; }

  // The compressed id a position takes (kDead for an image position).
  std::int32_t compress(std::int32_t token) const;

  // The rows position `pos` reads, for every engram layer: out[slot * cols
  // + c]. `hist` holds the compressed ids of positions 0..pos (its last
  // entry the position's own).
  void hash(std::span<const std::int32_t> hist, std::int64_t* out) const;

  // numpy.random.default_rng(seed).integers(0, high, n) for int64, high >
  // 2^32 -- what the multipliers are drawn with (exposed for tests).
  static std::vector<std::int64_t> numpy_integers(std::uint64_t seed,
                                                  std::int64_t high, int n);
  // A token's text as the training normalizer keys it (exposed for
  // tests): NFKC, NFD, marks stripped, lowercase, whitespace collapsed,
  // stripped (a single space survives).
  static std::string normalize(const std::string& utf8);

private:
  std::vector<std::int32_t> _map;
  int _cvocab = 0;
  int _ngram = 0, _heads = 0, _cols = 0;
  std::int32_t _pad = 0;
  std::vector<std::vector<std::int64_t>> _mult;      // [layers][ngram]
  std::vector<std::vector<std::int64_t>> _primes;    // [layers][cols]
  std::vector<std::vector<std::int64_t>> _offsets;   // [layers][cols]
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_DEEPSEEK_V41_ENGRAM_H
