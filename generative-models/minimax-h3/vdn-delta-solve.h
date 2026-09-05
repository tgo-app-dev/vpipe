#ifndef GENERATIVE_MODELS_MINIMAX_H3_VDN_DELTA_SOLVE_H
#define GENERATIVE_MODELS_MINIMAX_H3_VDN_DELTA_SOLVE_H

// VDN-H3's `vdn_solve` delta rule: how one frame's statistics become the
// bidirectional scan's transition and injection.
//
//     S_out = (S_in Diag(alpha) + B) (I + A)^-1
//     A = sum_s beta_s k_s k_s^T     [d_k, d_k]  symmetric PSD
//     B = sum_s beta_s v_s k_s^T     [d_v, d_k]
//
// A IS NOT SMALL, and that is the whole reason this is a solve rather
// than a truncation. The branch L2-normalises its keys, so
//
//     trace(A) = sum_s beta_s     EXACTLY
//
// -- A's strength is set by the token count of a frame times the mean
// beta, and at production geometry (S ~ 1008, beta centred on 0.5, since
// beta_proj carries no bias) that is a trace in the hundreds over 128
// dimensions. The released checkpoints run the exact inverse; the
// first-order form (I - c^2 A) is a DIFFERENT operator there, not an
// approximation of one, and swapping them produces a render that looks
// entirely normal.
//
// The ordered-product spelling -- the WY / DeltaNet chunk form -- is not
// a substitute either, and the reason is worth writing down because the
// two are so nearly the same algorithm. Woodbury gives
//
//     (I + K^T D K)^-1        = I - K^T (I +      D G )^-1 D K
//     prod_s (I - beta k k^T) = I - K^T (I + triu(D G))^-1 D K
//
// (G = K K^T, D = diag(beta)) -- ONE contraction over two masks of the
// same S x S matrix. But a product FACTORISES over chunks and a solve
// over a sum does not: (I + A1 + A2)^-1 != (I + A1)^-1 (I + A2)^-1. The
// triangular mask is exactly what makes the DeltaNet form chunkable, and
// this rule does not have it. Nor does Woodbury help: S >> d_k here, so
// the S x S side is the expensive one.
//
// Which leaves a batched SPD factorisation at d_k = 128. This header is
// the CPU oracle for it.

#include <cstddef>

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace vdn {

// Cholesky of M = I + A, lower factor, in place-free form.
//
// Reads only the LOWER triangle of `a`, which is what
// torch.linalg.cholesky does and therefore what the reference's
// numbers come from. That is not a detail: A is formed as (k*beta)^T k,
// so its (i,j) and (j,i) entries multiply differently-rounded operands
// and the matrix is only symmetric to the working precision. The
// producer symmetrises; this consumer still reads one triangle, so the
// two agree about which matrix was factorised.
//
// False when the matrix is not positive definite -- which for I + A
// (eigenvalues >= 1 in exact arithmetic) means the statistics upstream
// were computed in a precision that lost the property, and is worth a
// loud failure rather than a NaN.
bool cholesky_lower(const float* a, int d, float* l);

// Solve M X = C for X, given M's lower Cholesky factor. C and X are
// [d, cols], column count free. Forward then back substitution.
void cholesky_solve(const float* l, int d, const float* c, int cols,
                    float* x);

// M^-1, formed from the factor as L^-T L^-1.
void cholesky_inverse(const float* l, int d, float* inv);

// The reference's factor_apply, batched over frames and heads.
//
//   transition[f,h] = Diag(alpha[f,h]) (I + A[f,h])^-1     [d_k, d_k]
//   injection[f,h]  = B[f,h] (I + A[f,h])^-1               [d_v, d_k]
//
// alpha scales the ROWS of the inverse, because the scan applies the
// transition on the right of a [d_v, d_k] state.
//
// The explicit inverse is what the reference stores, and the reason is
// that the scan reads each transition twice (once per direction). A
// caller that would rather not materialise it can factor once and call
// cholesky_solve twice instead -- see solve_state below, which is the
// same arithmetic in the shape a Metal port wants.
bool factor_apply(const float* a, const float* b, const float* alpha,
                  int frames, int heads, int d_k, int d_v,
                  float* transition, float* injection);

// One scan step without forming an inverse:
//
//     X (I + A) = S_in Diag(alpha) + B      ->  X
//
// I + A is symmetric, so this is M X^T = (S_in Diag(alpha) + B)^T. Same
// numbers as multiplying by the transition, one fewer d x d product per
// step, and no bank of inverses to hold.
bool solve_state(const float* a, const float* alpha, const float* state_in,
                 const float* b, int d_k, int d_v, float* state_out);

}  // namespace vdn
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
