// ByteTrack tracker stage. The tracker algorithm, the STrack book-
// keeping, and the dense lapjv linear-assignment solver below are
// adapted from FoundationVision/ByteTrack
// (https://github.com/FoundationVision/ByteTrack), MIT-licensed; see
// THIRD_PARTY_LICENSES.md for the verbatim license text. The
// reference uses Eigen for an 8-dimensional Kalman filter and OpenCV
// for box geometry; both dependencies are replaced here by inline
// std::array arithmetic.

#include "stages/vision/byte-track-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/oport-policy.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

// =============================================================
// 8-state Kalman filter for [cx, cy, aspect, h, vx, vy, va, vh].
//
// Replaces the Eigen-backed KalmanFilter in the reference. F has
// the constant-velocity structure [[I_4, dt*I_4], [0_4, I_4]] with
// dt=1, so F·x and F·P·F^T have closed forms (see predict_()).
// H = [I_4 | 0_4], so H·x and H·P·H^T are simple submatrix reads.
// The only non-trivial linear-algebra op is solving a 4x4 SPD
// system during the measurement update; we do that with an inline
// Cholesky.
// =============================================================

using Vec4 = array<float, 4>;
using Vec8 = array<float, 8>;
using Mat4 = array<float, 16>;   // row-major
using Mat8 = array<float, 64>;   // row-major

inline float& m4_(Mat4& m, int r, int c) { return m[r * 4 + c]; }
inline float  m4_(const Mat4& m, int r, int c) { return m[r * 4 + c]; }
inline float& m8_(Mat8& m, int r, int c) { return m[r * 8 + c]; }
inline float  m8_(const Mat8& m, int r, int c) { return m[r * 8 + c]; }

struct KFState {
  Vec8 mean{};
  Mat8 cov{};
};

constexpr float kStdWeightPos = 1.0f / 20.0f;
constexpr float kStdWeightVel = 1.0f / 160.0f;

KFState
kf_initiate_(const Vec4& measurement)
{
  KFState s;
  s.mean[0] = measurement[0];
  s.mean[1] = measurement[1];
  s.mean[2] = measurement[2];
  s.mean[3] = measurement[3];
  // velocities init to 0

  const float h = measurement[3];
  const float std_pos = 2.0f * kStdWeightPos * h;
  const float std_a   = 1e-2f;
  const float std_vel = 10.0f * kStdWeightVel * h;
  const float std_va  = 1e-5f;
  const float diag[8] = {
    std_pos * std_pos,
    std_pos * std_pos,
    std_a   * std_a,
    std_pos * std_pos,
    std_vel * std_vel,
    std_vel * std_vel,
    std_va  * std_va,
    std_vel * std_vel,
  };
  for (int i = 0; i < 8; ++i) {
    m8_(s.cov, i, i) = diag[i];
  }
  return s;
}

// Predict step. With F = [[I,I],[0,I]] (dt=1):
//   mean'_i = mean_i + mean_{i+4}   for i in 0..3
//   mean'_i = mean_i                for i in 4..7
// For P' = F·P·F^T with P partitioned into 4x4 blocks
// [[A,B],[C,D]] (A is P[0..3,0..3], etc.):
//   F·P·F^T = [[A+B+C+D, B+D],
//              [C+D,     D   ]]
// Then add Q (diagonal motion covariance).
void
kf_predict_(Vec8& mean, Mat8& cov)
{
  const float h       = mean[3];
  const float std_pos = kStdWeightPos * h;
  const float std_a   = 1e-2f;
  const float std_vel = kStdWeightVel * h;
  const float std_va  = 1e-5f;
  const float q[8] = {
    std_pos * std_pos,
    std_pos * std_pos,
    std_a   * std_a,
    std_pos * std_pos,
    std_vel * std_vel,
    std_vel * std_vel,
    std_va  * std_va,
    std_vel * std_vel,
  };

  // mean update
  for (int i = 0; i < 4; ++i) {
    mean[i] += mean[i + 4];
  }

  // covariance update using block sums. Read out the four 4x4
  // blocks first since the new top-left depends on all four.
  float A[16], B[16], C[16], D[16];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      A[r * 4 + c] = m8_(cov, r,     c    );
      B[r * 4 + c] = m8_(cov, r,     c + 4);
      C[r * 4 + c] = m8_(cov, r + 4, c    );
      D[r * 4 + c] = m8_(cov, r + 4, c + 4);
    }
  }
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      m8_(cov, r,     c    ) = A[r * 4 + c] + B[r * 4 + c]
                             + C[r * 4 + c] + D[r * 4 + c];
      m8_(cov, r,     c + 4) = B[r * 4 + c] + D[r * 4 + c];
      m8_(cov, r + 4, c    ) = C[r * 4 + c] + D[r * 4 + c];
      m8_(cov, r + 4, c + 4) = D[r * 4 + c];
    }
  }
  for (int i = 0; i < 8; ++i) {
    m8_(cov, i, i) += q[i];
  }
}

// Project to measurement space. Returns the projected mean
// (top 4 entries of state) and the projected 4x4 covariance
// (top-left block of P + diagonal measurement noise R).
struct ProjectOut {
  Vec4 mean;
  Mat4 cov;
};

ProjectOut
kf_project_(const Vec8& mean, const Mat8& cov)
{
  ProjectOut out;
  for (int i = 0; i < 4; ++i) {
    out.mean[i] = mean[i];
  }
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      m4_(out.cov, r, c) = m8_(cov, r, c);
    }
  }
  const float h       = mean[3];
  const float std_pos = kStdWeightPos * h;
  const float std_a   = 1e-1f;
  const float r2[4]   = {
    std_pos * std_pos,
    std_pos * std_pos,
    std_a   * std_a,
    std_pos * std_pos,
  };
  for (int i = 0; i < 4; ++i) {
    m4_(out.cov, i, i) += r2[i];
  }
  return out;
}

// Cholesky factor (lower triangle) of a 4x4 SPD matrix in-place.
// Returns false on numerical breakdown (negative pivot); the caller
// must fall back rather than dividing by zero.
bool
chol4_(Mat4& a)
{
  for (int j = 0; j < 4; ++j) {
    float sum = m4_(a, j, j);
    for (int k = 0; k < j; ++k) {
      sum -= m4_(a, j, k) * m4_(a, j, k);
    }
    if (sum <= 0.0f) {
      return false;
    }
    const float ljj = sqrtf(sum);
    m4_(a, j, j) = ljj;
    for (int i = j + 1; i < 4; ++i) {
      float s = m4_(a, i, j);
      for (int k = 0; k < j; ++k) {
        s -= m4_(a, i, k) * m4_(a, j, k);
      }
      m4_(a, i, j) = s / ljj;
    }
  }
  // Zero out the strict upper triangle so subsequent solve calls
  // don't accidentally read garbage; we only consult the lower
  // triangle, but explicit zeros make the matrix trivially print-
  // and inspect-friendly during debug.
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      m4_(a, i, j) = 0.0f;
    }
  }
  return true;
}

// Solve L·y = b for y, where L is the lower-triangular Cholesky
// factor of a 4x4 SPD matrix (L is in `l`'s lower triangle).
void
chol4_fwd_(const Mat4& l, const float b[4], float y[4])
{
  for (int i = 0; i < 4; ++i) {
    float s = b[i];
    for (int j = 0; j < i; ++j) {
      s -= m4_(l, i, j) * y[j];
    }
    y[i] = s / m4_(l, i, i);
  }
}

// Solve L^T·x = y for x (back-substitution).
void
chol4_back_(const Mat4& l, const float y[4], float x[4])
{
  for (int i = 3; i >= 0; --i) {
    float s = y[i];
    for (int j = i + 1; j < 4; ++j) {
      s -= m4_(l, j, i) * x[j];
    }
    x[i] = s / m4_(l, i, i);
  }
}

// Kalman update. Given prior mean/cov and measurement z (4-vec),
// produce the posterior mean/cov.
//
// Math:
//   S  = H·P·H^T + R   (the projected 4x4)
//   K  = P·H^T·S^-1    (8x4 Kalman gain)
//   x' = x + K·(z - H·x)
//   P' = P - K·S·K^T
//
// P·H^T is the left 8x4 block of P (columns 0..3). We solve
// S·G = (P·H^T)^T row by row (4 solves) to get G = S^-1·(P·H^T)^T;
// K is then G^T (transposed back, 8x4).
void
kf_update_(Vec8& mean, Mat8& cov, const Vec4& z)
{
  ProjectOut p = kf_project_(mean, cov);
  Mat4 L = p.cov;
  if (!chol4_(L)) {
    // Numerical breakdown: fall back to no update. The state
    // continues coasting on prediction alone; in practice this
    // happens only if the covariance has become degenerate.
    return;
  }

  // PH^T = first 4 columns of P, as an 8x4 matrix. Solve, row-by-
  // row in the *PH^T-row* sense, S·g = ph_t_row to get K rows.
  // (K is 8x4; K[r][:] = S^-1 · PH^T[r][:].)
  float K[8][4];
  for (int r = 0; r < 8; ++r) {
    float ph_t_row[4];
    for (int c = 0; c < 4; ++c) {
      ph_t_row[c] = m8_(cov, r, c);
    }
    float y[4];
    chol4_fwd_(L, ph_t_row, y);
    chol4_back_(L, y, K[r]);
  }

  // Innovation in measurement space.
  float innov[4];
  for (int i = 0; i < 4; ++i) {
    innov[i] = z[i] - p.mean[i];
  }

  // mean' = mean + K·innov.
  for (int r = 0; r < 8; ++r) {
    float s = 0.0f;
    for (int c = 0; c < 4; ++c) {
      s += K[r][c] * innov[c];
    }
    mean[r] += s;
  }

  // cov' = cov - K·S·K^T. Compute KS = K·S first (8x4), then
  // (K·S)·K^T to get the 8x8 to subtract.
  float KS[8][4];
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 4; ++c) {
      float s = 0.0f;
      for (int k = 0; k < 4; ++k) {
        s += K[r][k] * m4_(p.cov, k, c);
      }
      KS[r][c] = s;
    }
  }
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      float s = 0.0f;
      for (int k = 0; k < 4; ++k) {
        s += KS[r][k] * K[c][k];
      }
      m8_(cov, r, c) -= s;
    }
  }
}

// =============================================================
// STrack
// =============================================================

enum TrackState { kNew = 0, kTracked, kLost, kRemoved };

struct STrack {
  // tlwh as observed at construction time (frozen).
  Vec4 _tlwh{};
  // Current tlwh derived from the KF mean (or _tlwh while New).
  Vec4 tlwh{};
  // Current tlbr derived from tlwh (cached).
  Vec4 tlbr{};

  // Kalman state. mean[0..3] is [cx, cy, a, h]; mean[4..7] is the
  // matching velocity vector. cov is the 8x8 covariance.
  Vec8 mean{};
  Mat8 cov{};

  bool is_activated = false;
  int  track_id     = 0;
  int  state        = kNew;

  int   frame_id     = 0;
  int   tracklet_len = 0;
  int   start_frame  = 0;
  float score        = 0.0f;
  int   class_id     = -1;
};

Vec4
tlbr_to_tlwh_(const Vec4& tlbr)
{
  return { tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1] };
}

Vec4
tlwh_to_xyah_(const Vec4& tlwh)
{
  // (top-left x,y,w,h) -> (center x, center y, aspect, h)
  const float h = tlwh[3];
  return {
    tlwh[0] + tlwh[2] * 0.5f,
    tlwh[1] + tlwh[3] * 0.5f,
    h > 0.0f ? tlwh[2] / h : 0.0f,
    h,
  };
}

void
strack_refresh_tlwh_(STrack& t)
{
  if (t.state == kNew) {
    t.tlwh = t._tlwh;
  } else {
    // mean is [cx, cy, a, h] -> tlwh
    t.tlwh[3] = t.mean[3];
    t.tlwh[2] = t.mean[2] * t.tlwh[3];
    t.tlwh[0] = t.mean[0] - t.tlwh[2] * 0.5f;
    t.tlwh[1] = t.mean[1] - t.tlwh[3] * 0.5f;
  }
  t.tlbr = {
    t.tlwh[0],
    t.tlwh[1],
    t.tlwh[0] + t.tlwh[2],
    t.tlwh[1] + t.tlwh[3],
  };
}

// Per-stage track-id counter. Stored on the BYTETracker instance
// (see ByteTracker below) rather than as a function-local static
// so that two pipelines (or two stage instances) don't interleave
// ids.
struct TrackIdCounter {
  int next() { return ++_n; }
  int _n = 0;
};

void
strack_activate_(STrack& t, TrackIdCounter& ids, int frame_id)
{
  t.track_id = ids.next();
  const Vec4 xyah = tlwh_to_xyah_(t._tlwh);
  KFState s = kf_initiate_(xyah);
  t.mean = s.mean;
  t.cov  = s.cov;
  t.state = kTracked;
  t.tracklet_len = 0;
  t.is_activated = (frame_id == 1);
  t.frame_id = frame_id;
  t.start_frame = frame_id;
  strack_refresh_tlwh_(t);
}

void
strack_re_activate_(STrack& self, const STrack& new_track,
                    TrackIdCounter& ids, int frame_id, bool new_id)
{
  const Vec4 xyah = tlwh_to_xyah_(new_track.tlwh);
  kf_update_(self.mean, self.cov, xyah);
  self.tracklet_len = 0;
  self.state = kTracked;
  self.is_activated = true;
  self.frame_id = frame_id;
  self.score = new_track.score;
  self.class_id = new_track.class_id;
  if (new_id) {
    self.track_id = ids.next();
  }
  strack_refresh_tlwh_(self);
}

void
strack_update_(STrack& self, const STrack& new_track, int frame_id)
{
  self.frame_id = frame_id;
  self.tracklet_len += 1;
  const Vec4 xyah = tlwh_to_xyah_(new_track.tlwh);
  kf_update_(self.mean, self.cov, xyah);
  self.state = kTracked;
  self.is_activated = true;
  self.score = new_track.score;
  self.class_id = new_track.class_id;
  strack_refresh_tlwh_(self);
}

void
strack_multi_predict_(vector<STrack*>& tracks)
{
  for (STrack* t : tracks) {
    if (t->state != kTracked) {
      // Zero out the height-velocity component for non-Tracked
      // states (the reference does the same), so a lost track
      // doesn't keep shrinking.
      t->mean[7] = 0.0f;
    }
    kf_predict_(t->mean, t->cov);
    strack_refresh_tlwh_(*t);
  }
}

// =============================================================
// IoU & cost matrix
// =============================================================

float
box_iou_(const Vec4& a, const Vec4& b)
{
  // The reference utils.cpp adds +1 to widths/heights when
  // computing IoU (a leftover from Faster-R-CNN-style pixel
  // counting). We keep the +1 for parity so the cost matrix
  // values match the reference behaviour byte-for-byte.
  const float iw =
      min(a[2], b[2]) - max(a[0], b[0]) + 1.0f;
  if (iw <= 0.0f) { return 0.0f; }
  const float ih =
      min(a[3], b[3]) - max(a[1], b[1]) + 1.0f;
  if (ih <= 0.0f) { return 0.0f; }
  const float inter = iw * ih;
  const float aa = (a[2] - a[0] + 1.0f) * (a[3] - a[1] + 1.0f);
  const float bb = (b[2] - b[0] + 1.0f) * (b[3] - b[1] + 1.0f);
  const float uni = aa + bb - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

// cost[i][j] = 1 - IoU(a[i], b[j]). Returns empty matrix if either
// side is empty.
vector<vector<float>>
iou_cost_(const vector<Vec4>& a, const vector<Vec4>& b)
{
  vector<vector<float>> out;
  if (a.empty() || b.empty()) {
    return out;
  }
  out.assign(a.size(), vector<float>(b.size(), 1.0f));
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < b.size(); ++j) {
      out[i][j] = 1.0f - box_iou_(a[i], b[j]);
    }
  }
  return out;
}

// =============================================================
// LapJV solver. Dense Jonker-Volgenant variant. Adapted directly
// from FoundationVision/ByteTrack's deploy/ncnn/cpp/src/lapjv.cpp
// (MIT-licensed). The interface uses std::vector<vector<double>>
// instead of raw double**; behaviour and step structure are
// preserved.
// =============================================================

constexpr double kLapLarge = 1e9;

int
lap_ccrrt_dense_(int n, const vector<vector<double>>& cost,
                 vector<int>& free_rows, vector<int>& x,
                 vector<int>& y, vector<double>& v)
{
  for (int i = 0; i < n; ++i) {
    x[i] = -1;
    v[i] = kLapLarge;
    y[i] = 0;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const double c = cost[i][j];
      if (c < v[j]) {
        v[j] = c;
        y[j] = i;
      }
    }
  }
  vector<char> unique(n, 1);
  for (int j = n - 1; j >= 0; --j) {
    const int i = y[j];
    if (x[i] < 0) {
      x[i] = j;
    } else {
      unique[i] = 0;
      y[j] = -1;
    }
  }
  int n_free = 0;
  for (int i = 0; i < n; ++i) {
    if (x[i] < 0) {
      free_rows[n_free++] = i;
    } else if (unique[i]) {
      const int j = x[i];
      double minv = kLapLarge;
      for (int j2 = 0; j2 < n; ++j2) {
        if (j2 == j) { continue; }
        const double c = cost[i][j2] - v[j2];
        if (c < minv) { minv = c; }
      }
      v[j] -= minv;
    }
  }
  return n_free;
}

int
lap_carr_dense_(int n, const vector<vector<double>>& cost,
                int n_free_rows, vector<int>& free_rows,
                vector<int>& x, vector<int>& y, vector<double>& v)
{
  int current = 0;
  int new_free_rows = 0;
  int rr_cnt = 0;
  while (current < n_free_rows) {
    rr_cnt += 1;
    const int free_i = free_rows[current++];
    int j1 = 0;
    double v1 = cost[free_i][0] - v[0];
    int j2 = -1;
    double v2 = kLapLarge;
    for (int j = 1; j < n; ++j) {
      const double c = cost[free_i][j] - v[j];
      if (c < v2) {
        if (c >= v1) {
          v2 = c;
          j2 = j;
        } else {
          v2 = v1;
          v1 = c;
          j2 = j1;
          j1 = j;
        }
      }
    }
    int i0 = y[j1];
    const double v1_new = v[j1] - (v2 - v1);
    const bool   v1_lower = v1_new < v[j1];
    if (rr_cnt < current * n) {
      if (v1_lower) {
        v[j1] = v1_new;
      } else if (i0 >= 0 && j2 >= 0) {
        j1 = j2;
        i0 = y[j2];
      }
      if (i0 >= 0) {
        if (v1_lower) {
          free_rows[--current] = i0;
        } else {
          free_rows[new_free_rows++] = i0;
        }
      }
    } else {
      if (i0 >= 0) {
        free_rows[new_free_rows++] = i0;
      }
    }
    x[free_i] = j1;
    y[j1] = free_i;
  }
  return new_free_rows;
}

int
lap_find_dense_(int n, int lo, const vector<double>& d,
                vector<int>& cols)
{
  int hi = lo + 1;
  double mind = d[cols[lo]];
  for (int k = hi; k < n; ++k) {
    int j = cols[k];
    if (d[j] <= mind) {
      if (d[j] < mind) {
        hi = lo;
        mind = d[j];
      }
      cols[k] = cols[hi];
      cols[hi++] = j;
    }
  }
  return hi;
}

int
lap_scan_dense_(int n, const vector<vector<double>>& cost,
                int& plo, int& phi, vector<double>& d,
                vector<int>& cols, vector<int>& pred,
                const vector<int>& y, const vector<double>& v)
{
  int lo = plo;
  int hi = phi;
  while (lo != hi) {
    int j = cols[lo++];
    const int i = y[j];
    const double mind = d[j];
    const double h = cost[i][j] - v[j] - mind;
    for (int k = hi; k < n; ++k) {
      j = cols[k];
      const double cred_ij = cost[i][j] - v[j] - h;
      if (cred_ij < d[j]) {
        d[j] = cred_ij;
        pred[j] = i;
        if (cred_ij == mind) {
          if (y[j] < 0) {
            plo = lo;
            phi = hi;
            return j;
          }
          cols[k] = cols[hi];
          cols[hi++] = j;
        }
      }
    }
  }
  plo = lo;
  phi = hi;
  return -1;
}

int
lap_find_path_dense_(int n, const vector<vector<double>>& cost,
                     int start_i, const vector<int>& y,
                     vector<double>& v, vector<int>& pred)
{
  int lo = 0, hi = 0;
  int final_j = -1;
  int n_ready = 0;
  vector<int>    cols(n);
  vector<double> d(n);
  for (int i = 0; i < n; ++i) {
    cols[i] = i;
    pred[i] = start_i;
    d[i] = cost[start_i][i] - v[i];
  }
  while (final_j == -1) {
    if (lo == hi) {
      n_ready = lo;
      hi = lap_find_dense_(n, lo, d, cols);
      for (int k = lo; k < hi; ++k) {
        const int j = cols[k];
        if (y[j] < 0) {
          final_j = j;
        }
      }
    }
    if (final_j == -1) {
      final_j = lap_scan_dense_(n, cost, lo, hi, d, cols, pred, y, v);
    }
  }
  const double mind = d[cols[lo]];
  for (int k = 0; k < n_ready; ++k) {
    const int j = cols[k];
    v[j] += d[j] - mind;
  }
  return final_j;
}

int
lap_ca_dense_(int n, const vector<vector<double>>& cost,
              int n_free_rows, const vector<int>& free_rows,
              vector<int>& x, vector<int>& y, vector<double>& v)
{
  vector<int> pred(n);
  for (int p = 0; p < n_free_rows; ++p) {
    int i = -1;
    int j = lap_find_path_dense_(n, cost, free_rows[p], y, v, pred);
    if (j < 0 || j >= n) { return -1; }
    int k = 0;
    while (i != free_rows[p]) {
      i = pred[j];
      y[j] = i;
      std::swap(j, x[i]);
      k += 1;
      if (k >= n) { return -1; }
    }
  }
  return 0;
}

// Solve the LAP on a square cost matrix of size n x n. On success
// fills x[i] (column assigned to row i) and y[j] (row assigned to
// column j) and returns 0; on failure returns nonzero. Pure
// algorithm port — no I/O.
int
lapjv_internal_(int n, const vector<vector<double>>& cost,
                vector<int>& x, vector<int>& y)
{
  vector<int>    free_rows(n);
  vector<double> v(n);
  int ret = lap_ccrrt_dense_(n, cost, free_rows, x, y, v);
  int i = 0;
  while (ret > 0 && i < 2) {
    ret = lap_carr_dense_(n, cost, ret, free_rows, x, y, v);
    i += 1;
  }
  if (ret > 0) {
    ret = lap_ca_dense_(n, cost, ret, free_rows, x, y, v);
  }
  return ret;
}

// =============================================================
// Cost-bounded linear assignment over a rectangular matrix.
// Extends to square via dummy rows/columns (per the reference),
// then accepts matches only when their original cost is below
// thresh. Outputs row->col and col->row arrays with -1 markers
// for unmatched.
// =============================================================
void
linear_assignment_(const vector<vector<float>>& cost,
                   int n_rows, int n_cols, float thresh,
                   vector<pair<int, int>>* matches,
                   vector<int>* unmatched_rows,
                   vector<int>* unmatched_cols)
{
  matches->clear();
  unmatched_rows->clear();
  unmatched_cols->clear();

  if (cost.empty() || n_rows == 0 || n_cols == 0) {
    for (int i = 0; i < n_rows; ++i) { unmatched_rows->push_back(i); }
    for (int j = 0; j < n_cols; ++j) { unmatched_cols->push_back(j); }
    return;
  }

  // Build an (n_rows + n_cols) square extended matrix. Real
  // entries in the top-left, thresh/2 fill in the
  // dummy-rows/dummy-cols cross blocks, 0 in the bottom-right
  // dummy/dummy block. Matches landing on a dummy row or column
  // mean "no real assignment for that row/col."
  const int n = n_rows + n_cols;
  vector<vector<double>> ext(n, vector<double>(n,
      static_cast<double>(thresh) * 0.5));
  for (int i = 0; i < n_rows; ++i) {
    for (int j = 0; j < n_cols; ++j) {
      ext[i][j] = static_cast<double>(cost[i][j]);
    }
  }
  for (int i = n_rows; i < n; ++i) {
    for (int j = n_cols; j < n; ++j) {
      ext[i][j] = 0.0;
    }
  }

  vector<int> x(n, -1);
  vector<int> y(n, -1);
  if (lapjv_internal_(n, ext, x, y) != 0) {
    // Solver failure: degrade gracefully (all unmatched). This is
    // very rare; the ref aborts the process here.
    for (int i = 0; i < n_rows; ++i) { unmatched_rows->push_back(i); }
    for (int j = 0; j < n_cols; ++j) { unmatched_cols->push_back(j); }
    return;
  }

  for (int i = 0; i < n_rows; ++i) {
    int j = x[i];
    if (j < 0 || j >= n_cols) {
      unmatched_rows->push_back(i);
    } else {
      matches->emplace_back(i, j);
    }
  }
  for (int j = 0; j < n_cols; ++j) {
    int i = y[j];
    if (i < 0 || i >= n_rows) {
      unmatched_cols->push_back(j);
    }
  }
}

// =============================================================
// Joint / sub / dedupe helpers.
// =============================================================

vector<STrack*>
joint_stracks_(const vector<STrack*>& a, vector<STrack>& b)
{
  map<int, int> seen;
  vector<STrack*> out;
  out.reserve(a.size() + b.size());
  for (STrack* t : a) {
    seen[t->track_id] = 1;
    out.push_back(t);
  }
  for (auto& t : b) {
    if (!seen.count(t.track_id)) {
      seen[t.track_id] = 1;
      out.push_back(&t);
    }
  }
  return out;
}

vector<STrack>
joint_stracks_(const vector<STrack>& a, const vector<STrack>& b)
{
  map<int, int> seen;
  vector<STrack> out;
  out.reserve(a.size() + b.size());
  for (const auto& t : a) {
    seen[t.track_id] = 1;
    out.push_back(t);
  }
  for (const auto& t : b) {
    if (!seen.count(t.track_id)) {
      seen[t.track_id] = 1;
      out.push_back(t);
    }
  }
  return out;
}

vector<STrack>
sub_stracks_(const vector<STrack>& a, const vector<STrack>& b)
{
  map<int, STrack> by_id;
  for (const auto& t : a) {
    by_id.emplace(t.track_id, t);
  }
  for (const auto& t : b) {
    by_id.erase(t.track_id);
  }
  vector<STrack> out;
  out.reserve(by_id.size());
  for (auto& kv : by_id) {
    out.push_back(std::move(kv.second));
  }
  return out;
}

void
remove_duplicate_stracks_(vector<STrack>& a, vector<STrack>& b)
{
  vector<Vec4> tlbr_a, tlbr_b;
  tlbr_a.reserve(a.size());
  tlbr_b.reserve(b.size());
  for (auto& t : a) { tlbr_a.push_back(t.tlbr); }
  for (auto& t : b) { tlbr_b.push_back(t.tlbr); }

  vector<vector<float>> cost = iou_cost_(tlbr_a, tlbr_b);
  vector<char> dup_a(a.size(), 0);
  vector<char> dup_b(b.size(), 0);
  for (size_t i = 0; i < cost.size(); ++i) {
    for (size_t j = 0; j < cost[i].size(); ++j) {
      if (cost[i][j] < 0.15f) {
        const int age_a = a[i].frame_id - a[i].start_frame;
        const int age_b = b[j].frame_id - b[j].start_frame;
        if (age_a > age_b) {
          dup_b[j] = 1;
        } else {
          dup_a[i] = 1;
        }
      }
    }
  }
  vector<STrack> ra, rb;
  ra.reserve(a.size());
  rb.reserve(b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (!dup_a[i]) { ra.push_back(std::move(a[i])); }
  }
  for (size_t j = 0; j < b.size(); ++j) {
    if (!dup_b[j]) { rb.push_back(std::move(b[j])); }
  }
  a = std::move(ra);
  b = std::move(rb);
}

// =============================================================
// ByteTracker: the main update() loop. Mirrors the reference
// implementation step-for-step; comments below mark each step.
// =============================================================

struct Detection {
  Vec4  tlbr;
  float score;
  int   class_id;
};

class ByteTracker {
public:
  ByteTracker(int frame_rate, int track_buffer,
              float track_thresh, float high_thresh,
              float match_thresh)
    : _track_thresh(track_thresh)
    , _high_thresh(high_thresh)
    , _match_thresh(match_thresh)
    , _max_time_lost(static_cast<int>(
          static_cast<float>(frame_rate) / 30.0f
        * static_cast<float>(track_buffer)))
  {}

  // Update with one frame's worth of detections. Returns the
  // tracker's currently-confirmed STrack snapshots.
  vector<STrack>
  update(const vector<Detection>& dets)
  {
    _frame_id += 1;

    // Step 1: Split detections by score and stage them as STracks
    // (one short-lived STrack per detection — these are not the
    // tracker's bookkeeping objects, just measurement carriers).
    vector<STrack> hi, lo;
    hi.reserve(dets.size());
    for (const auto& d : dets) {
      STrack s;
      s._tlwh = tlbr_to_tlwh_(d.tlbr);
      s.tlwh  = s._tlwh;
      s.tlbr  = d.tlbr;
      s.score = d.score;
      s.class_id = d.class_id;
      if (d.score >= _track_thresh) {
        hi.push_back(s);
      } else {
        lo.push_back(s);
      }
    }

    // Partition the existing book into "confirmed tracked" and
    // "unconfirmed" (track that only had a New activation).
    vector<STrack*> unconfirmed;
    vector<STrack*> tracked_ptrs;
    for (auto& t : _tracked) {
      if (t.is_activated) {
        tracked_ptrs.push_back(&t);
      } else {
        unconfirmed.push_back(&t);
      }
    }

    // Step 2: First association — high-score detections against
    // active tracks AND lost tracks, IoU + lapjv.
    vector<STrack*> pool = joint_stracks_(tracked_ptrs, _lost);
    strack_multi_predict_(pool);

    vector<Vec4> pool_tlbr;
    pool_tlbr.reserve(pool.size());
    for (auto* t : pool) { pool_tlbr.push_back(t->tlbr); }
    vector<Vec4> hi_tlbr;
    hi_tlbr.reserve(hi.size());
    for (auto& t : hi) { hi_tlbr.push_back(t.tlbr); }

    vector<vector<float>> dists = iou_cost_(pool_tlbr, hi_tlbr);
    vector<pair<int, int>> matches;
    vector<int> u_track, u_det;
    linear_assignment_(dists, static_cast<int>(pool.size()),
                       static_cast<int>(hi.size()),
                       _match_thresh, &matches, &u_track, &u_det);

    vector<STrack> activated;
    vector<STrack> refind;
    vector<STrack> new_lost;
    vector<STrack> new_removed;

    for (auto& m : matches) {
      STrack* track = pool[m.first];
      STrack& det   = hi[m.second];
      if (track->state == kTracked) {
        strack_update_(*track, det, _frame_id);
        activated.push_back(*track);
      } else {
        strack_re_activate_(*track, det, _ids, _frame_id, false);
        refind.push_back(*track);
      }
    }

    // Step 3: Second association — unmatched still-tracked tracks
    // against low-score detections (the ByteTrack trick).
    vector<STrack> det_remainder;
    det_remainder.reserve(u_det.size());
    for (int idx : u_det) {
      det_remainder.push_back(hi[idx]);
    }
    vector<STrack*> r_tracked;
    for (int idx : u_track) {
      if (pool[idx]->state == kTracked) {
        r_tracked.push_back(pool[idx]);
      }
    }

    vector<Vec4> r_tlbr, lo_tlbr;
    r_tlbr.reserve(r_tracked.size());
    for (auto* t : r_tracked) { r_tlbr.push_back(t->tlbr); }
    lo_tlbr.reserve(lo.size());
    for (auto& t : lo) { lo_tlbr.push_back(t.tlbr); }

    dists = iou_cost_(r_tlbr, lo_tlbr);
    matches.clear();
    u_track.clear();
    vector<int> u_det2;
    linear_assignment_(dists, static_cast<int>(r_tracked.size()),
                       static_cast<int>(lo.size()), 0.5f, &matches,
                       &u_track, &u_det2);
    for (auto& m : matches) {
      STrack* track = r_tracked[m.first];
      STrack& det   = lo[m.second];
      if (track->state == kTracked) {
        strack_update_(*track, det, _frame_id);
        activated.push_back(*track);
      } else {
        strack_re_activate_(*track, det, _ids, _frame_id, false);
        refind.push_back(*track);
      }
    }
    for (int idx : u_track) {
      STrack* track = r_tracked[idx];
      if (track->state != kLost) {
        track->state = kLost;
        new_lost.push_back(*track);
      }
    }

    // Unconfirmed pass: match remaining high-score unmatched
    // detections against unconfirmed tracks.
    vector<Vec4> u_tlbr, drem_tlbr;
    u_tlbr.reserve(unconfirmed.size());
    for (auto* t : unconfirmed) { u_tlbr.push_back(t->tlbr); }
    drem_tlbr.reserve(det_remainder.size());
    for (auto& t : det_remainder) { drem_tlbr.push_back(t.tlbr); }

    dists = iou_cost_(u_tlbr, drem_tlbr);
    matches.clear();
    vector<int> u_unconfirmed;
    vector<int> u_det3;
    linear_assignment_(dists, static_cast<int>(unconfirmed.size()),
                       static_cast<int>(det_remainder.size()),
                       0.7f, &matches, &u_unconfirmed, &u_det3);
    for (auto& m : matches) {
      strack_update_(*unconfirmed[m.first], det_remainder[m.second],
                     _frame_id);
      activated.push_back(*unconfirmed[m.first]);
    }
    for (int idx : u_unconfirmed) {
      STrack* track = unconfirmed[idx];
      track->state = kRemoved;
      new_removed.push_back(*track);
    }

    // Step 4: Init new tracks from leftover unmatched high-score
    // detections, gated by high_thresh.
    for (int idx : u_det3) {
      STrack& d = det_remainder[idx];
      if (d.score < _high_thresh) { continue; }
      strack_activate_(d, _ids, _frame_id);
      activated.push_back(d);
    }

    // Step 5: Retire stale lost tracks.
    for (auto& t : _lost) {
      if (_frame_id - t.frame_id > _max_time_lost) {
        t.state = kRemoved;
        new_removed.push_back(t);
      }
    }

    // Rebuild _tracked from the still-Tracked entries plus the
    // newly-activated/refound tracks.
    vector<STrack> next_tracked;
    next_tracked.reserve(_tracked.size());
    for (auto& t : _tracked) {
      if (t.state == kTracked) {
        next_tracked.push_back(std::move(t));
      }
    }
    _tracked = std::move(next_tracked);
    _tracked = joint_stracks_(_tracked, activated);
    _tracked = joint_stracks_(_tracked, refind);

    _lost = sub_stracks_(_lost, _tracked);
    for (auto& t : new_lost) {
      _lost.push_back(std::move(t));
    }
    _lost = sub_stracks_(_lost, _removed);
    // _removed only needs to carry the most recent frame's removals:
    // it is consumed exactly once (the sub_stracks_ above, on the
    // FOLLOWING frame) to drop just-removed tracks from _lost, and a
    // removed track never re-enters _lost. Accumulating it across all
    // frames -- as the reference ByteTrack's removed_stracks.extend()
    // does -- is fine for short clips but an unbounded heap leak on an
    // indefinite RTSP stream (one ~640-byte STrack per retired track,
    // forever). Replace, don't append.
    _removed = std::move(new_removed);

    remove_duplicate_stracks_(_tracked, _lost);

    vector<STrack> out;
    out.reserve(_tracked.size());
    for (auto& t : _tracked) {
      if (t.is_activated) {
        out.push_back(t);
      }
    }
    return out;
  }

  int frame_id() const noexcept { return _frame_id; }

private:
  float _track_thresh;
  float _high_thresh;
  float _match_thresh;
  int   _max_time_lost;
  int   _frame_id = 0;

  vector<STrack>   _tracked;
  vector<STrack>   _lost;
  vector<STrack>   _removed;
  TrackIdCounter   _ids;
};

}  // namespace

// =============================================================
// Stage glue.
// =============================================================

struct ByteTrackStage::Impl {
  // Config attribute values; defaults live in kSpec.attrs and are read
  // in the constructor via attr_*. Declarations carry no non-zero
  // default.
  float    track_thresh{};
  float    high_thresh{};
  float    match_thresh{};
  int      frame_rate{};
  int      track_buffer{};
  unsigned oport_capacity{};

  unique_ptr<ByteTracker> tracker;
};

ByteTrackStage::ByteTrackStage(const SessionContextIntf* s,
                               string                    id,
                               vector<InEdge>            iports,
                               FlexData                  config)
  : TypedStage<ByteTrackStage>(s, std::move(id), std::move(iports),
                               std::move(config))
  , _impl(make_unique<Impl>())
{
  // Attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _impl->track_thresh   = static_cast<float>(attr_real("track_thresh"));
  _impl->high_thresh    = static_cast<float>(attr_real("high_thresh"));
  _impl->match_thresh   = static_cast<float>(attr_real("match_thresh"));
  _impl->frame_rate     = static_cast<int>(attr_int("frame_rate"));
  _impl->track_buffer   = static_cast<int>(attr_int("track_buffer"));
  _impl->oport_capacity =
      static_cast<unsigned>(attr_uint("oport_capacity"));

  // Validation is deferred to launch (see Stage::fail_config).
  if (!(_impl->track_thresh > 0.0f)) {
    fail_config(fmt(
        "ByteTrackStage('{}'): track_thresh must be > 0 (got {})",
        this->id(), _impl->track_thresh));
  }
  if (!(_impl->high_thresh > 0.0f)) {
    fail_config(fmt(
        "ByteTrackStage('{}'): high_thresh must be > 0 (got {})",
        this->id(), _impl->high_thresh));
  }
  if (!(_impl->match_thresh > 0.0f)) {
    fail_config(fmt(
        "ByteTrackStage('{}'): match_thresh must be > 0 (got {})",
        this->id(), _impl->match_thresh));
  }
  if (_impl->frame_rate <= 0) {
    fail_config(fmt(
        "ByteTrackStage('{}'): frame_rate must be > 0 (got {})",
        this->id(), _impl->frame_rate));
  }
  if (_impl->track_buffer <= 0) {
    fail_config(fmt(
        "ByteTrackStage('{}'): track_buffer must be > 0 (got {})",
        this->id(), _impl->track_buffer));
  }
  if (_impl->oport_capacity == 0) {
    _impl->oport_capacity = 4;
  }

  // Only build the tracker for a valid config; a stage with a config
  // error is skipped at launch so process() (the only user of the
  // tracker) never runs.
  if (config_error().empty()) {
    _impl->tracker = make_unique<ByteTracker>(
        _impl->frame_rate, _impl->track_buffer,
        _impl->track_thresh, _impl->high_thresh,
        _impl->match_thresh);
  }

  allocate_oports(1);
  set_oport_policy(0,
      {_impl->oport_capacity, OverrunPolicy::Backpressure});
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "track_thresh", .type = ConfigType::Real,
   .doc = "score >= this enters the high-score pool", .def_real = 0.5},
  {.key = "high_thresh", .type = ConfigType::Real,
   .doc = "bar an unmatched det must clear to spawn a track",
   .def_real = 0.6},
  {.key = "match_thresh", .type = ConfigType::Real,
   .doc = "cost cap (1 - IoU) for first association", .def_real = 0.8},
  {.key = "frame_rate", .type = ConfigType::Int,
   .doc = "with track_buffer, derives max lost frames", .def_int = 30},
  {.key = "track_buffer", .type = ConfigType::Int,
   .doc = "frames a lost track waits before removal", .def_int = 30},
  {.key = "oport_capacity", .type = ConfigType::Uint,
   .doc = "output ring capacity", .def_uint = 4},
};
const PortSpec kIports[] = {
  {.name = "detections", .doc = "FlexData yolo-detection frame record "
                                "{frame_width,frame_height,detections[]}",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "tracks", .doc = "same FlexData shape; confirmed detections "
                            "only, each enriched with an integer track_id",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "byte-track",
  .doc       = "ByteTrack multi-object tracker: assigns persistent "
               "track_ids to yolo detections across frames (Kalman + "
               "two-stage IoU/Hungarian). Forwards confirmed detections.",
  .display_name = "ByteTrack",
  .category  = StageCategory::Video,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
ByteTrackStage::spec() const noexcept
{
  return kSpec;
}

ByteTrackStage::~ByteTrackStage() = default;

Job
ByteTrackStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  if (!in) {
    ctx.signal_done();
    co_return;
  }
  const FlexDataPayload* fdp =
      dynamic_cast<const FlexDataPayload*>(in.get());
  if (!fdp || !fdp->data.is_object()) {
    session()->warn(fmt(
        "ByteTrackStage('{}'): ignoring non-object FlexData beat",
        this->id()));
    co_return;
  }

  const FlexData& fd = fdp->data;
  auto root = fd.as_object();

  // Pull detections into Detection structs. Drop entries that are
  // missing geometry; pass everything else (including very-low
  // score detections — the tracker uses them for the second
  // association pass).
  vector<Detection> dets;
  if (root.contains("detections")) {
    FlexData dfd = root.at("detections");
    if (dfd.is_array()) {
      auto av = dfd.as_array();
      dets.reserve(av.size());
      for (size_t i = 0, n = av.size(); i < n; ++i) {
        FlexData entry = av.at(i);
        if (!entry.is_object()) { continue; }
        auto e = entry.as_object();
        if (!e.contains("x1") || !e.contains("y1")
            || !e.contains("x2") || !e.contains("y2")) {
          continue;
        }
        Detection d;
        d.tlbr[0] = static_cast<float>(e.at("x1").as_real(0.0));
        d.tlbr[1] = static_cast<float>(e.at("y1").as_real(0.0));
        d.tlbr[2] = static_cast<float>(e.at("x2").as_real(0.0));
        d.tlbr[3] = static_cast<float>(e.at("y2").as_real(0.0));
        d.score   = e.contains("score")
                  ? static_cast<float>(e.at("score").as_real(0.0))
                  : 0.0f;
        d.class_id = e.contains("class_id")
                   ? static_cast<int>(e.at("class_id").as_int(-1))
                   : -1;
        if (!(d.tlbr[2] > d.tlbr[0]) || !(d.tlbr[3] > d.tlbr[1])) {
          continue;   // degenerate
        }
        dets.push_back(d);
      }
    }
  }

  vector<STrack> tracks = _impl->tracker->update(dets);

  // Emit a FlexData record mirroring the upstream shape, with
  // each surviving detection enriched with `track_id` and the
  // tracker-smoothed bbox. We pull class_id / class_name back
  // from the matched input detection by IoU — that's the cheapest
  // way to preserve labels without forcing class_id through the
  // tracker.
  FlexData out_fd = FlexData::make_object();
  auto out_root = out_fd.as_object();
  if (root.contains("frame_width")) {
    out_root.insert("frame_width",
        FlexData::make_int(root.at("frame_width").as_int(0)));
  }
  if (root.contains("frame_height")) {
    out_root.insert("frame_height",
        FlexData::make_int(root.at("frame_height").as_int(0)));
  }

  FlexData out_dets = FlexData::make_array();
  auto out_dets_view = out_dets.as_array();
  out_dets_view.reserve(tracks.size());

  // Re-pull the input detections so we can echo class_name back.
  vector<FlexData> in_entries;
  if (root.contains("detections")) {
    auto dfd = root.at("detections");
    if (dfd.is_array()) {
      auto av = dfd.as_array();
      in_entries.reserve(av.size());
      for (size_t i = 0, n = av.size(); i < n; ++i) {
        in_entries.push_back(av.at(i));
      }
    }
  }

  for (auto& t : tracks) {
    // Snap the smoothed tlbr to the original input box (highest
    // IoU) so class_id / class_name survive even when the tracker
    // emits a slightly drifted box.
    int    best = -1;
    float  best_iou = 0.0f;
    for (size_t i = 0; i < in_entries.size(); ++i) {
      FlexData entry = in_entries[i];
      if (!entry.is_object()) { continue; }
      auto e = entry.as_object();
      if (!e.contains("x1")) { continue; }
      Vec4 b = {
        static_cast<float>(e.at("x1").as_real(0.0)),
        static_cast<float>(e.at("y1").as_real(0.0)),
        static_cast<float>(e.at("x2").as_real(0.0)),
        static_cast<float>(e.at("y2").as_real(0.0)),
      };
      const float iou = box_iou_(t.tlbr, b);
      if (iou > best_iou) {
        best_iou = iou;
        best     = static_cast<int>(i);
      }
    }

    FlexData entry = FlexData::make_object();
    auto out_e = entry.as_object();
    out_e.insert("track_id", FlexData::make_int(t.track_id));
    out_e.insert("score",    FlexData::make_real(t.score));
    out_e.insert("x1",       FlexData::make_real(t.tlbr[0]));
    out_e.insert("y1",       FlexData::make_real(t.tlbr[1]));
    out_e.insert("x2",       FlexData::make_real(t.tlbr[2]));
    out_e.insert("y2",       FlexData::make_real(t.tlbr[3]));
    if (best >= 0) {
      auto in_e = in_entries[static_cast<size_t>(best)].as_object();
      if (in_e.contains("class_id")) {
        out_e.insert("class_id",
            FlexData::make_int(in_e.at("class_id").as_int(-1)));
      } else if (t.class_id >= 0) {
        out_e.insert("class_id", FlexData::make_int(t.class_id));
      }
      if (in_e.contains("class_name")) {
        out_e.insert("class_name", FlexData::make_string(
            in_e.at("class_name").as_string("")));
      }
    } else if (t.class_id >= 0) {
      out_e.insert("class_id", FlexData::make_int(t.class_id));
    }
    out_dets_view.push_back(std::move(entry));
  }

  out_root.insert("detections", std::move(out_dets));

  co_await ctx.write(0,
      make_payload<FlexDataPayload>(std::move(out_fd)));
}

VPIPE_REGISTER_STAGE(ByteTrackStage)
VPIPE_REGISTER_SPEC(ByteTrackStage, kSpec)

}
