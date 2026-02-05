// linear_cuda_demo.cu
// Forward of torch.nn.Linear: Y = X @ W^T + b
// X[B,K], W[N,K], b[N] -> Y[B,N]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>

#define CHECK_CUDA(cmd) do {                                      \
  cudaError_t e = (cmd);                                          \
  if (e != cudaSuccess) {                                         \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                               \
    std::exit(1);                                                 \
  }                                                               \
} while (0)

static void cpu_linear(const float* X, const float* W, const float* b,
                       float* Y, int B, int N, int K) {
  for (int i = 0; i < B; ++i) {
    for (int n = 0; n < N; ++n) {
      float acc = b ? b[n] : 0.0f;
      const float* xrow = X + i*K;
      const float* wrow = W + n*K; // W[n,:]
      for (int k = 0; k < K; ++k) acc += xrow[k] * wrow[k];
      Y[i*N + n] = acc;
    }
  }
}

/* -------------------- 1) Naive kernel -------------------- */
// Each thread computes one Y[b,n]
__global__ void linear_naive_kernel(
    const float* __restrict__ X,   // [B,K]
    const float* __restrict__ W,   // [N,K]
    const float* __restrict__ bias,// [N] or nullptr
    float* __restrict__ Y,         // [B,N]
    int B, int N, int K)
{
  int b = blockIdx.y * blockDim.y + threadIdx.y; // row in Y (batch)
  int n = blockIdx.x * blockDim.x + threadIdx.x; // col in Y (out feature)

  if (b >= B || n >= N) return;

  float acc = bias ? bias[n] : 0.0f;
  const float* xrow = X + b * K;
  const float* wrow = W + n * K; // W[n, :]
  for (int k = 0; k < K; ++k) {
    acc += xrow[k] * wrow[k];
  }
  Y[b * N + n] = acc;
}

/* -------------------- 2) Tiled shared-memory kernel -------------------- */
// Tile sizes (tweak for your GPU)
constexpr int BM = 128; // rows of Y per block (B dimension)
constexpr int BN = 128; // cols of Y per block (N dimension)
constexpr int BK = 32;  // K chunk per iteration
// Thread tile (blockDim = (16,16)): each thread computes a TM x TN micro-tile
constexpr int TM = 8;
constexpr int TN = 8;

// Y = X[B,K] * W[N,K]^T + b[N]
__global__ void linear_tiled_kernel(
    const float* __restrict__ X,   // [B,K] row-major
    const float* __restrict__ W,   // [N,K] row-major
    const float* __restrict__ bias,// [N] or nullptr
    float* __restrict__ Y,         // [B,N] row-major
    int B, int N, int K)
{
  // Block origin in output tile space
  int b0 = blockIdx.y * BM;
  int n0 = blockIdx.x * BN;

  // Thread coordinates inside block
  int tx = threadIdx.x; // 0..15
  int ty = threadIdx.y; // 0..15

  // Shared memory tiles
  __shared__ float sX[BM][BK]; // 128 x 32
  __shared__ float sW[BN][BK]; // 128 x 32

  // Per-thread accumulators
  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.f;
    }
  }

  // This thread computes a TM x TN micro-tile starting at:
  int b_base = b0 + ty * TM; // output row start
  int n_base = n0 + tx * TN; // output col start

  // Iterate over K dimension in chunks of BK
  for (int k0 = 0; k0 < K; k0 += BK) {
    // Load X tile: rows [b0..b0+BM), cols [k0..k0+BK)
    for (int r = ty; r < BM; r += blockDim.y) {
      for (int kk = tx; kk < BK; kk += blockDim.x) {
        int b_idx = b0 + r;
        int k_idx = k0 + kk;
        sX[r][kk] = (b_idx < B && k_idx < K) ? X[b_idx * K + k_idx] : 0.f;
      }
    }
    // Load W tile: rows [n0..n0+BN), cols [k0..k0+BK)
    for (int r = ty; r < BN; r += blockDim.y) {
      for (int kk = tx; kk < BK; kk += blockDim.x) {
        int n_idx = n0 + r;
        int k_idx = k0 + kk;
        sW[r][kk] = (n_idx < N && k_idx < K) ? W[n_idx * K + k_idx] : 0.f;
      }
    }
    __syncthreads();

    // Compute this micro-tile from the shared slabs
    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float xreg[TM];
      float wreg[TN];

      // Load a column from sX for this thread's TM rows
      #pragma unroll
      for (int ii = 0; ii < TM; ++ii) {
        int r = ty * TM + ii;
        xreg[ii] = sX[r][kk];
      }
      // Load a column from sW for this thread's TN cols
      #pragma unroll
      for (int jj = 0; jj < TN; ++jj) {
        int c = tx * TN + jj;
        wreg[jj] = sW[c][kk];
      }

      // FMA
      #pragma unroll
      for (int ii = 0; ii < TM; ++ii) {
        #pragma unroll
        for (int jj = 0; jj < TN; ++jj) {
          acc[ii][jj] += xreg[ii] * wreg[jj];
        }
      }
    }
    __syncthreads();
  }

  // Write back with bias
  #pragma unroll
  for (int ii = 0; ii < TM; ++ii) {
    int b = b_base + ii;
    if (b >= B) continue;
    #pragma unroll
    for (int jj = 0; jj < TN; ++jj) {
      int n = n_base + jj;
      if (n >= N) continue;
      float v = acc[ii][jj];
      if (bias) v += bias[n];
      Y[b * N + n] = v;
    }
  }
}

/* -------------------- Host helpers -------------------- */
static void rand_fill(std::vector<float>& v, float scale=0.01f) {
  for (auto& x : v) x = scale * float(rand()) / float(RAND_MAX) - 0.5f*scale;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  assert(a.size() == b.size());
  float m = 0.f;
  for (size_t i=0;i<a.size();++i) {
    m = fmaxf(m, fabsf(a[i]-b[i]));
  }
  return m;
}

int main() {
  // Problem sizes (change as you like)
  int B = 256;   // batch
  int K = 1024;  // in_features
  int N = 2048;  // out_features

  printf("B=%d, K=%d, N=%d\n", B, K, N);

  size_t szX = size_t(B) * K;
  size_t szW = size_t(N) * K;
  size_t szY = size_t(B) * N;

  std::vector<float> hX(szX), hW(szW), hb(N), hY_ref(szY), hY_naive(szY), hY_tiled(szY);
  rand_fill(hX);
  rand_fill(hW);
  rand_fill(hb, 0.1f);

  // CPU reference
  cpu_linear(hX.data(), hW.data(), hb.data(), hY_ref.data(), B, N, K);

  // Allocate device
  float *dX, *dW, *db, *dY;
  CHECK_CUDA(cudaMalloc(&dX, szX * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dW, szW * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&db, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dY, szY * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dX, hX.data(), szX*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dW, hW.data(), szW*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, hb.data(), N*sizeof(float), cudaMemcpyHostToDevice));

  // ---- Launch naive
  dim3 blockN(16, 16, 1);
  dim3 gridN((N + blockN.x - 1)/blockN.x, (B + blockN.y - 1)/blockN.y, 1);
  linear_naive_kernel<<<gridN, blockN>>>(dX, dW, db, dY, B, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(hY_naive.data(), dY, szY*sizeof(float), cudaMemcpyDeviceToHost));
  printf("naive max |diff| vs CPU: %.6f\n", max_abs_diff(hY_naive, hY_ref));

  // ---- Launch tiled
  dim3 blockT(16, 16, 1);
  dim3 gridT((N + BN - 1)/BN, (B + BM - 1)/BM, 1);
  linear_tiled_kernel<<<gridT, blockT>>>(dX, dW, db, dY, B, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(hY_tiled.data(), dY, szY*sizeof(float), cudaMemcpyDeviceToHost));
  printf("tiled  max |diff| vs CPU: %.6f\n", max_abs_diff(hY_tiled, hY_ref));

  CHECK_CUDA(cudaFree(dX));
  CHECK_CUDA(cudaFree(dW));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dY));

  return 0;
}
