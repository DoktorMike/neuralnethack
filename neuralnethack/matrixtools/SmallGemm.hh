#ifndef __SmallGemm_hh__
#define __SmallGemm_hh__

#include <cstdint>

namespace nnh::smallgemm {

using uint = unsigned int;

/// Row-major double-precision GEMM microkernels for the small matrix
/// shapes that dominate MLP training (batch x layer-width products).
/// A cblas_dgemm call costs ~1 us in dispatch/packing overhead, which
/// exceeds the arithmetic cost below roughly 64^3 FLOPs. These kernels
/// are direct AVX-512 loops with no packing and no dispatch, and fall
/// back to plain (auto-vectorizable) loops when AVX-512 is unavailable.

/// True when the shape is small enough that bypassing BLAS wins.
inline bool small(uint m, uint n, uint k) {
	return static_cast<std::uint64_t>(m) * n * k <= 65536u;
}

/// C[M x N] = A[M x K] * B[N x K]^T
/// (forward pass: Out = Input * W^T, ldb skips the bias column)
void gemmNT(uint M, uint N, uint K, const double* A, uint lda, const double* B, uint ldb, double* C,
            uint ldc);

/// C[M x N] = A[M x K] * B[K x N]
/// (backprop: delta_curr = delta_next * W_next)
void gemmNN(uint M, uint N, uint K, const double* A, uint lda, const double* B, uint ldb, double* C,
            uint ldc);

/// C[M x N] += A[K x M]^T * B[K x N]
/// (gradient accumulation: grad += delta^T * input, ldc skips bias column)
void gemmTNAcc(uint M, uint N, uint K, const double* A, uint lda, const double* B, uint ldb,
               double* C, uint ldc);

} // namespace nnh::smallgemm

#endif
