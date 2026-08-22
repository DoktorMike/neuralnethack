#include "SmallGemm.hh"

#ifdef __AVX512F__
#include <immintrin.h>
#include <vector>
#endif

namespace nnh::smallgemm {

#ifdef __AVX512F__

// Design notes. At these sizes the limiter is FMA latency (4-5 cycles),
// not bandwidth: a single accumulator per output block serializes the
// k-loop. All kernels therefore process 4 output rows per pass -- four
// independent FMA chains that share every load of B. Tails are handled
// with masked loads/stores, so any runtime shape works. gemmNT packs
// B^T once per call (O(K*N) against O(M*N*K) FLOPs) to avoid horizontal
// reductions, except when N is too narrow to vectorize, where the
// dot-product form wins.

namespace {

thread_local std::vector<double> tScratch;

// C[M x N] (=|+=) A' * B where A'[m,k] is either A[m*lda+k] (TransA=false)
// or A[k*lda+m] (TransA=true). B rows contiguous (ldb). Quad-row blocked.
template <bool TransA, bool Acc>
inline void broadcastKernel(uint M, uint N, uint K, const double* A, uint lda, const double* B,
                            uint ldb, double* C, uint ldc) {
	const uint nTail = N & 7u;
	const __mmask8 nMask = nTail ? static_cast<__mmask8>((1u << nTail) - 1u) : 0;
	const auto elemA = [&](uint m, uint k) { return TransA ? A[k * lda + m] : A[m * lda + k]; };

	uint m = 0;
	for (; m + 4 <= M; m += 4) {
		double* c0 = C + (m + 0) * ldc;
		double* c1 = C + (m + 1) * ldc;
		double* c2 = C + (m + 2) * ldc;
		double* c3 = C + (m + 3) * ldc;
		uint n = 0;
		for (; n + 8 <= N; n += 8) {
			__m512d a0, a1, a2, a3;
			if constexpr (Acc) {
				a0 = _mm512_loadu_pd(c0 + n);
				a1 = _mm512_loadu_pd(c1 + n);
				a2 = _mm512_loadu_pd(c2 + n);
				a3 = _mm512_loadu_pd(c3 + n);
			} else {
				a0 = a1 = a2 = a3 = _mm512_setzero_pd();
			}
			for (uint k = 0; k < K; ++k) {
				const __m512d vb = _mm512_loadu_pd(B + k * ldb + n);
				a0 = _mm512_fmadd_pd(_mm512_set1_pd(elemA(m + 0, k)), vb, a0);
				a1 = _mm512_fmadd_pd(_mm512_set1_pd(elemA(m + 1, k)), vb, a1);
				a2 = _mm512_fmadd_pd(_mm512_set1_pd(elemA(m + 2, k)), vb, a2);
				a3 = _mm512_fmadd_pd(_mm512_set1_pd(elemA(m + 3, k)), vb, a3);
			}
			_mm512_storeu_pd(c0 + n, a0);
			_mm512_storeu_pd(c1 + n, a1);
			_mm512_storeu_pd(c2 + n, a2);
			_mm512_storeu_pd(c3 + n, a3);
		}
		if (nTail) {
			__m512d a0, a1, a2, a3;
			if constexpr (Acc) {
				a0 = _mm512_maskz_loadu_pd(nMask, c0 + n);
				a1 = _mm512_maskz_loadu_pd(nMask, c1 + n);
				a2 = _mm512_maskz_loadu_pd(nMask, c2 + n);
				a3 = _mm512_maskz_loadu_pd(nMask, c3 + n);
			} else {
				a0 = a1 = a2 = a3 = _mm512_setzero_pd();
			}
			for (uint k = 0; k < K; ++k) {
				const __m512d vb = _mm512_maskz_loadu_pd(nMask, B + k * ldb + n);
				a0 = _mm512_fmadd_pd(_mm512_set1_pd(elemA(m + 0, k)), vb, a0);
				a1 = _mm512_fmadd_pd(_mm512_set1_pd(elemA(m + 1, k)), vb, a1);
				a2 = _mm512_fmadd_pd(_mm512_set1_pd(elemA(m + 2, k)), vb, a2);
				a3 = _mm512_fmadd_pd(_mm512_set1_pd(elemA(m + 3, k)), vb, a3);
			}
			_mm512_mask_storeu_pd(c0 + n, nMask, a0);
			_mm512_mask_storeu_pd(c1 + n, nMask, a1);
			_mm512_mask_storeu_pd(c2 + n, nMask, a2);
			_mm512_mask_storeu_pd(c3 + n, nMask, a3);
		}
	}
	for (; m < M; ++m) { // row tail
		double* c = C + m * ldc;
		uint n = 0;
		for (; n + 8 <= N; n += 8) {
			__m512d acc = Acc ? _mm512_loadu_pd(c + n) : _mm512_setzero_pd();
			for (uint k = 0; k < K; ++k)
				acc = _mm512_fmadd_pd(_mm512_set1_pd(elemA(m, k)), _mm512_loadu_pd(B + k * ldb + n),
				                      acc);
			_mm512_storeu_pd(c + n, acc);
		}
		if (nTail) {
			__m512d acc = Acc ? _mm512_maskz_loadu_pd(nMask, c + n) : _mm512_setzero_pd();
			for (uint k = 0; k < K; ++k)
				acc = _mm512_fmadd_pd(_mm512_set1_pd(elemA(m, k)),
				                      _mm512_maskz_loadu_pd(nMask, B + k * ldb + n), acc);
			_mm512_mask_storeu_pd(c + n, nMask, acc);
		}
	}
}

// Dot-product form for narrow C (N < 8): C[M x N] = A[M x K] * B[N x K]^T,
// vectorized over K, 2 k-accumulators per output to break the FMA chain.
inline void dotKernelNT(uint M, uint N, uint K, const double* A, uint lda, const double* B,
                        uint ldb, double* C, uint ldc) {
	const uint kTail = K & 7u;
	const __mmask8 kMask = kTail ? static_cast<__mmask8>((1u << kTail) - 1u) : 0;
	for (uint m = 0; m < M; ++m) {
		const double* a = A + m * lda;
		double* c = C + m * ldc;
		for (uint n = 0; n < N; ++n) {
			const double* b = B + n * ldb;
			__m512d s0 = _mm512_setzero_pd(), s1 = _mm512_setzero_pd();
			uint k = 0;
			for (; k + 16 <= K; k += 16) {
				s0 = _mm512_fmadd_pd(_mm512_loadu_pd(a + k), _mm512_loadu_pd(b + k), s0);
				s1 = _mm512_fmadd_pd(_mm512_loadu_pd(a + k + 8), _mm512_loadu_pd(b + k + 8), s1);
			}
			if (k + 8 <= K) {
				s0 = _mm512_fmadd_pd(_mm512_loadu_pd(a + k), _mm512_loadu_pd(b + k), s0);
				k += 8;
			}
			if (kTail)
				s1 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(kMask, a + k),
				                     _mm512_maskz_loadu_pd(kMask, b + k), s1);
			c[n] = _mm512_reduce_add_pd(_mm512_add_pd(s0, s1));
		}
	}
}

} // namespace

void gemmNN(uint M, uint N, uint K, const double* A, uint lda, const double* B, uint ldb, double* C,
            uint ldc) {
	broadcastKernel<false, false>(M, N, K, A, lda, B, ldb, C, ldc);
}

void gemmNT(uint M, uint N, uint K, const double* A, uint lda, const double* B, uint ldb, double* C,
            uint ldc) {
	// Packing B^T costs O(K*N) and only amortizes across enough rows of A;
	// at M=1 (single-pattern inference) it equals the whole FLOP count.
	if (N < 8 || M < 8) {
		dotKernelNT(M, N, K, A, lda, B, ldb, C, ldc);
		return;
	}
	// Pack B^T so the broadcast kernel streams contiguous rows.
	tScratch.resize(static_cast<std::size_t>(K) * N);
	double* Bt = tScratch.data();
	for (uint n = 0; n < N; ++n) {
		const double* b = B + n * ldb;
		for (uint k = 0; k < K; ++k)
			Bt[k * N + n] = b[k];
	}
	broadcastKernel<false, false>(M, N, K, A, lda, Bt, N, C, ldc);
}

void gemmTNAcc(uint M, uint N, uint K, const double* A, uint lda, const double* B, uint ldb,
               double* C, uint ldc) {
	// C[m,n] += sum_k A[k,m] * B[k,n]: broadcast kernel with A transposed.
	// The strided broadcasts stay in L1; the quad-row ILP is what matters.
	broadcastKernel<true, true>(M, N, K, A, lda, B, ldb, C, ldc);
}

#else // scalar fallbacks; -O3 -ffast-math auto-vectorizes these

void gemmNT(uint M, uint N, uint K, const double* A, uint lda, const double* B, uint ldb, double* C,
            uint ldc) {
	for (uint m = 0; m < M; ++m) {
		const double* a = A + m * lda;
		for (uint n = 0; n < N; ++n) {
			const double* b = B + n * ldb;
			double sum = 0.0;
			for (uint k = 0; k < K; ++k)
				sum += a[k] * b[k];
			C[m * ldc + n] = sum;
		}
	}
}

void gemmNN(uint M, uint N, uint K, const double* A, uint lda, const double* B, uint ldb, double* C,
            uint ldc) {
	for (uint m = 0; m < M; ++m) {
		double* c = C + m * ldc;
		for (uint n = 0; n < N; ++n)
			c[n] = 0.0;
		for (uint k = 0; k < K; ++k) {
			const double a = A[m * lda + k];
			const double* b = B + k * ldb;
			for (uint n = 0; n < N; ++n)
				c[n] += a * b[n];
		}
	}
}

void gemmTNAcc(uint M, uint N, uint K, const double* A, uint lda, const double* B, uint ldb,
               double* C, uint ldc) {
	for (uint m = 0; m < M; ++m) {
		double* c = C + m * ldc;
		for (uint k = 0; k < K; ++k) {
			const double a = A[k * lda + m];
			const double* b = B + k * ldb;
			for (uint n = 0; n < N; ++n)
				c[n] += a * b[n];
		}
	}
}

#endif

} // namespace nnh::smallgemm
