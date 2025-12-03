// Tunable wrapper for esmm kernel (case 10)
// This file is used by kernel_tuner for autotuning
// Uses preprocessor defines instead of templates for kernel_tuner compatibility

#define WARPSIZE 32

/*
 * Tuning parameters (set via preprocessor defines):
 * BM - The threadblock size for M dimension SMEM caching.
 * BN - The threadblock size for N dimension SMEM caching.
 * BK - The threadblock size for K dimension SMEM caching. [FIXED=8]
 * WM - M dim of continuous tile computed by each warp
 * WN - N dim of continuous tile computed by each warp
 * WMITER - The number of subwarp tiling steps in M dimension (computed)
 * WNITER - The number of subwarp tiling steps in N dimension.
 * TM - The per-thread tile size for M dimension. [FIXED=1]
 * TN - The per-thread tile size for N dimension.
 * NUM_THREADS - Number of threads per block
 */

extern "C" __global__ void __launch_bounds__(NUM_THREADS)
	esmm(int M, int N, int K, float *A, float *B, float *C) {
	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	const uint warpIdx = threadIdx.x / WARPSIZE;
	const uint warpCol = warpIdx % (BN / WN);
	const uint warpRow = warpIdx / (BN / WN);

	const int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
	const int WSUBM = WM / WMITER;
	const int WSUBN = WN / WNITER;

	const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
	const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
	const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

	__shared__ float As[BK * BM];
	__shared__ float Bs[BK * BN];

	A += cRow * BM * K;
	B += cCol * BN;
	C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

	const uint innerRowA = threadIdx.x / (BK / 4);
	const uint innerColA = threadIdx.x % (BK / 4);
	const int rowStrideA = (NUM_THREADS * 4) / BK;
	const uint innerRowB = threadIdx.x / (BN / 4);
	const uint innerColB = threadIdx.x % (BN / 4);
	const int rowStrideB = NUM_THREADS / (BN / 4);

	float threadResults[WMITER * TM * WNITER * TN] = {0.0};
	float regM[WMITER * TM] = {0.0};
	float regN[WNITER * TN] = {0.0};

	for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
		for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
			const float4 tmp = reinterpret_cast<const float4 *>(
				&A[(innerRowA + offset) * K + innerColA * 4])[0];
			As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
			As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
			As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
			As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
		}
		for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
			reinterpret_cast<float4 *>(
				&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
				reinterpret_cast<const float4 *>(
					&B[(innerRowB + offset) * N + innerColB * 4])[0];
		}
		__syncthreads();
		for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
			for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
				regM[wSubRowIdx] = As[(dotIdx * BM) + warpRow * WM +
					wSubRowIdx * WSUBM + threadRowInWarp * TM];
			}
			/* Load TN values into the register */
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				regN[wSubColIdx * TN + 0] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 0];
				regN[wSubColIdx * TN + 1] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 1];
				regN[wSubColIdx * TN + 2] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 2];
				regN[wSubColIdx * TN + 3] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 3];
				regN[wSubColIdx * TN + 4] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 4];
				regN[wSubColIdx * TN + 5] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 5];
				regN[wSubColIdx * TN + 6] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 6];
				regN[wSubColIdx * TN + 7] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 7];
			}
			for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
				for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
					const int regNBase = wSubColIdx * 8;
					const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
					const float regM_val = regM[wSubRowIdx];
					threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
					threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
					threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
					threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
					threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
					threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
					threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
					threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
				}
			}
		}
		A += BK;
		B += BK * N;
		__syncthreads();
	}

	for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
			for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
				for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
					float4 tmp;
					const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
						wSubColIdx * TN + resIdxN;
					tmp.x = threadResults[i + 0];
					tmp.y = threadResults[i + 1];
					tmp.z = threadResults[i + 2];
					tmp.w = threadResults[i + 3];
					reinterpret_cast<float4 *>(
							&C_interim[(threadRowInWarp * TM + resIdxM) * N +
							threadColInWarp * TN + resIdxN])[0] = tmp;
				}
			}
		}
	}
}
