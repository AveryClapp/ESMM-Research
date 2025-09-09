"""
Kernel Tuner setup for autotuning K11 double buffered GEMM kernel
Based on the CUDA MMM implementation from your project knowledge
"""

import json
from typing import Dict, List, Tuple

import numpy as np
from kernel_tuner import tune_kernel


def setup_k11_autotuning():
    """
    Complete autotuning setup for your K11 double buffered kernel
    """

    kernel_string = """
    #define WARP_SIZE 32

    template<
        const uint BM,     // Block tile size for M dimension
        const uint BN,     // Block tile size for N dimension  
        const uint BK,     // Block tile size for K dimension
        const uint TM,     // Thread tile size for M dimension
        const uint TN,     // Thread tile size for N dimension
        const uint WSUBN,  // Warp subtile size for N
        const uint WSUBM,  // Warp subtile size for M
        const uint WNITER, // Warp N iterations
        const uint WMITER, // Warp M iterations
        const uint NUM_STAGES // Double buffering stages (2 or 3)
    >
    __global__ void k11_double_buffered_gemm(
        float* __restrict__ A,
        float* __restrict__ B, 
        float* __restrict__ C,
        const int M,
        const int N,
        const int K
    ) {
        // Shared memory double buffering
        __shared__ float As[NUM_STAGES][BM * BK];
        __shared__ float Bs[NUM_STAGES][BK * BN];

        // Register file buffers for double buffering SMEM->RF
        float regA[NUM_STAGES][TM];
        float regB[NUM_STAGES][TN];
        float threadResults[TM * TN] = {0.0f};

        // Thread and warp identification
        const uint threadRow = threadIdx.x / BN;
        const uint threadCol = threadIdx.x % BN;
        const uint warpId = threadIdx.x / WARP_SIZE;
        const uint laneId = threadIdx.x % WARP_SIZE;

        // Warp tiling coordinates
        const uint warpRow = (warpId / (BN / WSUBM)) * WSUBM;
        const uint warpCol = (warpId % (BN / WSUBM)) * WSUBN;

        // Global memory pointers
        A += blockIdx.y * BM * K;
        B += blockIdx.x * BN;
        C += blockIdx.y * BM * N + blockIdx.x * BN;

        // Pipeline stages
        int stage = 0;
        int load_stage = 0;
        int compute_stage = 0;

        // Initial load into stage 0
        load_gmem_to_smem(A, B, As[0], Bs[0], threadRow, threadCol, K, N);
        __syncthreads();

        // Main double buffered loop
        for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
            // Determine next stages
            load_stage = (stage + 1) % NUM_STAGES;
            compute_stage = stage;

            // Async load next tile to SMEM while computing current
            if (bkIdx + BK < K) {
                load_gmem_to_smem(A + BK, B + BK * N, As[load_stage], Bs[load_stage], 
                                threadRow, threadCol, K, N);
            }

            // Load current tile from SMEM to registers (first stage of RF double buffering)
            load_smem_to_regs(As[compute_stage], Bs[compute_stage], regA[0], regB[0], 
                            threadRow, threadCol, warpRow, warpCol);

            // Inner loop with register file double buffering
            for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                uint reg_load_stage = (dotIdx + 1) % NUM_STAGES;
                uint reg_compute_stage = dotIdx % NUM_STAGES;

                // Prefetch next iteration to registers
                if (dotIdx + 1 < BK) {
                    load_smem_to_regs_next(As[compute_stage], Bs[compute_stage], 
                                         regA[reg_load_stage], regB[reg_load_stage],
                                         dotIdx + 1, threadRow, threadCol, warpRow, warpCol);
                }

                // Compute using current register values
                compute_warp_tile(regA[reg_compute_stage], regB[reg_compute_stage], 
                                threadResults, warpRow, warpCol);
            }

            // Synchronize before next iteration
            __syncthreads();

            // Advance pointers and stage
            A += BK;
            B += BK * N;
            stage = load_stage;
        }

        // Write results back to global memory
        write_results_to_gmem(C, threadResults, threadRow, threadCol, M, N);
    }
    
    // Helper device functions would be implemented here
    __device__ void load_gmem_to_smem(...) { /* Implementation */ }
    __device__ void load_smem_to_regs(...) { /* Implementation */ }
    __device__ void load_smem_to_regs_next(...) { /* Implementation */ }
    __device__ void compute_warp_tile(...) { /* Implementation */ }
    __device__ void write_results_to_gmem(...) { /* Implementation */ }
    """

    # Test matrix dimensions - start with sizes where you have performance gaps vs cuBLAS
    test_sizes = [
        (512, 512, 512),  # Small matrices where cuBLAS dominates
        (1024, 1024, 1024),  # Medium matrices
        (2048, 2048, 2048),  # Large matrices where you're competitive
        (4096, 4096, 4096),  # Very large matrices
    ]

    # Comprehensive parameter space based on your existing autotuning
    tune_params = {
        # Block tiling parameters - expand from your current best configs
        "BM": [64, 128, 256],
        "BN": [64, 128, 256],
        "BK": [8, 16, 32],
        # Thread tiling - optimize for register usage
        "TM": [1],
        "TN": [4, 8, 16],
        # Warp tiling parameters - critical for double buffering efficiency
        "WSUBN": [16, 32, 64],
        "WSUBM": [16, 32, 64],
        "WNITER": [1, 2, 4],
        "WMITER": [1, 2, 4],
        # Double buffering stages - key parameter for your k11 kernel
        "NUM_STAGES": [2, 3, 4],  # 2=basic double buf, 3+ = deeper pipelining
    }

    # Constraints to ensure valid configurations
    restrictions = [
        # Block size constraints
        "BM >= TM * WMITER * WSUBM",
        "BN >= TN * WNITER * WSUBN",
        "BM * BN <= 1024",  # Max threads per block
        # Warp constraints
        "WSUBM * WSUBN <= 32",  # Warp size
        "BM % (WSUBM * WMITER) == 0",
        "BN % (WSUBN * WNITER) == 0",
        # Memory constraints - ensure SMEM fits
        "NUM_STAGES * (BM * BK + BK * BN) * 4 <= 49152",  # 48KB SMEM limit
        # Register pressure constraints
        "TM * TN * NUM_STAGES <= 64",  # Rough register estimate
    ]

    return kernel_string, tune_params, restrictions, test_sizes


def run_comprehensive_autotuning():
    """
    Run systematic autotuning across multiple matrix sizes and optimization strategies
    """
    kernel_string, tune_params, restrictions, test_sizes = setup_k11_autotuning()

    results = {}

    for M, N, K in test_sizes:
        print(f"\n=== Autotuning for matrix size {M}x{N}x{K} ===")

        # Setup test data
        np.random.seed(42)  # Reproducible results
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        args = [A, B, C, np.int32(M), np.int32(N), np.int32(K)]

        # Grid dimensions
        grid_size = ((N + 128 - 1) // 128, (M + 128 - 1) // 128, 1)

        # Custom metrics for ML workloads
        def custom_metrics(gpu_args):
            # Calculate effective bandwidth and FLOP/s
            flops = 2 * M * N * K  # FMA operations
            memory_bytes = (M * K + K * N + M * N) * 4  # float32

            return {
                "GFLOPS": flops / (gpu_args["time"] * 1e9),
                "GB_per_s": memory_bytes / (gpu_args["time"] * 1e9),
                "arithmetic_intensity": flops / memory_bytes,
            }

        # Multi-objective optimization strategies
        optimization_strategies = [
            {"strategy": "minimize", "objective": "time"},
            {"strategy": "minimize", "objective": "energy"},  # For sustainable ML
            {"strategy": "maximize", "objective": "GFLOPS"},
            {"strategy": "bayes_opt", "max_fevals": 200},  # Smart search
        ]

        size_results = {}

        for strategy_config in optimization_strategies:
            strategy_name = f"{strategy_config['strategy']}_{strategy_config.get('objective', 'adaptive')}"
            print(f"Running {strategy_name} optimization...")

            try:
                result = tune_kernel(
                    "k11_double_buffered_gemm",
                    kernel_string,
                    grid_size,
                    args,
                    tune_params,
                    restrictions=restrictions,
                    metrics=custom_metrics,
                    **strategy_config,
                    cache="k11_cache.json",  # Cache results
                    verbose=True,
                )

                size_results[strategy_name] = {
                    "best_config": result[0],
                    "best_time": result[1],
                    "all_results": result[2],
                }

                print(f"Best {strategy_name} config: {result[0]}")
                print(f"Performance: {result[1]:.6f} ms")

            except Exception as e:
                print(f"Strategy {strategy_name} failed: {e}")
                continue

        results[f"{M}x{N}x{K}"] = size_results

        # Analyze double buffering effectiveness
        analyze_double_buffering_impact(size_results)

    return results


def analyze_double_buffering_impact(results):
    """
    Analyze how different NUM_STAGES values affect performance
    """
    print("\n=== Double Buffering Analysis ===")

    stage_performance = {}
    for strategy, data in results.items():
        if "all_results" in data:
            for config, perf_data in data["all_results"].items():
                stages = config.get("NUM_STAGES", 2)
                if stages not in stage_performance:
                    stage_performance[stages] = []
                stage_performance[stages].append(perf_data["time"])

    # Report optimal staging strategy
    for stages, times in stage_performance.items():
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"NUM_STAGES={stages}: {avg_time:.6f}±{std_time:.6f} ms")


def generate_production_templates(results):
    """
    Generate optimized kernel templates for different use cases
    """
    templates = {
        "small_matrices": {},  # For transformer attention, small batch inference
        "large_matrices": {},  # For training, large batch inference
        "energy_optimal": {},  # For edge deployment
        "throughput_optimal": {},  # For data center training
    }

    # Extract best configurations for each category
    for size, size_results in results.items():
        M, N, K = map(int, size.split("x"))

        if M <= 1024:
            category = "small_matrices"
        else:
            category = "large_matrices"

        # Find best throughput config
        if "maximize_GFLOPS" in size_results:
            templates[category] = size_results["maximize_GFLOPS"]["best_config"]

        # Find best energy config
        if "minimize_energy" in size_results:
            templates["energy_optimal"] = size_results["minimize_energy"]["best_config"]

    return templates


if __name__ == "__main__":
    print("Starting K11 Double Buffered Kernel Autotuning...")
    print("This will systematically optimize your double buffering implementation")

    # Run comprehensive autotuning
    results = run_comprehensive_autotuning()

    # Generate production-ready templates
    templates = generate_production_templates(results)

    # Save results for analysis
    with open("k11_autotuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("k11_optimized_templates.json", "w") as f:
        json.dump(templates, f, indent=2)

    print(
        "\nAutotuning complete! Check k11_autotuning_results.json for detailed results"
    )
    print("Optimized templates saved to k11_optimized_templates.json")
