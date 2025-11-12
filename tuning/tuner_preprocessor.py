#!/usr/bin/env python3
"""
A-Matrix Preprocessor Autotuner using kernel_tuner

Tunes the preprocessing kernel that generates sparsity patterns.
Since preprocessing is ~9% of total runtime, even 10-30× speedup
here only yields 5-9% overall improvement. Consider this a lower
priority than B-sparsity implementation.
"""

import numpy as np
import json
from kernel_tuner import tune_kernel
from collections import OrderedDict

# Matrix configuration
M = 2048
K = 2048
BK = 8
num_patterns = M * (K // BK)

print(f"=== A-Matrix Preprocessor Tuning ===")
print(f"Matrix: {M}×{K}")
print(f"Patterns to generate: {num_patterns:,}")
print(f"Output size: {num_patterns / 1024:.1f} KB")
print()

# Generate test data (50% sparse)
np.random.seed(42)
h_A = np.random.rand(M, K).astype(np.float32)
h_A[h_A < 0.5] = 0.0  # 50% sparsity

# Output array
h_patterns = np.zeros(num_patterns, dtype=np.uint8)

# Kernel arguments (use numpy types for scalars)
args = [np.int32(M), np.int32(K), h_A, h_patterns]

# Tunable parameters
tune_params = OrderedDict()
tune_params["NUM_THREADS"] = [128, 256, 512]
tune_params["ROWS_PER_THREAD"] = [1, 2, 4, 8]
tune_params["USE_SHARED_MEM"] = [0]  # Start simple, can add [0, 1] later
tune_params["VECTORIZE"] = [1, 2, 4]
tune_params["K_BATCH_SIZE"] = [1, 2, 4, 8]

# Grid computation function
def grid_func(params):
    """Calculate grid dimensions based on parameters"""
    rows_per_block = params["NUM_THREADS"] * params["ROWS_PER_THREAD"]
    grid_x = (M + rows_per_block - 1) // rows_per_block
    return (grid_x, 1, 1)

# Restrictions to avoid invalid configurations
# kernel_tuner passes individual parameters, not a dict
def restrictions_func(NUM_THREADS, ROWS_PER_THREAD, USE_SHARED_MEM, VECTORIZE, K_BATCH_SIZE):
    """Filter out invalid parameter combinations"""
    # VECTORIZE=4 requires BK divisible by 4 (always true since BK=8)
    # VECTORIZE=2 requires BK divisible by 2 (always true)
    # No restrictions needed for our case
    return True

# Performance metrics
metrics = OrderedDict()
metrics["GB/s"] = lambda p: (M * K * 4 + num_patterns) / (p["time"] / 1000) / 1e9
metrics["patterns/sec"] = lambda p: num_patterns / (p["time"] / 1000)

print("Starting tuning run...")
print(f"Total configurations: {len(tune_params['NUM_THREADS']) * len(tune_params['ROWS_PER_THREAD']) * len(tune_params['VECTORIZE']) * len(tune_params['K_BATCH_SIZE'])}")
print()

# Run tuning
results, env = tune_kernel(
    "preprocess_a_patterns",
    "tuning/a_preprocessor_tune.cu",
    (M, K),
    args,
    tune_params,
    grid_div_x=["NUM_THREADS * ROWS_PER_THREAD"],  # Dynamic grid sizing
    block_size_names=["NUM_THREADS"],  # Specify thread block parameter
    restrictions=restrictions_func,
    metrics=metrics,
    verbose=True,
    iterations=32,  # Average over 32 runs for stability
)

# Display results
print("\n" + "="*80)
print("TUNING RESULTS")
print("="*80)

if results:
    # Sort by execution time
    sorted_results = sorted(results, key=lambda x: x["time"])

    print("\n🏆 Top 5 Configurations:\n")
    for i, result in enumerate(sorted_results[:5], 1):
        time_ms = result["time"]
        gbps = metrics["GB/s"](result)
        patterns_per_sec = metrics["patterns/sec"](result)

        print(f"{i}. Time: {time_ms:.4f} ms | {gbps:.1f} GB/s | {patterns_per_sec/1e9:.2f}G patterns/sec")
        print(f"   NUM_THREADS={result['NUM_THREADS']}, "
              f"ROWS_PER_THREAD={result['ROWS_PER_THREAD']}, "
              f"VECTORIZE={result['VECTORIZE']}, "
              f"K_BATCH_SIZE={result['K_BATCH_SIZE']}")
        print()

    # Best configuration
    best = sorted_results[0]
    print("="*80)
    print("BEST CONFIGURATION:")
    print("="*80)
    print(f"""
const uint NUM_THREADS = {best['NUM_THREADS']};
const uint ROWS_PER_THREAD = {best['ROWS_PER_THREAD']};
const uint USE_SHARED_MEM = {best['USE_SHARED_MEM']};
const uint VECTORIZE = {best['VECTORIZE']};
const uint K_BATCH_SIZE = {best['K_BATCH_SIZE']};

// Performance:
// Time: {best['time']:.4f} ms
// Bandwidth: {metrics["GB/s"](best):.1f} GB/s
// Throughput: {metrics["patterns/sec"](best)/1e9:.2f}G patterns/sec
""")

    # Save to JSON
    output = {
        "best_config": {
            "NUM_THREADS": int(best['NUM_THREADS']),
            "ROWS_PER_THREAD": int(best['ROWS_PER_THREAD']),
            "USE_SHARED_MEM": int(best['USE_SHARED_MEM']),
            "VECTORIZE": int(best['VECTORIZE']),
            "K_BATCH_SIZE": int(best['K_BATCH_SIZE']),
        },
        "performance": {
            "time_ms": float(best['time']),
            "bandwidth_gbps": float(metrics["GB/s"](best)),
            "patterns_per_sec": float(metrics["patterns/sec"](best)),
        },
        "matrix_size": {"M": M, "K": K},
        "all_results": [
            {
                "config": {k: int(v) for k, v in r.items() if k in tune_params},
                "time_ms": float(r["time"]),
            }
            for r in sorted_results[:10]
        ]
    }

    with open("tuning/preprocessor_tuning_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n✓ Results saved to: tuning/preprocessor_tuning_results.json")

    # Impact analysis
    baseline_time = 0.6  # ms (from performance analysis)
    speedup = baseline_time / best['time']

    # Overall impact (assuming kernel is 6.0 ms)
    kernel_time = 6.0
    new_total = best['time'] + kernel_time
    old_total = baseline_time + kernel_time
    overall_speedup = old_total / new_total
    overall_improvement = ((old_total - new_total) / old_total) * 100

    print("\n" + "="*80)
    print("IMPACT ANALYSIS")
    print("="*80)
    print(f"Preprocessing speedup: {speedup:.1f}×")
    print(f"Overall speedup: {overall_speedup:.3f}× ({overall_improvement:.1f}% improvement)")
    print()
    print(f"Before: {old_total:.2f} ms total ({baseline_time:.2f} ms prep + {kernel_time:.2f} ms kernel)")
    print(f"After:  {new_total:.2f} ms total ({best['time']:.2f} ms prep + {kernel_time:.2f} ms kernel)")

else:
    print("❌ Tuning failed - no valid results")
