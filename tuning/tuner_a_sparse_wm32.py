#!/usr/bin/env python3
"""
A-Sparse Kernel Tuner with WM=32 Fixed

Finds optimal configuration for esmm_a_sparse_blockwise kernel with:
- WM = 32 (matches preprocessor granularity)
- BK = 8 (matches preprocessor)
"""

import json
import subprocess
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Config:
    NUM_THREADS: int
    BM: int
    BN: int
    BK: int = 8  # Fixed
    WM: int = 32  # Fixed
    WN: int = 32
    WNITER: int = 2
    TM: int = 1  # Fixed
    TN: int = 8  # Fixed

    def is_valid(self) -> bool:
        """Check if configuration satisfies all constraints"""
        # Warp tiling constraints
        if self.BM % self.WM != 0:
            return False
        if self.BN % self.WN != 0:
            return False

        # Number of warps must match thread count
        warps_M = self.BM // self.WM
        warps_N = self.BN // self.WN
        warps_total = warps_M * warps_N
        warps_avail = self.NUM_THREADS // 32

        if warps_total != warps_avail:
            return False

        # WMITER calculation
        WMITER = (self.WM * self.WN) // (32 * self.TM * self.TN * self.WNITER)
        if WMITER < 1:
            return False
        if self.WM % WMITER != 0:
            return False

        # WN must be divisible by WNITER
        if self.WN % self.WNITER != 0:
            return False

        # Memory loading constraints
        rowStrideA = (self.NUM_THREADS * 4) // self.BK
        if rowStrideA > self.BM:
            # Can't load A tile in one pass
            return False
        if self.BM % rowStrideA != 0:
            # Need clean divisibility for loading
            return False

        rowStrideB = self.NUM_THREADS // (self.BN // 4)
        if rowStrideB > self.BK:
            # Can't load B tile
            return False

        # Shared memory constraint (48KB on A10G)
        # Note: As is padded to (BM+1)*BK
        smem_bytes = (self.BM + 1) * self.BK * 4 + self.BN * self.BK * 4
        if smem_bytes > 48000:
            return False

        # Register file constraints
        WSUBM = self.WM // WMITER
        WSUBN = self.WN // self.WNITER
        threadResults = WMITER * self.TM * self.WNITER * self.TN
        if threadResults > 128:  # Conservative register limit
            return False

        return True

    def to_dict(self) -> Dict:
        return {
            'NUM_THREADS': self.NUM_THREADS,
            'BM': self.BM,
            'BN': self.BN,
            'BK': self.BK,
            'WM': self.WM,
            'WN': self.WN,
            'WNITER': self.WNITER,
            'TM': self.TM,
            'TN': self.TN
        }

    def __str__(self) -> str:
        return (f"T{self.NUM_THREADS}_BM{self.BM}x{self.BN}_"
                f"WM{self.WM}x{self.WN}_WNI{self.WNITER}")


def generate_configs() -> List[Config]:
    """Generate all valid configurations with WM=32"""
    configs = []

    for NUM_THREADS in [128, 256]:
        for BM in [64, 128, 256]:
            for BN in [64, 128, 256]:
                for WN in [32, 64]:
                    for WNITER in [1, 2, 4]:
                        cfg = Config(
                            NUM_THREADS=NUM_THREADS,
                            BM=BM,
                            BN=BN,
                            WN=WN,
                            WNITER=WNITER
                        )
                        if cfg.is_valid():
                            configs.append(cfg)

    return configs


def modify_runner(config: Config) -> None:
    """Modify runners.cuh with new config"""
    runner_path = "include/runners.cuh"

    with open(runner_path, 'r') as f:
        content = f.read()

    # Pattern to find the A-sparse config section
    pattern = r'(bool run_esmm_a_sparse_blockwise_no_check.*?\n)(  const uint NUM_THREADS = \d+;.*?\n)(  const uint BN = \d+;.*?\n)(  const uint BM = \d+;.*?\n)(  const uint BK = \d+;.*?\n)(  const uint WN = \d+;.*?\n)(  const uint WM = \d+;.*?\n)(  const uint WNITER = \d+;.*?\n)(  const uint TN = \d+;.*?\n)(  const uint TM = \d+;.*?\n)'

    replacement = (
        r'\1'
        f'  const uint NUM_THREADS = {config.NUM_THREADS};\n'
        f'  const uint BN = {config.BN};\n'
        f'  const uint BM = {config.BM};\n'
        f'  const uint BK = {config.BK};\n'
        f'  const uint WN = {config.WN};\n'
        f'  const uint WM = {config.WM};  // Fixed to match preprocessor\n'
        f'  const uint WNITER = {config.WNITER};\n'
        f'  const uint TN = {config.TN};\n'
        f'  const uint TM = {config.TM};\n'
    )

    content_new = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(runner_path, 'w') as f:
        f.write(content_new)


def compile_and_run(config: Config) -> Optional[float]:
    """Compile and benchmark a configuration"""
    print(f"Testing: {config}")

    # Modify runner with config
    modify_runner(config)

    # Compile
    compile_cmd = "nvcc -o driver driver.cu -I. -std=c++17 -lcublas 2>&1"
    result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ❌ Compilation failed")
        return None

    # Run benchmark (10 iterations, no checking for speed)
    bench_cmd = "./driver 16 10 --no-check 2>&1"
    result = subprocess.run(bench_cmd, shell=True, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"  ❌ Runtime error")
        return None

    # Extract kernel time (average across iterations)
    kernel_match = re.search(r'Kernel time:.*?avg: (\d+\.\d+) ms', result.stdout)
    if not kernel_match:
        print(f"  ❌ Could not extract kernel time")
        return None

    kernel_time = float(kernel_match.group(1))

    # Also track preprocessing time
    preprocess_match = re.search(r'GPU preprocessing.*?(\d+\.\d+) ms', result.stdout)
    preprocess_time = float(preprocess_match.group(1)) if preprocess_match else 0.0

    print(f"  ✓ Kernel: {kernel_time:.3f} ms, Preprocess: {preprocess_time:.2f} ms")

    return kernel_time


def benchmark_all_configs():
    """Benchmark all valid configurations"""
    configs = generate_configs()
    print(f"Generated {len(configs)} valid configurations with WM=32\n")

    results = []

    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] ", end='')
        time = compile_and_run(cfg)

        if time is not None:
            results.append({
                'config': cfg.to_dict(),
                'config_str': str(cfg),
                'time_ms': time
            })

    # Sort by time
    results.sort(key=lambda x: x['time_ms'])

    return results


def print_results(results: List[Dict]):
    """Print top 10 configurations"""
    print("\n" + "=" * 80)
    print("Top 10 Configurations (WM=32 fixed)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Config':<40} {'Time (ms)':<12}")
    print("-" * 80)

    for i, res in enumerate(results[:10], 1):
        print(f"{i:<5} {res['config_str']:<40} {res['time_ms']:<12.3f}")

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION:")
    print("=" * 80)
    best = results[0]
    cfg = best['config']
    print(f"Time: {best['time_ms']:.3f} ms")
    print(f"\nConfiguration:")
    for key, val in cfg.items():
        print(f"  {key:15} = {val}")

    # Save results
    with open('tuning/results_a_sparse_wm32.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to tuning/results_a_sparse_wm32.json")


if __name__ == "__main__":
    print("A-Sparse Kernel Tuner (WM=32 Fixed)")
    print("=" * 80)
    print("Target: ESMM A-sparse kernel with warp-granularity patterns")
    print("Matrix: 4096x4096, Pattern: 11110000 (50% sparse)")
    print("Fixed: WM=32 (matches preprocessor), BK=8, TM=1, TN=8")
    print()

    results = benchmark_all_configs()

    if results:
        print_results(results)
    else:
        print("No valid configurations found!")
