#!/usr/bin/env python3
"""
Parallel Benchmark Runner for ESMM Kernels

Automates running ncu profiles across multiple matrix sizes and sparsity patterns.
Generates organized .ncu-rep files and consolidated CSV with key metrics.

Usage:
    ./scripts/benchmark.py --kernel 17
    ./scripts/benchmark.py -k 17,22,23 --sizes 1024,2048 --parallel 2
    ./scripts/benchmark.py -k 17 --cold-start  # For cold-start measurements
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Default configurations
DEFAULT_SIZES = [1024, 2048, 4096]
DEFAULT_SPARSITY = {
    "100pct": "11111111",  # 100% density (0% sparse)
    "50pct": "11110000",  # 50% density
    "25pct": "11000000",  # 25% density
    "12pct": "10000000",  # 12.5% density
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parallel ncu benchmarks across sizes and sparsity patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --kernel 17
  %(prog)s -k 17,22,23 --sizes 1024,2048 --parallel 2
  %(prog)s -k 17 --sparsity 11111111,10101010 --output my_experiment
  %(prog)s -k 17 --cold-start  # Cold-start measurements (matches manual runs)
        """,
    )

    parser.add_argument(
        "-k",
        "--kernel",
        required=True,
        help="Kernel number(s) to benchmark (comma-separated)",
    )
    parser.add_argument(
        "-s",
        "--sizes",
        default=",".join(map(str, DEFAULT_SIZES)),
        help=f"Matrix sizes to test (comma-separated, default: {','.join(map(str, DEFAULT_SIZES))})",
    )
    parser.add_argument(
        "-p",
        "--sparsity",
        default=None,
        help="Sparsity patterns (comma-separated 8-bit strings, default: 100%%,50%%,25%%,12.5%%)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of concurrent ncu runs (default: 1 for accurate measurements)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory name (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--executable",
        default="./exec_dev",
        help="Path to executable (default: ./exec_dev)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of warmup/timing runs per kernel (default: 1)",
    )
    parser.add_argument(
        "--metrics",
        default="gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed",
        help="NCU metrics to collect (comma-separated)",
    )
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Reset GPU between runs for cold-start measurements (slower but accurate, matches manual runs)",
    )
    parser.add_argument(
        "--reset-delay",
        type=float,
        default=2.0,
        help="Delay in seconds after GPU reset (default: 2.0)",
    )
    parser.add_argument(
        "--unload-driver",
        action="store_true",
        help="Unload/reload NVIDIA driver between runs (VERY slow, but guarantees cold-start)",
    )

    return parser.parse_args()


def parse_kernel_list(kernel_str):
    """Parse comma-separated kernel numbers"""
    return [int(k.strip()) for k in kernel_str.split(",")]


def parse_size_list(size_str):
    """Parse comma-separated sizes"""
    return [int(s.strip()) for s in size_str.split(",")]


def parse_sparsity_patterns(sparsity_str):
    """Parse sparsity patterns or use defaults"""
    if sparsity_str is None:
        return DEFAULT_SPARSITY

    patterns = {}
    for i, pattern in enumerate(sparsity_str.split(",")):
        pattern = pattern.strip()
        if len(pattern) != 8 or not all(c in "01" for c in pattern):
            raise ValueError(
                f"Invalid pattern '{pattern}': must be 8 characters of 0s and 1s"
            )

        # Calculate density percentage
        density = pattern.count("1") / 8.0 * 100
        label = f"{density:.1f}pct".replace(".", "_")
        patterns[label] = pattern

    return patterns


def create_output_dir(args, kernels):
    """Create timestamped output directory"""
    if args.output:
        dirname = args.output
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        kernel_str = "k" + "_".join(map(str, kernels))
        mode_suffix = "_coldstart" if args.cold_start else ""
        dirname = f"{timestamp}_{kernel_str}{mode_suffix}"

    output_dir = Path("benchmarks") / dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def clear_cuda_cache():
    """Clear CUDA JIT compilation cache to ensure cold-start"""
    import shutil

    cache_paths = [
        Path.home() / ".nv" / "ComputeCache",
        Path(os.environ.get("CUDA_CACHE_PATH", "/nonexistent")),
    ]

    cleared = False
    for cache_path in cache_paths:
        if cache_path.exists() and cache_path.is_dir():
            try:
                # Remove and recreate to clear cache
                shutil.rmtree(cache_path)
                cache_path.mkdir(parents=True, exist_ok=True)
                print(f"[Cache] Cleared {cache_path}")
                cleared = True
            except Exception as e:
                print(f"[Warning] Could not clear cache at {cache_path}: {e}")

    return cleared


def unload_reload_driver(delay=5.0):
    """Unload and reload NVIDIA driver - nuclear option for cold-start

    WARNING: This is risky and may fail. Only use if absolutely necessary.
    """
    try:
        # Check if any processes are using the GPU
        print("[Driver] Checking for GPU processes...")
        check_result = subprocess.run(
            ["sudo", "fuser", "-v", "/dev/nvidia*"],
            capture_output=True,
            text=True,
        )

        # fuser exits with 1 if no processes found (which is what we want)
        if check_result.returncode == 0:
            print("[Warning] GPU is in use by other processes. Skipping driver reload.")
            print(f"[Warning] Processes: {check_result.stdout}")
            return False

        # Try to unload all nvidia modules in correct order
        print("[Driver] Unloading NVIDIA driver modules...")
        modules_to_unload = ["nvidia_uvm", "nvidia_drm", "nvidia_modeset", "nvidia"]

        for module in modules_to_unload:
            result = subprocess.run(
                ["sudo", "modprobe", "-r", module],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Ignore errors - some modules might not be loaded
            if result.returncode == 0:
                print(f"[Driver] Unloaded {module}")

        time.sleep(2)

        # Reload driver modules
        print("[Driver] Reloading NVIDIA driver modules...")
        result = subprocess.run(
            ["sudo", "modprobe", "nvidia"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"[ERROR] Failed to reload nvidia module: {result.stderr}")
            return False

        # Wait for driver to initialize
        time.sleep(delay)

        # Verify driver is working
        verify = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=10,
        )

        if verify.returncode == 0:
            print("[Driver] Driver reloaded successfully")
            return True
        else:
            print("[ERROR] Driver loaded but nvidia-smi failed")
            return False

    except Exception as e:
        print(f"[ERROR] Driver reload failed: {e}")
        print("[ERROR] You may need to manually reload the driver or reboot")
        return False


def reset_gpu(delay=2.0):
    """Reset GPU hardware state (optional, may not work on all systems)"""
    try:
        result = subprocess.run(
            ["sudo", "nvidia-smi", "-r"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            # This is expected to fail on many EC2 instances
            return False

        # Wait for reset to complete
        time.sleep(delay)
        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False


def run_ncu_profile(config):
    """Run a single ncu profile"""
    kernel, size, sparsity_label, sparsity_pattern, output_file, args = config

    cmd = [
        "sudo",
        "/usr/local/cuda-12.1/bin/ncu",
        "--set",
        "full",
        "--target-processes",
        "all",
        "--export",
        str(output_file),
        "--force-overwrite",
        args.executable,
        str(kernel),
        "--size",
        str(size),
        "--pattern",
        sparsity_pattern,
        "--no-check",  # Skip verification for speed
    ]

    print(f"[Running] Kernel {kernel}, Size {size}, Sparsity {sparsity_label}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Clear caches if cold-start mode enabled
        if args.cold_start:
            # Clear CUDA JIT cache (this is the critical one)
            clear_cuda_cache()

            if args.unload_driver:
                # Nuclear option: unload/reload driver
                unload_reload_driver()
            else:
                # Try GPU reset (may fail on EC2, but worth trying)
                reset_gpu(args.reset_delay)

        print(f"[Done] Kernel {kernel}, Size {size}, Sparsity {sparsity_label}")
        return {
            "success": True,
            "kernel": kernel,
            "size": size,
            "sparsity_label": sparsity_label,
            "sparsity_pattern": sparsity_pattern,
            "output_file": output_file,
        }
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Kernel {kernel}, Size {size}, Sparsity {sparsity_label}")
        print(f"  {e.stderr}")
        return {
            "success": False,
            "kernel": kernel,
            "size": size,
            "sparsity_label": sparsity_label,
            "error": str(e),
        }


def extract_metrics_from_report(ncu_rep_file, metrics_list):
    """Extract metrics from .ncu-rep file using ncu --import

    Returns dict with:
        - main_kernel: {kernel_time_us, memory_throughput_pct, compute_throughput_pct}
        - preprocess_kernels: list of {kernel_name, kernel_time_us, memory_throughput_pct, compute_throughput_pct}
        - total_preprocess_time_us: sum of all preprocessing kernel times
    """
    try:
        cmd = [
            "sudo",
            "/usr/local/cuda-12.1/bin/ncu",
            "--import",
            str(ncu_rep_file),
            "--csv",
            "--page",
            "details",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Each row: "ID","PID","Process","Host","Kernel","Context","Stream","Block Size","Grid Size","Device","CC","Section","Metric Name","Metric Unit","Metric Value",...
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return None

        main_metrics = {}
        # Track each preprocessing kernel separately by kernel name
        preprocess_kernels = {}

        for line in lines[1:]:  # Skip header
            if not line.strip():
                continue

            # Parse CSV line (handle quoted commas)
            parts = []
            current = ""
            in_quotes = False
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == "," and not in_quotes:
                    parts.append(current.strip('"'))
                    current = ""
                    continue
                current += char
            parts.append(current.strip('"'))

            if len(parts) < 15:
                continue

            kernel_name = parts[4]  # "Kernel Name" column
            metric_name = parts[12]  # "Metric Name" column
            metric_unit = parts[13]  # "Metric Unit" column
            metric_value = parts[14]  # "Metric Value" column

            # Determine if this is a preprocessing kernel (any kernel with "preprocess" in name)
            is_preprocess = "preprocess" in kernel_name.lower()

            if is_preprocess:
                # Create entry for this preprocessing kernel if not exists
                if kernel_name not in preprocess_kernels:
                    preprocess_kernels[kernel_name] = {"kernel_name": kernel_name}
                target_metrics = preprocess_kernels[kernel_name]
            else:
                target_metrics = main_metrics

            # Extract Duration in microseconds
            if metric_name == "Duration":
                metric_value = float(metric_value.replace(",", ""))
                if metric_unit == "second":
                    metric_value *= 1_000_000  # Convert seconds to microseconds
                elif metric_unit == "msecond":
                    metric_value *= 1000  # Convert milliseconds to microseconds
                # else: already in microseconds ("usecond")
                try:
                    target_metrics["kernel_time_us"] = metric_value
                except:
                    pass

            # Extract Memory Throughput percentage
            elif metric_name == "Memory Throughput" and metric_unit == "%":
                try:
                    target_metrics["memory_throughput_pct"] = float(
                        metric_value.replace(",", "")
                    )
                except:
                    pass

            # Extract Compute (SM) Throughput percentage
            elif metric_name == "Compute (SM) Throughput" and metric_unit == "%":
                try:
                    target_metrics["compute_throughput_pct"] = float(
                        metric_value.replace(",", "")
                    )
                except:
                    pass

        # Calculate total preprocessing time
        total_preprocess_time = sum(
            kernel.get("kernel_time_us", 0.0)
            for kernel in preprocess_kernels.values()
        )

        return {
            "main_kernel": main_metrics if main_metrics else None,
            "preprocess_kernels": list(preprocess_kernels.values()) if preprocess_kernels else [],
            "total_preprocess_time_us": total_preprocess_time,
        }

    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not extract metrics from {ncu_rep_file}: {e.stderr}")
        return None


def generate_summary_csv(results, output_dir, cold_start_mode):
    """Generate summary CSV from all results

    Includes preprocessing time once per matrix size (not per kernel)
    AND adds preprocessing time to kernel time for total end-to-end time
    """
    csv_path = output_dir / "summary.csv"

    # Track preprocessing time per size for adding to kernel times
    preprocess_time_by_size = {}
    preprocess_reported = {}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Add metadata header
        writer.writerow([f"# Cold-start mode: {'enabled' if cold_start_mode else 'disabled'}"])

        writer.writerow(
            [
                "kernel",
                "size",
                "sparsity",
                "pattern",
                "kernel_time_us",
                "memory_throughput_pct",
                "compute_throughput_pct",
                "ncu_report",
                "kernel_name",
            ]
        )

        for result in results:
            if not result["success"]:
                continue

            # Extract metrics from .ncu-rep file
            metrics = extract_metrics_from_report(result["output_file"], [])

            if not metrics:
                # No metrics extracted
                writer.writerow(
                    [
                        result["kernel"],
                        result["size"],
                        result["sparsity_label"],
                        result["sparsity_pattern"],
                        "N/A",
                        "N/A",
                        "N/A",
                        result["output_file"].name,
                        "",
                    ]
                )
                continue

            # Cache total preprocessing time per size
            size = result["size"]
            total_preprocess_time = metrics.get("total_preprocess_time_us", 0.0)
            preprocess_kernels = metrics.get("preprocess_kernels", [])

            if total_preprocess_time > 0 and size not in preprocess_time_by_size:
                preprocess_time_by_size[size] = total_preprocess_time

            # Write preprocessing entries once per size (all preprocessing kernels)
            if preprocess_kernels and size not in preprocess_reported:
                preprocess_reported[size] = True

                # Write each preprocessing kernel separately
                for preproc in preprocess_kernels:
                    writer.writerow(
                        [
                            "PREPROCESS",  # Special kernel ID
                            size,
                            result["sparsity_label"],
                            result["sparsity_pattern"],
                            (
                                f"{preproc.get('kernel_time_us', 'N/A'):.3f}"
                                if isinstance(preproc.get("kernel_time_us"), float)
                                else "N/A"
                            ),
                            (
                                f"{preproc.get('memory_throughput_pct', 'N/A'):.2f}"
                                if isinstance(preproc.get("memory_throughput_pct"), float)
                                else "N/A"
                            ),
                            (
                                f"{preproc.get('compute_throughput_pct', 'N/A'):.2f}"
                                if isinstance(preproc.get("compute_throughput_pct"), float)
                                else "N/A"
                            ),
                            result["output_file"].name,
                            preproc.get("kernel_name", ""),  # Actual preprocessing kernel name
                        ]
                    )

            # Write main kernel metrics (with preprocessing time added if applicable)
            main = metrics.get("main_kernel")
            if main:
                # Get kernel time and add preprocessing if it exists for this size
                kernel_time = main.get("kernel_time_us")
                if isinstance(kernel_time, float) and size in preprocess_time_by_size:
                    # Add preprocessing time to kernel time for total end-to-end time
                    total_time = kernel_time + preprocess_time_by_size[size]
                    kernel_time_str = f"{total_time:.3f}"
                elif isinstance(kernel_time, float):
                    kernel_time_str = f"{kernel_time:.3f}"
                else:
                    kernel_time_str = "N/A"

                writer.writerow(
                    [
                        result["kernel"],
                        result["size"],
                        result["sparsity_label"],
                        result["sparsity_pattern"],
                        kernel_time_str,
                        (
                            f"{main.get('memory_throughput_pct', 'N/A'):.2f}"
                            if isinstance(main.get("memory_throughput_pct"), float)
                            else "N/A"
                        ),
                        (
                            f"{main.get('compute_throughput_pct', 'N/A'):.2f}"
                            if isinstance(main.get("compute_throughput_pct"), float)
                            else "N/A"
                        ),
                        result["output_file"].name,
                        "",  # No kernel_name for main kernels (could add if needed)
                    ]
                )
            else:
                writer.writerow(
                    [
                        result["kernel"],
                        result["size"],
                        result["sparsity_label"],
                        result["sparsity_pattern"],
                        "N/A",
                        "N/A",
                        "N/A",
                        result["output_file"].name,
                        "",
                    ]
                )

    print(f"\nSummary CSV written to: {csv_path}")
    return csv_path


def main():
    args = parse_args()

    # Parse inputs
    kernels = parse_kernel_list(args.kernel)
    sizes = parse_size_list(args.sizes)
    sparsity_patterns = parse_sparsity_patterns(args.sparsity)

    # Warn if cold-start + parallel > 1
    if args.cold_start and args.parallel > 1:
        print("[Warning] Cold-start mode with parallel > 1 may have inconsistent results")
        print("[Warning] GPU reset affects ALL GPUs, not just the one running the kernel")
        print("[Warning] Consider using --parallel 1 for cold-start benchmarks\n")

    # Create output directory
    output_dir = create_output_dir(args, kernels)
    print(f"Output directory: {output_dir}")
    print(f"Cold-start mode: {'ENABLED' if args.cold_start else 'DISABLED'}")
    if args.cold_start:
        print(f"Reset delay: {args.reset_delay}s")
        # Clear cache BEFORE first run for true cold-start
        print("\n[Initializing] Clearing CUDA cache for cold-start...")
        clear_cuda_cache()
    print()

    # Generate all configurations
    configs = []
    for kernel in kernels:
        for size in sizes:
            for sparsity_label, sparsity_pattern in sparsity_patterns.items():
                output_file = output_dir / f"k{kernel}_{size}_{sparsity_label}.ncu-rep"
                configs.append(
                    (kernel, size, sparsity_label, sparsity_pattern, output_file, args)
                )

    print(f"Total configurations to run: {len(configs)}")
    print(f"Parallelism: {args.parallel}\n")

    # Run all configurations
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_config = {
            executor.submit(run_ncu_profile, config): config for config in configs
        }

        for future in as_completed(future_to_config):
            result = future.result()
            results.append(result)

    # Generate summary CSV
    print("\n" + "=" * 60)
    print("Extracting metrics from reports...")
    csv_path = generate_summary_csv(results, output_dir, args.cold_start)

    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print("\n" + "=" * 60)
    print(f"Benchmark complete!")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  Output: {output_dir}")
    print(f"  CSV Summary: {csv_path}")
    print(f"  Cold-start: {'YES' if args.cold_start else 'NO'}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
