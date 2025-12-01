#!/usr/bin/env python3
"""
Parallel Benchmark Runner for ESMM Kernels

Automates running ncu profiles across multiple matrix sizes and sparsity patterns.
Generates organized .ncu-rep files and consolidated CSV with key metrics.

Usage:
    ./scripts/benchmark.py --kernel 17
    ./scripts/benchmark.py -k 17,22,23 --sizes 1024,2048 --parallel 2
"""

import argparse
import csv
import os
import re
import subprocess
import sys
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
        dirname = f"{timestamp}_{kernel_str}"

    output_dir = Path("benchmarks") / dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
        str(args.runs),
        "--size",
        str(size),
        "--pattern",
        sparsity_pattern,
        "--no-check",  # Skip verification for speed
    ]

    print(f"[Running] Kernel {kernel}, Size {size}, Sparsity {sparsity_label}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
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
    """Extract metrics from .ncu-rep file using ncu --import"""
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

        metrics = {}
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

            metric_name = parts[12]  # "Metric Name" column
            metric_unit = parts[13]  # "Metric Unit" column
            metric_value = parts[14]  # "Metric Value" column

            # Extract Duration in microseconds
            if metric_name == "Duration" and metric_unit == "usecond":
                try:
                    metrics["kernel_time_us"] = float(metric_value.replace(",", ""))
                except:
                    pass

            # Extract Memory Throughput percentage
            elif metric_name == "Memory Throughput" and metric_unit == "%":
                try:
                    metrics["memory_throughput_pct"] = float(
                        metric_value.replace(",", "")
                    )
                except:
                    pass

            # Extract Compute (SM) Throughput percentage
            elif metric_name == "Compute (SM) Throughput" and metric_unit == "%":
                try:
                    metrics["compute_throughput_pct"] = float(
                        metric_value.replace(",", "")
                    )
                except:
                    pass

        return metrics

    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not extract metrics from {ncu_rep_file}: {e.stderr}")
        return None


def generate_summary_csv(results, output_dir):
    """Generate summary CSV from all results"""
    csv_path = output_dir / "summary.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
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
            ]
        )

        for result in results:
            if not result["success"]:
                continue

            # Extract metrics from .ncu-rep file
            metrics = extract_metrics_from_report(result["output_file"], [])

            if metrics:
                writer.writerow(
                    [
                        result["kernel"],
                        result["size"],
                        result["sparsity_label"],
                        result["sparsity_pattern"],
                        (
                            f"{metrics.get('kernel_time_us', 'N/A'):.3f}"
                            if isinstance(metrics.get("kernel_time_us"), float)
                            else "N/A"
                        ),
                        (
                            f"{metrics.get('memory_throughput_pct', 'N/A'):.2f}"
                            if isinstance(metrics.get("memory_throughput_pct"), float)
                            else "N/A"
                        ),
                        (
                            f"{metrics.get('compute_throughput_pct', 'N/A'):.2f}"
                            if isinstance(metrics.get("compute_throughput_pct"), float)
                            else "N/A"
                        ),
                        result["output_file"].name,
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

    # Create output directory
    output_dir = create_output_dir(args, kernels)
    print(f"Output directory: {output_dir}\n")

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
    csv_path = generate_summary_csv(results, output_dir)

    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print("\n" + "=" * 60)
    print(f"Benchmark complete!")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  Output: {output_dir}")
    print(f"  CSV Summary: {csv_path}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
