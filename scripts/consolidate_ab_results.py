#!/usr/bin/env python3
"""
Consolidate all A×B combination results into a single CSV file.

Usage:
    ./scripts/consolidate_ab_results.py benchmarks/2025-XX-XX_XXXXXX_k25_ab_grid/
    ./scripts/consolidate_ab_results.py benchmarks/2025-XX-XX_XXXXXX_k25_ab_grid/ --output results.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def consolidate_results(grid_dir, output_file=None):
    """Consolidate all summary.csv files from A×B grid into one master CSV"""
    grid_dir = Path(grid_dir)

    if not grid_dir.exists():
        print(f"Error: Directory {grid_dir} does not exist")
        return 1

    # Read the combinations_summary.csv to get metadata
    combinations_file = grid_dir / "combinations_summary.csv"
    if not combinations_file.exists():
        print(f"Error: {combinations_file} not found")
        print("This doesn't look like an A×B grid directory from benchmark_all_ab_combinations.sh")
        return 1

    # Parse combinations metadata
    combinations = {}
    with open(combinations_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            output_dir = Path(row['output_dir'])
            dir_name = output_dir.name
            combinations[dir_name] = {
                'pattern_a': row['pattern_a'],
                'pattern_b': row['pattern_b'],
                'density_a_pct': row['density_a_pct'],
                'density_b_pct': row['density_b_pct'],
            }

    print(f"Found {len(combinations)} A×B combinations")

    # Collect all results
    all_results = []
    missing_count = 0

    for subdir in sorted(grid_dir.iterdir()):
        if not subdir.is_dir():
            continue

        summary_file = subdir / "summary.csv"
        if not summary_file.exists():
            print(f"Warning: {summary_file} not found")
            missing_count += 1
            continue

        # Get metadata for this combination
        dir_name = subdir.name
        if dir_name not in combinations:
            print(f"Warning: {dir_name} not in combinations_summary.csv")
            continue

        metadata = combinations[dir_name]

        # Read the summary.csv (skip header comment line)
        with open(summary_file, 'r') as f:
            lines = f.readlines()
            # Skip comment lines
            data_lines = [line for line in lines if not line.startswith('#')]

            reader = csv.DictReader(data_lines)
            for row in reader:
                # Add A/B metadata to each row
                result = {
                    'pattern_a': metadata['pattern_a'],
                    'pattern_b': metadata['pattern_b'],
                    'density_a_pct': metadata['density_a_pct'],
                    'density_b_pct': metadata['density_b_pct'],
                    **row  # Add all columns from summary.csv
                }
                all_results.append(result)

    if missing_count > 0:
        print(f"Warning: {missing_count} subdirectories missing summary.csv")

    if not all_results:
        print("Error: No results found!")
        return 1

    # Determine output file
    if output_file is None:
        output_file = grid_dir / "consolidated_results.csv"
    else:
        output_file = Path(output_file)

    # Write consolidated CSV
    fieldnames = ['pattern_a', 'pattern_b', 'density_a_pct', 'density_b_pct'] + [
        k for k in all_results[0].keys()
        if k not in ['pattern_a', 'pattern_b', 'density_a_pct', 'density_b_pct']
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✅ Consolidated {len(all_results)} results")
    print(f"   Output: {output_file}")

    # Print summary statistics
    print(f"\n📊 Summary:")
    print(f"   Total runs: {len(all_results)}")

    if 'kernel_time_us' in all_results[0]:
        times = [float(r['kernel_time_us']) for r in all_results if r['kernel_time_us'] not in ['N/A', '']]
        if times:
            print(f"   Kernel time range: {min(times):.1f} - {max(times):.1f} µs")
            print(f"   Kernel time mean: {sum(times)/len(times):.1f} µs")

    # Count unique A/B combinations
    unique_combos = set((r['density_a_pct'], r['density_b_pct']) for r in all_results)
    print(f"   Unique A×B combinations: {len(unique_combos)}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate A×B combination benchmark results into single CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s benchmarks/2025-12-07_180000_k25_ab_grid/
  %(prog)s benchmarks/2025-12-07_180000_k25_ab_grid/ --output my_results.csv
        """
    )

    parser.add_argument(
        'grid_dir',
        help="Path to A×B grid directory (output from benchmark_all_ab_combinations.sh)"
    )
    parser.add_argument(
        '-o', '--output',
        help="Output CSV file (default: grid_dir/consolidated_results.csv)"
    )

    args = parser.parse_args()

    return consolidate_results(args.grid_dir, args.output)


if __name__ == '__main__':
    sys.exit(main())
