#!/usr/bin/env python3
"""
Print clean summary of A×B combination benchmark results.

Usage:
    ./scripts/summarize_ab_results.py benchmarks/TIMESTAMP_k25_ab_grid/
    ./scripts/summarize_ab_results.py benchmarks/TIMESTAMP_k25_ab_grid/ --format table
    ./scripts/summarize_ab_results.py benchmarks/TIMESTAMP_k25_ab_grid/ --format csv
    ./scripts/summarize_ab_results.py benchmarks/TIMESTAMP_k25_ab_grid/ --format heatmap
"""

import argparse
import csv
import sys
from pathlib import Path


def load_consolidated_results(grid_dir):
    """Load consolidated results, creating it if needed"""
    grid_dir = Path(grid_dir)

    # Check for consolidated results
    consolidated = grid_dir / "consolidated_results.csv"

    # If in nested benchmarks/benchmarks, look there
    if not consolidated.exists():
        alt_dir = grid_dir.parent / "benchmarks" / grid_dir.name
        if alt_dir.exists():
            consolidated = alt_dir / "consolidated_results.csv"

    if not consolidated.exists():
        print("Error: consolidated_results.csv not found")
        print("Run: python3 scripts/consolidate_ab_results.py <grid_dir>")
        return None

    # Load and filter to main kernel only
    results = []
    with open(consolidated, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['kernel'] == '25':  # Main kernel only
                results.append({
                    'density_a': float(row['density_a_pct']),
                    'density_b': float(row['density_b_pct']),
                    'time_us': float(row['kernel_time_us']),
                    'pattern_a': row['pattern_a'],
                    'pattern_b': row['pattern_b'],
                })

    return sorted(results, key=lambda x: (x['density_a'], x['density_b']), reverse=True)


def print_list(results):
    """Print simple list format"""
    print(f"\n{'A%':<6} {'B%':<6} {'Time (µs)':<12} {'Time (ms)':<10} {'Speedup vs Dense':<18}")
    print("=" * 70)

    dense_time = next((r['time_us'] for r in results if r['density_a'] == 100.0 and r['density_b'] == 100.0), None)

    for r in results:
        speedup = dense_time / r['time_us'] if dense_time else 1.0
        print(f"{r['density_a']:<6.1f} {r['density_b']:<6.1f} {r['time_us']:<12.1f} {r['time_us']/1000:<10.3f} {speedup:<18.2f}x")


def print_table(results):
    """Print as table with A densities as rows, B densities as columns"""
    # Get unique densities
    a_densities = sorted(set(r['density_a'] for r in results), reverse=True)
    b_densities = sorted(set(r['density_b'] for r in results), reverse=True)

    # Create lookup
    lookup = {(r['density_a'], r['density_b']): r['time_us'] for r in results}

    print("\nKernel Time (µs): A-density (rows) × B-density (cols)")
    print("=" * 80)

    # Header
    print(f"{'A%':<8}", end="")
    for b in b_densities:
        print(f"{b:>8.1f}", end="")
    print()
    print("-" * 80)

    # Rows
    for a in a_densities:
        print(f"{a:<8.1f}", end="")
        for b in b_densities:
            time = lookup.get((a, b), 0)
            print(f"{time:>8.0f}", end="")
        print()


def print_csv(results):
    """Print as CSV"""
    print("density_a_pct,density_b_pct,kernel_time_us,kernel_time_ms")
    for r in results:
        print(f"{r['density_a']:.1f},{r['density_b']:.1f},{r['time_us']:.1f},{r['time_us']/1000:.3f}")


def print_heatmap(results):
    """Print ASCII heatmap"""
    # Get unique densities
    a_densities = sorted(set(r['density_a'] for r in results), reverse=True)
    b_densities = sorted(set(r['density_b'] for r in results), reverse=True)

    # Create lookup
    lookup = {(r['density_a'], r['density_b']): r['time_us'] for r in results}

    # Get min/max for scaling
    min_time = min(r['time_us'] for r in results)
    max_time = max(r['time_us'] for r in results)

    # Symbols from fastest to slowest
    symbols = [' ', '░', '▒', '▓', '█']

    print("\nPerformance Heatmap (darker = slower)")
    print(f"Range: {min_time:.0f} µs (lightest) to {max_time:.0f} µs (darkest)")
    print("=" * 80)

    # Header
    print(f"{'A%':<8}", end="")
    for b in b_densities:
        print(f"{b:>6.0f}%", end="")
    print()
    print("-" * 80)

    # Rows
    for a in a_densities:
        print(f"{a:<7.0f}%", end="")
        for b in b_densities:
            time = lookup.get((a, b), 0)
            # Normalize to 0-1
            normalized = (time - min_time) / (max_time - min_time) if max_time > min_time else 0
            # Pick symbol
            idx = int(normalized * (len(symbols) - 1))
            symbol = symbols[idx]
            print(f"  {symbol}{symbol}  ", end="")
        print()

    print()


def print_stats(results):
    """Print summary statistics"""
    times = [r['time_us'] for r in results]

    fastest = min(results, key=lambda x: x['time_us'])
    slowest = max(results, key=lambda x: x['time_us'])
    dense = next((r for r in results if r['density_a'] == 100.0 and r['density_b'] == 100.0), None)

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total combinations: {len(results)}")
    print(f"Time range: {min(times):.1f} - {max(times):.1f} µs")
    print(f"Mean time: {sum(times)/len(times):.1f} µs")
    print(f"Median time: {sorted(times)[len(times)//2]:.1f} µs")
    print()

    print("Fastest combination:")
    print(f"  A={fastest['density_a']:.1f}%, B={fastest['density_b']:.1f}%")
    print(f"  Time: {fastest['time_us']:.1f} µs ({fastest['time_us']/1000:.3f} ms)")
    if dense:
        print(f"  Speedup vs dense: {dense['time_us']/fastest['time_us']:.2f}x")
    print()

    print("Slowest combination:")
    print(f"  A={slowest['density_a']:.1f}%, B={slowest['density_b']:.1f}%")
    print(f"  Time: {slowest['time_us']:.1f} µs ({slowest['time_us']/1000:.3f} ms)")
    print()

    if dense:
        print("Fully dense baseline:")
        print(f"  A=100%, B=100%")
        print(f"  Time: {dense['time_us']:.1f} µs ({dense['time_us']/1000:.3f} ms)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Print clean summary of A×B benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Format options:
  list     - Simple list with speedups (default)
  table    - Table with A as rows, B as columns
  csv      - CSV format for importing
  heatmap  - ASCII heatmap visualization
  stats    - Summary statistics only

Examples:
  %(prog)s benchmarks/2025-12-07_180000_k25_ab_grid/
  %(prog)s benchmarks/2025-12-07_180000_k25_ab_grid/ --format table
  %(prog)s benchmarks/2025-12-07_180000_k25_ab_grid/ --format heatmap
        """
    )

    parser.add_argument(
        'grid_dir',
        help="Path to A×B grid directory"
    )
    parser.add_argument(
        '--format',
        choices=['list', 'table', 'csv', 'heatmap', 'stats'],
        default='list',
        help="Output format (default: list)"
    )

    args = parser.parse_args()

    # Load results
    results = load_consolidated_results(args.grid_dir)
    if not results:
        return 1

    # Print in requested format
    if args.format == 'list':
        print_list(results)
        print_stats(results)
    elif args.format == 'table':
        print_table(results)
        print_stats(results)
    elif args.format == 'csv':
        print_csv(results)
    elif args.format == 'heatmap':
        print_heatmap(results)
        print_stats(results)
    elif args.format == 'stats':
        print_stats(results)

    return 0


if __name__ == '__main__':
    sys.exit(main())
