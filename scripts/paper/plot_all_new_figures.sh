#!/usr/bin/env bash
# Generate all 6 paper figures (Figures 1–6 of ESMM paper).
# Figures 3 and 4 use hardcoded NCU-verified data from Tables 1–3.
# Figure 3 attempts NCU extraction for a 7-point curve, falls back to 4-point table data.
# Figure 6 reads results/real_weights_benchmark.csv.
#
# Output: results/paper_figures/fig{1..6}_*.{pdf,png}

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Figure 1: Skip hierarchy diagram ==="
python3 plot_fig1_skip_hierarchy.py

echo "=== Figure 2: K-tile layout / bitmask encoding ==="
python3 plot_fig2_ktile_layout.py

echo "=== Figure 3: Speedup curve (compute + e2e) ==="
python3 plot_fig3_speedup.py

echo "=== Figure 4: Kernel ablation compute times ==="
python3 plot_fig4_ablation.py

echo "=== Figure 5: Skip rate breakdown ==="
python3 plot_fig5_skiprates.py

echo "=== Figure 6: Real weights evaluation ==="
python3 plot_fig6_realweights.py

echo ""
echo "All figures saved to: results/paper_figures/"
ls -lh "$(dirname "$SCRIPT_DIR")/results/paper_figures/"
