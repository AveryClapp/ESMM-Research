"""
Shared matplotlib style for ESMM paper figures.
Import and call apply() at the top of every plot script.

Column widths (ACM/IEEE two-column):
  - Single column:  3.33 in  (~85 mm)
  - Double column:  7.00 in  (~178 mm)
  - Full page:      9.00 in  (~229 mm)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Consistent color palette (colorblind-friendly)
COLORS = {
    "cuBLAS":               "#555555",
    "AB-Cached-64":         "#4e79a7",   # blue
    "AB-Cached-32-noskip":  "#f28e2b",   # orange
    "AB-Cached-32":         "#59a14f",   # green
    "AB-Stream-32":         "#b07aa1",   # purple
    "ESMM":                 "#e15759",   # red (main contribution)
    "cuSPARSE":             "#76b7b2",   # teal
}

# Hatches for grayscale-safe plots (optional)
HATCHES = {
    "cuBLAS":               "",
    "AB-Cached-64":         "//",
    "AB-Cached-32-noskip":  "xx",
    "AB-Cached-32":         "\\\\",
    "AB-Stream-32":         "..",
    "ESMM":                 "",
}

# Single-column and double-column widths
W1 = 3.33   # single column
W2 = 7.00   # double column
W3 = 9.00   # full page


def apply(usetex=False):
    """Apply paper-quality rcParams. Call once at module level."""
    mpl.rcParams.update({
        # Font
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":            9,
        "axes.titlesize":       10,
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.fontsize":      8,
        "legend.title_fontsize":8,

        # Lines and markers
        "lines.linewidth":      1.5,
        "lines.markersize":     5,
        "patch.linewidth":      0.6,

        # Axes
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.8,
        "axes.grid":            True,
        "axes.grid.axis":       "y",
        "grid.alpha":           0.25,
        "grid.linewidth":       0.5,

        # Ticks
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "xtick.major.size":     3.5,
        "ytick.major.size":     3.5,
        "xtick.direction":      "out",
        "ytick.direction":      "out",

        # Legend
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#cccccc",
        "legend.borderpad":     0.4,
        "legend.handlelength":  1.5,

        # Figure
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,

        # PDF/vector output
        "pdf.fonttype":         42,   # embed fonts as Type 42 (TrueType)
        "ps.fonttype":          42,
    })

    if usetex:
        mpl.rcParams.update({
            "text.usetex":      True,
            "font.family":      "serif",
            "font.serif":       ["Computer Modern"],
        })
