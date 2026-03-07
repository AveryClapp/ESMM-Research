#!/usr/bin/env python3
"""
Analyze real LLM weight tensors from Wanda/SparseGPT pruning.
Computes element-wise and block sparsity, exports float32 binaries for ESMM benchmarking.

Usage:
  # Explore structure of .pt files
  python3 scripts/analyze_real_weights.py --explore weight_permutations/

  # Full analysis + binary export
  python3 scripts/analyze_real_weights.py --analyze weight_permutations/ --out exports/real_weights
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

BM = 32   # row tile size — matches ESMM preprocessor TILE_M_A
BK = 8    # col tile size — matches ESMM BK


# ── Exploration ────────────────────────────────────────────────────────────────

def explore_file(pt_path: Path):
    print(f"\n=== {pt_path.name} ===")
    obj = torch.load(pt_path, map_location="cpu", weights_only=True)

    if isinstance(obj, torch.Tensor):
        t = obj
        arr = t.float().numpy()
        zeros = np.sum(arr == 0)
        total = arr.size
        print(f"  Tensor  shape={t.shape}  dtype={t.dtype}  "
              f"sparsity={zeros/total*100:.1f}%")
    elif isinstance(obj, dict):
        print(f"  dict  keys={list(obj.keys())[:8]}")
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                arr = v.float().numpy()
                sparsity = np.sum(arr == 0) / arr.size
                print(f"    [{k}] shape={v.shape} dtype={v.dtype} "
                      f"sparsity={sparsity*100:.1f}%")
            else:
                print(f"    [{k}] {type(v).__name__} = {str(v)[:60]}")
    else:
        print(f"  {type(obj).__name__}: {str(obj)[:200]}")


# ── Block sparsity ─────────────────────────────────────────────────────────────

def compute_block_sparsity(arr: np.ndarray, bm: int = BM, bk: int = BK):
    """
    Returns (block_sparsity_fraction, heatmap_2d).
    heatmap[i,j] = 1.0 if tile (i,j) is entirely zero, else 0.0.
    Array must already be divisible by (bm, bk).
    """
    M, K = arr.shape
    nM, nK = M // bm, K // bk
    tiles = arr.reshape(nM, bm, nK, bk)
    all_zero = (np.abs(tiles).max(axis=(1, 3)) == 0).astype(np.float32)
    return float(all_zero.mean()), all_zero   # scalar, (nM x nK) heatmap


def pad_to_tile(arr: np.ndarray, bm: int = BM, bk: int = BK) -> np.ndarray:
    M, K = arr.shape
    M_pad = ((M + bm - 1) // bm) * bm
    K_pad = ((K + bk - 1) // bk) * bk
    if M_pad == M and K_pad == K:
        return arr
    out = np.zeros((M_pad, K_pad), dtype=arr.dtype)
    out[:M, :K] = arr
    return out


# ── Export ─────────────────────────────────────────────────────────────────────

def plot_heatmap(heatmap: np.ndarray, title: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="All-zero tile")
    ax.set_xlabel(f"K-blocks (BK={BK} cols each)")
    ax.set_ylabel(f"M-blocks (BM={BM} rows each)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=80)
    plt.close()


def load_tensor_2d(pt_path: Path) -> tuple:
    """Load .pt file, return (arr_float32, orig_shape). Skips non-2D tensors."""
    obj = torch.load(pt_path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        # Take the largest 2D float tensor
        candidates = {k: v for k, v in obj.items()
                      if isinstance(v, torch.Tensor) and v.dim() == 2
                      and v.dtype in (torch.float32, torch.float16, torch.bfloat16)}
        if not candidates:
            return None, None
        key = max(candidates, key=lambda k: candidates[k].numel())
        t = candidates[key]
    elif isinstance(obj, torch.Tensor) and obj.dim() == 2:
        t = obj
    else:
        return None, None

    return t.float().numpy(), tuple(t.shape)


def analyze_and_export(pt_path: Path, out_dir: Path, config_meta: dict):
    arr, orig_shape = load_tensor_2d(pt_path)
    if arr is None:
        print(f"  [SKIP] No 2D tensor in {pt_path.name}")
        return None

    orig_M, orig_K = orig_shape
    elem_sparsity = float(np.sum(arr == 0) / arr.size)

    # Pad to tile boundaries
    arr_padded = pad_to_tile(arr)
    M, K = arr_padded.shape
    block_sparsity, heatmap = compute_block_sparsity(arr_padded)

    # Unique name: {pruner}_{grp}_{perm}_sp{sparsity}_{stem}
    pruner = config_meta.get("pruner", "unk")
    grp    = config_meta.get("group_size", "unk")
    perm   = config_meta.get("perm_type", "unk")
    sp     = config_meta.get("sparsity", "unk").replace(".", "")
    name = f"{pruner}_grp{grp}_{perm}_sp{sp}_{pt_path.stem}"
    bin_path = out_dir / f"{name}.bin"
    json_path = out_dir / f"{name}.json"
    heatmap_path = out_dir / f"{name}_heatmap.png"

    # Write binary (padded, row-major float32)
    arr_padded.astype(np.float32).tofile(str(bin_path))

    # Write metadata
    meta = {
        **config_meta,
        "source": str(pt_path),
        "name": name,
        "layer": pt_path.stem,
        "orig_M": orig_M, "orig_K": orig_K,
        "M": M, "K": K,
        "elem_sparsity": elem_sparsity,
        "block_sparsity_bm32_bk8": block_sparsity,
        "bin_file": bin_path.name,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Heatmap
    title = (f"{name}  [{orig_M}×{orig_K}]  "
             f"elem={elem_sparsity*100:.1f}%  block={block_sparsity*100:.1f}%")
    plot_heatmap(heatmap, title, heatmap_path)

    print(f"  [{orig_M}×{orig_K}] elem={elem_sparsity*100:.1f}%  "
          f"block={block_sparsity*100:.1f}%  → {bin_path.name}")
    return meta


def parse_config_from_path(pt_path: Path) -> dict:
    """Extract pruner/group_size/perm_type/sparsity from directory structure."""
    parts = pt_path.parts
    config = {"pruner": "", "group_size": "", "perm_type": "", "sparsity": ""}
    for p in parts:
        if p in ("wanda", "sparsegpt"):
            config["pruner"] = p
        if "grp_8" in p:
            config["group_size"] = "8"
        elif "grp_16" in p:
            config["group_size"] = "16"
        if "col_perm" in p or "columns_permuted" in p or "col_" in p:
            config["perm_type"] = "col"
        elif "row_perm" in p or "rows_permuted" in p or "row_" in p:
            config["perm_type"] = "row"
        if "sparsity_" in p:
            config["sparsity"] = p.split("sparsity_")[1].rstrip("/")
    return config


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze real LLM weight tensors")
    parser.add_argument("--explore", help="Path to .pt file or dir — print structure only")
    parser.add_argument("--analyze", help="Path to .pt file or dir — full analysis + export")
    parser.add_argument("--out", default="exports/real_weights",
                        help="Output dir for binaries, JSONs, heatmaps (default: exports/real_weights)")
    args = parser.parse_args()

    if args.explore:
        target = Path(args.explore)
        if target.is_file():
            explore_file(target)
        else:
            pt_files = sorted(target.rglob("*.pt"))
            print(f"Found {len(pt_files)} .pt files")
            for f in pt_files[:20]:
                explore_file(f)

    elif args.analyze:
        target = Path(args.analyze)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        pt_files = sorted(target.rglob("*_permuted.pt")) if target.is_dir() else [target]
        print(f"Analyzing {len(pt_files)} weight tensors → {out_dir}\n")

        all_meta = []
        for f in pt_files:
            config = parse_config_from_path(f)
            label = (f"{config['pruner']}/{config['group_size']}/"
                     f"{config['perm_type']}/sp{config['sparsity']}/"
                     f"{f.stem}")
            print(f"{label}")
            meta = analyze_and_export(f, out_dir, config)
            if meta:
                all_meta.append(meta)

        # Summary table
        if all_meta:
            print(f"\n{'─'*90}")
            print(f"{'Name':<45} {'Shape':>12} {'Elem%':>7} {'Block%':>8} {'Pruner':<10} {'Sp':>6}")
            print(f"{'─'*90}")
            for r in all_meta:
                name = r["name"][:45]
                shape = f"({r['orig_M']},{r['orig_K']})"
                print(f"{name:<45} {shape:>12} "
                      f"{r['elem_sparsity']*100:>6.1f}% "
                      f"{r['block_sparsity_bm32_bk8']*100:>7.1f}% "
                      f"{r['pruner']:<10} {r['sparsity']:>6}")
            print(f"\n{len(all_meta)} tensors exported to {out_dir}/")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
