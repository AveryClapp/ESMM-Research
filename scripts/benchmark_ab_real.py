#!/usr/bin/env python3
"""
Benchmark ESMM kernels vs cuBLAS on real LLM weight pairs (both A and B loaded).

Finds all compatible A×B pairs from two directories, benchmarks each under NCU,
and writes results to CSV.

Usage:
  python3 scripts/benchmark_ab_real.py \
    --weights-a weight_permutations/.../sparsity_0.99/ \
    --weights-b weight_permutations/.../sparsity_0.99/ \
    --kernels 15,29
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
_candidates = ["exec_prod", "exec_dev", "exec"]
EXEC = next((REPO_ROOT / e for e in _candidates if (REPO_ROOT / e).exists()), REPO_ROOT / "exec_dev")

NCU_PATH = "/usr/local/cuda-12.1/bin/ncu"
NCU_ENV = {**os.environ, "LD_LIBRARY_PATH": f"/usr/local/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"}

BM = 32
BK = 8
MAX_K = 1024 * BK  # 8192 — K29 smem limit


def pad_to_tile(arr: np.ndarray) -> np.ndarray:
    M, K = arr.shape
    M_pad = ((M + BM - 1) // BM) * BM
    K_pad = ((K + BK - 1) // BK) * BK
    if M_pad == M and K_pad == K:
        return arr
    out = np.zeros((M_pad, K_pad), dtype=arr.dtype)
    out[:M, :K] = arr
    return out


def compute_block_sparsity(arr: np.ndarray) -> float:
    M, K = arr.shape
    tiles = arr.reshape(M // BM, BM, K // BK, BK)
    all_zero = np.abs(tiles).max(axis=(1, 3)) == 0
    return float(all_zero.mean())


def parse_config(pt_path: Path) -> dict:
    parts = pt_path.parts
    config = {"pruner": "", "group_size": "", "perm_type": "", "sparsity": ""}
    for p in parts:
        if p in ("wanda", "sparsegpt"):
            config["pruner"] = p
        if "grp_8" in p:
            config["group_size"] = "8"
        elif "grp_16" in p:
            config["group_size"] = "16"
        if "col_perm" in p or "columns_permuted" in p:
            config["perm_type"] = "col"
        elif "row_perm" in p or "rows_permuted" in p:
            config["perm_type"] = "row"
        if "sparsity_" in p:
            config["sparsity"] = p.split("sparsity_")[1].rstrip("/")
    return config


def _parse_csv_line(line: str) -> list:
    """Parse a single CSV line handling quoted commas."""
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
    return parts


def find_compatible_pairs(a_files: list, b_files: list) -> list:
    """Return (a_path, b_path) tuples where A.cols == B.rows and K <= MAX_K."""
    b_shapes = {}
    for b_path in b_files:
        try:
            b_obj = torch.load(b_path, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"  [SKIP B] {b_path.name}: {e}")
            continue
        if not isinstance(b_obj, torch.Tensor) or b_obj.dim() != 2:
            print(f"  [SKIP B] {b_path.name}: not a 2D tensor")
            continue
        b_shapes[b_path] = b_obj.shape

    pairs = []
    for a_path in a_files:
        try:
            a_obj = torch.load(a_path, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if not isinstance(a_obj, torch.Tensor) or a_obj.dim() != 2:
            continue
        a_K = a_obj.shape[1]
        if a_K > MAX_K:
            print(f"  [SKIP A] {a_path.name}: K={a_K} > {MAX_K}")
            continue
        for b_path, b_shape in b_shapes.items():
            if a_path == b_path:
                continue
            if b_shape[0] == a_K:
                pairs.append((a_path, b_path))
    return pairs


def run_pair(kernel: int, a_bin: str, b_bin: str, rows: int, inners: int, cols: int):
    """Run kernel under NCU with both A and B loaded from binary files.

    Returns (compute_ms, preprocess_ms) or (None, None) on failure.
    """
    tmp_dir = tempfile.mkdtemp()
    ncu_base = os.path.join(tmp_dir, "profile")
    ncu_rep = ncu_base + ".ncu-rep"

    ncu_cmd = [
        "sudo", NCU_PATH,
        "--set", "basic",
        "--target-processes", "all",
        "--export", ncu_base,
        "--force-overwrite",
        str(EXEC), str(kernel),
        "--load-a", a_bin,
        "--load-b", b_bin,
        "--dims", str(rows), str(inners), str(cols),
        "--no-check",
    ]

    try:
        result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=300, env=NCU_ENV)
        if result.returncode != 0:
            print(f"    [NCU ERROR] kernel={kernel}: {result.stderr[:300]}")
            return None, None

        import_cmd = [
            "sudo", NCU_PATH,
            "--import", ncu_rep,
            "--csv", "--page", "details",
        ]
        imp = subprocess.run(import_cmd, capture_output=True, text=True, timeout=60, env=NCU_ENV)
        if imp.returncode != 0:
            print(f"    [NCU IMPORT ERROR] kernel={kernel}: {imp.stderr[:200]}")
            return None, None

        compute_us = 0.0
        preprocess_us = 0.0

        lines = imp.stdout.strip().split("\n")
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = _parse_csv_line(line)
            if len(parts) < 15:
                continue

            kernel_name = parts[4]
            metric_name = parts[12]
            metric_unit = parts[13]
            metric_value_str = parts[14]

            if metric_name != "Duration":
                continue

            try:
                val = float(metric_value_str.replace(",", ""))
            except ValueError:
                continue

            if metric_unit == "second":
                val_us = val * 1e6
            elif metric_unit == "msecond":
                val_us = val * 1e3
            else:
                val_us = val  # usecond

            kn_lower = kernel_name.lower()
            if "preprocess" in kn_lower or "analyze" in kn_lower:
                preprocess_us += val_us
            else:
                compute_us += val_us

        if compute_us == 0.0:
            print(f"    [WARN] No Duration for kernel={kernel}. stdout[:300]:\n{imp.stdout[:300]}")
            return None, None

        return compute_us / 1000.0, preprocess_us / 1000.0

    except subprocess.TimeoutExpired:
        print(f"    [NCU TIMEOUT] kernel={kernel}")
        return None, None
    except Exception as e:
        print(f"    [ERROR] kernel={kernel}: {e}")
        return None, None
    finally:
        try:
            os.unlink(ncu_rep)
        except OSError:
            pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
