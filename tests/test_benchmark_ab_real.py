import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from benchmark_ab_real import pad_to_tile, compute_block_sparsity, BM, BK


def test_pad_to_tile_no_op():
    arr = np.ones((64, 16), dtype=np.float32)
    result = pad_to_tile(arr)
    assert result.shape == (64, 16)
    assert np.array_equal(result, arr)


def test_pad_to_tile_pads_rows_and_cols():
    arr = np.ones((33, 9), dtype=np.float32)
    result = pad_to_tile(arr)
    assert result.shape == (64, 16)       # ceil(33/32)*32=64, ceil(9/8)*8=16
    assert result[33, 9] == 0.0           # padded region is zero
    assert result[0, 0] == 1.0            # original data preserved


def test_compute_block_sparsity_fully_dense():
    arr = np.ones((64, 16), dtype=np.float32)
    assert compute_block_sparsity(arr) == 0.0


def test_compute_block_sparsity_fully_sparse():
    arr = np.zeros((64, 16), dtype=np.float32)
    assert compute_block_sparsity(arr) == 1.0


def test_compute_block_sparsity_half():
    arr = np.zeros((64, 16), dtype=np.float32)
    arr[:32, :] = 1.0    # first row of tiles is non-zero
    result = compute_block_sparsity(arr)
    assert result == pytest.approx(0.5)


from benchmark_ab_real import find_compatible_pairs


def _make_pt(tmp_path, name, shape):
    """Write a fake .pt tensor file and return its Path."""
    p = tmp_path / name
    torch.save(torch.zeros(shape), p)
    return p


def test_find_compatible_pairs_matching(tmp_path):
    a = _make_pt(tmp_path, "a_permuted.pt", (4096, 4096))
    b = _make_pt(tmp_path, "b_permuted.pt", (4096, 4096))
    pairs = find_compatible_pairs([a], [b])
    assert pairs == [(a, b)]


def test_find_compatible_pairs_shape_mismatch(tmp_path):
    a = _make_pt(tmp_path, "a_permuted.pt", (4096, 4096))
    b = _make_pt(tmp_path, "b_permuted.pt", (512, 4096))   # B.rows=512 != A.cols=4096
    pairs = find_compatible_pairs([a], [b])
    assert pairs == []


def test_find_compatible_pairs_skips_k_too_large(tmp_path):
    a = _make_pt(tmp_path, "a_permuted.pt", (4096, 11008))  # K=11008 > 8192
    b = _make_pt(tmp_path, "b_permuted.pt", (11008, 4096))
    pairs = find_compatible_pairs([a], [b])
    assert pairs == []


def test_find_compatible_pairs_skips_identical_path(tmp_path):
    a = _make_pt(tmp_path, "a_permuted.pt", (4096, 4096))
    pairs = find_compatible_pairs([a], [a])
    assert pairs == []
