import sys
from pathlib import Path
import numpy as np
import pytest

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
