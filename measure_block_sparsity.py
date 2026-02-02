#!/usr/bin/env python3
"""
Block Sparsity Improvement Analysis

Measures how much row clustering improves block-structured sparsity capture.
This tells you if the clustering preprocessing is worth implementing for your workload.

Usage:
    python measure_block_sparsity.py

Requirements:
    pip install torch
"""

import torch
import time

def count_zero_blocks(A, bm, bn):
    """Count fraction of bm×bn blocks that are completely zero."""
    M, K = A.shape
    zero_blocks = 0
    total_blocks = (M // bm) * (K // bn)
    
    for i in range(0, M, bm):
        for j in range(0, K, bn):
            if A[i:i+bm, j:j+bn].abs().max() == 0:
                zero_blocks += 1
    
    return zero_blocks / total_blocks


def count_zero_blocks_fast(A, bm, bn):
    """Vectorized version - much faster for large matrices."""
    M, K = A.shape
    # Reshape into blocks
    A_blocks = A[:M//bm*bm, :K//bn*bn].reshape(M//bm, bm, K//bn, bn)
    # Check if each block is all zeros
    block_maxes = A_blocks.abs().amax(dim=(1, 3))  # Max over bm and bn dims
    zero_blocks = (block_maxes == 0).sum().item()
    total_blocks = (M // bm) * (K // bn)
    return zero_blocks / total_blocks


def measure_block_improvement(activations, block_m=8, block_n=32, verbose=True):
    """
    Measure block sparsity before and after row clustering.
    
    Args:
        activations: [M, K] tensor
        block_m: Row block size (default 8, matching K25)
        block_n: Column block size (default 32, matching K25's WN)
    
    Returns:
        (before, after, improvement_ratio)
    """
    M, K = activations.shape
    
    # Before: original order
    before = count_zero_blocks_fast(activations, block_m, block_n)
    
    # After: sort rows by density (most sparse rows first)
    row_zeros = (activations == 0).sum(dim=1)
    perm = torch.argsort(row_zeros, descending=True)  # Most zeros first
    reordered = activations[perm]
    
    after = count_zero_blocks_fast(reordered, block_m, block_n)
    
    improvement = after / max(before, 0.001)
    
    if verbose:
        print(f"  Block sparsity: {before*100:.1f}% → {after*100:.1f}% ({improvement:.2f}x improvement)")
    
    return before, after, improvement


def generate_sparse_activations(M, K, element_sparsity, pattern="random"):
    """
    Generate synthetic sparse activations.
    
    Patterns:
        "random": Uniform random sparsity (worst case for blocking)
        "structured": Some rows are denser than others (more realistic)
        "relu": Simulates ReLU output (values are 0 or positive)
    """
    if pattern == "random":
        # Uniform random - each element independently zero with probability p
        mask = torch.rand(M, K) < element_sparsity
        A = torch.randn(M, K)
        A[mask] = 0
        
    elif pattern == "structured":
        # Some rows are very sparse, others are denser
        # This is more realistic for transformer activations
        A = torch.randn(M, K)
        row_sparsity = torch.rand(M) * 0.3 + (element_sparsity - 0.15)  # Vary ±15%
        row_sparsity = row_sparsity.clamp(0, 1)
        for i in range(M):
            mask = torch.rand(K) < row_sparsity[i]
            A[i, mask] = 0
            
    elif pattern == "relu":
        # ReLU: negative values become 0
        A = torch.randn(M, K)
        # Shift distribution to get desired sparsity
        # For ~90% sparsity, shift mean negative
        shift = torch.distributions.Normal(0, 1).icdf(torch.tensor(element_sparsity))
        A = A - shift
        A = torch.relu(A)
        
    elif pattern == "block_structured":
        # Some K-regions are sparser than others (simulates attention patterns)
        A = torch.randn(M, K)
        # Divide K into regions with different sparsity
        num_regions = K // 64
        for r in range(num_regions):
            region_sparsity = torch.rand(1).item() * 0.4 + (element_sparsity - 0.2)
            region_sparsity = max(0, min(1, region_sparsity))
            mask = torch.rand(M, 64) < region_sparsity
            A[:, r*64:(r+1)*64][mask] = 0
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return A


def run_experiment(M, K, element_sparsity, pattern, block_m=8, block_n=32):
    """Run a single experiment and return results."""
    A = generate_sparse_activations(M, K, element_sparsity, pattern)
    
    # Verify actual element sparsity
    actual_sparsity = (A == 0).float().mean().item()
    
    before, after, improvement = measure_block_improvement(A, block_m, block_n, verbose=False)
    
    return {
        "M": M,
        "K": K,
        "target_sparsity": element_sparsity,
        "actual_sparsity": actual_sparsity,
        "pattern": pattern,
        "block_m": block_m,
        "block_n": block_n,
        "before": before,
        "after": after,
        "improvement": improvement,
    }


def main():
    print("=" * 70)
    print("Block Sparsity Improvement from Row Clustering")
    print("=" * 70)
    print()
    print("This measures how much block-structured sparsity can be improved")
    print("by reordering rows to cluster sparse regions together.")
    print()
    print("Your K25 kernel uses 8×32 blocks. For it to skip computation,")
    print("entire 8×32 blocks must be zero - not just individual elements.")
    print()
    
    # Test configurations
    sizes = [(4096, 4096), (2048, 8192), (8192, 2048)]
    sparsities = [0.5, 0.7, 0.9, 0.95]
    patterns = ["random", "structured", "relu", "block_structured"]
    
    print("=" * 70)
    print("Experiment 1: Effect of Sparsity Level")
    print("=" * 70)
    print(f"Matrix size: 4096 × 4096, Pattern: structured")
    print()
    
    for sparsity in sparsities:
        A = generate_sparse_activations(4096, 4096, sparsity, "structured")
        actual = (A == 0).float().mean().item()
        print(f"Element sparsity: {actual*100:.1f}%")
        measure_block_improvement(A, block_m=8, block_n=32)
        print()
    
    print("=" * 70)
    print("Experiment 2: Effect of Sparsity Pattern")
    print("=" * 70)
    print(f"Matrix size: 4096 × 4096, Element sparsity: 90%")
    print()
    
    for pattern in patterns:
        A = generate_sparse_activations(4096, 4096, 0.9, pattern)
        actual = (A == 0).float().mean().item()
        print(f"Pattern: {pattern} (actual sparsity: {actual*100:.1f}%)")
        measure_block_improvement(A, block_m=8, block_n=32)
        print()
    
    print("=" * 70)
    print("Experiment 3: Effect of Block Size")
    print("=" * 70)
    print(f"Matrix size: 4096 × 4096, Element sparsity: 90%, Pattern: structured")
    print()
    
    A = generate_sparse_activations(4096, 4096, 0.9, "structured")
    block_configs = [(8, 32), (8, 64), (16, 32), (32, 32)]
    
    for bm, bn in block_configs:
        print(f"Block size: {bm}×{bn}")
        measure_block_improvement(A, block_m=bm, block_n=bn)
        print()
    
    print("=" * 70)
    print("Experiment 4: Different Matrix Sizes")
    print("=" * 70)
    print(f"Element sparsity: 90%, Pattern: structured")
    print()
    
    for M, K in sizes:
        A = generate_sparse_activations(M, K, 0.9, "structured")
        print(f"Matrix size: {M} × {K}")
        measure_block_improvement(A, block_m=8, block_n=32)
        print()
    
    print("=" * 70)
    print("SUMMARY: Decision Guide")
    print("=" * 70)
    print()
    print("For row clustering to be worthwhile with your K25 kernel:")
    print("  1. Block sparsity AFTER clustering should be ≥40%")
    print("     (Below this, cuBLAS wins based on your crossover analysis)")
    print()
    print("  2. Improvement ratio should be ≥1.5x")
    print("     (Otherwise overhead isn't justified)")
    print()
    print("  3. Element sparsity should be ≥70%")
    print("     (Lower sparsity rarely clusters well into blocks)")
    print()
    
    # Final recommendation
    print("=" * 70)
    print("TEST WITH YOUR ACTUAL DATA")
    print("=" * 70)
    print()
    print("To test with real LLM activations:")
    print()
    print("  # Load your activations (example)")
    print("  activations = torch.load('your_activations.pt')")
    print("  measure_block_improvement(activations, block_m=8, block_n=32)")
    print()
    print("Or modify the generate_sparse_activations() function to load")
    print("real activation tensors from your model.")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if CUDA is available (not required, but faster)
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        torch.set_default_device('cuda')
    else:
        print("Using CPU (install CUDA for faster execution)")
    print()
    
    main()

