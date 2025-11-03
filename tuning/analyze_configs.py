#!/usr/bin/env python3
"""
Analyze and enumerate valid ESMM kernel configurations for BK=8, TM=1, TN=8.
This helps verify the parameter space before running the full autotuner.
"""

import itertools


def check_config(config):
    """Check if a configuration satisfies all ESMM kernel constraints."""
    BM = config["BM"]
    BN = config["BN"]
    BK = config["BK"]
    WM = config["WM"]
    WN = config["WN"]
    TM = config["TM"]
    TN = config["TN"]
    WNITER = config["WNITER"]
    NUM_THREADS = config["NUM_THREADS"]

    constraints = [
        ("BN % WN == 0", BN % WN == 0),
        ("BM % WM == 0", BM % WM == 0),
        (
            "(BN // WN) * (BM // WM) == NUM_THREADS // 32",
            (BN // WN) * (BM // WM) == NUM_THREADS // 32,
        ),
        (
            "(WM * WN) % (32 * TM * TN * WNITER) == 0",
            (WM * WN) % (32 * TM * TN * WNITER) == 0,
        ),
        ("WN % WNITER == 0", WN % WNITER == 0),
        ("NUM_THREADS >= BK // 4", NUM_THREADS >= BK // 4),
        ("NUM_THREADS >= BN // 4", NUM_THREADS >= BN // 4),
        ("(NUM_THREADS * 4) % BK == 0", (NUM_THREADS * 4) % BK == 0),
        ("(NUM_THREADS * 4) % BN == 0", (NUM_THREADS * 4) % BN == 0),
        ("(BN * BK + BM * BK) * 4 <= 48000", (BN * BK + BM * BK) * 4 <= 48000),
        ("BM >= 64", BM >= 64),
        ("BN >= 64", BN >= 64),
        ("WM <= BM", WM <= BM),
        ("WN <= BN", WN <= BN),
    ]

    # Check WMITER constraints
    if (WM * WN) % (32 * TM * TN * WNITER) == 0:
        WMITER = (WM * WN) // (32 * TM * TN * WNITER)
        constraints.extend(
            [
                ("WM % WMITER == 0", WM % WMITER == 0 if WMITER > 0 else False),
                (
                    "WMITER * TM * WNITER * TN <= 64",
                    WMITER * TM * WNITER * TN <= 64,
                ),
            ]
        )
    else:
        WMITER = None

    # Check loading constraints
    constraints.extend(
        [
            ("(BM * BK) % (4 * NUM_THREADS) == 0", (BM * BK) % (4 * NUM_THREADS) == 0),
            ("(BN * BK) % (4 * NUM_THREADS) == 0", (BN * BK) % (4 * NUM_THREADS) == 0),
        ]
    )

    # Check all constraints
    failed = [name for name, check in constraints if not check]

    return len(failed) == 0, failed, WMITER


def main():
    # Parameter space (matching tuner.py)
    params = {
        "NUM_THREADS": [128, 256],
        "BM": [64, 128, 256],
        "BN": [64, 128, 256],
        "BK": [8],
        "TM": [1],
        "TN": [8],
        "WM": [32, 64, 128],
        "WN": [32, 64, 128],
        "WNITER": [1, 2, 4, 8],
    }

    print("=" * 80)
    print("ESMM Kernel Configuration Analysis (BK=8, TM=1, TN=8)")
    print("=" * 80)
    print()

    # Generate all combinations
    keys = list(params.keys())
    values = [params[k] for k in keys]
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total possible combinations: {len(all_configs)}")

    # Filter valid configurations
    valid_configs = []
    invalid_reasons = {}

    for config in all_configs:
        is_valid, failed, wmiter = check_config(config)
        if is_valid:
            config["WMITER"] = wmiter
            valid_configs.append(config)
        else:
            for reason in failed:
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    print(f"Valid configurations: {len(valid_configs)}")
    print(f"Invalid configurations: {len(all_configs) - len(valid_configs)}")
    print()

    if invalid_reasons:
        print("Top reasons for invalid configurations:")
        for reason, count in sorted(
            invalid_reasons.items(), key=lambda x: -x[1]
        )[:5]:
            print(f"  - {reason}: {count} configs")
        print()

    # Group valid configs by key properties
    print("Valid configurations grouped by warp structure:")
    print()

    warp_groups = {}
    for cfg in valid_configs:
        key = (cfg["WM"], cfg["WN"], cfg["WNITER"], cfg["WMITER"])
        if key not in warp_groups:
            warp_groups[key] = []
        warp_groups[key].append(cfg)

    for (wm, wn, wniter, wmiter), configs in sorted(warp_groups.items()):
        print(
            f"  WM={wm}, WN={wn}, WNITER={wniter}, WMITER={wmiter} "
            f"→ {len(configs)} configs"
        )
        print(f"    Warp tile: {wm}×{wn}, Threads per warp: {32}")
        print(f"    Elements per thread: {wmiter}×{wniter*8} (M×N)")
        print(
            f"    Registers per thread: {wmiter * wniter * 8} floats "
            f"{'✓' if wmiter * wniter * 8 <= 64 else '✗'}"
        )
        # Show a few example block configs
        block_examples = set(
            (c["BM"], c["BN"], c["NUM_THREADS"]) for c in configs[:3]
        )
        for bm, bn, nt in sorted(block_examples):
            warps = (bm // wm) * (bn // wn)
            print(f"      - BM={bm}, BN={bn}, NUM_THREADS={nt} ({warps} warps)")
        if len(configs) > 3:
            print(f"      ... and {len(configs) - 3} more block size combinations")
        print()

    # Highlight recommended configs
    print("=" * 80)
    print("Recommended starting configurations for manual testing:")
    print("=" * 80)
    print()

    # Find configs with good balance
    recommended = [
        cfg
        for cfg in valid_configs
        if cfg["BM"] == 128
        and cfg["BN"] == 128
        and cfg["WMITER"] * cfg["WNITER"] <= 4  # Low register pressure
    ]

    if recommended:
        for i, cfg in enumerate(recommended[:5], 1):
            print(f"{i}. BM={cfg['BM']}, BN={cfg['BN']}, WM={cfg['WM']}, "
                  f"WN={cfg['WN']}, WNITER={cfg['WNITER']}, "
                  f"NUM_THREADS={cfg['NUM_THREADS']}")
            print(f"   → WMITER={cfg['WMITER']}, "
                  f"Registers/thread={cfg['WMITER']*cfg['WNITER']*8}, "
                  f"Warps={(cfg['BM']//cfg['WM'])*(cfg['BN']//cfg['WN'])}")
            print()

    print(f"Total valid configurations to tune: {len(valid_configs)}")
    print()


if __name__ == "__main__":
    main()
