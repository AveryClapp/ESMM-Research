#!/bin/bash
# Test script to compare cold vs warm starts

KERNEL=17
SIZE=1024
PATTERN="11111111"

echo "=== Test 1: Manual cold starts (3 separate invocations) ==="
for i in {1..3}; do
    echo "Run $i:"
    rm -rf ~/.nv/ComputeCache
    sudo /usr/local/cuda-12.1/bin/ncu --set full --target-processes all \
        ./exec_dev $KERNEL --size $SIZE --pattern $PATTERN --no-check 2>&1 | \
        grep -E "Duration|kernel_time"
    sleep 2
done

echo ""
echo "=== Test 2: Your benchmark script WITHOUT --cold-start ==="
python3 scripts/benchmark.py -k $KERNEL --sizes $SIZE --sparsity $PATTERN --output test_warm

echo ""
echo "=== Test 3: Your benchmark script WITH --cold-start ==="
python3 scripts/benchmark.py -k $KERNEL --sizes $SIZE --sparsity $PATTERN --output test_cold --cold-start

echo ""
echo "=== Results Comparison ==="
echo "Manual (should be slowest - true cold start):"
echo "Warm (should be fastest - cached):"
cat benchmarks/test_warm/summary.csv | grep -v "^#" | tail -n 1
echo "Cold-start mode (should match manual):"
cat benchmarks/test_cold/summary.csv | grep -v "^#" | tail -n 1
