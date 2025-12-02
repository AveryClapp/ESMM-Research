#!/bin/bash
# Safe test of driver unload/reload

set -e

echo "=== Testing NVIDIA Driver Unload/Reload on EC2 ==="
echo ""

echo "1. Current GPU status:"
nvidia-smi --query-gpu=name,driver_version,pstate --format=csv
echo ""

echo "2. Checking for GPU processes:"
sudo fuser -v /dev/nvidia* 2>&1 || echo "No processes using GPU (good!)"
echo ""

echo "3. Checking loaded nvidia modules:"
lsmod | grep nvidia
echo ""

echo "4. Attempting to unload nvidia_uvm (safest module to test)..."
if sudo modprobe -r nvidia_uvm 2>/dev/null; then
    echo "   ✓ nvidia_uvm unloaded successfully"

    echo "5. Reloading nvidia_uvm..."
    sudo modprobe nvidia_uvm
    echo "   ✓ nvidia_uvm reloaded"

    echo "6. Verifying GPU still works:"
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo "   ✓ GPU accessible"

    echo ""
    echo "SUCCESS: Driver reload works on this EC2 instance!"
    echo "It should be safe to use --unload-driver"
else
    echo "   ✗ Could not unload nvidia_uvm"
    echo ""
    echo "This is normal on some EC2 instances."
    echo "Driver reload won't work, but --cold-start (cache clearing) will still work."
fi
