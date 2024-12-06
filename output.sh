#!/bin/bash

# Initialize variables
total=0
count=30

echo "Running ./a.out $count times..."

## Run the program count times
for ((i=1; i<=count; i++))
  do
    # Run a.out and capture output
    output=$(./a.out)
    # Extract timing value using grep and awk
    time=$(echo "$output" | grep "GPU Timing:" | awk '{print $3}')
    # Add to total
    total=$(echo "$total + $time" | bc -l)
    done
# Calculate average
avg=$(echo "scale=6; $total / $count" | bc -l)
# Print results
echo "Average over $count runs: $avg ms"
