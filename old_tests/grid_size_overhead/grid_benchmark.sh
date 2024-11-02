#!/bin/bash

RUNS=100

declare -A times
GRIDS=("1,1" "2,2" "4,4" "8,8" "16,16" "32,32" "64,64" "128,128")

for ((i=1; i<=RUNS; i++)); do
  echo "Run $i of $RUNS"
    ./grid_tests | while read line; do
      if [[ $line =~ Grid[[:space:]]size:[[:space:]]\((.*)\)[[:space:]]Launch[[:space:]]Time:[[:space:]]([0-9.]+) ]]; then
        grid="${BASH_REMATCH[1]}"
        time="${BASH_REMATCH[2]}"
        echo "$time" >> "times_${grid}.txt"
      fi
    done
done

echo -e "\nAverages over $RUNS runs:"
echo "------------------------"
for grid in "${GRIDS[@]}"; do
  avg=$(awk '{sum+=$1} END {print sum/NR}' "times_${grid}.txt")
  echo "Grid size ($grid): $avg ms"
  rm "times_${grid}.txt"
done
