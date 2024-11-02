#!/bin/bash

NUM_RUNS=30
EXECUTABLE="./ck"
TEMP_DIR="temp_data"

mkdir -p $TEMP_DIR
AGGREGATED_FILE="aggregated_data.csv"
rm -f $AGGREGATED_FILE

declare -A experimental_times
declare -A regular_times
declare -A counts

PATTERNS=("dense" "01010101" "10101010" "00010000" "00100000")

for pattern in "${PATTERNS[@]}"; do
experimental_times[$pattern]=0
regular_times[$pattern]=0
counts[$pattern]=0
done

for ((i=1; i<=NUM_RUNS; i++)); do
echo "Run #$i"
$EXECUTABLE > /dev/null
cp performance_data.csv "$TEMP_DIR/performance_data_$i.csv"

while IFS=',' read -r pattern experimental_time regular_time; do
if [ "$pattern" != "Pattern" ]; then
experimental_times[$pattern]=$(echo "${experimental_times[$pattern]} + $experimental_time" | bc)
regular_times[$pattern]=$(echo "${regular_times[$pattern]} + $regular_time" | bc)
counts[$pattern]=$((counts[$pattern] + 1))
  fi
  done < performance_data.csv
  done

  echo "Pattern,AverageExperimentalTime(ms),AverageRegularTime(ms)" > $AGGREGATED_FILE

  for pattern in "${PATTERNS[@]}"; do
  avg_experimental=$(echo "scale=6; ${experimental_times[$pattern]} / ${counts[$pattern]}" | bc)
  avg_regular=$(echo "scale=6; ${regular_times[$pattern]} / ${counts[$pattern]}" | bc)
  echo "$pattern,$avg_experimental,$avg_regular" >> $AGGREGATED_FILE
  done

  echo "Average Execution Times over $NUM_RUNS runs:"
  column -s, -t $AGGREGATED_FILE

  rm -r $TEMP_DIR
