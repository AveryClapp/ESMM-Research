#!/bin/bash
RUNS=30

> speedups.txt
> avg_lengths.txt
> max_lengths.txt
> full_times.txt
> early_times.txt

for ((i=1; i<=RUNS; i++)); do
echo "Run $i of $RUNS"
./et2 | while read line; do
if [[ $line =~ Speedup:[[:space:]]([0-9.]+)x ]]; then
speedup="${BASH_REMATCH[1]}"
echo "$speedup" >> "speedups.txt"
fi
if [[ $line =~ Average[[:space:]]column[[:space:]]length:[[:space:]]([0-9.]+)[[:space:]]\(([0-9.]+)% ]]; then
length="${BASH_REMATCH[1]}"
percent="${BASH_REMATCH[2]}"
echo "$percent" >> "avg_lengths.txt"
fi
if [[ $line =~ Max[[:space:]]column[[:space:]]length:[[:space:]]([0-9.]+)[[:space:]]\(([0-9.]+)% ]]; then
max_length="${BASH_REMATCH[1]}"
max_percent="${BASH_REMATCH[2]}"
echo "$max_percent" >> "max_lengths.txt"
fi
if [[ $line =~ Full[[:space:]]computation[[:space:]]time:[[:space:]]([0-9.]+)[[:space:]]ms ]]; then
full_time="${BASH_REMATCH[1]}"
echo "$full_time" >> "full_times.txt"
fi
if [[ $line =~ Early[[:space:]]termination[[:space:]]time:[[:space:]]([0-9.]+)[[:space:]]ms ]]; then
early_time="${BASH_REMATCH[1]}"
echo "$early_time" >> "early_times.txt"
fi
done
done

echo -e "\nResults over $RUNS runs:"
echo "------------------------"
avg_speedup=$(awk '{sum+=$1} END {print sum/NR}' "speedups.txt")
avg_length=$(awk '{sum+=$1} END {print sum/NR}' "avg_lengths.txt")
max_length=$(awk '{sum+=$1} END {print sum/NR}' "max_lengths.txt")
avg_full_time=$(awk '{sum+=$1} END {print sum/NR}' "full_times.txt")
avg_early_time=$(awk '{sum+=$1} END {print sum/NR}' "early_times.txt")

echo "Average speedup: $avg_speedup x"
echo "Average column length: $avg_length%"
echo "Average max column length: $max_length%"
echo "Average full computation time: $avg_full_time ms"
echo "Average early termination time: $avg_early_time ms"

rm speedups.txt avg_lengths.txt max_lengths.txt full_times.txt early_times.txt
