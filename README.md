# MMMResearch
## TODO
- [ ] Get rid of Atomic Add
- [ ] Shared Memory/Memory Optimizations

## Current Findings:
30 run AVG for atomic add: .1536
30 run AVG for non atomic add: .1437
Atomic add leads to a 7% increase in runtime ((.1536 - .1437) / .1437)
Interstingly, dividing the A matrix into havles leads to a time of .1370.

32x32 Tiles of B:
By having B be a square matrix and allowing there to be one thread per column of B, we get a time of .1862. This isa longer time obviously but we are doing more work as now C and B have 4x the # of columns

32x32 SHMEM of B (WIP) but load B into __shared__ array.
