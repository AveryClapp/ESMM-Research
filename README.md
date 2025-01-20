# MMMResearch
## TODO

## Current Findings:
30 run AVG for atomic add: .1536
30 run AVG for non atomic add: .1437
Atomic add leads to a 7% increase in runtime ((.1536 - .1437) / .1437)
Interstingly, dividing the A matrix into havles leads to a time of .1370.

32x32 Tiles of B:
By having B be a square matrix and allowing there to be one thread per column of B, we get a time of .1862. This is longer time obviously but we are doing more work as now C and B have 4x the # of columns

By loading elements of B into SHMEM, the average time is .229
The reason for such a significant slowdown is that we add all of this overhead of loading elements into a 32x32 matrix, and then synchronizing the warp when B elements aren't being reused for these computations. (One thread strides each row therefore elements are only accessed once in the lifespan of these kernels. 
