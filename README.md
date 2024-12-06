# MMMResearch
## TODO
- [ ] Get rid of Atomic Add
- [ ] Shared Memory/Memory Optimizations

## Current Findings:
30 run AVG for atomic add: .1536
30 run AVG for non atomic add: .1437
Atomic add leads to a 7% increase in runtime ((.1536 - .1437) / .1437)
Interstingly, dividing the A matrix into havles leads to a time of .1370.

