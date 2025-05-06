## Investigating A Sparsity
### 1024 x 1024
No A-Sparsity Check: 159.4 ns (profiler results)
0% Sparsity in A (w/ `50% in B)  
    - 226.91 us     
50% Sparsity in A (w/ 50% in B)      
    - 194.46 us    
75% Sparsity in A    
    - 178.50 us    
87.5% Sparsity in A    
    - 169.95 us    
   
### 2048 x 2048    
No A-Sparsity Check: .94 ms   
0% Sparsity in A (w/ 50% in B)    
    - 1.05 ms    
50% Sparsity in A     
    - .94 ms   
75% Sparsity in A    
    - .92 ms   
87.5% Sparsity in A    
    - .915    
### 4096 x 4096    
No A-Sparsity Check: 5.78 ms   
0% Sparsity in A    
    - 6.48 ms   
50% Sparsity in A    
    - 5.75 ms   
75% Sparsity in A   
    - 5.67 ms   
87.5% Sparsity in A   
    - 5.65 ms   

### 100% Dense B with no A-Sparsity Check
1024 x 1024: 233.06 us
2048 x 2048: 1510 us
4096 x 4096: 9620 us

### 100% Dense B and 100% Dense A
1024 x 1024: 327.36 us
2048 x 2048: 1860 us
4096 x 4096: 12010 us

### 100% Dense B and 0% Dense A
1024 x 1024: 222.18 us
2048 x 2048: 1320 us
4096 x 4096: 8350 us

### 100% Dense B and 50% Dense A (investigating divergence)
1024 x 1024: 274.08 us (divergent threads avg 791.65, 253328 sum)
2048 x 2048: 1540 us (6,280 avg, 2,0009,860 sum)
4096 x 4096: 9820 us (50,380.2 avg, 16121664 sum)
## Switch Table on Unrolled Sparsity Kernels
