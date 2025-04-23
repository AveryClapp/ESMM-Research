# April 21st - April 28th   
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
