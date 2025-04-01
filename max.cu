#include <stdio.h>
#include <cuda_runtime.h>

int main() {
		    int device = 0;  // Assuming you want to query the first GPU
			    cudaSetDevice(device);
				    
				    int sharedMemPerMultiprocessor;
					    cudaDeviceGetAttribute(&sharedMemPerMultiprocessor, 
										                           cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
																                              device);
						    
						    printf("Maximum Shared Memory Per Multiprocessor: %d bytes (0x%X)\n", 
											           sharedMemPerMultiprocessor, sharedMemPerMultiprocessor);
							    
							int sharedMemPerBlock;
							cudaDeviceGetAttribute(&sharedMemPerBlock, 
											                      cudaDevAttrMaxSharedMemoryPerBlockOptin, 
																                        device);
							    
							printf("Maximum Shared Memory Per Block: %d bytes (0x%X)\n", 
											       sharedMemPerBlock, sharedMemPerBlock);
							    return 0;
}
