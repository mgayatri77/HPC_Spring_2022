#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

// Function to compute inner product between vectors using OMP
void vec_inner_product_omp(double * z, const double * x, const double * y, long N) {
    #pragma omp parallel for schedule(static)
    for(long i = 0; i < N; i++)
        z[i] = x[i] * y[i]; 
}

// Function to multiply a matrix with a vector using OMP
void mat_vec_product_omp(double * z, double * M, const double * y, long N) {
    for (long row = 0; row < N; row++) {
        double sum = 0; 
        #pragma omp parallel for schedule(static) reduction(+:sum)
        // reduction over columns
        for (long col = 0; col < N; col++) {
            sum += M[(row * N) + col] * y[col];
        }
        z[row] = sum; 
    }
}

// Kernel to compute inner product between two vectors with CUDA 
__global__ void vec_inner_product(double * z, const double * x, const double * y, long N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // compute if smaller than N
    if(idx < N) 
        z[idx] = x[idx] * y[idx]; 
}

// Kernel to multiply matrix with a vector with CUDA 
__global__ void mat_vec_product(double * z, double * M, const double * y, long N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // compute if smaller than N
    if (row < N) {
        double sum = 0;
        // sum over columns
        for (long col = 0; col < N; col++) 
            sum += (M[(row * N) + col] * y[col]); 
        z[row] = sum; 
    }
}

// Check latest CUDA error and print
void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
    long N = (1UL<<12); // 2^12 

    // allocate vectors and matrices
    double* M = (double*) malloc(N * N * sizeof(double)); 
    double* x = (double*) malloc(N * sizeof(double));
    double* y = (double*) malloc(N * sizeof(double));
    double* z = (double*) malloc(N * sizeof(double));
    double* z_ref = (double*) malloc(N * sizeof(double));

    // initialize vectors and matrix using OMP
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
        x[i] = i+2;
        y[i] = 1.0/(i+1);
        z[i] = 0;
        z_ref[i] = 0;
        for (long j = 0; j < N; j++)
            M[i*N + j] = i+j+2; 
    }

    /*// code to run compute inner product between 2 vectors on OMP and print time
    double tt = omp_get_wtime();
    vec_inner_product_omp(z_ref, x, y, N);
    printf("CPU %f s\n", omp_get_wtime()-tt);
    printf("CPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    // code to allocate arrays for GPU
    double *x_d, *y_d, *z_d; 
    cudaMalloc(&x_d, N*sizeof(double));
    Check_CUDA_Error("malloc x failed");
    cudaMalloc(&y_d, N*sizeof(double));
    cudaMalloc(&z_d, N*sizeof(double));
    
    // code to copy input matrices to GPU
    tt = omp_get_wtime();
    cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
    // code to compute inner product on GPU
    double ttinner = omp_get_wtime();
    vec_inner_product<<<N/1024,1024>>>(z_d, x_d, y_d, N);
    cudaDeviceSynchronize();
    ttinner = omp_get_wtime() - ttinner;
    // copy result back and print time
    cudaMemcpy(z, z_d, N*sizeof(double), cudaMemcpyDeviceToHost);
    printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);
    printf("GPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    // compute error between CPU and GPU matrices
    double err = 0;
    for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
    printf("Error = %f\n", err);*/

    // code to run matrix vector multiplication on OMP and print time
    double tt = omp_get_wtime();
    mat_vec_product_omp(z_ref, M, y, N);
    printf("CPU %f s\n", omp_get_wtime()-tt);
    printf("CPU Bandwidth = %f GB/s\n", (2*N*N + N)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    // code to allocate arrays for GPU
    double *M_d, *y_d, *z_d; 
    cudaMalloc(&M_d, N*N*sizeof(double));
    Check_CUDA_Error("malloc m failed");
    cudaMalloc(&y_d, N*sizeof(double));
    cudaMalloc(&z_d, N*sizeof(double));
    
    // code to copy input matrices to GPU
    tt = omp_get_wtime();
    cudaMemcpy(M_d, M, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
    // code to run matrix vector multiplication on GPU
    double ttinner = omp_get_wtime();
    mat_vec_product<<<N/1024, 1024>>>(z_d, M_d, y_d, N);
    cudaDeviceSynchronize();
    ttinner = omp_get_wtime() - ttinner;
    // copy result back and print time
    cudaMemcpy(z, z_d, N*sizeof(double), cudaMemcpyDeviceToHost);
    printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);
    printf("GPU Bandwidth = %f GB/s\n", (2*N*N + N)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    // compute error between CPU and GPU matrices
    double err = 0;
    for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
    printf("Error = %f\n", err);
    
    // free GPU memory
    // cudaFree(x_d); 
    cudaFree(M_d);
    cudaFree(y_d);
    cudaFree(z_d);

    // free CPU memory
    // free(x); 
    free(M); 
    free(y);
    free(z);
    free(z_ref);

    return 0;
}
