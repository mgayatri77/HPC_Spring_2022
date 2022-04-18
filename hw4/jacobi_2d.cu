#include <omp.h>
#include <algorithm>
#include <stdio.h>
#include "math.h"

// Set max number of iterations
#define MAX_ITERATIONS 5000

// Kernel to perform Jacobi iteration on given arrays
__global__ void jacobi_iter (double * u, double * u_prev, double * f, int N, double h) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; 
    int j = blockDim.y * blockIdx.y + threadIdx.y; 
    
    // compute if non-boundary points 
    if (i > 0 && j > 0 && i < N+1 && j < N+1) {
        double u_lo = u_prev[(i-1)*(N+2) + j] + u_prev[i*(N+2) + j-1];
        double u_hi = u_prev[(i+1)*(N+2) + j] + u_prev[i*(N+2) + j+1];
        // jacobi update
        u[i*(N+2) + j] = 0.25*(f[i*(N+2) + j]*h*h + u_lo + u_hi);        
    }
}

// Kernel to copy current u array into u_prev 
__global__ void jacobi_copy (double * u, double * u_prev, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; 
    int j = blockDim.y * blockIdx.y + threadIdx.y; 

    // copy non-boundary points
    if (i > 0 && j > 0 && i < N+1 && j < N+1) {
        u_prev[i*(N+2) + j] = u[i*(N+2) + j]; 
    }
}

// Function to run Jacobi method using GPU kernels 
void jacobi_2d_gpu (double * u, double * u_prev, double * f, int N, double h) {
    // declare blocks and threads 
    dim3 blocksPerGrid(N/32+1, N/32+1, 1); 
    dim3 threadsPerBlock(32, 32, 1); 

    // run jacobi until max iterations
    for (int it = 0; it < MAX_ITERATIONS; it++) {
        // call kernels
        jacobi_iter<<<blocksPerGrid, threadsPerBlock>>>(u, u_prev, f, N, h);
        jacobi_copy<<<blocksPerGrid, threadsPerBlock>>>(u, u_prev, N);
    }
}

// Function to run Jacobi method using OpenMP
void jacobi_2d_omp (double * u, double * u_prev, double * f, int N, double h) {
    int i, j; 
    // run Jacobi until max iterations
    for (int it = 0; it < MAX_ITERATIONS; it++) {
        // parallelize jacobi iterations using OpenMP
        #pragma omp parallel for collapse(2) private(i,j)
        for (i = 1; i < N+1; i++) {
            for (j = 1; j < N+1; j++) {
                // perform jacobi iteration
                double u_lo = u_prev[(i-1)*(N+2) + j] + u_prev[i*(N+2) + j-1];
                double u_hi = u_prev[(i+1)*(N+2) + j] + u_prev[i*(N+2) + j+1];
                // jacobi update
                u[i*(N+2) + j] = 0.25*(f[i*(N+2) + j]*h*h + u_lo + u_hi);
            }
        }

        //  copy values from u to u_prev using OpenMP
        #pragma omp parallel for collapse(2) private(i,j)
        for (i = 1; i < N+1; i++) {
            for (j = 1; j < N+1; j++) {
                u_prev[i*(N+2) + j] = u[i*(N+2) + j];  
            } 
        }
    }
}

// Function to compute the norm of the residual 
double get_residual_norm (int N, double * u, double * f) {
    int i, j; 
    double res_norm = 0, rhs = 0;  
    double h = 1.0 / (N+1);

    // parallelize residual norm computation using OpenMP
    #pragma omp parallel for collapse(2) reduction (+:res_norm) private(i,j)
    for (i = 1; i < N+1; i++) {
        for (j = 1; j < N+1; j++) {
            double u_lo = u[(i-1)*(N+2) + j] + u[i*(N+2) + j-1];
            double u_hi = u[(i+1)*(N+2) + j] + u[i*(N+2) + j+1];
            // compute rhs and residual norm
            rhs = (4*u[i*(N+2) + j] - u_lo - u_hi); 
            res_norm += pow((f[i*(N+2) + j] - rhs/(pow(h, 2))), 2); 
        }
    }
    return pow(res_norm, 0.5); 
}

// Check latest CUDA error and print
void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main (int argc, char ** argv) {
    // declare N
    int N = 1000;

    // allocate variables/arrays for h, u,  u_prev, u and f
    double h = 1.0 / (N+1);
    double * u = (double*) calloc ((N+2)*(N+2), sizeof(double)); 
    double * u_ref = (double*) calloc ((N+2)*(N+2), sizeof(double));
    double * u_prev = (double*) calloc ((N+2)*(N+2), sizeof(double));
    double * f = (double*) calloc ((N+2)*(N+2), sizeof(double));

    // initialize f values to 1's using OpenMP
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N+2; i++) {
        for (int j = 0; j < N+2; j++)
            f[i*(N+2) + j] = 1.0;
    }
    
    // code to run 2D Jacobi method on OMP and print time
    double tt = omp_get_wtime();
    jacobi_2d_omp(u_ref, u_prev, f, N, h);
    printf("CPU %f s\n", omp_get_wtime()-tt);

    // reset u_prev values to 0's using OpenMP
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (N+2)*(N+2); i++) {
        u_prev[i] = 0.0;
    }

    // code to allocate arrays for GPU
    double *u_prev_d, *u_d, *f_d; 
    cudaMalloc(&u_prev_d, (N+2)*(N+2)*sizeof(double));
    Check_CUDA_Error("malloc m failed");
    cudaMalloc(&u_d, (N+2)*(N+2)*sizeof(double));
    cudaMalloc(&f_d, (N+2)*(N+2)*sizeof(double));
    
    // code to copy input matrices to GPU
    tt = omp_get_wtime();
    cudaMemcpy(u_prev_d, u_prev, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u_d, u, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(f_d, f, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
    // code to run Jacobi 2D method on GPU
    double ttinner = omp_get_wtime();
    jacobi_2d_gpu(u_d, u_prev_d, f_d, N, h); 
    ttinner = omp_get_wtime() - ttinner;
    // copy result from GPU back and print time
    cudaMemcpy(u, u_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost);
    printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);

    // compute error between u and u_ref
    double err = 0;
    for (long i = 0; i < (N+2)*(N+2); i++) err += fabs(u[i]-u_ref[i]);
    printf("Error = %f\n", err);

    // compute residual norms for CPU, GPU and print
    double res_norm_cpu = get_residual_norm(N, u_ref, f);
    double res_norm_gpu = get_residual_norm(N, u, f);
    printf("CPU Residual Norm: %f\n", res_norm_cpu); 
    printf("GPU Residual Norm: %f\n", res_norm_gpu);

    // free GPU memory
    cudaFree(u_d);
    cudaFree(u_prev_d);
    cudaFree(f_d);

    // free CPU memory
    free(u); 
    free(u_ref);
    free(u_prev);
    free(f);

    return 0;
}
