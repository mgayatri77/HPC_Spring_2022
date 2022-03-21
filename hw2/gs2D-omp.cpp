#include <stdio.h>
#include "utils.h"
#include "math.h"

// max number of iterations for Gauss-Siedel loop
#define MAX_ITERATIONS 5000

// Function to compute the norm of the residual 
double get_residual_norm (int N, double ** u, double ** f) {
    int i, j; 
    double res_norm = 0, rhs = 0;  
    double h = 1.0 / (N+1);

    #pragma omp parallel for collapse(2) reduction (+:res_norm) private(i,j)
    for (i = 1; i < N+1; i++) {
        for (j = 1; j < N+1; j++) {
            rhs = (4*u[i][j] - u[i-1][j] - u[i+1][j] - u[i][j-1] - u[i][j+1]); 
            res_norm += pow((f[i][j] - rhs/(pow(h, 2))), 2); 
        }
    }
    return pow(res_norm, 0.5); 
}

// implement Gauss-Siedel Method for 2D system discretized by N
void gauss_seidel_2D(int N) {
    // declare variables for h, u and u_prev
    double h = 1.0 / (N+1);
    double** u = (double**) calloc (N+2, sizeof(double*)); 
    double** f = (double**) calloc (N+2, sizeof(double*));

    // initialize arrays
    for (int i = 0; i < N+2; i++) {
        u[i] = (double*) calloc (N+2, sizeof(double));
        f[i] = (double*) calloc (N+2, sizeof(double));
        for (int j = 0; j < N+2; j++)
            f[i][j] = 1.0;
    }
    
    // initialize constants for running loop 
    int i, j; 
    int it = 0; 
    double tol = 1e6; 
    double res_init = get_residual_norm(N, u, f);
    double res_curr = res_init; 
    
    printf("Iteration %d, Residual Norm = %f\n", it, res_curr); 

    // Initialize timer
    Timer t;
    t.tic();
    
    // run jacobi loop while iterations < MAX and residual is large
    while (it < MAX_ITERATIONS && (res_init / res_curr) < tol) {
        // perform gauss-siedel iteration for red points
        #pragma omp parallel for collapse(2) private(i,j)
        for (i = 1; i < N+1; i++) {
            for (j = 1; j < N+1; j++) {
                if ((i + j) % 2 == 0) {
                    u[i][j] = 0.25*(f[i][j]*pow(h, 2) + u[i-1][j] + u[i][j-1]); 
                    u[i][j] += 0.25*(u[i+1][j] + u[i][j+1]);
                }
            }
        }

        // perform gauss-siedel iteration for black points
        #pragma omp parallel for collapse(2) private(i,j)
        for (i = 1; i < N+1; i++) {
            for (j = 1; j < N+1; j++) {
                if ((i + j) % 2 == 1) {
                    u[i][j] = 0.25*(f[i][j]*pow(h, 2) + u[i-1][j] + u[i][j-1]); 
                    u[i][j] += 0.25*(u[i+1][j] + u[i][j+1]);
                }
            }
        }

        // compute and output residual
        // res_curr = get_residual_norm(N, u, f);
        // printf("Iteration %d, Residual Norm = %f\n", it+1, res_curr);
        it++; 
    }
    
    // print time taken to compete jacobi method
    double time = t.toc();
    res_curr = get_residual_norm(N, u, f);
    printf("Final Residual Norm = %f\n", res_curr);
    printf("Time taken to run %i iterations = %f seconds\n", it, time);
    
    // printf("Final u values\n");
    // for (int i = 0; i < N+2; i++) {
    //     for (int j = 0; j < N+2; j++) {
    //         printf("%f ", u[i][j]);
    //     }
    //     printf("\n");
    // }

    for (int i = 0; i < N+2; i++) {
        free(u[i]);
        free(f[i]); 
    }
    free(u);
    free(f); 
}

int main(int argc, char** argv) {
    // call gauss-siedel method with specified N 
    gauss_seidel_2D(1000); 
}