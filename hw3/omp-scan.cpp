#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#define NUM_THREADS 32

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

// Parallelized version of scan using OpenMP
void scan_omp(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0; 

  // compute size of array that each thread will operate on
  long chunk_size = (n/NUM_THREADS); 
  
  #pragma omp parallel num_threads(NUM_THREADS)
  {  
    // get thread number and start and end indices for current thread
    int t_num = omp_get_thread_num();
    long start = (t_num * chunk_size); 
    long end = (t_num == (NUM_THREADS-1)) ? n : (t_num + 1) * chunk_size; 
    
    // set prefix sum at start
    if (start > 0) 
      prefix_sum[start] = A[start-1]; 
    
    // compute prefix sums and synchronize
    for (long i = start+1; i < end; i++)
        prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    #pragma omp barrier

    // compute partial sums and synchronize
    long partial_sum = 0; 
    for (int i = 1; i <= t_num; i++)
      partial_sum += prefix_sum[(i*chunk_size)-1]; 
    #pragma omp barrier

    // offset prefix by partial sums
    if(t_num > 0) {
      for (long i = start; i < end; i++)
        prefix_sum[i] += partial_sum; 
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}