#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#define NUM_THREADS 8

// void print_array(long* arr, long N) {
//   fprintf(stderr, "[");
//   for(int i = 0; i < N; i++) {
//     fprintf(stderr, "%li ", arr[i]);
//   }
//   fprintf(stderr, "]\n");
// }

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;

  // compute size each thread will operate on
  long chunk_size = (n/NUM_THREADS); 
  
  #pragma omp parallel num_threads(NUM_THREADS)
  {  
    // get thread number and start and end indices for current thread
    long t_num = omp_get_thread_num();
    long start = (t_num * chunk_size); 
    long end = (t_num + 1) * chunk_size; 
    
    // make last thread run to end of array
    if (t_num == (NUM_THREADS-1))
      end = n; 
    
    // set prefix sum at start
    if (start > 0) 
      prefix_sum[start] = A[start-1];
    else 
      prefix_sum[start] = 0.; 
    
    // compute prefix sums
    for (long i = start+1; i < end; i++) {
        prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }
  }

  // loop over threads
  for (int i = 1; i < NUM_THREADS; i++) {
      // set start and end indices for chunks
      long start = i*chunk_size;
      long end = (i+1)*chunk_size;
      if (i == NUM_THREADS-1)
        end = n; 
      
      // add partial sum to section of thread i in parallel
      # pragma omp parallel for num_threads(NUM_THREADS)
      for (long j = start; j < end; j++) {
        prefix_sum[j] += prefix_sum[start-1];
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

//   fprintf(stderr, "A:\n");
//   print_array(A, N); 
//   fprintf(stderr, "Scan Seq:\n");
//   print_array(B0, N);
//   fprintf(stderr, "Scan Par:\n");
//   print_array(B1, N);

  free(A);
  free(B0);
  free(B1);
  return 0;
}