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

// Parallelized version of scan using OpenMP
void scan_omp(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  
  long chunk_size = n/NUM_THREADS;
  #pragma omp parallel num_threads(NUM_THREADS)
  { 
    long my_start = 1 + (omp_get_thread_num()*chunk_size);
    long my_end = (omp_get_thread_num() == NUM_THREADS-1) ? n : my_start + chunk_size;  
    prefix_sum[my_start] = A[my_start-1];
    // fprintf(stderr, "Start: %li, End: %li\n", my_start, my_start+chunk_size); 
    for (long i = my_start+1; i < my_end; i++) {
        prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }
  }

  long partial_sum = 0; 
  for (int i = 1; i < NUM_THREADS; i++) {
      partial_sum = prefix_sum[i*chunk_size];
      for (int j = i*chunk_size + 1; j <= (i+1)*chunk_size; j++)
        prefix_sum[j] += partial_sum; 
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