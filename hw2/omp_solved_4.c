// Modified by Mohit Gayatri (mag9528)
/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

/* Comments
The variable a is declared private, so each thread makes it own copy.
However, with OpenMP, the variable is stored on the stack by default, which is not
large enough to store a 1048*1048 double array. To fix the problem, a is declared
as a double ** and stored on the heap using malloc. In the parallel section it is
marked shared so that each thread does not make its own copy. 
*/

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;
double ** a = (double**)malloc(N * sizeof(double*)); 
for(i = 0; i < N; i++)
  a[i] = (double*)malloc(N * sizeof(double)); 

/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(a,nthreads) private(i,j,tid)
  {

  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);

  /* Each thread works on its own private copy of the array */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i][j] = tid + i + j;

  /* For confirmation */
  printf("Thread %d done. Last element= %f\n",tid,a[N-1][N-1]);

  }  /* All threads join master thread and disband */

}