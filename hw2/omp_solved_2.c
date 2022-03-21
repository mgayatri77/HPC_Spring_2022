// Modified by Mohit Gayatri (mag9528)
/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Comments
The following changes needed to be made to fix bugs: 
1) Thread numbers being printed on line 38 were incorrect
   due to the "tid" variable being shared. This was fixed by 
   making the "tid" variable private. 
2) The "total" being printed by threads on line 47 was the same
   for each thread and could also be 0 for some iterations. This 
   was again fixed by making the "total" variable private. 
*/

int main (int argc, char *argv[]) 
{
int nthreads, i, tid;
float total;  

/*** Spawn parallel region ***/
#pragma omp parallel private(tid,total)
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  total = 0.0;
  #pragma omp for schedule(dynamic, 10)
  for (i=0; i<1000000; i++) 
     total = total + i*1.0;
  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}