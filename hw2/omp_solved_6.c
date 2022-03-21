// Modified by Mohit Gayatri (mag9528)
/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

/* Comments
The code would not compile initially, due to the following error: 
"error: reduction variable ‘sum’ is private in outer context
   32 | #pragma omp for reduction(+:sum)"

This can be fixed by bring the for loop from the "dotprod" subroutine 
into "main" and changing line 46 to the following: 
"#pragma omp parallel for reduction(+:sum)"
*/

float a[VECLEN], b[VECLEN];

int main (int argc, char *argv[]) {
int i, tid;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

#pragma omp parallel for reduction(+:sum)
  for (i=0; i < VECLEN; i++) {
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
  }
printf("Sum = %f\n",sum);

}