#include <stdlib.h>
#include <stdio.h>
#include <mkl.h>
#include <omp.h>
#include <sys/time.h>

/* 
This program demonstrates the OpenMP parallelization of a
computationally intensive loop where the work per iteration is allowed
to vary. Within the main loop, matrix is generated, the eigenvalue
solver DYSEV is called and the largest eigenvalue is saved.

DYSEV documentation can be found here:
http://www.netlib.org/lapack/explore-html/dd/d4c/dsyev_8f.html

Note that this is not necessarily the preferred way to calculate
eigenvalues and was used purely as a time consuming example for which
the work per iteration could be easily varied.

To compile using Intel C++ compiler and linking MKL routine

    icpc openmp_eigen_dynamic.c -mkl -openmp

To run

    export OMP_NUM_THREADS=N; ./a.out X Y Z

where

    N = number of OpenMP threads
    X = dimension of array
    Y = number of iterations (number of eigenvalue problems solved)
    Z = 'E' for even amount of work per iteration
        'U' for uneven amount of work per iteration
*/


int main(int argc, char **argv) {
  char choice;
  int n, niter;
  double elapsed, *eigmax;
  struct timeval tv_start, tv_end;


  // Make sure we use serial version of Intel MKL routine
  mkl_set_num_threads(1);

  // Process command line arguments
  if (argc < 4) {
    printf("\nThree command line arguments required\n");
    printf("  Dimension of array\n");
    printf("  Number of iterations\n");
    printf("  Choice: 'E' for even / 'U' for uneven work per iteration\n\n");
    return(0);
  }

  n      = atoi(argv[1]);
  niter  = atoi(argv[2]);
  choice = argv[3][0];

  if (choice != 'E' && choice != 'U') {
    printf("\nThird argument must be 'E' or 'U' for even or uneven\n");
    printf("work per iteration, respectively\n\n");
    return(0);
  }


  // Allocate vector to store results
  eigmax = (double *) malloc(niter * sizeof(double));
  
  // Solve eigenvalue problem for "niter" random matrices and print largest eigenvector

  // Get timestamp at start of loop
  gettimeofday(&tv_start, NULL); 


#pragma omp parallel for schedule(dynamic, 5)
  for (int j = 0; j < niter; j++) {

    int m, lda, info, lwork;
    double wkopt, *a, *w, *work;

    // Define the problem size. If choice set to uneven, allow problem
    // to grow for later iterations
    if (choice == 'E') {
      m = n;
    } else {
      m = n + j/5;
    }

    // Setup work space
    lda = m;
    lwork = -1;
    dsyev("Vectors", "Upper", &m, a, &lda, w, &wkopt, &lwork, &info);
    lwork = (int)wkopt;

    // Allocate arrays
    a      = (double *) malloc(m * m * sizeof(double));
    w      = (double *) malloc(m * sizeof(double));
    work   = (double*)malloc( lwork*sizeof(double) );

    // Initialize array for eigenvalue problem
    for (int i=0; i< m*m; i++) {
      a[i] = (double) ((i+j)%17) / (2.0 + j);
    }

    // Calculate eigenvalues and save the largest value
    dsyev("Vectors", "Upper", &m, a, &lda, w, work, &lwork, &info);
    eigmax[j] = w[m-1];

    // Free memory
    free(a);
    free(w);
    free(work);
  }

  // Get timestamp at end of loop
  gettimeofday(&tv_end, NULL);
  
  // Calculate elapsed time
  elapsed = (tv_end.tv_sec - tv_start.tv_sec) +
    (tv_end.tv_usec - tv_start.tv_usec) / 1000000.0;


  printf("array dimension = %d\n", n);
  printf("number of iterations = %d\n", niter);
  printf("wall time = %f\n", elapsed);

  // Following code is included to prevent compiler from optimizing
  // away the eigenvalue calculations. Provides the possibility that the
  // results will be used.
  if (choice == 'A') {
    for (int j = 0; j < niter; j++) {
      printf("%f\n", eigmax[j]);
    }
  }
  
  free(eigmax);
}
