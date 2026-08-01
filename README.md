# undergraduate_navier_stokes_least_squares

The two codes of the repository were developed working with undergraduate students at Francis Marion University.

## Serial Code
* The serial code utilizes the deal.II software library - https://dealii.org/
* The code solves the matrix system utilizing the UMFPACK direct solver
  * for further information regarding UMFPACK with deal.II see the ReadMe in Downloads at https://dealii.org/ 
* The code has been tested, compiled, and run with deal.II version 9.3.3

## Parallel Code
* The parallel code utilizes deal.II, p4est, and Trilinos software libraries
  * For further information about Trilinos and p4est with deal.II see the ReadMe in Downloads at https://dealii.org/
  * The website below may be of interest for compiling deal.II on supercomputers
    * https://github.com/geodynamics/aspect/wiki/Compiling-and-Running-ASPECT-on-TACC-Stampede2  
* The code employs deal.II's Trilinos wrappers to utilize Trilinos parallel data structures with deal.II
* The code solves the parallel matrix system using the Trilinos parallel direct solver
* The code has been tested, complied, and run with deal.II version 9.7.1, Trilinos 16.2.0, and p4est 2.8.7 
