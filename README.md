# Poisson FEM Solver

This repository contains a Finite Element solver for the Poisson equation on structured triangular meshes.

## Table of Contents

- [Installation](#installation)
- [Examples](#examples)
- [Documentation](#documentation)
- [Notes to the reader](#notes-to-the-reader)

## How to run the code

Clone the repository with the following:

```
git clone git@github.com:Alberto1Artoni/PoissonSolver.git
```

install the repo with:
```
pip install -e .
```

## Examples
  Three test cases are proposed to validate the implementation of the library.
  In each folder, find both a jupyter notebook and a python script that runs the solver.

  ### 00-Solver
  The solver is validated on a simple problem with known solution.
  The considered domain is $\(\Omega = [0,3]\times[0,2]\)$ and the following manufactured solution $u(x,y) = \sin(\pi x) \cos(\pi y)$ was taken. 
  For this case, we present both the case where we impose strong Dirichlet boundary conditions and the case where we impose the Dirichlet boundary conditions with the lift operator.
  To run the case with the lift operator, run the following command:
  ```
  cd ./examples/00-Solver
  python3 Solver.py
  ```

  To run the case with the strong Dirichlet, run the following command:
  ```
  cd ./examples/00-Solver
  python3 SolverDirichlet.py
  ```

  ### 01-Refinement
  This example shows the refinement of the mesh. The implementation has been tested only on structured triangular grids.
  To run the example, run the following command:
  ```
  cd ./examples/01-Refinement
  python3 Refinement.py
  ```

  ### 02-Convergence
  This example shows the convergence of the error with respect to the mesh size, numerically veryfing the theoretical convergence rate of the FEM method.

  To run the example, run the following command:
  ```
  cd ./examples/02-Convergence
  python3 ConvergenceStudy.py
  ```

## Documentation
To generate the documentation, run the following commands:
```
cd doc
sphinx-apidoc -o ./source ../PoissonSolver
```

