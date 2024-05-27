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
  The considered domain is \(\Omega = [0,3]\times[0,2]\) and the following manufactured solution $u(x,y) = \sin(\pi x) \cos(\pi y)$ was taken. 

  ### 01-Refinement
  This example shows the refinement of the mesh.

  ### 02-Convergence
  This example shows the convergence of the error with respect to the mesh size.

## Documentation
To generate the documentation, run the following commands:
```
cd doc
sphinx-apidoc -o ./source ../PoissonSolver
```

## Notes to the reader
