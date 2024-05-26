import sys
sys.path.append('../../')

from src.mesh    import Mesh
from src.problem import Problem
from src.utils   import l2Norm

import json
from sympy import symbols, sin, cos, pi
from sympy.utilities.lambdify import lambdify

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def main():
    # read input
    filePath = "data.json"
    with open(filePath, 'r') as file:
        data = json.load(file)

    # generate mesh
    mesh = Mesh(data)

    for i in range(0,5):
        problem = Problem(mesh, data)
        uh = problem.solveLifting()
        normL2 = l2Norm(mesh, uh);
        print("L2 norm: ", normL2)
        mesh.refine()


if __name__ == "__main__":
    main()
