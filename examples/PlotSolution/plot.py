import sys
sys.path.append('../../')

from src.mesh    import Mesh
from src.problem import Problem
from src.utils   import l2Norm, parseFunction

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

    # parse source term
    bcdata_str = data['exact']
    g = parseFunction(bcdata_str)

    # assemble problem
    problem = Problem(mesh, data)

    # solve linear system
    uh = problem.solve()

    # post-process
    x = np.linspace(data['a'], data['b'], data['nx'])
    y = np.linspace(data['c'], data['d'], data['ny'])
    x, y = np.meshgrid(x, y)
    z = g(x,y)

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, uh.reshape(data["nx"], data["ny"]) - z, cmap='viridis')
    ax.set_title("Error")
    fig.colorbar(surf)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, uh.reshape(data["nx"], data["ny"]), cmap='viridis')
    ax.set_title("Numerical solution")
    fig.colorbar(surf)
    plt.show()


if __name__ == "__main__":
    main()
