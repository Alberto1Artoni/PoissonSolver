from PoissonSolver import mesh, problem, utils
import numpy as np
import json
import matplotlib.pyplot as plt


def main():
    filePath = "data.json"
    with open(filePath, 'r') as file:
       data = json.load(file)

    # generate mesh
    grid = mesh.Mesh(data)

    # employ custom utility to read the forcing and the solution from the data.json file
    f = utils.parseFunction(data['source'])
    g = utils.parseFunction(data['exact'])

    # assemble the problem
    pb = problem.Problem(grid, data)

    # assemble matrix stiff
    pb.stiff(grid)

    # assemble source term
    pb.source(grid, f)

    # impose dirichlet bc with lifting
    pb.lifting(grid, g)

    # solve the algebric problem
    uh = pb.solveLifting()

    # post-process
    x = np.linspace(data['a'], data['b'], data['nx'])
    y = np.linspace(data['c'], data['d'], data['ny'])
    x, y = np.meshgrid(x, y)
    z = g(x,y)

    # Create a figure and a 3D axis for the solution plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, uh.reshape(data["ny"], data["nx"]), cmap='viridis')
    ax.set_title("Numerical solution")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u_h')
    fig.colorbar(surf)
    plt.show()

    # Create a figure and a 3D axis for the error plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, uh.reshape(data["ny"], data["nx"]) - z, cmap='viridis')
    ax.set_title("Error")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u - u_h')
    fig.colorbar(surf)
    plt.show()


if __name__ == "__main__":
    main()

