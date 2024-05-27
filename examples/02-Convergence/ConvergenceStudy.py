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

    # declare list to collect h refinement and L^2 error result
    hList = []
    normL2list = []

    # We now loop and refine the solution computing at each iteration the L^2 error
    for i in range(0,5):
        # start with a finer refinement
        grid.refine()

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

        # compute the L^2 norm
        normL2 = utils.l2Norm(grid, uh)
        hList.append(grid.h())
        normL2list.append(normL2)

    # plot the convergence
    plt.figure()
    plt.loglog(hList, normL2list, '-o')
    plt.loglog(hList, np.array(hList)**2, 'k-.')
    plt.xlabel('h')
    plt.ylabel('L2 norm')
    plt.title('Convergence test')
    plt.legend(['L2 norm', 'h^2'])
    plt.show()

if __name__ == "__main__":
    main()

