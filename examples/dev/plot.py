from PoissonSolver import mesh, problem, utils

import numpy as np
import json


def main():
    # read input
    filePath = "data.json"
    with open(filePath, 'r') as file:
        data = json.load(file)

    # generate mesh
    grid = mesh.Mesh(data)

    f = utils.parseFunction(data['source'])
    g = utils.parseFunction(data['exact'])

    for i in range(0,5):
        pb = problem.Problem(grid, data)
        pb.stiff(grid)
        pb.source(grid, f)
        pb.lifting(grid, g)
        uh = pb.solveLifting()

        normL2 = utils.l2Norm(grid, uh);
        print("L2 norm: ", normL2)
        grid.refine()


if __name__ == "__main__":
    main()
