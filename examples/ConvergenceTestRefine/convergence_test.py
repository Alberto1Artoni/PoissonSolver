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


    # parse source term
    # this needs to be checked, could be slow
    function_str = data['source']
    x = symbols('x')
    y = symbols('y')
    f_expr = eval(function_str, {"sin": sin, "cos": cos, "pi": pi, "x": x, "y": y})
    f = lambdify((x, y), f_expr, 'numpy')

    hList = [];
    normL2list = [];
    refs = [11,21,41,81];
    data['nx'] = 11;
    data['ny'] = 11;
    mesh = Mesh(data)

    for ref in refs:

        # generate mesh
        mesh.refine()

        # assemble problem
        problem = Problem(mesh, data)

        # solve linear system
        uh = problem.solveLifting()
        hList.append(mesh.h());
        normL2 = l2Norm(mesh, uh);
        normL2list.append(normL2);
        print("L2 norm: ", normL2)

#   # plot convergence
#   plt.figure()
#   plt.loglog(hList, normL2list, '-o')
#   plt.loglog(hList, np.array(hList)**2, 'k-.')
#   plt.xlabel('h')
#   plt.ylabel('L2 norm')
#   plt.title('Convergence test')
#   plt.legend(['L2 norm', 'h^2'])
#   plt.show()

if __name__ == "__main__":
    main()
