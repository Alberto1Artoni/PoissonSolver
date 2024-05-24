from .mesh import Mesh
import numpy as np
from sympy import symbols, sin, cos, pi
from sympy.utilities.lambdify import lambdify

def l2Norm(mesh, uh):

    # here I need a higher order quadrature rule
    [weig, xnod, ynod] = get_quad( );
    shape = eval_shape(xnod, ynod);

    l2norm = 0.0;

    for ie in range(mesh.ne):
        [jac, invJac] = mesh.compute_jacobian(ie);
        x = np.zeros(3);
        y = np.zeros(3);
        for i in range(3):
            vett  = np.array([xnod[i], ynod[i]])[:,None];
            trans = np.array([mesh.coord_x[mesh.elements[ie,0]], mesh.coord_y[mesh.elements[ie,0]]])[:, None];
            vphys = jac@vett + trans
            x[i] = vphys[0];
            y[i] = vphys[1];
        

        dofs = mesh.elements[ie,:]

        local_uh = np.zeros(3);
        # evaluate the function at the quadrature points
        for i in range(3):
            local_uh[i] = local_uh[i] + np.dot(uh[dofs],shape[i,:]);

        areaPhys = np.abs(np.linalg.det(jac)) / 2.0;
        # evaluate the integral
        l2norm += np.dot(weig , (np.sin(2*np.pi * x) * np.sin(2*np.pi * y) - local_uh)**2) * areaPhys;

    l2norm = np.sqrt(l2norm);
    return l2norm

def eval_shape(xnod, ynod):
    shape = np.zeros((3,3));
    shape[0,:] = np.ones(3) - xnod - ynod;
    shape[1,:] = xnod;
    shape[2,:] = ynod;

    return shape

def get_quad():

    # hard coded quadrature rule
    w = np.array([1.0/3, 1.0/3, 1.0/3]);
    x = np.array([1.0/2, 1.0/2, 0.0]);
    y = np.array([1.0/2, 0.0, 1.0/2]);

    return w, x, y


def parseFunction(fun_str ):
    """ helper function to parse a string and return a function """
    x = symbols('x')
    y = symbols('y')
    h_expr = eval(fun_str, {"sin": sin, "cos": cos, "pi": pi, "x": x, "y": y})
    h = lambdify((x, y), h_expr, 'numpy')

    return h
