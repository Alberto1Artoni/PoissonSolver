import numpy as np
from sympy import symbols, sin, cos, pi
from sympy.utilities.lambdify import lambdify


def l2Norm(mesh, uh):
    """ Compute the L2 norm of the error between the exact solution
        and the numerical solution

    Parameters
    ----------
    mesh : Mesh
        Mesh object
    uh : numpy array
        Numerical solution

    Returns
    -------
    l2norm : float
        L2 norm of the error

    Notes
    -----
        At the moment the analytical solution is hard coded

    """

    # I need a higher order quadrature rule
    # mid point quad rule is not enough accurate
    [weig, xnod, ynod] = get_quad()

    # returns the shape functions evaluated at the quadrature points
    shape = eval_shape(xnod, ynod)

    l2norm = 0.0

    # loop over the elements
    for ie in range(mesh.ne):
        # get the jacobian and the inverse of the jacobian
        [jac, invJac] = mesh.compute_jacobian(ie)
        x = np.zeros(3)
        y = np.zeros(3)
        for i in range(3):
            vett = np.array([xnod[i], ynod[i]])[:, None]
            trans = np.array([mesh.coord_x[mesh.elements[ie, 0]],
                              mesh.coord_y[mesh.elements[ie, 0]]])[:, None]
            vphys = jac@vett + trans
            x[i] = vphys[0]
            y[i] = vphys[1]

        dofs = mesh.elements[ie, :]

        local_uh = np.zeros(3)
        # evaluate the function at the quadrature points
        for i in range(3):
            local_uh[i] = local_uh[i] + np.dot(uh[dofs], shape[i, :])

        areaPhys = np.abs(np.linalg.det(jac)) / 2.0

        # evaluate the integral
        l2norm += np.dot(weig, (np.sin(2*np.pi * x) *
                                np.sin(2*np.pi * y) - local_uh)**2) * areaPhys

    l2norm = np.sqrt(l2norm)
    return l2norm


def eval_shape(xnod, ynod):
    """ Evaluate the shape functions at the quadrature points.
    Shape functions are P1 linear shape functions.

    Parameters
    ----------
    xnod : numpy array
        x nodes
    ynod : numpy array
        y nodes

    Returns
    -------
    shape : numpy array
        Shape functions evaluated in the node points
    """

    shape = np.zeros((3, 3))

    # Lagrangian linear function: 1-x-y
    shape[0, :] = np.ones(3) - xnod - ynod

    # Lagrangian linear function: x
    shape[1, :] = xnod

    # Lagrangian linear function: y
    shape[2, :] = ynod

    return shape


def get_quad():
    """ Get mid-point edge quadrature for a triangular element

    Returns
    -------
    w : numpy array
        Weights
    x : numpy array
        x quadrature nodes
    y : numpy array
        y quadrature nodes
    """

    # hard coded quadrature rule
    w = np.array([1.0/3, 1.0/3, 1.0/3])
    x = np.array([1.0/2, 1.0/2, 0.0])
    y = np.array([1.0/2, 0.0, 1.0/2])

    return w, x, y


def parseFunction(fun_str):
    """ Helper function to parse a string and return a function

    Parameters
    ----------
    fun_str : str
        String representing the function

    Returns
    -------
    h : function
        Lambda function representing the function

    Notes
    -----
        Check if all the symbols are defined in the function

    @TODO understand if something else is also possible
    """

    x = symbols('x')
    y = symbols('y')
    h_expr = eval(fun_str, {"sin": sin, "cos": cos, "pi": pi, "x": x, "y": y})
    h = lambdify((x, y), h_expr, 'numpy')

    return h
