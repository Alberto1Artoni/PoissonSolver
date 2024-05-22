import numpy as np

class Mesh:
    def __init__(self, data, path = None):

        if path is not None:
            self.load(path)
        else:
            """ Generates a structured triangular mesh """

            """ 1. generates the grid on the unit square """
            """     a. generates the nodes """
            self.generate_nodes(data)

            """     b. generates the elements """
            self.generate_structured_elements(data)


    def generate_nodes(self, data):
        self.nx = data['nx']        
        self.ny = data['ny']

        """ generate the coordinates """
        self.coord_x =   np.tile(np.linspace(0, 1, self.nx), self.ny)
        self.coord_y = np.repeat(np.linspace(0, 1, self.ny), self.nx)

    def generate_structured_elements(self, data):
        """ generates element local connectivity """
        # number of elements given a structured triangular mesh
        self.ne = 2 * (self.nx - 1) * (self.ny - 1)

        #Odd elements are lower triangles, while even elements are upper triangles
        self.elements = np.zeros( (self.ne, 3), dtype=int)

        # store element connectivity in counter-clockwise order
        lower = np.array([0, 1, self.nx]);
        upper = np.array([self.nx + 1, self.nx, 1]);        
        
        for i in range(0, self.nx - 1):
            for j in range(0, self.ny - 1):
                k = 2 * (i * (self.ny - 1) + j)
                self.elements[k, :]     = lower + i + j * self.nx
                self.elements[k + 1, :] = upper + i + j * self.nx

    def compute_jacobian(self, ie):
        # compute the Jacobian
        jac = np.zeros((2, 2))
        jac[:, 0] = self.coord_x[self.elements[ie, 1:]] - \
                      self.coord_x[self.elements[ie,   0]]
        jac[:, 1] = self.coord_y[self.elements[ie, 1:]] - \
                      self.coord_y[self.elements[ie,   0]]

        # @TODO implement the explicit formula for the inverse of a 2x2 matrix
        invJac = np.linalg.inv(jac)
        return jac, invJac


    def h(self):
        # for the moment keep it simple
        return 1.0 / self.nx


