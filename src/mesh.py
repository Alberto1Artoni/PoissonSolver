import numpy as np

class Mesh:
    def __init__(self, data, path = None):

        if path is not None:
            self.load(path)
        else:
            """ Generates a structured triangular mesh """
            self.nx = data['nx']    # nodes in the x direction
            self.ny = data['ny']    # nodes in the y direction

            self.a = data['a']      # left x 
            self.b = data['b']      # right x
            self.c = data['c']      # bottom y
            self.d = data['d']      # top y

            """     a. generates the nodes """
            self.generate_nodes()

            """     b. generates the elements """
            self.generate_structured_elements()

            """     c. generates the boundary nodes """
            self.generate_boundary_nodes()


    def generate_nodes(self):
        self.coord_x =   np.tile(np.linspace(self.a, self.b, self.nx), self.ny)
        self.coord_y = np.repeat(np.linspace(self.c, self.d, self.ny), self.nx)

    def generate_boundary_nodes(self):
        """ generates the boundary nodes """
        # returns a vector collecting the boundary nodes
        """
        self.boundary_dofs = np.zeros(4 * (self.nx + self.ny - 2), dtype=int)
        self.boundary_dofs[0:self.nx] = np.arange(0, self.nx)
        self.boundary_dofs[self.nx : self.nx + self.ny - 1] = \
                                        np.arange(self.nx, self.nx*(self.ny-2) , self.nx)
        self.boundary_dofs[self.nx +   self.ny - 1: 
                           self.nx + 2*self.ny - 1] = \
                                        np.arange(self.nx, self.nx*(self.ny-2) , self.nx)
        """
        # hard coding for the moment
        self.boundary_dofs = np.zeros(4 * (self.nx + self.ny - 2), dtype=int)
        k = 0;
        for i in range(self.nx*self.ny):
            if self.coord_x[i] == 0 or self.coord_x[i] == 1 or \
               self.coord_y[i] == 0 or self.coord_y[i] == 1:
                self.boundary_dofs[k] = i
                k += 1


    def generate_structured_elements(self):
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
        jac[:, 0] = self.coord_x[self.elements[ie,  1:]] - \
                    self.coord_x[self.elements[ie,   0]]
        jac[:, 1] = self.coord_y[self.elements[ie,  1:]] - \
                    self.coord_y[self.elements[ie,   0]]

        det    = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
        invJac = (1 / det) * np.array([[jac[1,1], -jac[0,1]], [-jac[1,0], jac[0,0] ]])
        return jac, invJac


    def h(self):
        # for the moment keep it simple
        return 1.0 / self.nx


