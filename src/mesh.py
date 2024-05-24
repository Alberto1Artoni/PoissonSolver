import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    def __init__(self, data):
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
        # @TODO fix this

#       self.boundary_dofs = np.zeros(4 * (self.nx + self.ny - 2), dtype=int)
#       self.boundary_dofs[0:self.nx] = np.arange(0, self.nx)
#       self.boundary_dofs[self.nx : self.nx + self.ny - 1] = \
#                                       np.arange(self.nx, self.nx*(self.ny-2) , self.nx)
#       self.boundary_dofs[self.nx +   self.ny - 1: 
#                          self.nx + 2*self.ny - 1] = \
#                                       np.arange(self.nx, self.nx*(self.ny-2) , self.nx)

        # hard coding for the moment
        self.boundary_dofs = np.zeros(2 * (self.nx + self.ny - 2), dtype=int)
        k = 0;
        for i in range(self.nx*self.ny):
            if self.coord_x[i] == self.a or self.coord_x[i] == self.b or \
               self.coord_y[i] == self.c or self.coord_y[i] == self.d:
                self.boundary_dofs[k] = i
                k += 1


    def generate_structured_elements(self):
        """ generates element local connectivity """
        # number of elements given a structured triangular mesh
        self.ne = 2 * (self.nx - 1) * (self.ny - 1)

        # odd elements are lower triangles, while even elements are upper triangles
        self.elements = np.zeros( (self.ne, 3), dtype=int)

        # store element connectivity in counter-clockwise order
        lower = np.array([1, self.nx + 1, 0]);
        upper = np.array([self.nx, 0, self.nx+1]);        
        
        for j in range(0, self.ny - 1):
            for i in range(0, self.nx - 1):
                k = 2 * (j * (self.nx - 1) + i)
                self.elements[k, :]     = lower + i + j * self.nx
                self.elements[k + 1, :] = upper + i + j * self.nx

    def compute_jacobian(self, ie):
        # compute the Jacobian
        jac = np.zeros((2, 2))
        jac[0, :] = self.coord_x[self.elements[ie,  1:]] - \
                    self.coord_x[self.elements[ie,   0]]
        jac[1, :] = self.coord_y[self.elements[ie,  1:]] - \
                    self.coord_y[self.elements[ie,   0]]

        det    = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
        invJac = (1 / det) * np.array([[jac[1,1], -jac[0,1]], [-jac[1,0], jac[0,0] ]])
        return jac, invJac

    def refine(self):
        """ Refine the mesh """
        # Midpoint edge refinement

        # A general implementation requires a better implementation 
        # of the mesh data structure.
        # This probably should be called inside the constructor,
        # but since this is a poc I will leave it here.

        #   1. Connectivity of the mesh
        # compute vertex connectivity for faster computations
        own = [[] for _ in range(self.nx * self.ny)]
        for ie in range(self.ne):
            for iv in range(3):
                own[self.elements[ie,iv]].append(ie)

        #   2. Compute neighobours
        #  set to -1 boundary neig
        neig = -np.ones([self.ne,3], dtype=int)
        for ie in range(self.ne):
            ineig = 0
            for iv in range(3):
                i = self.elements[ie,iv]
                j = self.elements[ie,(iv+1)%3]
                for je in own[i]:
                    if je != ie:
                        for jv in range(3):
                            if self.elements[je,jv] == j:
                                neig[ie,ineig] = je
                                ineig += 1
        
        isRefined = np.zeros(self.ne)

        #   3. Refine
        new_nodes = [];     # empty list collecting the owners of the new nodes
        new_coords = [];    # empty list collecting the coordinates of the new nodes
        for ie in range(self.ne):
            for iv in range(3):
                if (neig[ie,iv] == -1): # check if it is a boundary edge
                    # refine
                    print("Refining element ", ie)
                    new_nodes.append([ie, -1])
                else:
                    if (isRefined[neig[ie,iv]] == 1):
                        # do nothing
                        print("do nothing")
                    else:
                        i = self.elements[ie,  iv]
                        j = self.elements[ie, (iv+1)%3]
                        new_nodes.append([ie, neig[ie,iv]])
                        x = 0.5 * (self.coord_x[i] + self.coord_x[j])
                        y = 0.5 * (self.coord_y[i] + self.coord_y[j])
                        new_coords.append([x, y])

            isRefined[ie] = 1

        print(new_nodes)
        print(isRefined)

        

    def plot(self):
        """ Utility to plot the mesh """
        plt.figure()
        for ie in range(self.ne):
            x = np.zeros(4)
            y = np.zeros(4)
            for i in range(3):
                x[i] = self.coord_x[self.elements[ie, i]]
                y[i] = self.coord_y[self.elements[ie, i]]
            x[3] = x[0]
            y[3] = y[0]
            plt.plot(x, y, 'b')
        plt.show()

    def h(self):
        """ returns the mesh size for a structured triangular grid """
        return np.sqrt(2)/2.0 * max(self.coord_x[1] - self.coord_x[0], self.coord_y[self.nx] - self.coord_y[0])

