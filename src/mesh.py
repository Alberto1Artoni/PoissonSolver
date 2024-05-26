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
#       self.boundary_dofs = np.concatenate((\
#                                np.arange(0, self.nx),                             \
#                                np.arange(self.nx, self.nx*(self.ny-1) , self.nx), \
#                                np.arange(2*self.nx-1, self.nx*(self.ny-1) , self.nx), \
#                                np.arange(self.nx*(self.ny-1), self.nx*self.ny)))
        # hard coding for the moment
        self.boundary_dofs = np.zeros(2 * (self.nx + self.ny - 2), dtype=int)
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

        # odd elements are lower triangles, while even elements are upper triangles
        self.elements = np.zeros( (self.ne, 3), dtype=int)

        # store element connectivity in counter-clockwise order
        lower = np.array([1, self.nx + 1, 0]);
        upper = np.array([self.nx, 0, self.nx+1]);        
        

        # @TODO this can be vectorized
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
                                neig[ie,iv] = je
        
        isRefined = np.zeros(self.ne)

        #   3. Refine

        # generates the refined skeleton
        new_nodes = [];     # empty list collecting the owners of the new nodes
        new_coords = [];    # empty list collecting the coordinates of the new nodes
        new_own = [[] for _ in range(self.ne)]
                                                # empty list collecting the owner of each 
                                                # new node

        shift = self.nx*self.ny
        for ie in range(self.ne):
            for iv in range(3):
                if (neig[ie,iv] == -1): # check if it is a boundary edge
                    # refine
                    new_nodes.append([ie, -1])
                    i = self.elements[ie,  iv]
                    j = self.elements[ie, (iv+1)%3]
                    x = 0.5 * (self.coord_x[i] + self.coord_x[j])
                    y = 0.5 * (self.coord_y[i] + self.coord_y[j])
                    new_coords.append([x, y])
                    new_own[ie].append([i,len(new_nodes)-1 + shift])
                    new_own[ie].append([len(new_nodes)-1 + shift,j])

                else:
                    if (isRefined[neig[ie,iv]] == 1):
                        # push the edge 
                        nie = neig[ie,iv]
                        # need to get the proper edge
                        # current edge on ie is iv, iv+1
                        # I am looking for the edge on nie
                        for kv in range(3):
                            for jv in range(3):
                                if (self.elements[nie, jv]      == self.elements[ie, kv]) and \
                                   (self.elements[nie,(jv+1)%3] == self.elements[ie,(kv+1)%3]):
                                   neig_jv = jv;
                                   neig_kv = kv;
                                   break
                                if (self.elements[nie,(jv+1)%3] == self.elements[ie, kv]) and \
                                   (self.elements[nie,(jv)]     == self.elements[ie,(kv+1)%3]):
                                   neig_kv = kv;
                                   neig_jv = jv;
                                   break
                        # adding these elements doesn't preseve the order
                        # it needs to be checked:
                        if (self.elements[ie,neig_kv] == new_own[nie][2*neig_jv][0]):
                            new_own[ie].append(new_own[nie][2*neig_jv])
                            new_own[ie].append(new_own[nie][2*neig_jv+1])

                        if (self.elements[ie,neig_kv] == new_own[nie][2*neig_jv+1][1]):
                            new_own[ie].append([new_own[nie][2*neig_jv+1][1], new_own[nie][2*neig_jv+1][0]])
                            new_own[ie].append([new_own[nie][2*neig_jv][1], new_own[nie][2*neig_jv][0]])

                    else:
                        i = self.elements[ie,  iv]
                        j = self.elements[ie, (iv+1)%3]
                        new_nodes.append([ie, neig[ie,iv]])
                        x = 0.5 * (self.coord_x[i] + self.coord_x[j])
                        y = 0.5 * (self.coord_y[i] + self.coord_y[j])
                        new_coords.append([x, y])
                        new_own[ie].append([i,len(new_nodes)-1 + shift])
                        new_own[ie].append([len(new_nodes)-1 + shift,j])

            isRefined[ie] = 1

        #   4. Update the mesh
        new_elements = []

        for ie in range(self.ne):
            for iv in range(1,len(new_own[ie]),2):
               new_elements.append([new_own[ie][(iv  )%6][0], \
                                    new_own[ie][(iv  )%6][1], \
                                    new_own[ie][(iv+1)%6][1]])
            new_elements.append([new_own[ie][0][1], new_own[ie][2][1], new_own[ie][4][1]])

        self.elements = np.array(new_elements)
        coord_x = np.concatenate((self.coord_x, [x[0] for x in new_coords]))
        coord_y = np.concatenate((self.coord_y, [y[1] for y in new_coords]))
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.ne = len(self.elements)
        self.nx = 2*self.nx - 1
        self.ny = 2*self.ny - 1

        # update boundary nodes
        self.generate_boundary_nodes()


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

