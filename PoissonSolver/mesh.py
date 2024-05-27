import numpy as np
import matplotlib.pyplot as plt


class Mesh:
    def __init__(self, data):
        """ Generates a structured triangular mesh

        Parameters
        ----------

        data : dict
            a dictionary containing the following keys:
            nx : int
                number of nodes in the x direction
            ny : int
                number of nodes in the y direction
            a : float
                domain limits, [a, b] x [c, d]
            b : float
                domain limits, [a, b] x [c, d]
            c : float
                domain limits, [a, b] x [c, d]
            d : float
                domain limits, [a, b] x [c, d]

        Attributes
        ----------
        coord_x : numpy.ndarray
            Stored array of x-coordinates of the nodes of the grid.
        coord_y : numpy.ndarray
            Stored array of y-coordinates of the nodes of the grid.
        elements : numpy.ndarray
            Stored array of element connectivity.
        boundary_dofs : numpy.ndarray
            Stored array of boundary nodes.
        """

        self.nx = data['nx']    # number of nodes in the x direction
        self.ny = data['ny']    # number of nodes in the y direction
        self.a = data['a']      # left x
        self.b = data['b']      # right x
        self.c = data['c']      # bottom y
        self.d = data['d']      # top y

        # generates the nodes of the grid
        self.generate_nodes()

        # generates the element connectivity
        self.generate_structured_elements()

        # tag the boundary nodes
        self.generate_boundary_nodes()

    def generate_nodes(self):
        """ Generates the nodes of a structured mesh

        Notes
        -----
            After the method is called, the following attributes are set:
            coord_x : numpy.ndarray
                Stored array of x-coordinates of the nodes of the grid.
            coord_y : numpy.ndarray
                Stored array of y-coordinates of the nodes of the grid.
        """

        # generate the x coordinates
        self.coord_x = np.tile(np.linspace(self.a, self.b, self.nx), self.ny)

        # generate the y coordinates
        self.coord_y = np.repeat(np.linspace(self.c, self.d, self.ny), self.nx)

    def generate_boundary_nodes(self):
        """ Generates the boundary nodes

        Notes
        -----
            The method assumes that the boundary nodes are ordered
            After the method is called, the following attributes are set:
            boundary_dofs : numpy.ndarray
                Stored array of boundary nodes.
        """
        self.boundary_dofs = np.concatenate((
                         np.arange(0, self.nx),
                         np.arange(self.nx, self.nx*(self.ny-1), self.nx),
                         np.arange(2*self.nx-1, self.nx*(self.ny-1), self.nx),
                         np.arange(self.nx*(self.ny-1), self.nx*self.ny)))

    def generate_structured_elements(self):
        """ Generates the element local connectivity of a
            triangular structured mesh

        Notes
        -----
            The method assumes that the mesh is triangular and structured
            After the method is called, the following attributes are set:
            elements : numpy.ndarray
                Stored array of element connectivity.
        """

        # number of elements given a structured triangular mesh
        self.ne = 2 * (self.nx - 1) * (self.ny - 1)

        # odd elements are lower triangles,
        # while even elements are upper triangles
        self.elements = np.zeros((self.ne, 3), dtype=int)

        # store element connectivity in counter-clockwise order
        lower = np.array([1, self.nx + 1, 0])
        upper = np.array([self.nx, 0, self.nx+1])

        # loop over the elements
        for j in range(0, self.ny - 1):
            for i in range(0, self.nx - 1):
                k = 2 * (j * (self.nx - 1) + i)
                # lower triangle
                self.elements[k, :] = lower + i + j * self.nx
                # upper triangle
                self.elements[k + 1, :] = upper + i + j * self.nx

    def compute_jacobian(self, ie):
        """ Compute the Jacobian map and its inverse

        Parameters
        ----------
        ie : int
            element index

        Returns
        -------
        jac : numpy.ndarray
            Jacobian map
        invJac : numpy.ndarray
            Inverse of the Jacobian map
        """

        # compute the Jacobian
        jac = np.zeros((2, 2))
        jac[0, :] = (
                self.coord_x[self.elements[ie, 1:]] -
                self.coord_x[self.elements[ie, 0]]
                )
        jac[1, :] = (
                self.coord_y[self.elements[ie, 1:]] -
                self.coord_y[self.elements[ie, 0]]
                )

        # compute the determinant of the Jacobian
        det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]

        # compute the inverse of the Jacobian
        invJac = (1 / det) * np.array([[jac[1, 1], -jac[0, 1]],
                                       [-jac[1, 0], jac[0, 0]]])
        return jac, invJac

    def refine(self):
        """ Midpoint edge refinement

        The implementation wants to be more general and to take into account
        also the non-structured case.
        To handle the general case, we need to enrich the mesh class with
        more information on the geometry.
        We make the following steps:
        1. We compute the owner of each vertex.
            The idea is to have faster computations.

        2. We compute the neighbours of each element.
            We set to -1 the boundary neighbours.

        3. We compute the skeleton of the grid and we refine the edges.
            This step is delicate.

        4. We update the mesh.

        @TODO: The code is not very efficient and needs to be improved,
               but it works.

        """

        #   1. Compute element owner of each vertex
        # compute vertex connectivity for faster computations
        own = [[] for _ in range(self.nx * self.ny)]
        for ie in range(self.ne):
            for iv in range(3):
                own[self.elements[ie, iv]].append(ie)

        #   2. Compute neighbours
        #  set to -1 boundary neig
        neig = -np.ones([self.ne, 3], dtype=int)
        for ie in range(self.ne):
            for iv in range(3):
                i = self.elements[ie, iv]
                j = self.elements[ie, (iv+1) % 3]
                for je in own[i]:
                    if je != ie:
                        for jv in range(3):
                            if self.elements[je, jv] == j:
                                neig[ie, iv] = je

        isRefined = np.zeros(self.ne)

        #   3. Refine: mid-point edge refinement

        # empty list collecting the owners of the new nodes
        new_nodes = []

        # empty list collecting the coordinates of the new nodes
        new_coords = []

        # empty list collecting the owner of each new node
        new_own = [[] for _ in range(self.ne)]

        # shift to avoid overlapping with the old nodes
        shift = self.nx*self.ny

        for ie in range(self.ne):
            for iv in range(3):
                if (neig[ie, iv] == -1):  # check if it is a boundary edge
                    # if boundary edge, I have no neighbours

                    new_nodes.append([ie, -1])
                    i = self.elements[ie, iv]
                    j = self.elements[ie, (iv+1) % 3]
                    x = 0.5 * (self.coord_x[i] + self.coord_x[j])
                    y = 0.5 * (self.coord_y[i] + self.coord_y[j])
                    new_coords.append([x, y])
                    new_own[ie].append([i, len(new_nodes)-1 + shift])
                    new_own[ie].append([len(new_nodes)-1 + shift, j])

                else:
                    if (isRefined[neig[ie, iv]] == 1):
                        # push the edge on the  new_own list
                        nie = neig[ie, iv]

                        # need to get the proper edge
                        # current edge on ie is iv, iv+1
                        # I am looking for the edge on nie
                        for kv in range(3):
                            for jv in range(3):
                                if ((self.elements[nie, jv] ==
                                     self.elements[ie, kv]) and
                                    (self.elements[nie, (jv+1) % 3] ==
                                     self.elements[ie, (kv+1) % 3])):

                                    neig_jv = jv
                                    neig_kv = kv
                                    break
                                if ((self.elements[nie, (jv+1) % 3] ==
                                     self.elements[ie, kv]) and
                                    (self.elements[nie, (jv)] ==
                                     self.elements[ie, (kv+1) % 3])):

                                    neig_kv = kv
                                    neig_jv = jv
                                    break

                        # adding these elements doesn't preseve the order
                        # it needs to be checked:

                        if (self.elements[ie, neig_kv] ==
                                new_own[nie][2*neig_jv][0]):

                            new_own[ie].append(new_own[nie][2*neig_jv])
                            new_own[ie].append(new_own[nie][2*neig_jv+1])

                        if (self.elements[ie, neig_kv] ==
                                new_own[nie][2*neig_jv+1][1]):

                            new_own[ie].append([new_own[nie][2*neig_jv+1][1],
                                                new_own[nie][2*neig_jv+1][0]])
                            new_own[ie].append([new_own[nie][2*neig_jv][1],
                                                new_own[nie][2*neig_jv][0]])

                    else:
                        i = self.elements[ie, iv]
                        j = self.elements[ie, (iv+1) % 3]
                        new_nodes.append([ie, neig[ie, iv]])
                        x = 0.5 * (self.coord_x[i] + self.coord_x[j])
                        y = 0.5 * (self.coord_y[i] + self.coord_y[j])
                        new_coords.append([x, y])
                        new_own[ie].append([i, len(new_nodes)-1 + shift])
                        new_own[ie].append([len(new_nodes)-1 + shift, j])

            isRefined[ie] = 1

        #   4. Update the mesh
        new_elements = []

        # exploit the topology of the old elements to build the new elements
        for ie in range(self.ne):
            for iv in range(1, len(new_own[ie]), 2):
                new_elements.append([new_own[ie][(iv) % 6][0],
                                    new_own[ie][(iv) % 6][1],
                                    new_own[ie][(iv+1) % 6][1]])
            new_elements.append([new_own[ie][0][1],
                                 new_own[ie][2][1], new_own[ie][4][1]])

        # update mesh elements
        self.elements = np.array(new_elements)
        coord_x = np.concatenate((self.coord_x, [x[0] for x in new_coords]))
        coord_y = np.concatenate((self.coord_y, [y[1] for y in new_coords]))
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.ne = len(self.elements)
        self.nx = 2*self.nx - 1
        self.ny = 2*self.ny - 1

        # update mesh boundary conditions
        self.boundary_dofs = np.zeros(2 * (self.nx + self.ny - 2), dtype=int)
        tol = 1e-9    # handle floating point errors

        # Create masks for conditions
        mask_x_a = np.isclose(self.coord_x, self.a, atol=tol)
        mask_x_b = np.isclose(self.coord_x, self.b, atol=tol)
        mask_y_c = np.isclose(self.coord_y, self.c, atol=tol)
        mask_y_d = np.isclose(self.coord_y, self.d, atol=tol)

        # Combine masks for x and y coordinates
        mask_x = np.logical_or(mask_x_a, mask_x_b)
        mask_y = np.logical_or(mask_y_c, mask_y_d)
        mask = np.logical_or(mask_x, mask_y)

        # Find indices where any of the conditions are True
        boundary_indices = np.nonzero(mask)[0]

        # Fill boundary_dofs array
        self.boundary_dofs[:len(boundary_indices)] = boundary_indices

    def plot(self):
        """ Utility to plot the mesh

        Represents each of the triangular elements in the mesh
        """

        # open a figure
        plt.figure()

        # loop over the elements
        for ie in range(self.ne):

            # get local coordinates
            x = np.zeros(4)
            y = np.zeros(4)
            for i in range(3):
                x[i] = self.coord_x[self.elements[ie, i]]
                y[i] = self.coord_y[self.elements[ie, i]]

            # get last elements to close the edge loop
            x[3] = x[0]
            y[3] = y[0]

            # plot the element
            plt.plot(x, y, 'b')

        # show the grid
        plt.show()

    def h(self):
        """ Returns the mesh size for a structured triangular grid

        Returns
        -------
        h : float
            Mesh size

        Notes
        -----
            The mesh size is computed as the distance between the mid point of
            the hypotenuse and the vertex sqaured corner of the triangle.
        """
        return np.sqrt(2)/2.0 * max(self.coord_x[1] - self.coord_x[0],
                                    self.coord_y[self.nx] - self.coord_y[0])
