import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


class Problem:
    def __init__(self, mesh, data):
        """ Constructor for the Poisson problem

        Parameters
        ----------
        mesh : Mesh
            Contains mesh information, see mesh.py
        data : dict
            Contains the problem data, see mesh.py

        Attributes
        ----------
        ndof : int
            Number of degrees of freedom. For P1 is the same
            as the number of nodes of the grid.
        K : csr_matrix
            Stiffness matrix
        fh : ndarray
            Source vector

        """

        self.ndof = mesh.nx * mesh.ny

    def localStiff(self, mesh, ie):
        """ Compute local stiffness matrix for element ie

        Parameters
        ----------
        mesh : Mesh
            Contains mesh information, see mesh.py
        ie : int
            Element index

        Returns
        -------
        localStiff : ndarray
            Local stiffness matrix of the element ie
        """

        # precompute gradient of basis functions
        gradLambda = np.zeros((2, 3))
        gradLambda[:, 0] = [-1, -1]
        gradLambda[:, 1] = [1, 0]
        gradLambda[:, 2] = [0, 1]

        localStiff = np.zeros((3, 3))

        # get the jacobian
        [jac, invJac] = mesh.compute_jacobian(ie)

        # area of the element
        area = np.linalg.det(jac) / 2.0

        for i in range(3):
            for j in range(3):
                localStiff[i, j] = (gradLambda[:, i].T @ invJac @
                                    invJac.T @ gradLambda[:, j] * area)

        return localStiff

    def stiff(self, mesh):
        """ Assemble the stiffness matrix
        Loop over the elements and assemble the global stiffness matrix
        from the local stiffness matrix

        Parameters
        ----------
        mesh : Mesh
            Contains mesh information, see mesh.py

        Notes
        -----
            The stiffness matrix is assembled in COO format
        """

        # initialize sparse matrix in coo format
        coo_i = np.zeros(mesh.ne * 9, dtype=int)
        coo_j = np.zeros(mesh.ne * 9, dtype=int)
        val = np.zeros(mesh.ne * 9, dtype=float)

        # iterator to fill the sparse matrix
        k = np.arange(0, 9)

        # loop over elements, assemble global stiffness
        for ie in range(mesh.ne):
            localStiff = self.localStiff(mesh, ie)
            ii = np.repeat(np.arange(3), 3)
            jj = np.tile(np.arange(3), 3)
            coo_i[k] = mesh.elements[ie, ii]
            coo_j[k] = mesh.elements[ie, jj]
            val[k] = localStiff.flatten()
            k += 9

        # returns the stiff matrix
        self.K = sp.coo_matrix((val, (coo_i, coo_j)),
                               shape=(self.ndof, self.ndof))
        self.K.sum_duplicates()

    def source(self, mesh, f):
        """ Assemble the source vector.
        Loop over the elements and assemble the global source vector
        with a mid point quadrature rule.

        Parameters
        ----------
        mesh : Mesh
            Contains mesh information, see mesh.py
        f : function
            Source function

        Notes
        -----
            The source vector is assembled in a numpy array.
        """

        # preallocate the source vector
        fh = np.zeros(self.ndof)

        # loop over elements, assemble global source
        for ie in range(mesh.ne):

            # get jacobian
            [jac, invJac] = mesh.compute_jacobian(ie)

            # area of the element
            area = np.linalg.det(jac) / 2.0

            # get the barycenter of the element
            bar_x = np.sum(mesh.coord_x[mesh.elements[ie, :]]) / 3.0
            bar_y = np.sum(mesh.coord_y[mesh.elements[ie, :]]) / 3.0

            # compute the source term
            fh[mesh.elements[ie, :]] += (f(bar_x, bar_y) * area
                                         * np.ones(3)) / 3.0

        # store the source vector
        self.fh = fh

    def dirichlet(self, mesh, g):
        """ Enforce Dirichlet boundary conditions in a strong sense.

        Parameters
        ----------
        mesh : Mesh
            Contains mesh information, see mesh.py
        g : function
            Dirichlet boundary function

        Notes
        -----
            The function modifies the stiffness matrix and the source vector
            to enforce the Dirichlet boundary conditions in a strong sense.
            After the modification, the matrix is converted to csr format.
        """

        # enforce Dirichlet boundary conditions on nodes
        self.fh[mesh.boundary_dofs] = g(mesh.coord_x[mesh.boundary_dofs],
                                        mesh.coord_y[mesh.boundary_dofs])

        # get COO data format
        data = self.K.data
        row = self.K.row
        col = self.K.col

        # enforce Dirichlet boundary conditions on nodes
        mask = np.isin(row, mesh.boundary_dofs)
        indices = np.where(mask)
        data[indices] = np.where(row[indices] == col[indices], 1.0, 0.0)


        K = sp.coo_matrix((data, (row, col)),
                               shape=(self.ndof, self.ndof))

        # convert to csr format for faster algebra
        self.K = K.tocsr()

    def solve(self):
        """ Solve the algebraic problem

        Returns
        -------
        uh : ndarray
            Finite element solution
        """

        # solve the problem
        uh = spsolve(self.K, self.fh)

        return uh

    def lifting(self, mesh, g):
        """ Implementation of the lifting operator for
        Dirichlet boundary conditions

        Parameters
        ----------
        mesh : Mesh
            Contains mesh information, see mesh.py
        g : function
            Dirichlet boundary function

        Notes
        -----
            The function modifies the stiffness matrix and the source vector
            to enforce the Dirichlet boundary conditions with
            the lifting operator.
            After the modification, the matrix is converted to csr format.
            Also, after the call to the solve method,
            it requires the update of the solution.
            Use hence the solveLifting method.
        """

        self.ug = np.zeros(self.ndof)

        # enforce Dirichlet boundary conditions on nodes
        self.ug[mesh.boundary_dofs] = g(mesh.coord_x[mesh.boundary_dofs],
                                        mesh.coord_y[mesh.boundary_dofs])

        # lift the source
        self.fg = self.fh - self.K * self.ug

        # modify the entries of the stiffness matrix
        # get COO data format
        data = self.K.data
        row = self.K.row
        col = self.K.col

        # enforce Dirichlet boundary conditions on nodes
        mask_row = np.isin(row, mesh.boundary_dofs)
        mask_col = np.isin(col, mesh.boundary_dofs)
        mask = mask_row | mask_col
        indices = np.where(mask)
        data[indices] = np.where(col[indices] == row[indices], 1.0, 0.0)

        # update the stiffness matrix in COO format
        self.K = sp.coo_matrix((data, (row, col)),
                               shape=(self.ndof, self.ndof))

        # convert to csr format for faster algebra
        self.K = self.K.tocsr()

        # set the source vector to zero on the boundary
        self.fg[mesh.boundary_dofs] = 0

    def solveLifting(self):
        """ Solve the algebraic problem with the lifting operator

        Returns
        -------
        uh : ndarray
            Finite element solution

        Notes
        -----
            The function solves the algebraic problem with the lifting
            operator.
            Next, the solution is lifted.
        """

        # solve the homogeneous problem
        uh = spsolve(self.K, self.fg)

        # lift the solution
        uh = uh + self.ug
        return uh
