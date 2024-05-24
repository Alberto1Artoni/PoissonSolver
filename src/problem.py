from .mesh import Mesh
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

class Problem:
    """ class defining the Poisson problem """

    def __init__(self, mesh, f):
        """ Constructor """
        # assemble stiffness
        self.K  = self.stiff(mesh)

        # assemble source
        self.fh = self.source(mesh, f)

        # enforce Dirichlet boundary conditions
        self.dirichlet(mesh)

    
    def localStiff(self, mesh, ie):
        """ compute local stiffness matrix """

        # precompute gradient of basis functions
        gradLambda = np.zeros((2,3))
        gradLambda[:,0] = [-1,-1]
        gradLambda[:,1] = [ 1, 0]
        gradLambda[:,2] = [ 0, 1]

        localStiff = np.zeros((3,3))

        # get the jacobian
        [jac, invJac] = mesh.compute_jacobian(ie)
        halfArea = np.linalg.det(jac) / 2.0  # area of the element

        for i in range(3):
            for j in range(3):
                localStiff[i,j] =  gradLambda[:,i].T @ invJac \
                                                     @ invJac.T @ gradLambda[:,j] * halfArea

        return localStiff

    def stiff(self, mesh):
        """ assemble global stiffness matrix """

        ndof = mesh.nx * mesh.ny        # true only for P1 elements

        # initialize sparse matrix in coo format
        coo_i = np.zeros( mesh.ne * 9 )
        coo_j = np.zeros( mesh.ne * 9 )
        val   = np.zeros( mesh.ne * 9 )

        # iterator to fill the sparse matrix
        k = 0

        # loop over elements, assemble global stiffness
        for ie in range(mesh.ne):
            localStiff = self.localStiff(mesh, ie)
            for i in range(3):
                for j in range(3):
                    coo_i[k] = mesh.elements[ie, i]
                    coo_j[k] = mesh.elements[ie, j]
                    val[k]   = localStiff[i, j]
                    k += 1

        # return the stiff matrix
        K = sp.coo_matrix((val, (coo_i, coo_j)), shape=(ndof, ndof)) 
        return K

    def source(self, mesh, f):
        # loop over elements, assemble global source
        ndof = mesh.nx * mesh.ny
        fh = np.zeros(ndof)

        # @TODO improve this loop
        # this is not efficient, but works
        for ie in range(mesh.ne):
            [jac, invJac] = mesh.compute_jacobian(ie)
            det = np.linalg.det(jac)
            bar_x = np.sum(mesh.coord_x[mesh.elements[ie,:]]) / 3.0
            bar_y = np.sum(mesh.coord_y[mesh.elements[ie,:]]) / 3.0
            fh[mesh.elements[ie,:]] = f(bar_x, bar_y) * det * np.ones(3)

        return fh


    def dirichlet(self, mesh):
        """ enforce Dirichlet boundary conditions """

        # @TODO implement lifting
        # enforce Dirichlet boundary conditions on nodes
        for i in mesh.boundary_dofs:
            self.fh[i] = 0
        
        # get COO data format
        data = self.K.data
        row  = self.K.row
        col  = self.K.col

        ndof = mesh.nx * mesh.ny

        # enforce Dirichlet boundary conditions on nodes
        for k in range(data.size):
            if ( row[k] in  mesh.boundary_dofs):
                if col[k] == row[k]:
                    data[k] = 1.0
                else:
                    data[k] = 0.0
        self.K = sp.coo_matrix((data, (row, col)), shape=(ndof, ndof))

        # convert to csr format for faster algebra
        self.K = self.K.tocsr()

    def solve(self):
        # solve the problem
        uh = spsolve(self.K, self.fh)
        return uh

