from mesh import Mesh
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

class Problem:
    """ class defining the Poisson problem """
    def __init__(self, mesh, f):
        #self.K  = self.stiff(mesh);
        self.K  = self.stiffSparse(mesh);
        self.fh = self.source(mesh, f);
        self.dirichletSparse(mesh);

    def localStiff(self, mesh, ie):
        # build local stiffness
        gradLambda = np.zeros((2,3));

        gradLambda[:,0] = [-1,-1];
        gradLambda[:,1] = [ 1, 0];
        gradLambda[:,2] = [ 0, 1];
        localStiff = np.zeros((3,3));

        [jac, invJac] = mesh.compute_jacobian(ie);
        halfArea = np.linalg.det(jac) / 2.0;  # area of the element
        for i in range(3):
            for j in range(3):
                localStiff[i,j] =  gradLambda[:,i].T @ invJac \
                                                     @ invJac.T @ gradLambda[:,j] * halfArea;
        return localStiff;

    def stiff(self, mesh):
        # loop over elements, assemble global stiffness
        ndof = mesh.nx * mesh.ny;   # true only for P1 elements
        K = np.zeros((ndof, ndof));

        # map dof to element is the same as the map vertex to element
        for ie in range(mesh.ne):
            localStiff = self.localStiff(mesh, ie);
            for i in range(3):
                for j in range(3):
                    K[mesh.elements[ie, i], mesh.elements[ie, j]] += localStiff[i, j];
        return K

    def stiffSparse(self, mesh):
        # loop over elements, assemble global stiffness
        ndof = mesh.nx * mesh.ny;   # true only for P1 elements
        coo_i = np.zeros( mesh.ne * 9 )
        coo_j = np.zeros( mesh.ne * 9 )
        val   = np.zeros( mesh.ne * 9 )

        k = 0
        # map dof to element is the same as the map vertex to element
        for ie in range(mesh.ne):
            localStiff = self.localStiff(mesh, ie);
            for i in range(3):
                for j in range(3):
                    coo_i[k] = mesh.elements[ie, i]
                    coo_j[k] = mesh.elements[ie, j]
                    val[k]   = localStiff[i, j];
                    k += 1
        K = sp.coo_matrix((val, (coo_i, coo_j)), shape=(ndof, ndof)) 
        return K

    def source(self, mesh, f):
        # loop over elements, assemble global source
        ndof = mesh.nx * mesh.ny;
        fh = np.zeros(ndof);

        # this is not efficient, but works
        for ie in range(mesh.ne):
            [jac, invJac] = mesh.compute_jacobian(ie);
            det = np.linalg.det(jac);
            bar_x = np.sum(mesh.coord_x[mesh.elements[ie,:]]) / 3.0;
            bar_y = np.sum(mesh.coord_y[mesh.elements[ie,:]]) / 3.0;
            fh[mesh.elements[ie,:]] = f(bar_x, bar_y) * det * np.ones(3);

        return fh

    def dirichlet(self, mesh):
        # enforce Dirichlet boundary conditions on nodes
        for i in mesh.boundary_dofs:
            self.K[i, :] = 0
            self.K[i, i] = 1
            self.fh[i] = 0

        # @TODO implement lifting

    def dirichletSparse(self, mesh):
        # enforce Dirichlet boundary conditions on nodes
        for i in mesh.boundary_dofs:
            self.fh[i] = 0

        data = self.K.data
        row  = self.K.row
        col  = self.K.col
        ndof = mesh.nx * mesh.ny
        for k in range(data.size):
            if ( row[k] in  mesh.boundary_dofs):
                if col[k] == row[k]:
                    data[k] = 1.0
                else:
                    data[k] = 0.0
        self.K = sp.coo_matrix((data, (row, col)), shape=(ndof, ndof))
        # convert to csr format
        self.K = self.K.tocsr()

    def solve(self):
        # solve the problem
        uh = spsolve(self.K, self.fh)
        return uh
