from mesh import Mesh
import numpy as np


class Problem:
    """ class defining the Poisson problem """
    def __init__(self, mesh, f):
        self.K  = self.stiff(mesh);
        self.fh = self.source(mesh, f);
        self.dirichlet(mesh);

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

    def solve(self):
        # solve the problem
        uh = np.linalg.solve(self.K, self.fh)
        return uh
