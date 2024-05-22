from mesh import Mesh
import numpy as np

class Problem:
    """ class defining the Poisson problem """
    def __init__(self, mesh):
        self.stiff(mesh);

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
                                                     @ jac.T @ gradLambda[:,j] * halfArea;
        return localStiff;

    def stiff(self, mesh):
        # loop over elements, assemble global stiffness
        ndof = mesh.nx * mesh.ny;   # true only for P1 elements
        self.K = np.zeros((ndof, ndof));

        # map dof to element is the same as the map vertex to element
        for ie in range(mesh.ne):
            localStiff = self.localStiff(mesh, ie);
            for i in range(3):
                for j in range(3):
                    self.K[mesh.elements[ie, i], mesh.elements[ie, j]] += localStiff[i, j];
"""
    def source(self, vec):
        # build source

    def solve(self, u):
    # solve the problem
        print("Non yet implemented")
"""
