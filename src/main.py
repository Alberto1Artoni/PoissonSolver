from mesh    import Mesh
from problem import Problem
import json

def main():
    # read input
    filePath = "data.json"
    with open(filePath, 'r') as file:
        data = json.load(file)

    # generate mesh
    mesh = Mesh(data)

    print("Dumping mesh coordinates...")
    print(mesh.coord_x)

    print("Dumping mesh connectivity...")
    print(mesh.elements)

    # assemble matrices
    problem = Problem(mesh)

    # assemble right-hand side

    # solve linear system

    # post-process

if __name__ == "__main__":
    main()
