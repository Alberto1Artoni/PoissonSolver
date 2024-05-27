from PoissonSolver import mesh
import json
import matplotlib.pyplot as plt

def main():
    filePath = "data.json"
    with open(filePath, 'r') as file:
       data = json.load(file)

    # generate mesh
    grid = mesh.Mesh(data)
    grid.plot()

    # We now loop and refine
    for i in range(0,3):
        # start with a finer refinement
        grid.refine()
        grid.plot()

if __name__ == "__main__":
    main()
