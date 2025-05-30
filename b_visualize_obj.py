import sys
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main(obj_path):
    # Load the mesh
    mesh = trimesh.load(obj_path, process=False, force="mesh")

    # If loaded as a scene (multi-part), take the first geometry
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = np.array(mesh.vertices)

    # Plot the vertices
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, c='b')

    ax.set_title(f"Visualization of: {obj_path}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_obj_matplotlib.py path/to/model.obj")
    else:
        main(sys.argv[1])