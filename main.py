import algorithms as alg
import numpy as np

if __name__ == "__main__":
    image = np.array([[0, 1, 2], [3, 4, 5]])
    s = (0, 0)
    t = (1, 1)
    is_four_neighbors = True
    sizes, edges = alg.graph_by_image(image.shape[0], image.shape[1], s, t, is_four_neighbors)
    print("image: ")
    print(image)
    print()
    print("sizes: ", sizes)
    print("edges: ", list(edges.keys()))


