from .algorithms import *
import numpy as np
from PIL import Image


def algorithm(file_name, verifier_name, obj_pixels, bg_pixels, is_four_neighbors, lyambda, sigma):
    # test data
    # white = 255
    # black = 0
    # image = np.array([[200, 180, 20], [180, 160, 10], [160, 20, 10]])
    # obj_pixels = {(1, 1)}
    # bg_pixels = {(2, 1)}
    # is_four_neighbors = True
    # lyambda = 1
    # sigma = 1
    # verifier = np.array([[white, white, black], [white, white, black], [white, black, black]])

    img = Image.open(file_name).convert('L')
    image = np.asarray(img)
    ver_img = Image.open(verifier_name).convert('L')
    verifier = np.asarray(ver_img)

    print(f"Object pixels are: ")
    print(obj_pixels)
    print(f"Background pixels are: ")
    print(bg_pixels)
    print(f"Lambda={lyambda}, Sigma={sigma}")
    # print(image[0][0])
    # print(verifier[0][0])

    # 0. preparatory stage
    m = image.shape[0]
    n = image.shape[1]

    # 1. build graph by image
    graph = graph_by_image(m, n, is_four_neighbors)
    print("Graph was created")
    # print(graph)
    # print("=================")
    # print()

    # 2. get weights for graph
    get_weights(graph, image, obj_pixels, bg_pixels, lyambda, sigma)
    print("Weights were got")
    # print(graph)
    # print("=============")
    # print()

    # 3. get min cut for graph
    obj, bg = get_min_cut(graph)
    print("Cut was got")
    # print(obj)
    # print(bg)

    # 4. get image by min cut
    img = get_splitting(m, n, obj)
    jpg = Image.fromarray(img)
    jpg.save('test.bmp')
    print("Splitting was got")

    # 5. get metrics
    tfm, tsm = get_metrics(img, verifier)  # the first metric, the second metric
    print(f"The first metric: {np.round(tfm, 3)}\nThe second metric: {np.round(tsm, 3)}")
    return (jpg)


def main():
    file_name = 'banana2-gr.jpg'
    verifier_name = 'banana2.bmp'
    obj_pixels = {(211, 61), (306, 215), (375, 151), (354, 307), (303, 397), (391, 458), (284, 538), (165, 538), (94, 542)}
    bg_pixels = {(16, 20), (63, 305), (53, 576), (246, 354), (379, 34), (8, 631), (473, 12), (471, 330), (452, 615), (439, 303)}
    is_four_neighbors = False
    lyambda = 1
    sigma = 0.01
    algorithm(file_name, verifier_name, obj_pixels, bg_pixels, is_four_neighbors, lyambda, sigma)


if __name__ == "__main__":
    # test()
    # print("=================")
    main()
