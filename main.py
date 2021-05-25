import algorithms as alg
import numpy as np
from PIL import Image


def main():
    # test data
    """white = 255
    black = 0
    image = np.array([[200, 180, 10], [180, 10, 0]])
    obj_pixels = {(0, 0)}
    bg_pixels = {(1, 1)}
    is_four_neighbors = True
    lyambda = 1
    sigma = 1
    verifier = np.array([[white, white, black], [white, black, black]])"""

    img = Image.open('test1.jpeg')
    image = np.asarray(img)
    obj_pixels = {(320, 360)}
    bg_pixels = {(1, 1)}
    is_four_neighbors = True
    lyambda = 1
    sigma = 1
    ver_img = Image.open('test2.bmp')
    verifier = np.asarray(ver_img)

    # 0. preparatory stage
    m = image.shape[0]
    n = image.shape[1]

    # 1. build graph by image
    graph = alg.graph_by_image(m, n, is_four_neighbors)
    print("Graph was created")

    # 2. get weights for graph
    alg.get_weights(graph, image, obj_pixels, bg_pixels, lyambda, sigma)
    print("Weights were got")

    # 3. get min cut for graph
    obj, bg = alg.get_min_cut(graph)
    print("Cut was got")

    # 4. get image by min cut
    img = alg.get_splitting(m, n, obj)
    jpg = Image.fromarray(img)
    jpg.save('test.jpeg')
    print("Splitting was got")

    # 5. get metrics
    tfm, tsm = alg.get_metrics(img, verifier)  # the first metric, the second metric
    print(f"The first metric: {tfm}\nThe second metric: {tsm}")


def test():
    img = Image.open('test1.jpeg')
    # print(img)
    data = np.asarray(img)
    print(type(data))
    print(data.shape)
    # print(data)
    img2 = Image.fromarray(data)
    print(type(img2))
    print(img2.mode)
    print(img2.size)
    # print(img2)
    img2.save('test2.jpeg')


if __name__ == "__main__":
    # test()
    # print("=================")
    main()
