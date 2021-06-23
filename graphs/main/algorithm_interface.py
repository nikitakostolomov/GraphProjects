from PIL import Image
import uuid
from time import time
import numpy as np
from .algorithms import graph_by_image, get_metrics, get_min_cut, get_splitting, improve_result

default_precision = 3
default_blocking = (4, 4)
default_intensity_object_key = 0


def open_images(file_name, verifier_name, needed_to_print=True):
    img = Image.open(file_name).convert('L')
    if needed_to_print:
        print(f"Image mode: {img.mode}")
    image = np.asarray(img)

    ver_img = Image.open(verifier_name).convert('L')
    if needed_to_print:
        print(f"Verifier mode: {ver_img.mode}\n")
    verifier = np.asarray(ver_img)

    print(image[0][0])
    print(verifier[0][0])

    return image, verifier


# if intensity_obj_key == 2 then object is brighter than background, if ... == 1 then the opposite
# if ... == 0 then do not use this feature
def start_algorithm(file_name, verifier_name, obj_pixels, bg_pixels, is_four_neighbors, lyambda, sigma,
                    group_by=default_blocking, precision=default_precision,
                    intensity_obj_key=default_intensity_object_key):
    print(f"\n\n\n\n====================\nALGORITHM START\n=================\n")

    image, verifier = open_images(file_name, verifier_name)

    # print(f"Object pixels are: {obj_pixels}")
    # print(f"\nBackground pixels are: {bg_pixels}")
    print(f"\nLambda={lyambda}, Sigma={sigma}\n")

    # print(f"Seventh row of image: {image[61][61]};\nSeventh row of verifier: {verifier[61][61]}\n")
    # print(f"{image[181][301]} {image[301][301]}\n")

    # 0. preparatory stage
    m = image.shape[0]
    n = image.shape[1]

    # 1. build graph by image
    graph, k = graph_by_image(image, obj_pixels, bg_pixels, lyambda, sigma, is_four_neighbors, group_by,
                              precision, intensity_obj_key)
    print("Graph was created\n")
    # print(graph)
    # print("=================")
    # print()

    graph_to_txt(graph, "original_graph.txt")

    # 2. get min cut for graph
    # start = time()
    # (old_obj, old_bg), graph_to_save_old = old_get_min_cut(graph)
    # end = time()
    # print(f"Old min cut was got in {end - start}\n")
    # # print(f"Old obj is {old_obj}\nOld bg is {old_bg}\n")

    start = time()
    (obj, bg), graph_to_save = get_min_cut(graph)
    end = time()
    print(f"Min cut was got in {end - start}\n")
    # print(f"Obj is {obj}\nBg is {bg}\n")

    graph_to_txt(graph_to_save, "graph_to_save.txt")

    # obj1, bg1 = standard_get_min_cut(graph)
    # print(f"Nikita's obj: {obj}")
    # print(f"Standard's obj: {obj1}")

    # 3. get image by min cut
    result = get_splitting(m, n, obj, group_by)
    result_pic = Image.fromarray(result)
    print("Splitting was got\n")

    # 5. get metrics
    tfm, tsm = get_metrics(result, verifier)  # the first metric, the second metric
    print(f"The first metric: {np.round(tfm, 3)}\nThe second metric: {np.round(tsm, 3)}")

    print(f"\n====================\nALGORITHM END\n=================\n\n")

    return graph_to_string(graph_to_save), k, result_pic, tfm, tsm
    # print(f"Graph to save: {graph_to_save}")
    # graph_string = graph_to_string(graph_to_save[1], graph_to_save[0][0], graph_to_save[0][1])
    # print(f"Got string: {graph_string}")
    # graph_from_string = string_to_graph(graph_string)
    # print(f"Got graph back: {graph_from_string}")
    # return graph_to_string(graph_to_save[1], graph_to_save[0][0]), k, jpg, tfm, tsm
    # return jpg


def start_improving(file_name, verifier_name, graph, obj_pixels_to_add, bg_pixels_to_add, k,
                    group_by=default_blocking):
    print(f"\n\n\n\n====================\nRESULT IMPROVING START\n=================\n")

    image, verifier = open_images(file_name, verifier_name)

    # print(f"Object pixels to add: {obj_pixels_to_add}\n")
    # print(f"Background pixels to add: {bg_pixels_to_add}\n")

    # 0. preparatory stage
    m = image.shape[0]
    n = image.shape[1]

    pixels = dict()
    pixels['add'] = dict()
    pixels['add']['obj'] = obj_pixels_to_add
    pixels['add']['bg'] = bg_pixels_to_add

    # 1. get new weights and use them to get new min cut
    graph = string_to_graph(graph)
    graph_to_txt(graph, "loaded_from_save_graph.txt")
    (obj, bg), graph_to_save = improve_result(graph, n, group_by, pixels, k)
    print("Cut was got")

    graph_to_txt(graph_to_save, "graph_to_save_after_improving.txt")

    # 2. get image by min cut
    result = get_splitting(m, n, obj, group_by)
    result_pic = Image.fromarray(result)
    print("Splitting was got")

    # 3. get metrics
    tfm, tsm = get_metrics(result, verifier)  # the first metric, the second metric
    print(f"The first metric: {np.round(tfm, 3)}\nThe second metric: {np.round(tsm, 3)}")

    print(f"\n====================\nRESULT IMPROVING END\n=================\n\n")

    return graph_to_string(graph_to_save), result_pic, tfm, tsm


def graph_to_txt(graph, filename=None):
    v = graph[0][0]
    e = graph[0][1]
    edges = graph[1]

    if e != len(edges):
        raise Exception(f"Number of edges: {e} is not equal to length of the dict of edges: {len(edges)}")

    if not filename:
        file = open(f"MyFile_{uuid.uuid4().hex[-5:-1]}.txt", "w")
    else:
        file = open(filename, "w")

    file.write(f"{v} {e}\n")
    for i, j in edges:
        file.write(f"{i + 1} {j + 1} {edges[(i, j)]}\n")
    file.close()


def graph_to_string(graph):
    v = graph[0][0]
    e = graph[0][1]
    edges = graph[1]

    if e != len(edges):
        raise Exception(f"Number of edges: {e} is not equal to length of the dict of edges: {len(edges)}")

    result = f"{v} {e}"
    for i, j in edges:
        result += f" {i} {j} {edges[(i, j)]}"
    return result


def string_to_graph(string):
    split_string = string.split(' ')
    sizes = (int(split_string[0]), int(split_string[1]))
    edges = dict()
    for i in range(0, sizes[1]):
        edges[(int(split_string[2 + 3 * i]), int(split_string[2 + 3 * i + 1]))] = int(split_string[2 + 3 * i + 2])

    if sizes[1] != len(edges):
        raise Exception(f"Number of edges: {sizes[1]} is not equal to length of the dict of edges: {len(edges)}")
    return sizes, edges
