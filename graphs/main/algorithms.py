import numpy as np
# import networkx as nx
from .maxflow import Graph
from PIL import Image
import uuid
from math import ceil

white = 255
default_blocking = (6, 6)


def graph_by_image(image, obj_pixels, bg_pixels, lyambda, sigma, is_four_neighbors, group_by, precision):
    m = image.shape[0]
    n = image.shape[1]

    multiplier = pow(10, precision)

    ceil_m = int(ceil(m / group_by[0]))
    ceil_n = int(ceil(n / group_by[1]))

    edges = dict()

    # ===============
    # adding n-links
    # ===============

    sqrt2 = np.sqrt(2)
    sigma_coefficient = 2 * pow(sigma, 2)

    exps = dict()
    v = ceil_m * ceil_n + 2
    sum_of_bpq = np.zeros(v - 2, dtype=int)

    # compute weight to n_link
    def n_link_w(intensity_diff, dist):
        if intensity_diff not in exps:
            exps[intensity_diff] = np.exp(-pow(intensity_diff, 2) / sigma_coefficient)
        return exps[intensity_diff] / dist

    # absolute of difference
    def diff_abs(_x, _y):
        if _x >= _y:
            return _x - _y
        else:
            return _y - _x

    def block_intensity(_i, _j):
        block = image[_i * group_by[0]:(_i + 1) * group_by[0], _j * group_by[1]:(_j + 1) * group_by[1]]
        return int(np.round(block.mean()))

    for i in range(ceil_m):
        for j in range(ceil_n):
            right = j < ceil_n - 1
            up = i > 0
            left = j > 0
            down = i < ceil_m - 1

            w = [0, 0, 0, 0, 0, 0, 0, 0]  # right, right_up, ..., down, right_down

            w[0] = int(multiplier * round(n_link_w(diff_abs(block_intensity(i, j), block_intensity(i, j + 1)), 1.0),
                                          precision)) if right else 0
            w[2] = int(multiplier * round(n_link_w(diff_abs(block_intensity(i, j), block_intensity(i - 1, j)), 1.0),
                                          precision)) if up else 0
            w[4] = int(multiplier * round(n_link_w(diff_abs(block_intensity(i, j), block_intensity(i, j - 1)), 1.0),
                                          precision)) if left else 0
            w[6] = int(multiplier * round(n_link_w(diff_abs(block_intensity(i, j), block_intensity(i + 1, j)), 1.0),
                                          precision)) if down else 0

            if not is_four_neighbors:
                w[1] = int(multiplier * round(n_link_w(diff_abs(block_intensity(i, j), block_intensity(i - 1, j + 1)),
                                                       sqrt2), precision)) if right and up else 0
                w[3] = int(multiplier * round(n_link_w(diff_abs(block_intensity(i, j), block_intensity(i - 1, j - 1)),
                                                       sqrt2), precision)) if left and up else 0
                w[5] = int(multiplier * round(n_link_w(diff_abs(block_intensity(i, j), block_intensity(i + 1, j - 1)),
                                                       sqrt2), precision)) if left and down else 0
                w[7] = int(multiplier * round(n_link_w(diff_abs(block_intensity(i, j), block_intensity(i + 1, j + 1)),
                                                       sqrt2), precision)) if right and down else 0

            u = i * ceil_n + j + 1
            sum_of_bpq[u - 1] = sum_of_bpq[u - 1] + sum(w)
            if w[0] > 0:
                edges[(u, u + 1)] = w[0]
            if w[1] > 0:
                edges[(u - ceil_n, u + 1)] = w[1]
            if w[2] > 0:
                edges[(u - ceil_n, u)] = w[2]
            if w[3] > 0:
                edges[(u - ceil_n, u - 1)] = w[3]
            if w[4] > 0:
                edges[(u, u - 1)] = w[4]
            if w[5] > 0:
                edges[(u + ceil_n, u - 1)] = w[5]
            if w[6] > 0:
                edges[(u + ceil_n, u)] = w[6]
            if w[7] > 0:
                edges[(u + ceil_n, u + 1)] = w[7]

    k = 1 + max(sum_of_bpq)

    # ===============
    # adding t-links
    # ===============

    replace_inf = 1000.0
    hist_range = (0, 256)
    s = 0
    t = v - 1

    def binary_search(arr, el, start=0, end=None):
        if end is None:
            end = len(arr)
        if el <= arr[start]:
            return start
        if arr[start] == arr[end - 1]:
            return start
        if arr[start + (end - start) // 2] > el:
            return binary_search(arr, el, start, start + (end - start) // 2)
        else:
            return binary_search(arr, el, start + (end - start) // 2, end)

    def get_hist(_blocks):
        hist = np.histogram([0], 1, hist_range, density=True)
        if blocks:
            intensity_list = []
            for _i, _j in _blocks:
                block = image[_i * group_by[0]:(_i + 1) * group_by[0], _j * group_by[1]:(_j + 1) * group_by[1]]
                intensity_list.extend(np.reshape(block, block.size))
            bins_number = (len(intensity_list) + 1) if len(intensity_list) <= 255 else 255
            hist = np.histogram(intensity_list, bins_number, hist_range, density=True)
        return hist

    blocks = dict()
    obj_blocks = []
    bg_blocks = []

    for i, j in obj_pixels:
        u = (i // group_by[0], j // group_by[1])
        blocks[u] = blocks.get(u, 0) + 1
    for i, j in bg_pixels:
        u = (i // group_by[0], j // group_by[1])
        blocks[u] = blocks.get(u, 0) - 1

    for x in blocks:
        if blocks[x] > 0:
            obj_blocks.append(x)
        elif blocks[x] < 0:
            bg_blocks.append(x)

    obj_hist = get_hist(obj_blocks)
    bg_hist = get_hist(bg_blocks)

    pr_obj_save = dict()
    pr_bg_save = dict()

    for i in range(ceil_m):
        for j in range(ceil_n):
            p = i * ceil_n + j + 1
            if (i, j) in obj_blocks:
                edges[(s, p)] = int(round(k))
            elif (i, j) in bg_blocks:
                edges[(p, t)] = int(round(k))
            elif lyambda != 0.0:
                ip = block_intensity(i, j)

                if ip not in pr_obj_save:
                    pr_obj_save[ip] = obj_hist[0][binary_search(obj_hist[1], ip)]
                if ip not in pr_bg_save:
                    pr_bg_save[ip] = bg_hist[0][binary_search(bg_hist[1], ip)]

                pr_obj = pr_obj_save[ip]
                pr_bg = pr_bg_save[ip]

                if pr_obj == 0 and pr_bg == 0:
                    pr_obj = pr_bg = 0.5
                else:
                    pr_obj = pr_obj / (pr_obj + pr_bg)
                    pr_bg = 1.0 - pr_obj

                if pr_obj == 0.0:
                    r_obj = replace_inf
                else:
                    r_obj = -np.log(pr_obj)

                if pr_bg == 0.0:
                    r_bg = replace_inf
                else:
                    r_bg = -np.log(pr_bg)

                if r_bg != 0.0:
                    edges[(s, p)] = int(multiplier * round(lyambda * r_bg, precision))
                if r_obj != 0.0:
                    edges[(p, t)] = int(multiplier * round(lyambda * r_obj, precision))

    return ((v, len(edges)), edges), k


# def standard_get_min_cut(graph):
#     g = nx.DiGraph()
#     for i, j in graph[1]:
#         g.add_edge(i, j, capacity=graph[1][(i, j)])
#     for i, j in graph[2]:
#         g.add_edge(i, j, capacity=graph[2][(i, j)])
#
#     cut_value, partition = nx.minimum_cut(g, graph[0][0] - 2, graph[0][0] - 1)
#     reachable, non_reachable = partition
#     print(f"Cut value is {np.round(cut_value, 3)}")
#
#     return reachable, non_reachable


def get_min_cut(graph):
    s = 0
    t = graph[0][0] - 1

    g = Graph(amount_of_vertex_and_edges=graph[0], edges_and_throughput=graph[1])
    print(f"\nMax flow: {g.push_relabel_max_flow(s, t)}\n")
    obj, bg, cut = g.get_min_cut()

    print(f"Minimal cut: {cut}\n")

    return (obj, bg), g.get_edges_and_res_cap_dict()


def get_splitting(m, n, obj, group_by):
    ceil_m = int(ceil(m / group_by[0]))
    ceil_n = int(ceil(n / group_by[1]))
    v = ceil_m * ceil_n
    s = 0
    t = v - 1

    img = np.zeros((m, n), dtype=np.uint8)

    for p in obj:
        if p != s and p != t:
            i = (p - 1) // ceil_n
            j = (p - 1) % ceil_n
            img[i * group_by[0]:(i + 1) * group_by[0], j * group_by[1]:(j + 1) * group_by[1]] = white
    return img


def get_metrics(img, verifier):
    if img.shape != verifier.shape:
        raise Exception(f"Sizes of images are not equal! \nSize of our image: {img.shape} " +
                        f"\nSize of the verifier: {verifier.shape}")
    m = img.shape[0]
    n = img.shape[1]
    counter = 0  # correct pixels
    cups = 0  # number of pixels in the union of objects in two images
    caps = 0  # number of pixels in the intersection of objects in two images
    for i in range(m):
        for j in range(n):
            if img[i][j] == white:
                if verifier[i][j] == white:
                    caps = caps + 1
                cups = cups + 1
            elif verifier[i][j] == white:
                cups = cups + 1
            if img[i][j] == verifier[i][j]:
                counter = counter + 1

    tsm = 1
    if cups != 0:
        tsm = caps / cups

    return counter / (m * n), tsm


def open_images(file_name, verifier_name, needed_to_print=True):
    img = Image.open(file_name)
    if needed_to_print:
        print(f"Image mode: {img.mode}")
    img.convert('L')
    image = np.asarray(img)

    ver_img = Image.open(verifier_name)
    if needed_to_print:
        print(f"Verifier mode: {ver_img.mode}\n")
    ver_img.convert('L')
    verifier = np.asarray(ver_img)
    return image, verifier


def start_algorithm(file_name, verifier_name, obj_pixels, bg_pixels, is_four_neighbors, lyambda, sigma,
                    group_by=default_blocking, precision=3):
    print(f"\n\n\n\n====================\nALGORITHM START\n=================\n")

    image, verifier = open_images(file_name, verifier_name)

    print(f"Object pixels are: ")
    print(obj_pixels)
    print(f"\nBackground pixels are: ")
    print(bg_pixels)
    print(f"\nLambda={lyambda}, Sigma={sigma}\n")

    # print(f"Seventh row of image: {image[61][61]};\nSeventh row of verifier: {verifier[61][61]}\n")
    # print(f"{image[181][301]} {image[301][301]}\n")

    # 0. preparatory stage
    m = image.shape[0]
    n = image.shape[1]

    # 1. build graph by image
    graph, k = graph_by_image(image, obj_pixels, bg_pixels, lyambda, sigma, is_four_neighbors, group_by, precision)
    print("Graph was created\n")
    # print(graph)
    # print("=================")
    # print()

    graph_to_txt(graph, "original_graph.txt")

    # 2. get min cut for graph
    (obj, bg), graph_to_save = get_min_cut(graph)
    print("Min cut was got\n")

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


def improve_result(graph, n, group_by, pixels, k):
    edges = graph[1]

    ceil_n = int(ceil(n / group_by[1]))

    s = 0
    t = graph[0][0] - 1

    for i, j in pixels['add']['obj']:
        p = (i // group_by[0]) * ceil_n + (j // group_by[1]) + 1
        rp_bg = edges.get((s, p), 0)
        rp_obj = edges.get((p, t), 0)

        edges[(s, p)] = int(round(rp_bg + rp_obj + k))
        if rp_bg + rp_obj > 0:
            edges[(p, t)] = int(round(rp_bg + rp_obj))

    for i, j in pixels['add']['bg']:
        p = (i // group_by[0]) * ceil_n + (j // group_by[1]) + 1
        rp_bg = edges.get((s, p), 0)
        rp_obj = edges.get((p, t), 0)

        if rp_bg + rp_obj > 0:
            edges[(s, p)] = int(round(rp_bg + rp_obj))
        edges[(p, t)] = int(round(rp_bg + rp_obj + k))

    return get_min_cut(graph)


def improve_algorithm(file_name, verifier_name, graph, obj_pixels_to_add, bg_pixels_to_add, k,
                      group_by=default_blocking):
    print(f"\n\n\n\n====================\nRESULT IMPROVING START\n=================\n")

    image, verifier = open_images(file_name, verifier_name)

    print(f"Object pixels to add: {obj_pixels_to_add}\n")
    print(f"Background pixels to add: {bg_pixels_to_add}\n")

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
