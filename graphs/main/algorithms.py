import numpy as np
from math import ceil
from .newmaxflow import push_relabel_max_flow
# from .maxflow import Graph
# import networkx as nx

white = 255


def graph_by_image(image, obj_pixels, bg_pixels, lyambda, sigma, is_four_neighbors, group_by,
                   precision, intensity_obj_key):
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
        if intensity_obj_key == 2:
            if _x <= _y:
                return 0
            else:
                return _x - _y
        elif intensity_obj_key == 1:
            if _x >= _y:
                return 0
            else:
                return _y - _x
        else:
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

    k = 2 + max(sum_of_bpq)

    # ===============
    # adding t-links
    # ===============

    replace_inf = k - 1
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
        if _blocks:
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
                    r_obj = multiplier * round(-np.log(pr_obj) * lyambda, precision)
                    if r_obj > replace_inf:
                        r_obj = replace_inf

                if pr_bg == 0.0:
                    r_bg = replace_inf
                else:
                    r_bg = multiplier * round(-np.log(pr_bg) * lyambda, precision)
                    if r_bg > replace_inf:
                        r_bg = replace_inf

                if r_bg != 0.0:
                    edges[(s, p)] = int(r_bg)
                if r_obj != 0.0:
                    edges[(p, t)] = int(r_obj)

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


# def old_get_min_cut(graph, need_to_print_value=True, need_to_print_edges=False):
#     s = 0
#     t = graph[0][0] - 1
#
#     g = Graph(amount_of_vertex_and_edges=graph[0], edges_and_throughput=graph[1])
#     max_flow = g.push_relabel_max_flow(s, t)
#     obj, bg, cut = g.get_min_cut()
#
#     if need_to_print_value:
#         print(f"\nMax flow: {max_flow}\n")
#     if need_to_print_edges:
#         print(f"Minimal cut: {cut}\n")
#
#     return (obj, bg), g.get_edges_and_res_cap_dict()


def get_min_cut(graph, need_to_print_cut_value=True, need_to_print_cut_edges=False):
    s = 0
    t = graph[0][0] - 1

    (obj, bg), graph_to_save, min_cut_value, min_cut_edges = push_relabel_max_flow(graph, s, t, need_to_print_cut_edges)

    if need_to_print_cut_value:
        print(f"\nMax flow: {min_cut_value}\n")
    if need_to_print_cut_edges:
        print(f"Minimal cut: {min_cut_edges}\n")

    return (obj, bg), graph_to_save


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
