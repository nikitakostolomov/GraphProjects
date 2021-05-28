import numpy as np
from scipy.spatial import distance
import networkx as nx
from .maxflow import Graph as min_cut
from PIL import Image

white = 255


def graph_by_image(m, n, is_four_neighbors=True):
    n_links = dict()
    a = 1  # initial weight of an edge

    # ===============
    # adding n-links
    # ===============

    # case (0, 0)
    u = 0 * n + 0
    n_links.update({(u, u + 1): a, (u, u + n): a})
    if not is_four_neighbors:
        n_links.update({(u, u + n + 1): a})

    # case (0, j), 0 < j < n - 1
    for j in range(1, n - 1):
        u = 0 * n + j
        n_links.update({(u, u - 1): a, (u, u + 1): a, (u, u + n): a})
        if not is_four_neighbors:
            n_links.update({(u, u + n - 1): a, (u, u + n + 1): a})

    # case (0, n - 1)
    u = 0 * n + n - 1
    n_links.update({(u, u - 1): a, (u, u + n): a})
    if not is_four_neighbors:
        n_links.update({(u, u + n - 1): a})

    # case (i, 0), 0 < i < m - 1
    for i in range(1, m - 1):
        u = i * n + 0
        n_links.update({(u, u - n): a, (u, u + 1): a, (u, u + n): a})
        if not is_four_neighbors:
            n_links.update({(u, u - n + 1): a, (u, u + n + 1): a})

    # case (i, j), 0 < i < m - 1, 0 < j < n - 1
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            u = i * n + j
            n_links.update({(u, u - n): a, (u, u - 1): a, (u, u + 1): a, (u, u + n): a})
            if not is_four_neighbors:
                n_links.update({(u, u - n - 1): a, (u, u - n + 1): a, (u, u + n - 1): a, (u, u + n + 1): a})

    # case (i, n - 1), 0 < i < m - 1
    for i in range(1, m - 1):
        u = i * n + n - 1
        n_links.update({(u, u - n): a, (u, u - 1): a, (u, u + n): a})
        if not is_four_neighbors:
            n_links.update({(u, u - n - 1): a, (u, u + n - 1): a})

    # case (m - 1, 0)
    u = (m - 1) * n + 0
    n_links.update({(u, u - n): a, (u, u + 1): a})
    if not is_four_neighbors:
        n_links.update({(u, u - n + 1): a})

    # case (m - 1, j), 0 < j < n - 1
    for j in range(1, n - 1):
        u = (m - 1) * n + j
        n_links.update({(u, u - n): a, (u, u - 1): a, (u, u + 1): a})
        if not is_four_neighbors:
            n_links.update({(u, u - n - 1): a, (u, u - n + 1): a})

    # case (m - 1, n - 1)
    u = (m - 1) * n + n - 1
    n_links.update({(u, u - n): a, (u, u - 1): a})
    if not is_four_neighbors:
        n_links.update({(u, u - n - 1): a})

    # ===============
    # adding t-links
    # ===============

    t_links = dict()
    for v in range(n * m):
        t_links.update({(n * m, v): a})
        t_links.update({(v, n * m + 1): a})
    return (m * n + 2, len(n_links) + len(t_links)), n_links, t_links


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


def get_hist(image, pixels, hist_range):
    hist = np.histogram([0], 1, hist_range, density=True)
    if pixels:
        intensity_list = [image[i][j] for i, j in pixels]
        bins_number = len(intensity_list) if len(intensity_list) <= 255 else 255
        hist = np.histogram(intensity_list, bins_number, hist_range, density=True)
    return hist


def get_weights(graph, image, obj_pixels, bg_pixels, lyambda, sigma):
    m = image.shape[0]
    n = image.shape[1]
    n_links = graph[1]
    t_links = graph[2]
    sum_of_bpq = dict()
    for p, q in list(n_links.keys()):
        ip = image[p // n][p % n]
        iq = image[q // n][q % n]
        dist = distance.euclidean((p // n, p % n), (q // n, q % n))
        n_links[(p, q)] = 1 if ip <= iq else np.exp(-pow((ip - iq), 2) / (2 * pow(sigma, 2))) / dist
        sum_of_bpq[p] = sum_of_bpq.get(p, 0) + n_links[(p, q)]
    k = 1 + sum_of_bpq[max(sum_of_bpq, key=lambda i: sum_of_bpq[i])]
    # print(k)
    hist_range = (0, 256)

    obj_hist = get_hist(image, obj_pixels, hist_range)
    bg_hist = get_hist(image, bg_pixels, hist_range)

    # print(obj_pixels)
    # print(bg_pixels)
    for p in range(m * n):
        if (p // n, p % n) in obj_pixels:
            # print(f"{p} in obj_pixels")
            t_links[(m * n, p)] = k
            t_links[(p, m * n + 1)] = 0
        elif (p // n, p % n) in bg_pixels:
            # print(f"{p} in bg_pixels")
            t_links[(m * n, p)] = 0
            t_links[(p, m * n + 1)] = k
        else:
            # print(f"{p} not in obj_pixels and bg_pixels")
            ip = image[p // n][p % n]
            pr_obj = obj_hist[0][binary_search(obj_hist[1], ip)]
            pr_bg = bg_hist[0][binary_search(bg_hist[1], ip)]

            pr_obj = pr_obj / (pr_obj + pr_bg)
            pr_bg = 1.0 - pr_obj
            r_obj = -np.log(pr_obj)
            r_bg = -np.log(pr_bg)
            # print(p, pr_obj, pr_bg, r_obj, r_bg)

            t_links[(m * n, p)] = lyambda * r_bg
            t_links[(p, m * n + 1)] = lyambda * r_obj
    return k


def get_splitting(m, n, obj):
    img = np.zeros((m, n), dtype=int)
    for x in obj:
        if x == m * n:
            # print("Ola!")
            pass
        elif x == m * n + 1:
            # print("WRONG!")
            pass
        else:
            img[x // n][x % n] = white
    return img.astype(np.uint8)


def convert_graph(graph):
    edges = {}
    for i, j in graph[1]:
        edges[(i + 2, j + 2)] = graph[1][(i, j)]
    for i, j in graph[2]:
        if i == graph[0][0] - 2:
            edges[(1, j + 2)] = graph[2][(i, j)]
        else:
            edges[(i + 2, j + 1)] = graph[2][(i, j)]

    return edges


def graph_to_txt(edges, n, k):
    file = open("MyFile.txt", "w")
    file.write(f"{n} {k}\n")
    for i, j in edges:
        file.write(f"{i} {j} {edges[(i, j)]}\n")
    file.close()


def convert_answer_back(obj, bg, n):
    ans_obj = []
    for i in obj:
        if i != 1:
            ans_obj.append(i - 2)
    ans_bg = []
    for i in bg:
        if i != n:
            ans_bg.append(i - 2)
    return ans_obj, ans_bg


def standard_get_min_cut(graph):
    G = nx.DiGraph()
    for i, j in graph[1]:
        G.add_edge(i, j, capacity=graph[1][(i, j)])
    for i, j in graph[2]:
        G.add_edge(i, j, capacity=graph[2][(i, j)])

    cut_value, partition = nx.minimum_cut(G, graph[0][0] - 2, graph[0][0] - 1)
    reachable, non_reachable = partition
    print(f"Cut value is {np.round(cut_value, 3)}")

    return reachable, non_reachable


def get_min_cut(graph):
    edges = convert_graph(graph)
    # graph_to_txt(edges, graph[0][0], graph[0][1])

    g = min_cut(amount_of_vertex_and_edges=graph[0], edges_and_throughput=edges)
    g.push_relabel_max_flow()
    g.get_min_cut()

    return convert_answer_back(g.min_cut_object, g.min_cut_background, graph[0][0]), g


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

    return counter / (m * n), caps / cups


def improve_result(graph, n, pixels, k):
    edges = graph[1]
    s = 1
    t = graph[0][0]

    for i, j in pixels['add']['obj']:
        p = i * n + j + 2
        rp_bg = edges.get((s, p), 0)
        rp_obj = edges.get((p, t), 0)
        edges[(s, p)] = rp_bg + rp_obj + k
        edges[(p, t)] = rp_bg + rp_obj

    for i, j in pixels['add']['bg']:
        p = i * n + j + 2
        rp_bg = edges.get((s, p), 0)
        rp_obj = edges.get((p, t), 0)
        edges[(s, p)] = rp_bg + rp_obj
        edges[(p, t)] = rp_bg + rp_obj + k

    g = min_cut(amount_of_vertex_and_edges=graph[0], edges_and_throughput=edges)
    g.push_relabel_max_flow()
    g.get_min_cut()

    return convert_answer_back(g.min_cut_object, g.min_cut_background, graph[0][0]), g


def start_algorithm(file_name, verifier_name, obj_pixels, bg_pixels, is_four_neighbors, lyambda, sigma):
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
    k = get_weights(graph, image, obj_pixels, bg_pixels, lyambda, sigma)
    print("Weights were got")
    # print(graph)
    # print("=============")
    # print()

    # 3. get min cut for graph
    (obj, bg), graph_to_save = get_min_cut(graph)
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
    return graph_to_save, k, jpg, tfm, tsm


def improve_algorithm(file_name, verifier_name, graph, obj_pixels_to_add, bg_pixels_to_add, k):
    img = Image.open(file_name).convert('L')
    image = np.asarray(img)
    ver_img = Image.open(verifier_name).convert('L')
    verifier = np.asarray(ver_img)

    # 0. preparatory stage
    m = image.shape[0]
    n = image.shape[1]
    pixels = dict()
    pixels['add'] = dict()
    pixels['add']['obj'] = obj_pixels_to_add
    pixels['add']['bg'] = bg_pixels_to_add

    # 1. get new weights and use them to get new min cut
    (obj, bg), graph_to_save = improve_result(graph, n, pixels, k)
    print("Cut was got")

    # 2. get image by min cut
    img = get_splitting(m, n, obj)
    jpg = Image.fromarray(img)
    jpg.save('test.bmp')
    print("Splitting was got")

    # 3. get metrics
    tfm, tsm = get_metrics(img, verifier)  # the first metric, the second metric
    print(f"The first metric: {np.round(tfm, 3)}\nThe second metric: {np.round(tsm, 3)}")
    return graph_to_save, jpg, tfm, tsm
