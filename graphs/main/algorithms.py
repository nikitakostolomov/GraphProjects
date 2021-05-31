import numpy as np
import networkx as nx
from .maxflow import Graph
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
        bins_number = (len(intensity_list) + 1) if len(intensity_list) <= 255 else 255
        hist = np.histogram(intensity_list, bins_number, hist_range, density=True)
    return hist


def get_weights(graph, image, obj_pixels, bg_pixels, lyambda, sigma):
    m = image.shape[0]
    n = image.shape[1]
    n_links = graph[1]
    t_links = graph[2]

    exps = dict()
    sqrt2 = np.sqrt(2)
    sigma_coefficient = 2 * pow(sigma, 2)

    sum_of_bpq = dict()
    for p, q in list(n_links.keys()):
        p_div = p // n
        p_mod = p % n
        q_div = q // n
        q_mod = q % n

        ip = image[p_div][p_mod]
        iq = image[q_div][q_mod]

        intensity_diff = ip - iq
        if intensity_diff < 0:
            intensity_diff = -intensity_diff
        if intensity_diff not in exps:
            exps[intensity_diff] = np.exp(-pow(intensity_diff, 2) / sigma_coefficient)

        dist = sqrt2
        diff = p - q
        if diff == n or diff == -1 or diff == -n or diff == 1:
            dist = 1

        if exps[intensity_diff] > 0.0:
            n_links[(p, q)] = exps[intensity_diff] / dist
            sum_of_bpq[p] = sum_of_bpq.get(p, 0) + n_links[(p, q)]
        else:
            del n_links[(p, q)]
    k = 1.0 + max(sum_of_bpq.values())
    # print(k)
    hist_range = (0, 256)

    obj_hist = get_hist(image, obj_pixels, hist_range)
    bg_hist = get_hist(image, bg_pixels, hist_range)

    pr_obj_save = dict()
    pr_bg_save = dict()
    # print(obj_pixels)
    # print(bg_pixels)
    for p in range(m * n):
        if (p // n, p % n) in obj_pixels:
            # print(f"{p} in obj_pixels")
            t_links[(m * n, p)] = k
            del t_links[(p, m * n + 1)]
        elif (p // n, p % n) in bg_pixels:
            # print(f"{p} in bg_pixels")
            t_links[(p, m * n + 1)] = k
            del t_links[(m * n, p)]
        elif lyambda != 0.0:
            # print(f"{p} not in obj_pixels and bg_pixels")
            ip = image[p // n][p % n]

            if ip not in pr_obj_save:
                pr_obj_save[ip] = obj_hist[0][binary_search(obj_hist[1], ip)]
            if ip not in pr_bg_save:
                pr_bg_save[ip] = bg_hist[0][binary_search(bg_hist[1], ip)]

            pr_obj = pr_obj_save[ip]
            pr_bg = pr_bg_save[ip]

            pr_obj = pr_obj / (pr_obj + pr_bg)
            pr_bg = 1.0 - pr_obj
            r_obj = -np.log(pr_obj)
            r_bg = -np.log(pr_bg)
            # print(p, pr_obj, pr_bg, r_obj, r_bg)

            if r_bg != 0.0:
                t_links[(m * n, p)] = lyambda * r_bg
            else:
                del t_links[(m * n, p)]
            if r_obj != 0.0:
                t_links[(p, m * n + 1)] = lyambda * r_obj
            else:
                del t_links[(p, m * n + 1)]
        else:
            del t_links[(m * n, p)]
            del t_links[(p, m * n + 1)]

    return ((graph[0][0], len(graph[1]) + len(graph[2])), n_links, t_links), k


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


def graph_to_txt(edges, n):
    file = open("MyFile.txt", "w")
    file.write(f"{n} {len(edges)}\n")
    for i, j in edges:
        file.write(f"{i} {j} {edges[(i, j)]}\n")
    file.close()


def graph_to_string(edges, n):
    result = f"{n} {len(edges)}"
    for i, j in edges:
        result += f" {i} {j} {edges[(i, j)]}"
    return result


def string_to_graph(string):
    split_string = string.split(' ')
    sizes = (int(split_string[0]), int(split_string[1]))
    edges = dict()
    print(sizes)
    for i in range(0, sizes[1]):
        print(i, split_string[2 + 3 * i], split_string[2 + 3 * i + 1], split_string[2 + 3 * i + 2])
        edges[(int(split_string[2 + 3 * i]), int(split_string[2 + 3 * i + 1]))] = float(split_string[2 + 3 * i + 2])
    return sizes, edges


def convert_graph(graph):
    edges = {}
    for i, j in graph[1]:
        edges[(i + 1, j + 1)] = graph[1][(i, j)]
    for i, j in graph[2]:
        if i == graph[0][0] - 2:
            edges[(0, j + 1)] = graph[2][(i, j)]
        else:
            edges[(i + 1, j)] = graph[2][(i, j)]

    return edges


def convert_answer_back(obj, bg, n):
    ans_obj = []
    for i in obj:
        if i != 0:
            ans_obj.append(i - 1)
    ans_bg = []
    for i in bg:
        if i != n - 1:
            ans_bg.append(i - 1)
    return ans_obj, ans_bg


def standard_get_min_cut(graph):
    g = nx.DiGraph()
    for i, j in graph[1]:
        g.add_edge(i, j, capacity=graph[1][(i, j)])
    for i, j in graph[2]:
        g.add_edge(i, j, capacity=graph[2][(i, j)])

    cut_value, partition = nx.minimum_cut(g, graph[0][0] - 2, graph[0][0] - 1)
    reachable, non_reachable = partition
    print(f"Cut value is {np.round(cut_value, 3)}")

    return reachable, non_reachable


def get_min_cut(graph):
    # edges = convert_graph(graph)
    edges = {}
    for i, j in graph[1]:
        edges[(i, j)] = graph[1][(i, j)]
    for i, j in graph[2]:
        edges[(i, j)] = graph[2][(i, j)]
    # edges = graph[1]
    # print(graph)
    graph_to_txt(edges, graph[0][0])

    g = Graph(amount_of_vertex_and_edges=graph[0], edges_and_throughput=edges)
    print(f"Max flow: {g.push_relabel_max_flow(graph[0][0] - 2, graph[0][0] - 1)}")
    g.get_min_cut()

    # result = convert_answer_back(g.min_cut_object, g.min_cut_background, graph[0][0])
    result = (g.min_cut_object, g.min_cut_background)

    return result, g.get_edges_and_res_cap_dict()


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

    g = Graph(amount_of_vertex_and_edges=graph[0], edges_and_throughput=edges)
    g.push_relabel_max_flow()
    g.get_min_cut()

    result = (g.min_cut_object, g.min_cut_background)

    return result, g.get_edges_and_res_cap_dict()


def start_algorithm(file_name, verifier_name, obj_pixels, bg_pixels, is_four_neighbors, lyambda, sigma):
    print(f"\n\n\n\n====================\nALGORITHM START\n=================\n")

    img = Image.open(file_name)
    print(f"Image mode: {img.mode}")
    img.convert('L')
    image = np.asarray(img)

    ver_img = Image.open(verifier_name)
    print(f"Verifier mode: {ver_img.mode}")
    ver_img.convert('L')
    verifier = np.asarray(ver_img)

    print(f"Object pixels are: ")
    print(obj_pixels)
    print(f"Background pixels are: ")
    print(bg_pixels)
    print(f"Lambda={lyambda}, Sigma={sigma}")

    # print(file_name, verifier_name)

    # print(f"seventh row of image: {image[7]}")
    # print(f"seventh row of verifier: {verifier[7]}")

    # print(image[0][0])
    # print(verifier[7][7])

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
    graph, k = get_weights(graph, image, obj_pixels, bg_pixels, lyambda, sigma)
    print("Weights were got")
    # print(graph)
    # print("=============")
    # print()

    # 3. get min cut for graph
    (obj, bg), graph_to_save = get_min_cut(graph)
    # obj1, bg1 = standard_get_min_cut(graph)

    # print(f"Nikita's obj: {obj}")
    # print(f"Standard's obj: {obj1}")

    # 4. get image by min cut
    img = get_splitting(m, n, obj)
    jpg = Image.fromarray(img)
    jpg.save('test.bmp')
    print("Splitting was got")

    # 5. get metrics
    tfm, tsm = get_metrics(img, verifier)  # the first metric, the second metric
    print(f"The first metric: {np.round(tfm, 3)}\nThe second metric: {np.round(tsm, 3)}")

    return graph_to_string(graph_to_save[1], graph_to_save[0][0]), k, jpg, tfm, tsm
    # print(f"Graph to save: {graph_to_save}")
    # graph_string = graph_to_string(graph_to_save[1], graph_to_save[0][0], graph_to_save[0][1])
    # print(f"Got string: {graph_string}")
    # graph_from_string = string_to_graph(graph_string)
    # print(f"Got graph back: {graph_from_string}")
    # return graph_to_string(graph_to_save[1], graph_to_save[0][0]), k, jpg, tfm, tsm
    # return jpg


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
    return graph_to_string(graph_to_save[1], graph_to_save[0][0]), jpg, tfm, tsm
