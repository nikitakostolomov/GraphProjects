import numpy as np
from scipy.spatial import distance

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
        n_links[(p, q)] = 1 if ip <= iq else np.exp(-pow((ip - iq), 2) / 2 / pow(sigma, 2)) / dist
        sum_of_bpq[p] = sum_of_bpq.get(p, 0) + n_links[(p, q)]
    k = 1 + sum_of_bpq[max(sum_of_bpq, key=lambda i: sum_of_bpq[i])]
    hist_range = (0, 256)

    obj_hist = np.histogram([0], 1, hist_range, density=True)
    if obj_pixels:
        obj_intensity_list = [image[i][j] for i, j in obj_pixels]
        obj_bins_number = len(obj_intensity_list) if len(obj_intensity_list) <= 255 else 255
        obj_hist = np.histogram(obj_intensity_list, obj_bins_number, hist_range, density=True)

    bg_hist = np.histogram([0], 1, hist_range, density=True)
    if bg_pixels:
        bg_intensity_list = [image[i][j] for i, j in bg_pixels]
        bg_bins_number = len(bg_intensity_list) if len(bg_intensity_list) <= 255 else 255
        bg_hist = np.histogram(bg_intensity_list, bg_bins_number, hist_range, density=True)

    for p in range(m * n):
        if p in obj_pixels:
            t_links[(m * n, p)] = k
            t_links[(p, m * n + 1)] = 0
        elif p in bg_pixels:
            t_links[(m * n, p)] = 0
            t_links[(p, m * n + 1)] = k
        else:
            ip = image[p // n][p % n]
            pr_obj = obj_hist[0][binary_search(obj_hist[1], ip)]
            pr_bg = bg_hist[0][binary_search(bg_hist[1], ip)]

            pr_obj = pr_obj / (pr_obj + pr_bg)
            pr_bg = 1.0 - pr_obj
            r_obj = -np.log(pr_obj)
            r_bg = -np.log(pr_bg)

            t_links[(m * n, p)] = lyambda * r_bg
            t_links[(p, m * n + 1)] = lyambda * r_obj


def get_splitting(m, n, obj):
    img = np.zeros((m, n), dtype=int)
    for x in obj:
        img[x // n][x % n] = white
    return img.astype(np.uint8)


def get_min_cut(graph):
    return [i for i in range(graph[0][0] // 2)], [i for i in range(graph[0][0] // 2, graph[0][0])]


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
            if np.array_equal(img[i][j], white):
                if np.array_equal(verifier[i][j], white):
                    caps = caps + 1
                cups = cups + 1
            elif np.array_equal(verifier[i][j], white):
                cups = cups + 1
            if np.array_equal(img[i][j], verifier[i][j]):
                counter = counter + 1

    return counter / (m * n), caps / cups
