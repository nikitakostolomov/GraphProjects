def graph_by_image(m, n, s, t, is_four_neighbors=True):
    if s == t:
        raise Exception("Pixel 's' cannot be equal to pixel 't'!")

    edges = dict()
    a = 1  # initial weight of an edge
    s_pix = s[0] * n + s[1]
    t_pix = t[0] * n + t[1]

    # ===============
    # adding n-links
    # ===============

    # case (0, 0)
    u = 0 * n + 0
    edges.update({(u, u + 1): a, (u, u + n): a})
    if not is_four_neighbors:
        edges.update({(u, u + n + 1): a})

    # case (0, j), 0 < j < n - 1
    for j in range(1, n - 1):
        u = 0 * n + j
        edges.update({(u, u - 1): a, (u, u + 1): a, (u, u + n): a})
        if not is_four_neighbors:
            edges.update({(u, u + n - 1): a, (u, u + n + 1): a})

    # case (0, n - 1)
    u = 0 * n + n - 1
    edges.update({(u, u - 1): a, (u, u + n): a})
    if not is_four_neighbors:
        edges.update({(u, u + n - 1): a})

    # case (i, 0), 0 < i < m - 1
    for i in range(1, m - 1):
        u = i * n + 0
        edges.update({(u, u - n): a, (u, u + 1): a, (u, u + n): a})
        if not is_four_neighbors:
            edges.update({(u, u - n + 1): a, (u, u + n + 1): a})

    # case (i, j), 0 < i < m - 1, 0 < j < n - 1
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            u = i * n + j
            edges.update({(u, u - n): a, (u, u - 1): a, (u, u + 1): a, (u, u + n): a})
            if not is_four_neighbors:
                edges.update({(u, u - n - 1): a, (u, u - n + 1): a, (u, u + n - 1): a, (u, u + n + 1): a})

    # case (i, n - 1), 0 < i < m - 1
    for i in range(1, m - 1):
        u = i * n + n - 1
        edges.update({(u, u - n): a, (u, u - 1): a, (u, u + n): a})
        if not is_four_neighbors:
            edges.update({(u, u - n - 1): a, (u, u + n - 1): a})

    # case (m - 1, 0)
    u = (m - 1) * n + 0
    edges.update({(u, u - n): a, (u, u + 1): a})
    if not is_four_neighbors:
        edges.update({(u, u - n + 1): a})

    # case (m - 1, j), 0 < j < n - 1
    for j in range(1, n - 1):
        u = (m - 1) * n + j
        edges.update({(u, u - n): a, (u, u - 1): a, (u, u + 1): a})
        if not is_four_neighbors:
            edges.update({(u, u - n - 1): a, (u, u - n + 1): a})

    # case (m - 1, n - 1)
    u = (m - 1) * n + n - 1
    edges.update({(u, u - n): a, (u, u - 1): a})
    if not is_four_neighbors:
        edges.update({(u, u - n - 1): a})

    # ===============
    # adding t-links
    # ===============

    for v in range(n * m):
        if v != s_pix:
            edges.update({(s_pix, v): a})
        if v != t_pix:
            edges.update({(v, t_pix): a})
    return (m * n, len(edges)), edges


def get_weights(graph, image):
    return 0


def get_splitting(m, n, obj, bg):
    return 1
