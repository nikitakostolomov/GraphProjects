from queue import Queue


def bfs(ver, s, using_back_edges=False):
    distances = [-1] * len(ver)
    distances[s] = 0
    queue = Queue(maxsize=len(ver))
    queue.put(s)
    while not queue.empty():
        u = queue.get()
        if using_back_edges:
            for v in ver[u]['adj_from']:
                if distances[v] == -1:
                    queue.put(v)
                    distances[v] = distances[u] + 1
        else:
            for e in ver[u]['adj_to']:
                if distances[e['v']] == -1:
                    queue.put(e['v'])
                    distances[e['v']] = distances[u] + 1
    return distances


def initialize_preflow(ver, s, t, queue):
    for edge in ver[s]['adj_to']:
        push(ver, s, edge, True)
        ver[edge['v']]['adj_from'].remove(s)
        if edge['v'] != t and edge['v'] != s:
            queue.append(edge['v'])

    ver[s]['adj_to'] = []
    ver[s]['h'] = len(ver)


def global_relabeling(ver, s, t):
    distances = bfs(ver, t, True)
    visited = [False] * len(ver)
    need_to_second_bfs = False
    for u in range(len(ver)):
        if distances[u] > ver[u]['h']:
            ver[u]['h'] = distances[u]
            visited[u] = True
        elif distances[u] == -1 and u != s:
            need_to_second_bfs = True
    if need_to_second_bfs:
        distances = bfs(ver, s, True)
        for u in range(len(ver)):
            if not visited[u] and distances[u] > ver[u]['h']:
                ver[u]['h'] = distances[u]
                visited[u] = True


def push(ver, u, e, is_initial=False):
    v = e['v']

    need_to_del_edge = False
    if is_initial:
        flow = e['c']
    else:
        flow = min(ver[u]['e'], e['c'])
        if flow == e['c']:
            need_to_del_edge = True

    if v not in ver[u]['adj_from']:
        ver[v]['adj_to'].append(
            {'v': u, 'f': 0, 'c': 0})
        ver[u]['adj_from'].append(v)
    rev_e = next(x for x in ver[v]['adj_to'] if x['v'] == u)

    ver[u]['e'] = ver[u]['e'] - flow
    ver[v]['e'] = ver[v]['e'] + flow
    e['f'] = e['f'] + flow
    e['c'] = e['c'] - flow
    rev_e['f'] = rev_e['f'] - flow
    rev_e['c'] = rev_e['c'] + flow

    return need_to_del_edge


def relabel(ver, u):
    ver[u]['h'] = 1 + ver[min(ver[u]['adj_to'], key=lambda e: ver[e['v']]['h'])['v']]['h']


def push_relabel_max_flow(graph, s, t, print_cut_edges=False):
    m = graph[0][1]

    # stats
    # counter_push = 0
    # counter_relabel = 0
    # counter_global_relabeling = 1

    edges = graph[1]

    ver = []
    queue = []

    for i in range(graph[0][0]):
        ver.append({'h': 0, 'e': 0, 'adj_to': [], 'adj_from': []})

    for i, j in edges:
        ver[i]['adj_to'].append({'v': j, 'f': 0, 'c': edges[(i, j)]})
        ver[j]['adj_from'].append(i)

    initialize_preflow(ver, s, t, queue)
    global_relabeling(ver, s, t)

    counter = 0

    while queue:
        if counter >= m:
            # counter_global_relabeling = counter_global_relabeling + 1
            global_relabeling(ver, s, t)
            counter = 0
        u = queue.pop(0)
        to_del = set()
        for e in ver[u]['adj_to']:
            v = e['v']
            if ver[u]['h'] == ver[v]['h'] + 1 and e['c'] > 0:
                # counter_push = counter_push + 1
                counter = counter + 1
                if push(ver, u, e):
                    to_del.add(v)
                if v != s and v != t and v not in queue:
                    queue.append(v)
                if ver[u]['e'] == 0:
                    break
        if to_del:
            for e in ver[u]['adj_to']:
                if e['v'] in to_del:
                    ver[e['v']]['adj_from'].remove(u)
            ver[u]['adj_to'] = [e for e in ver[u]['adj_to'] if not e['v'] in to_del]
        if ver[u]['e'] > 0:
            # counter_relabel = counter_relabel + 1
            counter = counter + 1
            relabel(ver, u)
            queue.append(u)

    distances = bfs(ver, s)
    obj = set()
    bg = set()

    for i in range(len(ver)):
        if distances[i] == -1:
            bg.add(i)
        else:
            obj.add(i)

    edges_to_save = dict()
    for u in range(len(ver)):
        for edge in ver[u]['adj_to']:
            v = edge['v']
            c = edge['c']
            edges_to_save[(u, v)] = c

    # print(f"\nNumber of pushing: {counter_push}")
    # print(f"Number of relabeling: {counter_relabel}")
    # print(f"Number of global relabeling: {counter_global_relabeling}\n")

    if print_cut_edges:
        cut = set()
        for edge in edges:
            if edge[0] in obj and edge[1] in bg:
                cut.add(edge)
    else:
        cut = None

    return (obj, bg), ((len(ver), len(edges_to_save)), edges_to_save), ver[t]['e'], cut
