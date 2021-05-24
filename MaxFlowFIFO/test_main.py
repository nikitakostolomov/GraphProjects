import pytest

import MaxFlowFIFO.main as mn


@pytest.mark.parametrize(
    [
        "path",
        "source",
        "amount_of_vertex_and_edges",
        "edges_and_throughput",
        "expected_result",
    ],
    [
        (
            "MaxFlow-tests/test_1.txt",
            1,
            (3, 2),
            {(1, 2): 5, (1, 3): 10},
            (
                {1: [3, -15], 2: [0, 5], 3: [0, 10]},
                {(1, 2): [5, 0], (1, 3): [10, 0], (2, 1): [-5, 5], (3, 1): [-10, 10]},
            ),
        ),
        (
            "MaxFlow-tests/test_1.txt",
            1,
            (3, 3),
            {(1, 2): 10, (2, 1): 5, (1, 3): 10},
            (
                {1: [3, -20], 2: [0, 10], 3: [0, 10]},
                {
                    (1, 2): [10, 0],
                    (1, 3): [10, 0],
                    (2, 1): [-10, 15],
                    (3, 1): [-10, 10],
                },
            ),
        ),
    ],
)
def test_initialize_pre_flow(
    path, source, amount_of_vertex_and_edges, edges_and_throughput, expected_result
):
    graph = mn.Graph(path)
    graph.initialize_pre_flow(source, amount_of_vertex_and_edges, edges_and_throughput)
    actual_result = (
        graph.vertex_and_height_excess,
        graph.edges_and_flow_residual_capacity,
    )
    assert actual_result == expected_result


@pytest.mark.parametrize(
    [
        "path",
        "edge",
        "vertex_and_height_excess",
        "edges_and_flow_residual_capacity",
        "expected_result",
    ],
    [
        (
            "MaxFlow-tests/test_1.txt",
            (1, 2),
            {1: [1, 10], 2: [0, 0]},
            {(1, 2): [0, 10], (2, 1): [0, 0]},
            ({1: [1, 0], 2: [0, 10]}, {(1, 2): [10, 0], (2, 1): [-10, 10]}),
        ),
        (
            "MaxFlow-tests/test_1.txt",
            (1, 2),
            {1: [1, 10], 2: [0, 5]},
            {(1, 2): [0, 5], (2, 1): [0, 0]},
            ({1: [1, 5], 2: [0, 10]}, {(1, 2): [5, 0], (2, 1): [-5, 5]}),
        ),
        (
            "MaxFlow-tests/test_1.txt",
            (1, 2),
            {1: [1, 5], 2: [0, 5]},
            {(1, 2): [0, 10], (2, 1): [0, 0]},
            ({1: [1, 0], 2: [0, 10]}, {(1, 2): [5, 5], (2, 1): [-5, 5]}),
        ),
        (
            "MaxFlow-tests/test_1.txt",
            (1, 2),
            {1: [1, 5], 2: [0, 5]},
            {(1, 2): [0, 10], (2, 1): [0, 5]},
            ({1: [1, 0], 2: [0, 10]}, {(1, 2): [5, 5], (2, 1): [-5, 10]}),
        ),
        (
            "MaxFlow-tests/test_1.txt",
            (1, 2),
            {1: [1, 5], 2: [1, 5]},
            {(1, 2): [0, 10], (2, 1): [0, 5]},
            ({1: [1, 0], 2: [1, 10]}, {(1, 2): [5, 5], (2, 1): [-5, 10]}),
        ),
    ],
)
def test_push(
    path,
    edge,
    vertex_and_height_excess,
    edges_and_flow_residual_capacity,
    expected_result,
):
    graph = mn.Graph(path)
    graph.push(edge, vertex_and_height_excess, edges_and_flow_residual_capacity)
    actual_result = (
        graph.vertex_and_height_excess,
        graph.edges_and_flow_residual_capacity,
    )
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ["path", "vertex", "edges_and_flow_residual_capacity", "expected_result"],
    [
        ("MaxFlow-tests/test_1.txt", 1, {(1, 2): [0, 10], (1, 3): [0, 10]}, [2, 3]),
        (
            "MaxFlow-tests/test_1.txt",
            1,
            {(1, 2): [10, 0], (2, 1): [-10, 10], (1, 3): [0, 10], (3, 1): [0, 0]},
            [3],
        ),
        (
            "MaxFlow-tests/test_1.txt",
            1,
            {(1, 2): [10, 0], (2, 1): [-10, 10], (1, 3): [10, 0], (3, 1): [-10, 10]},
            [],
        ),
    ],
)
def test_find_adjacent_vertices(
    path, vertex, edges_and_flow_residual_capacity, expected_result
):
    graph = mn.Graph(path)
    actual_result = graph.find_adjacent_vertices(
        vertex, edges_and_flow_residual_capacity
    )
    assert actual_result == expected_result


@pytest.mark.parametrize(
    [
        "path",
        "vertex",
        "vertex_and_height_excess",
        "edges_and_flow_residual_capacity",
        "expected_result",
    ],
    [
        (
            "MaxFlow-tests/test_1.txt",
            2,
            {1: [3, -20], 2: [0, 10], 3: [0, 10]},
            {
                (1, 2): [0, 10],
                (2, 1): [-10, 10],
                (1, 3): [0, 10],
                (3, 1): [-10, 10],
                (2, 3): [0, 5],
                (3, 2): [0, 0],
            },
            (
                {1: [3, -20], 2: [1, 10], 3: [0, 10]},
                {
                    (1, 2): [0, 10],
                    (2, 1): [-10, 10],
                    (1, 3): [0, 10],
                    (3, 1): [-10, 10],
                    (2, 3): [0, 5],
                    (3, 2): [0, 0],
                },
            ),
        ),
        (
            "MaxFlow-tests/test_1.txt",
            2,
            {1: [3, -20], 2: [0, 10], 3: [1, 10]},
            {
                (1, 2): [0, 10],
                (2, 1): [-10, 10],
                (1, 3): [0, 10],
                (3, 1): [-10, 10],
                (2, 3): [0, 5],
                (3, 2): [0, 0],
            },
            (
                {1: [3, -20], 2: [2, 10], 3: [1, 10]},
                {
                    (1, 2): [0, 10],
                    (2, 1): [-10, 10],
                    (1, 3): [0, 10],
                    (3, 1): [-10, 10],
                    (2, 3): [0, 5],
                    (3, 2): [0, 0],
                },
            ),
        ),
        (
            "MaxFlow-tests/test_1.txt",
            2,
            {1: [3, -20], 2: [0, 10], 3: [0, 10]},
            {
                (1, 2): [0, 10],
                (2, 1): [-10, 10],
                (1, 3): [0, 10],
                (3, 1): [-10, 10],
                (2, 3): [5, 0],
                (3, 2): [-5, 5],
            },
            (
                {1: [3, -20], 2: [4, 10], 3: [0, 10]},
                {
                    (1, 2): [0, 10],
                    (2, 1): [-10, 10],
                    (1, 3): [0, 10],
                    (3, 1): [-10, 10],
                    (2, 3): [5, 0],
                    (3, 2): [-5, 5],
                },
            ),
        ),
    ],
)
def test_relabel(
    path,
    vertex,
    vertex_and_height_excess,
    edges_and_flow_residual_capacity,
    expected_result,
):
    graph = mn.Graph(path)
    graph.relabel(vertex, vertex_and_height_excess, edges_and_flow_residual_capacity)
    actual_result = (
        graph.vertex_and_height_excess,
        graph.edges_and_flow_residual_capacity,
    )
    assert actual_result == expected_result
