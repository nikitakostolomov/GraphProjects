import pytest

import MaxFlowFIFO.main as mn


@pytest.mark.parametrize(
    ["path", "expected_result"],
    [
        (
            "TestFiles/TestDataFromTask",
            {(1, 2): 10000, (1, 3): 10000, (2, 3): 1, (3, 4): 10000, (2, 4): 10000},
        ),
        ("TestFiles/TestData1", {(1, 2): 10}),
    ],
)
def test_get_edges_and_throughput_from_file(path, expected_result):
    """
    :param path: путь до файла с данными для задачи
    :param expected_result: словарь, у которого ключ - это ребро (кортеж двух чисел например: (1, 2)),
    а значение - это его пропускная способность
    """
    graph = mn.Graph(path)
    actual_result = graph.get_edges_and_throughput_from_file()
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ["path", "expected_result"],
    [
        ("TestFiles/TestDataFromTask", (4, 5)),
        ("TestFiles/TestData1", (2, 1)),
    ],
)
def test_get_amount_of_vertex_and_edges_from_file(path, expected_result):
    """
    :param path: путь до файла с данными для задачи
    :param expected_result: кортеж, у которого на первом месте находится количество вершин, на втором - количество ребер
    """
    graph = mn.Graph(path)
    actual_result = graph.get_amount_of_vertex_and_edges_from_file()
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ["path", "source", "expected_result"],
    [
        (
            "TestFiles/TestDataFromTask",
            1,
            (
                {1: [4, 0], 2: [0, 10000], 3: [0, 10000], 4: [0, 0]},
                {
                    (1, 2): [10000, 0],
                    (2, 1): [-10000, 10000],
                    (1, 3): [10000, 0],
                    (3, 1): [-10000, 10000],
                    (2, 3): [0, 1],
                    (3, 2): [0, 0],
                    (3, 4): [0, 10000],
                    (4, 3): [0, 0],
                    (2, 4): [0, 10000],
                    (4, 2): [0, 0],
                },
            ),
        ),
        (
            "TestFiles/TestData1",
            1,
            ({1: [2, 0], 2: [0, 10]}, {(1, 2): [10, 0], (2, 1): [-10, 10]}),
        ),
    ],
)
def test_initialize_pre_flow(path, source, expected_result):
    """
    :param path: путь до файла с данными для задачи
    :param expected_result: кортеж из двух словарей:
        1) словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка
        2) словарь, у которого ключ - это ребро (кортеж двух чисел например: (1, 2)),
    а значение - это список потока (изначально нулевой), также ставим поток для обратных ребер и остаточной пропускной
    способности
    """
    graph = mn.Graph(path)
    actual_result = graph.initialize_pre_flow(source)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    [
        "edge",
        "vertex_and_height_excess",
        "edges_and_flow_residual_capacity",
        "expected_result",
    ],
    [
        (
            (1, 2),
            {1: [2, 10], 2: [0, 0]},
            {(1, 2): [0, 10], (2, 1): [0, 0]},
            ({1: [2, 0], 2: [0, 10]}, {(1, 2): [10, 0], (2, 1): [-10, 10]}),
        ),
        (
            (1, 2),
            {1: [2, 10], 2: [0, 0]},
            {(1, 2): [0, 5], (2, 1): [0, 0]},
            ({1: [2, 5], 2: [0, 5]}, {(1, 2): [5, 0], (2, 1): [-5, 5]}),
        ),
        (
            (1, 2),
            {1: [2, 5], 2: [0, 0]},
            {(1, 2): [0, 10], (2, 1): [0, 0]},
            ({1: [2, 0], 2: [0, 5]}, {(1, 2): [5, 5], (2, 1): [-5, 5]}),
        ),
    ],
)
def test_push(
    edge, vertex_and_height_excess, edges_and_flow_residual_capacity, expected_result
):
    """
    1) Если делаем push по ребру и у отдающей вершины избыток равен пропускной способности, то пропускная способность
    ребра становится равной нулю
    2) Если делаем push по ребру и у отдающей вершины избыток больше пропускной способности, то этот избыток в ней
    остается
    3) Если делаем push по ребру и у отдающей вершины избыток меньше пропускной способности, то пропускная
    способность ребра станет равна c(u, v) - e(u)
    :param edge: ребро (кортеж), по которому делаем проталкивание
    :param vertex_and_height_excess: словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка
    :param edges_and_flow_residual_capacity: edges_and_flow_residual_capacity: словарь, у которого ключ - это ребро (кортеж двух чисел например: (1, 2)),
    а значение - это список потока и остаточной пропускной способности
    :param expected_result: кортеж из двух словарей:
        1) словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка
        2) словарь, у которого ключ - это ребро (кортеж двух чисел например: (1, 2)),
    а значение - это список потока и остаточной пропускной способности
    """
    actual_result = mn.Graph.push(
        edge, vertex_and_height_excess, edges_and_flow_residual_capacity
    )
    assert actual_result == expected_result


@pytest.mark.parametrize(
    [
        "vertex",
        "vertex_and_height_excess",
        "edges_and_flow_residual_capacity",
        "expected_result",
    ],
    [
        (
            1,
            {1: [0, 5], 2: [1, 0]},
            {(1, 2): [5, 10], (2, 1): [0, 0]},
            {1: [2, 5], 2: [1, 0]},
        ),
        (
            1,
            {1: [0, 5], 2: [5, 0]},
            {(1, 2): [5, 10], (2, 1): [0, 0]},
            {1: [6, 5], 2: [5, 0]},
        ),
        (
            1,
            {1: [0, 5], 2: [1, 0], 3: [2, 0]},
            {(1, 2): [5, 10], (2, 1): [0, 0], (1, 3): [5, 10], (3, 1): [0, 0]},
            {1: [2, 5], 2: [1, 0], 3: [2, 0]},
        ),
    ],
)
def test_relabel(
    vertex, vertex_and_height_excess, edges_and_flow_residual_capacity, expected_result
):
    """
    :param vertex: вершина, которую нужно поднять
    :param vertex_and_height_excess: словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка
    :param edges_and_flow_residual_capacity: edges_and_flow_residual_capacity: словарь, у которого ключ - это ребро
    (кортеж двух чисел например: (1, 2)), а значение - это список потока и остаточной пропускной способности
    :param expected_result: словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка, но
    высота увеличена на единицу
    """
    actual_result = mn.Graph.relabel(
        vertex, vertex_and_height_excess, edges_and_flow_residual_capacity
    )
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ["vertex", "edges_and_flow_residual_capacity", "expected_result"],
    [
        (1, {(1, 2): [0, 10], (2, 1): [0, 0]}, [2]),
        (1, {(1, 2): [0, 10], (2, 1): [0, 0], (1, 3): [0, 10], (3, 1): [0, 0]}, [2, 3]),
        (1, {(1, 2): [10, 0], (2, 1): [-10, 10], (1, 3): [0, 10], (3, 1): [0, 0]}, [2, 3]),
    ],
)
def test_find_adjacent_vertices(
    vertex, edges_and_flow_residual_capacity, expected_result
):
    """
    :param vertex: вершина, для которой нужно найти смежные вершины
    :param edges_and_flow_residual_capacity: словарь, у которого ключ - это ребро (кортеж двух чисел например: (1, 2)),
    а значение - это список потока и остаточной пропускной способности
    :param expected_result: список смежных вершин, у ребер которых остаточная пропускная способность больше нуля
    """
    actual_result = mn.Graph.find_adjacent_vertices(vertex, edges_and_flow_residual_capacity)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ["source", "sink", "path", "expected_result"],
    [
        (1, 4, "TestFiles/TestDataFromTask", 20000),
        (1, 2, "TestFiles/TestData1", 10),
    ],
)
def test_push_relabel_max_flow(source, sink, path, expected_result):
    graph = mn.Graph(path)
    actual_result = graph.push_relabel_max_flow(source, sink)
    assert actual_result == expected_result
