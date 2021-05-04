import pytest

import MaxFlowFIFO.main as mn


@pytest.mark.parametrize(
    ["path", "expected_result"],
    [
        (
            "TestFiles/TestDataFromTask",
            (
                (4, 5),
                {(1, 2): 10000, (1, 3): 10000, (2, 3): 1, (3, 4): 10000, (2, 4): 10000},
            ),
        ),
        ("TestFiles/TestData1", ((2, 1), {(1, 2): 10})),
    ],
)
def test_get_data_from_file(path, expected_result):
    """
    :param path: путь до файла с данными для задачи
    :param expected_result: кортеж:
        1) Кортеж из количества вершин и ребер
        2) Словарь ребер и их пропускных способностей.
    """
    graph = mn.Graph(path)
    actual_result = graph.get_data_from_file()
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
    graph.initialize_pre_flow(source)
    actual_result = (
        graph.vertex_and_height_excess,
        graph.edges_and_flow_residual_capacity,
    )
    assert actual_result == expected_result


@pytest.mark.parametrize(
    [
        "path",
        "edge",
        "expected_result",
    ],
    [
        (
            "TestFiles/TestDataFromTask",
            (2, 3),
            {
                (1, 2): [10000, 0],
                (2, 1): [-10000, 10000],
                (1, 3): [10000, 0],
                (3, 1): [-10000, 10000],
                (2, 3): [1, 0],
                (3, 2): [-1, 1],
                (3, 4): [0, 10000],
                (4, 3): [0, 0],
                (2, 4): [0, 10000],
                (4, 2): [0, 0],
            },
        ),
        (
            "TestFiles/TestDataFromTask",
            (2, 4),
            {
                (1, 2): [10000, 0],
                (2, 1): [-10000, 10000],
                (1, 3): [10000, 0],
                (3, 1): [-10000, 10000],
                (2, 3): [0, 1],
                (3, 2): [0, 0],
                (3, 4): [0, 10000],
                (4, 3): [0, 0],
                (2, 4): [10000, 0],
                (4, 2): [-10000, 10000],
            },
        ),
    ],
)
def test_push(path, edge, expected_result):
    """
    1) Если делаем push по ребру и у отдающей вершины избыток равен пропускной способности, то пропускная способность
    ребра становится равной нулю
    2) Если делаем push по ребру и у отдающей вершины избыток больше пропускной способности, то этот избыток в ней
    остается
    3) Если делаем push по ребру и у отдающей вершины избыток меньше пропускной способности, то пропускная
    способность ребра станет равна c(u, v) - e(u)
    :param path: путь до файла с задачей
    :param edge: ребро (кортеж), по которому делаем проталкивание
    :param expected_result: кортеж из двух словарей:
        1) словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка
        2) словарь, у которого ключ - это ребро (кортеж двух чисел например: (1, 2)),
    а значение - это список потока и остаточной пропускной способности
    """
    graph = mn.Graph(path)
    graph.initialize_pre_flow(1)
    graph.push(edge)
    actual_result = graph.edges_and_flow_residual_capacity
    assert actual_result == expected_result


@pytest.mark.parametrize(
    [
        "path",
        "vertex",
        "expected_result",
    ],
    [
        (
            "TestFiles/TestDataFromTask",
            3,
            {1: [4, 0], 2: [0, 10000], 3: [1, 10000], 4: [0, 0]},
        ),
        (
            "TestFiles/TestDataFromTask",
            2,
            {1: [4, 0], 2: [1, 10000], 3: [0, 10000], 4: [0, 0]},
        ),
        (
            "TestFiles/TestDataFromTask",
            1,
            {1: [4, 0], 2: [0, 10000], 3: [0, 10000], 4: [0, 0]},
        ),
    ],
)
def test_relabel(path, vertex, expected_result):
    """
    :param path: путь до файла с задачей
    :param vertex: вершина, которую нужно поднять
    :param expected_result: словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка, но
    высота увеличена на единицу
    """
    graph = mn.Graph(path)
    graph.initialize_pre_flow(1)
    graph.relabel(vertex)
    actual_result = graph.vertex_and_height_excess
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ["path", "vertex", "expected_result"],
    [
        ("TestFiles/TestDataFromTask", 1, [2, 3]),
        ("TestFiles/TestDataFromTask", 2, [1, 3, 4]),
        ("TestFiles/TestDataFromTask", 3, [1, 2, 4]),
    ],
)
def test_find_adjacent_vertices(path, vertex, expected_result):
    """
    :param path: путь до файла с задачей
    :param vertex: вершина, для которой нужно найти смежные вершины
    :param expected_result: список смежных вершин, у ребер которых остаточная пропускная способность больше нуля
    """
    graph = mn.Graph(path)
    graph.initialize_pre_flow(1)
    actual_result = graph.find_adjacent_vertices(vertex)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ["path", "source", "destination", "expected_result"],
    [
        ("TestFiles/TestDataFromTask", 2, 4, 1),
        ("TestFiles/TestDataFromTask", 3, 4, 1),
        ("TestFiles/TestDataFromTask", 4, 1, False),
        ("TestFiles/TestDataFromTask", 1, 4, False),
    ],
)
def test_bfs(path, source, destination, expected_result):
    """
    Если сток или исток недостижимы, то вернется False, иначе расстояние до пункта назначения
    :param path: путь до файла с задачей
    :param source: вершина, из которой начинается bfs
    :param destination: сток или исток, если сток недостижим
    :param expected_result: расстояние от вершины до пункта назначения
    """
    graph = mn.Graph(path)
    graph.initialize_pre_flow(1)
    actual_result = graph.bfs(source, destination)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ["path", "glob_value", "expected_result"],
    [
        ("TestFiles/TestDataFromTask", 1, 1),
        ("TestFiles/TestDataFromTask", 5, 5),
    ],
)
def test_try_global_relabeling(path, glob_value, expected_result):
    """
    :param path: путь до файла с задачей
    :param glob_value: число повторений, через которое можно запускать global relabeling (по умолчанию - 1)
    :param expected_result:
    """
    graph = mn.Graph(path)
    graph.push_relabel_max_flow(1, 4, glob_value)
    actual_result = graph.glob_rel_value
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ["source", "sink", "glob_value", "path", "expected_result"],
    [
        (1, 4, 1, "TestFiles/TestDataFromTask", 20000),
        (1, 2, 1, "TestFiles/TestData1", 10),
    ],
)
def test_push_relabel_max_flow(source, sink, path, glob_value, expected_result):
    """
    :param source: источник
    :param sink: сток
    :param path: путь до файла с задачей
    :param glob_value: число, global_relabeling будет запускаться через каждые glob_rel_value элементарных
        операций (push, relabel)
    :param expected_result: избыток стока
    """
    graph = mn.Graph(path)
    actual_result = graph.push_relabel_max_flow(source, sink, glob_value)
    assert actual_result == expected_result
