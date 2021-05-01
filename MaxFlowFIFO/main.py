from collections import deque
from typing import Tuple


class Graph:
    """
    Класс для поиска максимального потока с оптимизацией FIFO и global relabeling.

    FIFO:
    Вершины с положительным избытком обрабатываются (просматриваются) в порядке first-in, first-out.
    Вершина извлекается из списка и делаются операции push пока это возможно. Новые вершины с избытком добавляются в
    конец списка. Если операции push для обрабатываемой вершины больше нельзя выполнить, и в вершине есть остаток, то
    выполняется операция подъем, а вершина добавляется в конец списка.

    Global relabeling:
    Расстояние до стока. h(v) ≤ d(v), где d(v) — расстояние от вершины v до стока по ненасыщенным ребрам или расстояние
    от вершины v до истока по ненасыщенным ребрам, если сток уже не достижим. d(v)  — корректная высотная функция.
    Посчитаем все d(v) (два bfs-а — от стока и от истока по обратным ребрам).
    Положим h(v) = d(v). Некоторые высоты увеличились, а предпоток остался корректным.
    Global Relabeling можно запускать через каждые m элементарных операций (push - relabel).
    """
    def __init__(self, path: str):
        """
        Конструктор класса
        :param path: путь до файла с задачей
        """
        self.path = path
        self.edges_and_throughput = self.get_edges_and_throughput_from_file()
        self.amount_of_vertex_and_edges = (
            self.get_amount_of_vertex_and_edges_from_file()
        )

    def get_edges_and_throughput_from_file(self) -> dict:
        """
        Принимаем путь до файла, читаем его и возвращаем словарь ребер и их пропускных способностей.
        :return: словарь, у которого ключ - это ребро (кортеж двух чисел например: (1, 2)), а значение - это его
        пропускная способность
        """
        edges_and_throughput = {}
        with open(self.path) as f:
            f.readline()
            for line in f:
                edge_throughput = list(map(int, line.split()))
                edges_and_throughput[
                    (edge_throughput[0], edge_throughput[1])
                ] = edge_throughput[2]

        return edges_and_throughput

    def get_amount_of_vertex_and_edges_from_file(self) -> tuple:
        """
        Принимаем путь до файла, читаем его и возвращаем кортеж из количества вершин и ребер.
        :return: кортеж, у которого на первом месте находится количество вершин, на втором - количество ребер
        """
        with open(self.path) as f:
            list_of_vertexes_edges = list(map(int, f.readline().strip().split()))
            vertexes_edges = list_of_vertexes_edges[0], list_of_vertexes_edges[1]

        return vertexes_edges

    def push_relabel_max_flow(self, source: int, sink: int) -> int:
        """
        Вершины с положительным избытком обрабатываются (просматриваются) в порядке first-in, first-out.
        Вершина извлекается из списка и делаются операции push пока это возможно. Новые вершины с избытком добавляются в
        конец списка (только если вершина не является стоком). Если операции push для обрабатываемой вершины больше
        нельзя выполнить, и в вершине есть остаток, то выполняется операция подъем, а вершина добавляется в конец списка
        (опять же только если вершина не является стоком).
        :param source: источник
        :param sink: сток
        :return: избыток стока
        """
        (
            vertex_and_height_excess,
            edges_and_flow_residual_capacity,
        ) = self.initialize_pre_flow(source)
        vertices_with_excess = []

        for (
            vertex,
            height_and_excess,
        ) in vertex_and_height_excess.items():  # ищем вершины с положительным избытком
            if height_and_excess[1] > 0:
                vertices_with_excess.append(vertex)

        while vertices_with_excess:  # пока список не пуст, цикл работает
            vertex_with_positive_excess = (
                vertices_with_excess.pop()
            )  # извлекаем последний элемент

            adjacent_vertices = self.find_adjacent_vertices(
                vertex_with_positive_excess, edges_and_flow_residual_capacity
            )  # находим смежные вершины

            adjacent_vertices_with_positive_residual_capacity = (
                []
            )  # пустой список смежных вершин, у ребер которых пропускная способность больше нуля
            for (
                vertex
            ) in (
                adjacent_vertices
            ):  # среди смежных вершин ищем те, у ребер которых пропускная способность больше нуля
                if (
                    edges_and_flow_residual_capacity[
                        vertex_with_positive_excess, vertex
                    ][1]
                    > 0
                ):
                    adjacent_vertices_with_positive_residual_capacity.append(vertex)

            for (
                vertex
            ) in (
                adjacent_vertices_with_positive_residual_capacity
            ):  # выполняем push пока можем
                if vertex_and_height_excess[vertex_with_positive_excess][1] > 0 and (
                    vertex_and_height_excess[vertex_with_positive_excess][0]
                    == vertex_and_height_excess[vertex][0] + 1
                ):  # push выполняем только если избыток вершины больше нуля и высота вершины больше на единицу,
                    # чем у смежной
                    (
                        vertex_and_height_excess,
                        edges_and_flow_residual_capacity,
                    ) = self.push(
                        (vertex_with_positive_excess, vertex),
                        vertex_and_height_excess,
                        edges_and_flow_residual_capacity,
                    )
                    if (
                        vertex != sink
                    ):  # если вершина не является стоком, добавляем ее в конец списка
                        vertices_with_excess.append(vertex)

            if (
                vertex_and_height_excess[vertex_with_positive_excess][1] > 0
                and vertex_with_positive_excess != sink
            ):  # relabel выполняем только если избыток вершины больше нуля и вершина не является стоком
                vertex_and_height_excess = self.relabel(
                    vertex_with_positive_excess,
                    vertex_and_height_excess,
                    edges_and_flow_residual_capacity,
                )
                vertices_with_excess.append(vertex_with_positive_excess)

        return vertex_and_height_excess[sink][1]

    @staticmethod
    def find_adjacent_vertices(
        vertex: int, edges_and_flow_residual_capacity: dict
    ) -> list:
        """
        Принимает вершину, для которой нужно найти смежные, также принимает словарь ребер и список их потока и
        пропускной способности. Если вершина смежная, то добавляем вершину в список смежных вершин.
        :param vertex: вершина, для которой нужно найти смежные вершины
        :param edges_and_flow_residual_capacity: словарь, у которого ключ - это ребро (кортеж двух чисел например:
        (1, 2)), а значение - это список потока и остаточной пропускной способности
        :return: список смежных вершин
        """
        adjacent_vertices = []
        for (
            edge,
            flow_and_residual_capacity,
        ) in edges_and_flow_residual_capacity.items():
            if edge[0] == vertex:
                adjacent_vertices.append(edge[1])

        return adjacent_vertices

    def initialize_pre_flow(self, source) -> Tuple[dict, dict]:
        """
        Для начала проинициализируем предпоток. Пропустим максимально возможный поток по рёбрам, инцидентным истоку,
        увеличив избыточный поток для каждой смежной с истоком вершиной на соответствующую величину.
        Все остальные потока не несут, следовательно, для вершин не смежных с истоком избыточный поток изначально
        будет нулевым. Также для всех вершин кроме истока установим высоту, равную нулю. Для истока устанавливаем высоту
        равную количеству вершин.
        :param source: число (вершина), которое будет являться источником
        :return: кортеж из двух словарей:
            1) словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка
            2) словарь, у которого ключ - это ребро (кортеж двух чисел например: (1, 2)),
        а значение - это список потока (изначально нулевой), (также ставим поток для обратных ребер) и остаточной
        пропускной способности
        """
        vertex_and_height_excess = (
            {}
        )  # словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка
        edges_and_flow_residual_capacity = (
            {}
        )  # словарь, у которого ключ - это ребро (кортеж двух чисел например: (1, 2)),
        # а значение - это список потока (изначально нулевой) и остаточной пропускной способности

        for vertex in range(1, self.amount_of_vertex_and_edges[0] + 1):
            if vertex != source:
                vertex_and_height_excess[vertex] = [
                    0,
                    0,
                ]  # для вершин, не являющихся источником, устанавливаем высоту и избыток равными нулю

            if vertex == source:
                vertex_and_height_excess[vertex] = [
                    self.amount_of_vertex_and_edges[0]
                ]  # для вершины, являющейся источником, устанавливаем высоту равную количеству вершин

        for (
            edge,
            throughput,
        ) in (
            self.edges_and_throughput.items()
        ):  # ставим нулевой поток для прямого и обратного ребра,
            # а также устанавливаем остаточную пропускную способность
            edges_and_flow_residual_capacity[edge] = [0, throughput]
            edges_and_flow_residual_capacity[edge[::-1]] = [0, 0]

        excess_in_source = 0
        for edge in self.edges_and_throughput:  # находим избыток источника
            if edge[0] == source:
                excess_in_source += self.edges_and_throughput[edge]

        vertex_and_height_excess[source].append(
            excess_in_source
        )  # источнику в словаре добавляем его избыток

        for (
            edge
        ) in (
            self.edges_and_throughput
        ):  # по ребрам, инцидентных источнику, пускаем поток
            if edge[0] == source:
                throughput = self.edges_and_throughput[
                    edge
                ]  # получили пропускную способность для ребра, инцидентого источнику
                edges_and_flow_residual_capacity[edge][
                    0
                ] = throughput  # пустили поток по ребру
                edges_and_flow_residual_capacity[edge][
                    1
                ] -= throughput  # уменьшили пропускную способность ребра
                edges_and_flow_residual_capacity[edge[::-1]][
                    0
                ] = -throughput  # пустили поток по обратному ребру
                edges_and_flow_residual_capacity[edge[::-1]][
                    1
                ] = throughput  # установили пропускную способность обратного ребра
                vertex_and_height_excess[edge[1]][
                    1
                ] = throughput  # увеличили избыток вершины, в которую идет поток
                vertex_and_height_excess[edge[0]][
                    1
                ] -= throughput  # уменьшили избыток источника

        return vertex_and_height_excess, edges_and_flow_residual_capacity

    @staticmethod
    def push(
        edge: tuple,
        vertex_and_height_excess: dict,
        edges_and_flow_residual_capacity: dict,
    ) -> Tuple[dict, dict]:
        """
        По ребру (u,v) пропускается максимально возможный поток, то есть минимум из избытка вершины u
        и остаточной пропускной способности ребра (u,v), вследствие чего избыток вершины u,
        остаточная пропускная способность ребра (u,v) и поток по обратному ребру (v,u) уменьшаются на величину потока,
        а избыток вершины v, поток по ребру (u,v) и остаточная пропускная способность обратного ребра (v,u)
        увеличиваются на эту же величину.
        :param edge: ребро (кортеж), по которому делаем проталкивание
        :param vertex_and_height_excess: словарь, у которого ключ - это вершина, а значение - это список ее высоты и
        избытка
        :param edges_and_flow_residual_capacity: словарь, у которого ключ - это ребро (кортеж двух чисел например:
        (1, 2)), а значение - это список потока и остаточной пропускной способности
        :return: кортеж из двух словарей:
            1) словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка
            2) словарь, у которого ключ - это ребро (кортеж двух чисел например: (1, 2)),
        а значение - это список потока и остаточной пропускной способности
        """
        excess = vertex_and_height_excess[edge[0]][1]  # избыток у исходящей вершины
        residual_capacity = edges_and_flow_residual_capacity[edge][
            1
        ]  # остаточная пропускная способность ребра

        max_flow = min(
            excess, residual_capacity
        )  # находим максимальный поток, который можем пустить по ребру

        edges_and_flow_residual_capacity[edge][0] += max_flow  # увеличиваем поток ребра
        edges_and_flow_residual_capacity[edge][
            1
        ] -= max_flow  # уменьшаем пропускную способность ребра
        edges_and_flow_residual_capacity[edge[::-1]][
            0
        ] = -edges_and_flow_residual_capacity[edge][
            0
        ]  # уменьшаем поток обратного ребра
        edges_and_flow_residual_capacity[edge[::-1]][
            1
        ] += max_flow  # увеличиваем пропусную способность обратного ребра
        vertex_and_height_excess[edge[0]][
            1
        ] -= max_flow  # уменьшаем избыток у исходящей вершины
        vertex_and_height_excess[edge[1]][
            1
        ] += max_flow  # увеличиваем избыток у принимающей вершины

        return vertex_and_height_excess, edges_and_flow_residual_capacity

    @staticmethod
    def relabel(
        vertex: int,
        vertex_and_height_excess: dict,
        edges_and_flow_residual_capacity: dict,
    ) -> dict:
        """
        Для переполненной вершины u применима операция подъёма, если все вершины, для которых в остаточной сети есть
        рёбра из u, расположены не ниже u. Следовательно, операцию проталкивания для вершины u произвести нельзя.
        В результате подъёма высота текущей вершины становится на единицу больше высоты самый низкой смежной вершины в
        остаточной сети, вследствие чего появляется как минимум одно ребро, по которому можно протолкнуть поток.
        :param vertex: вершина, которую нужно поднять
        :param vertex_and_height_excess: словарь, у которого ключ - это вершина, а значение - это список ее высоты и
        избытка
        :param edges_and_flow_residual_capacity: словарь, у которого ключ - это ребро (кортеж двух чисел например:
        (1, 2)), а значение - это список потока и остаточной пропускной способности
        :return: словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка, но высота
        увеличена на единицу
        """
        adjacent_vertices = {}  # словарь смежных вершин и их высот
        for edge in edges_and_flow_residual_capacity:
            if edge[0] == vertex and (
                edges_and_flow_residual_capacity[edge][0]
                - edges_and_flow_residual_capacity[edge][1]
                < 0
            ):  # первое условие для того, чтобы найти смежную вершину, воторое для того, чтобы поток не превышал
                # пропускную способность ребра(?)
                adjacent_vertex = edge[1]
                adjacent_vertices[adjacent_vertex] = vertex_and_height_excess[
                    adjacent_vertex
                ][0]

        vertex_with_min_height = min(
            adjacent_vertices, key=lambda x: adjacent_vertices[x]
        )  # находим вершину с минимальной высотой
        vertex_and_height_excess[vertex][0] += (
            adjacent_vertices[vertex_with_min_height] + 1
        )  # поднимаем нашу вершину

        return vertex_and_height_excess
