import collections
import os
import time
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
        amount_of_vertex_and_edges: кортеж из количества вершин и ребер
        edges_and_throughput: словарь, у которого ключ - это ребро (кортеж, например(1, 2)), а значение - это
            пропускная способность ребра
        _source: int, источник
        _sink: int, сток
        _vertex_and_height_excess: словарь, у которого ключ - это вершина, а значение - это список ее высоты и
            избытка
        _edges_and_flow_residual_capacity: словарь, у которого ключ - это ребро (кортеж двух чисел например:
            (1, 2)), а значение - это список потока и остаточной пропускной способности
        _amount_of_push_and_rel: количество повторений элементарных операций (push - relabel)
        _glob_rel_value: количество повторений, через которое можно запускать global relabeling (равно количеству ребер)
        _max_flow: значение маскимального потока
        _min_cut: значение минимального разреза
        _min_cut_object: вершины, находящиеся слева от разреза
        _min_cut_background: вершины, находящиеся справа от разреза
        """
        self.path = path
        (
            self.amount_of_vertex_and_edges,
            self.edges_and_throughput,
        ) = self.get_data_from_file()

        self.start = 1
        self.end = self.amount_of_vertex_and_edges[0]

        self._source = 0
        self._sink = 0

        self._vertex_and_height_excess = {}
        self._edges_and_flow_residual_capacity = {}

        self._amount_of_push_and_rel = 0
        self._glob_rel_value = 1

        self._max_flow = 0
        self._min_cut = 0
        self._min_cut_object = set()
        self._min_cut_background = set()

    def get_data_from_file(self) -> Tuple[tuple, dict]:
        """
        Принимаем путь до файла, читаем его и возвращаем кортеж:
            1) Кортеж из количества вершин и ребер
            2) Словарь ребер и их пропускных способностей.
        :return: кортеж из кортежа и словаря
        """
        edges_and_throughput = {}
        with open(self.path) as f:
            list_of_vertexes_edges = list(map(int, f.readline().strip().split()))
            vertexes_edges = list_of_vertexes_edges[0], list_of_vertexes_edges[1]
            for line in f:
                edge_throughput = list(map(int, line.split()))
                edges_and_throughput[
                    (edge_throughput[0], edge_throughput[1])
                ] = edge_throughput[2]

        return vertexes_edges, edges_and_throughput

    def initialize_pre_flow(self, *args) -> None:
        """
        Для начала проинициализируем предпоток. Пропустим максимально возможный поток по рёбрам, инцидентным истоку,
        увеличив избыточный поток для каждой смежной с истоком вершиной на соответствующую величину.
        Все остальные потока не несут, следовательно, для вершин не смежных с истоком избыточный поток изначально
        будет нулевым. Также для всех вершин кроме истока установим высоту, равную нулю. Для истока устанавливаем высоту
        равную количеству вершин.
        """
        if len(args) == 1:  # для тестов, сюда можно не смотреть
            self.source = args[0]
        elif len(args) > 1:  # для тестов, сюда можно не смотреть
            self.source = args[0]
            self.amount_of_vertex_and_edges = args[1]
            self.edges_and_throughput = args[2]

        for vertex in range(1, self.amount_of_vertex_and_edges[0] + 1):
            if vertex != self.source:
                self.vertex_and_height_excess[vertex] = [
                    0,
                    0,
                ]  # для вершин, не являющихся источником, устанавливаем высоту и избыток равными нулю
            else:
                self.vertex_and_height_excess[vertex] = [
                    self.amount_of_vertex_and_edges[0]
                ]  # для вершины, являющейся источником, устанавливаем высоту равную количеству вершин

        self.vertex_and_height_excess[self.source].append(
            0
        )  # источнику в словаре добавляем его избыток

        for (
            edge,
            throughput,
        ) in (
            self.edges_and_throughput.items()
        ):  # ставим нулевой поток для прямого и обратного ребра,
            # а также устанавливаем остаточную пропускную способность
            self.edges_and_flow_residual_capacity[edge] = [0, throughput]
            if edge[::-1] not in self.edges_and_throughput:
                self.edges_and_flow_residual_capacity[edge[::-1]] = [0, 0]

        for (
            edge
        ) in (
            self.edges_and_throughput
        ):  # по ребрам, инцидентных источнику, пускаем поток
            if edge[0] == self.source:
                throughput = self.edges_and_throughput[
                    edge
                ]  # получили пропускную способность для ребра, инцидентого источнику
                self.edges_and_flow_residual_capacity[edge][
                    0
                ] = throughput  # пустили поток по ребру
                self.edges_and_flow_residual_capacity[edge][
                    1
                ] -= throughput  # уменьшили пропускную способность ребра
                self.edges_and_flow_residual_capacity[edge[::-1]][
                    0
                ] = -throughput  # пустили поток по обратному ребру
                self.edges_and_flow_residual_capacity[edge[::-1]][
                    1
                ] += throughput  # установили пропускную способность обратного ребра
                self.vertex_and_height_excess[edge[1]][
                    1
                ] = throughput  # увеличили избыток вершины, в которую идет поток
                self.vertex_and_height_excess[edge[0]][
                    1
                ] -= throughput  # уменьшили избыток источника

    def push(self, edge: tuple, *args) -> None:
        """
        По ребру (u,v) пропускается максимально возможный поток, то есть минимум из избытка вершины u
        и остаточной пропускной способности ребра (u,v), вследствие чего избыток вершины u,
        остаточная пропускная способность ребра (u,v) и поток по обратному ребру (v,u) уменьшаются на величину потока,
        а избыток вершины v, поток по ребру (u,v) и остаточная пропускная способность обратного ребра (v,u)
        увеличиваются на эту же величину.
        :param edge: ребро (кортеж), по которому делаем проталкивание
        """
        if len(args) == 2:  # для тестов, сюда можно не смотреть
            self.vertex_and_height_excess = args[0]
            self.edges_and_flow_residual_capacity = args[1]

        excess = self.vertex_and_height_excess[edge[0]][
            1
        ]  # избыток у исходящей вершины
        residual_capacity = self.edges_and_flow_residual_capacity[edge][
            1
        ]  # остаточная пропускная способность ребра

        max_flow = min(
            excess, residual_capacity
        )  # находим максимальный поток, который можем пустить по ребру

        self.edges_and_flow_residual_capacity[edge][
            0
        ] += max_flow  # увеличиваем поток ребра
        self.edges_and_flow_residual_capacity[edge][
            1
        ] -= max_flow  # уменьшаем пропускную способность ребра
        self.edges_and_flow_residual_capacity[edge[::-1]][
            0
        ] = -self.edges_and_flow_residual_capacity[edge][
            0
        ]  # уменьшаем поток обратного ребра
        self.edges_and_flow_residual_capacity[edge[::-1]][
            1
        ] += max_flow  # увеличиваем пропусную способность обратного ребра
        self.vertex_and_height_excess[edge[0]][
            1
        ] -= max_flow  # уменьшаем избыток у исходящей вершины
        self.vertex_and_height_excess[edge[1]][
            1
        ] += max_flow  # увеличиваем избыток у принимающей вершины

    def relabel(self, vertex: int, *args) -> None:
        """
        Для переполненной вершины u применима операция подъёма, если все вершины, для которых в остаточной сети есть
        рёбра из u, расположены не ниже u. Следовательно, операцию проталкивания для вершины u произвести нельзя.
        В результате подъёма высота текущей вершины становится на единицу больше высоты самый низкой смежной вершины в
        остаточной сети, вследствие чего появляется как минимум одно ребро, по которому можно протолкнуть поток.
        :param vertex: вершина, которую нужно поднять
        """
        if len(args) == 2:  # для тестов, сюда можно не смотреть
            self.vertex_and_height_excess = args[0]
            self.edges_and_flow_residual_capacity = args[1]

        adjacent_vertices_and_height = {}  # словарь смежных вершин и их высот
        adjacent_vertices = self.find_adjacent_vertices(vertex)  # нашли смежные вершины

        for adj in adjacent_vertices:
            edge = (vertex, adj)
            if (
                self.edges_and_flow_residual_capacity[edge][0]
                < self.edges_and_flow_residual_capacity[edge][1]
            ):  # если поток ребра меньше его пропускной способности, то высоту смежной вершины добавляем в словарь
                adjacent_vertices_and_height[adj] = self.vertex_and_height_excess[adj][
                    0
                ]

        if adjacent_vertices_and_height:
            vertex_with_min_height = min(
                adjacent_vertices_and_height,
                key=lambda x: adjacent_vertices_and_height[x],
            )  # находим вершину с минимальной высотой
            self.vertex_and_height_excess[vertex][0] = (
                adjacent_vertices_and_height[vertex_with_min_height] + 1
            )  # поднимаем нашу вершину

    def find_adjacent_vertices(self, vertex: int, *args) -> list:
        """
        Принимает вершину, для которой нужно найти смежные. Если вершина смежная, то добавляем вершину в список смежных
        вершин.
        :param vertex: вершина, для которой нужно найти смежные вершины
        :return: список смежных вершин
        """
        if len(args) == 1:  # для тестов, сюда можно не смотреть
            self.edges_and_flow_residual_capacity = args[0]

        adjacent_vertices = []  # пустой список смежных вершин
        for (
            edge,
            flow_and_residual_capacity,
        ) in self.edges_and_flow_residual_capacity.items():
            if (
                edge[0] == vertex and flow_and_residual_capacity[1] > 0
            ):  # если первая вершина ребра сопадает с переданной вершиной и пропускная способность ребра больше нуля,
                # то вторая вершина ребра является смежной
                adjacent_vertices.append(edge[1])

        return adjacent_vertices

    def push_relabel_max_flow(self, source: int = 0, sink: int = 0) -> int:
        """
        Вершины с положительным избытком обрабатываются (просматриваются) в порядке first-in, first-out.
        Вершина извлекается из списка и делаются операции push пока это возможно. Новые вершины с избытком добавляются в
        конец списка (только если вершина не является стоком). Если операции push для обрабатываемой вершины больше
        нельзя выполнить, и в вершине есть остаток, то выполняется операция подъем, а вершина добавляется в конец списка
        (опять же только если вершина не является стоком).
        Global relabeling можно запускать через каждые m элементарных операций (push - relabel). Здесь glob_rel_value -
        это m.
        :param source: исток, по умолчанию вершиной будет число 1, можно выбрать другую
        :param sink: сток, по умолчанию вершиной будет первое число из файла с задачей , можно выбрать другую
        :return: избыток стока
        """
        if source == 0:
            self.source = self.start

        if sink == 0:
            self.sink = self.end

        self.glob_rel_value = self.amount_of_vertex_and_edges[
            1
        ]  # количество повторений, через которое можно запускать global relabeling (равно количеству ребер)

        self.initialize_pre_flow()  # инициализируем предпоток

        vertices_with_excess = (
            collections.deque()
        )  # пустая очередь вершин с положительным избытком

        for (
            vertex,
            height_and_excess,
        ) in (
            self.vertex_and_height_excess.items()
        ):  # ищем вершины с положительным избытком
            if height_and_excess[1] > 0:
                vertices_with_excess.append(vertex)

        self.global_relabeling()  # инициализируем высоты вершин

        while vertices_with_excess:  # пока список не пуст, цикл работает

            vertex_with_positive_excess = (
                vertices_with_excess.popleft()
            )  # извлекаем первый элемент

            adjacent_vertices = self.find_adjacent_vertices(
                vertex_with_positive_excess
            )  # находим смежные вершины

            check_push = True  # если True, значит мы не выполнили push, и вершину можно будет поднять

            for vertex in adjacent_vertices:  # выполняем push пока можем
                if self.vertex_and_height_excess[vertex_with_positive_excess][
                    1
                ] > 0 and (
                    self.vertex_and_height_excess[vertex_with_positive_excess][0]
                    == self.vertex_and_height_excess[vertex][0] + 1
                ):  # push выполняем только если избыток вершины больше нуля и высота вершины больше на 1, чем у смежной
                    self.push(
                        (vertex_with_positive_excess, vertex),
                    )
                    check_push = (
                        False  # выполнили push, поэтому поднимать вершину не будем
                    )

                    if (
                        vertex != self.sink and vertex not in vertices_with_excess
                    ):  # если вершина не является стоком и ее еще нет в очереди, добавляем ее в конец очереди
                        vertices_with_excess.append(vertex)

            if (
                self.vertex_and_height_excess[vertex_with_positive_excess][1] > 0
                and vertex_with_positive_excess != self.sink
            ):  # если в вершине остался избыток, то добавляем ее в конец очереди
                vertices_with_excess.append(vertex_with_positive_excess)

            if (
                check_push
                and self.vertex_and_height_excess[vertex_with_positive_excess][1] > 0
                and vertex_with_positive_excess != self.sink
                and vertex_with_positive_excess != self.source
            ):  # relabel выполняем, если мы не сделали push, избыток вершины больше нуля и вершина не является стоком
                self.relabel(vertex_with_positive_excess)
                if (
                    vertex_with_positive_excess not in vertices_with_excess
                ):  # если вершины еще нет в очереди, добавляем ее в конец
                    vertices_with_excess.append(vertex_with_positive_excess)

            # self.try_global_relabeling()

        self.max_flow = self.vertex_and_height_excess[self.sink][1]

        return self.max_flow

    def bfs(self, source: int, destination: int = -1) -> int or False or set:
        """
        Начинаем обход графа с вершины (source) и идем, пока не дойдем до пункта назначения (destination), также считаем
        расстояния от вершины до пункта назначения. Далее возвращаем расстояние до пункта назнаечния.
        Если пункт назначения не достижим, то возвращаем False.
        :param source: вершина, из которой начинается bfs
        :param destination: сток или исток, если сток недостижим
        :return: расстояние от вершины до пункта назначения
        """
        visited, queue = set(), collections.deque(
            [source]
        )  # посещенные вершины - множество, смежные с ними попадают
        # в очередь
        if destination != -1:
            bfs_with_destination = True  # значит bfs используется для global relabeling
        else:
            bfs_with_destination = (
                False  # значит bfs используется для нахождения min cut
            )

        vertices_dist = {}  # пустой словарь вершин и расстояний

        if bfs_with_destination:
            vertices_dist = {
                vertex: 1_000_000 for vertex in self.vertex_and_height_excess
            }  # словарь вершин и расстояний

            vertices_dist[
                source
            ] = 0  # расстояние от вершины до нее самой ставим равным нулю

        visited.add(source)

        while queue:
            vertex = queue.popleft()
            neighbours = self.find_adjacent_vertices(vertex)

            for neighbour in neighbours:
                res_cap = self.edges_and_flow_residual_capacity[vertex, neighbour][1]

                if bfs_with_destination:
                    flow = self.edges_and_flow_residual_capacity[vertex, neighbour][0]

                    if (
                        neighbour not in visited and flow < res_cap
                    ):  # если вершина еще не посещена и ребро не насыщено
                        visited.add(neighbour)  # то вершину добавляем в посещенные
                        vertices_dist[neighbour] = vertices_dist[vertex] + 1
                        queue.append(neighbour)

                        if neighbour == destination:
                            return vertices_dist[neighbour]
                else:
                    if (
                        neighbour not in visited and res_cap > 0
                    ):  # если вершина еще не посещена и пропускная способность ребра больше нуля
                        visited.add(neighbour)  # то вершину добавляем в посещенные
                        queue.append(neighbour)

        if bfs_with_destination:
            return False
        else:
            return visited

    def get_min_cut(self) -> Tuple[set, set]:
        """
        Находит минимальный разрез следующим способом: пытаемся добраться до всех вершин из истока в остаточной сети
        с помощью bfs, все найденные вершины будут слева от разреза, остальные справа. Минимальным разрезом будет сумма
        пропускных способностей ребер, соединяющих вершины слева и справа от разреза.
        """
        self.min_cut_object = self.bfs(
            self.source
        )  # ищем вершины, до которых можно добраться из истока

        # for edge in self.edges_and_throughput:
        #     if (
        #         edge[0] in self.min_cut_object
        #         and edge[1] not in self.min_cut_object
        #     ):
        #         self.min_cut += self.edges_and_throughput[edge]  # нахождение значения минимального разреза

        for vertex in range(1, self.amount_of_vertex_and_edges[0] + 1):
            if vertex not in self.min_cut_object:
                self.min_cut_background.add(vertex)

        return self.min_cut_object, self.min_cut_background

    def global_relabeling(self) -> None:
        """
        Для каждой вершины запускаем bfs до стока, если расстояние больше чем высота вершины, то высоту полагаем равной
        расстоянию от вершины до стока.
        Если сток не достижим, то запускаем bfs от вершины до истока, если расстояние больше чем высота вершины, то
        высоту полагаем равной расстоянию от вершины до истока.
        """
        for vertex in self.vertex_and_height_excess:
            distance = self.bfs(vertex, self.sink)
            if distance > 0 or vertex == self.sink:
                if self.vertex_and_height_excess[vertex][0] < distance:
                    self.vertex_and_height_excess[vertex][0] = distance
            else:
                distance = self.bfs(vertex, self.source)
                if distance > 0 or vertex == self.source:
                    if self.vertex_and_height_excess[vertex][0] < distance:
                        self.vertex_and_height_excess[vertex][0] = distance

    def try_global_relabeling(self) -> None:
        """
        Если количество посторений элементарных операций равно заданному количество повторений, то запускаем global
        relabeling, иначе увеличваем количество повторений на единицу
        """
        if self.amount_of_push_and_rel < self.glob_rel_value:
            self.amount_of_push_and_rel += 1
        else:
            self.global_relabeling()
            self.amount_of_push_and_rel = 0

    def check_equality_min_cut_and_max_flow(self):
        if self.min_cut == self.max_flow:
            print("Значения минимального разреза и максимального потока равны")
        else:
            print("Значения минимального разреза и максимального потока не равны")

    @property
    def source(self):
        """Getter для истока"""
        return self._source

    @source.setter
    def source(self, source):
        """Setter для истока"""
        self._source = source

    @property
    def sink(self):
        """Getter для стока"""
        return self._sink

    @sink.setter
    def sink(self, sink):
        """Setter для стока"""
        self._sink = sink

    @property
    def vertex_and_height_excess(self):
        """Getter для словаря, у которого ключ - это вершина, а значение - это список ее высоты и избытка"""
        return self._vertex_and_height_excess

    @vertex_and_height_excess.setter
    def vertex_and_height_excess(self, vertex_and_height_excess):
        """Setter для словаря, у которого ключ - это вершина, а значение - это список ее высоты и избытка"""
        self._vertex_and_height_excess = vertex_and_height_excess

    @property
    def edges_and_flow_residual_capacity(self):
        """Getter для словаря, у которого ключ - это ребро (кортеж двух чисел например:
        (1, 2)), а значение - это список потока и остаточной пропускной способности"""
        return self._edges_and_flow_residual_capacity

    @edges_and_flow_residual_capacity.setter
    def edges_and_flow_residual_capacity(self, edges_and_flow_residual_capacity):
        """Setter для словаря, у которого ключ - это ребро (кортеж двух чисел например:
        (1, 2)), а значение - это список потока и остаточной пропускной способности"""
        self._edges_and_flow_residual_capacity = edges_and_flow_residual_capacity

    @property
    def amount_of_push_and_rel(self):
        """Getter для количества повторений элементарных операций (push - relabel)"""
        return self._amount_of_push_and_rel

    @amount_of_push_and_rel.setter
    def amount_of_push_and_rel(self, amount_of_push_and_rel):
        """Setter для количества повторений элементарных операций (push - relabel)"""
        self._amount_of_push_and_rel = amount_of_push_and_rel

    @property
    def glob_rel_value(self):
        """Getter для числа повторений, через которое можно запускать global relabeling"""
        return self._glob_rel_value

    @glob_rel_value.setter
    def glob_rel_value(self, glob_rel_value):
        """Setter для числа повторений, через которое можно запускать global relabeling"""
        self._glob_rel_value = glob_rel_value

    @property
    def max_flow(self):
        """Getter для ответа"""
        return self._max_flow

    @max_flow.setter
    def max_flow(self, max_flow):
        """Setter для ответа"""
        self._max_flow = max_flow

    @property
    def min_cut(self):
        """Getter для минимального разреза"""
        return self._min_cut

    @min_cut.setter
    def min_cut(self, min_cut):
        """Setter для минимального разреза"""
        self._min_cut = min_cut

    @property
    def min_cut_object(self):
        """Getter для вершин слева от разреза"""
        return self._min_cut_object

    @min_cut_object.setter
    def min_cut_object(self, min_cut_object):
        """Setter для вершин слева от разреза"""
        self._min_cut_object = min_cut_object

    @property
    def min_cut_background(self):
        """Getter для вершин справа от разреза"""
        return self._min_cut_background

    @min_cut_background.setter
    def min_cut_background(self, min_cut_background):
        """Setter для вершин справа от разреза"""
        self._min_cut_background = min_cut_background


def read_files_and_find_max_flow(directory):
    files = os.listdir(directory)
    for file in files:
        print("-" * 30, file, "-" * 30)
        start = time.time()
        g = Graph(directory + "/" + file)
        g.push_relabel_max_flow()
        g.get_min_cut()
        print("Значение максимального потока:", g.max_flow)
        print("Вершины слева от разреза:", g.min_cut_object)
        print("Вершины справа от разреза:", g.min_cut_background)
        end = time.time()
        print("Времени заняло:", round(end - start, 3))


if __name__ == "__main__":
    read_files_and_find_max_flow(
        "MaxFlow-tests"
    )  # для работы с директорией, в которой находятся тесты

    # g = Graph("MaxFlow-tests/test_4.txt")  # для работы с конкретным файлом, лучше указывать полный путь
    # g.push_relabel_max_flow()
    # g.get_min_cut()
    # print(g.max_flow, g.min_cut_object, g.min_cut_background)
