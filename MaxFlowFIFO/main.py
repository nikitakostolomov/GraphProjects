import collections
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
        _glob_rel_value: число повторений, через которое можно запускать global relabeling (по умолчанию - 1)
        """
        self.path = path
        (
            self.amount_of_vertex_and_edges,
            self.edges_and_throughput,
        ) = self.get_data_from_file()

        self._source = 0
        self._sink = 0

        self._vertex_and_height_excess = {}
        self._edges_and_flow_residual_capacity = {}

        self._amount_of_push_and_rel = 0
        self._glob_rel_value = 1

        self._max_flow = 0
        self._min_cut = 0

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

    def push_relabel_max_flow(
        self
    ) -> None:
        """
        Вершины с положительным избытком обрабатываются (просматриваются) в порядке first-in, first-out.
        Вершина извлекается из списка и делаются операции push пока это возможно. Новые вершины с избытком добавляются в
        конец списка (только если вершина не является стоком). Если операции push для обрабатываемой вершины больше
        нельзя выполнить, и в вершине есть остаток, то выполняется операция подъем, а вершина добавляется в конец списка
        (опять же только если вершина не является стоком).
        Global relabeling можно запускать через каждые m элементарных операций (push - relabel). Здесь glob_rel_value -
        это m.
        :return: избыток стока
        """

        self.initialize_pre_flow(self.source)

        vertices_with_excess = collections.deque()  # пустой список вершин с положительным избытком

        for (
            vertex,
            height_and_excess,
        ) in (
            self.vertex_and_height_excess.items()
        ):  # ищем вершины с положительным избытком
            if height_and_excess[1] > 0:
                vertices_with_excess.append(vertex)

        while vertices_with_excess:  # пока список не пуст, цикл работает
            vertex_with_positive_excess = (
                vertices_with_excess.popleft()
            )  # извлекаем последний элемент

            adjacent_vertices = self.find_adjacent_vertices(
                vertex_with_positive_excess
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
                    self.edges_and_flow_residual_capacity[
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
                if self.vertex_and_height_excess[vertex_with_positive_excess][
                    1
                ] > 0 and (
                    self.vertex_and_height_excess[vertex_with_positive_excess][0]
                    == self.vertex_and_height_excess[vertex][0] + 1
                ):  # push выполняем только если избыток вершины больше нуля и высота вершины больше на единицу,
                    # чем у смежной
                    self.push(
                        (vertex_with_positive_excess, vertex),
                    )

                    self.try_global_relabeling()  # запускаем global relabeling, если количество повторений элементарных
                    # операций равно заданному количеству повторений

                    if (
                        vertex != self.sink
                    ):  # если вершина не является стоком, добавляем ее в конец списка
                        vertices_with_excess.append(vertex)

            if (
                self.vertex_and_height_excess[vertex_with_positive_excess][1] > 0
                and vertex_with_positive_excess != self.sink
            ):  # relabel выполняем только если избыток вершины больше нуля и вершина не является стоком
                self.relabel(
                    vertex_with_positive_excess,
                )
                vertices_with_excess.append(vertex_with_positive_excess)

                self.try_global_relabeling()  # запускаем global relabeling, если количество повторений элементарных
                # операций равно заданному количеству повторений

        self.max_flow = self.vertex_and_height_excess[self.sink][1]


    def find_adjacent_vertices(self, vertex: int) -> list:
        """
        Принимает вершину, для которой нужно найти смежные. Если вершина смежная, то добавляем вершину в список смежных
        вершин.
        :param vertex: вершина, для которой нужно найти смежные вершины
        :return: список смежных вершин
        """
        adjacent_vertices = []
        for (
            edge,
            flow_and_residual_capacity,
        ) in self.edges_and_flow_residual_capacity.items():
            if edge[0] == vertex:
                adjacent_vertices.append(edge[1])

        return adjacent_vertices

    def initialize_pre_flow(self, source) -> None:
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
        for vertex in range(1, self.amount_of_vertex_and_edges[0] + 1):
            if vertex != source:
                self.vertex_and_height_excess[vertex] = [
                    0,
                    0,
                ]  # для вершин, не являющихся источником, устанавливаем высоту и избыток равными нулю

            if vertex == source:
                self.vertex_and_height_excess[vertex] = [
                    self.amount_of_vertex_and_edges[0]
                ]  # для вершины, являющейся источником, устанавливаем высоту равную количеству вершин

        for (
            edge,
            throughput,
        ) in (
            self.edges_and_throughput.items()
        ):  # ставим нулевой поток для прямого и обратного ребра,
            # а также устанавливаем остаточную пропускную способность
            self.edges_and_flow_residual_capacity[edge] = [0, throughput]
            self.edges_and_flow_residual_capacity[edge[::-1]] = [0, 0]

        excess_in_source = 0
        for edge in self.edges_and_throughput:  # находим избыток источника
            if edge[0] == source:
                excess_in_source += self.edges_and_throughput[edge]

        self.vertex_and_height_excess[source].append(
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
                ] = throughput  # установили пропускную способность обратного ребра
                self.vertex_and_height_excess[edge[1]][
                    1
                ] = throughput  # увеличили избыток вершины, в которую идет поток
                self.vertex_and_height_excess[edge[0]][
                    1
                ] -= throughput  # уменьшили избыток источника

    def push(
        self,
        edge: tuple,
    ) -> None:
        """
        По ребру (u,v) пропускается максимально возможный поток, то есть минимум из избытка вершины u
        и остаточной пропускной способности ребра (u,v), вследствие чего избыток вершины u,
        остаточная пропускная способность ребра (u,v) и поток по обратному ребру (v,u) уменьшаются на величину потока,
        а избыток вершины v, поток по ребру (u,v) и остаточная пропускная способность обратного ребра (v,u)
        увеличиваются на эту же величину.
        :param edge: ребро (кортеж), по которому делаем проталкивание
        :return: кортеж из двух словарей:
            1) словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка
            2) словарь, у которого ключ - это ребро (кортеж двух чисел например: (1, 2)),
        а значение - это список потока и остаточной пропускной способности
        """
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

    def relabel(
        self,
        vertex: int,
    ) -> None:
        """
        Для переполненной вершины u применима операция подъёма, если все вершины, для которых в остаточной сети есть
        рёбра из u, расположены не ниже u. Следовательно, операцию проталкивания для вершины u произвести нельзя.
        В результате подъёма высота текущей вершины становится на единицу больше высоты самый низкой смежной вершины в
        остаточной сети, вследствие чего появляется как минимум одно ребро, по которому можно протолкнуть поток.
        :param vertex: вершина, которую нужно поднять
        :return: словарь, у которого ключ - это вершина, а значение - это список ее высоты и избытка, но высота
        увеличена на единицу
        """
        adjacent_vertices = {}  # словарь смежных вершин и их высот
        for edge in self.edges_and_flow_residual_capacity:
            if edge[0] == vertex and (
                self.edges_and_flow_residual_capacity[edge][0]
                - self.edges_and_flow_residual_capacity[edge][1]
                < 0
            ):  # первое условие для того, чтобы найти смежную вершину, воторое для того, чтобы поток не превышал
                # пропускную способность ребра(?)
                adjacent_vertex = edge[1]
                adjacent_vertices[adjacent_vertex] = self.vertex_and_height_excess[
                    adjacent_vertex
                ][0]

        if adjacent_vertices:
            vertex_with_min_height = min(
                adjacent_vertices, key=lambda x: adjacent_vertices[x]
            )  # находим вершину с минимальной высотой
            self.vertex_and_height_excess[vertex][0] += (
                adjacent_vertices[vertex_with_min_height] + 1
            )  # поднимаем нашу вершину

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
            bfs_with_destination = False  # значит bfs используется для нахождения min cut

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

    def get_min_cut(self):
        """
        Находит минимальный разрез следующим способом: пытаемся добраться до всех вершин из истока в остаточной сети
        с помощью bfs, все найденные вершины будут слева от разреза, остальные справа. Минимальный разрезом будет сумма
        пропускных способностей ребер, соединяющих вершины слева и справа от разреза.
        """
        vertices_accessible_from_source = self.bfs(self.source)  # ищем вершины, до которых можно добраться из истока

        for edge in self.edges_and_throughput:
            if edge[0] in vertices_accessible_from_source and edge[1] not in vertices_accessible_from_source:
                self.min_cut += self.edges_and_throughput[edge]

    def do_all_work(self, source: int, sink: int, glob_rel_value: int = 1) -> None:
        """
        Запускает функцию на нахождения максимального потока и функцию на нахождения минимального разреза.
        :param source: источник
        :param sink: сток
        :param glob_rel_value: число, global_relabeling будет запускаться через каждые glob_rel_value элементарных
        операций (push, relabel)
        :return:
        """
        self.source = source
        self.sink = sink
        self.glob_rel_value = glob_rel_value

        self.push_relabel_max_flow()
        self.get_min_cut()

        print(f"Максимальный поток: {self.max_flow}\nМинимальный разрез: {self.min_cut}")
        print(f"Равны ли они: {self.max_flow == self.min_cut}")

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


if __name__ == "__main__":
    g = Graph("TestFiles/TestDataFromTask")
    g.do_all_work(1, 4)
