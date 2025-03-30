#1 сделайте простую хеш-таблицу, но сначала импортируем что нужно
#2 бинарное дерево поиска (BST) - потому что деревья это красиво
#3 алгоритм Дейкстры для поиска кратчайшего пути - потому что GPS тоже чем-то пользуется
#4 сортировка слиянием - потому что быстрее пузырька и красивая рекурсия
#5 очередь с приоритетом - потому что heapq не всегда очевидна
#6 алгоритм A* для поиска пути - потому что игры тоже хотят находить пути
#7 обход графа в ширину (BFS) - потому что иногда нужно идти по уровням
#8 алгоритм Крускала для минимального остовного дерева - потому что сети должны быть связными
#9 алгоритм Флойда-Уоршелла для всех кратчайших путей - потому что иногда нужно знать всё

#10 алгоритм Косарайю для поиска сильно связных компонент - потому что графы бывают разными


print("\n-----------------------------------------------------------------------1-----")
# hash table implementation - потому что словари в питоне это круто, но как они работают внутри?
from collections import defaultdict  
# это как обычный dict, но с дефолтными значениями

class HashTable:
    """наша самопальная хеш-таблица с обработкой коллизий методом цепочек"""
    def __init__(self, size=10):
        self.size = size
        # self.table = [[] for _ in range(size)] 
        self.table = defaultdict(list)  #defaultdict для упрощения
    
    def _hash(self, key):
        """простейшая хеш-функция - остаток от деления"""
        return hash(key) % self.size  # встроенная hash() + модуль
    
    def insert(self, key, value):
        """добавляем пару ключ-значение"""
        hash_key = self._hash(key)
        # проверяем, нет ли уже такого ключа
        for i, (k, v) in enumerate(self.table[hash_key]):
            if k == key:
                self.table[hash_key][i] = (key, value)  
                #обновляем значение
                
                return
        self.table[hash_key].append((key, value))  #новая пару
    
    def get(self, key):
        """получаем значение по ключу"""
        hash_key = self._hash(key)
        for k, v in self.table[hash_key]:
            if k == key:
                return v
        raise KeyError(f"Key {key} not found")  # если ключа нет
    
    def __str__(self):
        """красиво печатаем таблицу"""
        return "\n".join(f"{i}: {items}" for i, items in enumerate(self.table) if items)

# тестируем хеш-таблицу
ht = HashTable()
ht.insert("apple", 5)
ht.insert("banana", 7)
ht.insert("orange", 3)
# ht.insert("apple", 10)  

print("Наша хеш-таблица:")
print(ht)
# print(ht.get("banana"))  #  7
# print(ht.get("grape"))  # KeyError

print("-----------------------------------------------------------------------2-----")

class Node:
    """узел дерева - как лист на ветке"""
    def __init__(self, value):
        self.value = value
        self.left = None  # левая ветка
        self.right = None  # правая ветка

class BST:
    """само бинарное дерево поиска"""
    def __init__(self):
        self.root = None  
        # корень дерева
    
    def insert(self, value):
        """вставляем новое значение в дерево"""
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        """помогатель для рекурсивной вставки"""
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def inorder(self):
        """обход дерева в порядке 'левый-корень-правый'"""
        return self._inorder_recursive(self.root, [])
    
    def _inorder_recursive(self, node, result):
        """рекурсивный помощник для обхода"""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)
        return result

#дерево и его значения
bst = BST()
numbers = [8, 3, 10, 1, 6, 14, 4, 7, 13]
for num in numbers:
    bst.insert(num)

print("Обход BST в порядке возрастания:")
print(bst.inorder())  # [1, 3, 4, 6, 7, 8, 10, 13, 14]
# print(bst.root.left.right.value)  #6

print("\n-----------------------------------------------------------------------3-----")
import heapq  #понадобится очередь с приоритетом

def dijkstra(graph, start):
    """ищем кратчайшие пути от стартовой вершины до всех остальных"""
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]  
    #очередь с приоритетом
    

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        #если найден более короткий путь - пропускаем
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# пример графа (взвешенного)
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

print("Кратчайшие расстояния от точки A:")
print(dijkstra(graph, 'A'))  # {'A': 0, 'B': 1, 'C': 3, 'D': 4}
# print(dijkstra(graph, 'C'))  # раскомментируй чтобы увидеть другие стартовые точки

print("\n-----------------------------------------------------------------------4-----")
def merge_sort(arr):
    """сортируем массив методом слияния"""
    if len(arr) > 1:
        mid = len(arr) // 2  #середину
        left = arr[:mid]  # левая половина
        right = arr[mid:]  # правая половина
        
        merge_sort(left)  # сортировка левую
        merge_sort(right)  # сортировка правую
        
        # слияние двух отсортированных половин
        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
#добавляем оставшиеся элементы
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1

data = [12, 11, 13, 5, 6, 7]
print("Исходный массив:", data)
merge_sort(data)
print("Отсортированный массив:", data)  # [5, 6, 7, 11, 12, 13]

print("\n-----------------------------------------------------------------------5-----")
class PriorityQueue:
    """наша собственная очередь с приоритетами"""
    def __init__(self):
        self._heap = []
        self._index = 0  # для обработки с одинаковым приоритетом
    
    def push(self, item, priority):
        """добавляем элемент с приоритетом"""
        heapq.heappush(self._heap, (-priority, self._index, item))
        self._index += 1
    
    def pop(self):
        """извлекаем элемент с наивысшим приоритетом"""
        return heapq.heappop(self._heap)[-1]
    
    def __str__(self):
        return str([(item, -priority) for priority, _, item in sorted(self._heap)])

pq = PriorityQueue()
pq.push("task1", 3)
pq.push("task2", 1)
pq.push("task3", 2)
pq.push("task4", 5)

print("Очередь с приоритетами:")
print(pq)  
# [('task4', 5), ('task1', 3), ('task3', 2), ('task2', 1)]
print("Обрабатываем задачи по приоритету:")
# print(pq.pop())  #task4
# print(pq.pop())  #task1

print("\n-----------------------------------------------------------------------6-----")
def a_star(start, goal, graph, heuristic):
    """алгоритм A* для поиска кратчайшего пути"""
    open_set = {start}
    came_from = {}
    g_score = {vertex: float('infinity') for vertex in graph}
    g_score[start] = 0
    f_score = {vertex: float('infinity') for vertex in graph}
    f_score[start] = heuristic(start, goal)
    
    while open_set:
        current = min(open_set, key=lambda vertex: f_score[vertex])
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        open_set.remove(current)
        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + graph[current][neighbor]
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                if neighbor not in open_set:
                    open_set.add(neighbor)
    
    return None  # если путь не найден




grid_graph = {
    'A': {'B': 1, 'D': 3},
    'B': {'A': 1, 'C': 2, 'D': 4},
    'C': {'B': 2, 'D': 1, 'E': 5},
    'D': {'A': 3, 'B': 4, 'C': 1, 'E': 1},
    'E': {'C': 5, 'D': 1}
}

def manhattan(a, b):
    """манхэттенское расстояние между узлами"""
    return abs(ord(a) - ord(b))  #эвристика

path = a_star('A', 'E', grid_graph, manhattan)
print("Найденный путь A*:", path) 
# print(a_star('B', 'E', grid_graph, manhattan))  

print("\n-----------------------------------------------------------------------7-----")

from collections import deque  
#двусторонняя очередь для эффективности


def bfs(graph, start):
    """обход графа в ширину"""
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

#граф (не взвешенный)
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("Обход графа в ширину (BFS):")
print(bfs(graph, 'A'))  
# print(bfs(graph, 'C'))  

print("\n-----------------------------------------------------------------------8-----")
class DisjointSet:
    """система непересекающихся множеств для Крускала"""
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}
    
    def find(self, item):
        """находим корень множества"""
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  
            # path compression

        return self.parent[item]
    
    def union(self, set1, set2):
        """объединяем два множества"""
        root1 = self.find(set1)
        root2 = self.find(set2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1

def kruskal(graph):
    """алгоритм Крускала для минимального остовного дерева"""
    edges = []
    for u in graph:
        for v, weight in graph[u].items():
            edges.append((weight, u, v))
    edges.sort()  
# сортируем ребра по весу
    
    vertices = set(graph.keys())
    ds = DisjointSet(vertices)
    mst = []
    
    for edge in edges:
        weight, u, v = edge
        if ds.find(u) != ds.find(v):
            ds.union(u, v)
            mst.append(edge)
    
    return mst

# пример графа
graph = {
    'A': {'B': 2, 'D': 6},
    'B': {'A': 2, 'C': 3, 'D': 8, 'E': 5},
    'C': {'B': 3, 'E': 7},
    'D': {'A': 6, 'B': 8, 'E': 9},
    'E': {'B': 5, 'C': 7, 'D': 9}
}

print("Минимальное остовное дерево (Крускал):")
print(kruskal(graph))  

print("\n-----------------------------------------------------------------------9-----")
def floyd_warshall(graph):
    """алгоритм Флойда-Уоршелла для всех пар кратчайших путей"""
    vertices = list(graph.keys())
    n = len(vertices)
    dist = [[float('infinity')] * n for _ in range(n)]
    
    #матрица расстояний
    for i in range(n):
        dist[i][i] = 0
    for u in graph:
        for v in graph[u]:
            i = vertices.index(u)
            j = vertices.index(v)
            dist[i][j] = graph[u][v]
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return {vertices[i]: {vertices[j]: dist[i][j] for j in range(n)} for i in range(n)}

graph = {
    'A': {'B': 3, 'C': 8, 'E': -4},
    'B': {'D': 1, 'E': 7},
    'C': {'B': 4},
    'D': {'A': 2, 'C': -5},
    'E': {'D': 6}
}

print("Все кратчайшие пути (Флойд-Уоршелл):")
result = floyd_warshall(graph)
for u in result:
    print(f"{u}: {result[u]}")
# вывод --- матрица кратчайших расстояний между всеми парами вершин

print("\n----------------------------------------------------------------------10-----")
def kosaraju(graph):
    """алгоритм Косарайю для сильно связных компонент"""
    visited = set()
    order = []
    
    def dfs(node):    #первый проход (прямой граф)
        stack = [(node, False)]
        while stack:
            n, processed = stack.pop()
            if processed:
                order.append(n)
                continue
            if n in visited:
                continue
            visited.add(n)
            stack.append((n, True))
            for neighbor in graph.get(n, []):
                stack.append((neighbor, False))
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    reversed_graph = {}#обратный граф
    for node in graph:
        for neighbor in graph[node]:
            if neighbor not in reversed_graph:
                reversed_graph[neighbor] = []
            reversed_graph[neighbor].append(node)
    
#второй проход (обратный граф)
    visited = set()
    components = []
    
    while order:
        node = order.pop()
        if node not in visited:
            stack = [node]
            visited.add(node)
            component = []
            while stack:
                n = stack.pop()
                component.append(n)
                for neighbor in reversed_graph.get(n, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            components.append(component)
    
    return components

#пример графа
graph = {
    'A': ['B'],
    'B': ['C', 'D'],
    'C': ['A'],
    'D': ['E'],
    'E': ['F'],
    'F': ['D'],
    'G': ['F', 'H'],
    'H': ['I'],
    'I': ['J'],
    'J': ['G', 'K'],
    'K': []
}

print("Сильно связные компоненты (Косарайю):")
print(kosaraju(graph))  
### [['A', 'C', 'B'], ['D', 'F', 'E'], ['G', 'J', 'I', 'H'], ['K']]

print("\n----------------------------------------------------------------------------")
        