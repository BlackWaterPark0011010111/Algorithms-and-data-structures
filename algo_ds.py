#1 сделайте простую хеш-таблицу, но сначала импортируем что нужно
#2 бинарное дерево поиска (BST) - потому что деревья это красиво
#3 алгоритм Дейкстры для поиска кратчайшего пути - потому что GPS тоже чем-то пользуется
#4 сортировка слиянием - потому что быстрее пузырька и красивая рекурсия

#9алгоритм Косарайю для поиска сильно связных компонент - потому что графы бывают разными


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
        