#1 make a simple hash table, but first let's import what we need
#2 binary search tree (BST) - because trees are beautiful
#3 Dijkstra's algorithm for shortest path finding - because GPS uses something too
#4 merge sort - because it's faster than bubble sort and has beautiful recursion
#5 priority queue - because heapq isn't always obvious
#6 A* pathfinding algorithm - because games also want to find paths
#7 breadth-first search (BFS) - because sometimes you need to go level by level
#8 Kruskal's algorithm for minimum spanning tree - because networks need to be connected
#9 Floyd-Warshall algorithm for all shortest paths - because sometimes you need to know everything

#10 Kosaraju's algorithm for strongly connected components - because graphs can be different

  
print("\n-----------------------------------------------------------------------1-----")
# hash table implementation - because Python dictionaries are cool, but how do they work inside?
from collections import defaultdict  
# this is like a regular dict but with default values

class HashTable:
    """our custom hash table with chaining collision resolution"""
    def __init__(self, size=10):
        self.size = size
        # self.table = [[] for _ in range(size)] 
        self.table = defaultdict(list)  #defaultdict for simplicity
    
    def _hash(self, key):
        """simplest hash function - modulo division"""
        return hash(key) % self.size  # built-in hash() + modulo
    
    def insert(self, key, value):
        """add key-value pair"""
        hash_key = self._hash(key)
        # check if key already exists
        for i, (k, v) in enumerate(self.table[hash_key]):
            if k == key:
                self.table[hash_key][i] = (key, value)  
                #update value
                return
        self.table[hash_key].append((key, value))  #new pair
    
    def get(self, key):
        """get value by key"""
        hash_key = self._hash(key)
        for k, v in self.table[hash_key]:
            if k == key:
                return v
        raise KeyError(f"Key {key} not found")  # if key doesn't exist
    
    def __str__(self):
        """pretty print the table"""
        return "\n".join(f"{i}: {items}" for i, items in enumerate(self.table) if items)

# test the hash table
ht = HashTable()
ht.insert("apple", 5)
ht.insert("banana", 7)
ht.insert("orange", 3)
# ht.insert("apple", 10)  

print("Our hash table:")
print(ht)
# print(ht.get("banana"))  #  7
# print(ht.get("grape"))  # KeyError

print("-----------------------------------------------------------------------2-----")

class Node:
    """tree node - like a leaf on a branch"""
    def __init__(self, value):
        self.value = value
        self.left = None  # left branch
        self.right = None  # right branch

class BST:
    """the binary search tree itself"""
    def __init__(self):
        self.root = None  
        # tree root
    
    def insert(self, value):
        """insert new value into the tree"""
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        """helper for recursive insertion"""
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
        """in-order tree traversal 'left-root-right'"""
        return self._inorder_recursive(self.root, [])
    
    def _inorder_recursive(self, node, result):
        """recursive helper for traversal"""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)
        return result

# tree and its values
bst = BST()
numbers = [8, 3, 10, 1, 6, 14, 4, 7, 13]
for num in numbers:
    bst.insert(num)

print("BST in-order traversal:")
print(bst.inorder())  # [1, 3, 4, 6, 7, 8, 10, 13, 14]
# print(bst.root.left.right.value)  #6

print("\n-----------------------------------------------------------------------3-----")
import heapq  #we'll need priority queue

def dijkstra(graph, start):
    """find shortest paths from start vertex to all others"""
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]  
    #priority queue
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        #if we found a shorter path - skip
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# example graph (weighted)
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

print("Shortest distances from point A:")
print(dijkstra(graph, 'A'))  # {'A': 0, 'B': 1, 'C': 3, 'D': 4}
# print(dijkstra(graph, 'C'))  # uncomment to see other starting points

print("\n-----------------------------------------------------------------------4-----")
def merge_sort(arr):
    """sort array using merge sort"""
    if len(arr) > 1:
        mid = len(arr) // 2  #middle
        left = arr[:mid]  # left half
        right = arr[mid:]  # right half
        
        merge_sort(left)  # sort left
        merge_sort(right)  # sort right
        
        # merge two sorted halves
        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
#add remaining elements
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1

data = [12, 11, 13, 5, 6, 7]
print("Original array:", data)
merge_sort(data)
print("Sorted array:", data)  # [5, 6, 7, 11, 12, 13]


print("\n-----------------------------------------------------------------------5-----")
class PriorityQueue:
    """our own priority queue"""
    def __init__(self):
        self._heap = []
        self._index = 0  # for handling same priority
    
    def push(self, item, priority):
        """add item with priority"""
        heapq.heappush(self._heap, (-priority, self._index, item))
        self._index += 1
    
    def pop(self):
        """get item with highest priority"""
        return heapq.heappop(self._heap)[-1]
    
    def __str__(self):
        return str([(item, -priority) for priority, _, item in sorted(self._heap)])

pq = PriorityQueue()
pq.push("task1", 3)
pq.push("task2", 1)
pq.push("task3", 2)
pq.push("task4", 5)

print("Priority queue:")
print(pq)  
# [('task4', 5), ('task1', 3), ('task3', 2), ('task2', 1)]
print("Processing tasks by priority:")
# print(pq.pop())  #task4
# print(pq.pop())  #task1

print("\n-----------------------------------------------------------------------6-----")
def a_star(start, goal, graph, heuristic):
    """A* algorithm for shortest path finding"""
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
    
    return None  # if path not found

grid_graph = {
    'A': {'B': 1, 'D': 3},
    'B': {'A': 1, 'C': 2, 'D': 4},
    'C': {'B': 2, 'D': 1, 'E': 5},
    'D': {'A': 3, 'B': 4, 'C': 1, 'E': 1},
    'E': {'C': 5, 'D': 1}
}

def manhattan(a, b):
    """Manhattan distance between nodes"""
    return abs(ord(a) - ord(b))  #heuristic

path = a_star('A', 'E', grid_graph, manhattan)
print("A* found path:", path) 
# print(a_star('B', 'E', grid_graph, manhattan))  

print("\n-----------------------------------------------------------------------7-----")

from collections import deque  
#double-ended queue for efficiency

def bfs(graph, start):
    """breadth-first graph traversal"""
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

#graph (unweighted)
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("Breadth-first search (BFS):")
print(bfs(graph, 'A'))  
# print(bfs(graph, 'C'))  

print("\n-----------------------------------------------------------------------8-----")
class DisjointSet:
    """disjoint set system for Kruskal"""
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}
    
    def find(self, item):
        """find set root"""
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  
            # path compression
        return self.parent[item]
    
    def union(self, set1, set2):
        """union two sets"""
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
    """Kruskal's algorithm for minimum spanning tree"""
    edges = []
    for u in graph:
        for v, weight in graph[u].items():
            edges.append((weight, u, v))
    edges.sort()  
# sort edges by weight
    
    vertices = set(graph.keys())
    ds = DisjointSet(vertices)
    mst = []
    
    for edge in edges:
        weight, u, v = edge
        if ds.find(u) != ds.find(v):
            ds.union(u, v)
            mst.append(edge)
    
    return mst

# example graph
graph = {
    'A': {'B': 2, 'D': 6},
    'B': {'A': 2, 'C': 3, 'D': 8, 'E': 5},
    'C': {'B': 3, 'E': 7},
    'D': {'A': 6, 'B': 8, 'E': 9},
    'E': {'B': 5, 'C': 7, 'D': 9}
}

print("Minimum spanning tree (Kruskal):")
print(kruskal(graph))  

print("\n-----------------------------------------------------------------------9-----")
def floyd_warshall(graph):
    """Floyd-Warshall algorithm for all pairs shortest paths"""
    vertices = list(graph.keys())
    n = len(vertices)
    dist = [[float('infinity')] * n for _ in range(n)]
    
    #distance matrix
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

print("All shortest paths (Floyd-Warshall):")
result = floyd_warshall(graph)
for u in result:
    print(f"{u}: {result[u]}")
# output --- matrix of shortest paths between all vertex pairs

print("\n----------------------------------------------------------------------10-----")
def kosaraju(graph):
    """Kosaraju's algorithm for strongly connected components"""
    visited = set()
    order = []
    
    def dfs(node):    #first pass (direct graph)
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
    
    reversed_graph = {}#reversed graph
    for node in graph:
        for neighbor in graph[node]:
            if neighbor not in reversed_graph:
                reversed_graph[neighbor] = []
            reversed_graph[neighbor].append(node)
    
#second pass (reversed graph)
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

#example graph
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

print("Strongly connected components (Kosaraju):")
print(kosaraju(graph))  
### [['A', 'C', 'B'], ['D', 'F', 'E'], ['G', 'J', 'I', 'H'], ['K']]

print("\n----------------------------------------------------------------------------")