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
