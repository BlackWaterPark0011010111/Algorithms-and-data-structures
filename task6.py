# Бинарное дерево - это как генеалогическое древо, но у каждого родителя максимум двое детей

class TreeNode:
    """один 'узел' дерева (как в родословной)"""
    def __init__(self, value):
        self.value = value  # Значение узла (имя человека)
        self.left = None    
        # левый потомок
        self.right = None   
        # правый потомок

class BinaryTree:
    """Само бинарное дерево"""
    def __init__(self):
        self.root = None  #корень дерева (прародитель семьи)
    
    def add(self, value):
        """новый узел в дерево"""
        if self.root is None:  # Если дерево пустое
            self.root = TreeNode(value)
        else:
            self._add_recursive(self.root, value)
    
    def _add_recursive(self, node, value):
        """вспомогательная рекурсивная функция для добавления"""
        if value < node.value:  # Если меньше текущего идем влево
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._add_recursive(node.left, value)
        else:  # идем вправо
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._add_recursive(node.right, value)
    
    def print_tree(self):
        """Печатаем дерево 'по порядку'"""
        self._print_recursive(self.root)
    
    def _print_recursive(self, node):
        """Рекурсивный обход дерева"""
        if node is not None:
            self._print_recursive(node.left)  #левое поддерево
            print(node.value)                 #сам узел
            self._print_recursive(node.right) #правое поддерево

#создание
tree = BinaryTree()
numbers = [5, 3, 7, 1, 4, 6, 8]

for num in numbers:
    tree.add(num)

print("Числа в отсортированном порядке:")
tree.print_tree()
