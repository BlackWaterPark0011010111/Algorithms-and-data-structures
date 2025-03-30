# Стек - это как стопка блинов: последний блин, который положили, будет съеден первым
# (LIFO - Last In First Out)

class Stack:
    def __init__(self):
        self.items = []  #наши блины
    
    def push(self, item):
        """Кладем новый элемент на вершину стека"""
        self.items.append(item) 
    
    def pop(self):
        """Берем верхний элемент со стека"""
        if not self.is_empty():  # Проверяем стек, пустой или нет
            return self.items.pop()  # удалить и вернуть последний элемент
        else:
            print("Стек пуст! Нечего брать!")
            return None
    
    def peek(self):
        """Смотрим, что лежит на вершине, но не забираем"""
        if not self.is_empty():
            return self.items[-1] 
         #смотрим последний элемент

        else:
            print("Стек пуст!")
            return None
    
    def is_empty(self):
        """Проверяем, пустой ли стек"""
        return len(self.items) == 0
    
    def size(self):
        """Сколько элементов в стеке"""
        return len(self.items)

browser_history = Stack()
#браузер и кнопка "назад"

browser_history.push("google.com")         # посещаемые страницы
browser_history.push("youtube.com")
browser_history.push("youtube.com/coding_tutorial")

print(f"Текущая страница: {browser_history.peek()}")

# юзер нажимает "Назад"
last_page = browser_history.pop()
print(f"Вернулись на: {browser_history.peek()}")