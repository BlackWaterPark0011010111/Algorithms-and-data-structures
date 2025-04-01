    # Очередь - это как очередь в магазине: кто первым пришел, тот первым и обслуживается
# (FIFO - First In First Out)

class Queue:
    def __init__(self):
        self.items = []  # Здесь будем хранить элементы очереди
    
    def enqueue(self, item):
        """Добавляем элемент в конец очереди"""
       
        self.items.append(item)  # Просто добавляем в конец списка
    
    def dequeue(self):
        """Забираем элемент из начала очереди"""

        if not self.is_empty():
            return self.items.pop(0)  # Удаляем и возвращаем первый элемент
        else:
            print("Очередь пуста! Нечего обслуживать!")
            return None
    
    def is_empty(self):
        """Проверяем, пустая ли очередь"""

        return len(self.items) == 0
    
    def size(self):
        """Сколько элементов в очереди"""
        return len(self.items)
    
    def front(self):

        """Смотрим, кто первый в очереди, но не обслуживаем"""
        if not self.is_empty():
            return self.items[0]
        else:
            print("Очередь пуста!")

            return None

#очередь на печать
printer_queue = Queue()

printer_queue.enqueue("Отчет.xlsx")        #документы на печать
printer_queue.enqueue("Презентация.pptx")
printer_queue.enqueue("Фото котика.jpg")

print(f"Сейчас печатается: {printer_queue.front()}")

printed = printer_queue.dequeue()#принтер обрабатывает первый документ
print(f"Распечатан: {printed}")
print(f"Следующий на печать: {printer_queue.front()}")