
# Связный список - как поезд, где каждый вагон (узел) знает только про следующий вагон
# (в отличие от массива, где все элементы стоят рядом)

class Node:
    """Это один 'вагон' нашего поезда-списка"""
    def __init__(self, data):
        self.data = data  #здесь храним данные
        self.next = None  
        #здесь - ссылку на следующий узел

class LinkedList:
    """Сам 'поезд' - связный список"""
    def __init__(self):
        self.head = None  
        #вагон изначально пуст
    
    def add_node(self, data):
        """Добавляем новый вагон в конец поезда"""
        new_node = Node(data)
        
        if self.head is None:  # Если поезд пустой
            self.head = new_node
        else:
            #идем до последнего вагона
            last = self.head
            while last.next:
                last = last.next
            last.next = new_node  #прицепляем новый вагон
    
    def print_list(self):
        """Печатаем весь поезд (для наглядности)"""
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")  #конец списка

# Создаем поезд-список
my_train = LinkedList()
my_train.add_node("Локомотив")
my_train.add_node("Вагон с углем")
my_train.add_node("Пассажирский вагон")
my_train.add_node("Вагон-ресторан")

my_train.print_list()  
