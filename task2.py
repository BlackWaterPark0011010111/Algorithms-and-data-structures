#Окей, линейный поиск - это как искать в темноте, а бинарный - уже с фонариком!
# Но есть условие: список должен быть отсортирован. Как в телефонной книге.

def binary_search(sorted_list, target):
    """
    Бинарный поиск - делим список пополам и отбрасываем ненужную часть
    """
    left = 0  # Левая граница поиска
    right = len(sorted_list) - 1  # Правая граница
    
    while left <= right:
        middle = (left + right) // 2  # середина
        
        if sorted_list[middle] == target:
            return middle  #находим
        elif sorted_list[middle] < target:
            left = middle + 1  #ищем в правой половине
        else:
            right = middle - 1  #ищем в левой половине
    
    return -1  # ничего не нашли

# ищем страницу в книге
book_pages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
page = 70
result = binary_search(book_pages, page)

if result != -1:
    print(f"Страница {page} найдена на позиции {result}")
else:
    print("Такой страницы нет в книге :(")