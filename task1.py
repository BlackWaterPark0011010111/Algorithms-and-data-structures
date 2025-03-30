
# Представь, что ты ищешь ключи в темной комнате - ты просто шаришь руками по столу,
# Привет! Это мой первый алгоритм - линейный поиск.
# пока не нащупаешь нужную связку. 
# Вот так и этот алгоритм!

def linear_search(items, target):
    """
    Простейший поиск: проверяем каждый элемент по очереди
    items - список, в котором ищем
    target - что ищем
    """
    for i in range(len(items)):
        if items[i] == target: 
            return i  # Возвращаем индекс, где нашли элемент
    return -1 
 # Если ничего не нашли, возвращаем -1 (как "не найдено")

#тест
my_stuff = ["телефон", "ключи", "кошелек", "очки"]
where_keys = linear_search(my_stuff, "ключи")

print(f"Ключи лежат на {where_keys+1}-м месте в списке")  # +1 потому что считаем с 1, а не с 0
