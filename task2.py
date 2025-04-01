# Okay, linear search is like searching in the dark, but binary search is like using a flashlight!
# But there's a condition: the list must be sorted. Like in a phone book.

def binary_search(sorted_list, target):
    """Binary search - we divide the list in half and discard the unnecessary part"""
    left = 0  # Left search boundary
    right = len(sorted_list) - 1  #Right boundary
    
    while left <= right:
        middle = (left + right) // 2  # middle point
        
        if sorted_list[middle] == target:
            return middle  # found it
        elif sorted_list[middle] < target:
            left = middle + 1  #search in right half
        else:
            right = middle - 1  #search in left half
    
    return -1  # nothing found

book_pages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]#searching for a page in a book
page = 70
result = binary_search(book_pages, page)

if result != -1:
    print(f"Page {page} found at position {result}")
else:
    print("This page doesn't exist in the book :(")