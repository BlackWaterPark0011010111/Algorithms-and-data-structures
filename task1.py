# linear search
# you're searching for keys in a dark room - you just feel around the table with your hands
# until you find the right set. That's how this algorithm works!

"""
    Simplest search: check each item one by one
    items - list to search in
    target - what we're looking for
    """
def linear_search(items, target):
    for i in range(len(items)):
        if items[i] == target:  # found it
            return i  # return the index where we found the item
    return -1  # if nothing found, return -1 as not found

my_stuff = ["phone", "keys", "wallet", "glasses"]
where_keys = linear_search(my_stuff, "keys")

print(f"Keys are in position {where_keys+1} in the list")  # +1 because we count from 1, not 0 