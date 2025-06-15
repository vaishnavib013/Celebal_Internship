# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 22:08:55 2025

@author: dell
"""

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None  # Fixed typo from mext to next

class LL:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        """Add a node at the end of the list"""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def L_list(self):
        """Display the linked list"""
        if not self.head:
            print("List is empty")
            return

        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_any(self, n):
        """Delete the nth node (1-based index)"""
        if not self.head:
            raise IndexError("Cannot delete from empty list")

        if n <= 0:
            raise ValueError("Index must be a positive integer")

        if n == 1:
            print(f"Deleting node at position {n} with value {self.head.data}")
            self.head = self.head.next
            return

        current = self.head
        prev = None
        count = 1

        while current and count < n:
            prev = current
            current = current.next
            count += 1

        if not current:
            raise IndexError("Index out of range")

        print(f"Deleting node at position {n} with value {current.data}")
        prev.next = current.next


# Now let's execute the linked list
if __name__ == "__main__":
    l = LL()

    # Adding nodes
    l.add_node(10)
    l.add_node(20)
    l.add_node(30)
    l.add_node(40)
    l.add_node(50)

    print("Original List:")
    l.L_list()

    try:
        l.delete_any(3)
        print("\nAfter Deleting 3rd node:")
        l.L_list()

        l.delete_any(10)  # This will raise an exception
    except Exception as e:
        print("Error:", e)

    try:
        empty_l = LL()
        empty_l.delete_any(1)  # Deleting from an empty list
    except Exception as e:
        print("\nError (empty list):", e)
