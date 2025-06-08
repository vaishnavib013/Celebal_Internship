rows = int(input("Enter the number of rows: "))

print("\nUpper Triangular Pattern:")
for i in range(rows, 0, -1):
    spaces = "  " * (rows - i) 
    stars = "* " * i
    print(spaces + stars)