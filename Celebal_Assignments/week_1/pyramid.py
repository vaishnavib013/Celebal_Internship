rows = int(input("Enter the number of rows: "))

print("\n3. Centered Pyramid Pattern:")
for i in range(rows):
    spaces = ' ' * (rows - i - 1)
    stars = '*' * (2 * i + 1)
    print(spaces + stars)