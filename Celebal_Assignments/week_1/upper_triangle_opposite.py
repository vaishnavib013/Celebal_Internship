rows = int(input("Enter the number of rows: "))

print("\nUpper Triangular Pattern from Opposite Side):")
for i in range(rows, 0, -1):
    print("* " * i)