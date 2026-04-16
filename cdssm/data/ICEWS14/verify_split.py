def get_range(file):
    with open(file) as f:
        times = [int(line.split()[3]) for line in f]
    return min(times), max(times)

print("Train:", get_range("train_split.txt"))
print("Valid:", get_range("valid.txt"))
print("Test :", get_range("test.txt"))