import numpy as np

input_file = "train.txt"

with open(input_file, "r") as f:
    lines = f.readlines()

# sort by timestamp (4th column)
lines.sort(key=lambda x: int(x.split()[3]))

n = len(lines)

train_split = int(0.8 * n)
valid_split = int(0.9 * n)

train_lines = lines[:train_split]
valid_lines = lines[train_split:valid_split]
test_extra = lines[valid_split:]  # optional

with open("train_split.txt", "w") as f:
    f.writelines(train_lines)

with open("valid.txt", "w") as f:
    f.writelines(valid_lines)

print("Temporal split done!")