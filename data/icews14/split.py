import random

input_file = "train.txt"
train_out = "train_split.txt"
valid_out = "valid.txt"

with open(input_file, "r") as f:
    lines = f.readlines()

random.shuffle(lines)

split_idx = int(0.8 * len(lines))

train_lines = lines[:split_idx]
valid_lines = lines[split_idx:]

with open(train_out, "w") as f:
    f.writelines(train_lines)

with open(valid_out, "w") as f:
    f.writelines(valid_lines)

print("Split done!")