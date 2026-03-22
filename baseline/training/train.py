import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import load_data, get_stats
from models.baseline import BaselineModel


# ========================
# CONFIG
# ========================
train_path = "../../data/icews14/train_split.txt"
batch_size = 512
epochs = 5
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# LOAD DATA
# ========================
train_data = load_data(train_path)

# 🔥 reduce for testing (remove later)
train_data = train_data[:10000]

num_entities, num_relations = get_stats(train_data)

print("Entities:", num_entities)
print("Relations:", num_relations)


# ========================
# PREPARE TENSORS
# ========================
s_list = []
r_list = []
o_list = []

for s, r, o, t in train_data:
    s_list.append(s)
    r_list.append(r)
    o_list.append(o)

s_tensor = torch.tensor(s_list, dtype=torch.long)
r_tensor = torch.tensor(r_list, dtype=torch.long)
o_tensor = torch.tensor(o_list, dtype=torch.long)


# ========================
# MODEL
# ========================
model = BaselineModel(num_entities, num_relations).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()


# ========================
# TRAINING LOOP
# ========================
num_samples = len(train_data)

for epoch in range(epochs):
    total_loss = 0

    # shuffle indices
    perm = torch.randperm(num_samples)

    for i in range(0, num_samples, batch_size):
        idx = perm[i:i+batch_size]

        s_batch = s_tensor[idx].to(device)
        r_batch = r_tensor[idx].to(device)
        o_batch = o_tensor[idx].to(device)

        scores = model(s_batch, r_batch)

        loss = loss_fn(scores, o_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")