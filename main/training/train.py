import torch
import torch.optim as optim
import random
import os

from data.loader import load_data
from models.node_embeddings import NodeEmbedding
from models.graph_mamba import GraphMamba
from models.graph_ssm import GraphSSM
from models.predictor import TKGScorer
from models.csi.csi_model import CSIModel
from models.csi.loss import csi_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_data("../data/icews14/train.txt")[:10000]

num_entities = max(max(s, o) for s, _, o, _ in data) + 1
num_relations = max(r for _, r, _, _ in data) + 1

node_emb = NodeEmbedding(num_entities, 128).to(device)
mamba = GraphMamba(128).to(device)
ssm = GraphSSM(128).to(device)
pred = TKGScorer(num_entities, num_relations, 128).to(device)

model = CSIModel(128, mamba, ssm, pred).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    total_loss = 0
    random.shuffle(data)

    X = node_emb()

    for (s, r, o, t) in data[:2000]:
        s = torch.tensor([s]).to(device)
        r = torch.tensor([r]).to(device)
        o = torch.tensor([o]).to(device)

        edge_index = torch.randint(0, num_entities, (2, 500)).to(device)
        edge_type = torch.zeros(500).long().to(device)

        query = X[s]

        pred_c, pred_s, pred_i = model(
            X, edge_index, edge_type, query, r
        )

        loss = csi_loss(pred_c, pred_s, pred_i, o)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")