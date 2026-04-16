import torch
import json
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import load_data
from models.node_embeddings import NodeEmbedding
from models.graph_mamba import GraphMamba
from models.graph_ssm import GraphSSM
from models.predictor import TKGScorer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# LOAD CONFIG + MODEL
# ========================
result_dir = "../results/fusion_False_causal_False_uq_False"

with open(f"{result_dir}/config.json") as f:
    config = json.load(f)

checkpoint = torch.load(f"{result_dir}/best_model.pt", map_location=device)


# ========================
# USE TRAIN CONFIG (CRITICAL)
# ========================
num_entities = config["num_entities"]
num_relations = config["num_relations"]


# ========================
# LOAD DATA
# ========================
test_data = load_data("../../data/icews14/test.txt")

snapshots = torch.load("../snapshots.pt")
times = sorted(snapshots.keys())[:config["snapshots_used"]]


# ========================
# LOAD MODELS
# ========================
node_emb = NodeEmbedding(num_entities, config["hidden_dim"]).to(device)
graph_mamba = GraphMamba(config["hidden_dim"]).to(device)
graph_ssm = GraphSSM(config["hidden_dim"]).to(device)
predictor = TKGScorer(num_entities, num_relations, config["hidden_dim"]).to(device)

node_emb.load_state_dict(checkpoint["node_emb"])
graph_mamba.load_state_dict(checkpoint["graph_mamba"])
graph_ssm.load_state_dict(checkpoint["graph_ssm"])
predictor.load_state_dict(checkpoint["predictor"])


# ========================
# BUILD REPRESENTATION
# ========================
node_emb.eval()
graph_mamba.eval()
graph_ssm.eval()

with torch.no_grad():
    X = node_emb()

    H_seq = []
    for t in times:
        edge_index, _ = snapshots[t]
        edge_index = edge_index.to(device)

        H_t = graph_mamba(X, edge_index)
        H_seq.append(H_t)

    S_seq = graph_ssm(H_seq)
    h = S_seq[-1]


# ========================
# EVALUATION
# ========================
ranks = []

print("\nEvaluating...")

for (s, r, o, t) in tqdm(test_data[:2000]):
    s_tensor = torch.tensor([s]).to(device)
    r_tensor = torch.tensor([r]).to(device)

    h_s = h[s_tensor]

    # ONLY predictor (correct)
    scores = predictor(h_s, r_tensor).squeeze()

    # ranking
    _, indices = torch.sort(scores, descending=True)

    rank = (indices == o).nonzero(as_tuple=True)[0].item() + 1
    ranks.append(rank)


# ========================
# METRICS
# ========================
ranks = torch.tensor(ranks, dtype=torch.float)

mrr = torch.mean(1.0 / ranks).item()
hits1 = torch.mean((ranks <= 1).float()).item()
hits3 = torch.mean((ranks <= 3).float()).item()
hits10 = torch.mean((ranks <= 10).float()).item()


# ========================
# RESULTS
# ========================
print("\nEvaluation Results:")
print(f"MRR: {mrr:.4f}")
print(f"Hits@1: {hits1:.4f}")
print(f"Hits@3: {hits3:.4f}")
print(f"Hits@10: {hits10:.4f}")