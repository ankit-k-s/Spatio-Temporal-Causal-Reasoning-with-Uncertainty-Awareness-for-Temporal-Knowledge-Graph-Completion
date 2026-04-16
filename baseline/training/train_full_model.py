import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import load_data
from models.node_embeddings import NodeEmbedding
from models.graph_mamba import GraphMamba
from models.graph_ssm import GraphSSM
from models.fusion import GatedFusion
from models.causal import CausalModule
from models.uncertainty import UncertaintyHead
from models.predictor import TKGScorer


# ========================
# CONFIG
# ========================
config = {
    "hidden_dim": 128,
    "epochs": 100,
    "snapshots_used": 20,
    "lr": 0.001,
    "batch_size": 512,
    "early_stopping_patience": 5,

    # ABLATION FLAGS
    "use_fusion": False,
    "use_causal": False,
    "use_uncertainty": False,

    "num_envs": 3,
    "lambda_inv": 0.1
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# RESULT DIR
# ========================
exp_name = f"fusion_{config['use_fusion']}_causal_{config['use_causal']}_uq_{config['use_uncertainty']}"
result_dir = f"../results/{exp_name}"
os.makedirs(result_dir, exist_ok=True)


# ========================
# LOAD DATA
# ========================
train_data = load_data("../../data/icews14/train_split.txt")
train_data = train_data[:10000]

num_entities = max(max(s, o) for s, _, o, _ in train_data) + 1
num_relations = max(r for _, r, _, _ in train_data) + 1

#  IMPORTANT FIX (ADD THIS)
config["num_entities"] = num_entities
config["num_relations"] = num_relations

snapshots = torch.load("../snapshots.pt")
times = sorted(snapshots.keys())[:config["snapshots_used"]]


# ========================
# MODELS
# ========================
node_emb = NodeEmbedding(num_entities, config["hidden_dim"]).to(device)
graph_mamba = GraphMamba(config["hidden_dim"]).to(device)
graph_ssm = GraphSSM(config["hidden_dim"]).to(device)
predictor = TKGScorer(num_entities, num_relations, config["hidden_dim"]).to(device)

fusion = GatedFusion(config["hidden_dim"]).to(device) if config["use_fusion"] else None
causal = CausalModule(config["hidden_dim"]).to(device) if config["use_causal"] else None
uncertainty = UncertaintyHead(config["hidden_dim"], num_entities).to(device) if config["use_uncertainty"] else None


# ========================
# OPTIMIZER
# ========================
params = list(node_emb.parameters()) + \
         list(graph_mamba.parameters()) + \
         list(graph_ssm.parameters()) + \
         list(predictor.parameters())

if fusion:
    params += list(fusion.parameters())

if causal:
    params += list(causal.parameters())

if uncertainty:
    params += list(uncertainty.parameters())

optimizer = optim.Adam(params, lr=config["lr"])
loss_fn = nn.CrossEntropyLoss()


# ========================
# TRAINING LOOP
# ========================
logs = []

best_loss = float("inf")
patience_counter = 0

for epoch in range(config["epochs"]):
    random.shuffle(train_data)

    X = node_emb()
    env_losses = []

    num_envs = config["num_envs"]
    env_size = len(times) // num_envs

    for env_idx in range(num_envs):
        start = env_idx * env_size
        end = (env_idx + 1) * env_size
        env_times = times[start:end]

        H_seq_env = []

        for t in env_times:
            edge_index, _ = snapshots[t]
            edge_index = edge_index.to(device)

            H_t = graph_mamba(X, edge_index)
            H_seq_env.append(H_t)

        S_seq_env = graph_ssm(H_seq_env)

        H_t_env = H_seq_env[-1]
        S_t_env = S_seq_env[-1]

        # FORWARD
        h = S_t_env

        if fusion:
            h = fusion(H_t_env, S_t_env)

        if causal:
            h = causal(h)

        # BATCH
        batch = train_data[:config["batch_size"]]

        s = torch.tensor([x[0] for x in batch]).to(device)
        r = torch.tensor([x[1] for x in batch]).to(device)
        o = torch.tensor([x[2] for x in batch]).to(device)

        h_s = h[s]

        # LOSS
        if uncertainty:
            mu, log_var = uncertainty(h_s)

            log_var = torch.clamp(log_var, min=-5, max=5)
            sigma = torch.exp(log_var)
            sigma = torch.clamp(sigma, min=1e-3, max=10)

            target_mu = mu.gather(1, o.unsqueeze(1))
            target_log_var = log_var.gather(1, o.unsqueeze(1))
            target_sigma = sigma.gather(1, o.unsqueeze(1))

            loss_env = torch.mean(
                (target_mu - 1) ** 2 / target_sigma + target_log_var
            )

        else:
            scores = predictor(h_s, r)
            loss_env = loss_fn(scores, o)

        env_losses.append(loss_env)

    # CAUSAL LOSS
    env_losses_tensor = torch.stack(env_losses)

    loss_mean = env_losses_tensor.mean()
    loss_var = env_losses_tensor.var()

    loss = loss_mean + config["lambda_inv"] * loss_var

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss = loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    logs.append({
        "epoch": epoch + 1,
        "loss": total_loss,
        "loss_mean": loss_mean.item(),
        "loss_var": loss_var.item()
    })

    # EARLY STOPPING
    if total_loss < best_loss:
        best_loss = total_loss
        patience_counter = 0

        torch.save({
            "node_emb": node_emb.state_dict(),
            "graph_mamba": graph_mamba.state_dict(),
            "graph_ssm": graph_ssm.state_dict(),
            "fusion": fusion.state_dict() if fusion else None,
            "causal": causal.state_dict() if causal else None,
            "uncertainty": uncertainty.state_dict() if uncertainty else None,
            "predictor": predictor.state_dict()
        }, f"{result_dir}/best_model.pt")

    else:
        patience_counter += 1

    if patience_counter >= config["early_stopping_patience"]:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break


# SAVE LOGS
with open(f"{result_dir}/logs.json", "w") as f:
    json.dump(logs, f, indent=4)

with open(f"{result_dir}/config.json", "w") as f:
    json.dump(config, f, indent=4)

print(f"\nResults saved in {result_dir}")