import torch
import torch.nn as nn

class TKGEmbedding(nn.Module):
    def __init__(self, num_entities, num_relations, num_timestamps, dim):
        super().__init__()

        self.dim = dim

        # Entity embeddings
        self.ent_emb = nn.Embedding(num_entities, dim)

        # Relation embeddings (already includes inverse)
        self.rel_emb = nn.Embedding(num_relations, dim)

        # Time embeddings
        self.time_emb = nn.Embedding(num_timestamps, dim)

        # Time projection (important for time-aware entities)
        self.time_proj = nn.Linear(dim, dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.xavier_uniform_(self.time_emb.weight)
        nn.init.xavier_uniform_(self.time_proj.weight)

    def forward(self, subjects, relations, objects, time_ids):
        """
        Inputs:
            subjects: (B,)
            relations: (B,)
            objects: (B,)
            time_ids: (B,)

        Returns:
            e_s: (B, D)  - Time-aware subject embedding
            e_r: (B, D)  - Relation embedding
            e_o: (B, D)  - Time-aware object embedding
            t_proj: (B, D) - Projected time embedding
        """
        # Base embeddings
        e_s = self.ent_emb(subjects)
        e_r = self.rel_emb(relations)
        e_o = self.ent_emb(objects)

        # Time embeddings
        t_emb = self.time_emb(time_ids)
        t_proj = self.time_proj(t_emb)

        # Time-aware entity embeddings
        e_s = e_s + t_proj
        e_o = e_o + t_proj

        return e_s, e_r, e_o, t_proj

    def get_all_entity_embeddings(self, time_ids):
        """
        Used for scoring against all entities later.
        time_ids: (B,) or scalar
        Returns:
            all_entities: (B, N, D)
        """
        all_ent = self.ent_emb.weight  # (N, D)

        if isinstance(time_ids, int):
            time_ids = torch.tensor([time_ids], device=all_ent.device)

        t_emb = self.time_emb(time_ids)  # (B, D)
        t_proj = self.time_proj(t_emb)   # (B, D)

        # Broadcast time to all entities
        all_ent = all_ent.unsqueeze(0) + t_proj.unsqueeze(1)

        return all_ent  # (B, N, D)

# ==========================================
# CP-2 VERIFICATION TEST
# ==========================================
if __name__ == "__main__":
    # Exact dimensions from your CP-1
    NUM_ENTITIES = 12498
    NUM_RELATIONS = 520
    NUM_TIMESTAMPS = 348
    DIM = 200

    model = TKGEmbedding(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        num_timestamps=NUM_TIMESTAMPS,
        dim=DIM
    )

    # Dummy batch of size 4
    s = torch.randint(0, NUM_ENTITIES, (4,))
    r = torch.randint(0, NUM_RELATIONS, (4,))
    o = torch.randint(0, NUM_ENTITIES, (4,))
    t = torch.randint(0, NUM_TIMESTAMPS, (4,))

    e_s, e_r, e_o, t_p = model(s, r, o, t)
    
    print("\n" + "="*40)
    print("====== CP-2 VERIFICATION ======")
    print("="*40)
    print(f"Time-aware Subject : {list(e_s.shape)} (Expected: [4, 200])")
    print(f"Relation           : {list(e_r.shape)} (Expected: [4, 200])")
    print(f"Time-aware Object  : {list(e_o.shape)} (Expected: [4, 200])")
    print(f"Projected Time     : {list(t_p.shape)} (Expected: [4, 200])")
    
    # Test the broadcast scoring function
    all_ents = model.get_all_entity_embeddings(t)
    print(f"All Entities (Eval): {list(all_ents.shape)} (Expected: [4, 12498, 200])")
    print("="*40 + "\n")