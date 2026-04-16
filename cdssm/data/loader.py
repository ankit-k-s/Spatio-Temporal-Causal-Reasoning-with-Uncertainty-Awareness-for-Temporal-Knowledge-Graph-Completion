import os
import torch
import numpy as np
from collections import defaultdict

class TKGDataloader:
    def __init__(self, data_dir: str):
        """
        CP-1: TKG Input Pipeline (Research Grade)
        Loads ICEWS14, adds reverse edges, chronologically sorts, and normalizes Time IDs.
        """
        self.data_dir = data_dir
        self.num_entities = 0
        self.num_relations_original = 0
        self.num_relations_total = 0
        self.time2id = {}
        
        # 1. Global Scan: Find exact entities, relations, and universal timestamps
        self._compute_global_stats()
        
        # 2. Build Snapshots: Now utilizing reverse edges and normalized time IDs
        self.train_snapshots = self._load_file("train_split.txt")
        self.valid_snapshots = self._load_file("valid.txt")
        self.test_snapshots = self._load_file("test.txt")
        
        # 3. CP-1 Verification printout
        self._verify_cp1()

    def _compute_global_stats(self):
        """Scans all files to find unique entities, relations, and map timestamps."""
        max_ent, max_rel = 0, 0
        all_times = set()
        
        for filename in ["train_split.txt", "valid.txt", "test.txt"]:
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                continue
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    h, r, t, tau = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    max_ent = max(max_ent, h, t)
                    max_rel = max(max_rel, r)
                    all_times.add(tau)
                    
        self.num_entities = max_ent + 1
        self.num_relations_original = max_rel + 1
        # Double the relations to account for reverse edges
        self.num_relations_total = self.num_relations_original * 2 
        
        # Create chronological time ID mapping (e.g., 0, 24, 48 -> 0, 1, 2)
        sorted_times = sorted(list(all_times))
        self.time2id = {tau: idx for idx, tau in enumerate(sorted_times)}

    def _load_file(self, filename: str):
        filepath = os.path.join(self.data_dir, filename)
        snapshots_raw = defaultdict(list)
        
        if not os.path.exists(filepath):
            return {}

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                h, r, t, tau = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                
                # Forward Edge
                snapshots_raw[tau].append((h, t, r))
                # Reverse Edge (Tail becomes Head, Relation gets offset)
                snapshots_raw[tau].append((t, h, r + self.num_relations_original))
        
        formatted_snapshots = {}
        # Iterate through explicitly sorted timestamps
        for tau in sorted(snapshots_raw.keys()):
            edges = snapshots_raw[tau]
            edges_np = np.array(edges)
            
            # edge_index shape: [2, E] (Head in row 0, Tail in row 1)
            edge_index = torch.tensor(edges_np[:, :2].T, dtype=torch.long)
            
            # edge_type shape: [E]
            edge_type = torch.tensor(edges_np[:, 2], dtype=torch.long)
            
            # Normalized ordinal time ID
            normalized_time_id = self.time2id[tau]
            
            formatted_snapshots[tau] = (edge_index, edge_type, normalized_time_id)
            
        return formatted_snapshots

    def _verify_cp1(self):
        """Prints the exact verification metrics requested for CP-1."""
        all_times = set(self.train_snapshots.keys()) | \
                    set(self.valid_snapshots.keys()) | \
                    set(self.test_snapshots.keys())
        
        print("\n" + "="*40)
        print("====== CP-1 VERIFICATION (REFINED) ======")
        print("="*40)
        print(f"Total Entities         : {self.num_entities}")
        print(f"Original Relations     : {self.num_relations_original}")
        print(f"Total Rel (w/ Inverse) : {self.num_relations_total}")
        print(f"Total Unique Timestamps: {len(all_times)}")
        print("-" * 40)
        print(f"Train Snapshots : {len(self.train_snapshots)}")
        print(f"Valid Snapshots : {len(self.valid_snapshots)}")
        print(f"Test Snapshots  : {len(self.test_snapshots)}")
        print("-" * 40)
        
        if len(self.train_snapshots) > 0:
            sample_tau = list(self.train_snapshots.keys())[0]
            edge_index, edge_type, time_id = self.train_snapshots[sample_tau]
            print(f"Sample Snapshot (Raw Tau: {sample_tau})")
            print(f" -> Normalized Time ID : {time_id}")
            print(f" -> edge_index shape   : {list(edge_index.shape)} (Should be [2, E])")
            print(f" -> edge_type shape    : {list(edge_type.shape)} (Should be [E])")
        print("="*40 + "\n")


if __name__ == "__main__":
    loader = TKGDataloader(data_dir="ICEWS14")