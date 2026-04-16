import torch

def load_data(file_path):
    data = []
    
    with open(file_path, "r") as f:
        for line in f:
            s, r, o, t, _ = map(int, line.strip().split())
            data.append((s, r, o, t))
    
    return data


def get_stats(data):
    entities = set()
    relations = set()
    
    for s, r, o, t in data:
        entities.add(s)
        entities.add(o)
        relations.add(r)
    
    num_entities = max(entities) + 1
    num_relations = max(relations) + 1
    
    return num_entities, num_relations