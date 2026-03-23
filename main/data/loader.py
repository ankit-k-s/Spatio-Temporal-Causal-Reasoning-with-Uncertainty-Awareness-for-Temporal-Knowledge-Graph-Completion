def load_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            s, r, o, t, *_ = map(int, line.strip().split())
            data.append((s, r, o, t))
    return data