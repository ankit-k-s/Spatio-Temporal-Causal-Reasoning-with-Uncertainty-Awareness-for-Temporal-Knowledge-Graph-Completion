def split_graph(edge_index, edge_type, attn, threshold=0.5):
    mask = attn > threshold

    c_edge = edge_index[:, mask]
    s_edge = edge_index[:, ~mask]

    c_type = edge_type[mask]
    s_type = edge_type[~mask]

    return (c_edge, c_type), (s_edge, s_type)