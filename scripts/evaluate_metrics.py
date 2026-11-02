import numpy as np
from sklearn import preprocessing
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                              silhouette_score, davies_bouldin_score)

BASE = "/Users/harshalshah05/Documents/4-1 SEM/Genome Analysis"


def load_embedding(path):
    nodes, feats = [], []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            nodes.append(int(parts[0]))
            feats.append([float(x) for x in parts[1:]])
    return np.array(nodes), np.array(feats)


def load_compartments(path):
    # returns dict: (chr,start,end) -> label string, and dict node_id(row idx) -> label
    coord_label = {}
    idx_label = {}
    with open(path) as f:
        next(f)
        for i, line in enumerate(f):
            chrom, start, end, label = line.rstrip("\n").split("\t")
            coord_label[(chrom, start, end)] = label
            idx_label[i] = label
    return coord_label, idx_label


def load_reference(path):
    ref = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            chrom, start, end, label = parts[0], parts[1], parts[2], parts[3]
            ref[(chrom, start, end)] = label
    return ref


ref = load_reference(f"{BASE}/repos/sci/predictions/GM12878_SCI_sub_compartments.bed")

# also need coord lookup by node id (row index) from the master compartments file
master_coords = []
with open(f"{BASE}/results/repro_test_compartments.txt") as f:
    next(f)
    for line in f:
        chrom, start, end, _ = line.rstrip("\n").split("\t")
        master_coords.append((chrom, start, end))


def evaluate(name, embedding_data, node_ids, coord_label):
    # labels for clustering-quality metrics: use only nodes with a non-NA predicted label
    labels = np.array([coord_label[master_coords[n]] for n in node_ids])
    valid = labels != "NA"
    X = embedding_data[valid]
    y = labels[valid]
    y_num = preprocessing.LabelEncoder().fit_transform(y)

    # full pairwise distance matrix on ~27k points is too heavy; subsample for silhouette
    sil = silhouette_score(X, y_num, sample_size=5000, random_state=42)
    db = davies_bouldin_score(X, y_num)

    # ARI / NMI against reference bed, on bins present in both
    pred_map = {master_coords[n]: coord_label[master_coords[n]] for n in node_ids
                if coord_label[master_coords[n]] != "NA"}
    common = sorted(set(pred_map) & set(ref))
    pred_labels = [pred_map[c] for c in common]
    ref_labels = [ref[c] for c in common]
    ari = adjusted_rand_score(ref_labels, pred_labels)
    nmi = normalized_mutual_info_score(ref_labels, pred_labels)

    print(f"\n=== {name} ===")
    print(f"clustered bins (internal metrics): {len(y_num)}")
    print(f"Silhouette Score:      {sil:.5f}")
    print(f"Davies-Bouldin Index:  {db:.4f}")
    print(f"overlap bins with reference: {len(common)}")
    print(f"Adjusted Rand Index (ARI): {ari:.5f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.5f}")


# --- LINE: order-1 + order-2 concatenated, L2-normalized (matches Compartments.py) ---
nodes1, feats1 = load_embedding(f"{BASE}/results/repro_test_HiC_graph_order_1_samples_1M.embedding")
nodes2, feats2 = load_embedding(f"{BASE}/results/repro_test_HiC_graph_order_2_samples_1M.embedding")
assert np.array_equal(nodes1, nodes2), "order-1/order-2 node lists differ"
line_data = np.concatenate((feats1, feats2), axis=1)
line_data = preprocessing.normalize(line_data, norm='l2')
line_coord_label, _ = load_compartments(f"{BASE}/results/repro_test_compartments.txt")
evaluate("LINE (order 1+2 concatenated)", line_data, nodes1, line_coord_label)

# --- HOPE ---
nodes_h, feats_h = load_embedding(f"{BASE}/results/repro_test_HOPE_graph.embedding")
hope_coord_label, _ = load_compartments(f"{BASE}/results/repro_test_HOPE_compartments.txt")
evaluate("HOPE (order-1 Katz)", feats_h, nodes_h, hope_coord_label)
