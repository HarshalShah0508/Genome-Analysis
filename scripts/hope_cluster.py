import numpy as np
from sklearn.cluster import KMeans

BASE = "/Users/harshalshah05/Documents/4-1 SEM/Genome Analysis"

# --- load genome-wide bin metadata (chr,start,end) in bin-id order ---
meta = []
with open(f"{BASE}/results/repro_test_compartments.txt") as f:
    next(f)
    for line in f:
        chrom, start, end, _ = line.rstrip("\n").split("\t")
        meta.append((chrom, int(start), int(end)))

# --- load HOPE embedding (LINE-compatible word2vec-style format) ---
nodes, feats = [], []
with open(f"{BASE}/results/repro_test_HOPE_graph.embedding") as f:
    next(f)
    for line in f:
        parts = line.split()
        nodes.append(int(parts[0]))
        feats.append([float(x) for x in parts[1:]])

nodes = np.array(nodes)
data = np.array(feats)
print(f"embedding matrix: {data.shape}")

# --- cluster, same k and random_state as the LINE/SCI run for a fair comparison ---
model = KMeans(n_clusters=5, random_state=12345)
model.fit(data)
labels = model.predict(data)

compartments = np.full(len(meta), -10)
compartments[nodes] = labels

out_path = f"{BASE}/results/repro_test_HOPE_compartments.txt"
with open(out_path, "w") as f:
    f.write("chr\tstart\tend\tcompartment\n")
    for coord, c in zip(meta, compartments):
        value = "NA" if c == -10 else str(c)
        f.write(f"{coord[0]}\t{coord[1]}\t{coord[2]}\t{value}\n")

print(f"wrote {out_path}")
