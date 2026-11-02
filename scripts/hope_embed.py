import time, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

t0 = time.time()
BASE = "/Users/harshalshah05/Documents/4-1 SEM/Genome Analysis"

# --- 1. load bin metadata (chr,start,end) in bin-id order from existing compartments file ---
meta = []
with open(f"{BASE}/results/repro_test_compartments.txt") as f:
    next(f)
    for line in f:
        chrom, start, end, _ = line.rstrip("\n").split("\t")
        meta.append((chrom, int(start), int(end)))
N = len(meta)
print(f"[{time.time()-t0:.1f}s] loaded {N} genome-wide bins", flush=True)

# --- 2. load HiC graph edges into sparse adjacency matrix ---
rows, cols, vals = [], [], []
with open(f"{BASE}/results/repro_test_HiC_graph.txt") as f:
    for line in f:
        a, b, w = line.split()
        rows.append(int(a)); cols.append(int(b)); vals.append(float(w))
print(f"[{time.time()-t0:.1f}s] read {len(rows)} directed edge entries", flush=True)

A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
print(f"[{time.time()-t0:.1f}s] built adjacency matrix, nnz={A.nnz}", flush=True)

# --- 3. order-1 Katz proximity: S = beta*A (2-hop A@A term dropped: 726M nnz made SVD infeasible) ---
max_deg = A.sum(axis=1).max()
beta = 0.5 / max_deg
print(f"[{time.time()-t0:.1f}s] max weighted degree={max_deg:.2f}, beta={beta:.6g}", flush=True)

S = beta * A
print(f"[{time.time()-t0:.1f}s] built Katz similarity S, nnz={S.nnz}", flush=True)

# --- 4. low-rank SVD -> HOPE embedding (symmetric graph: U ~ V) ---
DIM = 100
U, s, Vt = svds(S.asfptype(), k=DIM)
order = np.argsort(-s)
U, s = U[:, order], s[order]
embedding = U * np.sqrt(np.clip(s, 0, None))
print(f"[{time.time()-t0:.1f}s] SVD done, embedding shape={embedding.shape}", flush=True)

# --- 5. write embedding in LINE-compatible format (only nodes present in graph) ---
present_nodes = sorted(set(rows) | set(cols))
out_path = f"{BASE}/results/repro_test_HOPE_graph.embedding"
with open(out_path, "w") as f:
    f.write(f"{len(present_nodes)} {DIM}\n")
    for nid in present_nodes:
        vec = embedding[nid]
        f.write(str(nid) + " " + " ".join(f"{v:.6f}" for v in vec) + "\n")
print(f"[{time.time()-t0:.1f}s] wrote embedding to {out_path}", flush=True)
