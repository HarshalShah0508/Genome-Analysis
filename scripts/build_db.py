"""Build results/genome.db from the SCI/HOPE reproduction result files.

Run: python3 scripts/build_db.py [--skip-graph]
"""
import argparse
import os
import sqlite3
import sys
import time

import numpy as np

BASE = "/Users/harshalshah05/Documents/4-1 SEM/Genome Analysis"
DB_PATH = f"{BASE}/results/genome.db"

SCHEMA = """
CREATE TABLE bins (
    bin_id INTEGER PRIMARY KEY,
    chrom  TEXT NOT NULL,
    start  INTEGER NOT NULL,
    end    INTEGER NOT NULL
);

CREATE TABLE labels (
    method TEXT NOT NULL,
    bin_id INTEGER NOT NULL,
    label  TEXT,
    PRIMARY KEY (method, bin_id),
    FOREIGN KEY (bin_id) REFERENCES bins(bin_id)
);

CREATE TABLE embeddings (
    method TEXT NOT NULL,
    bin_id INTEGER NOT NULL,
    dim    INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (method, bin_id),
    FOREIGN KEY (bin_id) REFERENCES bins(bin_id)
);

CREATE TABLE graph_edges (
    node_a INTEGER NOT NULL,
    node_b INTEGER NOT NULL,
    weight REAL NOT NULL
);

CREATE TABLE metrics (
    method      TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value       REAL NOT NULL,
    PRIMARY KEY (method, metric_name)
);
"""

INDEXES = """
CREATE INDEX idx_bins_coord ON bins(chrom, start, end);
CREATE INDEX idx_bins_chrom ON bins(chrom);
CREATE INDEX idx_labels_bin ON labels(bin_id);
CREATE INDEX idx_labels_method ON labels(method, label);
CREATE INDEX idx_embeddings_bin ON embeddings(bin_id);
CREATE INDEX idx_edges_a ON graph_edges(node_a);
CREATE INDEX idx_edges_b ON graph_edges(node_b);
"""

METRICS = {
    "line": {
        "clustered_bins": 26977,
        "silhouette_score": 0.00569,
        "davies_bouldin_index": 9.0750,
        "overlap_bins_with_reference": 26226,
        "ari": -0.00003,
        "nmi": 0.00017,
    },
    "hope": {
        "clustered_bins": 26977,
        "silhouette_score": 0.06380,
        "davies_bouldin_index": 5.6569,
        "overlap_bins_with_reference": 26226,
        "ari": 0.34236,
        "nmi": 0.36976,
    },
}


def log(t0, msg):
    print(f"[{time.time()-t0:.1f}s] {msg}", flush=True)


def load_bins(conn, t0):
    path = f"{BASE}/results/repro_test_compartments.txt"
    rows = []
    with open(path) as f:
        next(f)
        for i, line in enumerate(f):
            chrom, start, end, _ = line.rstrip("\n").split("\t")
            rows.append((i, chrom, int(start), int(end)))
    conn.executemany("INSERT INTO bins VALUES (?,?,?,?)", rows)
    log(t0, f"loaded bins: {len(rows)} rows")
    return len(rows)


def load_compartment_labels(conn, t0, path, method):
    rows = []
    with open(path) as f:
        next(f)
        for i, line in enumerate(f):
            _, _, _, label = line.rstrip("\n").split("\t")
            rows.append((method, i, None if label == "NA" else label))
    conn.executemany("INSERT INTO labels VALUES (?,?,?)", rows)
    log(t0, f"loaded labels[{method}]: {len(rows)} rows")


def load_reference_labels(conn, t0):
    path = f"{BASE}/repos/sci/predictions/GM12878_SCI_sub_compartments.bed"
    cur = conn.cursor()
    rows = []
    unmatched = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            chrom, start, end, label = parts[0], parts[1], parts[2], parts[3]
            cur.execute(
                "SELECT bin_id FROM bins WHERE chrom=? AND start=? AND end=?",
                (chrom, int(start), int(end)),
            )
            result = cur.fetchone()
            if result is None:
                unmatched.append((chrom, start, end))
                continue
            rows.append(("sci_reference", result[0], label))
    if unmatched:
        raise RuntimeError(
            f"reference BED: {len(unmatched)} rows failed coordinate match, "
            f"e.g. {unmatched[:5]}"
        )
    conn.executemany("INSERT INTO labels VALUES (?,?,?)", rows)
    log(t0, f"loaded labels[sci_reference]: {len(rows)} rows")
    assert len(rows) == 26226, f"expected 26226 reference labels, got {len(rows)}"


def load_embedding_file(conn, t0, path, method):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            bin_id = int(parts[0])
            vec = np.array(parts[1:], dtype="<f4")
            rows.append((method, bin_id, len(vec), vec.tobytes()))
    conn.executemany("INSERT INTO embeddings VALUES (?,?,?,?)", rows)
    log(t0, f"loaded embeddings[{method}]: {len(rows)} rows")


def load_graph_edges(conn, t0):
    path = f"{BASE}/results/repro_test_HiC_graph.txt"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found (gitignored, local-only). "
            "Run from a machine that has it, or pass --skip-graph."
        )
    batch = []
    total = 0
    BATCH_SIZE = 500_000
    with open(path) as f:
        for line in f:
            a, b, w = line.split()
            batch.append((int(a), int(b), float(w)))
            if len(batch) >= BATCH_SIZE:
                conn.executemany("INSERT INTO graph_edges VALUES (?,?,?)", batch)
                total += len(batch)
                batch = []
                log(t0, f"graph_edges: {total} rows loaded so far")
    if batch:
        conn.executemany("INSERT INTO graph_edges VALUES (?,?,?)", batch)
        total += len(batch)
    log(t0, f"loaded graph_edges: {total} rows")
    return total


def load_metrics(conn, t0):
    rows = [
        (method, metric_name, value)
        for method, metric_dict in METRICS.items()
        for metric_name, value in metric_dict.items()
    ]
    conn.executemany("INSERT INTO metrics VALUES (?,?,?)", rows)
    log(t0, f"loaded metrics: {len(rows)} rows")


def verify(conn, skip_graph):
    cur = conn.cursor()
    checks = [
        ("SELECT COUNT(*) FROM bins", 28823),
        ("SELECT COUNT(*) FROM labels WHERE method='sci_line_repro'", 28823),
        ("SELECT COUNT(*) FROM labels WHERE method='hope_repro'", 28823),
        ("SELECT COUNT(*) FROM labels WHERE method='sci_reference'", 26226),
        ("SELECT COUNT(*) FROM embeddings WHERE method='line_order1'", 26977),
        ("SELECT COUNT(*) FROM embeddings WHERE method='line_order2'", 26977),
        ("SELECT COUNT(*) FROM embeddings WHERE method='hope'", 26977),
        ("SELECT COUNT(*) FROM metrics", 12),
    ]
    if not skip_graph:
        checks.append(("SELECT COUNT(*) FROM graph_edges", 16704136))

    for sql, expected in checks:
        actual = cur.execute(sql).fetchone()[0]
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  [{status}] {sql} -> {actual} (expected {expected})")
        if actual != expected:
            raise RuntimeError(f"verification failed: {sql} -> {actual} != {expected}")

    dist = dict(cur.execute(
        "SELECT COALESCE(label,'NA'), COUNT(*) FROM labels "
        "WHERE method='hope_repro' GROUP BY 1"
    ).fetchall())
    expected_dist = {"0": 647, "1": 14, "2": 13984, "3": 8103, "4": 4229, "NA": 1846}
    assert dist == expected_dist, f"HOPE label distribution mismatch: {dist}"
    print(f"  [OK] hope_repro label distribution matches {expected_dist}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-graph", action="store_true",
                         help="skip loading the 337MB graph_edges table")
    args = parser.parse_args()

    t0 = time.time()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    log(t0, "created schema")

    with conn:
        load_bins(conn, t0)
        load_compartment_labels(conn, t0, f"{BASE}/results/repro_test_compartments.txt", "sci_line_repro")
        load_compartment_labels(conn, t0, f"{BASE}/results/repro_test_HOPE_compartments.txt", "hope_repro")
        load_reference_labels(conn, t0)
        load_embedding_file(conn, t0, f"{BASE}/results/repro_test_HiC_graph_order_1_samples_1M.embedding", "line_order1")
        load_embedding_file(conn, t0, f"{BASE}/results/repro_test_HiC_graph_order_2_samples_1M.embedding", "line_order2")
        load_embedding_file(conn, t0, f"{BASE}/results/repro_test_HOPE_graph.embedding", "hope")
        load_metrics(conn, t0)
        if not args.skip_graph:
            load_graph_edges(conn, t0)
        else:
            log(t0, "skipped graph_edges per --skip-graph")

    log(t0, "building indexes...")
    conn.executescript(INDEXES)
    conn.execute("ANALYZE")
    conn.commit()
    log(t0, "indexes built")

    print("\nverification:")
    verify(conn, args.skip_graph)

    conn.close()
    log(t0, f"done -> {DB_PATH}")


if __name__ == "__main__":
    main()
