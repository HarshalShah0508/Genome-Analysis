"""Query processor over results/genome.db built by build_db.py.

Importable function library:
    from scripts.query import get_label, confusion_matrix, nearest_neighbors, ...
"""
import sqlite3
from collections import defaultdict

import numpy as np

BASE = "/Users/harshalshah05/Documents/4-1 SEM/Genome Analysis"
DB_PATH = f"{BASE}/results/genome.db"
CHROM_SIZES_PATH = f"{BASE}/repos/sci/chromosome_sizes/hg19.chrom.sizes"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _with_conn(conn):
    """Return (conn, should_close) — opens a default connection if none given."""
    if conn is not None:
        return conn, False
    return _connect(), True


def _decode_vec(blob):
    return np.frombuffer(blob, dtype="<f4")


def get_bin(bin_id, conn=None):
    conn, close = _with_conn(conn)
    try:
        row = conn.execute("SELECT * FROM bins WHERE bin_id=?", (bin_id,)).fetchone()
        return dict(row) if row else None
    finally:
        if close:
            conn.close()


def find_bin_by_coord(chrom, start, end, conn=None):
    conn, close = _with_conn(conn)
    try:
        row = conn.execute(
            "SELECT bin_id FROM bins WHERE chrom=? AND start=? AND end=?",
            (chrom, start, end),
        ).fetchone()
        return row[0] if row else None
    finally:
        if close:
            conn.close()


def get_label(method, bin_id, conn=None):
    conn, close = _with_conn(conn)
    try:
        row = conn.execute(
            "SELECT label FROM labels WHERE method=? AND bin_id=?", (method, bin_id)
        ).fetchone()
        return row[0] if row else None
    finally:
        if close:
            conn.close()


def get_labels_for_chromosome(method, chrom, conn=None):
    conn, close = _with_conn(conn)
    try:
        rows = conn.execute(
            """
            SELECT b.bin_id, b.start, b.end, l.label
            FROM bins b
            LEFT JOIN labels l ON l.bin_id = b.bin_id AND l.method = ?
            WHERE b.chrom = ?
            ORDER BY b.start
            """,
            (method, chrom),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        if close:
            conn.close()


def confusion_matrix(method_a, method_b, conn=None):
    conn, close = _with_conn(conn)
    try:
        rows = conn.execute(
            """
            SELECT la.label AS label_a, lb.label AS label_b, COUNT(*) AS n
            FROM labels la
            JOIN labels lb ON la.bin_id = lb.bin_id
            WHERE la.method = ? AND lb.method = ?
              AND la.label IS NOT NULL AND lb.label IS NOT NULL
            GROUP BY la.label, lb.label
            ORDER BY la.label, lb.label
            """,
            (method_a, method_b),
        ).fetchall()
        matrix = defaultdict(dict)
        for r in rows:
            matrix[r["label_a"]][r["label_b"]] = r["n"]
        return dict(matrix)
    finally:
        if close:
            conn.close()


def label_distribution(method, chrom=None, conn=None):
    conn, close = _with_conn(conn)
    try:
        if chrom is None:
            rows = conn.execute(
                """
                SELECT COALESCE(l.label, 'NA') AS label, COUNT(*) AS n
                FROM bins b LEFT JOIN labels l ON l.bin_id = b.bin_id AND l.method = ?
                GROUP BY label ORDER BY n DESC
                """,
                (method,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT COALESCE(l.label, 'NA') AS label, COUNT(*) AS n
                FROM bins b LEFT JOIN labels l ON l.bin_id = b.bin_id AND l.method = ?
                WHERE b.chrom = ?
                GROUP BY label ORDER BY n DESC
                """,
                (method, chrom),
            ).fetchall()
        return {r["label"]: r["n"] for r in rows}
    finally:
        if close:
            conn.close()


def get_embedding(method, bin_id, conn=None):
    conn, close = _with_conn(conn)
    try:
        row = conn.execute(
            "SELECT vector FROM embeddings WHERE method=? AND bin_id=?", (method, bin_id)
        ).fetchone()
        return _decode_vec(row[0]) if row else None
    finally:
        if close:
            conn.close()


def embedding_similarity(method, bin_id_a, bin_id_b, metric="cosine", conn=None):
    conn, close = _with_conn(conn)
    try:
        rows = conn.execute(
            "SELECT bin_id, vector FROM embeddings WHERE method=? AND bin_id IN (?,?)",
            (method, bin_id_a, bin_id_b),
        ).fetchall()
        vecs = {r["bin_id"]: _decode_vec(r["vector"]) for r in rows}
        if bin_id_a not in vecs or bin_id_b not in vecs:
            return None
        va, vb = vecs[bin_id_a], vecs[bin_id_b]
        if metric == "cosine":
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            return float(np.dot(va, vb) / denom) if denom else 0.0
        elif metric == "euclidean":
            return float(np.linalg.norm(va - vb))
        raise ValueError(f"unknown metric: {metric}")
    finally:
        if close:
            conn.close()


def nearest_neighbors(method, bin_id, k=10, metric="cosine", conn=None):
    conn, close = _with_conn(conn)
    try:
        rows = conn.execute(
            "SELECT bin_id, vector FROM embeddings WHERE method=?", (method,)
        ).fetchall()
        ids = np.array([r["bin_id"] for r in rows])
        mat = np.stack([_decode_vec(r["vector"]) for r in rows])
        query_idx = np.where(ids == bin_id)[0]
        if len(query_idx) == 0:
            return None
        q = mat[query_idx[0]]

        if metric == "cosine":
            norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(q)
            norms[norms == 0] = 1e-12
            sims = (mat @ q) / norms
            dist = 1 - sims
        elif metric == "euclidean":
            dist = np.linalg.norm(mat - q, axis=1)
        else:
            raise ValueError(f"unknown metric: {metric}")

        order = np.argsort(dist)
        result = []
        for i in order:
            if ids[i] == bin_id:
                continue
            result.append((int(ids[i]), float(dist[i])))
            if len(result) >= k:
                break
        return result
    finally:
        if close:
            conn.close()


def get_neighbors(bin_id, min_weight=None, conn=None):
    conn, close = _with_conn(conn)
    try:
        if min_weight is None:
            rows = conn.execute(
                "SELECT node_b, weight FROM graph_edges WHERE node_a=? ORDER BY weight DESC",
                (bin_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT node_b, weight FROM graph_edges WHERE node_a=? AND weight>=? "
                "ORDER BY weight DESC",
                (bin_id, min_weight),
            ).fetchall()
        return [(r["node_b"], r["weight"]) for r in rows]
    finally:
        if close:
            conn.close()


def degree(bin_id, weighted=True, conn=None):
    conn, close = _with_conn(conn)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt, SUM(weight) AS wsum FROM graph_edges WHERE node_a=?",
            (bin_id,),
        ).fetchone()
        if weighted:
            return row["wsum"] or 0.0
        return row["cnt"] or 0
    finally:
        if close:
            conn.close()


def top_contacts(bin_id, n=10, conn=None):
    conn, close = _with_conn(conn)
    try:
        rows = conn.execute(
            "SELECT node_b, weight FROM graph_edges WHERE node_a=? ORDER BY weight DESC LIMIT ?",
            (bin_id, n),
        ).fetchall()
        return [(r["node_b"], r["weight"]) for r in rows]
    finally:
        if close:
            conn.close()


def get_metrics(method=None, conn=None):
    conn, close = _with_conn(conn)
    try:
        if method is None:
            rows = conn.execute("SELECT method, metric_name, value FROM metrics").fetchall()
        else:
            rows = conn.execute(
                "SELECT method, metric_name, value FROM metrics WHERE method=?", (method,)
            ).fetchall()
        result = defaultdict(dict)
        for r in rows:
            result[r["method"]][r["metric_name"]] = r["value"]
        return dict(result) if method is None else result.get(method, {})
    finally:
        if close:
            conn.close()


def compare_metrics(metric_name, conn=None):
    conn, close = _with_conn(conn)
    try:
        rows = conn.execute(
            "SELECT method, value FROM metrics WHERE metric_name=? ORDER BY value DESC",
            (metric_name,),
        ).fetchall()
        return [(r["method"], r["value"]) for r in rows]
    finally:
        if close:
            conn.close()


def disagreements(method_a, method_b, label_map=None, conn=None):
    conn, close = _with_conn(conn)
    try:
        rows = conn.execute(
            """
            SELECT b.chrom, b.start, b.end, la.label AS label_a, lb.label AS label_b
            FROM bins b
            JOIN labels la ON la.bin_id = b.bin_id AND la.method = ?
            JOIN labels lb ON lb.bin_id = b.bin_id AND lb.method = ?
            WHERE la.label IS NOT NULL AND lb.label IS NOT NULL
            """,
            (method_a, method_b),
        ).fetchall()
        result = []
        for r in rows:
            la, lb = r["label_a"], r["label_b"]
            if label_map is not None:
                la = label_map.get(la, la)
                lb = label_map.get(lb, lb)
            if la != lb:
                result.append({"chrom": r["chrom"], "start": r["start"], "end": r["end"],
                                "label_a": la, "label_b": lb})
        return result
    finally:
        if close:
            conn.close()


def chromosome_summary(method, chrom, conn=None):
    conn, close = _with_conn(conn)
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total_bins,
                   SUM(CASE WHEN l.label IS NULL THEN 1 ELSE 0 END) AS na_count
            FROM bins b LEFT JOIN labels l ON l.bin_id = b.bin_id AND l.method = ?
            WHERE b.chrom = ?
            """,
            (method, chrom),
        ).fetchone()
        return {
            "chrom": chrom,
            "total_bins": row["total_bins"],
            "na_count": row["na_count"],
            "label_distribution": label_distribution(method, chrom, conn=conn),
        }
    finally:
        if close:
            conn.close()


def chromosome_sizes():
    sizes = {}
    with open(CHROM_SIZES_PATH) as f:
        for line in f:
            chrom, size = line.split()
            sizes[chrom] = int(size)
    return sizes
