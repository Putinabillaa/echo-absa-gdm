import argparse
import csv
import numpy as np
from pathlib import Path
from itertools import combinations
import ast

def cosine_similarity(v1, v2):
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def calculate_global_mean(values):
    """Calculate mean of values"""
    if not values:
        return 0.0
    return float(np.mean(values))

def calculate_consensus(user_pairs, aspects_data, p_inner=2):
    if not user_pairs:
        return None, None

    m = len(next(iter(aspects_data.values())))
    SV = []
    for u, v in user_pairs:
        if u in aspects_data and v in aspects_data:
            sims = [
                cosine_similarity(aspects_data[u][k], aspects_data[v][k])
                if not (np.allclose(aspects_data[u][k], 0) and np.allclose(aspects_data[v][k], 0))
                else None
                for k in range(m)
            ]
            SV.append(sims)

    if not SV:
        return None, None

    SV = np.array(SV, dtype=object)
    ac_values = []
    active_aspects = 0

    for z in range(m):
        sims_for_aspect = np.array([s for s in SV[:, z] if s is not None], dtype=float)
        if sims_for_aspect.size:
            active_aspects += 1
            ac_values.append((np.sum(sims_for_aspect ** p_inner) / len(sims_for_aspect)) ** (1/p_inner))
        else:
            ac_values.append(0.0)

    cc = ((np.sum(np.array([ac for ac in ac_values if ac > 0]) ** p_inner) / active_aspects) ** (1/p_inner)
          if active_aspects else 0.0)

    return cc, ac_values

def main():
    parser = argparse.ArgumentParser(description="Calculate GDM Consensus (mean only)")
    parser.add_argument("aspects_csv")
    parser.add_argument("-p", "--p-inner", type=float, default=1, help="Power parameter for inner consensus calculation (default: 1)")
    parser.add_argument("--min-members", type=int, default=5, help="Minimum members in a community to be considered (default: 5)")
    parser.add_argument("-o", "--output", default="summary.txt")
    parser.add_argument("--details", default="details.txt")
    args = parser.parse_args()

    aspects_data, communities = {}, {}
    with open(args.aspects_csv, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        aspect_cols = [c for c in reader.fieldnames if c not in ("id", "community")]
        for row in reader:
            uid = row["id"]
            aspects = []
            for col in aspect_cols:
                val = row[col].strip()
                try:
                    vec = ast.literal_eval(val) if val.lower() != "nan" and val != "" else [0.0, 0.0, 0.0]
                    aspects.append(vec if isinstance(vec, list) else [float(vec)])
                except:
                    aspects.append([0.0, 0.0, 0.0])
            comm = row["community"]
            aspects_data[uid] = aspects
            communities.setdefault(comm, {})[uid] = aspects

    filtered = {c: m for c, m in communities.items() if len(m) >= args.min_members}
    if not filtered:
        print("No communities meet the minimum member requirement.")
        return

    detail_lines, within_ccs, between_ccs = [], [], []

    detail_lines.append("=== WITHIN-COMMUNITY CONSENSUS ===")
    for comm, members in filtered.items():
        comm_pairs = list(combinations(members.keys(), 2))
        cc, ac_values = calculate_consensus(comm_pairs, members, p_inner=args.p_inner)
        if cc is not None:
            within_ccs.append(cc)
            detail_lines.append(f"Community {comm} (size={len(members)}): CC(within)={cc:.4f}")
            # NEW: print each aspect’s consensus
            for idx, val in enumerate(ac_values, start=1):
                detail_lines.append(f"   Aspect {idx}: {val:.4f}")

    detail_lines.append("\n=== BETWEEN-COMMUNITY CONSENSUS ===")
    keys = list(filtered.keys())
    for a, b in combinations(keys, 2):
        cross_pairs = [(u, v) for u in filtered[a] for v in filtered[b]]
        cc_between, _ = calculate_consensus(cross_pairs, aspects_data, p_inner=args.p_inner)
        if cc_between is not None:
            between_ccs.append(cc_between)
            detail_lines.append(f"Between {a} & {b}: CC={cc_between:.4f}")

    # Only calculate mean
    within_mean = calculate_global_mean(within_ccs)
    between_mean = calculate_global_mean(between_ccs)
    ecr_mean = within_mean / between_mean if between_mean > 0 else float("inf")

    summary_lines = [
        "=== GLOBAL WITHIN-COMMUNITY CONSENSUS ===",
        f"Mean = {within_mean:.4f}",
        "",
        "=== GLOBAL BETWEEN-COMMUNITY CONSENSUS ===",
        f"Mean = {between_mean:.4f}",
        "",
        "=== ECHO CHAMBER RATIOS (ECR) ===",
        f"ECR (Mean) = {ecr_mean:.4f}"
    ]

    Path(args.output).write_text("\n".join(summary_lines), encoding="utf-8")
    Path(args.details).write_text("\n".join(detail_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\nSummary saved to {args.output}")
    print(f"Details saved to {args.details}")

if __name__ == "__main__":
    main()
