import argparse
import csv
import numpy as np
from pathlib import Path
from itertools import combinations
import ast
import random
import math

def cosine_similarity(v1, v2):
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def power_mean(values, p, weights=None, eps=1e-12):
    vals = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(vals)
    else:
        weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(vals)
    vals, weights = vals[mask], weights[mask]
    if vals.size == 0:
        return 0.0
    if p == 0:
        vals = np.clip(vals, eps, None)
        return float(np.exp(np.sum(weights * np.log(vals)) / np.sum(weights)))
    return float((np.sum(weights * (vals ** p)) / np.sum(weights)) ** (1.0 / p))

def calculate_consensus(user_pairs, aspects_data, p=2):
    if not user_pairs:
        return None, None, None

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
        return None, None, None

    SV = np.array(SV, dtype=object)
    ac_values = []
    active_aspects = 0

    for z in range(m):
        sims_for_aspect = np.array([s for s in SV[:, z] if s is not None], dtype=float)
        if sims_for_aspect.size:
            active_aspects += 1
            ac_values.append((np.sum(np.abs(sims_for_aspect) ** p) / len(sims_for_aspect)) ** (1/p))
        else:
            ac_values.append(0.0)

    cc = ((np.sum(np.array([ac for ac in ac_values if ac > 0]) ** p) / active_aspects) ** (1/p)
          if active_aspects else 0.0)

    return cc, ac_values, SV.tolist()

def bootstrap_ci(data, n_boot=1000, ci=95, agg_fn=None):
    if not data:
        return 0.0, (0.0, 0.0)
    arr = np.array(data, dtype=float)
    if agg_fn is None:
        agg_fn = lambda x: float(np.mean(x))
    boot_stats = [agg_fn(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boot_stats, (100-ci)/2)
    upper = np.percentile(boot_stats, 100-(100-ci)/2)
    return float(agg_fn(arr)), (lower, upper)

def size_balanced_bootstrap_cc(communities, aspects_data, p, min_members=2, n_boot=1000):
    filtered = {c: m for c, m in communities.items() if len(m) >= min_members}
    if not filtered:
        return []

    min_size = min(len(members) for members in filtered.values())
    if min_size < 2:
        return []

    results = []
    members_list = [list(m.keys()) for m in filtered.values()]
    for _ in range(n_boot):
        boot_within = []
        for mems in members_list:
            sampled = random.sample(mems, min_size)
            cc, _, _ = calculate_consensus(list(combinations(sampled, 2)), aspects_data, p)
            if cc is not None:
                boot_within.append(cc)
        if boot_within:
            results.append(np.mean(boot_within))
    return results

def main():
    parser = argparse.ArgumentParser(description="Calculate GDM Consensus")
    parser.add_argument("edges_csv")
    parser.add_argument("aspects_csv")
    parser.add_argument("-p", type=float, default=2)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--min-members", type=int, default=3)
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
        cc, _, _ = calculate_consensus(comm_pairs, members, p=args.p)
        if cc is not None:
            within_ccs.append(cc)
            detail_lines.append(f"Community {comm} (size={len(members)}): CC(within)={cc:.4f}")

    detail_lines.append("\n=== BETWEEN-COMMUNITY CONSENSUS ===")
    keys = list(filtered.keys())
    for a, b in combinations(keys, 2):
        cross_pairs = [(u, v) for u in filtered[a] for v in filtered[b]]
        cc_between, _, _ = calculate_consensus(cross_pairs, aspects_data, p=args.p)
        if cc_between is not None:
            between_ccs.append(cc_between)
            detail_lines.append(f"Between {a} & {b}: CC={cc_between:.4f}")

    # Global aggregation using power mean
    within_weights = [math.log(len(members)+1) for members in filtered.values()]
    global_within_cc = power_mean(within_ccs, p=args.p, weights=within_weights)
    global_between_cc = power_mean(between_ccs, p=args.p) if between_ccs else 0.0

    if args.bootstrap > 0:
        pm_agg = lambda x: power_mean(x, p=args.p)
        b_mean, b_ci = bootstrap_ci(between_ccs, n_boot=args.bootstrap, agg_fn=pm_agg) if between_ccs else (0.0, (0.0, 0.0))

        balanced = size_balanced_bootstrap_cc(filtered, aspects_data, p=args.p,
                                              min_members=args.min_members, n_boot=args.bootstrap)
        if balanced:
            w_mean_balanced = np.mean(balanced)
            w_ci_balanced = (np.percentile(balanced, 2.5), np.percentile(balanced, 97.5))
            ecr_list = [w/b_mean if b_mean>0 else float("inf") for w in balanced]
            ecr_mean = np.mean(ecr_list)
            ecr_ci = (np.percentile(ecr_list,2.5), np.percentile(ecr_list,97.5))
        else:
            w_mean_balanced = ecr_mean = 0.0
            w_ci_balanced = ecr_ci = (0.0, 0.0)

        summary_lines = [
            f"Global Within CC (power mean, weighted by log size) = {global_within_cc:.4f}",
            f"Global Within CC (size-balanced bootstrap) = {w_mean_balanced:.4f} (95% CI: {w_ci_balanced[0]:.4f} - {w_ci_balanced[1]:.4f})",
            f"Global Between CC (power mean) = {b_mean:.4f} (95% CI: {b_ci[0]:.4f} - {b_ci[1]:.4f})",
            f"ECR (balanced) = {ecr_mean:.4f} (95% CI: {ecr_ci[0]:.4f} - {ecr_ci[1]:.4f})"
        ]
    else:
        ecr = global_within_cc/global_between_cc if global_between_cc>0 else float("inf")
        summary_lines = [
            f"Global Within CC (power mean, weighted by log size) = {global_within_cc:.4f}",
            f"Global Between CC (power mean) = {global_between_cc:.4f}",
            f"ECR (Weighted, power mean) = {ecr:.4f}"
        ]

    Path(args.output).write_text("\n".join(summary_lines), encoding="utf-8")
    Path(args.details).write_text("\n".join(detail_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\nSummary saved to {args.output}")
    print(f"Details saved to {args.details}")

if __name__ == "__main__":
    main()
