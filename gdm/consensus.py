import argparse
import csv
import numpy as np
from pathlib import Path
from itertools import combinations
import ast


def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def calculate_consensus(user_pairs, aspects_data, p=2):
    """Calculate AC values and CC from given user pairs, skipping zero–zero pairs."""
    if not user_pairs:
        return None, None, None

    m = len(next(iter(aspects_data.values())))  # number of aspects
    SV = []
    for u, v in user_pairs:
        if u in aspects_data and v in aspects_data:
            sims = []
            for k in range(m):
                v1, v2 = aspects_data[u][k], aspects_data[v][k]
                # Skip if both are zero vectors
                if np.allclose(v1, 0) and np.allclose(v2, 0):
                    sims.append(None)  # mark as skipped
                else:
                    sims.append(cosine_similarity(v1, v2))
            SV.append(sims)

    if not SV:
        return None, None, None

    SV = np.array(SV, dtype=object)
    ac_values = []
    for z in range(m):
        sims_for_aspect = [s for s in SV[:, z] if s is not None]
        if not sims_for_aspect:
            ac_values.append(0.0)
        else:
            sims_for_aspect = np.array(sims_for_aspect, dtype=float)
            ac_z = (np.sum(np.abs(sims_for_aspect) ** p) / len(sims_for_aspect)) ** (1 / p)
            ac_values.append(ac_z)

    cc = (np.sum(np.array(ac_values) ** p) / m) ** (1 / p)
    return cc, ac_values, SV.tolist()



def main():
    parser = argparse.ArgumentParser(description="Calculate GDM Consensus (within & between communities)")
    parser.add_argument("edges_csv", help="Path to edges CSV file (source,target) — unused for consensus calculation")
    parser.add_argument("aspects_csv", help="Path to ABSA & community CSV file (id,...,community)")
    parser.add_argument("-p", type=float, default=2, help="p-norm value (default=2)")
    parser.add_argument("-o", "--output", default="summary.txt", help="Summary TXT output file")
    parser.add_argument("--details", default="details.txt", help="Detailed TXT output file")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Only include top N communities by within CC for consensus calculations")
    args = parser.parse_args()

    # Load aspects & communities
    aspects_data = {}
    communities = {}
    with open(args.aspects_csv, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        aspect_cols = [col for col in reader.fieldnames if col not in ("id", "community")]

        for row in reader:
            uid = row["id"]
            aspects = []
            for col in aspect_cols:
                val = row[col].strip()
                if val == "" or val.lower() == "nan":
                    aspects.append([0.0, 0.0, 0.0])
                else:
                    try:
                        vec = ast.literal_eval(val)
                        aspects.append(vec if isinstance(vec, list) else [float(vec)])
                    except Exception:
                        aspects.append([0.0, 0.0, 0.0])
            comm = row["community"]
            aspects_data[uid] = aspects
            communities.setdefault(comm, {})[uid] = aspects

    detail_lines = ["=== WITHIN-COMMUNITY CONSENSUS ==="]
    community_ac = {}
    within_results = []

    # ===== WITHIN-COMMUNITY =====
    for comm, members in communities.items():
        comm_pairs = list(combinations(members.keys(), 2))
        cc, ac_values, SV = calculate_consensus(comm_pairs, members, p=args.p)
        if cc is None:
            continue
        community_ac[comm] = ac_values
        within_results.append((comm, cc))
        detail_lines.append(f"\nCommunity {comm}:")
        detail_lines.append(f"  Level 1 - Similarity vectors (per pair): {SV}")
        for idx, ac in enumerate(ac_values, 1):
            detail_lines.append(f"  Level 2 - Aspect {idx} AC = {ac:.4f}")
        detail_lines.append(f"  Level 3 - CC (within) = {cc:.4f}")

    within_results.sort(key=lambda x: x[1], reverse=True)

    # ===== FILTER TOP-N COMMUNITIES =====
    if args.top_n is not None:
        sorted_comms = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
        communities = dict(sorted_comms[:args.top_n])
        # Keep community_ac in sync
        community_ac = {k: v for k, v in community_ac.items() if k in communities}

    summary_lines = [f"Community {comm} (within): CC = {cc:.4f}" for comm, cc in within_results if comm in communities]

    # ===== BETWEEN-COMMUNITIES =====
    detail_lines.append("\n=== IN-BETWEEN-COMMUNITY CONSENSUS ===")
    between_values = []
    for comm_a, comm_b in combinations(communities.keys(), 2):
        cross_pairs = [(u, v) for u in communities[comm_a].keys() for v in communities[comm_b].keys()]
        cc_between, _, _ = calculate_consensus(cross_pairs, aspects_data, p=args.p)
        if cc_between is not None:
            between_values.append(cc_between)
            summary_lines.append(f"Communities {comm_a}-{comm_b} (between): CC = {cc_between:.4f}")
            detail_lines.append(f"Between {comm_a} & {comm_b}: CC (between) = {cc_between:.4f}")

    if between_values:
        cc_global_between = sum(between_values) / len(between_values)
        summary_lines.append(f"Global in-between communities: CC = {cc_global_between:.4f}")
        detail_lines.append(f"\nGlobal in-between CC = {cc_global_between:.4f}")

    # Save
    Path(args.output).write_text("\n".join(summary_lines), encoding="utf-8")
    Path(args.details).write_text("\n".join(detail_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"\nSummary saved to {args.output}")
    print(f"Details saved to {args.details}")

if __name__ == "__main__":
    main()
