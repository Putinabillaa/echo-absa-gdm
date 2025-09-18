import csv
import argparse
import networkx as nx
import pymetis
import community  # python-louvain
from collections import defaultdict
import copy


def build_graph(tweets, available_columns):
    """Build graph with flexible column handling and multiple edge rules."""
    G = nx.Graph()

    # Initialize mapping dictionaries
    status_id_map = defaultdict(list)   # status_id -> tweet IDs
    screen_name_map = defaultdict(list) # screen_name -> tweet IDs
    name_map = defaultdict(list)        # name -> tweet IDs

    # First pass: collect mappings and add nodes
    for t in tweets:
        node = t["id"]

        G.add_node(
            node,
            text=t.get("text", ""),
            in_reply_to_screen_name=t.get("in_reply_to_screen_name", ""),
            in_reply_to_status_id=t.get("in_reply_to_status_id", ""),
            name=t.get("name", "")
        )

        # Build status_id mapping
        if "in_reply_to_status_id" in available_columns and t.get("in_reply_to_status_id"):
            status_id_map[t["in_reply_to_status_id"]].append(node)

        # Build screen_name mapping
        if "in_reply_to_screen_name" in available_columns and t.get("in_reply_to_screen_name"):
            sn = t["in_reply_to_screen_name"]
            screen_name_map[sn].append(node)

        # Build name mapping
        if "name" in available_columns and t.get("name"):
            name_map[t["name"]].append(node)

    # Second pass: add edges
    for t in tweets:
        node = t["id"]

        # Rule 1: Connect tweets replying to the same status_id
        if ("in_reply_to_status_id" in available_columns and
            t.get("in_reply_to_status_id") and
            t["in_reply_to_status_id"] in status_id_map):

            targets = status_id_map[t["in_reply_to_status_id"]]
            for target in targets:
                if target != node and not G.has_edge(node, target):
                    G.add_edge(node, target)

        # Rule 2: Connect tweets replying to the same screen_name
        if "in_reply_to_screen_name" in available_columns and t.get("in_reply_to_screen_name"):
            sn = t["in_reply_to_screen_name"]
            if sn in screen_name_map:
                targets = screen_name_map[sn]
                for target in targets:
                    if target != node and not G.has_edge(node, target):
                        G.add_edge(node, target)

        # Rule 3: Connect replies to a screen_name with tweets from that user
        if ("in_reply_to_screen_name" in available_columns and
            "name" in available_columns and
            t.get("in_reply_to_screen_name")):

            sn = t["in_reply_to_screen_name"]
            if sn in name_map:
                targets = name_map[sn]
                for target in targets:
                    if target != node and not G.has_edge(node, target):
                        G.add_edge(node, target)

        # Rule 4: Connect tweets replying to a status_id with the original tweet
        if ("in_reply_to_status_id" in available_columns and
            t.get("in_reply_to_status_id")):

            if t["in_reply_to_status_id"] in [tweet["id"] for tweet in tweets]:
                target = t["in_reply_to_status_id"]
                if target != node and not G.has_edge(node, target):
                    G.add_edge(node, target)

    return G


def pymetis_partition(G, k):
    """Partition a NetworkX graph using PyMetis."""
    nodes = list(G.nodes())
    if len(nodes) < 2 or k < 2:
        return {node: 0 for node in nodes}

    k = min(k, len(nodes))
    node_index = {node: idx for idx, node in enumerate(nodes)}
    adjacency = [list(map(node_index.get, G.neighbors(node))) for node in nodes]

    try:
        _, membership = pymetis.part_graph(k, adjacency=adjacency)
        unique_labels = sorted(set(membership))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return {nodes[i]: label_map[membership[i]] for i in range(len(nodes))}
    except Exception:
        return {node: i % k for i, node in enumerate(nodes)}


def detect_communities(G, algo="louvain", k=2):
    """Detect communities using specified algorithm."""
    if len(G.nodes()) == 0:
        return {}

    if algo == "louvain":
        return community.best_partition(G, resolution=0.5)
    elif algo == "metis":
        return pymetis_partition(G, k)
    else:
        raise ValueError("Unsupported algorithm. Use 'louvain' or 'metis'.")


def merge_graph_by_user(G, tweets, partition, available_columns):
    """Merge tweets by username+community into single nodes."""
    if "name" not in available_columns:
        info = {}
        for n in G.nodes():
            info[n] = {
                "id": n,
                "community": partition.get(n, -1),
                "text": G.nodes[n].get("text", ""),
                "in_reply_to_screen_name": G.nodes[n].get("in_reply_to_screen_name", ""),
                "in_reply_to_status_id": G.nodes[n].get("in_reply_to_status_id", ""),
                "name": G.nodes[n].get("name", "")
            }
        return copy.deepcopy(G), info

    merged_graph = nx.Graph()
    merged_texts = defaultdict(str)
    merged_info = {}
    node_map = {}
    first_ids = {}

    for t in tweets:
        comm = partition.get(t["id"], -1)
        user = t.get("name", t["id"]) or t["id"]
        group_key = (user, comm)

        if group_key not in first_ids:
            first_ids[group_key] = t["id"]

        merged_id = first_ids[group_key]
        node_map[t["id"]] = merged_id

        if "text" in available_columns and t.get("text"):
            merged_texts[merged_id] += " " + t["text"]

        if merged_id not in merged_info:
            merged_info[merged_id] = {
                "id": merged_id,
                "name": user,
                "community": comm,
                "in_reply_to_screen_name": t.get("in_reply_to_screen_name", ""),
                "in_reply_to_status_id": t.get("in_reply_to_status_id", "")
            }

    for merged_id, info in merged_info.items():
        if "text" in available_columns:
            info["text"] = merged_texts[merged_id].strip()
        for k, v in info.items():
            if v is None:
                info[k] = ""
        merged_graph.add_node(merged_id, **info)

    for u, v in G.edges():
        new_u, new_v = node_map[u], node_map[v]
        if new_u != new_v:
            merged_graph.add_edge(new_u, new_v)

    return merged_graph, merged_info


def main():
    parser = argparse.ArgumentParser(description="Build interaction graph and detect communities.")
    parser.add_argument("input_csv", help="Input CSV file path.")
    parser.add_argument("--algo", choices=["louvain", "metis"], default="louvain")
    parser.add_argument("-k", type=int, default=2, help="Number of communities (for METIS).")
    parser.add_argument("-o", "--out-graph", default="graph.gml")
    parser.add_argument("--format", choices=["gml", "graphml"], default="gml")
    parser.add_argument("--out-csv", default="tweets_with_communities.csv")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    tweets = []
    with open(args.input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        available_columns = set(reader.fieldnames)

        if "id" not in available_columns:
            raise ValueError("Required 'id' column not found in CSV")

        for row in reader:
            tweet = {"id": row["id"]}

            if "in_reply_to_screen_name" in available_columns:
                tweet["in_reply_to_screen_name"] = (
                    row["in_reply_to_screen_name"].strip().lower()
                    if row["in_reply_to_screen_name"] else ""
                )
            if "in_reply_to_status_id" in available_columns:
                tweet["in_reply_to_status_id"] = row["in_reply_to_status_id"].strip() if row["in_reply_to_status_id"] else ""
            if "name" in available_columns:
                tweet["name"] = row["name"].strip().lower() if row["name"] else ""
            if "text" in available_columns:
                tweet["text"] = row["text"]

            tweets.append(tweet)

    print(f"Available columns: {sorted(available_columns)}")

    print(f"\nBuilding graph from {args.input_csv}...")
    G = build_graph(tweets, available_columns)
    print(f"Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.")

    if args.verbose:
        print(f"Connected components: {nx.number_connected_components(G)}")

    if len(G.nodes()) > 0:
        print(f"Running {args.algo} community detection...")
        partition = detect_communities(G, algo=args.algo, k=args.k)
        num_communities = len(set(partition.values()))
        print(f"Detected {num_communities} communities.")
        nx.set_node_attributes(G, partition, "community")
        G, merged_info = merge_graph_by_user(G, tweets, partition, available_columns)
    else:
        partition = {}
        merged_info = {}

    if len(G.nodes()) > 0:
        if args.format == "gml":
            nx.write_gml(G, args.out_graph)
        else:
            nx.write_graphml(G, args.out_graph)
        print(f"Graph saved to {args.out_graph}.")
    else:
        print("Empty graph - skipping graph file output.")

    print(f"Saving updated CSV to {args.out_csv}...")
    fieldnames = ["id"]
    if "text" in available_columns: fieldnames.append("text")
    if "in_reply_to_screen_name" in available_columns: fieldnames.append("in_reply_to_screen_name")
    if "in_reply_to_status_id" in available_columns: fieldnames.append("in_reply_to_status_id")
    if "name" in available_columns: fieldnames.append("name")
    fieldnames.append("community")

    with open(args.out_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, info in merged_info.items():
            row = {
                "id": info["id"],
                "community": info["community"],
                "text": info.get("text", ""),
                "in_reply_to_screen_name": info.get("in_reply_to_screen_name", ""),
                "in_reply_to_status_id": info.get("in_reply_to_status_id", ""),
                "name": info.get("name", "")
            }
            # ✅ Only keep allowed columns
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered_row)


    print(f"Updated CSV saved to {args.out_csv}")


if __name__ == "__main__":
    main()
