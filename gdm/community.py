import csv
import argparse
import networkx as nx
import pymetis
import community  # python-louvain
from collections import defaultdict


def build_graph(tweets):
    G = nx.Graph()
    status_id_map = defaultdict(list)
    screen_name_map = defaultdict(list)

    # First pass: just collect mapping info
    for t in tweets:
        node = t["id"]
        G.add_node(node)  # add node first

        if t["in_reply_to_status_id"]:
            status_id_map[t["in_reply_to_status_id"]].append(node)
        if t["in_reply_to_screen_name"]:
            screen_name_map[t["in_reply_to_screen_name"]].append(node)

    # Second pass: add edges based on reply relationships
    for t in tweets:
        node = t["id"]

        if t["in_reply_to_status_id"]:
            for target in status_id_map[t["in_reply_to_status_id"]]:
                if target != node:
                    if G.has_edge(node, target):
                        G[node][target]['weight'] += 1
                    else:
                        G.add_edge(node, target, weight=1)

        if t["in_reply_to_screen_name"]:
            for target in screen_name_map[t["in_reply_to_screen_name"]]:
                if target != node:
                    if G.has_edge(node, target):
                        G[node][target]['weight'] += 1
                    else:
                        G.add_edge(node, target, weight=1)

    return G


def pymetis_partition(G, k):
    """Partition a NetworkX graph using PyMetis."""
    nodes = list(G.nodes())
    if len(nodes) < 2 or k < 2:
        # Too few nodes or invalid k → all in one community
        return {node: 0 for node in nodes}

    # Clamp k so it never exceeds number of nodes
    k = min(k, len(nodes))

    node_index = {node: idx for idx, node in enumerate(nodes)}
    adjacency = [list(map(node_index.get, G.neighbors(node))) for node in nodes]

    _, membership = pymetis.part_graph(k, adjacency=adjacency)
    return {nodes[i]: membership[i] for i in range(len(nodes))}


def detect_communities(G, algo="louvain", k=2):
    if algo == "louvain":
        return community.best_partition(G)

    elif algo == "metis":
        # If you only want exactly k communities for the whole graph:
        if nx.is_connected(G):
            return pymetis_partition(G, k)

        # If graph has multiple components, partition each separately
        partition = {}
        comm_offset = 0
        for comp in nx.connected_components(G):
            subG = G.subgraph(comp).copy()
            this_k = min(k, len(subG))
            sub_partition = pymetis_partition(subG, this_k)
            for node, comm_id in sub_partition.items():
                partition[node] = comm_offset + comm_id
            comm_offset += this_k
        return partition

    else:
        raise ValueError("Unsupported algorithm. Use 'louvain' or 'metis'.")



def detect_communities(G, algo="louvain", k=2):
    if algo == "louvain":
        return community.best_partition(G)

    elif algo == "metis":
        partition = {}
        comm_offset = 0
        components = list(nx.connected_components(G))

        if len(components) == 1:
            partition = pymetis_partition(G, k)
        else:
            per_comp_k = max(1, k // len(components))
            remaining = k

            for comp in components:
                subG = G.subgraph(comp).copy()
                this_k = min(per_comp_k, remaining)
                if len(subG) <= this_k:
                    for idx, node in enumerate(subG.nodes()):
                        partition[node] = comm_offset + idx
                    comm_offset += len(subG)
                    remaining -= len(subG)
                else:
                    sub_partition = pymetis_partition(subG, this_k)
                    for node, comm_id in sub_partition.items():
                        partition[node] = comm_offset + comm_id
                    comm_offset += this_k
                    remaining -= this_k

        return partition

    else:
        raise ValueError("Unsupported algorithm. Use 'louvain' or 'metis'.")


def main():
    parser = argparse.ArgumentParser(description="Construct interaction graph, detect communities, and append community column to CSV.")
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("--algo", choices=["louvain", "metis"], default="louvain",
                        help="Community detection algorithm (default: louvain)")
    parser.add_argument("-k", type=int, default=2,
                        help="Number of communities for METIS (default: 2; ignored for Louvain)")
    parser.add_argument("-o", "--out-graph", default="graph.gml",
                        help="Path to save the graph file (default: graph.gml)")
    parser.add_argument("--format", choices=["gml", "graphml"], default="gml",
                        help="Graph format to save (default: gml)")
    parser.add_argument("--out-csv", default="tweets_with_communities.csv",
                        help="Path to save updated CSV with community column.")
    args = parser.parse_args()

    # Read CSV
    tweets = []
    with open(args.input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweets.append({
                "id": row["id"],
                "in_reply_to_status_id": row["in_reply_to_status_id"].strip() if row["in_reply_to_status_id"] else None,
                "in_reply_to_screen_name": row["in_reply_to_screen_name"].strip().lower() if row["in_reply_to_screen_name"] else None
            })

    # Build graph
    print(f"Building graph from {args.input_csv}...")
    G = build_graph(tweets)
    print(f"Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.")

    # Detect communities
    print(f"Running {args.algo} community detection...")
    partition = detect_communities(G, algo=args.algo, k=args.k)
    num_communities = len(set(partition.values()))
    print(f"Detected {num_communities} communities.")

    # Add community attribute to graph
    nx.set_node_attributes(G, partition, "community")

    # Save graph
    if args.format == "gml":
        nx.write_gml(G, args.out_graph)
    else:
        nx.write_graphml(G, args.out_graph)
    print(f"Graph saved to {args.out_graph} in {args.format} format.")

    # Save CSV with communities
    with open(args.out_csv, "w", newline='', encoding="utf-8") as f:
        fieldnames = ["id", "in_reply_to_status_id", "in_reply_to_screen_name", "community"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in tweets:
            comm = partition.get(t["id"], -1)
            writer.writerow({
                "id": t["id"],
                "in_reply_to_status_id": t["in_reply_to_status_id"] or "",
                "in_reply_to_screen_name": t["in_reply_to_screen_name"] or "",
                "community": comm
            })
    print(f"Updated CSV with communities saved to {args.out_csv}")

    # Save edges CSV
    edges_csv = args.out_csv.replace(".csv", "_edges.csv")
    with open(edges_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        for u, v in G.edges():
            writer.writerow([u, v])
    print(f"Edges CSV saved to {edges_csv}")


if __name__ == "__main__":
    main()
