import csv
import argparse
import networkx as nx
import pymetis
import community  # python-louvain
from collections import defaultdict

def build_graph(tweets, available_columns):
    """Build graph with flexible column handling and multiple edge rules."""
    G = nx.Graph()
    
    # Initialize mapping dictionaries based on available columns
    status_id_map = defaultdict(list)  # Maps status_id -> list of tweet IDs that have this status_id
    screen_name_map = defaultdict(list)  # Maps screen_name -> list of tweet IDs from this user
    name_map = defaultdict(list)  # Maps name -> list of tweet IDs from this user
    
    # First pass: collect all mappings and add nodes
    for t in tweets:
        node = t["id"]
        G.add_node(node)
        
        # Build status_id mapping if column exists
        if "in_reply_to_status_id" in available_columns and t.get("in_reply_to_status_id"):
            status_id_map[t["in_reply_to_status_id"]].append(node)
        
        # Build screen_name mapping if column exists
        if "in_reply_to_screen_name" in available_columns and t.get("in_reply_to_screen_name"):
            screen_name_map[t["in_reply_to_screen_name"]].append(node)
        
        # Build name mapping if column exists
        if "name" in available_columns and t.get("name"):
            name_map[t["name"]].append(node)
    
    # Second pass: add edges based on reply relationships
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
        if ("in_reply_to_screen_name" in available_columns and 
            t.get("in_reply_to_screen_name") and 
            t["in_reply_to_screen_name"] in screen_name_map):
            
            targets = screen_name_map[t["in_reply_to_screen_name"]]
            for target in targets:
                if target != node and not G.has_edge(node, target):
                    G.add_edge(node, target)
        
        # Rule 3: Connect tweets replying to a screen_name with tweets from users with that name
        if ("in_reply_to_screen_name" in available_columns and 
            "name" in available_columns and
            t.get("in_reply_to_screen_name") and
            t["in_reply_to_screen_name"] in name_map):
            
            targets = name_map[t["in_reply_to_screen_name"]]
            for target in targets:
                if target != node and not G.has_edge(node, target):
                    G.add_edge(node, target)
        
        # Rule 4: Connect tweets replying to a status_id with tweets from users with matching ID
        if ("in_reply_to_status_id" in available_columns and
            t.get("in_reply_to_status_id")):
            
            # Check if there's a tweet with ID matching the replied-to status_id
            if t["in_reply_to_status_id"] in [tweet["id"] for tweet in tweets]:
                target = t["in_reply_to_status_id"]
                if target != node and not G.has_edge(node, target):
                    G.add_edge(node, target)
    
    return G

def pymetis_partition(G, k):
    """Partition a NetworkX graph using PyMetis, ensuring exactly k communities."""
    nodes = list(G.nodes())
    if len(nodes) < 2 or k < 2:
        return {node: 0 for node in nodes}
    
    k = min(k, len(nodes))  # cannot have more communities than nodes
    node_index = {node: idx for idx, node in enumerate(nodes)}
    adjacency = [list(map(node_index.get, G.neighbors(node))) for node in nodes]
    
    try:
        _, membership = pymetis.part_graph(k, adjacency=adjacency)
        # Ensure communities are labeled 0..k-1
        unique_labels = sorted(set(membership))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return {nodes[i]: label_map[membership[i]] for i in range(len(nodes))}
    except Exception:
        # Fallback if PyMetis fails
        return {node: i % k for i, node in enumerate(nodes)}

def detect_communities(G, algo="louvain", k=2):
    """Detect communities using specified algorithm."""
    if len(G.nodes()) == 0:
        return {}
    
    if algo == "louvain":
        return community.best_partition(G)
    elif algo == "metis":
        return pymetis_partition(G, k)
    else:
        raise ValueError("Unsupported algorithm. Use 'louvain' or 'metis'.")

def merge_graph_by_user(G, tweets, partition, available_columns):
    """
    Merge tweets from the same username inside the same community into single nodes,
    concatenate their texts, and rebuild the graph.
    If 'name' is not in available_columns, no merging is done (returns original graph with consistent info dict).
    """
    from collections import defaultdict
    import copy

    # Case 1: no username info -> skip merging
    if "name" not in available_columns:
        info = {}
        for n in G.nodes():
            info[n] = {
                "id": n,
                "community": partition.get(n, -1),
            }
            if "text" in available_columns:
                info[n]["text"] = G.nodes[n].get("text", "")
            if "in_reply_to_screen_name" in available_columns:
                info[n]["in_reply_to_screen_name"] = G.nodes[n].get("in_reply_to_screen_name", "")
            if "in_reply_to_status_id" in available_columns:
                info[n]["in_reply_to_status_id"] = G.nodes[n].get("in_reply_to_status_id", "")
            if "name" in available_columns:
                info[n]["name"] = G.nodes[n].get("name", "")
        return copy.deepcopy(G), info

    # Case 2: merge by username + community
    merged_graph = nx.Graph()
    merged_texts = defaultdict(str)
    merged_info = {}
    node_map = {}   # maps old node -> new merged node id
    first_ids = {}  # store first tweet id for (user, comm) groups

    for t in tweets:
        comm = partition.get(t["id"], -1)
        user = t.get("name", t["id"]) or t["id"]
        group_key = (user, comm)

        # assign representative id (first tweet id encountered)
        if group_key not in first_ids:
            first_ids[group_key] = t["id"]

        merged_id = first_ids[group_key]
        node_map[t["id"]] = merged_id

        # concatenate text
        if "text" in available_columns and t.get("text"):
            merged_texts[merged_id] += " " + t["text"]

        # store representative attributes (once per merged node)
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
            info["text"] = merged_texts[merged_id].strip() if merged_texts[merged_id] else ""
        # Ensure no None values
        for k, v in info.items():
            if v is None:
                info[k] = ""
        merged_graph.add_node(merged_id, **info)


    # rebuild edges between merged nodes
    for u, v in G.edges():
        new_u, new_v = node_map[u], node_map[v]
        if new_u != new_v:  # avoid self-loops from merging
            merged_graph.add_edge(new_u, new_v)

    return merged_graph, merged_info


def main():
    parser = argparse.ArgumentParser(description="Construct interaction graph with flexible edge rules, detect communities, and append community column to CSV.")
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
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed information about edge building rules")
    
    args = parser.parse_args()
    
    # Read CSV and detect available columns
    tweets = []
    available_columns = set()
    
    with open(args.input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        available_columns = set(reader.fieldnames)
        
        # Check for required ID column
        if "id" not in available_columns:
            raise ValueError("Required 'id' column not found in CSV")
        
        for row in reader:
            tweet = {"id": row["id"]}
            
            # Add optional columns if they exist
            if "in_reply_to_screen_name" in available_columns:
                tweet["in_reply_to_screen_name"] = row["in_reply_to_screen_name"].strip().lower() if row["in_reply_to_screen_name"] else None
            
            if "in_reply_to_status_id" in available_columns:
                tweet["in_reply_to_status_id"] = row["in_reply_to_status_id"].strip() if row["in_reply_to_status_id"] else None
            
            if "name" in available_columns:
                tweet["name"] = row["name"].strip().lower() if row["name"] else None
            
            if "text" in available_columns:
                tweet["text"] = row["text"]
            
            tweets.append(tweet)
    
    # Print available columns and active rules
    print(f"Available columns: {sorted(available_columns)}")
    print("Active edge building rules:")
    
    rule_count = 0
    if "in_reply_to_status_id" in available_columns:
        print("  ✓ Rule 1: Connect tweets replying to same status_id")
        print("  ✓ Rule 4: Connect tweets replying to status_id with original tweet")
        rule_count += 2
    else:
        print("  ✗ Rule 1 & 4: Skipped (no 'in_reply_to_status_id' column)")
    
    if "in_reply_to_screen_name" in available_columns:
        print("  ✓ Rule 2: Connect tweets replying to same screen_name")
        rule_count += 1
    else:
        print("  ✗ Rule 2: Skipped (no 'in_reply_to_screen_name' column)")
    
    if "in_reply_to_screen_name" in available_columns and "name" in available_columns:
        print("  ✓ Rule 3: Connect replies to screen_name with tweets from that user")
        rule_count += 1
    else:
        print("  ✗ Rule 3: Skipped (missing 'in_reply_to_screen_name' or 'name' column)")
    
    if rule_count == 0:
        print("Warning: No edge building rules are active. Graph will only contain isolated nodes.")
    
    # Build graph
    print(f"\nBuilding graph from {args.input_csv}...")
    G = build_graph(tweets, available_columns)
    print(f"Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.")
    
    if args.verbose:
        print(f"Connected components: {nx.number_connected_components(G)}")
        if len(G.edges()) > 0:
            print(f"Average degree: {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}")
    
    # Detect communities
    if len(G.nodes()) > 0:
        print(f"Running {args.algo} community detection...")
        partition = detect_communities(G, algo=args.algo, k=args.k)
        num_communities = len(set(partition.values()))
        print(f"Detected {num_communities} communities.")
        
        # Add community attribute to graph
        nx.set_node_attributes(G, partition, "community")
        print("Merging tweets by username within each community...")
        G, merged_info = merge_graph_by_user(G, tweets, partition, available_columns)
        print(f"Merged graph has {len(G.nodes())} nodes and {len(G.edges())} edges.")

    else:
        print("Empty graph - no communities to detect.")
        partition = {}
        merged_info = {}
    
    # Save graph
    if len(G.nodes()) > 0:
        if args.format == "gml":
            nx.write_gml(G, args.out_graph)
        else:
            nx.write_graphml(G, args.out_graph)
        print(f"Graph saved to {args.out_graph} in {args.format} format.")
    else:
        print("Empty graph - skipping graph file output.")
    
    # Save CSV with communities
    print(f"Saving updated CSV to {args.out_csv}...")
    
    # Determine output fieldnames based on input columns
    output_fieldnames = ["id"]
    if "text" in available_columns:
        output_fieldnames.append("text")
    if "in_reply_to_screen_name" in available_columns:
        output_fieldnames.append("in_reply_to_screen_name")
    if "in_reply_to_status_id" in available_columns:
        output_fieldnames.append("in_reply_to_status_id")
    if "name" in available_columns:
        output_fieldnames.append("name")
    output_fieldnames.append("community")
    
    with open(args.out_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for _, info in merged_info.items():
            row = {
                "id": info["id"],
                "community": info["community"]
            }
            if "text" in available_columns:
                row["text"] = info.get("text", "")
            if "in_reply_to_screen_name" in available_columns:
                row["in_reply_to_screen_name"] = info.get("in_reply_to_screen_name", "")
            if "in_reply_to_status_id" in available_columns:
                row["in_reply_to_status_id"] = info.get("in_reply_to_status_id", "")
            if "name" in available_columns:
                row["name"] = info.get("name", "")
            
            writer.writerow(row)
    
    print(f"Updated CSV with communities saved to {args.out_csv}")

    
    # Save edges CSV if there are edges
    if len(G.edges()) > 0:
        edges_csv = args.out_csv.replace(".csv", "_edges.csv")
        with open(edges_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target"])
            for u, v in G.edges():
                writer.writerow([u, v])
        print(f"Edges CSV saved to {edges_csv}")
    else:
        print("No edges to save.")

if __name__ == "__main__":
    main()