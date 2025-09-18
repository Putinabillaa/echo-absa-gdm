#!/usr/bin/env python3
"""
null_compare_with_consensus.py

Runs attribute-permutation nulls and topology-rewiring nulls,
but calls the user's original consensus.py script to compute ECRs.

Usage:
  python null_compare_with_consensus.py --aspects aspects.csv --method attribute --n-runs 100 \
    --out-prefix null/attr --consensus-script consensus.py

  python null_compare_with_consensus.py --aspects aspects.csv --edge-list edges.txt \
    --method topology --n-runs 50 --out-prefix null/rewire --consensus-script consensus.py

Dependencies: numpy, networkx, python-louvain, tqdm, matplotlib
"""

import argparse, csv, json, ast, os, subprocess, tempfile
import numpy as np
import networkx as nx
import community as community_louvain
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------
# Helpers for aspects I/O
# ---------------------------
def read_aspects_csv(path):
    aspects_data = {}
    communities = defaultdict(dict)
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        aspect_cols = [c for c in fieldnames if c not in ("id","community")]
        for row in reader:
            uid = row["id"]
            vecs = []
            for col in aspect_cols:
                val = (row.get(col) or "").strip()
                try:
                    v = ast.literal_eval(val)
                except Exception:
                    try:
                        v = json.loads(val)
                    except Exception:
                        v = None
                if isinstance(v,(int,float)):
                    v = [float(v)]
                vecs.append(v)
            aspects_data[uid] = vecs
            comm = row.get("community","")
            communities[comm][uid] = vecs
    return aspects_data, communities, aspect_cols

def write_aspects_csv(aspects_data, communities, aspect_cols, out_path):
    header = ["id","community"]+aspect_cols
    with open(out_path,"w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for comm, members in communities.items():
            for uid in members.keys():
                vecs = aspects_data[uid]
                row = [uid, comm] + [json.dumps(v) for v in vecs]
                writer.writerow(row)

# ---------------------------
# Attribute null
# ---------------------------
def shuffle_aspects(aspects_data, rng):
    uids = list(aspects_data.keys())
    vecs = [aspects_data[u] for u in uids]
    perm = rng.permutation(len(uids))
    return {uids[i]: vecs[perm[i]] for i in range(len(uids))}

# ---------------------------
# Topology null
# ---------------------------
def read_edge_list(path):
    G = nx.Graph()
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if "," in line:
                u,v=line.split(",",1)
            else:
                parts=line.split()
                if len(parts)<2: continue
                u,v=parts[0],parts[1]
            G.add_edge(u.strip(),v.strip())
    return G

def read_gml_graph(path):
    """
    Reads a GML graph using networkx.
    Preserves node labels as strings (important to match aspects.csv 'id').
    """
    G = nx.read_gml(path, label="label")  # use node 'label' field as node id
    # Make sure node ids are strings to match aspects.csv ids
    mapping = {n: str(n) for n in G.nodes()}
    return nx.relabel_nodes(G, mapping)

def rewire_partition(G, seed=None, nswap_mult=5):
    Gc = G.copy()
    nswap = nswap_mult*G.number_of_edges()
    nx.double_edge_swap(Gc, nswap=nswap, max_tries=nswap*10, seed=seed)
    part = community_louvain.best_partition(Gc)
    comms = defaultdict(list)
    for node, comm in part.items():
        comms[comm].append(str(node))
    return comms

# ---------------------------
# Run consensus.py
# ---------------------------
def run_consensus(consensus_script, aspects_csv):
    # run your consensus.py and parse ECR
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as fout:
        summary_path=fout.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as fdet:
        details_path=fdet.name
    cmd=["python3", consensus_script, aspects_csv, "-o", summary_path, "--details", details_path]
    subprocess.run(cmd, check=True, capture_output=True)
    # parse summary file
    ecr=None
    with open(summary_path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("ECR (Mean)"):
                ecr=float(line.strip().split("=")[1])
    return ecr

# ---------------------------
# Null ensemble runners
# ---------------------------
def run_attribute_null(aspects_path, consensus_script, n_runs, out_prefix, seed=1):
    rng=np.random.default_rng(seed)
    aspects_data, communities, aspect_cols = read_aspects_csv(aspects_path)
    ecrs=[]
    for i in tqdm(range(n_runs), desc="attribute nulls"):
        shuffled=shuffle_aspects(aspects_data,rng)
        # build new aspects with old communities
        out_csv=f"{out_prefix}_attr_run{i+1}.csv"
        write_aspects_csv(shuffled, communities, aspect_cols, out_csv)
        ecr=run_consensus(consensus_script,out_csv)
        ecrs.append(ecr)
    return np.array(ecrs)

def run_topology_null(aspects_path, edge_list, consensus_script, n_runs, out_prefix, seed=1):
    rng=np.random.default_rng(seed)
    aspects_data, _, aspect_cols = read_aspects_csv(aspects_path)
    G=read_edge_list(edge_list)
    uids=set(aspects_data.keys())
    ecrs=[]
    for i in tqdm(range(n_runs), desc="topology nulls"):
        comms=rewire_partition(G, seed=int(rng.integers(1,1_000_000)))
        # build new aspects CSV with new comm labels
        communities={str(c):{u:aspects_data[u] for u in members if u in uids} for c,members in comms.items()}
        out_csv=f"{out_prefix}_topo_run{i+1}.csv"
        write_aspects_csv(aspects_data, communities, aspect_cols, out_csv)
        ecr=run_consensus(consensus_script,out_csv)
        ecrs.append(ecr)
    return np.array(ecrs)

def run_topology_null_gml(aspects_path, gml_path, consensus_script, n_runs, out_prefix, nswap_per_node=5, seed=1):
    rng = np.random.default_rng(seed)
    aspects_data, _, aspect_cols = read_aspects_csv(aspects_path)
    G = read_gml_graph(gml_path)
    uids = set(aspects_data.keys())

    # observed ECR: run Louvain on original graph
    partition_orig = community_louvain.best_partition(G)
    comms_obs = defaultdict(dict)
    for node, comm in partition_orig.items():
        if node in aspects_data:
            comms_obs[str(comm)][node] = aspects_data[node]
    obs_ecr = run_consensus(consensus_script, aspects_path)
    print("Observed ECR (topology):", obs_ecr)

    null_ecrs=[]
    nswap=int(nswap_per_node*G.number_of_edges())
    for i in tqdm(range(n_runs), desc="topology null runs"):
        G_rewired = G.copy()
        nx.double_edge_swap(G_rewired, nswap=nswap, max_tries=nswap*10,
                            seed=int(rng.integers(1,1_000_000)))
        part = community_louvain.best_partition(G_rewired)
        # build communities with aspects
        comms = defaultdict(dict)
        for node, comm in part.items():
            if node in aspects_data:
                comms[str(comm)][node] = aspects_data[node]
        # write aspects CSV with new community labels
        out_csv=f"{out_prefix}_topo_run{i+1}.csv"
        write_aspects_csv(aspects_data, comms, aspect_cols, out_csv)
        ecr = run_consensus(consensus_script, out_csv)
        null_ecrs.append(ecr)

    nulls=np.array(null_ecrs)
    summarize_and_plot(obs_ecr,nulls,"topology",out_prefix)
    return nulls

# ---------------------------
# Summary stats + plot
# ---------------------------
def summarize_and_plot(obs, nulls, label, out_prefix):
    mean,std=nulls.mean(),nulls.std(ddof=1)
    z=(obs-mean)/std if std>0 else float("nan")
    p=(nulls>=obs).sum()/len(nulls)
    stats={"obs":obs,"null_mean":mean,"null_std":std,"z":z,"p_emp":p}
    print(label,stats)
    with open(out_prefix+f"_{label}_stats.json","w") as f:
        json.dump(stats,f,indent=2)
    plt.hist(nulls,bins=30,alpha=0.8)
    plt.axvline(obs,color="r",linestyle="--",label=f"obs={obs:.3f}")
    plt.axvline(mean,color="k",linestyle=":",label=f"null_mean={mean:.3f}")
    plt.legend();plt.title(f"Null distribution: {label}");plt.xlabel("ECR");plt.ylabel("freq")
    plt.tight_layout();plt.savefig(out_prefix+f"_{label}_hist.png");plt.close()

# ---------------------------
# CLI
# ---------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--aspects",required=True)
    ap.add_argument("--graph",help="needed for topology null")
    ap.add_argument("--consensus-script",required=True,help="path to consensus.py")
    ap.add_argument("--method",choices=["attribute","topology","both"],default="both")
    ap.add_argument("--n-runs",type=int,default=100)
    ap.add_argument("--out-prefix",default="null/compare")
    ap.add_argument("--obs-ecr",type=float,help="Observed ECR (if already computed).")
    args=ap.parse_args()

    obs=args.obs_ecr
    if obs is None:
        # compute observed once using consensus.py directly
        obs=run_consensus(args.consensus_script,args.aspects)
        print("Observed ECR =",obs)

    if args.method in ("attribute","both"):
        attr_ecrs=run_attribute_null(args.aspects,args.consensus_script,args.n_runs,args.out_prefix)
        summarize_and_plot(obs,attr_ecrs,"attribute",args.out_prefix)

    if args.method in ("topology","both"):
        if not args.graph: raise ValueError("Need --graph (GML file) for topology null")
        topo_ecrs=run_topology_null_gml(args.aspects,args.graph,args.consensus_script,args.n_runs,args.out_prefix)


if __name__=="__main__":
    main()
