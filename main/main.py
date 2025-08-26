import csv
import subprocess
import argparse
import os
import sys
import glob

def run_cmd(cmd, description):
    print(f"\n[INFO] Running: {description}")
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {description} failed:\n{result.stderr}")
        sys.exit(1)
    else:
        print(f"[DONE] {description} complete.")
        if result.stdout.strip():
            print(result.stdout.strip())


def create_absa_input_from_community(community_csv_path, absa_input_path):
    """
    Reads community_input CSV which must contain at least 'id' and 'text' columns,
    and creates absa_input.csv containing only 'id,text'.
    """
    with open(community_csv_path, newline="", encoding="utf-8") as infile, \
         open(absa_input_path, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        fieldnames = ["id", "text"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if "id" not in row or "text" not in row:
                print("[ERROR] community_input CSV must contain 'id' and 'text' columns")
                sys.exit(1)
            writer.writerow({"id": row["id"], "text": row["text"]})


def main():
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator")
    parser.add_argument("--community_input", required=True,
                        help="CSV with columns: id,text,in_reply_to_status_id,in_reply_to_screen_name,name")
    parser.add_argument("--absa_aspect", required=True,
                        help="CSV with columns: aspect,desc")
    parser.add_argument("--workdir", default="pipeline_out",
                        help="Directory for intermediate outputs")
    parser.add_argument("--model", choices=["gemini", "dp"], default="gemini",
                        help="Main model/algorithm for ABSA")
    parser.add_argument("--model_name",
                        help="Specific model checkpoint or nlp processor (gemini: gemini-2.5-flash, gemini-2.0-flash; dp: stanza, udpipe)")
    parser.add_argument("--vector_mode", choices=["sentence", "tfidf", "fasttext"], default="tfidf",
                        help="Vector mode for DP model (default: tfidf)")
    parser.add_argument("--batch_size", type=int, default=25,
                        help="Batch size for ABSA models (default: 25)")
    parser.add_argument("--algo", choices=["louvain", "metis"], default="louvain",
                        help="Community detection algorithm (default: louvain)")
    parser.add_argument("-k", type=int, default=2,
                        help="Number of communities for METIS (default: 2; ignored for Louvain)")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)

    # Auto-create absa_input.csv
    absa_input_csv = os.path.join(args.workdir, "absa_input.csv")
    create_absa_input_from_community(args.community_input, absa_input_csv)

    # Step 1: ABSA model execution
    aspect_output_folder = os.path.join(args.workdir, "aspect_out")

    if args.model == "gemini":
        model_value = args.model_name if args.model_name else "gemini-2.0-flash"
        run_cmd([
            "gemini",
            "--input", absa_input_csv,
            "--aspects", args.absa_aspect,
            "--output", aspect_output_folder,
            "--model", model_value,
            "--batch_size", str(args.batch_size),
        ], "Component 1: Gemini")

    elif args.model == "dp":
        vactor_mode_value = args.vector_mode if args.vector_mode else "tfidf"
        model_value = args.model_name if args.model_name else "stanza"
        run_cmd([
            "dp",
            "--input", absa_input_csv,
            "--aspects", args.absa_aspect,
            "--lexicon", "lexicon.csv",   # ⚠️ must exist or be passed
            "--output", aspect_output_folder,
            "--vector_mode", vactor_mode_value,
            "--processor", model_value,
        ], "Component 1: Double Propagation (DP)")

    # Find ABSA output CSV
    if args.model == "gemini":
        pattern = os.path.join(aspect_output_folder,
                               f"absa_input_*.csv")
    else:
        pattern = os.path.join(aspect_output_folder, "*.csv")

    matching_files = glob.glob(pattern)
    absa_output_csv = matching_files[0] if matching_files else None

    # Step 2: Community detection
    community_output_csv = os.path.join(args.workdir, "community_out.csv")
    edges_output_csv = community_output_csv.replace('.csv', '_edges.csv')
    graph_output_gml = os.path.join(args.workdir, "community_graph.gml")

    run_cmd([
        "community",
        "--out-csv", community_output_csv,
        "--out-graph", graph_output_gml,
        args.community_input,
        "--algo", args.algo,
        "-k", str(args.k),
    ], "Component 2: Community detection (nodes)")

    # Step 3: ABSA + community merge
    merged_output_csv = os.path.join(args.workdir, "absa_community.csv")
    run_cmd([
        "absa_community_merge",
        "--aspect", absa_output_csv,
        "--meta", community_output_csv,
        "--output", merged_output_csv,
    ], "Component 3: ABSA + Community Merge")

    # Step 4: Consensus
    consensus_output_txt = os.path.join(args.workdir, "consensus.txt")
    consensus_details_txt = os.path.join(args.workdir, "consensus_details.txt")
    run_cmd([
        "consensus",
        "-o", consensus_output_txt,
        "--details", consensus_details_txt,
        edges_output_csv, 
        merged_output_csv
    ], "Component 4: Consensus")

    print("\n[PIPELINE COMPLETE]")
    print(f"Final consensus file: {consensus_output_txt}")


if __name__ == "__main__":
    main()
