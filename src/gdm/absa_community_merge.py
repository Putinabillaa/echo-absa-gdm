import csv
import argparse
from collections import defaultdict

def merge_csv(aspect_file, meta_file, output_file):
    # Read aspect file and collect aspects per id
    aspect_data = defaultdict(dict)
    all_aspects = set()

    with open(aspect_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweet_id = row["id"]
            aspect = row["aspect_category"]
            prob = row["sentiment_prob"]
            aspect_data[tweet_id][aspect] = prob
            all_aspects.add(aspect)

    # Read meta file and merge with aspect data
    merged_rows = []
    with open(meta_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweet_id = row["id"]
            merged_row = {"id": tweet_id}

            # Fill each aspect column with sentiment_prob or blank
            for aspect in sorted(all_aspects):
                merged_row[aspect] = aspect_data.get(tweet_id, {}).get(aspect, "")

            # Add community
            merged_row["community"] = row.get("community", "")

            merged_rows.append(merged_row)

    # Write output
    fieldnames = ["id"] + sorted(all_aspects) + ["community"]

    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"✅ Merged file saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge aspect sentiment CSV with metadata CSV into wide format.")
    parser.add_argument("--aspect", required=True, help="CSV file containing id,text,aspect_category,sentiment_prob")
    parser.add_argument("--meta", required=True, help="CSV file containing id,in_reply_to_status_id,in_reply_to_screen_name,community")
    parser.add_argument("--output", required=True, help="Output CSV file path")

    args = parser.parse_args()
    merge_csv(args.aspect, args.meta, args.output)
    

if __name__ == "__main__":
    main()