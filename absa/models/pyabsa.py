import csv
import os
import time
import re
import argparse
from pathlib import Path
from pyabsa import AspectPolarityClassification as APC


# --------------------------
# Helpers
# --------------------------

def normalize_text(text):
    text = text.strip().strip('"').strip("'")
    text = re.sub(r'\s+', ' ', text)
    return text


# --------------------------
# Main CLI
# --------------------------

def main():
    parser = argparse.ArgumentParser(description="ABSA Batch Pipeline with PyABSA Zero-Shot (No Confidence)")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--aspects", required=True, help="Aspects CSV file path")
    parser.add_argument("--output", required=True, help="Output folder path")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--model", default="multilingual", help="PyABSA pretrained model name or path (default: multilingual zero-shot)")
    args = parser.parse_args()

    start_time = time.time()

    input_path = Path(args.input).resolve()
    output_folder = Path(args.output).resolve()
    os.makedirs(output_folder, exist_ok=True)

    # --------------------------
    # Load PyABSA zero-shot model
    # --------------------------
    print("📥 Loading PyABSA model...")
    if args.model.lower() == "multilingual":
        model = APC.SentimentClassifier(
            checkpoint="multilingual",  # multilingual zero-shot model
            auto_device=True
        )
    else:
        model = APC.SentimentClassifier(
            checkpoint=args.model,
            auto_device=True
        )

    # --------------------------
    # Load tweets
    # --------------------------
    tweets = []
    with open(input_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweets.append({
                "id": row["id"],
                "text": normalize_text(row["text"])
            })

    # --------------------------
    # Load aspects
    # --------------------------
    aspect_categories = []
    with open(args.aspects, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            aspect_categories.append(row["aspect_category"].strip())

    # --------------------------
    # Collect results
    # --------------------------
    all_parsed_results = []

    for i in range(0, len(tweets), args.batch_size):
        tweets_batch = tweets[i:i + args.batch_size]

        for tweet in tweets_batch:
            # Pass all aspects at once (list) instead of looping per aspect
            results = model.predict(
                text=tweet["text"],
                aspect=aspect_categories,  # list of aspects
                print_result=False
            )

            # `results` will be a list of predictions (one per aspect)
        for res in results:
            if isinstance(res, str):
                # res is just the predicted label
                all_parsed_results.append({
                    "id": tweet["id"],
                    "text": tweet["text"],
                    "aspect_category": res,  # maybe map to your aspect if needed
                    "sentiment_prob": None   # no probabilities in zero-shot
                })
            else:
                # res is a proper prediction object
                polarity_probs = [round(float(p), 4) for p in res.probs]
                all_parsed_results.append({
                    "id": tweet["id"],
                    "text": tweet["text"],
                    "aspect_category": res.aspect,
                    "sentiment_prob": polarity_probs
                })


        print(f"✅ Processed tweets {i + 1}-{i + len(tweets_batch)}")
        time.sleep(0.5)

    # --------------------------
    # Save results
    # --------------------------
    output_path = output_folder / f"{input_path.stem}_absa_{int(time.time())}.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "text", "aspect_category", "sentiment_prob"])
        writer.writeheader()
        for row in all_parsed_results:
            row["sentiment_prob"] = str(row["sentiment_prob"])
            writer.writerow(row)

    print(f"📝 Results saved to {output_path}")

    elapsed_time = time.time() - start_time
    print(f"⏱️ Total time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
