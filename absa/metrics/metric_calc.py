import argparse
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compute ABSA evaluation metrics")
    parser.add_argument("--pred", required=True, help="Predicted CSV file path")
    parser.add_argument("--gold", required=True, help="Gold CSV file path")
    parser.add_argument("--output", required=True, help="Output file OR folder path for metrics.txt")

    args = parser.parse_args()

    pred_file = Path(args.pred).resolve()
    gold_file = Path(args.gold).resolve()
    output_path = Path(args.output).resolve()

    # If output is a folder or ends with "/", treat it as a folder
    if output_path.is_dir() or str(output_path).endswith("/"):
        output_path = output_path / f"{pred_file.stem}_metrics_{int(time.time())}.txt"

    # Resolve to absolute paths to compare safely
    output_resolved = output_path.resolve()
    pred_resolved = pred_file.resolve()
    gold_resolved = gold_file.resolve()

    # Check for accidental overwrite
    if output_resolved in [pred_resolved, gold_resolved]:
        raise ValueError(
            f"❌ Output file must not overwrite input files!\n"
            f"Pred: {pred_file}\nGold: {gold_file}\nOutput: {output_path}"
        )
    
    os.makedirs(output_path.parent, exist_ok=True)

    # === Load files ===
    pred_df = pd.read_csv(pred_file)
    gold_df = pd.read_csv(gold_file)

    # === Group predictions by tweet_id ===
    pred_groups = pred_df.groupby("id")

    # === Containers ===
    total_gold_aspects = 0
    total_pred_aspects = 0
    total_correct_aspects = 0
    soft_accuracies = []
    exact_match_count = 0
    total_tweets = len(gold_df)

    for _, row in gold_df.iterrows():
        tweet_id = row["id"]
        gold_aspect = str(row["aspect_category_final"]).strip() if pd.notna(row["aspect_category_final"]) else None
        gold_sentiment = np.array(eval(row["hard_label_final"])) if pd.notna(row["hard_label_final"]) else None

        # --- Gold aspects ---
        gold_aspects = set()
        gold_sentiments = {}
        if gold_aspect and gold_sentiment is not None:
            gold_aspects.add(gold_aspect)
            gold_sentiments[gold_aspect] = gold_sentiment

        # --- Predicted aspects ---
        if tweet_id in pred_groups.groups:
            pred_group = pred_groups.get_group(tweet_id)
            pred_aspects = set(pred_group["aspect_category"].str.strip())
        else:
            pred_aspects = set()

        # --- Correct aspects ---
        correct_aspects = gold_aspects & pred_aspects

        # --- Soft accuracy for matched aspects ---
        for aspect in correct_aspects:
            pred_rows = pred_group[pred_group["aspect_category"].str.strip() == aspect]
            pred_probs = np.array([eval(p) for p in pred_rows["sentiment_prob"]])
            pred_mean = np.mean(pred_probs, axis=0)
            gold_prob = gold_sentiments[aspect]
            mse = np.mean((pred_mean - gold_prob) ** 2)
            soft_acc = 1 - mse
            soft_accuracies.append(soft_acc)

        # --- Aspect stats ---
        total_gold_aspects += len(gold_aspects)
        total_pred_aspects += len(pred_aspects)
        total_correct_aspects += len(correct_aspects)

        # --- Exact match ---
        if gold_aspects == pred_aspects:
            exact_match_count += 1

    # === Aspect detection P/R/F1 ===
    precision = total_correct_aspects / total_pred_aspects if total_pred_aspects else 0.0
    recall = total_correct_aspects / total_gold_aspects if total_gold_aspects else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # === Sentiment soft accuracy ===
    mean_soft_acc = np.mean(soft_accuracies) if soft_accuracies else 0.0

    # === Exact match ===
    exact_match_rate = exact_match_count / total_tweets if total_tweets else 0.0

    # === Format results ===
    result_lines = [
        f"Aspect Detection Precision: {precision:.4f}",
        f"Aspect Detection Recall:    {recall:.4f}",
        f"Aspect Detection F1:        {f1:.4f}",
        f"Sentiment Soft Accuracy:    {mean_soft_acc:.4f}",
        f"Exact Match Rate:           {exact_match_rate:.4f}"
    ]

    for line in result_lines:
        print(line)

    # === Save ===
    with open(output_path, "w", encoding="utf-8") as f:
        for line in result_lines:
            f.write(line + "\n")

    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
