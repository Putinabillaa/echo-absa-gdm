import argparse
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Compute ABSA evaluation metrics with Macro & Micro F1")
    parser.add_argument("--pred", required=True, help="Predicted CSV file path")
    parser.add_argument("--gold", required=True, help="Gold CSV file path")
    parser.add_argument("--output", required=True, help="Output file OR folder path for metrics.txt")

    args = parser.parse_args()

    pred_file = Path(args.pred).resolve()
    gold_file = Path(args.gold).resolve()
    output_path = Path(args.output).resolve()

    if output_path.is_dir() or str(output_path).endswith("/"):
        output_path = output_path / f"{pred_file.stem}_metrics_{int(time.time())}.txt"

    output_resolved = output_path.resolve()
    pred_resolved = pred_file.resolve()
    gold_resolved = gold_file.resolve()

    if output_resolved in [pred_resolved, gold_resolved]:
        raise ValueError(
            f"❌ Output file must not overwrite input files!\n"
            f"Pred: {pred_file}\nGold: {gold_file}\nOutput: {output_path}"
        )
    
    os.makedirs(output_path.parent, exist_ok=True)

    # Load files
    pred_df = pd.read_csv(pred_file)
    gold_df = pd.read_csv(gold_file)

    pred_groups = pred_df.groupby("id")

    total_gold_aspects = 0
    total_pred_aspects = 0
    total_correct_aspects = 0
    soft_accuracies = []
    exact_match_count = 0
    total_tweets = len(gold_df)

    # For macro F1 stats
    aspect_tp = defaultdict(int)
    aspect_fp = defaultdict(int)
    aspect_fn = defaultdict(int)

    for _, row in gold_df.iterrows():
        tweet_id = row["id"]
        gold_aspect = str(row["aspect_category_final"]).strip() if pd.notna(row["aspect_category_final"]) else None
        gold_sentiment = np.array(eval(row["hard_label_final"])) if pd.notna(row["hard_label_final"]) else None

        gold_aspects = set()
        gold_sentiments = {}
        if gold_aspect and gold_sentiment is not None:
            gold_aspects.add(gold_aspect)
            gold_sentiments[gold_aspect] = gold_sentiment

        if tweet_id in pred_groups.groups:
            pred_group = pred_groups.get_group(tweet_id)
            pred_aspects = set(pred_group["aspect_category"].str.strip())
        else:
            pred_aspects = set()

        correct_aspects = gold_aspects & pred_aspects

        # Sentiment soft accuracy
        for aspect in correct_aspects:
            pred_rows = pred_group[pred_group["aspect_category"].str.strip() == aspect]
            pred_probs = np.array([eval(p) for p in pred_rows["sentiment_prob"]])
            pred_mean = np.mean(pred_probs, axis=0)
            gold_prob = gold_sentiments[aspect]
            mse = np.mean((pred_mean - gold_prob) ** 2)
            soft_acc = 1 - mse
            soft_accuracies.append(soft_acc)

        # Micro stats
        total_gold_aspects += len(gold_aspects)
        total_pred_aspects += len(pred_aspects)
        total_correct_aspects += len(correct_aspects)

        # Macro stats per aspect
        for asp in correct_aspects:
            aspect_tp[asp] += 1
        for asp in pred_aspects - gold_aspects:
            aspect_fp[asp] += 1
        for asp in gold_aspects - pred_aspects:
            aspect_fn[asp] += 1

        if gold_aspects == pred_aspects:
            exact_match_count += 1

    # Micro Precision/Recall/F1
    micro_precision = total_correct_aspects / total_pred_aspects if total_pred_aspects else 0.0
    micro_recall = total_correct_aspects / total_gold_aspects if total_gold_aspects else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0

    # Macro F1
    aspect_f1_scores = []
    for asp in set(list(aspect_tp.keys()) + list(aspect_fp.keys()) + list(aspect_fn.keys())):
        tp = aspect_tp[asp]
        fp = aspect_fp[asp]
        fn = aspect_fn[asp]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        aspect_f1_scores.append(f1)
    macro_f1 = np.mean(aspect_f1_scores) if aspect_f1_scores else 0.0

    mean_soft_acc = np.mean(soft_accuracies) if soft_accuracies else 0.0
    exact_match_rate = exact_match_count / total_tweets if total_tweets else 0.0

    result_lines = [
        f"Aspect Detection Micro Precision: {micro_precision:.4f}",
        f"Aspect Detection Micro Recall:    {micro_recall:.4f}",
        f"Aspect Detection Micro F1:        {micro_f1:.4f}",
        f"Aspect Detection Macro F1:        {macro_f1:.4f}",
        f"Sentiment Soft Accuracy:          {mean_soft_acc:.4f}",
        f"Exact Match Rate:                 {exact_match_rate:.4f}"
    ]

    for line in result_lines:
        print(line)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in result_lines:
            f.write(line + "\n")

    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
