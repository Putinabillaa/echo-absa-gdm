import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, mean_squared_error, log_loss
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import ast


def safe_eval_list(x):
    """Convert string representation of list to actual list of floats."""
    if pd.isna(x):
        return None
    if isinstance(x, list):
        return x
    try:
        return list(map(float, ast.literal_eval(x)))
    except Exception:
        return None


def multiclass_brier_score(y_true, y_prob):
    """Compute proper multi-class Brier score (mean squared error per sample)."""
    return np.mean(np.sum((y_prob - y_true) ** 2, axis=1))


def baseline_brier_score(y_true):
    """Compute baseline Brier score by predicting the class distribution for all samples."""
    class_freq = np.mean(y_true, axis=0)  # empirical distribution of classes
    baseline_probs = np.tile(class_freq, (y_true.shape[0], 1))
    return np.mean(np.sum((baseline_probs - y_true) ** 2, axis=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gold_aspect_col", default="aspect_category_final")
    parser.add_argument("--gold_sentiment_col", default="hard_label_final")
    parser.add_argument("--pred_aspect_col", default="aspect_category")
    parser.add_argument("--pred_sentiment_col", default="sentiment_prob")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred)
    gold_df = pd.read_csv(args.gold)
    output_path = Path(args.output)

    # Convert sentiment strings to numeric lists
    gold_df[args.gold_sentiment_col] = gold_df[args.gold_sentiment_col].apply(safe_eval_list)
    pred_df[args.pred_sentiment_col] = pred_df[args.pred_sentiment_col].apply(safe_eval_list)

    all_ids = set(pred_df['id']).union(set(gold_df['id']))
    all_aspects = sorted(set(gold_df[args.gold_aspect_col].dropna()).union(
                          set(pred_df[args.pred_aspect_col].dropna())))

    # For multilabel binarization
    mlb = MultiLabelBinarizer(classes=all_aspects)

    Y_true, Y_pred = [], []
    mismatches = []
    exact_match_count = 0

    sentiment_true, sentiment_pred = [], []
    sentiment_pairs_count = 0

    for tweet_id in all_ids:
        gold_rows = gold_df[gold_df['id'] == tweet_id]
        pred_rows = pred_df[pred_df['id'] == tweet_id] if tweet_id in pred_df['id'].values else pd.DataFrame()

        # FIXED: Build aspect->sentiment dictionaries to ensure proper pairing
        gold_aspect_sentiment = {}
        if not gold_rows.empty:
            for _, row in gold_rows.iterrows():
                aspect = row[args.gold_aspect_col]
                sentiment = row[args.gold_sentiment_col]
                if pd.notna(aspect) and sentiment is not None:
                    gold_aspect_sentiment[aspect] = sentiment

        pred_aspect_sentiment = {}
        if not pred_rows.empty:
            for _, row in pred_rows.iterrows():
                aspect = row[args.pred_aspect_col]
                sentiment = row[args.pred_sentiment_col]
                if pd.notna(aspect) and sentiment is not None:
                    pred_aspect_sentiment[aspect] = sentiment

        # Get unique aspects
        gold_aspects = list(gold_aspect_sentiment.keys())
        pred_aspects = list(pred_aspect_sentiment.keys())

        # Save multilabel aspect sets
        Y_true.append(set(gold_aspects))
        Y_pred.append(set(pred_aspects))

        # Exact match: compare (aspect, argmax sentiment) pairs
        def sent_label(s):
            if s is None:
                return None
            if isinstance(s, (list, np.ndarray)):
                return int(np.argmax(s))
            return int(s)

        # FIXED: Use dictionaries for proper aspect-sentiment pairing
        gold_pairs = set()
        for aspect, sentiment in gold_aspect_sentiment.items():
            sentiment_label = sent_label(sentiment)
            if sentiment_label is not None:
                gold_pairs.add((aspect, sentiment_label))
        
        pred_pairs = set()
        for aspect, sentiment in pred_aspect_sentiment.items():
            sentiment_label = sent_label(sentiment)
            if sentiment_label is not None:
                pred_pairs.add((aspect, sentiment_label))

        if gold_pairs == pred_pairs:
            exact_match_count += 1

        # Mismatches logging
        for ga in gold_aspects:
            if ga not in pred_aspects:
                mismatches.append(f"Tweet {tweet_id}: GOLD={ga} PRED=none")
        for pa in pred_aspects:
            if pa not in gold_aspects:
                mismatches.append(f"Tweet {tweet_id}: GOLD=none PRED={pa}")

        # FIXED: Sentiment metrics using dictionaries for proper matching
        for aspect in gold_aspects:
            if aspect in pred_aspects:
                gs = gold_aspect_sentiment[aspect]
                ps = pred_aspect_sentiment[aspect]
                
                if gs is not None and ps is not None:
                    sentiment_true.append(np.array(gs, dtype=float))
                    sentiment_pred.append(np.array(ps, dtype=float))
                    sentiment_pairs_count += 1

    # FIXED: Fit the binarizer first, then transform both
    Y_true_bin = mlb.fit_transform(Y_true)
    Y_pred_bin = mlb.transform(Y_pred)

    # Aspect detection metrics
    precision_per_class = precision_score(Y_true_bin, Y_pred_bin, average=None, zero_division=0)
    recall_per_class = recall_score(Y_true_bin, Y_pred_bin, average=None, zero_division=0)
    f1_per_class = f1_score(Y_true_bin, Y_pred_bin, average=None, zero_division=0)

    f1_macro = f1_score(Y_true_bin, Y_pred_bin, average='macro')
    f1_micro = f1_score(Y_true_bin, Y_pred_bin, average='micro')

    # FIXED: Confusion matrix construction
    labels_with_none = all_aspects + ["none"]
    y_true_aspects, y_pred_aspects = [], []
    
    # For each sample (tweet), check each aspect
    for yt, yp in zip(Y_true, Y_pred):
        # Add true positives and false negatives
        for a in yt:
            if a in yp:
                y_true_aspects.append(a)
                y_pred_aspects.append(a)
            else:
                y_true_aspects.append(a)
                y_pred_aspects.append("none")
        
        # Add false positives
        for a in yp:
            if a not in yt:
                y_true_aspects.append("none")
                y_pred_aspects.append(a)
    
    # Handle empty predictions case
    if not y_true_aspects:
        y_true_aspects = ["none"]
        y_pred_aspects = ["none"]
    
    cm = confusion_matrix(y_true_aspects, y_pred_aspects, labels=labels_with_none)

    # Sentiment metrics
    if sentiment_pairs_count > 0:
        true_mat = np.vstack(sentiment_true)
        pred_mat = np.vstack(sentiment_pred)

        # FIXED: Add validation for matrix dimensions
        if true_mat.shape[1] != pred_mat.shape[1]:
            print(f"Warning: Dimension mismatch - true_mat: {true_mat.shape}, pred_mat: {pred_mat.shape}")
            min_cols = min(true_mat.shape[1], pred_mat.shape[1])
            true_mat = true_mat[:, :min_cols]
            pred_mat = pred_mat[:, :min_cols]

        # Convert one-hot to class indices for log_loss
        true_classes = np.argmax(true_mat, axis=1)

        sentiment_mse = mean_squared_error(true_mat, pred_mat)
        
        # FIXED: Add error handling for log_loss
        try:
            sentiment_logloss = log_loss(true_classes, pred_mat, labels=list(range(pred_mat.shape[1])))
        except ValueError as e:
            print(f"Warning: Could not compute log loss: {e}")
            sentiment_logloss = None
            
        sentiment_brier = multiclass_brier_score(true_mat, pred_mat)
        sentiment_brier_baseline = baseline_brier_score(true_mat)
    else:
        sentiment_mse = None
        sentiment_logloss = None
        sentiment_brier = None
        sentiment_brier_baseline = None

    exact_match_ratio = exact_match_count / len(all_ids) if len(all_ids) > 0 else 0

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)  # FIXED: Ensure output directory exists
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("PER-CLASS METRICS (ASPECT DETECTION):\n")
        for label, p, r, f1c in zip(all_aspects, precision_per_class, recall_per_class, f1_per_class):
            f.write(f"{label:<15s} Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1c:.4f}\n")
        f.write(f"\nF1 macro: {f1_macro:.4f}\nF1 micro: {f1_micro:.4f}\n")
        f.write(f"Exact match (aspect+sentiment): {exact_match_ratio:.4f}\n")
        f.write(f"Sentiment MSE (on {sentiment_pairs_count} overlapping aspects): {sentiment_mse}\n")
        f.write(f"Sentiment Log Loss: {sentiment_logloss}\n")
        f.write(f"Sentiment Brier Score: {sentiment_brier}\n")
        f.write(f"Baseline Brier Score: {sentiment_brier_baseline}\n\n")

        f.write("CONFUSION MATRIX (with 'none'):\n")
        f.write(",".join(labels_with_none) + "\n")
        for row_label, row in zip(labels_with_none, cm):
            f.write(row_label + "," + ",".join(map(str,row)) + "\n")

        f.write("\nMISMATCHED PREDICTIONS:\n")
        for m in mismatches[:50]:  # limit to first 50
            f.write(m + "\n")

    # Save confusion matrix as PNG
    plt.figure(figsize=(max(8,len(labels_with_none)), max(6,len(labels_with_none))))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels_with_none, yticklabels=labels_with_none, cmap='Blues')
    plt.xlabel("Predicted Aspect")
    plt.ylabel("True Aspect")
    plt.title("Aspect Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_png_path = output_path.parent / f"{output_path.stem}_confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    print(f"✅ Confusion matrix saved to {cm_png_path}")


if __name__ == "__main__":
    main()