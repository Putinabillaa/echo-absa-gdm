import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    mean_squared_error,
    log_loss,
    multilabel_confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import warnings


# -----------------------------
# Helpers
# -----------------------------
def safe_eval_list(x):
    """Convert string representation of list to actual list of floats."""
    if pd.isna(x):
        return None
    if isinstance(x, (list, np.ndarray)):
        return list(map(float, x))
    try:
        return list(map(float, ast.literal_eval(str(x))))
    except Exception:
        return None


def multiclass_brier_score(y_true, y_prob):
    """
    Compute Penalized Brier Score (PBS) for multiclass classification.
    Adds a penalty (R-1)/R when the predicted class != true class.
    Returns PBS averaged per class.
    
    y_true : np.ndarray, shape (N, R) one-hot true labels
    y_prob : np.ndarray, shape (N, R) predicted probabilities
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    N, R = y_true.shape

    # Standard multiclass Brier Score per instance
    bs_per_instance = np.sum((y_prob - y_true) ** 2, axis=1)

    # Check incorrect argmax predictions
    true_classes = np.argmax(y_true, axis=1)
    pred_classes = np.argmax(y_prob, axis=1)
    penalties = ((R - 1) / R) * (pred_classes != true_classes)

    # Add penalty
    pbs_per_instance = bs_per_instance + penalties

    # Average PBS per class
    class_sums = np.zeros(R)
    class_counts = np.zeros(R)
    for i in range(N):
        class_idx = true_classes[i]
        class_sums[class_idx] += pbs_per_instance[i]
        class_counts[class_idx] += 1

    # Avoid division by zero
    class_avg = np.divide(class_sums, class_counts, out=np.zeros_like(class_sums), where=class_counts!=0)

    # Overall average across classes
    overall_avg = np.mean(class_avg[class_counts > 0])  # Only average non-empty classes

    return overall_avg, class_avg


def baseline_brier_score(y_true):
    """Calculate baseline Penalized Brier Score (PBS) using class frequencies."""
    class_freq = np.mean(y_true, axis=0)
    baseline_probs = np.tile(class_freq, (y_true.shape[0], 1))
    # Use the same PBS calculation as the main metric for fair comparison
    baseline_pbs_overall, _ = multiclass_brier_score(y_true, baseline_probs)
    return baseline_pbs_overall


def sent_label_from_probs(s):
    """Extract sentiment label from probability vector."""
    if s is None:
        return None
    return int(np.argmax(np.asarray(s, dtype=float)))


def normalize_probs(vec):
    """Normalize probability vector to sum to 1."""
    if vec is None:
        return None
    v = np.asarray(vec, dtype=float)
    s = v.sum()
    if s > 0:
        return (v / s).tolist()
    else:
        # If all zeros, return uniform distribution
        return (np.ones_like(v) / len(v)).tolist()


def check_vector_length(v, expected_len, row_idx, colname, labels):
    """Raise error if sentiment vector length mismatched."""
    if v is not None and len(v) != expected_len:
        raise ValueError(
            f"Sentiment vector length mismatch in column '{colname}' at row {row_idx}: "
            f"expected {expected_len} ({labels}), got {len(v)} with value {v}"
        )


def pairwise_mismatches(gold_aspects, pred_aspects, tweet_id):
    """Return mismatches + aligned gold/pred aspect lists for CM."""
    mismatches = []
    y_true_aspects, y_pred_aspects = [], []

    gold_only = list(gold_aspects - pred_aspects)
    pred_only = list(pred_aspects - gold_aspects)

    # True positives
    for a in gold_aspects & pred_aspects:
        y_true_aspects.append(a)
        y_pred_aspects.append(a)

    # Pair leftover aspects
    for ga, pa in zip(gold_only, pred_only):
        y_true_aspects.append(ga)
        y_pred_aspects.append(pa)
        mismatches.append(f"Tweet {tweet_id}: GOLD={ga} PRED={pa}")

    # Handle unpaired leftovers
    for ga in gold_only[len(pred_only):]:
        y_true_aspects.append(ga)
        y_pred_aspects.append("none")
        mismatches.append(f"Tweet {tweet_id}: GOLD={ga} PRED=none")

    for pa in pred_only[len(gold_only):]:
        y_true_aspects.append("none")
        y_pred_aspects.append(pa)
        mismatches.append(f"Tweet {tweet_id}: GOLD=none PRED={pa}")

    return mismatches, y_true_aspects, y_pred_aspects


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate aspect-based sentiment analysis predictions")
    parser.add_argument("--pred", required=True, help="Path to predictions CSV file")
    parser.add_argument("--gold", required=True, help="Path to gold standard CSV file")
    parser.add_argument("--output", required=True, help="Path to output results file")
    parser.add_argument("--gold_aspect_col", default="aspect_category_final", help="Gold aspect column name")
    parser.add_argument("--gold_sentiment_col", default="hard_label_final", help="Gold sentiment column name")
    parser.add_argument("--pred_aspect_col", default="aspect_category", help="Predicted aspect column name")
    parser.add_argument("--pred_sentiment_col", default="sentiment_prob", help="Predicted sentiment column name")
    parser.add_argument("--sentiment_order", default="neg,neu,pos", help="Sentiment label order")
    parser.add_argument("--normalize_sentiment_probs", action="store_true", help="Normalize sentiment probabilities")
    args = parser.parse_args()

    SENTIMENT_LABELS = [s.strip() for s in args.sentiment_order.split(",") if s.strip()]
    NUM_SENTIMENTS = len(SENTIMENT_LABELS)

    # Load data with error handling
    try:
        pred_df = pd.read_csv(args.pred)
        gold_df = pd.read_csv(args.gold)
    except FileNotFoundError as e:
        print(f"Error: Could not find input file - {e}")
        return
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Validate required columns
    required_cols_pred = ['id', args.pred_aspect_col, args.pred_sentiment_col]
    required_cols_gold = ['id', args.gold_aspect_col, args.gold_sentiment_col]
    
    missing_pred = [col for col in required_cols_pred if col not in pred_df.columns]
    missing_gold = [col for col in required_cols_gold if col not in gold_df.columns]
    
    if missing_pred:
        print(f"Error: Missing columns in predictions file: {missing_pred}")
        return
    if missing_gold:
        print(f"Error: Missing columns in gold file: {missing_gold}")
        return

    output_path = Path(args.output)

    # Parse probability vectors
    for df, col in [(gold_df, args.gold_sentiment_col), (pred_df, args.pred_sentiment_col)]:
        df[col] = df[col].apply(safe_eval_list)
        if args.normalize_sentiment_probs:
            df[col] = df[col].apply(normalize_probs)

    # Check vector lengths
    for i, v in enumerate(gold_df[args.gold_sentiment_col].values):
        check_vector_length(v, NUM_SENTIMENTS, i, args.gold_sentiment_col, SENTIMENT_LABELS)
    for i, v in enumerate(pred_df[args.pred_sentiment_col].values):
        check_vector_length(v, NUM_SENTIMENTS, i, args.pred_sentiment_col, SENTIMENT_LABELS)

    # Collect all ids/aspects
    all_ids = sorted(set(pred_df["id"]) | set(gold_df["id"]))
    if not all_ids:
        print("Error: No common IDs found between prediction and gold files")
        return
        
    all_aspects = sorted(set(gold_df[args.gold_aspect_col].dropna()) |
                         set(pred_df[args.pred_aspect_col].dropna()))
    
    if not all_aspects:
        print("Warning: No aspects found in the data")
        all_aspects = ['unknown']
        
    mlb = MultiLabelBinarizer(classes=all_aspects)

    Y_true, Y_pred, mismatches = [], [], []
    sentiment_true, sentiment_pred = [], []
    sentiment_true_classes, sentiment_pred_classes = [], []
    exact_match_count, sentiment_pairs_count = 0, 0

    y_true_aspects, y_pred_aspects = [], []

    for tweet_id in all_ids:
        g_rows = gold_df[gold_df["id"] == tweet_id]
        p_rows = pred_df[pred_df["id"] == tweet_id]

        g_dict = {r[args.gold_aspect_col]: r[args.gold_sentiment_col]
                  for _, r in g_rows.iterrows() if pd.notna(r[args.gold_aspect_col]) and r[args.gold_sentiment_col] is not None}
        p_dict = {r[args.pred_aspect_col]: r[args.pred_sentiment_col]
                  for _, r in p_rows.iterrows() if pd.notna(r[args.pred_aspect_col]) and r[args.pred_sentiment_col] is not None}
        
        g_aspects, p_aspects = set(g_dict.keys()), set(p_dict.keys())
        Y_true.append(g_aspects)
        Y_pred.append(p_aspects)

        g_pairs = {(a, sent_label_from_probs(s)) for a, s in g_dict.items() if sent_label_from_probs(s) is not None}
        p_pairs = {(a, sent_label_from_probs(s)) for a, s in p_dict.items() if sent_label_from_probs(s) is not None}
        if g_pairs == p_pairs:
            exact_match_count += 1

        mm, yt, yp = pairwise_mismatches(g_aspects, p_aspects, tweet_id)
        mismatches.extend(mm)
        y_true_aspects.extend(yt)
        y_pred_aspects.extend(yp)

        # Sentiment metrics for overlapping aspects
        for a in g_aspects & p_aspects:
            gs, ps = g_dict[a], p_dict[a]
            if gs is not None and ps is not None:
                # Normalize here if not done earlier to avoid double normalization
                if not args.normalize_sentiment_probs:
                    gs_norm = normalize_probs(gs)
                    ps_norm = normalize_probs(ps)
                else:
                    gs_norm, ps_norm = gs, ps
                
                sentiment_true.append(np.array(gs_norm, dtype=float))
                sentiment_pred.append(np.array(ps_norm, dtype=float))
                
                # Store classes for macro F1
                sentiment_true_classes.append(sent_label_from_probs(gs))
                sentiment_pred_classes.append(sent_label_from_probs(ps))
                
                sentiment_pairs_count += 1

    # Multilabel metrics
    if len(Y_true) == 0:
        print("Error: No valid data found for evaluation")
        return
        
    Y_true_bin, Y_pred_bin = mlb.fit_transform(Y_true), mlb.transform(Y_pred)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision_per_class = precision_score(Y_true_bin, Y_pred_bin, average=None, zero_division=0)
        recall_per_class = recall_score(Y_true_bin, Y_pred_bin, average=None, zero_division=0)
        f1_per_class = f1_score(Y_true_bin, Y_pred_bin, average=None, zero_division=0)
        f1_macro = f1_score(Y_true_bin, Y_pred_bin, average="macro", zero_division=0)
        f1_micro = f1_score(Y_true_bin, Y_pred_bin, average="micro", zero_division=0)

    # Use all_aspects + ["none"] for confusion matrix labels, but handle empty case
    cm_labels = all_aspects + ["none"] if all_aspects else ["none"]
    cm_custom = confusion_matrix(y_true_aspects or ["none"], y_pred_aspects or ["none"],
                                labels=cm_labels)
    ml_cms = multilabel_confusion_matrix(Y_true_bin, Y_pred_bin)

    # Sentiment metrics
    sentiment_mse = sentiment_logloss = sentiment_brier_overall = sentiment_brier_baseline = None
    sentiment_macro_f1 = sentiment_micro_f1 = sentiment_weighted_f1 = None
    sentiment_classification_report = None
    
    if sentiment_pairs_count > 0:
        true_mat, pred_mat = np.vstack(sentiment_true), np.vstack(sentiment_pred)
        true_classes = np.argmax(true_mat, axis=1)

        sentiment_mse = mean_squared_error(true_mat, pred_mat)
        
        try:
            sentiment_logloss = log_loss(true_classes, pred_mat, labels=list(range(NUM_SENTIMENTS)))
        except ValueError as e:
            print(f"Warning: Could not compute log loss: {e}")
            
        sentiment_brier_overall, _ = multiclass_brier_score(true_mat, pred_mat)  # Fixed: unpack tuple
        all_gold_mat = np.vstack([
            normalize_probs(s) if s is not None else np.ones(NUM_SENTIMENTS)/NUM_SENTIMENTS
            for s in gold_df[args.gold_sentiment_col].values
        ])
        sentiment_brier_baseline = baseline_brier_score(all_gold_mat)

        
        # Calculate sentiment F1 scores using argmax predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sentiment_macro_f1 = f1_score(sentiment_true_classes, sentiment_pred_classes, 
                                         average='macro', labels=list(range(NUM_SENTIMENTS)), zero_division=0)
            sentiment_micro_f1 = f1_score(sentiment_true_classes, sentiment_pred_classes, 
                                         average='micro', labels=list(range(NUM_SENTIMENTS)), zero_division=0)
            sentiment_weighted_f1 = f1_score(sentiment_true_classes, sentiment_pred_classes, 
                                            average='weighted', labels=list(range(NUM_SENTIMENTS)), zero_division=0)
        
        # Generate classification report for sentiment
        sentiment_classification_report = classification_report(
            sentiment_true_classes, sentiment_pred_classes,
            target_names=SENTIMENT_LABELS,
            labels=list(range(NUM_SENTIMENTS)),
            zero_division=0
        )

    exact_match_ratio = exact_match_count / len(all_ids) if all_ids else 0

    # -----------------------------
    # Save report
    # -----------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total tweets evaluated: {len(all_ids)}\n")
        f.write(f"Sentiment pairs calculated: {sentiment_pairs_count}\n")
        f.write(f"Sentiment calculation note: Only calculated for overlapping aspects between gold and predicted\n\n")
        
        f.write("PER-CLASS METRICS (ASPECT DETECTION):\n")
        f.write("-" * 50 + "\n")
        for label, p, r, f1c in zip(all_aspects, precision_per_class, recall_per_class, f1_per_class):
            f.write(f"{label:<25s} Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1c:.4f}\n")
        
        f.write(f"\nASPECT DETECTION SUMMARY:\n")
        f.write(f"F1 macro: {f1_macro:.4f}\n")
        f.write(f"F1 micro: {f1_micro:.4f}\n")
        
        f.write(f"\nSENTIMENT ANALYSIS RESULTS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Sentiment pairs evaluated: {sentiment_pairs_count}\n")
        if sentiment_pairs_count > 0:
            f.write(f"Sentiment Macro F1 (argmax): {sentiment_macro_f1:.4f}\n")
            f.write(f"Sentiment Micro F1 (argmax): {sentiment_micro_f1:.4f}\n")
            f.write(f"Sentiment Weighted F1 (argmax): {sentiment_weighted_f1:.4f}\n")
            f.write(f"Sentiment MSE: {sentiment_mse:.4f}\n")
            if sentiment_logloss is not None:
                f.write(f"Sentiment Log Loss: {sentiment_logloss:.4f}\n")
            f.write(f"Sentiment Brier Score: {sentiment_brier_overall:.4f}\n")
            f.write(f"Baseline Brier Score: {sentiment_brier_baseline:.4f}\n")
        else:
            f.write("No sentiment pairs to evaluate\n")
        
        f.write(f"\nCOMBINED METRICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Exact match (aspect+sentiment): {exact_match_ratio:.4f}\n")
        f.write(f"Fixed sentiment order: {SENTIMENT_LABELS}\n")

        if sentiment_classification_report:
            f.write(f"\nSENTIMENT CLASSIFICATION REPORT:\n")
            f.write("-" * 50 + "\n")
            f.write(sentiment_classification_report)

        f.write("\nCUSTOM CONFUSION MATRIX (with 'none'):\n")
        f.write("-" * 50 + "\n")
        f.write(",".join(cm_labels) + "\n")
        for lbl, row in zip(cm_labels, cm_custom):
            f.write(lbl + "," + ",".join(map(str, row)) + "\n")

        f.write("\nMULTILABEL CONFUSION MATRICES:\n")
        f.write("-" * 50 + "\n")
        for label, m in zip(all_aspects, ml_cms):
            f.write(f"\nAspect: {label}\n{m}\n")

        f.write("\nMISMATCHED PREDICTIONS (first 50):\n")
        f.write("-" * 50 + "\n")
        for m in mismatches[:50]:
            f.write(m + "\n")

    # -----------------------------
    # Print summary to console
    # -----------------------------
    print("EVALUATION COMPLETE!")
    print("=" * 50)
    print(f"Total tweets evaluated: {len(all_ids)}")
    print(f"Sentiment pairs calculated: {sentiment_pairs_count}")
    print(f"Aspect F1 Macro: {f1_macro:.4f}")
    if sentiment_macro_f1 is not None:
        print(f"Sentiment F1 Macro (argmax): {sentiment_macro_f1:.4f}")
    if sentiment_brier_overall is not None:
        print(f"Sentiment Brier Score: {sentiment_brier_overall:.4f}")
    print(f"Exact Match Ratio: {exact_match_ratio:.4f}")
    print(f"Results saved to: {output_path}")

    # -----------------------------
    # Save plots
    # -----------------------------
    try:
        # Aspect confusion matrix
        plt.figure(figsize=(max(8, len(cm_labels)), max(6, len(cm_labels))))
        sns.heatmap(cm_custom, annot=True, fmt="d", xticklabels=cm_labels,
                    yticklabels=cm_labels, cmap="Blues")
        plt.xlabel("Predicted Aspect")
        plt.ylabel("True Aspect")
        plt.title("Aspect Confusion Matrix (Custom, with 'none')")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path.parent / f"{output_path.stem}_confusion_matrix.png", dpi=300)
        plt.close()

        # Create sentiment confusion matrix if we have sentiment data
        if sentiment_pairs_count > 0:
            plt.figure(figsize=(8, 6))
            sent_cm = confusion_matrix(sentiment_true_classes, sentiment_pred_classes, 
                                      labels=list(range(NUM_SENTIMENTS)))
            sns.heatmap(sent_cm, annot=True, fmt="d", xticklabels=SENTIMENT_LABELS,
                        yticklabels=SENTIMENT_LABELS, cmap="Blues")
            plt.xlabel("Predicted Sentiment")
            plt.ylabel("True Sentiment")
            plt.title(f"Sentiment Confusion Matrix ({sentiment_pairs_count} pairs)")
            plt.tight_layout()
            plt.savefig(output_path.parent / f"{output_path.stem}_sentiment_confusion_matrix.png", dpi=300)
            plt.close()

        # Multilabel confusion matrices
        if len(all_aspects) > 0:
            n_aspects, cols = len(all_aspects), min(3, len(all_aspects))
            rows = int(np.ceil(n_aspects / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            
            # Handle different subplot configurations
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for idx, (label, m) in enumerate(zip(all_aspects, ml_cms)):
                sns.heatmap(m, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"], 
                            ax=axes[idx])
                axes[idx].set_title(f"Aspect: {label}")
            
            # Hide unused subplots
            for j in range(len(all_aspects), len(axes)):
                axes[j].axis("off")
                
            plt.tight_layout()
            plt.savefig(output_path.parent / f"{output_path.stem}_multilabel_confusion_matrices.png", dpi=300)
            plt.close()
            
    except Exception as e:
        print(f"Warning: Could not save plots: {e}")


if __name__ == "__main__":
    main()