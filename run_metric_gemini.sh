#!/bin/bash

# Model name (used for folder structure)
MODEL="gemini-2.0-flash"

# Base directories
RESULTS_DIR="/Users/putinabillaaidira/Downloads/KULIAH/TA/echo-absa-gdm/absa/results"
GOLD_DIR="/Users/putinabillaaidira/Downloads/KULIAH/TA/echo-absa-gdm/data/labeled/for_absa"

GEMINI_OUTPUT_DIR="$RESULTS_DIR/gemini/$MODEL"
METRICS_OUTPUT_DIR="$RESULTS_DIR/metrics/gemini/$MODEL"

# Loop over timestamped subfolders
for timestamp_dir in "$GEMINI_OUTPUT_DIR"/*/; do
    echo "Processing folder: $timestamp_dir"
    
    # Preserve relative folder structure for metrics
    relative_dir="${timestamp_dir#$GEMINI_OUTPUT_DIR/}"
    metrics_subdir="$METRICS_OUTPUT_DIR/$relative_dir"
    mkdir -p "$metrics_subdir"
    
    # Process all CSV prediction files in this timestamp folder
    find "$timestamp_dir" -type f -name "*.csv" | while read -r output_file; do
        base_filename=$(basename "$output_file" .csv)
        echo "Calculating metrics for: $base_filename"

        # Split filename into parts
        IFS='_' read -r -a parts <<< "$base_filename"

        # --- gold_corpus_name (drop last 2: conf + timestamp) ---
        gold_corpus_name="${parts[0]}"
        for ((i=1; i<${#parts[@]}-2; i++)); do
            gold_corpus_name="${gold_corpus_name}_${parts[i]}"
        done

        # --- metrics_corpus_name (drop only timestamp, keep conf) ---
        metrics_corpus_name="${parts[0]}"
        for ((i=1; i<${#parts[@]}-1; i++)); do
            metrics_corpus_name="${metrics_corpus_name}_${parts[i]}"
        done

        gold_file="$GOLD_DIR/${gold_corpus_name}.csv"

        if [[ -f "$gold_file" ]]; then
            # Save metrics with conf threshold, without timestamp
            metric_calc --pred "$output_file" \
                        --gold "$gold_file" \
                        --output "$metrics_subdir/${metrics_corpus_name}_metrics.txt"
        else
            echo "⚠️ Warning: Gold file not found for $gold_corpus_name"
        fi
    done
done

echo "✅ All metric calculations completed."
