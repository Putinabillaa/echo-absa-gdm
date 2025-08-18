#!/bin/bash

# Processor and vector modes
PROCESSOR="stanza"
VECTOR_MODES=("fasttext" "tfidf" "sentence")

# Base directories
RESULTS_DIR="/Users/putinabillaaidira/Downloads/KULIAH/TA/echo-absa-gdm/absa/results"
GOLD_DIR="/Users/putinabillaaidira/Downloads/KULIAH/TA/echo-absa-gdm/data/labeled/for_absa"
METRICS_BASE_DIR="$RESULTS_DIR/metrics/dp/$PROCESSOR"

# Loop over vector modes
for VECTOR_MODE in "${VECTOR_MODES[@]}"; do
    DP_OUTPUT_DIR="$RESULTS_DIR/dp/$PROCESSOR/$VECTOR_MODE"
    METRICS_OUTPUT_DIR="$METRICS_BASE_DIR/$VECTOR_MODE"

    echo "--- Processing vector mode: $VECTOR_MODE ---"

    # Loop over timestamped subfolders (or just direct results if no timestamp)
    for timestamp_dir in "$DP_OUTPUT_DIR"/*/; do
        echo "Processing folder: $timestamp_dir"
        
        # Preserve relative folder structure for metrics
        relative_dir="${timestamp_dir#$DP_OUTPUT_DIR/}"
        metrics_subdir="$METRICS_OUTPUT_DIR/$relative_dir"
        mkdir -p "$metrics_subdir"

        # Process all CSV prediction files in this folder
        find "$timestamp_dir" -type f -name "*.csv" | while read -r output_file; do
            base_filename=$(basename "$output_file" .csv)
            echo "Calculating metrics for: $base_filename"

            gold_file="$GOLD_DIR/${base_filename}.csv"

            if [[ -f "$gold_file" ]]; then
                metric_calc --pred "$output_file" \
                            --gold "$gold_file" \
                            --output "$metrics_subdir/${base_filename}_metrics.txt"
            else
                echo "⚠️ Warning: Gold file not found for $base_filename"
            fi
        done
    done
done

echo "✅ All metric calculations completed."
