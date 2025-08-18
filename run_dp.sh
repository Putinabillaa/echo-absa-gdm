#!/bin/bash

# Processor to run
PROCESSOR="stanza"
VECTOR_MODES=("fasttext")

# Base directories
CORPUS_BASE_DIR="/Users/putinabillaaidira/Downloads/KULIAH/TA/echo-absa-gdm/data/corpus_final/for_absa"
CORPUS_DIR="$CORPUS_BASE_DIR/dp"
ASPECT_DIR="$CORPUS_BASE_DIR/aspect_list"
LEXICON_DIR="/Users/putinabillaaidira/Downloads/KULIAH/TA/echo-absa-gdm/data/lexicon"
GOLD_DIR="/Users/putinabillaaidira/Downloads/KULIAH/TA/echo-absa-gdm/data/labeled/for_absa"
RESULTS_DIR="/Users/putinabillaaidira/Downloads/KULIAH/TA/echo-absa-gdm/absa/results"

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for VECTOR_MODE in "${VECTOR_MODES[@]}"; do
    echo "--- Using vector mode: $VECTOR_MODE ---"

    DP_OUTPUT_DIR="$RESULTS_DIR/dp/$PROCESSOR/$VECTOR_MODE"
    METRICS_OUTPUT_DIR="$RESULTS_DIR/metrics/dp/$PROCESSOR/$VECTOR_MODE"

    DP_TIMESTAMP_DIR="$DP_OUTPUT_DIR/$TIMESTAMP"
    METRICS_TIMESTAMP_DIR="$METRICS_OUTPUT_DIR/$TIMESTAMP"

    mkdir -p "$DP_TIMESTAMP_DIR"
    mkdir -p "$METRICS_TIMESTAMP_DIR"

    # Loop over corpus files
    for corpus_file in "$CORPUS_DIR"/*.csv; do
        # Skip if the file is in aspect_list folder
        if [[ "$corpus_file" == *"aspect_list"* ]]; then
            continue
        fi

        base_filename=$(basename "$corpus_file" .csv)
        echo "Processing corpus: $base_filename with $PROCESSOR and $VECTOR_MODE"

        aspect_file="$ASPECT_DIR/${base_filename}.csv"
        gold_file="$GOLD_DIR/${base_filename}.csv"
        lexicon_file="$LEXICON_DIR/idopinionwords_combined.csv"

        # Run dp
        dp --input "$corpus_file" \
           --aspects "$aspect_file" \
           --lexicon "$lexicon_file" \
           --output "$DP_TIMESTAMP_DIR/${base_filename}.csv" \
           --vector_mode "$VECTOR_MODE"

        sleep 2

        # Calculate metrics
        output_file="$DP_TIMESTAMP_DIR/${base_filename}.csv"
        if [[ -f "$output_file" ]]; then
            echo "Calculating metrics for $base_filename ($PROCESSOR, $VECTOR_MODE)"
            metric_calc --pred "$output_file" \
                        --gold "$gold_file" \
                        --output "$METRICS_TIMESTAMP_DIR/${base_filename}_metrics.txt"
        else
            echo "⚠️ Warning: Output file for $base_filename with $PROCESSOR and $VECTOR_MODE not found!"
        fi
        echo "Done processing $base_filename with $PROCESSOR and $VECTOR_MODE"
    done
done

echo "All vector modes done with processor $PROCESSOR."
