#!/bin/bash

export OPENAI_API_KEY="sk-8raWH28nCaj9mCn4Y5hHbCGt0hCJZlic7nJyb8E32mT3BlbkFJJeT-RcHPFAxvSSprZ-cKOF50NllyFCANsKML3jpvcA"

MODEL="gpt-4.1"  # or gpt-4.1-mini

# Base directories
CORPUS_BASE_DIR="/Users/putinabillaaidira/Downloads/KULIAH/TA/echo-absa-gdm/data/corpus_final/for_absa"
CORPUS_DIR="$CORPUS_BASE_DIR/llm"
ASPECT_DIR="$CORPUS_BASE_DIR/aspect_list"
GOLD_DIR="/Users/putinabillaaidira/Downloads/KULIAH/TA/echo-absa-gdm/data/labeled/for_absa"
RESULTS_DIR="/Users/putinabillaaidira/Downloads/KULIAH/TA/echo-absa-gdm/absa/results"
GPT_OUTPUT_DIR="$RESULTS_DIR/gpt/$MODEL"
METRICS_OUTPUT_DIR="$RESULTS_DIR/metrics/gpt/$MODEL"

CONF_THRESHOLDS="0.0,0.2,0.4,0.6,0.8,1.0"
BATCH_SIZE=30

mkdir -p "$GPT_OUTPUT_DIR"
mkdir -p "$METRICS_OUTPUT_DIR"

# Create a timestamp folder for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

GPT_TIMESTAMP_DIR="$GPT_OUTPUT_DIR/$TIMESTAMP"
METRICS_TIMESTAMP_DIR="$METRICS_OUTPUT_DIR/$TIMESTAMP"

mkdir -p "$GPT_TIMESTAMP_DIR"
mkdir -p "$METRICS_TIMESTAMP_DIR"

# Loop over corpus files
for corpus_file in "$CORPUS_DIR"/*.csv; do
    base_filename=$(basename "$corpus_file" .csv)
    echo "Processing corpus: $base_filename"

    aspect_file="$ASPECT_DIR/${base_filename}.csv"
    gold_file="$GOLD_DIR/${base_filename}.csv"

    # Run GPT ABSA
    echo "Running gptabsa for all thresholds: $CONF_THRESHOLDS"
    gptabsa --input "$corpus_file" \
            --aspects "$aspect_file" \
            --output "$GPT_TIMESTAMP_DIR" \
            --mode block \
            --batch_size $BATCH_SIZE \
            --conf_thresholds "$CONF_THRESHOLDS" \
            --model "$MODEL"
    
    sleep 3

    # Calculate metrics for each threshold
    for conf in ${CONF_THRESHOLDS//,/ }; do
        conf_str=${conf//./}
        output_file=$(ls "$GPT_TIMESTAMP_DIR/${base_filename}_${conf_str}_"*.csv 2>/dev/null | head -n 1)

        if [[ -f "$output_file" ]]; then
            echo "Calculating metrics for conf_threshold=$conf"
            metric_calc --pred "$output_file" \
                        --gold "$gold_file" \
                        --output "$METRICS_TIMESTAMP_DIR/${base_filename}_${conf_str}_metrics.txt"
        else
            echo "⚠️ Warning: Output file for threshold $conf not found!"
        fi
    done

    echo "Done processing $base_filename"
done

echo "All done."
