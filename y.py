import pandas as pd

# File paths
raw_text_file = "data/corpus_final/for_echo_chamber_detection/indo_vaccination_labeled_317t.csv"
meta_file = "data/raw/Indo_vaccination_labeled.csv"
output_file = "indo_vaccination_labeled_317t.csv"

# Load CSVs
df_text = pd.read_csv(raw_text_file)
df_meta = pd.read_csv(meta_file)

# Merge on 'id'
df = pd.merge(df_text, df_meta, on="id", how="inner")

# Rename 'aspect_category' to 'aspect_category_final'
df.rename(columns={"aspect_category": "aspect_category_final"}, inplace=True)

# Map sentiment to hard_label_final
sentiment_map = {
    "Positive": "[1, 0, 0]",
    "Neutral": "[0, 1, 0]",
    "Negative": "[0, 0, 1]"
}
df["hard_label_final"] = df["aspect_sentiment"].map(sentiment_map)

# Keep only desired columns in order
df = df[["id", "text", "aspect_category_final", "aspect_sentiment", "community", "hard_label_final"]]

# Save to CSV
df.to_csv(output_file, index=False)

print(f"Merged file saved to {output_file}")
