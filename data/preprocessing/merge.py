#merge.py
import pandas as pd

def merge_aspects_text(text_csv, aspects_csv, output_csv):
    # Read both CSVs with id as string
    df_text = pd.read_csv(text_csv, dtype={'id': str})
    df_aspects = pd.read_csv(aspects_csv, dtype={'id': str})

    # Remove leading apostrophes and whitespace from id in aspects CSV
    df_aspects['id'] = df_aspects['id'].str.strip().str.lstrip("'")

    # Merge on 'id'
    df_merged = pd.merge(df_text, df_aspects[['id', 'subdomain', 'final label']], on='id', how='inner')

    # Save output
    df_merged.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"Done! Merged CSV saved to: {output_csv}")

if __name__ == "__main__":
    TEXT_CSV = 'cleaned/biodiversity_tweet.csv'     # CSV that has 'id' + 'text'
    ASPECTS_CSV = 'raw/biodiversity_labeled.csv'                # CSV that has 'id' + 'aspect_category' + 'aspect_sentiment'
    OUTPUT_CSV = 'labeled/biodiversity_tweet.csv'

    merge_aspects_text(TEXT_CSV, ASPECTS_CSV, OUTPUT_CSV)
