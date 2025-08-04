#cleaning.py
import re
import pandas as pd

def clean_line(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.splitlines())     # removes all linebreaks robustly
    text = re.sub(r'\s+', ' ', text)  
    text = text.strip()
    return text

def clean_csv_column(input_csv: str, output_csv: str, text_column: str, id_column: str = 'id'):
    """
    Membersihkan teks dari kolom tertentu pada CSV.
    Menyimpan ID, teks asli, dan teks yang sudah dibersihkan ke CSV baru.
    """
    df = pd.read_csv(input_csv)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV columns: {df.columns.tolist()}")

    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in CSV columns: {df.columns.tolist()}")

    df['cleaned_text'] = df[text_column].apply(clean_line)

    output_df = df[[id_column, 'cleaned_text']]

    output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"Done! Cleaned text saved to: {output_csv}")

if __name__ == "__main__":
    INPUT_CSV = 'raw/biodiversity_labeled_hydrated.csv'
    OUTPUT_CSV = 'cleaned/biodiversity_tweet.csv'
    TEXT_COLUMN = 'text'
    ID_COLUMN = 'id'  # Pastikan kolom ID di CSV-mu memang bernama 'id'
    clean_csv_column(INPUT_CSV, OUTPUT_CSV, TEXT_COLUMN, ID_COLUMN)
