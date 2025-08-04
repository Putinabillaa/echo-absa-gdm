# helper.py

import os
import re
import csv
import pandas as pd

def resolve_output_path(input_path: str, output_arg: str) -> str:
    """
    If output_arg is a folder, reuse input filename.
    Raises ValueError if input and output paths would be the same.
    """
    input_abs = os.path.abspath(input_path)

    if os.path.isdir(output_arg):
        filename = os.path.basename(input_path)
        output_abs = os.path.abspath(os.path.join(output_arg, filename))
    else:
        output_abs = os.path.abspath(output_arg)

    if input_abs == output_abs:
        raise ValueError(f"❌ Output path must not overwrite input: {input_abs}")

    return output_abs

def add_id(input_csv: str, output_csv: str):
    output_csv = resolve_output_path(input_csv, output_csv)
    df = pd.read_csv(input_csv)
    df['id'] = df.groupby('cleaned_text').ngroup() + 1
    df.to_csv(output_csv, index=False)
    print(f"✅ Done! IDs added and saved to {output_csv}")

def clean_line(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.splitlines())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_csv_column(input_csv: str, output_csv: str, text_column: str, id_column: str = 'id'):
    output_csv = resolve_output_path(input_csv, output_csv)
    df = pd.read_csv(input_csv)
    if text_column not in df.columns or id_column not in df.columns:
        raise ValueError(f"Columns '{text_column}' or '{id_column}' not found in {df.columns.tolist()}")
    df['cleaned_text'] = df[text_column].apply(clean_line)
    df[[id_column, 'cleaned_text']].to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Done! Cleaned text saved to {output_csv}")

def csv_to_txt(input_csv: str, output_txt: str, text_column: str = 'normalized_text'):
    output_txt = resolve_output_path(input_csv, output_txt)
    with open(input_csv, 'r', encoding='utf-8') as csvfile, open(output_txt, 'w', encoding='utf-8') as txtfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row[text_column].strip()
            if text:
                txtfile.write(text + '\n')
    print(f"✅ Done! Text saved to {output_txt}")

def extract_unique_aspects(input_csv: str, output_csv: str, aspect_column: str = "aspect_category_final"):
    output_csv = resolve_output_path(input_csv, output_csv)
    unique_aspects = set()
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            unique_aspects.add(row[aspect_column].strip())
    with open(output_csv, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["aspect_category"])
        for aspect in sorted(unique_aspects):
            writer.writerow([aspect])
    print(f"✅ Saved {len(unique_aspects)} unique aspect categories to {output_csv}")

def extract_corpus(input_csv: str, output_csv: str):
    output_csv = resolve_output_path(input_csv, output_csv)
    df = pd.read_csv(input_csv)
    df_selected = df[["id", "normalized_text"]].drop_duplicates()
    df_selected.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df_selected)} unique rows to {output_csv}")

def filter_stopwords(stopwords_file: str, safe_keep: set):
    with open(stopwords_file, encoding="utf-8") as f:
        full_stopwords = set(line.strip() for line in f if line.strip())
    safe_stopwords = full_stopwords - safe_keep
    with open(stopwords_file, "w", encoding="utf-8") as f:
        for word in sorted(safe_stopwords):
            f.write(word + "\n")
    print(f"✅ Filtered stopwords saved back to {stopwords_file}")

def merge_aspects_text(text_csv: str, aspects_csv: str, output_csv: str):
    output_csv = resolve_output_path(text_csv, output_csv)
    df_text = pd.read_csv(text_csv, dtype={'id': str})
    df_aspects = pd.read_csv(aspects_csv, dtype={'id': str})
    df_aspects['id'] = df_aspects['id'].str.strip().str.lstrip("'")
    df_merged = pd.merge(df_text, df_aspects, on='id', how='inner')
    df_merged.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Merged CSV saved to {output_csv}")

def replace_special_phrase(input_csv: str, output_csv: str, phrase: str, replacement: str):
    output_csv = resolve_output_path(input_csv, output_csv)
    with open(input_csv, mode='r', encoding='utf-8') as infile, \
         open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row['normalized_text'] = row['normalized_text'].replace(phrase, replacement)
            writer.writerow(row)
    print(f"✅ Replaced '{phrase}' with '{replacement}' and saved to {output_csv}")
