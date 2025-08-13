# helper.py

import os
import re
import csv
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LEXICON_DIR = os.path.join(BASE_DIR, "data", "lexicon")

def normalize_csv(
    input_csv: str,
    output_csv: str,
    text_column: str = "text",
    use_slangid: bool = True,
    use_txt_dict: bool = True,
    use_huggingface: bool = True,
    use_stopwords: bool = True
):
    output_csv = resolve_output_path(input_csv, output_csv)

    # === SlangID ===
    if use_slangid:
        from slangid import Translator
        slangid_normalizer = Translator()
    else:
        slangid_normalizer = None

    # === Local txt dict ===
    slang_txt_map = {}
    if use_txt_dict:
        with open(os.path.join(LEXICON_DIR, "Indonesian_Slang_Dictionary.txt")) as f:
            for line in f:
                line = line.strip()
                if line and ":" in line:
                    slang, formal = line.split(":", 1)
                    slang_txt_map[slang.strip()] = formal.strip()

    # === Hugging Face dict ===
    slang_hf_map = {}
    if use_huggingface:
        slang_df = pd.read_csv(os.path.join(LEXICON_DIR, "slang-indo.csv"))
        slang_hf_map = dict(zip(slang_df['slang'], slang_df['formal']))

    # === Stopwords ===
    stopwords = set()
    if use_stopwords:
        with open(os.path.join(LEXICON_DIR, "stop_words.txt"), encoding="utf-8") as f:
            stopwords = {line.strip() for line in f if line.strip()}

    # === Combine maps ===
    combined_map = {}
    combined_map.update(slang_txt_map)
    combined_map.update(slang_hf_map)

    def normalize_text(text: str) -> str:
        if pd.isnull(text) or not isinstance(text, str):
            return ""
        if slangid_normalizer:
            text = slangid_normalizer.translate(text)
        tokens = text.split()
        tokens = [combined_map.get(tok, tok) for tok in tokens]
        if use_stopwords:
            tokens = [tok for tok in tokens if tok.lower() not in stopwords]
        return " ".join(tokens)

    df = pd.read_csv(input_csv)
    print(f"Input rows: {len(df)}")

    df["text"] = df[text_column].apply(normalize_text)
    df.to_csv(output_csv, index=False)
    print(f"✅ Done! Normalized CSV saved to {output_csv}")
    
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
    df['id'] = df.groupby('text').ngroup() + 1
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

import re

def min_clean_line(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[,]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = ' '.join(text.splitlines())
    return text

def clean_csv_column(input_csv: str, output_csv: str, text_column: str, id_column: str = 'id'):
    output_csv = resolve_output_path(input_csv, output_csv)
    df = pd.read_csv(input_csv)
    if text_column not in df.columns or id_column not in df.columns:
        raise ValueError(f"Columns '{text_column}' or '{id_column}' not found in {df.columns.tolist()}")
    df.dropna(subset=[text_column], inplace=True)
    df['text'] = df[text_column].apply(clean_line)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Done! Cleaned text saved to {output_csv}")

def csv_to_txt(input_csv: str, output_txt: str, text_column: str = 'text'):
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
    df_selected = df[["id", "text"]].drop_duplicates()
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
    df_merged.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Merged CSV saved to {output_csv}")

def replace_special_phrase(input_csv: str, output_csv: str, phrase: str, replacement: str):
    output_csv = resolve_output_path(input_csv, output_csv)
    with open(input_csv, mode='r', encoding='utf-8') as infile, \
         open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row['text'] = row['text'].replace(phrase, replacement)
            writer.writerow(row)
    print(f"✅ Replaced '{phrase}' with '{replacement}' and saved to {output_csv}")

def drop_columns_by_names(input_csv, output_csv, columns, invert=False):
    output_csv = resolve_output_path(input_csv, output_csv)
    df = pd.read_csv(input_csv)
    if invert:
        # Keep only the listed columns (if exist)
        cols_to_keep = [col for col in columns if col in df.columns]
        df = df[cols_to_keep]
    else:
        # Drop listed columns if exist
        df = df.drop(columns=columns, errors='ignore')
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Saved file after {'keeping' if invert else 'dropping'} columns {columns} to {output_csv}")

def drop_rows_by_condition(input_csv, output_csv, condition, invert=False):
    output_csv = resolve_output_path(input_csv, output_csv)
    df = pd.read_csv(input_csv)
    try:
        filtered = df.query(condition)
    except Exception as e:
        print(f"⚠️ Error in condition '{condition}': {e}")
        filtered = df  # fallback: no filtering

    if not invert:
        df_result = df.loc[~df.index.isin(filtered.index)]
    else:
        df_result = filtered

    df_result.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Saved file after {'keeping' if not invert else 'dropping'} rows matching condition '{condition}' to {output_csv}")

def min_clean(input_csv: str, output_csv: str, text_column: str = "text"):
    output_csv = resolve_output_path(input_csv, output_csv)
    df = pd.read_csv(input_csv)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {df.columns.tolist()}")
    df.dropna(subset=[text_column], inplace=True)
    df[text_column] = df[text_column].apply(min_clean_line)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Minimal cleaning done! Cleaned CSV saved to {output_csv}")