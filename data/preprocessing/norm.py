#norm.py
import re
import pandas as pd

# === CONFIG ===
INPUT_CSV = "labeled/biodiversity_labeled_108t.csv"
TEXT_COLUMN = "cleaned_text"
OUTPUT_CSV = "normalized/biodiversity_labeled_108t.csv"

USE_SLANGID = True
USE_TXT_DICT = True
USE_HUGGINGFACE = True
USE_STOPWORDS = True

# === OUTPUT CSV suffixes based on enabled features ===
suffixes = []

if USE_SLANGID:
    suffixes.append("slangid")
if USE_TXT_DICT:
    suffixes.append("txtdict")
if USE_HUGGINGFACE:
    suffixes.append("hf")
if USE_STOPWORDS:
    suffixes.append("stopwords")

if suffixes:
    suffix_str = "_".join(suffixes)
    OUTPUT_CSV = OUTPUT_CSV.replace(".csv", f"_{suffix_str}.csv")

# === 1. Load SlangID ===
if USE_SLANGID:
    from slangid import Translator
    slangid_normalizer = Translator()

# === 2. Load Indonesian_Slang_Dictionary.txt ===
slang_txt_map = {}
if USE_TXT_DICT:
    with open("lexicon/Indonesian_Slang_Dictionary.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            slang, formal = line.split(":", 1)
            slang_txt_map[slang.strip()] = formal.strip()

# === 3. Load Hugging Face slang dictionary ===
slang_hf_map = {}
if USE_HUGGINGFACE:
    slang_df = pd.read_csv("lexicon/slang-indo.csv")  # theonlydo/indonesia-slang
    slang_hf_map = dict(zip(slang_df['slang'], slang_df['formal']))

# === 4. Combine all maps ===
combined_slang_map = {}
combined_slang_map.update(slang_txt_map)
combined_slang_map.update(slang_hf_map)

# === 5. Load stopwords ===
stopwords = set()
if USE_STOPWORDS:
    with open("lexicon/stop_words.txt", "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)

# === 6. Normalizer ===
def normalize_text(text: str) -> str:
    if pd.isnull(text) or not isinstance(text, str):
        return ""

    # slangid normalization
    if USE_SLANGID:
        text = slangid_normalizer.translate(text)

    tokens = text.split()

    # Apply combined slang map
    normalized_tokens = [combined_slang_map.get(tok, tok) for tok in tokens]

    # Remove stopwords
    if USE_STOPWORDS:
        normalized_tokens = [tok for tok in normalized_tokens if tok.lower() not in stopwords]

    return " ".join(normalized_tokens)

# === 7. Load CSV, normalize, save ===
def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Input rows: {len(df)}")

    df["normalized_text"] = df[TEXT_COLUMN].apply(normalize_text)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
