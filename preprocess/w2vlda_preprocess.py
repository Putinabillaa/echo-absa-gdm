import re

# === Config ===
CORPUS_FILE = "corpus_final/kampus_merdeka_labeled_150t_slangid_txtdict_hf_stopwords.txt"
STOPWORDS_FILE = "lexicon/combined_stop_words.txt"
POS_LEXICON = "lexicon/idopinionwords_positive.txt"
NEG_LEXICON = "lexicon/idopinionwords_negative.txt"
OUTPUT_FILE = "corpus_final/kampus_merdeka_labeled_150t_w2vlda.txt"

# === Load stopwords ===
with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
    stopwords = set(line.strip() for line in f if line.strip())

# === Load phrases only ===
def load_phrases(file_path):
    phrases = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            phrase = line.strip()
            if not phrase:
                continue
            if " " in phrase:  # Only multi-word
                phrases.append(phrase)
    return phrases

pos_phrases = load_phrases(POS_LEXICON)
neg_phrases = load_phrases(NEG_LEXICON)

# === Merge all unique phrases ===
ALL_PHRASES = sorted(set(pos_phrases + neg_phrases), key=len, reverse=True)

print(f"Total unique phrases to merge: {len(ALL_PHRASES)}")

# === Preprocessing helper ===
def preprocess_line(text):
    # Tokenize by splitting on spaces
    tokens = text.strip().split()
    text = " ".join(tokens)

    # Merge known phrases first
    for phrase in ALL_PHRASES:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        replacement = phrase.replace(" ", "_")
        text = re.sub(pattern, replacement, text)

    # Additional: merge "tidak" with next word
    # Example: "tidak suka" -> "tidak_suka"
    tokens = text.strip().split()
    merged_tokens = []
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token.lower() == "tidak" and i + 1 < len(tokens):
            merged_tokens.append(f"tidak_{tokens[i + 1]}")
            skip_next = True
        else:
            merged_tokens.append(token)
    tokens = merged_tokens

    # Remove stopwords after merging
    tokens = [w for w in tokens if w.lower() not in stopwords]

    return " ".join(tokens)

# === Process corpus ===
with open(CORPUS_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        cleaned = preprocess_line(line)
        fout.write(cleaned + "\n")

print(f"✅ Preprocessed corpus saved to {OUTPUT_FILE}")

# === Merge all phrases in POS and NEG lexicons ===
def merge_phrases_file(file_path):
    merged_phrases = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            phrase = line.strip()
            if not phrase:
                continue
            if " " in phrase:
                phrase = phrase.replace(" ", "_")
            merged_phrases.append(phrase)
    return merged_phrases

# Overwrite POS file
merged_pos = merge_phrases_file(POS_LEXICON)
with open(POS_LEXICON, "w", encoding="utf-8") as f:
    for phrase in merged_pos:
        f.write(phrase + "\n")

# Overwrite NEG file
merged_neg = merge_phrases_file(NEG_LEXICON)
with open(NEG_LEXICON, "w", encoding="utf-8") as f:
    for phrase in merged_neg:
        f.write(phrase + "\n")

print(f"✅ Merged phrases written back to {POS_LEXICON} and {NEG_LEXICON}")
