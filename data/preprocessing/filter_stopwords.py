# filter_stopwords.py
KEEP_NEGATORS = {"tidak", "tak", "bukan", "jangan", "belum", "enggak", "gak"}
KEEP_CONJUNCTIONS = {"dan", "atau", "tetapi", "namun", "tapi", "serta", "sementara", "sedangkan", '&'}
KEEP_CONTRAST = {"walaupun", "meskipun", "padahal", "biarpun", "sekalipun"}
KEEP_INTENSIFIERS = {"sangat", "banget", "amat"}
KEEP_MODAL = {"harus", "bisa", "mesti", "boleh", "dapat", "masih", "sudah"}

SAFE_KEEP = KEEP_NEGATORS | KEEP_CONJUNCTIONS | KEEP_CONTRAST | KEEP_INTENSIFIERS | KEEP_MODAL

with open("lexicon/stop_words.txt", encoding="utf-8") as f:
    full_stopwords = set([line.strip() for line in f if line.strip()])

# === 3️. Filter ===
safe_stopwords = full_stopwords - SAFE_KEEP

print(f"Total stopwords: {len(full_stopwords)}")
print(f"Safe to remove: {len(safe_stopwords)}")

# === 4️. Save ===
with open("lexicon/stop_words.txt", "w", encoding="utf-8") as f:
    for word in sorted(safe_stopwords):
        f.write(word + "\n")

print("✅")
