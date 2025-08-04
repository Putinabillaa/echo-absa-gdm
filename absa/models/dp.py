import stanza
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# For fastText
from gensim.models import KeyedVectors

# For Sentence Transformers
from sentence_transformers import SentenceTransformer

# === 0. Config ===
VECTOR_MODE = "sentence"  # Options: "tfidf", "fasttext", "sentence"

# === 1. Load INSET Lexicon ===
lexicon_df = pd.read_csv("../data/lexicon/inset_combined.csv")
seed_opinions = dict(zip(lexicon_df['word'], lexicon_df['weight']))

# === 2. Load Tweets ===
tweets_df = pd.read_csv("../data/normalized/indo_vaccination_labeled_175t_slangid_txtdict_hf.csv")
tweets = tweets_df['normalized_text'].tolist()

# === 3. Setup NLP ===
# stanza.download('id')
nlp = stanza.Pipeline(lang='id', processors='tokenize,mwt,pos,lemma,depparse')

# === 4. Aspect Categories ===
aspect_categories = pd.read_csv("../data/labeled/aspect_list/indo_vaccination_labeled_175t.csv")['aspect_category'].tolist()

# === 5. Prepare Vectorizer ===
if VECTOR_MODE == "tfidf":
    vectorizer = TfidfVectorizer().fit(aspect_categories)
    aspect_vecs = vectorizer.transform(aspect_categories)

elif VECTOR_MODE == "fasttext":
    print("Loading fastText models...")
    ft_id = KeyedVectors.load_word2vec_format("../data/cc.id.300.vec")
    ft_en = KeyedVectors.load_word2vec_format("../data/cc.en.300.vec")
    print("Done loading.")

    aspect_vecs = []
    for cat in aspect_categories:
        tokens = cat.lower().split()
        vecs = [ft_en[t] for t in tokens if t in ft_en]
        if vecs:
            vec = np.mean(vecs, axis=0)
        else:
            vec = np.zeros(300)
        aspect_vecs.append(vec)
    aspect_vecs = np.vstack(aspect_vecs)

elif VECTOR_MODE == "sentence":
    print("Loading Sentence Transformer...")
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    print("Encoding aspect categories...")
    aspect_vecs = model.encode(aspect_categories)

else:
    raise ValueError(f"Invalid VECTOR_MODE: {VECTOR_MODE}")

# === 6. Containers ===
results = []

# === 7. Double Propagation Core ===
for tweet in tweets:
    doc = nlp(tweet)
    found_pairs = set()

    for sent in doc.sentences:
        for word in sent.words:
            w = word.text.lower()
            if w in seed_opinions:
                for child in sent.words:
                    if child.head == word.id and child.upos == "NOUN":
                        found_pairs.add((child.text.lower(), w))
                    if word.upos == "ADJ" and word.head > 0:
                        head = sent.words[word.head - 1]
                        if head.upos == "NOUN":
                            found_pairs.add((head.text.lower(), w))

    if not found_pairs:
        continue

    targets_opinions = defaultdict(list)
    for target, opinion in found_pairs:
        weight = seed_opinions.get(opinion, 0)
        targets_opinions[target].append((opinion, weight))
        
    for target, opinions in targets_opinions.items():
        if VECTOR_MODE == "tfidf":
            target_vec = vectorizer.transform([target])
            sims = cosine_similarity(target_vec, aspect_vecs).flatten()
        elif VECTOR_MODE == "fasttext":
            if target in ft_id:
                target_vec = ft_id[target].reshape(1, -1)
                sims = cosine_similarity(target_vec, aspect_vecs).flatten()
            else:
                sims = np.zeros(len(aspect_categories))
        elif VECTOR_MODE == "sentence":
            target_vec = model.encode([target])
            sims = cosine_similarity(target_vec, aspect_vecs).flatten()

        best_idx = sims.argmax()
        best_aspect_category = aspect_categories[best_idx]
        best_sim = sims[best_idx]

        total_weight = sum(w for _, w in opinions)

        score = np.tanh(total_weight)
        pos = max(score, 0)
        neg = abs(min(score, 0))
        neu = max(0, 1 - (pos + neg))
        prob = [round(pos, 4), round(neu, 4), round(neg, 4)]

        results.append({
            "tweet": tweet,
            "aspect_term": target,
            "aspect_category": best_aspect_category,
            "similarity": round(best_sim, 4),
            "matched_opinions": ", ".join(f"{w}({wt})" for w, wt in opinions),
            "total_weight": total_weight,
            "sentiment_prob": prob
        })

# === 8. Save ===
results_df = pd.DataFrame(results)
results_df.to_csv(f"double_propagation_result_{VECTOR_MODE}.csv", index=False)
print(f"Done. Saved to double_propagation_result_{VECTOR_MODE}.csv")
