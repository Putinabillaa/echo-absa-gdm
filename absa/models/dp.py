import os
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import stanza
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer

# --------------------------
# Constants
# --------------------------
NEGATION_WORDS = {"tidak", "tak", "bukan", "jangan", "belum", "enggak"}

# --------------------------
# Main
# --------------------------

def main():
    parser = argparse.ArgumentParser(description="Dependency Parsing ABSA Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--aspects", required=True, help="Aspects CSV file path")
    parser.add_argument("--lexicon", required=True, help="Lexicon CSV file path")
    parser.add_argument("--output", required=True, help="Output file OR folder path")
    parser.add_argument("--vector_mode", choices=["tfidf", "fasttext", "sentence"], default="tfidf")
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if output_path.is_dir() or args.output.endswith("/"):
        output_path = output_path / f"{input_path.stem}_{int(time.time())}.csv"

    if output_path.resolve() == input_path:
        raise ValueError(f"❌ Output file must not overwrite input file!\nInput: {input_path}\nOutput: {output_path}")

    os.makedirs(output_path.parent, exist_ok=True)

    start_time = time.time()

    CONFIG = {
        "VECTOR_MODE": args.vector_mode,
        "MAX_ITER": args.max_iter,
        "USE_CLAUSE_PRUNING": True,
        "USE_GLOBAL_PRUNING": True,
        "MIN_FREQ_GLOBAL": 2,
        "Q_NOUNS": 2,
        "K_ADJ": 1,
        "NEGATION_WINDOW": 5,
    }

    # === 1. Load Lexicon ===
    lexicon_df = pd.read_csv(args.lexicon)
    seed_opinions = dict(zip(lexicon_df['word'], lexicon_df['weight']))

    # === 2. Load Tweets ===
    tweets_df = pd.read_csv(input_path)
    tweets = tweets_df.to_dict("records")

    # === 3. NLP ===
    nlp = stanza.Pipeline(lang='id', processors='tokenize,mwt,pos,lemma,depparse', download_method=None)

    # === 4. Aspect Categories ===
    aspect_df = pd.read_csv(args.aspects)
    aspect_categories = aspect_df['aspect_category'].tolist()
    aspect_descs = aspect_df['desc'].tolist()
    aspect_texts = [f"{cat}. {desc}" for cat, desc in zip(aspect_categories, aspect_descs)]

    # === 5. Vectorizer ===
    if CONFIG["VECTOR_MODE"] == "tfidf":
        vectorizer = TfidfVectorizer().fit(aspect_texts)
        aspect_vecs = vectorizer.transform(aspect_texts)
    elif CONFIG["VECTOR_MODE"] == "fasttext":
        print("Loading fastText...")
        ft_id = KeyedVectors.load_word2vec_format("../data/lexicon/cc.id.300.vec.gz")
        aspect_vecs = []
        for cat in aspect_texts:
            tokens = cat.lower().split()
            vecs = [ft_id[t] for t in tokens if t in ft_id]
            vec = np.mean(vecs, axis=0) if vecs else np.zeros(300)
            aspect_vecs.append(vec)
        aspect_vecs = np.vstack(aspect_vecs)
    elif CONFIG["VECTOR_MODE"] == "sentence":
        print("Loading Sentence Transformer...")
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        aspect_vecs = model.encode(aspect_texts)

    # === 6. Core ===
    known_opinions = set(seed_opinions.keys())
    known_targets = set()
    found_pairs = defaultdict(set)

    for row in tweets:
        row["nlp_doc"] = nlp(row["text"])

    iteration = 0
    while iteration < CONFIG["MAX_ITER"]:
        iteration += 1
        print(f"Iteration {iteration}")
        new_opinions, new_targets = set(), set()

        for row in tweets:
            tweet_id = row["id"]
            doc = row["nlp_doc"]
            for sent in doc.sentences:
                words = sent.words
                for word in words:
                    w = word.text.lower()
                    # DP extraction logic same as before...
                    for child in words:
                        if w in known_opinions:
                            if child.head == word.id and child.upos == "NOUN":
                                new_targets.add(child.text.lower())
                                found_pairs[tweet_id].add((child.text.lower(), w))
                            if child.head == word.id and child.deprel == "conj" and child.upos == "ADJ":
                                new_opinions.add(child.text.lower())
                        if w in known_targets:
                            if child.head == word.id and child.upos == "ADJ":
                                new_opinions.add(child.text.lower())
                            if child.head == word.id and child.deprel == "conj" and child.upos == "NOUN":
                                new_targets.add(child.text.lower())

        if not new_opinions and not new_targets:
            print(f"✅ Converged after {iteration} iterations.")
            break

        known_opinions.update(new_opinions)
        known_targets.update(new_targets)

    # === 7. Clause pruning ===
    if CONFIG["USE_CLAUSE_PRUNING"]:
        target_counter = Counter([t for v in found_pairs.values() for t, _ in v])
        for row in tweets:
            tweet_id = row["id"]
            doc = row["nlp_doc"]
            pairs = list(found_pairs[tweet_id])
            keep_targets = set()
            for sent in doc.sentences:
                clause_targets = [t for t, _ in pairs if t in [w.text.lower() for w in sent.words]]
                if len(clause_targets) <= 1: continue
                keep = {w.text.lower() for w in sent.words if w.deprel == "conj"}
                sorted_clause = sorted(clause_targets, key=lambda x: -target_counter[x])
                keep.add(sorted_clause[0])
                pairs = [(t, o) for t, o in pairs if t in keep]
            found_pairs[tweet_id] = set(pairs)

    # === 8. Phrase expansion & global prune ===
    expanded = []
    for row in tweets:
        tweet_id, text = row["id"], row["text"]
        doc = row["nlp_doc"]
        for sent in doc.sentences:
            words = sent.words
            for idx, word in enumerate(words):
                w = word.text.lower()
                if any(w == t for t, _ in found_pairs[tweet_id]):
                    phrase = []
                    if idx-CONFIG["K_ADJ"] >= 0 and words[idx-CONFIG["K_ADJ"]].upos == "ADJ":
                        phrase.append(words[idx-CONFIG["K_ADJ"]].text.lower())
                    for i in range(CONFIG["Q_NOUNS"]):
                        if idx-(i+1) >= 0 and words[idx-(i+1)].upos == "NOUN":
                            phrase.insert(0, words[idx-(i+1)].text.lower())
                    phrase.append(w)
                    for i in range(CONFIG["Q_NOUNS"]):
                        if idx+(i+1) < len(words) and words[idx+(i+1)].upos == "NOUN":
                            phrase.append(words[idx+(i+1)].text.lower())
                    expanded.append(" ".join(phrase))

    if CONFIG["USE_GLOBAL_PRUNING"]:
        target_counter.update(expanded)
        expanded = [p for p in expanded if target_counter[p.split()[-1]] >= CONFIG["MIN_FREQ_GLOBAL"]]

    # === 9. Final scoring ===
    results = []
    for row in tweets:
        tweet_id, text = row["id"], row["text"]
        doc = row["nlp_doc"]
        pairs = found_pairs[tweet_id]
        token_index = [w.text.lower() for s in doc.sentences for w in s.words]
        unique_targets = {t for t, _ in pairs}
        for target in unique_targets:
            matched = {o for t, o in pairs if t == target}
            weight = 0
            for matched_op in matched:
                raw_weight = seed_opinions.get(matched_op, 0)
                if raw_weight == 0:
                    continue
                flipped = False
                for sent in doc.sentences:
                    sent_tokens = [w.text.lower() for w in sent.words]
                    if matched_op in sent_tokens:
                        idx = sent_tokens.index(matched_op)
                        window = sent_tokens[max(0, idx-CONFIG["NEGATION_WINDOW"]):idx+CONFIG["NEGATION_WINDOW"]+1]
                        if any(w in NEGATION_WORDS for w in window):
                            flipped = True
                            break
                if flipped:
                    raw_weight *= -1
                weight += raw_weight
            if weight == 0:
                review_weight = sum(seed_opinions.get(w, 0) for w in token_index)
                weight = 1 if review_weight > 0 else -1 if review_weight < 0 else 0

            context_phrase = target
            for sent in doc.sentences:
                sent_tokens = [w.text.lower() for w in sent.words]
                if target in sent_tokens:
                    idx = sent_tokens.index(target)
                    left = sent_tokens[idx-1] if idx-1 >= 0 else ""
                    right = sent_tokens[idx+1] if idx+1 < len(sent_tokens) else ""
                    context_phrase = f"{left} {target} {right}".strip()
                    break

            if CONFIG["VECTOR_MODE"] == "tfidf":
                vec = vectorizer.transform([context_phrase])
                sims = cosine_similarity(vec, aspect_vecs).flatten()
            elif CONFIG["VECTOR_MODE"] == "fasttext":
                vecs = [ft_id[t] for t in context_phrase.split() if t in ft_id]
                vec = np.mean(vecs, axis=0).reshape(1,-1) if vecs else np.zeros((1,300))
                sims = cosine_similarity(vec, aspect_vecs).flatten()
            elif CONFIG["VECTOR_MODE"] == "sentence":
                vec = model.encode([context_phrase])
                sims = cosine_similarity(vec, aspect_vecs).flatten()

            best_idx = sims.argmax()
            best_sim = np.clip(sims[best_idx], 0, 1)
            raw_pos = max(weight, 0)
            raw_neg = abs(min(weight, 0))
            raw_neu = 1
            total = raw_pos + raw_neg + raw_neu
            pos = raw_pos / total
            neg = raw_neg / total
            neu = raw_neu / total

            results.append({
                "id": tweet_id,
                "original_text": text,
                "aspect_term": target,
                "aspect_category": aspect_categories[best_idx],
                "similarity": round(best_sim, 4),
                "matched_opinions": ",".join(matched),
                "total_weight": weight,
                "sentiment_prob": [round(pos, 4), round(neu, 4), round(neg, 4)],
                "context_phrase": context_phrase
            })

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ Done saved to {output_path}")
    elapsed_time = time.time() - start_time
    print(f"⏱️ Total time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
