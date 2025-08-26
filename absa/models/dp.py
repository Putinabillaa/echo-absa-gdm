import os
import time
import argparse
import requests
from pathlib import Path
from collections import defaultdict, Counter

import stanza
# Remove this line since we'll use API instead:
# from ufal.udpipe import Model, Pipeline

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

# UDPipe API configuration
UDPIPE_API_URL = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
UDPIPE_MODEL = "indonesian-csui-ud-2.12-230717" 


# --------------------------
# UDPipe API Helper
# --------------------------
def udpipe_api_process(text, model=UDPIPE_MODEL):
    """Process text using UDPipe API"""
    try:
        data = {
            'data': text,
            'model': model,
            'tokenizer': '',
            'tagger': '',
            'parser': '',
            'output': 'conllu'
        }
        
        response = requests.post(UDPIPE_API_URL, data=data, timeout=30)
        response.raise_for_status()
        
        # The API returns JSON with 'result' field containing CoNLL-U format
        result = response.json()
        if 'result' in result:
            return result['result']
        else:
            print(f"⚠️ Unexpected API response format: {result}")
            return ""
            
    except requests.exceptions.RequestException as e:
        print(f"❌ UDPipe API error: {e}")
        return ""
    except Exception as e:
        print(f"❌ Error processing text: {e}")
        return ""


# --------------------------
# NLP Loader
# --------------------------
def load_nlp(processor="stanza"):
    if processor == "stanza":
        return stanza.Pipeline(
            lang="id", processors="tokenize,mwt,pos,lemma,depparse", download_method=None
        )
    elif processor == "udpipe":
        # For API mode, we don't need to load a local model
        # Just return a placeholder that indicates API mode
        return "udpipe_api"
    else:
        raise ValueError(f"Unknown processor: {processor}")


def process_text(nlp, text: str, processor="stanza"):
    sentences = []
    if processor == "stanza":
        doc = nlp(text)
        for sent in doc.sentences:
            words = []
            for w in sent.words:
                words.append({
                    "id": w.id,
                    "text": w.text,
                    "lemma": w.lemma,
                    "pos": w.upos,
                    "head": w.head,
                    "deprel": w.deprel,
                })
            sentences.append({"words": words})
    elif processor == "udpipe":
        # Use API instead of local model
        processed = udpipe_api_process(text)
        if not processed:
            print(f"⚠️ Failed to process text via API: {text[:50]}...")
            return []
            
        # Parse CoNLL-U format from API response
        for sent_str in processed.strip().split("\n\n"):
            if not sent_str.strip():
                continue
            lines = [l for l in sent_str.split("\n") if l.strip() and not l.startswith("#")]
            words = []
            for line in lines:
                parts = line.split("\t")
                if len(parts) < 8:
                    continue
                try:
                    idx, form, lemma, upos, xpos, feats, head, deprel = parts[:8]
                    # Handle multi-word tokens (skip them)
                    if "-" in idx:
                        continue
                    words.append({
                        "id": int(idx),
                        "text": form,
                        "lemma": lemma,
                        "pos": upos,
                        "head": int(head) if head.isdigit() else 0,
                        "deprel": deprel,
                    })
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Error parsing line: {line} - {e}")
                    continue
            if words:
                sentences.append({"words": words})
    return sentences


# --------------------------
# Enhanced error handling and batching for API
# --------------------------
def process_texts_batch(nlp, texts, processor="stanza", batch_size=10):
    """Process multiple texts with batching for API efficiency"""
    results = []
    
    if processor == "udpipe":
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            batch_results = []
            for text in batch:
                result = process_text(nlp, text, processor)
                batch_results.append(result)
                # Small delay to be respectful to the API
                time.sleep(0.1)
            
            results.extend(batch_results)
    else:
        # For stanza, process normally
        for text in texts:
            results.append(process_text(nlp, text, processor))
    
    return results


# --------------------------
# Sentiment calculation (unchanged)
# --------------------------
def calculate_sentiment_probabilities(weight, confidence_factor=2.0):
    if weight == 0:
        return (0.0, 1.0, 0.0)
    abs_weight = abs(weight)
    scaled_weight = abs_weight * confidence_factor
    if weight > 0:
        pos_raw = scaled_weight
        neg_raw = 0.1
        neu_raw = max(0.1, 1.0 - abs_weight * 0.5)
    else:
        pos_raw = 0.1
        neg_raw = scaled_weight
        neu_raw = max(0.1, 1.0 - abs_weight * 0.5)
    total = pos_raw + neu_raw + neg_raw
    return (pos_raw / total, neu_raw / total, neg_raw / total)


def calculate_contextual_sentiment(target, matched_opinions, doc, seed_opinions, negation_window=5):
    total_weight = 0
    opinion_count = 0
    for matched_op in matched_opinions:
        raw_weight = seed_opinions.get(matched_op, 0)
        if raw_weight == 0:
            continue
        opinion_count += 1
        flipped = False
        for sent in doc:
            sent_tokens = [w["text"].lower() for w in sent["words"]]
            if matched_op in sent_tokens:
                idx = sent_tokens.index(matched_op)
                window = sent_tokens[max(0, idx - negation_window):idx + negation_window + 1]
                if any(w in NEGATION_WORDS for w in window):
                    flipped = True
                    break
        if flipped:
            raw_weight *= -1
        total_weight += raw_weight
    if opinion_count == 0:
        token_index = [w["text"].lower() for s in doc for w in s["words"]]
        doc_sentiment = sum(seed_opinions.get(w, 0) for w in token_index)
        if doc_sentiment != 0:
            total_weight = doc_sentiment * 0.3
        else:
            total_weight = 0
    return total_weight, opinion_count


# --------------------------
# Main (modified to use batch processing)
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Dependency Parsing ABSA Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--aspects", required=True, help="Aspects CSV file path")
    parser.add_argument("--lexicon", required=True, help="Lexicon CSV file path")
    parser.add_argument("--output", required=True, help="Output file OR folder path")
    parser.add_argument("--vector_mode", choices=["tfidf", "fasttext", "sentence"], default="tfidf")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--confidence_factor", type=float, default=2.0)
    parser.add_argument("--processor", choices=["stanza", "udpipe"], default="stanza")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for UDPipe API processing")
    parser.add_argument("--udpipe_model", default="indonesian-csui-ud-2.12-230717", help="UDPipe model to use")
    
    args = parser.parse_args()

    # Set global UDPipe model if specified
    global UDPIPE_MODEL
    UDPIPE_MODEL = args.udpipe_model

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if output_path.is_dir() or args.output.endswith("/"):
        output_path = output_path / f"{input_path.stem}_{int(time.time())}.csv"
    if output_path.resolve() == input_path:
        raise ValueError("❌ Output file must not overwrite input file!")
    os.makedirs(output_path.parent, exist_ok=True)
    start_time = time.time()

    CONFIG = {
        "VECTOR_MODE": args.vector_mode,
        "MAX_ITER": args.max_iter,
        "USE_CLAUSE_PRUNING": True,
        "USE_GLOBAL_PRUNING": True,
        "MIN_FREQ_GLOBAL": 1,
        "Q_NOUNS": 2,
        "K_ADJ": 1,
        "NEGATION_WINDOW": 5,
        "CONFIDENCE_FACTOR": args.confidence_factor,
    }

    lexicon_df = pd.read_csv(args.lexicon)
    seed_opinions = dict(zip(lexicon_df["word"], lexicon_df["weight"]))

    tweets_df = pd.read_csv(input_path, dtype={"id": str})
    tweets = tweets_df.to_dict("records")

    print(f"Loading NLP pipeline with {args.processor}...")
    nlp = load_nlp(args.processor)
    
    # Process texts with batching for API efficiency
    texts = [row["text"] for row in tweets]
    if args.processor == "udpipe":
        print(f"Processing {len(texts)} texts via UDPipe API (batch size: {args.batch_size})")
        processed_docs = process_texts_batch(nlp, texts, processor=args.processor, batch_size=args.batch_size)
    else:
        processed_docs = [process_text(nlp, text, processor=args.processor) for text in texts]
    
    # Assign processed documents back to tweets
    for row, doc in zip(tweets, processed_docs):
        row["nlp_doc"] = doc

    aspect_df = pd.read_csv(args.aspects)
    aspect_categories = aspect_df["aspect_category"].tolist()
    aspect_descs = aspect_df["desc"].tolist()
    aspect_texts = [f"{cat}. {desc}" for cat, desc in zip(aspect_categories, aspect_descs)]

    print(f"Initializing {CONFIG['VECTOR_MODE']} vectorizer...")
    if CONFIG["VECTOR_MODE"] == "tfidf":
        vectorizer = TfidfVectorizer().fit(aspect_texts)
        aspect_vecs = vectorizer.transform(aspect_texts)
    elif CONFIG["VECTOR_MODE"] == "fasttext":
        print("Loading fastText...")
        ft_id = KeyedVectors.load_word2vec_format("/Users/putinabillaaidira/Downloads/KULIAH/TA/echo-absa-gdm/data/lexicon/cc.id.300.vec.gz")
        aspect_vecs = []
        for cat in aspect_texts:
            tokens = cat.lower().split()
            vecs = [ft_id[t] for t in tokens if t in ft_id]
            vec = np.mean(vecs, axis=0) if vecs else np.zeros(300)
            aspect_vecs.append(vec)
        aspect_vecs = np.vstack(aspect_vecs)
    elif CONFIG["VECTOR_MODE"] == "sentence":
        print("Loading Sentence Transformer...")
        model = SentenceTransformer("firqaaa/indo-sentence-bert-base")
        aspect_vecs = model.encode(aspect_texts)

    # === Core Iterative Expansion (unchanged) ===
    print("Starting iterative opinion/target expansion...")
    known_opinions = set(seed_opinions.keys())
    known_targets = set()
    found_pairs = defaultdict(set)

    # adapt POS rules
    target_pos = {"NOUN"}
    opinion_pos = {"ADJ"}
    if args.processor == "udpipe":
        target_pos |= {"PROPN"}
        opinion_pos |= {"ADV"}

    iteration = 0
    while iteration < CONFIG["MAX_ITER"]:
        iteration += 1
        print(f"Iteration {iteration}")
        new_opinions, new_targets = set(), set()
        for row in tweets:
            tweet_id = row["id"]
            doc = row["nlp_doc"]
            for sent in doc:
                words = sent["words"]
                for word in words:
                    w = word["text"].lower()
                    for child in words:
                        if w in known_opinions:
                            if child["head"] == word["id"] and child["pos"] in target_pos:
                                new_targets.add(child["text"].lower())
                                found_pairs[tweet_id].add((child["text"].lower(), w))
                            if child["head"] == word["id"] and child["deprel"] == "conj" and child["pos"] in opinion_pos:
                                new_opinions.add(child["text"].lower())
                        if w in known_targets:
                            if child["head"] == word["id"] and child["pos"] in opinion_pos:
                                new_opinions.add(child["text"].lower())
                            if child["head"] == word["id"] and child["deprel"] == "conj" and child["pos"] in target_pos:
                                new_targets.add(child["text"].lower())
        if not new_opinions and not new_targets:
            print(f"✅ Converged after {iteration} iterations.")
            break
        known_opinions.update(new_opinions)
        known_targets.update(new_targets)

    print(f"Found {len(known_opinions)} opinion terms and {len(known_targets)} target terms")

    # === Rest of the processing (unchanged) ===
    print("Expanding phrases...")
    expanded = []
    for row in tweets:
        tweet_id, text = row["id"], row["text"]
        doc = row["nlp_doc"]
        for sent in doc:
            words = sent["words"]
            for idx, word in enumerate(words):
                w = word["text"].lower()
                if any(w == t for t, _ in found_pairs[tweet_id]):
                    phrase = []
                    if idx - CONFIG["K_ADJ"] >= 0 and words[idx - CONFIG["K_ADJ"]]["pos"] == "ADJ":
                        phrase.append(words[idx - CONFIG["K_ADJ"]]["text"].lower())
                    for i in range(CONFIG["Q_NOUNS"]):
                        if idx - (i + 1) >= 0 and words[idx - (i + 1)]["pos"] == "NOUN":
                            phrase.insert(0, words[idx - (i + 1)]["text"].lower())
                    phrase.append(w)
                    for i in range(CONFIG["Q_NOUNS"]):
                        if idx + (i + 1) < len(words) and words[idx + (i + 1)]["pos"] == "NOUN":
                            phrase.append(words[idx + (i + 1)]["text"].lower())
                    expanded.append(" ".join(phrase))

    if CONFIG["USE_GLOBAL_PRUNING"]:
        print("Applying global pruning...")
        target_counter = Counter([t for v in found_pairs.values() for t, _ in v])
        target_counter.update(expanded)
        expanded = [p for p in expanded if target_counter[p.split()[-1]] >= CONFIG["MIN_FREQ_GLOBAL"]]

    # === Final scoring (unchanged) ===
    print("Calculating final scores...")
    results = []
    for row in tweets:
        tweet_id, text = row["id"], row["text"]
        doc = row["nlp_doc"]
        pairs = found_pairs[tweet_id]
        unique_targets = {t for t, _ in pairs}
        
        for target in unique_targets:
            matched = {o for t, o in pairs if t == target}
            
            weight, opinion_count = calculate_contextual_sentiment(
                target, matched, doc, seed_opinions, CONFIG["NEGATION_WINDOW"]
            )
            
            pos, neu, neg = calculate_sentiment_probabilities(weight, CONFIG["CONFIDENCE_FACTOR"])

            context_phrase = target
            for sent in doc:
                sent_tokens = [w["text"].lower() for w in sent["words"]]
                if target in sent_tokens:
                    idx = sent_tokens.index(target)
                    left = sent_tokens[idx - 1] if idx - 1 >= 0 else ""
                    right = sent_tokens[idx + 1] if idx + 1 < len(sent_tokens) else ""
                    context_phrase = f"{left} {target} {right}".strip()
                    break

            if CONFIG["VECTOR_MODE"] == "tfidf":
                vec = vectorizer.transform([context_phrase])
                sims = cosine_similarity(vec, aspect_vecs).flatten()
            elif CONFIG["VECTOR_MODE"] == "fasttext":
                vecs = [ft_id[t] for t in context_phrase.split() if t in ft_id]
                vec = np.mean(vecs, axis=0).reshape(1, -1) if vecs else np.zeros((1, 300))
                sims = cosine_similarity(vec, aspect_vecs).flatten()
            elif CONFIG["VECTOR_MODE"] == "sentence":
                vec = model.encode([context_phrase])
                sims = cosine_similarity(vec, aspect_vecs).flatten()

            best_idx = sims.argmax()
            best_sim = np.clip(sims[best_idx], 0, 1)

            results.append({
                "id": tweet_id,
                "original_text": text,
                "aspect_term": target,
                "aspect_category": aspect_categories[best_idx],
                "similarity": round(best_sim, 4),
                "matched_opinions": ",".join(matched),
                "total_weight": weight,
                "sentiment_prob": [round(pos, 4), round(neu, 4), round(neg, 4)],
                "context_phrase": context_phrase,
                "opinion_count": opinion_count
            })

    # Save results
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ Done! Results saved to {output_path}")
    elapsed_time = time.time() - start_time
    print(f"⏱️ Total time: {elapsed_time:.2f} seconds")
    print(f"📊 Processed {len(tweets)} tweets, found {len(results)} aspect-sentiment pairs")

if __name__ == "__main__":
    main()