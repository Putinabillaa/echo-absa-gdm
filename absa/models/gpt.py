import csv
import os
import time
import re
import argparse
from typing import List, Optional
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from difflib import SequenceMatcher
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------
# Pydantic Schema for JSON mode
# --------------------------
class AspectSentiment(BaseModel):
    aspect: str
    present: bool
    confidence: float
    sentiment: Optional[List[float]] = Field(default=None)

class TweetABSA(BaseModel):
    tweet: str
    aspects: List[AspectSentiment]

class ABSAResponse(BaseModel):
    results: List[TweetABSA]

# --------------------------
# Helpers
# --------------------------
def normalize_text(text):
    """Normalize text for comparison"""
    if not text:
        return ""
    text = str(text).strip().strip('"').strip("'")
    text = re.sub(r'\s+', ' ', text)
    return text

def fuzzy_match_text(text1, text2, threshold=0.8):
    """Check if two texts match with fuzzy matching"""
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio() >= threshold

def safe_find_tweet_id(parsed_text, tweets_batch, fallback_index=None):
    """Safely find tweet ID with multiple fallback strategies"""
    parsed_norm = normalize_text(parsed_text)
    
    # Strategy 1: Exact match
    for tweet in tweets_batch:
        if normalize_text(tweet["text"]) == parsed_norm:
            return tweet["id"]
    
    # Strategy 2: Fuzzy match
    best_match_id = None
    best_ratio = 0
    for tweet in tweets_batch:
        ratio = SequenceMatcher(None, parsed_norm, normalize_text(tweet["text"])).ratio()
        if ratio > best_ratio and ratio >= 0.8:
            best_ratio = ratio
            best_match_id = tweet["id"]
    
    if best_match_id:
        logger.warning(f"Using fuzzy match for tweet (ratio: {best_ratio:.3f}): {parsed_text[:50]}...")
        return best_match_id
    
    # Strategy 3: Positional fallback
    if fallback_index is not None and 0 <= fallback_index < len(tweets_batch):
        logger.warning(f"Using positional fallback for tweet: {parsed_text[:50]}...")
        return tweets_batch[fallback_index]["id"]
    
    logger.error(f"Could not match tweet: {parsed_text[:50]}...")
    return "NA"

def load_few_shots(few_shot_path):
    """Load few-shot examples with error handling"""
    examples = []
    try:
        with open(few_shot_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                examples.append({
                    "text": row["text"].strip(),
                    "aspect_category": row["aspect_category"].strip(),
                    "sentiment_prob": row["sentiment_prob"].strip()
                })
        logger.info(f"Loaded {len(examples)} few-shot examples")
    except Exception as e:
        logger.error(f"Error loading few-shot examples: {e}")
        raise
    return examples

def parse_probs(sent: str):
    """Parse probability string with better error handling"""
    if not sent:
        logger.warning("Empty sentiment string")
        return [0.0, 0.0, 0.0]
    
    # remove brackets and labels if present
    sent_clean = sent.strip().replace("[", "").replace("]", "")
    # remove sentiment labels like "positive=", "neg=", etc.
    sent_clean = re.sub(r"[a-zA-Z=]", "", sent_clean)
    # split and filter out empty strings
    parts = [p for p in sent_clean.split(",") if p.strip() != ""]
    
    try:
        probs = [round(float(x.strip()), 4) for x in parts]
        
        # Validate probabilities
        if len(probs) != 3:
            logger.warning(f"Expected 3 probabilities, got {len(probs)}: {sent}")
            return [0.0, 0.0, 0.0]
        
        # Check if probabilities sum to approximately 1.0
        prob_sum = sum(probs)
        if abs(prob_sum - 1.0) > 0.1:
            logger.warning(f"Probabilities don't sum to 1.0 (sum={prob_sum}): {sent}")
            # Normalize if close to 1.0
            if prob_sum > 0:
                probs = [p/prob_sum for p in probs]
            else:
                return [0.0, 0.0, 0.0]
        
        return probs
        
    except ValueError as e:
        logger.error(f"Could not parse probabilities: {sent!r} -> {parts!r}. Error: {e}")
        return [0.0, 0.0, 0.0]

def normalize_aspect_name(pred_aspect: str, aspect_categories: list) -> str:
    """Normalize aspect name with fuzzy matching"""
    if not pred_aspect:
        return ""
    
    pred_norm = pred_aspect.strip().lower()
    
    # Exact match first
    for a in aspect_categories:
        if pred_norm == a["category"].strip().lower():
            return a["category"]
    
    # Fuzzy match as fallback
    best_match = None
    best_ratio = 0
    for a in aspect_categories:
        ratio = SequenceMatcher(None, pred_norm, a["category"].strip().lower()).ratio()
        if ratio > best_ratio and ratio >= 0.8:
            best_ratio = ratio
            best_match = a["category"]
    
    if best_match:
        logger.info(f"Fuzzy matched aspect '{pred_aspect}' -> '{best_match}' (ratio: {best_ratio:.3f})")
        return best_match
    
    logger.warning(f"Could not normalize aspect: '{pred_aspect}'")
    return pred_aspect.strip()  # fallback: return as-is

# --------------------------
# Build Prompt
# --------------------------
def build_prompt(tweets_batch, aspects, mode, parser, few_shots=None):
    """Build consistent prompts for both modes"""
    few_shot_text = ""
    if few_shots:
        few_shot_text = "\n\nContoh:\n"
        for ex in few_shots:
            few_shot_text += (
                f'Tweet: {ex["text"]}\n'
                f'Aspect: {ex["aspect_category"]}\n'
                f'Sentiment: {ex["sentiment_prob"]}\n'
                f'Confidence: 0.9\n\n'
            )

    tweets_list = "\n".join([f'- "{t["text"]}"' for t in tweets_batch])
    aspects_list = "\n".join([f'- "{a["category"]}": {a["description"]}' for a in aspects])

    if mode == "json":
        return f"""
Anda adalah model ABSA (Aspect-Based Sentiment Analysis).

Tweets:
{tweets_list}

Daftar aspek:
{aspects_list}

Tugas:
1. Untuk setiap tweet, evaluasi SEMUA aspek satu per satu
2. Jika aspek relevan, tulis "present": true, "confidence": 0.x, dan tentukan "sentiment": [pos, neu, neg] (jumlah 1.0)
3. Jika aspek TIDAK relevan, tulis "present": false dan "confidence": 0.x
4. Format HARUS SESUAI skema berikut:

{parser.get_format_instructions()}

{few_shot_text}
"""
    elif mode == "block":
        return f"""
    Anda adalah model ABSA (Aspect-Based Sentiment Analysis).
    
    Input:
    Tweets berikut:
    {tweets_list}
    
    Daftar aspek (dengan deskripsi singkat):
    {aspects_list}
    
    Instruksi:
    1. Baca setiap tweet dengan teliti.
    2. Tentukan aspek-aspek yang benar-benar disebutkan atau tersirat dalam tweet. 
    - Jika aspek tidak relevan dengan tweet, jangan keluarkan.
    - Jika ada beberapa aspek relevan, keluarkan semuanya.
    3. Untuk setiap aspek relevan, keluarkan hasil dalam format berikut:
    
    Tweet: <tweet_text_exact>
    Aspect: <aspect_name>
    Sentiment: [p_pos, p_neu, p_neg]  # probabilitas untuk positif, netral, negatif (jumlah = 1.0)
    Confidence: <angka antara 0 dan 1>  # seberapa yakin model terhadap klasifikasi aspek
    
    4. Probabilitas harus dalam format desimal 2 angka, misalnya [0.70, 0.20, 0.10].
    5. Jangan keluarkan aspek yang tidak relevan.
    
    Contoh output:
    Tweet: Mobil ini sangat irit bensin tetapi performanya kurang
    Aspect: bensin
    Sentiment: [0.85, 0.10, 0.05]
    Confidence: 0.90
    
    Tweet: Mobil ini sangat irit bensin tetapi performanya kurang
    Aspect: performa
    Sentiment: [0.10, 0.15, 0.75]
    Confidence: 0.90
    
    {few_shot_text}
    """

# --------------------------
# Main CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="ABSA Batch Pipeline with LangChain & OpenAI GPT")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--aspects", required=True, help="Aspects CSV file path")
    parser.add_argument("--output", required=True, help="Output folder path")
    parser.add_argument("--mode", choices=["json", "block"], default="json", help="Prompt mode")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--few_shot", help="Optional few-shot examples CSV")
    parser.add_argument(
        "--conf_thresholds", 
        type=str, 
        default="0.6", 
        help="Comma separated list of confidence thresholds, e.g. '0.5,0.6,0.7'"
    )
    parser.add_argument(
        "--model", 
        choices=["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"], 
        default="gpt-4o-mini",
        help="OpenAI model to use"
    )
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries for API calls")
    parser.add_argument("--retry_delay", type=int, default=10, help="Delay between retries (seconds)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    input_path = Path(args.input).resolve()
    output_folder = Path(args.output).resolve()
    os.makedirs(output_folder, exist_ok=True)
    
    # Validate inputs
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not Path(args.aspects).exists():
        raise FileNotFoundError(f"Aspects file not found: {args.aspects}")

    conf_thresholds = [float(t.strip()) for t in args.conf_thresholds.split(",")]

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    llm = ChatOpenAI(
        model=args.model,
        api_key=OPENAI_API_KEY,
        temperature=0.1  # Lower temperature for more consistent results
    )

    output_parser = PydanticOutputParser(pydantic_object=ABSAResponse)

    # Load tweets
    tweets = []
    try:
        with open(input_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                tweets.append({
                    "id": row["id"],
                    "text": row["text"]
                })
        logger.info(f"Loaded {len(tweets)} tweets")
    except Exception as e:
        logger.error(f"Error loading tweets: {e}")
        raise

    # Load aspects
    aspect_categories = []
    try:
        with open(args.aspects, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                aspect_categories.append({
                    "category": row["aspect_category"].strip(),
                    "description": row["desc"].strip()
                })
        logger.info(f"Loaded {len(aspect_categories)} aspect categories")
    except Exception as e:
        logger.error(f"Error loading aspects: {e}")
        raise

    few_shots = load_few_shots(args.few_shot) if args.few_shot else None

    # Initialize output files and tracking
    timestamp = int(time.time())
    all_parsed_results = []
    failed_batches = []
    
    # Create output files for each confidence threshold
    output_files = {}
    writers = {}
    for threshold in conf_thresholds:
        threshold_str = str(threshold).replace('.', '')
        output_path = output_folder / f"{input_path.stem}_{threshold_str}_{timestamp}.csv"
        output_files[threshold] = open(output_path, "w", newline="", encoding="utf-8")
        writers[threshold] = csv.DictWriter(
            output_files[threshold], 
            fieldnames=["id", "text", "aspect_category", "sentiment_prob", "confidence"]
        )
        writers[threshold].writeheader()
    
    # Create full output file
    full_output_path = output_folder / f"{input_path.stem}_all_full_{timestamp}.csv"
    full_output_file = open(full_output_path, "w", newline="", encoding="utf-8")
    full_writer = csv.DictWriter(
        full_output_file, 
        fieldnames=["id", "text", "aspect_category", "sentiment_prob", "confidence"]
    )
    full_writer.writeheader()

    # Process batches with retry logic
    for i in range(0, len(tweets), args.batch_size):
        tweets_batch = tweets[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        
        logger.info(f"Processing batch {batch_num}: tweets {i + 1}-{i + len(tweets_batch)}")
        
        prompt_text = build_prompt(tweets_batch, aspect_categories, args.mode, output_parser, few_shots)
        
        # Retry logic for API calls
        success = False
        for attempt in range(args.max_retries):
            try:
                raw_result = llm.invoke(prompt_text)
                success = True
                break
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}/{args.max_retries}): {e}")
                if attempt < args.max_retries - 1:
                    time.sleep(args.retry_delay)
                else:
                    failed_batches.append(batch_num)
                    continue

        if not success:
            logger.error(f"Failed to process batch {batch_num} after {args.max_retries} attempts")
            continue

        # Parse results based on mode and save immediately
        batch_results = []
        try:
            if args.mode == "json":
                result: ABSAResponse = output_parser.invoke(raw_result)
                
                # Process each tweet result with better ID matching
                for tweet_idx, tweet_data in enumerate(result.results):
                    tweet_id = safe_find_tweet_id(
                        tweet_data.tweet, 
                        tweets_batch, 
                        fallback_index=tweet_idx
                    )
                    
                    for asp in tweet_data.aspects:
                        if asp.present and asp.sentiment:
                            # Validate and normalize aspect
                            normalized_aspect = normalize_aspect_name(asp.aspect, aspect_categories)
                            
                            # Validate sentiment probabilities
                            if len(asp.sentiment) == 3 and abs(sum(asp.sentiment) - 1.0) <= 0.1:
                                batch_results.append({
                                    "id": tweet_id,
                                    "text": tweet_data.tweet,
                                    "aspect_category": normalized_aspect,
                                    "sentiment_prob": asp.sentiment,
                                    "confidence": asp.confidence
                                })
                            else:
                                logger.warning(f"Invalid sentiment probabilities for tweet {tweet_id}: {asp.sentiment}")

            else:  # block mode
                pattern = re.compile(
                    r"Tweet:\s*(.*?)\nAspect:\s*(.*?)\nSentiment:\s*\[([^\]]+)\]\s*\nConfidence:\s*([0-9]*\.?[0-9]+)",
                    re.DOTALL
                )
                
                content = raw_result.content if hasattr(raw_result, "content") else str(raw_result)
                matches = pattern.findall(content)
                
                for tweet_text, aspect, sent, conf in matches:
                    probs = parse_probs(sent)
                    
                    try:
                        confidence = float(conf)
                    except ValueError:
                        logger.warning(f"Invalid confidence value: {conf}")
                        confidence = 0.0
                    
                    tweet_id = safe_find_tweet_id(tweet_text.strip(), tweets_batch)
                    normalized_aspect = normalize_aspect_name(aspect.strip(), aspect_categories)
                    
                    # Only add if probabilities are valid
                    if sum(probs) > 0:
                        batch_results.append({
                            "id": tweet_id,
                            "text": tweet_text.strip(),
                            "aspect_category": normalized_aspect,
                            "sentiment_prob": probs,
                            "confidence": confidence
                        })

            # Save batch results immediately to all output files
            for row in batch_results:
                # Add to memory collection for summary stats
                all_parsed_results.append(row)
                
                # Save to full output file
                row_copy = row.copy()
                row_copy["sentiment_prob"] = str(row["sentiment_prob"])
                full_writer.writerow(row_copy)
                
                # Save to threshold-filtered files
                for threshold in conf_thresholds:
                    if row["confidence"] >= threshold:
                        writers[threshold].writerow(row_copy)
            
            # Flush all files to ensure data is written
            full_output_file.flush()
            for f in output_files.values():
                f.flush()
            
            logger.info(f"✅ Successfully processed and saved batch {batch_num} ({len(batch_results)} results)")
            
        except Exception as e:
            logger.error(f"Error parsing results for batch {batch_num}: {e}")
            failed_batches.append(batch_num)
            continue

        # Rate limiting
        time.sleep(5)

    # Close all output files
    full_output_file.close()
    for f in output_files.values():
        f.close()

    # Report results
    if failed_batches:
        logger.warning(f"Failed to process {len(failed_batches)} batches: {failed_batches}")

    logger.info(f"Collected {len(all_parsed_results)} total results")
    logger.info(f"📝 Full unfiltered results saved to {full_output_path}")
    
    # Report threshold-filtered file locations and counts
    for threshold in conf_thresholds:
        threshold_str = str(threshold).replace('.', '')
        output_path = output_folder / f"{input_path.stem}_{threshold_str}_{timestamp}.csv"
        filtered_count = len([row for row in all_parsed_results if row["confidence"] >= threshold])
        logger.info(f"▶️ Filtered results (confidence >= {threshold}) saved to {output_path} ({filtered_count} rows)")

    elapsed_time = time.time() - start_time
    logger.info(f"⏱️ Total time: {elapsed_time:.2f} seconds")
    
    # Summary statistics
    logger.info(f"📊 Processing Summary:")
    logger.info(f"  - Total tweets processed: {len(tweets)}")
    logger.info(f"  - Total results collected: {len(all_parsed_results)}")
    logger.info(f"  - Failed batches: {len(failed_batches)}")
    logger.info(f"  - Success rate: {((len(tweets) - len(failed_batches) * args.batch_size) / len(tweets) * 100):.1f}%")

if __name__ == "__main__":
    main()