import csv
import os
import time
import re
import argparse
from typing import List, Optional, Dict
from pathlib import Path
from difflib import SequenceMatcher
import logging
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------
# Google Generative AI Client
# --------------------------
class GoogleGenerativeAI:
    def __init__(self, model: str, api_key: str, temperature: float = 0.0):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    
    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """Call Google Generative AI API with system and user prompts"""
        url = f"{self.base_url}/{self.model}:generateContent"
        
        headers = {"Content-Type": "application/json"}
        
        # Structure the conversation with system instruction and user message
        data = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
            }
        }
        params = {"key": self.api_key}
        
        response = requests.post(url, headers=headers, json=data, params=params)
        response.raise_for_status()
        
        result = response.json()
        
        if "candidates" in result and result["candidates"]:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0]["text"]
        
        raise Exception(f"No valid response from API: {result}")

# --------------------------
# Unified Text Processing Utils
# --------------------------
class TextMatcher:
    """Unified text matching utility with fuzzy matching capabilities"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        text = str(text).strip().strip('"').strip("'")
        return re.sub(r'\s+', ' ', text)
    
    @staticmethod
    def fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> float:
        """Get fuzzy match ratio between two texts"""
        norm1 = TextMatcher.normalize_text(text1)
        norm2 = TextMatcher.normalize_text(text2)
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    @staticmethod
    def find_best_match(target: str, candidates: List[Dict], 
                       text_key: str, threshold: float = 0.8) -> Optional[Dict]:
        """Find best matching candidate from a list"""
        target_norm = TextMatcher.normalize_text(target)
        
        # Try exact match first
        for candidate in candidates:
            if TextMatcher.normalize_text(candidate[text_key]) == target_norm:
                return candidate
        
        # Try fuzzy match
        best_match = None
        best_ratio = 0
        
        for candidate in candidates:
            ratio = TextMatcher.fuzzy_match(target, candidate[text_key])
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = candidate
        
        if best_match:
            logger.info(f"Fuzzy matched '{target}' -> '{best_match[text_key]}' (ratio: {best_ratio:.3f})")
        
        return best_match

# --------------------------
# Data Processing Utils
# --------------------------
def find_tweet_id(parsed_text: str, tweets_batch: List[Dict], 
                 fallback_index: Optional[int] = None) -> str:
    """Find tweet ID with fallback strategies"""
    match = TextMatcher.find_best_match(parsed_text, tweets_batch, "text", threshold=0.8)
    
    if match:
        return match["id"]
    
    # Positional fallback
    if fallback_index is not None and 0 <= fallback_index < len(tweets_batch):
        logger.warning(f"Using positional fallback for tweet: {parsed_text[:50]}...")
        return tweets_batch[fallback_index]["id"]
    
    logger.error(f"Could not match tweet: {parsed_text[:50]}...")
    return "NA"

def normalize_aspect_name(pred_aspect: str, aspect_categories: List[Dict]) -> Optional[str]:
    """Normalize aspect name with fuzzy matching"""
    if not pred_aspect:
        logger.warning("Empty aspect name provided")
        return None
    
    match = TextMatcher.find_best_match(pred_aspect, aspect_categories, "category", threshold=0.7)
    
    if match:
        return match["category"]
    
    logger.warning(f"Could not match aspect '{pred_aspect}' to any category. Skipping.")
    return None

def parse_sentiment_probabilities(sent: str) -> List[float]:
    """Parse probability string with error handling"""
    if not sent:
        logger.warning("Empty sentiment string")
        return [0.0, 0.0, 0.0]
    
    # Clean the string: remove brackets and labels
    sent_clean = re.sub(r'[\[\]a-zA-Z=]', '', sent.strip())
    parts = [p.strip() for p in sent_clean.split(",") if p.strip()]
    
    try:
        probs = [round(float(x), 4) for x in parts]
        
        if len(probs) != 3:
            logger.warning(f"Expected 3 probabilities, got {len(probs)}: {sent}")
            return [0.0, 0.0, 0.0]
        
        # Validate and normalize probabilities
        prob_sum = sum(probs)
        if abs(prob_sum - 1.0) > 0.1:
            logger.warning(f"Probabilities don't sum to 1.0 (sum={prob_sum}): {sent}")
            if prob_sum > 0:
                probs = [p/prob_sum for p in probs]
            else:
                return [0.0, 0.0, 0.0]
        
        return probs
        
    except ValueError as e:
        logger.error(f"Could not parse probabilities: {sent!r}. Error: {e}")
        return [0.0, 0.0, 0.0]

# --------------------------
# Response Parser
# --------------------------
class ResponseParser:
    """Unified response parser with multiple parsing strategies"""
    
    def __init__(self, tweets_batch: List[Dict], aspect_categories: List[Dict]):
        self.tweets_batch = tweets_batch
        self.aspect_categories = aspect_categories
    
    def parse(self, content: str) -> List[Dict]:
        """Main parsing method with fallback strategies"""
        # Strategy 1: Structured regex parsing
        results = self._structured_parse(content)
        if results:
            logger.info(f"✅ Structured parsing successful: {len(results)} results")
            return results
        
        # Strategy 2: Flexible block parsing
        logger.warning("Structured parsing failed, trying flexible parsing...")
        results = self._flexible_parse(content)
        if results:
            logger.info(f"✅ Flexible parsing successful: {len(results)} results")
            return results
        
        logger.error("All parsing methods failed for this batch")
        return []
    
    def _structured_parse(self, content: str) -> List[Dict]:
        """Parse using structured regex pattern"""
        pattern = r"(?i)(?:tweet|text):\s*[\"']?(.*?)[\"']?\s*[\n\r]+\s*(?:aspect|category):\s*[\"']?(.*?)[\"']?\s*[\n\r]+\s*(?:sentiment|prob):\s*\[([^\]]+)\]"
        
        try:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            return self._process_matches(matches)
        except Exception as e:
            logger.error(f"Error in structured parsing: {e}")
            return []
    
    def _flexible_parse(self, content: str) -> List[Dict]:
        """Parse using flexible block-based approach"""
        results = []
        blocks = re.split(r'\n\s*\n', content)
        
        for block in blocks:
            if len(block.strip()) < 20:
                continue
            
            # Extract components using separate patterns
            tweet_match = re.search(r'(?i)(?:tweet|text)[:\s]+(.*?)(?=\n|aspect|category|sentiment|$)', block, re.DOTALL)
            aspect_match = re.search(r'(?i)(?:aspect|category)[:\s]+(.*?)(?=\n|sentiment|$)', block, re.DOTALL)
            sentiment_match = re.search(r'(?i)(?:sentiment|prob)[:\s]*\[([^\]]+)\]', block)
            
            if all([tweet_match, aspect_match, sentiment_match]):
                matches = [(
                    tweet_match.group(1).strip().strip('"').strip("'"),
                    aspect_match.group(1).strip().strip('"').strip("'"),
                    sentiment_match.group(1).strip()
                )]
                results.extend(self._process_matches(matches))
        
        return results
    
    def _process_matches(self, matches: List[tuple]) -> List[Dict]:
        """Process regex matches into result dictionaries"""
        results = []
        
        for tweet_text, aspect, sentiment in matches:
            # Skip empty matches
            if not all([tweet_text.strip(), aspect.strip(), sentiment.strip()]):
                continue
            
            # Parse sentiment probabilities
            probs = parse_sentiment_probabilities(sentiment)
            if sum(probs) == 0:
                continue
            
            # Find tweet ID and normalize aspect
            tweet_id = find_tweet_id(tweet_text, self.tweets_batch)
            normalized_aspect = normalize_aspect_name(aspect, self.aspect_categories)
            
            if normalized_aspect is None:
                continue
            
            results.append({
                "id": tweet_id,
                "text": tweet_text,
                "aspect_category": normalized_aspect,
                "sentiment_prob": probs
            })
        
        return results

# --------------------------
# File Utilities
# --------------------------
def load_csv_data(file_path: str, required_columns: List[str]) -> List[Dict]:
    """Load CSV data with validation"""
    data = []
    try:
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check if required columns exist
            missing_columns = set(required_columns) - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            for row in reader:
                # Clean and validate row data
                cleaned_row = {col: row[col].strip() if row[col] else "" for col in required_columns}
                if any(cleaned_row.values()):  # Skip completely empty rows
                    data.append(cleaned_row)
        
        logger.info(f"Loaded {len(data)} records from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise

# --------------------------
# Prompt Building
# --------------------------
def build_system_prompt(aspects: List[Dict], few_shots: Optional[List[Dict]] = None) -> str:
    """Build system prompt with task definition and examples"""
    
    # Few-shot examples
    few_shot_text = ""
    if few_shots:
        few_shot_text = "\n\nContoh format yang benar:\n"
        for ex in few_shots:
            few_shot_text += (
                f'Tweet: {ex["text"]}\n'
                f'Aspect: {ex["aspect_category"]}\n'
                f'Sentiment: {ex["sentiment_prob"]}\n\n'
            )

    # Aspects list
    aspects_list = "\n".join([f'- "{a["category"]}": {a["description"]}' for a in aspects])

    return f"""Anda adalah model ABSA (Aspect-Based Sentiment Analysis) yang ahli dalam menganalisis sentimen berbasis aspek pada teks berbahasa Indonesia.

ASPEK YANG TERSEDIA:
{aspects_list}

TUGAS ANDA:
1. Untuk setiap tweet yang diberikan, identifikasi aspek mana yang disebutkan atau tersirat
2. Hanya sertakan aspek yang benar-benar relevan dengan tweet
3. Untuk setiap aspek yang relevan, analisis sentimennya (positif, netral, negatif)

FORMAT OUTPUT YANG WAJIB:
Tweet: <teks_tweet_persis_sama>
Aspect: <nama_aspek_dari_daftar>
Sentiment: [p_pos, p_neu, p_neg]

ATURAN PENTING:
- Probabilitas sentimen harus berjumlah 1.0 (contoh: [0.70, 0.20, 0.10])
- Gunakan nama aspek yang PERSIS SAMA dengan daftar yang diberikan
- Setiap tweet memiliki setidaknya satu aspek
- PASTIKAN format output sama persis seperti yang ditentukan

{few_shot_text}"""

def build_user_prompt(tweets_batch: List[Dict]) -> str:
    """Build user prompt with tweets to analyze"""
    tweets_list = "\n".join([f'"{t["text"]}"' for t in tweets_batch])
    
    return f"""Analisis tweet-tweet berikut untuk aspek dan sentimen yang relevan:

{tweets_list}

Berikan analisis ABSA untuk setiap tweet yang relevan dengan format yang telah ditentukan."""

# --------------------------
# Main CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="ABSA Batch Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--aspects", required=True, help="Aspects CSV file path")
    parser.add_argument("--output", required=True, help="Output folder path")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--few_shot", help="Optional few-shot examples CSV")
    parser.add_argument(
        "--model", 
        choices=["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"], 
        required=True, 
        help="Model to use"
    )
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries for API calls")
    parser.add_argument("--retry_delay", type=int, default=10, help="Delay between retries (seconds)")

    args = parser.parse_args()

    start_time = time.time()
    input_path = Path(args.input).resolve()
    output_folder = Path(args.output).resolve()
    os.makedirs(output_folder, exist_ok=True)

    # Validate API key
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    # Initialize Google AI client
    llm = GoogleGenerativeAI(model=args.model, api_key=GOOGLE_API_KEY, temperature=0.0)

    # Load data
    tweets = load_csv_data(args.input, ["id", "text"])
    aspect_categories = load_csv_data(args.aspects, ["aspect_category", "desc"])
    
    # Rename desc to description for consistency
    for aspect in aspect_categories:
        aspect["description"] = aspect.pop("desc")
        aspect["category"] = aspect.pop("aspect_category")

    few_shots = None
    if args.few_shot:
        few_shots = load_csv_data(args.few_shot, ["text", "aspect_category", "sentiment_prob"])

    # Initialize output
    timestamp = int(time.time())
    output_path = output_folder / f"{input_path.stem}_results_{timestamp}.csv"
    
    all_results = []
    failed_batches = []
    
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, 
            fieldnames=["id", "text", "aspect_category", "sentiment_prob"]
        )
        writer.writeheader()

        # Process batches
        for i in range(0, len(tweets), args.batch_size):
            tweets_batch = tweets[i:i + args.batch_size]
            batch_num = i // args.batch_size + 1
            
            logger.info(f"Processing batch {batch_num}: tweets {i + 1}-{i + len(tweets_batch)}")
            
            # Build prompts
            system_prompt = build_system_prompt(aspect_categories, few_shots)
            user_prompt = build_user_prompt(tweets_batch)
            
            # Retry logic for API calls
            raw_content = None
            for attempt in range(args.max_retries):
                try:
                    raw_content = llm.invoke(system_prompt, user_prompt)
                    break
                except Exception as e:
                    logger.error(f"API call failed (attempt {attempt + 1}/{args.max_retries}): {e}")
                    if attempt < args.max_retries - 1:
                        time.sleep(args.retry_delay)

            if not raw_content:
                logger.error(f"Failed to process batch {batch_num} after {args.max_retries} attempts")
                failed_batches.append(batch_num)
                continue

            # Parse results
            try:
                parser_instance = ResponseParser(tweets_batch, aspect_categories)
                batch_results = parser_instance.parse(raw_content)

                # Save results
                for result in batch_results:
                    all_results.append(result)
                    row_copy = result.copy()
                    row_copy["sentiment_prob"] = str(result["sentiment_prob"])
                    writer.writerow(row_copy)

                output_file.flush()
                logger.info(f"✅ Successfully processed batch {batch_num} ({len(batch_results)} results)")

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                failed_batches.append(batch_num)

            # Rate limiting
            time.sleep(5)

    # Report results
    elapsed_time = time.time() - start_time
    
    logger.info(f"📝 Results saved to {output_path}")
    logger.info(f"📊 Processing Summary:")
    logger.info(f"  - Total tweets processed: {len(tweets)}")
    logger.info(f"  - Total results collected: {len(all_results)}")
    logger.info(f"  - Failed batches: {len(failed_batches)}")
    logger.info(f"  - Success rate: {((len(tweets) - len(failed_batches) * args.batch_size) / len(tweets) * 100):.1f}%")
    logger.info(f"⏱️ Total time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()