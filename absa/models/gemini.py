import csv
import os
import time
import re
import argparse
from typing import List, Optional
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


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
    text = text.strip().strip('"').strip("'")
    text = re.sub(r'\s+', ' ', text)
    return text

def load_few_shots(few_shot_path):
    examples = []
    with open(few_shot_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            examples.append({
                "text": row["text"].strip(),
                "aspect_category": row["aspect_category"].strip(),
                "sentiment_prob": row["sentiment_prob"].strip()
            })
    return examples


# --------------------------
# Build Prompt
# --------------------------

def build_prompt(tweets_batch, aspects, mode, parser, few_shots=None):
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

    common_instruction = """
Hanya keluarkan aspek yang secara eksplisit atau sangat jelas tersirat dari teks.
Jika ragu, jangan keluarkan aspek tersebut.
Maksimal 2 aspek per tweet kecuali ada bukti sangat kuat untuk lebih.
"""

    if mode == "json":
        return f"""
Anda adalah model ABSA (Aspect-Based Sentiment Analysis).

{common_instruction}

Tweets:
{tweets_list}

Daftar aspek:
{aspects_list}

Tugas:
1. Untuk setiap tweet, evaluasi SEMUA aspek satu per satu.
2. Jika aspek relevan, tulis "present": true, "confidence": 0.x, dan tentukan "sentiment": [pos, neu, neg] (jumlah 1.0).
3. Jika aspek TIDAK relevan, tulis "present": false dan "confidence": 0.x.
4. Format HARUS SESUAI skema berikut:
{parser.get_format_instructions()}

{few_shot_text}
"""
    elif mode == "block":
        return f"""
Anda adalah pengklasifikasi sentimen berbasis aspek (ABSA).

{common_instruction}

Tweets:
{tweets_list}

Aspects (with description):
{aspects_list}

Setiap tweet di bawah ini mungkin membahas satu atau lebih aspek.

Tugas Anda:
1. Untuk setiap tweet, baca teksnya baik-baik.
2. Untuk setiap aspek yang relevan, keluarkan:
   Tweet: <tweet>
   Aspect: <aspect>
   Sentiment: [x.x, x.x, x.x] (jumlah 1.0)
   Confidence: 0.x
3. Lewati aspek yang tidak relevan.
4. Jangan ubah teks tweet aslinya.

{few_shot_text}
"""


# --------------------------
# Main CLI
# --------------------------

def main():
    parser = argparse.ArgumentParser(description="ABSA Batch Pipeline with LangChain & Gemini")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--aspects", required=True, help="Aspects CSV file path")
    parser.add_argument("--output", required=True, help="Output folder path")  # expect folder now
    parser.add_argument("--mode", choices=["json", "block"], default="json", help="Prompt mode")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--few_shot", help="Optional few-shot examples CSV")
    parser.add_argument(
        "--conf_thresholds",
        type=str,
        default="0.6",
        help="Comma separated list of confidence thresholds, e.g. '0.5,0.6,0.7'"
    )
    parser.add_argument("--model", choices=["gemini-2.5-pro", "gemini-2.0-pro", "gemini-2.5-flash", "gemini-2.0-flash"], required=True, help="Model to use")
    args = parser.parse_args()

    start_time = time.time()

    input_path = Path(args.input).resolve()
    output_folder = Path(args.output).resolve()
    os.makedirs(output_folder, exist_ok=True)

    conf_thresholds = [float(t.strip()) for t in args.conf_thresholds.split(",")]

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model=args.model,
        google_api_key=GOOGLE_API_KEY
    )
    output_parser = PydanticOutputParser(pydantic_object=ABSAResponse)

    # Load tweets
    tweets = []
    with open(input_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweets.append({
                "id": row["id"],
                "text": row["text"]
            })

    # Load aspects
    aspect_categories = []
    with open(args.aspects, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            aspect_categories.append({
                "category": row["aspect_category"].strip(),
                "description": row["desc"].strip()
            })

    few_shots = load_few_shots(args.few_shot) if args.few_shot else None

    # 1) Collect ALL parsed results from all batches here:
    all_parsed_results = []

    # Process batches: CALL API once per batch, parse ALL results, collect in all_parsed_results
    for i in range(0, len(tweets), args.batch_size):
        tweets_batch = tweets[i:i + args.batch_size]
        prompt_text = build_prompt(tweets_batch, aspect_categories, args.mode, output_parser, few_shots)

        raw_result = llm.invoke(prompt_text)

        if args.mode == "json":
            result: ABSAResponse = output_parser.invoke(raw_result)
            for tweet_data in result.results:
                parsed_text = normalize_text(tweet_data.tweet)
                tweet_id = next((t["id"] for t in tweets_batch if normalize_text(t["text"]) == parsed_text), "NA")
                for asp in tweet_data.aspects:
                    if asp.present:
                        all_parsed_results.append({
                            "id": tweet_id,
                            "text": tweet_data.tweet,
                            "aspect_category": asp.aspect,
                            "sentiment_prob": asp.sentiment,
                            "confidence": asp.confidence
                        })
        else:
            pattern = re.compile(
                r"Tweet: (.*?)\nAspect: (.*?)\nSentiment: \[(.*?)\]\nConfidence: ([0-9]*\.?[0-9]+)",
                re.DOTALL
            )
            matches = pattern.findall(raw_result.content if hasattr(raw_result, "content") else str(raw_result))
            for tweet_text, aspect, sent, conf in matches:
                probs = [round(float(x.strip()), 4) for x in sent.split(",")]
                confidence = float(conf)
                parsed_text = normalize_text(tweet_text)
                tweet_id = next((t["id"] for t in tweets_batch if normalize_text(t["text"]) == parsed_text), "NA")
                all_parsed_results.append({
                    "id": tweet_id,
                    "text": tweet_text,
                    "aspect_category": aspect.strip(),
                    "sentiment_prob": probs,
                    "confidence": confidence
                })

        print(f"✅ Processed tweets {i + 1}-{i + len(tweets_batch)}")
        time.sleep(5)

    # 2) Save full unfiltered results once (optional)
    full_output_path = output_folder / f"{input_path.stem}_all_full_{int(time.time())}.csv"
    with open(full_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "text", "aspect_category", "sentiment_prob", "confidence"])
        writer.writeheader()
        for row in all_parsed_results:
            row["sentiment_prob"] = str(row["sentiment_prob"])
            writer.writerow(row)

    print(f"📝 Full unfiltered results saved to {full_output_path}")

    # 3) Now for each threshold, filter from all_parsed_results and save filtered CSV
    for threshold in conf_thresholds:
        threshold_str = str(threshold).replace('.', '')
        timestamp = int(time.time())
        output_path = output_folder / f"{input_path.stem}_{threshold_str}_{timestamp}.csv"

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["id", "text", "aspect_category", "sentiment_prob", "confidence"])
            writer.writeheader()

            filtered_rows = [row for row in all_parsed_results if row["confidence"] >= threshold]

            for row in filtered_rows:
                row["sentiment_prob"] = str(row["sentiment_prob"])
                writer.writerow(row)

        print(f"▶️ Filtered results with confidence >= {threshold} saved to {output_path}")

    elapsed_time = time.time() - start_time
    print(f"⏱️ Total time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
