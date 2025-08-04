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

# --------------------------
# Build Prompt
# --------------------------

def build_prompt(tweets_batch, aspects, mode, parser):
    tweets_list = "\n".join([f'- "{t["cleaned_text"]}"' for t in tweets_batch])
    aspects_list = "\n".join([f'- "{a["category"]}": {a["description"]}' for a in aspects])

    if mode == "json":
        return f"""
Anda adalah model ABSA (Aspect-Based Sentiment Analysis).

Tweets:
{tweets_list}

Daftar aspek:
{aspects_list}

Tugas:
1. Untuk setiap tweet, evaluasi SEMUA aspek satu per satu.
2. Jika aspek relevan, tulis "present": true dan tentukan "sentiment": [pos, neu, neg] (jumlah 1.0).
3. Jika aspek TIDAK relevan, tulis "present": false.
4. Format HARUS SESUAI skema berikut:
{parser.get_format_instructions()}
"""
    elif mode == "block":
        return f"""
Anda adalah pengklasifikasi sentimen berbasis aspek (ABSA).

Tweets:
{tweets_list}

Aspects (with description):
{aspects_list}

Setiap tweet di bawah ini mungkin membahas satu atau lebih aspek.

Tugas Anda:
1. Untuk setiap tweet, baca teksnya baik-baik.
2. Untuk setiap aspek yang relevan/tersirat, keluarkan:
   Tweet: <tweet>
   Aspect: <aspect>
   Sentiment: [x.x, x.x, x.x] (jumlah 1.0)
3. Lewati aspek yang tidak relevan.
4. Jangan ubah teks tweet aslinya.

Contoh:
Tweet: Program Kampus Merdeka bermanfaat bagi mahasiswa namun anggarannya bengkak.
Aspect: Kebijakan dan Dampak
Sentiment: [0.8, 0.1, 0.1]

Tweet: Program Kampus Merdeka bermanfaat bagi mahasiswa namun anggarannya bengkak.
Aspect: Anggaran
Sentiment: [0.1, 0.3, 0.6]
"""

# --------------------------
# Main CLI
# --------------------------

def main():
    parser = argparse.ArgumentParser(description="ABSA Batch Pipeline with LangChain & Gemini")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--aspects", required=True, help="Aspects CSV file path")
    parser.add_argument("--output", required=True, help="Output file OR folder path")
    parser.add_argument("--mode", choices=["json", "block"], default="json", help="Prompt mode")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    # If output is a directory → build filename from input basename
    if output_path.is_dir() or args.output.endswith("/"):
        output_path = output_path / f"{input_path.stem}_{int(time.time())}.csv"

    # Check if output == input → prevent overwrite
    if output_path.resolve() == input_path:
        raise ValueError(f"❌ Output file must not overwrite input file!\nInput: {input_path}\nOutput: {output_path}")

    os.makedirs(output_path.parent, exist_ok=True)

    # Setup LLM
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
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
                "cleaned_text": row["normalized_text"]
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

    # Write CSV header
    fieldnames = ["id", "text", "aspect_category", "sentiment_prob"]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Process batches
    for i in range(0, len(tweets), args.batch_size):
        tweets_batch = tweets[i:i+args.batch_size]
        prompt_text = build_prompt(tweets_batch, aspect_categories, args.mode, output_parser)

        raw_result = llm.invoke(prompt_text)

        parsed = []
        if args.mode == "json":
            result: ABSAResponse = output_parser.invoke(raw_result)
            for tweet_data in result.results:
                parsed_text = normalize_text(tweet_data.tweet)
                tweet_id = next((t["id"] for t in tweets_batch if normalize_text(t["cleaned_text"]) == parsed_text), "NA")
                for asp in tweet_data.aspects:
                    if asp.present:
                        parsed.append({
                            "id": tweet_id,
                            "text": tweet_data.tweet,
                            "aspect_category": asp.aspect,
                            "sentiment_prob": asp.sentiment
                        })
        else:
            pattern = re.compile(r"Tweet: (.*?)\nAspect: (.*?)\nSentiment: \[(.*?)\]", re.DOTALL)
            matches = pattern.findall(raw_result.content if hasattr(raw_result, "content") else str(raw_result))
            for tweet_text, aspect, sent in matches:
                probs = [round(float(x.strip()), 4) for x in sent.split(",")]
                parsed_text = normalize_text(tweet_text)
                tweet_id = next((t["id"] for t in tweets_batch if normalize_text(t["cleaned_text"]) == parsed_text), "NA")
                parsed.append({
                    "id": tweet_id,
                    "text": tweet_text,
                    "aspect_category": aspect.strip(),
                    "sentiment_prob": probs
                })

        # Append to CSV
        with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for row in parsed:
                row["sentiment_prob"] = str(row["sentiment_prob"])
                writer.writerow(row)

        print(f"✅ Saved tweets {i+1}-{i+len(tweets_batch)} to {output_path}")
        time.sleep(5)

    print(f"✅ Done! Full results saved to {output_path}")

if __name__ == "__main__":
    main()
