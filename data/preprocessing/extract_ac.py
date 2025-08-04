#extract_ac.py
import csv

# Input CSV file
input_file = "labeled/provider_labeled_135t.csv"

# Output CSV file
output_file = "labeled/aspect_list/provider_labeled_135t.csv"

# Store unique aspects
unique_aspects = set()

# Read input CSV
with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        unique_aspects.add(row["aspect_category_final"].strip())

# Write unique aspects to output CSV
with open(output_file, "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["aspect_category"])  # header
    for aspect in sorted(unique_aspects):
        writer.writerow([aspect])

print(f"✅ Saved {len(unique_aspects)} unique aspect categories to {output_file}")
