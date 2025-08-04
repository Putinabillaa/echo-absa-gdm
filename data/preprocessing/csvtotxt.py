#csvtotxt.py
import csv

input_file = 'corpus_final/kampus_merdeka_labeled_150t_slangid_txtdict_hf_stopwords.csv'
output_file = 'corpus_final/kampus_merdeka_labeled_150t_slangid_txtdict_hf_stopwords.txt'

with open(input_file, 'r', encoding='utf-8') as csvfile, open(output_file, 'w', encoding='utf-8') as txtfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['normalized_text'].strip()
        if text:
            txtfile.write(text + '\n')

print(f'Done! Saved to {output_file}')
