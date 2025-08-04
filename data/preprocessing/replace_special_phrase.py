#replace_special_phrase.py

import csv

# Input and output file paths
input_file = 'corpus_final/kampus_merdeka_labeled_150t_slangid_txtdict_hf_stopwords.csv'
output_file = 'corpus_final/kampus_merdeka_labeled_150t_slangid_txtdict_hf_stopwords.csv'
special_phrase = 'kampus_merdeka'
replacement_phrase = 'nama_program'

# Open the input file and output file
with open(input_file, mode='r', encoding='utf-8') as infile, \
     open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        row['normalized_text'] = row['normalized_text'].replace(special_phrase, replacement_phrase)
        writer.writerow(row)

print(f'Done! Output saved to {output_file}')
