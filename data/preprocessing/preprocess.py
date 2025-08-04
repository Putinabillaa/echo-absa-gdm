# preprocess.py

import argparse
from helper import (
    add_id, clean_csv_column, csv_to_txt,
    extract_unique_aspects, extract_corpus,
    filter_stopwords, merge_aspects_text,
    replace_special_phrase
)

KEEP_NEGATORS = {"tidak", "tak", "bukan", "jangan", "belum", "enggak", "gak"}
KEEP_CONJUNCTIONS = {"dan", "atau", "tetapi", "namun", "tapi", "serta", "sementara", "sedangkan", '&'}
KEEP_CONTRAST = {"walaupun", "meskipun", "padahal", "biarpun", "sekalipun"}
KEEP_INTENSIFIERS = {"sangat", "banget", "amat"}
KEEP_MODAL = {"harus", "bisa", "mesti", "boleh", "dapat", "masih", "sudah"}
SAFE_KEEP = KEEP_NEGATORS | KEEP_CONJUNCTIONS | KEEP_CONTRAST | KEEP_INTENSIFIERS | KEEP_MODAL

def main():
    parser = argparse.ArgumentParser(description="Preprocessing CLI")
    subparsers = parser.add_subparsers(dest='command')

    p1 = subparsers.add_parser("add_id")
    p1.add_argument("input_csv")
    p1.add_argument("output_csv")

    p2 = subparsers.add_parser("clean")
    p2.add_argument("input_csv")
    p2.add_argument("output_csv")
    p2.add_argument("--text_column", default="text")
    p2.add_argument("--id_column", default="id")

    p3 = subparsers.add_parser("csvtotxt")
    p3.add_argument("input_csv")
    p3.add_argument("output_txt")
    p3.add_argument("--text_column", default="normalized_text")

    p4 = subparsers.add_parser("extract_ac")
    p4.add_argument("input_csv")
    p4.add_argument("output_csv")

    p5 = subparsers.add_parser("extract_corpus")
    p5.add_argument("input_csv")
    p5.add_argument("output_csv")

    p6 = subparsers.add_parser("filter_stopwords")
    p6.add_argument("stopwords_file")

    p7 = subparsers.add_parser("merge")
    p7.add_argument("text_csv")
    p7.add_argument("aspects_csv")
    p7.add_argument("output_csv")

    p8 = subparsers.add_parser("replace_phrase")
    p8.add_argument("input_csv")
    p8.add_argument("output_csv")
    p8.add_argument("phrase")
    p8.add_argument("replacement")

    args = parser.parse_args()

    if args.command == "add_id":
        add_id(args.input_csv, args.output_csv)
    elif args.command == "clean":
        clean_csv_column(args.input_csv, args.output_csv, args.text_column, args.id_column)
    elif args.command == "csvtotxt":
        csv_to_txt(args.input_csv, args.output_txt, args.text_column)
    elif args.command == "extract_ac":
        extract_unique_aspects(args.input_csv, args.output_csv)
    elif args.command == "extract_corpus":
        extract_corpus(args.input_csv, args.output_csv)
    elif args.command == "filter_stopwords":
        filter_stopwords(args.stopwords_file, SAFE_KEEP)
    elif args.command == "merge":
        merge_aspects_text(args.text_csv, args.aspects_csv, args.output_csv)
    elif args.command == "replace_phrase":
        replace_special_phrase(args.input_csv, args.output_csv, args.phrase, args.replacement)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
