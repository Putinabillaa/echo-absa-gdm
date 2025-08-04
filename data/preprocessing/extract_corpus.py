#extract_corpus.py
import pandas as pd

df = pd.read_csv("normalized/provider_labeled_135t_slangid_txtdict_hf_stopwords.csv")
df_selected = df[["id", "normalized_text"]].drop_duplicates()
df_selected.to_csv("corpus_final/provider_labeled_135t_slangid_txtdict_hf_stopwords.csv", index=False)

print(f"✅ Saved {len(df_selected)} unique row")
