#add_id.py
import pandas as pd

df = pd.read_csv('labeled/provider_labeled_135t.csv')
df['id'] = df.groupby('cleaned_text').ngroup() + 1
df.to_csv('labeled/provider_labeled_135t.csv', index=False)

print("✅ Done! Saved")
