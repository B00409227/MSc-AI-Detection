from datasets import load_dataset
import pandas as pd
import os

ds = load_dataset('artem9k/ai-text-detection-pile', split='train')
df = ds.to_pandas()

human_sample = df[df['source'] == 'human'].sample(1000, random_state=42)
ai_sample    = df[df['source'] == 'ai'].sample(1000, random_state=42)
sample       = pd.concat([human_sample, ai_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

sample['label']      = sample['source'].map({'human': 0, 'ai': 1})
sample['label_name'] = sample['source']
sample               = sample[['text', 'label', 'label_name']].copy()
sample['word_count'] = sample['text'].apply(lambda t: len(str(t).split()))

os.makedirs('data/raw/m4', exist_ok=True)
save_path = 'data/raw/m4/cross_dataset_test.csv'
sample.to_csv(save_path, index=False)

print('Saved:', save_path)
print('Total rows :', len(sample))
print('Human (0)  :', (sample['label'] == 0).sum())
print('AI (1)     :', (sample['label'] == 1).sum())
print('Avg words  :', round(sample['word_count'].mean()))
