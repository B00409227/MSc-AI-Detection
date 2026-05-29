"""
Prepares 100 AI samples for ChatGPT manual rewriting.
Run this once — it creates a template CSV and a text file with all samples.
"""
import pandas as pd
import os

# Load the AI samples
df = pd.read_csv('data/adversarial/ai_samples_500_for_colab.csv')

# Take first 100 samples, truncate to 100 words each (easier to paste into ChatGPT)
samples = df.head(100).copy()
samples['text_short'] = samples['text'].apply(
    lambda t: ' '.join(str(t).split()[:100])
)

# Save a template CSV — user fills in the rewritten_text column
template = pd.DataFrame({
    'sample_id'     : range(1, 101),
    'original_text' : samples['text_short'].values,
    'rewritten_text': [''] * 100,   # user fills this in
    'label'         : 1,
    'label_name'    : 'ai'
})

os.makedirs('data/adversarial/chatgpt', exist_ok=True)
template_path = 'data/adversarial/chatgpt/chatgpt_template.csv'
template.to_csv(template_path, index=False)
print(f'Template saved: {template_path}')

# Also save a plain text file with all samples numbered
txt_path = 'data/adversarial/chatgpt/samples_to_rewrite.txt'
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write('CHATGPT REWRITING SAMPLES\n')
    f.write('=' * 60 + '\n')
    f.write('Prompt to use each time:\n')
    f.write('"Rewrite this text to sound more natural and human-written\n')
    f.write('while keeping the same meaning. Do not add new information."\n')
    f.write('=' * 60 + '\n\n')
    for i, row in template.iterrows():
        f.write(f'--- SAMPLE {row["sample_id"]} ---\n')
        f.write(row['original_text'] + '\n\n')

print(f'Text file saved: {txt_path}')
print(f'\nTotal samples: {len(template)}')
print('Open chatgpt_template.csv in Excel and fill in the rewritten_text column.')
print('Save as chatgpt_samples.csv when done.')
