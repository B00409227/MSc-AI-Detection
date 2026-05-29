"""
QuillBot-Style Attack — 500 AI samples rewritten using T5 paraphraser (GPU)
Student: Abdul Hannaan Mohammed | B00409227 | UWS

Uses Vamsi/T5_Paraphrase_Paws with higher diversity settings than the
first attack to produce distinctly different paraphrases.

Run: python run_pegasus_local_attack.py
Expected time: ~30-45 minutes on RTX 3060
"""

import os
import time
import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── Settings ───────────────────────────────────────────────────────────────────
MODEL_NAME       = 'Vamsi/T5_Paraphrase_Paws'
MAX_WORDS        = 80
N_SAMPLES        = 500
CHECKPOINT_EVERY = 50

OUTPUT_DIR  = os.path.join('data', 'adversarial', 'quillbot')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'quillbot_samples.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
print(f'Loading {MODEL_NAME}...')
tokeniser = T5Tokenizer.from_pretrained(MODEL_NAME)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()
print('Model loaded.')

# ── Load samples ───────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join('data', 'adversarial', 'ai_samples_500_for_colab.csv'))
df = df.head(N_SAMPLES).copy()
df['text_short'] = df['text'].apply(
    lambda t: ' '.join(str(t).split()[:MAX_WORDS])
)
print(f'Loaded {len(df)} samples.')

# ── Load existing progress if resuming ────────────────────────────────────────
results   = []
start_idx = 0

if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
    try:
        existing  = pd.read_csv(OUTPUT_FILE)
        if len(existing) > 0 and existing['success'].sum() > 0:
            results   = existing.to_dict('records')
            start_idx = len(results)
            print(f'Resuming from sample {start_idx + 1}')
    except Exception:
        print('Starting fresh.')

# ── Paraphrase function ────────────────────────────────────────────────────────
def paraphrase(text):
    """Paraphrase using T5 with sampling for more varied output."""
    input_text = f'paraphrase: {text} </s>'
    inputs = tokeniser(
        input_text,
        return_tensors='pt',
        max_length=256,
        truncation=True
    )
    input_ids      = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            do_sample=True,
            top_k=200,
            top_p=0.95,
            num_return_sequences=1
        )
    return tokeniser.decode(output[0], skip_special_tokens=True)

# ── Test one sample first ──────────────────────────────────────────────────────
print('\nTesting on one sample...')
test_result = paraphrase(df['text_short'].iloc[0])
print(f'Test OK: {test_result[:100]}')
print('\nStarting main loop...')
print('-' * 50)

# ── Main loop ──────────────────────────────────────────────────────────────────
errors     = 0
start_time = time.time()

for i in range(start_idx, len(df)):
    row      = df.iloc[i]
    original = row['text_short']

    try:
        rewritten = paraphrase(original)
        success   = True
    except Exception as e:
        print(f'  ERROR on sample {i+1}: {type(e).__name__}: {str(e)[:80]}')
        rewritten = original
        success   = False
        errors   += 1

    results.append({
        'original_text'  : row['text'],
        'rewritten_text' : rewritten,
        'label'          : 1,
        'label_name'     : 'ai',
        'success'        : success
    })

    done      = i + 1
    elapsed   = time.time() - start_time
    per_item  = elapsed / (done - start_idx) if done > start_idx else 1
    remaining = per_item * (len(df) - done)

    if done % 10 == 0:
        print(f'  [{done:3d}/{len(df)}]  '
              f'Elapsed: {elapsed/60:.1f}m  '
              f'ETA: {remaining/60:.1f}m  '
              f'Errors: {errors}')

    if done % CHECKPOINT_EVERY == 0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        print(f'  >>> Checkpoint saved: {done} samples')

final_df = pd.DataFrame(results)
final_df.to_csv(OUTPUT_FILE, index=False)

elapsed    = time.time() - start_time
successful = int(final_df['success'].sum())

print('\n' + '=' * 50)
print('QUILLBOT-STYLE ATTACK COMPLETE')
print(f'  Total      : {len(final_df)}')
print(f'  Successful : {successful}')
print(f'  Errors     : {errors}')
print(f'  Time       : {elapsed/60:.1f} minutes')
print(f'  Saved to   : {OUTPUT_FILE}')
print('=' * 50)
