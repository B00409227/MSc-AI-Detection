"""
Automated ChatGPT Attack — 500 AI samples rewritten via OpenAI API
Student: Abdul Hannaan Mohammed | B00409227 | UWS

Setup:
1. pip install openai
2. Create a file called .env in the project root with:
   OPENAI_API_KEY=sk-your-key-here

Run: python run_chatgpt_attack.py
Cost estimate: ~$0.50 for 500 samples using gpt-3.5-turbo
"""

import os
import time
import json
import pandas as pd
from openai import OpenAI

# ── Load API key ───────────────────────────────────────────────────────────────
# Set via terminal before running:
# PowerShell: $env:OPENAI_API_KEY = "sk-your-key-here"
API_KEY = os.environ.get('OPENAI_API_KEY', '')
if not API_KEY:
    print('ERROR: No API key found.')
    print('Run this in your terminal first:')
    print('  $env:OPENAI_API_KEY = "sk-your-key-here"')
    exit(1)

client = OpenAI(api_key=API_KEY)

# ── Settings ───────────────────────────────────────────────────────────────────
MODEL           = 'gpt-3.5-turbo'
MAX_WORDS       = 150        # truncate input to this many words
N_SAMPLES       = 500
CHECKPOINT_EVERY = 50        # save progress every N samples
SLEEP_BETWEEN    = 0.5       # seconds between API calls (avoid rate limits)

OUTPUT_DIR  = os.path.join('data', 'adversarial', 'chatgpt')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'chatgpt_samples.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT_TEMPLATE = (
    "Rewrite this text to sound more natural and human-written while keeping "
    "the same meaning. Do not add new information. Only give me the rewritten "
    "text, nothing else.\n\n{text}"
)

# ── Load samples ───────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join('data', 'adversarial', 'ai_samples_500_for_colab.csv'))
df = df.head(N_SAMPLES).copy()
df['text_short'] = df['text'].apply(
    lambda t: ' '.join(str(t).split()[:MAX_WORDS])
)

print(f'Loaded {len(df)} AI samples.')
print(f'Model: {MODEL}')
print(f'Output: {OUTPUT_FILE}')
print(f'Estimated cost: ~${len(df) * 0.001:.2f}')
print('-' * 50)

# ── Load existing progress if resuming ────────────────────────────────────────
results = []
start_idx = 0

if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
    try:
        existing  = pd.read_csv(OUTPUT_FILE)
        results   = existing.to_dict('records')
        start_idx = len(results)
        print(f'Resuming from sample {start_idx + 1} ({start_idx} already done)')
    except Exception:
        print('Existing file unreadable — starting fresh.')
        results   = []
        start_idx = 0

# ── Main rewriting loop ────────────────────────────────────────────────────────
errors = 0
start_time = time.time()

for i in range(start_idx, len(df)):
    row = df.iloc[i]
    original = row['text_short']

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'user', 'content': PROMPT_TEMPLATE.format(text=original)}
            ],
            max_tokens=300,
            temperature=0.7
        )
        rewritten = response.choices[0].message.content.strip()
        success   = True

    except Exception as e:
        print(f'  ERROR on sample {i+1}: {e}')
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

    # Progress update
    done    = i + 1
    elapsed = time.time() - start_time
    per_item = elapsed / (done - start_idx) if done > start_idx else 1
    remaining = per_item * (len(df) - done)

    if done % 10 == 0:
        print(f'  [{done:3d}/{len(df)}]  '
              f'Elapsed: {elapsed/60:.1f}m  '
              f'ETA: {remaining/60:.1f}m  '
              f'Errors: {errors}')

    # Checkpoint save
    if done % CHECKPOINT_EVERY == 0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        print(f'  >>> Checkpoint saved: {done} samples')

    time.sleep(SLEEP_BETWEEN)

# Final save
final_df = pd.DataFrame(results)
final_df.to_csv(OUTPUT_FILE, index=False)

elapsed    = time.time() - start_time
successful = final_df['success'].sum()

print('\n' + '=' * 50)
print('CHATGPT ATTACK COMPLETE')
print(f'  Total      : {len(final_df)}')
print(f'  Successful : {successful}')
print(f'  Errors     : {errors}')
print(f'  Time       : {elapsed/60:.1f} minutes')
print(f'  Saved to   : {OUTPUT_FILE}')
print('=' * 50)
