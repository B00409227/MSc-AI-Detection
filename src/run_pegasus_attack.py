"""
Pegasus Paraphrase Attack — 500 AI samples rewritten using tuner007/pegasus_paraphrase
Student: Abdul Hannaan Mohammed | B00409227 | UWS

Uses tuner007/pegasus_paraphrase as specified in the project specification.
The model works sentence-by-sentence (max 60 tokens per sentence), so each
text is split into sentences, paraphrased individually, then recombined.

Run:   python src/run_pegasus_attack.py
Time:  ~30-45 minutes on RTX 3060 (6GB VRAM)
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
import nltk

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# ── Settings ───────────────────────────────────────────────────────────────────
MODEL_NAME       = 'tuner007/pegasus_paraphrase'
MAX_SENTENCES    = 6       # paraphrase first 6 sentences per text
NUM_BEAMS        = 10      # standard beam search setting for this model
N_SAMPLES        = 500
CHECKPOINT_EVERY = 50

INPUT_FILE  = os.path.join('data', 'adversarial', 'ai_samples_500_for_colab.csv')
BACKUP_FILE = os.path.join('data', 'adversarial', 'pegasus_rewritten_500_t5_backup.csv')
OUTPUT_FILE = os.path.join('data', 'adversarial', 'pegasus_rewritten_500.csv')

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU   : {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM  : {vram:.1f} GB')

# ── Back up existing T5 output (safety) ────────────────────────────────────────
if os.path.exists(OUTPUT_FILE):
    existing = pd.read_csv(OUTPUT_FILE)
    existing.to_csv(BACKUP_FILE, index=False)
    print(f'\nBacked up existing file to: {BACKUP_FILE}')
    print('(This is the old T5 output — safe to delete after you verify Pegasus results)')

# ── Load model ─────────────────────────────────────────────────────────────────
print(f'\nLoading {MODEL_NAME}...')
print('(First run downloads ~570MB — may take a few minutes)')
tokeniser = PegasusTokenizer.from_pretrained(MODEL_NAME)
model     = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()
print('Model loaded.')

# ── Load input samples ─────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
df = df.head(N_SAMPLES).copy()
print(f'\nLoaded {len(df)} AI samples from {INPUT_FILE}')

# ── Paraphrase function ────────────────────────────────────────────────────────
def paraphrase_sentence(sentence):
    """Paraphrase one sentence using Pegasus. Returns paraphrased text."""
    sentence = sentence.strip()
    if not sentence:
        return sentence

    inputs = tokeniser(
        [sentence],
        truncation=True,
        padding='longest',
        max_length=60,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=60,
            num_beams=NUM_BEAMS,
            num_return_sequences=1,
            temperature=1.5,
            early_stopping=True
        )

    return tokeniser.decode(output[0], skip_special_tokens=True)


def paraphrase_text(text):
    """Split text into sentences, paraphrase each, recombine."""
    text = str(text).strip()
    sentences = sent_tokenize(text)

    # Use first MAX_SENTENCES sentences
    sentences = sentences[:MAX_SENTENCES]
    if not sentences:
        return text

    paraphrased = []
    for sent in sentences:
        try:
            para = paraphrase_sentence(sent)
            paraphrased.append(para)
        except Exception:
            paraphrased.append(sent)   # keep original if paraphrase fails

    return ' '.join(paraphrased)

# ── Test on one sample before full run ────────────────────────────────────────
print('\nTesting on sample 1...')
test_text = df['text'].iloc[0]
test_result = paraphrase_text(test_text)
print(f'Original  : {test_text[:150]}...')
print(f'Paraphrased: {test_result[:150]}...')
print('\nTest OK. Starting full run...')
print('-' * 55)

# ── Resume support ────────────────────────────────────────────────────────────
results   = []
start_idx = 0

if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 100:
    try:
        existing_out  = pd.read_csv(OUTPUT_FILE)
        # Only resume if this looks like a Pegasus run (not the T5 backup)
        if len(existing_out) > 0 and len(existing_out) < N_SAMPLES:
            results   = existing_out.to_dict('records')
            start_idx = len(results)
            print(f'Resuming from sample {start_idx + 1}')
    except Exception:
        pass

# ── Main loop ──────────────────────────────────────────────────────────────────
errors     = 0
start_time = time.time()

for i in range(start_idx, len(df)):
    row      = df.iloc[i]
    original = str(row['text']).strip()

    try:
        rewritten = paraphrase_text(original)
        success   = True
    except Exception as e:
        print(f'  ERROR sample {i+1}: {type(e).__name__}: {str(e)[:60]}')
        rewritten = original
        success   = False
        errors   += 1

    results.append({
        'original_text'  : original,
        'rewritten_text' : rewritten,
        'label'          : 1,
        'label_name'     : 'ai',
        'success'        : success
    })

    done      = i + 1
    elapsed   = time.time() - start_time
    per_item  = elapsed / (done - start_idx) if done > start_idx else 1
    remaining = per_item * (N_SAMPLES - done)

    if done % 10 == 0:
        print(f'  [{done:3d}/{N_SAMPLES}]  '
              f'Elapsed: {elapsed/60:.1f}m  '
              f'ETA: {remaining/60:.1f}m  '
              f'Errors: {errors}')

    if done % CHECKPOINT_EVERY == 0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        print(f'  >>> Checkpoint saved ({done} samples)')

# ── Final save ────────────────────────────────────────────────────────────────
final_df = pd.DataFrame(results)
final_df.to_csv(OUTPUT_FILE, index=False)

elapsed    = time.time() - start_time
successful = int(final_df['success'].sum())

print()
print('=' * 55)
print('PEGASUS ATTACK COMPLETE')
print(f'  Total      : {len(final_df)}')
print(f'  Successful : {successful}')
print(f'  Errors     : {errors}')
print(f'  Time       : {elapsed/60:.1f} minutes')
print(f'  Saved to   : {OUTPUT_FILE}')
print('=' * 55)
print()
print('NEXT STEPS:')
print('  1. Open notebooks/10_evaluate_all_models.ipynb')
print('  2. Run Cell 4 (main evaluation loop) — ~30 mins')
print('  3. Run Cell 5 (save results CSV)')
print('  4. Run Cells 6-9 (regenerate charts)')
print('  5. Take screenshot of the Pegasus results output')
print('  6. Run notebooks/12_consolidate_results.ipynb')
print('  7. Run notebooks/13_generate_dissertation_charts.ipynb')
