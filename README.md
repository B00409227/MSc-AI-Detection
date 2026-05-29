# MSc AI Dissertation — AI-Generated Text Detection Robustness

**Title:** Experimental Evaluation of AI-Generated Text Detection Robustness Against Rewriting-Based Attacks  
**Student:** Abdul Hannaan Mohammed | Banner ID: B00409227  
**Programme:** MSc Artificial Intelligence | University of the West of Scotland (UWS)  
**Supervisor:** Dr Tahir Mehmood  
**Academic Year:** 2025/26  
**Status:** Experiments complete — dissertation writing in progress

---

## Research Question

> To what extent do rewriting and paraphrasing attacks reduce the detection accuracy of transformer-based AI-generated text classifiers, and what does any observed performance degradation reveal about the robustness limitations of current supervised detection approaches under realistic adversarial conditions?

---

## Project Overview

This project compares five AI-text detection methods against three paraphrase-based adversarial attacks, using two datasets. Three transformer classifiers (RoBERTa, BERT, DistilBERT) were fine-tuned on the HC3 dataset. A pre-trained detector (Hello-SimpleAI) and a traditional Logistic Regression baseline were also evaluated. All five models were tested on clean samples, three adversarial attack conditions, and a cross-dataset generalisation test (M4). A human evaluation study with 15 participants was also conducted.

---

## Pipeline

```
HC3 Dataset (Guo et al., 2023)
    └── Preprocess & Split (70/15/15)
            ├── Fine-tune RoBERTa-base          (nb03 — local)
            ├── Fine-tune BERT-base             (nb08 — local)
            ├── Fine-tune DistilBERT            (nb11 — Colab)
            ├── Load Hello-SimpleAI detector    (nb10 — pre-trained, no training)
            └── Train Logistic Regression       (nb10 — TF-IDF baseline)
                    │
                    ├── Evaluate on clean HC3 test set (nb04, nb10)
                    │
                    ├── Generate adversarial samples:
                    │       ├── T5-Paraphrase Attack  (nb05 — Colab, T5_Paraphrase_Paws)
                    │       ├── QuillBot-Style Attack  (src/run_pegasus_local_attack.py — local)
                    │       └── ChatGPT Rewrite        (src/run_chatgpt_attack.py — local)
                    │
                    ├── Evaluate all 5 models × 3 attacks (nb10)
                    ├── Cross-dataset generalisation test on M4 (nb10)
                    └── Human evaluation study — 15 participants (nb07)
```

---

## Results Summary

| Model | Clean F1 | Pegasus ASR | QuillBot ASR | ChatGPT ASR | M4 F1 |
|-------|----------|-------------|--------------|-------------|-------|
| RoBERTa-base | 0.9913 | **93.4%** | 13.4% | 1.6% | 0.7389 |
| BERT-base | 0.9845 | 18.4% | 12.2% | 3.2% | 0.5999 |
| DistilBERT | 0.9922 | 27.2% | 19.0% | 6.8% | 0.4316 |
| Hello-SimpleAI | 0.9929 | 92.8% | 13.8% | 2.6% | 0.5442 |
| Logistic Regression | 0.9524 | 29.4% | 20.4% | **39.4%** | 0.3356 |

**ASR = Attack Success Rate** (percentage of AI samples evading detection after rewriting)

**Human evaluation:** 15 participants, 41.8% overall accuracy. Against rewritten AI text: 30.7% (below chance).

---

## Repository Structure

```
MSc-AI-Detection/
├── data/
│   ├── raw/                        # Raw HC3 dataset files (gitignored)
│   ├── processed/                  # Train/val/test CSVs (gitignored)
│   ├── supplementary/              # Extra ChatGPT-generated samples
│   └── adversarial/                # Attack output CSVs
│       ├── ai_samples_500_for_colab.csv
│       ├── t5_paraphrase/          # T5-Paraphrase attack samples
│       ├── quillbot/               # QuillBot-style attack samples
│       └── chatgpt/                # ChatGPT rewrite samples
├── notebooks/
│   ├── 01_explore_hc3.ipynb                # Dataset exploration
│   ├── 02_preprocess.ipynb                 # Cleaning and 70/15/15 split
│   ├── 03_train_roberta.ipynb              # RoBERTa fine-tuning (local)
│   ├── 04_evaluate_original.ipynb          # RoBERTa baseline evaluation
│   ├── 05_paraphrase_attack_colab.ipynb    # T5 paraphrase attack (Colab)
│   ├── 06_evaluate_adversarial.ipynb       # RoBERTa adversarial evaluation
│   ├── 07_human_eval_analysis.ipynb        # Human study analysis
│   ├── 08_train_bert.ipynb                 # BERT fine-tuning (local)
│   ├── 09_train_bert_colab.ipynb           # BERT fine-tuning (Colab backup)
│   ├── 10_evaluate_all_models.ipynb        # All 5 models × all conditions
│   ├── 11_train_distilbert_colab.ipynb     # DistilBERT fine-tuning (Colab)
│   ├── 12_consolidate_results.ipynb        # Master results CSV
│   └── 13_generate_dissertation_charts.ipynb  # All dissertation figures
├── src/
│   ├── prepare_chatgpt_samples.py          # Prepare 500 AI samples
│   ├── run_pegasus_local_attack.py         # QuillBot-style T5 attack (local)
│   ├── run_chatgpt_attack.py               # ChatGPT rewrite via OpenAI API
│   └── save_cross_dataset.py               # M4 cross-dataset evaluation
├── models/
│   └── checkpoints/                        # Saved model weights (gitignored)
├── results/
│   ├── metrics/                            # JSON/CSV evaluation results
│   ├── figures/                            # Dissertation figures (DPI=300)
│   └── tables/                             # Results tables (CSV)
├── human_evaluation/                       # Google Form, responses, analysis
├── dissertation/
│   ├── chapters/dissertation.docx          # Dissertation draft
│   └── figures/                            # Comparison tables
├── app/
│   └── streamlit_app.py                    # Interactive demo platform
├── screenshots/                            # Evidence screenshots (nb##_name.png)
├── logs/                                   # Training logs
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Models Used

| Model | Role | Training | Hardware |
|-------|------|----------|----------|
| `roberta-base` | Fine-tuned classifier | HC3, 3 epochs | Local 6GB VRAM |
| `bert-base-uncased` | Fine-tuned classifier | HC3, 3 epochs | Local 6GB VRAM |
| `distilbert-base-uncased` | Fine-tuned classifier | HC3, 3 epochs | Google Colab T4 |
| `Hello-SimpleAI/chatgpt-detector-roberta` | Pre-trained detector | No training needed | Local inference |
| Logistic Regression + TF-IDF | Traditional baseline | HC3 | Local CPU |

---

## Attack Methods

| Attack | Model Used | Where Run | Notes |
|--------|-----------|-----------|-------|
| T5-Paraphrase (Pegasus-style) | `Vamsi/T5_Paraphrase_Paws` | Google Colab | Beam search decoding |
| QuillBot-style | `Vamsi/T5_Paraphrase_Paws` | Local GPU | Sampling (top-k=200, top-p=0.95) |
| ChatGPT Rewrite | OpenAI `gpt-3.5-turbo` | Local (API) | Prompt: rewrite to sound human |

Note: DIPPER (Krishna et al., 2023) was originally planned but excluded due to its 11B parameter size exceeding available GPU memory. The T5_Paraphrase_Paws model was used as a practical alternative.

---

## Datasets

| Dataset | Source | Size | Use |
|---------|--------|------|-----|
| HC3 | `Hello-SimpleAI/HC3` (HuggingFace) | 11,820 test samples | Training + primary evaluation |
| M4 | `artem9k/ai-text-detection-pile` | 2,000 balanced samples | Cross-dataset generalisation test |

---

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## Running the Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

Requires model checkpoints in `models/checkpoints/`. Download from Google Drive or retrain using the training notebooks.

---

## Key References

- Devlin et al. (2019) BERT. doi: 10.18653/v1/N19-1423
- Liu et al. (2019) RoBERTa. arXiv: 1907.11692
- Guo et al. (2023) HC3. arXiv: 2301.07597
- Krishna et al. (2023) DIPPER. arXiv: 2303.13408
- Mitchell et al. (2023) DetectGPT. arXiv: 2301.11305
- Sadasivan et al. (2023) Detection reliability. arXiv: 2303.11156
- Wang et al. (2023) M4 dataset. arXiv: 2305.14902
