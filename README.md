# MSc AI Dissertation — AI-Generated Text Detection Robustness

**Title:** Experimental Evaluation of AI-Generated Text Detection Robustness Against Rewriting-Based Attacks  
**Student:** Abdul Hannaan Mohammed | Banner ID: B00409227  
**Programme:** MSc Artificial Intelligence | University of the West of Scotland (UWS)  
**Supervisor:** Dr Tahir Mehmood  
**Academic Year:** 2025/26

---

## Research Question

> To what extent do rewriting and paraphrasing attacks reduce the detection accuracy of a transformer-based AI-generated text classifier (RoBERTa), and what does any observed performance degradation reveal about the robustness limitations of current supervised detection approaches when applied under realistic adversarial conditions?

---

## Project Overview

This project investigates the robustness of AI-generated text detectors against adversarial rewriting attacks. A RoBERTa-base classifier is fine-tuned on the HC3 dataset to distinguish human-written from AI-generated text. The DIPPER paraphrasing model is then used to rewrite AI-generated samples, and classifier performance is re-evaluated to measure degradation.

---

## Pipeline

```
HC3 Dataset
    └── Preprocess & Split (70/15/15)
            └── Fine-tune RoBERTa-base
                    └── Evaluate on original test set
                            └── Rewrite AI samples with DIPPER
                                    └── Evaluate on rewritten test set
                                            └── Compare metrics + Human Evaluation
```

---

## Repository Structure

```
MSc-AI-Detection/
├── data/
│   ├── raw/                  # Raw HC3 dataset files
│   ├── processed/            # Preprocessed splits (train/val/test CSVs)
│   ├── supplementary/        # Extra ChatGPT-generated samples
│   └── adversarial/          # DIPPER-rewritten samples
├── notebooks/
│   ├── 01_explore_hc3.ipynb          # Dataset exploration
│   ├── 02_preprocess.ipynb           # Data cleaning and splitting
│   ├── 03_train_roberta.ipynb        # RoBERTa fine-tuning
│   ├── 04_evaluate_original.ipynb    # Evaluation on original samples
│   ├── 05_dipper_colab.ipynb         # DIPPER rewriting (run on Colab)
│   ├── 06_evaluate_adversarial.ipynb # Evaluation on rewritten samples
│   └── 07_human_eval_analysis.ipynb  # Human evaluation analysis
├── src/
│   ├── preprocess.py         # Data preprocessing utilities
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation utilities
│   └── visualise.py          # Plotting and graph utilities
├── models/
│   └── checkpoints/          # Saved model weights (gitignored)
├── results/
│   ├── metrics/              # JSON/CSV metric outputs
│   ├── figures/              # Saved plots and graphs
│   ├── tables/               # LaTeX/CSV result tables
│   └── confusion_matrices/   # Confusion matrix data
├── human_evaluation/
│   ├── form_screenshots/     # Google Form screenshots
│   ├── responses/            # Raw Google Sheets CSV export
│   └── analysis/             # Analysis outputs
├── dissertation/
│   ├── chapters/             # Draft chapter documents
│   └── figures/              # Figures for dissertation
├── logs/                     # Training logs
├── screenshots/              # Evidence screenshots for appendices
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Models Used

| Model | Purpose | Hardware |
|-------|---------|----------|
| `roberta-base` | AI-text classifier | Local (6GB VRAM) |
| `kalpeshk2011/dipper-paraphraser-xxl` | Adversarial rewriting | Google Colab T4 |
| `tuner007/pegasus_paraphrase` | Secondary paraphrasing | Local / Colab |

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve, Precision-Recall Curve
- Performance comparison: original vs rewritten samples
- Human evaluation scores (13–14 participants)

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

## Week-by-Week Timeline

| Week | Tasks |
|------|-------|
| 1–2 | Setup, HC3 exploration, literature review writing |
| 3–4 | Preprocessing, RoBERTa fine-tuning, wandb tracking, methodology chapter |
| 5–6 | DIPPER adversarial generation (Colab), implementation chapter |
| 7 | Full evaluation, results tables and graphs, interim report |
| 8–9 | Human evaluation (Google Forms), collection and analysis |
| 10–11 | Results and discussion chapters |
| 12 | Full dissertation draft, appendices, references |
| 13 | Proofread, polish, presentation slides, submit |

---

## Key References

- Devlin et al. (2019) BERT. doi: 10.18653/v1/N19-1423
- Liu et al. (2019) RoBERTa. arXiv: 1907.11692
- Guo et al. (2023) HC3. arXiv: 2301.07597
- Krishna et al. (2023) DIPPER. arXiv: 2303.13408
- Mitchell et al. (2023) DetectGPT. arXiv: 2301.11305
- Sadasivan et al. (2023) Detection reliability. arXiv: 2303.11156
