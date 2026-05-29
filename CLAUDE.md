# CLAUDE.md — Project Context for AI Coding Assistant

## Project Identity
- **Title:** Experimental Evaluation of AI-Generated Text Detection Robustness Against Rewriting-Based Attacks
- **Student:** Abdul Hannaan Mohammed | Banner ID: B00409227
- **Programme:** MSc Artificial Intelligence | University of the West of Scotland (UWS)
- **Supervisor:** Dr Tahir Mehmood
- **Academic Year:** 2025/26
- **Current Week:** Week 7 of 13
- **Status:** All experiments complete. Dissertation writing in progress.

## Research Question
To what extent do rewriting and paraphrasing attacks reduce the detection accuracy of transformer-based AI-generated text classifiers, and what does any observed performance degradation reveal about the robustness limitations of current supervised detection approaches when applied under realistic adversarial conditions?

## What Has Been Completed
1. HC3 dataset explored, preprocessed, split 70/15/15 (11,820 test samples)
2. Three transformer classifiers fine-tuned on HC3: RoBERTa-base, BERT-base, DistilBERT
3. Two additional models evaluated without training: Hello-SimpleAI pre-trained detector, Logistic Regression + TF-IDF baseline
4. Three adversarial attack sets generated (500 samples each):
   - T5-Paraphrase attack using `Vamsi/T5_Paraphrase_Paws` (Colab, beam search)
   - QuillBot-style attack using same T5 model with sampling (top-k=200, top-p=0.95) (local)
   - ChatGPT rewrite using OpenAI gpt-3.5-turbo API (local)
5. All 5 models evaluated against all 3 attacks = 15 adversarial result sets
6. Cross-dataset generalisation test on M4 dataset (2,000 samples)
7. Human evaluation study: 15 participants, 15 questions, Google Forms
8. Master results table (25 rows), all dissertation figures generated
9. Streamlit demo platform built (app/streamlit_app.py)
10. Chapter 2 (Literature Review) written in dissertation.docx

## What Remains
- Dissertation chapters: 1 (Introduction), 3 (Methodology), 4 (Implementation), 5 (Results), 6 (Discussion), 7 (Conclusion)
- Abstract (300 words)
- Appendices A–F
- Interim report due Week 7

## Hardware
- Local: Windows 11, 6GB VRAM GPU
- Google Colab free T4 (15GB VRAM) — used for DistilBERT training and T5 paraphrase attack

## Models (Actual — as used)
- `roberta-base` — fine-tuned classifier (HuggingFace), trained locally
- `bert-base-uncased` — fine-tuned classifier, trained locally
- `distilbert-base-uncased` — fine-tuned classifier, trained on Colab
- `Hello-SimpleAI/chatgpt-detector-roberta` — pre-trained detector, used as-is
- Logistic Regression + TF-IDF — traditional ML baseline, trained locally

## Attack Models (Actual — as used)
- `Vamsi/T5_Paraphrase_Paws` — used for both T5-Paraphrase and QuillBot-style attacks
- OpenAI `gpt-3.5-turbo` API — used for ChatGPT rewrite attack
- NOTE: DIPPER (`kalpeshk2011/dipper-paraphraser-xxl`) was originally planned but excluded due to 11B parameter size exceeding GPU memory. Still cited in literature review as what Krishna et al. (2023) used.

## Datasets
- HC3 (Hello-SimpleAI/HC3) from HuggingFace Hub — primary dataset
- M4 (artem9k/ai-text-detection-pile) — cross-dataset generalisation test
- Split: 70% train / 15% validation / 15% test

## Key Results
| Model | Clean F1 | Pegasus ASR | QuillBot ASR | ChatGPT ASR | M4 F1 |
|-------|----------|-------------|--------------|-------------|-------|
| RoBERTa-base | 0.9913 | 1.2% | 13.4% | 1.6% | 0.7389 |
| BERT-base | 0.9845 | 1.8% | 12.2% | 3.2% | 0.5999 |
| DistilBERT | 0.9922 | 5.2% | 19.0% | 6.8% | 0.4316 |
| Hello-SimpleAI | 0.9929 | 0.4% | 14.0% | 2.8% | 0.5442 |
| Logistic Regression | 0.9524 | 29.0% | 26.8% | 39.4% | 0.3356 |
Human accuracy: 41.8% overall; 30.7% against rewritten AI text (below chance)
Pegasus ASR = tuner007/pegasus_paraphrase (actual model). Human study used T5_Paraphrase_Paws samples.

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Attack Success Rate (ASR) = 1 - Recall on AI-only samples
- Confusion Matrix, ROC Curve, Precision-Recall Curve
- Before vs after rewriting comparison
- Human evaluation accuracy by condition

## Tech Stack
Python 3.10+, PyTorch, HuggingFace Transformers/Datasets/Accelerate/Evaluate,
Pandas, NumPy, NLTK, scikit-learn, Matplotlib, Seaborn, wandb,
Google Colab, Git/GitHub, Google Forms/Sheets, OpenAI API, Streamlit

## Notebook Guide (01–13)
| Notebook | Purpose | Where Run |
|----------|---------|-----------|
| 01_explore_hc3 | Dataset exploration | Local |
| 02_preprocess | Cleaning and 70/15/15 split | Local |
| 03_train_roberta | RoBERTa fine-tuning | Local |
| 04_evaluate_original | RoBERTa baseline eval | Local |
| 05_paraphrase_attack_colab | T5 paraphrase attack (500 samples) | Colab |
| 06_evaluate_adversarial | RoBERTa adversarial eval | Local |
| 07_human_eval_analysis | Human study: sample prep + analysis | Local |
| 08_train_bert | BERT fine-tuning | Local |
| 09_train_bert_colab | BERT fine-tuning (Colab backup) | Colab |
| 10_evaluate_all_models | All 5 models × all conditions | Local |
| 11_train_distilbert_colab | DistilBERT fine-tuning | Colab |
| 12_consolidate_results | Master results CSV (25 rows) | Local |
| 13_generate_dissertation_charts | All 7 dissertation figures | Local |

## Folder Structure
- `data/raw/` — raw HC3 files (gitignored)
- `data/processed/` — train/val/test CSVs (gitignored)
- `data/supplementary/` — ChatGPT-generated samples
- `data/adversarial/` — all 3 attack output CSVs
- `notebooks/` — all Jupyter notebooks (numbered 01–13)
- `src/` — attack scripts and utilities
- `app/streamlit_app.py` — interactive demo platform
- `models/checkpoints/` — saved model weights (gitignored — too large for git)
- `results/metrics/` — JSON/CSV results (25 result sets)
- `results/figures/` — dissertation figures at DPI=300
- `results/tables/` — master results CSV
- `human_evaluation/` — Google Form, responses, analysis
- `dissertation/` — chapter drafts and comparison figures
- `screenshots/` — evidence screenshots (nb##_description.png convention)
- `logs/` — training logs

## Rules for Claude
- Always tell the user exactly what to do next — never assume prior knowledge
- State which week of 13 we are in whenever relevant
- Every code block must be fully commented
- Always specify when to take a screenshot and what to name it
- Always specify when to commit and push to GitHub
- Always specify when content should go into the dissertation
- Generate all graphs with proper titles, axis labels, and legends
- Save all results automatically to the correct results/ subfolder
- Save all data automatically to the correct data/ subfolder
- Keep dissertation word counts in mind
- User is a full-time worker — be efficient and move fast without sacrificing quality
- User is on Windows with 6GB VRAM
- Attack model is T5_Paraphrase_Paws NOT DIPPER — be accurate in all dissertation writing

## Dissertation Structure
1. Title Page
2. Abstract (300 words)
3. Acknowledgements
4. Table of Contents
5. List of Figures / List of Tables
6. Chapter 1 — Introduction (1,500 words)
7. Chapter 2 — Literature Review (4,000 words) — COMPLETE
8. Chapter 3 — Research Design and Methodology (3,500 words)
9. Chapter 4 — Implementation (2,500 words)
10. Chapter 5 — Results and Evaluation (2,500 words)
11. Chapter 6 — Discussion and Analysis (2,500 words)
12. Chapter 7 — Conclusion (1,000 words)
13. References (Harvard style)
14. Appendix A–F

## Key References (Harvard, with DOIs)
- Devlin, J. et al. (2019) BERT. doi:10.18653/v1/N19-1423
- Gehrmann, S. et al. (2019) GLTR. doi:10.18653/v1/P19-3019
- Guo, B. et al. (2023) HC3. arXiv:2301.07597
- Kirchenbauer, J. et al. (2023) Watermarking. arXiv:2301.10226
- Krishna, K. et al. (2023) DIPPER. arXiv:2303.13408
- Liu, Y. et al. (2019) RoBERTa. arXiv:1907.11692
- Mitchell, E. et al. (2023) DetectGPT. arXiv:2301.11305
- Sadasivan, V.S. et al. (2023) Detection reliability. arXiv:2303.11156
- Sokolova, M. and Lapalme, G. (2009) Performance measures. doi:10.1016/j.ipm.2009.03.002
- Wang, Y. et al. (2023) M4 dataset. arXiv:2305.14902

## 13-Week Timeline (Actual Progress)
| Week | Tasks | Status |
|------|-------|--------|
| 1–2 | Setup, HC3 exploration, literature review writing | DONE |
| 3–4 | Preprocessing, RoBERTa fine-tuning, wandb, methodology | DONE |
| 5–6 | T5 paraphrase attacks, BERT/DistilBERT training, all evaluations | DONE |
| 7 | Full evaluation, all results, interim report | IN PROGRESS |
| 8–9 | Human evaluation, collection and analysis | DONE (early) |
| 10–11 | Results and discussion chapters | PENDING |
| 12 | Full dissertation draft, appendices, references | PENDING |
| 13 | Proofread, polish, presentation slides, submit | PENDING |
