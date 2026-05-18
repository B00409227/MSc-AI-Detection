# CLAUDE.md — Project Context for AI Coding Assistant

## Project Identity
- **Title:** Experimental Evaluation of AI-Generated Text Detection Robustness Against Rewriting-Based Attacks
- **Student:** Abdul Hannaan Mohammed | Banner ID: B00409227
- **Programme:** MSc Artificial Intelligence | University of the West of Scotland (UWS)
- **Supervisor:** Dr Tahir Mehmood
- **Academic Year:** 2025/26
- **Current Week:** Week 1 of 13

## Research Question
To what extent do rewriting and paraphrasing attacks reduce the detection accuracy of a transformer-based AI-generated text classifier (RoBERTa), and what does any observed performance degradation reveal about the robustness limitations of current supervised detection approaches when applied under realistic adversarial conditions?

## Pipeline Summary
1. Fine-tune RoBERTa-base on HC3 dataset (human vs AI-generated text)
2. Generate adversarial samples using DIPPER paraphrasing model
3. Evaluate classifier on original + rewritten samples
4. Measure performance degradation (Accuracy, Precision, Recall, F1)
5. Human evaluation via Google Forms (13–14 participants)
6. Analyse and discuss robustness limitations

## Hardware
- Local: Windows 11, 6GB VRAM GPU (for RoBERTa training)
- Google Colab free T4 (15GB VRAM) — DIPPER only

## Models
- `roberta-base` — main classifier (HuggingFace)
- `kalpeshk2011/dipper-paraphraser-xxl` — adversarial rewriting (Colab only)
- `tuner007/pegasus_paraphrase` — secondary paraphrasing

## Dataset
- HC3 (Hello-SimpleAI/HC3) from HuggingFace Hub
- Split: 70% train / 15% validation / 15% test
- Supplementary: 500–1000 ChatGPT-generated samples

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix, ROC Curve, Precision-Recall Curve
- Before vs after rewriting comparison
- Human evaluation scores

## Tech Stack
Python 3.10+, PyTorch, HuggingFace Transformers/Datasets/Accelerate/Evaluate,
Pandas, NumPy, NLTK, scikit-learn, Matplotlib, Seaborn, wandb,
Google Colab, Git/GitHub, Google Forms/Sheets

## Folder Structure
- `data/raw/` — raw HC3 files
- `data/processed/` — train/val/test CSVs
- `data/supplementary/` — ChatGPT-generated samples
- `data/adversarial/` — DIPPER-rewritten samples
- `notebooks/` — all Jupyter notebooks (numbered 01–07)
- `src/` — reusable Python modules
- `models/checkpoints/` — saved model weights (gitignored)
- `results/metrics/` — JSON/CSV results
- `results/figures/` — all plots
- `results/tables/` — results tables
- `results/confusion_matrices/` — confusion matrix data
- `human_evaluation/` — Google Form, responses, analysis
- `dissertation/` — chapter drafts and figures
- `logs/` — training logs
- `screenshots/` — evidence screenshots

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

## Dissertation Structure
1. Title Page
2. Abstract (300 words)
3. Acknowledgements
4. Table of Contents
5. List of Figures / List of Tables
6. Chapter 1 — Introduction (1,500 words)
7. Chapter 2 — Literature Review (4,000 words)
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

## 13-Week Timeline
| Week | Tasks |
|------|-------|
| 1–2 | Setup, HC3 exploration, literature review writing |
| 3–4 | Preprocessing, RoBERTa fine-tuning, wandb, methodology chapter |
| 5–6 | DIPPER adversarial generation on Colab, implementation chapter |
| 7 | Full evaluation, all results, interim report |
| 8–9 | Human evaluation, collection and analysis |
| 10–11 | Results and discussion chapters |
| 12 | Full dissertation draft, appendices, references |
| 13 | Proofread, polish, presentation slides, submit |
