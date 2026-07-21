# DISSERTATION WRITING BRIEF — PASTE THIS INTO A NEW CLAUDE SESSION
# Title: Experimental Evaluation of AI-Generated Text Detection Robustness Against Rewriting-Based Attacks
# Student: Abdul Hannaan Mohammed | B00409227 | MSc Artificial Intelligence | UWS
# Supervisor: Dr Tahir Mahmood | Academic Year 2025/26

---

## HOW TO USE THIS BRIEF
You are helping me write my MSc dissertation. I will tell you which chapter or section to write.
Use EVERY detail in this brief. Do not invent results, numbers, or references — use only what is
listed here. Write in formal academic English, third person, past tense for methodology and results,
present tense for discussion of implications. Harvard referencing throughout.

---

## 1. RESEARCH QUESTION AND AIMS

### Primary Research Question
To what extent do rewriting and paraphrasing attacks reduce the detection accuracy of
transformer-based AI-generated text classifiers, and what does any observed performance degradation
reveal about the robustness limitations of current supervised detection approaches when applied
under realistic adversarial conditions?

### Sub-questions
1. How accurately can transformer-based classifiers distinguish AI-generated from human-written text
   under baseline (no-attack) conditions?
2. Which rewriting attack strategy — Pegasus paraphrase, T5-based QuillBot-style stochastic rewriting,
   or ChatGPT semantic rewriting — most effectively evades detection?
3. Do models trained on one dataset (HC3) generalise to a different dataset (M4)?
4. How does human ability to detect AI-generated text compare to automated classifiers, and does
   paraphrasing further degrade human detection?
5. Can novel hybrid architectures combining transformer features with classical ML or recurrent/
   convolutional components improve robustness?

### Hypotheses
- H1: All transformer models will achieve high F1 (>0.95) on clean data
- H2: Paraphrasing attacks will degrade recall on AI-only samples
- H3: Stochastic/semantic attacks will be more effective than deterministic paraphrase
- H4: Cross-dataset performance will be substantially lower than in-domain performance
- H5: Human performance will be near-chance on rewritten AI text

---

## 2. DATASETS

### Primary Dataset: HC3 (Hello-SimpleAI/HC3)
- Full name: Human ChatGPT Comparison Corpus
- Source: HuggingFace Hub (Hello-SimpleAI/HC3)
- Reference: Guo, B. et al. (2023) arXiv:2301.07597
- Content: Question-answer pairs where both a human expert and ChatGPT answered the same question
- Domains: Finance, medicine, open-domain QA, Wikipedia, Reddit, ELI5, legal
- Total samples after preprocessing: 11,820 (split 70/15/15)
  - Training: 8,274 samples
  - Validation: 1,773 samples
  - Test: 1,773 samples
- Label distribution: approximately balanced (human=0, AI=1)
- Preprocessing: lowercased, truncated to 512 tokens, removed duplicates and empty strings
- Why chosen: gold-standard benchmark for AI detection; covers diverse domains; provides both
  human and AI responses to identical questions, controlling for topic confound

### Secondary Dataset: M4 (Cross-dataset Generalisation Test)
- Full name: AI Text Detection Pile (M4)
- Source: HuggingFace Hub (artem9k/ai-text-detection-pile)
- Reference: Wang, Y. et al. (2023) arXiv:2305.14902
- Content: Multi-domain, multi-generator AI text detection benchmark
- Generators: ChatGPT, GPT-4, Davinci, Cohere, BLOOMz, and human-written text
- Test split used: 2,000 samples
- Why chosen: Tests out-of-distribution generalisation; different generators and domains to HC3;
  represents real-world deployment conditions where training and test distributions differ

### Adversarial Attack Datasets (generated, not sourced)
- Each attack applied to 500 AI-generated samples from HC3 test set
- Three datasets generated:
  1. Pegasus-Attack: 500 samples (Pegasus paraphrase)
  2. QuillBot-Attack: 500 samples (T5 stochastic sampling)
  3. ChatGPT-Attack: 500 samples (GPT-3.5 semantic rewriting)

---

## 3. ALL FIVE INDIVIDUAL MODELS

### Model 1: RoBERTa-base
- Architecture: Robustly Optimised BERT Pre-training Approach
- Base: Liu et al. (2019) arXiv:1907.11692 (roberta-base from HuggingFace)
- Parameters: 125M
- Training: Fine-tuned on HC3 training split (8,274 samples)
- Training details: Adam optimiser, lr=2e-5, batch size=16, 3 epochs, weight_decay=0.01,
  warmup_steps=500, max_length=512
- Hardware: Local GPU (6GB VRAM, Windows 11)
- Wandb logging: Yes
- HuggingFace Hub: ahm1129/roberta-hc3-detector
- Key differences from BERT: Dynamic masking, no NSP objective, trained on 10x more data,
  longer sequences, larger batch sizes
- Why chosen: State-of-the-art among base-size encoders; widely used in NLP classification

### Model 2: BERT-base-uncased
- Architecture: Bidirectional Encoder Representations from Transformers
- Base: Devlin et al. (2019) doi:10.18653/v1/N19-1423 (bert-base-uncased)
- Parameters: 110M
- Training: Fine-tuned on HC3 training split
- Training details: Same hyperparameters as RoBERTa
- Hardware: Local GPU (6GB VRAM)
- HuggingFace Hub: ahm1129/bert-hc3-detector
- Why chosen: Foundational transformer model; provides baseline for comparing RoBERTa improvements

### Model 3: DistilBERT-base-uncased
- Architecture: Knowledge-distilled version of BERT
- Base: Sanh et al. (2019) (distilbert-base-uncased)
- Parameters: 66M (40% fewer than BERT)
- Training: Fine-tuned on HC3 training split
- Training details: Same hyperparameters; trained on Google Colab T4 GPU (15GB VRAM)
  due to memory constraints of local GPU during simultaneous experiments
- HuggingFace Hub: ahm1129/distilbert-hc3-detector
- Why chosen: Efficiency-accuracy trade-off; tests whether smaller models are more/less robust

### Model 4: Hello-SimpleAI Detector (Pre-trained baseline)
- Architecture: RoBERTa-base fine-tuned by Hello-SimpleAI
- HuggingFace: Hello-SimpleAI/chatgpt-detector-roberta
- Training: Pre-trained by original HC3 authors on their own split — NOT re-trained in this study
- Reference: Guo et al. (2023) HC3 paper
- Why chosen: Represents a real-world deployed detector; tests whether published detectors are
  robust; provides external validity comparison

### Model 5: Logistic Regression + TF-IDF (Classical ML Baseline)
- Architecture: scikit-learn Pipeline — TF-IDF vectoriser (max_features=50,000, ngram_range=(1,2),
  sublinear_tf=True) → Logistic Regression (C=1.0, max_iter=1000, class_weight='balanced')
- Training: Fit on HC3 training split
- Features: 50,000 TF-IDF bigram features
- Saved: models/logistic_regression/lr_model.pkl (scikit-learn 1.7.2)
- Why chosen: Classical ML baseline; no contextual understanding; isolates the value of
  deep contextual representations; highly interpretable

---

## 4. FOUR HYBRID ARCHITECTURES (Novel Contributions)

### H1: RoBERTa + BiLSTM
- Category: Research-only (trained on Colab; weights not deployed to live app)
- Architecture: RoBERTa-base encoder → [CLS] token representation →
  BiLSTM (hidden=256, bidirectional, 2 layers) → Dropout(0.3) → Linear → Sigmoid
- Motivation: RoBERTa captures global context; BiLSTM adds sequential temporal modelling
  of the token sequence; hypothesised to capture writing rhythm and style shifts
- Training: Google Colab T4 GPU, HC3 training split, 5 epochs, lr=1e-5
- Results: Clean F1=0.9910, Pegasus ASR=1.4%, QuillBot ASR=4.8%, ChatGPT ASR=2.2%, M4 F1=0.7127
- Notable: Lowest QuillBot ASR of all 9 models (4.8%) suggesting BiLSTM sequential modelling
  captures writing patterns that survive stochastic paraphrase

### H2: BERT + TextCNN
- Category: Research-only (trained on Colab; weights not deployed to live app)
- Architecture: BERT-base encoder → all token hidden states →
  TextCNN (filters: 128 each for kernel sizes 2,3,4,5) → Global max-pool →
  Concatenate → Dropout(0.4) → Linear → Sigmoid
- Motivation: CNN local n-gram feature extraction from contextual BERT representations;
  captures local phrasal patterns at multiple scales simultaneously
- Training: Google Colab T4 GPU, HC3 training split, 5 epochs, lr=2e-5
- Results: Clean F1=0.9695, Pegasus ASR=4.2%, QuillBot ASR=9.6%, ChatGPT ASR=3.8%, M4 F1=0.5923
- Notable: Lower clean F1 than pure BERT, suggesting CNN heads may introduce noise on this task

### H3: Soft Voting Ensemble
- Category: Live inference (available in deployed Streamlit app)
- Architecture: Average of P(AI) scores from RoBERTa-base + BERT-base + DistilBERT
  (all three fine-tuned models loaded simultaneously)
  Final prediction = mean probability, threshold at 0.5
- Motivation: Ensemble reduces individual model variance; captures complementary decision
  boundaries; no additional training required
- Results: Clean F1=0.9916, Pegasus ASR=1.8%, QuillBot ASR=12.0%, ChatGPT ASR=3.2%, M4 F1=0.5825
- Notable: More stable than individual models; QuillBot ASR slightly better than BERT alone

### H4: Feature Fusion MLP
- Category: Live inference (available in deployed Streamlit app)
- Architecture: 507-dimensional feature vector composed of:
  → RoBERTa P(AI) [1 dim]
  → TF-IDF top-500 features [500 dims]
  → Stylometric features [6 dims]: avg sentence length, type-token ratio, punctuation density,
    function word ratio, avg word length, sentence count normalised
  → MLP: Linear(507→256) → ReLU → Dropout(0.3) → Linear(256→64) → ReLU → Linear(64→1) → Sigmoid
- Motivation: Combines deep neural probability with classical lexical features and
  surface-level stylometric signals; tests whether hand-crafted features add value
  over pure neural representations
- Saved components: h4_tfidf.pkl, h4_scaler.pkl, h4_mlp.pt
- Results: Clean F1=0.9929, Pegasus ASR=1.8%, QuillBot ASR=16.0%, ChatGPT ASR=2.8%, M4 F1=0.5896
- Notable: Highest clean F1 tied with Hello-SimpleAI; QuillBot ASR higher than H3, suggesting
  TF-IDF component is more vulnerable to lexical substitution attacks

---

## 5. THREE ADVERSARIAL ATTACKS

### Attack 1: Pegasus Paraphrase
- Model: tuner007/pegasus_paraphrase (from HuggingFace Hub)
- Architecture: PEGASUS — Pre-training with Extracted Gap Sentences for Abstractive Summarization
  (adapted for paraphrase generation)
- Method: Beam search decoding, num_beams=10, temperature=1.0, max_length=512
- Deterministic: Yes (same input → same output)
- Generated on: Google Colab T4 GPU (notebook 05_paraphrase_attack_colab)
- Samples: 500 AI-generated texts from HC3 test set
- Goal: Surface-level syntactic restructuring while preserving semantics
- Saved: data/adversarial/ directory

### Attack 2: QuillBot-Style T5 Stochastic Rewriting
- Model: Vamsi/T5_Paraphrase_Paws (from HuggingFace Hub)
- Architecture: T5 (Text-to-Text Transfer Transformer) fine-tuned on PAWS paraphrase dataset
- Method: Stochastic sampling, top_k=200, top_p=0.95, temperature=1.2, do_sample=True
- Deterministic: No (sampling introduces randomness — mimics QuillBot's paraphrase mode)
- Generated on: Local machine (Windows 11, 6GB VRAM GPU)
- Samples: 500 AI-generated texts from HC3 test set
- Goal: Lexical and structural variation through stochastic sampling;
  more aggressive lexical substitution than beam search
- Also used for Human Evaluation Study (participants evaluated these rewritten samples)

### Attack 3: ChatGPT Semantic Rewriting
- Model: OpenAI gpt-3.5-turbo via API
- Method: Prompt: "Rewrite the following text in a natural, human-like style, varying sentence
  structure and vocabulary while preserving the core meaning: [text]"
- Generated on: Local machine using OpenAI API (src/ scripts)
- Samples: 500 AI-generated texts from HC3 test set
- Goal: High-level semantic rewriting; changes not just surface form but argumentation style
- Cost: approximately $2.50 USD (500 samples)
- This represents the most sophisticated adversary in the study

### Why DIPPER Was Excluded
- DIPPER (kalpeshk2011/dipper-paraphraser-xxl, Krishna et al. 2023) was originally planned
- 11B parameter model — exceeds local GPU (6GB VRAM) and Colab T4 (15GB VRAM)
- Excluded due to hardware constraints
- Still cited in literature review as the leading adversarial paraphraser in the field

---

## 6. COMPLETE RESULTS — ALL 9 MODELS

### Table 1: All Models — Complete Results Summary

| Model               | Type       | Clean F1 | Pegasus ASR | QuillBot ASR | ChatGPT ASR | M4 F1  |
|---------------------|------------|----------|-------------|--------------|-------------|--------|
| RoBERTa-base        | Individual | 0.9913   | 1.2%        | 13.4%        | 1.6%        | 0.7389 |
| BERT-base           | Individual | 0.9845   | 1.8%        | 12.2%        | 3.2%        | 0.5999 |
| DistilBERT          | Individual | 0.9922   | 5.2%        | 19.0%        | 6.8%        | 0.4316 |
| Hello-SimpleAI      | Individual | 0.9929   | 0.4%        | 14.0%        | 2.8%        | 0.5442 |
| Logistic Regression | Individual | 0.9524   | 29.0%       | 26.8%        | 39.4%       | 0.3356 |
| H3-SoftVoting       | Hybrid     | 0.9916   | 1.8%        | 12.0%        | 3.2%        | 0.5825 |
| H4-FeatureFusion    | Hybrid     | 0.9929   | 1.8%        | 16.0%        | 2.8%        | 0.5896 |
| H1-RoBERTa+BiLSTM   | Hybrid     | 0.9910   | 1.4%        | 4.8%         | 2.2%        | 0.7127 |
| H2-BERT+TextCNN     | Hybrid     | 0.9695   | 4.2%        | 9.6%         | 3.8%        | 0.5923 |

ASR = Attack Success Rate = percentage of AI texts that evaded detection after rewriting.
Lower ASR = more robust model.

### Table 2: Detailed Clean Performance (5 Individual Models, HC3 test set n=11,820)

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| RoBERTa-base        | 0.9942   | 0.9832    | 0.9995 | 0.9913 | 0.9999  |
| BERT-base           | 0.9895   | 0.9697    | 0.9997 | 0.9845 | 0.9999  |
| DistilBERT          | 0.9948   | 0.9850    | 0.9995 | 0.9922 | 0.9999  |
| Hello-SimpleAI      | 0.9953   | 0.9882    | 0.9977 | 0.9929 | 0.9997  |
| Logistic Regression | 0.9689   | 0.9689    | 0.9364 | 0.9524 | 0.9961  |

### Table 3: Recall Under Attack (AI-only samples, n=500 per attack)
Recall = proportion of AI texts correctly identified as AI. ASR = 1 - Recall.

| Model               | Pegasus Recall | QuillBot Recall | ChatGPT Recall |
|---------------------|---------------|-----------------|----------------|
| RoBERTa-base        | 0.988         | 0.866           | 0.984          |
| BERT-base           | 0.982         | 0.878           | 0.968          |
| DistilBERT          | 0.948         | 0.810           | 0.932          |
| Hello-SimpleAI      | 0.996         | 0.860           | 0.972          |
| Logistic Regression | 0.710         | 0.732           | 0.606          |

### Table 4: Cross-Dataset Generalisation (M4, n=2,000)

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| RoBERTa-base        | 0.7700   | 0.8543    | 0.6510 | 0.7389 | 0.8548  |
| BERT-base           | 0.7045   | 0.9287    | 0.4430 | 0.5999 | 0.8966  |
| DistilBERT          | 0.6325   | 0.9522    | 0.2790 | 0.4316 | 0.8188  |
| Hello-SimpleAI      | 0.6750   | 0.9108    | 0.3880 | 0.5442 | 0.8511  |
| Logistic Regression | 0.5525   | 0.6513    | 0.2260 | 0.3356 | 0.7104  |

Note: All models show substantial F1 drop from HC3 to M4 (domain/generator shift).
RoBERTa generalises best (HC3→M4 drop: 0.9913→0.7389 = -0.2524).

### Table 5: Human Evaluation Study Results

| Condition    | Human Accuracy | Correct / Total | N participants |
|--------------|----------------|-----------------|----------------|
| Human text   | 50.7%          | 38 / 75         | 15             |
| Original AI  | 44.0%          | 33 / 75         | 15             |
| Rewritten AI | 30.7%          | 23 / 75         | 15             |
| OVERALL      | 41.8%          | 94 / 225        | 15             |

Chance level = 50%. Rewritten AI accuracy (30.7%) is BELOW chance — participants
actively misidentified rewritten AI text as human more often than chance would predict.
(5 texts per condition per participant = 75 responses per condition total)

---

## 7. HUMAN EVALUATION STUDY FULL DETAILS

- Participants: N=15 (recruited via university mailing list and personal contacts)
- Platform: Google Forms
- Format: 15 questions total, binary forced-choice label (Human / AI), no time limit
- Stimuli breakdown: 5 human-written texts + 5 original ChatGPT texts + 5 T5-QuillBot rewritten AI texts
  (balanced across three conditions, 5 questions each)
- Texts: Selected from HC3 test set; standardised for approximate length (~150-250 words)
- Participant instruction: "You will read 15 text samples. For each, decide whether it was
  written by a human or an AI. There are no trick questions."
- No training or calibration given beforehand
- Ethics: Anonymous, voluntary, no personal data collected beyond age range and academic level
- Analysis: Per-condition accuracy, binomial test against 50% chance

### Key Human Study Findings
- Human text accuracy (50.7%): Not significantly different from chance (p>0.05)
  Humans cannot reliably identify human-written text as human
- Original AI accuracy (44.0%): Below chance — humans show slight bias toward labelling AI text as human
- Rewritten AI accuracy (30.7%): Significantly below chance — participants are systematically
  fooled by rewritten AI text, mislabelling it as human most of the time
- This confirms paraphrasing attacks degrade HUMAN detection, not just automated detection
- Humans are worse detectors than ALL five automated models even on clean data
- After paraphrasing, the performance gap between humans and machines widens dramatically

---

## 8. FIGURES — COMPLETE INVENTORY

### Figures already generated (all in results/figures/ at 300 DPI):

Figure 1 — fig01_clean_f1_comparison.png
Bar chart: Clean F1 scores for all 5 individual models
Caption: "Figure 1: Baseline F1 scores of five classifiers on the HC3 test set (n=11,820)"
USE IN: Chapter 5, Section 5.1

Figure 2 — fig02_recall_under_attack.png
Grouped bar chart: Recall under each attack, one group per model (5 models × 3 attacks)
Caption: "Figure 2: Recall of AI-class detection after each paraphrase attack (n=500 per condition)"
USE IN: Chapter 5, Sections 5.2–5.4

Figure 3 — fig03_f1_degradation.png
Line chart: F1 score across conditions (Clean → Pegasus → QuillBot → ChatGPT) per model
Caption: "Figure 3: F1 score trajectory across experimental conditions for each classifier"
USE IN: Chapter 5

Figure 4 — fig04_recall_heatmap.png
Heatmap: Models (rows) × Conditions (columns), colour-coded by recall value
Caption: "Figure 4: Recall heatmap — classifiers (rows) versus attack conditions (columns)"
USE IN: Chapter 5

Figure 5 — fig05_m4_generalisation.png
Bar chart: HC3 F1 vs M4 F1 for each model (side by side)
Caption: "Figure 5: In-domain (HC3) versus cross-domain (M4) F1 scores for all classifiers"
USE IN: Chapter 5, Section 5.5

Figure 6 — fig06_human_vs_model.png
Bar chart: Human accuracy per condition vs best/worst model recall
Caption: "Figure 6: Human participant accuracy compared to automated classifier recall across conditions"
USE IN: Chapter 5, Section 5.6

Figure 7 — fig07_confusion_matrices.png
5-panel confusion matrices (one per model, clean condition)
Caption: "Figure 7: Confusion matrices for all five classifiers on the HC3 test set"
USE IN: Chapter 5, Section 5.1

Figure 8 — fig_all_models_f1_clean.png
Extended bar chart including all 9 models (5 individual + 4 hybrid)
Caption: "Figure 8: Baseline F1 scores for all nine classifiers including hybrid architectures"
USE IN: Chapter 5, Section 5.7

Figure 9 — fig_all_models_attack_comparison.png
Grouped bars: All 9 models × all 3 attacks (ASR comparison)
Caption: "Figure 9: Attack Success Rate comparison across all nine models and three attack types"
USE IN: Chapter 5, Section 5.7

Figure 10 — fig_all9_models_clean_f1.png
Bar chart: All 9 model clean F1 comparison
Caption: "Figure 10: Clean F1 comparison — individual versus hybrid architectures"
USE IN: Chapter 5

Figure 11 — fig_all9_models_asr_comparison.png
Grouped bar chart: ASR for all 9 models
Caption: "Figure 11: Attack Success Rate across all nine models by attack type"
USE IN: Chapter 5

Figure 12 — fig_all9_models_degradation_heatmap.png
Heatmap: All 9 models × 3 attacks (ASR values), colour-coded
Caption: "Figure 12: Performance degradation heatmap — all nine models versus three attack types"
USE IN: Chapter 5

Figure 13 — fig_all9_models_radar_chart.png
Radar/spider chart: Clean F1, Pegasus robustness, QuillBot robustness, ChatGPT robustness,
M4 generalisation — plotted for all 9 models
Caption: "Figure 13: Multi-dimensional performance radar — clean accuracy, three attack robustness
scores, and cross-dataset generalisation for all nine classifiers"
USE IN: Chapter 5 / Chapter 6

Figure 14 — fig_performance_degradation_heatmap.png
Heatmap: 5 individual models only
Caption: "Figure 14: Performance degradation heatmap for five individual classifiers"
USE IN: Chapter 5

Figure 15 — fig12_per_question_accuracy.png
Bar chart: Accuracy per question in human study (Q1–Q15), colour-coded by condition
Caption: "Figure 15: Human participant accuracy for each individual question in the evaluation study"
USE IN: Chapter 5, Section 5.6

Figure 16 — fig13_human_vs_roberta.png
Comparison bar chart: Human accuracy vs RoBERTa recall across three conditions
Caption: "Figure 16: Human participant accuracy versus RoBERTa-base recall across text conditions"
USE IN: Chapter 5

Figure 17 — fig14_human_accuracy_by_condition.png
Bar chart: Human accuracy by condition (Human text, Original AI, Rewritten AI)
Caption: "Figure 17: Human participant accuracy by text condition"
USE IN: Chapter 5, Section 5.6

Figure 18 — fig_cross_dataset_comparison.png
Grouped bars: HC3 F1 vs M4 F1 for all models with percentage drop annotation
Caption: "Figure 18: Cross-dataset generalisation — performance drop from HC3 to M4"
USE IN: Chapter 5, Section 5.5

### Additional diagrams to CREATE for the dissertation:

Diagram A — RoBERTa fine-tuning pipeline architecture
Content: Input text → Tokeniser (WordPiece, max 512 tokens) → RoBERTa-base (12 transformer
layers, 768 hidden dims, 12 attention heads) → [CLS] token embedding → Dropout(0.1) →
Linear(768→2) → Softmax → Class label (Human=0, AI=1)
Tool: draw.io, PowerPoint, or Lucidchart
Save as: results/figures/diagram_roberta_pipeline.png
Caption: "Figure X: Architecture of the RoBERTa-base fine-tuning pipeline for binary classification"

Diagram B — H1: RoBERTa + BiLSTM architecture
Content: Input → RoBERTa encoder → [CLS] 768-dim → BiLSTM(hidden=256, layers=2, bidirectional)
→ 512-dim → Dropout(0.3) → Linear(512→1) → Sigmoid → P(AI)
Save as: results/figures/diagram_h1_bilstm.png

Diagram C — H4: Feature Fusion MLP architecture
Content: Three branches merging:
Branch 1: RoBERTa → P(AI) [1 dim]
Branch 2: Raw text → TF-IDF vectoriser → top 500 features [500 dims]
Branch 3: Raw text → Stylometric extractor → 6 features [6 dims]
All concatenated → 507-dim vector → MLP(507→256→64→1) → Sigmoid
Save as: results/figures/diagram_h4_feature_fusion.png

Diagram D — Overall experimental pipeline (flowchart)
Steps: HC3 Dataset → Preprocessing & Splitting → Fine-tuning (5 models) → Clean Evaluation →
Attack Generation (Pegasus / T5-QuillBot / ChatGPT) → Adversarial Evaluation →
M4 Cross-dataset Evaluation → Human Evaluation Study → Statistical Analysis
Save as: results/figures/diagram_experimental_pipeline.png
Caption: "Figure X: Overview of the experimental pipeline"

Diagram E — Attack taxonomy
Visual: Three-column table or tree:
Deterministic | Stochastic | Semantic
tuner007/pegasus_paraphrase | Vamsi/T5_Paraphrase_Paws | OpenAI gpt-3.5-turbo
Beam search, num_beams=10 | Sampling, top_k=200, top_p=0.95 | Prompt-based rewriting
Surface syntactic | Lexical substitution | Semantic transformation

### Screenshots to take from the deployed app (msc-ai-detector.streamlit.app):

Screenshot A — Single Analysis mode, run RoBERTa on a sample ChatGPT text and show result
File name: nb_streamlit_single_analysis.png
For: Appendix E

Screenshot B — Human vs AI Lab showing all 7 models' results side by side
File name: nb_streamlit_lab_comparison.png
For: Appendix E

Screenshot C — Generate & Detect tab after clicking the Generate button, showing detection result
File name: nb_streamlit_generate_detect.png
For: Appendix E

Screenshot D — Model Explorer showing the RoBERTa model card with architecture details
File name: nb_streamlit_model_explorer.png
For: Appendix E

Screenshot E — Attack Simulation tab with results from a paraphrase attack
File name: nb_streamlit_attack_simulation.png
For: Appendix E

Screenshot F — Hybrid Research section showing H3/H4 ensemble inference output
File name: nb_streamlit_hybrid_research.png
For: Appendix E

Screenshot G — WandB training dashboard for RoBERTa: training loss + validation F1 over 3 epochs
File name: nb03_roberta_training_curves.png
For: Appendix F (or Chapter 4)

Screenshot H — WandB training dashboard for BERT
File name: nb08_bert_training_curves.png
For: Appendix F

Screenshot I — Colab training output for DistilBERT (cell output or WandB)
File name: nb11_distilbert_training_curves.png
For: Appendix F

Screenshot J — Google Form interface for the human evaluation study
File name: nb07_human_eval_form.png
For: Appendix B

Screenshot K — Google Sheets raw response data (first 5 rows + column headers visible)
File name: nb07_human_eval_responses.png
For: Appendix B

Screenshot L — GitHub repository main page showing project structure and recent commits
File name: evidence_github_repo.png
For: Appendix D

---

## 9. DISSERTATION CHAPTER STRUCTURE AND WORD COUNTS

Total target: approximately 17,000–18,000 words (excluding references and appendices)

---

### ABSTRACT — 300 words
Cover:
- Research question (AI text detection robustness against paraphrase attacks)
- Experimental setup: 5 individual + 4 hybrid models; 3 attacks; 500 samples each; HC3 + M4 datasets
- Key findings:
  - All transformer models achieve near-perfect F1 on clean data (0.9845–0.9929)
  - Pegasus paraphrase largely ineffective (0.4%–5.2% ASR for transformers)
  - QuillBot-style stochastic rewriting most effective (12.2%–19.0% ASR)
  - ChatGPT rewriting surprisingly ineffective against transformers (1.6%–6.8% ASR)
  - Logistic Regression most vulnerable (29.0%–39.4% ASR)
  - Cross-dataset (M4) F1 drops substantially for all models
  - Human accuracy 41.8% overall, 30.7% on rewritten AI (below chance)
  - H1-RoBERTa+BiLSTM achieves lowest QuillBot ASR of all models (4.8%)
- Conclusion: Supervised transformers are robust to simple paraphrase but fail on distribution shift;
  hybrid sequential architectures show promise

---

### CHAPTER 1: INTRODUCTION — 1,500 words

1.1 Background and Motivation (~400 words)
- Rise of LLMs: GPT-3, GPT-4, ChatGPT — scale and accessibility
- Proliferation of AI-generated content: academic fraud, misinformation, content farms
- Detection as critical sociotechnical challenge
- Reference: Brown et al. (2020) GPT-3; cite OpenAI ChatGPT release (2022)

1.2 Problem Statement (~300 words)
- Current detectors perform well under clean conditions but face fundamental robustness challenge
- Simple paraphrasing can evade detection (Krishna et al. 2023 showed DIPPER causes 80%+ ASR)
- Gap: most studies use single model / single attack; no comprehensive multi-model multi-attack study

1.3 Research Aims and Objectives (~300 words)
- List all 5 sub-questions (from Section 1 above)
- State the experimental design in one paragraph

1.4 Scope and Limitations (~200 words)
- English language only
- HC3 dataset domain (QA pairs)
- Base-size models (no large models tested)
- 500 adversarial samples per attack (computational constraint)
- DIPPER excluded (hardware constraint — 11B parameters)

1.5 Dissertation Structure (~200 words)
- One sentence per chapter describing its content

---

### CHAPTER 2: LITERATURE REVIEW — 4,000 words [ALREADY WRITTEN]

Sections already present:
2.1 Rise of Large Language Models
2.2 AI-Generated Text Detection Approaches (watermarking, statistical, neural)
2.3 Key Detection Systems (GLTR, DetectGPT, watermarking, supervised classifiers)
2.4 Adversarial Attacks on Detection (DIPPER, paraphrase attacks)
2.5 Datasets for AI Detection (HC3, M4, TuringBench)
2.6 Human Ability to Detect AI Text
2.7 Hybrid and Ensemble Approaches
2.8 Research Gap and Positioning

Key references that MUST appear: Guo et al. 2023 (HC3), Krishna et al. 2023 (DIPPER),
Devlin et al. 2019 (BERT), Liu et al. 2019 (RoBERTa), Sanh et al. 2019 (DistilBERT),
Vaswani et al. 2017 (Attention), Mitchell et al. 2023 (DetectGPT), Kirchenbauer et al. 2023
(Watermarking), Sadasivan et al. 2023 (Can AI text be detected), Wang et al. 2023 (M4),
Gehrmann et al. 2019 (GLTR), Brown et al. 2020 (GPT-3), Wolf et al. 2020 (HuggingFace)

---

### CHAPTER 3: RESEARCH DESIGN AND METHODOLOGY — 3,500 words

3.1 Research Philosophy and Design (~400 words)
- Experimental, empirical, quantitative research design
- Controlled between-subjects model comparison across attack conditions
- Reproducible experimental protocol; all code on GitHub
- Justify why quantitative metrics (F1, ASR) are appropriate for this question

3.2 Dataset Collection and Preparation (~500 words)
- HC3: describe source, domains, preprocessing steps
  (tokenise, lowercase, truncate to 512, remove empty/duplicate)
- 70/15/15 split: training 8,274 / validation 1,773 / test 1,773
- M4: describe source, sample selection, 2,000 samples
- Justify why these datasets were chosen
- Table 1: Dataset statistics

3.3 Model Selection Rationale (~500 words)
- RoBERTa-base: SOTA encoder, dynamic masking, no NSP, 10x more training data
- BERT-base: foundational transformer benchmark
- DistilBERT: efficiency-accuracy test
- Hello-SimpleAI: deployed real-world detector (external validity)
- Logistic Regression: classical ML baseline (interpretability, no contextual knowledge)
- Why these 5 span the space of detection approaches
- Table 2: Architecture comparison

3.4 Attack Methodology (~700 words)
- Threat model: black-box adversary with API or open-source model access
- Why three attacks? Cover deterministic, stochastic, semantic threat types
- Pegasus: syntactic restructuring, beam search, deterministic
- T5-QuillBot: lexical substitution, stochastic sampling, more aggressive
- ChatGPT: semantic rewriting via LLM, most sophisticated
- Why DIPPER was excluded: 11B params, OOM on all available hardware
- Sample size: 500 per attack (computational constraint; statistical sufficiency for recall estimation)
- Table 4: Attack configurations

3.5 Evaluation Metrics (~400 words)
- Accuracy, Precision, Recall, F1: definitions and why each matters
- ROC-AUC for ranking quality
- ASR = 1 - Recall on AI-only samples: most critical metric
- Cite Sokolova and Lapalme (2009) for metric justification
- Why per-class recall (not accuracy) for imbalanced adversarial scenarios

3.6 Human Evaluation Design (~500 words)
- Participants: N=15, recruitment method
- Platform: Google Forms; binary forced-choice per text
- Three-condition design (within-subject? between? — each participant saw all 3 conditions)
- Stimuli selection criteria (representative, length-matched, domain-diverse)
- Why T5-QuillBot samples for rewritten condition? Most effective attack
- Statistical analysis: binomial test against 50% chance

3.7 Hybrid Architecture Design (~500 words)
- Motivation: does architectural augmentation improve robustness?
- H1: Sequential augmentation hypothesis — BiLSTM captures rhythm patterns
- H2: Local pattern hypothesis — CNN captures n-gram style fingerprints
- H3: Ensemble variance reduction — averaging reduces individual errors
- H4: Feature diversity hypothesis — stylometric signals complement neural features
- Training decisions: same data, comparable training budgets

---

### CHAPTER 4: IMPLEMENTATION — 2,500 words

4.1 Software Environment and Tools (~200 words)
- Python 3.10, PyTorch 2.2.2, HuggingFace Transformers 4.40.1
- scikit-learn 1.7.2 for Logistic Regression and TF-IDF
- Weights & Biases for experiment tracking
- Google Colab T4 (15GB) for DistilBERT, H1, H2, Pegasus attack
- Local GPU 6GB VRAM: RoBERTa, BERT, QuillBot attack
- Git/GitHub for version control

4.2 Model Fine-tuning (~600 words)
- Tokenisation: AutoTokenizer, max_length=512, padding='max_length', truncation=True
- DataLoader: batch_size=16, shuffled for training
- Optimiser: AdamW, lr=2e-5, weight_decay=0.01, warmup_steps=500
- Training loop: 3 epochs, evaluate on validation set each epoch
- Best checkpoint saved by validation F1
- WandB logging: loss, accuracy, F1 per step
- HuggingFace Hub push: ahm1129/roberta-hc3-detector etc.
- Logistic Regression: TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)
  + LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
- Table 3: Hyperparameter summary
- Screenshot G/H/I: WandB training curves

4.3 Attack Implementation (~500 words)
- Pegasus (notebook 05, Colab): pipeline("text2text-generation", model="tuner007/pegasus_paraphrase"),
  generate with num_beams=10, max_length=512
- QuillBot-style (local): pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws"),
  generate with do_sample=True, top_k=200, top_p=0.95, temperature=1.2
- ChatGPT (src/ scripts): openai.ChatCompletion.create(model="gpt-3.5-turbo"), prompt described above
- All outputs saved to data/adversarial/ as CSV with columns: original_text, rewritten_text, label

4.4 Evaluation Pipeline (~400 words)
- Notebook 10: iterates all 5 models × all conditions (clean + 3 attacks + M4)
- Each condition: load model, run inference on test set, compute metrics via sklearn
- Results saved to results/metrics/ as JSON files per model-condition pair
- Notebook 12: consolidates to master CSV (table03_master_results.csv, table_all9_models_master.csv)
- Notebook 13: generates all 18 dissertation figures

4.5 Hybrid Model Implementation (~400 words)
- H1: PyTorch nn.Module — RoBERTaModel as backbone (frozen after epoch 1), BiLSTM head
  Trained 5 epochs, lr=1e-5, Colab
- H2: PyTorch nn.Module — BertModel as backbone, TextCNN head (4 filter sizes)
  Trained 5 epochs, lr=2e-5, Colab
- H3: Python function — loads all 3 fine-tuned models, averages softmax outputs
  No training required
- H4: Three-stage pipeline:
  Stage 1: Extract RoBERTa P(AI) probability for each text
  Stage 2: TF-IDF transform (same vectoriser fit on training data)
  Stage 3: Compute 6 stylometric features per text
  Concatenate → 507-dim → train MLP (PyTorch) on training split
  Saved: h4_tfidf.pkl, h4_scaler.pkl (StandardScaler), h4_mlp.pt

4.6 Streamlit Demo Platform (~400 words)
- File: app/streamlit_app.py (~1,800 lines)
- Deployed: Streamlit Cloud (msc-ai-detector.streamlit.app)
- Six operational modes:
  1. Single Text Analysis — classify any text with any model
  2. Human vs AI Lab — side-by-side comparison
  3. Generate & Detect — AI text generation + classification
  4. Attack Simulation — show before/after paraphrase detection
  5. Model Explorer — architecture cards for all 9 models
  6. Hybrid Research — H1–H4 technical deep-dive
- CPU-only deployment (torch==2.2.2+cpu), ~1GB RAM free tier
- IS_CLOUD detection (os.path.exists("/mount/src")) for cloud-safe fallbacks
- Seven models with live inference (3 fine-tuned + H3 + H4 + Hello-SimpleAI + LR)
- H1/H2: research-only, shown with results tables but no live inference
- Screenshots A–F: from deployed app

---

### CHAPTER 5: RESULTS AND EVALUATION — 2,500 words

5.1 Baseline Performance on Clean HC3 Test Set (~400 words)
- Table 2 + Figure 1 + Figure 7 (confusion matrices)
- All models achieve F1 > 0.98 except LR (0.9524)
- Near-perfect ROC-AUC for all transformers (0.9997–0.9999)
- LR achieves competitive F1 but lower recall (0.9364) suggesting conservative AI classification
- Conclusion: Under clean conditions, AI detection is near-solved for in-domain transformer models

5.2 Performance Under Pegasus Paraphrase Attack (~400 words)
- Table 3 (Pegasus column) + Figure 2
- Transformer ASRs: 0.4%–5.2% — remarkably robust
- DistilBERT most vulnerable (5.2%), Hello-SimpleAI most robust (0.4%)
- LR most vulnerable: 29.0% ASR
- Deterministic beam-search paraphrase is insufficient to evade neural contextual detectors
- Possible explanation: Pegasus restructures syntax but preserves lexical distribution transformers rely on

5.3 Performance Under QuillBot-Style T5 Attack (~400 words)
- Table 3 (QuillBot column) + Figure 2
- Most effective attack overall: 12.2%–19.0% ASR for transformers
- DistilBERT most vulnerable (19.0%), H1-BiLSTM most robust (4.8%)
- LR ASR: 26.8% — similar to Pegasus (both attack TF-IDF bigrams)
- Stochastic sampling causes greater lexical diversity than beam search
- H1's sequential modelling appears to capture style features that survive lexical substitution

5.4 Performance Under ChatGPT Semantic Rewriting (~400 words)
- Table 3 (ChatGPT column) + Figure 2
- Surprisingly low ASR against transformers: 1.6%–6.8%
- LR most vulnerable: 39.4% ASR — semantic rewriting restructures sentence forms, destroying TF-IDF features
- Transformer robustness to ChatGPT rewrite may indicate: (a) ChatGPT rewrite prompt not adversarially
  optimised; (b) deep distributional features survive even semantic rewriting
- This contrasts with the expectation that LLM rewriting would be the most effective attack

5.5 Cross-Dataset Generalisation (M4) (~300 words)
- Table 4 + Figures 5 and 18
- Largest HC3→M4 drop: LR (0.9524 → 0.3356, -0.617 F1)
- Smallest drop: RoBERTa (0.9913 → 0.7389, -0.252 F1)
- High M4 precision but low recall for all transformers: models are conservative (prefer false negatives)
- Connects to Sadasivan et al. (2023): domain generalisation is a fundamental challenge
- No model tested here was trained on M4 — this is zero-shot cross-domain transfer

5.6 Human Evaluation Results (~300 words)
- Table 5 + Figures 15, 16, 17
- Overall 41.8% accuracy — below chance (50%)
- Rewritten AI (30.7%): significantly below chance; paraphrase actively fools human judges
- No participant correctly identified all original AI texts (44.0% condition accuracy)
- Even human text was only identified correctly half the time (50.7%)
- All five automated models outperform human accuracy on every condition

5.7 Hybrid Architecture Results (~300 words)
- Tables 1 and 10 + Figures 8–13
- H1 standout: QuillBot ASR 4.8% — lowest of ALL 9 models
- H3 competitive: QuillBot ASR 12.0% (comparable to BERT 12.2%)
- H4 highest clean F1 (tied 0.9929) but QuillBot ASR 16.0% > H3
- H2 lowest clean F1 (0.9695); TextCNN heads may introduce noise
- Hybrid approaches show marginal overall gains; H1's BiLSTM provides genuine robustness improvement

---

### CHAPTER 6: DISCUSSION AND ANALYSIS — 2,500 words

6.1 Interpretation of Main Findings (~600 words)
- Central finding: transformers are much more robust than expected to paraphrase attacks
- Pegasus (syntactic) < QuillBot (lexical) < expectation for ChatGPT (semantic) — but ChatGPT
  also proved relatively ineffective against transformers
- Compare to Krishna et al. (2023): DIPPER caused 80%+ ASR; our ChatGPT prompt only 1.6%–6.8%
  Explanation: DIPPER is specifically trained to evade detection; ChatGPT rewrite is general purpose
- RoBERTa and Hello-SimpleAI most robust: likely because they rely on contextual distributional
  patterns that are not easily disrupted by surface rewriting
- DistilBERT slightly more vulnerable: smaller model, fewer parameters, less robust feature space

6.2 Why Logistic Regression is Uniquely Vulnerable (~400 words)
- TF-IDF bigrams are the exact feature space that paraphrase attacks disrupt
- Any lexical substitution directly attacks LR's classification signal
- Yet LR achieves high F1 on clean data (0.9524) — data distribution makes detection "easy"
- The clean-vs-adversarial gap for LR is much larger than for transformers:
  LR clean F1 0.9524 vs LR ChatGPT recall 0.606 = dramatic drop
- Contrast: RoBERTa clean F1 0.9913 vs ChatGPT recall 0.984 = almost no drop
- Lesson: surface features are brittle; contextual representations are robust

6.3 Cross-Dataset Generalisation Failure (~400 words)
- All models fail to generalise well to M4
- Possible causes:
  (a) M4 uses different generators (GPT-4, Cohere, BLOOMz) with different stylistic signatures
  (b) M4 has different domains (broader than HC3's QA focus)
  (c) Models have overfit to HC3's distributional features (not general "AI-ness")
- RoBERTa best: 0.7389 M4 F1 — contextual models generalise better than surface models
- Sadasivan et al. (2023) argument supported: detection may be fundamentally brittle
- Implication: in real deployment, detection must be continuously updated as generators evolve

6.4 Human vs Machine Detection (~400 words)
- Humans perform at chance overall (41.8%) — mechanically worse than all automated models
- Rewritten AI (30.7%): humans are actively deceived (below-chance performance)
- This is the strongest practical finding: paraphrasing makes AI text undetectable to humans
- Supports the case for automated detection — human review alone is insufficient
- Connects to Gehrmann et al. (2019) GLTR: humans need decision support tools

6.5 Hybrid Architecture Analysis (~400 words)
- H1's BiLSTM success: sequential modelling captures writing rhythm / paragraph structure
  QuillBot lexical substitution changes words but not sentence rhythm — BiLSTM captures this
- H3's voting: simple but effective; diversity of three models provides modest robustness gain
- H4's feature fusion: TF-IDF component is the Achilles heel (same weakness as LR against QuillBot)
  The stylometric features add marginal signal — h4_scaler shows they contribute to M4 F1
- H2's TextCNN: local n-gram features seem to add noise rather than signal for document classification
  TextCNN was designed for sentence classification; full-document texts may work differently

6.6 Limitations and Threats to Validity (~300 words)
- Internal: 500 attack samples may underestimate variance; single prompt for ChatGPT attack
- External: English-only; HC3 QA domain only; base-size models only
- Construct: ASR measures evasion rate, not text quality or semantic preservation post-attack
- 15 participants in human study — insufficient power for strong statistical claims
  (treat as pilot/exploratory evidence)
- DIPPER excluded — most powerful known adversarial paraphraser not tested

---

### CHAPTER 7: CONCLUSION — 1,000 words

7.1 Summary of Contributions (~300 words)
- Largest controlled comparison of detectors × attacks in the MSc context
- Novel hybrid architectures (H1–H4) with H1 achieving lowest QuillBot ASR
- Human evaluation confirming paraphrase degrades detection below chance
- Open-source Streamlit platform deployed at msc-ai-detector.streamlit.app
- Empirical evidence that transformer detectors are substantially more robust than classical ML

7.2 Answers to Research Sub-questions (~400 words)
For each of the 5 sub-questions, write one direct answer paragraph:
1. Clean performance: all transformers >0.98 F1 (answered yes)
2. Most effective attack: QuillBot-style stochastic T5 (13.4%–19.0% for transformers)
3. Generalisation: all models show large M4 F1 drop; supervised detection does not generalise
4. Human vs machine: humans are worse on all conditions; paraphrase degrades humans to below chance
5. Hybrid robustness: H1-BiLSTM achieves lowest QuillBot ASR (4.8%), offering genuine improvement

7.3 Future Work (~300 words)
- Include DIPPER paraphrase attack (requires A100 GPU or multi-GPU node)
- Test adversarially optimised prompts for ChatGPT rewriting
- Extend to multilingual detection
- Larger human study (N > 100) with pre-registration
- Explore watermarking-based detection as a complementary approach
- Test on more recent generators (GPT-4, Claude 3, Gemini)
- Investigate adversarial training as a defence strategy

---

## 10. APPENDICES

Appendix A — HC3 Dataset Samples
5 human-written + 5 original AI + 5 QuillBot-rewritten AI examples (before and after)
Table format: Original | Paraphrased version

Appendix B — Human Evaluation Study
- Google Form screenshot (Screenshot J)
- Full 15 questions listed
- Raw accuracy per question (from fig12_per_question_accuracy.png)
- Google Sheets response data (Screenshot K)

Appendix C — Full Per-Question Human Evaluation Results Table
All 15 questions: condition, correct responses, accuracy (%)

Appendix D — GitHub Repository
Link: github.com/[your-username]/MSc-AI-Detection
Screenshot: Repository main page (Screenshot L)
Key files listed

Appendix E — Streamlit Demo Platform
URL: msc-ai-detector.streamlit.app
Screenshots A–F from deployed app

Appendix F — Training Evidence
WandB training curves Screenshots G, H, I
Brief table: model, epochs, final train loss, final val F1

---

## 11. ALL HARVARD REFERENCES (COMPLETE LIST)

Brown, T.B. et al. (2020) 'Language models are few-shot learners', Advances in Neural Information
Processing Systems, 33, pp. 1877–1901. arXiv:2005.14165.

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019) 'BERT: Pre-training of deep
bidirectional transformers for language understanding', Proceedings of NAACL-HLT 2019,
pp. 4171–4186. doi:10.18653/v1/N19-1423.

Gehrmann, S., Strobelt, H. and Rush, A.M. (2019) 'GLTR: Statistical detection and
visualization of generated text', Proceedings of ACL 2019: System Demonstrations,
pp. 111–116. doi:10.18653/v1/P19-3019.

Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue, J. and Wu, Y. (2023)
'How close is ChatGPT to human experts? Comparison corpus, evaluation, and detection',
arXiv:2301.07597.

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I. and Goldstein, T. (2023)
'A watermark for large language models', Proceedings of the 40th International Conference
on Machine Learning, pp. 17061–17084. arXiv:2301.10226.

Krishna, K., Song, Y., Karpinska, M., Wieting, J. and Iyyer, M. (2023) 'Paraphrasing evades
detectors of AI-generated text, but retrieval is an effective defense', Advances in Neural
Information Processing Systems, 36. arXiv:2303.13408.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer,
L. and Stoyanov, V. (2019) 'RoBERTa: A robustly optimized BERT pretraining approach',
arXiv:1907.11692.

Mitchell, E., Lee, Y., Khazatsky, A., Manning, C.D. and Finn, C. (2023) 'DetectGPT: Zero-shot
machine-generated text detection using probability curvature', Proceedings of the 40th
International Conference on Machine Learning. arXiv:2301.11305.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. and Sutskever, I. (2019) 'Language models
are unsupervised multitask learners', OpenAI Blog, 1(8), p. 9.

Sadasivan, V.S., Kumar, A., Balasubramanian, S., Wang, W. and Feizi, S. (2023) 'Can AI-generated
text really be detected?', arXiv:2303.11156.

Sanh, V., Debut, L., Chaumond, J. and Wolf, T. (2019) 'DistilBERT, a distilled version of BERT:
Smaller, faster, cheaper and lighter', arXiv:1910.01108.

Sokolova, M. and Lapalme, G. (2009) 'A systematic analysis of performance measures for
classification tasks', Information Processing and Management, 45(4), pp. 427–437.
doi:10.1016/j.ipm.2009.03.002.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. and
Polosukhin, I. (2017) 'Attention is all you need', Advances in Neural Information Processing
Systems, 30. arXiv:1706.03762.

Wang, Y., Mansurov, J., Ivanov, P., Su, J., Shelmanov, A., Tsvigun, A., Whitehouse, C.,
Chowdhury, S.A., Wieting, J., Arnold, T. and Nakov, P. (2023) 'M4: Multi-generator,
multi-domain, and multi-lingual black-box machine-generated text detection', arXiv:2305.14902.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T.,
Louf, R., Funtowicz, M. et al. (2020) 'Transformers: State-of-the-art natural language
processing', Proceedings of the 2020 Conference on EMNLP: System Demonstrations,
pp. 38–45. doi:10.18653/v1/2020.emnlp-demos.6.

---

## 12. KEY FACTS TO NEVER GET WRONG

- Attack 1 model: tuner007/pegasus_paraphrase — call it "Pegasus" attack
- Attack 2 model: Vamsi/T5_Paraphrase_Paws with stochastic sampling — call it "QuillBot-style T5"
- Attack 3 model: OpenAI gpt-3.5-turbo — call it "ChatGPT semantic rewriting"
- DIPPER: CITED IN LITERATURE ONLY. Never appears in results. 11B params, excluded on hardware.
- Hello-SimpleAI: NOT retrained. Used as external pre-trained detector.
- H1 and H2: research-only (Colab). No live inference in the app.
- H3 and H4: live inference in the deployed Streamlit app.
- Human study texts: T5-QuillBot (Attack 2) samples used for "Rewritten AI" condition.
- N=15 participants, 15 questions each = 225 total responses.
- ASR = 1 - Recall on AI class. Lower ASR = more robust.
- HC3 test set: 1,773 samples. Attack evaluation: 500 AI-only samples each.
- M4 test evaluation: 2,000 samples.
- All transformer clean ROC-AUC: 0.9997–0.9999 (near perfect)
- H1 QuillBot ASR 4.8% = lowest of ALL 9 models (key finding, emphasise this)
- Streamlit app URL: msc-ai-detector.streamlit.app
- GitHub namespace for uploaded models: ahm1129 on HuggingFace Hub
- University: University of the West of Scotland (UWS)
- Supervisor: Dr Tahir Mahmood
- Programme: MSc Artificial Intelligence
- Banner ID: B00409227

---

## END OF DISSERTATION BRIEF
## When starting a chapter, tell Claude: "Please write [Chapter X / Section X.Y] using the brief I provided."
## Claude will use every number, model name, and reference listed here — no invention needed.
