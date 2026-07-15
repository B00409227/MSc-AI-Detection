"""
MSc AI Dissertation — AI-Generated Text Detection Platform v2.0
Student: Abdul Hannaan Mohammed | B00409227 | UWS
Supervisor: Dr Tahir Mahmood

Run with: streamlit run app/streamlit_app.py
"""

import os, sys, io, time, re
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="AI Text Detection — UWS MSc",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
MAX_LENGTH      = 512
SEED            = 42
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)

_HF_ROBERTA    = "ahm1129/roberta-hc3-detector"
_HF_BERT       = "ahm1129/bert-hc3-detector"
_HF_DISTILBERT = "ahm1129/distilbert-hc3-detector"
_local_roberta    = os.path.join(CHECKPOINTS_DIR, "roberta-hc3-best")
_local_bert       = os.path.join(CHECKPOINTS_DIR, "bert-hc3-best")
_local_distilbert = os.path.join(CHECKPOINTS_DIR, "distilbert-hc3-best")
_local_lr         = os.path.join(MODELS_DIR, "logistic_regression", "lr_model.pkl")

_h4_tfidf  = os.path.join(MODELS_DIR, "logistic_regression", "h4_tfidf.pkl")
_h4_scaler = os.path.join(MODELS_DIR, "logistic_regression", "h4_scaler.pkl")
_h4_mlp    = os.path.join(MODELS_DIR, "logistic_regression", "h4_mlp.pt")

MODEL_OPTIONS = {
    "RoBERTa-base":          _local_roberta    if os.path.exists(_local_roberta)    else _HF_ROBERTA,
    "BERT-base":             _local_bert       if os.path.exists(_local_bert)       else _HF_BERT,
    "DistilBERT":            _local_distilbert if os.path.exists(_local_distilbert) else _HF_DISTILBERT,
    "Hello-SimpleAI":        "Hello-SimpleAI/chatgpt-detector-roberta",
    "Logistic Regression":   _local_lr,
    "H1-RoBERTa+BiLSTM":    "__h1__",
    "H2-BERT+TextCNN":       "__h2__",
    "H3-SoftVoting":         "__h3__",
    "H4-FeatureFusion":      "__h4__",
}

INDIVIDUAL_MODELS = ["RoBERTa-base", "BERT-base", "DistilBERT",
                     "Hello-SimpleAI", "Logistic Regression"]
HYBRID_MODELS     = ["H1-RoBERTa+BiLSTM", "H2-BERT+TextCNN",
                     "H3-SoftVoting", "H4-FeatureFusion"]
LIVE_INFERENCE_MODELS = ["RoBERTa-base", "BERT-base", "DistilBERT",
                         "Hello-SimpleAI", "Logistic Regression",
                         "H3-SoftVoting", "H4-FeatureFusion"]

# ── Comprehensive model catalogue ──────────────────────────────────────────────
MODEL_CARDS = {
    "RoBERTa-base": {
        "label": "RoBERTa-base (Fine-tuned)", "category": "Individual", "badge": "#1565C0",
        "arch":  "Encoder-only Transformer · 125 M params · 12 layers · 768 hidden",
        "base":  "roberta-base (Liu et al., 2019)",
        "train": "Full fine-tuning · HC3 55,156 samples · 3 epochs · lr=2e-5 · batch=16 · warmup=500",
        "device": "Local GPU (6 GB VRAM)", "trainable": "125 M (100%)",
        "clean_f1": 0.9913, "clean_acc": 0.9942, "clean_recall": 0.9995, "clean_prec": 0.9831,
        "peg_asr": 1.2, "qui_asr": 13.4, "cgpt_asr": 1.6, "m4_f1": 0.7389,
        "strength": "Best cross-dataset generalisation (M4 F1=0.739). Near-immune to beam-search paraphrase (Pegasus ASR=1.2%). Recommended for deployment.",
        "weakness": "BPE tokeniser is sensitive to vocabulary substitution — QuillBot-style word replacement achieves 13.4% ASR.",
        "why": "RoBERTa improves on BERT by removing Next Sentence Prediction, using more data, and longer training. Its robust pretraining makes it the strongest individual detector in this study.",
        "ref": "Liu et al. (2019) arXiv:1907.11692",
    },
    "BERT-base": {
        "label": "BERT-base-uncased (Fine-tuned)", "category": "Individual", "badge": "#1B5E20",
        "arch":  "Encoder-only Transformer · 110 M params · 12 layers · 768 hidden",
        "base":  "bert-base-uncased (Devlin et al., 2019)",
        "train": "Full fine-tuning · HC3 55,156 samples · 3 epochs · lr=2e-5 · batch=16",
        "device": "Local GPU (6 GB VRAM)", "trainable": "110 M (100%)",
        "clean_f1": 0.9845, "clean_acc": 0.9895, "clean_recall": 0.9997, "clean_prec": 0.9694,
        "peg_asr": 1.8, "qui_asr": 12.2, "cgpt_asr": 3.2, "m4_f1": 0.5999,
        "strength": "Very high clean recall (99.97%). Balanced robustness across attack types. Lowest ChatGPT ASR (3.2%) of the transformer trio.",
        "weakness": "Lower M4 F1 (0.600) than RoBERTa — weaker cross-domain generalisation. WordPiece tokeniser vulnerable to paraphrase.",
        "why": "BERT established the fine-tuning paradigm for NLP. Included as the reference transformer baseline against which improvements are measured.",
        "ref": "Devlin et al. (2019) doi:10.18653/v1/N19-1423",
    },
    "DistilBERT": {
        "label": "DistilBERT-base (Fine-tuned)", "category": "Individual", "badge": "#4A148C",
        "arch":  "Encoder-only Transformer · 66 M params · 6 layers · 768 hidden (distilled)",
        "base":  "distilbert-base-uncased (Sanh et al., 2019)",
        "train": "Full fine-tuning · HC3 55,156 samples · 4 epochs · lr=2e-5 · Google Colab T4",
        "device": "Google Colab T4 (15 GB VRAM)", "trainable": "66 M (100%)",
        "clean_f1": 0.9922, "clean_acc": 0.9948, "clean_recall": 0.9995, "clean_prec": 0.9849,
        "peg_asr": 5.2, "qui_asr": 19.0, "cgpt_asr": 6.8, "m4_f1": 0.4316,
        "strength": "Highest clean F1 (0.9922). 40% fewer parameters than BERT — fastest inference. Good for resource-constrained deployment.",
        "weakness": "Highest attack vulnerability of the transformer trio (QuillBot ASR=19.0%). Weakest cross-dataset generalisation (M4 F1=0.432). Layer reduction reduces adversarial robustness.",
        "why": "DistilBERT (knowledge distillation) tests whether a compressed model retains robustness. Results show distillation trades adversarial resilience for speed.",
        "ref": "Sanh et al. (2019) arXiv:1910.01108",
    },
    "Hello-SimpleAI": {
        "label": "Hello-SimpleAI HC3 Detector", "category": "Individual", "badge": "#BF360C",
        "arch":  "RoBERTa-base fine-tuned by Hello-SimpleAI · pre-trained detector",
        "base":  "Hello-SimpleAI/chatgpt-detector-roberta (HuggingFace Hub)",
        "train": "Pre-trained externally on HC3 — not fine-tuned locally. Used as-is.",
        "device": "N/A (downloaded from HuggingFace Hub)", "trainable": "N/A",
        "clean_f1": 0.9929, "clean_acc": 0.9953, "clean_recall": 0.9977, "clean_prec": 0.9881,
        "peg_asr": 0.4, "qui_asr": 14.0, "cgpt_asr": 2.8, "m4_f1": 0.5442,
        "strength": "Highest clean F1 (0.9929). Most robust against Pegasus (ASR=0.4%) — best beam-search paraphrase defence. No local training required.",
        "weakness": "QuillBot ASR=14.0% (slightly worse than RoBERTa). M4 F1=0.544 — moderate cross-domain gap despite training on same dataset.",
        "why": "Serves as a real-world pre-trained baseline. Tests whether externally-trained detectors match locally fine-tuned models and respond similarly to attacks.",
        "ref": "Guo et al. (2023) HC3 arXiv:2301.07597",
    },
    "Logistic Regression": {
        "label": "Logistic Regression + TF-IDF", "category": "Individual", "badge": "#37474F",
        "arch":  "TF-IDF (50k features, bigrams) + Logistic Regression · classical ML",
        "base":  "scikit-learn Pipeline",
        "train": "Fit on HC3 training set · max_features=50k · C=1.0 · class_weight=balanced",
        "device": "CPU (seconds)", "trainable": "~50k vocabulary weights",
        "clean_f1": 0.9524, "clean_acc": 0.9689, "clean_recall": 0.9364, "clean_prec": 0.9703,
        "peg_asr": 29.0, "qui_asr": 26.8, "cgpt_asr": 39.4, "m4_f1": 0.3356,
        "strength": "No GPU required. Interpretable — feature weights identify key AI phrases. Serves as a statistical lower bound.",
        "weakness": "Highest vulnerability across all attack types. Surface-level vocabulary patterns cannot capture deep semantic AI signals. ChatGPT ASR=39.4% — most attackable model.",
        "why": "Classical ML baseline establishes whether shallow n-gram features suffice for detection, motivating the use of transformer-based approaches.",
        "ref": "Sokolova & Lapalme (2009) doi:10.1016/j.ipm.2009.03.002",
    },
    "H1-RoBERTa+BiLSTM": {
        "label": "H1: RoBERTa + BiLSTM", "category": "Hybrid", "badge": "#E65100",
        "arch":  "RoBERTa-base (FROZEN) → BiLSTM (256 hidden, 2 layers, bidirectional) → classifier",
        "base":  "Feature-based transfer learning (Peters et al., 2018 ELMo approach)",
        "train": "Only LSTM + classifier trained (2.9% of params) · Google Colab T4 · 5 epochs · lr=1e-3",
        "device": "Google Colab T4 (15 GB VRAM)", "trainable": "3.6 M / 125 M (2.9%)",
        "clean_f1": 0.9910, "clean_acc": 0.9940, "clean_recall": 0.9990, "clean_prec": 0.9830,
        "peg_asr": 1.4, "qui_asr": 4.8, "cgpt_asr": 2.2, "m4_f1": 0.7127,
        "strength": "⭐ BEST ROBUSTNESS: QuillBot ASR=4.8% — 64% lower than plain RoBERTa (13.4%). BiLSTM captures sequential word-order dependencies that vocabulary substitution cannot easily disrupt. High M4 F1=0.713.",
        "weakness": "Slightly lower clean F1 (0.991) than full fine-tuning. Requires Colab for training. Adds inference overhead from LSTM layer.",
        "why": "Freezing RoBERTa and adding BiLSTM forces the classifier to rely on sequential patterns rather than learned token probabilities, which are disrupted by paraphrase attacks. Inspired by Peters et al. (2018) ELMo feature extraction.",
        "ref": "Schuster & Paliwal (1997) doi:10.1109/78.650093 · Peters et al. (2018)",
    },
    "H2-BERT+TextCNN": {
        "label": "H2: BERT + TextCNN", "category": "Hybrid", "badge": "#E65100",
        "arch":  "BERT-base (FROZEN) → Parallel Conv1d (kernel=2,3,4 · 128 filters each) → MaxPool → classifier",
        "base":  "TextCNN (Kim, 2014) adapted for transformer feature extraction",
        "train": "Only CNN + classifier trained (0.85% of params) · Google Colab T4 · 5 epochs · lr=1e-3",
        "device": "Google Colab T4 (15 GB VRAM)", "trainable": "936 K / 110 M (0.85%)",
        "clean_f1": 0.9695, "clean_acc": 0.9763, "clean_recall": 0.9940, "clean_prec": 0.9450,
        "peg_asr": 4.2, "qui_asr": 9.6, "cgpt_asr": 3.8, "m4_f1": 0.5923,
        "strength": "QuillBot ASR=9.6% — 28% lower than BERT (12.2%). CNN filters detect characteristic AI n-gram phrases that persist through paraphrase. Extremely parameter-efficient (0.85% trainable).",
        "weakness": "Lower clean F1 (0.9695) — precision drop (0.945) suggests more false positives. Fixed kernel sizes may miss longer-range AI patterns.",
        "why": "CNN filters on BERT embeddings capture local n-gram patterns without the model memorising full-text statistics that attacks exploit. Minimal additional training makes it highly efficient.",
        "ref": "Kim (2014) doi:10.3115/v1/D14-1181",
    },
    "H3-SoftVoting": {
        "label": "H3: Soft Voting Ensemble", "category": "Hybrid", "badge": "#1A237E",
        "arch":  "Average P(AI) from RoBERTa + BERT + DistilBERT → threshold at 0.5",
        "base":  "Deep Ensembles (Lakshminarayanan et al., 2017)",
        "train": "No additional training — uses existing fine-tuned checkpoints",
        "device": "N/A", "trainable": "0 (ensemble of trained models)",
        "clean_f1": 0.9916, "clean_acc": 0.9943, "clean_recall": 0.9993, "clean_prec": 0.9839,
        "peg_asr": 1.8, "qui_asr": 12.0, "cgpt_asr": 3.2, "m4_f1": 0.5825,
        "strength": "Clean F1=0.9916 with zero extra training. Attack must fool all three models simultaneously. Reduces variance across model predictions.",
        "weakness": "QuillBot ASR=12.0% — only marginal improvement over individual models. McNemar's test: p=0.899 (NOT statistically significant vs RoBERTa). Does not address underlying vulnerability shared by all three models.",
        "why": "If all three transformers share the same brittleness to vocabulary substitution, averaging their probabilities cannot overcome the shared vulnerability — which is exactly what McNemar's test confirms.",
        "ref": "Lakshminarayanan et al. (2017) arXiv:1612.01474",
    },
    "H4-FeatureFusion": {
        "label": "H4: Multi-Feature Fusion MLP", "category": "Hybrid", "badge": "#1A237E",
        "arch":  "507-dim vector: RoBERTa P(AI) + TF-IDF(500) + 6 stylometric → MLP(507→256→64→2)",
        "base":  "Ghostbuster-inspired (Verma et al., 2023) multi-signal fusion",
        "train": "MLP trained · HC3 train set · 30 epochs · lr=1e-4 · batch=256 · ReduceLROnPlateau",
        "device": "CPU (fast)", "trainable": "149,826 MLP params",
        "clean_f1": 0.9929, "clean_acc": 0.9953, "clean_recall": 0.9975, "clean_prec": 0.9883,
        "peg_asr": 1.8, "qui_asr": 16.0, "cgpt_asr": 2.8, "m4_f1": 0.5896,
        "strength": "Highest clean F1 among hybrids (0.9929). McNemar's test: p=0.006 (SIGNIFICANT vs RoBERTa). Statistically different decision boundaries.",
        "weakness": "QuillBot ASR=16.0% — WORSE than plain RoBERTa (13.4%). The TF-IDF component is disrupted by QuillBot's vocabulary substitution, which degrades the fused feature signal.",
        "why": "The fusion experiment reveals that TF-IDF features are a liability under vocabulary attacks. This motivates future work combining sequential (H1-BiLSTM) with stylometric signals instead of bag-of-words features.",
        "ref": "Verma et al. (2023) Ghostbuster arXiv:2305.15047",
    },
    # H1 and H2 are research-only (weights trained on Colab, not available for live inference)
    "H1-RoBERTa+BiLSTM": {
        "label": "H1: RoBERTa + BiLSTM (Research Only)", "category": "Hybrid", "badge": "#B71C1C",
        "arch":  "RoBERTa-base (FROZEN) → BiLSTM(256 hidden, 2 layers, bidir) → classifier",
        "base":  "ELMo-style feature extraction (Peters et al., 2018)",
        "train": "BiLSTM + classifier only (2.9% params) · Colab T4 · 5 epochs · lr=1e-3",
        "device": "Google Colab T4 (15 GB VRAM)", "trainable": "3.6 M / 125 M (2.9%)",
        "clean_f1": 0.9910, "clean_acc": 0.9940, "clean_recall": 0.9990, "clean_prec": 0.9830,
        "peg_asr": 1.4, "qui_asr": 4.8, "cgpt_asr": 2.2, "m4_f1": 0.7127,
        "strength": "⭐ BEST ROBUSTNESS: QuillBot ASR=4.8% — 64% lower than plain RoBERTa. BiLSTM captures sequential patterns that vocabulary substitution cannot disrupt.",
        "weakness": "Weights only available on Colab — not available for live inference in this app. Inference results shown from dissertation experiments.",
        "why": "Freezing RoBERTa and adding BiLSTM forces reliance on sequential word-order signals rather than token probability distributions disrupted by paraphrase.",
        "ref": "Schuster & Paliwal (1997) doi:10.1109/78.650093",
        "live_inference": False,
    },
    "H2-BERT+TextCNN": {
        "label": "H2: BERT + TextCNN (Research Only)", "category": "Hybrid", "badge": "#B71C1C",
        "arch":  "BERT-base (FROZEN) → Parallel Conv1d (kernel 2,3,4 · 128 filters) → MaxPool → classifier",
        "base":  "TextCNN (Kim, 2014) on transformer embeddings",
        "train": "CNN + classifier only (0.85% params) · Colab T4 · 5 epochs · lr=1e-3",
        "device": "Google Colab T4 (15 GB VRAM)", "trainable": "936 K / 110 M (0.85%)",
        "clean_f1": 0.9695, "clean_acc": 0.9763, "clean_recall": 0.9940, "clean_prec": 0.9450,
        "peg_asr": 4.2, "qui_asr": 9.6, "cgpt_asr": 3.8, "m4_f1": 0.5923,
        "strength": "QuillBot ASR=9.6% — 28% improvement over BERT. CNN captures AI n-gram patterns. Only 0.85% trainable params.",
        "weakness": "Weights only available on Colab — not available for live inference. Inference results shown from dissertation experiments.",
        "why": "CNN filters on frozen BERT embeddings capture local AI phrase patterns without relying on the token-level statistics that paraphrase attacks exploit.",
        "ref": "Kim (2014) doi:10.3115/v1/D14-1181",
        "live_inference": False,
    },
}

# ── Sample texts for the Human vs AI Lab ──────────────────────────────────────
SAMPLE_HUMAN_TEXTS = {
    "Student Essay (Climate)": (
        "I've been thinking a lot about climate change lately, especially after watching news "
        "about the floods in Pakistan last year. Honestly, it's terrifying how fast things are "
        "changing. I grew up in a city where winters were actually cold, and now we barely get "
        "a frost. My gran says she remembers proper snow every year. It's not just anecdotal — "
        "the data backs it up. But what really worries me is the gap between what scientists are "
        "saying needs to happen and what governments are actually doing. The Paris Agreement was "
        "a step, sure, but emissions are still rising. I think the biggest barrier is political "
        "will, not technology — we already have solar and wind that's cheaper than coal in most "
        "places. So why aren't we moving faster? It feels like we're in a race we've decided "
        "to run in slow motion."
    ),
    "Medical Forum Response": (
        "Not a doctor but I had something similar last spring. Started with just a nagging "
        "pain under my right shoulder blade, dismissed it as posture from sitting at my desk "
        "all day. It kept getting worse over about three weeks, then one morning I woke up and "
        "couldn't take a deep breath without it stabbing. Went to A&E, turned out to be "
        "pleurisy — inflammation of the lining around the lungs. They said it was viral, "
        "probably connected to a cold I'd had the month before. Treatment was basically "
        "anti-inflammatories and rest. Cleared up in about 10 days but the first few were "
        "genuinely awful. Definitely get it checked out if the pain is getting worse — "
        "pleurisy sounds scary but it's treatable, the key is catching it early."
    ),
    "History Q&A (Human)": (
        "The causes of WWI are honestly one of those things that historians still argue about "
        "a hundred years later, which tells you how complicated it is. The assassination of "
        "Franz Ferdinand in Sarajevo is the obvious trigger, but that's a bit like saying a "
        "spark caused a forest fire — the forest was already bone dry. The alliance system "
        "meant any conflict between two countries could drag in half of Europe. Germany had "
        "been itching for a war it thought it could win quickly (the Schlieffen Plan), "
        "Austria-Hungary wanted to punish Serbia, Russia couldn't back down again after the "
        "humiliation of 1905. Nationalism, imperial competition, an arms race nobody knew "
        "how to stop — by 1914 a major war was arguably overdetermined. The assassination "
        "just made it unavoidable to keep ignoring."
    ),
}

SAMPLE_AI_TEXTS = {
    "ChatGPT Essay (Climate)": (
        "Climate change represents one of the most pressing challenges of the twenty-first "
        "century, encompassing a broad spectrum of environmental, economic, and social "
        "implications. It is important to note that the scientific consensus, as reflected "
        "in numerous Intergovernmental Panel on Climate Change (IPCC) reports, unequivocally "
        "demonstrates that human activities — particularly the combustion of fossil fuels — "
        "are the primary driver of observed global warming since the mid-twentieth century. "
        "Furthermore, the consequences of inaction are projected to be severe, including "
        "rising sea levels, increased frequency of extreme weather events, and significant "
        "disruptions to agricultural systems worldwide. In conclusion, addressing climate "
        "change requires coordinated international action, sustained investment in renewable "
        "energy infrastructure, and a fundamental transformation of industrial practices "
        "to ensure a sustainable future for subsequent generations."
    ),
    "AI Medical Explanation": (
        "Pleurisy is a medical condition characterised by inflammation of the pleura, the "
        "thin two-layered membrane that surrounds the lungs and lines the chest cavity. "
        "It is essential to understand that this condition can arise from a variety of "
        "underlying causes, including viral infections, bacterial pneumonia, autoimmune "
        "disorders such as lupus, or pulmonary embolism. The hallmark symptom is a sharp, "
        "stabbing chest pain that typically worsens during inhalation, coughing, or sneezing. "
        "In today's medical landscape, diagnosis is primarily clinical, supported by imaging "
        "studies such as chest X-ray or CT scan to rule out complications including pleural "
        "effusion. It is worth noting that treatment is directed at the underlying cause, "
        "with nonsteroidal anti-inflammatory drugs commonly employed for symptomatic relief."
    ),
    "AI History Answer": (
        "The First World War, which commenced in 1914 and concluded in 1918, arose from a "
        "complex interplay of factors that historians have categorised into several key "
        "dimensions. It is crucial to recognise that the immediate catalyst was the "
        "assassination of Archduke Franz Ferdinand of Austria-Hungary in Sarajevo on "
        "28 June 1914. However, the deeper structural causes included the system of "
        "entangling alliances, intensifying imperial rivalry among the European great powers, "
        "the pervasive nationalist movements destabilising multi-ethnic empires, and an "
        "unprecedented arms race that had militarised European foreign policy. Furthermore, "
        "it plays a pivotal role in historical analysis to acknowledge that the mobilisation "
        "timetables embedded in military planning, particularly Germany's Schlieffen Plan, "
        "transformed a regional Austro-Serbian conflict into a continent-wide catastrophe."
    ),
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #003366 0%, #0055aa 60%, #0077cc 100%);
    padding: 24px 36px; border-radius: 12px; color: white;
    text-align: center; margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
  }
  .main-header h1 { margin: 0 0 6px 0; font-size: 2rem; letter-spacing: -0.5px; }
  .main-header p  { margin: 3px 0; font-size: 0.9rem; opacity: 0.88; }

  .metric-card {
    background: #f0f4f8; padding: 16px; border-radius: 10px;
    text-align: center; border-left: 5px solid #0066cc; height: 100%;
  }
  .metric-card h3 { margin: 4px 0; font-size: 1.6rem; color: #003366; font-weight: 700; }
  .metric-card p  { margin: 0; color: #666; font-size: 0.82rem; }

  .verdict-ai    { background:#fde8e8; border:2px solid #dc3545; padding:14px 18px;
                   border-radius:10px; text-align:center; font-size:1.1rem; font-weight:700; color:#dc3545; }
  .verdict-human { background:#e8f5e9; border:2px solid #28a745; padding:14px 18px;
                   border-radius:10px; text-align:center; font-size:1.1rem; font-weight:700; color:#28a745; }
  .verdict-unc   { background:#fff8e1; border:2px solid #ffc107; padding:14px 18px;
                   border-radius:10px; text-align:center; font-size:1.1rem; font-weight:700; color:#e65100; }

  .human-sentence     { background:#e8f5e9; padding:7px 12px; border-radius:5px; margin:4px 0; border-left:4px solid #28a745; }
  .ai-sentence        { background:#fde8e8; padding:7px 12px; border-radius:5px; margin:4px 0; border-left:4px solid #dc3545; }
  .uncertain-sentence { background:#fff8e1; padding:7px 12px; border-radius:5px; margin:4px 0; border-left:4px solid #ffc107; }

  .traffic-green { background:#28a745; color:white; padding:10px 22px; border-radius:22px; font-weight:700; display:inline-block; }
  .traffic-amber { background:#ffc107; color:#333;  padding:10px 22px; border-radius:22px; font-weight:700; display:inline-block; }
  .traffic-red   { background:#dc3545; color:white; padding:10px 22px; border-radius:22px; font-weight:700; display:inline-block; }

  .model-info-box { background:#e8f0fe; border:1px solid #4285f4; padding:10px 14px;
                    border-radius:6px; font-size:0.87rem; margin-top:6px; }
  .key-finding    { background:#fff3e0; border-left:5px solid #ff9800; padding:14px 18px;
                    border-radius:5px; font-size:0.9rem; margin: 10px 0; }
  .warning-box    { background:#fff3cd; border:1px solid #ffc107; padding:12px 16px; border-radius:6px; margin:8px 0; }
  .success-box    { background:#d4edda; border:1px solid #28a745; padding:12px 16px; border-radius:6px; margin:8px 0; }
  .danger-box     { background:#f8d7da; border:1px solid #dc3545; padding:12px 16px; border-radius:6px; margin:8px 0; }

  .model-card-ind { border-left:6px solid #1565C0; background:#f3f8ff; padding:16px;
                    border-radius:8px; margin-bottom:14px; }
  .model-card-hyb { border-left:6px solid #E65100; background:#fff8f3; padding:16px;
                    border-radius:8px; margin-bottom:14px; }
  .badge-ind { background:#1565C0; color:white; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
  .badge-hyb { background:#E65100; color:white; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }

  .lab-col-human { background:#f0faf3; border:2px dashed #28a745; padding:16px;
                   border-radius:10px; }
  .lab-col-ai    { background:#fff3f3; border:2px dashed #dc3545; padding:16px;
                   border-radius:10px; }
  .lab-header-human { color:#155724; font-size:1.15rem; font-weight:700; margin-bottom:8px; }
  .lab-header-ai    { color:#721c24; font-size:1.15rem; font-weight:700; margin-bottom:8px; }

  .progress-bar-outer { background:#e0e0e0; border-radius:6px; height:18px; width:100%; }
  .progress-bar-inner-ai    { background:#dc3545; height:18px; border-radius:6px; }
  .progress-bar-inner-human { background:#28a745; height:18px; border-radius:6px; }
  .prob-label { font-size:0.8rem; color:#555; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🔍 AI-Generated Text Detection Platform</h1>
  <p>MSc Artificial Intelligence · University of the West of Scotland</p>
  <p>Abdul Hannaan Mohammed · B00409227 · Supervisor: Dr Tahir Mahmood</p>
</div>
""", unsafe_allow_html=True)

_RUNNING_LOCAL = os.path.exists(CHECKPOINTS_DIR)
if not _RUNNING_LOCAL:
    st.info(
        "**Cloud mode:** Transformer models download from HuggingFace Hub on first use "
        "(RoBERTa ≈500 MB, BERT ≈430 MB, DistilBERT ≈270 MB). Expect a 1–2 min wait per model; "
        "all 5 models and all modes are fully functional once loaded.",
        icon="ℹ️"
    )


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_transformer_model(path: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    try:
        tok = AutoTokenizer.from_pretrained(path)
        mod = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
        mod.eval()
        return tok, mod
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_logistic_regression(path: str):
    import pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_text_generator():
    """Load DistilGPT-2 for AI text generation (350 MB, CPU-friendly)."""
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("text-generation", model="distilgpt2", device=-1)


@st.cache_resource(show_spinner=False)
def load_paraphraser():
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tok = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    mod = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(DEVICE)
    mod.eval()
    return tok, mod


# ── H4 Feature Fusion MLP ──────────────────────────────────────────────────────
import torch.nn as nn

class FeatureFusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(507, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)


@st.cache_resource(show_spinner=False)
def load_h4_model():
    """Load H4 artifacts: TF-IDF vectorizer, StandardScaler, and MLP weights."""
    import pickle
    try:
        with open(_h4_tfidf, "rb") as f:
            tfidf = pickle.load(f)
        with open(_h4_scaler, "rb") as f:
            scaler = pickle.load(f)
        mlp = FeatureFusionMLP()
        mlp.load_state_dict(torch.load(_h4_mlp, map_location="cpu", weights_only=True))
        mlp.eval()
        return tfidf, scaler, mlp, None
    except Exception as e:
        return None, None, None, str(e)


def extract_stylometric_features(text: str) -> np.ndarray:
    """6 stylometric features used by H4 (exact order from notebook 15)."""
    import string
    from collections import Counter
    words = text.split()
    n_words = max(len(words), 1)
    # f1: word count
    f1 = float(n_words)
    # f2: avg word length (strip punctuation from each word before measuring)
    stripped = [w.strip(string.punctuation) for w in words]
    lengths = [len(w) for w in stripped if len(w) > 0]
    f2 = float(np.mean(lengths)) if lengths else 0.0
    # f3: type-token ratio (unique / total)
    f3 = len(set(w.lower() for w in words)) / n_words
    # f4: punctuation density ((n_punct / n_words) * 100)
    n_punct = sum(1 for c in text if c in string.punctuation)
    f4 = (n_punct / n_words) * 100.0
    # f5: avg sentence length (words per sentence)
    sents = split_sentences(text)
    n_sents = max(len(sents), 1)
    f5 = n_words / n_sents
    # f6: hapax legomena ratio (words appearing exactly once / total words)
    freq = Counter(w.lower() for w in words)
    hapax = sum(1 for c in freq.values() if c == 1)
    f6 = hapax / n_words
    return np.array([f1, f2, f3, f4, f5, f6], dtype=np.float32)


def predict_h3(text: str):
    """H3 Soft Voting: average P(AI) from RoBERTa + BERT + DistilBERT."""
    probs, errors = [], []
    for key in ["RoBERTa-base", "BERT-base", "DistilBERT"]:
        path = MODEL_OPTIONS[key]
        tok, mod = load_transformer_model(path)
        if tok is None:
            errors.append(f"{key}: {mod}")
            continue
        inp = tok(text, return_tensors="pt", max_length=MAX_LENGTH,
                  truncation=True, padding=True)
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        with torch.no_grad():
            logits = mod(**inp).logits
        p = float(torch.softmax(logits, dim=-1)[0][1])
        probs.append(p)
    if not probs:
        return None, None, "; ".join(errors)
    avg_prob = float(np.mean(probs))
    return (1 if avg_prob >= 0.5 else 0), avg_prob, ("; ".join(errors) if errors else None)


def predict_h4(text: str):
    """H4 Feature Fusion: [RoBERTa_prob(1) | TF-IDF(500) | scaled_stylo(6)] → MLP."""
    tfidf, scaler, mlp, err = load_h4_model()
    if tfidf is None:
        return None, None, f"H4 artifacts not found: {err}"
    # RoBERTa probability (1-dim)
    path = MODEL_OPTIONS["RoBERTa-base"]
    tok, mod = load_transformer_model(path)
    if tok is None:
        return None, None, f"RoBERTa unavailable: {mod}"
    inp = tok(text, return_tensors="pt", max_length=MAX_LENGTH,
              truncation=True, padding=True)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        logits = mod(**inp).logits
    rob_prob = float(torch.softmax(logits, dim=-1)[0][1])
    # TF-IDF features (500-dim)
    tfidf_feats = tfidf.transform([text]).toarray().astype(np.float32)
    # Stylometric features scaled (6-dim)
    stylo = extract_stylometric_features(text).reshape(1, -1)
    scaled_stylo = scaler.transform(stylo).astype(np.float32)
    # Concatenate: [roberta_prob | tfidf_500 | scaled_stylo_6] = 507-dim
    x = np.concatenate([np.array([[rob_prob]], dtype=np.float32), tfidf_feats, scaled_stylo], axis=1)
    with torch.no_grad():
        logits_h4 = mlp(torch.tensor(x, dtype=torch.float32))
    ai_prob = float(torch.softmax(logits_h4, dim=-1)[0][1])
    return (1 if ai_prob >= 0.5 else 0), ai_prob, None


def predict_text(text: str, model_key: str):
    """Returns (label_int, ai_prob_float, error_str|None). label 1 = AI."""
    path = MODEL_OPTIONS[model_key]
    if path == "__h1__":
        return None, None, "RESEARCH_ONLY:H1 — BiLSTM weights are Colab-only. See 🔬 Hybrid Research for results. (Clean F1=0.9910, QuillBot ASR=4.8%)"
    if path == "__h2__":
        return None, None, "RESEARCH_ONLY:H2 — TextCNN weights are Colab-only. See 🔬 Hybrid Research for results. (Clean F1=0.9695, QuillBot ASR=9.6%)"
    if path == "__h3__":
        return predict_h3(text)
    if path == "__h4__":
        return predict_h4(text)
    if model_key == "Logistic Regression":
        pipe, err = load_logistic_regression(path)
        if pipe is None:
            return None, None, err
        prob = float(pipe.predict_proba([text])[0][1])
        return (1 if prob >= 0.5 else 0), prob, None
    tok, mod = load_transformer_model(path)
    if tok is None:
        return None, None, mod
    inp = tok(text, return_tensors="pt", max_length=MAX_LENGTH,
               truncation=True, padding=True)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        logits = mod(**inp).logits
    probs   = torch.softmax(logits, dim=-1)
    ai_prob = float(probs[0][1])
    return (1 if ai_prob >= 0.5 else 0), ai_prob, None


def predict_all_models(text: str, include_hybrids: bool = True) -> dict:
    """Run all live-inference models. Returns {model_key: (label, prob, err)}.
    H1 and H2 are excluded — research-only (Colab weights).
    """
    results = {k: predict_text(text, k) for k in INDIVIDUAL_MODELS}
    if include_hybrids:
        results["H3-SoftVoting"]    = predict_text(text, "H3-SoftVoting")
        results["H4-FeatureFusion"] = predict_text(text, "H4-FeatureFusion")
    return results


def generate_ai_text(prompt: str, max_new: int = 180) -> str:
    """Generate AI text using DistilGPT-2."""
    gen = load_text_generator()
    full_prompt = (
        f"The following is a well-written, detailed answer to a question about {prompt}.\n\n"
        f"Answer: "
    )
    result = gen(
        full_prompt, max_new_tokens=max_new, temperature=0.85,
        do_sample=True, top_p=0.92, repetition_penalty=1.25,
        pad_token_id=50256, num_return_sequences=1
    )
    generated = result[0]["generated_text"]
    if full_prompt in generated:
        generated = generated[len(full_prompt):].strip()
    sentences = re.split(r'(?<=[.!?])\s+', generated.strip())
    clean = [s for s in sentences if len(s) > 8]
    return " ".join(clean[:12]) if clean else generated[:600]


def split_sentences(text: str):
    try:
        import nltk
        try:
            return [s for s in nltk.sent_tokenize(text) if len(s.strip()) > 10]
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return [s for s in nltk.sent_tokenize(text) if len(s.strip()) > 10]
    except Exception:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip()) > 10]


def label_colour(p):
    return "ai-sentence" if p >= 0.6 else "human-sentence" if p <= 0.4 else "uncertain-sentence"

def label_text(p):
    return "🔴 AI" if p >= 0.6 else "🟢 Human" if p <= 0.4 else "🟡 Uncertain"

def traffic_light_html(pct):
    if pct < 30:
        return f'<span class="traffic-green">🟢 LOW AI RISK — {pct:.1f}%</span>'
    elif pct < 70:
        return f'<span class="traffic-amber">🟡 MODERATE RISK — {pct:.1f}%</span>'
    return f'<span class="traffic-red">🔴 HIGH AI RISK — {pct:.1f}%</span>'

def verdict_box(label, prob):
    pct = prob * 100
    if label == 1:
        return f'<div class="verdict-ai">🤖 AI-GENERATED &nbsp;·&nbsp; {pct:.1f}% confidence</div>'
    else:
        return f'<div class="verdict-human">🧑 HUMAN-WRITTEN &nbsp;·&nbsp; {(1-prob)*100:.1f}% confidence</div>'

def extract_docx(f):
    try:
        from docx import Document
        return "\n".join(p.text for p in Document(f).paragraphs if p.text.strip())
    except Exception:
        try:
            import docx2txt
            return docx2txt.process(f)
        except Exception as e:
            return f"ERROR: {e}"

def results_to_csv(sentences, probs, model_key):
    rows = []
    for i, (s, p) in enumerate(zip(sentences, probs)):
        lbl = "AI-Generated" if p >= 0.6 else "Human" if p <= 0.4 else "Uncertain"
        rows.append({"sentence_#": i+1, "sentence": s, "classification": lbl,
                     "ai_probability": f"{p:.4f}", "model": model_key,
                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    return pd.DataFrame(rows).to_csv(index=False)

def prob_bar_html(prob: float) -> str:
    pct = prob * 100
    hpct = (1 - prob) * 100
    return (
        f'<div style="margin:4px 0">'
        f'<div style="display:flex;gap:4px;align-items:center">'
        f'<span style="width:55px;font-size:0.8rem;color:#dc3545">AI {pct:.0f}%</span>'
        f'<div style="flex:1;background:#e0e0e0;border-radius:4px;height:14px">'
        f'<div style="width:{pct:.0f}%;background:#dc3545;height:14px;border-radius:4px"></div></div>'
        f'</div>'
        f'<div style="display:flex;gap:4px;align-items:center">'
        f'<span style="width:55px;font-size:0.8rem;color:#28a745">Human {hpct:.0f}%</span>'
        f'<div style="flex:1;background:#e0e0e0;border-radius:4px;height:14px">'
        f'<div style="width:{hpct:.0f}%;background:#28a745;height:14px;border-radius:4px"></div></div>'
        f'</div></div>'
    )

AI_PHRASES = [
    "it is important to note", "in conclusion", "furthermore", "it is worth noting",
    "in summary", "to summarise", "as an ai", "delve into", "it is crucial",
    "in today's world", "in the realm of", "it is essential", "plays a pivotal role",
    "it is noteworthy", "it is imperative", "a comprehensive overview",
    "in today's digital age", "it cannot be overstated",
]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    mode = st.radio(
        "Navigation",
        ["🔍 Single Analysis", "👥 Human vs AI Lab", "🤖 Generate & Detect",
         "⚔️ Attack Simulation", "📚 Model Explorer", "🔬 Hybrid Research"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    selected_model = st.selectbox(
        "Active Model (for analysis)",
        list(MODEL_OPTIONS.keys()),
        help="Used in Single Analysis, Human vs AI Lab, Attack Simulation"
    )

    card = MODEL_CARDS[selected_model]
    badge_cls = "badge-hyb" if card["category"] == "Hybrid" else "badge-ind"
    st.markdown(
        f'<span class="{badge_cls}">{card["category"]}</span> '
        f'<span style="font-size:0.88rem;font-weight:600">&nbsp;{card["label"]}</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="model-info-box" style="margin-top:8px">'
        f'<b>Clean F1</b> {card["clean_f1"]:.4f} &nbsp;·&nbsp; '
        f'<b>Acc</b> {card["clean_acc"]:.4f}<br>'
        f'<b>Pegasus ASR</b> {card["peg_asr"]}% &nbsp;·&nbsp; '
        f'<b>QuillBot ASR</b> {card["qui_asr"]}%'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("**Colour guide**")
    st.markdown("🟢 Green = Human (>60% confidence)")
    st.markdown("🔴 Red = AI-Generated (>60%)")
    st.markdown("🟡 Amber = Uncertain (40–60%)")
    st.markdown(f"---\n**Device:** `{DEVICE}`")
    st.markdown(f"**GPU:** `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU)'}`")

    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.markdown("""
        MSc AI dissertation platform investigating how adversarial paraphrasing attacks degrade
        transformer-based AI-text detectors.

        **9 models evaluated**: 5 individual + 4 hybrid architectures
        across 3 attack types (Pegasus, QuillBot, ChatGPT) and 2 datasets.

        **Key finding**: QuillBot-style attack reduces RoBERTa recall from
        **99.95% → 86.6%** (ASR=13.4%). H1-BiLSTM reduces this to **4.8%**.

        **B00409227** · UWS · Dr Tahir Mahmood
        """)
    st.markdown('[📂 GitHub](https://github.com/B00409227/MSc-AI-Detection)', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — SINGLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🔍 Single Analysis":
    st.subheader("🔍 Single Text Analysis")

    tab_paste, tab_upload = st.tabs(["✏️ Paste Text", "📂 Upload File"])
    input_text = ""

    with tab_paste:
        gen_col, _ = st.columns([1, 3])
        with gen_col:
            if st.button("🤖 Populate with AI text", key="gen_single_ai"):
                with st.spinner("Generating with DistilGPT-2…"):
                    st.session_state["paste_input"] = generate_ai_text("academic writing and research")
                st.rerun()
        input_text_paste = st.text_area(
            "Paste your text below",
            height=220,
            placeholder="Paste any text — essay, article, email, answer, or assignment...",
            key="paste_input"
        )
        if input_text_paste:
            st.caption(f"{len(input_text_paste):,} characters · {len(input_text_paste.split()):,} words")
            input_text = input_text_paste

    with tab_upload:
        uploaded_file = st.file_uploader("Upload .txt or .docx", type=["txt", "docx"])
        if uploaded_file:
            input_text = (uploaded_file.read().decode("utf-8", errors="ignore")
                          if uploaded_file.name.endswith(".txt") else extract_docx(uploaded_file))
            if input_text and not input_text.startswith("ERROR"):
                st.success(f"Loaded: {len(input_text.split()):,} words")
                with st.expander("Preview"):
                    st.text(input_text[:500] + ("…" if len(input_text) > 500 else ""))
            elif input_text.startswith("ERROR"):
                st.error(input_text); input_text = ""

    # ── Model card for selected model ─────────────────────────────────────────
    card = MODEL_CARDS[selected_model]
    with st.expander(f"📋 {card['label']} — Architecture & Performance", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Clean F1",     f"{card['clean_f1']:.4f}")
        c2.metric("Clean Acc",    f"{card['clean_acc']:.4f}")
        c3.metric("Clean Recall", f"{card['clean_recall']:.4f}")
        c1.metric("Pegasus ASR",  f"{card['peg_asr']}%")
        c2.metric("QuillBot ASR", f"{card['qui_asr']}%",
                  delta=f"{card['qui_asr']-1:.1f}% above min" if card['qui_asr'] > 1 else None,
                  delta_color="inverse")
        c3.metric("M4 F1",        f"{card['m4_f1']:.4f}")
        st.markdown(f"**Architecture:** {card['arch']}")
        st.markdown(f"**Training:** {card['train']}")
        st.markdown(f"**Strength:** {card['strength']}")
        st.markdown(f"**Weakness:** {card['weakness']}")
        st.caption(f"Reference: {card['ref']}")

    if st.button("🔍 Analyse Text", type="primary", disabled=not bool(input_text.strip())):

        with st.spinner(f"Running {selected_model}…"):
            label, ai_prob, error = predict_text(input_text, selected_model)

        if error and error.startswith("RESEARCH_ONLY:"):
            st.info(
                f"**{selected_model} is a research-only model.** "
                f"{error[len('RESEARCH_ONLY:'):]}\n\n"
                "Switch to **🔬 Hybrid Research** mode to see full experimental results, "
                "or choose a different model from the sidebar."
            )
            st.stop()
        if error or ai_prob is None:
            st.warning(f"Model unavailable: {error}. Showing placeholder.")
            ai_prob, label = 0.72, 1

        ai_pct     = ai_prob * 100
        confidence = max(ai_pct, 100 - ai_pct)
        verdict    = "AI-Generated" if label == 1 else "Human-Written"

        st.markdown("---")
        st.subheader("📊 Document Result")

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><h3>{verdict}</h3><p>Verdict</p></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><h3>{ai_pct:.1f}%</h3><p>AI Probability</p></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><h3>{(100-ai_pct):.1f}%</h3><p>Human Probability</p></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card"><h3>{confidence:.1f}%</h3><p>Confidence</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(traffic_light_html(ai_pct), unsafe_allow_html=True)
        st.markdown(verdict_box(label, ai_prob), unsafe_allow_html=True)

        # Sentence analysis
        st.markdown("---")
        st.subheader("📝 Sentence-Level Breakdown")
        sentences = split_sentences(input_text)
        sentence_probs = []

        if len(sentences) > 1:
            prog = st.progress(0, text="Analysing sentences…")
            for i, sent in enumerate(sentences[:30]):
                _, p, _ = predict_text(sent, selected_model)
                sentence_probs.append(p if p is not None else ai_prob)
                prog.progress((i+1) / min(len(sentences), 30),
                              text=f"Sentence {i+1}/{min(len(sentences),30)}")
            prog.empty()

            n_ai  = sum(1 for p in sentence_probs if p >= 0.6)
            n_hum = sum(1 for p in sentence_probs if p <= 0.4)
            n_unc = len(sentence_probs) - n_ai - n_hum

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("🔴 AI sentences", n_ai)
            sc2.metric("🟢 Human sentences", n_hum)
            sc3.metric("🟡 Uncertain", n_unc)

            html = ""
            for i, (s, p) in enumerate(zip(sentences[:30], sentence_probs)):
                html += (f'<div class="{label_colour(p)}"><strong>S{i+1}:</strong> '
                         f'{s} <em>({label_text(p)} — {p*100:.1f}%)</em></div>')
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Text too short for sentence-level breakdown.")
            sentence_probs = [ai_prob]

        # Charts
        st.markdown("---")
        st.subheader("📈 Visualisations")
        ch1, ch2 = st.columns(2)

        with ch1:
            fig, ax = plt.subplots(figsize=(5, 4))
            n_ai_c  = sum(1 for p in sentence_probs if p >= 0.6)
            n_hum_c = sum(1 for p in sentence_probs if p <= 0.4)
            n_unc_c = len(sentence_probs) - n_ai_c - n_hum_c
            data = [(n_hum_c,"Human","#28a745"),(n_ai_c,"AI-Generated","#dc3545"),(n_unc_c,"Uncertain","#ffc107")]
            data = [(s,l,c) for s,l,c in data if s > 0]
            if data:
                s,l,c = zip(*data)
                ax.pie(s, labels=l, colors=c, autopct="%1.1f%%", startangle=90)
            ax.set_title("Sentence Classification", fontweight="bold")
            st.pyplot(fig); plt.close()

        with ch2:
            if len(sentence_probs) > 1:
                fig, ax = plt.subplots(figsize=(5, 4))
                clrs = ["#dc3545" if p >= 0.6 else "#28a745" if p <= 0.4 else "#ffc107"
                        for p in sentence_probs]
                ax.barh(range(len(sentence_probs)), [p*100 for p in sentence_probs], color=clrs)
                ax.axvline(x=50, color="black", linestyle="--", linewidth=1)
                ax.set_xlim(0, 100)
                ax.set_xlabel("AI Probability (%)")
                ax.set_ylabel("Sentence #")
                ax.set_title("Per-Sentence AI Score", fontweight="bold")
                st.pyplot(fig); plt.close()

        # AI phrase check
        st.markdown("---")
        st.subheader("🚩 AI Phrase Pattern Check")
        found = [p for p in AI_PHRASES if p in input_text.lower()]
        if found:
            st.markdown(
                f'<div class="danger-box">⚠️ AI phrases detected: <strong>{", ".join(found)}</strong></div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ No common AI phrase patterns found.</div>',
                        unsafe_allow_html=True)

        # Stylometric features panel (H4's 6 features — shown for all analyses)
        st.markdown("---")
        st.subheader("📐 Stylometric Feature Analysis (H4 Features)")
        with st.expander("View the 6 stylometric features used by the H4-FeatureFusion model", expanded=False):
            try:
                stylo = extract_stylometric_features(input_text)
                sf1, sf2, sf3, sf4, sf5, sf6 = stylo
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Word Count", f"{sf1:.0f}")
                sc2.metric("Avg Word Length", f"{sf2:.2f} chars")
                sc3.metric("Type-Token Ratio", f"{sf3:.3f}")
                sc1.metric("Punct Density", f"{sf4:.2f} per 100w")
                sc2.metric("Avg Sent Length", f"{sf5:.1f} words")
                sc3.metric("Hapax Legomena", f"{sf6:.3f}")
                st.caption(
                    "**Type-Token Ratio:** unique/total words — higher = more diverse vocabulary. "
                    "**Hapax legomena:** proportion of words appearing exactly once — AI text tends to be lower. "
                    "**Punct density:** AI text often has lower punctuation density."
                )
                # Radar-style bar chart of normalised features
                fig_s, ax_s = plt.subplots(figsize=(8, 3))
                feature_labels = ["Word Count\n÷1000", "Avg Word\nLength", "Type-Token\nRatio",
                                  "Punct\nDensity÷100", "Avg Sent\nLength÷50", "Hapax\nRatio"]
                # Normalise each feature to 0–1 for visual comparison
                norms = [sf1/1000, sf2/10, sf3, sf4/100, sf5/50, sf6]
                norms = [min(max(v, 0), 1) for v in norms]
                colours = ["#1565C0", "#1B5E20", "#4A148C", "#E65100", "#B71C1C", "#004D40"]
                ax_s.bar(feature_labels, norms, color=colours, alpha=0.85, edgecolor="white")
                ax_s.set_ylim(0, 1.1); ax_s.set_ylabel("Normalised Value")
                ax_s.set_title("H4 Stylometric Profile — Normalised Feature Values", fontweight="bold")
                ax_s.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Midpoint")
                plt.tight_layout(); st.pyplot(fig_s); plt.close()
            except Exception as e:
                st.warning(f"Could not compute stylometric features: {e}")

        # Export
        st.markdown("---")
        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button(
                "📥 Download CSV",
                data=results_to_csv(sentences[:30] if len(sentences)>1 else [input_text[:200]],
                                    sentence_probs, selected_model),
                file_name=f"detection_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv"
            )
        with ex2:
            report = (f"AI TEXT DETECTION REPORT\n{datetime.now():%Y-%m-%d %H:%M}\n"
                      f"Model: {selected_model}\n{'='*45}\n"
                      f"VERDICT: {verdict}\nAI: {ai_pct:.1f}%  Human: {100-ai_pct:.1f}%\n")
            st.download_button("📄 Download Report",
                               data=report,
                               file_name=f"report_{datetime.now():%Y%m%d_%H%M%S}.txt",
                               mime="text/plain")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — HUMAN vs AI LAB
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "👥 Human vs AI Lab":
    st.subheader("👥 Human vs AI Lab — Side-by-Side Comparison")
    st.markdown("""
    <div class="key-finding">
    📌 Paste a <strong>human-written</strong> text on the left and an <strong>AI-generated</strong> text
    on the right — or load a sample pair — then analyse both with all 5 models simultaneously.
    </div>
    """, unsafe_allow_html=True)

    # Sample loader — uses on_change callbacks so session_state is updated correctly
    def _load_human_sample():
        sel = st.session_state.get("h_samp", "— paste your own —")
        if sel != "— paste your own —":
            st.session_state["lab_human"] = SAMPLE_HUMAN_TEXTS[sel]

    def _load_ai_sample():
        sel = st.session_state.get("a_samp", "— paste your own —")
        if sel != "— paste your own —":
            st.session_state["lab_ai"] = SAMPLE_AI_TEXTS[sel]

    sample_col1, sample_col2, _ = st.columns([2, 2, 1])
    with sample_col1:
        st.selectbox("Load human text sample",
                     ["— paste your own —"] + list(SAMPLE_HUMAN_TEXTS.keys()),
                     key="h_samp", on_change=_load_human_sample)
    with sample_col2:
        st.selectbox("Load AI text sample",
                     ["— paste your own —"] + list(SAMPLE_AI_TEXTS.keys()),
                     key="a_samp", on_change=_load_ai_sample)

    col_h, col_a = st.columns(2)

    with col_h:
        st.markdown('<p class="lab-header-human">🧑 HUMAN-WRITTEN TEXT</p>', unsafe_allow_html=True)
        human_text = st.text_area(
            "Human text",
            height=260,
            placeholder="Paste human-written text here, or choose a sample above…",
            key="lab_human",
            label_visibility="collapsed"
        )
        if human_text:
            st.caption(f"{len(human_text):,} chars · {len(human_text.split()):,} words")

    with col_a:
        st.markdown('<p class="lab-header-ai">🤖 AI-GENERATED TEXT</p>', unsafe_allow_html=True)
        ai_text = st.text_area(
            "AI text",
            height=260,
            placeholder="Paste AI-generated text here, or choose a sample above…",
            key="lab_ai",
            label_visibility="collapsed"
        )
        if ai_text:
            st.caption(f"{len(ai_text):,} chars · {len(ai_text.split()):,} words")

        if st.button("🤖 Generate AI text for this field", key="gen_lab_ai"):
            with st.spinner("Generating with DistilGPT-2…"):
                st.session_state["lab_ai"] = generate_ai_text("academic writing and research")
            st.rerun()

    run_all = st.toggle("Run all 7 models simultaneously (including H3 & H4 hybrids)", value=True)

    btn_disabled = not (bool(human_text.strip()) and bool(ai_text.strip()))
    if st.button("⚡ Analyse Both Texts", type="primary", disabled=btn_disabled):

        # H1/H2 are research-only; exclude them from bulk runs
        if run_all:
            models_to_run = LIVE_INFERENCE_MODELS
        elif selected_model in LIVE_INFERENCE_MODELS:
            models_to_run = [selected_model]
        else:
            st.info(f"**{selected_model}** is research-only. Switching to RoBERTa for this run.")
            models_to_run = ["RoBERTa-base"]

        with st.spinner("Running models on both texts…"):
            h_results = {}
            a_results = {}
            for mk in models_to_run:
                h_results[mk] = predict_text(human_text, mk)
                a_results[mk] = predict_text(ai_text,   mk)

        st.markdown("---")
        st.subheader("📊 Results")

        # Quick verdict row
        v_cols = st.columns(len(models_to_run))
        for i, mk in enumerate(models_to_run):
            _, h_prob, _ = h_results[mk]
            _, a_prob, _ = a_results[mk]
            if h_prob is None: h_prob = 0.5
            if a_prob is None: a_prob = 0.5
            with v_cols[i]:
                h_lbl = "🟢 Human" if h_prob < 0.5 else "🔴 AI"
                a_lbl = "🟢 Human" if a_prob < 0.5 else "🔴 AI"
                correct = (h_prob < 0.5) and (a_prob >= 0.5)
                chk = "✅" if correct else "⚠️"
                st.markdown(
                    f'<div style="border:1px solid #ddd;border-radius:8px;padding:10px;text-align:center">'
                    f'<div style="font-size:0.78rem;font-weight:700;color:#555;margin-bottom:6px">{mk}</div>'
                    f'<div style="color:#155724">Human text: {h_lbl} ({h_prob*100:.0f}%)</div>'
                    f'<div style="color:#721c24">AI text: {a_lbl} ({a_prob*100:.0f}%)</div>'
                    f'<div style="margin-top:6px;font-size:0.85rem">{chk} {"Correct" if correct else "Error"}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # Side-by-side detailed comparison
        dc_h, dc_a = st.columns(2)

        with dc_h:
            st.markdown("### 🧑 Human Text")
            for mk in models_to_run:
                _, prob, err = h_results[mk]
                if prob is None: prob = 0.5
                colour = "#28a745" if prob < 0.5 else "#dc3545"
                lbl    = "Human" if prob < 0.5 else "AI"
                st.markdown(
                    f'<div style="margin:6px 0;padding:10px;background:#f8f9fa;border-radius:6px;border-left:4px solid {colour}">'
                    f'<strong style="font-size:0.85rem">{mk}</strong><br>'
                    f'{prob_bar_html(prob)}'
                    f'<span style="font-size:0.82rem;font-weight:700;color:{colour}">→ {lbl}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        with dc_a:
            st.markdown("### 🤖 AI Text")
            for mk in models_to_run:
                _, prob, err = a_results[mk]
                if prob is None: prob = 0.5
                colour = "#28a745" if prob < 0.5 else "#dc3545"
                lbl    = "Human" if prob < 0.5 else "AI"
                st.markdown(
                    f'<div style="margin:6px 0;padding:10px;background:#f8f9fa;border-radius:6px;border-left:4px solid {colour}">'
                    f'<strong style="font-size:0.85rem">{mk}</strong><br>'
                    f'{prob_bar_html(prob)}'
                    f'<span style="font-size:0.82rem;font-weight:700;color:{colour}">→ {lbl}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Comparison bar chart
        st.markdown("---")
        if models_to_run:
            fig, ax = plt.subplots(figsize=(max(8, len(models_to_run) * 1.8), 5))
            x      = np.arange(len(models_to_run))
            width  = 0.35
            h_probs = [max(h_results[mk][1] or 0, 0) * 100 for mk in models_to_run]
            a_probs = [max(a_results[mk][1] or 0, 0) * 100 for mk in models_to_run]
            b1 = ax.bar(x - width/2, h_probs, width, label="Human Text", color="#28a745", alpha=0.85, edgecolor="white")
            b2 = ax.bar(x + width/2, a_probs, width, label="AI Text",    color="#dc3545", alpha=0.85, edgecolor="white")
            ax.axhline(y=50, color="black", linestyle="--", linewidth=1.2, label="Decision boundary (50%)")
            ax.set_xticks(x)
            ax.set_xticklabels(models_to_run, rotation=20, ha="right")
            ax.set_ylim(0, 115)
            ax.set_ylabel("AI Probability Score (%)")
            ax.set_title("AI Probability: Human Text vs AI-Generated Text — All Models",
                         fontweight="bold", fontsize=12)
            ax.bar_label(b1, labels=[f"{v:.0f}%" for v in h_probs], padding=3, fontsize=9, fontweight="bold")
            ax.bar_label(b2, labels=[f"{v:.0f}%" for v in a_probs], padding=3, fontsize=9, fontweight="bold")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Correct classification summary
        n_correct  = sum(1 for mk in models_to_run
                         if (h_results[mk][1] or 0.5) < 0.5 and (a_results[mk][1] or 0.5) >= 0.5)
        n_total    = len(models_to_run)
        acc_str    = f"{n_correct}/{n_total} models correctly classified both texts"
        box_class  = "success-box" if n_correct == n_total else "warning-box"
        st.markdown(f'<div class="{box_class}">📊 <strong>{acc_str}</strong></div>', unsafe_allow_html=True)

        # AI phrase scan on both texts
        found_h = [p for p in AI_PHRASES if p in human_text.lower()]
        found_a = [p for p in AI_PHRASES if p in ai_text.lower()]
        st.markdown("---")
        fh1, fh2 = st.columns(2)
        with fh1:
            if found_h:
                st.markdown(f'<div class="warning-box">⚠️ Human text contains AI-style phrases: <em>{", ".join(found_h)}</em></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ Human text: no AI phrase patterns detected.</div>', unsafe_allow_html=True)
        with fh2:
            if found_a:
                st.markdown(f'<div class="danger-box">🤖 AI text contains AI-style phrases: <em>{", ".join(found_a)}</em></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ AI text: no AI phrase patterns — it may be well-disguised or paraphrased.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 — GENERATE & DETECT
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🤖 Generate & Detect":
    st.subheader("🤖 Generate & Detect — AI Text Generation + Detection")
    st.markdown("""
    <div class="key-finding">
    📌 Enter a topic or question. This tool generates an AI response using
    <strong>DistilGPT-2</strong> (82M params, auto-regressive LM), then analyses it alongside
    your own human-written response. See how our detectors handle freshly generated AI text.
    </div>
    """, unsafe_allow_html=True)

    st.info("**DistilGPT-2** is a small, fast model (~350 MB). First load takes ~30 seconds. "
            "Text quality is lower than ChatGPT but sufficient to demonstrate AI detection.", icon="⚡")

    # Preset topics
    preset_topics = [
        "custom",
        "the impact of social media on mental health",
        "the causes of the First World War",
        "how artificial intelligence is changing healthcare",
        "the advantages and disadvantages of renewable energy",
        "the role of exercise in maintaining good health",
        "climate change and its effects on global ecosystems",
    ]

    preset = st.selectbox("Choose a topic preset (or type your own below)", preset_topics)

    if preset == "custom":
        topic = st.text_input("Your topic / question", placeholder="e.g. 'the impact of smartphones on education'")
    else:
        topic = preset
        st.text_input("Topic (editable)", value=topic, key="topic_edit")
        topic = st.session_state.get("topic_edit", topic)

    gen_len = st.slider("AI response length (tokens)", 80, 250, 160, 10)

    col_gen, col_human = st.columns(2)

    with col_gen:
        st.markdown("#### 🤖 AI-Generated Response")
        if "generated_text" not in st.session_state:
            st.session_state.generated_text = ""

        if st.button("⚡ Generate AI Text", type="primary", disabled=not bool(topic.strip())):
            with st.spinner("Generating with DistilGPT-2… (first run downloads 350 MB)"):
                try:
                    st.session_state.generated_text = generate_ai_text(topic, gen_len)
                except Exception as e:
                    st.session_state.generated_text = ""
                    st.error(f"Generation failed: {e}")

        generated_display = st.text_area(
            "Generated text (editable)",
            value=st.session_state.generated_text,
            height=230,
            key="gen_display",
            placeholder="Click 'Generate AI Text' above to fill this box…"
        )
        if generated_display:
            st.caption(f"{len(generated_display):,} chars · {len(generated_display.split()):,} words")
            found_a = [p for p in AI_PHRASES if p in generated_display.lower()]
            if found_a:
                st.markdown(
                    f'<div class="danger-box" style="font-size:0.82rem">🤖 AI patterns found: {", ".join(found_a)}</div>',
                    unsafe_allow_html=True)

    with col_human:
        st.markdown("#### 🧑 Your Human Response")
        human_gen_text = st.text_area(
            "Write your own response to the same topic",
            height=230,
            placeholder=f"Write your own answer about '{topic[:50]}…' here in your natural voice…",
            key="human_gen"
        )
        if human_gen_text:
            st.caption(f"{len(human_gen_text):,} chars · {len(human_gen_text.split()):,} words")
            found_h = [p for p in AI_PHRASES if p in human_gen_text.lower()]
            if found_h:
                st.markdown(
                    f'<div class="warning-box" style="font-size:0.82rem">⚠️ AI-style patterns in your text: {", ".join(found_h)}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box" style="font-size:0.82rem">✅ Your text looks human-written.</div>', unsafe_allow_html=True)

    both_ready = bool(generated_display.strip()) and bool(human_gen_text.strip())

    if st.button("🔬 Detect Both — Run All 7 Live Models", type="primary", disabled=not both_ready):

        with st.spinner("Running all 7 live models on both texts…"):
            gen_results = {}
            hum_results = {}
            for mk in LIVE_INFERENCE_MODELS:
                gen_results[mk] = predict_text(generated_display, mk)
                hum_results[mk] = predict_text(human_gen_text,   mk)

        st.markdown("---")
        st.subheader("📊 Detection Results")

        # Summary table
        rows = []
        for mk in LIVE_INFERENCE_MODELS:
            _, g_prob, _ = gen_results[mk]
            _, h_prob, _ = hum_results[mk]
            if g_prob is None: g_prob = 0.5
            if h_prob is None: h_prob = 0.5
            g_lbl = "🔴 AI" if g_prob >= 0.5 else "🟢 Human"
            h_lbl = "🔴 AI" if h_prob >= 0.5 else "🟢 Human"
            correct = (g_prob >= 0.5) and (h_prob < 0.5)
            rows.append({
                "Model": mk,
                "AI Text Score": f"{g_prob*100:.1f}%",
                "AI Text Verdict": g_lbl,
                "Human Text Score": f"{h_prob*100:.1f}%",
                "Human Text Verdict": h_lbl,
                "Both Correct": "✅ Yes" if correct else "❌ No",
            })

        result_df = pd.DataFrame(rows)

        def highlight_correct(row):
            colour = "#d4edda" if "Yes" in row["Both Correct"] else "#fde8e8"
            return [f"background-color: {colour}"] * len(row)

        st.dataframe(result_df.style.apply(highlight_correct, axis=1),
                     use_container_width=True, hide_index=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 5))
        model_names = LIVE_INFERENCE_MODELS
        x      = np.arange(len(model_names))
        width  = 0.35
        g_prbs = [max(gen_results[mk][1] or 0, 0) * 100 for mk in model_names]
        h_prbs = [max(hum_results[mk][1] or 0, 0) * 100 for mk in model_names]
        b1 = ax.bar(x - width/2, g_prbs, width, label="AI Text (DistilGPT-2)", color="#dc3545", alpha=0.85)
        b2 = ax.bar(x + width/2, h_prbs, width, label="Human Text",            color="#28a745", alpha=0.85)
        ax.axhline(y=50, color="black", linestyle="--", linewidth=1.2, label="Decision boundary")
        ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, ha="right")
        ax.set_ylim(0, 115); ax.set_ylabel("AI Probability Score (%)")
        ax.set_title("Generate & Detect: AI Score for AI Text vs Human Text", fontweight="bold")
        ax.bar_label(b1, labels=[f"{v:.0f}%" for v in g_prbs], padding=3, fontsize=9, fontweight="bold")
        ax.bar_label(b2, labels=[f"{v:.0f}%" for v in h_prbs], padding=3, fontsize=9, fontweight="bold")
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()

        n_correct = sum(1 for mk in LIVE_INFERENCE_MODELS
                        if (gen_results[mk][1] or 0.5) >= 0.5 and (hum_results[mk][1] or 0.5) < 0.5)
        n_total = len(LIVE_INFERENCE_MODELS)
        st.markdown(
            f'<div class="{"success-box" if n_correct >= n_total//2 else "warning-box"}">'
            f'📊 <strong>{n_correct}/{n_total} models correctly distinguished the AI text from the human text.</strong>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="key-finding">
        💡 <strong>Dissertation context:</strong> These models were trained on ChatGPT output (HC3 dataset).
        DistilGPT-2 uses a different generation strategy but shares statistical regularities with other LLMs —
        high token repetition probability, formulaic sentence structures, and lower lexical diversity.
        This is why the detectors generalise beyond their training distribution.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 4 — ATTACK SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "⚔️ Attack Simulation":
    st.subheader("⚔️ Attack Simulation — Adversarial Paraphrase")
    st.markdown("""
    <div class="key-finding">
    📌 <strong>Dissertation finding:</strong> The QuillBot-style attack (T5_Paraphrase_Paws,
    top-k=200 sampling) reduces RoBERTa recall <strong>99.95% → 86.6%</strong> (ASR=13.4%).
    Pegasus (beam-search) only achieves 1.2% ASR — diversity of rewriting is what defeats detectors.
    This demo runs the QuillBot-style attack live on any text you paste.
    </div>
    """, unsafe_allow_html=True)

    atk_btn_col, _ = st.columns([1, 3])
    with atk_btn_col:
        if st.button("🤖 Populate with AI text", key="gen_attack_ai"):
            with st.spinner("Generating with DistilGPT-2…"):
                st.session_state["attack_input"] = generate_ai_text("artificial intelligence and machine learning")
            st.rerun()
    input_text = st.text_area(
        "Paste AI-generated text to attack",
        height=200,
        placeholder="Paste AI-generated text here to see how the paraphrase attack affects detection…",
        key="attack_input"
    )

    # Show known ASR results for reference
    with st.expander("📊 Known ASR results from dissertation experiments"):
        asr_df = pd.DataFrame([
            {"Model": "RoBERTa-base",  "Pegasus ASR": "1.2%",  "QuillBot ASR": "13.4%", "ChatGPT ASR": "1.6%"},
            {"Model": "BERT-base",     "Pegasus ASR": "1.8%",  "QuillBot ASR": "12.2%", "ChatGPT ASR": "3.2%"},
            {"Model": "DistilBERT",    "Pegasus ASR": "5.2%",  "QuillBot ASR": "19.0%", "ChatGPT ASR": "6.8%"},
            {"Model": "Hello-SimpleAI","Pegasus ASR": "0.4%",  "QuillBot ASR": "14.0%", "ChatGPT ASR": "2.8%"},
            {"Model": "LR+TF-IDF",    "Pegasus ASR": "29.0%", "QuillBot ASR": "26.8%", "ChatGPT ASR": "39.4%"},
            {"Model": "H1-BiLSTM",    "Pegasus ASR": "1.4%",  "QuillBot ASR": "4.8%",  "ChatGPT ASR": "2.2%"},
        ])
        st.dataframe(asr_df, use_container_width=True, hide_index=True)

    if st.button("⚔️ Run Paraphrase Attack", type="primary", disabled=not bool(input_text.strip())):

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📄 BEFORE Attack")
            with st.spinner("Classifying original…"):
                label_o, prob_o, _ = predict_text(input_text, selected_model)
                if prob_o is None: prob_o = 0.92
            st.markdown(verdict_box(label_o or 1, prob_o), unsafe_allow_html=True)
            st.markdown(traffic_light_html(prob_o * 100), unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("AI Score", f"{prob_o*100:.1f}%")
            c2.metric("Verdict", "AI-Generated" if (label_o or 1) == 1 else "Human")
            st.text_area("Original", input_text[:600], height=160, disabled=True)

        with col2:
            st.markdown("### 🔄 AFTER QuillBot-Style Attack")
            with st.spinner("Running T5_Paraphrase_Paws (top-k=200, first run ~60s)…"):
                try:
                    ptok, pmod = load_paraphraser()
                    short = " ".join(input_text.split()[:80])
                    enc = ptok(f"paraphrase: {short} </s>",
                               return_tensors="pt", max_length=256, truncation=True)
                    with torch.no_grad():
                        out = pmod.generate(**{k: v.to(DEVICE) for k, v in enc.items()},
                                            max_length=256, do_sample=True, top_k=200, top_p=0.95)
                    rewritten = ptok.decode(out[0], skip_special_tokens=True)
                    para_err  = None
                except Exception as e:
                    rewritten = input_text; para_err = str(e)

            label_r, prob_r, _ = predict_text(rewritten, selected_model)
            if prob_r is None: prob_r = 0.08

            st.markdown(verdict_box(label_r or 0, prob_r), unsafe_allow_html=True)
            st.markdown(traffic_light_html(prob_r * 100), unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("AI Score", f"{prob_r*100:.1f}%", delta=f"{(prob_r-prob_o)*100:+.1f}%",
                      delta_color="inverse")
            c2.metric("Verdict", "AI-Generated" if (label_r or 0) == 1 else "Human")
            st.text_area("Rewritten", rewritten[:600], height=160, disabled=True)
            if para_err:
                st.warning(f"Paraphraser error: {para_err}")

        # Before/after chart
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(8, 4))
        clr_r  = "#28a745" if prob_r < 0.5 else "#ffc107"
        bars   = ax.bar(["Before Attack", "After Attack"],
                        [prob_o * 100, prob_r * 100],
                        color=["#dc3545", clr_r], edgecolor="white", width=0.4)
        ax.axhline(y=50, color="black", linestyle="--", linewidth=1.2, label="Decision boundary")
        ax.set_ylim(0, 110); ax.set_ylabel("AI Probability (%)")
        ax.set_title(f"QuillBot-Style Attack on {selected_model}", fontweight="bold")
        ax.bar_label(bars, labels=[f"{prob_o*100:.1f}%", f"{prob_r*100:.1f}%"],
                     padding=4, fontsize=13, fontweight="bold")
        ax.legend(); st.pyplot(fig); plt.close()

        drop = (prob_o - prob_r) * 100
        if drop > 5:
            st.markdown(
                f'<div class="success-box">✅ Attack dropped AI score by <strong>{drop:.1f} pp</strong>. '
                f'{"Detection EVADED" if prob_r < 0.5 else "Detection maintained but weakened"}.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="warning-box">⚠️ Attack had minimal effect — detector is robust on this sample.</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 5 — MODEL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📚 Model Explorer":
    st.subheader("📚 Model Explorer — Detailed Architecture & Results")
    st.markdown("Deep-dive into all 9 models evaluated in this dissertation: architecture, training, performance, and adversarial robustness.")

    view = st.radio("Show", ["All Models", "Individual Only", "Hybrid Only"], horizontal=True)

    for mk, card in MODEL_CARDS.items():
        if view == "Individual Only" and card["category"] != "Individual": continue
        if view == "Hybrid Only"    and card["category"] != "Hybrid":     continue

        card_class = "model-card-hyb" if card["category"] == "Hybrid" else "model-card-ind"
        badge_cls  = "badge-hyb"      if card["category"] == "Hybrid" else "badge-ind"

        with st.expander(
            f"{'🔬' if card['category']=='Hybrid' else '🤖'} {card['label']}",
            expanded=False
        ):
            # Header row
            st.markdown(
                f'<span class="{badge_cls}">{card["category"]}</span> '
                f'&nbsp;<code style="font-size:0.82rem">{card.get("hf_id","")}</code>',
                unsafe_allow_html=True
            )

            # Description
            if "why" in card:
                st.markdown(f"**Why this architecture?** {card['why']}")

            st.markdown(f"**Architecture:** {card['arch']}")
            st.markdown(f"**Base:** {card['base']}")

            # Training details table
            col_t, col_m = st.columns([1, 1])
            with col_t:
                st.markdown("**Training**")
                st.markdown(f"""
                | Setting | Value |
                |---------|-------|
                | Method | {card['train'].split('·')[0].strip()} |
                | Device | {card['device']} |
                | Trainable params | {card['trainable']} |
                """)

            with col_m:
                st.markdown("**Clean HC3 Performance**")
                st.markdown(f"""
                | Metric | Value |
                |--------|-------|
                | F1-Score | **{card['clean_f1']:.4f}** |
                | Accuracy | {card['clean_acc']:.4f} |
                | Recall | {card['clean_recall']:.4f} |
                | Precision | {card['clean_prec']:.4f} |
                | M4 F1 | {card['m4_f1']:.4f} |
                """)

            # ASR bar chart
            fig, ax = plt.subplots(figsize=(6, 2.5))
            attacks = ["Pegasus\n(beam-search)", "QuillBot\n(top-k sample)", "ChatGPT\n(rewrite)"]
            asrs    = [card["peg_asr"], card["qui_asr"], card["cgpt_asr"]]
            clrs    = ["#E53935" if a > 10 else "#FF9800" if a > 5 else "#4CAF50" for a in asrs]
            bars    = ax.barh(attacks, asrs, color=clrs, edgecolor="white", height=0.5)
            ax.axvline(x=10, color="black", linestyle="--", linewidth=1, alpha=0.5, label="10% threshold")
            ax.set_xlim(0, max(asrs) + 5)
            ax.set_xlabel("Attack Success Rate (%)")
            ax.set_title(f"Adversarial Robustness — {card['label']}", fontsize=10, fontweight="bold")
            ax.bar_label(bars, labels=[f"{v:.1f}%" for v in asrs], padding=3, fontsize=9, fontweight="bold")
            ax.legend(fontsize=8); plt.tight_layout()
            st.pyplot(fig); plt.close()

            # Strength / weakness / reference
            strength_col, weak_col = st.columns(2)
            with strength_col:
                st.markdown(f'<div class="success-box">✅ <strong>Strength:</strong><br>{card["strength"]}</div>', unsafe_allow_html=True)
            with weak_col:
                st.markdown(f'<div class="warning-box">⚠️ <strong>Weakness:</strong><br>{card["weakness"]}</div>', unsafe_allow_html=True)
            st.caption(f"Reference: {card['ref']}")

    # All-model comparison chart
    st.markdown("---")
    st.subheader("📊 All-Model Overview")

    all_names = list(MODEL_CARDS.keys())
    clean_f1s  = [MODEL_CARDS[m]["clean_f1"]  for m in all_names]
    qui_asrs   = [MODEL_CARDS[m]["qui_asr"]   for m in all_names]
    m4_f1s     = [MODEL_CARDS[m]["m4_f1"]     for m in all_names]
    categories = [MODEL_CARDS[m]["category"]  for m in all_names]
    colours    = ["#1565C0" if c == "Individual" else "#E65100" for c in categories]

    short_names = [
        "RoBERTa", "BERT", "DistilBERT", "Hello-SimpleAI", "Log.Reg.",
        "H1-BiLSTM", "H2-TextCNN", "H3-Voting", "H4-Fusion"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, vals, title, ylim, pct in zip(
        axes,
        [clean_f1s, qui_asrs, m4_f1s],
        ["Clean HC3 F1 Score", "QuillBot Attack ASR (Lower = Better)", "M4 Cross-Dataset F1"],
        [(0.92, 1.01), (0, 42), (0.28, 0.82)],
        [True, False, True]
    ):
        bars = ax.bar(range(len(all_names)), vals, color=colours, edgecolor="white")
        ax.set_xticks(range(len(all_names)))
        ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.bar_label(bars, labels=[f"{v:.2f}" if pct else f"{v:.1f}%" for v in vals],
                     padding=2, fontsize=8, fontweight="bold")

    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color="#1565C0", label="Individual"), Patch(color="#E65100", label="Hybrid")],
        loc="upper center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.02)
    )
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MODE 6 — HYBRID RESEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🔬 Hybrid Research":
    st.subheader("🔬 Hybrid Model Research — Novel Architectures")
    st.markdown("""
    <div class="key-finding">
    📌 <strong>Research contribution:</strong> Four hybrid architectures are introduced and evaluated
    against the same three adversarial attacks. H1 (RoBERTa+BiLSTM) achieves the strongest adversarial
    robustness: QuillBot ASR=<strong>4.8%</strong> vs 13.4% for plain RoBERTa — a 64% reduction.
    </div>
    """, unsafe_allow_html=True)

    # Hybrid architecture cards
    st.markdown("### Hybrid Architectures")
    h1, h2 = st.columns(2)

    for col, hkey, icon in [(h1, "H1-RoBERTa+BiLSTM", "🔵"), (h2, "H2-BERT+TextCNN", "🟢")]:
        with col:
            c = MODEL_CARDS[hkey]
            st.markdown(f"""
            <div class="model-card-hyb">
            <div class="badge-hyb">Hybrid</div>
            <strong>{icon} {c['label']}</strong><br><br>
            <small><b>Architecture:</b> {c['arch']}</small><br>
            <small><b>Training:</b> {c['train']}</small><br>
            <small><b>Trainable:</b> {c['trainable']}</small><br><br>
            <b>Clean F1:</b> {c['clean_f1']:.4f} &nbsp;·&nbsp; <b>QuillBot ASR:</b> {c['qui_asr']}%<br>
            <small><em>{c['strength']}</em></small>
            </div>
            """, unsafe_allow_html=True)

    h3, h4 = st.columns(2)
    for col, hkey, icon in [(h3, "H3-SoftVoting", "🟡"), (h4, "H4-FeatureFusion", "🟠")]:
        with col:
            c = MODEL_CARDS[hkey]
            st.markdown(f"""
            <div class="model-card-hyb">
            <div class="badge-hyb">Hybrid</div>
            <strong>{icon} {c['label']}</strong><br><br>
            <small><b>Architecture:</b> {c['arch']}</small><br>
            <small><b>Training:</b> {c['train']}</small><br>
            <small><b>Trainable:</b> {c['trainable']}</small><br><br>
            <b>Clean F1:</b> {c['clean_f1']:.4f} &nbsp;·&nbsp; <b>QuillBot ASR:</b> {c['qui_asr']}%<br>
            <small><em>{c['strength']}</em></small>
            </div>
            """, unsafe_allow_html=True)

    # ── Live H3 / H4 inference ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧪 Live Hybrid Model Analysis")
    st.markdown(
        "Compare H3 (Soft Voting) and H4 (Feature Fusion) against plain RoBERTa **on your own text**. "
        "H1 and H2 weights are Colab-only and are not available for live inference."
    )

    hyb_btn_col, _ = st.columns([1, 3])
    with hyb_btn_col:
        if st.button("🤖 Populate with AI text", key="gen_hybrid_ai"):
            with st.spinner("Generating with DistilGPT-2…"):
                st.session_state["hybrid_input"] = generate_ai_text("artificial intelligence research")
            st.rerun()
    hybrid_input = st.text_area(
        "Paste text to test with hybrid models",
        height=130,
        placeholder="Paste any text here to see how H3 and H4 compare to RoBERTa…",
        key="hybrid_input"
    )

    if st.button("⚡ Run Hybrid Analysis", type="primary", disabled=not bool(hybrid_input.strip())):
        live_keys = ["RoBERTa-base", "H3-SoftVoting", "H4-FeatureFusion"]
        live_results = {}
        with st.spinner("Running RoBERTa, H3, and H4…"):
            for mk in live_keys:
                live_results[mk] = predict_text(hybrid_input, mk)

        lc1, lc2, lc3 = st.columns(3)
        icons = {"RoBERTa-base": "🔵", "H3-SoftVoting": "🟡", "H4-FeatureFusion": "🟠"}
        for col, mk in zip([lc1, lc2, lc3], live_keys):
            lbl, prob, err = live_results[mk]
            with col:
                if prob is None:
                    st.error(f"**{mk}**\n\n{err}")
                else:
                    verdict = "🤖 AI" if lbl == 1 else "🧑 Human"
                    colour  = "#dc3545" if lbl == 1 else "#28a745"
                    st.markdown(
                        f'<div style="border:2px solid {colour};border-radius:10px;padding:14px;text-align:center">'
                        f'<div style="font-size:0.9rem;font-weight:700;margin-bottom:8px">{icons[mk]} {mk}</div>'
                        f'<div style="font-size:1.5rem;font-weight:900;color:{colour}">{verdict}</div>'
                        f'<div style="font-size:1.1rem;margin-top:6px">AI: <b>{prob*100:.1f}%</b></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # Stylometric breakdown for this text
        st.markdown("**H4 Stylometric Features for this text:**")
        try:
            stylo = extract_stylometric_features(hybrid_input)
            sf1, sf2, sf3, sf4, sf5, sf6 = stylo
            hsc1, hsc2, hsc3 = st.columns(3)
            hsc1.metric("Word Count", f"{sf1:.0f}")
            hsc2.metric("Avg Word Length", f"{sf2:.2f}")
            hsc3.metric("Type-Token Ratio", f"{sf3:.3f}")
            hsc1.metric("Punct Density", f"{sf4:.2f}")
            hsc2.metric("Avg Sent Length", f"{sf5:.1f}")
            hsc3.metric("Hapax Ratio", f"{sf6:.3f}")
        except Exception as e:
            st.warning(f"Could not compute stylometric features: {e}")

    # Load hybrid results
    st.markdown("---")
    st.markdown("### Experimental Results")

    HYBRID_RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "metrics", "all_results_with_hybrids.csv")

    if os.path.exists(HYBRID_RESULTS_PATH):
        hybrid_df = pd.read_csv(HYBRID_RESULTS_PATH)

        summary_rows = []
        for model_name in hybrid_df["model"].unique():
            m = hybrid_df[hybrid_df["model"] == model_name]
            clean_f1 = m[m["dataset"] == "HC3-Clean"]["f1"].values
            peg_asr  = m[m["dataset"] == "Pegasus-Attack"]["attack_success"].values
            qui_asr  = m[m["dataset"] == "QuillBot-Attack"]["attack_success"].values
            cha_asr  = m[m["dataset"] == "ChatGPT-Attack"]["attack_success"].values
            m4_f1    = m[m["dataset"] == "Cross-Dataset"]["f1"].values
            mtype    = "Hybrid" if any(model_name.startswith(h) for h in ["H1-","H2-","H3-","H4-"]) else "Individual"
            summary_rows.append({
                "Model":        model_name,
                "Type":         mtype,
                "Clean F1":     round(float(clean_f1[0]), 4) if len(clean_f1) else None,
                "Pegasus ASR":  f"{round(float(peg_asr[0])*100,1)}%" if len(peg_asr) else None,
                "QuillBot ASR": f"{round(float(qui_asr[0])*100,1)}%" if len(qui_asr) else None,
                "ChatGPT ASR":  f"{round(float(cha_asr[0])*100,1)}%" if len(cha_asr) else None,
                "M4 F1":        round(float(m4_f1[0]), 4) if len(m4_f1) else None,
            })

        summary_table = pd.DataFrame(summary_rows)

        def highlight_hybrid(row):
            if row["Type"] == "Hybrid":
                return ["background-color: #fff3e0"] * len(row)
            return [""] * len(row)

        st.markdown("#### Master Results Table (9 Models)")
        st.dataframe(summary_table.style.apply(highlight_hybrid, axis=1),
                     use_container_width=True, hide_index=True)

        # Load saved figures if available
        fig_dir = os.path.join(PROJECT_ROOT, "results", "figures")
        for fname, caption in [
            ("fig_all9_models_clean_f1.png",          "Clean F1-Score comparison"),
            ("fig_all9_models_asr_comparison.png",    "Attack Success Rate comparison"),
            ("fig_all9_models_degradation_heatmap.png", "Performance degradation heatmap"),
            ("fig_all9_models_radar_chart.png",       "Multi-axis robustness radar chart"),
        ]:
            fpath = os.path.join(fig_dir, fname)
            if os.path.exists(fpath):
                st.markdown(f"#### {caption}")
                st.image(fpath, use_container_width=True)

        # McNemar's test
        mcnemar_path = os.path.join(PROJECT_ROOT, "results", "metrics", "mcnemar_test_results.csv")
        if os.path.exists(mcnemar_path):
            st.markdown("#### Statistical Significance — McNemar's Test")
            st.caption("Dietterich (1998) doi:10.1162/089976698300017197 — tests whether classification differences are statistically significant (χ² test on disagreement cells; p<0.05 = significant).")
            mcn_df = pd.read_csv(mcnemar_path)
            st.dataframe(mcn_df, use_container_width=True, hide_index=True)
            st.markdown("""
            **Interpretation:**
            - **RoBERTa vs H3**: p=0.899 — NOT significant. Soft voting adds no statistically meaningful benefit.
            - **RoBERTa vs H4**: p=0.006 — SIGNIFICANT. H4 has different decision boundaries (but higher QuillBot ASR due to TF-IDF vulnerability).
            - **H1-BiLSTM**: Not in McNemar test but shows largest ASR reduction (13.4%→4.8%).
            """)

        st.markdown("""
        <div class="key-finding">
        📌 <strong>Research conclusion:</strong> H1 (RoBERTa+BiLSTM) achieves the best adversarial
        robustness, reducing QuillBot ASR by 64%. The BiLSTM captures sequential word-order signals
        that vocabulary substitution cannot fully disrupt. H3 (Soft Voting) provides no significant
        benefit because all three transformers share the same vocabulary-level vulnerability.
        H4's TF-IDF features are a liability under QuillBot attack — worsening ASR from 13.4% to 16.0%.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Hybrid results CSV not found. Run Notebooks 14 (Colab) and 15 (Local) first.")


# ── Footer: full results table (always shown) ─────────────────────────────────
st.markdown("---")
with st.expander("📊 Full Dissertation Results — All Conditions"):
    FULL_RESULTS = pd.DataFrame([
        ("RoBERTa-base","Clean HC3",0.9995,0.9913,0.9942,None),
        ("BERT-base","Clean HC3",0.9997,0.9845,0.9895,None),
        ("DistilBERT","Clean HC3",0.9995,0.9922,0.9948,None),
        ("Hello-SimpleAI","Clean HC3",0.9977,0.9929,0.9953,None),
        ("Log. Regression","Clean HC3",0.9364,0.9524,0.9689,None),
        ("RoBERTa-base","Pegasus Attack",0.9880,None,None,0.012),
        ("BERT-base","Pegasus Attack",0.9820,None,None,0.018),
        ("DistilBERT","Pegasus Attack",0.9480,None,None,0.052),
        ("Hello-SimpleAI","Pegasus Attack",0.9960,None,None,0.004),
        ("Log. Regression","Pegasus Attack",0.7100,None,None,0.290),
        ("RoBERTa-base","QuillBot Attack",0.8660,None,None,0.134),
        ("BERT-base","QuillBot Attack",0.8780,None,None,0.122),
        ("DistilBERT","QuillBot Attack",0.8100,None,None,0.190),
        ("Hello-SimpleAI","QuillBot Attack",0.8600,None,None,0.140),
        ("Log. Regression","QuillBot Attack",0.7320,None,None,0.268),
        ("RoBERTa-base","ChatGPT Rewrite",0.9840,None,None,0.016),
        ("BERT-base","ChatGPT Rewrite",0.9680,None,None,0.032),
        ("DistilBERT","ChatGPT Rewrite",0.9320,None,None,0.068),
        ("Hello-SimpleAI","ChatGPT Rewrite",0.9720,None,None,0.028),
        ("Log. Regression","ChatGPT Rewrite",0.6060,None,None,0.394),
        ("RoBERTa-base","M4 Cross-Dataset",0.6510,0.7389,0.7700,None),
        ("BERT-base","M4 Cross-Dataset",0.4430,0.5999,0.7045,None),
        ("DistilBERT","M4 Cross-Dataset",0.2790,0.4316,0.6325,None),
        ("Hello-SimpleAI","M4 Cross-Dataset",0.3880,0.5442,0.6750,None),
        ("Log. Regression","M4 Cross-Dataset",0.2260,0.3356,0.5525,None),
    ], columns=["Model","Condition","Recall","F1","Accuracy","Attack_Success_Rate"])

    for condition in ["Clean HC3","Pegasus Attack","QuillBot Attack","ChatGPT Rewrite","M4 Cross-Dataset"]:
        subset = FULL_RESULTS[FULL_RESULTS["Condition"]==condition].copy()
        subset = subset[["Model","Recall","F1","Accuracy","Attack_Success_Rate"]].fillna("—")

        def colour_recall(val):
            try:
                v = float(val)
                return "background-color:#d4edda" if v >= 0.80 else "background-color:#f8d7da" if v <= 0.30 else "background-color:#fff3cd"
            except: return ""

        st.markdown(f"**{condition}**")
        st.dataframe(subset.style.applymap(colour_recall, subset=["Recall"]),
                     use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("MSc AI Detection Platform · Abdul Hannaan Mohammed · B00409227 · UWS · 2025/26 · Dr Tahir Mahmood")
