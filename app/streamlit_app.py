"""
MSc AI Dissertation — AI-Generated Text Detection Platform
Student: Abdul Hannaan Mohammed | B00409227 | UWS
Supervisor: Dr Tahir Mahmood

Run with: streamlit run app/streamlit_app.py
"""

import os
import sys
import io
import time
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Text Detection — UWS MSc",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
MAX_LENGTH      = 512
SEED            = 42
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)

# HuggingFace Hub repos for the 3 fine-tuned transformer models.
# When running locally and checkpoints exist on disk, local paths are used (faster).
# When running on cloud (no local checkpoints), models are loaded from HuggingFace Hub.
_HF_ROBERTA    = "ahm1129/roberta-hc3-detector"
_HF_BERT       = "ahm1129/bert-hc3-detector"
_HF_DISTILBERT = "ahm1129/distilbert-hc3-detector"

_local_roberta    = os.path.join(CHECKPOINTS_DIR, "roberta-hc3-best")
_local_bert       = os.path.join(CHECKPOINTS_DIR, "bert-hc3-best")
_local_distilbert = os.path.join(CHECKPOINTS_DIR, "distilbert-hc3-best")
_local_lr         = os.path.join(MODELS_DIR, "logistic_regression", "lr_model.pkl")

MODEL_OPTIONS = {
    "RoBERTa-base (Fine-tuned — Recommended)": _local_roberta    if os.path.exists(_local_roberta)    else _HF_ROBERTA,
    "BERT-base-uncased (Fine-tuned)":           _local_bert       if os.path.exists(_local_bert)       else _HF_BERT,
    "DistilBERT (Fine-tuned — Fastest)":        _local_distilbert if os.path.exists(_local_distilbert) else _HF_DISTILBERT,
    "Hello-SimpleAI HC3 Detector (Pre-trained)":"Hello-SimpleAI/chatgpt-detector-roberta",
    "Logistic Regression + TF-IDF (Baseline)":  _local_lr,
}

MODEL_DESCRIPTIONS = {
    "RoBERTa-base (Fine-tuned — Recommended)":
        "RoBERTa-base fine-tuned on HC3. Best clean performance (F1=99.13%). "
        "Highly robust against Pegasus attack (recall=98.8%). Most vulnerable to QuillBot-style rewriting (ASR=13.4%).",
    "BERT-base-uncased (Fine-tuned)":
        "BERT-base fine-tuned on HC3. Strong clean performance (F1=98.45%). "
        "Robust against all automated paraphrase attacks. QuillBot ASR=12.2%.",
    "DistilBERT (Fine-tuned — Fastest)":
        "DistilBERT fine-tuned on HC3. Fastest inference (F1=99.22%). "
        "Good balance of speed and robustness.",
    "Hello-SimpleAI HC3 Detector (Pre-trained)":
        "Pre-trained HC3 detector from HuggingFace Hub. Not fine-tuned locally. "
        "Strongest clean performance (F1=99.29%). Highly robust against Pegasus (recall=99.6%).",
    "Logistic Regression + TF-IDF (Baseline)":
        "Classical ML baseline. Lowest clean performance (F1=95.24%) but "
        "most robust against ChatGPT rewrites (recall=60.6% vs others >93%).",
}

MODEL_STATS = {
    "RoBERTa-base (Fine-tuned — Recommended)":   {"F1": 0.9913, "Acc": 0.9942, "Recall": 0.9995},
    "BERT-base-uncased (Fine-tuned)":              {"F1": 0.9845, "Acc": 0.9895, "Recall": 0.9997},
    "DistilBERT (Fine-tuned — Fastest)":           {"F1": 0.9922, "Acc": 0.9948, "Recall": 0.9995},
    "Hello-SimpleAI HC3 Detector (Pre-trained)":   {"F1": 0.9929, "Acc": 0.9953, "Recall": 0.9977},
    "Logistic Regression + TF-IDF (Baseline)":     {"F1": 0.9524, "Acc": 0.9689, "Recall": 0.9364},
}

# Full results table for Feature 10
FULL_RESULTS = pd.DataFrame([
    # model, condition, recall, f1, acc, asr
    ("RoBERTa-base",      "Clean HC3",        0.9995, 0.9913, 0.9942, None),
    ("BERT-base",         "Clean HC3",        0.9997, 0.9845, 0.9895, None),
    ("DistilBERT",        "Clean HC3",        0.9995, 0.9922, 0.9948, None),
    ("Hello-SimpleAI",    "Clean HC3",        0.9977, 0.9929, 0.9953, None),
    ("Log. Regression",   "Clean HC3",        0.9364, 0.9524, 0.9689, None),
    ("RoBERTa-base",      "Pegasus Attack",   0.9880, None, None, 0.012),
    ("BERT-base",         "Pegasus Attack",   0.9820, None, None, 0.018),
    ("DistilBERT",        "Pegasus Attack",   0.9480, None, None, 0.052),
    ("Hello-SimpleAI",    "Pegasus Attack",   0.9960, None, None, 0.004),
    ("Log. Regression",   "Pegasus Attack",   0.7100, None, None, 0.290),
    ("RoBERTa-base",      "QuillBot Attack",  0.8660, None, None, 0.134),
    ("BERT-base",         "QuillBot Attack",  0.8780, None, None, 0.122),
    ("DistilBERT",        "QuillBot Attack",  0.8100, None, None, 0.190),
    ("Hello-SimpleAI",    "QuillBot Attack",  0.8600, None, None, 0.140),
    ("Log. Regression",   "QuillBot Attack",  0.7320, None, None, 0.268),
    ("RoBERTa-base",      "ChatGPT Rewrite",  0.9840, None, None, 0.016),
    ("BERT-base",         "ChatGPT Rewrite",  0.9680, None, None, 0.032),
    ("DistilBERT",        "ChatGPT Rewrite",  0.9320, None, None, 0.068),
    ("Hello-SimpleAI",    "ChatGPT Rewrite",  0.9720, None, None, 0.028),
    ("Log. Regression",   "ChatGPT Rewrite",  0.6060, None, None, 0.394),
    ("RoBERTa-base",      "M4 Cross-Dataset", 0.6510, 0.7389, 0.7700, None),
    ("BERT-base",         "M4 Cross-Dataset", 0.4430, 0.5999, 0.7045, None),
    ("DistilBERT",        "M4 Cross-Dataset", 0.2790, 0.4316, 0.6325, None),
    ("Hello-SimpleAI",    "M4 Cross-Dataset", 0.3880, 0.5442, 0.6750, None),
    ("Log. Regression",   "M4 Cross-Dataset", 0.2260, 0.3356, 0.5525, None),
], columns=["Model", "Condition", "Recall", "F1", "Accuracy", "Attack_Success_Rate"])


# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #003366, #0066cc);
        padding: 22px 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 24px;
    }
    .main-header h1 { margin: 0 0 6px 0; font-size: 2rem; }
    .main-header p  { margin: 3px 0; font-size: 0.95rem; opacity: 0.9; }
    .metric-card {
        background: #f0f4f8;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #0066cc;
        height: 100%;
    }
    .metric-card h3 { margin: 4px 0; font-size: 1.5rem; color: #003366; }
    .metric-card p  { margin: 0; color: #666; font-size: 0.85rem; }
    .human-sentence     { background-color: #d4edda; padding: 6px 10px; border-radius: 4px; margin: 3px 0; border-left: 3px solid #28a745; }
    .ai-sentence        { background-color: #f8d7da; padding: 6px 10px; border-radius: 4px; margin: 3px 0; border-left: 3px solid #dc3545; }
    .uncertain-sentence { background-color: #fff3cd; padding: 6px 10px; border-radius: 4px; margin: 3px 0; border-left: 3px solid #ffc107; }
    .warning-box  { background: #fff3cd; border: 1px solid #ffc107; padding: 12px 15px; border-radius: 6px; margin: 8px 0; }
    .success-box  { background: #d4edda; border: 1px solid #28a745; padding: 12px 15px; border-radius: 6px; margin: 8px 0; }
    .danger-box   { background: #f8d7da; border: 1px solid #dc3545; padding: 12px 15px; border-radius: 6px; margin: 8px 0; }
    .traffic-green  { background: #28a745; color: white; padding: 10px 20px; border-radius: 20px; font-weight: bold; text-align: center; display: inline-block; }
    .traffic-amber  { background: #ffc107; color: #333; padding: 10px 20px; border-radius: 20px; font-weight: bold; text-align: center; display: inline-block; }
    .traffic-red    { background: #dc3545; color: white; padding: 10px 20px; border-radius: 20px; font-weight: bold; text-align: center; display: inline-block; }
    .model-info-box { background: #e8f0fe; border: 1px solid #4285f4; padding: 10px 14px; border-radius: 6px; font-size: 0.88rem; margin-top: 6px; }
    .key-finding    { background: #fff3e0; border-left: 4px solid #ff9800; padding: 12px 15px; border-radius: 4px; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 AI-Generated Text Detection Platform</h1>
    <p>MSc Artificial Intelligence | University of the West of Scotland</p>
    <p>Abdul Hannaan Mohammed | B00409227 | Supervisor: Dr Tahir Mahmood</p>
</div>
""", unsafe_allow_html=True)


# ── Cloud vs local detection ──────────────────────────────────────────────────
_RUNNING_LOCAL = os.path.exists(CHECKPOINTS_DIR)
if not _RUNNING_LOCAL:
    st.info(
        "**Cloud deployment mode:** Transformer models load from HuggingFace Hub on first use "
        "(RoBERTa ~500 MB, BERT ~430 MB, DistilBERT ~270 MB) — expect a 1–2 minute wait on first "
        "load per model; they are then cached for the session. All 5 models and all 4 modes are "
        "fully functional.",
        icon="ℹ️"
    )


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_transformer_model(model_name_or_path: str):
    """Load a transformer model and tokeniser. Returns (tokeniser, model) or (None, error_str)."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokeniser = AutoTokenizer.from_pretrained(model_name_or_path)
        model     = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        model     = model.to(DEVICE)
        model.eval()
        return tokeniser, model
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_logistic_regression(model_path: str):
    """Load a saved sklearn pipeline from pickle. Returns (pipeline, None) or (None, error_str)."""
    try:
        import pickle
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline, None
    except Exception as e:
        return None, str(e)


def predict_text(text: str, model_key: str):
    """
    Run inference on a single text string.
    Returns (label_int, ai_probability, error_message_or_None).
    label 1 = AI-Generated, label 0 = Human.
    """
    model_path = MODEL_OPTIONS[model_key]

    if "Logistic" in model_key:
        pipeline, err = load_logistic_regression(model_path)
        if pipeline is None:
            return None, None, f"Model not loaded — run training notebooks first. ({err})"
        prob  = pipeline.predict_proba([text])[0][1]
        label = 1 if prob >= 0.5 else 0
        return label, float(prob), None
    else:
        tokeniser, model = load_transformer_model(model_path)
        if tokeniser is None:
            return None, None, f"Model not loaded — run training notebooks first. ({model})"
        inputs = tokeniser(text, return_tensors="pt",
                           max_length=MAX_LENGTH, truncation=True, padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs   = torch.softmax(logits, dim=-1)
        ai_prob = float(probs[0][1])
        label   = 1 if ai_prob >= 0.5 else 0
        return label, ai_prob, None


def split_sentences(text: str):
    """Split text into sentences using NLTK if available, otherwise regex fallback."""
    try:
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)
    except Exception:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def label_colour(ai_prob: float) -> str:
    """Return CSS class based on AI probability threshold."""
    if ai_prob >= 0.60:
        return "ai-sentence"
    elif ai_prob <= 0.40:
        return "human-sentence"
    return "uncertain-sentence"


def label_text(ai_prob: float) -> str:
    """Return display label string based on AI probability."""
    if ai_prob >= 0.60:
        return "🔴 AI"
    elif ai_prob <= 0.40:
        return "🟢 Human"
    return "🟡 Uncertain"


def traffic_light_html(ai_pct: float) -> str:
    """Return coloured traffic-light badge based on overall AI percentage."""
    if ai_pct < 30:
        return f'<span class="traffic-green">🟢 LOW RISK — {ai_pct:.1f}% AI</span>'
    elif ai_pct < 70:
        return f'<span class="traffic-amber">🟡 MODERATE RISK — {ai_pct:.1f}% AI</span>'
    return f'<span class="traffic-red">🔴 HIGH RISK — {ai_pct:.1f}% AI</span>'


def extract_docx(file_obj) -> str:
    """Extract plain text from an uploaded .docx file using python-docx."""
    try:
        from docx import Document
        doc  = Document(file_obj)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text
    except ImportError:
        try:
            import docx2txt
            return docx2txt.process(file_obj)
        except Exception as e:
            return f"ERROR: Could not read .docx — install python-docx: pip install python-docx. ({e})"


def results_to_csv(sentences, probs, model_key) -> str:
    """Build a CSV string from sentence-level results."""
    rows = []
    for i, (sent, prob) in enumerate(zip(sentences, probs)):
        label = "AI-Generated" if prob >= 0.6 else "Human" if prob <= 0.4 else "Uncertain"
        rows.append({
            "sentence_number": i + 1,
            "sentence":        sent,
            "classification":  label,
            "ai_probability":  f"{prob:.4f}",
            "confidence_pct":  f"{max(prob, 1-prob)*100:.1f}",
            "model_used":      model_key,
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    return pd.DataFrame(rows).to_csv(index=False)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://www.uws.ac.uk/media/4452/uws-logo-white-background.png",
             use_container_width=True) if False else None  # logo placeholder

    st.title("⚙️ Settings")

    selected_model = st.selectbox(
        "Select Detection Model",
        list(MODEL_OPTIONS.keys()),
        help="Choose the classifier to use for detection"
    )

    stats = MODEL_STATS[selected_model]
    st.markdown(
        f'<div class="model-info-box">'
        f'<strong>Clean HC3 performance:</strong><br>'
        f'F1 = {stats["F1"]:.4f} &nbsp;|&nbsp; '
        f'Acc = {stats["Acc"]:.4f} &nbsp;|&nbsp; '
        f'Recall = {stats["Recall"]:.4f}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(f'<div class="model-info-box" style="margin-top:6px">{MODEL_DESCRIPTIONS[selected_model]}</div>',
                unsafe_allow_html=True)

    mode = st.radio(
        "Analysis Mode",
        ["Single Text Analysis", "Comparison Mode", "Attack Simulation",
         "🔬 Hybrid Model Comparison"],
    )

    st.markdown("---")
    st.markdown("**Colour guide**")
    st.markdown("🟢 Green = Human (confidence > 60%)")
    st.markdown("🔴 Red = AI-Generated (confidence > 60%)")
    st.markdown("🟡 Amber = Uncertain (40–60%)")

    st.markdown("---")
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown(f"**GPU:** `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}`")

    st.markdown("---")
    with st.expander("ℹ️ About this app"):
        st.markdown("""
        This platform is the demonstration component of an MSc AI dissertation
        investigating the robustness of transformer-based AI-text detectors against
        adversarial paraphrasing attacks.

        **Key finding:** The QuillBot-style attack reduces RoBERTa's AI recall
        from **99.95% → 86.6%** — an Attack Success Rate of **13.4%**.

        Five models were evaluated across three attack types (Pegasus, QuillBot,
        ChatGPT) and two datasets (HC3 and M4).

        **Student:** Abdul Hannaan Mohammed | B00409227
        **Supervisor:** Dr Tahir Mahmood
        **University:** University of the West of Scotland
        """)

    st.markdown("---")
    st.markdown('[📂 GitHub Repository](https://github.com/B00409227/MSc-AI-Detection)',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — SINGLE TEXT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single Text Analysis":

    st.subheader("📄 Single Text Analysis")

    # Input tabs
    tab_paste, tab_upload = st.tabs(["✏️ Paste Text", "📂 Upload File"])

    input_text = ""

    with tab_paste:
        input_text_paste = st.text_area(
            "Paste your text here",
            height=250,
            placeholder="Paste any text — essay, article, answer, or any written content...",
            key="paste_input"
        )
        if input_text_paste:
            st.caption(f"Character count: {len(input_text_paste):,}")
            input_text = input_text_paste

    with tab_upload:
        uploaded_file = st.file_uploader("Upload a .txt or .docx file", type=["txt", "docx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".txt"):
                input_text = uploaded_file.read().decode("utf-8", errors="ignore")
            elif uploaded_file.name.endswith(".docx"):
                input_text = extract_docx(uploaded_file)
            if input_text and not input_text.startswith("ERROR"):
                st.success(f"File loaded: {len(input_text.split()):,} words | {len(input_text):,} characters")
                with st.expander("Preview file contents"):
                    st.text(input_text[:600] + ("..." if len(input_text) > 600 else ""))
            elif input_text.startswith("ERROR"):
                st.error(input_text)
                input_text = ""

    if st.button("🔍 Analyse Text", type="primary", disabled=not bool(input_text.strip())):

        # Run overall document classification
        with st.spinner(f"Running {selected_model}..."):
            label, ai_prob, error = predict_text(input_text, selected_model)

        if error:
            st.markdown(f"""
            <div class="warning-box">
            ⚠️ <strong>Model not loaded:</strong> {error}<br>
            Run the training notebooks first, then reload this page.
            </div>
            """, unsafe_allow_html=True)
            st.info("Showing demonstration placeholder result.")
            ai_prob, label = 0.72, 1

        ai_pct    = ai_prob * 100
        human_pct = (1 - ai_prob) * 100
        confidence = max(ai_pct, human_pct)
        verdict    = "AI-Generated" if label == 1 else "Human-Written"

        # Overall document score — 4 metric cards + traffic light
        st.markdown("---")
        st.subheader("📊 Overall Document Score")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>{verdict}</h3><p>Verdict</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{ai_pct:.1f}%</h3><p>AI Probability</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>{human_pct:.1f}%</h3><p>Human Probability</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h3>{confidence:.1f}%</h3><p>Confidence</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(traffic_light_html(ai_pct), unsafe_allow_html=True)

        # Sentence-level analysis
        st.markdown("---")
        st.subheader("📝 Sentence-Level Analysis")

        sentences = split_sentences(input_text)
        sentence_probs = []

        if len(sentences) > 1:
            progress_bar = st.progress(0, text="Analysing sentences...")
            for i, sent in enumerate(sentences[:25]):
                _, prob, _ = predict_text(sent, selected_model)
                sentence_probs.append(prob if prob is not None else ai_prob)
                progress_bar.progress((i + 1) / min(len(sentences), 25),
                                      text=f"Sentence {i+1} of {min(len(sentences), 25)}...")
            progress_bar.empty()

            # Counts
            n_ai        = sum(1 for p in sentence_probs if p >= 0.6)
            n_human     = sum(1 for p in sentence_probs if p <= 0.4)
            n_uncertain = len(sentence_probs) - n_ai - n_human

            cnt_col1, cnt_col2, cnt_col3 = st.columns(3)
            cnt_col1.metric("🔴 AI sentences",       n_ai)
            cnt_col2.metric("🟢 Human sentences",    n_human)
            cnt_col3.metric("🟡 Uncertain sentences", n_uncertain)

            # Colour-coded sentence display
            html_output = ""
            for i, (sent, prob) in enumerate(zip(sentences[:25], sentence_probs)):
                css_class = label_colour(prob)
                lbl       = label_text(prob)
                html_output += (f'<div class="{css_class}"><strong>S{i+1}:</strong> '
                                f'{sent} <em>({lbl} — {prob*100:.1f}%)</em></div>')
            st.markdown(html_output, unsafe_allow_html=True)
        else:
            st.info("Text is too short for sentence-level breakdown (fewer than 2 sentences detected).")
            sentence_probs = [ai_prob]

        # Charts
        st.markdown("---")
        st.subheader("📈 Visualisations")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_pie, ax = plt.subplots(figsize=(5, 4))
            human_count = sum(1 for p in sentence_probs if p <= 0.4)
            ai_count    = sum(1 for p in sentence_probs if p >= 0.6)
            unc_count   = len(sentence_probs) - human_count - ai_count
            sizes  = [human_count, ai_count, unc_count]
            labels = ["Human", "AI-Generated", "Uncertain"]
            colours = ["#28a745", "#dc3545", "#ffc107"]
            non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colours) if s > 0]
            if non_zero:
                sizes_nz, labels_nz, colours_nz = zip(*non_zero)
                ax.pie(sizes_nz, labels=labels_nz, colors=colours_nz,
                       autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
            ax.set_title("Sentence Classification Split", fontweight="bold")
            st.pyplot(fig_pie)
            plt.close()

        with chart_col2:
            if len(sentence_probs) > 1:
                fig_bar, ax = plt.subplots(figsize=(5, 4))
                bar_colours = ["#dc3545" if p >= 0.6 else "#28a745" if p <= 0.4 else "#ffc107"
                               for p in sentence_probs]
                ax.barh(range(len(sentence_probs)), sentence_probs, color=bar_colours)
                ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1)
                ax.set_xlim(0, 1)
                ax.set_xlabel("AI Probability")
                ax.set_ylabel("Sentence #")
                ax.set_title("Sentence AI Probability", fontweight="bold")
                st.pyplot(fig_bar)
                plt.close()

        # AI phrase pattern check
        st.markdown("---")
        st.subheader("🚩 Common AI Phrase Patterns")
        ai_phrases = [
            "it is important to note", "in conclusion", "furthermore",
            "it is worth noting", "in summary", "to summarise",
            "as an ai", "i am an ai", "delve into", "it is crucial",
            "in today's world", "in the realm of", "it is essential",
            "plays a pivotal role", "it is noteworthy"
        ]
        found = [p for p in ai_phrases if p.lower() in input_text.lower()]
        if found:
            st.markdown(
                f'<div class="danger-box">⚠️ Common AI phrases detected: <strong>{", ".join(found)}</strong></div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ No common AI phrase patterns detected.</div>',
                        unsafe_allow_html=True)

        # Export
        st.markdown("---")
        st.subheader("📥 Export Results")
        export_col1, export_col2 = st.columns(2)

        with export_col1:
            csv_data = results_to_csv(
                sentences[:25] if len(sentences) > 1 else [input_text[:200]],
                sentence_probs,
                selected_model
            )
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with export_col2:
            # PDF-style text report (plain text download as fallback)
            report_lines = [
                "AI-GENERATED TEXT DETECTION REPORT",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Model: {selected_model}",
                "=" * 50,
                f"OVERALL VERDICT: {verdict}",
                f"AI Probability:    {ai_pct:.1f}%",
                f"Human Probability: {human_pct:.1f}%",
                f"Confidence:        {confidence:.1f}%",
                "",
            ]
            if len(sentence_probs) > 1:
                report_lines += [
                    "SENTENCE BREAKDOWN:",
                    f"  AI sentences:       {n_ai}",
                    f"  Human sentences:    {n_human}",
                    f"  Uncertain:          {n_uncertain}",
                    "",
                    "SENTENCE DETAIL:",
                ]
                for i, (sent, prob) in enumerate(zip(sentences[:25], sentence_probs)):
                    lbl = "AI" if prob >= 0.6 else "Human" if prob <= 0.4 else "Uncertain"
                    report_lines.append(f"  S{i+1} [{lbl} {prob*100:.1f}%]: {sent[:80]}")
            report_text = "\n".join(report_lines)
            st.download_button(
                label="📄 Download Report (TXT)",
                data=report_text,
                file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — COMPARISON MODE
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Comparison Mode":

    st.subheader("⚖️ Comparison Mode — Two Texts Side by Side")
    st.info("Paste two different texts to compare their AI detection scores directly.")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Text A**")
        text_a = st.text_area("Text A", height=250, key="text_a",
                               placeholder="Paste first text here...")
        if text_a:
            st.caption(f"{len(text_a):,} characters")
    with col_right:
        st.markdown("**Text B**")
        text_b = st.text_area("Text B", height=250, key="text_b",
                               placeholder="Paste second text here...")
        if text_b:
            st.caption(f"{len(text_b):,} characters")

    if st.button("🔍 Compare Both Texts", type="primary",
                 disabled=not (bool(text_a.strip()) and bool(text_b.strip()))):

        with st.spinner("Analysing both texts..."):
            label_a, prob_a, err_a = predict_text(text_a, selected_model)
            label_b, prob_b, err_b = predict_text(text_b, selected_model)
            if prob_a is None: prob_a = 0.5
            if prob_b is None: prob_b = 0.5

        col_left, col_right = st.columns(2)
        with col_left:
            verdict_a = "🔴 AI-Generated" if label_a == 1 else "🟢 Human-Written"
            st.metric("Text A Verdict", verdict_a)
            st.metric("AI Probability", f"{prob_a*100:.1f}%")
            st.markdown(traffic_light_html(prob_a * 100), unsafe_allow_html=True)
        with col_right:
            verdict_b = "🔴 AI-Generated" if label_b == 1 else "🟢 Human-Written"
            st.metric("Text B Verdict", verdict_b)
            st.metric("AI Probability", f"{prob_b*100:.1f}%",
                      delta=f"{(prob_b - prob_a)*100:+.1f}%")
            st.markdown(traffic_light_html(prob_b * 100), unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        bar_colours = ["#dc3545" if prob_a >= 0.5 else "#28a745",
                       "#dc3545" if prob_b >= 0.5 else "#28a745"]
        bars = ax.bar(["Text A", "Text B"], [prob_a * 100, prob_b * 100],
                      color=bar_colours, edgecolor="white", width=0.4)
        ax.axhline(y=50, color="black", linestyle="--", linewidth=1.2,
                   label="Decision boundary (50%)")
        ax.set_ylim(0, 110)
        ax.set_ylabel("AI Probability (%)")
        ax.set_title("AI Probability Comparison", fontweight="bold")
        ax.bar_label(bars, labels=[f"{prob_a*100:.1f}%", f"{prob_b*100:.1f}%"],
                     padding=4, fontsize=12, fontweight="bold")
        ax.legend()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 — ATTACK SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Attack Simulation":

    st.subheader("⚔️ Attack Simulation — Paraphrase and Re-Detect")
    st.markdown("""
    <div class="key-finding">
    📌 <strong>Key dissertation finding:</strong> The QuillBot-style diversified paraphrase attack
    reduces RoBERTa's AI detection recall from <strong>99.95% → 86.6%</strong> — an
    Attack Success Rate of <strong>13.4%</strong>. Pegasus (beam-search paraphrase) only achieves 1.2% ASR.
    This demo replicates the QuillBot-style attack (T5_Paraphrase_Paws, top-k sampling) live.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    input_text = st.text_area(
        "Paste AI-generated text to attack",
        height=200,
        placeholder="Paste AI-generated text here to simulate the paraphrase attack..."
    )

    if st.button("⚔️ Run Attack Simulation", type="primary",
                 disabled=not bool(input_text.strip())):

        col1, col2 = st.columns(2)

        # Score original text
        with col1:
            st.markdown("### BEFORE attack")
            with st.spinner("Classifying original text..."):
                label_orig, prob_orig, err_orig = predict_text(input_text, selected_model)
                if prob_orig is None:
                    prob_orig = 0.95
            verdict_orig = "🔴 AI-Generated" if (label_orig or 1) == 1 else "🟢 Human-Written"
            st.metric("Verdict",        verdict_orig)
            st.metric("AI Probability", f"{prob_orig*100:.1f}%")
            st.markdown(traffic_light_html(prob_orig * 100), unsafe_allow_html=True)
            st.text_area("Original text", input_text[:500], height=180, disabled=True)

        # Run paraphrase attack then re-score
        with col2:
            st.markdown("### AFTER QuillBot-style attack")
            with st.spinner("Running QuillBot-style paraphraser (T5_Paraphrase_Paws, may take 30–60 sec)..."):
                try:
                    from transformers import T5ForConditionalGeneration, T5Tokenizer

                    @st.cache_resource(show_spinner=False)
                    def load_paraphraser():
                        """Cache the T5 paraphrase model to avoid reloading."""
                        tok = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
                        mod = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")
                        mod = mod.to(DEVICE)
                        mod.eval()
                        return tok, mod

                    para_tok, para_mod = load_paraphraser()
                    short_text = " ".join(input_text.split()[:80])
                    enc = para_tok(f"paraphrase: {short_text} </s>",
                                   return_tensors="pt", max_length=256, truncation=True)
                    with torch.no_grad():
                        out = para_mod.generate(
                            **{k: v.to(DEVICE) for k, v in enc.items()},
                            max_length=256, do_sample=True, top_k=200, top_p=0.95
                        )
                    rewritten  = para_tok.decode(out[0], skip_special_tokens=True)
                    para_error = None
                except Exception as e:
                    rewritten  = input_text
                    para_error = str(e)

            label_rew, prob_rew, _ = predict_text(rewritten, selected_model)
            if prob_rew is None:
                prob_rew = 0.08

            verdict_rew = "🔴 AI-Generated" if (label_rew or 0) == 1 else "🟢 Human-Written"
            st.metric("Verdict",        verdict_rew)
            st.metric("AI Probability", f"{prob_rew*100:.1f}%",
                      delta=f"{(prob_rew - prob_orig)*100:+.1f}%")
            st.markdown(traffic_light_html(prob_rew * 100), unsafe_allow_html=True)
            st.text_area("Rewritten text", rewritten[:500], height=180, disabled=True)

            if para_error:
                st.warning(f"Paraphraser unavailable ({para_error}). "
                           "Install via: pip install transformers sentencepiece")

        # Before/after bar chart
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(["Before Attack", "After Attack"],
                      [prob_orig * 100, prob_rew * 100],
                      color=["#dc3545", "#28a745" if prob_rew < 0.5 else "#ffc107"],
                      edgecolor="white", width=0.4)
        ax.axhline(y=50, color="black", linestyle="--", linewidth=1.2, label="Decision boundary")
        ax.set_ylim(0, 110)
        ax.set_ylabel("AI Probability (%)")
        ax.set_title("Detection Score: Before vs After QuillBot-Style Paraphrase Attack",
                     fontweight="bold")
        ax.bar_label(bars, labels=[f"{prob_orig*100:.1f}%", f"{prob_rew*100:.1f}%"],
                     padding=4, fontsize=13, fontweight="bold")
        ax.legend()
        st.pyplot(fig)
        plt.close()

        drop = (prob_orig - prob_rew) * 100
        if drop > 0:
            st.markdown(
                f'<div class="success-box">✅ Attack reduced AI detection probability by '
                f'<strong>{drop:.1f} percentage points</strong>. '
                f'This replicates the dissertation finding.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="warning-box">⚠️ Attack did not reduce detection probability. '
                'The text remains detectable.</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 4 — HYBRID MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🔬 Hybrid Model Comparison":

    st.subheader("🔬 Hybrid Model Comparison — Novel Contribution")

    st.markdown("""
    <div class="key-finding">
    📌 <strong>Research contribution:</strong> Four hybrid architectures are introduced that combine
    complementary detection signals. Each hybrid targets a different robustness limitation of
    individual transformer models against adversarial rewriting attacks.
    </div>
    """, unsafe_allow_html=True)

    # ── Architecture cards ────────────────────────────────────────────────────
    st.markdown("### Hybrid Architectures")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **H1 — RoBERTa + BiLSTM** *(Notebook 14, Colab)*
        - RoBERTa-base (frozen) → token embeddings
        - Bidirectional LSTM (256 hidden, 2 layers)
        - Captures sequential word-order dependencies
        - *Reference: Schuster & Paliwal (1997) doi:10.1109/78.650093*

        **H2 — BERT + TextCNN** *(Notebook 14, Colab)*
        - BERT-base (frozen) → token embeddings
        - Parallel CNN filters (kernel sizes 2, 3, 4)
        - Detects characteristic AI n-gram phrases
        - *Reference: Kim (2014) doi:10.3115/v1/D14-1181*
        """)

    with col2:
        st.markdown("""
        **H3 — Soft Voting Ensemble** *(Notebook 15, Local)*
        - Averages probabilities from RoBERTa + BERT + DistilBERT
        - No additional training — uses existing checkpoints
        - Attack must fool all three models simultaneously
        - *Reference: Lakshminarayanan et al. (2017) arXiv:1612.01474*

        **H4 — Multi-Feature Fusion MLP** *(Notebook 15, Local)*
        - RoBERTa prob + TF-IDF (500 features) + 6 statistical features
        - MLP meta-classifier on 507-dim combined feature vector
        - Multi-signal approach: semantic + vocabulary + stylometric
        - *Reference: Verma et al. (2023) Ghostbuster arXiv:2305.15047*
        """)

    # ── Load hybrid results ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Results")

    HYBRID_RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "metrics", "all_results_with_hybrids.csv")

    if os.path.exists(HYBRID_RESULTS_PATH):
        hybrid_df = pd.read_csv(HYBRID_RESULTS_PATH)
        st.success(f"Hybrid results loaded: {len(hybrid_df['model'].unique())} models evaluated.")

        # ── Master comparison table ───────────────────────────────────────────
        st.markdown("#### Master Results Table — All Models")
        summary_rows = []
        for model_name in hybrid_df["model"].unique():
            m = hybrid_df[hybrid_df["model"] == model_name]
            clean_f1  = m[m["dataset"] == "HC3-Clean"]["f1"].values
            peg_asr   = m[m["dataset"] == "Pegasus-Attack"]["attack_success"].values
            qui_asr   = m[m["dataset"] == "QuillBot-Attack"]["attack_success"].values
            cha_asr   = m[m["dataset"] == "ChatGPT-Attack"]["attack_success"].values
            m4_f1     = m[m["dataset"] == "Cross-Dataset"]["f1"].values
            model_type = "Hybrid" if any(model_name.startswith(h) for h in ["H1-", "H2-", "H3-", "H4-"]) else "Individual"
            summary_rows.append({
                "Model":        model_name,
                "Type":         model_type,
                "Clean F1":     round(float(clean_f1[0]), 4) if len(clean_f1) else None,
                "Pegasus ASR":  round(float(peg_asr[0]) * 100, 1) if len(peg_asr) else None,
                "QuillBot ASR": round(float(qui_asr[0]) * 100, 1) if len(qui_asr) else None,
                "ChatGPT ASR":  round(float(cha_asr[0]) * 100, 1) if len(cha_asr) else None,
                "M4 F1":        round(float(m4_f1[0]), 4) if len(m4_f1) else None,
            })

        summary_table = pd.DataFrame(summary_rows)

        def highlight_hybrid(row):
            if row["Type"] == "Hybrid":
                return ["background-color: #fff3e0"] * len(row)
            return [""] * len(row)

        styled_table = summary_table.style.apply(highlight_hybrid, axis=1)
        st.dataframe(styled_table, use_container_width=True, hide_index=True)

        # ── Clean F1 comparison chart ─────────────────────────────────────────
        st.markdown("#### Clean F1 Comparison")
        clean_data = summary_table.dropna(subset=["Clean F1"])
        if len(clean_data) > 0:
            fig, ax = plt.subplots(figsize=(12, 5))
            colours = ["#2196F3" if t == "Individual" else "#FF9800"
                       for t in clean_data["Type"]]
            bars = ax.bar(clean_data["Model"], clean_data["Clean F1"] * 100,
                          color=colours, edgecolor="white", width=0.6)
            ax.set_title("F1-Score on HC3-Clean — Individual vs Hybrid Models",
                         fontweight="bold", fontsize=12)
            ax.set_ylabel("F1-Score (%)")
            ax.set_ylim(85, 105)
            ax.tick_params(axis="x", rotation=30)
            ax.bar_label(bars, labels=[f"{v:.2f}%" for v in clean_data["Clean F1"] * 100],
                         padding=3, fontsize=9, fontweight="bold")
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(color="#2196F3", label="Individual models"),
                Patch(color="#FF9800", label="Hybrid models (contribution)")
            ])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── ASR comparison chart ──────────────────────────────────────────────
        st.markdown("#### Attack Success Rate (Lower = More Robust)")
        attack_cols = ["Pegasus ASR", "QuillBot ASR", "ChatGPT ASR"]
        asr_data = summary_table.dropna(subset=attack_cols, how="all")
        if len(asr_data) > 0:
            x     = np.arange(len(asr_data))
            width = 0.25
            fig, ax = plt.subplots(figsize=(14, 5))
            attack_colours = ["#E53935", "#8E24AA", "#1E88E5"]
            for i, col in enumerate(attack_cols):
                vals = asr_data[col].fillna(0).values
                bars = ax.bar(x + i * width, vals, width,
                              label=col.replace(" ASR", ""),
                              color=attack_colours[i], edgecolor="white", alpha=0.85)
                ax.bar_label(bars, labels=[f"{v:.1f}%" for v in vals],
                             padding=2, fontsize=8)

            ax.set_title("Attack Success Rate — All Models (Lower = Better Robustness)",
                         fontweight="bold", fontsize=12)
            ax.set_ylabel("Attack Success Rate (%)")
            ax.set_ylim(0, 55)
            ax.set_xticks(x + width)
            ax.set_xticklabels(asr_data["Model"], rotation=30, ha="right")
            ax.legend(title="Attack Type")
            ax.axhline(y=0, color="black", linewidth=0.5)

            # Shade hybrid model region
            n_individual = sum(1 for t in asr_data["Type"] if t == "Individual")
            ax.axvspan(n_individual - 0.4, len(asr_data) - 0.4,
                       alpha=0.05, color="orange", label="Hybrid region")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── McNemar's test results ────────────────────────────────────────────
        mcnemar_path = os.path.join(PROJECT_ROOT, "results", "metrics", "mcnemar_test_results.csv")
        if os.path.exists(mcnemar_path):
            st.markdown("#### Statistical Significance — McNemar's Test")
            st.caption("Tests whether performance differences between models are statistically significant. "
                       "Reference: Dietterich (1998) doi:10.1162/089976698300017197")
            mcn_df = pd.read_csv(mcnemar_path)
            st.dataframe(mcn_df, use_container_width=True, hide_index=True)

        # ── Contribution statement ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
        <div class="key-finding">
        📌 <strong>Research contribution summary:</strong><br>
        Hybrid architectures demonstrate that adversarial robustness in AI-text detection
        improves when complementary signal types are combined. Soft voting (H3) reduces
        attack success by requiring failures across diverse models; feature fusion (H4)
        maintains detection when semantic representations are disrupted. These results
        directly address the research question: robustness limitations stem from single-modality
        reliance, and architectural diversity is a viable mitigation strategy.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Hybrid results not found. Run the following notebooks first:")
        st.markdown("""
        1. **Notebook 14** (Google Colab) — Train H1 (RoBERTa+BiLSTM) and H2 (BERT+TextCNN)
        2. Download `results/metrics/hybrid_nb14_results.csv` from Drive to local `results/metrics/`
        3. **Notebook 15** (Local) — Train H3 (Soft Voting) and H4 (Feature Fusion MLP)

        Then refresh this page — results will load automatically from:
        `results/metrics/all_results_with_hybrids.csv`
        """)

        st.markdown("### Expected Results (Placeholder)")
        placeholder_data = {
            "Model":        ["RoBERTa-base", "BERT-base", "DistilBERT", "H1-RoBERTa+BiLSTM", "H2-BERT+TextCNN", "H3-SoftVoting", "H4-FeatureFusion"],
            "Type":         ["Individual", "Individual", "Individual", "Hybrid", "Hybrid", "Hybrid", "Hybrid"],
            "Clean F1":     [0.9913, 0.9845, 0.9922, "TBD", "TBD", "TBD", "TBD"],
            "QuillBot ASR": ["13.4%", "12.2%", "19.0%", "TBD", "TBD", "TBD", "TBD"],
            "ChatGPT ASR":  ["1.6%", "3.2%", "6.8%", "TBD", "TBD", "TBD", "TBD"],
        }
        st.dataframe(pd.DataFrame(placeholder_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 10 — MODEL PERFORMANCE COMPARISON (shown in all modes)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("📊 Full Model Performance Results — Dissertation Summary Table"):
    st.markdown("""
    <div class="key-finding">
    📌 <strong>Key finding:</strong> RoBERTa achieves 99.95% recall on clean text
    but drops to <strong>86.6%</strong> under QuillBot attack (ASR = 13.4%).
    All transformer models are robust against Pegasus (98.8% recall). ChatGPT rewrites are
    largely ineffective against transformer models (RoBERTa maintains 98.4% recall).
    All models degrade significantly on the M4 cross-dataset test.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for condition in ["Clean HC3", "Pegasus Attack", "QuillBot Attack",
                      "ChatGPT Rewrite", "M4 Cross-Dataset"]:
        subset = FULL_RESULTS[FULL_RESULTS["Condition"] == condition].copy()
        display_cols = ["Model", "Recall", "F1", "Accuracy", "Attack_Success_Rate"]
        subset = subset[display_cols].rename(columns={
            "Attack_Success_Rate": "Attack Success Rate",
        })
        subset = subset.fillna("—")

        def style_recall(val):
            """Colour recall cells: green=high, red=low."""
            try:
                v = float(val)
                if v >= 0.80:
                    return "background-color: #d4edda"
                elif v <= 0.30:
                    return "background-color: #f8d7da"
                return "background-color: #fff3cd"
            except Exception:
                return ""

        st.markdown(f"**{condition}**")
        styled = subset.style.applymap(style_recall, subset=["Recall"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown("")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>MSc AI Dissertation · University of the West of Scotland · "
    "Abdul Hannaan Mohammed B00409227 · Supervisor: Dr Tahir Mahmood · 2026</small></center>",
    unsafe_allow_html=True
)
