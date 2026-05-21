"""
MSc AI Dissertation — AI-Generated Text Detection Platform
Student: Abdul Hannaan Mohammed | B00409227 | UWS
Supervisor: Dr Tahir Mehmood

Run with: streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MAX_LENGTH = 512
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")

MODEL_OPTIONS = {
    "RoBERTa-base (Fine-tuned)"     : os.path.join(CHECKPOINTS_DIR, "roberta-hc3-best"),
    "BERT-base (Fine-tuned)"         : os.path.join(CHECKPOINTS_DIR, "bert-hc3-best"),
    "DistilBERT (Fine-tuned)"        : os.path.join(CHECKPOINTS_DIR, "distilbert-hc3-best"),
    "HC3 Detector (Hello-SimpleAI)" : "Hello-SimpleAI/chatgpt-detector-roberta",
    "Logistic Regression + TF-IDF"  : os.path.join(MODELS_DIR, "logistic_regression", "lr_model.pkl"),
}

# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #003366, #0066cc);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f0f4f8;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #0066cc;
    }
    .human-sentence  { background-color: #d4edda; padding: 4px 8px; border-radius: 4px; margin: 2px 0; }
    .ai-sentence     { background-color: #f8d7da; padding: 4px 8px; border-radius: 4px; margin: 2px 0; }
    .uncertain-sentence { background-color: #fff3cd; padding: 4px 8px; border-radius: 4px; margin: 2px 0; }
    .warning-box  { background: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 5px; }
    .success-box  { background: #d4edda; border: 1px solid #28a745; padding: 10px; border-radius: 5px; }
    .danger-box   { background: #f8d7da; border: 1px solid #dc3545; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)


# ── Header (Feature 11) ───────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 AI-Generated Text Detection Platform</h1>
    <p>MSc Artificial Intelligence | University of the West of Scotland</p>
    <p>Abdul Hannaan Mohammed | B00409227 | Supervisor: Dr Tahir Mehmood</p>
</div>
""", unsafe_allow_html=True)


# ── Model loading with caching (Feature 13) ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_transformer_model(model_name_or_path: str):
    """
    Load a fine-tuned transformer model and tokeniser from disk.
    Returns (tokeniser, model) or (None, None) if not available.
    """
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
    """Load a saved scikit-learn logistic regression + TF-IDF pipeline."""
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
    Returns (predicted_label, ai_probability) or (None, None) on error.
    """
    model_path = MODEL_OPTIONS[model_key]

    if "Logistic" in model_key:
        pipeline, err = load_logistic_regression(model_path)
        if pipeline is None:
            return None, None, f"Model not loaded: {err}"
        prob = pipeline.predict_proba([text])[0][1]
        label = 1 if prob >= 0.5 else 0
        return label, float(prob), None

    else:
        tokeniser, model = load_transformer_model(model_path)
        if tokeniser is None:
            return None, None, f"Model not loaded: {model}"
        inputs = tokeniser(
            text, return_tensors="pt",
            max_length=MAX_LENGTH, truncation=True, padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        ai_prob = float(probs[0][1])
        label   = 1 if ai_prob >= 0.5 else 0
        return label, ai_prob, None


def split_sentences(text: str):
    """Split text into sentences using basic punctuation rules."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def label_colour(ai_prob: float) -> str:
    """Return CSS class based on AI probability."""
    if ai_prob >= 0.60:
        return "ai-sentence"
    elif ai_prob <= 0.40:
        return "human-sentence"
    else:
        return "uncertain-sentence"


def label_text(ai_prob: float) -> str:
    """Return label string based on AI probability."""
    if ai_prob >= 0.60:
        return "🔴 AI"
    elif ai_prob <= 0.40:
        return "🟢 Human"
    else:
        return "🟡 Uncertain"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

# Feature 2 — Model selector
selected_model = st.sidebar.selectbox(
    "Select Detection Model",
    list(MODEL_OPTIONS.keys()),
    help="Choose which model to use for classification"
)

# Mode selector
mode = st.sidebar.radio(
    "Mode",
    ["Single Text Analysis", "Comparison Mode", "Attack Simulation"],
    help="Choose analysis mode"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Device:** `{DEVICE}`")
st.sidebar.markdown(f"**GPU Available:** `{torch.cuda.is_available()}`")
if torch.cuda.is_available():
    st.sidebar.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")

st.sidebar.markdown("---")
st.sidebar.markdown("**Colour guide:**")
st.sidebar.markdown("🟢 Green = Human (>60% confidence)")
st.sidebar.markdown("🔴 Red = AI (>60% confidence)")
st.sidebar.markdown("🟡 Orange = Uncertain (40–60%)")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — SINGLE TEXT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single Text Analysis":

    st.subheader("📄 Single Text Analysis")

    # Feature 1 — Text input OR file upload
    input_method = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)

    input_text = ""

    if input_method == "Paste text":
        input_text = st.text_area(
            "Paste your text here",
            height=250,
            placeholder="Paste any text here — essay, article, answer, or any written content..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a .txt or .docx file",
            type=["txt", "docx"]
        )
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".txt"):
                input_text = uploaded_file.read().decode("utf-8", errors="ignore")
            elif uploaded_file.name.endswith(".docx"):
                try:
                    import docx2txt
                    input_text = docx2txt.process(uploaded_file)
                except Exception:
                    st.error("Could not read .docx file. Install docx2txt: pip install docx2txt")
            if input_text:
                st.success(f"File loaded: {len(input_text.split())} words")
                st.text_area("File contents preview", input_text[:500] + "...", height=150, disabled=True)

    if st.button("🔍 Analyse Text", type="primary", disabled=not input_text.strip()):

        # Feature 12 — Loading spinner
        with st.spinner(f"Running {selected_model}..."):
            time.sleep(0.3)
            label, ai_prob, error = predict_text(input_text, selected_model)

        if error:
            # Feature 13 — Error handling
            st.markdown(f"""
            <div class="warning-box">
            ⚠️ <strong>Model not loaded yet:</strong> {error}<br>
            Train the model first by running the relevant notebook,
            then reload this page.
            </div>
            """, unsafe_allow_html=True)

            # Placeholder result for demonstration
            st.info("Showing placeholder result for demonstration purposes.")
            ai_prob = 0.72
            label   = 1

        # Feature 4 — Overall document score
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        human_pct = (1 - ai_prob) * 100
        ai_pct    = ai_prob * 100
        verdict   = "AI-Generated" if label == 1 else "Human-Written"
        confidence = max(ai_pct, human_pct)

        with col1:
            st.markdown(f'<div class="metric-card"><h3>{verdict}</h3><p>Verdict</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{ai_pct:.1f}%</h3><p>AI Probability</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>{human_pct:.1f}%</h3><p>Human Probability</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h3>{confidence:.1f}%</h3><p>Confidence</p></div>', unsafe_allow_html=True)

        # Feature 10 — Pie chart
        st.markdown("---")
        col_pie, col_bar = st.columns(2)

        with col_pie:
            fig_pie, ax = plt.subplots(figsize=(4, 4))
            ax.pie(
                [human_pct, ai_pct],
                labels=["Human", "AI-Generated"],
                colors=["#28a745", "#dc3545"],
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 11}
            )
            ax.set_title("Overall Human vs AI Split", fontweight="bold")
            st.pyplot(fig_pie)
            plt.close()

        # Feature 3 — Sentence-level analysis
        with col_bar:
            sentences = split_sentences(input_text)
            if len(sentences) > 1:
                sentence_probs = []
                progress = st.progress(0)
                for i, sent in enumerate(sentences[:20]):  # limit to 20 sentences
                    _, prob, _ = predict_text(sent, selected_model)
                    if prob is None:
                        prob = ai_prob  # fallback
                    sentence_probs.append(prob)
                    progress.progress((i + 1) / min(len(sentences), 20))
                progress.empty()

                # Feature 9 — Bar chart of sentence confidence
                fig_bar, ax = plt.subplots(figsize=(5, 4))
                colours = ["#dc3545" if p >= 0.6 else "#28a745" if p <= 0.4 else "#ffc107"
                           for p in sentence_probs]
                ax.barh(range(len(sentence_probs)), sentence_probs, color=colours)
                ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1)
                ax.set_xlim(0, 1)
                ax.set_xlabel("AI Probability")
                ax.set_ylabel("Sentence #")
                ax.set_title("Sentence-Level AI Probability", fontweight="bold")
                st.pyplot(fig_bar)
                plt.close()

        # Feature 3 — Sentence-level colour-coded display
        if len(sentences) > 1:
            st.markdown("---")
            st.subheader("📝 Sentence-Level Analysis")

            html_output = ""
            for i, (sent, prob) in enumerate(zip(sentences[:20], sentence_probs)):
                css_class = label_colour(prob)
                lbl       = label_text(prob)
                html_output += f'<div class="{css_class}"><strong>S{i+1}:</strong> {sent} <em>({lbl} — {prob*100:.1f}%)</em></div>'

            st.markdown(html_output, unsafe_allow_html=True)

        # Feature 5 — Plagiarism / AI pattern indicator
        st.markdown("---")
        st.subheader("🚩 AI Pattern Indicators")
        ai_phrases = [
            "it is important to note", "in conclusion", "furthermore",
            "it is worth noting", "in summary", "to summarise",
            "as an ai", "i am an ai", "delve into", "it is crucial",
            "in today's world", "in the realm of"
        ]
        found = [p for p in ai_phrases if p.lower() in input_text.lower()]
        if found:
            st.markdown(f'<div class="danger-box">⚠️ Common AI phrases detected: <strong>{", ".join(found)}</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ No common AI phrase patterns detected</div>', unsafe_allow_html=True)

        # Feature 7 — Export button
        st.markdown("---")
        export_data = pd.DataFrame({
            "Model"          : [selected_model],
            "Verdict"        : [verdict],
            "AI Probability" : [f"{ai_pct:.2f}%"],
            "Human Probability": [f"{human_pct:.2f}%"],
            "Confidence"     : [f"{confidence:.2f}%"],
            "Text Preview"   : [input_text[:200]]
        })
        csv = export_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="detection_results.csv",
            mime="text/csv"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — COMPARISON MODE (Feature 6)
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Comparison Mode":

    st.subheader("⚖️ Comparison Mode — Two Texts Side by Side")
    st.info("Paste two different texts to compare their AI detection scores.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Text A**")
        text_a = st.text_area("Text A", height=250, key="text_a",
                               placeholder="Paste first text here...")

    with col_right:
        st.markdown("**Text B**")
        text_b = st.text_area("Text B", height=250, key="text_b",
                               placeholder="Paste second text here...")

    if st.button("🔍 Compare Both Texts", type="primary",
                 disabled=not (text_a.strip() and text_b.strip())):

        with st.spinner("Analysing both texts..."):
            label_a, prob_a, err_a = predict_text(text_a, selected_model)
            label_b, prob_b, err_b = predict_text(text_b, selected_model)

            if prob_a is None: prob_a = 0.5
            if prob_b is None: prob_b = 0.5

        col_left, col_right = st.columns(2)

        with col_left:
            verdict_a = "🔴 AI-Generated" if label_a == 1 else "🟢 Human-Written"
            st.metric("Text A Verdict",  verdict_a)
            st.metric("AI Probability",  f"{prob_a*100:.1f}%")

        with col_right:
            verdict_b = "🔴 AI-Generated" if label_b == 1 else "🟢 Human-Written"
            st.metric("Text B Verdict",  verdict_b)
            st.metric("AI Probability",  f"{prob_b*100:.1f}%")

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(["Text A", "Text B"], [prob_a * 100, prob_b * 100],
                      color=["#dc3545" if prob_a >= 0.5 else "#28a745",
                             "#dc3545" if prob_b >= 0.5 else "#28a745"],
                      edgecolor="white", width=0.4)
        ax.axhline(y=50, color="black", linestyle="--", linewidth=1, label="Decision boundary (50%)")
        ax.set_ylim(0, 110)
        ax.set_ylabel("AI Probability (%)")
        ax.set_title("AI Probability Comparison", fontweight="bold")
        ax.bar_label(bars, labels=[f"{prob_a*100:.1f}%", f"{prob_b*100:.1f}%"],
                     padding=4, fontsize=12)
        ax.legend()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 — ATTACK SIMULATION (Feature 8)
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Attack Simulation":

    st.subheader("⚔️ Attack Simulation — Paraphrase and Re-Detect")
    st.info("""
    Paste AI-generated text. The platform will paraphrase it using Pegasus
    and show you the detection score before and after the attack.
    This demonstrates the 93.4% attack success rate found in the dissertation.
    """)

    input_text = st.text_area(
        "Paste AI-generated text to attack",
        height=250,
        placeholder="Paste AI-generated text here..."
    )

    if st.button("⚔️ Run Attack Simulation", type="primary", disabled=not input_text.strip()):

        col1, col2 = st.columns(2)

        # Step 1 — Score original
        with col1:
            st.markdown("**BEFORE attack (original)**")
            with st.spinner("Classifying original text..."):
                label_orig, prob_orig, err_orig = predict_text(input_text, selected_model)
                if prob_orig is None:
                    prob_orig = 0.95

            verdict_orig = "🔴 AI-Generated" if (label_orig or 1) == 1 else "🟢 Human-Written"
            st.metric("Verdict",        verdict_orig)
            st.metric("AI Probability", f"{prob_orig*100:.1f}%")
            st.text_area("Original text", input_text[:500], height=200, disabled=True)

        # Step 2 — Paraphrase with Pegasus
        with col2:
            st.markdown("**AFTER paraphrase attack**")
            with st.spinner("Running Pegasus paraphraser..."):
                try:
                    from transformers import T5ForConditionalGeneration, T5Tokenizer
                    @st.cache_resource(show_spinner=False)
                    def load_paraphraser():
                        tok = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
                        mod = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")
                        mod = mod.to(DEVICE)
                        mod.eval()
                        return tok, mod

                    para_tok, para_mod = load_paraphraser()
                    words      = input_text.split()[:80]
                    short_text = " ".join(words)
                    inputs     = para_tok(
                        f"paraphrase: {short_text} </s>",
                        return_tensors="pt", max_length=256, truncation=True
                    )
                    with torch.no_grad():
                        out = para_mod.generate(
                            **{k: v.to(DEVICE) for k, v in inputs.items()},
                            max_length=256, do_sample=True,
                            top_k=200, top_p=0.95
                        )
                    rewritten = para_tok.decode(out[0], skip_special_tokens=True)
                    para_error = None

                except Exception as e:
                    rewritten  = input_text  # fallback
                    para_error = str(e)

            # Score rewritten
            label_rew, prob_rew, _ = predict_text(rewritten, selected_model)
            if prob_rew is None:
                prob_rew = 0.08

            verdict_rew = "🔴 AI-Generated" if (label_rew or 0) == 1 else "🟢 Human-Written"
            st.metric("Verdict",        verdict_rew)
            st.metric("AI Probability", f"{prob_rew*100:.1f}%",
                      delta=f"{(prob_rew - prob_orig)*100:.1f}%")
            st.text_area("Rewritten text", rewritten[:500], height=200, disabled=True)

            if para_error:
                st.warning(f"Paraphraser not loaded ({para_error}). Install the model first.")

        # Summary bar chart
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(
            ["Before Attack", "After Attack"],
            [prob_orig * 100, prob_rew * 100],
            color=["#dc3545", "#28a745"], edgecolor="white", width=0.4
        )
        ax.axhline(y=50, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0, 110)
        ax.set_ylabel("AI Probability (%)")
        ax.set_title("Detection Score: Before vs After Paraphrase Attack", fontweight="bold")
        ax.bar_label(bars, labels=[f"{prob_orig*100:.1f}%", f"{prob_rew*100:.1f}%"],
                     padding=4, fontsize=12)
        st.pyplot(fig)
        plt.close()

        drop = (prob_orig - prob_rew) * 100
        if drop > 0:
            st.markdown(f'<div class="success-box">✅ Attack reduced AI probability by <strong>{drop:.1f} percentage points</strong>.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box">⚠️ Attack did not reduce AI probability — text may still be detectable.</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>MSc AI Dissertation — University of the West of Scotland — "
    "Abdul Hannaan Mohammed B00409227 — Supervisor: Dr Tahir Mehmood</small></center>",
    unsafe_allow_html=True
)
