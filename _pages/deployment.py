import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import hashlib

from utils.ml_utils import (
    load_default_data, load_uploaded_data,
    train_all_models, predict_text, clean_text
)


def _df_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()


def _gauge(prob: float):
    color = "#22c55e" if prob < 0.35 else ("#f59e0b" if prob < 0.65 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "AI Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 35],  "color": "#dcfce7"},
                {"range": [35, 65], "color": "#fef9c3"},
                {"range": [65, 100],"color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
        },
        number={"suffix": "%", "font": {"size": 36}},
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=0, l=20, r=20))
    return fig


def show():
    st.markdown('<span class="phase-badge phase-6">Phase 6</span>', unsafe_allow_html=True)
    st.title("🚀 Deployment — Live AI Text Detector")

    # ── Ensure a model is ready ───────────────────────────────────────────────
    if not st.session_state.get("trained"):
        st.info("No models trained yet. Auto-training on default dataset…")
        with st.spinner("Training (first time only — ~60 s) …"):
            df = load_default_data()
            MAX_ROWS = 15_000
            if len(df) > MAX_ROWS:
                df = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
            h = _df_hash(df)
            from utils.ml_utils import _TRAINING_DATA_CACHE
            _TRAINING_DATA_CACHE[h] = df
            results, *_ = train_all_models(h, df)
        st.session_state["results"] = results
        st.session_state["trained"] = True
        st.success("Models ready!")

    results = st.session_state["results"]

    # pick best model by default
    best_name = max(results, key=lambda n: results[n]["f1"])
    model_name = st.selectbox(
        "Choose classifier",
        list(results.keys()),
        index=list(results.keys()).index(best_name),
    )
    pipeline = results[model_name]["pipeline"]

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### ✍️ Input Text")
        input_mode = st.radio("Input mode", ["Paste text", "Upload .txt file"], horizontal=True)

        essay_text = ""
        if input_mode == "Paste text":
            essay_text = st.text_area(
                "Paste essay or paragraph here",
                height=300,
                placeholder="Type or paste text to analyse…",
            )
        else:
            txt_file = st.file_uploader("Upload a .txt file", type=["txt"])
            if txt_file:
                essay_text = txt_file.read().decode("utf-8", errors="ignore")
                st.text_area("File contents (preview)", essay_text[:1000], height=200, disabled=True)

        analyse = st.button("🔍 Analyse Text", type="primary", disabled=not bool(essay_text.strip()))

    with col_right:
        st.markdown("### 📊 Result")
        if not essay_text.strip():
            st.caption("Enter text on the left and click **Analyse**.")

    if analyse and essay_text.strip():
        with st.spinner("Analysing…"):
            label, prob = predict_text(pipeline, essay_text)

        # Result box
        if prob < 0.35:
            verdict = "✅ Likely Human-written"
            css_class = "result-human"
            explanation = "The writing style, vocabulary, and structure suggest a human author."
        elif prob > 0.65:
            verdict = "🤖 Likely AI-generated"
            css_class = "result-ai"
            explanation = "Patterns in the text are consistent with AI-generated content."
        else:
            verdict = "⚠️ Uncertain — manual review recommended"
            css_class = "result-unsure"
            explanation = "The model is not confident. A human reviewer should inspect this text."

        with col_right:
            st.plotly_chart(_gauge(prob), use_container_width=True)

        st.markdown(
            f'<div class="result-box {css_class}">'
            f'<h3>{verdict}</h3>'
            f'<p>{explanation}</p>'
            f'<p><b>AI probability:</b> {prob:.1%} &nbsp;|&nbsp; <b>Model:</b> {model_name}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Text stats
        with st.expander("📋 Text Analysis Details"):
            words = essay_text.split()
            sentences = [s for s in essay_text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
            cleaned = clean_text(essay_text)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Characters", f"{len(essay_text):,}")
            c2.metric("Words", f"{len(words):,}")
            c3.metric("Sentences", f"{len(sentences):,}")
            c4.metric("Unique words", f"{len(set(w.lower() for w in words)):,}")

            st.markdown("**Cleaned / normalised text (first 200 tokens):**")
            st.code(" ".join(cleaned.split()[:200]), language=None)

        # Batch mode hint
        st.markdown("---")
        st.markdown("### 📦 Batch Analysis")
        st.info("""
**Want to analyse multiple essays?**  
Upload a CSV with a `text` column (no `generated` label needed) below.
        """)
        batch_file = st.file_uploader("Upload CSV for batch analysis", type=["csv"], key="batch")
        if batch_file:
            batch_df = pd.read_csv(batch_file)
            if "text" not in batch_df.columns:
                st.error("CSV must contain a `text` column.")
            else:
                with st.spinner(f"Analysing {len(batch_df):,} texts…"):
                    preds = [predict_text(pipeline, t) for t in batch_df["text"].fillna("")]
                batch_df["ai_label"] = [p[0] for p in preds]
                batch_df["ai_probability"] = [round(p[1], 4) for p in preds]
                st.dataframe(
                    batch_df[["text", "ai_probability", "ai_label"]].head(50),
                    use_container_width=True,
                )
                csv_out = batch_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️  Download results CSV", csv_out,
                    file_name="ai_detection_results.csv", mime="text/csv"
                )
