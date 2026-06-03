import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.ml_utils import load_default_data, load_uploaded_data, clean_text, STOPWORDS


def show():
    st.markdown('<span class="phase-badge phase-3">Phase 3</span>', unsafe_allow_html=True)
    st.title("Data Preparation")

    with st.sidebar:
        st.markdown("### 📂 Dataset Source")
        src = st.radio("Use", ["Default dataset", "Upload CSV"], key="dp_src")
        uploaded = None
        if src == "Upload CSV":
            uploaded = st.file_uploader("CSV", type="csv", key="dp_up")

    with st.spinner("Loading…"):
        try:
            df = load_uploaded_data(uploaded) if (src == "Upload CSV" and uploaded) else load_default_data()
        except Exception as e:
            st.error(str(e)); return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧹 Data Cleaning", "🔤 Text Normalisation", "⚖️ Class Balancing", "🔀 Train/Test Split"
    ])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Before Cleaning")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total rows", f"{len(df):,}")
        c2.metric("Missing texts", str(df["text"].isna().sum()))
        c3.metric("Duplicate rows", str(df.duplicated().sum()))

        st.markdown("### Steps Applied")
        steps = {
            "1. Drop NaN": "Remove any row where `text` or `generated` is missing.",
            "2. Drop duplicates": "Remove identical (text, label) pairs to avoid data leakage.",
            "3. Strip whitespace": "Normalise leading/trailing spaces and newlines.",
            "4. Label validation": "Ensure `generated` ∈ {0, 1}; drop others.",
        }
        for step, desc in steps.items():
            st.markdown(f"**{step}** — {desc}")

        df_clean = df.dropna().drop_duplicates(subset=["text"]).copy()
        df_clean["text"] = df_clean["text"].str.strip()
        df_clean = df_clean[df_clean["generated"].isin([0, 1])]

        st.markdown("### After Cleaning")
        c1, c2 = st.columns(2)
        c1.metric("Rows remaining", f"{len(df_clean):,}")
        c2.metric("Rows removed", f"{len(df) - len(df_clean):,}")

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Text Normalisation Pipeline")
        steps_norm = [
            "Lowercase all text",
            "Remove URLs (http/https)",
            "Remove digits",
            "Remove punctuation",
            "Remove stopwords (176 English words)",
            "Remove very short tokens (< 3 chars)",
        ]
        for s in steps_norm:
            st.markdown(f"✅ {s}")

        st.markdown("### Interactive Demo")
        demo_text = st.text_area(
            "Enter text to normalise",
            value="The quick brown fox jumps over the lazy dog! Visit https://example.com for more.",
            height=100,
        )
        if demo_text:
            cleaned = clean_text(demo_text)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original**")
                st.code(demo_text, language=None)
                st.caption(f"{len(demo_text.split())} words")
            with col_b:
                st.markdown("**Cleaned**")
                st.code(cleaned, language=None)
                st.caption(f"{len(cleaned.split())} words")

        st.markdown("### Stopwords Used")
        with st.expander(f"Show all {len(STOPWORDS)} stopwords"):
            st.write(", ".join(sorted(STOPWORDS)))

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Class Imbalance Problem")
        counts = df_clean["generated"].value_counts()
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.pie(
                values=counts.values, names=["Human", "AI"],
                color_discrete_sequence=["#6366f1", "#ec4899"],
                title="Original class distribution",
                hole=0.4,
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("#### Balancing strategy")
            strategy = st.selectbox(
                "Method",
                ["Oversampling (minority → majority size)", "Undersampling (majority → minority size)"],
            )
            minority_n = counts.min()
            majority_n = counts.max()

            if "Over" in strategy:
                balanced_n = majority_n
                info = f"Minority class upsampled: {minority_n:,} → {majority_n:,}"
            else:
                balanced_n = minority_n
                info = f"Majority class downsampled: {majority_n:,} → {minority_n:,}"

            st.info(info)
            fig2 = px.bar(
                x=["Human", "AI"], y=[balanced_n, balanced_n],
                color=["Human", "AI"],
                color_discrete_map={"Human": "#6366f1", "AI": "#ec4899"},
                title="After balancing",
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 4 ─────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Train / Validation / Test Split")
        test_size = st.slider("Test set size (%)", 10, 40, 20, step=5)
        val_size  = st.slider("Validation set size (% of train)", 0, 30, 10, step=5)

        n = len(df_clean)
        n_test  = int(n * test_size / 100)
        n_val   = int((n - n_test) * val_size / 100)
        n_train = n - n_test - n_val

        fig = px.pie(
            values=[n_train, n_val, n_test],
            names=["Train", "Validation", "Test"],
            color_discrete_sequence=["#6366f1", "#8b5cf6", "#ec4899"],
            hole=0.4,
            title=f"Split of {n:,} samples",
        )
        fig.update_traces(textinfo="percent+label+value")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Stratification** is applied to preserve the original class ratio in each split.")
        st.success(f"Train: **{n_train:,}** | Val: **{n_val:,}** | Test: **{n_test:,}**")
