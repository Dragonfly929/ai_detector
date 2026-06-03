import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

from utils.ml_utils import load_default_data, load_uploaded_data, extract_stylometric_features


def show():
    st.markdown('<span class="phase-badge phase-2">Phase 2</span>', unsafe_allow_html=True)
    st.title("Data Understanding")

    # ── Dataset selector ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📂 Dataset Source")
        src = st.radio("Use", ["Default dataset", "Upload CSV"], key="du_src")
        uploaded = None
        if src == "Upload CSV":
            uploaded = st.file_uploader("CSV with `text` & `generated` columns", type="csv", key="du_up")

    with st.spinner("Loading data…"):
        try:
            if src == "Upload CSV" and uploaded:
                df = load_uploaded_data(uploaded)
                st.success(f"Loaded uploaded file — {len(df):,} rows")
            else:
                df = load_default_data()
                st.info(f"Using default dataset — {len(df):,} rows")
        except Exception as e:
            st.error(str(e)); return

    df = df.copy()
    df["text_len"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["label"] = df["generated"].map({0: "Human", 1: "AI"})

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "📏 Text Length", "📖 Vocabulary", "🔬 Feature Peek"
    ])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total essays", f"{len(df):,}")
        c2.metric("Human (0)", f"{(df.generated==0).sum():,}")
        c3.metric("AI (1)", f"{(df.generated==1).sum():,}")
        c4.metric("Balance ratio", f"{(df.generated==1).mean():.1%}")

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.pie(
                df, names="label",
                color="label",
                color_discrete_map={"Human": "#6366f1", "AI": "#ec4899"},
                title="Class Distribution",
                hole=0.45,
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            counts = df["label"].value_counts().reset_index()
            counts.columns = ["Label", "Count"]
            fig2 = px.bar(
                counts, x="Label", y="Count",
                color="Label",
                color_discrete_map={"Human": "#6366f1", "AI": "#ec4899"},
                title="Absolute Class Counts",
                text="Count",
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Sample Essays")
        for lbl in [0, 1]:
            with st.expander(f"{'🤖 AI' if lbl else '👤 Human'} sample"):
                sample = df[df.generated == lbl].sample(1, random_state=7)["text"].values[0]
                st.write(sample[:800] + "…")

    # ── Tab 2: Text Length ────────────────────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(
                df, x="word_count", color="label",
                color_discrete_map={"Human": "#6366f1", "AI": "#ec4899"},
                nbins=60, barmode="overlay", opacity=0.7,
                title="Word Count Distribution by Class",
                labels={"word_count": "Words per essay"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.box(
                df, x="label", y="word_count", color="label",
                color_discrete_map={"Human": "#6366f1", "AI": "#ec4899"},
                title="Word Count Boxplot",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Descriptive Statistics — Word Count")
        st.dataframe(
            df.groupby("label")["word_count"].describe().round(1),
            use_container_width=True,
        )

    # ── Tab 3: Vocabulary ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Top 20 Words per Class")
        from utils.ml_utils import STOPWORDS, clean_text

        col_a, col_b = st.columns(2)
        for idx, (lbl, col) in enumerate([(0, col_a), (1, col_b)]):
            texts = df[df.generated == lbl]["text"].dropna()
            all_words = []
            for t in texts:
                all_words.extend(clean_text(t).split())
            top = pd.DataFrame(Counter(all_words).most_common(20), columns=["word", "count"])
            fig = px.bar(
                top, x="count", y="word", orientation="h",
                title=f"{'AI' if lbl else 'Human'} — Top 20 words",
                color_discrete_sequence=["#ec4899" if lbl else "#6366f1"],
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            with col:
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Stylometric feature peek ──────────────────────────────────────
    with tab4:
        st.markdown("### Stylometric Feature Distributions")
        st.caption("Computed on a 1 000-row sample for speed.")
        sample_df = df.sample(min(1000, len(df)), random_state=42).reset_index(drop=True)
        feats = extract_stylometric_features(sample_df["text"])
        feats["label"] = sample_df["generated"].map({0: "Human", 1: "AI"}).values

        feat_cols = [c for c in feats.columns if c != "label"]
        chosen = st.selectbox("Feature to plot", feat_cols, index=3)
        fig = px.histogram(
            feats, x=chosen, color="label",
            color_discrete_map={"Human": "#6366f1", "AI": "#ec4899"},
            nbins=40, barmode="overlay", opacity=0.7,
            title=f"Distribution of `{chosen}` by class",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Correlation heatmap (stylometric features)")
        corr = feats[feat_cols].corr()
        fig2 = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix",
        )
        st.plotly_chart(fig2, use_container_width=True)
