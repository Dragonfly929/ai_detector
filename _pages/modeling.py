import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hashlib

from utils.ml_utils import load_default_data, load_uploaded_data, train_all_models


def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()


def show():
    st.markdown('<span class="phase-badge phase-4">Phase 4</span>', unsafe_allow_html=True)
    st.title("Modeling")

    with st.sidebar:
        st.markdown("### 📂 Dataset Source")
        src = st.radio("Use", ["Default dataset", "Upload CSV"], key="mod_src")
        uploaded = None
        if src == "Upload CSV":
            uploaded = st.file_uploader("CSV", type="csv", key="mod_up")
        st.markdown("---")
        st.markdown("### ⚙️ Training Options")
        max_feats = st.select_slider(
            "TF-IDF max features", [5_000, 10_000, 20_000, 50_000], value=20_000
        )
        st.caption("Larger → slower but potentially better.")

    with st.spinner("Loading data…"):
        try:
            df = load_uploaded_data(uploaded) if (src == "Upload CSV" and uploaded) else load_default_data()
        except Exception as e:
            st.error(str(e)); return

    # sub-sample for speed inside the app
    MAX_ROWS = 15_000
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
        st.info(f"Sub-sampled to {MAX_ROWS:,} rows for interactive speed.")

    tab1, tab2, tab3 = st.tabs(["🏗️ Architecture", "🚀 Train Models", "🔍 Inspect Models"])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Feature Engineering — TF-IDF")
        st.markdown("""
**TF-IDF** (Term Frequency–Inverse Document Frequency) converts raw essay text into
a numerical vector. Key parameters:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `max_features` | 20 000 | Controls vocabulary size; balances speed vs richness |
| `ngram_range` | (1, 2) | Captures single words **and** adjacent word pairs |
| `sublinear_tf` | True | Dampens effect of very frequent terms |
| `min_df` | 2 | Ignores terms that appear in fewer than 2 documents |
        """)

        st.markdown("### Classifiers Compared")
        classifiers = {
            "Logistic Regression": {
                "type": "Linear",
                "pros": "Fast, interpretable coefficients, works well with high-dimensional TF-IDF.",
                "cons": "Assumes linear decision boundary.",
            },
            "Naive Bayes (Multinomial)": {
                "type": "Probabilistic",
                "pros": "Very fast, handles sparse data naturally, good baseline.",
                "cons": "Independence assumption between features is unrealistic.",
            },
            "Random Forest": {
                "type": "Ensemble (bagging)",
                "pros": "Robust to noise, captures non-linear patterns, built-in feature importance.",
                "cons": "Slower to train; memory-intensive on very large vocabularies.",
            },
            "Gradient Boosting": {
                "type": "Ensemble (boosting)",
                "pros": "Often achieves highest accuracy; corrects previous errors iteratively.",
                "cons": "Slow to train; prone to over-fitting without careful tuning.",
            },
        }
        for name, info in classifiers.items():
            with st.expander(f"**{name}** ({info['type']})"):
                st.markdown(f"✅ **Pros:** {info['pros']}")
                st.markdown(f"⚠️ **Cons:** {info['cons']}")

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Train All Models")
        st.markdown(f"Training on **{len(df):,}** essays · 80/20 stratified split.")

        if st.button("▶️  Train Now", type="primary"):
            h = _df_hash(df)
            from utils.ml_utils import _TRAINING_DATA_CACHE
            _TRAINING_DATA_CACHE[h] = df
            with st.spinner("Training models — this may take 1–2 minutes…"):
                results, X_train, X_test, y_train, y_test = train_all_models(h, df)
            st.session_state["results"] = results
            st.session_state["trained"] = True
            st.success("Training complete! Navigate to **Evaluation** for detailed results.")

        if st.session_state.get("trained"):
            results = st.session_state["results"]
            rows = []
            for name, r in results.items():
                rows.append({
                    "Model": name,
                    "Accuracy": f"{r['accuracy']:.4f}",
                    "Precision": f"{r['precision']:.4f}",
                    "Recall": f"{r['recall']:.4f}",
                    "F1": f"{r['f1']:.4f}",
                    "ROC-AUC": f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        if not st.session_state.get("trained"):
            st.info("Train models first in the **Train Models** tab.")
            return

        results = st.session_state["results"]
        model_name = st.selectbox("Select model to inspect", list(results.keys()))
        r = results[model_name]
        pipe = r["pipeline"]

        st.markdown(f"### Top TF-IDF Features — {model_name}")
        try:
            tfidf = pipe["tfidf"]
            clf = pipe["clf"]
            vocab = tfidf.get_feature_names_out()

            if hasattr(clf, "coef_"):
                coefs = clf.coef_[0]
                top_ai_idx = np.argsort(coefs)[-20:][::-1]
                top_human_idx = np.argsort(coefs)[:20]
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = px.bar(
                        x=coefs[top_ai_idx], y=vocab[top_ai_idx],
                        orientation="h", title="Top AI-associated words",
                        color_discrete_sequence=["#ec4899"],
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    fig = px.bar(
                        x=np.abs(coefs[top_human_idx]), y=vocab[top_human_idx],
                        orientation="h", title="Top Human-associated words",
                        color_discrete_sequence=["#6366f1"],
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
            elif hasattr(clf, "feature_importances_"):
                imp = clf.feature_importances_
                top_idx = np.argsort(imp)[-20:][::-1]
                fig = px.bar(
                    x=imp[top_idx], y=vocab[top_idx],
                    orientation="h", title="Feature Importances",
                    color_discrete_sequence=["#8b5cf6"],
                )
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature visualization not available for this model type.")
        except Exception as e:
            st.warning(f"Could not extract feature importances: {e}")
