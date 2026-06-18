import streamlit as st


def show():
    st.markdown('<p class="hero-title">AI-Generated Text Detector</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">A full data-science project following the <b>CRISP-DM</b> methodology</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### What is this project?")
        st.markdown("""
This application detects whether a piece of text was written by a **human** or generated
by an **AI language model** (e.g. GPT, LLaMA, PaLM).

The project is structured end-to-end following **CRISP-DM** — the industry-standard
cross-industry process for data mining — as taught in the
*Platforme pentru Analiza Avansată de Date* master's course at UTM.
        """)

        st.markdown("### The CRISP-DM Pipeline")

        phases = [
            ("phase-1", "1", "Business Understanding", "Define the problem: detect AI-generated essays to protect academic integrity."),
            ("phase-2", "2", "Data Understanding", "Explore ~30 k labelled essays; analyse text length, class balance, vocabulary."),
            ("phase-3", "3", "Data Preparation", "Clean text, remove stopwords, engineer stylometric features, balance classes."),
            ("phase-4", "4", "Modeling", "Train TF-IDF + classifiers: Logistic Regression, Naive Bayes, Random Forest, Gradient Boosting."),
            ("phase-5", "5", "Evaluation", "Compare models by Accuracy, F1, ROC-AUC; inspect confusion matrices."),
            ("phase-6", "6", "Deployment", "Interactive demo: paste or upload text and get an AI probability score."),
        ]

        for cls, num, title, desc in phases:
            st.markdown(
                f'<span class="phase-badge {cls}">Phase {num}</span> '
                f'**{title}** — {desc}',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### Quick Stats")
        st.markdown("""
<div class="metric-row">
  <div class="metric-card"><div class="val">~30k</div><div class="lbl">Training essays</div></div>
  <div class="metric-card"><div class="val">4</div><div class="lbl">ML Models</div></div>
</div>
<div class="metric-row">
  <div class="metric-card"><div class="val">TF-IDF</div><div class="lbl">Text vectorizer</div></div>
  <div class="metric-card"><div class="val">97%+</div><div class="lbl">Best accuracy</div></div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("### Dataset")
        st.info(
            "**Default:** `Training_Essay_Data.csv`  \n"
            "Human-written essays labelled `0` and AI-generated essays labelled `1`.  \n\n"
            "You can **upload your own CSV** (same schema) in the sidebar of most pages."
        )
        st.markdown(
            "🔗 [View dataset on Kaggle](https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset)",)

        st.markdown("### Technologies")
        for tech in ["Python · scikit-learn", "Streamlit", "TF-IDF (bi-grams)", "Plotly / Matplotlib", "CRISP-DM framework"]:
            st.markdown(f"- {tech}")
