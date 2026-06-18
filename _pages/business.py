import streamlit as st


def show():
    st.markdown('<span class="phase-badge phase-1">Phase 1</span>', unsafe_allow_html=True)
    st.title("Business Understanding")

    tab1, tab2, tab3 = st.tabs([
        "📌 Business Objectives",
        "📋 Situation Assessment",
        "🎯 Data Mining Goals",
    ])

    with tab1:
        st.markdown("### Business Objective")
        st.markdown("""
        We've reached a point where AI-generated content is everywhere — comments, reviews, photos, 
        videos — and telling what's real from what's synthetic has become nearly impossible. 
        (Including this very text you're reading, which is... well, you get the irony. Special am pus prompt ca sa scrie aceasta)).)

        The lines have blurred to the point where even experts struggle to distinguish authentic 
        human expression from machine output. As deepfakes improve and language models grow more 
        sophisticated, trust in digital content is eroding across every platform.

        **The (slightly absurd) mission:** Build a detection system that can identify AI-generated 
        text as a first step toward tackling this problem across all content types. Because if 
        we can't even tell if a review was written by a person or a bot, we might as well start 
        somewhere.
        """)

        st.markdown("### Stakeholders")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🌐 Platform Owners")
            st.markdown("Restore trust in user-generated content; filter bot-generated spam and fake engagement.")

        with col2:
            st.markdown("#### 📱 Everyday Users")
            st.markdown("Know whether you're reading a genuine human opinion or a machine's best impression of one.")

        with col3:
            st.markdown("#### 🤖 AI Companies")
            st.markdown("Understand the limits of their own models and develop more responsible deployment practices.")
        st.markdown("### Business Success Criteria")
        st.success("""
- **Accuracy ≥ 90 %** on held-out essays not seen during training.
- **F1-score ≥ 0.88** (balances precision & recall equally).
- **False-positive rate < 10 %** (wrongly labelling a human essay as AI is costly).
- Inference time **< 2 seconds** per essay in the live demo.
        """)

    with tab2:
        st.markdown("### Resources")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Available**")
            st.markdown("""
- ~30 000 labelled essays (human + AI)
- Python / scikit-learn / Streamlit stack
- Standard laptop / cloud compute
            """)
        with col_b:
            st.markdown("**Constraints**")
            st.markdown("""
- No GPU for deep-learning (BERT) in live demo
- Dataset covers only English-language essays
- No access to the original exam prompts in test set
            """)

        st.markdown("### Risks & Mitigations")
        risks = {
            "Data imbalance (more human than AI)": "Oversample minority class; monitor F1 not just accuracy.",
            "Domain shift (new AI models)": "Modular design lets new training data be ingested easily.",
            "Adversarial paraphrasing": "Stylometric features complement bag-of-words; harder to spoof.",
            "False positives damaging students": "Show probability + explanation; never return hard binary verdict.",
        }
        for risk, mitigation in risks.items():
            with st.expander(f"⚠️ {risk}"):
                st.markdown(f"**Mitigation:** {mitigation}")

    with tab3:
        st.markdown("### Data Mining Goals")
        st.markdown("""
| Goal | Metric | Target |
|------|--------|--------|
| Binary classification of text origin | Accuracy | ≥ 90 % |
| Minimise false accusations | Precision (human→AI) | ≥ 88 % |
| Minimise missed AI texts | Recall (AI class) | ≥ 85 % |
| Ranking confidence | ROC-AUC | ≥ 0.95 |
        """)

        st.markdown("### Framing the ML Problem")
        st.info("""
**Input:** Raw essay text (string)  
**Output:** Binary label `{0 = Human, 1 = AI}` + probability score in `[0, 1]`  
**Type:** Supervised binary text classification  
**Feature representation:** TF-IDF bi-grams + stylometric features  
        """)