import streamlit as st


def show():
    st.markdown('<span class="phase-badge phase-1">Phase 1</span>', unsafe_allow_html=True)
    st.title("Business Understanding")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Business Objectives",
        "📋 Situation Assessment",
        "🎯 Data Mining Goals",
        "📅 Project Plan",
    ])

    with tab1:
        st.markdown("### Business Objective")
        st.markdown("""
The proliferation of large language models (GPT-4, LLaMA 3, PaLM 2, etc.) has made it
trivially easy to generate fluent academic text. Educational institutions face a growing
risk of **academic dishonesty** when students submit AI-generated essays as their own work.

**Primary goal:** Build a reliable, explainable classifier that can determine whether a
piece of text was authored by a human student or produced by an AI model.
        """)

        st.markdown("### Stakeholders")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🎓 Universities")
            st.markdown("Maintain academic integrity; flag suspicious submissions before grading.")
        with col2:
            st.markdown("#### 📝 Instructors")
            st.markdown("Fast, objective second opinion before launching a misconduct inquiry.")
        with col3:
            st.markdown("#### 👩‍🎓 Students")
            st.markdown("Fair evaluation; avoid false accusations with confidence scores.")

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

    with tab4:
        st.markdown("### Project Plan (aligned to course syllabus)")
        plan = [
            ("Week 1–2", "Business Understanding", "Define problem, success criteria, stakeholder map"),
            ("Week 3–4", "Data Understanding", "EDA: distributions, word clouds, length analysis"),
            ("Week 5–6", "Data Preparation", "Cleaning, stopword removal, feature engineering, balancing"),
            ("Week 7–8", "Modeling", "Train TF-IDF + 4 classifiers; hyperparameter search"),
            ("Week 9–10", "Evaluation", "Cross-validation, confusion matrices, ROC curves"),
            ("Week 11–12", "Deployment", "Streamlit app with live inference + upload support"),
            ("Week 13", "Documentation", "Final report + project presentation"),
        ]
        for week, phase, activity in plan:
            st.markdown(f"**{week}** · `{phase}` — {activity}")
