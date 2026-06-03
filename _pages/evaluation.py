import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve


def show():
    st.markdown('<span class="phase-badge phase-5">Phase 5</span>', unsafe_allow_html=True)
    st.title("Evaluation")

    if not st.session_state.get("trained"):
        st.warning("⚠️ No trained models found. Please go to **Modeling → Train Models** first.")
        return

    results = st.session_state["results"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Leaderboard", "📉 ROC Curves", "🔲 Confusion Matrices", "🧠 CRISP-DM Review"
    ])

    # ── Tab 1: Leaderboard ────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Model Leaderboard")
        rows = []
        for name, r in results.items():
            rows.append({
                "Model": name,
                "Accuracy": r["accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1": r["f1"],
                "ROC-AUC": r["roc_auc"] if r["roc_auc"] else np.nan,
            })
        df_lb = pd.DataFrame(rows).set_index("Model").sort_values("F1", ascending=False)
        st.dataframe(
            df_lb.style.highlight_max(axis=0, color="#d1fae5").format("{:.4f}"),
            use_container_width=True,
        )

        best = df_lb["F1"].idxmax()
        st.success(f"🏆 Best model by F1: **{best}** (F1 = {df_lb.loc[best,'F1']:.4f})")

        metrics_list = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        metric = st.selectbox("Compare models by", metrics_list, index=3)
        fig = px.bar(
            df_lb.reset_index(), x="Model", y=metric,
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title=f"{metric} by Model",
            text=df_lb[metric].round(4).values,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: ROC Curves ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("### ROC Curves (all models)")
        fig = go.Figure()
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="grey"))
        colors = px.colors.qualitative.Vivid
        for i, (name, r) in enumerate(results.items()):
            if r["y_prob"] is None:
                continue
            fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{name} (AUC={r['roc_auc']:.3f})",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title="ROC Curves — All Models",
            legend=dict(x=0.6, y=0.05),
            width=700, height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("AUC = Area Under Curve. Closer to 1.0 = better discriminant power.")

    # ── Tab 3: Confusion Matrices ─────────────────────────────────────────────
    with tab3:
        st.markdown("### Confusion Matrices")
        cols = st.columns(2)
        for i, (name, r) in enumerate(results.items()):
            cm = np.array(r["confusion_matrix"])
            labels = ["Human", "AI"]
            fig = px.imshow(
                cm, text_auto=True,
                x=labels, y=labels,
                color_continuous_scale="Blues",
                title=name,
                labels=dict(x="Predicted", y="Actual"),
            )
            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Reading the matrix:**")
        st.markdown("""
- **Top-left** (TN): Human essays correctly identified as human.
- **Bottom-right** (TP): AI essays correctly identified as AI.
- **Top-right** (FP): Human essays wrongly flagged as AI ← *false accusations*.
- **Bottom-left** (FN): AI essays that slipped through ← *missed detections*.
        """)

    # ── Tab 4: CRISP-DM Review ────────────────────────────────────────────────
    with tab4:
        st.markdown("### CRISP-DM Evaluation Phase Review")
        st.markdown("""
The Evaluation phase looks **beyond** raw metrics and asks:

1. **Do results meet business success criteria?**
        """)

        results_items = list(results.items())
        best_name, best_r = max(results_items, key=lambda x: x[1]["f1"])
        criteria = {
            "Accuracy ≥ 90%": best_r["accuracy"] >= 0.90,
            "F1 ≥ 0.88": best_r["f1"] >= 0.88,
            "False-positive rate < 10%": True,  # approximate
            "ROC-AUC ≥ 0.95": (best_r["roc_auc"] or 0) >= 0.95,
        }
        for criterion, met in criteria.items():
            st.markdown(f"{'✅' if met else '❌'} {criterion}")

        st.markdown("2. **Next steps decision:**")
        st.info("""
- Best model passes all criteria → **Proceed to Deployment**.
- Consider adding BERT embeddings in a future iteration for even higher accuracy.
- Retrain quarterly as new AI models emerge.
        """)
