import streamlit as st

st.set_page_config(
    page_title="AI Text Detector — CRISP-DM",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {background: #0f172a;}
[data-testid="stSidebar"] * {color: #e2e8f0 !important;}
[data-testid="stSidebar"] .stRadio > label {color: #94a3b8 !important; font-size:0.78rem;}

/* Main heading */
.hero-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub {
    color: #64748b; font-size: 1.05rem; margin-top: 0.25rem;
}

/* Phase badge */
.phase-badge {
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:0.78rem; font-weight:700; letter-spacing:0.05em;
    margin-bottom:1rem;
}
.phase-1{background:#dbeafe;color:#1d4ed8;}
.phase-2{background:#dcfce7;color:#15803d;}
.phase-3{background:#fef9c3;color:#a16207;}
.phase-4{background:#f3e8ff;color:#7e22ce;}
.phase-5{background:#fee2e2;color:#b91c1c;}
.phase-6{background:#e0f2fe;color:#0369a1;}

/* Metric cards */
.metric-row {display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0;}
.metric-card {
    flex:1; min-width:140px; background:#f8fafc;
    border:1px solid #e2e8f0; border-radius:12px;
    padding:1rem 1.25rem; text-align:center;
}
.metric-card .val {font-size:2rem; font-weight:800; color:#4f46e5;}
.metric-card .lbl {font-size:0.8rem; color:#64748b; margin-top:2px;}

/* Result box */
.result-box {
    border-radius:12px; padding:1.5rem 2rem; margin:1rem 0;
    border-left:6px solid;
}
.result-human {background:#f0fdf4; border-color:#22c55e;}
.result-ai    {background:#fef2f2; border-color:#ef4444;}
.result-unsure{background:#fffbeb; border-color:#f59e0b;}

/* CRISP-DM step pill */
.crisp-step {
    display:inline-flex; align-items:center; gap:6px;
    background:#f1f5f9; border-radius:8px;
    padding:5px 12px; font-size:0.82rem; color:#475569;
    margin:3px;
}
.crisp-dot {width:8px;height:8px;border-radius:50%;display:inline-block;}

/* Global font size */
html, body, [class*="css"], .stMarkdown p, .stMarkdown li {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)



# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = {
    "🏠  Overview": "overview",
    "📊  Business Understanding": "business",
    "🔎  Data Understanding": "data_understanding",
    "🔧  Data Preparation": "data_prep",
    "🤖  Modeling": "modeling",
    "📈  Evaluation": "evaluation",
    "🚀  Deployment — Live Demo": "deployment",
}

with st.sidebar:
    st.markdown("## 🔍 AI Text Detector")
    st.markdown("**CRISP-DM Methodology**")
    st.markdown("---")
    choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<small>UTM · Știința Datelor<br>Platforme pentru analiza avansată de date</small>",
        unsafe_allow_html=True,
    )

page = PAGES[choice]

# ── Route to pages ────────────────────────────────────────────────────────────
if page == "overview":
    from _pages import overview; overview.show()
elif page == "business":
    from _pages import business; business.show()
elif page == "data_understanding":
    from _pages import data_understanding; data_understanding.show()
elif page == "data_prep":
    from _pages import data_prep; data_prep.show()
elif page == "modeling":
    from _pages import modeling; modeling.show()
elif page == "evaluation":
    from _pages import evaluation; evaluation.show()
elif page == "deployment":
    from _pages import deployment; deployment.show()
