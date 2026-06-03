"""
utils/ml_utils.py
Shared ML helpers used across pages.
"""
import re
import string
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATA_PATH = Path(__file__).parent.parent / "data" / "Training_Essay_Data.csv"

# Fallback paths to search if the default doesn't exist
_FALLBACK_PATHS = [
    Path.home() / "Downloads" / "ai_text_detector_project (1)" / "ai_detector" / "data" / "Training_Essay_Data.csv",
    Path.home() / "Downloads" / "ai_text_detector_project" / "ai_detector" / "data" / "Training_Essay_Data.csv",
    Path("/mnt/user-data/uploads/Training_Essay_Data.csv"),
]

def _resolve_data_path() -> Path:
    if DATA_PATH.exists():
        return DATA_PATH
    for p in _FALLBACK_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Training_Essay_Data.csv not found. Expected at: {DATA_PATH}\n"
        "Please ensure the data folder exists inside the project directory."
    )

STOPWORDS = set("""i me my myself we our ours ourselves you you're you've you'll
you'd your yours yourself yourselves he him his himself she she's her hers
herself it it's its itself they them their theirs themselves what which who
whom this that that'll these those am is are was were be been being have has
had having do does did doing a an the and but if or because as until while of
at by for with about against between into through during before after above
below to from up down in out on off over under again further then once here
there when where why how all any both each few more most other some such
no nor not only own same so than too very s t can will just don don't should
should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't
doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn
mightn't mustn mustn't needn needn't shan shan't shouldn shouldn't wasn
wasn't weren weren't won won't wouldn wouldn't""".split())


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    df = pd.read_csv(_resolve_data_path())
    df.columns = df.columns.str.strip().str.lower()
    df = df[["text", "generated"]].dropna()
    df["generated"] = df["generated"].astype(int)
    return df


def load_uploaded_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()
    required = {"text", "generated"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must have columns: {required}. Found: {set(df.columns)}")
    df = df[["text", "generated"]].dropna()
    df["generated"] = df["generated"].astype(int)
    return df


# ── Text features ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


def extract_stylometric_features(texts: pd.Series) -> pd.DataFrame:
    feats = []
    for t in texts:
        words = t.split()
        sentences = re.split(r"[.!?]+", t)
        sentences = [s.strip() for s in sentences if s.strip()]
        unique_words = set(w.lower() for w in words)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        type_token = len(unique_words) / len(words) if words else 0
        punct_ratio = sum(1 for c in t if c in string.punctuation) / max(len(t), 1)
        comma_ratio = t.count(",") / max(len(words), 1)
        semicol_ratio = t.count(";") / max(len(words), 1)
        capital_ratio = sum(1 for c in t if c.isupper()) / max(len(t), 1)
        paragraph_count = t.count("\n\n") + 1
        feats.append({
            "word_count": len(words),
            "unique_word_ratio": type_token,
            "avg_word_length": avg_word_len,
            "avg_sentence_length": avg_sent_len,
            "sentence_count": len(sentences),
            "punct_ratio": punct_ratio,
            "comma_ratio": comma_ratio,
            "semicolon_ratio": semicol_ratio,
            "capital_ratio": capital_ratio,
            "paragraph_count": paragraph_count,
        })
    return pd.DataFrame(feats)


# ── Model building ────────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}


def build_tfidf_pipeline(clf, max_features=20_000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", clf),
    ])


# In-memory cache for DataFrames passed to training (avoids JSON serialization issues)
_TRAINING_DATA_CACHE: dict = {}


@st.cache_resource(show_spinner=False)
def train_all_models(data_hash: str, _df: pd.DataFrame = None):
    """Train all models. Pass the DataFrame directly; data_hash is used as cache key."""
    # Retrieve from our own cache if not passed directly (cache_resource re-use)
    df = _df if _df is not None else _TRAINING_DATA_CACHE.get(data_hash)
    if df is None:
        raise ValueError("DataFrame not found in cache. Please re-trigger training.")

    X, y = df["text"], df["generated"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results = {}
    for name, clf in MODELS.items():
        pipe = build_tfidf_pipeline(clf)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe["clf"], "predict_proba") else None
        results[name] = {
            "pipeline": pipe,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist() if y_prob is not None else None,
        }
    return results, X_train, X_test, y_train, y_test


def predict_text(pipeline, text: str):
    prob = pipeline.predict_proba([text])[0][1]
    label = pipeline.predict([text])[0]
    return label, prob
