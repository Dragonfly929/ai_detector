# AI-Generated Text Detector
### A CRISP-DM Data Science Project

**Course:** Platforme pentru Analiza Avansată de Date  
**Institution:** Universitatea Tehnică a Moldovei  
**Programme:** Știința Datelor (Master Year I)

---

## Overview

This Streamlit application detects whether a text was written by a human or generated
by an AI (GPT, LLaMA, PaLM, etc.). The project is structured following the
**CRISP-DM** methodology (Cross-Industry Standard Process for Data Mining).

## Project Structure

```
ai_detector/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── data/
│   └── Training_Essay_Data.csv   # Default dataset (~30 k essays)
├── pages/
│   ├── overview.py         # Landing page
│   ├── business.py         # Phase 1 – Business Understanding
│   ├── data_understanding.py  # Phase 2 – Data Understanding (EDA)
│   ├── data_prep.py        # Phase 3 – Data Preparation
│   ├── modeling.py         # Phase 4 – Modeling
│   ├── evaluation.py       # Phase 5 – Evaluation
│   └── deployment.py       # Phase 6 – Live Demo
└── utils/
    └── ml_utils.py         # Shared ML utilities
```

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset Schema

The default dataset (`Training_Essay_Data.csv`) and any uploaded replacement must have:

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Essay text |
| `generated` | int (0/1) | 0 = Human, 1 = AI |

## Models

| Model | Notes |
|-------|-------|
| Logistic Regression | Fast baseline; interpretable feature weights |
| Naive Bayes (Multinomial) | Probabilistic; great for sparse TF-IDF |
| Random Forest | Robust ensemble; feature importance |
| Gradient Boosting | Highest accuracy; slower to train |

## CRISP-DM Phases

1. **Business Understanding** – Define problem, success criteria, stakeholders
2. **Data Understanding** – EDA: distributions, word clouds, stylometric analysis
3. **Data Preparation** – Cleaning, normalisation, balancing, train/test split
4. **Modeling** – TF-IDF vectorisation + 4 classifiers
5. **Evaluation** – Accuracy, F1, ROC-AUC, confusion matrices
6. **Deployment** – Interactive Streamlit demo with batch upload support
