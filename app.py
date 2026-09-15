from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="CreditWise Loan Approval System", page_icon="🏦", layout="wide")

DATA_PATH = Path(__file__).resolve().parent / "loan_approval_data.csv"
TARGET_COLUMN = "Loan_Approved"
ID_COLUMN = "Applicant_ID"

NUMERIC_COLUMNS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
]

CATEGORICAL_COLUMNS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category",
]


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0})
    )
    return df.dropna(subset=[TARGET_COLUMN])


@st.cache_resource
def train_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame]:
    features = [col for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS if col in df.columns]
    x = df[features]
    y = df[TARGET_COLUMN].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = [c for c in NUMERIC_COLUMNS if c in features]
    categorical_features = [c for c in CATEGORICAL_COLUMNS if c in features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_prob),
    }

    importance_df = pd.DataFrame()
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        feature_scores = pipeline.named_steps["model"].feature_importances_
        importance_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": feature_scores})
            .sort_values("Importance", ascending=False)
            .head(12)
        )
        importance_df["Feature"] = importance_df["Feature"].str.replace("num__", "", regex=False)
        importance_df["Feature"] = importance_df["Feature"].str.replace("cat__", "", regex=False)
    except Exception:
        pass

    return pipeline, metrics, importance_df


def input_panel(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("🧾 Applicant Profile")
    payload: Dict[str, object] = {}

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        median = float(series.median()) if not series.empty else 0.0
        min_val = float(series.min()) if not series.empty else 0.0
        max_val = float(series.max()) if not series.empty else median + 1000.0
        step = 0.01 if col == "DTI_Ratio" else 1.0
        payload[col] = st.sidebar.number_input(
            col.replace("_", " "),
            min_value=min_val,
            max_value=max_val,
            value=median,
            step=step,
        )

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        options: List[str] = (
            df[col].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
        )
        options = sorted(options)
        payload[col] = st.sidebar.selectbox(col.replace("_", " "), options)

    return pd.DataFrame([payload])


def main() -> None:
    st.markdown(
        """
        <style>
        .main-title {
            padding: 1rem 1.25rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
            color: white;
            margin-bottom: 1rem;
        }
        .subtitle {
            font-size: 1rem;
            color: #cbd5e1;
            margin-top: 0.3rem;
        }
        .result-card {
            border: 1px solid #dbeafe;
            border-radius: 12px;
            padding: 1rem;
            background: #f8fafc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="main-title">
            <h1>🏦 CreditWise Loan Approval System</h1>
            <p class="subtitle">Live AI-driven recommendation engine for fast and consistent loan decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data(DATA_PATH)
    model, metrics, importance_df = train_model(df)

    applicant_df = input_panel(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{len(df):,}")
    c2.metric("Approval Rate", f"{(df[TARGET_COLUMN].mean() * 100):.1f}%")
    c3.metric("Avg Credit Score", f"{df['Credit_Score'].median():.0f}")
    c4.metric("Avg Loan Amount", f"₹{df['Loan_Amount'].median():,.0f}")

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("🔍 Instant Decision")
        if st.button("Predict Approval", type="primary", use_container_width=True):
            prediction = int(model.predict(applicant_df)[0])
            approval_prob = float(model.predict_proba(applicant_df)[0][1])
            status = "APPROVE ✅" if prediction == 1 else "REJECT ⚠️"
            color = "#15803d" if prediction == 1 else "#b91c1c"

            st.markdown(
                f"""
                <div class="result-card">
                    <h3 style="margin-bottom:0.25rem; color:{color};">Recommendation: {status}</h3>
                    <p style="margin:0;">Approval Confidence: <strong>{approval_prob * 100:.2f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(approval_prob, 0.0), 1.0))

        st.subheader("📊 Model Performance")
        m1, m2, m3 = st.columns(3)
        m4, m5 = st.columns(2)
        m1.metric("Accuracy", f"{metrics['Accuracy'] * 100:.2f}%")
        m2.metric("Precision", f"{metrics['Precision'] * 100:.2f}%")
        m3.metric("Recall", f"{metrics['Recall'] * 100:.2f}%")
        m4.metric("F1 Score", f"{metrics['F1 Score'] * 100:.2f}%")
        m5.metric("ROC AUC", f"{metrics['ROC AUC'] * 100:.2f}%")

    with right:
        st.subheader("🧠 Key Risk Drivers")
        if importance_df.empty:
            st.info("Feature importance is currently unavailable.")
        else:
            st.bar_chart(importance_df.set_index("Feature"))

        st.subheader("📁 Data Snapshot")
        st.dataframe(df.head(8), use_container_width=True)


if __name__ == "__main__":
    main()
