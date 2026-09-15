from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="CreditWise Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Professional Color Palette
COLORS = {
    "primary": "#1F3A7D",
    "secondary": "#2563EB",
    "success": "#059669",
    "danger": "#DC2626",
    "warning": "#F59E0B",
    "light_bg": "#F8FAFC",
    "border": "#E2E8F0",
    "text_dark": "#0F172A",
    "text_light": "#475569",
}


def apply_custom_styling() -> None:
    """Apply professional custom CSS styling."""
    st.markdown(
        f"""
        <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        [data-testid="stAppViewContainer"] {{
            background-color: #FFFFFF;
        }}

        [data-testid="stSidebar"] {{
            background-color: {COLORS['light_bg']};
            border-right: 1px solid {COLORS['border']};
        }}

        .main-header {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
            color: white;
            padding: 2rem 2.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        }}

        .main-header h1 {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
        }}

        .main-header p {{
            font-size: 0.95rem;
            opacity: 0.95;
            font-weight: 400;
        }}

        .metric-card {{
            background-color: white;
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            transition: all 0.3s ease;
        }}

        .metric-card:hover {{
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border-color: {COLORS['secondary']};
        }}

        .metric-label {{
            color: {COLORS['text_light']};
            font-size: 0.85rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }}

        .metric-value {{
            color: {COLORS['text_dark']};
            font-size: 1.75rem;
            font-weight: 700;
        }}

        .decision-card {{
            border-radius: 8px;
            padding: 1.5rem;
            border-left: 4px solid;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        }}

        .decision-approved {{
            border-left-color: {COLORS['success']};
        }}

        .decision-rejected {{
            border-left-color: {COLORS['danger']};
        }}

        .decision-status {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }}

        .decision-approved .decision-status {{
            color: {COLORS['success']};
        }}

        .decision-rejected .decision-status {{
            color: {COLORS['danger']};
        }}

        .confidence-bar {{
            background-color: {COLORS['light_bg']};
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
            margin-top: 0.75rem;
        }}

        .confidence-fill {{
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, {COLORS['secondary']}, {COLORS['primary']});
            transition: width 0.5s ease;
        }}

        .section-title {{
            color: {COLORS['text_dark']};
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid {COLORS['secondary']};
        }}

        .info-box {{
            background-color: {COLORS['light_bg']};
            border: 1px solid {COLORS['border']};
            border-left: 4px solid {COLORS['secondary']};
            border-radius: 6px;
            padding: 1rem;
            margin: 1rem 0;
        }}

        .risk-indicator {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            font-size: 0.9rem;
            font-weight: 500;
        }}

        .risk-high {{
            background-color: #FEE2E2;
            color: {COLORS['danger']};
            border: 1px solid #FECACA;
        }}

        .risk-medium {{
            background-color: #FEF3C7;
            color: {COLORS['warning']};
            border: 1px solid #FCD34D;
        }}

        .risk-low {{
            background-color: #DCFCE7;
            color: {COLORS['success']};
            border: 1px solid #BBEF63;
        }}

        .tab-content {{
            padding: 1.5rem 0;
        }}

        .sidebar-section {{
            margin-bottom: 1.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid {COLORS['border']};
        }}

        .sidebar-section:last-child {{
            border-bottom: none;
        }}

        .sidebar-title {{
            color: {COLORS['text_dark']};
            font-size: 0.95rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 1rem;
        }}

        [data-testid="stMetricDelta"] {{
            display: none;
        }}

        .comparison-table {{
            border-collapse: collapse;
            width: 100%;
        }}

        .comparison-table th {{
            background-color: {COLORS['light_bg']};
            border-bottom: 2px solid {COLORS['border']};
            padding: 0.75rem;
            text-align: left;
            font-weight: 700;
            color: {COLORS['text_dark']};
        }}

        .comparison-table td {{
            border-bottom: 1px solid {COLORS['border']};
            padding: 0.75rem;
            color: {COLORS['text_light']};
        }}

        .comparison-table tr:hover {{
            background-color: {COLORS['light_bg']};
        }}

        button[kind="primary"] {{
            background-color: {COLORS['primary']};
            border-color: {COLORS['primary']};
        }}

        button[kind="primary"]:hover {{
            background-color: {COLORS['secondary']};
            border-color: {COLORS['secondary']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Load and preprocess data."""
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
def train_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame, Dict]:
    """Train the Random Forest model and return metrics."""
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
        max_depth=15,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_prob),
    }

    # Feature importance
    importance_df = pd.DataFrame()
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        feature_scores = pipeline.named_steps["model"].feature_importances_
        importance_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": feature_scores})
            .sort_values("Importance", ascending=False)
            .head(10)
        )
        importance_df["Feature"] = importance_df["Feature"].str.replace("num__", "", regex=False)
        importance_df["Feature"] = importance_df["Feature"].str.replace("cat__", "", regex=False)
    except Exception:
        pass

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}

    return pipeline, metrics, importance_df, roc_data


def display_header() -> None:
    """Display professional header."""
    st.markdown(
        """
        <div class="main-header">
            <h1>CreditWise Loan Approval System</h1>
            <p>Advanced AI-Powered Lending Decision Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_metric_display(label: str, value: str, color: str = "primary") -> None:
    """Create a professional metric display."""
    color_hex = COLORS.get(color, COLORS["primary"])
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color: {color_hex};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def input_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Create sidebar input panel."""
    st.sidebar.markdown(
        f'<div class="sidebar-title">Applicant Information</div>',
        unsafe_allow_html=True,
    )

    payload: Dict[str, object] = {}

    # Numeric inputs
    with st.sidebar.expander("Financial Profile", expanded=True):
        for col in ["Applicant_Income", "Coapplicant_Income", "Savings", "Loan_Amount", "Loan_Term"]:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            median = float(series.median()) if not series.empty else 0.0
            min_val = float(series.min()) if not series.empty else 0.0
            max_val = float(series.max()) if not series.empty else median + 1000.0
            payload[col] = st.number_input(
                col.replace("_", " "),
                min_value=min_val,
                max_value=max_val,
                value=median,
                step=1.0,
            )

    with st.sidebar.expander("Credit & Loan Details", expanded=True):
        for col in ["Credit_Score", "Existing_Loans", "DTI_Ratio", "Collateral_Value"]:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            median = float(series.median()) if not series.empty else 0.0
            min_val = float(series.min()) if not series.empty else 0.0
            max_val = float(series.max()) if not series.empty else median + 1000.0
            step = 0.01 if col == "DTI_Ratio" else 1.0
            payload[col] = st.number_input(
                col.replace("_", " "),
                min_value=min_val,
                max_value=max_val,
                value=median,
                step=step,
            )

    with st.sidebar.expander("Personal Details", expanded=True):
        for col in ["Age", "Dependents"]:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            median = float(series.median()) if not series.empty else 0.0
            min_val = float(series.min()) if not series.empty else 0.0
            max_val = float(series.max()) if not series.empty else median + 1000.0
            payload[col] = st.number_input(
                col.replace("_", " "),
                min_value=min_val,
                max_value=max_val,
                value=median,
                step=1.0,
            )

    # Categorical inputs
    st.sidebar.markdown(
        f'<div class="sidebar-title" style="margin-top: 1.5rem;">Categorical Information</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("Employment & Background", expanded=True):
        for col in ["Employment_Status", "Employer_Category", "Education_Level", "Gender"]:
            if col not in df.columns:
                continue
            options: List[str] = (
                df[col].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
            )
            options = sorted(options)
            payload[col] = st.selectbox(col.replace("_", " "), options, key=col)

    with st.sidebar.expander("Family & Property", expanded=True):
        for col in ["Marital_Status", "Property_Area", "Loan_Purpose"]:
            if col not in df.columns:
                continue
            options: List[str] = (
                df[col].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
            )
            options = sorted(options)
            payload[col] = st.selectbox(col.replace("_", " "), options, key=col)

    return pd.DataFrame([payload])


def display_dashboard_metrics(df: pd.DataFrame, metrics: Dict[str, float]) -> None:
    """Display key dashboard metrics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        create_metric_display(
            "Total Applications",
            f"{len(df):,}",
            "primary"
        )

    with col2:
        approval_rate = df[TARGET_COLUMN].mean() * 100
        create_metric_display(
            "Approval Rate",
            f"{approval_rate:.1f}%",
            "success" if approval_rate > 50 else "warning"
        )

    with col3:
        avg_score = df['Credit_Score'].median()
        create_metric_display(
            "Avg Credit Score",
            f"{avg_score:.0f}",
            "primary"
        )

    with col4:
        avg_loan = df['Loan_Amount'].median()
        create_metric_display(
            "Avg Loan Amount",
            f"₹{avg_loan:,.0f}",
            "primary"
        )

    st.divider()

    # Model performance metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        create_metric_display(
            "Accuracy",
            f"{metrics['Accuracy'] * 100:.1f}%",
            "primary"
        )

    with col2:
        create_metric_display(
            "Precision",
            f"{metrics['Precision'] * 100:.1f}%",
            "primary"
        )

    with col3:
        create_metric_display(
            "Recall",
            f"{metrics['Recall'] * 100:.1f}%",
            "secondary"
        )

    with col4:
        create_metric_display(
            "F1 Score",
            f"{metrics['F1 Score'] * 100:.1f}%",
            "secondary"
        )

    with col5:
        create_metric_display(
            "ROC AUC",
            f"{metrics['ROC AUC'] * 100:.1f}%",
            "success"
        )


def display_prediction_results(prediction: int, confidence: float) -> None:
    """Display prediction decision with visualization."""
    decision_class = "decision-approved" if prediction == 1 else "decision-rejected"
    status_text = "APPROVED" if prediction == 1 else "REJECTED"

    st.markdown(
        f"""
        <div class="decision-card {decision_class}">
            <div class="decision-status">{status_text}</div>
            <div style="color: {COLORS['text_light']}; font-size: 0.9rem; margin-bottom: 0.5rem;">Approval Confidence</div>
            <div style="color: {COLORS['text_dark']}; font-size: 1.25rem; font-weight: 700;">{confidence * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence bar
    st.markdown(
        f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {min(max(confidence, 0.0), 1.0) * 100}%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_risk_assessment(applicant_data: pd.DataFrame, df: pd.DataFrame) -> None:
    """Display risk assessment for applicant."""
    st.markdown(
        f'<h3 class="section-title">Risk Assessment</h3>',
        unsafe_allow_html=True,
    )

    risks = []

    # Credit score risk
    credit_score = float(applicant_data["Credit_Score"].iloc[0])
    median_credit = float(pd.to_numeric(df["Credit_Score"], errors="coerce").median())
    if credit_score < median_credit * 0.8:
        risks.append(("High", "Low Credit Score", f"Score {credit_score} is below 80% of median"))

    # DTI ratio risk
    dti_ratio = float(applicant_data["DTI_Ratio"].iloc[0])
    if dti_ratio > 0.4:
        risks.append(("High", "High Debt-to-Income Ratio", f"DTI {dti_ratio:.2f} exceeds recommended 40%"))

    # Savings risk
    savings = float(applicant_data["Savings"].iloc[0])
    loan_amount = float(applicant_data["Loan_Amount"].iloc[0])
    if savings < loan_amount * 0.1:
        risks.append(("Medium", "Low Savings Buffer", "Savings less than 10% of loan amount"))

    # Loan term risk
    loan_term = float(applicant_data["Loan_Term"].iloc[0])
    if loan_term > 360:
        risks.append(("Medium", "Long Loan Term", f"Term of {int(loan_term)} months may indicate tight affordability"))

    if not risks:
        st.markdown(
            """
            <div class="info-box">
                <strong>Positive Assessment:</strong> No significant risk factors identified for this applicant.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for risk_level, risk_title, risk_desc in risks:
            risk_class = "risk-high" if risk_level == "High" else ("risk-medium" if risk_level == "Medium" else "risk-low")
            st.markdown(
                f"""
                <div class="risk-indicator {risk_class}">
                    <strong>{risk_title}:</strong> {risk_desc}
                </div>
                """,
                unsafe_allow_html=True,
            )


def create_feature_importance_chart(importance_df: pd.DataFrame) -> None:
    """Create interactive feature importance chart."""
    if not importance_df.empty:
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            labels={"Importance": "Feature Importance Score", "Feature": "Features"},
            color="Importance",
            color_continuous_scale="Blues",
            height=400
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="system-ui", size=11),
            coloraxis_showscale=False,
            xaxis_title="",
            yaxis_title="",
        )
        fig.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Feature importance data is not available.")


def create_roc_curve_chart(roc_data: Dict) -> None:
    """Create ROC curve visualization."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=roc_data["fpr"],
        y=roc_data["tpr"],
        name=f'ROC Curve (AUC = {roc_data["auc"]:.3f})',
        line=dict(color=COLORS["secondary"], width=2.5),
        mode='lines',
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash'),
        mode='lines',
    ))

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
        hovermode='closest',
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="system-ui", size=11),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_xaxes(gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(gridwidth=1, gridcolor="lightgray")

    st.plotly_chart(fig, use_container_width=True)


def create_approval_distribution(df: pd.DataFrame) -> None:
    """Create approval distribution visualization."""
    approval_data = df[TARGET_COLUMN].value_counts()
    labels = ["Approved", "Rejected"]
    values = [approval_data.get(1, 0), approval_data.get(0, 0)]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=[COLORS["success"], COLORS["danger"]]),
        textposition='inside',
        textinfo='label+percent',
        hoverinfo='label+value',
    )])

    fig.update_layout(
        title="Application Approval Distribution",
        height=400,
        paper_bgcolor="white",
        font=dict(family="system-ui", size=11),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def display_comparative_analysis(applicant_data: pd.DataFrame, df: pd.DataFrame) -> None:
    """Display how applicant compares to dataset."""
    st.markdown(
        f'<h3 class="section-title">Applicant Comparison Analysis</h3>',
        unsafe_allow_html=True,
    )

    comparison_data = []

    for col in ["Applicant_Income", "Credit_Score", "Age", "Loan_Amount", "DTI_Ratio"]:
        if col not in df.columns:
            continue

        applicant_val = float(applicant_data[col].iloc[0])
        dataset_values = pd.to_numeric(df[col], errors="coerce").dropna()
        dataset_median = float(dataset_values.median())
        percentile = (dataset_values <= applicant_val).sum() / len(dataset_values) * 100

        comparison_data.append({
            "Metric": col.replace("_", " "),
            "Applicant": f"{applicant_val:,.0f}",
            "Dataset Median": f"{dataset_median:,.0f}",
            "Percentile": f"{percentile:.1f}%",
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def main() -> None:
    """Main application function."""
    apply_custom_styling()
    display_header()

    # Load data and train model
    df = load_data(DATA_PATH)
    model, metrics, importance_df, roc_data = train_model(df)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Analytics", "Model Performance", "Data Explorer"])

    # ============= TAB 1: PREDICTION =============
    with tab1:
        st.markdown(
            f'<h3 class="section-title">Loan Application Prediction</h3>',
            unsafe_allow_html=True,
        )

        applicant_df = input_panel(df)

        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.markdown(
                f'<h4 style="color: {COLORS["text_dark"]}; font-weight: 700; margin-top: 1rem;">Decision Engine</h4>',
                unsafe_allow_html=True,
            )
            if st.button("Generate Prediction", type="primary", use_container_width=True):
                prediction = int(model.predict(applicant_df)[0])
                confidence = float(model.predict_proba(applicant_df)[0][1])

                display_prediction_results(prediction, confidence)

                display_risk_assessment(applicant_df, df)

        with col2:
            st.markdown(
                f'<h4 style="color: {COLORS["text_dark"]}; font-weight: 700; margin-top: 1rem;">Applicant Summary</h4>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="info-box">
                    <strong>Income:</strong> ₹{applicant_df['Applicant_Income'].values[0]:,.0f}<br>
                    <strong>Credit Score:</strong> {applicant_df['Credit_Score'].values[0]:.0f}<br>
                    <strong>Age:</strong> {applicant_df['Age'].values[0]:.0f} years<br>
                    <strong>Loan Amount:</strong> ₹{applicant_df['Loan_Amount'].values[0]:,.0f}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ============= TAB 2: ANALYTICS =============
    with tab2:
        st.markdown(
            f'<h3 class="section-title">Comprehensive Analytics</h3>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            create_approval_distribution(df)

        with col2:
            create_feature_importance_chart(importance_df)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f'<h4 style="color: {COLORS["text_dark"]}; font-weight: 700;">Income Distribution</h4>',
                unsafe_allow_html=True,
            )
            fig = px.histogram(
                df,
                x="Applicant_Income",
                nbins=50,
                labels={"Applicant_Income": "Applicant Income"},
                color_discrete_sequence=[COLORS["secondary"]]
            )
            fig.update_layout(
                height=350,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="system-ui", size=10),
                margin=dict(l=30, r=30, t=20, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col2:
            st.markdown(
                f'<h4 style="color: {COLORS["text_dark"]}; font-weight: 700;">Credit Score Distribution</h4>',
                unsafe_allow_html=True,
            )
            fig = px.histogram(
                df,
                x="Credit_Score",
                nbins=50,
                labels={"Credit_Score": "Credit Score"},
                color_discrete_sequence=[COLORS["success"]]
            )
            fig.update_layout(
                height=350,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="system-ui", size=10),
                margin=dict(l=30, r=30, t=20, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ============= TAB 3: MODEL PERFORMANCE =============
    with tab3:
        st.markdown(
            f'<h3 class="section-title">Model Performance Metrics</h3>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            create_metric_display("Accuracy", f"{metrics['Accuracy'] * 100:.2f}%", "primary")

        with col2:
            create_metric_display("Precision", f"{metrics['Precision'] * 100:.2f}%", "secondary")

        with col3:
            create_metric_display("Recall", f"{metrics['Recall'] * 100:.2f}%", "warning")

        with col4:
            create_metric_display("F1 Score", f"{metrics['F1 Score'] * 100:.2f}%", "success")

        with col5:
            create_metric_display("ROC AUC", f"{metrics['ROC AUC'] * 100:.2f}%", "secondary")

        st.divider()

        st.markdown(
            f'<h4 style="color: {COLORS["text_dark"]}; font-weight: 700; margin-top: 1.5rem;">ROC Curve Analysis</h4>',
            unsafe_allow_html=True,
        )
        create_roc_curve_chart(roc_data)

    # ============= TAB 4: DATA EXPLORER =============
    with tab4:
        st.markdown(
            f'<h3 class="section-title">Dataset Explorer</h3>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            create_metric_display("Total Records", f"{len(df):,}", "primary")

        with col2:
            create_metric_display("Approved", f"{int(df[TARGET_COLUMN].sum()):,}", "success")

        with col3:
            create_metric_display("Rejected", f"{int((1-df[TARGET_COLUMN]).sum()):,}", "danger")

        st.divider()

        # Data filtering
        st.markdown(
            f'<h4 style="color: {COLORS["text_dark"]}; font-weight: 700; margin-top: 1.5rem;">Filter Data</h4>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            min_income = st.slider(
                "Minimum Income",
                float(df["Applicant_Income"].min()),
                float(df["Applicant_Income"].max()),
                float(df["Applicant_Income"].min()),
            )

        with col2:
            min_credit = st.slider(
                "Minimum Credit Score",
                float(df["Credit_Score"].min()),
                float(df["Credit_Score"].max()),
                float(df["Credit_Score"].min()),
            )

        filtered_df = df[
            (df["Applicant_Income"] >= min_income) &
            (df["Credit_Score"] >= min_credit)
        ]

        st.markdown(
            f'<h4 style="color: {COLORS["text_dark"]}; font-weight: 700; margin-top: 1.5rem;">Filtered Data ({len(filtered_df)} records)</h4>',
            unsafe_allow_html=True,
        )
        st.dataframe(filtered_df.head(20), use_container_width=True, height=400)


if __name__ == "__main__":
    main()