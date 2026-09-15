# CreditWise Loan Approval System

**An intelligent machine learning system for automating loan approval decisions using historical applicant data.**

## Overview

CreditWise is a machine learning-based loan approval prediction system designed for SecureTrust Bank to automate and optimize the loan approval process. The system analyzes applicant information and predicts whether a loan should be approved or rejected, reducing manual effort while improving decision consistency and accuracy.

## Problem Statement

SecureTrust Bank faces significant challenges with manual loan verification:
- Good customers are sometimes rejected, leading to lost business opportunities
- High-risk customers are sometimes approved, resulting in financial losses
- Manual verification is time-consuming, biased, and inconsistent

This project addresses these challenges by developing an automated system that provides fast, accurate, and unbiased loan approval recommendations.

## Key Features

- Comprehensive data preprocessing and cleaning
- Exploratory data analysis with statistical insights
- Multiple machine learning models for loan approval prediction
- Handling of missing values and categorical variables
- Feature engineering and encoding
- Model evaluation with performance metrics
- Binary classification (Approved/Rejected)

## Technology Stack

- Python 3.x
- pandas - Data manipulation and analysis
- scikit-learn - Machine learning algorithms and preprocessing
- NumPy - Numerical computations
- Jupyter Notebook - Interactive development environment
- Streamlit - Live interactive web application UI

## Dataset

The dataset contains 18 features describing applicant demographics, financial status, and loan details:

| Feature | Description |
|---------|-------------|
| Applicant_Income | Monthly income of applicant |
| Coapplicant_Income | Monthly income of co-applicant |
| Employment_Status | Salaried / Self-Employed / Business |
| Age | Applicant age |
| Marital_Status | Married / Single |
| Credit_Score | Credit bureau score |
| DTI_Ratio | Debt-to-Income ratio |
| Loan_Amount | Loan amount requested |
| Loan_Term | Loan duration (months) |
| Loan_Approved | Target variable (1 = Approved, 0 = Rejected) |

## Project Structure

```
loan-approval-system/
├── app.py                          # Streamlit live prediction app
├── requirements.txt                # Python dependencies
├── loan_approval_modeling.ipynb    # Main project notebook
├── CreditWise_Loan_System.pdf      # Project specification
├── loan_approval_data.csv          # Dataset
└── README.md                       # This file
```

## Getting Started

### Prerequisites

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/creditwise-loan-approval.git
cd creditwise-loan-approval
```

2. Launch the Streamlit app:
```bash
streamlit run app.py
```

3. (Optional) Launch Jupyter Notebook for model experimentation:
```bash
jupyter notebook loan_approval_modeling.ipynb
```

4. In the Streamlit app:
   - Enter applicant financial and demographic details from the sidebar
   - Click **Predict Approval** to get instant recommendation with confidence
   - Review model quality metrics and top decision drivers

5. In the notebook, run cells sequentially to:
   - Load and explore the dataset
   - Perform data cleaning and preprocessing
   - Train machine learning models
   - Evaluate model performance

## Workflow

1. **Data Loading** - Import loan application dataset
2. **Exploratory Analysis** - Understand data distributions and patterns
3. **Data Cleaning** - Handle missing values and duplicates
4. **Feature Preprocessing** - Encode categorical variables
5. **Model Training** - Train classification models
6. **Model Evaluation** - Assess performance metrics

## Model Approach

The project implements classification models to predict loan approval status as a binary outcome:
- **Target Variable**: Loan_Approved (1 = Approved, 0 = Rejected)
- **Type**: Supervised Learning - Binary Classification
- **Preprocessing**: Handling missing values, categorical encoding, feature scaling

## Results and Insights

The model successfully:
- Identifies key factors influencing loan approval decisions
- Provides consistent predictions based on applicant data
- Reduces manual verification time and bias
- Improves overall decision accuracy

## Future Enhancements

- Hyperparameter tuning and model optimization
- Ensemble methods for improved predictions
- Feature importance analysis
- Model deployment as REST API
- Real-time prediction system integration

## Contributing

Contributions are welcome. Please feel free to submit pull requests or open issues for improvements.

## License

This project is provided as-is for educational and commercial purposes.

## Author

Machine Learning Engineer - SecureTrust Bank Project

---

**Note**: This project is based on the CreditWise Loan System specification developed for automating loan approval decisions in financial institutions.
