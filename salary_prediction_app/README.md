# 💰 Developer Salary Predictor

A machine learning web application that predicts developer salaries based on the [Stack Overflow 2022 Developer Survey](https://survey.stackoverflow.co/2022/) (73,268 responses).

Built with **CatBoost** and deployed via **Streamlit**.

---

## 🎯 Overview

This project walks through a complete end-to-end machine learning pipeline — from raw survey data to an interactive web app:

- **Data Cleaning** — handling missing values, outliers, and multi-select fields
- **Exploratory Data Analysis** — 12+ visualizations uncovering salary drivers
- **Feature Engineering** — ordinal encoding, log transforms, and categorical grouping
- **Model Comparison** — PyCaret benchmark across 25+ algorithms
- **Hyperparameter Tuning** — GridSearchCV with overfitting diagnostics
- **Interpretability** — SHAP analysis for global and local explanations
- **Deployment** — interactive Streamlit app for real-time predictions

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Model | CatBoost Regressor |
| Training samples | ~32,000 cleaned observations |
| Salary range | 10K – 300K USD |
| Test R² | **0.571** |
| Test MAE | **~$26,000** |
| Generalization gap | 0.005 (no overfitting) |

## 🚀 Run Locally

**1. Clone the repository**

    git clone https://github.com/yourusername/salary-prediction.git
    cd salary-prediction

**2. Install dependencies**

    pip install -r requirements.txt

**3. Run the Streamlit app**

    streamlit run salary_pred_app.py

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

    .
    ├── salary_pred_app.py                # Streamlit web app
    ├── salary_predictor_pipeline.pkl     # Trained CatBoost pipeline
    ├── model_metadata.json               # Feature categories and metadata
    ├── requirements.txt                  # Python dependencies
    ├── notebook.ipynb                    # Full analysis notebook (Kaggle)
    └── README.md

---

## 🔍 Key Insights from SHAP Analysis

- **Country (USA)** is the dominant salary driver, contributing up to +60K USD.
- **Years of professional experience** shows a strong, monotonic positive effect.
- **Organization size** (log-scaled) positively impacts predicted salary.
- **Remote work** commands a premium compared to full in-person roles.
- **Education** is a surprisingly weak signal compared to geography and experience.

---

## ⚠️ Limitations

- The model was trained on salaries between **10K and 300K USD**. Predictions outside this range may be unreliable.
- The dataset is heavily skewed toward male respondents (92%) and US-based developers, which may introduce bias.
- Salary is self-reported and may contain measurement errors.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **scikit-learn**, **CatBoost**, **LightGBM**, **XGBoost**
- **PyCaret** for rapid model benchmarking
- **SHAP** for model interpretability
- **Streamlit** for deployment
- **pandas**, **numpy**, **matplotlib**, **seaborn** for data analysis

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Stack Overflow Developer Survey 2022](https://survey.stackoverflow.co/2022/)
- [Kaggle](https://www.kaggle.com/datasets/soheiltehranipour/stackoverflow-developer-suvery-2022) for the notebook environment
