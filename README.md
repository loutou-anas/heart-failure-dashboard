# 🏥 Heart Failure Risk & Stay Duration Prediction

This project is a complete end-to-end data science solution designed to predict the risk of death and expected hospital stay duration for heart failure patients. It demonstrates both data analyst and data scientist skills, integrating EDA, feature engineering, machine learning, and deployment through a Streamlit dashboard.

---

## 🚀 Project Features

- 📊 **Exploratory Data Analysis (EDA)** using Pandas, Seaborn, and Matplotlib
- 🧠 **Classification Model**: Predicts death risk using Random Forest
- ⏳ **Regression Model**: Predicts follow-up duration using Gradient Boosting
- 🖥️ **Streamlit Dashboard** for interactive patient risk assessment
- 🧪 Fully reproducible notebooks and saved ML models
- 📁 Modular and clean project structure

---

## 🧱 Project Structure

```
project_root/
├── data/
│   ├── raw/
│   │   └── heart_failure_clinical_records_dataset.csv
|   └── processed/
├── models/
│   └── *.pkl
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_risk_scoring_model.ipynb
│   └── 04_length_of_stay_model.ipynb
├── dashboard/
│   └── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📊 Dataset

- Source: [UCI Heart Failure Clinical Records Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Failure+Clinical+Records)
- 299 patient records with 13 clinical features

---

## 🧪 Model Performance

### Classification (Death Risk)
- **Accuracy**: 73%
- **ROC-AUC**: 0.85
- **Model**: RandomForestClassifier

### Regression (Stay Length)
- **MAE**: ~68 days
- **RMSE**: ~82 days
- **Model**: GradientBoostingRegressor

---

## ▶️ Running the Dashboard

```bash
# # (Optional) Activate virtual environment

# If on PowerShell (Windows):
$env:Path += ";C:\Users\<YourUsername>\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts"

# Then run:
streamlit run dashboard/app.py
```

---

## 📄 Author & License

**Author**: Anas Loutou
**License**: MIT

---

## 🙌 Contributions

Contributions, issues, and feature requests are welcome!
