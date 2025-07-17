import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import plotly.graph_objects as go
import io
from fpdf import FPDF
from datetime import datetime
from PIL import Image


# Load models
risk_model = joblib.load('E:/hospital_analytics_project/models/risk_scoring_model.pkl')
stay_model = joblib.load('E:/hospital_analytics_project/models/stay_length_model.pkl')

# Title
st.title("🏥 Heart Failure Risk & Stay Prediction Dashboard")

st.markdown("Enter patient details to get predictions for:")
st.markdown("- Risk of death event")
st.markdown("- Expected hospital stay duration (follow-up time)")

# Input fields
age = st.slider("Age", 40, 95, 60)
ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 38)
serum_creatinine = st.slider("Serum Creatinine (mg/dL)", 0.5, 10.0, 1.2)
serum_sodium = st.slider("Serum Sodium (mEq/L)", 110, 150, 137)
platelets = st.number_input("Platelets", min_value=25000, max_value=850000, value=250000)
cpk = st.number_input("Creatinine Phosphokinase", min_value=20, max_value=8000, value=200)

sex = st.selectbox("Sex", ["Male", "Female"])
anaemia = st.checkbox("Anaemia")
diabetes = st.checkbox("Diabetes")
high_bp = st.checkbox("High Blood Pressure")
smoking = st.checkbox("Smoking")

# Derived features
is_elderly = int(age > 65)
low_ejection = int(ejection_fraction < 30)
high_creatinine = int(serum_creatinine > 1.5)
combined_risk_score = is_elderly + low_ejection + high_creatinine
sex_val = 1 if sex == "Male" else 0

# Common input dictionary
base_input = {
    "age": age,
    "anaemia": int(anaemia),
    "creatinine_phosphokinase": cpk,
    "diabetes": int(diabetes),
    "ejection_fraction": ejection_fraction,
    "high_blood_pressure": int(high_bp),
    "platelets": platelets,
    "serum_creatinine": serum_creatinine,
    "serum_sodium": serum_sodium,
    "sex": sex_val,
    "smoking": int(smoking),
    "is_elderly": is_elderly,
    "low_ejection": low_ejection,
    "high_creatinine": high_creatinine,
    "combined_risk_score": combined_risk_score
}

# Add 'time' for risk model input
risk_input = base_input.copy()
risk_input["time"] = 140  # Default or user-defined value

# Prepare dataframes
risk_input_df = pd.DataFrame([risk_input])
stay_input_df = pd.DataFrame([base_input])

# Reorder columns to match training
try:
    risk_input_df = risk_input_df[risk_model.feature_names_in_]
    stay_input_df = stay_input_df[stay_model.feature_names_in_]
except AttributeError:
    st.error("❌ The model was trained in an older version of scikit-learn that does not store feature names. Please retrain it with scikit-learn ≥1.0 or manually define the column order.")

# Prediction
# Prediction and Visualization
if st.button("Predict"):
    # Predict
    risk_prob = risk_model.predict_proba(risk_input_df)[0][1]
    stay_pred = stay_model.predict(stay_input_df)[0]

    # Gauge Chart
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "🧠 Estimated Risk of Death (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "crimson"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_prob * 100
            }
        }
    ))

    # Stay Length Chart
    stay_fig = go.Figure(go.Bar(
        x=["Predicted Stay Duration"],
        y=[stay_pred],
        marker_color='steelblue',
        text=[f"{stay_pred:.1f} days"],
        textposition='outside'
    ))
    stay_fig.update_layout(
        title="⏳ Predicted Follow-up Duration (days)",
        yaxis_title="Days",
        height=400
    )

    # Show charts and results
    st.plotly_chart(gauge_fig, use_container_width=True)
    st.plotly_chart(stay_fig, use_container_width=True)

    st.success(f"🧠 Risk of Death: {risk_prob*100:.2f}%")
    st.info(f"⏳ Predicted Stay: {stay_pred:.1f} days")

    # ===================== CSV Export =====================
    import io
    from fpdf import FPDF
    from datetime import datetime

    results_df = pd.DataFrame([{
        **base_input,
        "risk_probability (%)": round(risk_prob * 100, 2),
        "predicted_followup_days": round(stay_pred, 1)
    }])

    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False, sep=';')  # ← Use semicolon
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

    # ===================== PDF Export =====================
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 14)
            self.cell(0, 10, "Heart Failure Prediction Report", ln=True, align="C")  # Removed emoji
            self.ln(5)

        def footer(self):
            self.set_y(-20)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, 'C')

        def add_result_table(self, data):
            self.set_font("Arial", "", 10)
            for key, value in data.items():
                self.cell(60, 8, f"{key}:", border=0)
                self.cell(0, 8, f"{value}", ln=True, border=0)

        def add_signature(self):
            self.ln(10)
            self.set_font("Arial", "I", 10)
            self.cell(0, 10, "Doctor's Signature: ____________________", ln=True)
            self.ln(5)
            self.cell(0, 10, "Stamp/Official Note (Optional)", ln=True)
            
    # PDF content
    pdf_data_dict = {
        "Age": age,
        "Ejection Fraction (%)": ejection_fraction,
        "Serum Creatinine (mg/dL)": serum_creatinine,
        "Serum Sodium (mEq/L)": serum_sodium,
        "Platelets": platelets,
        "Creatinine Phosphokinase": cpk,
        "Sex": sex,
        "Anaemia": bool(anaemia),
        "Diabetes": bool(diabetes),
        "High Blood Pressure": bool(high_bp),
        "Smoking": bool(smoking),
        "Estimated Risk of Death (%)": f"{risk_prob * 100:.2f}",
        "Predicted Follow-up Duration (days)": f"{stay_pred:.1f}"

    }

    pdf = PDF()
    pdf.add_page()
    pdf.add_result_table(pdf_data_dict)
    pdf.add_signature()

    pdf_data = pdf.output(dest='S').encode('latin-1')

    st.download_button(
        label="📄 Download Results as PDF",
        data=pdf_data,
        file_name="prediction_report.pdf",
        mime="application/pdf"
    )
