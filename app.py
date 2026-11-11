import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Responsible AI Model Incorporating Bias Detection and Model Interpretability",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------
scaler = joblib.load("scaler.pkl")
model = joblib.load("responsible_ai_model.pkl")

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>Responsible AI Model Incorporating Bias Detection and Model Interpretability</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Fair • Transparent • Explainable</p>", unsafe_allow_html=True)

st.divider()

# ---------------- LAYOUT ----------------
left, right = st.columns(2, gap="large")

# ================= INPUT SECTION =================
with left:
    with st.container(border=True):
        st.subheader("User Profile")

        age = st.slider("Age", 18, 65, 30)
        income = st.number_input("Annual Income", 10000, 200000, 60000)
        credit_score = st.slider("Credit Score", 300, 850, 650)
        gender = st.selectbox("Gender", ["Female", "Male"])

        gender_val = 0 if gender == "Female" else 1
        columns = ["age", "income", "credit_score", "gender"]

        input_df = pd.DataFrame(
            [[age, income, credit_score, gender_val]],
            columns=columns
        )

        predict = st.button("Predict Bias", use_container_width=True, type="primary")

# ================= OUTPUT SECTION =================
with right:
    with st.container(border=True):
        st.subheader("Model Prediction & Explainability")

        if predict:
            scaled = pd.DataFrame(scaler.transform(input_df), columns=columns)
            prediction = int(model.predict(scaled)[0])
            probability = float(model.predict_proba(scaled)[0][prediction]) * 100
            decision = "Approved" if prediction == 1 else "Rejected"

            colA, colB = st.columns(2)
            with colA:
                st.metric("Decision", decision)
            with colB:
                st.metric("Confidence", f"{probability:.2f}%")

            # Counterfactual Bias Check
            st.divider()
            st.markdown("### Counterfactual Bias Check")

            female_df = pd.DataFrame([[age, income, credit_score, 0]], columns=columns)
            male_df = pd.DataFrame([[age, income, credit_score, 1]], columns=columns)

            f_scaled = pd.DataFrame(scaler.transform(female_df), columns=columns)
            m_scaled = pd.DataFrame(scaler.transform(male_df), columns=columns)

            f_pred = int(model.predict(f_scaled)[0])
            m_pred = int(model.predict(m_scaled)[0])

            f_approval_prob = float(model.predict_proba(f_scaled)[0][1]) * 100
            m_approval_prob = float(model.predict_proba(m_scaled)[0][1]) * 100
            prob_gap = abs(m_approval_prob - f_approval_prob)

            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "Female Profile",
                    "Approved" if f_pred == 1 else "Rejected",
                    f"{f_approval_prob:.1f}% Approval Odds"
                )
            with c2:
                st.metric(
                    "Male Profile",
                    "Approved" if m_pred == 1 else "Rejected",
                    f"{m_approval_prob:.1f}% Approval Odds"
                )

            if f_pred != m_pred:
                st.error(f"Bias Detected: Changing only gender flipped the decision (Approval odds gap: {prob_gap:.1f}%).")
            elif prob_gap > 5.0:
                st.warning(f"Potential Bias: Decision stayed the same, but approval odds shifted by {prob_gap:.1f}% based on gender.")
            else:
                st.success(f"Fair Outcome: Consistent decision across genders (Approval odds gap: {prob_gap:.1f}%).")

            # Feature Contribution Chart
            st.divider()
            st.markdown("### Feature Contribution")

            importance = model.feature_importances_
            features = ["Age", "Income", "Credit Score", "Gender"]

            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.barh(features, importance)
            ax.set_xlabel("Impact Score")
            ax.set_title("Global Model Explainability")
            st.pyplot(fig)
        else:
            st.info("Adjust the user profile on the left and click **Predict Bias** to view the decision and fairness analysis.")

st.divider()

# ---------------- BIAS SECTION ----------------
st.markdown(
    "<p style='text-align:center;color:#888;'>"
    "Bias Monitoring Enabled: Sensitive attributes are evaluated during inference to detect counterfactual disparities."
    "</p>",
    unsafe_allow_html=True
)


