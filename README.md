# Responsible AI: Bias Detection & Model Interpretability

An end-to-end ethical machine learning project built with Python, Scikit-Learn, and Streamlit. It demonstrates how machine learning models trained on historical financial data can inherit demographic bias—and how fairness auditing and explainability tools can expose those disparities in real time.

## Features
* **Demographic Parity Analysis:** Evaluates approval rate disparities across gender groups during model training.
* **Counterfactual Bias Detection:** Tests identical financial profiles across genders in real time and flags fairness alerts when predictions flip.
* **Model Interpretability:** Visualizes feature contribution scores to explain which attributes drive model decisions.

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model and generate artifacts:
   ```bash
   python responsible_ai_bias_interpretability.py
   ```
3. Launch the Streamlit dashboard:
   ```bash
   python -m streamlit run app.py
   ```
