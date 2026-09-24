# E-Commerce Customer Churn Prediction and Customer Behaviour Analytics

**Academic Internship Project:** IBM SkillsBuild Data Analytics with AI  
**Conducted by:** BharatCares in association with AICTE  
**Domain:** E-Commerce, Predictive Analytics, Customer Relationship Management (CRM)

---

## Project Overview

Customer churn represents one of the most critical operational risks for digital retail and e-commerce enterprises. Retaining an existing customer is widely estimated to be **5 to 25 times more cost-effective** than acquiring a replacement. This project delivers an end-to-end data analytics and machine learning solution that investigates customer purchasing behavior, performs behavioral RFM (Recency, Frequency, Monetary) segmentation, and trains supervised classification algorithms to detect at-risk customers before churn occurs.

The project encompasses a fully executed 19-section Jupyter notebook, an interactive decision-support web dashboard built with Streamlit, serialized scikit-learn machine learning pipelines, and a formal academic project report.

---

## Problem Statement

E-commerce businesses frequently struggle with:
1. **Detection Blindspots:** Lacking automated, real-time warning indicators to identify disengaging customers before attrition becomes permanent.
2. **Causal Ambiguity:** Inability to isolate and quantify the specific behavioral friction points (e.g., service grievances, logistics delay, low tenure) causing attrition.
3. **Inefficient Retention Spending:** Deploying blanket discount campaigns that erode profit margins rather than targeting high-risk, high-value cohorts.

---

## Objectives

- Ingest and preprocess a real-world e-commerce customer transaction dataset.
- Perform comprehensive exploratory data analysis (EDA) across univariate, bivariate, and multivariate distributions.
- Implement an **RFM (Recency, Frequency, Monetary)** behavioral segmentation to categorize customers into strategic cohorts (*Champions*, *Loyal Customers*, *Potential Loyalists*, *At Risk*, and *Hibernating*).
- Benchmark baseline **Logistic Regression** against an ensemble **Random Forest Classifier** with class-imbalance compensation.
- Evaluate models using multi-metric criteria: Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and ROC-AUC.
- Extract Gini impurity feature importances and logistic odds ratios to provide transparent model interpretability.
- Develop and deploy a local interactive **Streamlit** dashboard with live KPI tracking and customer churn simulation.

---

## Dataset

- **Dataset Source:** Kaggle E-Commerce Customer Churn and Retention Dataset
- **Dataset URL:** [https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
- **Author / Publisher:** Ankit Verma
- **License:** CC0 Public Domain
- **Records:** 5,630 customer observations
- **Attributes:** 20 features (14 numerical, 5 categorical, 1 target)
- **Target Variable:** `Churn` (Binary: `0 = Retained`, `1 = Churned`)
- **Empirical Churn Rate:** 16.84% (948 churned, 4,682 retained)

---

## Features & Schema

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `CustomerID` | Integer | Unique identifier for each customer |
| `Churn` | Integer (0/1) | Target label (1 = Churned, 0 = Retained) |
| `Tenure` | Float | Number of months the customer has engaged with the platform |
| `PreferredLoginDevice` | String | Customer authentication platform (`Mobile Phone`, `Computer`) |
| `CityTier` | Integer (1-3) | Municipal development tier |
| `WarehouseToHome` | Float | Distance in kilometers between fulfillment hub and customer home |
| `PreferredPaymentMode` | String | Payment method (`Debit Card`, `UPI`, `Credit Card`, `Cash on Delivery`, `E wallet`) |
| `Gender` | String | Customer gender (`Female`, `Male`) |
| `HourSpendOnApp` | Float | Average hours spent on app/website weekly |
| `NumberOfDeviceRegistered`| Integer | Total registered devices paired with customer profile |
| `PreferedOrderCat` | String | Main order category (`Laptop & Accessory`, `Mobile Phone`, `Fashion`, `Grocery`, `Others`) |
| `SatisfactionScore` | Integer (1-5)| Post-purchase customer satisfaction score |
| `MaritalStatus` | String | Marital status (`Single`, `Married`, `Divorced`) |
| `NumberOfAddress` | Integer | Count of shipping destinations added to profile |
| `Complain` | Binary (0/1) | Whether a complaint was logged in the preceding month |
| `OrderAmountHikeFromlastYear` | Float | Percentage increase in order value compared to prior year |
| `CouponUsed` | Float | Total promotional coupons redeemed in the preceding month |
| `OrderCount` | Float | Total order count placed in the preceding month |
| `DaySinceLastOrder` | Float | Recency metric: days elapsed since last purchase |
| `CashbackAmount` | Float | Average monetary cashback rewarded in the preceding month ($) |

---

## Technologies Used

- **Programming Language:** Python 3.11
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning & Preprocessing:** Scikit-Learn, Joblib
- **Web Application:** Streamlit
- **Computational Notebook:** Jupyter Notebook (`nbformat`, `nbclient`, `ipykernel`)
- **Reporting Engine:** Python-Docx

---

## Project Structure

```text
ecommerce-customer-churn/
│
├── data/
│   ├── dataset.csv                  # Raw dataset (CSV)
│   ├── dataset.xlsx                 # Original raw dataset with Data Dict (Excel)
│   └── cleaned_dataset.csv          # Preprocessed dataset with RFM segments
│
├── notebooks/
│   └── StudentName_EcommerceCustomerChurn.ipynb  # Fully executed 19-section notebook
│
├── models/
│   ├── churn_model.pkl              # Production Random Forest pipeline
│   ├── logistic_regression_model.pkl# Baseline Logistic Regression pipeline
│   └── model_metadata.json          # Benchmark metrics and schema metadata
│
├── reports/
│   └── figures/                     # 9 high-resolution generated analytical plots
│       ├── fig1_churn_distribution.png
│       ├── fig2_tenure_vs_churn.png
│       ├── fig3_complain_vs_churn.png
│       ├── fig4_category_performance.png
│       ├── fig5_rfm_segments.png
│       ├── fig6_correlation_heatmap.png
│       ├── fig7_confusion_matrices.png
│       ├── fig8_roc_curves.png
│       └── fig9_feature_importances.png
│
├── scripts/                         # Pipeline execution & reproduction scripts
│   ├── inspect_data.py              # Initial dataset inspection script
│   ├── train_and_evaluate.py        # ML training, evaluation & plot export script
│   ├── build_and_run_notebook.py    # Automated notebook builder & executor script
│   └── generate_report.py           # Programmatic Word report generator
│
├── app.py                           # Interactive Streamlit web application
├── churn_model.pkl                  # Model artifact for direct deployment
├── StudentName_EcommerceCustomerChurn.ipynb  # Root copy for submission compatibility
├── StudentName_ProjectReport.docx   # Comprehensive academic internship report
├── requirements.txt                 # Project library dependencies
├── .gitignore                       # Git ignore configuration
└── README.md                        # Project documentation
```

---

## Installation

### 1. Clone or Open the Workspace
Clone the repository from GitHub:
```bash
git clone https://github.com/mahekaggarwal17/E-Commerce-Customer-Churn-Prediction-and-Customer-Behaviour-Analytics.git
cd E-Commerce-Customer-Churn-Prediction-and-Customer-Behaviour-Analytics
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Notebook

To view and run the fully executed Jupyter Notebook:
```bash
jupyter notebook notebooks/StudentName_EcommerceCustomerChurn.ipynb
```
Or open the notebook within VS Code / Antigravity IDE and select the Python 3.11 virtual environment kernel.

---

## Running the Dashboard

Launch the interactive Streamlit analytics and simulation dashboard locally:
```bash
streamlit run app.py
```
The dashboard will open automatically in your default browser at `http://localhost:8501`.

### Key Dashboard Views:
1. **📊 Executive Overview:** Executive KPI summary cards, churn vs. active proportion donut chart, and RFM segment distribution.
2. **👥 Customer Behaviour & RFM:** Granular tables and interactive Plotly charts breaking down RFM cohorts, categories, and payment channels.
3. **⚠️ Churn Driver Analysis:** Empirical deep dive into tenure vulnerability, customer complaint friction, city tiers, and delivery distance.
4. **🤖 AI Model Evaluation:** Comparative performance table, confusion matrices, ROC curves, and feature importance charts.
5. **🎯 Live Churn Predictor:** Interactive customer profile input form generating real-time churn risk classifications, gauge probability meters, and tailored retention advice.

---

## Machine Learning Models

Two complementary algorithms were trained using an 80/20 stratified train-test split:
1. **Logistic Regression (Baseline):** Linear classifier regularized with L2 penalty, parameterized with `class_weight='balanced'` for direct odds-ratio interpretability.
2. **Random Forest Classifier (Ensemble):** Ensemble of 150 de-correlated decision trees (`max_depth=12`, `class_weight='balanced'`) capturing complex non-linear feature interactions without overfitting.

---

## Benchmark Results

All metrics were evaluated on the independent holdout test set (1,126 customers; 936 Retained, 190 Churned):

| Evaluation Metric | Logistic Regression (Balanced) | Random Forest Classifier (Balanced) |
| :--- | :---: | :---: |
| **Accuracy** | 79.22% | **96.98%** |
| **Precision (Churn Class)** | 43.99% | **85.45%** |
| **Recall (Churn Class)** | 84.74% | **98.95%** |
| **F1-Score (Churn Class)** | 0.5791 | **0.9171** |
| **ROC-AUC** | 0.8851 | **0.9971** |
| **Confusion Matrix (TN / FP / FN / TP)** | 731 / 205 / 29 / 161 | **904 / 32 / 2 / 188** |

*Note: In customer churn management, **Recall** is prioritized to capture at-risk accounts. Random Forest successfully identified **188 out of 190 actual churners** (98.95% recall) with only 2 false negatives.*

---

## Key Insights

1. **Tenure is the #1 Retention Determinant:** Over 40% of customers churn during the first 0 to 4 months of relationship tenure. Once a customer reaches 10+ months, retention stabilizes above 90%.
2. **Customer Service Complaints Multiply Attrition:** Customers logging a complaint in the preceding month exhibit a **31.67% churn rate**, nearly **3 times higher** than non-complainants (10.93%).
3. **RFM Cohort Disparity:** 'Potential Loyalists' exhibit the highest churn vulnerability (24.35%), proving that recent buyers without established recurring habits require structured onboarding.
4. **Logistics Friction in Outer Tiers:** Tier 3 cities face elevated churn (21.37%) compared to Tier 1 (14.51%), strongly correlated with physical distance from fulfillment centers.
5. **Product Category Stickiness:** 'Mobile Phone' buyers experience elevated churn (27.40%), whereas 'Grocery' consumers churn at only 4.88%, highlighting the retention value of everyday repeat essentials.

---

## Actionable Business Recommendations

1. **First-90-Days Concierge Onboarding:** Implement an automated welcoming journey with progressive milestone rewards to guide new accounts past the high-risk 4-month mark.
2. **Rapid Escalation & Service Recovery Protocol:** Auto-escalate complaints within 2 hours in the CRM, accompanied by instant compensatory cashback or expedited delivery vouchers.
3. **Segment-Specific RFM Re-engagement:** Deploy category-tailored bundles for 'Potential Loyalists' and aggressive win-back promotions for 'At Risk' and 'Hibernating' cohorts.
4. **Regional Logistics Modernization:** Form partnerships with localized 3PL carriers and micro-fulfillment centers in Tier 3 regions to mitigate delivery delays.
5. **Real-Time Automated CRM Scoring:** Embed the Random Forest inference pipeline into weekly CRM pipelines to automatically alert relationship managers when an account's churn probability exceeds 60%.

---

## Limitations

- **Temporal Snapshot:** The dataset represents a cross-sectional snapshot rather than multi-quarter event logs, precluding dynamic time-to-event survival modeling.
- **Financial Granularity:** Profit margins were inferred via cashback amounts rather than direct gross margin records.
- **Qualitative Context:** Customer sentiment was captured through discrete rating scores rather than raw support call/chat text logs.

---

## Future Scope

- **Survival Analysis:** Implementing Cox Proportional Hazards models to estimate the expected customer lifetime and expected time-to-churn.
- **NLP Sentiment Analysis:** Incorporating Natural Language Processing models on customer service transcripts to detect dissatisfaction prior to formal complaints.
- **Cloud MLOps Pipeline:** Deploying the pipeline on Google Cloud Vertex AI / AWS SageMaker with automated concept drift detection and continuous retraining.

---

## Author

- **Student Name:** [Your Name]
- **Academic Program:** IBM SkillsBuild Data Analytics with AI Academic Internship
- **Conducting Organization:** BharatCares in association with AICTE
- **Submission Date:** September 2026
