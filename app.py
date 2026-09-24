import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="E-Commerce Churn Analytics & AI Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 5px solid #2b5c8f;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 13px;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-sub {
        font-size: 12px;
        color: #10b981;
        font-weight: 500;
    }
    .metric-sub-neg {
        font-size: 12px;
        color: #ef4444;
        font-weight: 500;
    }
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #1e3a8a;
        margin-top: 15px;
        margin-bottom: 10px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data & Models with Caching ---
@st.cache_data
def load_data():
    if os.path.exists("data/cleaned_dataset.csv"):
        return pd.read_csv("data/cleaned_dataset.csv")
    elif os.path.exists("data/dataset.csv"):
        df = pd.read_csv("data/dataset.csv")
        df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({'Phone': 'Mobile Phone'})
        df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({'CC': 'Credit Card', 'COD': 'Cash on Delivery'})
        df['PreferedOrderCat'] = df['PreferedOrderCat'].replace({'Mobile': 'Mobile Phone'})
        return df
    else:
        st.error("Dataset not found! Please check data/ directory.")
        st.stop()

@st.cache_resource
def load_model_and_metadata():
    model_path = "models/churn_model.pkl" if os.path.exists("models/churn_model.pkl") else "churn_model.pkl"
    model = joblib.load(model_path)
    lr_path = "models/logistic_regression_model.pkl" if os.path.exists("models/logistic_regression_model.pkl") else "logistic_regression_model.pkl"
    lr_model = joblib.load(lr_path) if os.path.exists(lr_path) else None
    meta_path = "models/model_metadata.json" if os.path.exists("models/model_metadata.json") else "model_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, lr_model, meta

df = load_data()
model, lr_model, metadata = load_model_and_metadata()

# --- Sidebar Navigation ---
st.sidebar.image("https://img.icons8.com/isometric/100/shopping-cart-loaded.png", width=70)
st.sidebar.title("E-Commerce AI Hub")
st.sidebar.caption("IBM SkillsBuild & BharatCares Internship Project")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Executive Overview",
        "👥 Customer Behaviour & RFM",
        "⚠️ Churn Driver Analysis",
        "🤖 AI Model Evaluation",
        "🎯 Live Churn Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Academic Submission Info:**
- **Program:** IBM SkillsBuild Data Analytics with AI
- **Partner:** BharatCares & AICTE
- **Dataset:** Kaggle E-Commerce Churn (5,630 records)
- **Status:** Complete Submission
""")

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if menu == "📊 Executive Overview":
    st.title("🛍️ E-Commerce Customer Churn & Behaviour Analytics")
    st.markdown("Comprehensive executive overview of customer base dynamics, retention health, and revenue drivers.")
    
    total_customers = len(df)
    churned_customers = int(df['Churn'].sum())
    active_customers = total_customers - churned_customers
    churn_rate = (churned_customers / total_customers) * 100
    avg_tenure = df['Tenure'].mean()
    avg_cashback = df['CashbackAmount'].mean()
    complaint_rate = (df['Complain'].mean()) * 100
    
    # KPI Cards Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Customers</div>
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-sub">Active Base Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ef4444;">
            <div class="metric-title">Churn Rate</div>
            <div class="metric-value">{churn_rate:.2f}%</div>
            <div class="metric-sub-neg">{churned_customers:,} Churned Users</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #10b981;">
            <div class="metric-title">Active Customers</div>
            <div class="metric-value">{active_customers:,}</div>
            <div class="metric-sub">{100-churn_rate:.2f}% Retention</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f59e0b;">
            <div class="metric-title">Avg Tenure</div>
            <div class="metric-value">{avg_tenure:.1f} mo</div>
            <div class="metric-sub">Median: {df['Tenure'].median():.0f} mo</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #8b5cf6;">
            <div class="metric-title">Avg Cashback</div>
            <div class="metric-value">${avg_cashback:.2f}</div>
            <div class="metric-sub">Spending Proxy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Customer Base & Churn Distribution</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    
    with c1:
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Retained (Active)', 'Churned'],
            values=[active_customers, churned_customers],
            hole=.5,
            marker_colors=['#2b5c8f', '#ef4444'],
            textinfo='label+percent+value',
            pull=[0, 0.08]
        )])
        fig_donut.update_layout(
            title="Customer Retention vs Churn Proportion",
            title_font_size=15,
            showlegend=True,
            margin=dict(t=40, b=20, l=20, r=20),
            height=340
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        seg_counts = df['CustomerSegment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Customer Count']
        fig_seg = px.bar(
            seg_counts, x='Segment', y='Customer Count',
            color='Segment',
            color_discrete_sequence=px.colors.qualitative.Prism,
            title="Distribution Across RFM Customer Segments"
        )
        fig_seg.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=340, showlegend=False)
        st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("<div class='section-header'>Dataset Summary & Data Quality Inspection</div>", unsafe_allow_html=True)
    d1, d2 = st.columns([1.2, 0.8])
    with d1:
        st.dataframe(df.head(6), use_container_width=True)
    with d2:
        st.markdown(f"""
        - **Total Records:** `5,630` rows
        - **Total Attributes:** `20` features
        - **Target Variable:** `Churn` (Binary: 0 = Retained, 1 = Churned)
        - **Class Imbalance:** 83.16% Retained vs 16.84% Churned
        - **Cleaned Alias Fields:** 
          - *PreferredLoginDevice:* Phone consolidated to Mobile Phone
          - *PreferredPaymentMode:* CC $\\rightarrow$ Credit Card, COD $\\rightarrow$ Cash on Delivery
          - *PreferedOrderCat:* Mobile $\\rightarrow$ Mobile Phone
        """)

# ==========================================
# PAGE 2: CUSTOMER BEHAVIOUR & RFM
# ==========================================
elif menu == "👥 Customer Behaviour & RFM":
    st.title("👥 Customer Behaviour & RFM Segmentation")
    st.markdown("Detailed breakdown of customer purchasing frequency, spending behavior, device usage, and RFM tiers.")

    tab1, tab2, tab3 = st.tabs(["📊 RFM Segmentation", "📱 Category & Channel Analysis", "💳 Demographics & Spending"])

    with tab1:
        st.markdown("### RFM (Recency, Frequency, Monetary) Segmentation")
        st.markdown("""
        Customers are segmented into 5 strategic groups using **Recency** (`DaySinceLastOrder`), **Frequency** (`OrderCount`), and **Monetary** (`CashbackAmount` as proxy for order spending):
        - **Champions:** High Recency, High Frequency, High Spend
        - **Loyal Customers:** Frequent buyers with consistent spend
        - **Potential Loyalists:** Recent buyers with growing engagement
        - **At Risk:** High past spenders who haven't purchased recently
        - **Hibernating:** Infrequent buyers with high dormancy
        """)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            rfm_summary = df.groupby('CustomerSegment').agg(
                Customers=('CustomerID', 'count'),
                Avg_Days_Since_Last_Order=('DaySinceLastOrder', 'mean'),
                Avg_Orders=('OrderCount', 'mean'),
                Avg_Cashback=('CashbackAmount', 'mean'),
                Churn_Rate=('Churn', 'mean')
            ).reset_index()
            rfm_summary['Churn_Rate'] = (rfm_summary['Churn_Rate'] * 100).round(2)
            rfm_summary['Avg_Cashback'] = rfm_summary['Avg_Cashback'].round(2)
            rfm_summary['Avg_Orders'] = rfm_summary['Avg_Orders'].round(1)
            rfm_summary['Avg_Days_Since_Last_Order'] = rfm_summary['Avg_Days_Since_Last_Order'].round(1)
            st.dataframe(rfm_summary, use_container_width=True)
            
        with c2:
            fig_rfm_churn = px.bar(
                rfm_summary.sort_values('Churn_Rate', ascending=False),
                x='CustomerSegment', y='Churn_Rate',
                color='CustomerSegment',
                text='Churn_Rate',
                color_discrete_sequence=px.colors.sequential.Sunset,
                title="Churn Rate (%) Across RFM Segments"
            )
            fig_rfm_churn.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_rfm_churn.update_layout(yaxis_range=[0, 30], showlegend=False, height=350)
            st.plotly_chart(fig_rfm_churn, use_container_width=True)

    with tab2:
        st.markdown("### Category Preferences & Channel Engagement")
        col_cat1, col_cat2 = st.columns(2)
        with col_cat1:
            cat_df = df.groupby('PreferedOrderCat').agg(
                Orders=('OrderCount', 'sum'),
                Avg_Cashback=('CashbackAmount', 'mean')
            ).reset_index()
            fig_cat = px.bar(
                cat_df, x='PreferedOrderCat', y='Orders',
                color='Avg_Cashback',
                color_continuous_scale='Blues',
                title="Total Orders & Avg Cashback by Category"
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_cat2:
            fig_device = px.pie(
                df, names='PreferredLoginDevice',
                title="Preferred Login Device Distribution",
                hole=0.4,
                color_discrete_sequence=['#2b5c8f', '#60a5fa']
            )
            st.plotly_chart(fig_device, use_container_width=True)

    with tab3:
        st.markdown("### Demographics & Payment Behavior")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            pay_df = df.groupby('PreferredPaymentMode')['CustomerID'].count().reset_index()
            fig_pay = px.bar(
                pay_df, x='PreferredPaymentMode', y='CustomerID',
                title="Preferred Payment Modes",
                color='PreferredPaymentMode',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pay.update_layout(showlegend=False)
            st.plotly_chart(fig_pay, use_container_width=True)

        with col_d2:
            fig_box = px.box(
                df, x='MaritalStatus', y='CashbackAmount', color='MaritalStatus',
                title="Cashback Distribution by Marital Status"
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# PAGE 3: CHURN DRIVER ANALYSIS
# ==========================================
elif menu == "⚠️ Churn Driver Analysis":
    st.title("⚠️ Churn Drivers & Behavioral Insights")
    st.markdown("Uncovering the underlying factors that trigger e-commerce customer churn.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 1. The Critical Tenure Factor")
        fig_tenure = px.histogram(
            df, x='Tenure', color='Churn',
            barmode='overlay',
            color_discrete_map={0: '#2b5c8f', 1: '#ef4444'},
            labels={'Churn': 'Status'},
            title="Customer Tenure Distribution (Churned vs Retained)"
        )
        fig_tenure.update_layout(height=350)
        st.plotly_chart(fig_tenure, use_container_width=True)
        st.caption("📌 **Key Finding:** New customers (< 4 months tenure) experience the highest churn rate. Once a customer reaches 12+ months, churn drops dramatically.")

    with c2:
        st.markdown("#### 2. The Customer Complaint Multiplier")
        comp_df = df.groupby('Complain')['Churn'].value_counts(normalize=True).unstack() * 100
        comp_df = comp_df.reset_index()
        comp_df['Complain_Label'] = comp_df['Complain'].map({0: 'No Complaint', 1: 'Raised Complaint'})
        fig_comp = px.bar(
            comp_df, x='Complain_Label', y=[0, 1],
            barmode='stack',
            color_discrete_map={0: '#2b5c8f', 1: '#ef4444'},
            title="Impact of Complaints on Churn Proportion (%)",
            labels={'value': 'Percentage (%)', 'variable': 'Churn Status'}
        )
        fig_comp.update_layout(height=350)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption("📌 **Key Finding:** Customers with unresolved complaints in the last month churn at **31.67%**, nearly **3x higher** than customers without complaints (10.93%).")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 3. City Tier & Delivery Distance")
        city_churn = df.groupby('CityTier')['Churn'].mean().reset_index()
        city_churn['Churn_Rate'] = city_churn['Churn'] * 100
        fig_city = px.bar(
            city_churn, x='CityTier', y='Churn_Rate',
            color='CityTier',
            title="Churn Rate (%) Across City Tiers",
            text='Churn_Rate'
        )
        fig_city.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_city.update_layout(height=350, yaxis_range=[0, 25], showlegend=False)
        st.plotly_chart(fig_city, use_container_width=True)
        st.caption("📌 **Key Finding:** Tier 3 cities experience higher churn (21.37%) likely driven by longer delivery times and higher logistics friction.")

    with c4:
        st.markdown("#### 4. Warehouse-to-Home Distance")
        fig_dist = px.box(
            df, x='Churn', y='WarehouseToHome', color='Churn',
            color_discrete_map={0: '#2b5c8f', 1: '#ef4444'},
            title="Warehouse-to-Home Distance by Churn Status"
        )
        fig_dist.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption("📌 **Key Finding:** Churned customers reside further from fulfillment hubs on average, correlating with potential fulfillment delays.")

# ==========================================
# PAGE 4: AI MODEL EVALUATION
# ==========================================
elif menu == "🤖 AI Model Evaluation":
    st.title("🤖 Machine Learning Model Benchmarking")
    st.markdown("Rigorous evaluation of baseline **Logistic Regression** and non-linear **Random Forest Classifier**.")

    res = metadata['results']
    lr_res = res['Logistic Regression']
    rf_res = res['Random Forest Classifier']

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### 📊 Performance Comparison Table")
        comparison_df = pd.DataFrame({
            "Evaluation Metric": ["Accuracy", "Precision (Class 1)", "Recall (Class 1)", "F1-Score (Class 1)", "ROC-AUC"],
            "Logistic Regression (Balanced)": [
                f"{lr_res['accuracy']*100:.2f}%",
                f"{lr_res['precision']*100:.2f}%",
                f"{lr_res['recall']*100:.2f}%",
                f"{lr_res['f1_score']:.4f}",
                f"{lr_res['roc_auc']:.4f}"
            ],
            "Random Forest Classifier (Balanced)": [
                f"{rf_res['accuracy']*100:.2f}%",
                f"{rf_res['precision']*100:.2f}%",
                f"{rf_res['recall']*100:.2f}%",
                f"{rf_res['f1_score']:.4f}",
                f"{rf_res['roc_auc']:.4f}"
            ]
        })
        st.table(comparison_df)

        st.markdown("""
        **Metric Interpretation in Churn Context:**
        - **Recall (Sensitivity):** Crucial metric representing the percentage of actual churners identified. Missing a churner (False Negative) results in lost customer lifetime value.
        - **Precision:** Percentage of predicted churners who truly churned. High precision prevents wasteful retention discounts on active customers.
        - **ROC-AUC:** Measures discrimination capability across all classification thresholds. Random Forest achieves an exceptional **0.9971**.
        """)

    with col_m2:
        st.markdown("### 🎯 Confusion Matrices")
        cm_rf = np.array(rf_res['confusion_matrix'])
        fig_cm = px.imshow(
            cm_rf,
            labels=dict(x="Predicted Status", y="Actual Status", color="Count"),
            x=['Retained (0)', 'Churned (1)'],
            y=['Retained (0)', 'Churned (1)'],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Random Forest Classifier - Confusion Matrix"
        )
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(f"True Negatives: {cm_rf[0,0]} | False Positives: {cm_rf[0,1]} | False Negatives: {cm_rf[1,0]} | True Positives: {cm_rf[1,1]}")

    st.markdown("<div class='section-header'>Top 15 Feature Importances (Random Forest)</div>", unsafe_allow_html=True)
    feat_imp = pd.DataFrame(res['Top15_FeatureImportance_RF'])
    fig_feat = px.bar(
        feat_imp.sort_values('Importance', ascending=True),
        x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis',
        title="Predictive Weight of Attributes in Churn Determination"
    )
    fig_feat.update_layout(height=450, margin=dict(l=150, r=20, t=40, b=20))
    st.plotly_chart(fig_feat, use_container_width=True)

# ==========================================
# PAGE 5: LIVE CHURN PREDICTOR
# ==========================================
elif menu == "🎯 Live Churn Predictor":
    st.title("🎯 Customer Churn Risk Simulator")
    st.markdown("Input customer attributes below to generate real-time AI churn risk predictions and retention recommendations.")

    with st.form("churn_prediction_form"):
        st.markdown("#### 1. Customer Relationship & Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.number_input("Tenure (Months with Organization)", min_value=0, max_value=65, value=3, help="Duration since customer registered")
            gender = st.selectbox("Gender", ["Female", "Male"])
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
        with col2:
            warehouse_dist = st.slider("Warehouse-to-Home Distance (km)", min_value=1, max_value=130, value=15)
            addresses = st.number_input("Number of Registered Addresses", min_value=1, max_value=25, value=3)
            login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer"])
            devices_reg = st.number_input("Number of Devices Registered", min_value=1, max_value=6, value=4)
        with col3:
            satisfaction = st.slider("Satisfaction Score (1 = Lowest, 5 = Highest)", min_value=1, max_value=5, value=2)
            complaint = st.selectbox("Complaint Raised in Last Month?", ["No", "Yes"], index=1)
            hours_app = st.slider("Hours Spent on App / Website Weekly", min_value=0.0, max_value=5.0, value=2.5, step=0.5)

        st.markdown("#### 2. Purchasing & Transaction Dynamics")
        col4, col5, col6 = st.columns(3)
        with col4:
            order_cat = st.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"], index=1)
            pay_mode = st.selectbox("Preferred Payment Mode", ["Debit Card", "UPI", "Credit Card", "Cash on Delivery", "E wallet"], index=3)
        with col5:
            order_count = st.number_input("Order Count (Last Month)", min_value=1, max_value=20, value=2)
            day_since = st.slider("Days Since Last Order", min_value=0, max_value=50, value=3)
        with col6:
            cashback = st.number_input("Average Cashback Amount ($)", min_value=0.0, max_value=350.0, value=145.0)
            order_hike = st.slider("Order Amount Hike from Last Year (%)", min_value=10, max_value=30, value=13)
            coupons = st.number_input("Coupons Used (Last Month)", min_value=0, max_value=20, value=1)

        submit = st.form_submit_button("🔍 Run Churn Prediction Model", use_container_width=True)

    if submit:
        # Construct input DataFrame matching original schema
        input_dict = {
            'Tenure': [tenure],
            'CityTier': [city_tier],
            'WarehouseToHome': [warehouse_dist],
            'HourSpendOnApp': [hours_app],
            'NumberOfDeviceRegistered': [devices_reg],
            'SatisfactionScore': [satisfaction],
            'NumberOfAddress': [addresses],
            'Complain': [1 if complaint == "Yes" else 0],
            'OrderAmountHikeFromlastYear': [order_hike],
            'CouponUsed': [coupons],
            'OrderCount': [order_count],
            'DaySinceLastOrder': [day_since],
            'CashbackAmount': [cashback],
            'PreferredLoginDevice': [login_device],
            'PreferredPaymentMode': [pay_mode],
            'Gender': [gender],
            'PreferedOrderCat': [order_cat],
            'MaritalStatus': [marital]
        }
        input_df = pd.DataFrame(input_dict)
        
        # Inference
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        churn_prob = proba[1] * 100
        active_prob = proba[0] * 100

        st.markdown("---")
        st.markdown("<div class='section-header'>Prediction Output & Risk Assessment</div>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1, 1.2])
        
        with res_col1:
            if pred == 1:
                st.error("🚨 **PREDICTED STATUS: LIKELY TO CHURN**")
                st.markdown(f"""
                - **Churn Probability:** `{churn_prob:.1f}%`
                - **Retention Probability:** `{active_prob:.1f}%`
                - **Risk Category:** **HIGH RISK**
                """)
            else:
                st.success("✅ **PREDICTED STATUS: LIKELY ACTIVE (RETAINED)**")
                st.markdown(f"""
                - **Retention Probability:** `{active_prob:.1f}%`
                - **Churn Probability:** `{churn_prob:.1f}%`
                - **Risk Category:** **LOW RISK / HEALTHY**
                """)

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#ef4444" if churn_prob > 50 else "#10b981"},
                    'steps': [
                        {'range': [0, 30], 'color': "#dcfce7"},
                        {'range': [30, 60], 'color': "#fef9c3"},
                        {'range': [60, 100], 'color': "#fee2e2"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("#### 💡 Actionable Retention Strategy")
            if pred == 1:
                st.warning("""
                **Immediate High-Priority Interventions:**
                1. **Customer Support Escalation:** If complaints exist, trigger an automated high-priority ticket for dedicated concierge resolution within 2 hours.
                2. **Targeted Loyalty Incentive:** Deploy a personalized cashback bonus or free expedited shipping voucher for their preferred category.
                3. **Onboarding Re-engagement:** If tenure < 6 months, route into a welcome nurturing drip campaign highlighting app benefits.
                """)
            else:
                st.info("""
                **Customer Nurturing & Growth Plan:**
                1. **Cross-Selling:** Introduce personalized product bundles in adjacent categories.
                2. **Loyalty Advancement:** Invite customer into VIP tier with early-access product drops.
                3. **Review Request:** Solicit product ratings and reviews to amplify social proof.
                """)
            
            st.caption("⚠️ *Note: This output represents an AI model estimate based on historical behavioral patterns, not an absolute guarantee.*")
