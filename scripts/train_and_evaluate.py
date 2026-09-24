import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# 1. Load Data
df = pd.read_csv('data/dataset.csv')
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Data Cleaning & Standardization
df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({'Phone': 'Mobile Phone'})
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({'CC': 'Credit Card', 'COD': 'Cash on Delivery'})
df['PreferedOrderCat'] = df['PreferedOrderCat'].replace({'Mobile': 'Mobile Phone'})

# Median imputation for derived RFM columns
day_since_median = df['DaySinceLastOrder'].median()
order_count_median = df['OrderCount'].median()
df['DaySinceLastOrder_Clean'] = df['DaySinceLastOrder'].fillna(day_since_median)
df['OrderCount_Clean'] = df['OrderCount'].fillna(order_count_median)

# 3. RFM Analysis & Customer Segmentation
# R_score: fewer days since last order = more recent = higher score (5 to 1)
df['R_score'] = pd.qcut(df['DaySinceLastOrder_Clean'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
# F_score: more orders = higher score (1 to 5)
df['F_score'] = pd.qcut(df['OrderCount_Clean'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
# M_score: higher cashback = proxy for spend (1 to 5)
df['M_score'] = pd.qcut(df['CashbackAmount'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
df['RFM_Score'] = df['R_score'] + df['F_score'] + df['M_score']

def assign_segment(row):
    r, f, m = row['R_score'], row['F_score'], row['M_score']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 3 and f >= 2:
        return 'Potential Loyalists'
    elif r <= 2 and (f >= 3 or m >= 3):
        return 'At Risk'
    else:
        return 'Hibernating'

df['CustomerSegment'] = df.apply(assign_segment, axis=1)

# Save cleaned dataset for analytics & app
df.to_csv('data/cleaned_dataset.csv', index=False)
print("Saved data/cleaned_dataset.csv")

# 4. Generate Visualizations for Report and App
fig_dir = 'reports/figures'
os.makedirs(fig_dir, exist_ok=True)

# Fig 1: Churn Distribution
plt.figure(figsize=(7, 4.5))
ax = sns.countplot(data=df, x='Churn', hue='Churn', palette=['#2b5c8f', '#d9534f'], legend=False)
plt.title('Target Variable Distribution (Churn vs Retained)', fontsize=13, weight='bold', pad=12)
plt.xlabel('Customer Status (0 = Retained, 1 = Churned)', fontsize=11)
plt.ylabel('Number of Customers', fontsize=11)
for p in ax.patches:
    h = p.get_height()
    pct = (h / len(df)) * 100
    ax.annotate(f"{h:,}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., h / 2),
                ha='center', va='center', fontsize=11, color='white', weight='bold')
plt.tight_layout()
plt.savefig(f"{fig_dir}/fig1_churn_distribution.png")
plt.close()

# Fig 2: Tenure vs Churn
plt.figure(figsize=(8, 4.5))
sns.kdeplot(data=df[df['Churn']==0]['Tenure'], fill=True, color='#2b5c8f', label='Retained (0)', alpha=0.5)
sns.kdeplot(data=df[df['Churn']==1]['Tenure'], fill=True, color='#d9534f', label='Churned (1)', alpha=0.5)
plt.title('Customer Tenure Distribution by Churn Status', fontsize=13, weight='bold', pad=12)
plt.xlabel('Tenure (Months)', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(f"{fig_dir}/fig2_tenure_vs_churn.png")
plt.close()

# Fig 3: Complain vs Churn
plt.figure(figsize=(7, 4.5))
comp_churn = df.groupby('Complain')['Churn'].value_counts(normalize=True).unstack() * 100
comp_churn.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], figsize=(7, 4.5), edgecolor='black', alpha=0.85)
plt.title('Churn Proportion by Customer Complaint Status', fontsize=13, weight='bold', pad=12)
plt.xlabel('Complaint Raised in Last Month (0 = No, 1 = Yes)', fontsize=11)
plt.ylabel('Percentage (%)', fontsize=11)
plt.xticks(rotation=0)
plt.legend(['Retained', 'Churned'], title='Status', frameon=True)
plt.tight_layout()
plt.savefig(f"{fig_dir}/fig3_complain_vs_churn.png")
plt.close()

# Fig 4: Preferred Order Category & Churn
plt.figure(figsize=(9, 4.5))
cat_churn = df.groupby('PreferedOrderCat')['Churn'].agg(['count', 'mean']).reset_index()
cat_churn['mean'] = cat_churn['mean'] * 100
ax = sns.barplot(data=cat_churn, x='PreferedOrderCat', y='mean', palette='Blues_r', edgecolor='black')
plt.title('Churn Rate across Preferred Order Categories', fontsize=13, weight='bold', pad=12)
plt.xlabel('Order Category', fontsize=11)
plt.ylabel('Churn Rate (%)', fontsize=11)
plt.xticks(rotation=15)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.8),
                ha='center', va='bottom', fontsize=10, weight='bold')
plt.ylim(0, 35)
plt.tight_layout()
plt.savefig(f"{fig_dir}/fig4_category_performance.png")
plt.close()

# Fig 5: RFM Segments vs Churn
plt.figure(figsize=(9, 4.8))
rfm_stats = df.groupby('CustomerSegment')['Churn'].agg(['count', 'mean']).reset_index()
rfm_stats['mean'] = rfm_stats['mean'] * 100
rfm_stats = rfm_stats.sort_values('mean', ascending=False)
ax = sns.barplot(data=rfm_stats, x='CustomerSegment', y='mean', palette='flare', edgecolor='black')
plt.title('Churn Rate by RFM Customer Segment', fontsize=13, weight='bold', pad=12)
plt.xlabel('Customer Segment', fontsize=11)
plt.ylabel('Churn Rate (%)', fontsize=11)
plt.xticks(rotation=15)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.7),
                ha='center', va='bottom', fontsize=10, weight='bold')
plt.ylim(0, 30)
plt.tight_layout()
plt.savefig(f"{fig_dir}/fig5_rfm_segments.png")
plt.close()

# Fig 6: Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['CustomerID', 'R_score', 'F_score', 'M_score', 'RFM_Score', 'DaySinceLastOrder_Clean', 'OrderCount_Clean'], errors='ignore')
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.5, vmax=0.5, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix of Numerical Features with Churn', fontsize=13, weight='bold', pad=12)
plt.tight_layout()
plt.savefig(f"{fig_dir}/fig6_correlation_heatmap.png")
plt.close()

# 5. Machine Learning Pipeline Preparation
# Features to exclude from direct modeling to prevent data leakage / redundancy: CustomerID, RFM derived scores, clean duplicates
exclude_cols = ['CustomerID', 'Churn', 'R_score', 'F_score', 'M_score', 'RFM_Score', 'DaySinceLastOrder_Clean', 'OrderCount_Clean', 'CustomerSegment']
X = df.drop(columns=exclude_cols)
y = df['Churn']

num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'str']).columns.tolist()

print("Numerical features:", num_features)
print("Categorical features:", cat_features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# 6. Train Models
# Model 1: Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lr_model)
])
lr_pipe.fit(X_train, y_train)

# Model 2: Random Forest Classifier (Balanced)
rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, class_weight='balanced')
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])
rf_pipe.fit(X_train, y_train)

# 7. Model Evaluation
models = {'Logistic Regression': lr_pipe, 'Random Forest Classifier': rf_pipe}
results = {}

for name, pipe in models.items():
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    results[name] = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'confusion_matrix': cm,
        'classification_report': cr
    }
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

# Fig 7: Confusion Matrices Comparison
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for idx, (name, pipe) in enumerate(models.items()):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[idx].set_title(f"{name}\nConfusion Matrix", fontsize=12, weight='bold')
    axes[idx].set_xlabel('Predicted Label (0 = Retained, 1 = Churned)', fontsize=10)
    axes[idx].set_ylabel('Actual Label (0 = Retained, 1 = Churned)', fontsize=10)
    axes[idx].set_xticklabels(['Retained', 'Churned'])
    axes[idx].set_yticklabels(['Retained', 'Churned'])
plt.tight_layout()
plt.savefig(f"{fig_dir}/fig7_confusion_matrices.png")
plt.close()

# Fig 8: ROC Curves Comparison
plt.figure(figsize=(7.5, 5))
for name, pipe in models.items():
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {auc_val:.3f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.500)')
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, weight='bold', pad=12)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Recall)', fontsize=11)
plt.legend(loc='lower right', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig(f"{fig_dir}/fig8_roc_curves.png")
plt.close()

# Fig 9: Feature Importance from Random Forest
fitted_cat_enc = rf_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = fitted_cat_enc.get_feature_names_out(cat_features).tolist()
all_feat_names = num_features + cat_feature_names

rf_classifier = rf_pipe.named_steps['classifier']
importances = rf_classifier.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': all_feat_names, 'Importance': importances}).sort_values('Importance', ascending=True)

plt.figure(figsize=(9, 6.5))
top_15 = feat_imp_df.tail(15)
plt.barh(top_15['Feature'], top_15['Importance'], color='#2b5c8f', edgecolor='black', alpha=0.85)
plt.title('Top 15 Feature Importances (Random Forest Classifier)', fontsize=13, weight='bold', pad=12)
plt.xlabel('Relative Feature Importance (Gini Impurity Decrease)', fontsize=11)
plt.tight_layout()
plt.savefig(f"{fig_dir}/fig9_feature_importances.png")
plt.close()

# Logistic Regression Feature Coefficients
lr_classifier = lr_pipe.named_steps['classifier']
lr_coefs = lr_classifier.coef_[0]
lr_coef_df = pd.DataFrame({'Feature': all_feat_names, 'Coefficient': lr_coefs, 'OddsRatio': np.exp(lr_coefs)}).sort_values('Coefficient', ascending=False)
results['TopPositiveRiskFactors_LR'] = lr_coef_df.head(5).to_dict(orient='records')
results['TopProtectiveFactors_LR'] = lr_coef_df.tail(5).to_dict(orient='records')
results['Top15_FeatureImportance_RF'] = feat_imp_df.tail(15).iloc[::-1].to_dict(orient='records')

# 8. Save Pipeline Model and Metadata
joblib.dump(rf_pipe, 'models/churn_model.pkl')
joblib.dump(lr_pipe, 'models/logistic_regression_model.pkl')
print("Saved models/churn_model.pkl (Random Forest Pipeline) and logistic_regression_model.pkl")

# Save feature metadata for Streamlit input validation
metadata = {
    'results': results,
    'num_features': num_features,
    'cat_features': cat_features,
    'cat_categories': {col: df[col].unique().tolist() for col in cat_features},
    'num_summary': {col: {'min': float(df[col].min()), 'max': float(df[col].max()), 'median': float(df[col].median()), 'mean': float(df[col].mean())} for col in num_features}
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("Saved models/model_metadata.json")

print("\nModel training, figure generation, and metadata export completed successfully!")
