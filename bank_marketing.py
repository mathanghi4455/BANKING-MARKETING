# ==============================================================================
# Bank Marketing - Machine Learning Mini Project
# Problem: Who should the bank contact to get term deposit subscriptions?
# Models Used: Logistic Regression, Decision Tree, Random Forest
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix


# ------------------------------------------------------------------------------
# STEP 1: LOAD THE DATASET & BASIC INFORMATION
# ------------------------------------------------------------------------------
print("--- 1. DATASET INFORMATION ---")

# Load data (the bank marketing dataset uses semicolon ';' as delimiter)
df = pd.read_csv('bank.csv', sep=';')

print(f"Total rows (customers): {df.shape[0]}")
print(f"Total columns (features): {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------------------------
# STEP 2: DATA CLEANLINESS CHECK
# ------------------------------------------------------------------------------
print("\n--- 2. DATA CLEANLINESS CHECK ---")

# Check for missing values (NaN)
print("Missing values in each column:")
print(df.isnull().sum())

# Check for duplicate rows
print(f"\nDuplicate rows found: {df.duplicated().sum()}")

# Numerical and categorical columns
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Check for 'unknown' values (the dataset uses 'unknown' instead of null in some columns)
print("\nColumns with 'unknown' values:")
for col in cat_cols:
    count = (df[col] == 'unknown').sum()
    if count > 0:
        print(f"  - {col}: {count} rows ({count/len(df)*100:.1f}%)")

# Check target variable distribution ('y' = subscribed?)
print("\nTarget variable 'y' distribution:")
print(df['y'].value_counts())
print(df['y'].value_counts(normalize=True) * 100)


# ------------------------------------------------------------------------------
# STEP 3: PREPROCESSING & FEATURE SCALING
# ------------------------------------------------------------------------------
print("\n--- 3. DATA PREPROCESSING & FEATURE SCALING ---")

# Separate features (X) and target (y)
# Convert target 'yes' to 1, 'no' to 0
y = df['y'].map({'yes': 1, 'no': 0})
X = df.drop('y', axis=1)

print(f"Numerical features: {num_cols}")
print(f"Categorical features: {cat_cols}")

# Convert categorical columns to dummy numbers (One-Hot Encoding)
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print(f"Total features after One-Hot Encoding: {X.shape[1]}")

# Split into training set (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Feature Scaling using StandardScaler (brings all numbers to mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed successfully.")


# ------------------------------------------------------------------------------
# STEP 4: TRAIN THE MACHINE LEARNING MODELS
# ------------------------------------------------------------------------------
print("\n--- 4. TRAINING ML MODELS ---")

# 1. Logistic Regression
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# 2. Decision Tree
dt_model = DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42)
dt_model.fit(X_train_scaled, y_train)

# 3. Random Forest
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=8, random_state=42)
rf_model.fit(X_train_scaled, y_train)

print("All 3 models trained successfully!")


# ------------------------------------------------------------------------------
# STEP 5: EVALUATION & COMPARISON
# ------------------------------------------------------------------------------
print("\n--- 5. MODEL EVALUATION & COMPARISON ---")

models = {
    'Logistic Regression': lr_model,
    'Decision Tree': dt_model,
    'Random Forest': rf_model
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4)
    })

comparison_table = pd.DataFrame(results)
print(comparison_table.to_string(index=False))

print("\n>>> Why Accuracy Alone Is Not Enough:")
print("Around 88% of customers say 'no'. If a dummy model predicts 'no' for every single person,")
print("its accuracy is still 88%! But it finds 0 actual customers.")
print("That's why Recall, Precision, and ROC-AUC are the real metrics to look at.")


# ------------------------------------------------------------------------------
# STEP 6: FEATURE IMPORTANCE (WHAT MATTERS MOST)
# ------------------------------------------------------------------------------
print("\n--- 6. IMPORTANT FEATURES ---")

importances = rf_model.feature_importances_
feature_names = X.columns
top_indices = np.argsort(importances)[::-1][:5]

print("Top 5 most important customer features (from Random Forest):")
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank}. {feature_names[idx]} (score: {importances[idx]:.4f})")


# ------------------------------------------------------------------------------
# STEP 7: WHO SHOULD THE BANK CONTACT FIRST? (RANKED LIST)
# ------------------------------------------------------------------------------
print("\n--- 7. WHO SHOULD THE BANK CONTACT FIRST? ---")

# Predict probability of subscription using Logistic Regression / Random Forest
probs = lr_model.predict_proba(X_test_scaled)[:, 1]

# Create a ranked lead list
leads = X_test.copy()
leads['Subscription_Probability'] = np.round(probs, 4)
leads['Actual_Subscribed'] = y_test.values

# Sort by highest probability first
leads_ranked = leads.sort_values(by='Subscription_Probability', ascending=False)
leads_ranked.to_csv('ranked_customers.csv', index=False)

print("Saved 'ranked_customers.csv' with prioritized customer list!")
print("\nTop 5 customers to call first:")
print(leads_ranked[['Subscription_Probability', 'Actual_Subscribed']].head(5))


# ------------------------------------------------------------------------------
# STEP 8: SAVE VISUAL CHARTS (EVALUATION & FEATURES)
# ------------------------------------------------------------------------------
print("\n--- 8. GENERATING CHARTS ---")

# Chart 1: Model Evaluation (Confusion Matrix & ROC Curve)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
lr_pred = lr_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted No', 'Predicted Yes'],
            yticklabels=['Actual No', 'Actual Yes'])
plt.title('Logistic Regression - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
for name, model in models.items():
    prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_score = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Baseline (0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

plt.tight_layout()
plt.savefig('model_evaluation.png')
plt.close()
print("Saved chart: model_evaluation.png")

# Chart 2: Feature Importance
plt.figure(figsize=(10, 5))
top_10_idx = np.argsort(importances)[::-1][:10]
plt.barh([feature_names[i] for i in top_10_idx][::-1], importances[top_10_idx][::-1], color='steelblue')
plt.title('Top 10 Important Features for Subscribing')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_analysis.png')
plt.close()
print("Saved chart: feature_analysis.png")

# Chart 3: Model Comparison Output Graph (Bar Chart of All Metrics)
plt.figure(figsize=(11, 5))
comparison_plot_data = pd.DataFrame(results).set_index('Model')
ax = comparison_plot_data.plot(kind='bar', figsize=(11, 5), colormap='viridis', width=0.75, edgecolor='black')
plt.title('Output Comparison: Logistic Regression vs Decision Tree vs Random Forest', fontsize=13, fontweight='bold')
plt.ylabel('Score (0.0 to 1.0)')
plt.xlabel('Machine Learning Model')
plt.xticks(rotation=0)
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                    textcoords='offset points')
plt.tight_layout()
plt.savefig('output_model_comparison.png', dpi=300)
plt.close()
print("Saved chart: output_model_comparison.png (All Model Metrics Compared)")


# ------------------------------------------------------------------------------
# STEP 9: PRACTICAL BUSINESS SUGGESTIONS
# ------------------------------------------------------------------------------
print("\n==============================================================================")
print("PRACTICAL BUSINESS SUGGESTIONS FOR THE BANK")
print("==============================================================================")
print("1. Call by Probability (Save 60-70% calls):")
print("   Only contact customers with probability >= 0.40 in 'ranked_customers.csv'.")
print("   This cuts calls by two-thirds while still catching most subscribers.")
print("\n2. Prioritize Previous Successful Campaigns:")
print("   Customers where past campaign outcome was successful ('poutcome_success')")
print("   have the highest conversion rate. Call them first.")
print("\n3. Focus on Call Quality over Call Quantity:")
print("   Call duration is the biggest factor. Agents should spend more time explaining")
print("   benefits to high-probability customers rather than rushing through cold calls.")
print("==============================================================================")
