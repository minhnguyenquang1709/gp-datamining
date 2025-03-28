import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay,
                            precision_recall_fscore_support, precision_score, recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn import tree

if not os.path.exists('figures'):
    os.makedirs('figures')
    print("Created figures directory")

datadir_prep = 'datasets'
combined_df = pd.read_csv(os.path.join(datadir_prep, 'combined_df.csv'))

print("Columns in combined_df:", combined_df.columns.tolist())

print("\nNumber of NaN values in Cluster column:", combined_df['Cluster'].isna().sum())

combined_df = combined_df.dropna(subset=['Cluster'])

feature_columns = ['Gender', 'State', 'ReferralSource', 'Frequency', 'Recency', 'Duration', 'Customer_Profile']

existing_columns = [col for col in feature_columns if col in combined_df.columns]
missing_columns = [col for col in feature_columns if col not in combined_df.columns]
if missing_columns:
    print(f"Warning: The following requested columns are not in the dataset: {missing_columns}")

X = combined_df[existing_columns].copy()
y = combined_df['Cluster']

print("\nFeatures with NaN values:")
for col in X.columns:
    nan_count = X[col].isna().sum()
    if nan_count > 0:
        print(f"{col}: {nan_count} NaN values")

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
if len(numerical_features) > 0:
    imputer = SimpleImputer(strategy='median')
    X[numerical_features] = imputer.fit_transform(X[numerical_features])

print("\nFeatures used for the model:", X.columns.tolist())

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical features that need encoding:", categorical_features)

encoders = {}
for feature in categorical_features:
    X[feature] = X[feature].fillna('Unknown')
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature].astype(str))
    encoders[feature] = le

print("\nVerifying no NaN values remain in features:", X.isna().sum().sum())
print("Verifying no NaN values remain in target:", y.isna().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_classifier = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("\nBest Parameters:", best_params)

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")

precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

metrics_summary = pd.DataFrame({
    'Metric': ['Accuracy', 
              'Precision (macro)', 'Recall (macro)', 'F1-score (macro)',
              'Precision (weighted)', 'Recall (weighted)', 'F1-score (weighted)'],
    'Value': [accuracy, 
             precision_macro, recall_macro, f1_macro,
             precision_weighted, recall_weighted, f1_weighted]
})
print("\nMetrics Summary:")
print(metrics_summary.to_string(index=False))

labels = sorted(y.unique().astype(int))
cluster_names = [f'Cluster {i}' for i in labels]
y_test_bin = label_binarize(y_test, classes=labels)
n_classes = len(labels)

plt.figure(figsize=(10, 8))

auc_values = []
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    auc_values.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, label=f'{cluster_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('figures/roc_curves.png')
plt.show()

print("\nAUC-ROC Values:")
for i, label in enumerate(cluster_names):
    print(f"{label}: {auc_values[i]:.4f}")

print(f"Average AUC-ROC: {np.mean(auc_values):.4f}")

labels = sorted(y.unique().astype(int))
cluster_names = [f'Cluster {i}' for i in labels]

fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cluster_names)
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title('Confusion Matrix')
plt.savefig('figures/confusion_matrix.png')
plt.show()

plt.figure(figsize=(12, 10))
conf_matrix_norm = confusion_matrix(y_test, y_pred, normalize='true')
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=cluster_names,
            yticklabels=cluster_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix (Row Percentages)')
plt.savefig('figures/confusion_matrix_normalized.png')
plt.show()

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Decision Tree Model')
plt.tight_layout()
plt.savefig('figures/feature_importance.png')
plt.show()

plt.figure(figsize=(20, 15))
tree.plot_tree(best_model, max_depth=3, feature_names=X.columns, 
               class_names=[f'Cluster {i}' for i in sorted(y.unique().astype(int))],
               filled=True)
plt.title('Decision Tree (Limited to Depth 3 for Visualization)')
plt.savefig('figures/decision_tree.png')
plt.show()

print("\nModel training and evaluation complete. Visualizations displayed and saved to figures directory.") 