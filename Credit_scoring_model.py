import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc

file_path = "Credit Score Classification Dataset.csv"
credit_df = pd.read_csv(file_path)
credit_df.info()
print("Target values:\n", credit_df['Credit Score'].value_counts())

for col in ['Gender', 'Education', 'Marital Status', 'Home Ownership']:
    print(f"{col}: {credit_df[col].unique()}")

df = credit_df.copy()
le = LabelEncoder()
df['Credit Score'] = le.fit_transform(df['Credit Score'])
df = pd.get_dummies(df, columns=['Gender', 'Education', 'Marital Status', 'Home Ownership'], drop_first=True)

print("Final shape after encoding:", df.shape)

X = df.drop('Credit Score', axis=1)
y = df['Credit Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("Decision Tree")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Classification Report:\n", classification_report(y_test, dt_pred))

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Random Forest")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = model.predict_proba(X_test)

roc_auc = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')
print("ROC-AUC Score (Logistic Regression):", round(roc_auc, 4))

fpr, tpr, roc_auc_dict = dict(), dict(), dict()
n_classes = y_test_bin.shape[1]
labels = ['High', 'Low', 'Medium']
colors = ['blue', 'orange', 'green']

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc_dict[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"ROC curve for class {labels[i]} (area = {roc_auc_dict[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — Logistic Regression (Credit Score)')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()
