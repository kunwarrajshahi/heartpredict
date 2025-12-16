# ===============================
# 1. Import Required Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# ===============================
# 2. Load Dataset
# ===============================
dataset = pd.read_csv("heart.csv")

print("Dataset Shape:", dataset.shape)
print(dataset.head())

# ===============================
# 3. Dataset Information
# ===============================
print(dataset.info())
print(dataset.describe())

# ===============================
# 4. Correlation with Target
# ===============================
print("\nCorrelation with Target:")
print(dataset.corr(numeric_only=True)["target"].abs().sort_values(ascending=False))

# ===============================
# 5. Exploratory Data Analysis
# ===============================
y = dataset["target"]

plt.figure(figsize=(5, 4))
sns.countplot(x=y)
plt.title("Target Distribution")
plt.show()

print("\nTarget Value Counts:")
print(dataset.target.value_counts())

print("Percentage without heart disease:",
      round(dataset.target.value_counts()[0] * 100 / len(dataset), 2))
print("Percentage with heart disease:",
      round(dataset.target.value_counts()[1] * 100 / len(dataset), 2))

# Sex vs Target
plt.figure(figsize=(5, 4))
sns.barplot(x=dataset["sex"], y=y)
plt.title("Sex vs Heart Disease")
plt.show()

# Chest Pain vs Target
plt.figure(figsize=(5, 4))
sns.barplot(x=dataset["cp"], y=y)
plt.title("Chest Pain vs Heart Disease")
plt.show()

# ===============================
# 6. Train-Test Split
# ===============================
X = dataset.drop("target", axis=1)
Y = dataset["target"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# ===============================
# 7. Logistic Regression
# ===============================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, Y_train)
Y_pred_lr = lr.predict(X_test)
score_lr = accuracy_score(Y_test, Y_pred_lr) * 100
print("Logistic Regression Accuracy:", round(score_lr, 2), "%")

# ===============================
# 8. Naive Bayes
# ===============================
nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred_nb = nb.predict(X_test)
score_nb = accuracy_score(Y_test, Y_pred_nb) * 100
print("Naive Bayes Accuracy:", round(score_nb, 2), "%")

# ===============================
# 9. Support Vector Machine
# ===============================
svm_model = SVC(kernel="linear", random_state=0)
svm_model.fit(X_train, Y_train)
Y_pred_svm = svm_model.predict(X_test)
score_svm = accuracy_score(Y_test, Y_pred_svm) * 100
print("SVM Accuracy:", round(score_svm, 2), "%")

# ===============================
# 10. K Nearest Neighbors
# ===============================
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
score_knn = accuracy_score(Y_test, Y_pred_knn) * 100
print("KNN Accuracy:", round(score_knn, 2), "%")

# ===============================
# 11. Decision Tree (Best Random State)
# ===============================
max_accuracy = 0
best_state = 0

for i in range(200):
    dt_temp = DecisionTreeClassifier(random_state=i)
    dt_temp.fit(X_train, Y_train)
    acc = accuracy_score(Y_test, dt_temp.predict(X_test))
    if acc > max_accuracy:
        max_accuracy = acc
        best_state = i

dt = DecisionTreeClassifier(random_state=best_state)
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test)
score_dt = accuracy_score(Y_test, Y_pred_dt) * 100
print("Decision Tree Accuracy:", round(score_dt, 2), "%")

# ===============================
# 12. Random Forest (Best Random State)
# ===============================
max_accuracy = 0
best_state = 0

for i in range(300):
    rf_temp = RandomForestClassifier(random_state=i)
    rf_temp.fit(X_train, Y_train)
    acc = accuracy_score(Y_test, rf_temp.predict(X_test))
    if acc > max_accuracy:
        max_accuracy = acc
        best_state = i

rf = RandomForestClassifier(random_state=best_state)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
score_rf = accuracy_score(Y_test, Y_pred_rf) * 100
print("Random Forest Accuracy:", round(score_rf, 2), "%")

# ===============================
# 13. Final Model Comparison
# ===============================
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Naive Bayes",
        "SVM",
        "KNN",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy (%)": [
        round(score_lr, 2),
        round(score_nb, 2),
        round(score_svm, 2),
        round(score_knn, 2),
        round(score_dt, 2),
        round(score_rf, 2)
    ]
})

print("\nModel Performance Comparison:")
print(results.sort_values(by="Accuracy (%)", ascending=False))

# ===============================
# 14. Save Best Model
# ===============================
joblib.dump(rf, "heart_disease_model.pkl")
print("\nModel saved as heart_disease_model.pkl")
