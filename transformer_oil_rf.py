import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Realistic Dataset Creation
# (Overlapping + Noisy Data)
# -------------------------------
np.random.seed(42)
data = []

# Healthy oil
for _ in range(100):
    data.append([
        np.random.uniform(55, 80),
        np.random.uniform(5, 15),
        np.random.uniform(0.01, 0.05),
        np.random.uniform(35, 50),
        "Healthy"
    ])

# Moderate oil
for _ in range(100):
    data.append([
        np.random.uniform(35, 65),
        np.random.uniform(10, 30),
        np.random.uniform(0.04, 0.12),
        np.random.uniform(20, 40),
        "Moderate"
    ])

# Poor oil
for _ in range(100):
    data.append([
        np.random.uniform(20, 45),
        np.random.uniform(25, 55),
        np.random.uniform(0.10, 0.25),
        np.random.uniform(10, 25),
        "Poor"
    ])

df = pd.DataFrame(data, columns=["BDV", "Moisture", "Acidity", "IFT", "Condition"])

# -------------------------------
# Data Preprocessing
# -------------------------------
encoder = LabelEncoder()
df["Condition"] = encoder.fit_transform(df["Condition"])

X = df[["BDV", "Moisture", "Acidity", "IFT"]]
y = df["Condition"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# -------------------------------
# Random Forest Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,          # limits overfitting
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# Feature Importance
# -------------------------------
# Feature Importance
plt.figure(figsize=(6,4))
plt.bar(X.columns, model.feature_importances_, color='orange')
plt.title("Feature Importance of Chemical Parameters")
plt.xlabel("Parameters")
plt.ylabel("Importance")
plt.show()

