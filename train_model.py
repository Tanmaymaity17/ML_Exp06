from math import log2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset from local file
data = pd.read_csv("WineQT.csv")

# Inspect column names
print("Columns:", data.columns)

# Features and target (assuming target is 'quality')
X = data.drop("quality", axis=1)
y = data["quality"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RandomForestClassifier
model = RandomForestClassifier(criterion="entropy", max_features="log2", random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save trained model
joblib.dump(model, "wine_quality_rf.pkl")
print("Model saved as wine_quality_rf.pkl")
