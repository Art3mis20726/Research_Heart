from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\Vedang\Python Files\AIML\Heart 3\heart_attack_prediction_dataset.csv")
print("Missing values per column:\n", df.isnull().sum())
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop("Heart Attack Risk", axis=1)
y = df["Heart Attack Risk"]
print("Original class distribution:", Counter(y))

X_encoded = X.copy()
for column in X_encoded.select_dtypes(include=['object']).columns:
    X_encoded[column] = LabelEncoder().fit_transform(X_encoded[column])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Resampled class distribution:", Counter(y_train_resampled))

model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

model.fit(X_test, y_test)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
