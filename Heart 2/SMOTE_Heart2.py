from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import pandas as pd

# Step 1: Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset
df = pd.read_csv(r"D:\Vedang\Python Files\AIML\Heart 2\heart.csv")

# Separate features (X) and target (y)
X = df.drop("HeartDisease", axis=1)  # Replace 'target' with your actual target column name
y = df["HeartDisease"]

# Show the class distribution
print("Original class distribution:", Counter(y))

for column in X.select_dtypes(include=['object']).columns:
    X[column] = LabelEncoder().fit_transform(X[column])

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Show the new class distribution
print("Resampled class distribution:", Counter(y_train_resampled))

# Step 4: Train a model on the resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Step 5: Evaluate the model on the test set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
