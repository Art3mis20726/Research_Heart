from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import pandas as pd

# Load the dataset
df = pd.read_csv(r"D:\Vedang\Python Files\AIML\Heart 2\heart.csv")

# Separate features and target variable
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Check class distribution
print("Original class distribution:", Counter(y))

# Encode categorical variables
for column in X.select_dtypes(include=['object']).columns:
    X[column] = LabelEncoder().fit_transform(X[column])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Check resampled class distribution
print("Resampled class distribution:", Counter(y_train_resampled))

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=41)
model.fit(X_train_resampled, y_train_resampled)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
