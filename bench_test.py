import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("dehumidifier_sensor_data.csv")

# Encode the failure labels
label_encoder = LabelEncoder()
df['failure_encoded'] = label_encoder.fit_transform(df['failure_label'])

# All possible labels from the encoder
labels = label_encoder.transform(label_encoder.classes_)

# Print the unique labels for testing purposes
# print("Train labels:", y_train.value_counts())
# print("Test labels:", y_test.value_counts())

# Features (you can add more or engineer new ones!)
features = [
    'humidity (%)',
    'temp (°C)',
    'fan_rpm',
    'compressor_current (A)',
    'vibration (m/s²)',
    'pressure (psi)',
    'power (W)',
    'water_level (%)'
]

X = df[features]
y = df['failure_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Safe classification report
print("Classification Report:\n", classification_report(y_test, y_pred, labels=labels, target_names=label_encoder.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=labels))
