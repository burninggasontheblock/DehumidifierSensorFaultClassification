# Dehumidifier Sensor Fault Classification

This project uses machine learning to classify failure types in a dehumidifier system based on sensor data. It trains a **Random Forest Classifier** using various sensor features such as humidity, temperature, fan speed, and others to detect potential failure modes.

## 📁 Files

- `bench_test.py`: Main script to preprocess the dataset, train the model, and evaluate classification performance.
- `dehumidifier_sensor_data.csv`: The input dataset containing sensor readings and labeled failure modes (required to run the script).

## 🔍 Features Used

The model uses the following features from the dataset:

- Humidity (%)
- Temperature (°C)
- Fan RPM
- Compressor Current (A)
- Vibration (m/s²)
- Pressure (psi)
- Power (W)
- Water Level (%)

## ⚙️ Workflow

1. Load sensor data from CSV.
2. Encode failure labels using `LabelEncoder`.
3. Split the data into training and testing sets.
4. Train a `RandomForestClassifier` on the training data.
5. Evaluate model performance using classification report and confusion matrix.

## 📊 Output

The script prints out:
- **Classification Report**: Precision, recall, f1-score for each failure label.
- **Confusion Matrix**: Matrix showing predicted vs actual classifications.

## 🧪 Requirements

Ensure the following Python libraries are installed:

```bash
pip install pandas scikit-learn

