import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('predictive_maintenance.csv')

# Drop unused columns, but KEEP 'Type'
cols_to_drop = ['UDI', 'Product ID', 'Target', 'Failure Type']
X = df.drop(columns=cols_to_drop, errors='ignore')
y = df['Failure Type']

# 1. Feature Engineering
print("Engineering features...")
# One Hot Encode 'Type'
X = pd.get_dummies(X, columns=['Type'])

# Domain Specific Features
X['Temp_Diff'] = X['Process temperature [K]'] - X['Air temperature [K]']
X['Power'] = X['Rotational speed [rpm]'] * X['Torque [Nm]']
X['Strain'] = X['Tool wear [min]'] * X['Torque [Nm]']

# Ensure column order is fixed and save feature names
feature_names = list(X.columns)
joblib.dump(feature_names, 'feature_names.pkl')

print("Preparing data...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'label_encoder.pkl')

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# 2. Imbalance Handling & Modelling
print("Training Optimized Random Forest...")
# We use class_weight='balanced_subsample' to handle extreme imbalance natively
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

print("Evaluating...")
y_pred = rf_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Saving model...")
joblib.dump(rf_model, 'model.pkl')
print("Training complete and models exported!")
