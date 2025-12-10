import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import os

def run_ml_training():
    print("🚀 Starting Fraud Detection Model Training...")
    
    # 1. LOAD THE DATA
    csv_path = "claims.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: '{csv_path}' not found. Please save your CSV data in the Data_Engine folder.")
        return

    df = pd.read_csv(csv_path)
    print(f"✅ Data Loaded: {len(df)} rows found.")

    # 2. PRE-PROCESSING (Convert Text to Numbers for AI)
    print("⚙️  Processing Data for Machine Learning...")
    
    # We need to turn 'Cardiology', 'P001', etc. into numbers so the model understands.
    le_specialty = LabelEncoder()
    le_procedure = LabelEncoder()
    
    df['specialty_encoded'] = le_specialty.fit_transform(df['specialty'])
    df['procedure_encoded'] = le_procedure.fit_transform(df['procedure_code'])
    
    # Create Target Variable (0 = Clean, 1 = Fraud)
    # If injected_fraud_type is 'none', it is NOT fraud. Anything else is fraud.
    df['is_fraud'] = df['injected_fraud_type'].apply(lambda x: 0 if x == 'none' else 1)

    # 3. TRAIN THE MODEL
    print("🧠 Training Random Forest Model...")
    
    # Features: Specialty, Procedure, Amount
    X = df[['specialty_encoded', 'procedure_encoded', 'amount']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and Train
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    predictions = rf_model.predict(X_test)
    print("\n--- Model Performance Report ---")
    print(classification_report(y_test, predictions))
    
    # 4. PREDICT ON FULL DATASET (For the Chatbot UI)
    df['ml_prediction'] = rf_model.predict(X)
    df['prediction_label'] = df['ml_prediction'].apply(lambda x: "Suspicious" if x == 1 else "Safe")
    
    # 5. SAVE ARTIFACTS
    # Save the model
    with open('fraud_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Save the encoders (to decode later if needed)
    with open('encoders.pkl', 'wb') as f:
        pickle.dump((le_specialty, le_procedure), f)

    # Save the processed data for the Chatbot to read
    output_file = "processed_claims_ml.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Success! Trained Model saved to 'fraud_model.pkl'")
    print(f"✅ Processed Data saved to '{output_file}'")

if __name__ == "__main__":
    run_ml_training()