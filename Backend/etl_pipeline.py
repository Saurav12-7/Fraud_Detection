import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_raw_data(claims_path='claims.csv', providers_path='providers.csv'):
    """Load raw claims and providers data"""
    print("📥 Loading raw data...")
    
    if not os.path.exists(claims_path):
        raise FileNotFoundError(f"Claims file not found: {claims_path}")
    
    claims_df = pd.read_csv(claims_path)
    print(f"✅ Loaded {len(claims_df)} claims")
    
    providers_df = None
    if os.path.exists(providers_path):
        providers_df = pd.read_csv(providers_path)
        print(f"✅ Loaded {len(providers_df)} providers")
    else:
        print("⚠️  Providers file not found, skipping provider enrichment")
    
    return claims_df, providers_df

def clean_data(df):
    """Clean and validate data"""
    print("\n🧹 Cleaning data...")
    
    initial_count = len(df)
    
    # Convert dates
    df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
    
    # Remove rows with missing critical fields
    df = df.dropna(subset=['claim_id', 'patient_id', 'provider_id', 'amount'])
    
    # Remove negative amounts
    df = df[df['amount'] > 0]
    
    # Remove duplicates based on claim_id
    df = df.drop_duplicates(subset=['claim_id'], keep='first')
    
    cleaned_count = len(df)
    removed = initial_count - cleaned_count
    
    print(f"✅ Cleaned data: {cleaned_count} rows (removed {removed} invalid rows)")
    
    return df

def detect_duplicate_billing(df):
    """Rule 1: Detect duplicate billing"""
    print("\n🔍 Detecting duplicate billing...")
    
    # Group by patient, provider, date, procedure, and amount
    duplicate_cols = ['patient_id', 'provider_id', 'claim_date', 'procedure_code', 'amount']
    
    # Find duplicates
    df['is_duplicate'] = df.duplicated(subset=duplicate_cols, keep=False)
    
    duplicate_count = df['is_duplicate'].sum()
    print(f"⚠️  Found {duplicate_count} potential duplicate billing cases")
    
    return df

def detect_abnormal_amounts(df):
    """Rule 2: Detect abnormal amounts (>3 standard deviations)"""
    print("\n🔍 Detecting abnormal amounts...")
    
    # Calculate statistics by specialty
    specialty_stats = df.groupby('specialty')['amount'].agg(['mean', 'std']).reset_index()
    
    # Merge stats back to main dataframe
    df = df.merge(specialty_stats, on='specialty', how='left', suffixes=('', '_stat'))
    
    # Flag amounts > 3 standard deviations from mean
    df['is_abnormal_amount'] = (
        (df['amount'] > df['mean'] + 3 * df['std']) |
        (df['amount'] < df['mean'] - 3 * df['std'])
    )
    
    # Drop temporary columns
    df = df.drop(columns=['mean', 'std'])
    
    abnormal_count = df['is_abnormal_amount'].sum()
    print(f"⚠️  Found {abnormal_count} abnormal amount cases")
    
    return df

def detect_specialty_procedure_mismatch(df):
    """Rule 3: Detect specialty-procedure mismatches"""
    print("\n🔍 Detecting specialty-procedure mismatches...")
    
    # Define valid specialty-procedure mappings
    valid_mappings = {
        'cardiology': ['P001', 'P002', 'P003'],
        'oncology': ['P004', 'P005', 'P006'],
        'orthopedics': ['P007', 'P008', 'P009'],
        'general_medicine': ['P010', 'P011', 'P012'],
        'neurology': ['P013', 'P014', 'P015'],
        'gastroenterology': ['P016', 'P017', 'P018']
    }
    
    def check_mismatch(row):
        specialty = row['specialty']
        procedure = row['procedure_code']
        
        if specialty in valid_mappings:
            return procedure not in valid_mappings[specialty]
        return False
    
    df['is_specialty_mismatch'] = df.apply(check_mismatch, axis=1)
    
    mismatch_count = df['is_specialty_mismatch'].sum()
    print(f"⚠️  Found {mismatch_count} specialty-procedure mismatch cases")
    
    return df

def detect_medication_errors(df):
    """Rule 4: Detect medication errors (wrong medication for diagnosis)"""
    print("\n🔍 Detecting medication errors...")
    
    # Define valid diagnosis-medication mappings
    valid_med_mappings = {
        'D001': ['M001', 'M002'],  # Cardiology diagnoses
        'D002': ['M003', 'M004'],
        'D003': ['M002', 'M005'],
        'D004': ['M006'],  # Oncology
        'D005': ['M006'],
        'D006': ['M006'],
        'D007': ['M007', 'M008'],  # Orthopedics
        'D008': ['M007', 'M008'],
        'D009': ['M007', 'M009'],
        'D010': ['M010'],  # General medicine
        'D011': ['M010'],
        'D012': ['M011'],
        'D013': ['M012'],  # Neurology
        'D014': ['M013'],
        'D015': ['M014'],
        'D016': ['M015'],  # Gastroenterology
        'D017': ['M015'],
        'D018': ['M016']
    }
    
    def check_medication_error(row):
        diagnosis = row['diagnosis_code']
        medication = row['medication_code']
        
        if diagnosis in valid_med_mappings:
            return medication not in valid_med_mappings[diagnosis]
        return False
    
    df['is_medication_error'] = df.apply(check_medication_error, axis=1)
    
    med_error_count = df['is_medication_error'].sum()
    print(f"⚠️  Found {med_error_count} medication error cases")
    
    return df

def combine_fraud_flags(df):
    """Combine all fraud detection flags"""
    print("\n🔗 Combining fraud detection flags...")
    
    # Create rule-based fraud flag
    df['rule_based_fraud'] = (
        df['is_duplicate'] |
        df['is_abnormal_amount'] |
        df['is_specialty_mismatch'] |
        df['is_medication_error']
    )
    
    # Create fraud reason column
    def get_fraud_reasons(row):
        reasons = []
        if row['is_duplicate']:
            reasons.append('Duplicate Billing')
        if row['is_abnormal_amount']:
            reasons.append('Abnormal Amount')
        if row['is_specialty_mismatch']:
            reasons.append('Specialty-Procedure Mismatch')
        if row['is_medication_error']:
            reasons.append('Medication Error')
        
        return '; '.join(reasons) if reasons else 'None'
    
    df['fraud_reasons'] = df.apply(get_fraud_reasons, axis=1)
    
    rule_fraud_count = df['rule_based_fraud'].sum()
    print(f"⚠️  Total rule-based fraud cases: {rule_fraud_count}")
    
    return df

def enrich_with_providers(claims_df, providers_df):
    """Enrich claims with provider information"""
    if providers_df is None:
        print("\n⚠️  Skipping provider enrichment (no provider data)")
        return claims_df
    
    print("\n🔗 Enriching claims with provider data...")
    
    enriched_df = claims_df.merge(
        providers_df,
        on='provider_id',
        how='left',
        suffixes=('', '_provider')
    )
    
    print(f"✅ Enriched {len(enriched_df)} claims with provider information")
    
    return enriched_df

def save_processed_data(df, output_path='processed_claims_etl.csv'):
    """Save processed data"""
    print(f"\n💾 Saving processed data to {output_path}...")
    
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {len(df)} processed claims")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"  Total Claims: {len(df)}")
    print(f"  Rule-Based Fraud Cases: {df['rule_based_fraud'].sum()}")
    print(f"  Fraud Rate: {(df['rule_based_fraud'].sum() / len(df) * 100):.2f}%")
    
    if 'prediction_label' in df.columns:
        ml_fraud = (df['prediction_label'] == 'Suspicious').sum()
        print(f"  ML-Based Fraud Cases: {ml_fraud}")
        
        # Combined fraud (either rule-based OR ML)
        combined_fraud = (df['rule_based_fraud'] | (df['prediction_label'] == 'Suspicious')).sum()
        print(f"  Combined Fraud Cases: {combined_fraud}")
        print(f"  Combined Fraud Rate: {(combined_fraud / len(df) * 100):.2f}%")

def run_etl_pipeline(claims_path='claims.csv', providers_path='providers.csv'):
    """Main ETL pipeline execution"""
    print("=" * 60)
    print("🚀 Starting ETL Pipeline for Fraud Detection")
    print(f"📂 Claims: {claims_path}")
    print(f"📂 Providers: {providers_path}")
    print("=" * 60)
    
    try:
        # Step 1: Load raw data
        claims_df, providers_df = load_raw_data(claims_path, providers_path)
        
        # Step 2: Clean data
        claims_df = clean_data(claims_df)
        
        # Step 3: Apply rule-based fraud detection
        claims_df = detect_duplicate_billing(claims_df)
        claims_df = detect_abnormal_amounts(claims_df)
        claims_df = detect_specialty_procedure_mismatch(claims_df)
        claims_df = detect_medication_errors(claims_df)
        
        # Step 4: Combine fraud flags
        claims_df = combine_fraud_flags(claims_df)
        
        # Step 5: Enrich with provider data
        claims_df = enrich_with_providers(claims_df, providers_df)
        
        # Step 6: Save processed data
        save_processed_data(claims_df)
        
        print("\n" + "=" * 60)
        print("✅ ETL Pipeline Completed Successfully!")
        print("=" * 60)
        
        return claims_df
        
    except Exception as e:
        print(f"\n❌ ETL Pipeline Failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the ETL pipeline
    processed_df = run_etl_pipeline()
    
    print("\n💡 Next Steps:")
    print("  1. Run 'train_ml.py' to train the ML model")
    print("  2. Launch the Streamlit app with 'streamlit run ../Frontend/interface.py'")
