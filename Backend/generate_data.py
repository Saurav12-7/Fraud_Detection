"""
Data Generation Script for Fraud Detection System
Generates synthetic claims and providers data with injected fraud patterns

This script creates reproducible mock data for the ETL + RAG fraud detection system.
"""

import pandas as pd
import random
from datetime import datetime, timedelta
from faker import Faker

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
Faker.seed(SEED)
fake = Faker()

# Configuration
NUM_CLAIMS = 3000
NUM_PATIENTS = 1500
NUM_PROVIDERS = 100
FRAUD_INJECTION_RATE = 0.20  # 20% of claims will have fraud patterns

# Medical codes and specialties
SPECIALTIES = {
    'cardiology': {
        'diagnoses': ['D001', 'D002', 'D003'],
        'procedures': ['P001', 'P002', 'P003'],
        'medications': ['M001', 'M002', 'M003', 'M004', 'M005'],
        'amount_range': (2500, 17000)
    },
    'oncology': {
        'diagnoses': ['D004', 'D005', 'D006'],
        'procedures': ['P004', 'P005', 'P006'],
        'medications': ['M006'],
        'amount_range': (7000, 28000)
    },
    'orthopedics': {
        'diagnoses': ['D007', 'D008', 'D009'],
        'procedures': ['P007', 'P008', 'P009'],
        'medications': ['M007', 'M008', 'M009'],
        'amount_range': (1800, 48000)
    },
    'general_medicine': {
        'diagnoses': ['D010', 'D011', 'D012'],
        'procedures': ['P010', 'P011', 'P012'],
        'medications': ['M010', 'M011'],
        'amount_range': (450, 2200)
    },
    'neurology': {
        'diagnoses': ['D013', 'D014', 'D015'],
        'procedures': ['P013', 'P014', 'P015'],
        'medications': ['M012', 'M013', 'M014'],
        'amount_range': (2700, 9000)
    },
    'gastroenterology': {
        'diagnoses': ['D016', 'D017', 'D018'],
        'procedures': ['P016', 'P017', 'P018'],
        'medications': ['M015', 'M016'],
        'amount_range': (4500, 13000)
    }
}

CLAIM_STATUSES = ['submitted', 'approved', 'rejected']

def generate_claim_date():
    """Generate random claim date within last 6 months"""
    end_date = datetime(2025, 12, 10, 15, 20, 12, 348144)
    start_date = end_date - timedelta(days=180)
    random_days = random.randint(0, 180)
    return start_date + timedelta(days=random_days)

def generate_normal_claim(claim_id, specialty):
    """Generate a normal (non-fraudulent) claim"""
    spec_data = SPECIALTIES[specialty]
    
    return {
        'claim_id': f'CLAIM{claim_id:05d}',
        'patient_id': f'PAT{random.randint(1, NUM_PATIENTS):05d}',
        'provider_id': f'PROV{random.randint(1, NUM_PROVIDERS):04d}',
        'claim_date': generate_claim_date(),
        'specialty': specialty,
        'diagnosis_code': random.choice(spec_data['diagnoses']),
        'procedure_code': random.choice(spec_data['procedures']),
        'medication_code': random.choice(spec_data['medications']),
        'amount': round(random.uniform(*spec_data['amount_range']), 2),
        'status': random.choice(CLAIM_STATUSES),
        'injected_fraud_type': 'none'
    }

def inject_duplicate_billing(claims_list, claim_id):
    """Inject duplicate billing fraud pattern"""
    # Pick a random existing claim to duplicate
    if len(claims_list) > 10:
        original = random.choice(claims_list[-100:])
        duplicate = original.copy()
        duplicate['claim_id'] = f'CLAIM{claim_id:05d}'
        duplicate['injected_fraud_type'] = 'duplicate_billing'
        return duplicate
    return None

def inject_abnormal_amount(claim_id, specialty):
    """Inject abnormal amount fraud pattern"""
    claim = generate_normal_claim(claim_id, specialty)
    spec_data = SPECIALTIES[specialty]
    
    # Generate amount 5-10x the normal range
    multiplier = random.uniform(3, 5)
    normal_max = spec_data['amount_range'][1]
    claim['amount'] = round(normal_max * multiplier, 2)
    claim['injected_fraud_type'] = 'abnormal_amount'
    
    return claim

def inject_specialty_mismatch(claim_id, specialty):
    """Inject specialty-procedure mismatch fraud pattern"""
    claim = generate_normal_claim(claim_id, specialty)
    
    # Pick a procedure from a different specialty
    other_specialties = [s for s in SPECIALTIES.keys() if s != specialty]
    wrong_specialty = random.choice(other_specialties)
    claim['procedure_code'] = random.choice(SPECIALTIES[wrong_specialty]['procedures'])
    claim['injected_fraud_type'] = 'mismatch_specialty_procedure'
    
    return claim

def inject_wrong_medication(claim_id, specialty):
    """Inject wrong medication fraud pattern"""
    claim = generate_normal_claim(claim_id, specialty)
    
    # Pick a medication from a different specialty
    other_specialties = [s for s in SPECIALTIES.keys() if s != specialty]
    wrong_specialty = random.choice(other_specialties)
    claim['medication_code'] = random.choice(SPECIALTIES[wrong_specialty]['medications'])
    claim['injected_fraud_type'] = 'wrong_medication'
    
    return claim

def generate_claims_data():
    """Generate complete claims dataset with fraud patterns"""
    print(f"🏥 Generating {NUM_CLAIMS} synthetic claims...")
    
    claims = []
    fraud_counts = {
        'duplicate_billing': 0,
        'abnormal_amount': 0,
        'mismatch_specialty_procedure': 0,
        'wrong_medication': 0,
        'none': 0
    }
    
    for i in range(1, NUM_CLAIMS + 1):
        specialty = random.choice(list(SPECIALTIES.keys()))
        
        # Decide if this claim should have fraud
        if random.random() < FRAUD_INJECTION_RATE:
            # Choose fraud type
            fraud_type = random.choice([
                'duplicate_billing',
                'abnormal_amount',
                'mismatch_specialty_procedure',
                'wrong_medication'
            ])
            
            if fraud_type == 'duplicate_billing':
                claim = inject_duplicate_billing(claims, i)
                if claim is None:
                    claim = generate_normal_claim(i, specialty)
                    fraud_type = 'none'
            elif fraud_type == 'abnormal_amount':
                claim = inject_abnormal_amount(i, specialty)
            elif fraud_type == 'mismatch_specialty_procedure':
                claim = inject_specialty_mismatch(i, specialty)
            else:  # wrong_medication
                claim = inject_wrong_medication(i, specialty)
            
            fraud_counts[fraud_type] += 1
        else:
            claim = generate_normal_claim(i, specialty)
            fraud_counts['none'] += 1
        
        claims.append(claim)
    
    # Print fraud statistics
    print(f"\n📊 Fraud Pattern Distribution:")
    for fraud_type, count in fraud_counts.items():
        percentage = (count / NUM_CLAIMS) * 100
        print(f"  • {fraud_type}: {count} ({percentage:.1f}%)")
    
    return pd.DataFrame(claims)

def generate_providers_data():
    """Generate providers dataset"""
    print(f"\n🏥 Generating {NUM_PROVIDERS} synthetic providers...")
    
    providers = []
    locations = [
        'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
        'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
        'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
        'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Boston'
    ]
    
    for i in range(1, NUM_PROVIDERS + 1):
        specialty = random.choice(list(SPECIALTIES.keys()))
        is_high_risk = random.random() < 0.1  # 10% high-risk
        
        providers.append({
            'provider_id': f'PROV{i:04d}',
            'name': fake.name(),
            'specialty': specialty,
            'location': random.choice(locations),
            'years_experience': random.randint(1, 35),
            'license_number': f'LIC{random.randint(10000, 99999)}',
            'hospital_affiliation': fake.company() + ' Medical Center',
            'phone': fake.phone_number(),
            'email': f"{fake.user_name()}@{fake.domain_name()}",
            'avg_claims_per_month': random.randint(50, 150) if is_high_risk else random.randint(10, 50),
            'fraud_risk_score': round(random.uniform(0.6, 0.95), 3) if is_high_risk else round(random.uniform(0.0, 0.4), 3),
            'is_high_risk': is_high_risk
        })
    
    print(f"✅ Generated {NUM_PROVIDERS} providers")
    print(f"  • High-risk providers: {sum(p['is_high_risk'] for p in providers)}")
    
    return pd.DataFrame(providers)

def main():
    """Main execution function"""
    print("=" * 60)
    print("🚀 Fraud Detection Data Generation Script")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  • Random Seed: {SEED}")
    print(f"  • Number of Claims: {NUM_CLAIMS}")
    print(f"  • Number of Patients: {NUM_PATIENTS}")
    print(f"  • Number of Providers: {NUM_PROVIDERS}")
    print(f"  • Fraud Injection Rate: {FRAUD_INJECTION_RATE * 100}%")
    print("=" * 60)
    
    # Generate claims
    claims_df = generate_claims_data()
    
    # Generate providers
    providers_df = generate_providers_data()
    
    # Save to CSV
    print(f"\n💾 Saving data files...")
    claims_df.to_csv('claims.csv', index=False)
    print(f"✅ Saved claims.csv ({len(claims_df)} rows)")
    
    providers_df.to_csv('providers.csv', index=False)
    print(f"✅ Saved providers.csv ({len(providers_df)} rows)")
    
    print("\n" + "=" * 60)
    print("✅ Data Generation Completed Successfully!")
    print("=" * 60)
    
    print("\n💡 Next Steps:")
    print("  1. Run 'python etl_pipeline.py' to process the data")
    print("  2. Run 'python train_ml.py' to train the ML model")
    print("  3. Launch the app with 'streamlit run ../Frontend/interface.py'")

if __name__ == "__main__":
    main()
