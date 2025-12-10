import pandas as pd
import random
from faker import Faker

# Initialize Faker
fake = Faker()
Faker.seed(42)
random.seed(42)

def generate_providers_data(num_providers=100):
    """Generate synthetic provider data"""
    
    print(f"🏥 Generating {num_providers} synthetic providers...")
    
    specialties = [
        'cardiology',
        'oncology',
        'orthopedics',
        'general_medicine',
        'neurology',
        'gastroenterology'
    ]
    
    locations = [
        'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
        'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
        'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
        'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Boston'
    ]
    
    providers = []
    
    for i in range(1, num_providers + 1):
        provider_id = f"PROV{i:04d}"
        
        # Generate provider details
        name = fake.name()
        specialty = random.choice(specialties)
        location = random.choice(locations)
        years_experience = random.randint(1, 35)
        license_number = f"LIC{random.randint(10000, 99999)}"
        
        # Add some risk factors for certain providers
        # 10% of providers will have suspicious patterns
        is_high_risk = random.random() < 0.1
        
        if is_high_risk:
            # High-risk providers have more claims and higher amounts
            avg_claims_per_month = random.randint(50, 150)
            fraud_risk_score = random.uniform(0.6, 0.95)
        else:
            avg_claims_per_month = random.randint(10, 50)
            fraud_risk_score = random.uniform(0.0, 0.4)
        
        # Hospital affiliation
        hospital = fake.company() + " Medical Center"
        
        # Contact information
        phone = fake.phone_number()
        email = f"{name.lower().replace(' ', '.')}@{hospital.lower().replace(' ', '').replace('.', '')}hospital.com"
        
        providers.append({
            'provider_id': provider_id,
            'name': name,
            'specialty': specialty,
            'location': location,
            'years_experience': years_experience,
            'license_number': license_number,
            'hospital_affiliation': hospital,
            'phone': phone,
            'email': email,
            'avg_claims_per_month': avg_claims_per_month,
            'fraud_risk_score': round(fraud_risk_score, 3),
            'is_high_risk': is_high_risk
        })
    
    providers_df = pd.DataFrame(providers)
    
    print(f"✅ Generated {len(providers_df)} providers")
    print(f"   - High-risk providers: {providers_df['is_high_risk'].sum()}")
    print(f"   - Specialties distribution:")
    for specialty in specialties:
        count = (providers_df['specialty'] == specialty).sum()
        print(f"     • {specialty}: {count}")
    
    return providers_df

def save_providers_data(providers_df, output_path='providers.csv'):
    """Save providers data to CSV"""
    
    print(f"\n💾 Saving providers data to {output_path}...")
    providers_df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(providers_df)} providers to {output_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("🏥 Provider Data Generation Script")
    print("=" * 60)
    
    # Generate providers
    providers_df = generate_providers_data(num_providers=100)
    
    # Save to CSV
    save_providers_data(providers_df)
    
    print("\n" + "=" * 60)
    print("✅ Provider Data Generation Completed!")
    print("=" * 60)
    
    print("\n💡 Next Steps:")
    print("  1. Run 'etl_pipeline.py' to process claims with provider data")
    print("  2. The ETL pipeline will automatically enrich claims with provider information")
