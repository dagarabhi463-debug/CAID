"""
Synthetic Security Log Generator for CAID

This script generates realistic educational-platform activity records
for demonstration purposes. All data is synthetic and randomized.

IMPORTANT:
- Uses only documentation-only IP ranges (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24)
- No real Canvas infrastructure is accessed or simulated
- All user IDs and activity are randomly generated
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Documentation-only IP ranges (RFC 5737)
NORMAL_IP_PREFIX = "192.0.2."
SUSPICIOUS_IP_PREFIX = "198.51.100."
HIGH_RISK_IP_PREFIX = "203.0.113."

def generate_timestamp(base_date, hour_range=(0, 23), minute_range=(0, 59)):
    """Generate a random timestamp within the specified ranges."""
    hour = random.randint(*hour_range)
    minute = random.randint(*minute_range)
    return base_date.replace(hour=hour, minute=minute, second=random.randint(0, 59))

def generate_normal_activity(n_records=500):
    """Generate normal student activity records."""
    records = []
    base_date = datetime(2024, 1, 15)
    
    roles = ['student', 'instructor', 'ta']
    course_ids = [f"COURSE_{i:04d}" for i in range(1, 21)]
    event_types = ['login', 'course_view', 'resource_view', 'assignment_submit', 'discussion_post']
    
    for i in range(n_records):
        # Normal activity pattern: regular hours, moderate activity
        day_offset = random.randint(0, 30)
        timestamp = generate_timestamp(base_date + timedelta(days=day_offset), (8, 22))
        
        records.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': f"user_{random.randint(1000, 9999)}",
            'role': random.choice(roles),
            'source_ip': f"{NORMAL_IP_PREFIX}{random.randint(1, 254)}",
            'event_type': random.choice(event_types),
            'courses_accessed': random.randint(1, 5),
            'accounts_accessed': random.randint(1, 3),
            'api_requests': random.randint(10, 100),
            'data_mb': round(random.uniform(0.1, 5.0), 2),
            'token_age_minutes': random.randint(5, 120),
            'after_hours': 0,
            'label': 'normal'
        })
    
    return pd.DataFrame(records)

def generate_unusual_activity(n_records=100):
    """Generate unusual activity records."""
    records = []
    base_date = datetime(2024, 1, 15)
    
    roles = ['student', 'instructor']
    course_ids = [f"COURSE_{i:04d}" for i in range(1, 31)]
    event_types = ['login', 'course_view', 'resource_view', 'grade_view', 'enrollment_view']
    
    for i in range(n_records):
        # Unusual patterns: more courses, higher API usage, some after-hours
        day_offset = random.randint(0, 30)
        # Use before_hours or after_hours based on random choice
        if random.random() > 0.5:
            hour_range = (6, 22)  # Normal-ish hours
        else:
            hour_range = (22, 23)  # Late evening
            hour_range = (0, 2) if random.random() > 0.5 else hour_range
        timestamp = generate_timestamp(base_date + timedelta(days=day_offset), (0, 2) if random.random() > 0.5 else (6, 22))
        
        records.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': f"user_{random.randint(1000, 9999)}",
            'role': random.choice(roles),
            'source_ip': f"{SUSPICIOUS_IP_PREFIX}{random.randint(1, 254)}",
            'event_type': random.choice(event_types),
            'courses_accessed': random.randint(6, 15),
            'accounts_accessed': random.randint(4, 10),
            'api_requests': random.randint(101, 500),
            'data_mb': round(random.uniform(5.1, 20.0), 2),
            'token_age_minutes': random.randint(121, 480),
            'after_hours': random.randint(0, 1),
            'label': 'suspicious'
        })
    
    return pd.DataFrame(records)

def generate_high_risk_activity(n_records=50):
    """Generate high-risk activity records."""
    records = []
    base_date = datetime(2024, 1, 15)
    
    event_types = ['login', 'grade_download', 'student_export', 'course_export', 'bulk_download']
    
    for i in range(n_records):
        # High-risk patterns: very high activity, large data transfers
        day_offset = random.randint(0, 30)
        timestamp = generate_timestamp(base_date + timedelta(days=day_offset), (2, 5))
        
        records.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': f"user_{random.randint(1000, 9999)}",
            'role': random.choice(['student', 'instructor', 'admin']),
            'source_ip': f"{HIGH_RISK_IP_PREFIX}{random.randint(1, 254)}",
            'event_type': random.choice(event_types),
            'courses_accessed': random.randint(16, 50),
            'accounts_accessed': random.randint(11, 100),
            'api_requests': random.randint(501, 2000),
            'data_mb': round(random.uniform(20.1, 100.0), 2),
            'token_age_minutes': random.randint(481, 2880),
            'after_hours': 1,
            'label': 'high_risk'
        })
    
    return pd.DataFrame(records)

def generate_evaluation_dataset():
    """
    Generate a separate evaluation dataset with balanced class distribution.
    
    This dataset is used for proper model evaluation with:
    - 100 normal records (negative examples)
    - 50 suspicious records (positive in binary mode)
    - 50 high-risk records (positive in high-risk mode)
    
    IMPORTANT: This data is generated SEPARATELY from training data to avoid
    any overlap or data leakage.
    """
    print("Generating evaluation dataset...")
    
    # Generate evaluation records (completely separate from training)
    df_normal = generate_normal_activity(100)
    df_suspicious = generate_unusual_activity(50)
    df_high_risk = generate_high_risk_activity(50)
    
    # Combine all datasets
    df_eval = pd.concat([df_normal, df_suspicious, df_high_risk], ignore_index=True)
    
    # Shuffle the dataset
    df_eval = df_eval.sample(frac=1, random_state=RANDOM_SEED + 1000).reset_index(drop=True)
    
    # Save to CSV file
    df_eval.to_csv('data/processed/evaluation_data.csv', index=False)
    
    print(f"Generated evaluation dataset with {len(df_eval)} records:")
    print(f"  - Normal activity: {len(df_normal)} records")
    print(f"  - Suspicious activity: {len(df_suspicious)} records")
    print(f"  - High-risk activity: {len(df_high_risk)} records")
    print("\nSaved files:")
    print("  - data/processed/evaluation_data.csv")
    
    return df_eval


def generate_dataset():
    """Generate the complete synthetic dataset."""
    print("Generating synthetic security logs...")
    
    df_normal = generate_normal_activity(500)
    df_suspicious = generate_unusual_activity(100)
    df_high_risk = generate_high_risk_activity(50)
    
    # Combine all datasets
    df = pd.concat([df_normal, df_suspicious, df_high_risk], ignore_index=True)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Save to CSV files
    df.to_csv('data/raw/security_logs.csv', index=False)
    df_normal.to_csv('data/raw/normal_activity.csv', index=False)
    df_suspicious.to_csv('data/raw/suspicious_activity.csv', index=False)
    df_high_risk.to_csv('data/raw/high_risk_activity.csv', index=False)
    
    print(f"Generated {len(df)} total records:")
    print(f"  - Normal activity: {len(df_normal)} records")
    print(f"  - Suspicious activity: {len(df_suspicious)} records")
    print(f"  - High-risk activity: {len(df_high_risk)} records")
    print("\nSaved files:")
    print("  - data/raw/security_logs.csv")
    print("  - data/raw/normal_activity.csv")
    print("  - data/raw/suspicious_activity.csv")
    print("  - data/raw/high_risk_activity.csv")
    
    return df

if __name__ == '__main__':
    df = generate_dataset()
    print("\nSample data preview:")
    print(df.head(10).to_string())
    
    # Generate evaluation dataset
    df_eval = generate_evaluation_dataset()
    print("\nEvaluation dataset preview:")
    print(df_eval.head(10).to_string())
