import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def create_perfectly_balanced_dataset(df, samples_per_class=25000):
    """(Copied from train_model.py) Creates a perfectly balanced dataset."""
    # Ensure Attack_Binary is present for correct balancing
    if 'Attack_Binary' not in df.columns:
        df['Attack_Binary'] = df['Attack'].apply(lambda x: 'Normal' if x == 'Normal' else 'Attack')
    
    normal_samples = df[df['Attack_Binary'] == 'Normal'].sample(n=samples_per_class, random_state=42)
    attack_samples = df[df['Attack_Binary'] == 'Attack'].sample(n=samples_per_class, random_state=42)
    
    balanced_df = pd.concat([normal_samples, attack_samples])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

def simple_robust_preprocessing(df):
    """(Copied from train_model.py) Preprocessing with a reduced feature set."""
    # Ensure Attack_Binary is present for correct target separation
    if 'Attack_Binary' not in df.columns:
        df['Attack_Binary'] = df['Attack'].apply(lambda x: 'Normal' if x == 'Normal' else 'Attack')

    y = df['Attack_Binary']
    key_features = [
        'MI_dir_L0.1_mean', 'MI_dir_L0.1_variance',
        'H_L0.1_mean', 'H_L0.1_variance',
        'HH_L0.1_mean', 'HH_L0.1_std',
        'HpHp_L0.1_mean'
    ]
    available_features = [col for col in key_features if col in df.columns]
    X = df[available_features].copy()
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    return X, y

def generate_preprocessed_data_outputs():
    print("🚀 Generating Pre-processed Dataset Outputs 🚀")

    # --- 1. SETUP & DATA LOADING ---
    csv_path = 'dataset/botnet.csv' # IMPORTANT: Ensure this path is correct
    try:
        df_raw = pd.read_csv(csv_path)
        df_raw.columns = df_raw.columns.str.strip()
        print(f"✅ Raw dataset loaded successfully from: {csv_path}")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{csv_path}'. Please update the path and try again.")
        return
    
    # Create 'Attack_Binary' for stratification and balancing, if not already present
    df_raw['Attack_Binary'] = df_raw['Attack'].apply(lambda x: 'Normal' if x == 'Normal' else 'Attack')

    # Initial split to simulate the training pool (10% held out as unseen)
    training_pool_df, _ = train_test_split(
        df_raw, test_size=0.1, random_state=42, stratify=df_raw['Attack']
    )
    print(f"Dataset split into training pool ({len(training_pool_df):,} records) and unseen holdout.")

    # Apply balancing (undersampling) to the training pool
    df_balanced = create_perfectly_balanced_dataset(training_pool_df, samples_per_class=25000)
    print(f"Dataset balanced to {len(df_balanced):,} records (25,000 Normal, 25,000 Attack).")

    # Apply feature selection and basic cleaning (filling NaNs/infs)
    X_preprocessed, y_preprocessed = simple_robust_preprocessing(df_balanced)
    print(f"Features selected: {list(X_preprocessed.columns)}")

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_preprocessed)
    print(f"✅ Target variable encoded: {list(label_encoder.classes_)} -> {label_encoder.transform(label_encoder.classes_)}")

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_preprocessed)
    print("✅ Numerical features scaled using StandardScaler.")

    # Reconstruct the final pre-processed DataFrame (scaled features + encoded target) for display
    df_final_preprocessed = pd.DataFrame(X_scaled, columns=X_preprocessed.columns)
    df_final_preprocessed['Attack_Binary_Encoded'] = y_encoded
    print(f"Final pre-processed DataFrame shape: {df_final_preprocessed.shape}")
    print("\n" + "="*70)
    print("OUTPUTS FOR REPORT SECTION 4.3 - PRE-PROCESSED DATASET")
    print("="*70)


    # --- Display for each subsection ---

    # a) First five records of the dataset
    print("\n--- a) First five records of the pre-processed dataset ---")
    print(df_final_preprocessed.head())

    # b) Last five records of the dataset
    print("\n--- b) Last five records of the pre-processed dataset ---")
    print(df_final_preprocessed.tail())

    # c) Data Structure and Metadata
    print("\n--- c) Data Structure and Metadata ---")
    df_final_preprocessed.info()

    # d) Shape of the dataset
    print("\n--- d) Shape of the dataset ---")
    print(f"Shape of the final pre-processed dataset: {df_final_preprocessed.shape}")

    # e) List of columns
    print("\n--- e) List of columns ---")
    print("Features of the pre-processed dataset:")
    print(df_final_preprocessed.columns.tolist())

    # f) Data Types of columns
    print("\n--- f) Data Types of columns ---")
    print(df_final_preprocessed.dtypes)

    # g) Summary statistics of numerical features
    print("\n--- g) Summary statistics of numerical features ---")
    print(df_final_preprocessed.describe().T)

    # Dataset splitting
    print("\n--- Dataset Splitting ---")
    X_train, X_test, y_train, y_test = train_test_split(
        df_final_preprocessed.drop('Attack_Binary_Encoded', axis=1),
        df_final_preprocessed['Attack_Binary_Encoded'],
        test_size=0.2,
        random_state=42,
        stratify=df_final_preprocessed['Attack_Binary_Encoded']
    )

    print(f"Total records in pre-processed dataset: {len(df_final_preprocessed):,}")
    print(f"Training set size (75%): {len(X_train):,} records")
    print(f"Test set size (25%): {len(X_test):,} records")

    print("\nClass distribution in Training Set:")
    print(pd.Series(y_train).value_counts(normalize=True))
    print("\nClass distribution in Test Set:")
    print(pd.Series(y_test).value_counts(normalize=True))
    print("\n🎉 All outputs generated successfully!")


if __name__ == '__main__':
    generate_preprocessed_data_outputs()