import os
import kagglehub # Note: kagglehub is imported but not used in this specific script.
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings('ignore')

def find_csv_file(directory):
    """Find the first CSV file in the given directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                return os.path.join(root, file)
    return None

def save_unseen_data(df, output_dir='unseenData'):
    """Saves the raw, unseen holdout data before any processing."""
    print(f"\n--- Saving Unseen Holdout Data to '{output_dir}' folder ---")
    os.makedirs(output_dir, exist_ok=True)

    df_copy = df.copy()
    df_copy['Attack_Binary'] = df_copy['Attack'].apply(lambda x: 'Normal' if x == 'Normal' else 'Attack')

    normal_df = df_copy[df_copy['Attack_Binary'] == 'Normal']
    attack_df = df_copy[df_copy['Attack_Binary'] == 'Attack']

    normal_path = os.path.join(output_dir, 'unseen_normal_samples.csv')
    attack_path = os.path.join(output_dir, 'unseen_attack_samples.csv')
    mixed_path = os.path.join(output_dir, 'unseen_mixed_samples.csv')

    normal_df.to_csv(normal_path, index=False)
    attack_df.to_csv(attack_path, index=False)
    df.to_csv(mixed_path, index=False)

    print(f"✅ Saved unseen data files to the '{output_dir}' folder.")


def create_perfectly_balanced_dataset(df, samples_per_class=25000):
    """Create a perfectly balanced dataset by handling imbalance."""
    print("=== Creating Perfectly Balanced Dataset ===")
    # Ensure Attack_Binary is present before value_counts, as it might not be from initial split
    df['Attack_Binary'] = df['Attack'].apply(lambda x: 'Normal' if x == 'Normal' else 'Attack')
    print("Original distribution in the training pool:")
    print(df['Attack_Binary'].value_counts())

    class_counts = df['Attack_Binary'].value_counts()
    if len(class_counts) > 1 and class_counts.iloc[0] != class_counts.iloc[1]:
        print("\nDataset is imbalanced. Proceeding with Random Undersampling to handle it.")
    
    normal_samples = df[df['Attack_Binary'] == 'Normal'].sample(n=samples_per_class, random_state=42)
    attack_samples = df[df['Attack_Binary'] == 'Attack'].sample(n=samples_per_class, random_state=42)
    
    balanced_df = pd.concat([normal_samples, attack_samples])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced dataset created:")
    print(f"Normal: {len(normal_samples):,}")
    print(f"Attack: {len(attack_samples):,}")
    print(f"Total: {len(balanced_df):,}")
    
    return balanced_df

def simple_robust_preprocessing(df):
    """Preprocessing with a reduced feature set."""
    print("=== Simple Robust Preprocessing ===")
    # Ensure 'Attack_Binary' exists for the target variable 'y'
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
    print(f"Using {len(available_features)} key features: {available_features}.")
    return X, y

def save_single_test_file(X_test, y_test, scaler, label_encoder, feature_columns, filename='test.csv'):
    """Saves the entire test dataset split into a single CSV file."""
    print(f"\n--- Saving Test Split to a Single CSV File ---")
    # Inverse transform X_test from scaled to original values for saving
    X_test_original = scaler.inverse_transform(X_test)
    # Inverse transform y_test from encoded to original labels
    y_test_original = label_encoder.inverse_transform(y_test)
    test_df = pd.DataFrame(X_test_original, columns=feature_columns)
    test_df['Attack_Binary'] = y_test_original
    test_df.to_csv(filename, index=False)
    print(f"✅ Test data with {len(test_df)} samples saved to: {filename}")

def create_and_save_visualizations(model, X_test, y_test, label_encoder, output_dir='visualizations'):
    """Generates and saves a confusion matrix and ROC curve plot."""
    print(f"\n--- Creating and Saving Visualizations to '{output_dir}' folder ---")
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix'); plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path); plt.close()
    print(f"✅ Confusion Matrix saved to: {cm_path}")

    # ROC Curve
    # Ensure we get probabilities for the positive class.
    # We need to know which class is 'Attack' and use its probability column.
    try:
        attack_class_index = list(label_encoder.classes_).index('Attack')
    except ValueError:
        # Fallback if 'Attack' isn't explicitly named, assume it's the 0th or 1st class.
        # This needs to match how it was encoded.
        # From previous logs: Attack_Binary_Encoded mapping: ['Attack', 'Normal'] -> [0 1]
        # So 'Attack' is index 0.
        attack_class_index = 0 
        print(f"Warning: 'Attack' class not found by name, assuming positive class is at index {attack_class_index}.")

    y_pred_proba = model.predict_proba(X_test)[:, attack_class_index]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=attack_class_index)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve'); plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path); plt.close()
    print(f"✅ ROC Curve saved to: {roc_path}")

def train_and_evaluate_model():
    """Main workflow to train a model and create all assets."""
    print("=== MODEL TRAINING WITH UNSEEN HOLDOUT SET & VISUALIZATIONS ===")

    print("Step 1: Loading full dataset...")
    csv_path = 'dataset/botnet.csv' # Ensure this path is correct
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        print(f"Full dataset loaded: {df.shape}")
    except FileNotFoundError:
        print(f"❌ ERROR: Dataset not found at '{csv_path}'. Please check the path.")
        return False
    
    # Ensure 'Attack' column exists before stratifying
    if 'Attack' not in df.columns:
        print("❌ ERROR: 'Attack' column not found in the dataset. Cannot proceed with stratification.")
        return False

    print("\nStep 2: Creating Unseen Holdout Set (10% of full data)...")
    training_pool, unseen_holdout_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df['Attack'] 
    )
    save_unseen_data(unseen_holdout_df.copy())
    
    print("\nStep 3: Handling Imbalance and Creating Balanced Dataset...")
    df_balanced = create_perfectly_balanced_dataset(training_pool, samples_per_class=25000)
    
    print("\nStep 4: Preprocessing...")
    X, y = simple_robust_preprocessing(df_balanced)
    
    print("\nStep 5: Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print("\nStep 6: Feature scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nStep 7: Splitting the dataset for Train and Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    save_single_test_file(X_test, y_test, scaler, label_encoder, list(X.columns))
    
    model_dir = os.path.join('ids_app', 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    print("\nStep 8: Saving preprocessing components...")
    joblib.dump(list(X.columns), os.path.join(model_dir, 'model_columns.pkl'))
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    print("\nStep 9: Training the Random Forest model using the train dataset...")
   
    rf_model = RandomForestClassifier(
        n_estimators=200,    
        max_depth=10,      
        min_samples_leaf=5,   
        random_state=42,
        n_jobs=-1            
    )
    

    rf_model.fit(X_train, y_train)
    print("✅ Model training complete.")
    
    print("\n" + "="*60)
    print("MODEL EVALUATION (on the test dataset)")
    print("="*60)
    
    y_train_pred = rf_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    y_test_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

    create_and_save_visualizations(rf_model, X_test, y_test, label_encoder)

    print("\nSaving model...")
    model_filename = "random_forest_model.pkl"
    joblib.dump(rf_model, os.path.join(model_dir, model_filename))
    print(f"✅ Model saved as: {model_filename}")
    
    print("\nTraining completed successfully!")
    return True

if __name__ == '__main__':
    success = train_and_evaluate_model()
    if success:
        print("\n🎉 SUCCESS! Model, test.csv, unseen data, and visualization files are ready.")
    else:
        print("\n❌ Training failed. Please check the error messages above.")