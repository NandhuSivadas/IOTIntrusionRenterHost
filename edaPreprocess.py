import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Needed for initial unseen data split

import warnings
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# --- Re-using functions from train_model.py for consistency ---

def create_perfectly_balanced_dataset(df, samples_per_class=25000):
    """(Copied from train_model.py) Creates a perfectly balanced dataset."""
    # Ensure 'Attack_Binary' is created if not already present
    if 'Attack_Binary' not in df.columns:
        df['Attack_Binary'] = df['Attack'].apply(lambda x: 'Normal' if x == 'Normal' else 'Attack')
    
    normal_samples = df[df['Attack_Binary'] == 'Normal'].sample(n=samples_per_class, random_state=42)
    attack_samples = df[df['Attack_Binary'] == 'Attack'].sample(n=samples_per_class, random_state=42)
    
    balanced_df = pd.concat([normal_samples, attack_samples])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

def simple_robust_preprocessing(df):
    """(Copied from train_model.py) Preprocessing with a reduced feature set."""
    # Ensure 'Attack_Binary' is created if not already present
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

# --- New EDA Specific Functions ---

def run_full_eda():
    """
    Performs comprehensive EDA including class distributions, feature distributions,
    correlation, box plots, and class separation visualizations.
    """
    print("🚀 STARTING COMPREHENSIVE EDA 🚀")

    # --- 1. SETUP & DATA LOADING ---
    csv_path = 'dataset/botnet.csv'
    output_dir = 'eda_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    print(f"All EDA plots will be saved to the '{output_dir}' folder.")

    try:
        df_raw = pd.read_csv(csv_path)
        df_raw.columns = df_raw.columns.str.strip() # Clean column names
        print(f"✅ Raw dataset loaded successfully. Shape: {df_raw.shape}")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{csv_path}'.")
        return

    # Create Attack_Binary column for consistent use
    df_raw['Attack_Binary'] = df_raw['Attack'].apply(lambda x: 'Normal' if x == 'Normal' else 'Attack')

    # --- Initial Split for Training Pool (as in train_model.py) ---
    # This ensures EDA on the same 'training pool' data that would be balanced
    training_pool_df, _ = train_test_split(
        df_raw, test_size=0.1, random_state=42, stratify=df_raw['Attack'] 
    )
    print(f"Created training pool for balancing EDA. Shape: {training_pool_df.shape}")

    # --- 2. CLASS DISTRIBUTION PLOT (BEFORE BALANCING) ---
    print("\nStep 2: Generating Class Distribution Plot (Before Balancing)...")
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Attack_Binary', data=training_pool_df, palette='viridis')
    plt.title('Class Distribution Before Balancing (Training Pool)')
    plt.xlabel('Traffic Type'); plt.ylabel('Count')
    plot_path = os.path.join(output_dir, 'class_distribution_before_balancing.png')
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  ✅ Class distribution (before balancing) plot saved to: {plot_path}")

    # --- 3. DATA BALANCING (RANDOM UNDERSAMPLING) ---
    print("\nStep 3: Performing Data Balancing (Random Undersampling)...")
    df_balanced = create_perfectly_balanced_dataset(training_pool_df)
    print(f"  ✅ Dataset balanced. New shape: {df_balanced.shape}")

    # --- 4. CLASS DISTRIBUTION PLOT (AFTER BALANCING) ---
    print("\nStep 4: Generating Class Distribution Plot (After Balancing)...")
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Attack_Binary', data=df_balanced, palette='viridis')
    plt.title('Class Distribution After Balancing')
    plt.xlabel('Traffic Type'); plt.ylabel('Count')
    plot_path = os.path.join(output_dir, 'class_distribution_after_balancing.png')
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  ✅ Class distribution (after balancing) plot saved to: {plot_path}")

    # --- 5. FEATURE PREPROCESSING (Selection & Scaling) ---
    print("\nStep 5: Applying Feature Selection and Scaling for visualization...")
    X_unscaled, y = simple_robust_preprocessing(df_balanced) # X_unscaled for box plots
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_unscaled.columns)
    print(f"  ✅ Features selected and scaled. Number of features: {len(X_unscaled.columns)}")

    # --- 6. FEATURE DISTRIBUTIONS (POST-PREPROCESSING) ---
    print("\nStep 6: Generating Distribution Plots for Scaled Features (Histograms)...")
    plt.figure(figsize=(15, 10))
    X_scaled_df.hist(bins=30, figsize=(15, 10), layout=(3, 3))
    plt.suptitle("Distribution of Features After Standardization", size=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, 'scaled_feature_distributions.png')
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  ✅ Scaled feature distribution plots saved to: {plot_path}")
    print("     (Observe how all features are now centered around 0 with unit variance)")

    # --- 7. CORRELATION HEATMAP ---
    print("\nStep 7: Generating Correlation Heatmap for Selected Features...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_unscaled.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Matrix of Selected Features (Before Scaling)') # Use unscaled for interpretability here
    plot_path = os.path.join(output_dir, 'correlation_heatmap_selected_features.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  ✅ Correlation heatmap saved to: {plot_path}")

    # --- 8. BOX PLOTS FOR OUTLIER VISUALIZATION ---
    print("\nStep 8: Generating Box Plots for Selected Features (Before Scaling)...")
    num_features = len(X_unscaled.columns)
    rows = (num_features + 2) // 3  # Adjust rows dynamically
    plt.figure(figsize=(18, rows * 6))
    for i, column in enumerate(X_unscaled.columns):
        plt.subplot(rows, 3, i + 1)
        sns.boxplot(y=X_unscaled[column])
        plt.title(f'Box Plot of {column}')
        plt.ylabel('')
    plt.suptitle('Box Plots of Selected Features (Before Scaling) - Outlier Visualization', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plot_path = os.path.join(output_dir, 'box_plots_selected_features_unscaled.png')
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  ✅ Box plots saved to: {plot_path}")

    # --- 9. PAIR PLOT FOR CLASS SEPARATION ---
    print("\nStep 9: Generating Pair Plot for Class Separation (on a sample, this may take a moment)...")
    plot_df = X_scaled_df.copy() # Use scaled features for the pairplot
    plot_df['Attack_Binary'] = y

    # Sample a smaller DataFrame for pair plot to manage memory/time
    sample_size = min(5000, len(plot_df)) # Limit sample size to 5000 for performance
    sample_df = plot_df.sample(n=sample_size, random_state=42)

    sns.pairplot(sample_df, hue='Attack_Binary', palette={'Attack':'#ff6347', 'Normal':'#90ee90'})
    plt.suptitle("Pair Plot of Scaled Features by Class (Sampled)", y=1.02, size=16)
    plot_path = os.path.join(output_dir, 'class_separation_pairplot_scaled.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  ✅ Pair plot saved to: {plot_path}")
    print("     (This visualization helps understand how features interact to separate classes)")

    print("\n🎉 COMPREHENSIVE EDA COMPLETE! Check the 'eda_visualizations' folder.")

if __name__ == '__main__':
    run_full_eda()