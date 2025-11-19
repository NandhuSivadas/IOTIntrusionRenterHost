import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # Needed for initial unseen data split

def perform_eda():
    """
    Performs a full Exploratory Data Analysis on a local IoT dataset file,
    including handling class imbalance and visualizing the result.
    """
    print("🚀 STARTING EXPLORATORY DATA ANALYSIS 🚀")

    # --- 1. SETUP & DATA LOADING ---
    # IMPORTANT: Change this path to the location of your local CSV file.
    csv_path = 'dataset/botnet.csv'
    
    output_dir = 'eda_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to the '{output_dir}' folder.")

    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Dataset loaded successfully from: {csv_path}")
        print(f"Shape of the dataset: {df.shape}")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{csv_path}'. Please update the path and try again.")
        return
        
    # Clean column names to prevent errors from hidden whitespace
    df.columns = df.columns.str.strip()

    # --- 2. BASIC DATA OVERVIEW ---
    print("\n" + "="*50)
    print("SECTION 2: BASIC DATA OVERVIEW")
    print("="*50)
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info (Data Types, Non-Null Counts):")
    df.info()
    print("\nStatistical Summary of Numerical Columns:")
    print(df.describe().T)

    # --- 3. MISSING VALUES & 4. DUPLICATE ROWS ANALYSIS ---
    print("\n" + "="*50)
    print("SECTIONS 3 & 4: DATA QUALITY CHECK")
    print("="*50)
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"Duplicate Rows: {df.duplicated().sum():,}")

    # --- 5. TARGET VARIABLE ANALYSIS (IMBALANCE CHECK) ---
    print("\n" + "="*50)
    print("SECTION 5: IMBALANCE CHECK (BEFORE BALANCING)")
    print("="*50)
    
    if 'Attack' in df.columns:
        df['Attack_Binary'] = df['Attack'].apply(lambda x: 'Normal' if x == 'Normal' else 'Attack')
        class_distribution = df['Attack_Binary'].value_counts()
        print("Original Class Distribution:")
        print(class_distribution)

        plt.figure(figsize=(8, 6))
        ax = sns.countplot(x='Attack_Binary', data=df, palette={'Attack':'#ff6347', 'Normal':'#90ee90'}, order=['Attack', 'Normal'])
        plt.title('Distribution Before Balancing', fontsize=16)
        plt.xlabel('Traffic Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        
        plot_path = os.path.join(output_dir, 'class_distribution_imbalanced.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Imbalanced distribution plot saved to: {plot_path}")
    else:
        print("⚠️ 'Attack' column not found. Skipping imbalance check and subsequent steps.")
        return

    # --- 5.5: HANDLING IMBALANCE (UNDERSAMPLING) ---
    print("\n" + "="*50)
    print("SECTION 5.5: HANDLING IMBALANCE & VISUALIZING AFTER")
    print("="*50)
    
    df_majority = df[df.Attack_Binary == 'Attack']
    df_minority = df[df.Attack_Binary == 'Normal']

    # Undersample the majority class to match the minority class size
    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)

    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Applied Random Undersampling to balance the dataset.")
    print("\nNew Balanced Class Distribution:")
    print(df_balanced['Attack_Binary'].value_counts())

    # Visualize the new balanced distribution
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='Attack_Binary', data=df_balanced, palette={'Attack':'#ff6347', 'Normal':'#90ee90'}, order=['Attack', 'Normal'])
    plt.title('Distribution After Balancing (Undersampling)', fontsize=16)
    plt.xlabel('Traffic Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plot_path_balanced = os.path.join(output_dir, 'class_distribution_balanced.png')
    plt.savefig(plot_path_balanced, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Balanced distribution plot saved to: {plot_path_balanced}")
    
    # --- 6. OUTLIER ANALYSIS ---
    print("\n" + "="*50)
    print("SECTION 6: OUTLIER ANALYSIS (on Balanced Data)")
    print("="*50)
    features_to_check = ['MI_dir_L0.1_mean', 'H_L0.1_mean', 'HH_L0.1_mean', 'HH_L0.1_std', 'HpHp_L0.1_mean']
    
    for feature in features_to_check:
        if feature in df_balanced.columns:
            print(f"\n--- Analysis for '{feature}' ---")
            
            Q1 = df_balanced[feature].quantile(0.25)
            Q3 = df_balanced[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df_balanced[(df_balanced[feature] < lower_bound) | (df_balanced[feature] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df_balanced)) * 100

            print(f"  Outlier Boundaries: (Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f})")
            print(f"  Number of Outliers: {outlier_count:,}")
            print(f"  Percentage of Outliers: {outlier_percentage:.2f}%")

            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df_balanced[feature])
            plt.title(f'Box Plot for {feature} (Balanced Data)')
            plot_path = os.path.join(output_dir, f'boxplot_{feature}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Box plot saved.")

    # <<< START OF NEW CODE: OUTLIER AS ATTACK INDICATOR VISUALIZATION >>>
    print("\n" + "="*50)
    print("SECTION 6.1: VISUALIZING OUTLIERS AS ATTACK INDICATORS")
    print("="*50)

    # Use features that might show good separation for this illustrative plot
    feature1 = 'Packet_Length_mean' 
    feature2 = 'Flow_Duration'

    # Check if these features exist, if not, use alternatives or skip
    # (These specific features might not be in the reduced set, so we use full df for illustration)
    if 'Packet_Length_mean' in df.columns and 'Flow_Duration' in df.columns and 'Attack_Binary' in df.columns:
        # For this specific visualization, let's use a sample of the *original* df
        # to better represent the 'outlier' nature before balancing potentially hides it
        # and to include the 'Normal' vs 'Attack' distinction
        
        # Sample the full dataset for performance and clearer visualization of sparse outliers
        sample_df_for_outliers = df.sample(n=min(50000, len(df)), random_state=42)
        
        plt.figure(figsize=(10, 8))
        
        # Plot Normal Traffic (often forms dense clusters)
        sns.scatterplot(
            x=np.log1p(sample_df_for_outliers[sample_df_for_outliers['Attack_Binary'] == 'Normal'][feature1]), 
            y=np.log1p(sample_df_for_outliers[sample_df_for_outliers['Attack_Binary'] == 'Normal'][feature2]), 
            color='skyblue', label='Normal Traffic', s=20, alpha=0.6
        )
        
        # Plot Attack Traffic (often more scattered and forms outliers/distinct clusters)
        sns.scatterplot(
            x=np.log1p(sample_df_for_outliers[sample_df_for_outliers['Attack_Binary'] == 'Attack'][feature1]), 
            y=np.log1p(sample_df_for_outliers[sample_df_for_outliers['Attack_Binary'] == 'Attack'][feature2]), 
            color='red', label='Attack Traffic (Potential Outliers)', s=20, alpha=0.6
        )
        
        plt.title('Outliers as Attack Indicators: Normal vs. Malicious Traffic', fontsize=16)
        plt.xlabel(f'{feature1} (log scale)', fontsize=12)
        plt.ylabel(f'{feature2} (log scale)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_path = os.path.join(output_dir, 'outliers_as_attack_indicators.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  🖼️ Outliers as Attack Indicators plot saved to: {plot_path}")
        print("  (This plot illustrates how attack patterns often manifest as outliers distinct from normal traffic clusters.)")
    else:
        print(f"  ⚠️ Skipping 'Outliers as Attack Indicators' plot: Required features '{feature1}' or '{feature2}' not found in dataset.")
    # <<< END OF NEW CODE >>>

    # --- 7. FEATURE ANALYSIS ---
    print("\n" + "="*50)
    print("SECTION 7: FEATURE ANALYSIS (on Balanced Data)")
    print("="*50)
    
    high_perf_features = [
        'MI_dir_L0.1_weight', 'MI_dir_L0.1_mean', 'MI_dir_L0.1_variance', 'H_L0.1_weight', 'H_L0.1_mean', 
        'H_L0.1_variance', 'HH_L0.1_weight', 'HH_L0.1_mean', 'HH_L0.1_std', 'HH_L0.1_magnitude', 'HH_L0.1_radius', 
        'HH_L0.1_covariance', 'HpHp_L0.1_weight', 'HpHp_L0.1_mean', 'HpHp_L0.1_std', 'HpHp_L0.1_magnitude', 
        'HpHp_L0.1_radius', 'HpHp_L0.1_covariance'
    ]
    high_perf_features_exist = [f for f in high_perf_features if f in df_balanced.columns]

    if high_perf_features_exist:
        corr_matrix = df_balanced[high_perf_features_exist].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
        plt.title('Correlation Matrix of Key Features (Balanced Data)', fontsize=16)
        plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🔥 Correlation heatmap saved to: {plot_path}")

        print("\nCalculating feature importance...")
        y = df_balanced['Attack_Binary']
        X = df_balanced[high_perf_features_exist].fillna(0)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y_encoded)
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': high_perf_features_exist, 'importance': importances}).sort_values(by='importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
        plt.title('Feature Importance from RandomForest (Balanced Data)', fontsize=16)
        plot_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🌳 Feature importance plot saved to: {plot_path}")

    print("\n🎉 EDA COMPLETE! Check the terminal for analysis and the 'eda_visualizations' folder for plots.")


if __name__ == '__main__':
    perform_eda()