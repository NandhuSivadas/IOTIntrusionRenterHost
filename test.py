import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for potential inf/NaN handling if needed

def test_model():
    """
    Loads the trained model and preprocessors, then evaluates
    its performance using the 'test.csv' dataset.
    """
    print("=== MODEL TESTING PHASE ===")

    # Define paths
    model_dir = os.path.join('ids_app', 'model')
    test_csv_path = 'test.csv' # Assuming test.csv is in the root project directory
    output_dir = 'testVisualizations' # Output directory for plots
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    # --- 1. Load Preprocessing Components and Model ---
    print("\nStep 1: Loading model components and the trained model...")
    try:
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        model_columns = joblib.load(os.path.join(model_dir, 'model_columns.pkl'))
        rf_model = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))
        print("✅ All model components loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Required model file not found. Ensure '{model_dir}' exists and contains all .pkl files. Error: {e}")
        return

    # --- 2. Load the Test Dataset ---
    print(f"\nStep 2: Loading the test dataset from '{test_csv_path}'...")
    try:
        test_df = pd.read_csv(test_csv_path)
        print(f"✅ Test dataset loaded successfully. Shape: {test_df.shape}")
    except FileNotFoundError:
        print(f"❌ ERROR: '{test_csv_path}' not found. Please ensure it's in the correct directory.")
        return

    # --- 3. Prepare Test Data ---
    print("\nStep 3: Preparing test data for prediction...")
    
    y_true_labels = test_df['Attack_Binary']
    
    # Ensure raw test data has the same columns as the model was trained on
    # and handle potential NaNs/Infs that might creep into test.csv (though unlikely if saved from preprocessed data)
    X_test_raw = test_df[model_columns].copy()
    X_test_raw = X_test_raw.fillna(0).replace([np.inf, -np.inf], 0)


    X_test_scaled = scaler.transform(X_test_raw)
    y_true_encoded = label_encoder.transform(y_true_labels)
    print("✅ Test data prepared successfully.")

    # --- 4. Make Predictions ---
    print("\nStep 4: Making predictions on the test dataset...")
    y_pred_encoded = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled) # Get probabilities for ROC curve
    print("✅ Predictions made successfully.")

    # --- 5. Evaluate Performance ---
    print("\n" + "="*50)
    print("MODEL PERFORMANCE ON TEST.CSV")
    print("="*50)

    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    print(f"🎯 **Accuracy:** {accuracy:.4f} ({accuracy:.2%})")

    print("\n📊 **Classification Report:**")
    print(classification_report(y_true_encoded, y_pred_encoded, target_names=label_encoder.classes_))

    # <<< --- START: CONFUSION MATRIX CODE --- >>>
    print("\n📉 **Confusion Matrix:**")
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    
    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix on test.csv')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    cm_path = os.path.join(output_dir, 'test_confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"✅ Confusion Matrix plot saved to: {cm_path}")
    
    # Display the plot
    plt.show() 
    # <<< --- END: CONFUSION MATRIX CODE --- >>>


    # <<< --- START: ROC CURVE CODE --- >>>
    print("\nROC Curve:")
    # Determine which class is the positive class (e.g., 'Attack' often is)
    # Assuming 'Attack' is encoded as 0 and 'Normal' as 1 based on previous output
    # You need to explicitly find the index of the positive class
    
    # Find the index for the 'Attack' class in the label encoder's classes_ array
    try:
        attack_class_index = list(label_encoder.classes_).index('Attack')
    except ValueError:
        # Fallback if 'Attack' isn't found, assume 0 or 1 based on common practice
        # For a binary classification, if not 'Attack', then it must be the other class
        # (This handles cases where classes might be e.g., ['Benign', 'Malicious'])
        if label_encoder.classes_[0] == 'Normal': # If 'Normal' is 0, 'Attack' is 1
            attack_class_index = 1
        else: # If 'Attack' is 0, 'Normal' is 1
            attack_class_index = 0

    # Get the probabilities for the 'Attack' class
    # If attack_class_index is 0, then y_pred_proba[:, 0]
    # If attack_class_index is 1, then y_pred_proba[:, 1]
    y_pred_proba_attack = y_pred_proba[:, attack_class_index]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_encoded, y_pred_proba_attack, pos_label=attack_class_index)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve on test.csv')
    plt.legend(loc="lower right")

    # Save the plot
    roc_path = os.path.join(output_dir, 'test_roc_curve.png')
    plt.savefig(roc_path)
    print(f"✅ ROC Curve plot saved to: {roc_path}")

    # Display the plot
    plt.show()
    # <<< --- END: ROC CURVE CODE --- >>>


    print("\n=== Testing Complete ===")

if __name__ == '__main__':
    test_model()