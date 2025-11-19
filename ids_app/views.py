import os
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.conf import settings

# --- Load the single model and its supporting files ---
model_dir = os.path.join(settings.BASE_DIR, 'ids_app', 'model')
model, model_columns, label_encoder, scaler = None, None, None, None

try:
    model = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))
    model_columns = joblib.load(os.path.join(model_dir, 'model_columns.pkl'))
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    print("✅ Model and supporting files loaded successfully")
    print(f"Model expects {len(model_columns)} features: {model_columns}")
    print(f"Label encoder classes: {label_encoder.classes_}")
except FileNotFoundError as e:
    print(f"❌ Model loading error: {e}. Please ensure you have run the training script.")
except Exception as e:
    print(f"❌ Unexpected error during model loading: {e}")

# In ids_app/views.py

def preprocess_for_prediction(data):
    """
    Exact same preprocessing as the final training script, with improved error handling.
    """
    print("🔄 Starting prediction preprocessing...")
    
    key_features = [
        'MI_dir_L0.1_mean', 'MI_dir_L0.1_variance',
        'H_L0.1_mean', 'H_L0.1_variance',
        'HH_L0.1_mean', 'HH_L0.1_std',
        'HpHp_L0.1_mean'
    ]
    
    available_features = [feature for feature in key_features if feature in data.columns]
    
    # <<< START OF CHANGE: Improved Error Message >>>
    if not available_features:
        error_message = (
            "The uploaded CSV does not contain any of the required features. "
            f"The model expects at least one of these columns: {key_features}. "
            f"However, the uploaded file only has these columns: {list(data.columns)}."
        )
        raise ValueError(error_message)
    # <<< END OF CHANGE >>>
    
    print(f"Using {len(available_features)} available features: {available_features}")
    
    X = data[available_features].copy()
    
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    for col in model_columns:
        if col not in X.columns:
            X[col] = 0
            print(f"Added missing training column: {col}")
            
    X = X[model_columns]
    
    print(f"Features aligned with training set: {X.shape}")
    
    if scaler:
        print("🔧 Applying feature scaling...")
        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    else:
        raise ValueError("Scaler not found.")
    
    print(f"✅ Preprocessing complete: {X.shape}")
    return X

    
    """
    Exact same preprocessing as the final training script.
    """
    print("🔄 Starting prediction preprocessing...")
    
    # Use the exact same reduced set of 7 features from training.
    key_features = [
        'MI_dir_L0.1_mean', 'MI_dir_L0.1_variance',
        'H_L0.1_mean', 'H_L0.1_variance',
        'HH_L0.1_mean', 'HH_L0.1_std',
        'HpHp_L0.1_mean'
    ]
    
    # Keep only features that exist in the uploaded data
    available_features = [feature for feature in key_features if feature in data.columns]
    
    if not available_features:
        raise ValueError("The uploaded CSV does not contain any of the required features for prediction.")
    
    print(f"Using {len(available_features)} available features: {available_features}")
    
    X = data[available_features].copy()
    
    # Same cleaning as training
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Ensure all columns the model was trained on are present, in the correct order
    for col in model_columns:
        if col not in X.columns:
            X[col] = 0 # Add missing columns with a default value of 0
            print(f"Added missing training column: {col}")
            
    X = X[model_columns] # Enforce column order
    
    print(f"Features aligned with training set: {X.shape}")
    
    # Apply the same scaling
    if scaler:
        print("🔧 Applying feature scaling...")
        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    else:
        raise ValueError("Scaler not found.")
    
    print(f"✅ Preprocessing complete: {X.shape}")
    return X

def index(request):
    """
    Main prediction view, simplified for a single model.
    """
    context = {}
    
    # Check if all required components are loaded
    if not all([model, model_columns, label_encoder, scaler]):
        context['error'] = 'A required model component is missing. Please retrain the model.'
        return render(request, 'ids_app/index.html', context)

    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']

        if not csv_file.name.endswith('.csv'):
            context['error'] = 'Please upload a valid CSV file.'
            return render(request, 'ids_app/index.html', context)

        try:
            print(f"📁 Processing file: {csv_file.name}")
            data = pd.read_csv(csv_file)
            print(f"📊 Data loaded: {data.shape}")
            original_data = data.copy()
            
            # Preprocess data
            X_processed = preprocess_for_prediction(data)
            
            # Make predictions using the single loaded model
            print("🤖 Making predictions with Random Forest...")
            predictions = model.predict(X_processed)
            prediction_probs = model.predict_proba(X_processed)
            
            text_predictions = label_encoder.inverse_transform(predictions)
            
            # Add predictions to original data for display
            original_data['PREDICTION'] = text_predictions
            original_data['CONFIDENCE'] = np.max(prediction_probs, axis=1).round(3)
            
            # --- Start of context preparation ---
            headers_list = original_data.columns.tolist()
            try:
                # Find the 0-based index of the 'PREDICTION' column
                prediction_col_index = headers_list.index('PREDICTION')
            except ValueError:
                prediction_col_index = -1 # Will not match if column isn't found

            # Calculate statistics
            pred_counts = pd.Series(text_predictions).value_counts()
            total_samples = len(text_predictions)
            normal_count = pred_counts.get('Normal', 0)
            attack_count = pred_counts.get('Attack', 0)
            
            context.update({
                'headers': headers_list,
                'results': original_data.values.tolist(),
                'model_used': 'Random Forest',
                'total_samples': total_samples,
                'normal_count': normal_count,
                'attack_count': attack_count,
                'normal_percentage': round((normal_count / total_samples) * 100, 1) if total_samples > 0 else 0,
                'attack_percentage': round((attack_count / total_samples) * 100, 1) if total_samples > 0 else 0,
                'prediction_col_index': prediction_col_index,
            })
            
            print("📊 PREDICTION SUMMARY:")
            print(f"   Total samples: {total_samples}")
            print(f"   Normal predictions: {context['normal_count']} ({context['normal_percentage']}%)")
            print(f"   Attack predictions: {context['attack_count']} ({context['attack_percentage']}%)")
            
        except Exception as e:
            print(f"❌ Unexpected error during prediction: {e}")
            import traceback
            traceback.print_exc()
            context['error'] = f'An unexpected error occurred: {str(e)}'

    return render(request, 'ids_app/index.html', context)