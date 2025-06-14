import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class StressDetectionModel:
    def __init__(self, app=None):
        self.model = None
        self.scaler = None
        self.features = None
        self.app = app
        self.model_path = os.path.join(app.config['MODEL_FOLDER'], 'stress_model.joblib') if app else 'models/stress_model.joblib'
        self.scaler_path = os.path.join(app.config['MODEL_FOLDER'], 'scaler.joblib') if app else 'models/scaler.joblib'
        self.features_path = os.path.join(app.config['MODEL_FOLDER'], 'features.joblib') if app else 'models/features.joblib'
    
    def preprocess_data(self, df):
        # Ensure required columns exist in the dataset
        target_column = 'Stress_Level'
        required_features = ['Humidity', 'Temperature', 'Step_Count']
        
        # Check if the target column exists
        if target_column not in df.columns:
            # Try case-insensitive matching
            for col in df.columns:
                if col.lower() == target_column.lower():
                    # Rename to the expected format
                    df = df.rename(columns={col: target_column})
                    break
            
            # If still not found
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Check for required features (case-insensitive)
        for feature in required_features:
            if feature not in df.columns:
                # Try case-insensitive matching
                for col in df.columns:
                    if col.lower() == feature.lower():
                        # Rename to the expected format
                        df = df.rename(columns={col: feature})
                        break
                
                # If still not found after case-insensitive matching
                if feature not in df.columns:
                    raise ValueError(f"Required feature '{feature}' not found in dataset")
        
        # Drop any non-numeric columns except the target
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if target_column not in numeric_cols:
            numeric_cols.append(target_column)
        
        df = df[numeric_cols]
        
        # Handle missing values - use mean imputation
        for col in df.columns:
            if col != target_column:
                df[col].fillna(df[col].mean(), inplace=True)
        
        # Drop any remaining rows with NaN
        df = df.dropna()
        
        # Split features and target
        # Make sure to use only the required features for training
        feature_cols = [col for col in required_features if col in df.columns]
        X = df[feature_cols]
        y = df[target_column]
        
        # Store feature names
        self.features = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
    
    def train(self, df):
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names = self.preprocess_data(df)
        
        # Grid search for hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
        
        grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'best_params': grid_search.best_params_
        }
        
        # Save model, scaler, and features
        self.save()
        
        # Get feature importance if linear kernel
        feature_importance = None
        if self.model.kernel == 'linear':
            feature_importance = dict(zip(feature_names, self.model.coef_[0]))
        
        return metrics, feature_importance
    
    def predict(self, input_data):
        if self.model is None:
            self.load()
        
        if self.model is None:
            raise ValueError("Model could not be loaded. Please train the model first.")
            
        if isinstance(input_data, pd.DataFrame):
            # Ensure all required features are present
            missing_features = set(self.features or []) - set(input_data.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Reorder columns to match training data
            input_data = input_data[self.features]
        else:
            # Convert dictionary to DataFrame
            input_data = pd.DataFrame([input_data])
            
            # Check for missing features
            missing_features = set(self.features or []) - set(input_data.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Ensure columns are in the right order
            input_data = input_data[self.features]
        
        # Scale features
        if self.scaler is None:
            raise ValueError("Scaler is not available. Please train the model first.")
            
        scaled_input = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(scaled_input)
        probabilities = self.model.predict_proba(scaled_input)
        
        # Get the classes (handle case where classes_ might not be available)
        classes = getattr(self.model, 'classes_', [0, 1, 2])
        
        # Get the probability of the predicted class
        prediction_probability = []
        for i, pred in enumerate(prediction):
            try:
                pred_idx = list(classes).index(pred)
                prediction_probability.append(probabilities[i][pred_idx])
            except (ValueError, IndexError):
                # Default probability if class is not found or other errors
                prediction_probability.append(1.0)
        
        # Convert to standard Python types to ensure JSON serializability
        prediction_list = prediction.tolist() if hasattr(prediction, 'tolist') else list(map(int, prediction))
        
        # Convert classes to standard list
        try:
            # Try to convert numpy array to list
            if hasattr(classes, 'tolist'):
                classes_list = classes.tolist()
            # If it's already a list, just convert elements to int
            else:
                classes_list = list(map(int, classes))
        except (AttributeError, TypeError):
            # Fallback for any other type
            classes_list = [0, 1, 2]  # Default stress levels
            
        # Return prediction and confidence
        result = {
            'prediction': prediction_list,
            'probability': prediction_probability,
            'classes': classes_list
        }
        
        return result
    
    def save(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.features, self.features_path)
    
    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model file not found. Please train a model first.")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.features = joblib.load(self.features_path)
    
    def is_model_trained(self):
        return os.path.exists(self.model_path)
