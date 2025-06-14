import os
import pandas as pd
import numpy as np
import json
from flask import render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
from app import app
from forms import UploadForm, PredictionForm, SingleDataUploadForm
from models import StressDetectionModel
from utils import create_dynamic_form, get_form_from_features, parse_csv_preview

# Initialize the model
stress_model = StressDetectionModel(app)

@app.route('/')
def index():
    model_trained = stress_model.is_model_trained()
    return render_template('index.html', model_trained=model_trained)

@app.route('/train', methods=['GET', 'POST'])
def train():
    form = UploadForm()
    
    if form.validate_on_submit():
        # Save the uploaded file
        file = form.file.data
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Parse the CSV
            df = pd.read_csv(filepath)
            
            # Clear any previous session data
            session.pop('preview_data', None)
            session.pop('file_path', None)
            session.pop('column_info', None)
            session.pop('metrics', None)
            session.pop('feature_importance', None)
            session.pop('features', None)
            
            # Store preview data in session
            preview_data = parse_csv_preview(df)
            session['preview_data'] = preview_data
            session['file_path'] = filepath
            
            # Get column info for dynamic form creation
            column_info = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    column_info.append({
                        'name': col,
                        'dtype': str(df[col].dtype),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'missing': int(df[col].isna().sum())
                    })
                else:
                    column_info.append({
                        'name': col,
                        'dtype': str(df[col].dtype),
                        'unique_values': df[col].nunique(),
                        'missing': int(df[col].isna().sum())
                    })
            
            # Store column info in session
            session['column_info'] = column_info
            
            # Remove old model files if they exist
            try:
                model_files = [
                    os.path.join(app.config['MODEL_FOLDER'], 'stress_model.joblib'),
                    os.path.join(app.config['MODEL_FOLDER'], 'scaler.joblib'),
                    os.path.join(app.config['MODEL_FOLDER'], 'features.joblib')
                ]
                for file_path in model_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
            except Exception as e:
                app.logger.warning(f"Could not remove old model files: {str(e)}")
            
            # Begin training
            metrics, feature_importance = stress_model.train(df)
            
            # Store metrics in session
            session['metrics'] = metrics
            session['feature_importance'] = feature_importance
            
            # Store features for prediction form
            session['features'] = stress_model.features
            
            flash('Model successfully trained with new data!', 'success')
            return render_template('train.html', 
                                form=form, 
                                preview_data=preview_data,
                                column_info=column_info,
                                metrics=metrics,
                                feature_importance=feature_importance,
                                training_complete=True)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'danger')
            return render_template('train.html', form=form)
    
    # Add a reset option for GET requests with a query parameter
    if request.args.get('reset') == 'true':
        # Clear all session data for training
        session.pop('preview_data', None)
        session.pop('file_path', None)
        session.pop('column_info', None)
        session.pop('metrics', None)
        session.pop('feature_importance', None)
        session.pop('features', None)
        flash('Training data has been reset. You can now upload a new CSV file.', 'info')
        return redirect(url_for('train'))
        
    # If GET request or form not valid
    # Check if we have training results in session
    preview_data = session.get('preview_data')
    column_info = session.get('column_info')
    metrics = session.get('metrics')
    feature_importance = session.get('feature_importance')
    
    return render_template('train.html', 
                          form=form, 
                          preview_data=preview_data,
                          column_info=column_info,
                          metrics=metrics,
                          feature_importance=feature_importance,
                          training_complete=metrics is not None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Check if model is trained
    if not stress_model.is_model_trained():
        flash('Please train a model first.', 'warning')
        return redirect(url_for('train'))
    
    # Load the model if it's not loaded yet
    if stress_model.model is None:
        stress_model.load()
    
    # Get features from session or model
    features = session.get('features')
    if not features:
        stress_model.load()
        features = stress_model.features
    
    # Create dynamic form
    form = get_form_from_features(features)
    file_form = SingleDataUploadForm()
    
    prediction_result = None
    
    # Handle form submission
    if request.method == 'POST':
        if 'submit' in request.form:  # Manual input form submitted
            if form.validate_on_submit():
                # Get input data from form
                input_data = {}
                for field in form:
                    if field.name != 'submit' and field.name != 'csrf_token':
                        input_data[field.name] = float(field.data) if field.data else 0.0
                
                try:
                    # Make prediction
                    result = stress_model.predict(input_data)
                    prediction_result = {
                        'prediction': result['prediction'][0],
                        'probability': result['probability'][0],
                        'input_data': input_data
                    }
                    
                    flash('Prediction successful!', 'success')
                except Exception as e:
                    flash(f'Error making prediction: {str(e)}', 'danger')
        
        elif file_form.validate_on_submit():  # File upload form submitted
            try:
                # Save the uploaded file
                file = file_form.file.data
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Parse the CSV
                df = pd.read_csv(filepath)
                
                # Check if file has only one row
                if len(df) != 1:
                    flash('The uploaded CSV should contain exactly one row of data', 'danger')
                else:
                    # Make prediction
                    result = stress_model.predict(df)
                    prediction_result = {
                        'prediction': result['prediction'][0],
                        'probability': result['probability'][0],
                        'input_data': df.iloc[0].to_dict()
                    }
                    
                    flash('Prediction successful!', 'success')
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'danger')
    
    return render_template('predict.html', form=form, file_form=file_form, 
                          prediction=prediction_result, features=features)



@app.route('/api/model_info', methods=['GET'])
def model_info():
    if not stress_model.is_model_trained():
        return jsonify({'status': 'error', 'message': 'No model trained yet'})
    
    # Load the model if needed
    if stress_model.model is None:
        try:
            stress_model.load()
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    
    # Return model info
    model_params = {}
    if stress_model.model is not None:
        try:
            model_params = stress_model.model.get_params()
        except Exception:
            model_params = {"info": "Parameters not available"}
    
    return jsonify({
        'status': 'success',
        'features': stress_model.features,
        'model_type': type(stress_model.model).__name__ if stress_model.model else "Unknown",
        'model_params': model_params
    })
