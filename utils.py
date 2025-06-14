import pandas as pd
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SubmitField
from wtforms.validators import DataRequired, Optional

def parse_csv_preview(df, max_rows=5):
    """Extract a preview of CSV data for display"""
    preview = {
        'columns': df.columns.tolist(),
        'data': df.head(max_rows).values.tolist(),
        'shape': df.shape
    }
    return preview

def create_dynamic_form(feature_columns):
    """Dynamically create a form with fields based on feature columns"""
    class DynamicForm(FlaskForm):
        pass
    
    # Add fields to the form
    for column in feature_columns:
        setattr(DynamicForm, column, FloatField(column, validators=[Optional()]))
    
    # Add submit button
    setattr(DynamicForm, 'submit', SubmitField('Predict Stress Level'))
    
    return DynamicForm()

def get_form_from_features(features):
    """Create a form with fields based on feature list"""
    class DynamicForm(FlaskForm):
        pass
    
    # Add fields to the form
    for feature in features:
        setattr(DynamicForm, feature, FloatField(feature, validators=[Optional()]))
    
    # Add submit button
    setattr(DynamicForm, 'submit', SubmitField('Predict Stress Level'))
    
    return DynamicForm()
