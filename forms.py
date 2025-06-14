from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, FloatField, SubmitField, SelectField
from wtforms.validators import DataRequired, Optional

class UploadForm(FlaskForm):
    file = FileField('Upload CSV File', validators=[
        FileRequired(),
        FileAllowed(['csv'], 'CSV files only!')
    ])
    submit = SubmitField('Upload and Train')

class PredictionForm(FlaskForm):
    # This form will be dynamically generated based on the features used in training
    submit = SubmitField('Predict Stress Level')

class SingleDataUploadForm(FlaskForm):
    file = FileField('Upload CSV File with Single Data Point', validators=[
        FileRequired(),
        FileAllowed(['csv'], 'CSV files only!')
    ])
    submit = SubmitField('Upload and Predict')
