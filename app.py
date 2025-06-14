import os
import logging
import secrets
import datetime
from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = 'uploaded_data'
app.config['MODEL_FOLDER'] = 'models'
app.secret_key = os.environ.get("SESSION_SECRET", secrets.token_hex(16))

# Create necessary folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Add template context processor to provide current year
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

# Record app usage for debugging purposes
@app.before_request
def log_request_info():
    logging.debug('Request URL: %s', request.url)
    logging.debug('Remote IP: %s', request.remote_addr)

# Import routes after app is created to avoid circular imports
from routes import *
