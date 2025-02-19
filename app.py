from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
from werkzeug.middleware.proxy_fix import ProxyFix
from routes.payments import payments
from routes.webhooks import webhooks
from routes.ai import ai_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def validate_environment():
    """Validate required environment variables are set."""
    required_vars = {
        'SUPABASE_URL': os.getenv('SUPABASE_URL'),
        'SUPABASE_KEY': os.getenv('SUPABASE_KEY')
    }
    
    missing_vars = [key for key, value in required_vars.items() if not value]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    return True

def create_app():
    logger.info("Starting application initialization...")
    app = Flask(__name__)
    
    # Configure CORS with all necessary origins
    ALLOWED_ORIGINS = [
        "http://localhost:5173",
        "https://diz-nine.vercel.app",
        "https://bck-production-6927.up.railway.app"
    ]
    
    CORS(app, resources={
        r"/api/*": {
            "origins": ALLOWED_ORIGINS,
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-User-ID", "Accept", "Origin"],
            "expose_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True,
            "max_age": 86400
        }
    })
    
    # Add CORS preflight handler
    @app.after_request
    def after_request(response):
        origin = request.headers.get('Origin')
        if origin in ALLOWED_ORIGINS:
            response.headers.add('Access-Control-Allow-Origin', origin)
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-User-ID,Accept,Origin')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            response.headers.add('Access-Control-Max-Age', '86400')
        return response
    
    # Register blueprints
    app.register_blueprint(ai_bp, url_prefix='/api/ai')
    app.register_blueprint(payments, url_prefix='/api/payments')
    app.register_blueprint(webhooks, url_prefix='/api/webhooks')
    
    # Configure ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Health check endpoint - define this first to ensure it's available
    @app.route('/api/health')
    def health_check():
        env_valid = validate_environment()
        status = "healthy" if env_valid else "degraded"
        response = {
            "status": status,
            "environment": "valid" if env_valid else "missing required variables",
            "supabase_url": "configured" if os.getenv('SUPABASE_URL') else "missing",
            "supabase_key": "configured" if os.getenv('SUPABASE_KEY') else "missing"
        }
        return jsonify(response), 200
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000))) 