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
    
    # Configure CORS
    CORS(app, 
        resources={
            r"/*": {
                "origins": ["https://diz-nine.vercel.app", "http://localhost:5173"],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"],
                "expose_headers": ["Content-Type"],
                "supports_credentials": True,
                "max_age": 600
            }
        }
    )
    
    # Global after request handler for CORS
    @app.after_request
    def after_request(response):
        origin = request.headers.get('Origin')
        if origin in ["https://diz-nine.vercel.app", "http://localhost:5173"]:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = '*'
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Access-Control-Max-Age'] = '600'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Type'
        
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