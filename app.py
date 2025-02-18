from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
from werkzeug.middleware.proxy_fix import ProxyFix
from routes.payments import payments
from routes.webhooks import webhooks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_app():
    logger.info("Starting application initialization...")
    app = Flask(__name__)
    
    # Health check endpoint - define this first to ensure it's available
    @app.route('/api/health')
    def health_check():
        return jsonify({"status": "healthy"}), 200
    
    # Configure CORS to allow requests from frontend and health checks
    CORS(app, resources={
        r"/*": {  # Allow all routes
            "origins": ["*"],  # Allow all origins for now
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "Accept", "Stripe-Signature"],
        }
    })
    
    # Configure proxy settings for production
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    
    try:
        # Configure Supabase with defaults
        app.config['SUPABASE_URL'] = os.getenv('SUPABASE_URL', '')
        app.config['SUPABASE_KEY'] = os.getenv('SUPABASE_KEY', '')
        
        # Configure Redis with default
        app.config['REDIS_URL'] = os.getenv('REDIS_URL', '')
        
        # Register blueprints
        from routes.auth import auth_bp
        from routes.images import images_bp
        from routes.ai import ai_bp
        
        app.register_blueprint(auth_bp, url_prefix='/api/auth')
        app.register_blueprint(images_bp, url_prefix='/api/images')
        app.register_blueprint(ai_bp, url_prefix='/api/ai')
        app.register_blueprint(payments)
        app.register_blueprint(webhooks)
        logger.info("Successfully registered all blueprints")
    except Exception as e:
        logger.error(f"Error during app initialization: {str(e)}")
        # Continue anyway to allow health checks
    
    @app.route('/')
    def hello():
        return {'message': 'DiToolz Pro API is running!'}
    
    logger.info("Application initialization completed")
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 