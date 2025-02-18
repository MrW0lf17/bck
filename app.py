from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from werkzeug.middleware.proxy_fix import ProxyFix
from routes.payments import payments
from routes.webhooks import webhooks

# Load environment variables
load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Configure CORS to allow requests from frontend
    CORS(app, resources={
        r"/*": {  # Allow all routes
            "origins": [
                "http://localhost:3000",  # Development
                "http://localhost:3001",  # Alternative Development
                "http://localhost:5173",  # Vite dev server
                "http://127.0.0.1:3000",  # Alternative development
                "http://127.0.0.1:3001",  # Alternative development
                "http://127.0.0.1:5173",  # Alternative Vite dev server
                os.getenv("FRONTEND_URL", "")  # Production
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "Accept", "Stripe-Signature"],
            "supports_credentials": True,
            "expose_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Handle OPTIONS requests for all routes
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Stripe-Signature')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
    
    # Configure proxy settings for production
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    
    # Configure Supabase
    app.config['SUPABASE_URL'] = os.getenv('SUPABASE_URL')
    app.config['SUPABASE_KEY'] = os.getenv('SUPABASE_KEY')
    
    # Configure Redis
    app.config['REDIS_URL'] = os.getenv('REDIS_URL')
    
    # Register blueprints
    from routes.auth import auth_bp
    from routes.images import images_bp
    from routes.ai import ai_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(images_bp, url_prefix='/api/images')
    app.register_blueprint(ai_bp, url_prefix='/api/ai')
    app.register_blueprint(payments)
    app.register_blueprint(webhooks)
    
    @app.route('/api/health')
    def health_check():
        return jsonify({"status": "healthy"})
    
    @app.route('/')
    def hello():
        return {'message': 'DiToolz Pro API is running!'}
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 