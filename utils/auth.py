import jwt
from functools import wraps
from flask import request, jsonify
import os
from dotenv import load_dotenv

load_dotenv()

def verify_auth_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # Check if token is in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # Bearer <token>
            except IndexError:
                return jsonify({'message': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            # Verify the token
            jwt.decode(token, os.getenv('JWT_SECRET_KEY'), algorithms=["HS256"])
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
            
    return decorated_function 