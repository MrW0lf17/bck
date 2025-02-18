from flask import Blueprint, request, jsonify
from supabase import create_client, Client
import os

auth_bp = Blueprint('auth', __name__)

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv('SUPABASE_URL', ''),
    os.getenv('SUPABASE_KEY', '')
)

@auth_bp.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        # Sign up user with Supabase
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        
        if response.user:
            # Initialize user data
            supabase.table('users').insert({
                'id': response.user.id,
                'email': email,
                'created_at': 'now()',
                'updated_at': 'now()'
            }).execute()
        
        return jsonify({
            "success": True,
            "message": "User created successfully",
            "user": response.user
        }), 201
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@auth_bp.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        print(f"Attempting login for email: {email}")  # Debug log
        
        # Sign in user with Supabase
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        return jsonify({
            "success": True,
            "message": "Login successful",
            "session": response.session,
            "user": response.user
        }), 200
        
    except Exception as e:
        print(f"Login error: {str(e)}")  # Debug log
        return jsonify({
            "success": False,
            "error": str(e)
        }), 401

@auth_bp.route('/logout', methods=['POST'])
def logout():
    try:
        # Get the session token from the Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            # Invalidate the session
            supabase.auth.invalidate_session(token)
        
        # Sign out user from Supabase
        supabase.auth.sign_out()
        
        return jsonify({
            "success": True,
            "message": "Logout successful"
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@auth_bp.route('/test-login', methods=['POST'])
def test_login():
    try:
        # Authenticate with test credentials
        response = supabase.auth.sign_in_with_password({
            'email': 'test@example.com',
            'password': 'test123'
        })

        if not response.user:
            return jsonify({"error": "Authentication failed"}), 401

        # Return user data and session
        return jsonify({
            "user": {
                "id": response.user.id,
                "email": response.user.email,
                "isPremium": False
            },
            "session": {
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token
            }
        })

    except Exception as e:
        print(f"Error in test login: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/google-login', methods=['POST'])
def google_login():
    try:
        data = request.get_json()
        id_token = data.get('id_token')
        
        if not id_token:
            return jsonify({"error": "No ID token provided"}), 400

        # Verify the token with Google
        response = supabase.auth.sign_in_with_id_token({
            'provider': 'google',
            'token': id_token
        })

        if not response.user:
            return jsonify({"error": "Authentication failed"}), 401

        # Return user data and session
        return jsonify({
            "user": {
                "id": response.user.id,
                "email": response.user.email,
                "isPremium": False
            },
            "session": {
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token
            }
        })

    except Exception as e:
        print(f"Error in Google login: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500