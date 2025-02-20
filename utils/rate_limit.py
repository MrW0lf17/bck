from functools import wraps
from flask import request, jsonify
import time
from collections import defaultdict

# Store request timestamps for each user/IP
request_history = defaultdict(list)
RATE_LIMIT = 10  # requests per minute

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get user identifier (IP address or user ID)
        user_id = request.headers.get('X-User-ID') or request.remote_addr
        
        # Get current timestamp
        now = time.time()
        
        # Clean up old requests (older than 1 minute)
        request_history[user_id] = [
            timestamp for timestamp in request_history[user_id]
            if timestamp > now - 60
        ]
        
        # Check if rate limit is exceeded
        if len(request_history[user_id]) >= RATE_LIMIT:
            return jsonify({
                "success": False,
                "error": "Rate limit exceeded. Please try again later."
            }), 429
        
        # Add current request timestamp
        request_history[user_id].append(now)
        
        return f(*args, **kwargs)
    return decorated_function 