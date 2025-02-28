from flask import Blueprint, request, jsonify, current_app, send_file
import requests
import io
import uuid
from PIL import Image, ImageDraw, ImageFont
import os
from supabase import create_client, Client
import base64
from functools import wraps
import time
from datetime import datetime, timedelta
import jwt
from langdetect import detect
from deep_translator import GoogleTranslator
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO
import json
import textwrap
import subprocess
import replicate
from utils.auth import verify_auth_token
from utils.rate_limit import rate_limit

ai_bp = Blueprint('ai', __name__)

# Initialize Supabase client
try:
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        raise Exception("Missing Supabase configuration")
        
    print(f"Initializing Supabase client with URL: {supabase_url}")
    supabase = create_client(supabase_url, supabase_key)
    
    # Test the connection
    test_response = supabase.table('generated_images').select('*').limit(1).execute()
    print("Supabase connection test successful")
except Exception as e:
    print(f"Error initializing Supabase client: {str(e)}")
    supabase = None

# Together.xyz API configuration
TOGETHER_API_TOKEN = os.getenv('TOGETHER_API_TOKEN')
if not TOGETHER_API_TOKEN:
    raise Exception("Missing TOGETHER_API_TOKEN environment variable")
# Clean the token to remove any whitespace or newlines
TOGETHER_API_TOKEN = TOGETHER_API_TOKEN.strip()
TOGETHER_API_URL = "https://api.together.xyz/v1/images/generations"

# Rate limiting configuration
RATE_LIMIT_ANONYMOUS = 5  # requests per minute for anonymous users
RATE_LIMIT_AUTHENTICATED = 20  # requests per minute for authenticated users
rate_limit_dict = {}

def rate_limit():
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Get user identifier
            auth_header = request.headers.get('Authorization')
            is_authenticated = bool(auth_header and auth_header.startswith('Bearer '))
            user_id = verify_auth_token() if is_authenticated else request.remote_addr
            
            # Get appropriate rate limit
            limit = RATE_LIMIT_AUTHENTICATED if is_authenticated else RATE_LIMIT_ANONYMOUS
            
            # Check rate limit
            now = time.time()
            user_requests = rate_limit_dict.get(user_id, [])
            user_requests = [req for req in user_requests if req > now - 60]  # Keep only last minute
            
            if len(user_requests) >= limit:
                return jsonify({
                    "success": False,
                    "error": f"Rate limit exceeded. Please wait a minute and try again. Limit: {limit} requests per minute."
                }), 429
            
            rate_limit_dict[user_id] = user_requests + [now]
            return f(*args, **kwargs)
        return wrapped
    return decorator

def verify_auth_token():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    try:
        payload = jwt.decode(token, os.getenv('JWT_SECRET'), algorithms=['HS256'])
        return payload.get('user_id')
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def make_api_request_with_retries(url, headers, data, max_retries=3, timeout=60):
    """Helper function to make API requests with retries and various connection configurations"""
    last_error = None
    
    # List of configurations to try
    configs = [
        {"verify": True, "proxies": None},  # Try direct connection first
        {"verify": False, "proxies": None},  # Try without SSL verification
        {"verify": True, "proxies": {}},     # Try with empty proxies
        {"verify": False, "proxies": {}}     # Try without SSL and empty proxies
    ]
    
    for config in configs:
        for attempt in range(max_retries):
            try:
                print(f"Attempting request with config: {config}, attempt {attempt + 1}/{max_retries}")
                response = requests.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=timeout,
                    **config
                )
                if response.ok:
                    return response
                print(f"Request failed with status {response.status_code}")
            except requests.exceptions.SSLError as e:
                print(f"SSL Error: {str(e)}")
                last_error = e
            except requests.exceptions.ProxyError as e:
                print(f"Proxy Error: {str(e)}")
                last_error = e
            except requests.exceptions.ConnectionError as e:
                print(f"Connection Error: {str(e)}")
                last_error = e
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                last_error = e
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    raise Exception(f"All connection attempts failed: {str(last_error)}")

def query_together(prompt, params=None):
    try:
        print(f"Starting image generation with prompt: {prompt}")
        
        # Set default parameters for FLUX.1-schnell-Free
        default_params = {
            "model": "black-forest-labs/FLUX.1-schnell-Free",
            "prompt": prompt,
            "steps": 4,  # Reduced steps for faster generation
            "n": 1,
            "height": 1024,
            "width": 1024,
            "guidance": 3.5,  # Optimized for FLUX model
            "output_format": "jpeg",
            "response_format": "url"  # Keep this to maintain compatibility
        }
        
        if params:
            default_params.update(params)
            
        print(f"Using parameters: {json.dumps(default_params, indent=2)}")
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {TOGETHER_API_TOKEN}"
        }
        
        response = make_api_request_with_retries(
            TOGETHER_API_URL,
            headers=headers,
            data=default_params,
            timeout=60
        )
        
        print(f"Response status: {response.status_code}")
        result = response.json()
        print("Successfully received response from API")
        
        if 'data' in result and len(result['data']) > 0 and 'url' in result['data'][0]:
            image_url = result['data'][0]['url']
            print(f"Got image URL: {image_url}")
            
            # Download the image with retry logic
            for config in [
                {"verify": True, "proxies": None},
                {"verify": False, "proxies": None},
                {"verify": True, "proxies": {}},
                {"verify": False, "proxies": {}}
            ]:
                try:
                    img_response = requests.get(
                        image_url,
                        timeout=30,
                        **config
                    )
                    if img_response.ok:
                        break
                except Exception as e:
                    print(f"Image download attempt failed with config {config}: {str(e)}")
                    continue
            
            if not img_response.ok:
                raise Exception("Failed to download generated image after all attempts")
            
            image_base64 = base64.b64encode(img_response.content).decode('utf-8')
            return image_base64
        else:
            print(f"Unexpected API response format: {result}")
            raise Exception("No image URL in response")
            
    except Exception as e:
        print(f"Error in query_together: {str(e)}")
        raise Exception(f"Image generation failed: {str(e)}")

def query_together_translation(text, to_lang='english'):
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {TOGETHER_API_TOKEN}"
        }
        
        # Simple direct prompt for translation
        data = {
            "model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a translator. Translate any text to English. Only return the translation, nothing else."
                },
                {
                    "role": "user",
                    "content": f"Translate this to English: {text}"
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1,
            "top_p": 0.95,
            "context_length_exceeded_behavior": "error"
        }
        
        print(f"Making translation request for text: {text}")
        
        response = requests.post(
            url="https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if not response.ok:
            print(f"Translation API error: {response.status_code} - {response.text}")
            raise ValueError(f"Translation API error: {response.status_code}")
            
        response_data = response.json()
        print(f"Translation API response: {response_data}")
            
        if 'choices' in response_data and len(response_data['choices']) > 0:
            translated_text = response_data['choices'][0]['message']['content'].strip()
            # Clean up any potential prefixes or explanations
            translated_text = translated_text.replace("Translation:", "").strip()
            translated_text = translated_text.split('\n')[0].strip()
            
            if not translated_text:
                raise ValueError("Empty translation result")
                
            # Verify that the translation is different from the input and contains English characters
            if translated_text.lower() == text.lower() or not any(c.isascii() for c in translated_text):
                # Try one more time with a more forceful prompt
                data["messages"][1]["content"] = f"Translate this text to English (do not return the original text): {text}"
                retry_response = requests.post(
                    url="https://api.together.xyz/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if retry_response.ok:
                    retry_data = retry_response.json()
                    if 'choices' in retry_data and len(retry_data['choices']) > 0:
                        translated_text = retry_data['choices'][0]['message']['content'].strip()
                        translated_text = translated_text.replace("Translation:", "").strip()
                        translated_text = translated_text.split('\n')[0].strip()
                        
                        # Final verification
                        if translated_text.lower() == text.lower() or not any(c.isascii() for c in translated_text):
                            # Try Google Translate as a fallback
                            try:
                                translator = GoogleTranslator(source='auto', target='en')
                                translated_text = translator.translate(text)
                                if translated_text and translated_text.lower() != text.lower():
                                    return {
                                        "translatedText": translated_text,
                                        "to": "english",
                                        "success": True
                                    }
                            except:
                                pass
                            raise ValueError("Translation failed to produce English text")
                else:
                    raise ValueError("Translation retry failed")
                
            return {
                "translatedText": translated_text,
                "to": "english",
                "success": True
            }
        else:
            print(f"Unexpected API response format: {response_data}")
            raise ValueError("No translation in response")
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        # Try Google Translate as a last resort
        try:
            translator = GoogleTranslator(source='auto', target='en')
            translated_text = translator.translate(text)
            if translated_text and translated_text.lower() != text.lower():
                return {
                    "translatedText": translated_text,
                    "to": "english",
                    "success": True
                }
        except:
            pass
        raise Exception(f"Translation failed: {str(e)}")

@ai_bp.route('/detect-language', methods=['GET', 'POST', 'OPTIONS'])
def detect_language():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Try to get text from either POST JSON data or GET query parameter
        text = None
        if request.method == 'POST':
            data = request.get_json(silent=True)  # silent=True prevents error if JSON is invalid
            if data:
                text = data.get('text')
        if not text and request.args:
            text = request.args.get('text')
            
        if not text:
            return jsonify({
                "language": "en",
                "error": "No text provided, defaulting to English"
            }), 200
            
        try:
            language = detect(text)
            return jsonify({
                "language": language,
                "text": text,
                "success": True
            }), 200
        except Exception as lang_error:
            print(f"Language detection error: {str(lang_error)}")
            return jsonify({
                "language": "en",
                "text": text,
                "success": False,
                "error": "Language detection failed, defaulting to English"
            }), 200
        
    except Exception as e:
        print(f"Error in detect-language endpoint: {str(e)}")
        return jsonify({
            "language": "en",
            "success": False,
            "error": f"Error processing request: {str(e)}"
        }), 200

@ai_bp.route('/translate', methods=['POST', 'OPTIONS'])
def translate():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        try:
            # Always translate to English
            translation_result = query_together_translation(text, to_lang='english')
            
            if not translation_result or not isinstance(translation_result, dict):
                return jsonify({
                    "error": "Invalid translation response",
                    "translatedText": text,  # Return original text as fallback
                    "to": "english"
                }), 200
            
            return jsonify(translation_result), 200
            
        except Exception as translation_error:
            print(f"Translation error: {str(translation_error)}")
            return jsonify({
                "error": f"Translation failed: {str(translation_error)}",
                "translatedText": text,  # Return original text as fallback
                "to": "english"
            }), 200
            
    except Exception as e:
        print(f"Translation route error: {str(e)}")
        return jsonify({
            "error": str(e),
            "translatedText": text if text else "",
            "to": "english"
        }), 200

@ai_bp.route('/generate', methods=['POST', 'OPTIONS'])
@rate_limit()
def generate_image():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Log request details
        print("New image generation request received")
        print(f"Headers: {dict(request.headers)}")
        
        user_id = verify_auth_token()
        print(f"Authenticated user_id: {user_id}")
        
        data = request.get_json()
        prompt = data.get('prompt')
        print(f"Received prompt: {prompt}")
        
        if not prompt:
            print("Error: No prompt provided")
            return jsonify({
                "success": False,
                "error": "No prompt provided"
            }), 400
        
        try:
            # Try to detect language, but don't fail if it doesn't work
            try:
                detected_lang = detect(prompt)
                print(f"Detected language: {detected_lang}")
            except Exception as lang_error:
                detected_lang = "en"
                print(f"Language detection failed: {str(lang_error)}, defaulting to English")
            
            # Log Together API token status (masked)
            token_status = "Present" if TOGETHER_API_TOKEN else "Missing"
            print(f"Together API token status: {token_status}")
            
            # Generate image using Together.xyz
            print("Calling Together API for image generation...")
            image_base64 = query_together(prompt)
            
            if not image_base64:
                print("Error: No image data received from Together API")
                return jsonify({
                    "success": False,
                    "error": "Failed to generate image"
                }), 500
            
            print("Successfully received image data from Together API")
            
            try:
                # Convert base64 to bytes
                image_data = base64.b64decode(image_base64)
                
                # Generate a unique filename with timestamp and sanitized prompt
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                # Use the translated prompt for the filename
                sanitized_prompt = ''.join(c if c.isalnum() else '_' for c in prompt[:50])
                filename = f"generated_{timestamp}_{sanitized_prompt}_{uuid.uuid4()}.jpg"
                
                print(f"Attempting to save image with filename: {filename}")
                
                # Upload to Supabase Storage with retry logic
                max_retries = 3
                storage_success = False
                public_url = None
                
                for attempt in range(max_retries):
                    try:
                        print(f"Storage upload attempt {attempt + 1}")
                        print(f"Uploading to bucket: generated-images, filename: {filename}")
                        print(f"Image data size: {len(image_data)} bytes")
                        
                        storage_response = supabase.storage.from_('generated-images').upload(
                            filename,
                            image_data,
                            {
                                'content-type': 'image/jpeg',
                                'cache-control': 'public, max-age=31536000'
                            }
                        )
                        
                        print(f"Storage response: {storage_response}")
                        
                        if storage_response:
                            print(f"Successfully uploaded image on attempt {attempt + 1}")
                            storage_success = True
                            
                            # Get public URL
                            print("Getting public URL...")
                            public_url_response = supabase.storage.from_('generated-images').get_public_url(filename)
                            print(f"Public URL response: {public_url_response}")
                            
                            if isinstance(public_url_response, dict):
                                public_url = public_url_response.get('publicUrl')
                            else:
                                public_url = public_url_response
                                
                            if not public_url:
                                raise Exception("Failed to get public URL from response")
                                
                            print(f"Generated public URL: {public_url}")
                            
                            # Verify the image is accessible
                            print("Verifying image accessibility...")
                            verify_response = requests.head(public_url, timeout=10)
                            print(f"Verify response status: {verify_response.status_code}")
                            
                            if not verify_response.ok:
                                raise Exception(f"Uploaded image is not accessible: {verify_response.status_code}")
                                
                            print("Image URL verified as accessible")
                            break
                    except Exception as upload_error:
                        print(f"Upload attempt {attempt + 1} failed with error: {str(upload_error)}")
                        print(f"Error type: {type(upload_error)}")
                        if hasattr(upload_error, 'response'):
                            print(f"Response content: {upload_error.response.content}")
                        if attempt == max_retries - 1:
                            raise upload_error
                        time.sleep(1)  # Wait before retrying
                
                if not storage_success or not public_url:
                    raise Exception("Failed to upload image to storage")
                
                # Save metadata to database with retry logic
                metadata = {
                    'id': str(uuid.uuid4()),
                    'prompt': prompt,  # Store the original prompt
                    'image_url': public_url,
                    'created_at': datetime.utcnow().isoformat(),
                    'user_id': user_id if user_id else None
                }
                
                print(f"Attempting to save metadata to database: {metadata}")
                
                for attempt in range(max_retries):
                    try:
                        print(f"Database save attempt {attempt + 1}")
                        
                        # First, verify we can connect to the database
                        test_query = supabase.table('generated_images').select('*').limit(1).execute()
                        print("Database connection verified")
                        
                        # Attempt the insert
                        db_response = supabase.table('generated_images').insert(metadata).execute()
                        print(f"Database response: {db_response}")
                        
                        if not db_response:
                            print("No response from database insert")
                            raise Exception("No response from database insert")
                            
                        if not hasattr(db_response, 'data') or not db_response.data:
                            print("No data in database response")
                            raise Exception("No data in database response")
                        
                        saved_data = db_response.data[0]
                        print(f"Successfully saved to database: {saved_data}")
                        
                        # Verify the save by fetching the record
                        verify_response = supabase.table('generated_images').select('*').eq('id', metadata['id']).execute()
                        if not verify_response or not verify_response.data or not verify_response.data[0]:
                            print("Failed to verify saved data")
                            print(f"Verify response: {verify_response}")
                            raise Exception("Failed to verify saved data")
                            
                        print(f"Verified saved data: {verify_response.data[0]}")
                        break
                    except Exception as db_error:
                        error_msg = str(db_error)
                        print(f"Database save attempt {attempt + 1} failed with error: {error_msg}")
                        
                        if hasattr(db_error, 'response'):
                            print(f"Error response: {db_error.response.text if hasattr(db_error.response, 'text') else db_error.response}")
                        
                        if attempt == max_retries - 1:
                            raise Exception(f"Database save failed after {max_retries} attempts: {error_msg}")
                        
                        time.sleep(1)  # Wait before retrying
                
                return jsonify({
                    "success": True,
                    "message": "Image generated and saved successfully",
                    "image_url": f"data:image/jpeg;base64,{image_base64}",  # For immediate display
                    "stored_url": public_url,
                    "metadata": metadata
                }), 200
                
            except Exception as storage_error:
                print(f"Storage/Database error: {str(storage_error)}")
                # If storage/database fails, still return the base64 image
                return jsonify({
                    "success": True,
                    "message": "Image generated but storage failed",
                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                    "error_details": str(storage_error)
                }), 200
        
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    except Exception as e:
        print(f"Error in generate endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400

@ai_bp.route('/text-to-video', methods=['POST', 'OPTIONS'])
@rate_limit()
def text_to_video():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if len(text) > 500:
            return jsonify({"error": "Text is too long. Maximum 500 characters allowed."}), 400

        # Create frames using OpenCV
        width, height = 1280, 720
        fps = 30
        duration = 5  # seconds
        frames = []
        
        # Create frames with animated text
        font_scale = 2
        font_thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate text size and position
        lines = textwrap.wrap(text, width=30)
        line_height = 60
        
        for frame_idx in range(fps * duration):
            # Create a blank frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add gradient background
            for i in range(height):
                alpha = i / height
                frame[i, :] = [int(64 * alpha), int(128 * alpha), int(255 * alpha)]
            
            # Calculate animation offset based on frame
            offset = int(20 * np.sin(2 * np.pi * frame_idx / (fps * 2)))
            
            # Draw each line of text
            y = height // 4
            for line in lines:
                text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                x = (width - text_size[0]) // 2
                y += line_height + offset
                
                # Add shadow effect
                cv2.putText(frame, line, (x+2, y+2), font, font_scale, (0, 0, 0), font_thickness)
                cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), font_thickness)
            
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frames.append(base64.b64encode(buffer).decode('utf-8'))

        return jsonify({
            "success": True,
            "frames": frames,
            "fps": fps,
            "message": "Frames generated successfully!"
        }), 200

    except Exception as e:
        print(f"Error in text-to-video: {str(e)}")
        return jsonify({"error": str(e)}), 400

@ai_bp.route('/image-to-video', methods=['POST', 'OPTIONS'])
@rate_limit()
def image_to_video():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Get the name field from form data
        name = request.form.get('name')
        if not name:
            return jsonify({"error": "Name field is required"}), 400

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'webp'}
        if '.' not in image_file.filename or \
           image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({"error": "Invalid file type. Allowed types: PNG, JPG, JPEG, WEBP"}), 400
        
        # Check file size (10MB limit)
        if len(image_file.read()) > 10 * 1024 * 1024:  # 10MB in bytes
            return jsonify({"error": "File size too large. Maximum size is 10MB"}), 400
        
        # Reset file pointer after reading
        image_file.seek(0)
        
        # Read the input image
        image_array = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
            
        # Get and validate image dimensions
        height, width = image.shape[:2]
        max_dimension = 1920
        if width > max_dimension or height > max_dimension:
            # Resize image while maintaining aspect ratio
            scale = max_dimension / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            
        fps = 30
        duration = 5  # seconds
        frames = []

        # Create a zoom and pan effect
        for frame_idx in range(fps * duration):
            # Calculate zoom factor (1.0 to 1.2 and back)
            time = frame_idx / (fps * duration)
            zoom = 1.0 + 0.2 * np.sin(2 * np.pi * time)
            
            # Calculate pan offset
            offset_x = int(50 * np.sin(2 * np.pi * time))
            offset_y = int(50 * np.cos(2 * np.pi * time))
            
            # Create transformation matrix
            M = cv2.getRotationMatrix2D((width/2, height/2), 0, zoom)
            M[0,2] += offset_x
            M[1,2] += offset_y
            
            # Apply transformation
            frame = cv2.warpAffine(image, M, (width, height))
            
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frames.append(base64.b64encode(buffer).decode('utf-8'))

        return jsonify({
            "success": True,
            "frames": frames,
            "fps": fps,
            "message": "Frames generated successfully!"
        }), 200

    except Exception as e:
        print(f"Error in image-to-video: {str(e)}")
        return jsonify({"error": str(e)}), 400

@ai_bp.route('/chat', methods=['POST'])
@rate_limit()
def chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        message = data['message']
        
        # Configure the chat request with DIZ bot identity
        system_prompt = """You are DIZ bot, a helpful and friendly AI assistant. You should:
1. Always identify yourself as DIZ bot
2. Be concise and clear in your responses
3. Be helpful and friendly while maintaining a professional tone
4. Focus on providing accurate and relevant information
5. If you're not sure about something, be honest about it

Remember to stay in character as DIZ bot throughout the conversation."""

        user_prompt = f"{message}"
        
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\nDIZ bot: ",
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1,
            "stop": ["User:", "\nUser:", "System:", "\nSystem:"]
        }
        
        response = requests.post(
            "https://api.together.xyz/v1/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            return jsonify({'error': 'Failed to get AI response'}), 500
            
        response_data = response.json()
        ai_response = response_data['choices'][0]['text'].strip()
        
        return jsonify({
            'response': ai_response,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def base64_to_numpy(base64_str):
    """Convert base64 image to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Decode base64 to bytes
        img_data = base64.b64decode(base64_str)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        raise Exception(f"Failed to convert base64 to image: {str(e)}")

def numpy_to_base64(img_array):
    """Convert numpy array to base64 string."""
    try:
        # Encode image to jpg format
        _, buffer = cv2.imencode('.jpg', img_array)
        # Convert to base64
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        raise Exception(f"Failed to convert image to base64: {str(e)}")

def blend_images(original, generated, mask):
    """Blend original and generated images using the mask."""
    try:
        # Ensure all images have same dimensions
        if original.shape != generated.shape or original.shape[:2] != mask.shape[:2]:
            generated = cv2.resize(generated, (original.shape[1], original.shape[0]))
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        
        # Normalize mask to range [0, 1]
        mask_norm = mask.astype(float) / 255
        
        # Add channel dimension to mask if needed
        if len(mask_norm.shape) == 2:
            mask_norm = np.expand_dims(mask_norm, axis=-1)
        
        # Blend images
        result = original * (1 - mask_norm) + generated * mask_norm
        return result.astype(np.uint8)
    except Exception as e:
        raise Exception(f"Failed to blend images: {str(e)}")

@ai_bp.route('/enhance', methods=['POST', 'OPTIONS'])
@rate_limit()
def enhance_image():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        user_id = verify_auth_token()
        data = request.get_json()
        
        # Get image and settings from request
        image_data = data.get('image')
        settings = data.get('settings', {
            'enhance_level': 'medium',
            'style': 'natural',
            'focus': 'overall'
        })
        
        if not image_data:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400

        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        # Convert base64 to image for processing
        try:
            image_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (max 1024x1024)
            max_size = 1024
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert back to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as img_error:
            print(f"Image processing error: {str(img_error)}")
            return jsonify({
                "success": False,
                "error": f"Image processing failed: {str(img_error)}"
            }), 400

        # Prepare the API request
        payload = {
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "prompt": "enhance this photo, same composition, improve quality, maintain original colors and composition",
            "negative_prompt": "blur, pixelated, low quality, distorted, different composition, different colors",
            "input_image": image_data,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "strength": 0.3
        }
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {TOGETHER_API_TOKEN}"
        }

        try:
            print("Making API request to Together.xyz...")
            response = requests.post(
                "https://api.together.xyz/inference/stabilityai/stable-diffusion-xl-base-1.0/img2img",
                json=payload,
                headers=headers,
                timeout=60  # Increased timeout for image processing
            )
            
            print(f"API Response Status: {response.status_code}")
            print(f"API Response Headers: {response.headers}")
            
            if not response.ok:
                error_detail = response.text
                print(f"API error response: {error_detail}")
                return jsonify({
                    "success": False,
                    "error": f"API request failed: {error_detail}"
                }), response.status_code

            response_data = response.json()
            print("API response received successfully")

            if 'images' in response_data and response_data['images']:
                enhanced_image_base64 = response_data['images'][0]
                return jsonify({
                    "success": True,
                    "result": f"data:image/jpeg;base64,{enhanced_image_base64}"
                }), 200
            else:
                print(f"Unexpected API response format: {response_data}")
                return jsonify({
                    "success": False,
                    "error": "No enhanced image in API response"
                }), 500
            
        except requests.exceptions.Timeout:
            print("API request timed out")
            return jsonify({
                "success": False,
                "error": "Enhancement request timed out"
            }), 504
        except requests.exceptions.RequestException as api_error:
            print(f"API request error: {str(api_error)}")
            return jsonify({
                "success": False,
                "error": f"Enhancement request failed: {str(api_error)}"
            }), 500
        except Exception as api_error:
            print(f"Unexpected API error: {str(api_error)}")
            return jsonify({
                "success": False,
                "error": f"Enhancement failed: {str(api_error)}"
            }), 500
        
    except Exception as e:
        print(f"General error in enhance_image: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400 