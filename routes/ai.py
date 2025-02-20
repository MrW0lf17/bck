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
RATE_LIMIT = 10  # requests per minute
rate_limit_dict = {}

def rate_limit():
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Get user identifier (IP address or user ID)
            user_id = request.headers.get('X-User-ID') or request.remote_addr
            
            # Check rate limit
            now = time.time()
            user_requests = rate_limit_dict.get(user_id, [])
            user_requests = [req for req in user_requests if req > now - 60]  # Keep only last minute
            
            if len(user_requests) >= RATE_LIMIT:
                return jsonify({
                    "success": False,
                    "error": "Rate limit exceeded. Please try again later."
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
    
    # Verify API token is present and valid
    if not TOGETHER_API_TOKEN or len(TOGETHER_API_TOKEN.strip()) < 10:
        raise Exception("Invalid or missing API token")
    
    print(f"Making API request to {url}")
    print(f"Request data: {json.dumps(data, indent=2)}")
    print(f"Headers (excluding auth): {json.dumps({k:v for k,v in headers.items() if k != 'authorization'}, indent=2)}")
    
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
                print(f"\nAttempt {attempt + 1}/{max_retries} with config: {config}")
                
                # Create session with specific config
                session = requests.Session()
                session.verify = config["verify"]
                if config["proxies"] is not None:
                    session.proxies = config["proxies"]
                
                # Make the request
                response = session.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=timeout
                )
                
                print(f"Response status code: {response.status_code}")
                
                # Try to parse response content
                try:
                    response_data = response.json()
                    print(f"Response data: {json.dumps(response_data, indent=2)}")
                except:
                    print(f"Raw response content: {response.text[:500]}...")
                
                if response.ok:
                    return response
                    
                print(f"Request failed with status {response.status_code}")
                if response.status_code == 401:
                    raise Exception("API authentication failed. Please check your API token.")
                elif response.status_code == 403:
                    raise Exception("API access forbidden. Please check your API permissions.")
                elif response.status_code == 429:
                    raise Exception("API rate limit exceeded. Please try again later.")
                
            except requests.exceptions.SSLError as e:
                print(f"SSL Error: {str(e)}")
                last_error = e
            except requests.exceptions.ProxyError as e:
                print(f"Proxy Error: {str(e)}")
                last_error = e
            except requests.exceptions.ConnectionError as e:
                print(f"Connection Error: {str(e)}")
                last_error = e
            except requests.exceptions.Timeout as e:
                print(f"Timeout Error: {str(e)}")
                last_error = e
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                print(f"Error type: {type(e)}")
                last_error = e
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Waiting {wait_time} seconds before next attempt...")
                time.sleep(wait_time)
    
    error_msg = str(last_error) if last_error else "Unknown error"
    print(f"All connection attempts failed. Last error: {error_msg}")
    raise Exception(f"All connection attempts failed: {error_msg}")

def query_together(prompt, params=None):
    try:
        print(f"Starting image generation with prompt: {prompt}")
        
        # First, check if the model is available
        try:
            check_url = "https://api.together.xyz/v1/models"
            check_headers = {
                "accept": "application/json",
                "authorization": f"Bearer {TOGETHER_API_TOKEN}"
            }
            
            print("Checking model availability...")
            check_response = requests.get(check_url, headers=check_headers, timeout=30)
            
            if check_response.ok:
                models = check_response.json()
                # Check for the dev-lora model instead
                model_found = any(m.get('name') == "black-forest-labs/FLUX.1-dev-lora" for m in models)
                if not model_found:
                    print("FLUX LoRA model not found in available models")
                    # Fall back to Stable Diffusion XL
                    print("Falling back to Stable Diffusion XL")
                    model = "stabilityai/stable-diffusion-xl-base-1.0"
                    steps = 30
                    guidance = 7.5
                else:
                    print("FLUX LoRA model is available")
                    model = "black-forest-labs/FLUX.1-dev-lora"
                    steps = 28  # As recommended in docs
                    guidance = 7.5
            else:
                print("Failed to check model availability")
                # Default to Stable Diffusion XL
                model = "stabilityai/stable-diffusion-xl-base-1.0"
                steps = 30
                guidance = 7.5
        except Exception as e:
            print(f"Error checking model availability: {str(e)}")
            # Default to Stable Diffusion XL
            model = "stabilityai/stable-diffusion-xl-base-1.0"
            steps = 30
            guidance = 7.5
        
        default_params = {
            "model": model,
            "prompt": prompt,
            "steps": steps,
            "n": 1,
            "height": 768,  # Standard size from docs
            "width": 1024,  # Standard size from docs
            "guidance": guidance,
            "output_format": "url",  # As shown in docs
            "response_format": "url"  # As shown in docs
        }
        
        # Only add LoRA if using FLUX model
        if model == "black-forest-labs/FLUX.1-dev-lora":
            default_params["image_loras"] = [
                {
                    "path": "https://huggingface.co/XLabs-AI/flux-RealismLora",
                    "scale": 0.8
                },
                {
                    "path": "https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-add-details",
                    "scale": 0.8
                }
            ]
        
        if params:
            default_params.update(params)
            
        print(f"Using parameters: {json.dumps(default_params, indent=2)}")
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {TOGETHER_API_TOKEN}"
        }
        
        try:
            response = make_api_request_with_retries(
                TOGETHER_API_URL,
                headers=headers,
                data=default_params,
                timeout=60
            )
            
            print(f"Response status: {response.status_code}")
            result = response.json()
            print(f"API Response: {json.dumps(result, indent=2)}")  # Log full response for debugging
            
            if not response.ok:
                error_msg = result.get('error', {}).get('message', str(result))
                print(f"API Error: {error_msg}")
                raise Exception(f"API Error: {error_msg}")
            
            if 'data' not in result or not result['data']:
                raise Exception("No data in API response")
                
            if not isinstance(result['data'], list) or not result['data']:
                raise Exception("Invalid data format in API response")
                
            if 'url' not in result['data'][0]:
                raise Exception("No image URL in API response")
                
            image_url = result['data'][0]['url']
            print(f"Got image URL: {image_url}")
            
            # Download the image with retry logic
            img_response = None
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
            
            if not img_response or not img_response.ok:
                raise Exception("Failed to download generated image after all attempts")
            
            image_base64 = base64.b64encode(img_response.content).decode('utf-8')
            return image_base64
            
        except Exception as api_error:
            print(f"API Error in query_together: {str(api_error)}")
            raise Exception(f"API Error: {str(api_error)}")
            
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
        user_id = verify_auth_token()
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({
                "success": False,
                "error": "No prompt provided"
            }), 400
        
        try:
            # Try to detect language, but don't fail if it doesn't work
            try:
                detected_lang = detect(prompt)
                print(f"Detected language: {detected_lang}")
            except:
                detected_lang = "en"
                print("Language detection failed, defaulting to English")
            
            # Generate image using Together.xyz
            try:
                image_base64 = query_together(prompt)
                
                if not image_base64:
                    return jsonify({
                        "success": False,
                        "error": "Failed to generate image: No image data returned"
                    }), 500
                    
                return jsonify({
                    "success": True,
                    "message": "Image generated successfully",
                    "image_url": f"data:image/jpeg;base64,{image_base64}"
                }), 200
                
            except Exception as gen_error:
                error_msg = str(gen_error)
                print(f"Image generation error: {error_msg}")
                
                if "API Error" in error_msg:
                    # This is an API-specific error, return it directly
                    return jsonify({
                        "success": False,
                        "error": error_msg
                    }), 500
                else:
                    # This is an unexpected error
                    return jsonify({
                        "success": False,
                        "error": f"Failed to generate image: {error_msg}"
                    }), 500
                
        except Exception as e:
            print(f"Error in generate endpoint processing: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Error processing request: {str(e)}"
            }), 500
            
    except Exception as e:
        print(f"Error in generate endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

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