from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import io
import uuid
from supabase import create_client, Client

images_bp = Blueprint('images', __name__)

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv('SUPABASE_URL', ''),
    os.getenv('SUPABASE_KEY', '')
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@images_bp.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Create unique filename
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        
        # Upload to Supabase Storage
        file_data = file.read()
        res = supabase.storage.from_('ditoolz-media').upload(
            filename,
            file_data
        )
        
        # Get public URL
        public_url = supabase.storage.from_('ditoolz-media').get_public_url(filename)
        
        return jsonify({
            "message": "File uploaded successfully",
            "filename": filename,
            "url": public_url
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@images_bp.route('/api/images/enhance', methods=['POST'])
def enhance_image():
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400
            
        # Download image from Supabase
        response = supabase.storage.from_('ditoolz-media').download(image_url)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(response))
        
        # Basic enhancement
        enhanced = Image.fromarray(np.uint8(np.array(image) * 1.2))
        
        # Save enhanced image
        output = io.BytesIO()
        enhanced.save(output, format='PNG')
        output.seek(0)
        
        # Upload enhanced image
        new_filename = f"enhanced_{uuid.uuid4()}.png"
        supabase.storage.from_('ditoolz-media').upload(
            new_filename,
            output.getvalue()
        )
        
        # Get public URL
        public_url = supabase.storage.from_('ditoolz-media').get_public_url(new_filename)
        
        return jsonify({
            "message": "Image enhanced successfully",
            "enhanced_url": public_url
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400