from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

def base64_to_cv2(base64_string):
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    img = img.convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def enhance_palm_lines(image):
    """Enhanced preprocessing specifically for palm lines"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Strong contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    
    # Additional contrast stretch
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    # Bilateral filter - preserves edges while smoothing
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return filtered

def detect_lines_method1(enhanced, height, width):
    """Method 1: Adaptive threshold + morphological operations"""
    # Adaptive thresholding to find dark lines
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 3
    )
    
    # Morphological operations to connect broken lines
    kernel_line = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_line)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def detect_lines_method2(enhanced, height, width):
    """Method 2: Canny with very low thresholds for subtle lines"""
    # Very low thresholds to catch subtle palm creases
    edges = cv2.Canny(enhanced, 10, 50)
    
    # Dilate to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def detect_lines_method3(enhanced, height, width):
    """Method 3: Ridge detection using Laplacian"""
    # Laplacian for ridge detection
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F, ksize=3)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Threshold
    _, thresh = cv2.threshold(laplacian, 15, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def classify_line(points, height, width):
    """Classify a contour as a specific palm line based on position and shape"""
    if len(points) < 10:
        return None, 0
    
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()
    avg_y = y_coords.mean() / height
    avg_x = x_coords.mean() / width
    
    contour_width = (max_x - min_x) / width
    contour_height = (max_y - min_y) / height
    
    # Calculate aspect ratio and orientation
    aspect_ratio = contour_width / max(contour_height, 0.01)
    
    # Score each line type
    scores = {
        'heart_line': 0,
        'head_line': 0,
        'life_line': 0,
        'fate_line': 0
    }
    
    # Heart line: horizontal, upper part of palm (y: 15-40%)
    if 0.10 < avg_y < 0.45 and contour_width > 0.15 and aspect_ratio > 1.5:
        scores['heart_line'] = contour_width * 100 + (1 - abs(avg_y - 0.25)) * 50
    
    # Head line: horizontal, middle of palm (y: 30-55%)
    if 0.25 < avg_y < 0.60 and contour_width > 0.15 and aspect_ratio > 1.2:
        scores['head_line'] = contour_width * 100 + (1 - abs(avg_y - 0.42)) * 50
    
    # Life line: curved, left-center, more vertical (x: 20-50%)
    if 0.15 < avg_x < 0.55 and contour_height > 0.12 and aspect_ratio < 2.5:
        scores['life_line'] = contour_height * 100 + (1 - abs(avg_x - 0.35)) * 50
    
    # Fate line: vertical, center (x: 35-65%)
    if 0.30 < avg_x < 0.70 and contour_height > 0.10 and aspect_ratio < 1.0:
        scores['fate_line'] = contour_height * 100 + (1 - abs(avg_x - 0.5)) * 50
    
    # Return the best match
    best_line = max(scores, key=scores.get)
    best_score = scores[best_line]
    
    if best_score > 10:
        return best_line, best_score
    return None, 0

def simplify_line(points, num_points=15):
    """Simplify line to fewer points while maintaining shape"""
    if len(points) <= num_points:
        return points.tolist()
    
    # Sort by position for consistent ordering
    if np.std(points[:, 0]) > np.std(points[:, 1]):
        # More horizontal - sort by x
        sorted_idx = np.argsort(points[:, 0])
    else:
        # More vertical - sort by y
        sorted_idx = np.argsort(points[:, 1])
    
    sorted_points = points[sorted_idx]
    indices = np.linspace(0, len(sorted_points) - 1, num_points, dtype=int)
    return sorted_points[indices].tolist()

def detect_palm_lines(image):
    """Main function to detect palm lines using multiple methods"""
    height, width = image.shape[:2]
    
    # Enhance image for palm line detection
    enhanced = enhance_palm_lines(image)
    
    # Try multiple detection methods
    all_contours = []
    all_contours.extend(detect_lines_method1(enhanced, height, width))
    all_contours.extend(detect_lines_method2(enhanced, height, width))
    all_contours.extend(detect_lines_method3(enhanced, height, width))
    
    # Track best line for each type
    lines = {
        'heart_line': {'points': [], 'score': 0},
        'head_line': {'points': [], 'score': 0},
        'life_line': {'points': [], 'score': 0},
        'fate_line': {'points': [], 'score': 0}
    }
    
    for contour in all_contours:
        if len(contour) < 10:
            continue
        
        points = contour.reshape(-1, 2)
        line_type, score = classify_line(points, height, width)
        
        if line_type and score > lines[line_type]['score']:
            lines[line_type]['points'] = points
            lines[line_type]['score'] = score
    
    # Convert to output format (percentages)
    result = {}
    for line_name, data in lines.items():
        if len(data['points']) > 0:
            simplified = simplify_line(np.array(data['points']))
            result[line_name] = [
                {'x': float(p[0]) / width, 'y': float(p[1]) / height}
                for p in simplified
            ]
        else:
            result[line_name] = []
    
    return result

@app.route('/detect-lines', methods=['POST'])
def detect_lines():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        image = base64_to_cv2(data['image'])
        lines = detect_palm_lines(image)
        lines_found = sum(1 for l in lines.values() if len(l) > 0)
        
        return jsonify({
            'success': True,
            'lines_detected': lines_found,
            'lines': lines,
            'image_size': {'width': image.shape[1], 'height': image.shape[0]}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Palm Line Detection API v2',
        'endpoints': ['/detect-lines', '/health'],
        'methods': ['adaptive_threshold', 'canny_edge', 'laplacian_ridge']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
