{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from flask import Flask, request, jsonify\
from flask_cors import CORS\
import cv2\
import numpy as np\
import base64\
from io import BytesIO\
from PIL import Image\
import os\
\
app = Flask(__name__)\
CORS(app)\
\
def base64_to_cv2(base64_string):\
    if 'base64,' in base64_string:\
        base64_string = base64_string.split('base64,')[1]\
    img_data = base64.b64decode(base64_string)\
    img = Image.open(BytesIO(img_data))\
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)\
\
def simplify_line(points, num_points):\
    if len(points) <= num_points:\
        return points.tolist()\
    sorted_points = points[points[:, 0].argsort()]\
    indices = np.linspace(0, len(sorted_points) - 1, num_points, dtype=int)\
    return sorted_points[indices].tolist()\
\
def detect_palm_lines(image):\
    height, width = image.shape[:2]\
    \
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))\
    enhanced = clahe.apply(gray)\
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)\
    edges = cv2.Canny(blurred, 30, 80)\
    \
    kernel = np.ones((3, 3), np.uint8)\
    dilated = cv2.dilate(edges, kernel, iterations=1)\
    \
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)\
    \
    lines = \{\
        'heart_line': [],\
        'head_line': [],\
        'life_line': [],\
        'fate_line': []\
    \}\
    \
    for contour in contours:\
        if len(contour) < 30:\
            continue\
        \
        points = contour.reshape(-1, 2)\
        x_coords = points[:, 0]\
        y_coords = points[:, 1]\
        \
        min_x, max_x = x_coords.min(), x_coords.max()\
        min_y, max_y = y_coords.min(), y_coords.max()\
        avg_y = y_coords.mean() / height\
        avg_x = x_coords.mean() / width\
        contour_width = (max_x - min_x) / width\
        contour_height = (max_y - min_y) / height\
        \
        if (0.15 < avg_y < 0.38 and contour_width > 0.25 and contour_height < 0.2):\
            if len(points) > len(lines['heart_line']):\
                lines['heart_line'] = simplify_line(points, 15)\
        \
        elif (0.32 < avg_y < 0.55 and contour_width > 0.25 and contour_height < 0.25):\
            if len(points) > len(lines['head_line']):\
                lines['head_line'] = simplify_line(points, 15)\
        \
        elif (avg_x < 0.45 and contour_height > 0.15 and contour_width < contour_height * 1.5):\
            if len(points) > len(lines['life_line']):\
                lines['life_line'] = simplify_line(points, 15)\
        \
        elif (0.35 < avg_x < 0.65 and contour_height > 0.15 and contour_width < 0.15):\
            if len(points) > len(lines['fate_line']):\
                lines['fate_line'] = simplify_line(points, 10)\
    \
    result = \{\}\
    for line_name, points in lines.items():\
        if len(points) > 0:\
            result[line_name] = [\
                \{'x': float(p[0]) / width, 'y': float(p[1]) / height\}\
                for p in points\
            ]\
        else:\
            result[line_name] = []\
    \
    return result\
\
@app.route('/detect-lines', methods=['POST'])\
def detect_lines():\
    try:\
        data = request.json\
        if 'image' not in data:\
            return jsonify(\{'success': False, 'error': 'No image provided'\}), 400\
        \
        image = base64_to_cv2(data['image'])\
        lines = detect_palm_lines(image)\
        lines_found = sum(1 for l in lines.values() if len(l) > 0)\
        \
        return jsonify(\{\
            'success': True,\
            'lines_detected': lines_found,\
            'lines': lines,\
            'image_size': \{'width': image.shape[1], 'height': image.shape[0]\}\
        \})\
    except Exception as e:\
        return jsonify(\{'success': False, 'error': str(e)\}), 500\
\
@app.route('/health', methods=['GET'])\
def health():\
    return jsonify(\{'status': 'ok'\})\
\
@app.route('/', methods=['GET'])\
def home():\
    return jsonify(\{'message': 'Palm Line Detection API', 'endpoints': ['/detect-lines', '/health']\})\
\
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
