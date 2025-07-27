import os
import io
import base64
import numpy as np
import torch
import rasterio
from flask import Flask, request, render_template, send_file
from PIL import Image

# --- Step 1: ADD THIS IMPORT BACK ---
from model import MultiChannelUNet

# --- 1. Initialize App and Model ---
app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_flood_model.pth' # Use your original model file

# --- Step 2: ADD THESE TWO LINES BACK to define the model structure ---
model = MultiChannelUNet(in_channels=12, num_classes=2).to(device)
print("Model architecture created.")

try:
    # --- Step 3: CHANGE HOW YOU LOAD THE WEIGHTS ---
    # Load the entire dictionary checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Extract the state_dict and load it into your model
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully!")

except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    # Handle the error
except KeyError:
    # This is a fallback in case the file only contains the state_dict
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model state_dict loaded successfully!")
    
model.eval() # This will now work correctly!

# --- The rest of the file stays exactly the same ---
def normalize_channels(image):
    # ... (no changes here)
    normalized = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        channel = image[i].astype(np.float32)
        if i in [0, 1, 2, 3, 4, 5, 6]:
            p2, p98 = np.percentile(channel, (2, 98))
            channel = np.clip(channel, p2, p98)
            if (p98 - p2) > 0:
                channel = (channel - p2) / (p98 - p2)
        elif i in [8, 9]:
            std = np.std(channel)
            if std > 1e-8:
                channel = (channel - np.mean(channel)) / std
        else:
            min_val, max_val = np.min(channel), np.max(channel)
            if max_val > min_val:
                channel = (channel - min_val) / (max_val - min_val)
        normalized[i] = channel
    return normalized

def process_image(file_storage):
    # ... (no changes here)
    with rasterio.open(file_storage) as src:
        image = src.read()
    normalized_image = normalize_channels(image)
    tensor = torch.from_numpy(normalized_image).float().unsqueeze(0)
    return tensor

@app.route('/', methods=['GET'])
def index():
    # ... (no changes here)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # ... (no changes here)
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file and file.filename.endswith('.tif'):
        try:
            input_tensor = process_image(file).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                
            pred_mask = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8) * 255
            
            mask_image = Image.fromarray(pred_mask_np, mode='L')
            img_io = io.BytesIO()
            mask_image.save(img_io, 'PNG')
            img_io.seek(0)
            
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
            return render_template('index.html', prediction_image=img_base64)
        
        except Exception as e:
            return f"An error occurred: {str(e)}", 500

    return "Invalid file type. Please upload a .tif file.", 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')