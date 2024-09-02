from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os

app = Flask(__name__)

# Load pre-trained ResNet50 model for feature extraction
model = resnet50(weights='DEFAULT')
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.numpy().flatten()

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    similarities = {}
    if request.method == 'POST':
        uploaded_file = request.files['uploaded_image']
        if uploaded_file.filename != '':
            uploaded_image = Image.open(uploaded_file)
            uploaded_features = extract_features(uploaded_image)

            # Load sample images
            sample_images = {
                'Sample 1': 'sample1.png',
                'Sample 2': 'sample2.png',
                'Sample 3': 'sample3.png'
            }
            
            for label, file_name in sample_images.items():
                img_path = os.path.join('sample_images', file_name)
                sample_image = Image.open(img_path)
                sample_features = extract_features(sample_image)
                similarity = cosine_similarity([uploaded_features], [sample_features])[0][0]
                similarities[label] = similarity

    return render_template('index.html', similarities=similarities)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
