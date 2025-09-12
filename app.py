# app.py
from flask import Flask, request, jsonify, render_template
from model_builder import TinyVGG
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

classes=["Healthy","Powdery","Rust"]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyVGG(input_shape=3,
                hidden_units=10,
                output_shape=3)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return "No file uploaded!", 400
    
    img_file = request.files["image"]
    img_bytes = img_file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor.to(device))
        _, predicted = torch.max(outputs, 1)
        class_index = predicted.item()

    return jsonify({"prediction": classes[class_index]})

if __name__ == "__main__":
    app.run(debug=True)
