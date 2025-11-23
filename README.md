# Apple Leaf Disease Detection Web Application

A **Flask-based web application** for real-time apple leaf disease classification using a pretrained **TinyVGG** deep learning model. This interactive application allows users to upload apple leaf images and receive instant disease predictions, helping identify three common conditions: Healthy, Rust, and Powdery Mildew.

## Project Overview

This project demonstrates the deployment of a PyTorch-trained computer vision model as a user-friendly web application. Users can upload images of apple leaves through an intuitive web interface and receive immediate classification results powered by a lightweight TinyVGG convolutional neural network.

### Key Features

- **Web-Based Inference**: Upload and classify apple leaf images directly in your browser
- **Real-Time Predictions**: Instant disease classification with a single click
- **3-Class Detection**: Identifies Healthy, Rust, and Powdery Mildew conditions
- **Lightweight Model**: TinyVGG architecture optimized for fast inference
- **Interactive UI**: Clean, modern interface with image preview
- **Flask Framework**: Simple, scalable Python web backend
- **Pretrained Model**: Ready-to-use `best_model.pth` for immediate deployment

## Disease Classes

The application classifies apple leaves into three categories:

| Class              | Description                           | Visual Characteristics                       |
| ------------------ | ------------------------------------- | -------------------------------------------- |
| **Healthy**        | Normal, disease-free apple leaves     | Green, unblemished leaves with uniform color |
| **Rust**           | Apple rust disease (fungal infection) | Orange-brown spots, rust-colored lesions     |
| **Powdery Mildew** | Powdery mildew infection              | White, powdery coating on leaf surface       |

### Sample Images

Below are examples of the different apple leaf conditions the model can detect:

**Healthy Apple Leaf:**
![Healthy Apple Leaf](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/d3a51a51-bbae-4a73-8fab-e0a55f5bf02b)
_Clean, green leaf with no disease symptoms_

**Powdery Mildew Infected Leaf:**
![Powdery Mildew](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/4d540a71-d39a-4e3b-849c-c6b93f1d842b)
_White, powdery coating characteristic of powdery mildew infection_

## Project Structure

```
├── app.py                      # Flask application (main backend)
├── model_builder.py           # TinyVGG model architecture definition
├── best_model.pth             # Pretrained model weights (not in repo, download separately)
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html            # Frontend HTML template
├── static/
│   └── style.css             # CSS styling for the web interface
└── README.md                 # This file
```

## File Descriptions

### `app.py` **Main Application**

The Flask backend that handles web requests and model inference:

**Key Components:**

- **Flask App Setup**: Initializes Flask application and routes
- **Model Loading**: Loads pretrained TinyVGG model from `best_model.pth`
- **Device Selection**: Automatically uses GPU if available, otherwise CPU
- **Image Preprocessing**: Resizes images to 64×64 and converts to tensor
- **Routes**:
  - `GET /`: Serves the main HTML page
  - `POST /predict`: Accepts uploaded images and returns predictions

**Code Flow:**

```python
1. User uploads image via /predict endpoint
2. Image is read and converted to PIL Image (RGB)
3. Image is transformed (resize to 64×64, convert to tensor)
4. Model performs inference (forward pass)
5. Predicted class index extracted using torch.max()
6. Class name returned as JSON response
```

**Class Mapping:**

```python
classes = ["Healthy", "Powdery", "Rust"]
# Index 0 → Healthy
# Index 1 → Powdery Mildew
# Index 2 → Rust
```

### `model_builder.py`

Defines the TinyVGG architecture used for classification:

**Architecture:**

```
TinyVGG(
  input_shape=3,      # RGB channels
  hidden_units=10,    # Number of feature maps
  output_shape=3      # 3 disease classes
)
```

**Model Structure:**

- **Conv Block 1**:
  - Conv2d(3 → 10, kernel=3, padding=0) + ReLU
  - Conv2d(10 → 10, kernel=3, padding=0) + ReLU
  - MaxPool2d(kernel=2, stride=2)
- **Conv Block 2**:
  - Conv2d(10 → 10, kernel=3, padding=0) + ReLU
  - Conv2d(10 → 10, kernel=3, padding=0) + ReLU
  - MaxPool2d(kernel=2, stride=2)
- **Classifier**:
  - Flatten()
  - Linear(10×13×13 → 3)

**Input/Output:**

- Input: (batch, 3, 64, 64) RGB images
- Output: (batch, 3) logits for 3 classes

**Based on**: [CNN Explainer TinyVGG](https://poloclub.github.io/cnn-explainer/)

### `templates/index.html`

Frontend HTML template with interactive features:

**Features:**

- **File Upload Form**: Allows users to select image files
- **Image Preview**: Displays selected image before prediction
- **AJAX Submission**: Asynchronous form submission without page reload
- **Real-Time Results**: Displays prediction results dynamically
- **Responsive Design**: Clean, centered layout

**JavaScript Functionality:**

1. **Image Preview**: Shows thumbnail of uploaded image
2. **Form Submission**: AJAX POST request to `/predict`
3. **Loading State**: "Loading prediction..." message during inference
4. **Result Display**: Shows predicted class name

### `static/style.css`

Modern, clean styling for the web interface:

**Design Elements:**

- **Centered Layout**: Flexbox centering for main container
- **Card-Based UI**: White container with rounded corners and shadow
- **Color Scheme**:
  - Background: Light gray (#f3f4f6)
  - Primary: Blue (#007bff)
  - Success: Green (#2e7d32)
- **Responsive**: Max-width 500px for optimal viewing
- **Interactive**: Hover effects on buttons
- **Image Display**: Rounded corners, max 300px height

### `requirements.txt`

Python dependencies for the application:

**Key Dependencies:**

- **Flask 3.1.1**: Web framework
- **torch 2.7.1**: PyTorch for model inference
- **torchvision 0.22.1**: Image transformations
- **Pillow 11.3.0**: Image loading and processing
- **numpy 2.2.6**: Numerical operations
- **Werkzeug 3.1.3**: WSGI utilities for Flask
- **Jinja2 3.1.6**: Template engine

## Quick Start

### Prerequisites

- Python 3.8+
- Pretrained model file: `best_model.pth` (ensure this is in the project root)

### Installation

1. **Clone or download the project**

2. **Create a virtual environment (recommended)**

   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda
   conda create -n apple-disease python=3.9
   conda activate apple-disease
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model file exists**
   ```bash
   # Verify best_model.pth is in the project root directory
   ls best_model.pth
   ```

### Running the Application

1. **Start the Flask server**

   ```bash
   python app.py
   ```

2. **Access the web application**

   - Open your browser and navigate to: `http://127.0.0.1:5000/`
   - Or: `http://localhost:5000/`

3. **Use the application**
   - Click "Choose File" to select an apple leaf image
   - Preview appears automatically
   - Click "Predict" to get classification result
   - Result appears below the image (e.g., "Prediction: Healthy")

### Expected Output

```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
```

## User Interface

The web interface provides a clean, intuitive experience:

### Main Features

1. **Header**: " Plant Disease Detection" title
2. **Upload Section**:
   - File input with "Choose File" button
   - Accepts image formats (JPEG, PNG, etc.)
   - "Predict" button to submit
3. **Preview Area**:
   - Shows uploaded image before prediction
   - Rounded corners with border
   - Max 300px height for consistency
4. **Results Display**:
   - Green text showing prediction
   - Format: "Prediction: [Class Name]"
   - Updates asynchronously without page reload

## Customization

### Changing Model Parameters

If you retrain the model with different parameters, update `app.py`:

```python
# For a model with different hidden units or classes
model = TinyVGG(
    input_shape=3,           # RGB channels (keep as 3)
    hidden_units=20,         # Change if your model uses different hidden units
    output_shape=5           # Change to match number of disease classes
)

# Update class list
classes = ["Class1", "Class2", "Class3", "Class4", "Class5"]
```

### Modifying Image Size

To change input image dimensions (requires model retraining):

```python
# In app.py, modify the transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Change from (64, 64)
    transforms.ToTensor()
])

# Update model_builder.py Linear layer to match new dimensions
# For 128×128 input, calculate feature map size after conv blocks
```

### Adding More Disease Classes

1. **Retrain model** with additional classes
2. **Update class list** in `app.py`:
   ```python
   classes = ["Healthy", "Powdery", "Rust", "Scab", "Black Rot"]
   ```
3. **Update output_shape** in model initialization:
   ```python
   model = TinyVGG(input_shape=3, hidden_units=10, output_shape=5)
   ```

### Styling Changes

Modify `static/style.css` to customize appearance:

```css
/* Change primary color */
.main-heading {
  color: #1976d2; /* Blue instead of green */
}

/* Adjust button color */
input[type="submit"] {
  background-color: #4caf50; /* Green instead of blue */
}
```

## Deployment Options

### Local Development

Current setup with `app.run(debug=True)` is ideal for development.

### Production Deployment

For production, use a WSGI server like **Gunicorn**:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Cloud Deployment

**Heroku:**

1. Create `Procfile`: `web: gunicorn app:app`
2. Deploy: `git push heroku main`

**AWS/GCP/Azure:**

- Use containerization with Docker
- Deploy to App Service, Elastic Beanstalk, or Cloud Run

## Important Notes

### Model File

- **Ensure `best_model.pth` exists** in the project root before running
- Model must be trained with same architecture (TinyVGG with hidden_units=10, output_shape=3)
- Model expects 64×64 RGB images

### Image Requirements

- **Supported formats**: JPEG, JPG, PNG
- **Automatic preprocessing**: Images resized to 64×64
- **Color**: RGB (3 channels)
- **Best results**: Clear, well-lit leaf images

### Browser Compatibility

- Works on all modern browsers (Chrome, Firefox, Safari, Edge)
- JavaScript must be enabled for AJAX functionality

### Performance

- **CPU Inference**: Fast on modern CPUs (~0.1-0.5 seconds per image)
- **GPU Inference**: Near-instant if CUDA is available
- **Model Size**: Lightweight (~50KB), suitable for edge deployment

## Technical Details

### TinyVGG Architecture

The model uses a simplified VGG-style architecture:

**Advantages:**

- **Lightweight**: Only 10 feature maps, suitable for embedded systems
- **Fast Inference**: Small model size enables quick predictions
- **Effective**: Sufficient for 3-class plant disease detection
- **Interpretable**: Simple architecture easier to debug and understand

**Feature Extraction:**

- Two convolutional blocks progressively extract features
- ReLU activations introduce non-linearity
- MaxPooling reduces spatial dimensions and provides translation invariance

### Flask Framework

**Why Flask?**

- **Simplicity**: Minimal boilerplate for small applications
- **Flexibility**: Easy to extend with additional routes
- **Python Integration**: Seamless PyTorch model loading
- **Widely Supported**: Large community and ecosystem

### Image Preprocessing Pipeline

```python
1. Upload: User selects file from disk
2. Read: Flask receives file as binary stream
3. Decode: PIL.Image.open() converts bytes to image
4. Convert: .convert("RGB") ensures 3 channels
5. Transform: Resize to 64×64, convert to tensor
6. Normalize: ToTensor() scales to [0, 1]
7. Batch: .unsqueeze(0) adds batch dimension
8. Inference: Model processes (1, 3, 64, 64) tensor
```

## 🔬 Model Training (Not Included)

This repository contains only the inference application. To train your own model:

1. **Collect dataset**: Apple leaf images with labels (Healthy, Rust, Powdery)
2. **Preprocess**: Resize to 64×64, augment with flips/rotations
3. **Train**: Use PyTorch with CrossEntropyLoss and Adam optimizer
4. **Evaluate**: Test on held-out validation set
5. **Save**: `torch.save(model.state_dict(), 'best_model.pth')`
6. **Deploy**: Replace `best_model.pth` in this application

## References

- **TinyVGG Architecture**: [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- **Flask Documentation**: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)
- **Apple Diseases**: Plant pathology resources for rust and powdery mildew
