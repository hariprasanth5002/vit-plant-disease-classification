from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import io
import torchvision.transforms as transforms
import json

# -----------------------------
# INITIAL SETUP
# -----------------------------

app = FastAPI()

NUM_CLASSES = 38   # MUST match your training

# -----------------------------
# LOAD MODEL
# -----------------------------

model = models.vit_b_16(weights=None)

model.heads.head = nn.Linear(
    model.heads.head.in_features,
    NUM_CLASSES
)

model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

print("Model loaded successfully.")
print("Model output classes:", model.heads.head.out_features)

# -----------------------------
# LOAD CLASS NAMES
# -----------------------------

class_names = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Blueberry___healthy",
"Cherry_(including_sour)___Powdery_mildew",
"Cherry_(including_sour)___healthy",
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
"Corn_(maize)___Common_rust_",
"Corn_(maize)___Northern_Leaf_Blight",
"Corn_(maize)___healthy",
"Grape___Black_rot",
"Grape___Esca_(Black_Measles)",
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
"Grape___healthy",
"Orange___Haunglongbing_(Citrus_greening)",
"Peach___Bacterial_spot",
"Peach___healthy",
"Pepper,_bell___Bacterial_spot",
"Pepper,_bell___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Raspberry___healthy",
"Soybean___healthy",
"Squash___Powdery_mildew",
"Strawberry___Leaf_scorch",
"Strawberry___healthy",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Tomato_mosaic_virus",
"Tomato___healthy"
]


print("Total classes loaded:", len(class_names))

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# PREDICTION ROUTE
# -----------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_index = predicted.item()

    print("Predicted index:", predicted_index)
    print("Class list length:", len(class_names))

    # Safety check
    if predicted_index >= len(class_names):
        return {
            "error": "Prediction index exceeds class list length",
            "predicted_index": predicted_index,
            "class_list_length": len(class_names)
        }

    predicted_class = class_names[predicted_index]
    probs = torch.softmax(outputs, dim=1)
    confidence = torch.max(probs).item()

    return {
    "prediction": class_names[predicted.item()],
    "confidence": confidence
    }

