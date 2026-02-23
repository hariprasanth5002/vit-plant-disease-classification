🌿 Plant Disease Classification using Vision Transformer (ViT-B/16)

A deep learning–based plant disease classification system built using a Vision Transformer (ViT-B/16) fine-tuned on the PlantVillage dataset, achieving 99.59% test accuracy.

This project demonstrates:

Transformer-based image classification

Transfer learning from ImageNet

End-to-end model training

Model evaluation with metrics and visualizations

REST API deployment using FastAPI

📌 1. Problem Statement

Plant diseases significantly impact agricultural productivity. Early and accurate disease detection can help farmers prevent crop loss.

Traditional CNN-based architectures (e.g., ResNet, EfficientNet) focus on local texture extraction. In contrast, Vision Transformers (ViT) capture global relationships across leaf regions using self-attention.

This project explores how transformer architectures perform in plant disease diagnosis.

📊 2. Dataset

Dataset: PlantVillage

Total Classes: 38

Total Images Used: ~54,000+

Split Strategy:

70% Training

15% Validation

15% Testing

Each class represents a crop-disease combination such as:

Apple___Apple_scab

Tomato___Early_blight

Potato___Late_blight

etc.

Images are clean, centered leaf images with controlled backgrounds.

🧠 3. Model Architecture
Backbone: Vision Transformer (ViT-B/16)

Pretrained on ImageNet (1K classes)

Patch size: 16×16

Transformer encoder layers: 12

Embedding dimension: 768

Total parameters: ~86M

Modifications for Fine-Tuning

Replaced original 1000-class head

Added custom classifier: Linear(768 → 38)

Entire network fine-tuned (not just head)

Training Strategy

Loss Function: CrossEntropyLoss

Optimizer: AdamW

Learning Rate: 3e-5

Weight Decay: 1e-4

Early Stopping applied

This is full fine-tuning, not feature extraction.

🔄 4. Image Processing Pipeline

The image classification flow:

Input Image
→ Resize to 224×224
→ Convert to Tensor
→ Normalize (ImageNet Mean/Std)
→ Split into 16×16 patches
→ Linear patch embedding
→ Add positional encoding
→ Transformer attention blocks
→ Classification token extraction
→ Linear layer (38 classes)
→ Softmax probabilities
→ Predicted disease
📈 5. Model Performance
Test Accuracy: 99.59%
Training & Validation Curves

Available in:

results/training_curve.png
Confusion Matrix

Available in:

results/confusion_matrix.png

The confusion matrix shows strong diagonal dominance, indicating high classification precision across classes.

⚠️ 6. Real-World Observation (Important Discussion)

While the model achieved 99.59% accuracy on the PlantVillage test set:

Performance drops significantly on real-world mobile images.

Cause: Dataset distribution mismatch (Domain Shift).

PlantVillage images:

Clean background

Centered leaf

Controlled lighting

Real-world images:

Complex backgrounds

Shadows

Perspective distortions

This highlights an important limitation of controlled datasets in practical deployment.

Future work includes:

Domain adaptation

Swin Transformer comparison

ViT + Swin ensemble

Real-world dataset fine-tuning

🚀 7. How to Run the Project Locally
Step 1 — Clone the Repository
git clone https://github.com/yourusername/plant-disease-vit.git
cd plant-disease-vit
Step 2 — Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate  (Windows)
Step 3 — Install Dependencies
pip install -r requirements.txt
📦 8. Download Model Weights

Due to GitHub’s 100MB file limit, model weights are hosted externally.

Download from:

[Insert Google Drive Link Here]

After downloading:

Place the file:

best_model.pth

inside:

models/

Final structure:

models/
   best_model.pth
🧪 9. Run the API Server

Start FastAPI server:

uvicorn app.main:app --reload

Open browser:

http://127.0.0.1:8000/docs

Use the interactive Swagger UI to:

Upload leaf image

Get predicted disease

View prediction confidence

📂 10. Project Structure
plant-disease-vit/
│
├── app/
│   ├── main.py
│   └── model_loader.py
│
├── models/
│   └── best_model.pth
│
├── notebooks/
│   └── vit_training.ipynb
│
├── results/
│   ├── training_curve.png
│   └── confusion_matrix.png
│
├── requirements.txt
├── README.md
└── .gitignore
📚 11. Technologies Used

PyTorch

Torchvision

FastAPI

Uvicorn

Matplotlib

Scikit-learn

Vision Transformers (ViT-B/16)

🔬 12. Future Enhancements

Train Swin Transformer

ViT + Swin ensemble

Domain adaptation for field images

Model compression for mobile deployment

Web interface integration

👨‍🎓 Author

Hariprasanth U
Vision-Based Plant Disease Classification