import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms

# Model configuration
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 3)  # Assuming 3 classes
model.eval()  # Set model to evaluation mode
device = torch.device('cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the saved model
model_path = "model/saved_model.pt"
model.load_state_dict(torch.load(model_path))

# Define image transformation for consistency
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Set app title and description
st.title("Knee Osteoarthritis Classification")
st.write("Upload a knee X-ray image to classify its grade of Osteoarthritis.")


# Upload image widget
uploaded_image = st.file_uploader("Choose an image:", type="jpg,png")

if uploaded_image is not None:
  # Read the uploaded image
  image = Image.open(uploaded_image)

  # Preprocess the image
  image = transform(image).unsqueeze(0).to(device)

  # Make prediction
  with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    predicted_class = int(predicted.cpu().numpy()[0])

  # Display the uploaded image
  st.image(image.cpu().squeeze(0).permute(1, 2, 0).numpy(), width=256)

  # Map predicted class to label
  grade_labels = ["Healthy", "Moderate", "Severe"]
  predicted_label = grade_labels[predicted_class]

  # Display prediction result
  st.success(f"Predicted Grade: {predicted_label}")

