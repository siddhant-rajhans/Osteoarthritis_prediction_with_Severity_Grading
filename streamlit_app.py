import PIL
import streamlit as st
import torch
from torchvision import transforms
import cv2
import torch.nn as nn
import numpy as np
import pandas as pd

# Model loading (assuming you have a saved model file)
model_path = "model/saved_model.pt"  # Replace with your model path
model = torch.hub.load('pytorch/vision:v0.13.0', 'resnet18', pretrained=False)  # Load pre-trained model
model.fc = nn.Linear(model.fc.in_features, 3)  # Assuming 3 classes (Healthy, Moderate, Severe)
# model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Define image transformation (ensure normalization matches training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Assuming grayscale in training
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Replace with training values
])

def predict_osteoarthritis(image):
  """
  Preprocesses and predicts osteoarthritis grade on an image.

  Args:
      image: A PIL image object.

  Returns:
      A tuple containing the predicted class and probability vector.
  """
  # Convert to PIL Image if it's a NumPy array
  if isinstance(image, np.ndarray):
    image = PIL.Image.fromarray(image)  # Convert NumPy array to PIL Image
  image = transform(image)
  image = image.unsqueeze(0)  # Add batch dimension
  with torch.no_grad():
    output = model(image)
    probs = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
  return predicted_class, probs.squeeze().numpy() * 100 # Return class and probability vector

st.title("Osteoarthritis Classification")
st.write("Upload an X-ray image to predict the osteoarthritis grade.")

uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
  image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for PyTorch
  st.image(image, caption="Uploaded Image", use_column_width=True)

  # Preprocess and predict
  predicted_class, probabilities = predict_osteoarthritis(image)
  grade_labels = ["Healthy", "Moderate", "Severe"]  # Assuming class labels

  # Display predictions
  st.write(f"Predicted Grade: {grade_labels[predicted_class]}")
  st.write(f"Probabilities:")
  for i, label in enumerate(grade_labels):
    st.write(f"- {label}: {probabilities[i]:.2f}")
