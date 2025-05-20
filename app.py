import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Model definition must match the one used in training
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
# Save the trained model
# torch.save(model.state_dict(), "digit_cnn.pth")
print("Model saved to digit_cnn.pth")
# model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
model.load_state_dict(torch.load("digit_cnn.pth", map_location=device), strict=False)
model.eval()

st.title("Handwritten Digit Recognition")
st.write("Upload an image or draw a digit (0-9)")

uploaded_file = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert image (white digit on black)
    image = image.resize((28, 28))
    st.image(image, caption='Uploaded Image (28x28)', width=150)

    img_np = np.array(image)
    img_np = img_np.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        predicted = torch.argmax(output, dim=1).item()
    
    st.success(f"Predicted Digit: *{predicted}*")

else:
    st.info("Please upload a digit image for prediction.")
