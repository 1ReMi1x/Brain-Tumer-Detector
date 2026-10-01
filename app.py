import os
import cv2
import numpy as np
import pydicom
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
  checkpoint = torch.load('brain_tumor_model.pth', map_location='cpu')
  classes = checkpoint['classes']
  model = models.resnet18()
  model.fc = nn.Linear(model.fc.in_features, len(classes))
  model.load_state_dict(checkpoint['model'])
  model.eval()
  return model, classes


model, class_names = load_model()


def read_image(file_bytes, filename):
  ext = os.path.splitext(filename)[1].lower()
  if ext in ['.dcm', '.dicom']:
    ds = pydicom.dcmread(file_bytes)
    img = ds.pixel_array.astype(float)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5) * 255.0
    return Image.fromarray(img.astype(np.uint8)).convert('RGB')
  else:
    file_bytes = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


st.title('Brain Tumor Detection AI')
uploaded_file = st.file_uploader(
    'Upload Brain MRI Scan',
    type=['png', 'jpg', 'jpeg', 'dcm', 'dicom', 'bmp'],
)

if uploaded_file is not None:
  img = read_image(uploaded_file, uploaded_file.name)
  st.image(img, caption='Uploaded Image', width=300)

  tensor = transform(img).unsqueeze(0)
  with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1)
    conf, pred_idx = probs.max(1)
    result = class_names[pred_idx.item()]
    conf_score = conf.item() * 100

  if result == 'Not_Brain_MRI':
    st.error(
        f'❌ REJECTED: Not a Brain MRI Image (Confidence: {conf_score:.1f}%)'
    )
  elif result == 'No_Tumor':
    st.success(
        '✅ ACCEPTED: Brain MRI | Healthy / No Tumor Detected'
        f' (Confidence: {conf_score:.1f}%)'
    )
  else:
    st.warning(
        '⚠️ ACCEPTED: Brain MRI | Brain Tumor Detected (Confidence:'
        f' {conf_score:.1f}%)'
    )