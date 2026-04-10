# AI-Generated Face Detector

This project presents a deep learning-based system for detecting whether a face image is real or AI-generated. The model is built using EfficientNet-B0 and incorporates Grad-CAM to provide visual explanations of its predictions. A Streamlit interface is used for interactive testing and demonstration.

---

## Overview

The system classifies input face images into two categories: **Real** and **Fake**. To improve prediction quality, face detection is applied prior to classification. Grad-CAM is used to highlight the regions of the image that most influenced the model’s decision.

---

## Key Features

- EfficientNet-B0 based image classifier  
- Three-phase transfer learning strategy  
- Face detection using MTCNN  
- Grad-CAM visualization for interpretability  
- Batch image upload support  
- Confidence scores for predictions  
- Option to download results as CSV  
- Simple and interactive Streamlit interface  

---

## Model Details

- Architecture: EfficientNet-B0  
- Dataset: 140,000 real and fake face images  
- Accuracy: 91.69%  
- AUC Score: 0.9764  

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
