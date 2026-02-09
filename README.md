# ✋ Hand Gesture Recognition Using Deep Learning and Computer Vision

🔗 **Live Application (Hugging Face Space)**  
https://huggingface.co/spaces/leyuzak/Hand-Gesture-Recognition-Computer-Vision  

🔗 **Model Development & Training (Kaggle Notebook)**  
https://www.kaggle.com/code/leyuzakoksoken/hand-gesture-recognition-computer-vision  

---

## 1. Introduction

Hand gesture recognition is a fundamental problem in **human–computer interaction (HCI)**, enabling intuitive and contactless communication between humans and machines.  
This project presents an **end-to-end hand gesture recognition system**, from data preprocessing and model training to deployment as a **web-based application**.

The system leverages **deep convolutional neural networks (CNNs)** to classify hand gestures from RGB images and demonstrates how a trained model can be converted into a real-world, user-friendly application using **Streamlit** and **Hugging Face Spaces**.

---

## 2. Problem Definition

Traditional input devices (keyboard, mouse, touchscreens) are not always practical or accessible.  
Hand gesture recognition provides an alternative interaction modality that can be applied to:

- Assistive technologies
- Virtual and augmented reality systems
- Robotics and automation
- Smart home control
- Touchless user interfaces

The core challenge is to build a model that:
- Generalizes well to unseen hand images  
- Is lightweight enough for real-time or near real-time inference  
- Can be deployed easily without specialized hardware  

---

## 3. Dataset Description

The dataset consists of **labeled hand gesture images**, where each image corresponds to a specific gesture class.  
Class labels and metadata are stored in `leapgestrecog_metadata.csv`, allowing:

- Dynamic loading of class names
- Easy scalability to new gesture classes
- Clean separation of data and label logic

Each image is:
- Resized to **224 × 224**
- Converted to RGB format
- Normalized using ImageNet statistics

---

## 4. Model Architecture

The chosen architecture is **MobileNetV2**, selected for its balance between:

- High classification accuracy
- Low computational cost
- Suitability for deployment on CPU-based environments

### Key characteristics:
- Depthwise separable convolutions
- Inverted residual blocks
- Efficient parameter usage

The final classification layer is modified to match the number of gesture classes.

**Framework:** PyTorch  
**Saved model:** `MobileNetV2_best.pt`

---

## 5. Training and Evaluation

Model training and experimentation were conducted in a **Kaggle Notebook**, which includes:

- Data loading and preprocessing
- Model initialization and fine-tuning
- Loss and accuracy tracking
- Model selection based on validation performance

The final model represents the **best-performing checkpoint** and is exported for deployment.

For full training details, see the Kaggle notebook linked above.

---

## 6. Inference Pipeline

The inference pipeline follows these steps:

1. User provides an image (camera or upload)
2. Image is preprocessed (resize, normalization)
3. Model outputs raw logits
4. Softmax converts logits to probabilities
5. Top-1 prediction and Top-5 probabilities are displayed

This design ensures consistency between training and deployment environments.

---

## 7. Web Application

The trained model is deployed as an interactive **Streamlit application**, where users can:

- Capture an image using their device camera
- Upload gesture images
- Instantly receive predictions with confidence scores

### Application Features:
- Clean and minimal user interface
- CPU-only inference
- Automatic model and metadata loading
- Error-safe startup checks

---

## 8. Deployment with Docker and Hugging Face Spaces

The application is containerized using **Docker**, ensuring:

- Reproducibility
- Environment consistency
- Easy scalability

Deployment platform:
- **Hugging Face Spaces**
- SDK: **Docker**
- Port: **7860**

This setup allows the project to be publicly accessible without requiring users to install any dependencies locally.

