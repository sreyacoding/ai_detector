# 🔍 AI Detector: AI vs Human Detection for Text, Audio, and Image 🤖 vs 👤

🚀 [**🌐 Explore the live hosted website here!**](https://gen-ai-detector.streamlit.app/)

---

## 📑 Table of Contents  

- [🌟 Introduction](#introduction)  
- [🎧 Audio Detection](#audio-detection)  
  - [🎯 Objective](#objective)  
  - [📊 Dataset](#dataset)  
  - [🧠 Model Training](#model-training)  
  - [⚙️ Tech Stack](#tech-stack)  
  - [🏆 Conclusion](#conclusion)  
- [✍️ Text Detection](#text-detection)  
  - [🎯 Objective](#objective-1)  
  - [📊 Dataset](#dataset-1)  
  - [🧠 Model Training](#model-training-1)  
  - [⚙️ Tech Stack](#tech-stack-1)  
  - [🏆 Conclusion](#conclusion-1)  
- [🖼️ Image Detection](#image-detection)  
  - [🎯 Objective](#objective-2)  
  - [📊 Dataset](#dataset-2)  
  - [🧠 Model Training](#model-training-2)  
  - [⚙️ Tech Stack](#tech-stack-2)  
  - [🏆 Conclusion](#conclusion-2)  
- [🌟 Final Thoughts](#final-thoughts)  

---

## Introduction

In the age of **AI-generated content**, ensuring **authenticity** is more important than ever. This project aims to develop a robust model that can **differentiate between human-made** and **AI-generated content** across three domains: **text**, **audio**, and **images**.  

- 🌍 **Real-time predictions** via a user-friendly **website**.  
- 🔎 **Enhances content integrity** across multiple formats.  
- 🚀 **Deployed and accessible online** for easy testing and integration.  

[**🌐 Explore the live hosted website here!**](https://gen-ai-detector.streamlit.app/)  

---

## Audio Detection  

### Objective  
Create a model to detect **AI-generated deepfake voices** and differentiate them from **real human speech**, providing a tool to combat **synthetic audio manipulation**.  

### Dataset  
The model was trained using the following datasets:  
- **TIMIT-TTS**: Synthetic speech samples.  
- **LibriSpeech**: Real human speaker samples.  

### Model Training  
- **XGBoost** 🏆: The **top-performing model** with the highest precision, recall, F1 score, and accuracy.  
- **LightGBM**: Another model tested for comparison.  

**Selected Model**: **XGBoost** 🏆 for its **robust performance** across all evaluation metrics.  

### Tech Stack  
- **Librosa & IPython** 🎶: **Audio feature extraction** (MFCC, zero-crossing rates).  
- **Scikit-learn** 🔧: Used for training **XGBoost** and **LightGBM**.  
- **Streamlit** 🌐: Interactive website allowing real-time predictions.  

### Conclusion  
The **XGBoost model** was selected for its **outstanding performance** in identifying **synthetic audio**, making it an essential tool for combating **deepfake audio**.  

---

## Text Detection  

### Objective  
Develop a model capable of **classifying AI-generated text** and **human-written content**, specifically for detecting texts written by **Large Language Models (LLMs)** like **ChatGPT**.  

### Dataset  
The **LLM-Detect AI Generated Text Dataset** was used to train the model for **accurate classification** between **AI** and **human text**.  

### Model Training  
Models trained and evaluated include:  
- **Logistic Regression**: Basic approach using **TF-IDF** features.  
- **Random Forest**: Ensemble learning for **better generalization**.  
- **Support Vector Machine (SVM)** 🏆: Achieved the **highest F1 score**, selected as the final model for deployment.  
- **BERT**: Experimented with **transformers**, but **SVM** outperformed it for practical deployment.  

**Selected Model**: **SVM** 🏆, chosen for its **high F1 score** and **deployment efficiency**.  

### Tech Stack  
- **Scikit-learn** 🔧: For **model training** and evaluation.  
- **Transformers Library** 💻: Used for experimenting with **BERT** and other transformer models.  
- **Streamlit** 🌐: Created an **interactive web interface** to get **real-time predictions** on input text.  

### Conclusion  
**SVM** was selected for **optimal performance** in **AI text detection**, ensuring reliable classification of **AI-generated text**.  

---

## Image Detection  

### Objective  
Create a model that can differentiate between **AI-generated images** and **real photographs**, ensuring the **authenticity of visual content**.  

### Dataset  
The model was trained on the **CIFake dataset**, which contains a combination of **real and AI-generated images**.  

### Model Training  
- **ResNet-18** 🏆: A **pre-trained CNN** model utilizing **ImageNet weights** for **transfer learning**.  
- **Adam Optimizer**: Used with **cross-entropy loss** to optimize the model’s performance.  

**Selected Model**: **ResNet-18** 🏆, fine-tuned for **image authenticity detection**.  

### Tech Stack  
- **PyTorch** 🔥: Used for deep learning model training and evaluation. 
- **Streamlit** 🌐: Developed a **website** for users to upload images and get **real-time predictions**.  

### Conclusion  
The **ResNet-18 model**, fine-tuned on the **CIFake dataset**, is highly effective at distinguishing between **real and AI-generated images**, ensuring **image authenticity** in the digital age.  

---

## Final Thoughts  

This project demonstrates the power of **machine learning** in detecting **AI-generated content** and safeguarding **digital integrity** across text, audio, and image formats.  

- **Real-time predictions** via a **website** make this technology accessible and user-friendly.  
- The system is designed to **easily integrate** with other tools to combat the growing challenges of **synthetic media**.  
- **Versatile** across multiple content types, ensuring broad applications in **education**, **media**, **journalism**, and **forensics**.  

---

🚀 **Explore the AI-powered detection system and help ensure content authenticity!**  
[**🌐 View the live hosted website here!**](https://gen-ai-detector.streamlit.app/)  
