# ai_detector
AI vs Human Text and Audio Detection Model 

**Introduction**
This project develops a model to classify both text and audio as either AI-generated or human-made. Thus, the increasing application of AI in content development necessitates the establishment of authenticity and trust among producers and consumers.
We train models on both text and audio datasets, features that help to distinguish human input from AI output. The model is deployed via a Streamlit app where users can input text or upload audio files and immediately receive predictions, all with the value of ensuring content integrity in various domains.


Text Detection
**Objective:**
The goal of this project is to build a classification model that distinguishes between AI-generated text and human-written content. For example, the model can help detect whether a student or an LLM (Large Language Model), like ChatGPT, wrote an essay.    

**Dataset:** 
We used the LLM-Detect AI Generated Text Dataset to train the model and ensure accurate classification.

**Model Training:**
The following models were trained and evaluated for this task:

- Logistic Regression: Used to classify text based on TF-IDF features.
- Random Forest: Utilized ensemble learning on TF-IDF features for better generalization.
- Support Vector Machine (SVM): 
  - Achieved the highest F1 score among all models.
  - Selected as the final model for deployment due to superior performance.  
- BERT (Transformer Model): Fine-tuned on the dataset using the Transformers Library, but SVM outperformed it in terms of practical deployment.

**Tech Stack:**
- Scikit-learn: Used for training and evaluating the models.  
- Transformers Library: For experimentation with pre-trained models like BERT.  
- Streamlit: Built an interactive web app to allow users to input text and get predictions in real-time.

**Conclusion:**
The SVM model was chosen for deployment because it provided the best F1 score and overall performance. This project showcases how ML models can effectively identify text generated by AI systems, ensuring academic and professional integrity.


Audio Detection
**Objective:**
To build a model that can distinguish between an AI-generated deepfake voice vs real human speech.

**Dataset:** 
The model is trained using a combination of datasets like the TIMIT-TTS dataset, which contains a variety of synthetic audio files and LibriSpeech which contains a multitude of real human speakers' audio.

**Model Training:**
The following models were trained and evaluated for this task:
- XGBoost:
  - Achieved highest precision, recall, F1 score and accuracy among all models.
  - Selected as the final model due to superior performance.
- LightGBM
- VGG16

**Tech Stack:**
- Librosa & IPython: Used for feature extraction from audio files (e.g., MFCC, zero-crossing rates, chroma features).
- Scikit-learn: Used for XGBoost.
- Streamlit: Developed a web interface where users can upload audio files for real-time prediction.

**Conclusion:**
The XGBoost model was chosen due to its superior overall performance in metrics scores. The project successfully demonstrated how ML models can detect synthetic audio, offering a valuable tool to counteract the potential misuse of AI in audio content creation.
