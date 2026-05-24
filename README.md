📷 Image Caption Generator

An AI-powered web application that automatically generates descriptive captions for uploaded images — with text-to-speech audio playback.

<img width="1919" height="1134" alt="image" src="https://github.com/user-attachments/assets/394545f9-d6b9-4ff5-af33-049ddc55a7c1" />


📖 About the Project
Image Caption Generator is a deep learning project that combines Computer Vision and Natural Language Processing to automatically describe the content of an image in plain English.
Upload any image → the model analyzes it → generates a human-readable caption → reads it aloud using text-to-speech.
Built with:

VGG16 for image feature extraction (CNN)
LSTM for caption generation (RNN)
Flickr8k dataset for training
Streamlit for the interactive web UI
gTTS for audio narration of the caption


🌐 Live Demo

GitHub Repository: https://github.com/kajal9873/Image-Caption-Generator

Deploy on Streamlit Cloud:
https://image-caption-generator-a3zm3gykygwyty9h7uan7k.streamlit.app/

✨ Features

📤 Image Upload — supports JPG, JPEG, PNG formats
🤖 AI Caption Generation — LSTM model trained on 8,000+ images
🔊 Text-to-Speech — generated caption is read aloud using gTTS
🎧 Audio Player — built-in audio playback inside the app
💡 Smart Tips Panel — tells users which image types work best
⚡ Fast Inference — VGG16 feature extraction + LSTM prediction in seconds
📱 Responsive UI — works on desktop and mobile browsers


🧠 How It Works
User uploads image
        ↓
Image resized to 224×224
        ↓
VGG16 extracts 4096-dim feature vector
        ↓
LSTM model takes features + partial caption
        ↓
Predicts next word, repeats until "endseq"
        ↓
Caption cleaned (remove startseq/endseq)
        ↓
gTTS converts caption to audio
        ↓
Caption + Audio displayed to user ✅
Caption Generation (Greedy Search)
The model uses greedy search — at each step it predicts the most likely next word given the image features and the caption generated so far, starting from "startseq" and stopping at "endseq" or max_length = 35 words.

🏗 Model Architecture
Feature Extractor — VGG16 (CNN)

Pre-trained on ImageNet
Last classification layer removed
Output: 4096-dimensional feature vector per image
All 8,091 Flickr8k images pre-processed and saved in features.pkl

Caption Generator — LSTM (RNN)
Image Features (4096) ──→ Dense(256) ──→ ┐
                                          Add ──→ Dense(256, ReLU) ──→ Dense(vocab_size, Softmax)
Partial Caption ──→ Embedding ──→ LSTM(256) ──→ ┘
LayerDetailsInput 1Image features (4096-dim)Dense256 unitsInput 2Sequence (partial caption)Embeddingvocab_size × 256LSTM256 unitsAddMerge image + text featuresDense256 units, ReLUOutput Densevocab_size units, Softmax
Training:

Epochs: 12
Batch size: 32
Optimizer: Adam
Loss: Categorical Crossentropy


📊 Dataset
Flickr8k Dataset

8,091 images of people and animals doing activities
5 captions per image = ~40,000 caption-image pairs
Split: 80% train / 20% test
Captions preprocessed: lowercased, punctuation removed, startseq/endseq tokens added


⚠️ The model works best on outdoor scenes, activities, animals, and sports — as these are the dominant categories in Flickr8k.


📈 Model Performance
Evaluated using BLEU Score on the Flickr8k test set:
MetricScoreBLEU-1 (Unigram)~0.50–0.55BLEU-2 (Bigram)~0.30–0.35

BLEU scores are standard metrics for caption quality. Higher = closer to human-written captions.


🛠 Tech Stack
TechnologyVersionPurposePython3.11Core languageTensorFlow / Keras2.18.0 / 3.8.0Model training & inferenceVGG16ImageNet pretrainedImage feature extractionLSTMCustomCaption sequence generationStreamlit1.41.1Web application frameworkgTTS2.5.4Text-to-speech audioNumPy2.0.2Numerical computationsPillow11.1.0Image loading & processingh5py3.12.1Loading .h5 model filesGoogle Colab—Model training environment

⚠️ Known Limitations
LimitationReasonMay give wrong captions for selfies/facesFlickr8k has few close-up face imagesIndoor photos less accurateDataset is mostly outdoor scenesCartoons/drawings not supportedTrained only on real photographsDark or blurry imagesVGG16 needs clear visual features
Works best with: outdoor scenes, people doing activities (running, playing, swimming), dogs and common animals, sports photos, and well-lit clear images.

📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

<div align="center">
Built with ❤️ using TensorFlow, Streamlit & VGG16
</div>
