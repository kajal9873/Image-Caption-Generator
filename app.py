import gdown
import os

# Download model files if not present
if not os.path.exists('model.h5'):
    gdown.download(
        'https://drive.google.com/uc?id=1AMfzBUzS4CtDpxlaF9hSX4Xg2NlbmpMx',
        'model.h5', quiet=False
    )

if not os.path.exists('tokenizer.pkl'):
    gdown.download(
        'https://drive.google.com/uc?id=12vNGoIYG8_rVj1oFMBYtsu-xw8AKRV6b',
        'tokenizer.pkl', quiet=False
    )

if not os.path.exists('features.pkl'):
    gdown.download(
        'https://drive.google.com/uc?id=1cUmvDKk7HJvTcXcnaQb2yQg1S2EL0x47',
        'features.pkl', quiet=False
    )

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gtts import gTTS

# ✅ Sirf ek baar load hoga
@st.cache_resource
def load_all_models():
    vgg = VGG16()
    vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
    caption_model = load_model('model.h5')
    return vgg, caption_model

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

vgg_model, model = load_all_models()
tokenizer = load_tokenizer()

st.set_page_config(page_title="Caption Generator App", page_icon="📷")
st.title("Image Caption Generator")
st.markdown("Upload an image, and this app will generate a caption using a trained LSTM model.")

with st.expander("💡 Tips for best results — click to read", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.success("""
**✅ Works well with:**
- Outdoor scenes
- People doing activities (running, climbing, playing)
- Dogs and common animals
- Sports and nature photos
- Clear, well-lit images
        """)
    with col2:
        st.error("""
**❌ May give wrong captions for:**
- Selfies / close-up faces
- Indoor photos
- Cartoons or drawings
- Screenshots or memes
- Dark or blurry images
- Abstract or artistic photos
        """)
    st.info("ℹ️ This model was trained on the Flickr8k dataset. Results outside that scope may be inaccurate.")

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

def get_word_from_index(index, tokenizer):
    return next(
        (word for word, idx in tokenizer.word_index.items() if idx == index), None
    )

def predict_caption(model, image_features, tokenizer, max_caption_length=35):
    caption = "startseq"
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = get_word_from_index(predicted_index, tokenizer)
        caption += " " + predicted_word
        if predicted_word is None or predicted_word == "endseq":
            break
    return caption

if uploaded_image is not None:
    with st.spinner("Generating caption..."):
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, *image.shape))
        image = preprocess_input(image)

        image_features = vgg_model.predict(image, verbose=0)

        generated_caption = predict_caption(model, image_features, tokenizer)
        generated_caption = generated_caption.replace("startseq", "").replace("endseq", "").strip()

    st.subheader(generated_caption)

    tts = gTTS(generated_caption, lang='en')
    audio_path = "predicted_caption.mp3"
    tts.save(audio_path)

    st.audio(audio_path, format='audio/mp3')
    st.subheader("Uploaded Image")
    st.image(uploaded_image, width='stretch')
