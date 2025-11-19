import streamlit as st
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import wave
from streamlit_mic_recorder import mic_recorder
import tempfile
import base64
import io
import os 
import time
import librosa
from gtts import gTTS
from io import BytesIO

st.title("Nissay Project Prototype")

# ============================================
# 🆕 ENABLE AUDIO AUTOPLAY (ONE-TIME CLICK)
# ============================================
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = False
    st.session_state.last_prediction = None

if not st.session_state.audio_enabled:
    st.warning("⚠️ Click the button below to enable automatic audio feedback")
    if st.button("🔊 Enable Audio Feedback", type="primary"):
        st.session_state.audio_enabled = True
        st.rerun()
# ============================================

audio_interpreter = tf.lite.Interpreter(model_path="AudioSavedModel/soundclassifier_with_metadata.tflite")
audio_interpreter.allocate_tensors()

audio_input_details = audio_interpreter.get_input_details()
audio_output_details = audio_interpreter.get_output_details()

audio_labels = [label.upper().strip() for label in open("AudioSavedModel/labels.txt")]

# ============================================
# 🆕 TEXT-TO-SPEECH FUNCTION WITH AUTOPLAY
# ============================================
def play_speech_auto(text, placeholder):
    """Generate and automatically play speech in fixed placeholder"""
    try:
        # Create unique key for this prediction
        prediction_key = f"{text}_{time.time()}"
        
        # Only play if it's a new prediction
        if st.session_state.last_prediction != prediction_key:
            st.session_state.last_prediction = prediction_key
            
            tts = gTTS(text=text, lang='en', slow=False)
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # Convert to base64
            audio_base64 = base64.b64encode(fp.read()).decode()
            
            # Create autoplay HTML in the placeholder
            audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
            placeholder.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        placeholder.error(f"Could not generate speech: {e}")
# ============================================

uploaded_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "ogg"], key="audio_uploader")

# Create fixed placeholder for audio feedback at the top
audio_feedback_placeholder = st.empty()

audio_preds = None
if uploaded_audio:
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_audio.read())
            tmp_path = tmp_file.name
        
        # Load audio with librosa (supports more formats)
        audio_data, sr = librosa.load(tmp_path, sr=16000, mono=True)
        
        # Convert to int16 format
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Prepare for model
        input_shape = audio_input_details[0]['shape']
        audio_data = np.expand_dims(audio_data[:input_shape[1]], axis=0).astype(np.float32)

        audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_data)
        audio_interpreter.invoke()
        audio_preds = audio_interpreter.get_tensor(audio_output_details[0]['index'])[0]
        
        # Clean up temp file
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        st.info("Please upload a valid audio file (WAV, MP3, M4A, or OGG format)")

image_model = keras.layers.TFSMLayer(
    "img_savedmodel/model.savedmodel",
    call_endpoint="serving_default"
)
image_labels = [label.strip() for label in open("img_savedmodel/labels.txt")]

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png"], key="image_uploader")

image_preds = None
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    # Run inference
    output = image_model(img_array)
    image_preds = output["sequential_5"].numpy()[0]

# ============================================
# RESULTS WITH AUTOMATIC AUDIO FEEDBACK
# ============================================
if audio_preds is not None and image_preds is not None:
    # Extract only sad and happy
    audio_two = audio_preds[1:]   # drop background noise
    combined_preds = (audio_two + image_preds) / 2
    combined_idx = np.argmax(combined_preds)
    prediction_label = image_labels[combined_idx]
    confidence = float(combined_preds[combined_idx]) * 100
    
    st.subheader("Final Combined Result")
    st.write("Combined Prediction:", prediction_label)
    st.write(f"Combined Confidence: {confidence:.2f}%")
    
    # 🔊 Automatic audio feedback in fixed position
    if st.session_state.audio_enabled:
        play_speech_auto(f"{prediction_label}, {confidence:.0f} percent confidence", audio_feedback_placeholder)
    
elif audio_preds is not None:
    audio_idx = np.argmax(audio_preds)
    prediction_label = audio_labels[audio_idx]
    confidence = audio_preds[audio_idx] * 100
    
    st.success(f"Audio Prediction: {prediction_label}")
    st.info(f"Audio Confidence: {confidence:.2f}%")
    
    # 🔊 Automatic audio feedback in fixed position
    if st.session_state.audio_enabled:
        play_speech_auto(f"{prediction_label}, {confidence:.0f} percent confidence", audio_feedback_placeholder)
    
elif image_preds is not None:
    image_idx = np.argmax(image_preds)
    prediction_label = image_labels[image_idx]
    confidence = float(image_preds[image_idx]) * 100
    
    st.success(f"Image Prediction: {prediction_label}")
    st.info(f"Image Confidence: {confidence:.2f}%")
    
    # 🔊 Automatic audio feedback in fixed position
    if st.session_state.audio_enabled:
        play_speech_auto(f"{prediction_label}, {confidence:.0f} percent confidence", audio_feedback_placeholder)