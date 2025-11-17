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

st.title("Nissay Project Prototype")

interpreter = tf.lite.Interpreter(model_path="AudioSavedModel/soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = [label.strip() for label in open("AudioSavedModel/labels.txt")]

# st.subheader("Record Audio OR Upload WAV File")
# # Record audio
# recorded_audio = mic_recorder(
#     start_prompt="Start Recording",
#     stop_prompt="Stop Recording",
#     key="recorder"
# )

# if recorded_audio:

#     raw = recorded_audio["bytes"]

#     # -----------------------------
#     # Convert raw data to numpy int16
#     # -----------------------------
#     if isinstance(raw, str) and raw.startswith("data:audio"):
#         # Remove header and decode Base64
#         header, encoded = raw.split(",", 1)
#         pcm_bytes = np.frombuffer(base64.b64decode(encoded), dtype=np.int16)
#     elif isinstance(raw, str):
#         pcm_bytes = np.frombuffer(base64.b64decode(raw), dtype=np.int16)
#     else:
#         pcm_bytes = np.frombuffer(raw, dtype=np.int16)  # already bytes

#     # -----------------------------
#     # Save WAV with proper RIFF header
#     # -----------------------------
#     os.makedirs("recordings", exist_ok=True)
#     file_path = f"recordings/record_{int(time.time())}.wav"
#     wav = wave.open(file_path, 'rb')

#     audio_data = librosa.resample(wav.astype(np.float32), orig_sr=44100, target_sr=16000)

#     with wave.open(file_path, "wb") as wf:
#         wf.setnchannels(1)         # Mono
#         wf.setsampwidth(2)         # int16 = 2 bytes
#         wf.setframerate(16000)     # or your desired sample rate
#         wf.writeframes(pcm_bytes.tobytes())

#     st.success(f"Saved valid WAV to: {file_path}")
#     st.audio(file_path)
 
#     with open(file_path, "rb") as f:
#         wav = wave.open(f, 'rb')
#         frames = wav.readframes(wav.getnframes())
#         audio_data = np.frombuffer(frames, dtype=np.int16)

#         input_shape = input_details[0]['shape']
#         audio_data = np.expand_dims(audio_data[:input_shape[1]], axis=0).astype(np.float32)

#         interpreter.set_tensor(input_details[0]['index'], audio_data)
#         # interpreter.set_tensor(input_details['index'], audio_data)
#         interpreter.invoke()
#         preds = interpreter.get_tensor(output_details[0]['index'])[0]

#         idx = np.argmax(preds)
#         st.success(f"Prediction (Uploaded): {labels[idx]}")
#         st.info(f"Confidence: {preds[idx]}")

    

uploaded_audio = st.file_uploader("Upload WAV audio", type=["wav"])

if uploaded_audio:
    wav = wave.open(uploaded_audio, 'rb')
    frames = wav.readframes(wav.getnframes())
    audio_data = np.frombuffer(frames, dtype=np.int16)

    input_shape = input_details[0]['shape']
    audio_data = np.expand_dims(audio_data[:input_shape[1]], axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    idx = np.argmax(preds)
    st.success(f"Prediction (Uploaded): {labels[idx]}")
    st.info(f"Confidence: {preds[idx]}")




# Load SavedModel using TFSMLayer
model = keras.layers.TFSMLayer(
    "img_savedmodel/model.savedmodel",
    call_endpoint="serving_default"
)

labels = [label.strip() for label in open("img_savedmodel/labels.txt")]

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    # Run inference
    output = model(img_array)
    preds = output["sequential_5"].numpy()[0]

    # Show results
    idx = np.argmax(preds)
    st.write("Prediction:", labels[idx])
    st.write("Confidence:", float(preds[idx]))