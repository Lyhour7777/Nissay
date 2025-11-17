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

audio_interpreter = tf.lite.Interpreter(model_path="AudioSavedModel/soundclassifier_with_metadata.tflite")
audio_interpreter.allocate_tensors()

audio_input_details = audio_interpreter.get_input_details()
audio_output_details = audio_interpreter.get_output_details()

audio_labels = [label.upper().strip() for label in open("AudioSavedModel/labels.txt")]

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

#         input_shape = audio_input_details[0]['shape']
#         audio_data = np.expand_dims(audio_data[:input_shape[1]], axis=0).astype(np.float32)

#         audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_data)
#         # interpreter.set_tensor(input_details['index'], audio_data)
#         audio_interpreter.invoke()
#         audio_preds = audio_interpreter.get_tensor(audio_output_details[0]['index'])[0]

#         idx = np.argmax(audio_preds)
#         st.success(f"Prediction (Uploaded): {audio_labels[idx]}")
#         st.info(f"Confidence: {audio_preds[idx]}")

    

uploaded_audio = st.file_uploader("Upload WAV audio", type=["wav"], key="audio_uploader")

audio_preds = None
if uploaded_audio:
    wav = wave.open(uploaded_audio, 'rb')
    frames = wav.readframes(wav.getnframes())
    audio_data = np.frombuffer(frames, dtype=np.int16)

    input_shape = audio_input_details[0]['shape']
    audio_data = np.expand_dims(audio_data[:input_shape[1]], axis=0).astype(np.float32)

    audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_data)
    audio_interpreter.invoke()
    audio_preds = audio_interpreter.get_tensor(audio_output_details[0]['index'])[0]



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


if audio_preds is not None and image_preds is not None:
    # Extract only sad and happy
    audio_two = audio_preds[1:]   # drop background noise
    combined_preds = (audio_two + image_preds) / 2
    combined_idx = np.argmax(combined_preds)
    st.subheader("Final Combined Result")
    st.write("Combined Prediction:", image_labels[combined_idx])
    st.write(f"Combined Confidence: {float(combined_preds[combined_idx]) *100 :.2f}%")
elif audio_preds is not None:
    audio_idx = np.argmax(audio_preds)
    st.success(f"Audio Prediction: {audio_labels[audio_idx]}")
    st.info(f"Audio Confidence: {audio_preds[audio_idx] *100 :.2f}%")
elif image_preds is not None:
    image_idx = np.argmax(image_preds)
    st.success(f"Image Prediction: {image_labels[image_idx]}")
    st.info(f"Image Confidence: {float(image_preds[image_idx]) *100 :.2f}%")