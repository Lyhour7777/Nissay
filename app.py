import streamlit as st
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import tempfile
import base64
import os 
import time
import librosa
from gtts import gTTS
from io import BytesIO
import pandas as pd
from datetime import datetime
import random

# Page configuration for better accessibility
st.set_page_config(
    page_title="Nissay Emotion Detector",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visual design and accessibility
st.markdown("""
<style>
    /* High contrast and larger fonts */
    .main {
        background-color: #f8f9fa;
    }
    
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .uploadedFile {
        border: 3px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        background-color: white;
    }
    
    /* Large, high-contrast text */
    .big-text {
        font-size: 28px;
        font-weight: bold;
        color: #1f1f1f;
        padding: 20px;
        background: white;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    
    .result-box {
        font-size: 36px;
        font-weight: bold;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .happy-result {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        color: #2d3436;
    }
    
    .sad-result {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
    }
    
    .neutral-result {
        background: linear-gradient(135deg, #dfe6e9 0%, #b2bec3 100%);
        color: #2d3436;
    }
    
    /* Screen reader only content */
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0,0,0,0);
        border: 0;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .history-item {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Title with emoji for visual users
st.markdown("<h1 style='text-align: center; font-size: 48px;'>😊 Nissay Emotion Detector 😢</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 24px; color: #636e72;'>Upload audio or image to detect emotions</p>", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = False
    st.session_state.last_prediction = None
    st.session_state.last_empathetic_msg = None
    st.session_state.audio_queue = []
    st.session_state.conversation_stage = 0
    st.session_state.current_emotion = None
    st.session_state.initial_audio_played = False
    st.session_state.awaiting_choice = False
    st.session_state.current_exercise = {}
    st.session_state.show_resources = False
    st.session_state.history = []
    st.session_state.user_name = ""
    st.session_state.stats = {
        'total_analyses': 0,
        'happy_count': 0,
        'sad_count': 0,
        'audio_analyses': 0,
        'image_analyses': 0,
        'combined_analyses': 0
    }

# ============================================
# SIDEBAR SETTINGS & STATISTICS
# ============================================
with st.sidebar:
    st.header("🎛️ Settings")
    
    # User name input
    st.markdown("### 👤 Personalization")
    user_name = st.text_input(
        "Your name:", 
        value=st.session_state.get('user_name', ''),
        placeholder="Enter your name here", 
        help="Enter your name for a personalized experience",
        key="name_input"
    )
    
    # Update session state with proper fallback
    if user_name and user_name.strip():
        st.session_state.user_name = user_name.strip()
        st.success(f"👋 Welcome, {st.session_state.user_name}!")
    else:
        st.session_state.user_name = ""
        st.info("💡 Enter your name above for personalized responses")
    
    st.divider()
    
    # Audio toggle
    if not st.session_state.audio_enabled:
        st.warning("⚠️ Audio feedback is OFF")
        if st.button("🔊 Enable Audio Feedback", type="primary", help="Turn on automatic voice announcements"):
            st.session_state.audio_enabled = True
            st.rerun()
    else:
        st.success("✅ Audio feedback is ON")
        if st.button("🔇 Disable Audio Feedback", help="Turn off automatic voice announcements"):
            st.session_state.audio_enabled = False
            st.rerun()
    
    st.divider()
    
    # Statistics
    st.markdown("### 📊 Session Statistics")
    stats = st.session_state.stats
    st.metric("Total Analyses", stats['total_analyses'])
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("😊 Happy", stats['happy_count'])
    with col_b:
        st.metric("😢 Sad", stats['sad_count'])
    
    st.markdown("#### Analysis Types")
    st.write(f"🎤 Audio: {stats['audio_analyses']}")
    st.write(f"📸 Image: {stats['image_analyses']}")
    st.write(f"🎯 Combined: {stats['combined_analyses']}")
    
    if st.button("🔄 Reset Statistics"):
        st.session_state.stats = {
            'total_analyses': 0,
            'happy_count': 0,
            'sad_count': 0,
            'audio_analyses': 0,
            'image_analyses': 0,
            'combined_analyses': 0
        }
        st.session_state.history = []
        st.rerun()
    
    st.divider()
    
    # Instructions
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. **Upload audio** (crying, laughing sounds)
    2. **Upload image** (facial expressions)
    3. **Or upload both** for combined analysis
    4. Results will be announced automatically
    5. View history and statistics below
    """)
    
    st.divider()
    
    # Accessibility features
    st.markdown("### ♿ Accessibility Features")
    st.markdown("""
    - ✅ Voice announcements
    - ✅ High contrast design
    - ✅ Large text and buttons
    - ✅ Keyboard navigation
    - ✅ Screen reader compatible
    - ✅ ARIA labels
    """)

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_audio_model():
    interpreter = tf.lite.Interpreter(model_path="AudioSavedModel/soundclassifier_with_metadata.tflite")
    interpreter.allocate_tensors()
    labels = [label.upper().strip() for label in open("AudioSavedModel/labels.txt")]
    return interpreter, labels

@st.cache_resource
def load_image_model():
    model = keras.layers.TFSMLayer("img_savedmodel/model.savedmodel", call_endpoint="serving_default")
    labels = [label.strip() for label in open("img_savedmodel/labels.txt")]
    return model, labels

try:
    audio_interpreter, audio_labels = load_audio_model()
    image_model, image_labels = load_image_model()
    audio_input_details = audio_interpreter.get_input_details()
    audio_output_details = audio_interpreter.get_output_details()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ============================================
# HELPER FUNCTIONS
# ============================================
def play_combined_audio(announcement_text, empathetic_text, placeholder, force_play=False):
    """
    Combine both messages into ONE audio file and play it.
    """
    try:
        # Create unique key for this specific result
        result_key = f"{announcement_text[:30]}_{empathetic_text[:30]}_{time.time()}"
        
        # Only play if this is a new result OR force_play is True
        if force_play or st.session_state.last_prediction != result_key:
            st.session_state.last_prediction = result_key
            
            # Combine both messages with a pause indicator
            if announcement_text:
                combined_text = f"{announcement_text}. ... {empathetic_text}"
            else:
                combined_text = empathetic_text
            
            # Generate single audio file with both messages
            tts = gTTS(text=combined_text, lang='en', slow=False)
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_base64 = base64.b64encode(fp.read()).decode()
            
            # Play the combined audio with unique timestamp to force replay
            timestamp = int(time.time() * 1000)
            audio_html = f"""
            <audio autoplay aria-label="Analysis result and response" id="audio_{timestamp}">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
            placeholder.markdown(audio_html, unsafe_allow_html=True)
            
    except Exception as e:
        placeholder.error(f"Could not generate speech: {e}")

def update_statistics(analysis_type, emotion):
    """Update session statistics"""
    st.session_state.stats['total_analyses'] += 1
    
    if 'happy' in emotion.lower():
        st.session_state.stats['happy_count'] += 1
    elif 'sad' in emotion.lower():
        st.session_state.stats['sad_count'] += 1
    
    if analysis_type == 'audio':
        st.session_state.stats['audio_analyses'] += 1
    elif analysis_type == 'image':
        st.session_state.stats['image_analyses'] += 1
    elif analysis_type == 'combined':
        st.session_state.stats['combined_analyses'] += 1

def add_to_history(analysis_type, emotion, confidence):
    """Add result to history"""
    st.session_state.history.insert(0, {
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'type': analysis_type,
        'emotion': emotion,
        'confidence': confidence
    })
    # Keep only last 10 items
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[:10]

def get_emotion_emoji(emotion):
    """Get appropriate emoji for emotion"""
    emotion_lower = emotion.lower()
    if 'happy' in emotion_lower or 'laugh' in emotion_lower or 'joy' in emotion_lower:
        return "😊"
    elif 'sad' in emotion_lower or 'cry' in emotion_lower:
        return "😢"
    elif 'angry' in emotion_lower or 'mad' in emotion_lower:
        return "😠"
    elif 'surprise' in emotion_lower:
        return "😲"
    elif 'fear' in emotion_lower:
        return "😨"
    else:
        return "😐"

def get_result_class(emotion):
    """Get CSS class for result box"""
    emotion_lower = emotion.lower()
    if 'happy' in emotion_lower or 'laugh' in emotion_lower:
        return "happy-result"
    elif 'sad' in emotion_lower or 'cry' in emotion_lower:
        return "sad-result"
    else:
        return "neutral-result"

def get_empathetic_message(emotion, confidence, analysis_type='general'):
    """Generate empathetic response based on emotion and confidence"""
    user_name = st.session_state.get('user_name', '').strip()
    greeting = user_name if user_name else "friend"
    
    emotion_lower = emotion.lower()
    
    # High confidence responses (> 80%)
    if confidence > 80:
        if 'happy' in emotion_lower or 'joy' in emotion_lower or 'laugh' in emotion_lower:
            responses = [
                f"{greeting}, you know what? Celebrating positive moments is important too and I hear yours. Can you share what contributed to that feeling?",
                f"Aww, you look really warm and kind here, {greeting}. It's nice seeing this moment. What made you smile?",
                f"{greeting}, I hear a positive tone in your voice. What helped create this positive moment?",
                f"That's wonderful, {greeting}! Noticing positive emotions can be grounding. How can you keep this feeling going?",
                f"Such a beautiful moment, {greeting}. Let yourself feel proud and happy — you deserve to enjoy this fully.",
            ]
            return random.choice(responses)
        
        elif 'sad' in emotion_lower or 'cry' in emotion_lower:
            sad_opening_scripts = [
                f"Hey {greeting}, I'm sensing some heaviness in your voice and expression. It seems like things might be tough right now. If you feel ready, I'm here to listen to whatever's on your mind.",
                f"You look a little tired today, {greeting}. Have you been getting enough rest? No pressure to explain anything — but I'm here if you want to talk about what's been heavy for you lately.",
                f"It's totally okay to feel this way, {greeting}. Sadness doesn't mean something is wrong with you — it means you're human. We can explore it together at your pace."
            ]
            
            st.session_state.current_emotion = 'sad'
            st.session_state.conversation_stage = 0
            
            return random.choice(sad_opening_scripts)
    
    # Medium confidence - mixed emotions (40-80%)
    elif 40 <= confidence <= 80:
        mixed_responses = [
            f"I'm having a little trouble reading the situation, {greeting}. You seem a bit mixed. Do you feel more sad or happy right now?",
            f"I'm sensing a mix of emotions from you right now, {greeting}, almost like things are a bit complex. Sometimes we can feel hopeful and heavy at the same exact time. That is completely normal.",
            f"If you had to name the strongest feeling you have right now, {greeting}, even if it's just by 1%, what would it be?",
            f"I'm having a little trouble reading the signals clearly, {greeting}. Let's skip the labels for a second. On a scale of 1 to 10, how much energy do you feel you have right now?"
        ]
        return random.choice(mixed_responses)
    
    # Low confidence responses (< 40%)
    else:
        return f"Thank you for sharing, {greeting}. I'm here to listen if you'd like to tell me more about how you're feeling."

def get_conversation_followup(user_response):
    """
    Multi-stage conversation flow based on user's emotional state.
    """
    user_name = st.session_state.get('user_name', '').strip()
    greeting = user_name if user_name else "friend"
    
    current_emotion = st.session_state.get('current_emotion', None)
    stage = st.session_state.get('conversation_stage', 0)
    
    # Only continue conversation flow if emotion is sad
    if current_emotion != 'sad':
        return None
    
    # Increment stage
    st.session_state.conversation_stage = stage + 1
    new_stage = st.session_state.conversation_stage
    
    # Stage 1: After initial response - Ask about duration
    if new_stage == 1:
        return f"Thanks for sharing that with me, {greeting}. How long have you been feeling this way? Has it been a few hours, days, or longer?"
    
    # Stage 2: After duration - Ask about triggers
    elif new_stage == 2:
        return f"I hear you, {greeting}. Sometimes it helps to understand what might have triggered these feelings. Can you think of anything specific that happened recently, or does it feel more general?"
    
    # Stage 3: After triggers - Ask about support system
    elif new_stage == 3:
        return f"That makes sense, {greeting}. During difficult times, it can help to have support. Is there someone you trust that you've been able to talk to about this, or have you been managing these feelings on your own?"
    
    # Stage 4: After support system - Offer coping strategies
    elif new_stage == 4:
        return f"Thank you for being open with me, {greeting}. Would you like to try a quick grounding exercise together, or would you prefer some resources for additional support?"
    
    # Stage 5: Final stage - Offer resources or exercises
    elif new_stage == 5:
        return f"I'm here with you, {greeting}. Remember that what you're feeling is valid, and it's okay to ask for help. Would you like me to share some mental health resources, or shall we try a simple breathing exercise?"
    
    # Beyond stage 5 - Reset or offer resources
    else:
        st.session_state.conversation_stage = 5
        return f"You've been very brave in sharing, {greeting}. I want you to know that professional support can make a real difference. Let me share some resources that might help."


def show_mental_health_resources():
    """Display mental health resources and crisis hotlines"""
    st.markdown("## 🏥 Mental Health Resources")
    
    st.markdown("""
    <div class='metric-card' style='background: #fff3cd; border-left: 5px solid #ffc107;'>
        <h3 style='color: #856404;'>🚨 Crisis Support - Available 24/7</h3>
        <ul style='font-size: 18px; line-height: 2.0;'>
            <li><strong>National Suicide Prevention Lifeline:</strong> 988 (US)</li>
            <li><strong>Crisis Text Line:</strong> Text HOME to 741741 (US)</li>
            <li><strong>International Association for Suicide Prevention:</strong> <a href='https://www.iasp.info/resources/Crisis_Centres/' target='_blank'>Find your country's helpline</a></li>
            <li><strong>Emergency Services:</strong> 911 (US) or your local emergency number</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='metric-card' style='background: #d1ecf1; border-left: 5px solid #17a2b8; margin-top: 20px;'>
        <h3 style='color: #0c5460;'>💙 Mental Health Support</h3>
        <ul style='font-size: 18px; line-height: 2.0;'>
            <li><strong>NAMI HelpLine:</strong> 1-800-950-NAMI (6264) or text "HelpLine" to 62640</li>
            <li><strong>SAMHSA National Helpline:</strong> 1-800-662-4357 (substance abuse & mental health)</li>
            <li><strong>Therapy Resources:</strong> 
                <ul>
                    <li><a href='https://www.psychologytoday.com/us/therapists' target='_blank'>Psychology Today - Find a Therapist</a></li>
                    <li><a href='https://www.betterhelp.com/' target='_blank'>BetterHelp - Online Therapy</a></li>
                    <li><a href='https://www.talkspace.com/' target='_blank'>Talkspace - Online Therapy</a></li>
                </ul>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='metric-card' style='background: #d4edda; border-left: 5px solid #28a745; margin-top: 20px;'>
        <h3 style='color: #155724;'>🧘 Self-Care Resources</h3>
        <ul style='font-size: 18px; line-height: 2.0;'>
            <li><strong>Mindfulness Apps:</strong> Headspace, Calm, Insight Timer</li>
            <li><strong>Crisis Management:</strong> <a href='https://www.suicidepreventionlifeline.org/' target='_blank'>988 Lifeline Website</a></li>
            <li><strong>Mental Health Education:</strong> <a href='https://www.nami.org/' target='_blank'>NAMI.org</a></li>
            <li><strong>Peer Support:</strong> <a href='https://www.7cups.com/' target='_blank'>7 Cups - Free Emotional Support</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 Remember: Seeking help is a sign of strength, not weakness. You deserve support.")


def show_breathing_exercise():
    """Interactive breathing exercise to help with anxiety/stress"""
    st.markdown("## 🌬️ Guided Breathing Exercise")
    
    st.markdown("""
    <div class='metric-card' style='background: #e7f3ff; border-left: 5px solid #2196F3;'>
        <h3 style='color: #0d47a1;'>Box Breathing Technique</h3>
        <p style='font-size: 18px; line-height: 1.8;'>
            This simple technique can help calm your nervous system and reduce stress.
            Follow along with these steps:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card' style='text-align: center; background: #e3f2fd;'>
            <h2>1️⃣</h2>
            <p style='font-size: 20px;'><strong>Breathe IN</strong></p>
            <p style='font-size: 48px; color: #1976d2;'>4</p>
            <p>seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card' style='text-align: center; background: #fff3e0;'>
            <h2>2️⃣</h2>
            <p style='font-size: 20px;'><strong>HOLD</strong></p>
            <p style='font-size: 48px; color: #f57c00;'>4</p>
            <p>seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card' style='text-align: center; background: #f3e5f5;'>
            <h2>3️⃣</h2>
            <p style='font-size: 20px;'><strong>Breathe OUT</strong></p>
            <p style='font-size: 48px; color: #7b1fa2;'>4</p>
            <p>seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card' style='text-align: center; background: #e8f5e9;'>
            <h2>4️⃣</h2>
            <p style='font-size: 20px;'><strong>HOLD</strong></p>
            <p style='font-size: 48px; color: #388e3c;'>4</p>
            <p>seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='metric-card' style='background: #fff9c4; margin-top: 20px;'>
        <p style='font-size: 18px; line-height: 1.8;'>
            💡 <strong>Instructions:</strong> Find a comfortable position. 
            Repeat this cycle 4-5 times, focusing only on your breath. 
            Notice how your body feels with each cycle.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("✅ Done with Exercise", key="finish_breathing"):
        st.success("✅ Great job! How do you feel now?")
        st.session_state.conversation_stage = 5


def show_grounding_exercise():
    """5-4-3-2-1 grounding exercise for anxiety"""
    st.markdown("## 🌍 Grounding Exercise")
    
    st.markdown("""
    <div class='metric-card' style='background: #e8f5e9; border-left: 5px solid #4caf50;'>
        <h3 style='color: #1b5e20;'>5-4-3-2-1 Technique</h3>
        <p style='font-size: 18px; line-height: 1.8;'>
            This exercise helps bring you back to the present moment by engaging your senses.
            Take your time with each step:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    exercises = [
        ("👀 5 Things You Can SEE", "Look around and name 5 things you can see right now", "#e3f2fd"),
        ("✋ 4 Things You Can TOUCH", "Notice 4 things you can feel (texture, temperature)", "#fff3e0"),
        ("👂 3 Things You Can HEAR", "Listen carefully for 3 sounds around you", "#f3e5f5"),
        ("👃 2 Things You Can SMELL", "Identify 2 scents in your environment", "#ffebee"),
        ("👅 1 Thing You Can TASTE", "Notice 1 taste in your mouth right now", "#e8f5e9")
    ]
    
    for title, description, color in exercises:
        st.markdown(f"""
        <div class='metric-card' style='background: {color}; margin: 15px 0;'>
            <h4 style='font-size: 22px;'>{title}</h4>
            <p style='font-size: 18px;'>{description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("✅ Completed Exercise", key="finish_grounding"):
        st.success("✅ Well done! Grounding exercises can help manage overwhelming feelings.")
        st.session_state.conversation_stage = 5

# ============================================
# UPLOAD SECTIONS
# ============================================
audio_feedback_placeholder = st.empty()

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='big-text'>🎤 Audio Upload</div>", unsafe_allow_html=True)
    uploaded_audio = st.file_uploader(
        "Upload audio file (crying, laughing, speaking)",
        type=["wav", "mp3", "m4a", "ogg"],
        key="audio_uploader",
        help="Upload an audio file containing emotional sounds"
    )
    
    if uploaded_audio:
        st.audio(uploaded_audio, format='audio/wav')

with col2:
    st.markdown("<div class='big-text'>📸 Image Upload</div>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "Upload image (facial expression)",
        type=["jpg", "png", "jpeg"],
        key="image_uploader",
        help="Upload an image showing facial expression"
    )

# ============================================
# PROCESS AUDIO
# ============================================
audio_preds = None
if uploaded_audio:
    with st.spinner("🎵 Analyzing audio..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_audio.read())
                tmp_path = tmp_file.name
            
            audio_data, sr = librosa.load(tmp_path, sr=16000, mono=True)
            audio_data = (audio_data * 32767).astype(np.int16)
            
            input_shape = audio_input_details[0]['shape']
            audio_data = np.expand_dims(audio_data[:input_shape[1]], axis=0).astype(np.float32)

            audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_data)
            audio_interpreter.invoke()
            audio_preds = audio_interpreter.get_tensor(audio_output_details[0]['index'])[0]
            
            os.unlink(tmp_path)
            st.success("✅ Audio processed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
            audio_preds = None

# ============================================
# PROCESS IMAGE
# ============================================
image_preds = None
if uploaded_image:
    with st.spinner("🖼️ Analyzing image..."):
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)

            img = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

            output = image_model(img_array)
            image_preds = output["sequential_5"].numpy()[0]
            st.success("✅ Image processed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            image_preds = None

# ============================================
# COMBINED ANALYSIS & RESULTS
# ============================================
if audio_preds is not None or image_preds is not None:
    st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 36px;'>📊 Analysis Results</h2>", unsafe_allow_html=True)
    
    # Determine analysis type
    if audio_preds is not None and image_preds is not None:
        analysis_type = 'combined'
        st.info("🎯 Combined Analysis: Audio + Image")
    elif audio_preds is not None:
        analysis_type = 'audio'
        st.info("🎤 Audio-only Analysis")
    else:
        analysis_type = 'image'
        st.info("📸 Image-only Analysis")
    
    # Calculate combined predictions
    if audio_preds is not None and image_preds is not None:
        # Average predictions from both modalities
        combined_preds = (audio_preds + image_preds) / 2
        final_preds = combined_preds
        labels_to_use = audio_labels if len(audio_labels) == len(combined_preds) else image_labels
    elif audio_preds is not None:
        final_preds = audio_preds
        labels_to_use = audio_labels
    else:
        final_preds = image_preds
        labels_to_use = image_labels
    
    # Get top prediction
    top_idx = np.argmax(final_preds)
    confidence = float(final_preds[top_idx] * 100)
    predicted_emotion = labels_to_use[top_idx]
    
    # Update statistics and history
    update_statistics(analysis_type, predicted_emotion)
    add_to_history(analysis_type, predicted_emotion, confidence)
    
    # Display main result
    emoji = get_emotion_emoji(predicted_emotion)
    result_class = get_result_class(predicted_emotion)
    
    st.markdown(f"""
    <div class='result-box {result_class}' role='status' aria-live='polite'>
        <div style='font-size: 72px; margin-bottom: 20px;'>{emoji}</div>
        <div style='font-size: 42px; margin-bottom: 10px;'>{predicted_emotion}</div>
        <div style='font-size: 28px;'>Confidence: {confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed predictions
    with st.expander("📈 View Detailed Predictions", expanded=False):
        pred_data = []
        for i, label in enumerate(labels_to_use):
            pred_data.append({
                'Emotion': label,
                'Confidence': f"{final_preds[i] * 100:.2f}%",
                'Score': final_preds[i]
            })
        
        df = pd.DataFrame(pred_data).sort_values('Score', ascending=False)
        st.dataframe(df[['Emotion', 'Confidence']], use_container_width=True, hide_index=True)
        
        # Visual bar chart
        st.bar_chart(df.set_index('Emotion')['Score'])
    
    # Generate empathetic response
    empathetic_msg = get_empathetic_message(predicted_emotion, confidence, analysis_type)
    
    # Display empathetic message
    st.markdown(f"""
    <div class='metric-card' style='background: #e8f5e9; border-left: 5px solid #4caf50; margin: 20px 0;'>
        <h3 style='color: #2e7d32; font-size: 24px;'>💬 Empathetic Response</h3>
        <p style='font-size: 20px; line-height: 1.8; color: #1b5e20;'>{empathetic_msg}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Play audio announcement if enabled
    if st.session_state.audio_enabled:
        announcement = f"Analysis complete. Detected emotion: {predicted_emotion}. Confidence: {int(confidence)} percent."
        play_combined_audio(announcement, empathetic_msg, audio_feedback_placeholder)
    
    # ============================================
    # CONVERSATIONAL FLOW - SAD EMOTION SUPPORT
    # ============================================
    if 'sad' in predicted_emotion.lower() or 'cry' in predicted_emotion.lower():
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("## 💙 Let's Talk About It")
        
        # Text input for user response
        user_response = st.text_area(
            "Share your thoughts (optional):",
            placeholder="You can share as much or as little as you'd like...",
            height=120,
            key=f"response_input_{st.session_state.conversation_stage}"
        )
        
        if st.button("📤 Send Response", key=f"send_btn_{st.session_state.conversation_stage}"):
            if user_response.strip():
                # Display user's message
                st.markdown(f"""
                <div class='metric-card' style='background: #e3f2fd; margin: 15px 0;'>
                    <p style='font-size: 18px;'><strong>You:</strong> {user_response}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get follow-up response
                followup = get_conversation_followup(user_response)
                
                if followup:
                    st.markdown(f"""
                    <div class='metric-card' style='background: #f3e5f5; margin: 15px 0;'>
                        <p style='font-size: 18px;'><strong>Support Assistant:</strong> {followup}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Play audio for follow-up if enabled
                    if st.session_state.audio_enabled:
                        play_combined_audio("", followup, audio_feedback_placeholder, force_play=True)
                
                # Offer resources at appropriate stage
                if st.session_state.conversation_stage >= 4:
                    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button("🌬️ Try Breathing Exercise", key="breathing_btn", use_container_width=True):
                            show_breathing_exercise()
                    
                    with col_b:
                        if st.button("🌍 Try Grounding Exercise", key="grounding_btn", use_container_width=True):
                            show_grounding_exercise()
                    
                    if st.button("🏥 View Mental Health Resources", key="resources_btn", use_container_width=True):
                        show_mental_health_resources()
            else:
                st.warning("⚠️ Please enter a response before sending.")

# ============================================
# ANALYSIS HISTORY
# ============================================
if st.session_state.history:
    st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)
    st.markdown("## 📜 Recent Analysis History")
    
    for item in st.session_state.history:
        emoji = get_emotion_emoji(item['emotion'])
        type_emoji = "🎤" if item['type'] == 'audio' else "📸" if item['type'] == 'image' else "🎯"
        
        st.markdown(f"""
        <div class='history-item'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <span style='font-size: 24px;'>{emoji} {type_emoji}</span>
                    <strong style='font-size: 20px; margin-left: 10px;'>{item['emotion']}</strong>
                    <span style='color: #636e72; margin-left: 10px;'>({item['confidence']:.1f}%)</span>
                </div>
                <div style='color: #636e72;'>{item['timestamp']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #636e72; padding: 20px;'>
    <p style='font-size: 18px;'>💙 <strong>Nissay Emotion Detector</strong> - Supporting Your Emotional Wellbeing</p>
    <p style='font-size: 16px;'>This tool is designed to help identify emotions, but it is not a substitute for professional mental health support.</p>
    <p style='font-size: 16px;'>If you're experiencing a crisis, please contact emergency services or a mental health professional immediately.</p>
    <p style='font-size: 14px; margin-top: 20px;'>Crisis Hotline (US): 988 | Crisis Text Line: Text HOME to 741741</p>
</div>
""", unsafe_allow_html=True)

# Screen reader announcement for new results
if audio_preds is not None or image_preds is not None:
    st.markdown(f"""
    <div class='sr-only' role='alert' aria-live='assertive'>
        Analysis complete. Detected {predicted_emotion} with {int(confidence)} percent confidence.
    </div>
    """, unsafe_allow_html=True)