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
    st.session_state.awaiting_choice = False  # ADD THIS
    st.session_state.current_exercise = {}  # ADD THIS
    st.session_state.show_resources = False  # ADD THIS
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
# HELPER FUNCTIONS - FIXED AUDIO TIMING
# ============================================
def play_combined_audio(announcement_text, empathetic_text, placeholder, force_play=False):
    """
    Combine both messages into ONE audio file and play it.
    This is the most reliable approach for sequential playback.
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
    """Generate empathetic response based on emotion and confidence with multi-stage conversation"""
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
                f"You look a little tired today, {greeting}. Have you been getting enough rest? No pressure to explain anything — but I'm here if you want to talk about what's been heavy for you lately."
            ]
            return random.choice(responses)
        
        elif 'sad' in emotion_lower or 'cry' in emotion_lower:
            # Three different opening scripts for sadness
            sad_opening_scripts = [
                # Script 1: Direct empathy with grounding
                f"Hey {greeting}, I'm sensing some heaviness in your voice and expression. It seems like things might be tough right now. If you feel ready, I'm here to listen to whatever's on your mind.",
                
                # Script 2: Gentle check-in with gratitude focus
                f"You look a little tired today, {greeting}. Have you been getting enough rest? No pressure to explain anything — but I'm here if you want to talk about what's been heavy for you lately.",
                
                # Script 3: Normalizing sadness with support
                f"It's totally okay to feel this way, {greeting}. Sadness doesn't mean something is wrong with you — it means you're human. We can explore it together at your pace."
            ]
            
            # Store which script variant for later stages
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
    """Get the next stage of conversation based on user response"""
    user_name = st.session_state.get('user_name', '').strip()
    greeting = user_name if user_name else "Meymey"
    
    if st.session_state.current_emotion != 'sad':
        return None
    
    stage = st.session_state.conversation_stage
    
    # Stage 0 → 1: Acknowledgment
    if stage == 0:
        acknowledgment_responses = [
            f"Thank you for sharing that, {greeting}. Let's take this step by step. Maybe today we can explore one small thing that could help lighten your load — wanna try it?",
            f"Thank you for sharing that with me, {greeting}. I can hear how heavy that has been for you. You don't have to carry this alone — we can work through it together, one small step at a time.",
            f"What you're feeling makes a lot of sense, {greeting}. Anyone in your situation might feel the same. Let's think about one gentle thing you can offer yourself today — something that feels kind, not overwhelming.",
            f"I really appreciate your honesty, {greeting}. Even talking about this shows strength, even if it doesn't feel like it right now. Let's explore what you might need next — comfort, clarity, or just a moment to breathe. I'm right here with you."
        ]
        st.session_state.conversation_stage = 1
        return random.choice(acknowledgment_responses)
    
    # Stage 1 → 2: Pick ONE exercise and store it
    elif stage == 1:
        exercises = [
            {
                'name': 'grounding',
                'step1': f"Alright {greeting}, let's take it step by step. Grounding exercises help you focus on the present moment, calm your mind, and feel more stable and in control. First, take a slow breath in... and gently breathe out. Now, look around you and find something you can see or feel — just one thing {greeting}. It could be anything... a chair, a bag, the sky outside.",
                'step2': f"Good, now notice one thing you can touch or feel. Maybe your clothes... or your hands.",
                'step3': f"You're doing great, {greeting}! When you're ready take one more slow breath with me... In... And Out."
            },
            {
                'name': 'gratitude',
                'step1': f"Hey {greeting}, let's take a tiny gratitude adventure! First, take a slow, deep breath in... and sigh it out. Now, in your mind, spot one small thing today that made you smile — even a teeny little thing. A warm cup of tea, the sun peeking through the window, anything!",
                'step2': f"Nice! Take another slow breath. Picture that happy thing in your mind, and feel a tiny wave of warmth or joy spreading from it. Let it sit there for a moment.",
                'step3': f"Almost there! Take one more slow breath in... and let it out. Keep holding onto that little sparkle of happiness."
            },
            {
                'name': '5-4-3-2-1',
                'step1': f"Okay {greeting}, ready for a little grounding journey? 🌿 Take a slow breath in... and let it out. Now, let's spot 5 things you can see around you. Big, small, anything catches your eye!",
                'step2': f"Awesome! Now touch 4 things around you — maybe your clothes, your hands, or the chair. Really feel them with curiosity, like a detective exploring your own world.",
                'step3': f"Next, listen up — 3 things you can hear, then 2 things you can smell, and finally 1 thing you can taste. Take a slow, calming breath in... and out. You're anchored in the here and now."
            },
            {
                'name': 'visualization',
                'step1': f"Hey {greeting}, let's imagine a little happy scene in the future. Take a deep, calming breath in... and out. Picture a place or moment where you feel safe, proud, or joyful. Make it vivid in your mind!",
                'step2': f"Awesome! Notice all the little details — what do you see, hear, or feel? Imagine yourself right there, soaking in all the good vibes.",
                'step3': f"Now, take one tiny step toward that positive moment. Could be sending a kind message, finishing a small task, or just smiling at yourself. Breathe in... and out. You're planting seeds of hope!"
            }
        ]
        
        # Pick ONE exercise and save it
        chosen_exercise = random.choice(exercises)
        st.session_state.current_exercise = chosen_exercise
        st.session_state.conversation_stage = 2
        return chosen_exercise['step1']
    
    # Stage 2 → 3: Continue SAME exercise step 2
    elif stage == 2:
        exercise = st.session_state.get('current_exercise', {})
        st.session_state.conversation_stage = 3
        return exercise.get('step2', '')
    
    # Stage 3 → 4: Continue SAME exercise step 3
    elif stage == 3:
        exercise = st.session_state.get('current_exercise', {})
        st.session_state.conversation_stage = 4
        return exercise.get('step3', '')
    
    # Stage 4 → 5: Completion
    elif stage == 4:
        completion_messages = [
            f"You did really well, {greeting}. Even taking a small moment for yourself is a big step. I'm proud of you for trying.",
            f"You did wonderfully! Even noticing one little joy is a big win. I'm proud of you for giving yourself this moment.",
            f"Look at you! Even doing a tiny part of this journey helps you feel steady and present. I'm proud of you for exploring this moment.",
            f"Look at that! Even imagining one little positive step is meaningful. I'm proud of you for giving yourself this moment."
        ]
        st.session_state.conversation_stage = 5
        return random.choice(completion_messages)
    
    # Stage 5 → 6: Offer resources
    elif stage == 5:
        st.session_state.conversation_stage = 6
        return f"If you ever feel like you need more support, {greeting}, I can share some trusted mental-health places in Cambodia where real professionals listen and help. Just let me know if you'd like to see them."
    
    return None

def show_mental_health_resources():
    """Display mental health resources for Cambodia"""
    st.markdown("### 🏥 Mental Health Resources in Cambodia")
    
    with st.expander("1. Hospitals & Public Services", expanded=True):
        st.markdown("""
        **មន្ទីរពេទ្យរុស្សី (Russian Hospital)**  
        📞 092 288 7888  
        🏥 State-run hospital, severe psychiatric conditions  
        💰 Free / Public service  
        👥 For: General public
        
        **បណ្ដាញទូរស័ព្ទ ជំនួយកុមារនៅកម្ពុជា (Child Helpline Cambodia)**  
        📞 1280  
        🕒 24/7 phone consultation  
        💰 Free  
        👥 For: Children, youth, and general public
        """)
    
    with st.expander("2. Private Clinics & Centers"):
        st.markdown("""
        **ខ្លឹម (Khlem Counseling Clinic)**  
        📧 Khlem.discovery@gmail.com / 010 392 507  
        🏥 Private clinic, family counseling  
        💰 50% off for students  
        👥 For: Students
        
        **មជ្ឈមណ្ឌលស្នេហា (Love Center)**  
        📞 097 990 3333  
        🏥 General mental health counseling  
        💰 Fee-based  
        👥 For: General public
        
        **សំបុក សាយខូឡូជី (Sambok Psychology)**  
        📞 077 566 110  
        💰 $100+ per session  
        👥 For: General public
        """)
    
    with st.expander("3. NGO & Specialized Services"):
        st.markdown("""
        **អង្គការធីភីអូ កម្ពុជា (TPO Cambodia)**  
        📞 095 777 004  
        🏥 Supports general mental health in society  
        💰 Donation-based  
        👥 For: General public
        
        **ការីតាស — កម្មវិធីពន្លកនៃសេចក្ដីសង្ឃឹម (Karitas / Hope Program)**  
        📞 085 222 728  
        💰 Free / Public  
        👥 For: General public
        
        **មជ្ឈមណ្ឌល ផ្ទះយើង (Our Home Center)**  
        📞 095 614 956  
        💰 Free  
        👥 For: Factory workers only
        """)
    
    st.info("💙 Remember: Seeking help is a sign of strength, not weakness.")

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
# DISPLAY RESULTS WITH EMPATHY - FIXED TIMING
# ============================================
st.divider()

if audio_preds is not None and image_preds is not None:
    # Combined prediction
    audio_two = audio_preds[1:]
    combined_preds = (audio_two + image_preds) / 2
    combined_idx = np.argmax(combined_preds)
    prediction_label = image_labels[combined_idx]
    confidence = float(combined_preds[combined_idx]) * 100
    
    st.markdown("## 🎯 Combined Analysis Result")
    
    result_class = get_result_class(prediction_label)
    emoji = get_emotion_emoji(prediction_label)
    
    st.markdown(f"""
    <div class='result-box {result_class}' role='status' aria-live='polite'>
        {emoji} {prediction_label.upper()} {emoji}<br>
        Confidence: {confidence:.1f}%
    </div>
    """, unsafe_allow_html=True)
    
    # Add empathetic message - VISIBLE ON SCREEN
    empathetic_msg = get_empathetic_message(prediction_label, confidence, 'combined')
    if empathetic_msg:
        st.markdown("### 💬 Empathetic Response")
        st.markdown(f"""
        <div class='metric-card' style='border-left: 5px solid #4CAF50; background: #f0f9ff;'>
            <p style='font-size: 22px; color: #1e3a8a; line-height: 1.8; font-weight: 500;'>
                💭 {empathetic_msg}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add text input for user response with conversation flow
        user_response = st.text_area(
            "Share your thoughts (optional):",
            placeholder="What's on your mind?",
            height=100,
            key="response_combined"
        )
        
        if user_response:
            user_name = st.session_state.get('user_name', '').strip()
            display_name = user_name if user_name else "friend"
            st.success(f"✅ Thank you for sharing, {display_name}. Your feelings are valid and heard.")
            
            # Get conversation followup if in sad emotion flow
            followup = get_conversation_followup(user_response)
            if followup:
                st.markdown("### 💬 Next Step")
                st.markdown(f"""
                <div class='metric-card' style='border-left: 5px solid #9b59b6; background: #f3e5f5;'>
                    <p style='font-size: 20px; color: #4a148c; line-height: 1.8;'>
                        {followup}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Play followup audio - ALWAYS play for new conversation steps
                if st.session_state.audio_enabled:
                    followup_placeholder = st.empty()
                    play_combined_audio("", followup, followup_placeholder, force_play=True)
                
                # Show option to continue or get resources
                if st.session_state.conversation_stage == 5:
                    col_x, col_y = st.columns(2)
                    with col_x:
                        if st.button("🏥 Get Mental Health Resources", key="resources_combined"):
                            show_mental_health_resources()
                    with col_y:
                        if st.button("🔄 Start New Conversation", key="reset_combined"):
                            st.session_state.conversation_stage = 0
                            st.session_state.current_emotion = None
                            st.session_state.initial_audio_played = False  # Reset audio flag
                            st.rerun()
    
    # Show detailed breakdown
    with st.expander("📊 View Detailed Analysis"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Audio Predictions")
            audio_df = pd.DataFrame({
                'Emotion': audio_labels,
                'Confidence': [f"{p*100:.1f}%" for p in audio_preds]
            })
            st.dataframe(audio_df, hide_index=True)
        
        with col_b:
            st.markdown("### Image Predictions")
            image_df = pd.DataFrame({
                'Emotion': image_labels,
                'Confidence': [f"{p*100:.1f}%" for p in image_preds]
            })
            st.dataframe(image_df, hide_index=True)
    
    update_statistics('combined', prediction_label)
    add_to_history('Combined 🎯', prediction_label, confidence)
    
    # Play initial audio only once (first time result is shown)
    if st.session_state.audio_enabled and not st.session_state.initial_audio_played:
        announcement = f"Combined result: {prediction_label}, {confidence:.0f} percent confidence"
        play_combined_audio(announcement, empathetic_msg, audio_feedback_placeholder, force_play=True)
        st.session_state.initial_audio_played = True
    
elif audio_preds is not None:
    audio_idx = np.argmax(audio_preds)
    prediction_label = audio_labels[audio_idx]
    confidence = audio_preds[audio_idx] * 100
    
    st.markdown("## 🎤 Audio Analysis Result")
    
    result_class = get_result_class(prediction_label)
    emoji = get_emotion_emoji(prediction_label)
    
    st.markdown(f"""
    <div class='result-box {result_class}' role='status' aria-live='polite'>
        {emoji} {prediction_label} {emoji}<br>
        Confidence: {confidence:.1f}%
    </div>
    """, unsafe_allow_html=True)
    
    # Add empathetic message - VISIBLE ON SCREEN
    empathetic_msg = get_empathetic_message(prediction_label, confidence, 'audio')
    if empathetic_msg:
        st.markdown("### 💬 Empathetic Response")
        st.markdown(f"""
        <div class='metric-card' style='border-left: 5px solid #4CAF50; background: #f0f9ff;'>
            <p style='font-size: 22px; color: #1e3a8a; line-height: 1.8; font-weight: 500;'>
                💭 {empathetic_msg}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add text input for user response with conversation flow
        user_response = st.text_area(
            "Share your thoughts (optional):",
            placeholder="What's on your mind?",
            height=100,
            key="response_audio"
        )
        
        if user_response:
            user_name = st.session_state.get('user_name', '').strip()
            display_name = user_name if user_name else "friend"
            st.success(f"✅ Thank you for sharing, {display_name}. Your feelings are valid and heard.")
            
            # Get conversation followup if in sad emotion flow
            followup = get_conversation_followup(user_response)
            if followup:
                st.markdown("### 💬 Next Step")
                st.markdown(f"""
                <div class='metric-card' style='border-left: 5px solid #9b59b6; background: #f3e5f5;'>
                    <p style='font-size: 20px; color: #4a148c; line-height: 1.8;'>
                        {followup}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Play followup audio - ALWAYS play for new conversation steps
                if st.session_state.audio_enabled:
                    followup_placeholder = st.empty()
                    play_combined_audio("", followup, followup_placeholder, force_play=True)
                
                # Show option to continue or get resources
                if st.session_state.conversation_stage == 5:
                    col_x, col_y = st.columns(2)
                    with col_x:
                        if st.button("🏥 Get Mental Health Resources", key="resources_audio"):
                            show_mental_health_resources()
                    with col_y:
                        if st.button("🔄 Start New Conversation", key="reset_audio"):
                            st.session_state.conversation_stage = 0
                            st.session_state.current_emotion = None
                            st.session_state.initial_audio_played = False  # Reset audio flag
                            st.rerun()
    
    # Show all predictions
    with st.expander("📊 View All Predictions"):
        audio_df = pd.DataFrame({
            'Emotion': audio_labels,
            'Confidence': [f"{p*100:.1f}%" for p in audio_preds]
        }).sort_values('Confidence', ascending=False)
        st.dataframe(audio_df, hide_index=True)
    
    update_statistics('audio', prediction_label)
    add_to_history('Audio 🎤', prediction_label, confidence)
    
    # Play initial audio only once (first time result is shown)
    if st.session_state.audio_enabled and not st.session_state.initial_audio_played:
        announcement = f"Audio result: {prediction_label}, {confidence:.0f} percent confidence"
        play_combined_audio(announcement, empathetic_msg, audio_feedback_placeholder, force_play=True)
        st.session_state.initial_audio_played = True
    
elif image_preds is not None:
    image_idx = np.argmax(image_preds)
    prediction_label = image_labels[image_idx]
    confidence = float(image_preds[image_idx]) * 100
    
    st.markdown("## 📸 Image Analysis Result")
    
    result_class = get_result_class(prediction_label)
    emoji = get_emotion_emoji(prediction_label)
    
    st.markdown(f"""
    <div class='result-box {result_class}' role='status' aria-live='polite'>
        {emoji} {prediction_label.upper()} {emoji}<br>
        Confidence: {confidence:.1f}%
    </div>
    """, unsafe_allow_html=True)
    
    # Add empathetic message - VISIBLE ON SCREEN
    empathetic_msg = get_empathetic_message(prediction_label, confidence, 'image')
    if empathetic_msg:
        st.markdown("### 💬 Empathetic Response")
        st.markdown(f"""
        <div class='metric-card' style='border-left: 5px solid #4CAF50; background: #f0f9ff;'>
            <p style='font-size: 22px; color: #1e3a8a; line-height: 1.8; font-weight: 500;'>
                💭 {empathetic_msg}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add text input for user response with conversation flow
        user_response = st.text_area(
            "Share your thoughts (optional):",
            placeholder="What's on your mind?",
            height=100,
            key="response_image"
        )
        
        if user_response:
            user_name = st.session_state.get('user_name', '').strip()
            display_name = user_name if user_name else "friend"
            st.success(f"✅ Thank you for sharing, {display_name}. Your feelings are valid and heard.")
            
            # Get conversation followup if in sad emotion flow
            followup = get_conversation_followup(user_response)
            if followup:
                st.markdown("### 💬 Next Step")
                st.markdown(f"""
                <div class='metric-card' style='border-left: 5px solid #9b59b6; background: #f3e5f5;'>
                    <p style='font-size: 20px; color: #4a148c; line-height: 1.8;'>
                        {followup}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Play followup audio - ALWAYS play for new conversation steps
                if st.session_state.audio_enabled:
                    followup_placeholder = st.empty()
                    play_combined_audio("", followup, followup_placeholder, force_play=True)
                
                # Show option to continue or get resources
                if st.session_state.conversation_stage == 5:
                    col_x, col_y = st.columns(2)
                    with col_x:
                        if st.button("🏥 Get Mental Health Resources", key="resources_image"):
                            show_mental_health_resources()
                    with col_y:
                        if st.button("🔄 Start New Conversation", key="reset_image"):
                            st.session_state.conversation_stage = 0
                            st.session_state.current_emotion = None
                            st.session_state.initial_audio_played = False  # Reset audio flag
                            st.rerun()
    
    # Show all predictions
    with st.expander("📊 View All Predictions"):
        image_df = pd.DataFrame({
            'Emotion': image_labels,
            'Confidence': [f"{p*100:.1f}%" for p in image_preds]
        }).sort_values('Confidence', ascending=False)
        st.dataframe(image_df, hide_index=True)
    
    update_statistics('image', prediction_label)
    add_to_history('Image 📸', prediction_label, confidence)
    
    # Play initial audio only once (first time result is shown)
    if st.session_state.audio_enabled and not st.session_state.initial_audio_played:
        announcement = f"Image result: {prediction_label}, {confidence:.0f} percent confidence"
        play_combined_audio(announcement, empathetic_msg, audio_feedback_placeholder, force_play=True)
        st.session_state.initial_audio_played = True

else:
    st.info("👆 Please upload audio and/or image files to begin analysis")

# ============================================
# ANALYSIS HISTORY
# ============================================
if st.session_state.history:
    st.divider()
    st.markdown("## 📜 Recent Analysis History")
    
    for item in st.session_state.history:
        emoji = get_emotion_emoji(item['emotion'])
        st.markdown(f"""
        <div class='history-item'>
            <strong>{item['timestamp']}</strong> | 
            {item['type']} | 
            {emoji} <strong>{item['emotion']}</strong> | 
            Confidence: {item['confidence']:.1f}%
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #636e72; padding: 20px;'>
    <p style='font-size: 18px;'>♿ Designed with accessibility in mind</p>
    <p>Nissay Project Prototype © 2025</p>
    <p style='font-size: 14px; margin-top: 10px;'>
        Features: Real-time emotion detection | Audio feedback | Session tracking | Multi-modal analysis
    </p>
</div>
""", unsafe_allow_html=True)