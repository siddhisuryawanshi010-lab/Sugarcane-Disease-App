import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import time
import pandas as pd
import datetime
import os
import gdown # Library to download from Google Drive

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgriScan AI", page_icon="🌿", layout="wide")

# ==========================================
# 👇 PASTE YOUR GOOGLE DRIVE FILE ID HERE 👇
# ==========================================
file_id = '1XMMg6Ep99H5XvmCnlMa8IXOLImMDwpTy'
# ==========================================

# --- BACKEND MODEL LOADING (CLOUD VERSION) ---
@st.cache_resource
def load_model():
    model_path = 'sugarcane_model.h5'
    
    # 1. Check if model exists, if not, download from Drive
    if not os.path.exists(model_path):
        try:
            url = f'https://drive.google.com/uc?id={file_id}'
            print(f"Downloading model from Google Drive (ID: {file_id})...")
            gdown.download(url, model_path, quiet=False)
            print("Download complete.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None

    # 2. Load the model
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

model = load_model()

# --- SESSION STATE ---
if 'total_scans' not in st.session_state:
    st.session_state['total_scans'] = 0
if 'recent_scans' not in st.session_state:
    st.session_state['recent_scans'] = []

# --- TREATMENT DATABASE ---
TREATMENT_INFO = {
    "RedRot": {
        "Traditional": "Crop rotation with rice. Remove and burn infected clumps immediately.",
        "Chemical": "Dip setts in 0.1% Carbendazim solution for 15 mins before planting.",
        "Organic": "Soil application of Trichoderma viride (Bio-fungicide) mixed with compost."
    },
    "Mosaic": {
        "Traditional": "Use disease-free seed setts. Remove weeds that host aphids.",
        "Chemical": "Spray Malathion (0.1%) or Dimethoate to control aphid vectors.",
        "Organic": "Spray Neem Oil (2%) to repel aphids naturally."
    },
    "Rust": {
        "Traditional": "Avoid excess nitrogen fertilizer. Ensure proper field drainage.",
        "Chemical": "Spray Mancozeb (0.2%) or Propiconazole (0.1%) at 15-day intervals.",
        "Organic": "Foliar spray of Verticillium lecanii (a fungus that eats rust spores)."
    },
    "Yellow": {
        "Traditional": "Use hot water treated seed setts (50°C for 2 hours).",
        "Chemical": "Apply Imidacloprid to control the insect vectors spreading the disease.",
        "Organic": "Apply balanced organic manure to boost plant immunity."
    },
    "Healthy": {
        "Traditional": "Continue standard irrigation and monitoring.",
        "Chemical": "No chemical action needed.",
        "Organic": "Maintain soil health with vermicompost."
    }
}

class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
    }
    .stButton>button:hover { background-color: #1b5e20; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/188/188333.png", width=80)
st.sidebar.title("AgriScan Control")
app_mode = st.sidebar.radio("Navigate", ["Dashboard", "Live Analysis", "Reports"])
st.sidebar.markdown("---")
st.sidebar.info("System Status: **Online** ✅")

# --- PAGE 1: DASHBOARD ---
if app_mode == "Dashboard":
    st.title("📊 Farm Health Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Scans", f"{st.session_state['total_scans']}", "Live")
    col2.metric("System", "Active", "Ready")
    col3.metric("Accuracy", "92.5%", "High")
    
    if len(st.session_state['recent_scans']) > 0:
        st.markdown("### 🕒 Activity Log")
        st.dataframe(pd.DataFrame(st.session_state['recent_scans']), use_container_width=True)
    else:
        st.info("System Ready. Go to 'Live Analysis' to begin.")

# --- PAGE 2: LIVE ANALYSIS ---
elif app_mode == "Live Analysis":
    st.title("🌿 AI Disease Diagnostic Tool")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Upload Sample")
        uploaded_file = st.file_uploader("Drop Image Here", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Sample Preview", use_container_width=True)

    with col2:
        st.subheader("2. Analysis Results")
        if uploaded_file:
            if model is None:
                st.error("❌ Model failed to load. Check Google Drive ID.")
            else:
                if st.button("🔍 Run Deep Learning Scan"):
                    progress_bar = st.progress(0)
                    status = st.empty()
                    status.text("Processing...")
                    time.sleep(0.5)
                    progress_bar.progress(50)
                    
                    # --- CENTER CROP FIX ---
                    img_cropped = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    img_array = np.array(img_cropped)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    
                    predictions = model.predict(img_array)
                    predicted_class = class_names[np.argmax(predictions)]
                    confidence = np.max(predictions)
                    
                    progress_bar.progress(100)
                    status.empty()
                    
                    # Log
                    st.session_state['total_scans'] += 1
                    today = datetime.date.today().strftime("%Y-%m-%d")
                    st.session_state['recent_scans'].insert(0, {'Date': today, 'Location': 'Live', 'Status': predicted_class})

                    # Result
                    if predicted_class == 'Healthy':
                        st.success(f"✅ RESULT: {predicted_class}")
                        st.image(img_cropped, caption="Analysis Focus Area", width=200)
                    else:
                        st.error(f"⚠️ RESULT: {predicted_class}")
                    
                    st.write(f"**AI Confidence:** {confidence*100:.2f}%")
                    
                    st.markdown("---")
                    if predicted_class in TREATMENT_INFO:
                        info = TREATMENT_INFO[predicted_class]
                        t1, t2, t3 = st.columns(3)
                        with t1:
                            st.info("Traditional")
                            st.write(info["Traditional"])
                        with t2:
                            st.warning("Chemical")
                            st.write(info["Chemical"])
                        with t3:
                            st.success("Organic")
                            st.write(info["Organic"])

# --- PAGE 3: REPORTS ---
elif app_mode == "Reports":
    st.title("📑 Session Reports")
    if len(st.session_state['recent_scans']) > 0:
        df = pd.DataFrame(st.session_state['recent_scans'])
        st.table(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "report.csv", "text/csv")
    else:
        st.warning("No data available.")
