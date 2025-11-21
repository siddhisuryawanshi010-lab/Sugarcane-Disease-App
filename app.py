import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import pandas as pd
import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgriScan AI", page_icon="🌿", layout="wide")

# --- SESSION STATE (Start from ZERO) ---
if 'total_scans' not in st.session_state:
    st.session_state['total_scans'] = 0  # Starts at 0

if 'recent_scans' not in st.session_state:
    st.session_state['recent_scans'] = [] # Starts Empty

# --- TREATMENT DATABASE ---
TREATMENT_INFO = {
    "RedRot": {
        "Traditional": "Crop rotation with rice or green manure. Remove and burn infected clumps immediately.",
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
    .metric-container { background-color: white; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        # Ensure 'sugarcane_model.h5' is in the same folder!
        model = tf.keras.models.load_model('sugarcane_model.h5')
        return model
    except:
        return None

model = load_model()
class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

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
    col1.metric("Total Scans (Session)", f"{st.session_state['total_scans']}", "Live Count")
    col2.metric("System Status", "Active", "Ready")
    col3.metric("Model Accuracy", "92.5%", "High Precision")
    
    st.markdown("### 🕒 Recent Activity Log")
    
    if len(st.session_state['recent_scans']) == 0:
        st.info("No scans performed yet. Go to 'Live Analysis' to start.")
    else:
        df_log = pd.DataFrame(st.session_state['recent_scans'])
        st.dataframe(df_log, use_container_width=True)

# --- PAGE 2: LIVE ANALYSIS ---
elif app_mode == "Live Analysis":
    st.title("🌿 AI Disease Diagnostic Tool")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Upload Sample")
        uploaded_file = st.file_uploader("Drop Image Here", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Sample Preview", use_column_width=True)

    with col2:
        st.subheader("2. Analysis Results")
        if uploaded_file and model:
            if st.button("🔍 Run Deep Learning Scan"):
                # Fake Loading
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Preprocessing...")
                time.sleep(0.5)
                progress_bar.progress(50)
                status_text.text("Analyzing Leaf Patterns...")
                time.sleep(0.5)
                progress_bar.progress(100)
                status_text.empty()

                # Real Prediction
                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                predictions = model.predict(img_array)
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions)
                
                # Update Session State
                st.session_state['total_scans'] += 1
                today_date = datetime.date.today().strftime("%Y-%m-%d")
                new_log = {'Date': today_date, 'Location': 'Live Upload', 'Status': predicted_class}
                st.session_state['recent_scans'].insert(0, new_log)

                # Display Result
                if predicted_class == 'Healthy':
                    st.success(f"✅ RESULT: **{predicted_class}**")
                else:
                    st.error(f"⚠️ RESULT: **{predicted_class}**")
                
                st.write(f"**Confidence:** {confidence*100:.2f}%")
                st.progress(int(confidence * 100))
                
                st.markdown("---")
                st.subheader("💊 Recommended Treatments")
                
                # THE 3-WAY SOLUTION DISPLAY
                if predicted_class in TREATMENT_INFO:
                    info = TREATMENT_INFO[predicted_class]
                    
                    t1, t2, t3 = st.columns(3)
                    
                    with t1:
                        st.info("**🛠️ Traditional**")
                        st.write(info["Traditional"])
                    
                    with t2:
                        st.warning("**🧪 Chemical**")
                        st.write(info["Chemical"])
                        
                    with t3:
                        st.success("**🌿 Organic**")
                        st.write(info["Organic"])

# --- PAGE 3: REPORTS ---
elif app_mode == "Reports":
    st.title("📑 Session Reports")
    
    if len(st.session_state['recent_scans']) > 0:
        df_report = pd.DataFrame(st.session_state['recent_scans'])
        st.table(df_report)
        
        csv = df_report.to_csv(index=False).encode('utf-8')
        
        # Fixed the syntax error here by using arguments clearly
        st.download_button(
            label="📥 Download CSV Report",
            data=csv,
            file_name="scan_report.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data available to generate report. Please run a scan first.")