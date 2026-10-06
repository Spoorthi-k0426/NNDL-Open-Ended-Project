import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ==========================================
# Application Configuration & Constants
# ==========================================
IMG_SIZE = (224, 224)
MODEL_PATHS = [
    "pneumonia_resnet50v2_final.keras",
    "pneumonia_model.keras",
    "best_resnet50v2_finetuned.keras",
    "best_resnet50v2_head.keras"
]

# Default decision threshold (0.35 - 0.50 range for balanced sensitivity/specificity)
DEFAULT_THRESHOLD = 0.35

# Streamlit Page Setup
st.set_page_config(
    page_title="Pneumonia Detection from Chest X-Ray",
    page_icon="🫁",
    layout="centered"
)

# Custom Styling for Clean, Modern, Beginner-Friendly UI
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-card-normal {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
        text-align: center;
    }
    .result-card-pneumonia {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
        text-align: center;
    }
    .result-header {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 10px;
        margin-top: 1rem;
        text-align: center;
    }
    .metric-box {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 800;
    }
    .disclaimer {
        text-align: center;
        font-size: 0.85rem;
        color: #94a3b8;
        border-top: 1px solid #e2e8f0;
        padding-top: 1.25rem;
        margin-top: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# Model Loading (Cached for Performance)
# ==========================================
@st.cache_resource
def load_pneumonia_model():
    """Locate and load the trained Keras model file with caching."""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                return model, path
            except Exception as e:
                st.error(f"Error loading model from `{path}`: {e}")
    return None, None


# ==========================================
# Image Preprocessing Pipeline
# ==========================================
def preprocess_xray(image: Image.Image, model=None) -> np.ndarray:
    """
    Preprocess image aligned with the model architecture:
    1. Ensure RGB format (3 channels)
    2. Resize to (224, 224)
    3. Convert to float32 NumPy array in [0, 255]
    4. Expand batch dimension to (1, 224, 224, 3)
    5. Apply appropriate preprocess_input without double-scaling
    """
    # 1. RGB format
    rgb_image = image.convert("RGB")
    
    # 2. Resize to (224, 224)
    resized_image = rgb_image.resize(IMG_SIZE)
    
    # 3. Convert to float32 NumPy array (values 0.0 - 255.0)
    image_array = np.asarray(resized_image, dtype=np.float32)
    
    # 4. Expand batch dimension -> shape (1, 224, 224, 3)
    batch = np.expand_dims(image_array, axis=0)

    # 5. Determine backbone architecture and apply preprocess_input
    model_name = getattr(model, "name", "").lower() if model else ""
    layer_names = [l.name.lower() for l in model.layers] if model else []

    if "densenet" in model_name or any("densenet" in name for name in layer_names):
        from tensorflow.keras.applications.densenet import preprocess_input as densenet_prep
        preprocessed_batch = densenet_prep(batch.copy())
    else:
        from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_prep
        preprocessed_batch = resnet_prep(batch.copy())

    return preprocessed_batch


# ==========================================
# Prediction Handler
# ==========================================
def predict_pneumonia(image: Image.Image, model, threshold: float = DEFAULT_THRESHOLD):
    """
    Generate model predictions without hardcoding or saturation.
    Returns: label, confidence, prob_pneumonia, prob_normal
    """
    # Preprocess image
    img_batch = preprocess_xray(image, model=model)

    # Extract single sigmoid probability
    raw_pred = model.predict(img_batch, verbose=0)
    prob_pneumonia = float(raw_pred[0][0])
    prob_normal = 1.0 - prob_pneumonia

    # Decision logic based on threshold
    if prob_pneumonia >= threshold:
        label = "PNEUMONIA"
        confidence = prob_pneumonia * 100.0
    else:
        label = "NORMAL"
        confidence = prob_normal * 100.0

    return label, confidence, prob_pneumonia, prob_normal


# ==========================================
# User Interface
# ==========================================
st.markdown('<h1 class="main-title">Pneumonia Detection from Chest X-Ray</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload a chest X-ray image to predict whether it is Normal or Pneumonia.</p>',
    unsafe_allow_html=True
)

# Load model
model, loaded_path = load_pneumonia_model()

if model is None:
    st.warning(
        "⚠️ **Model file not found.** Please ensure `pneumonia_resnet50v2_final.keras` "
        "is placed in the project directory beside `app.py`."
    )

# File uploader
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supported image formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Open and display image preview
    pil_image = Image.open(uploaded_file)
    
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(
            pil_image,
            caption="Uploaded Chest X-Ray",
            use_container_width=True
        )

    # Predict Button
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    if predict_clicked:
        if model is None:
            st.error("Cannot run prediction because no trained model file was found.")
        else:
            with st.spinner("Analyzing chest X-ray..."):
                # Run prediction pipeline
                label, confidence, prob_pneumonia, prob_normal = predict_pneumonia(
                    pil_image,
                    model=model,
                    threshold=DEFAULT_THRESHOLD
                )

                # Render Result Card
                if label == "NORMAL":
                    st.markdown(
                        f"""
                        <div class="result-card-normal">
                            <div class="result-header" style="color: #16a34a;">✅ NORMAL</div>
                            <div class="metric-grid">
                                <div class="metric-box">
                                    <div class="metric-label">Confidence</div>
                                    <div class="metric-value" style="color: #16a34a;">{confidence:.2f}%</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-label">Normal Probability</div>
                                    <div class="metric-value" style="color: #16a34a;">{prob_normal * 100:.2f}%</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-label">Pneumonia Probability</div>
                                    <div class="metric-value" style="color: #475569;">{prob_pneumonia * 100:.2f}%</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-pneumonia">
                            <div class="result-header" style="color: #dc2626;">⚠️ PNEUMONIA</div>
                            <div class="metric-grid">
                                <div class="metric-box">
                                    <div class="metric-label">Confidence</div>
                                    <div class="metric-value" style="color: #dc2626;">{confidence:.2f}%</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-label">Pneumonia Probability</div>
                                    <div class="metric-value" style="color: #dc2626;">{prob_pneumonia * 100:.2f}%</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-label">Normal Probability</div>
                                    <div class="metric-value" style="color: #475569;">{prob_normal * 100:.2f}%</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Risk Meter / Visual Progress
                st.caption(f"Decision Threshold: {DEFAULT_THRESHOLD:.2f} ({DEFAULT_THRESHOLD * 100:.0f}%)")
                st.progress(
                    min(max(prob_pneumonia, 0.0), 1.0),
                    text=f"Pneumonia Risk Index: {prob_pneumonia * 100:.2f}%"
                )

# ==========================================
# Medical Disclaimer
# ==========================================
st.markdown(
    '<div class="disclaimer">This project is for educational purposes only and is not a medical diagnostic system.</div>',
    unsafe_allow_html=True
)
