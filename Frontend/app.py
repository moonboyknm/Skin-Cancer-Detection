import streamlit as st
from PIL import Image
import io
import time
import random
import requests
import numpy as np

# --- Import your model framework ---
from tensorflow.keras.models import load_model

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Helper function to process image for banner display ---
def get_banner_image(image_url, target_height=180, target_aspect_ratio=4/1):
    """
    Fetches an image from a URL, crops it to a target aspect ratio,
    and then resizes it to a specific height for banner display.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))

        original_width, original_height = img.size
        current_aspect_ratio = original_width / original_height

        if current_aspect_ratio > target_aspect_ratio:
            new_width = int(original_height * target_aspect_ratio)
            left = (original_width - new_width) / 2
            top = 0
            right = (original_width + new_width) / 2
            bottom = original_height
        else:
            new_height = int(original_width / target_aspect_ratio)
            left = 0
            top = (original_height - new_height) / 2
            right = original_width
            bottom = (original_height + new_height) / 2

        img_cropped = img.crop((left, top, right, bottom))
        cropped_width, cropped_height = img_cropped.size
        if cropped_height == 0:
            return None
        scale_factor = target_height / cropped_height
        final_width = int(cropped_width * scale_factor)
        img_final = img_cropped.resize((final_width, target_height), Image.LANCZOS)

        buf = io.BytesIO()
        img_final.save(buf, format="PNG")
        return buf.getvalue()

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching image from URL: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# --- Load the Model (this runs only once when the app starts) ---
@st.cache_resource
def load_my_model():
    try:
        model = load_model('skin cancer using resnet50.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}. Make sure 'skin cancer using resnet50.h5' is in the correct path and all dependencies (especially tensorflow==2.12.0) are installed.")
        return None

model = load_my_model()

# Define your class names (MUST match the order your model predicts)
CLASS_NAMES = [
    'pigmented benign keratosis',
    'melanoma',
    'vascular lesion',
    'actinic keratosis',
    'squamous cell carcinoma',
    'basal cell carcinoma',
    'seborrheic keratosis',
    'dermatofibroma',
    'nevus'
]

# --- Preprocessing function (MUST match your model's training preprocessing) ---
def preprocess_image_for_model(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    # --- IMPORTANT CHANGE HERE: Resize to (width, height) as expected by the model ---
    # The error message said "expected shape=(None, 75, 100, 3)"
    # This means height=75, width=100. PIL resize expects (width, height).
    image = image.resize((100, 75)) # Changed from (224, 224) to (100, 75)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 # Assuming 0-1 normalization as in your notebook

    return img_array


# --- Title and Description ---
st.title("🔬 Skin Cancer Classifier")

# --- Image URL for the banner ---
image_url = "https://plus.unsplash.com/premium_photo-1668487827039-ec0bedd0eb85?q=80&w=2012&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

# Process and display the banner image
banner_image_bytes = get_banner_image(image_url, target_height=180, target_aspect_ratio=4/1)

if banner_image_bytes:
    st.image(banner_image_bytes, use_container_width=True)
else:
    st.image("https://placehold.co/600x180/ADD8E6/000000?text=Medical+Image+Placeholder", use_container_width=True)

st.markdown("""
    Upload an image of a skin lesion to get a preliminary classification.
    **Disclaimer:** This is a prototype for demonstration purposes and does not provide medical advice.
    Always consult a qualified healthcare professional for any medical concerns.
""")

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

image = None
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        image = None

# --- Classification Button and Logic ---
if st.button("Classify Image", disabled=image is None or model is None):
    if image is not None and model is not None:
        with st.spinner('Classifying image...'):
            try:
                processed_img = preprocess_image_for_model(image)
                predictions = model.predict(processed_img)

                predicted_class_index = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions) * 100

                predicted_class_name = CLASS_NAMES[predicted_class_index]

                st.success("Classification Complete!")
                st.markdown(f"### Result: **{predicted_class_name}**")
                st.markdown(f"Confidence: **{confidence:.2f}%**")
                st.info("Remember, this is a preliminary result. For accurate diagnosis, consult a medical professional.")

            except Exception as e:
                st.error(f"Error during prediction: {e}. Please check your model loading and preprocessing steps. Ensure the input shape matches what your model expects.")
    else:
        if model is None:
            st.warning("The model could not be loaded. Please check the error message above.")
        else:
            st.warning("Please upload an image first.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by Kishor Sahoo and Kratik Mudgal")
