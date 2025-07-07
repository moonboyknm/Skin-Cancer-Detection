import streamlit as st
from PIL import Image
import io
import time
import random
import requests
import numpy as np
import tensorflow as tf # Import TensorFlow for Keras

# --- Import your model framework ---
from tensorflow.keras.models import load_model

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Initialize session state for file uploader reset ---
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

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

# --- IMPORTANT: Define x_train_mean and x_train_std from your Kaggle Notebook ---
# These are your global scalar mean and std values
x_train_mean = 163.41851373580246
x_train_std = 41.339967863159146

# --- Preprocessing function (MUST match your model's training preprocessing) ---
def preprocess_image_for_model(image, x_train_mean, x_train_std):
    # Ensure image is in RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image to match training dimensions (width, height)
    # The model expects (None, 75, 100, 3), so height=75, width=100.
    # PIL resize expects (width, height).
    img_resized = image.resize((100, 75), Image.LANCZOS) # Using LANCZOS for high quality downsampling

    # Convert PIL Image to NumPy array
    img_array = np.array(img_resized)

    # Normalize using training mean and std
    # NumPy broadcasting will correctly apply scalar mean/std to all elements
    img_norm = (img_array - x_train_mean) / x_train_std

    # Add batch dimension (becomes (1, 75, 100, 3))
    img_input = np.expand_dims(img_norm, axis=0)

    return img_input

# --- Function to clear uploaded files ---
def clear_uploads():
    st.session_state.uploader_key += 1 # Increment key to reset uploader
    # This will also implicitly clear `uploaded_files` on rerun

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
    Upload images of skin lesions (up to 5) to get preliminary classifications.
    **Disclaimer:** This is a prototype for demonstration purposes and does not provide medical advice.
    Always consult a qualified healthcare professional for any medical concerns.
""")

# --- Image Upload Section ---
# Use a column layout for the uploader and clear button
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Choose up to 5 images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key # IMPORTANT: Use the dynamic key here
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True) # Add some spacing
    if st.button("Clear Uploads", help="Click to clear all selected files"):
        clear_uploads()
        st.rerun() # Corrected: Use st.rerun() instead of st.experimental_rerun()


images = []
if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("You can upload a maximum of 5 images. Only the first 5 will be processed.")
        uploaded_files = uploaded_files[:5]

    st.subheader("Uploaded Images:")
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            image = Image.open(uploaded_file)
            images.append(image)
            st.image(image, caption=f'Uploaded Image {i+1}', width=200) # Display with fixed width
        except Exception as e:
            st.error(f"Error loading image {uploaded_file.name}: {e}")
            images = [] # Clear images if any fails to load
            break # Stop processing further files

# --- Classification Button and Logic ---
# Disable button if no images are uploaded or model is not loaded
if st.button("Classify Images", disabled=not images or model is None):
    if images and model is not None:
        st.subheader("Classification Results:")
        for i, image in enumerate(images):
            st.markdown(f"---")
            st.markdown(f"#### Image {i+1}")
            st.image(image, caption=f'Original Image {i+1}', use_container_width=True) # Display full size for results

            with st.spinner(f'Classifying Image {i+1}...'):
                try:
                    processed_img = preprocess_image_for_model(image, x_train_mean, x_train_std)
                    predictions = model.predict(processed_img)

                    predicted_class_index = np.argmax(predictions, axis=1)[0]
                    confidence = np.max(predictions) * 100

                    predicted_class_name = CLASS_NAMES[predicted_class_index]

                    st.success(f"Classification Complete for Image {i+1}!")
                    st.markdown(f"##### Result: **{predicted_class_name}**")
                    st.markdown(f"Confidence: **{confidence:.2f}%**")

                except Exception as e:
                    st.error(f"Error during prediction for Image {i+1}: {e}. Please check your model loading and preprocessing steps. Ensure the input shape matches what your model expects.")
                    st.exception(e) # Show full traceback for debugging
        
        st.info("Remember, these are preliminary results. For accurate diagnosis, consult a medical professional.")
    else:
        if model is None:
            st.warning("The model could not be loaded. Please check the error message above.")
        else:
            st.warning("Please upload at least one image first.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by Kishor Sahoo and Kratik Mudgal")