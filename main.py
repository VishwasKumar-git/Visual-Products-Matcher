import certifi
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union
from io import BytesIO
import urllib.request

# ===============================
# CONFIGURATION
# ===============================
TRAINED_DB_PATH = "db"
os.environ['SSL_CERT_FILE'] = certifi.where()
st.set_page_config(page_title="Visual Image Search Engine", layout="wide")

# ===============================
# MODEL LOADING (cached)
# ===============================
@st.cache_resource
def load_model() -> tf.keras.Model:
    """Load the pre-trained ResNet50 model (cached)."""
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features(image_path: Union[str, BytesIO], model: tf.keras.Model) -> Union[np.ndarray, None]:
    """Extract deep features from an image using ResNet50."""
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img, verbose=0).flatten()
        return features
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None

# ===============================
# DATABASE FEATURE EXTRACTION (NO MODEL ARG)
# ===============================
@st.cache_data(show_spinner=False)
def get_feature_vectors_from_db(db_path: str) -> tuple[np.ndarray, list[str]]:
    """Extract features from all images in the database."""
    model = load_model()
    feature_list = []
    image_paths = []

    try:
        for img_path in os.listdir(db_path):
            if img_path.lower().endswith(".jpg"):
                path = os.path.join(db_path, img_path)
                features = extract_features(path, model)
                if features is not None:
                    feature_list.append(features)
                    image_paths.append(path)

        if not feature_list:
            st.warning("No valid images found in the database folder.")
            return np.array([]), []

        feature_vectors = np.vstack(feature_list)
        return feature_vectors, image_paths

    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return np.array([]), []

# ===============================
# SIMILARITY SEARCH
# ===============================
def find_similar_images(
    image_path: Union[str, BytesIO],
    feature_vectors: np.ndarray,
    image_paths: list[str],
    threshold: float = 0.5,
    top_n: int = 5
) -> list[str]:
    """Find similar images based on cosine similarity."""
    model = load_model()
    query_features = extract_features(image_path, model)
    if query_features is None or feature_vectors.size == 0:
        return []

    similarities = cosine_similarity([query_features], feature_vectors)
    similarities_indices = [i for i in range(len(similarities[0])) if similarities[0][i] > threshold]
    similarities_indices = sorted(similarities_indices, key=lambda i: similarities[0][i], reverse=True)
    similar_images = [image_paths[i] for i in similarities_indices[:top_n]]
    return similar_images

# ===============================
# SESSION STATE INITIALIZATION
# ===============================
def init_session_state():
    """Initialize session state variables."""
    if "feature_vectors" not in st.session_state:
        st.session_state.feature_vectors = None
    if "image_paths" not in st.session_state:
        st.session_state.image_paths = None

# ===============================
# URL IMAGE LOADER
# ===============================
def load_image_from_url(url: str) -> Union[Image.Image, None]:
    """Load an image directly from a URL."""
    try:
        response = urllib.request.urlopen(url)
        img_data = response.read()
        img = Image.open(BytesIO(img_data))
        return img
    except Exception as e:
        st.error(f"Failed to load image from URL: {str(e)}")
        return None

# ===============================
# MAIN STREAMLIT APP
# ===============================
def main():
    st.title("🔍 Visual Image Search Engine")
    st.markdown(
        """
        Upload an image **or paste an image URL** to find visually similar images from the database.  
        It uses **ResNet50** for feature extraction and **cosine similarity** for matching.  
        _(Currently supports only `.jpg` images.)_
        """
    )

    init_session_state()

    # Load database once per session
    if st.session_state.feature_vectors is None:
        with st.spinner("Loading image database..."):
            st.session_state.feature_vectors, st.session_state.image_paths = get_feature_vectors_from_db(TRAINED_DB_PATH)
            st.success("Database loaded successfully! ✅")

    st.markdown("### 📤 Input Options")
    col1, col2 = st.columns(2)

    uploaded_img_file = col1.file_uploader("Upload an image file", type=["jpg"])
    image_url = col2.text_input("Or paste an image URL")

    # Load image either from file or URL
    img_source = None
    img_name = None

    if uploaded_img_file is not None:
        img_source = uploaded_img_file
        img_name = "Uploaded Image"
    elif image_url:
        img = load_image_from_url(image_url)
        if img is not None:
            buf = BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            img_source = buf
            img_name = "Image from URL"

    if img_source:
        img_display = Image.open(img_source)
        st.image(img_display, caption=img_name, use_column_width=True)

        threshold = st.slider("🔸 Similarity Threshold", 0.0, 1.0, 0.5, 0.01)
        top_n = st.slider("🔸 Number of Similar Images", 1, 10, 4)

        if st.button("Find Similar Images"):
            with st.spinner("Searching for similar images..."):
                similar_images = find_similar_images(
                    img_source,
                    st.session_state.feature_vectors,
                    st.session_state.image_paths,
                    threshold,
                    top_n,
                )

                if similar_images:
                    st.success("Similar images found! 🎯")
                    cols = st.columns(min(top_n, 4))
                    for i, img_path in enumerate(similar_images):
                        with cols[i % 4]:
                            image_display = Image.open(img_path)
                            st.image(image_display, caption=f"Match {i + 1}", use_column_width=True)
                else:
                    st.warning("No similar images found. Try adjusting the threshold.")

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    main()
