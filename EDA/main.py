import streamlit as st
from PIL import Image
import os

def load_image(image_path):
    """Load an image from a given path."""
    return Image.open(image_path)

st.set_page_config(page_title="CCTV Camera Model EDA", layout="wide")

st.title("📊 CCTV Camera Model EDA Dashboard")
st.markdown("Explore visualizations related to your CCTV camera model's performance.")

st.sidebar.header("Choose Model Category")
model_category = st.sidebar.selectbox(
    "Select a category:",
    ("accident detection", "fire", "license", "vehicle detection")
)

view_mode = st.toggle("🔄 Switch to Slideshow Mode")

static_folder = "static"
selected_folder = os.path.join(static_folder, model_category)

image_files = [
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "F1_curve.png",
    "labels.jpg",
    "labels_correlogram.jpg",
    "PR_curve.png",
    "P_curve.png",
    "results.png",
    "R_curve.png"
]

image_paths = [os.path.join(selected_folder, img) for img in image_files if os.path.exists(os.path.join(selected_folder, img))]
loaded_images = [load_image(img) for img in image_paths]

if not loaded_images:
    st.error("⚠ No images found in the selected category.")
else:
    if view_mode:
        st.header(f"🎞 Slideshow View - {model_category.capitalize()}")

        if "index" not in st.session_state:
            st.session_state.index = 0

        col1, col2, col3 = st.columns([1, 5, 1])

        with col1:
            if st.button("⬅ Previous"):
                st.session_state.index = (st.session_state.index - 1) % len(loaded_images)

        with col2:
            st.image(loaded_images[st.session_state.index], caption=image_files[st.session_state.index], use_container_width=True)

        with col3:
            if st.button("Next ➡"):
                st.session_state.index = (st.session_state.index + 1) % len(loaded_images)

    else:
        st.header(f"🖼 Grid View - {model_category.capitalize()}")

        num_columns = 3
        cols = st.columns(num_columns)

        for idx, img in enumerate(loaded_images):
            with cols[idx % num_columns]:
                st.image(img, caption=image_files[idx], use_container_width=True)
