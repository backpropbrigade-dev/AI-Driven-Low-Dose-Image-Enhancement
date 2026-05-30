import streamlit as st
import requests
from PIL import Image
import io

# Set up page configurations
st.set_page_config(page_title="AI Low-Dose Image Enhancer", layout="wide")

st.title("✨ AI-Driven Low-Dose Image Enhancement SaaS")
st.write("Upload your low-dose medical or low-light images to instantly enhance their quality using deep learning.")

# URL of your backend API (Points to local machine during testing)
API_URL = "http://127.0.0.1:8000/enhance"

# Sidebar for SaaS features / Branding
st.sidebar.header("⚙️ Subscription Tier: Free Plan")
st.sidebar.write("Upgrade to Premium for GPU processing and batch image uploads.")
if st.sidebar.button("Upgrade to Premium"):
    st.sidebar.info("Stripe integration sandbox: Payment checkout would open here.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a low-dose image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display processing layout side-by-side
    col1, col2 = st.columns(2)
    
    original_image = Image.open(uploaded_file)
    with col1:
        st.subheader("Original Low-Dose Image")
        st.image(original_image, use_container_width=True)
        
    with col2:
        st.subheader("Enhanced AI Image")
        with st.spinner("AI model processing... Please wait."):
            try:
                # Convert uploaded file to bytes to send via HTTP POST
                img_byte_arr = io.BytesIO()
                original_image.save(img_byte_arr, format=original_image.format if original_image.format else "PNG")
                img_byte_arr = img_byte_arr.getvalue()
                
                # Send request to FastAPI backend
                files = {"file": ("image.png", img_byte_arr, "image/png")}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    enhanced_image = Image.open(io.BytesIO(response.content))
                    st.image(enhanced_image, use_container_width=True)
                    
                    # Provide download button for the enhanced image
                    st.download_button(
                        label="Download Enhanced Image",
                        data=response.content,
                        file_name="enhanced_image.png",
                        mime="image/png"
                    )
                else:
                    st.error(f"Error from API server: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the Backend API. Make sure app.py is running on port 8000.")
