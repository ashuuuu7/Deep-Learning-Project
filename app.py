import streamlit as st 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title= "Cat vs Dog AI 🐱🐶",  page_icon="🐱🐶", layout="centered")

st.title("🐱🐶 Cats vs Dogs Classifier")
st.markdown("Upload an image and let AI predict whether it's a **Cat or Dog** 🔥")

@st.cache_resource
def load_my_model():
    return load_model("cats_dogs_classifier.keras")

model = load_my_model()

uploaded_file = st.file_uploader("📤 Upload Image (JPG / JPEG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="📷 Uploaded Image", use_container_width=True)

        with st.spinner("🤖 AI is analyzing the image..."):
            img = img.resize((64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])

        st.markdown("---")
        st.subheader("🧠 Prediction Result")

        if confidence > 0.5:
            st.success("🐶 It's a DOG!")
            st.write(f"Confidence: **{confidence:.2%}**")
        else:
            st.success("🐱 It's a CAT!")
            st.write(f"Confidence: **{(1-confidence):.2%}**")

        st.progress(float(confidence if confidence > 0.5 else 1-confidence))

    except Exception as e:
        st.error("⚠️ Error while processing image")
        st.write(e)

st.markdown("---")
st.markdown(
    """
    ---
    ### ⚡ Cat vs Dog Classifier
    Developed by **Ashutosh Giri**  
    AI/ML Project | CNN + Streamlit
    """
)