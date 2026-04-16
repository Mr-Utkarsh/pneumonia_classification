# Importing Libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input



# Settings
Img_size = 224
CLASSES=["PNEUMONIA", "NORMAL"]


# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/classifier_model.keras",
        compile=False
    )

model = load_model()

# Sidebar
st.sidebar.title("ℹ️ How to Use")
st.sidebar.markdown("""
1. Upload a **PNEUMONIA X-RAY image**
2. Supported formats: **JPG, PNG, JPEG**
3. The model will predict **PNEUMONIA or Normal**
4. Confidence score shows prediction certainty
""")

# UI
st.title("Pneumonia X-Ray Classification")
st.write("Upload a Pneumonia X-Ray image and the model will predict Pneumonia or Normal.")

uploaded_file = st.file_uploader(
    "Upload X-Ray Image",
    type=["jpg", "png", "jpeg"]
)

# Prediction
if uploaded_file is not None:

    ## Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    ## Preprocess
    image = image.resize((Img_size, Img_size))
    img = np.array(image)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    ## Predict
    ## Predict with loading spinner
    with st.spinner("Analyzing X-ray..."):
        prediction = model.predict(x)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    ## Result
    st.subheader("Prediction Result")
    if CLASSES[class_index] == "PNEUMONIA":
        st.error(f"⚠️ Pneumonia Detected")
    else:
        st.success(f"✅ Normal")

    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
