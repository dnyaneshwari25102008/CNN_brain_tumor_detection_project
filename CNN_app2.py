import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="brain_final.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Brain Tumor Detection 🧠")

uploaded_file = st.file_uploader(
    "Upload an Image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file)

    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    prob = prediction[0][0]

    if prob > 0.5:
        st.success("Yes, you have a brain tumor")
    else:
        st.success("No, you don't have a brain tumor")
