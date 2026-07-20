import streamlit as st
import pandas as pd
import numpy as np

import os
import requests

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

MODEL_PATH="brain_final.keras"

if not os.path.exists(MODEL_PATH):
    url="https://drive.google.com/file/d/1Bumn4TmX9ShBDZYYZk1fgOsqUhAfwQ6r/view?usp=sharing"
    r=request.get(url)
    with open(MODEL_PATH,"wb") as f:
        f.write(r.contect)

model = load_model("brain_final.keras")


st.title("Brain tumor Detection")

uploaded_file = st.file_uploader("Upload an Image ",type = ["jpg","jpeg","png"])

if uploaded_file is not None:
    st.image(uploaded_file)

    img = Image.open(uploaded_file)

    img = img.resize((128,128))  # resize image to 128 , 128 as this size we ussed while training

    img_array = image.img_to_array(img) / 255.0 #create array of uploaded image and normalise it
   
    img_array = np.expand_dims(img_array,axis  = 0)  # converts to expected dimension i.e 1,128,128

    prediction = model.predict(img_array)

    prob = prediction[0][0]

    if prob > 0.5 :
        st.success("Yes you have brain tumor")
    else :
        st.success("No you dont have brain tumor")
