import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#helper functions
def get_image(uploaded_file):
    st.image(Image.open(uploaded_file))
    img_array = np.array(Image.open(uploaded_file).convert('L').resize((150, 150))).reshape(1,150*150)
    return img_array

def get_image_type(model, uploaded_file):
    img_array = get_image(uploaded_file)
    return model.predict(img_array)[0]


# app
st.set_page_config( # head tag
    page_title="Image Classification Demo", 
    page_icon="random",
    menu_items={
        'Get Help': 'https://github.com/112523chen/Image-Classification-App',
        'Report a bug': "https://github.com/112523chen/Image-Classification-App/issues/new",
        'About': """
                ***Streamlit app*** that predicts what's inside an image using machine learning supervised learning
                """ 
    })

c1, c2, c3 = st.columns([1, 6, 1])
with c2:
    st.title("Image Classifier")
    uploaded_file = st.file_uploader("")
    if uploaded_file is not None:
        # ML Model
        model = pickle.load(open('RandomForest.pkl','rb'))
        #results
        image_type = get_image_type(model, uploaded_file)
        if image_type == "buildings":
            st.header(f"The image above has {image_type}")
        else:
            st.header(f"The image above has a {image_type}")
    else:
        st.info(
                f"""
                    ⬆️ Upload a .JPG file first.
                    """
            )
        st.stop()