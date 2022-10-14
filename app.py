import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#helper functions
def get_df(seg_type): #* Returns a DataFrame of train or test set of images
    labels = []
    pixels = []
    for image_type in list(os.listdir(f"archive/{seg_type}/{seg_type}")):
        for image in list(os.listdir(f"archive/{seg_type}/{seg_type}/{image_type}")):
            img = np.array(Image.open(f"archive/{seg_type}/{seg_type}/{image_type}/{image}").convert('L').resize((150, 150))) 
            img_y = (img.shape[0])
            img_x = (img.shape[1])
            img = img.reshape(1,img_y*img_x)
            pixels.append(img)
            labels.append(image_type)
    pixels = np.array(pixels)
    pixels = np.reshape(pixels, (-1, 150*150))
    df = pd.DataFrame(pixels)
    df.columns = [f"pixel{n}" for n in range(150*150)]
    df['labels'] = labels
    return df

def run_model(model, df, test_size): #* run and test ML Model
    X = df.drop("labels", axis=1)
    y = df["labels"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    return model

def get_image(uploaded_file):
    st.image(Image.open(uploaded_file))
    img_array = np.array(Image.open(uploaded_file).convert('L').resize((150, 150)))
    img_array.reshape(1,150*150)
    return img_array

def get_image_type(model, uploaded_file):
    img_array = get_image(uploaded_file)
    return model.predict(img_array)[0]
    

# ML Model
train_df = get_df('seg_train')
test_df = get_df('seg_test')
df = pd.concat([train_df,test_df])
model = run_model(RandomForestClassifier(), df, 0.2)


# app
c1, c2, c3 = st.columns([1, 6, 1])
with c2:
    st.title("Image Classifier")
    uploaded_file = st.file_uploader("")
    if uploaded_file is not None:
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