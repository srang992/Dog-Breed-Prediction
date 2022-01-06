"""
  This file containing all the necessary functions for making the streamlit app.
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import os
from PIL import Image
import streamlit as st

with open("dog_breeds_category.pickle", "rb") as file:
    classes = pickle.load(file)

st.session_state['model'] = load_model("models/feature_extractor.h5")
st.session_state['last_layer'] = load_model("models/full_connected_layer.h5")


def image_prediction(img):
    """
    This function is responsible for predicting the dog breed by taking the image.
    :param img: the image in which we are interested.
    :return: predicted label and the probability of the prediction
    """
    img_size = (331, 331, 3)

    img_g = image.load_img(img, target_size=img_size)
    img_g = np.expand_dims(img_g, axis=0)

    test_features = st.session_state['model'].predict(img_g)
    predg = st.session_state['last_layer'].predict(test_features)
    return classes[np.argmax(predg[0])], round(np.max(predg[0])) * 100


def save_uploaded_file(uploaded_file):
    """
     this function help us to take the uploaded image and save it in the local directory for further process.
    :param uploaded_file: the uploaded file
    :return: 1 or 0 denoting true or false.
    """
    try:
        with open(os.path.join('images', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def load_image(image):
    """
     this function will be called when we have to display the image in app after uploading.
    :param image: the uploaded image
    :return: image
    """
    img = Image.open(image)
    return img
