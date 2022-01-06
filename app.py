"""
  In this file, main app implementation is done.
"""
import streamlit as st

from helper_functions import *

st.set_page_config(layout="wide")


st.title("Dog Breed Prediction")

col1, col2 = st.columns([2, 2])

with col1:
    image_file = st.file_uploader("Choose a Image", type=["png", "jpg", "jpeg"])
    save_uploaded_file(image_file)
    submitted = st.button("Predict")
    if submitted:
        breed, pred_percent = image_prediction(os.path.join('images', image_file.name))
        os.remove('images/' + image_file.name)
        st.write("The Predicted Breed is")
        st.subheader(breed)
        st.write(f"Prediction Confidence {pred_percent} %")
with col2:
    if image_file is not None:
        st.image(load_image(image_file))





