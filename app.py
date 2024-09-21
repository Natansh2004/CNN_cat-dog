import streamlit as st
import cv2
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
import tempfile

model = load_model('./models/img_recognition.h5')  # model loaded

# FRONTEND
st.set_page_config(page_title='Computer Vision CNN')
st.header('Dog and Cat - Image Classification 👽')
uploaded_img = st.file_uploader('Choose any image of cat or dog:-',type=['jpeg','jpg','png'])

if uploaded_img is not None:
    with tempfile.NamedTemporaryFile(delete=False,suffix='.jpg') as temp_file:
        temp_file.write(uploaded_img.getbuffer())
        temp_file_path = temp_file.name()
        st.write(f'Temporary file path: {temp_file_path}')

display_img = Image.open(temp_file_path)
st.image(display_img,caption='image_uploaded',use_column_width=True)

submit = st.button('Predict')


# Image resizing and normalising

if submit:
    img_arr = cv2.imread(uploaded_img)
    resized_img = cv2.resize(img_arr,(150,150))
    x = resized_img/255.0

    # passing to the model
    
    pred = model.predict(x)  
    # it will give probabilities

    prediction = (pred>0.5).astype(int)
    # cat -- 1
    # dog -- 0
    if prediction == 1:
        st.write('The above image is of cat..')
    else:
        st.write('The above image is of dog..')
