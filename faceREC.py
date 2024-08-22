import streamlit as st 
import cv2 
import numpy as np 

def TOGRAY(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def face_detect(img):


    # face cascade 
    face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")  #  pretrain model 
    gray = TOGRAY(img)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y , w,h) in faces:
        cv2.rectangle(img, (x,y) , (x+w , y+h) , (0,0,255) , 2 )
    return img 



# meta data
uploaded_file = st.file_uploader("select an image" , type=["png" , "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()) , dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    input_col , output_cot = st.columns(2)
    with input_col:
        st.header("image")
        st.image(image, channels= "BGR", use_column_width= True)

    with output_cot:
        st.header("image with detect the faces")
        output = face_detect(image)
        st.image(output, channels= "BGR", use_column_width= True)    