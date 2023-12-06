import streamlit as st
import os
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
feature_list=np.array(pickle.load(open('embeddingsfinal.pkl','rb')))
file_names=pickle.load(open('filenamesfinal.pkl','rb'))

st.title("Smart Fashion")
def save_new_file(new_file):
    try:
        with open(os.path.join('inputs',new_file.name),'wb') as t:
            t.write(new_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expanded_img_array)
    res=model.predict(preprocessed_img).flatten()
    normalized_result=res/norm(res)
    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

new_file=st.file_uploader("Input an image")
if new_file is not None:
    if save_new_file(new_file):
        display_img=Image.open(new_file)
        st.image(display_img)
        features=feature_extraction(os.path.join('inputs',new_file.name),model)
        indices=recommend(features,feature_list)
        c1,c2,c3,c4,c5=st.columns(5)

        with c1:
            st.image(file_names[indices[0][0]])
        with c2:
            st.image(file_names[indices[0][1]])
        with c3:
            st.image(file_names[indices[0][2]])
        with c4:
            st.image(file_names[indices[0][3]])
        with c5:
            st.image(file_names[indices[0][4]])
    else:
        st.header("Error")

