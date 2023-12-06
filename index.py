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
from flask import Flask,request,render_template,url_for
import cv2

index=Flask(__name__)

@index.route('/')
def home():
    return render_template('smartfashion.html')


model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

feature_list=np.array(pickle.load(open('embeddingssmall.pkl', 'rb')))
file_names=pickle.load(open('filenamessmall.pkl', 'rb'))


def save_new_file(myfile):
    try:
        with open(os.path.join('inputs',myfile.name),'wb') as t:
            t.write(myfile.getbuffer())
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


@index.route('/predict',methods=['POST','GET'])
def predict():
    myfile = cv2.imread('myfile')
    if request.method == 'POST':
        if save_new_file(myfile):
            myfile = request.files['myfile']
            # image_path = "/sample" + myfile.filename
            # myfile.save(image_path)
            # display_img=Image.open(myfile)
            # st.image(display_img)
            features = feature_extraction(os.path.join('inputs', myfile.name), model)
            # st.text(features)
            indices = recommend(features, feature_list)
            # c1,c2,c3,c4,c5 = st.columns(5)
            # with c1:
            return cv2.imshow('output', cv2.resize(cv2.imread(file_names[indices[0][0]]), (512, 512)))
    return render_template('predict.html')


if __name__=='__main__':
    index.run(debug=True)
