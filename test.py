import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
feature_list=np.array(pickle.load(open('embeddingssmall.pkl','rb')))
#np.arange(len(feature_list)).reshape(1,-1)
file_names=pickle.load(open('filenamessmall.pkl','rb'))
model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('sample/1983.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
res = model.predict(preprocessed_img).flatten()
normalized_result = res / norm(res)
neighbors=NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)
distances,indices=neighbors.kneighbors([normalized_result])
#print(indices)
for file in indices[0]:
    print(file_names[file])