# !pip install mtcnn==0.1.0
# !pip install tensorflow==2.3.1
# !pip install keras==2.4.3
# !pip install keras-vggface==0.6
# !pip install keras_applications==1.0.8

# import os
# import pickle
# cricketer = os.listdir(r'C:\Users\avira\PycharmProjects\cric\indian cricketer')
# filenames = []
# for cricket in cricketer:
#     for file in os.listdir(os.path.join(r'C:\Users\avira\PycharmProjects\cric\indian cricketer',cricket)):
#         filenames.append(os.path.join(r'C:\Users\avira\PycharmProjects\cric\indian cricketer',cricket,file))
# pickle.dump(filenames,open('filenames.pkl','wb'))

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.layers import Flatten, Dense, Input
from keras.engine import  Model
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filenames.pkl','rb'))

hidden_dim=512
nb_class = 2
model = VGGFace(model='senet50',include_top=False,input_shape=(224,224,3),pooling='avg')
last_layer = model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(model.input, out)

def feature_extractor(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = custom_vgg_model.predict(preprocessed_img).flatten()

    return result

features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embedding_vgg.pkl','wb'))