import streamlit as st
import PIL
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import *
import keras
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import regularizers, initializers
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_hub as hub
import IPython.display as display
import matplotlib as mpl
from keras import backend as K
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import time
import functools
import matplotlib.cm as cm
from util_functions import *

densenet_model = tf.keras.applications.DenseNet169(weights="imagenet", include_top=True, input_tensor=Input(shape=(256, 256, 3)))

# densenet_model.trainable = False

combined_dense1 = Dense(512, activation='relu', kernel_initializer='he_normal')(densenet_model.layers[-2].output)
combined_drop1 = Dropout(0.5)(combined_dense1)
combined_dense2 = Dense(256, activation='relu', kernel_initializer='he_normal')(combined_drop1)
combined_drop2 = Dropout(0.5)(combined_dense2)
combined_dense3 = Dense(128, activation='relu', kernel_initializer='he_normal')(combined_drop2)
combined_drop3 = Dropout(0.5)(combined_dense3)
combined_output = Dense(5, activation='sigmoid', kernel_initializer='glorot_uniform', name = 'combined')(combined_drop3)


# # define new model
model = Model(inputs=densenet_model.inputs, outputs=combined_output)


from functools import partial, update_wrapper

def weighted_binary_crossentropy(train_dataset):
    """A weighted binary crossentropy loss function
    that works for multilabel classification
    """
    label_columns = [col for col in train_dataset.columns if col not in ['Movie_Id', 'Movie_Title','Poster_Link']]
    # obtain dataset here
    data = train_dataset[label_columns]
    # create a 2 by N array with weights for 0's and 1's
    weights = np.zeros((2, data.shape[1]))
    # calculates weights for each label in a for loop
    for i in range(data.shape[1]):
        weights_n, weights_p = (data.shape[0]/(2 * (data.iloc[:,i] == 0).sum())), (data.shape[0]/(2 * (data.iloc[:,i] == 1).sum()))
        # weights could be log-dampened to avoid extreme weights for extremly unbalanced data.
        weights[1, i], weights[0, i] = weights_p, weights_n

    # The below is needed to be able to work with keras' model.compile()
    def wrapped_partial(func, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        return partial_func

    def wrapped_weighted_binary_crossentropy(y_true, y_pred, class_weights):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    # cross-entropy loss with weighting
        out = -(y_true * K.log(y_pred)*class_weights[1] + (1.0 - y_true) * K.log(1.0 - y_pred)*class_weights[0])

        return K.mean(out, axis=-1)

    return wrapped_partial(wrapped_weighted_binary_crossentropy, class_weights=weights)
   

#initial_learning_rate = 0.00001 #0.0001
#lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True #learning rate was originally 0.96
#)

#model.compile(
#    loss=weighted_binary_crossentropy(train),#dyn_weighted_bincrossentropy,#,#f1_loss,#'binary_crossentropy',
#    metrics=['accuracy', f1],
#    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue = 1),
#)
#densenet_model = tf.keras.applications.DenseNet169(weights="imagenet", include_top=True, input_tensor=Input(shape=(256, 256, 3)))
#
#combined_dense1 = Dense(512, activation='relu', kernel_initializer='he_normal')(densenet_model.layers[-2].output)
#combined_drop1 = Dropout(0.5)(combined_dense1)
#combined_dense2 = Dense(256, activation='relu', kernel_initializer='he_normal')(combined_drop1)
#combined_drop2 = Dropout(0.5)(combined_dense2)
#combined_dense3 = Dense(128, activation='relu', kernel_initializer='he_normal')(combined_drop2)
#combined_drop3 = Dropout(0.5)(combined_dense3)
#combined_output = Dense(5, activation='sigmoid', kernel_initializer='glorot_uniform', name = 'combined')(combined_drop3)
#
#def f1(y_true, y_pred):
#    y_pred = K.round(y_pred)
#    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
#    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
#    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
#    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
#
#    p = tp / (tp + fp + K.epsilon())
#    r = tp / (tp + fn + K.epsilon())
#
#    f1 = 2*p*r / (p+r+K.epsilon())
#    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
#    return K.mean(f1)
#
#def f1_loss(y_true, y_pred):
#
#    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
#    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
#    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
#    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
#
#    p = tp / (tp + fp + K.epsilon())
#    r = tp / (tp + fn + K.epsilon())
#
#    f1 = 2*p*r / (p+r+K.epsilon())
#    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
#    return 1 - K.mean(f1)
    
# # define new model
model.load_weights('best_val_f1_1504211951.h5')


def import_and_predict2(image):

    list1 = model.predict(image)
#    l = ['Drama', 'Comedy', 'Action', 'Thriller', 'Horror']
    st.write('Prediction Results:')
    #TRY
    return list1
#    for i in range (5):
        #st.write(list1[0][i])
#        if (list1[0][i] > 0.5):
#            st.write(l[i])
    #UNCOMMENT
#    if (list1[0][0] > 0.7936):
#        st.write('Action') #Action
#    if (list1[0][1] > 0.8046):
#        st.write('Comedy') #Comedy
#    if (list1[0][2] > 0.6425):
#        st.write('Drama') #Drama
#    if (list1[0][3] > 0.8419):
#        st.write('Horror') #Horror
#    if (list1[0][4] > 0.7251):
#        st.write('Thriller') #Thriller
        
def style_transfer (image, intended):
    image = np.asarray(image)
    image = image.astype(np.float32)[np.newaxis, ...] / 255.
    #sample = tf.image.resize(sample, (256,256))

    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Stylize image.
    outputs = hub_module(tf.constant(image), tf.constant(intended))
    stylized_image = outputs[0]
    # st.image(stylized_image[0,:,:,:])
    #st.pyplot.imshow(sample)
    fig = plt.figure()
    plt.imshow(stylized_image[0,:,:,:])
    plt.axis('off')
    st.pyplot(fig)

st.title("""
         Classification of Movie Genre using Movie Images
         """)
st.write("CZ1105 REP Lab Group 1- Lim Sheng Jie, Ong Hiok Hian, Koh Yu Xuan")
st.write("Try our Image Clasification Model! Simply upload an image and it will show you the genre!")

user_input = st.text_input("Input the most desired genre (Action / Drama/ Thriller / Horror / Comedy): ")
if (user_input.lower() == 'action'):
    index = 0
if (user_input.lower() == 'comedy'):
    index = 1
if (user_input.lower() == 'drama'):
    index = 2
if (user_input.lower() == 'horror'):
    index = 3
if (user_input.lower() == 'thriller'):
    index = 4
    
if (user_input):
    file = st.file_uploader("Upload an image file", type=["jpg", "png"])
#    st.write(file)
    if file:
        image = Image.open(file).convert('RGB') #Open the Image and Convert
        st.image(image, use_column_width=True, caption = "Your input poster") #Print the image and show user
        size = (256,256)
        
        image = np.asarray(image.resize(size))
        
        
        
#        image1 = ImageOps.fit(image, size, Image.ANTIALIAS)
#        image = image.resize(size)

        preprocess_input = keras.applications.densenet.preprocess_input


        
        user_img = np.expand_dims(image, axis=0)
        x = np.expand_dims(preprocess_input(image), axis=0)
        #MODEL CALL
        list1 = import_and_predict2(x)

        if (list1[0][0] > 0.5276367):
            st.write('Action') #Action
            gradindex = 1
        if (list1[0][1] > 0.5806225):
            st.write('Comedy') #Comedy
            gradindex = 2
        if (list1[0][2] > 0.39837518):
            st.write('Drama') #Drama
            gradindex = 3
        if (list1[0][3] > 0.5521859):
            st.write('Horror') #Horror
            gradindex = 4
        if (list1[0][4] > 0.51436293):
            st.write('Thriller') #Thriller
            gradindex = 5
#        st.write(list1)
        #GradCAM
    #    st.button("Examine my poster feature ")
        col1, col2, col3 , col4, col5 = st.beta_columns(5)
        with col1:
            pass
        with col2:
            examine_button = st.button("Examine my poster")
        with col4:
            view_alt = st.button("View Alternatives")
        with col5:
            pass
        with col3 :
            pass
        list1 = list1.tolist()[0]
        thresholds = {0: 0.5276367, 1: 0.5806225, 2:0.39837518, 3:0.5521859, 4:0.51436293}
        mapbackgenre = {0: 'Action', 1: 'Comedy', 2:'Drama', 3:'Horror', 4:'Thriller'}
        
#        thresholds = {0: 0.7636, 1: 0.8046, 2:0.6425, 3:0.8419, 4:0.7251}
        if (examine_button):
        
            grad_indexes =[list1.index(i) for i in list1 if (i > thresholds[list1.index(i)] and list1.index(i) != index)]
            if len(grad_indexes) >0:
                cols = st.beta_columns(len(grad_indexes))
                for j in range(len(cols)):
                    with cols[j]:
                        st.write("Your poster could be mistaken for " + mapbackgenre[grad_indexes[j]])
                        runGradCAM(model, img = user_img[0,:,:,:], pred_index = grad_indexes[j], alpha = 0.5)
                st.write("Not sure how to change it? Click on View Alternatives for quick suggestions")
            else:
                st.write("Looks Good!")

    # Model that takes in user input
    #    user_input = st.text_input("If the prediction was not what you wanted, enter intended genre (Action / Drama / Thriller/ Horror / Comedy")
        if (view_alt):
            if (user_input == 'Horror'):
                col1, col2, col3 = st.beta_columns(3)
                with col1:
                    st.write("Choice 1")
                    horror1 = Image.open('Horror1.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    horror1 = np.asarray(horror1)
                    horror1 = horror1.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, horror1)
            #horror = tf.image.resize(horror, (256,256))
                with col2:
                    st.write("Choice 2")
                    horror2 = Image.open('Horror2.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    horror2 = np.asarray(horror2)
                    horror2 = horror2.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, horror2)
                with col3:
                    st.write("Choice 3")
                    horror3 = Image.open('Horror3.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    horror3 = np.asarray(horror3)
                    horror3 = horror3.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, horror3)

            elif (user_input == 'Drama'):
                col1, col2, col3 = st.beta_columns(3)
                with col1:
                    st.write("Choice 1")
                    drama1 = Image.open('Drama1.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    drama1 = np.asarray(drama1)
                    drama1 = horror.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, drama1)
                with col2:
                    st.write("Choice 2")
                    drama2 = Image.open('Drama2.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    drama2 = np.asarray(drama2)
                    drama2 = horror.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, drama2)
                with col3:
                    st.write("Choice 3")
                    drama3 = Image.open('Drama2.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    drama3 = np.asarray(drama3)
                    drama3 = horror.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, drama3)

            elif (user_input == 'Action'):
                col1, col2, col3 = st.beta_columns(3)
                with col1:
                    st.write("Choice 1")
                    action1 = Image.open('Action1.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    action1 = np.asarray(action1)
                    action1 = action1.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, action1)
                with col2:
                    st.write("Choice 2")
                    action2 = Image.open('Action2.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    action2 = np.asarray(action2)
                    action2 = action2.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, action2)
                with col3:
                    st.write("Choice 3")
                    action3 = Image.open('Action3.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    action3 = np.asarray(action3)
                    action3 = action3.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, action3)

            elif (user_input == 'Comedy'):
                col1, col2, col3 = st.beta_columns(3)
                with col1:
                    st.write("Choice 1")
                    comedy1 = Image.open('Comedy1.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    comedy1 = np.asarray(comedy1)
                    comedy1 = comedy1.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, comedy1)
                with col2:
                    st.write("Choice 2")
                    comedy2 = Image.open('Comedy2.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    comedy2 = np.asarray(comedy2)
                    comedy2 = comedy2.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, comedy2)
                with col3:
                    st.write("Choice 3")
                    comedy3 = Image.open('Comedy3.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    comedy3 = np.asarray(comedy3)
                    comedy3 = comedy1.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, comedy3)

            elif (user_input == 'Thriller'):
                col1, col2, col3 = st.beta_columns(3)
                with col1:
                    st.write("Choice 1")
                    thriller1 = Image.open('Thriller1.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    thriller1 = np.asarray(thriller1)
                    thriller1 = thriller1.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, thriller1)
                with col2:
                    st.write("Choice 2")
                    thriller2 = Image.open('Thriller2.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    thriller2 = np.asarray(thriller2)
                    thriller2 = thriller1.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, thriller2)
                with col3:
                    st.write("Choice 3")
                    thriller3 = Image.open('Thriller3.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    thriller3 = np.asarray(thriller3)
                    thriller3 = thriller3.astype(np.float32)[np.newaxis, ...] / 255.
                    style_transfer(image, thriller3)

            
