import numpy as np
import tensorflow as tf
import keras
# Display
import streamlit as st
import matplotlib.cm as cm
from keras import backend as K
from PIL import Image
import cv2
# class ourModel:
#     def __init__(self, ):
#         self.model = createBaseModel()
#         self.gradCAMModel = createGradCAMModel()
#         self.thresholds = {0: 0.5276367, 1: 0.5806225, 2:0.39837518, 3:0.5521859, 4:0.51436293}
#         self.size = (256,256)
#         self.mapBackGenre =  {0: 'Action', 1: 'Comedy', 2:'Drama', 3:'Horror', 4:'Thriller'}
#         self.preprocess_input = keras.applications.densenet.preprocess_input
    
#     def createBaseModel():
#         # create model

#         return model

#     def createGradCAMModel(model):

#         return gradCAMModel



def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array
    
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    base_img = keras.preprocessing.image.img_to_array(img)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((base_img.shape[1], base_img.shape[0]), Image.BICUBIC)
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + base_img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    # display grad CAM
    st.image(superimposed_img, use_column_width=True, caption = "Highlighted parts show parts that may be problematic")



def runGradCAM(model, img, pred_index, img_size = (256,256), last_conv_layer_name = 'conv5_block32_2_conv', alpha = 0.4):
    preprocess_input = keras.applications.densenet.preprocess_input
    # Prepare image
    img_array = np.asarray(img.resize(img_size))
    img_array = preprocess_input(img_array)
    img_array_pred = np.expand_dims(img_array, axis = 0)
    # Remove last layer's sigmoid
    model.layers[-1].activation = None

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array_pred, model, last_conv_layer_name, pred_index=pred_index)
    # # Display heatmap
    save_and_display_gradcam(img, heatmap, alpha = alpha)